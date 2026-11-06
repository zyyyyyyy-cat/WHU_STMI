"""Two CPU/Gloo ranks: real GradScaler overflow, skipped bank, then recovery."""
from datetime import timedelta
from pathlib import Path
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.amp_step import step_after_unscale
from processor.processor import PrototypeBank


def worker(rank, port):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:{}'.format(port),
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(1234)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=.008, momentum=.9)
        scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
        bank = PrototypeBank(2, 2, device='cpu')
        snapshot = [p.detach().clone() for p in model.parameters()]
        for step in range(2):
            optimizer.zero_grad()
            # Inject overflow in one rank's backward before gradient all-reduce;
            # both ranks must see it and make the same skip/backoff decision.
            handle = None
            if step == 0 and rank == 1:
                handle = model.module.weight.register_hook(lambda g:g*float('inf'))
            loss = model(torch.ones(2, 2)*(rank+1)).square().mean()
            scaler.scale(loss).backward()
            if handle is not None:
                handle.remove()
            scaler.unscale_(optimizer)
            outcome = step_after_unscale(scaler, optimizer, model.parameters(), distributed=True)
            if outcome['optimizer_step']:
                bank.update(torch.tensor([rank]), torch.tensor([[1., 1.]]), ddp=True)
            if step == 0:
                assert outcome['skipped'] and scaler.get_scale()==32768.
                assert len(optimizer.state)==0 and bank.prototypes.count_nonzero()==0
                for before,after in zip(snapshot, model.parameters()):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
            else:
                assert outcome['optimizer_step'] and bank.prototypes.count_nonzero()>0
            vector = torch.cat([p.detach().flatten() for p in model.parameters()]+[bank.prototypes.flatten(),
                                torch.tensor([scaler.get_scale()])])
            states = [torch.empty_like(vector) for _ in range(2)]
            dist.all_gather(states, vector)
            torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
        if rank == 0:
            print('PASS: native AMP overflow/backoff on both ranks; parameters/momentum/prototypes unchanged on skip; recovery and states synchronized')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0)); port=listener.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=2, join=True)
