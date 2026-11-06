"""Two CPU/Gloo ranks: QG gradient sync, empty masks, PKM and prototype sync.

Run: python tests/ddp_qg_smoke.py
This does not claim CUDA/NCCL validation.
"""
from datetime import timedelta
from pathlib import Path
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.sampler_ddp import PKMSampler_DDP
from model.backbones.vit_pytorch import TransReID
from processor.processor import PCA, GPD, PrototypeBank


def worker(rank, port):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:{}'.format(port),
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(1234)
        model = TransReID(img_size=(32, 32), stride_size=16, embed_dim=24, depth=12,
                         num_heads=3, num_classes=0, qg_sfm=True)
        model = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.008)
        bank = PrototypeBank(4, 24, device='cpu')
        modalities = ['RGB', 'IR', 'Thermal']
        dataset = {m: [('{}_{}_{}'.format(m, pid, k), pid, k, index + 1)
                       for pid in range(4) for k in range(2)]
                   for index, m in enumerate(modalities)}
        sampler = PKMSampler_DDP(dataset, batch_size=8, num_instances=2, modalities=modalities)
        samples = list(sampler)
        assert len(samples) == 4
        targets = torch.tensor([dataset['RGB'][item[0]][1] for item in samples])
        assert torch.equal(targets.reshape(-1, 2)[:, 0], targets.reshape(-1, 2)[:, 1])
        for item in samples:
            assert len({dataset[m][item[i]][1] for i, m in enumerate(modalities)}) == 1
        # One entire rank sees empty masks on step 0; all ranks see empty masks
        # on step 1. Both cases must retain all 24 parameters in the graph.
        for step in range(2):
            optimizer.zero_grad()
            torch.manual_seed(10 + rank + step)
            images = torch.randn(12, 3, 32, 32)
            coverage = torch.rand(12, 4)
            valid = torch.full((12,), bool(rank == 1 and step == 0))
            if not valid.any():
                coverage.zero_()
            feature = model(images, coverage=coverage, valid_flags=valid)
            pca, ids, centers = PCA(feature, targets, 3, group_size=2)
            loss = feature[:, :6].square().mean() + .01 * pca + GPD(feature, targets, bank, ids, centers, 3)
            loss.backward()
            qg = [p for n, p in model.named_parameters() if '.qg_' in n]
            assert len(qg) == 24
            for parameter in qg:
                assert parameter.grad is not None and torch.isfinite(parameter.grad)
            if step == 0:
                assert any(p.grad.abs().item() > 0 for p in qg)
            optimizer.step()
            bank.update(ids, centers, ddp=True)
            state = torch.cat([p.detach().reshape(-1) for p in qg] + [bank.prototypes.flatten()])
            states = [torch.empty_like(state) for _ in range(2)]
            dist.all_gather(states, state)
            torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
        # Match the existing validation scheme: ranks execute unequal numbers
        # of unwrapped forwards, then join at the next collective.
        model.eval()
        for buffer in model.module.buffers():
            dist.broadcast(buffer, src=0)
        with torch.no_grad():
            for _ in range(rank + 1):
                model.module(images[:1], coverage=coverage[:1], valid_flags=valid[:1])
        dist.barrier()
        if rank == 0:
            print('PASS: two CPU/Gloo ranks; 12 layers/24 scalars, empty-mask backward, PKM ordering, prototype equality, unequal evaluation forwards')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=2, join=True)
