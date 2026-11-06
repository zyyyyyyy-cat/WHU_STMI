"""CPU/Gloo recovery parity with real PKM DDP, BN, AMP and prototype state."""
import argparse
from datetime import timedelta
from pathlib import Path
import socket
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.sampler_ddp import PKMSampler_DDP
from test_checkpoint import RandomDataset, seed, make_run, assert_tree
from utils.checkpoint import save_training_checkpoint, load_training_checkpoint
from solver.amp_step import step_after_unscale
import unittest


def make_distributed_run(world):
    run = list(make_run())
    run[0] = torch.nn.parallel.DistributedDataParallel(run[0])
    dataset = RandomDataset()
    # Four identities suffice for 2/4 ranks with two identities per local batch.
    run[-1] = DataLoader(dataset, batch_size=4,
                         sampler=PKMSampler_DDP(dataset.dataset, world * 4, 2, ['RGB']))
    return run


def epoch(run, number, rank, world):
    model, optimizer, _, _, scheduler, scaler, bank, loader = run
    model.train()
    scheduler.step(number)
    trace = []
    for image, target in loader:
        optimizer.zero_grad()
        output = model(image)
        loss = torch.nn.functional.cross_entropy(output, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        result = step_after_unscale(scaler, optimizer, model.parameters(), distributed=True)
        assert result['optimizer_step']
        bank.update(target, image.new_ones(len(target), 3), ddp=True)
        trace.append((image.clone(), output.detach().clone(), loss.item()))
    # Same whole-modality assignment as the real evaluator: fourth rank is idle.
    for buffer in model.module.buffers():
        dist.broadcast(buffer, 0)
    model.eval()
    for mode in range(3):
        if mode % world == rank:
            for _ in DataLoader(torch.ones(3, 2), batch_size=2):
                pass
    return trace


def worker(rank, world, port, folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:{}'.format(port),
                            rank=rank, world_size=world, timeout=timedelta(seconds=90))
    try:
        test = unittest.TestCase()
        seed()
        original = make_distributed_run(world)
        epoch(original, 1, rank, world)
        # Also cover checkpoint epochs without validation, where BN is rank-local.
        with torch.no_grad():
            original[0].module[1].running_mean.add_(rank)
        buffers = {k: v.clone() for k, v in original[0].module.named_buffers()}
        path = Path(folder) / 'last.pth'
        save_training_checkpoint(path, 1, *original[:-1], original[-1], {'config': {}},
                                 dict(successful_updates=4, amp_skips=0, consecutive_amp_skips=0))
        expected = epoch(original, 2, rank, world)
        seed(9876)
        resumed = make_distributed_run(world)
        completed, _ = load_training_checkpoint(path, *resumed[:-1], resumed[-1], {'config': {}}, 4)
        assert completed == 1
        assert_tree(test, buffers, dict(resumed[0].module.named_buffers()))
        actual = epoch(resumed, 2, rank, world)
        # A fresh DDP reducer may use a different bucket layout; floating-point
        # all-reduce ordering is not guaranteed bitwise across reconstruction.
        assert_tree(test, expected, actual, rtol=1e-6, atol=1e-7)
        for a, b in zip(original[:-1], resumed[:-1]):
            assert_tree(test, a.state_dict(), b.state_dict(), rtol=1e-6, atol=1e-7)
        # Failure on one rank must propagate to every rank instead of hanging.
        bad_path = Path(folder) / 'missing.pth' if rank == world - 1 else path
        try:
            load_training_checkpoint(bad_path, *resumed[:-1], resumed[-1], {'config': {}}, 4)
        except RuntimeError as exc:
            assert 'Read/validate checkpoint failed' in str(exc)
        else:
            raise AssertionError('Missing rank-local file must fail on all ranks')
        try:
            # Parent is an existing file: only the writer sees the filesystem
            # error first, but every rank must receive the failure.
            save_training_checkpoint(path / 'child.pth', 2, *resumed[:-1], resumed[-1],
                                     {'config': {}}, {})
        except RuntimeError as exc:
            assert 'Save checkpoint failed' in str(exc)
        else:
            raise AssertionError('Rank-0 write failure must fail on all ranks')
        if rank == 0:
            print('PASS: {} ranks; exact rank-local BN restore; PKM/RNG recovery; parameters/momentum/scheduler/scaler/prototypes match uninterrupted training within rtol=1e-6 atol=1e-7; read/write failures propagated'.format(world))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(worker, args=(args.world_size, port, folder), nprocs=args.world_size, join=True)
