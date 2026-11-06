"""Resume parity and rejection/atomicity tests; no WHU data or CUDA required."""
import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg
from datasets.sampler import PKMSampler
from processor.processor import PrototypeBank
from solver.scheduler_factory import create_scheduler
from solver.amp_step import step_after_unscale
from utils.checkpoint import (atomic_torch_save, save_training_checkpoint,
                              load_training_checkpoint, training_signature)


class RandomDataset(Dataset):
    def __init__(self):
        self.dataset = {'RGB': [('{}.jpg'.format(i), i // 4, 0, 1) for i in range(16)]}

    def __len__(self):
        return 16

    def __getitem__(self, index):
        index = index[0]
        value = random.random() + float(np.random.random()) + torch.rand(()).item()
        return torch.tensor([index / 16., value], dtype=torch.float32), int(index // 4)


def seed(value=1234):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def make_run(workers=0):
    dataset = RandomDataset()
    loader = DataLoader(dataset, batch_size=4, num_workers=workers,
                        sampler=PKMSampler(dataset.dataset, 4, 2, ['RGB']))
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3),
                                torch.nn.Dropout(.3), torch.nn.Linear(3, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=.008, momentum=.9)
    center = torch.nn.Linear(3, 4)
    optimizer_center = torch.optim.SGD(center.parameters(), lr=.5, momentum=.9)
    config = cfg.clone()
    config.SOLVER.MAX_EPOCHS = 4
    scheduler = create_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler('cpu', init_scale=32.)
    bank = PrototypeBank(4, 3, device='cpu')
    return model, optimizer, optimizer_center, center, scheduler, scaler, bank, loader


def epoch(run, number):
    model, optimizer, _, _, scheduler, scaler, bank, loader = run
    scheduler.step(number)
    model.train()
    trace = []
    for image, target in loader:
        optimizer.zero_grad()
        output = model(image)
        loss = torch.nn.functional.cross_entropy(output, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        bank.update(target, image.new_ones(len(target), 3), ddp=False)
        trace.append((image.clone(), output.detach().clone(), loss.item()))
    # Simulate evaluation iterator RNG consumption before the checkpoint boundary.
    model.eval()
    for _ in DataLoader(torch.ones(3, 2), batch_size=2, num_workers=loader.num_workers):
        pass
    return trace


def assert_tree(test, left, right, rtol=0, atol=0):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree(test, left[key], right[key], rtol, atol)
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree(test, a, b, rtol, atol)
    else:
        test.assertEqual(left, right)


@unittest.skipUnless(hasattr(getattr(torch, 'amp', None), 'GradScaler'), 'Requires local CPU AMP')
class CheckpointTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_with_zero_and_two_workers(self):
        for workers in (0, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as folder:
                seed()
                original = make_run(workers)
                epoch(original, 1)
                path = Path(folder) / 'last.pth'
                counters = dict(successful_updates=4, amp_skips=2, consecutive_amp_skips=0)
                save_training_checkpoint(path, 1, *original[:-1], original[-1], {'test': 1}, counters)
                expected = epoch(original, 2)
                # Deliberately disturb construction and all RNG streams.
                seed(9876)
                resumed = make_run(workers)
                completed, restored = load_training_checkpoint(
                    path, *resumed[:-1], resumed[-1], {'test': 1}, 4)
                self.assertEqual(completed, 1)
                self.assertEqual(restored, counters)
                actual = epoch(resumed, 2)
                assert_tree(self, expected, actual)
                for a, b in zip(original[:-1], resumed[:-1]):
                    assert_tree(self, a.state_dict(), b.state_dict())

    def test_bad_resume_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = make_run()
            path = Path(folder) / 'last.pth'
            atomic_torch_save(run[0].state_dict(), path)
            with self.assertRaisesRegex(RuntimeError, 'model-only'):
                load_training_checkpoint(path, *run[:-1], run[-1], {'config': {}}, 4)
            save_training_checkpoint(path, 1, *run[:-1], run[-1], {'config': {}, 'x': 1}, {})
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                load_training_checkpoint(path, *run[:-1], run[-1], {'config': {}, 'x': 2}, 4)
            data = torch.load(path, weights_only=False)
            data['world_size'] = 2
            atomic_torch_save(data, path)
            with self.assertRaisesRegex(RuntimeError, 'same GPU/process count'):
                load_training_checkpoint(path, *run[:-1], run[-1], {'config': {}, 'x': 1}, 4)

    def test_atomic_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pth'
            atomic_torch_save({'epoch': 1}, path)
            with patch('utils.checkpoint.os.replace', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    atomic_torch_save({'epoch': 2}, path)
            self.assertEqual(torch.load(path, weights_only=False), {'epoch': 1})
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_scaler_backoff_and_prototypes_survive_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            run = make_run()
            epoch(run, 1)
            model, optimizer, _, _, _, scaler, bank, _ = run
            optimizer.zero_grad()
            scaler.scale(model(torch.ones(4, 2)).square().mean()).backward()
            next(model.parameters()).grad.fill_(float('inf'))
            scaler.unscale_(optimizer)
            result = step_after_unscale(scaler, optimizer, model.parameters())
            self.assertTrue(result['skipped'])
            self.assertEqual(scaler.get_scale(), 16.)
            path = Path(folder) / 'last.pth'
            save_training_checkpoint(path, 1, *run[:-1], run[-1], {'config': {}},
                                     dict(successful_updates=4, amp_skips=1, consecutive_amp_skips=1))
            restored = make_run()
            _, counters = load_training_checkpoint(path, *restored[:-1], restored[-1], {'config': {}}, 4)
            self.assertEqual(counters['consecutive_amp_skips'], 1)
            assert_tree(self, scaler.state_dict(), restored[5].state_dict())
            assert_tree(self, bank.state_dict(), restored[6].state_dict())
            assert_tree(self, optimizer.state_dict(), restored[1].state_dict())

    def test_changed_optimizer_order_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = make_run()
            path = Path(folder) / 'last.pth'
            save_training_checkpoint(path, 1, *run[:-1], run[-1], {'config': {}}, {})
            run[1].param_groups[0]['params'].reverse()
            with self.assertRaisesRegex(RuntimeError, 'group names/order'):
                load_training_checkpoint(path, *run[:-1], run[-1], {'config': {}}, 4)

    def test_signature_checks_protocol_and_order_but_allows_relocation(self):
        run = make_run()
        config = cfg.clone()
        before = training_signature(config, run[-1], [])
        config.SOLVER.RESUME = '/new/last.pth'
        config.DATASETS.ROOT_DIR = '/new/data'
        config.OUTPUT_DIR = '/new/output'
        self.assertEqual(before, training_signature(config, run[-1], []))
        run[-1].dataset.dataset['RGB'].reverse()
        self.assertNotEqual(before, training_signature(config, run[-1], []))
        run[-1].dataset.dataset['RGB'].reverse()
        config.SOLVER.IMS_PER_BATCH *= 2
        self.assertNotEqual(before, training_signature(config, run[-1], []))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
