"""Native AMP skip/backoff and optimizer/prototype safety on CPU."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver.amp_step import step_after_unscale


@unittest.skipUnless(hasattr(getattr(torch, 'amp', None), 'GradScaler'),
                     'CPU GradScaler tests require a newer local PyTorch; server uses CUDA scaler')
class AmpStepTests(unittest.TestCase):
    def test_native_overflow_skip_and_recovery(self):
        parameter = torch.nn.Parameter(torch.tensor(.1))
        optimizer = torch.optim.SGD([parameter], lr=.1, momentum=.9)
        scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
        loss = (parameter.half() * torch.tensor(2., dtype=torch.float16)).float()
        self.assertTrue(torch.isfinite(loss))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        before = parameter.detach().clone()
        outcome = step_after_unscale(scaler, optimizer, [parameter])
        self.assertTrue(outcome['skipped'])
        self.assertEqual(outcome['scale_after'], 32768.)
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        self.assertEqual(len(optimizer.state), 0)  # momentum must not advance
        optimizer.zero_grad()
        scaler.scale(parameter.square()).backward()
        scaler.unscale_(optimizer)
        recovered = step_after_unscale(scaler, optimizer, [parameter])
        self.assertTrue(recovered['optimizer_step'])
        self.assertLess(parameter.item(), before.item())
        self.assertIn('momentum_buffer', optimizer.state[parameter])

    def test_finite_step_matches_original_update(self):
        left = torch.nn.Parameter(torch.tensor([.3, -.4]))
        right = torch.nn.Parameter(left.detach().clone())
        opt_left = torch.optim.SGD([left], lr=.008, momentum=.9)
        opt_right = torch.optim.SGD([right], lr=.008, momentum=.9)
        for _ in range(3):
            scaler_left = torch.amp.GradScaler('cpu', init_scale=32.)
            scaler_right = torch.amp.GradScaler('cpu', init_scale=32.)
            for p, opt, scaler in ((left,opt_left,scaler_left),(right,opt_right,scaler_right)):
                opt.zero_grad(); scaler.scale((p*20).square().sum()).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([left], 1.)
            scaler_left.step(opt_left); scaler_left.update()
            outcome = step_after_unscale(scaler_right, opt_right, [right])
            self.assertFalse(outcome['skipped'])
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            torch.testing.assert_close(opt_left.state[left]['momentum_buffer'],
                                       opt_right.state[right]['momentum_buffer'], rtol=0, atol=0)

    def test_fp32_nonfinite_is_not_treated_as_recoverable(self):
        parameter = torch.nn.Parameter(torch.tensor(.1))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        parameter.grad = torch.tensor(float('nan'))
        with self.assertRaises(FloatingPointError):
            step_after_unscale(scaler, optimizer, [parameter])
        self.assertAlmostEqual(parameter.item(), .1)

    def test_nan_skip_preserves_existing_momentum(self):
        parameter = torch.nn.Parameter(torch.tensor(.1))
        optimizer = torch.optim.SGD([parameter], lr=.1, momentum=.9)
        scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
        scaler.scale(parameter.square()).backward(); scaler.unscale_(optimizer)
        step_after_unscale(scaler, optimizer, [parameter])
        before = parameter.detach().clone()
        momentum = optimizer.state[parameter]['momentum_buffer'].clone()
        optimizer.zero_grad()
        scaler.scale(parameter.square()).backward()
        parameter.grad.fill_(float('nan'))
        scaler.unscale_(optimizer)
        result = step_after_unscale(scaler, optimizer, [parameter])
        self.assertTrue(result['skipped'])
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[parameter]['momentum_buffer'], momentum, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
