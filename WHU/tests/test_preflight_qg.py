"""CPU regression checks for numerical failure reporting, no CUDA required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from preflight_qg import tensor_summary, loss_gradient_diagnostics, save_report


class PreflightDiagnosticsTests(unittest.TestCase):
    def test_missing_nan_and_inf_are_distinct(self):
        self.assertEqual(tensor_summary(None), {'missing': True})
        result = tensor_summary(torch.tensor([1., float('nan'), float('inf')]))
        self.assertFalse(result['missing'])
        self.assertFalse(result['finite'])
        self.assertEqual(result['nan'], 1)
        self.assertEqual(result['inf'], 1)
        self.assertEqual(result['finite_abs_max'], 1.)

    def test_scaled_fp16_path_overflows_with_finite_fp32_parameter(self):
        # A finite forward and an FP32 leaf do not guarantee that a loss-scaled
        # backward through saved FP16 intermediates remains finite.
        parameter = torch.nn.Parameter(torch.tensor(.1))
        loss = (parameter.half() * torch.tensor([1., 2.], dtype=torch.float16)).sum().float()
        self.assertTrue(torch.isfinite(loss))
        report = loss_gradient_diagnostics({'TOTAL': loss}, [('qg', parameter)], 65536.)
        self.assertEqual(report['TOTAL']['unscaled']['bad'], {})
        self.assertEqual(report['TOTAL']['scaled']['bad']['qg']['inf'], 1)
        self.assertIsNone(parameter.grad)
        # Probing preserves the graph and does not force the normal backward.
        loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad))

    def test_disconnected_gradient_and_failed_report_are_preserved(self):
        connected = torch.nn.Parameter(torch.tensor(1.))
        disconnected = torch.nn.Parameter(torch.tensor(2.))
        result = loss_gradient_diagnostics({'CE': connected.square()},
                                          [('connected', connected), ('absent', disconnected)], 1.)
        self.assertEqual(result['CE']['unscaled']['bad']['absent'], {'missing': True})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'nested'/'failure.json'
            save_report(path, dict(status='failed', failed_step=2, diagnostics=result))
            self.assertEqual(json.loads(path.read_text())['failed_step'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
