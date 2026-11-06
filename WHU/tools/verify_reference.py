"""Compare a preserved original WHU tree with this implementation on CPU.

Run from WHU: python tools/verify_reference.py --original-root ../tmp/original/WHU
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import cfg
from model.make_model import make_model
from processor.processor import PCA, GPD, PrototypeBank
from loss.wrt_loss import TripletLoss_WRT


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', required=True)
    args = parser.parse_args()
    original = Path(args.original_root)
    legacy_model = load_module('reference_model', original / 'model/__init__.py')
    legacy_processor = load_module('reference_processor', original / 'processor/processor.py')
    legacy_wrt = load_module('reference_wrt', original / 'loss/wrt_loss.py')
    configuration = cfg.clone()
    configuration.merge_from_file(str(ROOT / 'configs/UAD.yml'))
    configuration.MODEL.PRETRAIN_CHOICE = 'none'
    configuration.MODEL.DROP_PATH = 0.
    torch.manual_seed(1234)
    baseline = legacy_model.make_model(configuration, 2, 7, 3).eval()
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / 'manifest.json'
        manifest.write_text(json.dumps({'provenance': {'generator': 'synthetic-reference-v1'}}), encoding='utf-8')
        configuration.DATASETS.MASK_MANIFEST = str(manifest)
        torch.manual_seed(1234)
        improved = make_model(configuration, 2, 7, 3).eval()
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, improved.state_dict()[name], rtol=0, atol=0)
    print('PASS: same seed initializes all original parameters identically')
    improved.load_state_dict(baseline.state_dict(), strict=False)
    base_outputs, new_outputs = [], []
    hooks = []
    for left, right in zip(baseline.base.blocks, improved.base.blocks):
        for lm, rm in ((left.attn, right.attn), (left, right)):
            hooks += [lm.register_forward_hook(lambda m, i, o: base_outputs.append(o.detach())),
                      rm.register_forward_hook(lambda m, i, o: new_outputs.append(o.detach()))]
    images = [torch.randn(2, 3, 256, 128) for _ in range(3)]
    coverage = [torch.rand(2, 128) for _ in range(3)]
    valid = [torch.ones(2, dtype=torch.bool) for _ in range(3)]
    with torch.no_grad():
        expected = baseline(images)
        actual = improved(images, coverage=coverage, valid_flags=valid)
    for handle in hooks:
        handle.remove()
    for left, right in zip(base_outputs + list(expected), new_outputs + list(actual)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    print('PASS: actual original vs QG-zero: all 12 attention outputs, all blocks, CLS, BNNeck, classifier (exact FP32 equality)')
    feature = actual[1]
    targets = torch.tensor([0, 1])
    pca, ids, centers = PCA(feature, targets, 3, group_size=1)
    old_pca, old_ids, old_centers = legacy_processor.PCA(feature, targets, 3, group_size=1)
    bank = PrototypeBank(2, 768, device='cpu')
    old_bank = legacy_processor.PrototypeBank(2, 768, device='cpu')
    for current, reference in (
        (F.cross_entropy(actual[0], targets.repeat(3)), F.cross_entropy(expected[0], targets.repeat(3))),
        (TripletLoss_WRT()(feature, targets.repeat(3))[0], legacy_wrt.TripletLoss_WRT()(feature, targets.repeat(3))[0]),
        (pca, old_pca), (centers, old_centers), (ids, old_ids),
        (GPD(feature, targets, bank, ids, centers, 3),
         legacy_processor.GPD(feature, targets, old_bank, old_ids, old_centers, 3))):
        torch.testing.assert_close(current, reference, rtol=0, atol=0)
    bank.update(ids, centers, ddp=False)
    old_bank.update(old_ids, old_centers, ddp=False)
    torch.testing.assert_close(bank.prototypes, old_bank.prototypes, rtol=0, atol=0)
    print('PASS: original CE/WRT/ProCA/GPD, detached centers and prototype updates agree exactly')


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
