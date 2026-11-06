"""Compare production PNGs with frozen rule-comparison PNGs, without SAM/UAD."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def compare(generated_root, reference_root, rule, selection=None):
    generated_root, reference_root = Path(generated_root).resolve(), Path(reference_root).resolve()
    generated = json.loads((generated_root / 'provenance.json').read_text(encoding='utf-8'))
    reference = json.loads((reference_root / 'provenance.json').read_text(encoding='utf-8'))
    if generated.get('prompt_rule') != rule:
        raise ValueError('Production provenance does not use {}'.format(rule))
    for key in ('checkpoint_sha256', 'model_config', 'config_sha256', 'precision',
                'sam2_commit', 'torch_version', 'cuda_version'):
        if generated.get(key) != reference.get(key):
            raise ValueError('Generation conditions differ: {}'.format(key))
    names = reference['selection'] if selection is None else json.loads(
        Path(selection).read_text(encoding='utf-8'))
    if not names or len(set(names)) != len(names) or not set(names).issubset(reference['selection']):
        raise ValueError('Verification requires a nonempty unique subset of reference selection')
    mismatches = []
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Expected relative image paths')
        left = generated_root / relative.with_suffix('.png')
        right = reference_root / 'masks' / (rule + '__iou') / relative.with_suffix('.png')
        arrays = []
        for path in (left, right):
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                array = np.array(image)
            if array.ndim != 2 or not np.all((array == 0) | (array == 255)):
                raise ValueError('Expected binary 0/255 PNG: {}'.format(path))
            arrays.append(array)
        if arrays[0].shape != arrays[1].shape or not np.array_equal(*arrays):
            mismatches.append(name)
    return {'rule': rule, 'checked': len(names), 'identical': len(names) - len(mismatches),
            'mismatches': mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-root', required=True)
    parser.add_argument('--reference-root', required=True)
    parser.add_argument('--rule', choices=['R00_v1', 'R07_box_only', 'R08_box_center'], default='R08_box_center')
    parser.add_argument('--selection')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report = compare(args.generated_root, args.reference_root, args.rule, args.selection)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Checked {}; identical {}; mismatches {}'.format(
        report['checked'], report['identical'], len(report['mismatches'])))
    if report['mismatches']:
        raise SystemExit('Mask mismatch: inspect the report before full generation')


if __name__ == '__main__':
    main()
