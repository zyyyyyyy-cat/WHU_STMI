"""Build/audit a frozen image->mask manifest, without running segmentation.

Run from WHU: python tools/audit_masks.py --help
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg
from datasets.masks import MaskStore, JointMaskTransform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root', required=True, help='Dataset directory containing train/query/test')
    parser.add_argument('--mask-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--encoding', choices=['0/1', '0/255'], default='0/255')
    parser.add_argument('--build', action='store_true', help='Create a manifest using mirrored relative paths')
    parser.add_argument('--suffix', choices=['.png', '.npy'], default='.png')
    parser.add_argument('--provenance', help='JSON file recording generator/version/checkpoint/settings/binarization')
    parser.add_argument('--report', required=True, help='JSON report destination')
    args = parser.parse_args()
    image_root = Path(args.image_root).resolve()
    images = sorted(p for split in ('train', 'query', 'test')
                    for p in (image_root / split).glob('*/*.jpg'))
    if not images:
        raise ValueError('No WHU images found')
    manifest = Path(args.manifest)
    if args.build:
        if manifest.exists():
            raise FileExistsError('Refusing to overwrite a fixed manifest: {}'.format(manifest))
        if not args.provenance:
            parser.error('--build requires --provenance')
        with open(args.provenance, encoding='utf-8') as stream:
            provenance = json.load(stream)
        doc = {'version': 1, 'encoding': args.encoding, 'provenance': provenance,
               'images': {p.relative_to(image_root).as_posix():
                          p.relative_to(image_root).with_suffix(args.suffix).as_posix() for p in images}}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        # Write only after every mask has passed the audit below.
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / 'manifest.json'
            candidate.write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
            store = MaskStore(image_root, args.mask_root, candidate, args.encoding)
    else:
        doc = None
        store = MaskStore(image_root, args.mask_root, manifest, args.encoding)
    transform = JointMaskTransform(cfg, training=False)
    groups = defaultdict(lambda: {'count': 0, 'empty': 0, 'foreground': [], 'hist': np.zeros(10, dtype=np.int64)})
    for path in images:
        with Image.open(path) as source:
            source.load()
            mask = store.read(path, source.size)
            _, coverage, valid = transform(source.convert('RGB'), mask)
        # Original WHU camera IDs: 1..5 ground; 6..7 aerial.
        import re
        match = re.search(r'_c(\d+)', path.name)
        if match is None:
            raise ValueError('Cannot parse camera ID: {}'.format(path))
        platform = 'ground' if int(match.group(1)) <= 5 else 'aerial'
        relative = path.relative_to(image_root)
        key = '/'.join((relative.parts[0], relative.parts[1], platform))
        group = groups[key]
        group['count'] += 1
        group['empty'] += int(not valid)
        group['foreground'].append(float(coverage.mean()))
        group['hist'] += np.histogram(coverage.numpy(), bins=np.linspace(0, 1, 11))[0]
    report = {}
    for key, group in sorted(groups.items()):
        report[key] = {'count': group['count'], 'empty_count': group['empty'],
                       'empty_fraction': group['empty'] / group['count'],
                       'mean_foreground_coverage': float(np.mean(group['foreground'])),
                       'patch_coverage_bins': np.linspace(0, 1, 11).tolist(),
                       'patch_coverage_histogram': group['hist'].tolist()}
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if doc is not None:
        manifest.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Audited {} images; {} scene/split groups. Report: {}'.format(len(images), len(report), destination))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
