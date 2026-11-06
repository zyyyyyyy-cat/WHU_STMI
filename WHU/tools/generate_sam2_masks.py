"""Offline SAM2 masks. Run in wsam; this script never imports the UAD model.

Preview uses train only, sampled across modality/platform groups. Full mode
uses identical prompts on every train/query/test image. No PID-based sharing.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_images(root, preview_per_group, seed):
    images = sorted(p for split in ('train', 'query', 'test')
                    for modality in ('RGB', 'IR', 'Thermal')
                    for p in (root / split / modality).glob('*.jpg'))
    if not images:
        raise ValueError('No WHU images found')
    if not preview_per_group:
        return images
    rng = np.random.default_rng(seed)
    selected = []
    for modality in ('RGB', 'IR', 'Thermal'):
        for platform in ('ground', 'aerial'):
            group = []
            for path in images:
                relative = path.relative_to(root)
                if relative.parts[:2] != ('train', modality):
                    continue
                match = re.search(r'_c(\d+)', path.name)
                if match is None:
                    raise ValueError('Cannot parse camera: {}'.format(path))
                camera = int(match.group(1))
                if camera not in range(1, 8):
                    raise ValueError('Unexpected camera: {}'.format(path))
                if ('ground' if camera <= 5 else 'aerial') == platform:
                    group.append(path)
            if not group:
                raise ValueError('No train images for {}/{}'.format(modality, platform))
            count = min(preview_per_group, len(group))
            selected.extend(group[int(i)] for i in rng.choice(len(group), count, replace=False))
    return sorted(selected)


PROMPT_RULES = ('R00_v1', 'R07_box_only', 'R08_box_center')


def load_selection(root, selection):
    """Read a frozen train-only review list without resampling or reordering."""
    names = json.loads(Path(selection).read_text(encoding='utf-8'))
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError('Selection must be a nonempty list of relative image paths')
    if len(set(names)) != len(names):
        raise ValueError('Selection contains duplicate images')
    images = []
    for name in names:
        path = (root / name).resolve()
        relative = path.relative_to(root)
        if (relative.as_posix() != name or len(relative.parts) != 3
                or relative.parts[0] != 'train'
                or relative.parts[1] not in ('RGB', 'IR', 'Thermal')
                or relative.suffix != '.jpg' or not path.is_file()):
            raise ValueError('Selection requires existing normalized WHU train paths: {}'.format(name))
        images.append(path)
    return images


def prompts(width, height, rule='R00_v1'):
    # Fixed crop prior: the target person is expected near the crop center.
    # This is an experimental prompt rule, not a semantic person detector.
    if rule not in PROMPT_RULES:
        raise ValueError('Unknown prompt rule: {}'.format(rule))
    # Match compare_sam2_rules.prepare_prompt's float32 multiplication exactly.
    scale = np.array([width - 1, height - 1], dtype=np.float32)
    normalized_box = [0, 0, 1, 1] if rule == 'R00_v1' else [.10, .02, .90, .98]
    box = (np.array(normalized_box, dtype=np.float32).reshape(2, 2) * scale).reshape(4)
    if rule == 'R07_box_only':
        return None, None, box
    return (np.array([[.5, .5]], dtype=np.float32) * scale,
            np.array([1], dtype=np.int32), box)


def predict_mask(predictor, image, precision, rule):
    point, label, box = prompts(*image.size, rule=rule)
    context = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
    with torch.inference_mode(), context:
        predictor.set_image(np.asarray(image))
        masks, scores, _ = predictor.predict(point_coords=point, point_labels=label, box=box,
                                             multimask_output=True, return_logits=False)
    mask, score = choose_mask(masks, scores, image.height, image.width)
    return mask, score, int(np.argmax(scores))


def choose_mask(masks, scores, height, width):
    masks, scores = np.asarray(masks), np.asarray(scores)
    if masks.ndim != 3 or masks.shape[1:] != (height, width) or scores.shape != (len(masks),):
        raise ValueError('Unexpected SAM2 mask/score dimensions: {}, {}'.format(masks.shape, scores.shape))
    if not np.isfinite(scores).all() or not np.isfinite(masks).all():
        raise ValueError('SAM2 returned nonfinite results')
    if not np.all((masks == 0) | (masks == 1)):
        raise ValueError('Expected predictor binary masks with return_logits=False')
    index = int(np.argmax(scores))  # deterministic first-index tie breaking
    return masks[index].astype(np.uint8) * 255, float(scores[index])


def validate_existing(path, size):
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        array = np.array(image)
    if array.shape != (size[1], size[0]) or not np.all((array == 0) | (array == 255)):
        raise ValueError('Invalid existing mask: {}'.format(path))


def save_preview(image, mask, destination, title):
    rgb = np.asarray(image)
    overlay = rgb.copy()
    foreground = mask != 0
    overlay[foreground] = (rgb[foreground] * .55 + np.array([0, 255, 0]) * .45).astype(np.uint8)
    panels = [image, Image.fromarray(mask).convert('RGB'), Image.fromarray(overlay)]
    width, height = image.size
    # Native-size files remain untouched; only the review sheet is resized.
    panel_height = 320
    panel_width = max(1, round(width * panel_height / height))
    sheet = Image.new('RGB', (panel_width * 3, panel_height + 48), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, panel in enumerate(panels):
        sheet.paste(panel.resize((panel_width, panel_height)), (index * panel_width, 48))
    draw.text((4, 3), title, fill='black')
    draw.text((4, 23), 'image | binary mask | overlay', fill='black')
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--mask-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-config', default='configs/sam2/sam2_hiera_l.yaml')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('--prompt-rule', choices=PROMPT_RULES, default='R00_v1',
                        help='Explicitly use R08_box_center for the validated v3 experiment; default retains v1 behavior')
    parser.add_argument('--preview-per-group', type=int, default=0, help='>0: train-only stratified preview; 0: full dataset')
    parser.add_argument('--selection', help='Frozen train-only JSON list, e.g. the 300-image v3 selection')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--resume', action='store_true', help='Validate and reuse masks with matching provenance')
    args = parser.parse_args()
    if args.preview_per_group < 0:
        parser.error('--preview-per-group cannot be negative')
    if args.selection and args.preview_per_group:
        parser.error('--selection and --preview-per-group are mutually exclusive')
    if not torch.cuda.is_available():
        raise RuntimeError('Offline generation requires a visible CUDA GPU')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support BF16; explicitly choose --precision fp32')
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    root, output = Path(args.image_root).resolve(), Path(args.mask_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    package = Path(sam2.__file__).resolve().parent
    config_path = package / args.model_config
    if not config_path.is_file():
        raise FileNotFoundError('Cannot locate package config: {}'.format(config_path))
    git = subprocess.run(['git', '-C', str(package.parent), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True, check=True).stdout.strip()
    images = (load_selection(root, args.selection) if args.selection
              else select_images(root, args.preview_per_group, args.seed))
    names = [p.relative_to(root).as_posix() for p in images]
    selection_hash = hashlib.sha256(json.dumps(names, separators=(',', ':')).encode('utf-8')).hexdigest()
    provenance = {
        'generator': 'SAM2ImagePredictor', 'sam2_commit': git,
        'checkpoint_name': checkpoint.name, 'checkpoint_sha256': sha256(checkpoint),
        'model_config': args.model_config, 'config_sha256': sha256(config_path),
        'script_sha256': sha256(__file__), 'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda, 'precision': args.precision,
        'prompt_rule': args.prompt_rule,
        'prompt_box_normalized_xyxy': [0, 0, 1, 1] if args.prompt_rule == 'R00_v1' else [.10, .02, .90, .98],
        'prompt_positive_points_normalized': [] if args.prompt_rule == 'R07_box_only' else [[.5, .5]],
        'coordinate_convention': 'float32 normalized coordinates multiplied by W-1,H-1',
        'selection': 'multimask_output=True; maximum predicted IoU; first index on ties',
        'binarization': 'predictor threshold=0 logits; binary original-resolution output; PNG 0/255',
        'postprocessing': 'build_sam2 apply_postprocessing=False; no hole/sprinkle filtering; no dilation',
        'seed': args.seed,
        'tf32': False,
        'image_selection_sha256': selection_hash, 'image_count': len(images),
        'purpose': 'train-only review' if args.selection or args.preview_per_group else 'full train/query/test',
    }
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / 'provenance.json'
    if metadata.exists():
        if json.loads(metadata.read_text(encoding='utf-8')) != provenance:
            raise ValueError('Existing provenance differs; use a new mask-root instead of mixing settings')
        if not args.resume:
            raise FileExistsError('Output was already initialized; use --resume or a new mask-root')
    else:
        if any(output.iterdir()):
            raise ValueError('Output contains files without provenance; choose an empty mask-root')
        metadata.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    (output / ('preview_selection.json' if args.selection or args.preview_per_group else 'full_selection.json')).write_text(
        json.dumps(names, indent=2), encoding='utf-8')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_sam2(args.model_config, str(checkpoint), device='cuda', apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model, mask_threshold=0., max_hole_area=0., max_sprinkle_area=0.)
    start, generated, skipped = time.time(), 0, 0
    with open(output / 'generation.jsonl', 'a', encoding='utf-8') as log:
        for index, path in enumerate(images, 1):
            relative = path.relative_to(root)
            destination = output / relative.with_suffix('.png')
            with Image.open(path) as source:
                source.load()
                image = source.convert('RGB')
            if destination.exists():
                if not args.resume:
                    raise FileExistsError(destination)
                validate_existing(destination, image.size)
                skipped += 1
                continue
            mask, score, selected_candidate = predict_mask(predictor, image, args.precision, args.prompt_rule)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.tmp.png')
            Image.fromarray(mask).save(temporary)
            temporary.replace(destination)
            record = {'image': relative.as_posix(), 'size': list(image.size),
                      'prompt_rule': args.prompt_rule, 'selected_candidate': selected_candidate,
                      'predicted_iou': score, 'foreground_fraction': float((mask != 0).mean())}
            log.write(json.dumps(record) + '\n')
            log.flush()
            generated += 1
            if args.selection or args.preview_per_group:
                save_preview(image, mask, output / 'previews' / relative.with_suffix('.png'),
                             '{} score={:.3f} fg={:.3f}'.format(path.name, score, record['foreground_fraction']))
            if index % 50 == 0 or args.selection or args.preview_per_group or index == len(images):
                print('{}/{} generated={} skipped={} elapsed={:.1f}s {}'.format(
                    index, len(images), generated, skipped, time.time() - start, relative), flush=True)
    print('Done. generated={} skipped={}; run audit_masks.py in clipreid before training.'.format(generated, skipped))


if __name__ == '__main__':
    main()
