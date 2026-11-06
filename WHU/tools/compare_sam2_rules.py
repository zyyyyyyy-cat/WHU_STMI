"""Training-only WHU SAM2 prompt comparison. Does not modify UAD or v1 masks."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from generate_sam2_masks import sha256, validate_existing


def rules():
    # Coordinates normalized to [0,1] in the ORIGINAL crop, before padding.
    center = [[.5, .5]]
    corners = [[.02, .02], [.98, .02], [.02, .98], [.98, .98]]
    vertical = [[.5, .25], [.5, .5], [.5, .75]]
    cross = [[.5, .5], [.4, .5], [.6, .5]]
    return [
        dict(name='R00_v1', points=center, box=[0, 0, 1, 1]),
        dict(name='R01_center', points=center),
        dict(name='R02_upper', points=[[.5, .35]]),
        dict(name='R03_vertical', points=vertical),
        dict(name='R04_compact', points=cross),
        dict(name='R05_center_neg', points=center + corners, labels=[1] + [0]*4),
        dict(name='R06_vertical_neg', points=vertical + corners, labels=[1]*3 + [0]*4),
        dict(name='R07_box_only', box=[.10, .02, .90, .98]),
        dict(name='R08_box_center', points=center, box=[.10, .02, .90, .98]),
        dict(name='R09_compact_box', points=center, box=[.08, .08, .92, .92]),
        dict(name='R10_center_single', points=center, multi=False),
        dict(name='R11_vertical_single', points=vertical, multi=False),
        dict(name='R12_padding_center', points=center, encoding='padding'),
        dict(name='R13_padding_box', points=center, box=[0, 0, 1, 1], encoding='padding'),
        dict(name='R14_IR_contrast', points=center, encoding='ir_contrast'),
    ]


def input_image(image, encoding):
    if encoding == 'padding':
        margin = max(1, round(.12 * max(image.size)))
        array = np.array(image)
        boundary = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
        color = tuple(np.median(boundary, axis=0).astype(np.uint8).tolist())
        return ImageOps.expand(image, margin, fill=color), (margin, margin)
    if encoding == 'ir_contrast':
        # IR only: monotonic grayscale 1/99 percentile stretch. Not a thermal
        # temperature estimator; do not transform pseudo-colored Thermal.
        array = np.asarray(image.convert('L'), dtype=np.float32)
        low, high = np.percentile(array, [1, 99])
        if high > low:
            array = np.clip((array-low) * (255. / (high-low)), 0, 255)
        return Image.fromarray(array.astype(np.uint8)).convert('RGB'), (0, 0)
    return image, (0, 0)


def prepare_prompt(rule, original_size, offset):
    scale = np.array([original_size[0]-1, original_size[1]-1], dtype=np.float32)
    shift = np.array(offset, dtype=np.float32)
    points = np.array(rule['points'], dtype=np.float32) * scale + shift if 'points' in rule else None
    labels = np.array(rule.get('labels', [1]*len(points)), dtype=np.int32) if points is not None else None
    box = None
    if 'box' in rule:
        box = (np.array(rule['box'], dtype=np.float32).reshape(2, 2)*scale + shift).reshape(4)
    return dict(point_coords=points, point_labels=labels, box=box,
                multimask_output=rule.get('multi', True), return_logits=False)


def candidate_metrics(mask, score, rule):
    height, width = mask.shape
    area = float(mask.mean())
    border = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    border_coverage = float(border.mean())
    points = rule.get('points', [])
    labels = rule.get('labels', [1]*len(points))
    positive, negative = [], []
    for (x, y), label in zip(points, labels):
        value = float(mask[round(y*(height-1)), round(x*(width-1))])
        (positive if label else negative).append(value)
    pos = float(np.mean(positive)) if positive else 0.
    neg = float(np.mean(negative)) if negative else 0.
    # Smooth penalty, not an invalid-mask gate. No pixel edits, complements,
    # area clipping, or zero-mask fallback. Heuristics are comparison only.
    extreme = max(0., (area-.85)/.15) + max(0., (.03-area)/.03)
    rank = float(score) + .20*pos - .20*neg - .15*border_coverage - .30*extreme
    return dict(predicted_iou=float(score), foreground_fraction=area,
                border_coverage=border_coverage, positive_hit=pos,
                negative_hit=neg, extreme_penalty=extreme, geometry_rank=rank)


def overlay(image, mask, points=None, labels=None, box=None):
    array = np.array(image)
    array[mask] = (array[mask]*.55 + np.array([0, 255, 0])*.45).astype(np.uint8)
    panel = Image.fromarray(array)
    # Draw prompts on the panel after resizing to keep dots visible on tiny crops.
    panel = ImageOps.contain(panel, (260, 240))
    draw = ImageDraw.Draw(panel)
    sx, sy = panel.width/image.width, panel.height/image.height
    if points is not None:
        for (x, y), label in zip(points, labels):
            x, y = float(x)*sx, float(y)*sy
            draw.ellipse((x-3, y-3, x+3, y+3), fill='lime' if label else 'red', outline='black')
    if box is not None:
        draw.rectangle(tuple(float(v)*(sx if i%2 == 0 else sy) for i, v in enumerate(box)), outline='yellow', width=1)
    return panel


def render_sheet(tiles, destination):
    columns, cell_width, cell_height = 4, 290, 295
    sheet = Image.new('RGB', (columns*cell_width, ((len(tiles)+columns-1)//columns)*cell_height), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, (title, subtitle, panel) in enumerate(tiles):
        x, y = index%columns*cell_width, index//columns*cell_height
        draw.text((x+4, y+4), title, fill='black')
        draw.text((x+4, y+19), subtitle, fill='black')
        sheet.paste(panel, (x+(cell_width-panel.width)//2, y+48))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def run_image(predictor, image, relative, output, precision, enabled_rules):
    overview = [('original', str(relative), ImageOps.contain(image, (260, 240)))]
    candidate_tiles = [('original', str(relative), ImageOps.contain(image, (260, 240)))]
    records = []
    for encoding in ('native', 'padding', 'ir_contrast'):
        applicable = [r for r in enabled_rules if r.get('encoding', 'native') == encoding
                      and (encoding != 'ir_contrast' or relative.parts[1] == 'IR')]
        if not applicable:
            continue
        transformed, offset = input_image(image, encoding)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
        with torch.inference_mode(), context:
            predictor.set_image(np.asarray(transformed))  # once per encoding, cached for all prompts
            for rule in applicable:
                kwargs = prepare_prompt(rule, image.size, offset)
                masks, scores, _ = predictor.predict(**kwargs)
                masks, scores = np.asarray(masks), np.asarray(scores)
                if masks.ndim != 3 or masks.shape[1:] != (transformed.height, transformed.width) or scores.shape != (len(masks),):
                    raise ValueError('Unexpected predictor shapes')
                if not np.isfinite(masks).all() or not np.isfinite(scores).all() or not np.all((masks==0) | (masks==1)):
                    raise ValueError('Invalid mask/score')
                left, top = offset
                masks = masks[:, top:top+image.height, left:left+image.width].astype(bool)
                metrics = [candidate_metrics(mask, score, rule) for mask, score in zip(masks, scores)]
                prompt = prepare_prompt(rule, image.size, (0, 0))
                for index, (mask, metric) in enumerate(zip(masks, metrics)):
                    destination = output/'candidates'/rule['name']/relative.parent/(relative.stem+'__c{}.png'.format(index))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask.astype(np.uint8)*255).save(destination)
                    panel = overlay(image, mask, prompt['point_coords'], prompt['point_labels'], prompt['box'])
                    candidate_tiles.append((rule['name']+' c{}'.format(index),
                        'q={:.3f} a={:.3f} g={:.3f}'.format(metric['predicted_iou'], metric['foreground_fraction'], metric['geometry_rank']), panel))
                for selector in ('iou', 'geometry'):
                    key = 'predicted_iou' if selector == 'iou' else 'geometry_rank'
                    best = int(np.argmax([m[key] for m in metrics]))
                    variant = rule['name']+'__'+selector
                    destination = output/'masks'/variant/relative.with_suffix('.png')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(masks[best].astype(np.uint8)*255).save(destination)
                    metric = metrics[best]
                    overview.append((variant+' c{}'.format(best), 'q={:.3f} fg={:.3f}'.format(metric['predicted_iou'], metric['foreground_fraction']),
                                     overlay(image, masks[best], prompt['point_coords'], prompt['point_labels'], prompt['box'])))
                    records.append(dict(image=relative.as_posix(), size=list(image.size), variant=variant, selected_candidate=best,
                                        candidates=metrics, **metric))
    render_sheet(overview, output/'review/selected'/relative.with_suffix('.jpg'))
    render_sheet(candidate_tiles, output/'review/candidates'/relative.with_suffix('.jpg'))
    return records


def summarize(records):
    groups = {}
    for row in records:
        path = Path(row['image'])
        import re
        camera = int(re.search(r'_c(\d+)', path.name).group(1))
        group = path.parts[1] + ('_G' if camera <= 5 else '_A')
        key = row['variant']+'/'+group
        groups.setdefault(key, []).append(row)
    return {key: dict(count=len(rows), mean_area=float(np.mean([r['foreground_fraction'] for r in rows])),
                      area_ge_90=sum(r['foreground_fraction']>=.9 for r in rows),
                      area_le_03=sum(r['foreground_fraction']<=.03 for r in rows),
                      mean_predicted_iou=float(np.mean([r['predicted_iou'] for r in rows])))
            for key, rows in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--selection', required=True, help='v1 preview_selection.json; train images only')
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-config', default='configs/sam2/sam2_hiera_l.yaml')
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported')
    root, output = Path(args.image_root).resolve(), Path(args.output).resolve()
    selected = json.loads(Path(args.selection).read_text(encoding='utf-8'))
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError('Selection must contain unique image paths')
    for name in selected:
        path = (root/name).resolve()
        path.relative_to(root)
        relative = path.relative_to(root)
        if relative.parts[0] != 'train' or relative.parts[1] not in ('RGB', 'IR', 'Thermal') or not path.is_file():
            raise ValueError('Only existing WHU train images are allowed: {}'.format(name))
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    package = Path(sam2.__file__).resolve().parent
    provenance = dict(purpose='train-only comparison; not a production mask manifest',
                      checkpoint_sha256=sha256(args.checkpoint), model_config=args.model_config,
                      config_sha256=sha256(package/args.model_config), precision=args.precision,
                      torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                      sam2_commit=subprocess.run(['git','-C',str(package.parent),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
                      script_sha256=sha256(__file__), helper_sha256=sha256(Path(__file__).with_name('generate_sam2_masks.py')),
                      selection=selected, rules=rules(), geometry_formula='q+.20*positive-.20*negative-.15*border-.30*(max(0,(area-.85)/.15)+max(0,(.03-area)/.03))',
                      padding='12% of max(original W,H) on each side; constant median original border color; crop masks back',
                      ir_contrast='IR only: grayscale monotonic percentile 1/99 stretch; unchanged image if no intensity range',
                      postprocessing=False, mask_logit_threshold=0., tf32=False, seed=1234)
    output.mkdir(parents=True, exist_ok=True)
    metadata=output/'provenance.json'
    if metadata.exists():
        if not args.resume or json.loads(metadata.read_text(encoding='utf-8')) != provenance:
            raise ValueError('Existing output requires --resume and identical provenance')
    elif any(output.iterdir()):
        raise ValueError('Choose empty output directory')
    else:
        metadata.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=build_sam2(args.model_config,args.checkpoint,device='cuda',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model,mask_threshold=0.,max_hole_area=0.,max_sprinkle_area=0.)
    records=[]
    start=time.time()
    for index,name in enumerate(selected,1):
        relative=Path(name)
        marker=output/'records'/relative.with_suffix('.json')
        if marker.exists() and args.resume:
            rows=json.loads(marker.read_text(encoding='utf-8'))
            # Completed-image markers are written last; verify the associated
            # outputs before trusting them on resume.
            for row in rows:
                validate_existing(output/'masks'/row['variant']/relative.with_suffix('.png'),tuple(row['size']))
                rule_name=row['variant'].rsplit('__',1)[0]
                for candidate in range(len(row['candidates'])):
                    validate_existing(output/'candidates'/rule_name/relative.parent/
                                      (relative.stem+'__c{}.png'.format(candidate)),tuple(row['size']))
            for section in ('selected','candidates'):
                with Image.open(output/'review'/section/relative.with_suffix('.jpg')) as preview:
                    preview.verify()
        else:
            with Image.open(root/relative) as source:
                source.load()
                image=source.convert('RGB')
            rows=run_image(predictor,image,relative,output,args.precision,rules())
            marker.parent.mkdir(parents=True,exist_ok=True)
            temporary=marker.with_suffix('.tmp.json')
            temporary.write_text(json.dumps(rows,indent=2),encoding='utf-8')
            temporary.replace(marker)
        records.extend(rows)
        print('{}/{} elapsed={:.1f}s {}'.format(index,len(selected),time.time()-start,name),flush=True)
    (output/'results.jsonl').write_text('\n'.join(json.dumps(row) for row in records)+'\n',encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summarize(records),indent=2),encoding='utf-8')
    print('Done. Compare review/selected and review/candidates; lower extreme-area count is NOT segmentation accuracy.')


if __name__ == '__main__':
    main()
