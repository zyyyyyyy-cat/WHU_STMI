"""Strict per-image mask mapping and image/mask joint geometry.

No segmentation model is run during ReID training or evaluation.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from timm.data.random_erasing import RandomErasing


def _relative_file(root, name):
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError('Manifest path escapes its root: {}'.format(name))
    return path


class MaskStore:
    def __init__(self, image_root, mask_root, manifest, encoding):
        if not mask_root or not manifest:
            raise ValueError('QG-SFM requires DATASETS.MASK_ROOT and MASK_MANIFEST')
        if encoding not in ('0/1', '0/255'):
            raise ValueError('MASK_ENCODING must be 0/1 or 0/255')
        self.image_root = Path(image_root).resolve()
        self.mask_root = Path(mask_root).resolve()
        with open(manifest, encoding='utf-8') as stream:
            doc = json.load(stream)
        if doc.get('version') != 1 or doc.get('encoding') != encoding:
            raise ValueError('Mask manifest version/encoding disagrees with configuration')
        self.provenance = doc.get('provenance')
        if not isinstance(self.provenance, dict) or not self.provenance:
            raise ValueError('Manifest must record the fixed mask generator and settings in provenance')
        self.mapping = doc.get('images')
        if not isinstance(self.mapping, dict) or not self.mapping:
            raise ValueError('Manifest images must be a nonempty image->mask mapping')
        self.encoding = encoding
        # A reused RGB mask for IR/Thermal is an error, even if sizes coincide.
        resolved = set()
        for key, value in self.mapping.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError('Manifest paths must be strings')
            image = _relative_file(self.image_root, key)
            if image.relative_to(self.image_root).as_posix() != key:
                raise ValueError('Image keys must be normalized relative POSIX paths: {}'.format(key))
            mask = _relative_file(self.mask_root, value)
            if mask in resolved:
                raise ValueError('Each image needs its own mask file: {}'.format(mask))
            resolved.add(mask)

    def path_for(self, image_path):
        key = Path(image_path).resolve().relative_to(self.image_root).as_posix()
        if key not in self.mapping:
            raise KeyError('No mask mapped for {}'.format(key))
        mask = _relative_file(self.mask_root, self.mapping[key])
        if not mask.is_file():
            raise FileNotFoundError('Mask for {} is missing: {}'.format(key, mask))
        return mask

    def validate_records(self, records):
        for rows in records.values():
            for image_path, _, _, _ in rows:
                if not Path(image_path).is_file():
                    raise FileNotFoundError(image_path)
                self.path_for(image_path)

    def read(self, image_path, image_size):
        path = self.path_for(image_path)
        try:
            if path.suffix.lower() == '.npy':
                array = np.load(path, allow_pickle=False)
            elif path.suffix.lower() == '.png':
                # PIL's global truncated-image option is used by the legacy
                # image reader; verify PNG integrity before decoding masks.
                with Image.open(path) as mask:
                    mask.verify()
                with Image.open(path) as mask:
                    array = np.array(mask)
            else:
                raise ValueError('Only lossless single-channel PNG or 2D NPY masks are supported')
        except Exception as exc:
            raise ValueError('Cannot decode mask {}: {}'.format(path, exc)) from exc
        if array.ndim != 2 or array.shape != (image_size[1], image_size[0]):
            raise ValueError('Mask {} shape {} does not match image size {}'.format(path, array.shape, image_size))
        high = 1 if self.encoding == '0/1' else 255
        if not np.all((array == 0) | (array == high)):
            raise ValueError('Mask {} is not binary with declared encoding {}'.format(path, self.encoding))
        return Image.fromarray((array != 0).astype(np.uint8), mode='L')


class JointMaskTransform:
    def __init__(self, cfg, training):
        self.training = training
        self.size = tuple(cfg.INPUT.SIZE_TRAIN if training else cfg.INPUT.SIZE_TEST)
        if any(d % 16 for d in self.size) or tuple(cfg.MODEL.STRIDE_SIZE) != (16, 16):
            raise ValueError('QG-SFM requires a nonoverlapping 16x16 patch grid')
        self.flip_prob = cfg.INPUT.PROB
        self.padding = cfg.INPUT.PADDING
        # Match the original UAD: train bicubic, test default bilinear.
        self.image_resize = T.Resize(self.size, interpolation=3 if training else 2)
        self.mask_resize = T.Resize(self.size, interpolation=0)
        self.normalize = T.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        self.erase = RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')

    def __call__(self, image, mask):
        image, mask = self.image_resize(image), self.mask_resize(mask)
        if self.training:
            # Exactly one RNG draw per joint operation, as in the image-only path.
            if torch.rand(1) < self.flip_prob:
                image, mask = TF.hflip(image), TF.hflip(mask)
            image = TF.pad(image, self.padding)
            mask = TF.pad(mask, self.padding, fill=0)
            crop = T.RandomCrop.get_params(image, self.size)
            image, mask = TF.crop(image, *crop), TF.crop(mask, *crop)
        mask = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
        coverage = F.avg_pool2d(mask, kernel_size=16, stride=16).flatten()
        valid = mask.sum() > 0  # validity AFTER geometry, before CLS is prepended
        image = self.normalize(TF.to_tensor(image))
        if self.training:
            image = self.erase(image)
        return image, coverage, valid
