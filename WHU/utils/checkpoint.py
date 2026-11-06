"""Epoch-boundary recovery for both single-process and DDP training.

Full checkpoints contain Python/NumPy RNG state: load only your own trusted files.
Legacy model-only .pth files remain separate and are deliberately not resumable.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist


FORMAT = 'whu_training_v1'


def _distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _module(model):
    return model.module if hasattr(model, 'module') else model


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return copy.deepcopy(value)


def atomic_torch_save(value, path):
    """A failed/interrupted write leaves the previously completed file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _collect(value):
    if not _distributed():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def _check_errors(error, action):
    errors = _collect(error)
    if any(errors):
        raise RuntimeError('{} failed: {}'.format(action, '; '.join(
            'rank {}: {}'.format(i, e) for i, e in enumerate(errors) if e)))


def _sampler(loader):
    batch_sampler = getattr(loader, 'batch_sampler', None)
    sampler = getattr(batch_sampler, 'sampler', None)
    return sampler if sampler is not None else loader.sampler


def capture_rank_state(model, loader, scaler):
    sampler = _sampler(loader)
    return dict(
        python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        buffers={name: _cpu(buffer) for name, buffer in _module(model).named_buffers()},
        sampler={name: copy.deepcopy(getattr(sampler, name))
                 for name in ('length', '_seed') if hasattr(sampler, name)},
        scaler=_cpu(scaler.state_dict()),
    )


def restore_rank_state(state, model, loader, scaler):
    buffers = dict(_module(model).named_buffers())
    if set(buffers) != set(state['buffers']):
        raise ValueError('Per-rank model buffer names changed')
    with torch.no_grad():
        for name, saved in state['buffers'].items():
            buffers[name].copy_(saved)
    for name, value in state['sampler'].items():
        setattr(_sampler(loader), name, value)
    scaler.load_state_dict(state['scaler'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG state cannot be resumed without CUDA')
        torch.cuda.set_rng_state(state['cuda'])


def _optimizer_layout(model, optimizer):
    names = {id(parameter): name for name, parameter in _module(model).named_parameters()}
    return [[names[id(parameter)] for parameter in group['params']]
            for group in optimizer.param_groups]


def training_signature(cfg, train_loader, val_loaders):
    """Guard protocol/config and ordered image/identity mappings, without reading images.

    Image/mask bytes are not re-audited here; keep the already audited data frozen.
    Absolute data paths may change. Mask manifest content must stay identical.
    """
    def flatten(node, prefix=''):
        result = {}
        for key, value in node.items():
            name = prefix + key
            if hasattr(value, 'items'):
                result.update(flatten(value, name + '.'))
            else:
                result[name] = list(value) if isinstance(value, tuple) else value
        return result
    config = flatten(cfg)
    for name in ('OUTPUT_DIR', 'SOLVER.RESUME', 'SOLVER.CHECKPOINT_PERIOD',
                 'SOLVER.LOG_PERIOD', 'MODEL.DEVICE_ID', 'MODEL.PRETRAIN_PATH',
                 'MODEL.PRETRAIN_CHOICE', 'MODEL.QG_SFM.DIAGNOSTICS',
                 'DATASETS.ROOT_DIR', 'DATASETS.MASK_ROOT', 'DATASETS.MASK_MANIFEST',
                 'TEST.WEIGHT', 'TEST.DETAILED_DOMAIN_EVAL'):
        config.pop(name, None)
    digest = hashlib.sha256()
    if train_loader.persistent_workers or train_loader.generator is not None:
        raise ValueError('Epoch recovery requires nonpersistent workers and the default loader RNG')
    for index, loader in enumerate([train_loader] + list(val_loaders)):
        records = loader.dataset.dataset
        for modality, rows in records.items():
            for path, pid, camera, mode in rows:
                row = [index, modality, Path(path).name, int(pid), int(camera), int(mode)]
                digest.update((json.dumps(row, ensure_ascii=True) + '\n').encode('utf-8'))
    manifest_hash = None
    if cfg.MODEL.QG_SFM.ENABLED:
        manifest_hash = hashlib.sha256(Path(cfg.DATASETS.MASK_MANIFEST).read_bytes()).hexdigest()
    return dict(config=config, records_sha256=digest.hexdigest(), manifest_sha256=manifest_hash,
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda)


def save_training_checkpoint(path, epoch, model, optimizer, optimizer_center,
                             center_criterion, scheduler, scaler, prototype_bank,
                             train_loader, signature, counters):
    """All ranks must call after validation/epoch completion, even though only rank 0 writes."""
    local = None
    error = None
    try:
        local = capture_rank_state(model, train_loader, scaler)
    except Exception as exc:
        error = repr(exc)
    _check_errors(error, 'Capture checkpoint state')
    states = _collect(local)
    error = None
    if not _distributed() or dist.get_rank() == 0:
        try:
            payload = dict(format=FORMAT, epoch=int(epoch), world_size=len(states),
                           signature=signature, counters=dict(counters), rank_states=states,
                           model=_cpu(_module(model).state_dict()),
                           optimizer_layout=_optimizer_layout(model, optimizer),
                           optimizer=_cpu(optimizer.state_dict()),
                           optimizer_center=_cpu(optimizer_center.state_dict()),
                           center_criterion=_cpu(center_criterion.state_dict()),
                           scheduler=_cpu(scheduler.state_dict()),
                           prototype_bank=_cpu(prototype_bank.state_dict()) if prototype_bank is not None else None)
            atomic_torch_save(payload, path)
        except Exception as exc:
            error = repr(exc)
    _check_errors(error, 'Save checkpoint')


def load_training_checkpoint(path, model, optimizer, optimizer_center,
                             center_criterion, scheduler, scaler, prototype_bank,
                             train_loader, signature, max_epochs):
    """Strict recovery: same world size, training config, data ordering and software version."""
    error = None
    checkpoint = None
    try:
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:  # torch 1.12
            checkpoint = torch.load(path, map_location='cpu')
        if checkpoint.get('format') != FORMAT:
            raise ValueError('Not a full training checkpoint; model-only weights cannot resume training')
        world = dist.get_world_size() if _distributed() else 1
        if checkpoint['world_size'] != world:
            raise ValueError('Resume requires the same GPU/process count: saved={}, current={}'.format(checkpoint['world_size'], world))
        if checkpoint['signature'] != signature:
            saved = checkpoint['signature']
            changed = [key for key in sorted(set(saved['config']) | set(signature['config']))
                       if saved['config'].get(key) != signature['config'].get(key)]
            other = [key for key in signature if key != 'config' and saved.get(key) != signature[key]]
            raise ValueError('Resume config/data/software mismatch: {}'.format(changed + other))
        if not 0 < checkpoint['epoch'] <= max_epochs:
            raise ValueError('Invalid completed epoch in checkpoint')
        if len(checkpoint['rank_states']) != world:
            raise ValueError('Missing rank states')
        if (checkpoint['prototype_bank'] is None) != (prototype_bank is None):
            raise ValueError('Prototype bank configuration changed')
        if checkpoint['optimizer_layout'] != _optimizer_layout(model, optimizer):
            raise ValueError('Optimizer parameter group names/order changed')
    except Exception as exc:
        error = repr(exc)
    _check_errors(error, 'Read/validate checkpoint')
    error = None
    try:
        _module(model).load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_center.load_state_dict(checkpoint['optimizer_center'])
        center_criterion.load_state_dict(checkpoint['center_criterion'], strict=True)
        scheduler.load_state_dict(checkpoint['scheduler'])
        if prototype_bank is not None:
            prototype_bank.load_state_dict(checkpoint['prototype_bank'], strict=True)
        rank = dist.get_rank() if _distributed() else 0
        # DDP construction already completed: preserve each rank's own BN buffers.
        restore_rank_state(checkpoint['rank_states'][rank], model, train_loader, scaler)
    except Exception as exc:
        error = repr(exc)
    _check_errors(error, 'Restore checkpoint')
    return checkpoint['epoch'], checkpoint['counters']
