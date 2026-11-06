"""Finish an already-unscaled optimizer step without corrupting overflow grads."""
import torch
import torch.distributed as dist


def step_after_unscale(scaler, optimizer, parameters, max_norm=1., distributed=False):
    """Use GradScaler's native skip/backoff; never clip nonfinite gradients.

    Caller must have called scaler.unscale_(optimizer) exactly once. Prototype
    updates must be conditional on the returned optimizer_step flag.
    """
    parameters = list(parameters)
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        raise RuntimeError('No optimizer gradients were produced')
    finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all()
    if distributed:
        bad_ranks = (~finite).to(dtype=torch.int32)
        dist.all_reduce(bad_ranks, op=dist.ReduceOp.SUM)
        count = int(bad_ranks.item())
        if count not in (0, dist.get_world_size()):
            raise FloatingPointError('Gradient finiteness differs across ranks; refusing divergent optimizer steps')
    scale_before = float(scaler.get_scale())
    if not bool(finite.item()):
        if not scaler.is_enabled():
            raise FloatingPointError('Nonfinite gradients without AMP scaling; cannot recover by scale backoff')
        # unscale_ has already recorded found_inf. Native step must skip SGD,
        # preserving parameters and momentum; update resets the growth tracker.
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if scale_after >= scale_before:
            raise RuntimeError('Nonfinite gradients were not handled by GradScaler backoff')
        return dict(optimizer_step=False, skipped=True, scale_before=scale_before,
                    scale_after=scale_after, gradient_norm_before_clip=None)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite total norm despite finite gradient elements')
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if scale_after < scale_before:
        raise RuntimeError('GradScaler rejected gradients classified as finite')
    return dict(optimizer_step=True, skipped=False, scale_before=scale_before,
                scale_after=scale_after, gradient_norm_before_clip=float(norm.item()))
