"""A few real CUDA/AMP training steps and masked eval forwards before a long run.

Uses the existing loaders, model, losses, optimizer and NCCL setup. Does not save
training weights or modify train.py. Supports python and torchrun entrypoints.
"""
import argparse
import json
from pathlib import Path
import sys
import math

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from processor.processor import PCA, GPD, PrototypeBank, mask_kwargs
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from solver.amp_step import step_after_unscale
from train import setup_cuda_visible_devices, init_distributed, set_seed


def tensor_summary(value):
    if value is None:
        return {'missing': True}
    data = value.detach().float()
    finite = torch.isfinite(data)
    values = data[finite]
    return dict(missing=False, dtype=str(value.dtype), shape=list(value.shape),
                nan=int(torch.isnan(data).sum().item()),
                inf=int(torch.isinf(data).sum().item()),
                finite=bool(finite.all().item()),
                finite_abs_max=float(values.abs().max().item()) if values.numel() else None)


def loss_gradient_diagnostics(losses, parameters, scale):
    """Probe the same graph, without accumulating .grad or changing parameters.

    Single process only: DDP does not support this autograd.grad diagnostic.
    Unscaled/scaled comparison separates scale-induced overflow from an already
    invalid backward path; it is evidence, not automatic numerical repair.
    """
    output = {}
    for label, loss in losses.items():
        output[label] = {}
        for mode, factor in [('unscaled', 1.), ('scaled', scale)]:
            try:
                gradients = torch.autograd.grad(loss * factor, [p for _,p in parameters],
                                                retain_graph=True, allow_unused=True)
                summaries = {name:tensor_summary(g) for (name,_),g in zip(parameters, gradients)}
                output[label][mode] = dict(
                    bad={name:s for name,s in summaries.items() if s.get('missing') or not s.get('finite')},
                    max_finite_abs=max((s['finite_abs_max'] for s in summaries.values()
                                        if s.get('finite_abs_max') is not None), default=None))
            except RuntimeError as exc:
                output[label][mode] = {'error': str(exc)}
    return output


def save_report(path, report):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', default='configs/UAD.yml')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--max-amp-skips', type=int, default=8,
                        help='Bounded native GradScaler recovery; steps counts successful optimizer updates')
    parser.add_argument('--report', default='logs/qg_preflight.json')
    parser.add_argument('--precision', choices=['amp', 'fp32'], default='amp',
                        help='amp matches train.py; fp32 is an isolation experiment, not an equivalent AMP pass')
    parser.add_argument('--init-scale', type=float, default=65536., help='AMP loss scale; default matches train.py')
    parser.add_argument('--diagnose-loss-gradients', action='store_true',
                        help='Single GPU: probe each loss with/without loss scaling on the same graph')
    parser.add_argument('--detect-anomaly', action='store_true')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.max_amp_skips < 0:
        parser.error('--max-amp-skips cannot be negative')
    if not math.isfinite(args.init_scale) or args.init_scale <= 0:
        parser.error('--init-scale must be finite and positive')
    config = cfg.clone()
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)
    config.freeze()
    if config.SOLVER.RESUME:
        parser.error('Preflight does not resume training; use train.py with SOLVER.RESUME')
    if args.diagnose_loss_gradients and config.MODEL.DIST_TRAIN:
        parser.error('--diagnose-loss-gradients requires MODEL.DIST_TRAIN False')
    if not config.MODEL.QG_SFM.ENABLED:
        raise ValueError('This preflight checks the enabled QG-SFM experiment')
    setup_cuda_visible_devices(config)
    if not torch.cuda.is_available():
        raise RuntimeError('Run real-data preflight in the server clipreid CUDA environment')
    local_rank, rank, world = init_distributed(config, args)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    set_seed(config.SOLVER.SEED)
    records = []
    step = None
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    try:
        loader, _, eval_loaders, _, classes, cameras, views = make_dataloader(config)
        model = make_model(config, classes, cameras, views).to(device)
        base = model
        for block in base.base.blocks:
            block.attn.collect_qg_stats = True
        qg = [(n,p) for n,p in base.named_parameters() if n.endswith(('.qg_alpha_raw', '.qg_beta_raw'))]
        if len(qg) != 24 or any(p.detach().item() != 0 for _,p in qg):
            raise AssertionError('Expected 24 zero-initialized modulation scalars')
        loss_fn, center = make_loss(config, classes)
        optimizer, _ = make_optimizer(config, model, center)
        scheduler = create_scheduler(config, optimizer)
        scheduler.step(1)
        if world > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        bank = PrototypeBank(classes, 768, momentum=config.MODEL.GPD_MOMENTUM, device=device)
        scaler = torch.cuda.amp.GradScaler(enabled=args.precision == 'amp', init_scale=args.init_scale)
        model.train()
        iterator = iter(loader)
        torch.cuda.reset_peak_memory_stats(device)
        successful, skipped = 0, 0
        for step in range(args.steps + args.max_amp_skips):
            try:
                batch = next(iterator)
            except StopIteration:
                raise RuntimeError('Training loader ended before requested preflight steps')
            images, targets = [x.to(device) for x in batch[0]], batch[1].to(device)
            spatial = mask_kwargs(batch, device, training=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.precision == 'amp'):
                score, feature, _ = model(images, **spatial)
                loss, ce, wrt = loss_fn(score, feature, targets.repeat(len(images)))
                pca, ids, centers = PCA(feature, targets, len(images), config.DATALOADER.NUM_INSTANCE)
                gpd = GPD(feature, targets, bank, ids, centers, len(images))
                if 'pca' in config.SOLVER.LOSS_TYPE:
                    loss = loss + config.MODEL.PCA_LOSS_WEIGHT * pca
                if 'gpd' in config.SOLVER.LOSS_TYPE:
                    loss = loss + gpd
            diagnostics = [block.attn.qg_stats.detach().cpu().tolist() for block in base.base.blocks]
            row = dict(step=step+1, status='running', scale=scaler.get_scale(), precision=args.precision,
                       loss=loss.item(), ce=ce.item(), wrt=wrt.item(), pca=pca.item(), gpd=gpd.item(),
                       tuple_batch=len(targets), actual_images=sum(len(x) for x in images),
                       empty_masks=sum(int((~v.bool()).sum()) for v in spatial['valid_flags']),
                       qg_forward_stats=diagnostics,
                       logits=tensor_summary(score), features=tensor_summary(feature))
            records.append(row)
            if not torch.isfinite(loss):
                raise AssertionError('Nonfinite forward loss')
            if args.diagnose_loss_gradients:
                terms = dict(CE=config.MODEL.ID_LOSS_WEIGHT * ce,
                             WRT=config.MODEL.TRIPLET_LOSS_WEIGHT * wrt)
                if 'pca' in config.SOLVER.LOSS_TYPE:
                    terms['ProCA_weighted'] = config.MODEL.PCA_LOSS_WEIGHT * pca
                if 'gpd' in config.SOLVER.LOSS_TYPE:
                    terms['GPD'] = gpd
                terms['TOTAL'] = loss
                row['loss_gradient_diagnostics'] = loss_gradient_diagnostics(
                    terms, qg, scaler.get_scale())
            scaler.scale(loss).backward()
            row['scaled_bad_gradients'] = {
                name:tensor_summary(p.grad) for name,p in base.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()}
            scaler.unscale_(optimizer)
            row['qg_gradients'] = {name:tensor_summary(p.grad) for name,p in qg}
            for name, parameter in qg:
                if parameter.grad is None:
                    raise AssertionError('Missing modulation gradient (graph disconnected): {}'.format(name))
            outcome = step_after_unscale(scaler, optimizer, model.parameters(), distributed=world>1)
            row.update(outcome)
            if outcome['skipped']:
                skipped += 1
                row.update(status='amp_skipped', successful_updates=successful)
                if rank == 0:
                    print('AMP recovery: attempt={}, scale={} -> {}; optimizer/prototype update skipped'.format(
                        step+1, outcome['scale_before'], outcome['scale_after']), flush=True)
                if skipped > args.max_amp_skips:
                    raise FloatingPointError('AMP skip budget exceeded; numerical failure is persistent')
                continue
            successful += 1
            qg_grad_norm = torch.stack([p.grad.float().square() for _,p in qg]).sum().sqrt().item()
            bank.update(ids, centers, ddp=world>1)
            if world > 1:
                state = torch.cat([p.detach().reshape(-1) for _,p in qg] + [bank.prototypes.flatten()])
                states = [torch.empty_like(state) for _ in range(world)]
                dist.all_gather(states, state)
                for other in states[1:]:
                    torch.testing.assert_close(states[0], other, rtol=0, atol=0)
            row.update(status='passed', qg_gradient_norm=qg_grad_norm, successful_updates=successful)
            if rank == 0:
                print(json.dumps(row), flush=True)
            if successful == args.steps:
                break
        if successful != args.steps:
            raise FloatingPointError('AMP recovery did not reach the requested successful optimizer updates')
        # Release the last training graph before validation to measure a useful peak.
        del score, feature, loss, ce, wrt, pca, gpd
        for block in base.base.blocks:
            block.attn.collect_qg_stats = False
        model.eval()
        if world > 1:
            for buffer in base.buffers():
                dist.broadcast(buffer, src=0)
        evaluated = []
        with torch.no_grad():
            for mode, val_loader in enumerate(eval_loaders, 1):
                if (mode-1) % world != rank:
                    continue
                batch = next(iter(val_loader))
                result = base(batch[0].to(device), mode=mode, **mask_kwargs(batch, device))
                if result.shape != (len(batch[0]), 768) or not torch.isfinite(result).all():
                    raise AssertionError('Masked evaluation output is invalid')
                evaluated.append(mode)
        if world > 1:
            dist.barrier()
        if rank == 0:
            report = dict(status='passed', world_size=world, torch_version=str(torch.__version__),
                          precision=args.precision, init_scale=args.init_scale,
                          successful_updates=successful, amp_skips=skipped, attempts=len(records),
                          max_amp_skips=args.max_amp_skips,
                          cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(device),
                          steps=records, eval_modes_rank0=evaluated,
                          peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                          qg_parameters={n:p.detach().item() for n,p in qg},
                          config=config.dump(),
                          scope='real train steps and one eval batch per modality; not complete retrieval evaluation')
            save_report(args.report, report)
            label = 'CUDA/AMP' if args.precision == 'amp' else 'CUDA/FP32 diagnostic (AMP remains unverified)'
            print('PASS: real {} preflight; report {}'.format(label, args.report), flush=True)
    except Exception as exc:
        if records:
            records[-1]['status'] = 'failed'
        path = Path(args.report)
        if world > 1:
            path = path.with_name(path.stem + '.rank{}'.format(rank) + path.suffix)
        save_report(path, dict(status='failed', failed_step=None if step is None else step+1,
                              error_type=type(exc).__name__, error=str(exc),
                              precision=args.precision, init_scale=args.init_scale,
                              torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                              world_size=world, rank=rank, steps=records, config=config.dump()))
        print('FAIL: diagnostic report saved to {}'.format(path), flush=True)
        raise
    finally:
        if config.MODEL.DIST_TRAIN and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
