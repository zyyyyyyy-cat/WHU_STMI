import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
from solver.amp_step import step_after_unscale
from utils.checkpoint import (atomic_torch_save, training_signature,
                              save_training_checkpoint, load_training_checkpoint)


def mask_kwargs(batch, device, training=False):
    """Keep the original three/six metadata fields and append spatial inputs."""
    baseline_length = 3 if training else 6
    if len(batch) == baseline_length:
        return {}
    coverage, valid = batch[-2:]
    if training:
        coverage = [item.to(device) for item in coverage]
        valid = [item.to(device) for item in valid]
    else:
        coverage, valid = coverage.to(device), valid.to(device)
    return {'coverage': coverage, 'valid_flags': valid}


def qg_diagnostics(model, enabled, logger=None):
    base = model.module if hasattr(model, 'module') else model
    for index, block in enumerate(base.base.blocks, start=1):
        attn = block.attn
        attn.collect_qg_stats = enabled
        if logger is not None and attn.qg_stats is not None:
            a, b, alpha, beta, ratio = attn.qg_stats.cpu().tolist()
            logger.info('QG layer %02d rank0-local: raw=(%.6f, %.6f), effective=(%.6f, %.6f), R=%.6f',
                        index, a, b, alpha, beta, ratio)
            attn.qg_stats = None


def PCA(feat, target, num_modalities, group_size=4):
    """Progressive Center Alignment"""
    D = feat.shape[1]
    B = target.shape[0]

    feat_by_modality = feat.reshape(num_modalities, B, D)
    centers = []
    for m in range(num_modalities):
        fm = F.normalize(feat_by_modality[m], dim=1)
        cm = fm.reshape(-1, group_size, D).mean(1)
        cm = F.normalize(cm, dim=1)
        centers.append(cm)

    g_cent = F.normalize(torch.stack(centers, dim=0).mean(0), dim=1).detach()
    loss = sum((center - g_cent).pow(2).sum(1) for center in centers).mean() / num_modalities

    ids = target.reshape(-1, group_size)[:, 0].long()
    return loss, ids, g_cent


class PrototypeBank(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.2, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.m = momentum
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim, device=device))

    @torch.no_grad()
    def update(self, ids_local, gcent_local, ddp=True):
        ids_local = ids_local.to(self.prototypes.device).contiguous()
        gcent_local = gcent_local.to(self.prototypes.device).contiguous()

        if ddp and dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            ids_list = [torch.empty_like(ids_local) for _ in range(world)]
            gcent_list = [torch.empty_like(gcent_local) for _ in range(world)]
            dist.all_gather(ids_list, ids_local)
            dist.all_gather(gcent_list, gcent_local)
            ids_all = torch.cat(ids_list, dim=0)
            gc_all = torch.cat(gcent_list, dim=0)
        else:
            ids_all, gc_all = ids_local, gcent_local

        C, _ = self.prototypes.shape
        proto_sum = torch.zeros_like(self.prototypes)
        proto_count = torch.zeros(C, device=self.prototypes.device)

        proto_sum.index_add_(0, ids_all, gc_all)
        proto_count.index_add_(0, ids_all, torch.ones_like(ids_all, dtype=proto_count.dtype))

        mask = proto_count > 0
        new_centers = torch.zeros_like(self.prototypes)
        new_centers[mask] = proto_sum[mask] / proto_count[mask].unsqueeze(1)
        new_centers = F.normalize(new_centers, dim=1)

        old = self.prototypes[mask]
        upd = F.normalize((1 - self.m) * old + self.m * new_centers[mask], dim=1)
        self.prototypes[mask] = upd


def GPD(feat, target, bank: PrototypeBank, ids, g_cent, num_modalities, tau=0.03):
    """Global Prototype Discrimination"""
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat, dim=1)
    with torch.no_grad():
        P = bank.prototypes.clone()
        norms = P.norm(dim=1)
        need = norms[ids] < 1e-6
        if need.any():
            P[ids[need]] = g_cent[need]
        P = F.normalize(P, dim=1)

    logits = (f @ P.t()) / tau
    loss = F.cross_entropy(logits, target_rep)
    return loss


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loaders,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_querys, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = 'cuda'
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger('transreid.train')
    dist_on = (
        cfg.MODEL.DIST_TRAIN
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    rank = dist.get_rank() if dist_on else 0
    is_main = rank == 0
    if is_main:
        logger.info('start training')
    if device:
        model.to(local_rank)
        if dist_on:
            if is_main:
                logger.info('Using {} GPUs for DDP training'.format(dist.get_world_size()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # The ViT backbone retains its ImageNet classification head
                # (base.fc), while ReID training uses the outer classifier.
                # Consequently base.fc never participates in the loss.
                find_unused_parameters=True,
            )
            if local_rank == 0:
                torch.set_num_threads(16)
            else:
                torch.set_num_threads(4)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    meter_ls = [AverageMeter() for _ in range(4)]

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
    )
    scaler = amp.GradScaler()
    model_meta = model.module if hasattr(model, 'module') else model
    feat_dim = getattr(model_meta, 'in_planes', 768)
    num_classes = getattr(model_meta, 'num_classes', 500)
    loss_type = cfg.SOLVER.LOSS_TYPE.lower().strip()
    use_aux_modules = loss_type != 'base'
    proto_bank = None
    if is_main:
        logger.info('Feature dimension: {}; identities: {}'.format(feat_dim, num_classes))
    if use_aux_modules:
        proto_bank = PrototypeBank(
            num_classes=num_classes,
            feat_dim=feat_dim,
            momentum=cfg.MODEL.GPD_MOMENTUM,
            device=device,
        )

    group_size = cfg.DATALOADER.NUM_INSTANCE
    counters = dict(successful_updates=0, amp_skips=0, consecutive_amp_skips=0)
    signature = training_signature(cfg, train_loader, val_loaders)
    completed_epoch = 0
    if cfg.SOLVER.RESUME:
        completed_epoch, counters = load_training_checkpoint(
            cfg.SOLVER.RESUME, model, optimizer, optimizer_center, center_criterion,
            scheduler, scaler, proto_bank, train_loader, signature, epochs)
        if is_main:
            logger.info('Resumed full training state from %s: completed epoch=%s; next epoch=%s; AMP scale=%s',
                        cfg.SOLVER.RESUME, completed_epoch, completed_epoch + 1, scaler.get_scale())
            if completed_epoch == epochs:
                logger.info('All %s epochs are already complete; no additional training requested', epochs)
    consecutive_amp_skips = counters['consecutive_amp_skips']

    for epoch in range(completed_epoch + 1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for m in meter_ls:
            m.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        n_iter = 0
        for n_iter, batch in enumerate(train_loader):
            imgs, vid, camids = batch[:3]
            spatial_inputs = mask_kwargs(batch, device, training=True)
            if cfg.MODEL.QG_SFM.ENABLED:
                qg_diagnostics(model, cfg.MODEL.QG_SFM.DIAGNOSTICS and is_main
                               and (n_iter + 1) % log_period == 0)
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            imgs = [img.to(device) for img in imgs]
            camids = [cam.to(device) for cam in camids]
            target = vid.to(device)

            num_modalities = len(imgs)
            target_rep = target.repeat(num_modalities)

            with amp.autocast(enabled=True):
                cls_score, global_feat, feat = model(imgs, target, camids, **spatial_inputs)

                loss, il, tl = loss_fn(cls_score, global_feat, target_rep)
                meter_ls[0].update(il.item(), target_rep.shape[0])
                meter_ls[1].update(tl.item(), target_rep.shape[0])

                if use_aux_modules:
                    loss_pca, ids, g_cent = PCA(
                        global_feat,
                        target,
                        num_modalities=num_modalities,
                        group_size=group_size,
                    )
                    meter_ls[2].update(loss_pca.item(), max(1, target.shape[0] // group_size))

                    loss_gpd = GPD(
                        global_feat,
                        target,
                        proto_bank,
                        ids,
                        g_cent,
                        num_modalities=num_modalities,
                        tau=0.03,
                    )
                    meter_ls[3].update(loss_gpd.item(), max(1, target.shape[0] // group_size))

                    if 'pca' in loss_type:
                        loss += loss_pca * cfg.MODEL.PCA_LOSS_WEIGHT
                    if 'gpd' in loss_type:
                        loss += loss_gpd

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            outcome = step_after_unscale(scaler, optimizer, model.parameters(), distributed=dist_on)
            if outcome['skipped']:
                counters['amp_skips'] += 1
                consecutive_amp_skips += 1
                if is_main:
                    logger.warning('AMP skipped optimizer/prototype update: epoch=%s iteration=%s scale=%s -> %s',
                                   epoch, n_iter+1, outcome['scale_before'], outcome['scale_after'])
                if consecutive_amp_skips >= 8:
                    raise FloatingPointError('Eight consecutive AMP overflows; stop and inspect numerical stability')
            else:
                counters['successful_updates'] += 1
                consecutive_amp_skips = 0

            if use_aux_modules and outcome['optimizer_step']:
                with torch.no_grad():
                    proto_bank.update(ids, g_cent, ddp=cfg.MODEL.DIST_TRAIN)

            acc = (cls_score.max(1)[1] == target_rep).float().mean()
            loss_meter.update(loss.item(), target.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize(local_rank)

            if is_main and (n_iter + 1) % log_period == 0:
                logger.info(
                    'Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}'.format(
                        epoch,
                        (n_iter + 1),
                        len(train_loader),
                        loss_meter.avg,
                        acc_meter.avg,
                        scheduler._get_lr(epoch)[0],
                    )
                )
                metrics = [f'{m.avg:.3f}' for m in meter_ls]
                metrics_str = ', '.join(metrics)
                logger.info(f'Epoch[{epoch}] {metrics_str}')
                if cfg.MODEL.QG_SFM.ENABLED and cfg.MODEL.QG_SFM.DIAGNOSTICS:
                    qg_diagnostics(model, False, logger)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if dist_on:
            if is_main:
                logger.info(
                    'Epoch {} done. Time per batch: {:.3f}[s] Total: {:.1f}[s]'.format(
                        epoch, time_per_batch, end_time - start_time
                    )
                )
        else:
            logger.info(
                'Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'.format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % checkpoint_period == 0:
            if is_main:
                model_to_save = model.module if hasattr(model, 'module') else model
                atomic_torch_save(
                    model_to_save.state_dict(),
                    os.path.join(
                        cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)
                    ),
                )

        if epoch % eval_period == 0:
            model.eval()

            world_sz = dist.get_world_size() if dist_on else 1
            if dist_on:
                # Validation assigns whole modalities to ranks, so ranks execute
                # different numbers of forward passes (and rank 3 is idle with
                # three modalities). Synchronize rank-0 buffers once and use the
                # unwrapped module to avoid DDP's per-forward buffer collective.
                for buffer in model.module.buffers():
                    dist.broadcast(buffer, src=0)
                dist.barrier()
                eval_model = model.module
            else:
                eval_model = model

            for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
                evaluator.set_query_num(mode, num_query)

                if dist_on and (mode - 1) % world_sz != rank:
                    continue

                for batch in val_loader:
                    img, vid, camid, camidt, modid, _ = batch[:6]
                    with torch.no_grad():
                        img = img.to(device)
                        feat = eval_model(img, mode=mode, **mask_kwargs(batch, device))
                        evaluator.update((feat, vid, camid, modid), mode)

            evaluator.split_all()
            cmc, mAP, *_ = evaluator.compute()

            if cmc is not None:
                logger.info(f'Validation Results - Epoch: {epoch}')
                logger.info(f'mAP: {mAP:.2%}')
                for r in (1, 5, 10):
                    logger.info(f'CMC curve, Rank-{r:<2}: {cmc[r-1]:.2%}')
            torch.cuda.empty_cache()

        # Save AFTER validation: its DataLoader iterators also consume RNG state.
        # Every rank participates, including the idle fourth validation rank.
        counters['consecutive_amp_skips'] = consecutive_amp_skips
        save_training_checkpoint(
            os.path.join(cfg.OUTPUT_DIR or '.', 'checkpoint_last.pth'), epoch,
            model, optimizer, optimizer_center, center_criterion, scheduler,
            scaler, proto_bank, train_loader, signature, counters)
        if is_main:
            logger.info('Full checkpoint saved: epoch=%s, successful_updates=%s, AMP skips=%s, scale=%s',
                        epoch, counters['successful_updates'], counters['amp_skips'], scaler.get_scale())


def do_inference(cfg,
                 model,
                 val_loaders,
                 num_querys):
    device = 'cuda'
    logger = logging.getLogger('transreid.test')
    logger.info('Enter inferencing')

    domain_report_path = None
    if cfg.TEST.DETAILED_DOMAIN_EVAL:
        domain_report_path = os.path.join(
            cfg.OUTPUT_DIR or ".", "domain_protocol_results.csv")

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
        detailed_domain_eval=cfg.TEST.DETAILED_DOMAIN_EVAL,
        domain_report_path=domain_report_path,
    )

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
        evaluator.set_query_num(mode, num_query)

        for batch in val_loader:
            img, vid, camid, camidt, modid, imgpath = batch[:6]
            with torch.no_grad():
                img = img.to(device)

                feat = model(img, mode=mode, **mask_kwargs(batch, device))
                evaluator.update((feat, vid, camid, modid), mode)
                img_path_list.extend(imgpath)

    evaluator.split_all()
    cmc, mAP, *_ = evaluator.compute()

    logger.info('Validation Results ')
    logger.info('mAP: {:.2%}'.format(mAP))
    for r in [1, 5, 10]:
        logger.info('CMC curve, Rank-{:<3}:{:.2%}'.format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
