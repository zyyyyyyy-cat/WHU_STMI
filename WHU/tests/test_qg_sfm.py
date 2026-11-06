"""CPU correctness checks; no WHU data, weights, CUDA, or SAM required."""
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import cfg
from datasets.bases import ImageDataset, ImageDatasetTest
from datasets.make_dataloader import train_collate_fn, val_collate_fn, make_dataloader
from datasets.masks import MaskStore, JointMaskTransform
from model.backbones.vit_pytorch import Attention, TransReID
from model.make_model import make_model
from processor.processor import PCA, GPD, PrototypeBank, mask_kwargs
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler

torch.set_num_threads(2)


class QGMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_pretrained_loading_interpolation_and_missing_weight_guard(self):
        source = TransReID(img_size=(48, 48), stride_size=16, embed_dim=24, depth=12,
                           num_heads=3, num_classes=0)
        target = TransReID(img_size=(32, 32), stride_size=16, embed_dim=24, depth=12,
                           num_heads=3, num_classes=0, qg_sfm=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'imagenet.pth'
            state = source.state_dict()
            torch.save(state, path)
            target.load_param(str(path))
            torch.testing.assert_close(target.patch_embed.proj.weight, source.patch_embed.proj.weight)
            torch.testing.assert_close(target.pos_embed[:, :1], source.pos_embed[:, :1])
            self.assertEqual(target.pos_embed.shape, (1, 5, 24))
            self.assertTrue(all(p.item()==0 for n,p in target.named_parameters() if '.qg_' in n))
            state.pop('blocks.0.attn.qkv.weight')
            torch.save(state, path)
            with self.assertRaisesRegex(RuntimeError, 'missing backbone tensors'):
                target.load_param(str(path))

    def test_spatial_mechanism_and_full_foreground_limit(self):
        module = Attention(12, num_heads=3, qg_sfm=True)
        with torch.no_grad():
            module.qg_alpha_raw.fill_(.08)
            module.qg_beta_raw.fill_(.06)
        logits = torch.randn(1, 3, 5, 5)
        p = torch.tensor([[1., 1., 0., 0., 1.]], requires_grad=True)
        valid = torch.ones(1, dtype=torch.bool)
        changed = module.modulate_logits(logits, p, valid)
        # Foreground keys gain probability mass for foreground queries.
        original_prob = (logits * module.scale).softmax(-1)
        changed_prob = (changed * module.scale).softmax(-1)
        foreground = p.detach()[0].bool()
        before = original_prob[:, :, foreground, :][..., foreground].sum(-1)
        after = changed_prob[:, :, foreground, :][..., foreground].sum(-1)
        self.assertTrue((after > before).all())
        # Background query rows retain their own logits, not necessarily their
        # final multi-layer features, which can change through other queries.
        torch.testing.assert_close(changed[:, :, ~foreground], logits[:, :, ~foreground], rtol=0, atol=0)
        changed.sum().backward()
        self.assertIsNone(p.grad)
        alpha = module.qg_bound * torch.tanh(module.qg_alpha_raw.detach()/module.qg_bound)
        full = module.modulate_logits(logits, torch.ones_like(p), valid)
        torch.testing.assert_close((full*module.scale).softmax(-1),
                                   ((1-alpha)*logits*module.scale).softmax(-1))
        self.assertFalse(torch.equal((full*module.scale).softmax(-1), original_prob))

    def test_each_identity_loss_reaches_modulation_parameters(self):
        from loss.wrt_loss import TripletLoss_WRT
        network = TransReID(img_size=(32, 32), stride_size=16, embed_dim=24, depth=12,
                            num_heads=3, num_classes=0, qg_sfm=True).eval()
        images = torch.randn(12, 3, 32, 32)
        feature = network(images, coverage=torch.tensor([[0., .25, .75, 1.]]).repeat(12, 1),
                          valid_flags=torch.ones(12, dtype=torch.bool))
        target = torch.tensor([0, 0, 1, 1])
        pca, ids, centers = PCA(feature, target, 3, group_size=2)
        classifier = torch.nn.Linear(24, 2)
        losses = dict(CE=F.cross_entropy(classifier(feature), target.repeat(3)),
                      WRT=TripletLoss_WRT()(feature, target.repeat(3))[0], ProCA=pca,
                      GPD=GPD(feature, target, PrototypeBank(2, 24, device='cpu'), ids, centers, 3))
        parameters = [p for n,p in network.named_parameters() if '.qg_' in n]
        for name, loss in losses.items():
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), name)
            self.assertTrue(any(g.abs().item()>0 for g in gradients), name)

    def test_formula_and_extrema_gradient(self):
        module = Attention(12, num_heads=3, qg_sfm=True)
        with torch.no_grad():
            module.qg_alpha_raw.fill_(0.07)
            module.qg_beta_raw.fill_(-0.04)
        scores = torch.randn(2, 3, 5, 5, requires_grad=True)
        weights = torch.tensor([[1., 0., .2, .8, 1.], [1., 0., 0., 0., 0.]])
        valid = torch.tensor([True, False])
        output = module.modulate_logits(scores, weights, valid)
        # Independent scalar-loop reference to catch axis and broadcast errors.
        expected = scores.clone()
        alpha = float(.2 * torch.tanh(module.qg_alpha_raw.detach() / .2))
        beta = float(.2 * torch.tanh(module.qg_beta_raw.detach() / .2))
        for b in range(2):
            for h in range(3):
                for i in range(5):
                    for j in range(5):
                        row = scores[b, h, i]
                        expected[b, h, i, j] = row[j] + valid[b] * weights[b, i] * (
                            alpha * weights[b, j] * (row.max() - row[j])
                            - beta * (1 - weights[b, j]) * (row[j] - row.min()))
        torch.testing.assert_close(output, expected)
        gradient = torch.autograd.grad(output.sum(), scores)[0]
        reference = torch.autograd.grad(expected.sum(), scores)[0]
        torch.testing.assert_close(gradient, reference)
        torch.testing.assert_close(output[0, :, 1], scores[0, :, 1], rtol=0, atol=0)
        torch.testing.assert_close(output[1], scores[1], rtol=0, atol=0)

    def test_zero_init_attention_and_empty_backward(self):
        original = Attention(24, num_heads=3, qkv_bias=True).eval()
        improved = Attention(24, num_heads=3, qkv_bias=True, qg_sfm=True).eval()
        improved.load_state_dict(original.state_dict(), strict=False)
        x = torch.randn(2, 5, 24)
        weights = torch.tensor([[1., 0., .25, .75, 1.]]).repeat(2, 1)
        raw = torch.randn(2, 3, 5, 5)
        modulated = improved.modulate_logits(raw, weights, torch.ones(2))
        torch.testing.assert_close(raw, modulated, rtol=0, atol=0)
        torch.testing.assert_close((raw * improved.scale).softmax(-1),
                                   (modulated * improved.scale).softmax(-1), rtol=0, atol=0)
        torch.testing.assert_close(original(x), improved(x, weights, torch.ones(2)), rtol=0, atol=0)
        improved(x, weights, torch.ones(2)).square().sum().backward()
        self.assertGreater(improved.qg_alpha_raw.grad.abs().item(), 0)
        self.assertGreater(improved.qg_beta_raw.grad.abs().item(), 0)
        improved.zero_grad()
        with torch.no_grad():
            improved.qg_alpha_raw.fill_(.1)
            improved.qg_beta_raw.fill_(-.1)
        torch.testing.assert_close(original(x), improved(x, weights, torch.zeros(2)), rtol=0, atol=0)
        improved(x, weights, torch.zeros(2)).sum().backward()
        self.assertEqual(improved.qg_alpha_raw.grad.item(), 0)
        self.assertEqual(improved.qg_beta_raw.grad.item(), 0)

    def test_twelve_layers_and_four_loss_gradients(self):
        plain = TransReID(img_size=(32, 32), patch_size=16, stride_size=16,
                         embed_dim=24, depth=12, num_heads=3, num_classes=0).eval()
        improved = TransReID(img_size=(32, 32), patch_size=16, stride_size=16,
                            embed_dim=24, depth=12, num_heads=3, num_classes=0, qg_sfm=True).eval()
        improved.load_state_dict(plain.state_dict(), strict=False)
        self.assertEqual(sum(p.numel() for n, p in improved.named_parameters() if '.qg_' in n), 24)
        image = torch.randn(12, 3, 32, 32)
        coverage = torch.tensor([[0., .25, .75, 1.]]).repeat(12, 1)
        flags = torch.ones(12, dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(plain(image), improved(image, coverage=coverage, valid_flags=flags), rtol=0, atol=0)
        feature = improved(image, coverage=coverage, valid_flags=flags)
        classifier = torch.nn.Linear(24, 2)
        neck = torch.nn.BatchNorm1d(24)
        target = torch.tensor([0, 0, 1, 1])
        from loss.wrt_loss import TripletLoss_WRT
        ce = F.cross_entropy(classifier(neck(feature)), target.repeat(3))
        wrt = TripletLoss_WRT()(feature, target.repeat(3))[0]
        pca, ids, centers = PCA(feature, target, 3, group_size=2)
        bank = PrototypeBank(2, 24, device='cpu')
        gpd = GPD(feature, target, bank, ids, centers, 3)
        self.assertFalse(centers.requires_grad)
        self.assertFalse(bank.prototypes.requires_grad)
        (ce + wrt + .01 * pca + gpd).backward()
        for name, parameter in improved.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertTrue(torch.isfinite(neck.weight.grad).all())
        self.assertTrue(torch.isfinite(classifier.weight.grad).all())

    def test_ablations_and_low_precision(self):
        module = Attention(12, num_heads=3, qg_sfm=True, qg_mode='soft_sfm')
        with torch.no_grad():
            module.qg_alpha_raw.fill_(.05)
            module.qg_beta_raw.fill_(.05)
        scores = torch.randn(1, 3, 5, 5).to(torch.bfloat16)
        p = torch.tensor([[1., 0., .5, .8, 1.]])
        output = module.modulate_logits(scores, p, torch.ones(1))
        self.assertEqual(output.dtype, scores.dtype)
        self.assertFalse(torch.equal(output[:, :, 1], scores[:, :, 1]))
        output.float().sum().backward()
        self.assertTrue(torch.isfinite(module.qg_alpha_raw.grad))
        module.zero_grad()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            autocast_output = module(torch.randn(1, 5, 12), p, torch.ones(1))
        autocast_output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(module.qg_beta_raw.grad))
        uniform = TransReID(img_size=(32, 32), stride_size=16, embed_dim=12, depth=1,
                           num_heads=3, num_classes=0, qg_sfm=True, qg_uniform=True)
        captured = []
        handle = uniform.blocks[0].register_forward_pre_hook(lambda m, inputs: captured.append(inputs[1]))
        uniform(torch.randn(1, 3, 32, 32), coverage=p[:, 1:], valid_flags=torch.ones(1))
        handle.remove()
        torch.testing.assert_close(captured[0], torch.tensor([[1., .575, .575, .575, .575]]))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image_root = self.root / 'data'
        self.mask_root = self.root / 'masks'
        self.mapping = {}
        for split in ('train', 'query', 'test'):
            for index, modality in enumerate(('RGB', 'IR', 'Thermal')):
                for pid in (1000, 1001):
                    key = '{}/{}/{}_c{}.jpg'.format(split, modality, pid, 1 if split != 'test' else 2)
                    image = self.image_root / key
                    mask = self.mask_root / Path(key).with_suffix('.png')
                    image.parent.mkdir(parents=True, exist_ok=True)
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    array = np.zeros((32, 16), dtype=np.uint8)
                    array[:, index:index + 8] = 255
                    Image.fromarray(np.repeat(array[:, :, None], 3, axis=2)).save(image)
                    Image.fromarray(array).save(mask)
                    self.mapping[key] = mask.relative_to(self.mask_root).as_posix()
        self.manifest = self.root / 'manifest.json'
        self.document = {'version': 1, 'encoding': '0/255', 'provenance': {'generator': 'synthetic-test-v1'},
                         'images': self.mapping}
        self.manifest.write_text(json.dumps(self.document), encoding='utf-8')
        self.store = MaskStore(self.image_root, self.mask_root, self.manifest, '0/255')
        self.cfg = cfg.clone()
        self.cfg.merge_from_file(str(ROOT / 'configs/UAD.yml'))
        self.cfg.MODEL.PRETRAIN_CHOICE = 'none'
        self.cfg.DATASETS.ROOT_DIR = str(self.root)
        self.cfg.DATASETS.DATASET_DIR = 'data'
        self.cfg.DATASETS.MASK_ROOT = str(self.mask_root)
        self.cfg.DATASETS.MASK_MANIFEST = str(self.manifest)
        self.cfg.DATALOADER.NUM_WORKERS = 0
        self.cfg.DATALOADER.NUM_INSTANCE = 1
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.TEST.IMS_PER_BATCH = 2

    def tearDown(self):
        self.temp.cleanup()

    def test_strict_read_and_mapping(self):
        key = next(iter(self.mapping))
        image_path = self.image_root / key
        self.assertEqual(self.store.read(image_path, (16, 32)).size, (16, 32))
        with self.assertRaises(ValueError):
            self.store.read(image_path, (32, 16))
        mask_path = self.store.path_for(image_path)
        for invalid in (np.ones((32, 16), dtype=np.uint8), np.zeros((32, 16, 3), dtype=np.uint8)):
            Image.fromarray(invalid).save(mask_path)
            with self.assertRaises(ValueError):
                self.store.read(image_path, (16, 32))
        mask_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.store.read(image_path, (16, 32))
        with self.assertRaises(KeyError):
            self.store.path_for(self.image_root / 'unmapped.jpg')

    def test_npy_encoding_corruption_and_manifest_audit(self):
        key = next(iter(self.mapping))
        mask_path = self.mask_root / Path(key).with_suffix('.npy')
        array = np.zeros((32, 16), dtype=np.float32)
        array[:, :8] = 1.
        np.save(mask_path, array)
        self.document['encoding'] = '0/1'
        self.document['images'] = {key: mask_path.relative_to(self.mask_root).as_posix()}
        self.manifest.write_text(json.dumps(self.document), encoding='utf-8')
        store = MaskStore(self.image_root, self.mask_root, self.manifest, '0/1')
        self.assertEqual(np.array(store.read(self.image_root / key, (16, 32))).sum(), 256)
        mask_path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            store.read(self.image_root / key, (16, 32))
        provenance = self.root / 'provenance.json'
        provenance.write_text(json.dumps({'generator': 'synthetic-test-v1'}), encoding='utf-8')
        result = subprocess.run([sys.executable, str(ROOT / 'tools/audit_masks.py'),
                                 '--image-root', str(self.image_root), '--mask-root', str(self.mask_root),
                                 '--manifest', str(self.root / 'audited.json'), '--build',
                                 '--provenance', str(provenance), '--report', str(self.root / 'report.json')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.root / 'report.json').read_text())
        self.assertEqual(sum(item['count'] for item in report.values()), 18)

    def test_joint_geometry_and_image_rng_equivalence(self):
        transform = JointMaskTransform(self.cfg, training=True)
        old = T.Compose([T.Resize(self.cfg.INPUT.SIZE_TRAIN, interpolation=3),
                         T.RandomHorizontalFlip(self.cfg.INPUT.PROB), T.Pad(self.cfg.INPUT.PADDING),
                         T.RandomCrop(self.cfg.INPUT.SIZE_TRAIN), T.ToTensor(),
                         T.Normalize(self.cfg.INPUT.PIXEL_MEAN, self.cfg.INPUT.PIXEL_STD), transform.erase])
        key = next(iter(self.mapping))
        with Image.open(self.image_root / key) as source:
            image = source.convert('RGB')
        mask = self.store.read(self.image_root / key, image.size)
        for seed in range(4):
            torch.manual_seed(seed)
            random.seed(seed)
            expected = old(image)
            torch.manual_seed(seed)
            random.seed(seed)
            actual, coverage, valid = transform(image, mask)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(coverage.shape, (128,))
            self.assertTrue(((coverage >= 0) & (coverage <= 1)).all())
            self.assertTrue(valid)
        # Fixed geometry with one exact foreground patch checks flip/crop order.
        self.cfg.INPUT.SIZE_TRAIN = [32, 32]
        self.cfg.INPUT.PADDING = 16
        self.cfg.INPUT.PROB = 1.
        self.cfg.INPUT.RE_PROB = 0.
        self.cfg.INPUT.PIXEL_MEAN = [0., 0., 0.]
        self.cfg.INPUT.PIXEL_STD = [1., 1., 1.]
        array = np.zeros((32, 32), dtype=np.uint8)
        array[:16, :16] = 1
        image = Image.fromarray(np.repeat((array * 255)[:, :, None], 3, axis=2))
        with patch('datasets.masks.T.RandomCrop.get_params', return_value=(16, 16, 32, 32)):
            result, coverage, valid = JointMaskTransform(self.cfg, True)(image, Image.fromarray(array))
        torch.testing.assert_close(coverage, torch.tensor([0., 1., 0., 0.]))
        torch.testing.assert_close(F.avg_pool2d(result[:1], 16).flatten(), coverage)
        with patch('datasets.masks.T.RandomCrop.get_params', return_value=(32, 16, 32, 32)):
            _, cropped_coverage, cropped_valid = JointMaskTransform(self.cfg, True)(image, Image.fromarray(array))
        self.assertFalse(cropped_valid)
        self.assertEqual(cropped_coverage.sum(), 0)
        _, coverage, valid = JointMaskTransform(self.cfg, False)(image, Image.fromarray(array * 0))
        self.assertFalse(valid)
        self.assertEqual(coverage.sum(), 0)

    def test_loader_train_and_all_eval_paths(self):
        train, normal, vals, query_counts, classes, _, _ = make_dataloader(self.cfg)
        batch = next(iter(train))
        self.assertEqual(len(batch), 5)
        self.assertEqual(classes, 2)
        spatial = mask_kwargs(batch, 'cpu', training=True)
        self.assertEqual(len(spatial['coverage']), 3)
        self.assertEqual(spatial['coverage'][0].shape, (2, 128))
        self.assertFalse(torch.equal(spatial['coverage'][0], spatial['coverage'][1]))
        self.assertEqual(query_counts, [2, 2, 2])
        for loader in [normal] + vals:
            batch = next(iter(loader))
            self.assertEqual(len(batch), 8)
            self.assertEqual(mask_kwargs(batch, 'cpu')['coverage'].shape, (2, 128))
        self.cfg.MODEL.QG_SFM.ENABLED = False
        train, _, vals, *_ = make_dataloader(self.cfg)
        self.assertEqual(len(next(iter(train))), 3)
        self.assertEqual(len(next(iter(vals[0]))), 6)

    def test_full_vit_checkpoint_optimizer_and_zero_equivalence(self):
        torch.manual_seed(1234)
        model = make_model(self.cfg, num_class=2, camera_num=7, view_num=3).eval()
        image = torch.randn(1, 3, 256, 128)
        coverage = torch.rand(1, 128)
        flags = torch.ones(1, dtype=torch.bool)
        with torch.no_grad():
            before = model(image, mode=1, coverage=coverage, valid_flags=flags)
            # The baseline follows identical blocks with the QG branch disabled.
            for block in model.base.blocks:
                block.attn.qg_sfm = False
            baseline = model(image, mode=1, coverage=coverage, valid_flags=flags)
            torch.testing.assert_close(before, baseline, rtol=0, atol=0)
            for block in model.base.blocks:
                block.attn.qg_sfm = True
            for name, parameter in model.named_parameters():
                if name.endswith(('.qg_alpha_raw', '.qg_beta_raw')):
                    parameter.fill_(.03)
            expected = model(image, mode=1, coverage=coverage, valid_flags=flags)
        self.assertFalse(torch.equal(before, expected))
        center = torch.nn.Linear(1, 1)
        optimizer, _ = make_optimizer(self.cfg, model, center)
        group = [g for g in optimizer.param_groups if g.get('name') == 'qg_sfm'][0]
        self.assertEqual(len(group['params']), 24)
        self.assertEqual(group['lr'], .008)
        self.assertEqual(group['weight_decay'], 0)
        scheduler = create_scheduler(self.cfg, optimizer)
        for epoch in (1, 5, 40, 120):
            scheduler.step(epoch)
            self.assertEqual(group['lr'], optimizer.param_groups[0]['lr'])
        checkpoint = self.root / 'model.pth'
        torch.save(model.state_dict(), checkpoint)
        with torch.no_grad():
            model.base.blocks[0].attn.qg_alpha_raw.zero_()
        model.load_param(str(checkpoint))
        with torch.no_grad():
            torch.testing.assert_close(expected, model(image, mode=1, coverage=coverage, valid_flags=flags), rtol=0, atol=0)
        state = model.state_dict()
        state.pop('base.blocks.0.attn.qg_alpha_raw')
        torch.save(state, checkpoint)
        with self.assertRaisesRegex(RuntimeError, 'missing modulation'):
            model.load_param(str(checkpoint))
        state = model.state_dict()
        state['_qg_spec'] = torch.zeros(1, dtype=torch.uint8)
        torch.save(state, checkpoint)
        with self.assertRaisesRegex(RuntimeError, 'configuration'):
            model.load_param(str(checkpoint))
        with self.assertRaises(ValueError):
            model(image, mode=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
