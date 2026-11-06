import torch
import torch.nn as nn
import json
from .backbones.vit_pytorch import vit_base_in, vit_ics_lup

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.qg_enabled = cfg.MODEL.QG_SFM.ENABLED
        if self.qg_enabled:
            if cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_in':
                raise ValueError('The QG-SFM experiment requires the shared vit_base_in backbone')
            if list(cfg.INPUT.SIZE_TRAIN) != [256, 128] or list(cfg.INPUT.SIZE_TEST) != [256, 128]:
                raise ValueError('This experiment fixes train/test size to 256x128')
            with open(cfg.DATASETS.MASK_MANIFEST, encoding='utf-8') as stream:
                manifest = json.load(stream)
            specification = {
                'version': 1, 'bound': cfg.MODEL.QG_SFM.BOUND,
                'mode': cfg.MODEL.QG_SFM.MODE,
                'uniform': cfg.MODEL.QG_SFM.UNIFORM_COVERAGE,
                'size': list(cfg.INPUT.SIZE_TRAIN), 'stride': list(cfg.MODEL.STRIDE_SIZE),
                'mask_encoding': cfg.DATASETS.MASK_ENCODING,
                'mask_provenance': manifest['provenance'],
            }
            payload = json.dumps(specification, sort_keys=True, ensure_ascii=True).encode('utf-8')
            # Portable metadata travels with the usual raw state_dict checkpoint.
            # No absolute mask paths are saved, so the dataset may be relocated.
            self.register_buffer('_qg_spec', torch.tensor(list(payload), dtype=torch.uint8))

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base=factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                        qg_sfm=self.qg_enabled, qg_bound=cfg.MODEL.QG_SFM.BOUND,
                                                        qg_mode=cfg.MODEL.QG_SFM.MODE,
                                                        qg_uniform=cfg.MODEL.QG_SFM.UNIFORM_COVERAGE)

        if pretrain_choice == 'imagenet' and not cfg.SOLVER.RESUME:
            self.base.load_param(model_path)
            print('Loading pretrained model......from {}'.format(model_path))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
    def forward(self, x=None, label=None, camids=None, mode=0, coverage=None, valid_flags=None):
        if mode==0:
            if self.qg_enabled:
                if coverage is None or valid_flags is None or len(coverage) != len(x) or len(valid_flags) != len(x):
                    raise ValueError('Training requires one coverage/validity batch per modality')
                coverage = torch.cat(list(coverage), dim=0)
                valid_flags = torch.cat(list(valid_flags), dim=0)
            x = torch.cat(list(x), dim=0)
            global_feat = self.base(x, coverage=coverage, valid_flags=valid_flags)
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return cls_score, global_feat, feat
        else:
            global_feat = self.base(x, coverage=coverage, valid_flags=valid_flags)
            feat = self.bottleneck(global_feat)
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat


    def load_param(self, trained_path):
        self._load_reid_checkpoint(trained_path, skip_classifier=True)
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        self._load_reid_checkpoint(model_path, skip_classifier=False)
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def _load_reid_checkpoint(self, path, skip_classifier):
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:  # PyTorch 1.12
            checkpoint = torch.load(path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        state = {(key[7:] if key.startswith('module.') else key): value for key, value in state.items()}
        if self.qg_enabled:
            required = {name for name, _ in self.named_parameters()
                        if name.endswith(('.qg_alpha_raw', '.qg_beta_raw'))}
            missing = required - state.keys()
            if missing:
                raise RuntimeError('QG checkpoint is missing modulation parameters: {}'.format(sorted(missing)))
            if '_qg_spec' not in state or not torch.equal(state['_qg_spec'].cpu(), self._qg_spec.cpu()):
                raise RuntimeError('QG checkpoint configuration or mask provenance disagrees with current configuration')
        if skip_classifier:
            state = {key: value for key, value in state.items() if not key.startswith('classifier.')}
        # Validate everything before copying any tensor, avoiding partial loads.
        own = self.state_dict()
        expected = {key for key in own if not (skip_classifier and key.startswith('classifier.'))}
        if set(state) != expected:
            raise RuntimeError('Checkpoint keys mismatch; missing={}, unexpected={}'.format(
                sorted(expected - state.keys()), sorted(state.keys() - expected)))
        for key in expected:
            if own[key].shape != state[key].shape:
                raise RuntimeError('Checkpoint shape mismatch: {}'.format(key))
        self.load_state_dict(state, strict=not skip_classifier)


__factory_T_type = {
    'vit_base_in': vit_base_in,
    'vit_ics_lup': vit_ics_lup,
}

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model
