import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
import os

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    def __init__(self, weights_path=None, model_name='vit_base_patch16_224', **kwargs):
        super(Transformer, self).__init__()
        self.weights_path = weights_path
        self.model_name = model_name

        logger.info(f"[ViTTransformer-timm] Loading model: {model_name}")
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )
        
        # 从本地加载权重
        if weights_path is not None:
            self._load_local_weights(weights_path)
        else:
            logger.warning("[ViTTransformer-timm] No weights_path provided, using random initialization")
        
        self.embed_dim = self.backbone.embed_dim
        self.norm = nn.LayerNorm(self.embed_dim)
        
        logger.info(f"[ViTTransformer-timm] Model initialized, embed_dim={self.embed_dim}")
        logger.info(f"[ViTTransformer-timm] Number of blocks: {len(self.backbone.blocks)}")
        logger.info(f"[ViTTransformer-timm] dynamic_img_size=True, supports variable input sizes")

    def _load_local_weights(self, weights_path):
        """从本地路径加载权重"""
        # 支持目录或直接指定文件
        if os.path.isdir(weights_path):
            # 尝试常见的权重文件名
            possible_files = [
                'pytorch_model.bin',
                'model.safetensors', 
                'model.bin',
                'checkpoint.pth',
            ]
            weight_file = None
            for fname in possible_files:
                fpath = os.path.join(weights_path, fname)
                if os.path.exists(fpath):
                    weight_file = fpath
                    break
            if weight_file is None:
                raise FileNotFoundError(f"No weight file found in {weights_path}")
        else:
            weight_file = weights_path
        
        logger.info(f"[ViTTransformer-timm] Loading weights from: {weight_file}")
        
        # 加载权重
        if weight_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(weight_file)
        else:
            state_dict = torch.load(weight_file, map_location='cpu')
        
        # 处理可能的 state_dict 包装
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        # 加载权重（strict=False 允许部分匹配）
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"[ViTTransformer-timm] Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"[ViTTransformer-timm] Unexpected keys: {unexpected_keys[:5]}...")
        
        logger.info(f"[ViTTransformer-timm] Weights loaded successfully")

    def forward(self, x):
        
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0, :]  # [B, 768]
        x_norm = self.norm(features)   # [B, num_patches+1, 768]

        return x_norm, cls_token


class VitTransformerFeatureExtractor(nn.Module):
    def __init__(self, backbone_source, weights_path, fixed_blocks, pooling_size=7, 
                 model_name='vit_base_patch16_224'):
        super(VitTransformerFeatureExtractor, self).__init__()
        self.backbone_source = backbone_source
        self.weights_path = weights_path
        self.pooling_size = pooling_size

        self.fixed_blocks = fixed_blocks
        self.backbone = Transformer(weights_path=weights_path, model_name=model_name)

        self._init_modules()

    def _init_modules(self):
        self.unfreeze_base()

    def set_fixed_blocks(self, fixed_blocks):
        self.fixed_blocks = fixed_blocks

    def get_fixed_blocks(self):
        return self.fixed_blocks

    def unfreeze_base(self):
        assert (0 <= self.fixed_blocks < 4)

        backbone_model = self.backbone.backbone
        for param in backbone_model.parameters():
            param.requires_grad = True
        
        # timm ViT 使用 blocks 而不是 encoder.layer
        if hasattr(backbone_model, 'blocks'):
            blocks = backbone_model.blocks
            num_blocks = len(blocks)
            blocks_per_stage = num_blocks // 4
            freeze_blocks = self.fixed_blocks * blocks_per_stage
            for i in range(freeze_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = False
            logger.info(f'[ViTTransformer-timm] Frozen {freeze_blocks}/{num_blocks} blocks')
        else:
            logger.warning('[ViTTransformer-timm] Cannot find blocks in backbone, skipping freeze operation')
                    
        logger.info('[ViTTransformer-timm] backbone now has fixed_blocks={}'.format(self.fixed_blocks))

    def forward(self, images):
        features, cls_token = self.backbone(images)
        return features, cls_token

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.backbone.backbone.apply(set_bn_eval)
