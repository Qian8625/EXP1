import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import logging
import os

logger = logging.getLogger(__name__)


def resize_pos_embed(pos_embed, new_num_patches, num_prefix_tokens=1):
    """
    对 position_embeddings 进行插值以适配新的图像尺寸
    pos_embed: [1, N, C] 其中 N = num_patches + num_prefix_tokens
    """
    old_num_patches = pos_embed.shape[1] - num_prefix_tokens
    if old_num_patches == new_num_patches:
        return pos_embed
    
    prefix_pos = pos_embed[:, :num_prefix_tokens, :]
    patch_pos = pos_embed[:, num_prefix_tokens:, :]
    
    old_size = int(old_num_patches ** 0.5)
    new_size = int(new_num_patches ** 0.5)
    dim = pos_embed.shape[2]
    
    patch_pos = patch_pos.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bicubic', align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_size * new_size, dim)
    
    new_pos_embed = torch.cat([prefix_pos, patch_pos], dim=1)
    logger.info(f"[Transformer] Resized pos_embed: {pos_embed.shape} -> {new_pos_embed.shape}")
    return new_pos_embed


class Transformer(nn.Module):
    def __init__(self, weights_path=None, **kwargs):
        super(Transformer, self).__init__()
        self.weights_path = weights_path

        logger.info(f"[ViTTransformer] Loading from: {weights_path}")
        
        # 先加载原始配置的模型（224x224）获取原始 pos_embed
        original_model = ViTModel.from_pretrained(weights_path)
        original_pos_embed = original_model.embeddings.position_embeddings.data.clone()
        
        # 创建 384x384 配置的模型
        config = ViTConfig.from_pretrained(weights_path)
        config.image_size = 384
        self.backbone = ViTModel.from_pretrained(
            weights_path,
            config=config,
            ignore_mismatched_sizes=True,
        )
        
        # 手动插值 pos_embed 并替换
        # 384/16 = 24, 24*24 = 576 patches + 1 cls = 577
        new_num_patches = (384 // config.patch_size) ** 2
        new_pos_embed = resize_pos_embed(original_pos_embed, new_num_patches, num_prefix_tokens=1)
        self.backbone.embeddings.position_embeddings.data = new_pos_embed
        
        # 清理临时模型
        del original_model
        
        self.norm = nn.LayerNorm(768)
        logger.info("[ViTTransformer] Model initialized with interpolated pos_embed for 384x384")

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state  # [B, 577, 768]
        x_norm = self.norm(features)
        outs = [features]
        return x_norm, outs


class VitTransformerFeatureExtractor(nn.Module):
    def __init__(self, backbone_source, weights_path, fixed_blocks, pooling_size=7):
        super(VitTransformerFeatureExtractor, self).__init__()
        self.backbone_source = backbone_source
        self.weights_path = weights_path
        self.pooling_size = pooling_size

        self.fixed_blocks = fixed_blocks
        self.backbone = Transformer(weights_path=weights_path)

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
        
        # HuggingFace ViT 使用 encoder.layer 而不是 blocks
        if hasattr(backbone_model, 'encoder') and hasattr(backbone_model.encoder, 'layer'):
            layers = backbone_model.encoder.layer
            num_layers = len(layers)
            layers_per_stage = num_layers // 4
            freeze_layers = self.fixed_blocks * layers_per_stage
            for i in range(freeze_layers):
                for param in layers[i].parameters():
                    param.requires_grad = False
            logger.info(f'[ViTTransformer] Frozen {freeze_layers}/{num_layers} layers')
        else:
            logger.warning('Cannot find encoder.layer in backbone, skipping freeze operation')
                    
        logger.info('vit-transformer backbone now has fixed blocks {}'.format(self.fixed_blocks))

    def forward(self, images):
        features, out_put = self.backbone(images)
        return features, out_put

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.backbone.backbone.apply(set_bn_eval)
