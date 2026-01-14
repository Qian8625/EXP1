import torch
import os
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.layers import trunc_normal_  # 修复导入路径
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViTWithIntermediateLayers(VisionTransformer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        local_feature = x[:, 1:, :]  # 去除cls token，获取patch特征
        global_feature = x[:, 0:1, :]  # 提取cls token作为全局特征

        return x, global_feature

# vit_base_224 14*14 patch
def vit_base_patch16_224(**kwargs):
    model = ViTWithIntermediateLayers(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, **kwargs)
    return model

# vit_base_384 24*24 patch
def vit_base_patch16_384(**kwargs):
    model = ViTWithIntermediateLayers(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, **kwargs)
    return model


class VitTransformerFeatureExtractor(nn.Module):

    def __init__(self, backbone_source, weights_path, fixed_blocks=0, img_size=224, style='224', pth_style='timm'):
        super(VitTransformerFeatureExtractor, self).__init__()
        self.backbone_source = backbone_source
        self.weights_path = weights_path
        self.style = style
        self.pth_style = pth_style

        if self.style == '224':
            self.backbone = vit_base_patch16_224(img_size=img_size, num_classes=0)
        elif self.style == '384':
            self.backbone = vit_base_patch16_384(img_size=img_size, num_classes=0)
        else:
            raise NotImplementedError(f"Style {self.style} not implemented")
        
        # 初始化模块
        self._init_modules()

    def _init_modules(self):
        """初始化模型权重"""
        if not self.weights_path or not os.path.exists(self.weights_path):
            logger.warning('未指定预训练权重路径或文件不存在。模型将从头开始初始化。')
            self.backbone.apply(self._init_weights)
            return

        logger.info(f"开始从 '{self.weights_path}' (格式: {self.pth_style}) 加载预训练权重...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            checkpoint = torch.load(self.weights_path, map_location=device)
            state_dict = checkpoint.get('model', checkpoint)
            
            # 根据pth_style的值决定是否执行转换
            if self.pth_style == 'hg':
                logger.info("检测到 Hugging Face 权重源，开始转换键名...")
                new_state_dict = OrderedDict()

                for k, v in state_dict.items():
                    if k.startswith('head.') or k.startswith('classifier.'):
                        continue

                    name = k
                    # 转换Hugging Face格式的键名到timm格式
                    name = name.replace('vit.embeddings.patch_embeddings.projection', 'patch_embed.proj')
                    name = name.replace('vit.embeddings.cls_token', 'cls_token')
                    name = name.replace('vit.embeddings.position_embeddings', 'pos_embed')
                    name = name.replace('vit.encoder.layer', 'blocks')
                    name = name.replace('attention.attention', 'attn')
                    name = name.replace('intermediate.dense', 'mlp.fc1')
                    name = name.replace('attention.output.dense', 'attn.proj')
                    name = name.replace('output.dense', 'mlp.fc2')
                    name = name.replace('layernorm_before', 'norm1')
                    name = name.replace('layernorm_after', 'norm2')
                    name = name.replace('vit.layernorm', 'norm')

                    if 'attn.query' in name or 'attn.key' in name or 'attn.value' in name:
                        continue

                    new_state_dict[name] = v

                # 特殊处理qkv权重和偏置
                for i in range(len(self.backbone.blocks)):
                    # 检查是否存在对应的Hugging Face格式键
                    q_key = f'vit.encoder.layer.{i}.attention.attention.query.weight'
                    k_key = f'vit.encoder.layer.{i}.attention.attention.key.weight'
                    v_key = f'vit.encoder.layer.{i}.attention.attention.value.weight'
                    
                    if all(key in state_dict for key in [q_key, k_key, v_key]):
                        q_w = state_dict[q_key]
                        k_w = state_dict[k_key]
                        v_w = state_dict[v_key]
                        new_state_dict[f'blocks.{i}.attn.qkv.weight'] = torch.cat([q_w, k_w, v_w], dim=0)

                        q_b = state_dict[f'vit.encoder.layer.{i}.attention.attention.query.bias']
                        k_b = state_dict[f'vit.encoder.layer.{i}.attention.attention.key.bias']
                        v_b = state_dict[f'vit.encoder.layer.{i}.attention.attention.value.bias']
                        new_state_dict[f'blocks.{i}.attn.qkv.bias'] = torch.cat([q_b, k_b, v_b], dim=0)

                load_state_dict = new_state_dict

            # 默认情况 (pth_style == 'timm')，直接加载
            else:
                logger.info("检测到 timm 或其他原生权重源，直接加载。")
                for k in ['head.weight', 'head.bias']:
                    if k in state_dict:
                        logger.info(f"移除键 '{k}'")
                        del state_dict[k]

                load_state_dict = state_dict

            # 加载权重
            msg = self.backbone.load_state_dict(load_state_dict)

            if msg.missing_keys:
                logger.warning(f"权重加载警告: 存在缺失的键 (Missing keys): {msg.missing_keys}")
            if msg.unexpected_keys:
                logger.warning(f"权重加载警告: 存在意外的键 (Unexpected keys): {msg.unexpected_keys}")

            for k in msg.missing_keys:
                logger.warning(f"缺失权重: {k}")
            for k in msg.unexpected_keys:
                logger.warning(f"意外权重: {k}")
                
            if not msg.missing_keys and not msg.unexpected_keys:
                logger.info("所有权重已成功匹配并加载！")
            else:
                logger.info(f"权重加载完成，详细信息: {msg}")

        except Exception as e:
            logger.error(f"加载预训练权重时出错: {str(e)}")
            logger.warning("模型将从头开始初始化。")
            self.backbone.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, images):
        """
        前向传播
        Args:
            images: 输入图像张量 (B, C, H, W)
        Returns:
            x: 局部特征 (B, N+1, C)
            out_put: 全局特征 (B,1,C)
        """
        x, out_put = self.backbone.forward_features(images)
        # print(x.shape)
        
        return x, out_put

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            # 应用到所有子模块
            self.apply(self._set_bn_eval)
        return self

    def _set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()



