import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import logging
import os

logger = logging.getLogger(__name__)


def check_weights_loading(model, weights_path, model_name="Model"):
    """
    检查模型权重加载状态
    """
    from safetensors import safe_open
    
    model_keys = set(model.state_dict().keys())
    
    # 加载权重文件的 keys
    safetensors_path = os.path.join(weights_path, "model.safetensors")
    pytorch_path = os.path.join(weights_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        with safe_open(safetensors_path, framework="pt") as f:
            weight_keys = set(f.keys())
        logger.info(f"[{model_name}] 从 safetensors 加载权重 keys")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location="cpu")
        weight_keys = set(state_dict.keys())
        logger.info(f"[{model_name}] 从 pytorch_model.bin 加载权重 keys")
    else:
        logger.warning(f"[{model_name}] 未找到权重文件")
        return {"success": False, "missing_keys": [], "unexpected_keys": []}
    
    missing_keys = model_keys - weight_keys
    unexpected_keys = weight_keys - model_keys
    
    if missing_keys:
        logger.warning(f"[{model_name}] 模型中缺失的 keys: {list(missing_keys)}")
    if unexpected_keys:
        logger.warning(f"[{model_name}] 权重中多余的 keys: {list(unexpected_keys)}")
    
    success = len(missing_keys) == 0 and len(unexpected_keys) == 0
    if success:
        logger.info(f"[{model_name}] 权重加载检查通过，所有 keys 匹配")
    
    return {"success": success, "missing_keys": list(missing_keys), "unexpected_keys": list(unexpected_keys)}


class Transformer(nn.Module):
    def __init__(self, weights_path=None, **kwargs):
        super(Transformer, self).__init__()
        self.weights_path = weights_path

        logger.info(f"[ViTTransformer] Loading from: {weights_path}")
        
        self.backbone = ViTModel.from_pretrained(weights_path)
        
        self.norm = nn.LayerNorm(768)
        
        # 执行权重加载检查
        self.weights_check_result = check_weights_loading(self.backbone, weights_path, "ViTTransformer")
        
        logger.info("[ViTTransformer] Model initialized with interpolate_pos_encoding support")

    def forward(self, x):
        # 使用 interpolate_pos_encoding=True 自动处理不同尺寸图像的位置编码
        outputs = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        features = outputs.last_hidden_state  # [128, 577, 768]
        cls_token = features[:, 0, :]
        x_norm = self.norm(features)

        return x_norm, cls_token


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
