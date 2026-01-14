"""
预处理脚本：将长文本通过 CLIP 文本编码器预处理为特征并保存为 npy。
用法示例：
    python tools/precompute_clip_long_text.py \
        --data-root /root/cvpr26/data \
        --split train \
        --model-path /root/clip \
        --batch-size 64 \
        --output /root/cvpr26/data/LLava_train_clip_text.npy
"""

import argparse
import json
import logging
import os
from typing import List

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_long_captions(data_root: str, split: str) -> List[str]:
    jsonl_path = os.path.join(data_root, f"LLava_{split}.jsonl")
    captions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                captions.append(obj["text"])
            except json.JSONDecodeError as e:
                logger.warning("JSON decode error at line %d: %s", len(captions), e)
    if split == "test":
        # 与 RawImageDataset 保持一致：测试集重复 5 次
        captions = [cap for cap in captions for _ in range(5)]
    logger.info("Loaded %d captions from %s", len(captions), jsonl_path)
    return captions


def encode_captions(
    captions: List[str],
    model_path: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    clip_model = CLIPModel.from_pretrained(model_path)
    clip_processor = CLIPProcessor.from_pretrained(model_path)
    clip_model.to(device)
    clip_model.eval()

    features = []
    with torch.no_grad():
        for start in range(0, len(captions), batch_size):
            batch = captions[start : start + batch_size]
            inputs = clip_processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            feats = clip_model.get_text_features(**inputs)
            features.append(feats.cpu())
            logger.info("Encoded %d / %d captions", min(start + batch_size, len(captions)), len(captions))

    return torch.cat(features, dim=0).float().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CLIP text features for long captions.")
    parser.add_argument("--data-root", required=True, help="Root path containing LLava_{split}.jsonl")
    parser.add_argument("--split", choices=["train", "test", "dev", "val"], required=True)
    parser.add_argument("--model-path", default="/root/clip", help="Path to CLIP model/processor")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output",
        default=None,
        help="Output npy path. Default: {data_root}/LLava_{split}_clip_text.npy",
    )
    parser.add_argument("--device", default=None, help="Force device, e.g., cuda:0 or cpu. Default: auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    captions = load_long_captions(args.data_root, args.split)
    feats = encode_captions(captions, args.model_path, args.batch_size, device)

    out_path = args.output or os.path.join(args.data_root, f"LLava_{args.split}_clip_text.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, feats)
    logger.info("Saved features with shape %s to %s", feats.shape, out_path)


if __name__ == "__main__":
    main()

