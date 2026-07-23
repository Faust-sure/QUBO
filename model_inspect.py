# -*- coding: utf-8 -*-
"""
DistilBERT 模型结构分析
- 打印每层的名称、形状、参数量、FP32大小
- 输出对比表，帮助你理解"哪些层大、哪些层小"
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("distilbert-base-uncased")

# 类别：embedding层、每层Transformer（含Attention+FFN）、其他
layers = []
for name, param in model.named_parameters():
    layers.append((name, list(param.shape), param.numel()))

print("=" * 90)
print(f"{'层名':<50s} {'形状':<22s} {'参数量':>10s}  {'FP32大小':>10s}")
print("=" * 90)

total_params = 0
for name, shape, n in layers:
    size_mb = n * 4 / (1024 * 1024)
    total_params += n
    print(f"{name:<50s} {str(shape):<22s} {n:>10,}  {size_mb:>8.2f} MB")

print("=" * 90)
print(f"{'总参数量':<50s} {'':<22s} {total_params:>10,}  {total_params*4/1024/1024:>8.2f} MB")

# ===== 按Transformer层汇总 =====
print("\n\n========== 按 Transformer 层汇总（含Attention + FFN + LayerNorm） ==========")
print(f"{'层':<6s}  {'参数量':>10s}  {'FP32大小':>10s}  {'比重':>8s}")
print("-" * 45)

total_transformer = 0
for i in range(6):  # DistilBERT有6层transformer
    layer_params = 0
    for name, shape, n in layers:
        if f"transformer.layer.{i}." in name:
            layer_params += n
    total_transformer += layer_params
    pct = layer_params / total_params * 100
    print(f"Layer {i:<2d}  {layer_params:>10,}  {layer_params*4/1024/1024:>8.2f} MB  {pct:>7.2f}%")

# Embedding
embed_params = sum(n for name, shape, n in layers if "embedding" in name.lower())
pct = embed_params / total_params * 100
print(f"Embed  {embed_params:>10,}  {embed_params*4/1024/1024:>8.2f} MB  {pct:>7.2f}%")
print("-" * 45)
print(f"Total  {total_params:>10,}  {total_params*4/1024/1024:>8.2f} MB  {100:>7.2f}%")

# ===== 关键发现 =====
print("\n\n========== 关键发现 ==========")
print(f"  6 层 Transformer，每层约 {total_transformer // 6:,} 参数 ({total_transformer*4/1024/1024/6:.2f} MB)")
print(f"  Embedding 层: {embed_params:,} 参数 ({embed_params*4/1024/1024:.2f} MB) — 占了 {(embed_params/total_params)*100:.1f}%")
print(f"  每层结构: Attention(Q/K/V/Out) × 4 + FFN(lin1+lin2) × 2 + LayerNorm × 2")
print(f"  每层的最大矩阵: ffn.lin1/lin2 (3072×768, {2359296:,} 参数) — 是 Attention 的 4 倍")
