# -*- coding: utf-8 -*-
"""
qubo_model.py
输入: 已加载的 DistilBERT 模型
输出: QUBO 矩阵 Q (numpy 2D array) + 变量到层/精度的映射
"""

import torch
import numpy as np
from transformers import AutoModel
import os

# 配置
K = 5          # 精度种数（FP16=0, INT8=1, INT6=2, INT4=3, INT2=4）
λ1 = 10.0      # 罚项系数：每层必选一种精度
λ2 = 1.0       # 罚项系数：总预算（暂不用）
B = 200.0      # 总预算 200 MB（暂不用）

def load_and_analyze():
    """加载模型，按 Embedding + 每层 Attention/FFN 共 13 组独立收集敏感度"""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    model = AutoModel.from_pretrained("distilbert-base-uncased")

    params_per_layer = []
    sensitivity = []

    # --- 组 0：Embedding ---
    embed_params = 0
    embed_std_sum = 0.0
    embed_count = 0
    for name, param in model.named_parameters():
        if "embedding" in name.lower() and "weight" in name:
            embed_params += param.numel()
            embed_std_sum += param.std().item()
            embed_count += 1
    params_per_layer.append(embed_params)
    sensitivity.append(embed_std_sum / embed_count if embed_count > 0 else 0.0)

    # --- 组 1~12：每层 Transformer 拆为 Attention + FFN ---
    for layer_idx in range(6):
        attn_params = 0
        attn_std_sum = 0.0
        attn_count = 0
        ffn_params = 0
        ffn_std_sum = 0.0
        ffn_count = 0

        for name, param in model.named_parameters():
            if f"transformer.layer.{layer_idx}." in name and "weight" in name:
                if "attention" in name.lower():
                    attn_params += param.numel()
                    attn_std_sum += param.std().item()
                    attn_count += 1
                elif "ffn" in name.lower():
                    ffn_params += param.numel()
                    ffn_std_sum += param.std().item()
                    ffn_count += 1

        params_per_layer.append(attn_params)
        sensitivity.append(attn_std_sum / attn_count if attn_count > 0 else 0.0)
        params_per_layer.append(ffn_params)
        sensitivity.append(ffn_std_sum / ffn_count if ffn_count > 0 else 0.0)

    return params_per_layer, sensitivity

def compute_error_matrix(params_per_layer, sensitivity):
    """生成每层每种精度的误差表（K=5: FP16/INT8/INT6/INT4/INT2）"""
    L = len(params_per_layer)
    error = np.zeros((L, K))
    size  = np.zeros((L, K))

    for l in range(L):
        n_params = params_per_layer[l]
        sens = sensitivity[l]

        # FP16 (k=0): 几乎无损
        error[l][0] = sens * 0.01
        size[l][0]  = n_params * 2 / (1024 * 1024)

        # INT8 (k=1): 中等损失
        error[l][1] = sens * 0.5
        size[l][1]  = n_params * 1 / (1024 * 1024)

        # INT6 (k=2): 介于 INT8 与 INT4 之间
        error[l][2] = sens * 1.0
        size[l][2]  = n_params * 0.75 / (1024 * 1024)

        # INT4 (k=3): 较大损失
        error[l][3] = sens * 2.0
        size[l][3]  = n_params * 0.5 / (1024 * 1024)

        # INT2 (k=4): 极限压缩
        error[l][4] = sens * 4.0
        size[l][4]  = n_params * 0.25 / (1024 * 1024)

    return error, size

def build_qubo(error, size, lambda2=0.01,
               lambda_smooth=0.0, smooth_mode="linear", smooth_tau=2.0,
               smooth_gamma=1.0):
    """步骤3: 构建 QUBO 矩阵 Q

    参数:
      error         : L×K, error[l][k] = 第l层用第k种精度的精度损失
      size          : L×K, size[l][k]  = 第l层用第k种精度的大小(MB)
      lambda2       : 空间惩罚权重（越大→越倾向压缩）
      lambda_smooth : 层间平滑约束权重（0=不加, >0=加）
      smooth_mode   : "linear" / "exp" / "gaussian"  深度权重衰减方式
      smooth_tau    : 指数/高斯的衰减速度参数
    """
    L = error.shape[0]
    N = L * K
    Q = np.zeros((N, N))

    # --- 目标项：最小化总误差 ---
    for l in range(L):
        for k in range(K):
            i = l * K + k
            Q[i][i] += error[l][k]

    # --- 罚项1：每层必须且只能选一种精度 ---
    for l in range(L):
        for k in range(K):
            i = l * K + k
            Q[i][i] += λ1 * (1 - 2)
            for j in range(k + 1, K):
                jj = l * K + j
                Q[i][jj] += λ1 * 2
                Q[jj][i] += λ1 * 2

    # --- 罚项2：空间惩罚 ---
    if lambda2 is not None and lambda2 > 0:
        for l in range(L):
            for k in range(K):
                i = l * K + k
                Q[i][i] += lambda2 * size[l][k]

    # --- 罚项3：层间平滑约束（深度加权）---
    if lambda_smooth > 0:
        for l in range(L - 1):         # 第 l 层 与 第 l+1 层
            w = depth_weight(l, L, smooth_mode, smooth_tau, smooth_gamma)
            for k in range(K):
                for kk in range(K):
                    gap = abs(k - kk)
                    if gap > 0:
                        i = l * K + k
                        j = (l + 1) * K + kk
                        penalty = lambda_smooth * w * gap
                        Q[i][j] += penalty
                        Q[j][i] += penalty

    return Q


def depth_weight(l, L, mode="linear", tau=2.0, gamma=1.0):
    """计算第 l 层与第 l+1 层之间的平滑惩罚权重 w(l)

    l=0 是最浅交界(Embedding↔Layer0), l=L-2 是最深交界(Layer4↔Layer5)

    线性:     w = (L - l - 1) / (L - 1)     → 1.0 → 0.17
    指数:     w = exp(-l / tau)              → 1.0 → exp(-5/tau)
    高斯:     w = exp(-l² / (2 * tau²))      → 1.0 → exp(-25/(2τ²))
    反比例:   w = 1 / (1 + l / tau)          → 1.0 → 0.29 (缓慢衰减)
    余弦:     w = cos(π·l / (2·(L-1)))       → 1.0 → 0 (平稳衰减至零)

    gamma: 全局缩放因子 (默认 1.0)
      gamma=0.5 → 所有惩罚减半（允许更多浅层跳变）
      gamma=2.0 → 惩罚加倍（更严格平滑）
    """
    if mode == "linear":
        result = (L - l - 1) / max(L - 1, 1)
    elif mode == "exp":
        result = np.exp(-l / tau)
    elif mode == "gaussian":
        result = np.exp(-l * l / (2.0 * tau * tau))
    elif mode == "inverse":
        result = 1.0 / (1.0 + l / tau)
    elif mode == "cosine":
        result = np.cos(np.pi * l / (2.0 * (L - 1)))
    else:
        result = 1.0  # 均匀（uniform）
    return result * gamma

def var_index(l, k):
    """辅助函数: 层 l、精度 k → QUBO 变量索引"""
    return l * K + k

if __name__ == "__main__":
    params, sens = load_and_analyze()
    error, size = compute_error_matrix(params, sens)
    Q = build_qubo(error, size)
    print(f"QUBO 矩阵大小: {Q.shape}")
    print("Q 矩阵（仅显示前 6×6）:")
    print(Q[:6, :6])

