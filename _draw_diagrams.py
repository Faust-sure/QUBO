# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ===== 图1: 系统架构流程图 =====
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5)
ax.axis('off')

boxes = [
    (0.5, 2.5, "DistilBERT\n加载模型", "#E3F2FD"),
    (2.5, 2.5, "按13组统计\n敏感度 s_l\n参数量 n_l", "#E8F5E9"),
    (4.5, 2.5, "构建误差矩阵 ∈\n大小矩阵 M\n(5种精度)", "#FFF3E0"),
    (6.5, 3.5, "QUBO矩阵 Q\n(65x65)\nλ?+λ?+λ_s+Φ?", "#FCE4EC"),
    (9.0, 3.5, "SA经典求解\n20000步\n指数降温", "#E3F2FD"),
    (9.0, 1.5, "量子退火\nCIMOptimizer\nIsing转换", "#F3E5F5"),
    (11.5, 3.5, "解码精度\n配置向量", "#E8F5E9"),
    (11.5, 1.5, "解码精度\n配置向量", "#E8F5E9"),
]

for i, (x, y, text, color) in enumerate(boxes):
    rect = FancyBboxPatch((x, y-0.5), 1.8, 1.5, boxstyle="round,pad=0.1", facecolor=color, edgecolor="#333", linewidth=1)
    ax.add_patch(rect)
    ax.text(x+0.9, y+0.15, text, ha='center', va='center', fontsize=7, linespacing=1.3)

# Arrows
arrows = [(2.3,2.5,2.5,2.5), (4.3,2.5,4.5,2.5), (6.3,2.5,6.5,3.2), (8.3,3.5,9.0,3.5), (8.3,2.5,9.0,1.8), (10.8,3.5,11.5,3.5), (10.8,1.5,11.5,1.5)]
for x1,y1,x2,y2 in arrows:
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

# Labels
ax.text(7.75, 4.3, "求解器选择", fontsize=6, ha='center', color='#666')
ax.text(6.5, 1.0, "OUBO构造 (5项叠加)", fontsize=6, ha='center', color='#666')
ax.text(1.5, 4.5, "加载与建模阶段", fontsize=8, ha='center', fontweight='bold', color='#333')
ax.text(9.0, 4.5, "求解阶段", fontsize=8, ha='center', fontweight='bold', color='#333')
ax.text(11.5, 4.5, "输出阶段", fontsize=8, ha='center', fontweight='bold', color='#333')

ax.set_title("System Pipeline: Mixed-Precision Quantization via QUBO + Quantum Annealing", fontsize=10, fontweight='bold', pad=12)
plt.tight_layout()
fig.savefig(r"C:\Users\25841\Desktop\work\quantum_project\output\fig_system_pipeline.png", dpi=200, bbox_inches='tight')
print("Pipeline done")

# ===== 图2: 前向误差放大示意图 =====
fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.5); ax.axis('off')

layers = ["Input", "Emb", "L0\n(Attn)", "L0\n(FFN)", "L1\n(Attn)", "L1\n(FFN)", "...", "L5\n(FFN)", "Output"]
xpos = np.linspace(0.5, 9.5, len(layers))
colors = ['#333'] + ['#E3F2FD']*(len(layers)-2) + ['#333']

# Shallow error amplification
ax.annotate('', xy=(xpos[6], 2.5), xytext=(xpos[1], 2.5),
            arrowprops=dict(arrowstyle='->', color='#E53935', lw=2))
ax.text(4.5, 2.8, "shallow error ∝ ε_1 x ||W_2|| x ... x ||W_L||", fontsize=7, color='#E53935', ha='center')
ax.text(0.2, 2.8, "larger amplification", fontsize=6, color='#E53935')

# Deep error amplification  
ax.annotate('', xy=(xpos[7], 1.2), xytext=(xpos[6], 1.2),
            arrowprops=dict(arrowstyle='->', color='#43A047', lw=2))
ax.text(7.5, 1.5, "deep error ∝ ε_L  (minimal multiplication)", fontsize=7, color='#43A047', ha='center')

# Smoothness weight curve
w_vals = [np.exp(-i/2) for i in range(len(layers)-2)]
for i, (x, w) in enumerate(zip(xpos[1:-1], w_vals)):
    ax.plot(x, 0.2+w*0.8, 'o', color='#FF9800', markersize=w*10+4)
ax.plot(xpos[1:-1], [0.2+wi*0.8 for wi in w_vals], '--', color='#FF9800', alpha=0.5, lw=1)
ax.text(5, 0.1, "w(l) = exp(-l/τ)  (smoothness weight decays with depth)", fontsize=7, ha='center', color='#FF9800')

# Layer boxes
for x, lab, c in zip(xpos, layers, colors):
    rect = FancyBboxPatch((x-0.35, 1.6), 0.7, 0.7, boxstyle="round,pad=0.05", facecolor=c if c != '#333' else '#eee', edgecolor='#555', linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x, 1.95, lab, ha='center', va='center', fontsize=6, linespacing=1.2)

ax.annotate('', xy=(xpos[1], 2.5), xytext=(xpos[0], 2.6), arrowprops=dict(arrowstyle='->', color='#555', lw=1))
ax.annotate('', xy=(xpos[-1], 2.5), xytext=(xpos[-1], 2.6), arrowprops=dict(arrowstyle='->', color='#555', lw=1))
ax.annotate('', xy=(xpos[-1], 2.2), xytext=(xpos[-2], 2.2), arrowprops=dict(arrowstyle='->', color='#555', lw=0.5, alpha=0.5))

ax.set_title("Forward Propagation Error Amplification & Depth-Weighted Smoothness", fontsize=10, fontweight='bold', pad=10)
plt.tight_layout()
fig.savefig(r"C:\Users\25841\Desktop\work\quantum_project\output\fig_forward_error.png", dpi=200, bbox_inches='tight')
print("Error amplification done")

# ===== 图3: QUBO构造示意图 =====
fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 4.5); ax.axis('off')

components = [
    (0.5, 1.0, "① 目标项\n    ε[l][k] = s_l × α_k\n    填入对角线", "#E3F2FD"),
    (3.0, 1.0, "② 硬约束\n    Φ?=λ?Σ(Σx-1)2\n    对角线: -λ?\n    同层交叉: +2λ?", "#E8F5E9"),
    (5.5, 1.0, "③ 软约束\n    Φ?=λ?·size[l][k]\n    对角线: +λ?·size", "#FFF3E0"),
    (8.0, 2.5, "④ 平滑约束\n    Φ?=λ_s·w(l)·|k-k'|\n    跨层交叉项", "#FCE4EC"),
    (10.2, 1.0, "QUBO矩阵\n   Q (65x65)\n   x^T·Q·x", "#EDE7F6"),
]

for x, y, text, color in components:
    w = 2.2 if x < 8 else 2.0
    h = 1.8 if x < 8 else 1.8
    rect = FancyBboxPatch((x, y-0.5), w, h, boxstyle="round,pad=0.1", facecolor=color, edgecolor="#333", linewidth=1)
    ax.add_patch(rect)
    ax.text(x+w/2, y+0.3, text, ha='center', va='center', fontsize=6.5, linespacing=1.3)

ax.annotate('', xy=(10.2,2.3), xytext=(7.7,2.3), arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
ax.annotate('', xy=(7.5,2.3), xytext=(7.5,1.5), arrowprops=dict(arrowstyle='->', color='#555', lw=1))
ax.annotate('', xy=(7.5,1.0), xytext=(7.5,1.5), arrowprops=dict(arrowstyle='<->', color='#999', lw=0.8, ls='--'))
ax.annotate('', xy=(5.2,1.0), xytext=(5.2,1.5), arrowprops=dict(arrowstyle='->', color='#555', lw=1))
ax.annotate('', xy=(2.7,1.0), xytext=(2.7,1.5), arrowprops=dict(arrowstyle='->', color='#555', lw=1))

ax.text(1.5, 3.0, "Step 1: Build error/size matrices", fontsize=7, ha='center', color='#666')
ax.text(4.5, 3.0, "Step 2: Add penalty terms to Q", fontsize=7, ha='center', color='#666')
ax.text(9, 4.2, "Step 3: Solve & Decode", fontsize=7, ha='center', color='#666')

ax.set_title("QUBO Matrix Construction: Five-Term Assembly", fontsize=10, fontweight='bold', pad=12)
plt.tight_layout()
fig.savefig(r"C:\Users\25841\Desktop\work\quantum_project\output\fig_qubo_construction.png", dpi=200, bbox_inches='tight')
print("QUBO construction done")
print("All 3 diagrams saved")

