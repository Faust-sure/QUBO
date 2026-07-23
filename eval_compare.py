# -*- coding: utf-8 -*-
"""
最终对比实验：λ_s=0.01（甜点值），对比四种平滑策略
  ① 无平滑
  ② 均匀平滑
  ③ 线性深度加权
  ④ 指数深度加权
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from quantize_solver import (
    load_and_analyze, compute_error_matrix, build_qubo,
    solve_via_classical, decode_solution
)

OUTPUT_DIR = "./output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PRECISION = ["FP16", "INT8", "INT6", "INT4", "INT2"]


def jump_count(choices):
    return sum(1 for i in range(len(choices)-1)
               if abs(choices[i]-choices[i+1]) >= 2)


def run_one(l2, error, size, cfg):
    L = error.shape[0]
    Q = build_qubo(error, size, lambda2=l2, **cfg)
    x = solve_via_classical(Q, L)
    if x is None:
        return None
    c = decode_solution(x, L)
    return (l2, sum(size[l][c[l]] for l in range(L)),
            sum(error[l][c[l]] for l in range(L)), jump_count(c), c)


def experiment():
    print("load model ...")
    params, sens = load_and_analyze()
    error, size = compute_error_matrix(params, sens)
    L = error.shape[0]

    fp16_all = sum(size[l][0] for l in range(L))
    int8_all = sum(size[l][1] for l in range(L))
    int4_all = sum(size[l][2] for l in range(L))
    print(f"baseline: FP16={fp16_all:.0f}  INT8={int8_all:.0f}  INT4={int4_all:.0f}")

    l2_vals = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03, 0.05]

    groups = [
        ("no_smooth",    "no smoothness",             {"lambda_smooth": 0.0,  "smooth_mode": "linear",    "smooth_tau": 2.0}),
        ("uniform",      "uniform smooth",            {"lambda_smooth": 0.01, "smooth_mode": "uniform",   "smooth_tau": 2.0}),
        ("linear_dw",    "depth-weighted (linear)",   {"lambda_smooth": 0.01, "smooth_mode": "linear",    "smooth_tau": 2.0}),
        ("exp_dw",       "depth-weighted (exp)",      {"lambda_smooth": 0.01, "smooth_mode": "exp",       "smooth_tau": 2.0}),
        ("exp_gamma",    "exp (gamma=0.5)",           {"lambda_smooth": 0.01, "smooth_mode": "exp",       "smooth_tau": 2.0, "smooth_gamma": 0.5}),
        ("inverse_dw",   "depth-weighted (inverse)",  {"lambda_smooth": 0.01, "smooth_mode": "inverse",   "smooth_tau": 2.0}),
        ("cosine_dw",    "depth-weighted (cosine)",   {"lambda_smooth": 0.01, "smooth_mode": "cosine",    "smooth_tau": 2.0}),
    ]

    styles = {
        "no_smooth":   ("#333", "s"),
        "uniform":     ("#2196F3", "D"),
        "linear_dw":   ("#4CAF50", "o"),
        "exp_dw":      ("#FF9800", "^"),
        "exp_gamma":   ("#E91E63", "v"),
        "inverse_dw":  ("#9C27B0", "P"),
        "cosine_dw":   ("#00BCD4", "*"),
    }

    all_data = {}
    for key, label, cfg in groups:
        print(f"\n--- {label} ---")
        results = []
        for l2 in l2_vals:
            r = run_one(l2, error, size, cfg)
            if r:
                results.append(r)
                print(f"  l2={l2:.3f}  {r[1]:6.0f}MB  err={r[2]:.4f}  "
                      f"jumps={r[3]}  {[PRECISION[k] for k in r[4]]}")
        all_data[key] = results

    # ── Figure 1: jump count over lambda2 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, cfg in groups:
        if key not in all_data:
            continue
        xs = [r[0] for r in all_data[key]]
        ys = [r[3] for r in all_data[key]]
        ax.plot(xs, ys, marker=styles[key][1], color=styles[key][0],
                linewidth=2, markersize=8, label=label)
    ax.set_xlabel("lambda2 (compression)")
    ax.set_ylabel("jump count (>=2 levels)")
    ax.set_title("Precision smoothness: jump count vs lambda2 (lambda_s=0.01)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "final_jumps.png"), dpi=150)
    print("  -> final_jumps.png")

    # ── Figure 2: Pareto frontier (4 main strategies) ──
    main_keys = ["no_smooth", "uniform", "exp_dw", "exp_gamma"]
    main_labels = {"no_smooth": "no smoothness", "uniform": "uniform smooth",
                   "exp_dw": "depth-weighted (exp)", "exp_gamma": "exp (gamma=0.5)"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for key in main_keys:
        if key not in all_data:
            continue
        sizes = [r[1] for r in all_data[key]]
        errs  = [r[2] for r in all_data[key]]
        ax.plot(sizes, errs, marker=styles[key][1], color=styles[key][0],
                linewidth=2, markersize=6, label=main_labels.get(key, key))

    ax.axvline(x=fp16_all, color="gray", ls="--", alpha=0.3, label=f"FP16={fp16_all:.0f}MB")
    ax.axvline(x=int8_all, color="gray", ls=":", alpha=0.3, label=f"INT8={int8_all:.0f}MB")
    ax.axvline(x=int4_all, color="gray", ls=":", alpha=0.3, label=f"INT4={int4_all:.0f}MB")
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Total error")
    ax.set_title("Pareto frontier: smoothness comparison (lambda_s=0.01)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "final_pareto.png"), dpi=150)
    print("  -> final_pareto.png")

    # ── Figure 2b: grouped bar — model sizes at each lambda2 ──
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_l2s = [0.002, 0.003, 0.005, 0.008, 0.010, 0.012, 0.015, 0.020, 0.030]
    bar_keys = ["no_smooth", "uniform", "exp_dw", "exp_gamma"]
    bar_labels = {"no_smooth": "no smooth", "uniform": "uniform",
                  "exp_dw": "exp DW", "exp_gamma": "exp g=0.5"}
    bar_colors = {"no_smooth": "#333", "uniform": "#2196F3",
                  "exp_dw": "#FF9800", "exp_gamma": "#E91E63"}
    x = np.arange(len(bar_l2s))
    bar_w = 0.18
    for bi, k in enumerate(bar_keys):
        vals = []
        for l2 in bar_l2s:
            v = next((r[1] for r in all_data.get(k, []) if abs(r[0]-l2)<0.0005), None)
            vals.append(v if v is not None else 0)
        ax.bar(x + bi*bar_w, vals, bar_w, label=bar_labels[k], color=bar_colors[k], edgecolor='white')
    ax.set_xticks(x + bar_w*1.5)
    ax.set_xticklabels([str(l) for l in bar_l2s])
    ax.set_xlabel('lambda2 (compression strength)')
    ax.set_ylabel('Model size (MB)')
    ax.set_title('Model size comparison across strategies and lambda2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_size_bar_compare.png"), dpi=150)
    print("  -> fig_size_bar_compare.png")

    # ── Figure 2c: full Pareto (appendix) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, cfg in groups:
        if key not in all_data:
            continue
        sizes = [r[1] for r in all_data[key]]
        errs  = [r[2] for r in all_data[key]]
        ax.plot(sizes, errs, marker=styles[key][1], color=styles[key][0],
                linewidth=2, markersize=5, label=label)
    ax.axvline(x=fp16_all, color="gray", ls="--", alpha=0.3)
    ax.axvline(x=int8_all, color="gray", ls=":", alpha=0.3)
    ax.axvline(x=int4_all, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Total error")
    ax.set_title("Pareto frontier: all 7 strategies (appendix)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "final_pareto_full.png"), dpi=150)
    print("  -> final_pareto_full.png")

    # ── Figure 3: config bar at l2=0.020 (multi-precision visible) ──
    for key_l2, fig_suffix in [(0.005, "005"), (0.020, "020")]:
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, (key, label, cfg) in enumerate(groups):
            if key not in all_data:
                continue
            for r in all_data[key]:
                if abs(r[0] - key_l2) < 0.0005:
                    choices = r[4]
                    colors_bar = ["#66BB6A" if k == 0 else "#FFB74D" if k == 1 else
                                  "#FFD54F" if k == 2 else "#E57373" if k == 3 else "#AB47BC"
                                  for k in choices]
                    ax.barh(idx, [1]*L, left=range(L), color=colors_bar,
                            edgecolor="white", linewidth=1)
                    ax.text(L+0.3, idx, f"{r[1]:.0f}MB", va="center", fontsize=9)
                    break
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([g[1] for g in groups])
        ax.set_xlabel("Layer index (0=Emb, 1-12=Attn/FFN pairs)")
        ax.set_title(f"Precision allocation at lambda2={key_l2}")
        from matplotlib.patches import Patch
        ax.legend([Patch(color="#66BB6A"), Patch(color="#FFB74D"), Patch(color="#FFD54F"),
                   Patch(color="#E57373"), Patch(color="#AB47BC")],
                  ["FP16", "INT8", "INT6", "INT4", "INT2"], loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"final_config_bar_{fig_suffix}.png"), dpi=150)
        print(f"  -> final_config_bar_{fig_suffix}.png")

    print(f"\nDone. Charts in {OUTPUT_DIR}/")


if __name__ == "__main__":
    experiment()
