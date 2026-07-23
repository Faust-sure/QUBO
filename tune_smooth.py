# -*- coding: utf-8 -*-
"""
lambda_smooth 参数调优：在不同 λ_s 下扫描，寻找能压制极端跳变
但不消除中间混合解的甜点值。
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

PRECISION_NAMES = ["FP16", "INT8", "INT6", "INT4", "INT2"]


def count_jumps(choices, threshold=2):
    """统计跳变次数：相邻层精度差 >= threshold 的层对数"""
    return sum(1 for i in range(len(choices) - 1)
               if abs(choices[i] - choices[i + 1]) >= threshold)


def sweep():
    print("加载模型 ...")
    params, sens = load_and_analyze()
    error, size = compute_error_matrix(params, sens)
    L = error.shape[0]

    # 固定用均匀平滑（w(l)=1），只改变 lambda_smooth 的大小
    lambda_s_vals = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    l2_vals = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03, 0.05]

    print(f"{'l2':>6s}  {'ls=0':>6s} {'ls=.001':>6s} {'ls=.005':>6s} {'ls=.01':>6s} "
          f"{'ls=.02':>6s} {'ls=.05':>6s} {'ls=.1':>6s}  {'config'}")

    for l2 in l2_vals:
        best_config = None
        line = f"{l2:6.3f}"
        for ls in lambda_s_vals:
            Q = build_qubo(error, size, lambda2=l2,
                           lambda_smooth=ls, smooth_mode="uniform")
            x = solve_via_classical(Q, L)
            if x is not None:
                choices = decode_solution(x, L)
                sz = sum(size[l][choices[l]] for l in range(L))
                jc = count_jumps(choices, threshold=1)  # >=1档就算跳变
                line += f" {sz:6.0f}"
                if best_config is None:
                    best_config = (choices, sz, jc)
            else:
                line += f" {'--':>6s}"

        if best_config:
            choices, sz, jc = best_config
            line += f"  | {sz:.0f}MB jumps={jc} {[PRECISION_NAMES[c] for c in choices]}"
        print(line)


if __name__ == "__main__":
    sweep()
