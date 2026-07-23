# -*- coding: utf-8 -*-
"""
quantize_solver.py
- 调用 qubo_model 生成 QUBO 矩阵
- 用量子退火（Kaiwu CIMOptimizer）求解
- 用经典模拟退火（Kaiwu SimulatedAnnealingOptimizer）求解
- 对比两种方法的解
"""

import numpy as np
import kaiwu as kw
from kaiwu.cim import CIMOptimizer
from kaiwu.classical import SimulatedAnnealingOptimizer
from config import ACCESS_KEY, SECRET_KEY, DEFAULT_DEVICE, CHECKPOINT_DIR
from qubo_model import load_and_analyze, compute_error_matrix, build_qubo

# ── 参数 ──────────────────────────────────────────────
K = 5                    # 精度种数
PRECISION_NAMES = ["FP16", "INT8", "INT6", "INT4", "INT2"]
kw.common.CheckpointManager.save_dir = CHECKPOINT_DIR


def decode_solution(x_vec, L):
    """将 QUBO 解向量翻译为每层的精度选择

    输入: x_vec，可以是 QUBO 格式 (0/1) 或 Ising 格式 (+1/-1)
           自动检测并转换
    输出: choices = [第0层选k, 第1层选k, ...]  (长度 L)
    """
    # 如果是 Ising 格式 (+1/-1)，转为 QUBO 格式 (0/1)
    if np.min(x_vec) < 0:
        x_vec = (x_vec + 1) / 2  # Ising → QUBO: x = (s+1)/2

    choices = []
    for l in range(L):
        start = l * K
        end = start + K
        block = x_vec[start:end]
        k = int(np.argmax(block))
        choices.append(k)
    return choices


def print_solution(choices, error, size, label):
    """打印一个解的详情"""
    L = len(choices)
    total_error = sum(error[l][choices[l]] for l in range(L))
    total_size = sum(size[l][choices[l]] for l in range(L))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'层':<6s} {'精度':<8s} {'误差':>8s}  {'大小(MB)':>10s}")
    print(f"  {'-'*40}")
    for l in range(L):
        k = choices[l]
        print(f"  Layer{l:<2d}  {PRECISION_NAMES[k]:<8s} {error[l][k]:>8.4f}  {size[l][k]:>10.2f}")
    print(f"  {'-'*40}")
    print(f"  {'总计':<6s} {'':<8s} {total_error:>8.4f}  {total_size:>10.2f}")


def solve_via_quantum(Q, L):
    """用量子退火（相干光真机）求解 QUBO，返回最优解"""
    print("\n[量子退火] 提交 QUBO 任务到相干光量子计算机 ...")

    try:
        optimizer = CIMOptimizer(
            task_name="quantSolv01",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            device_id=DEFAULT_DEVICE,
            wait=True,
            interval=1,
        )
        solutions = optimizer.solve(Q)

        if solutions is not None and len(solutions) > 0:
            return solutions[0]
        print("  [警告] 量子退火未返回任何解")
    except Exception as e:
        print(f"  [提示] 量子真机暂时不可用: {e}")
    return None


def solve_via_classical(Q, L):
    """用自写模拟退火求解 QUBO（处理 0/1 二值变量）"""
    print("\n[经典 SA] 用本地模拟退火求解 ...")
    N = Q.shape[0]
    T_start = 10.0
    T_end = 0.01
    n_iter = 20000

    # 随机初始解，确保每层恰有一个 1
    x = np.zeros(N)
    for l in range(L):
        k = np.random.randint(0, K)
        x[l * K + k] = 1.0

    best_x = x.copy()
    best_val = x @ Q @ x

    for step in range(n_iter):
        T = T_start * (T_end / T_start) ** (step / n_iter)

        # 随机选一层，切换精度
        l = np.random.randint(0, L)
        current_k = int(np.argmax(x[l * K: l * K + K]))
        new_k = (current_k + np.random.randint(1, K)) % K

        old_pos = l * K + current_k
        new_pos = l * K + new_k

        # 计算 delta = 新解成本 - 老解成本
        x_old = x.copy()
        x_old[old_pos] = 0.0
        x_old[new_pos] = 1.0
        delta = (x_old @ Q @ x_old) - (x @ Q @ x)

        if delta < 0 or np.random.random() < np.exp(-delta / T):
            x[old_pos] = 0.0
            x[new_pos] = 1.0

        val = x @ Q @ x
        if val < best_val:
            best_val = val
            best_x = x.copy()

    return best_x


def solve_via_bruteforce(Q, L):
    """穷举最优解（仅限小规模问题，验证用）"""
    print("\n[穷举] 枚举所有组合找最优解 ...")
    N = L * K
    best_x = None
    best_val = float("inf")

    # QUBO: f(x) = x^T Q x, x ∈ {0,1}^N
    for i in range(2**N):
        x = np.array([(i >> j) & 1 for j in range(N)], dtype=float)
        val = x @ Q @ x
        if val < best_val:
            best_val = val
            best_x = x.copy()

    return best_x


def compare_all(params_per_layer, sensitivity,
                lambda2=0.01, lambda_smooth=0.0, smooth_mode="linear", smooth_tau=2.0):
    """完整对比流程"""
    error, size = compute_error_matrix(params_per_layer, sensitivity)
    Q = build_qubo(error, size, lambda2=lambda2,
                   lambda_smooth=lambda_smooth,
                   smooth_mode=smooth_mode, smooth_tau=smooth_tau)
    L = len(params_per_layer)
    N = L * K

    print(f"\nQUBO: {N}x{N}  lambda2={lambda2}  smooth={lambda_smooth}({smooth_mode})  tau={smooth_tau}")
    print(f"FP16={sum(size[l][0] for l in range(L)):.0f}MB  "
          f"INT8={sum(size[l][1] for l in range(L)):.0f}MB  "
          f"INT4={sum(size[l][2] for l in range(L)):.0f}MB")

    x_sa = solve_via_classical(Q, L)
    x_quantum = solve_via_quantum(Q, L)

    result = {"sizes": None, "err": None, "choices": None}
    if x_sa is not None:
        choices = decode_solution(x_sa, L)
        total_err = sum(error[l][choices[l]] for l in range(L))
        total_sz = sum(size[l][choices[l]] for l in range(L))
        result = {"sizes": total_sz, "err": total_err, "choices": choices}
        print_solution(choices, error, size,
                       f"SA (lambda2={lambda2}, smooth={lambda_smooth})")
        print(f"  → {jump_count(choices)} 次精度跳变(>=2档)")

    if x_quantum is not None:
        choices_q = decode_solution(x_quantum, L)
        print_solution(choices_q, error, size, "量子退火")

    return error, size, result


def jump_count(choices):
    """统计精度跳变次数：相邻层精度档位差 >= 2"""
    return sum(1 for i in range(len(choices) - 1)
               if abs(choices[i] - choices[i + 1]) >= 2)


if __name__ == "__main__":
    params, sens = load_and_analyze()
    compare_all(params, sens)
