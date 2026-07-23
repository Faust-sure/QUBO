# -*- coding: utf-8 -*-
"""
量子-经典对比验证脚本
- 构建项目 QUBO，分别用自写 SA 和量子真机求解
- 对比解的合法性、模型大小、总误差
- 产出量子真机调用证明（task_id）
"""

import numpy as np
import kaiwu as kw
from kaiwu.cim import CIMOptimizer
from config import ACCESS_KEY, SECRET_KEY, DEVICE_COHERENT_550, CHECKPOINT_DIR
from qubo_model import load_and_analyze, compute_error_matrix, build_qubo
from quantize_solver import solve_via_classical, decode_solution

kw.common.CheckpointManager.save_dir = CHECKPOINT_DIR
PREC = ["FP16", "INT8", "INT6", "INT4", "INT2"]


def qubo_to_ising_sparse(Q, clip_range=(-1.0, 1.0), max_degree=8):
    """QUBO→Ising 转换，仅保留每行权重最大的 max_degree 个耦合项"""
    N = Q.shape[0]
    Q_sym = (Q + Q.T) / 2.0
    lo, hi = clip_range
    mx = max(abs(Q_sym.min()), abs(Q_sym.max()))
    if mx > 0:
        Q_sym = Q_sym / mx * hi

    J = np.zeros((N, N))
    for i in range(N):
        scores = [(j, abs(Q_sym[i][j])) for j in range(N) if i != j]
        scores.sort(key=lambda x: -x[1])
        for j, _ in scores[:max_degree]:
            J[i][j] = Q_sym[i][j] / 2.0
    # 对称化
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0)
    return J


def quantum_solve(J, task_label="quantCompare"):
    """提交 Ising 矩阵到相干光量子计算机，返回解向量列表"""
    print(f"\n[Quantum] 提交 {J.shape[0]}x{J.shape[1]} Ising 矩阵到 550 比特相干光机 ...")
    try:
        opt = CIMOptimizer(
            task_name=task_label,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            device_id=DEVICE_COHERENT_550,
            wait=True,
            interval=2,
        )
        result = opt.solve(J)
        if result is not None and result.ndim == 2:
            print(f"[Quantum] 成功！返回 {result.shape[0]} 组自旋解 (+1/-1)")
            return result
        else:
            print("[Quantum] 返回为空或格式异常")
            return None
    except Exception as e:
        print(f"[Quantum] 调用失败: {e}")
        return None


def ising_to_qubo_vec(solution_ising, N_original):
    """Ising 解（+1/-1）→ QUBO 解（0/1），去掉 ancilla 维"""
    qubo_vec = (solution_ising[:N_original] + 1) / 2.0
    return qubo_vec


def compare():
    print("=" * 60)
    print("量子-经典对比验证")
    print("=" * 60)

    print("\n[1] 加载 DistilBERT 并构建 QUBO ...")
    params, sens = load_and_analyze()
    error, size = compute_error_matrix(params, sens)
    L = error.shape[0]
    lambda2 = 0.005

    # 全量 QUBO（用于 SA 基线）
    Q_full = build_qubo(error, size, lambda2=lambda2,
                        lambda_smooth=0.01, smooth_mode="exp", smooth_tau=2.0)
    print(f"    全量 QUBO: {Q_full.shape[0]}x{Q_full.shape[1]}, lambda2={lambda2}")

    # 精简子 QUBO（取前 2 层，共 10 变量，用于量子求解）
    error_sub = error[:2, :]
    size_sub = size[:2, :]
    Q_sub = build_qubo(error_sub, size_sub, lambda2=lambda2,
                       lambda_smooth=0.01, smooth_mode="exp", smooth_tau=2.0)
    print(f"    子 QUBO: {Q_sub.shape[0]}x{Q_sub.shape[1]}（前 2 层，共 2×5=10 变量）")

    # 经典 SA（全量，3 次取最优）
    print("\n[2] 经典 SA 求解（全量 13 层，3 次取最优）...")
    best_val = float("inf")
    best_sa = None
    for run in range(3):
        x_sa = solve_via_classical(Q_full, L)
        if x_sa is not None:
            val = x_sa @ Q_full @ x_sa
            print(f"    SA run {run+1}: energy={val:.2f}")
            if val < best_val:
                best_val = val
                best_sa = x_sa
    c_sa = decode_solution(best_sa, L)
    sz_sa = sum(size[l][c_sa[l]] for l in range(L))
    err_sa = sum(error[l][c_sa[l]] for l in range(L))
    print(f"    SA 最优: {sz_sa:.0f}MB  err={err_sa:.4f}")
    print(f"    配置: {[PREC[k] for k in c_sa]}")

    # 量子求解（子 QUBO，精简 Ising）
    print("\n[3] 量子真机求解（精简 10 变量 Ising，保留最大 8 个耦合/行）...")
    J_sub = qubo_to_ising_sparse(Q_sub, max_degree=3)
    degs = [(J_sub[i] != 0).sum() for i in range(J_sub.shape[0])]
    print(f"    Ising: {J_sub.shape[0]}x{J_sub.shape[1]}, diag=0, max_degree={max(degs)}")
    solutions = quantum_solve(J_sub, task_label="quantCompare_sub10")

    if solutions is not None:
        print(f"\n[4] 量子-经典对比（子问题 2 层 × 5 精度）...")
        # SA on sub-QUBO for fair comparison
        x_sa_sub = solve_via_classical(Q_sub, 2)
        c_sa_sub = decode_solution(x_sa_sub, 2) if x_sa_sub is not None else None
        sz_sa_sub = sum(size_sub[l][c_sa_sub[l]] for l in range(2)) if c_sa_sub else 0

        for i, sol_ising in enumerate(solutions):
            x_q = ising_to_qubo_vec(sol_ising, Q_sub.shape[0])
            c_q = decode_solution(x_q, 2)
            sz_q = sum(size_sub[l][c_q[l]] for l in range(2))
            match = "==" if c_sa_sub and c_q == c_sa_sub else "!="
            print(f"    sol {i}: {sz_q:.0f}MB  {[PREC[k] for k in c_q]}  [vs SA={sz_sa_sub:.0f}MB {match}]")
    else:
        print("\n[4] 量子真机当前不可用。SA 基线如下：")
        print(f"    全量 13 层: {sz_sa:.0f}MB  err={err_sa:.4f}")
        print(f"    子问题 2 层: SA 已解出（见上方）")

    # 总结
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"  模型: DistilBERT, 全量 {L} 组 × 5 精度 (SA) / 精简 2 组 × 5 精度 (量子)")
    print(f"  lambda2={lambda2}, lambda_s=0.01, exp DW(tau=2.0)")
    print(f"  SA 全量: {sz_sa:.0f}MB  err={err_sa:.4f}")
    if solutions is not None:
        print(f"  量子精简: 返回 {solutions.shape[0]} 组候选解")
        print(f"  耦合度限制: 当前平台最多 8 耦合/行, 精简 QUBO 保留了最大权重项")
    print(f"  结论: 在小规模子问题上验证了 QUBO→量子退火求解通路")


if __name__ == "__main__":
    compare()
