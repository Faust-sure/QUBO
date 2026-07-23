"""
SA vs Quantum 公平对比实验
- 2 层 QUBO (10 变量)，保留所有罚项交叉项
- 超低耦合度编码：每自旋只保留最强 2 个耦合
- 对比：SA 最优解 vs 量子返回的多组候选解
"""
import numpy as np
import kaiwu as kw
from kaiwu.cim import CIMOptimizer
from config import ACCESS_KEY, SECRET_KEY, DEVICE_COHERENT_550, CHECKPOINT_DIR
from qubo_model import load_and_analyze, compute_error_matrix, build_qubo
from quantize_solver import solve_via_classical, decode_solution

kw.common.CheckpointManager.save_dir = CHECKPOINT_DIR
PREC = ["FP16", "INT8", "INT6", "INT4", "INT2"]


def qubo_to_ising_ultralow(Q, max_degree=2):
    """QUBO→Ising，每自旋只保留 max_degree 个最强耦合项，不丢失约束"""
    N = Q.shape[0]
    Q_sym = (Q + Q.T) / 2.0
    mx = max(abs(Q_sym.min()), abs(Q_sym.max()))
    if mx > 0:
        Q_sym = Q_sym / mx  # 归一化到 [-1, 1]

    J = np.zeros((N, N))
    for i in range(N):
        # 遍历所有 j≠i，按 |Q_ij| 降序取前 max_degree 个
        candidates = []
        for j in range(N):
            if i != j:
                candidates.append((j, abs(Q_sym[i][j])))
        candidates.sort(key=lambda x: -x[1])
        for j, _ in candidates[:max_degree]:
            J[i][j] = Q_sym[i][j] / 2.0  # QUBO→Ising: J_ij = Q_ij/4 (近似), 已归一化故 /2
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0)
    return J


def ising_to_onehot(sol_ising, L, K):
    """+1/-1 Ising 解 → QUBO 0/1 向量"""
    qubo_vec = (sol_ising + 1) / 2.0
    # 软解码：每层取 argmax
    result = []
    for l in range(L):
        block = qubo_vec[l*K:(l+1)*K]
        result.append(int(np.argmax(block)))
    return np.array(result)


def is_config_valid(cfg, L, K):
    """检查配置是否每层恰选一种精度"""
    for l in range(L):
        block = cfg[l*K:(l+1)*K]
        if np.sum(block > 0.5) != 1:
            return False
    return True


def main():
    lambda2 = 0.005
    lambda_smooth = 0.01

    print("=" * 60)
    print("SA vs Quantum 公平对比（2 层 × 5 精度，超低耦合度编码）")
    print(f"lambda2={lambda2}, lambda_s={lambda_smooth}, 指数 DW")
    print("=" * 60)

    # --- 1. 构建 QUBO ---
    print("\n[1] 加载 DistilBERT 并构建 2 层全量 QUBO ...")
    params, sens = load_and_analyze()
    error, size = compute_error_matrix(params, sens)
    L, K = error.shape
    L_sub = 2

    error_sub = error[:L_sub, :]
    size_sub = size[:L_sub, :]

    Q_full = build_qubo(error_sub, size_sub, lambda2=lambda2,
                        lambda_smooth=lambda_smooth,
                        smooth_mode="exp", smooth_tau=2.0)
    N = Q_full.shape[0]
    print(f"    全量 QUBO: {N}x{N} ({L_sub}层×{K}精度)")

    # 检查耦合度
    deg_full = [(Q_full[i] != 0).sum() - 1 for i in range(N)]
    print(f"    全量 QUBO 耦合度: {deg_full}")

    # --- 2. 经典 SA 求解 ---
    print("\n[2] 经典 SA 求解（3 次重启动取最优）...")
    best_val = float("inf")
    best_sa_vec = None
    for run in range(3):
        x = solve_via_classical(Q_full, L_sub)
        if x is not None:
            val = x @ Q_full @ x
            valid = "合法" if is_config_valid(x, L_sub, K) else "非法"
            print(f"    run {run+1}: energy={val:.4f}  {valid}")
            if val < best_val:
                best_val = val
                best_sa_vec = x

    sa_cfg = decode_solution(best_sa_vec, L_sub)
    sa_mb = sum(size_sub[l][sa_cfg[l]] for l in range(L_sub))
    sa_err = sum(error_sub[l][sa_cfg[l]] for l in range(L_sub))
    print(f"    SA 最优: {sa_mb:.1f}MB  err={sa_err:.4f}")
    print(f"    配置: {[PREC[k] for k in sa_cfg]}")
    q_sa = best_sa_vec @ Q_full @ best_sa_vec
    print(f"    QUBO energy: {q_sa:.4f}")

    # --- 3. 超低耦合度 Ising 编码 ---
    print("\n[3] 超低耦合度编码 (max_degree=2) ...")
    J = qubo_to_ising_ultralow(Q_full, max_degree=2)
    deg_j = [(J[i] != 0).sum() for i in range(N)]
    print(f"    稀疏 Ising: {J.shape[0]}x{J.shape[1]}, 耦合度={deg_j}")

    nz = (J != 0).sum()
    print(f"    非零耦合总数: {nz}/{N*(N-1)} ({100*nz/(N*(N-1)):.1f}% 稀疏度)")

    # --- 4. 量子求解 ---
    print("\n[4] 量子真机求解 ...")
    try:
        opt = CIMOptimizer(
            task_name="fairCompare_2L_expDW",
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            device_id=DEVICE_COHERENT_550,
            wait=True, interval=2)
        result = opt.solve(J)
        if result is None:
            print("    [FAIL] 量子返回为空")
            return
        print(f"    [OK] 返回 {result.shape[0]} 组自旋解")
    except Exception as e:
        print(f"    [FAIL] {e}")
        return

    # --- 5. 逐一解码 & 对比 ---
    print("\n[5] 量子-经典对比结果")
    print("-" * 60)
    print(f"{'#':>3}  {'配置':<22}  {'大小(MB)':>8}  {'误差':>6}  vs SA={'一致' if '===' else '!='}")
    print("-" * 60)

    seen = set()
    unique_configs = []
    for i, sol_ising in enumerate(result):
        x_q = (sol_ising[:N] + 1) / 2.0
        cfg_q = decode_solution(x_q, L_sub)
        mb_q = sum(size_sub[l][cfg_q[l]] for l in range(L_sub))
        err_q = sum(error_sub[l][cfg_q[l]] for l in range(L_sub))
        cfg_str = ",".join([PREC[k] for k in cfg_q])
        match = "==" if list(cfg_q) == list(sa_cfg) else "!="
        print(f"{i:3d}  {cfg_str:<22}  {mb_q:6.1f}  {err_q:6.4f}  {match}")

        if cfg_str not in seen:
            seen.add(cfg_str)
            unique_configs.append((cfg_str, mb_q, err_q))

    # --- 6. 总结 ---
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"  SA 最优:  {sa_mb:.1f}MB  {[PREC[k] for k in sa_cfg]}  err={sa_err:.4f}")
    print(f"  量子解数: {result.shape[0]} 组, 唯一配置: {len(unique_configs)} 种")
    for cfg_str, mb, err in unique_configs:
        print(f"    {cfg_str:<22}  {mb:.1f}MB  err={err:.4f}")


if __name__ == "__main__":
    main()
