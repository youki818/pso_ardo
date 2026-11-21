import numpy as np

from common_tools import (ZHQ_OptimizationResult, ZHQ_convergence_factor, ZHQ_levy,
                          ZHQ_midpoint_repair, ZHQ_perturbation, ZHQ_raw_mobility,
                          ZHQ_roles_from_ranks, ZHQ_stable_ranks)
from config import ZH_PARAMS


def ZHQ_ardo(objective, initial, rng, iterations):
    # 初始化种群位置
    x = initial.copy()
    n, d = x.shape
    # 获取搜索上下界和搜索范围
    lower, upper = objective.lower, objective.upper
    span = upper - lower
    # 计算初始种群适应度
    values = objective.population(x)
    # 找到初始全局最优个体
    best = int(np.argmin(values))
    g, gf = x[best].copy(), float(values[best])
    # 初始化先驱者和协调者的比例
    wp, wc = ZH_PARAMS.pioneer_fraction, ZH_PARAMS.coordinator_fraction
    # 保存每一代的历史最优适应度
    curve = np.empty(iterations + 1)
    curve[0] = gf

    for t in range(1, iterations + 1):
        # 根据适应度排名划分先驱者、协调者和执行者
        counts = np.zeros(3, dtype=int)
        order, ranks = ZHQ_stable_ranks(values)
        roles = ZHQ_roles_from_ranks(ranks, wp, wc)

        # 计算迭代进度、扰动强度和种群收敛程度
        fraction = t / iterations
        fr = (1.0 - fraction)**3 * np.exp(-5.0 * fraction)
        crowd = ZHQ_convergence_factor(values, gf)
        previous_best = gf
        for i in range(n):
            role, rank = roles[i], ranks[i]
            # 先驱者负责利用 Levy 飞行进行远距离搜索
            if role == 0:
                candidate = x[i] + ZH_PARAMS.levy_scale * span * ZHQ_levy(rng, d)
            # 协调者负责根据不同个体之间的关系进行协调搜索
            elif role == 1:
                right = x[order[rng.integers(0, rank)]]                 # 选择排名更好的个体
                left = x[order[rng.integers(rank - 1, n)]]              # 选择排名更差的个体
                any_peer = x[order[rng.integers(0, n)]]
                # 生成随机扰动
                noise = ZHQ_perturbation(rng, d, fr, span)
                if rng.random() > crowd:
                    candidate = x[i] + rng.random(d) * (any_peer - x[i]) + noise
                else:
                    candidate = x[i] + rng.random(d) * (right - left) + noise
            # 执行者负责在先驱者和协调者的引导下进行局部搜索
            else:
                first = x[order[rng.integers(0, rank)]]
                second = x[order[rng.integers(0, rank)]]
                noise = ZHQ_perturbation(rng, d, fr, span)
                if rng.random() > crowd:
                    candidate = first + rng.random(d) * (second - x[i]) + noise
                else:
                    candidate = g + rng.random(d) * (second - x[i]) + noise
            # 修复越界位置并评价候选解        
            candidate = ZHQ_midpoint_repair(candidate, x[i], lower, upper)
            value = objective(candidate)
            # 采用更优个体，并更新全局最优解
            if value <= values[i]:
                values[i], x[i] = value, candidate
                if value <= gf:
                    gf, g = value, x[i].copy()
                    counts[role] += 1
        # 根据本轮优化效果调整角色流动性和角色比例
        with np.errstate(divide="ignore", invalid="ignore"):
            forward = np.tanh(np.divide(previous_best - gf, ZH_PARAMS.progress_scale * gf))
        mobility = float(np.fmax(ZH_PARAMS.mobility_floor, (1.0 - crowd) * forward))
        weights = ZHQ_raw_mobility(np.array([wp, wc, 1.0 - wp - wc]), counts, fraction, mobility)
        wp, wc = float(np.fmax(0.0, weights[0])), float(np.fmax(0.0, weights[1]))
        # 保存当前迭代的历史最优适应度
        curve[t] = gf

     # 返回最终最优值、最优位置、收敛曲线和评价次数
    return ZHQ_OptimizationResult(gf, g, curve, objective.count)
