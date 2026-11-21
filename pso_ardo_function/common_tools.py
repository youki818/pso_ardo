"""
负责提供ARDO 算法目标函数评价、优化结果保存、个体排名、
角色划分、收敛程度计算、Levy 飞行、随机扰动、越界修复
以及角色比例调整。

"""


from dataclasses import dataclass
from math import gamma, pi, sin
from typing import Callable

import numpy as np

from config import ZH_PARAMS


class ZHQ_CountedObjective:

    # 记录目标函数的评价次数
    def __init__(self, function: Callable, lower, upper):
        self.function = function
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.count = 0

    def __call__(self, x: np.ndarray) -> float:
        self.count += 1
        return float(self.function(x))

    def population(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray([self(x) for x in positions], dtype=float)


@dataclass
class ZHQ_OptimizationResult:
    # 保存算法最终结果
    best_value: float
    best_position: np.ndarray
    curve: np.ndarray
    evaluations: int


def ZHQ_linear_schedule(start: float, stop: float, t: int, total: int) -> float:
    # 在线性范围内计算当前迭代参数
    return start + (stop - start) * ((t - 1) / (total - 1) if total > 1 else 0.0)


def ZHQ_stable_ranks(fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 对适应度进行稳定排序并生成排名
    order = np.argsort(fitness, kind="stable")
    ranks = np.empty(len(fitness), dtype=int)
    ranks[order] = np.arange(1, len(fitness) + 1)
    return order, ranks


def ZHQ_roles_from_ranks(ranks, wp: float, wc: float) -> np.ndarray:
    # 根据排名划分先驱者、协调者和执行者
    n = len(ranks)
    return np.where(ranks <= wp * n, 0,
                    np.where(ranks <= (wp + wc) * n, 1, 2))


def ZHQ_convergence_factor(fitness: np.ndarray, best: float) -> float:
    # 根据个体适应度差异计算种群收敛程度
    scale = ZH_PARAMS.similarity_sigma * max(abs(best), ZH_PARAMS.score_floor)
    difference = fitness[:, None] - fitness[None, :]
    with np.errstate(over="ignore", under="ignore"):
        adjacency = np.exp(-(difference**2) / scale**2)
    return float(np.linalg.eigvalsh(adjacency)[-1] / len(fitness))


ZH_BETA = ZH_PARAMS.levy_beta
# 根据 Levy 分布参数计算缩放系数
ZH_LEVY_SIGMA = (gamma(1 + ZH_BETA) * sin(pi * ZH_BETA / 2)
                 / (gamma((1 + ZH_BETA) / 2) * ZH_BETA * 2**((ZH_BETA - 1) / 2)))**(1 / ZH_BETA)


def ZHQ_levy(rng: np.random.Generator, dimension: int) -> np.ndarray:
    u = rng.normal(0.0, ZH_LEVY_SIGMA, dimension)
    v = rng.normal(0.0, 1.0, dimension)
    return u / np.abs(v)**(1.0 / ZH_BETA)


def ZHQ_perturbation(rng, dimension: int, fr: float, span) -> np.ndarray:
    mask = rng.random(dimension) > ZH_PARAMS.perturbation_threshold
    return mask * rng.normal(0.0, fr * span, dimension)


def ZHQ_midpoint_repair(candidate, base, lower, upper) -> np.ndarray:

    # 将越界位置修复到当前个体和边界的中点
    high, low = candidate > upper, candidate < lower
    return np.where(high, (base + upper) / 2.0,
                    np.where(low, (base + lower) / 2.0, candidate))


def ZHQ_raw_mobility(weights, counts, fraction: float, mobility: float) -> np.ndarray:

    # 根据角色贡献情况调整三种角色的比例
    wp, wc, we = weights
    t1, t2 = counts[:2] / (np.sum(counts) + ZH_PARAMS.contribution_epsilon)
    t0, t00 = 0.5 * (1.0 - fraction), 1.0 - fraction
    p12 = mobility * ZH_PARAMS.demotion * max(0.0, 1.0 - t1 - t0)
    p21 = mobility * ZH_PARAMS.promotion * min(1.0, t1 + t0)
    p23 = mobility * ZH_PARAMS.demotion * max(0.0, 1.0 - t1 - t2 - t00)
    p32 = mobility * ZH_PARAMS.promotion * min(1.0, t1 + t2 + t00)
    new_wp = wp - wp * p12 + wc * p21
    new_wc = wc - wc * p21 - wc * p23 + wp * p12 + we * p32

    # 返回先驱者、协调者和执行者的新比例
    return np.array([new_wp, new_wc, 1.0 - new_wp - new_wc])
