
from dataclasses import dataclass
from math import gamma, pi, sin
from typing import Callable

import numpy as np

from config import PARAMS


class CountedObjective:

    def __init__(self, function: Callable, lower, upper, budget: int):
        self.function = function
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.budget = budget
        self.count = 0

    def __call__(self, x: np.ndarray) -> float:
        assert self.count < self.budget, "已超过函数评价预算"
        assert x.shape == self.lower.shape, "目标函数每次只接受一个决策向量"
        assert np.all(np.isfinite(x)), "候选解包含非有限值"
        assert np.all((x >= self.lower) & (x <= self.upper)), "候选解超出搜索范围"
        self.count += 1
        value = float(self.function(x))
        assert np.isfinite(value), "目标函数返回了非有限值"
        return value

    def population(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray([self(x) for x in positions], dtype=float)

    def assert_generation(self, population: int, generation: int) -> None:
        assert self.count == population * (generation + 1), (
            f"第 {generation} 代的评价次数应为 {population * (generation + 1)}，"
            f"实际为 {self.count}")



DIAGNOSTIC_COLUMNS = (
    "f_pc", "w_p", "w_c", "w_e", "n_p", "n_c", "n_e",
    "role_p", "role_c", "role_e", "rho", "pso_samples", "pso_hits", "gf",
)


@dataclass
class OptimizationResult:
    best_value: float
    best_position: np.ndarray
    curve: np.ndarray               
    evaluations: int
    diagnostics: dict[str, np.ndarray] | None = None
    current: np.ndarray | None = None
    personal_best: np.ndarray | None = None
    velocities: np.ndarray | None = None


def linear_schedule(start: float, stop: float, t: int, total: int) -> float:

    return start + (stop - start) * ((t - 1) / (total - 1) if total > 1 else 0.0)


def stable_ranks(fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(fitness, kind="stable")
    ranks = np.empty(len(fitness), dtype=int)
    ranks[order] = np.arange(1, len(fitness) + 1)
    return order, ranks


def roles_from_ranks(ranks, wp: float, wc: float) -> np.ndarray:
    n = len(ranks)
    return np.where(ranks <= wp * n, 0,
                    np.where(ranks <= (wp + wc) * n, 1, 2))


def convergence_factor(fitness: np.ndarray, best: float) -> float:
    scale = PARAMS.similarity_sigma * max(abs(best), PARAMS.score_floor)
    difference = fitness[:, None] - fitness[None, :]
    with np.errstate(over="ignore", under="ignore"):
        adjacency = np.exp(-(difference**2) / scale**2)
    return float(np.linalg.eigvalsh(adjacency)[-1] / len(fitness))


_BETA = PARAMS.levy_beta
_LEVY_SIGMA = (gamma(1 + _BETA) * sin(pi * _BETA / 2)
               / (gamma((1 + _BETA) / 2) * _BETA * 2**((_BETA - 1) / 2)))**(1 / _BETA)


def levy(rng: np.random.Generator, dimension: int) -> np.ndarray:
    u = rng.normal(0.0, _LEVY_SIGMA, dimension)
    v = rng.normal(0.0, 1.0, dimension)
    return u / np.abs(v)**(1.0 / _BETA)


def perturbation(rng, dimension: int, fr: float, span) -> np.ndarray:
    mask = rng.random(dimension) > PARAMS.perturbation_threshold
    return mask * rng.normal(0.0, fr * span, dimension)


def midpoint_repair(candidate, base, lower, upper) -> tuple[np.ndarray, np.ndarray]:
    high, low = candidate > upper, candidate < lower
    repaired = np.where(high, (base + upper) / 2.0,
                        np.where(low, (base + lower) / 2.0, candidate))
    return repaired, high | low


def raw_mobility(weights, counts, fraction: float, mobility: float) -> np.ndarray:

    wp, wc, we = weights
    t1, t2 = counts[:2] / (np.sum(counts) + PARAMS.contribution_epsilon)
    t0, t00 = 0.5 * (1.0 - fraction), 1.0 - fraction
    p12 = mobility * PARAMS.demotion * max(0.0, 1.0 - t1 - t0)
    p21 = mobility * PARAMS.promotion * min(1.0, t1 + t0)
    p23 = mobility * PARAMS.demotion * max(0.0, 1.0 - t1 - t2 - t00)
    p32 = mobility * PARAMS.promotion * min(1.0, t1 + t2 + t00)
    new_wp = wp - wp * p12 + wc * p21
    new_wc = wc - wc * p21 - wc * p23 + wp * p12 + we * p32
    return np.array([new_wp, new_wc, 1.0 - new_wp - new_wc])






