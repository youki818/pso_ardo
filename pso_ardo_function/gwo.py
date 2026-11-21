import numpy as np

from common_tools import ZHQ_OptimizationResult, ZHQ_linear_schedule
from config import ZH_PARAMS


def ZHQ_gwo(objective, initial, rng, iterations):
    x = initial.copy()
    n, d = x.shape
    leaders = np.zeros((3, d))
    scores = np.full(3, np.inf)

    def update_leaders(values):
        for i, value in enumerate(values):
            if value < scores[0]:
                scores[0], leaders[0] = value, x[i].copy()
            if scores[0] < value < scores[1]:
                scores[1], leaders[1] = value, x[i].copy()
            if scores[0] < value and scores[1] < value < scores[2]:
                scores[2], leaders[2] = value, x[i].copy()

    update_leaders(objective.population(x))
    curve = np.empty(iterations + 1)
    curve[0] = scores[0]
    for t in range(1, iterations + 1):
        a = ZHQ_linear_schedule(ZH_PARAMS.a_max, 0.0, t, iterations)
        randoms = rng.random((n, d, 3, 2))
        coefficient_a = 2.0 * a * randoms[..., 0] - a
        coefficient_c = 2.0 * randoms[..., 1]
        targets = leaders.T[None, :, :]
        proposals = targets - coefficient_a * np.abs(coefficient_c * targets - x[:, :, None])
        x = np.clip(np.mean(proposals, axis=2), objective.lower, objective.upper)
        update_leaders(objective.population(x))
        curve[t] = scores[0]
    return ZHQ_OptimizationResult(float(scores[0]), leaders[0].copy(), curve, objective.count)
