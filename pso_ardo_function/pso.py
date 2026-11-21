import numpy as np

from common_tools import ZHQ_OptimizationResult, ZHQ_linear_schedule
from config import ZH_PARAMS


def ZHQ_pso(objective, initial, rng, iterations):
    x = initial.copy()
    n, d = x.shape
    velocity = np.zeros_like(x)
    p = x.copy()
    pf = objective.population(x)
    best = int(np.argmin(pf))
    g, gf = p[best].copy(), float(pf[best])
    curve = np.empty(iterations + 1)
    curve[0] = gf
    limit = ZH_PARAMS.velocity_fraction * (objective.upper - objective.lower)
    for t in range(1, iterations + 1):
        omega = ZHQ_linear_schedule(ZH_PARAMS.omega_max, ZH_PARAMS.omega_min, t, iterations)
        r1, r2 = rng.random((n, d)), rng.random((n, d))
        velocity = omega * velocity + ZH_PARAMS.cognitive * r1 * (p - x) + ZH_PARAMS.social * r2 * (g - x)
        velocity = np.clip(velocity, -limit, limit)
        x = np.clip(x + velocity, objective.lower, objective.upper)
        values = objective.population(x)
        improved = values < pf
        p[improved], pf[improved] = x[improved], values[improved]
        best = int(np.argmin(pf))
        g, gf = p[best].copy(), float(pf[best])
        curve[t] = gf
    return ZHQ_OptimizationResult(gf, g, curve, objective.count)
