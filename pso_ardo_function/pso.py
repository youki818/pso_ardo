import numpy as np

from common_tools import OptimizationResult, linear_schedule
from config import PARAMS

def pso(objective, initial, rng, iterations):
    x = initial.copy()
    n, d = x.shape
    velocity = np.zeros_like(x)
    p = x.copy()
    pf = objective.population(x)
    objective.assert_generation(n, 0)
    best = int(np.argmin(pf))
    g, gf = p[best].copy(), float(pf[best])
    curve = np.empty(iterations + 1)
    curve[0] = gf
    limit = PARAMS.velocity_fraction * (objective.upper - objective.lower)
    for t in range(1, iterations + 1):
        omega = linear_schedule(PARAMS.omega_max, PARAMS.omega_min, t, iterations)
        r1, r2 = rng.random((n, d)), rng.random((n, d))
        velocity = omega * velocity + PARAMS.cognitive * r1 * (p - x) + PARAMS.social * r2 * (g - x)
        velocity = np.clip(velocity, -limit, limit)
        x = np.clip(x + velocity, objective.lower, objective.upper)
        values = objective.population(x)
        improved = values < pf
        p[improved], pf[improved] = x[improved], values[improved]
        best = int(np.argmin(pf))
        g, gf = p[best].copy(), float(pf[best])
        curve[t] = gf
        objective.assert_generation(n, t)
    return OptimizationResult(gf, g, curve, objective.count,
                              current=x, personal_best=p, velocities=velocity)
