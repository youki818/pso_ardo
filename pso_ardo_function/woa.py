import numpy as np

from common_tools import OptimizationResult, linear_schedule
from config import PARAMS

def woa(objective, initial, rng, iterations):

    x = initial.copy()
    n, d = x.shape
    values = objective.population(x)
    objective.assert_generation(n, 0)
    best = int(np.argmin(values))
    g, gf = x[best].copy(), float(values[best])
    curve = np.empty(iterations + 1)
    curve[0] = gf
    for t in range(1, iterations + 1):
        a = linear_schedule(PARAMS.a_max, 0.0, t, iterations)
        a2 = linear_schedule(-1.0, -2.0, t, iterations)
        for i in range(n):
            r1, r2 = rng.random(), rng.random()
            coefficient_a, coefficient_c = 2.0 * a * r1 - a, 2.0 * r2
            spiral_l = (a2 - 1.0) * rng.random() + 1.0
            probability = rng.random()
            if probability < 0.5:
                if abs(coefficient_a) >= 1.0:


                    for j in range(d):
                        peer = int(np.floor(n * rng.random()))
                        target = x[peer, j]
                        x[i, j] = target - coefficient_a * abs(coefficient_c * target - x[i, j])
                else:
                    x[i] = g - coefficient_a * np.abs(coefficient_c * g - x[i])
            else:
                x[i] = (np.abs(g - x[i]) * np.exp(PARAMS.woa_b * spiral_l)
                        * np.cos(2.0 * np.pi * spiral_l) + g)
        x = np.clip(x, objective.lower, objective.upper)
        values = objective.population(x)
        best = int(np.argmin(values))
        if values[best] < gf:
            g, gf = x[best].copy(), float(values[best])
        curve[t] = gf
        objective.assert_generation(n, t)
    return OptimizationResult(gf, g, curve, objective.count, current=x)
