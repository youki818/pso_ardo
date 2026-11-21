

import numpy as np


from common_tools import (DIAGNOSTIC_COLUMNS, OptimizationResult, convergence_factor, levy,
                    midpoint_repair, perturbation, raw_mobility, roles_from_ranks,
                    stable_ranks)
from config import PARAMS


def ardo(objective, initial, rng, iterations):
    x = initial.copy()
    n, d = x.shape
    lower, upper = objective.lower, objective.upper
    span = upper - lower
    values = objective.population(x)
    objective.assert_generation(n, 0)
    best = int(np.argmin(values))
    g, gf = x[best].copy(), float(values[best])
    wp, wc = PARAMS.pioneer_fraction, PARAMS.coordinator_fraction
    curve = np.empty(iterations + 1)
    curve[0] = gf

    diagnostic = {key: np.full(iterations, np.nan) for key in DIAGNOSTIC_COLUMNS}
    for t in range(1, iterations + 1):
        counts = np.zeros(3, dtype=int)
        order, ranks = stable_ranks(values)
        roles = roles_from_ranks(ranks, wp, wc)
        fraction = t / iterations
        fr = (1.0 - fraction)**3 * np.exp(-5.0 * fraction)

        crowd = convergence_factor(values, gf)
        previous_best = gf
        for i in range(n):
            role, rank = roles[i], ranks[i]
            if role == 0:
                candidate = x[i] + PARAMS.levy_scale * span * levy(rng, d)
            elif role == 1:

                right = x[order[rng.integers(0, rank)]]
                left = x[order[rng.integers(rank - 1, n)]]
                any_peer = x[order[rng.integers(0, n)]]
                noise = perturbation(rng, d, fr, span)
                if rng.random() > crowd:
                    candidate = x[i] + rng.random(d) * (any_peer - x[i]) + noise
                else:
                    candidate = x[i] + rng.random(d) * (right - left) + noise
            else:
                first = x[order[rng.integers(0, rank)]]
                second = x[order[rng.integers(0, rank)]]
                noise = perturbation(rng, d, fr, span)
                if rng.random() > crowd:
                    candidate = first + rng.random(d) * (second - x[i]) + noise
                else:
                    candidate = g + rng.random(d) * (second - x[i]) + noise
            candidate, _ = midpoint_repair(candidate, x[i], lower, upper)
            value = objective(candidate)
            if value <= values[i]:
                values[i], x[i] = value, candidate
                if value <= gf:
                    gf, g = value, x[i].copy()
                    counts[role] += 1




        with np.errstate(divide="ignore", invalid="ignore"):
            forward = np.tanh(np.divide(previous_best - gf, PARAMS.progress_scale * gf))
        mobility = float(np.fmax(PARAMS.mobility_floor, (1.0 - crowd) * forward))
        weights = raw_mobility(np.array([wp, wc, 1.0 - wp - wc]), counts, fraction, mobility)
        wp, wc = float(np.fmax(0.0, weights[0])), float(np.fmax(0.0, weights[1]))
        row = {"f_pc": crowd, "w_p": wp, "w_c": wc, "w_e": 1.0 - wp - wc,
               "n_p": counts[0], "n_c": counts[1], "n_e": counts[2],
               "role_p": np.count_nonzero(roles == 0), "role_c": np.count_nonzero(roles == 1),
               "role_e": np.count_nonzero(roles == 2), "pso_samples": 0, "pso_hits": 0, "gf": gf}
        for key, value in row.items():
            diagnostic[key][t - 1] = value
        curve[t] = gf
        objective.assert_generation(n, t)
    return OptimizationResult(gf, g, curve, objective.count, diagnostic, current=x)



