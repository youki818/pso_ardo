from dataclasses import asdict, dataclass

import numpy as np

from common_tools import (ZHQ_OptimizationResult, ZHQ_convergence_factor, ZHQ_levy, ZHQ_linear_schedule,
                          ZHQ_midpoint_repair, ZHQ_raw_mobility, ZHQ_stable_ranks)
from config import ZH_PARAMS as ZH_ARDO_PARAMS


@dataclass(frozen=True)
class ZHQ_CooperativeParameters:
    pso_share: float = 0.35
    share_min: float = 0.0
    share_max: float = 0.60
    ema_rate: float = 0.20
    omega_start: float = 0.85
    omega_end: float = 0.40
    cognitive: float = 1.50
    social: float = 1.50
    velocity_limit: float = 0.20
    elite_fraction: float = 0.20
    recovery_patience: int = 5
    recovery_scale: float = 0.02
    levy_scale: float = 0.02


ZH_PARAMETERS = ZHQ_CooperativeParameters()
ZH_PARAMETER_RECORD = asdict(ZH_PARAMETERS)


def ZHQ_reflect_unit(candidate):
    folded = np.mod(candidate, 2.0)
    return np.where(folded <= 1.0, folded, 2.0 - folded)


def ZHQ_role_assignment(values, weights):
    n = len(values)
    desired = (n - 3) * weights
    counts = np.floor(desired).astype(int) + 1
    rest = n - int(counts.sum())
    extra = np.argsort(-(desired - np.floor(desired)), kind="stable")[:rest]
    counts[extra] += 1
    order, ranks = ZHQ_stable_ranks(values)
    roles = np.empty(n, dtype=int)
    offset = 0
    for role, count in enumerate(counts):
        roles[order[offset:offset + count]] = role
        offset += count
    return roles, order, ranks


def ZHQ_ardo_proposal(archive, i, role, order, ranks, crowd, progress, rng, best):
    n, d = archive.shape
    base = archive[i]
    rank = ranks[i]
    fr = (1.0 - progress)**3 * np.exp(-5.0 * progress)
    noise = (rng.random(d) > 0.9) * rng.normal(0.0, fr, d)
    if role == 0:
        return base + ZH_PARAMETERS.levy_scale * ZHQ_levy(rng, d)
    if role == 1:
        better = archive[order[rng.integers(rank)]]
        worse = archive[order[rng.integers(rank - 1, n)]]
        peer = archive[rng.integers(n)]
        direction = peer - base if rng.random() > crowd else better - worse
        return base + rng.random(d) * direction + noise
    first = archive[order[rng.integers(rank)]]
    second = archive[order[rng.integers(rank)]]
    anchor = first if rng.random() > crowd else best
    return anchor + rng.random(d) * (second - base) + noise


def ZHQ_pso_ardo(objective, initial, rng, iterations):
    P = ZH_PARAMETERS
    n, d = initial.shape
    lower, span = objective.lower, objective.upper - objective.lower
    archive = (initial.copy() - lower) / span
    fitness = objective.population(initial)
    flights, velocity = archive.copy(), rng.uniform(-0.05, 0.05, archive.shape)
    failures = np.zeros(n, dtype=int)
    scale = max(float(np.std(fitness)), float(np.ptp(fitness)) * 0.1, 1e-12)
    weights = np.array([0.3, 0.4, 0.3])
    credit = np.full(2, 0.2)
    share = P.pso_share
    curve = np.empty(iterations + 1)
    curve[0] = float(fitness.min())

    for t in range(1, iterations + 1):
        previous_best = float(fitness.min())
        roles, order, ranks = ZHQ_role_assignment(fitness, weights)
        crowd = ZHQ_convergence_factor(fitness, previous_best)

        protected = [int(rng.choice(np.flatnonzero(roles == role))) for role in range(3)]
        available = np.setdiff1d(np.arange(n), protected)
        pso_count = min(len(available), max(1, int(round(share * n))))
        is_pso = np.zeros(n, dtype=bool)
        is_pso[rng.choice(available, pso_count, replace=False)] = True
        candidates = np.empty_like(archive)
        values = np.empty(n)
        parent = fitness.copy()
        hits = np.zeros(n, dtype=bool)
        best_index = int(np.argmin(fitness))
        elite = order[:max(2, int(np.ceil(P.elite_fraction * n)))]
        omega = ZHQ_linear_schedule(P.omega_start, P.omega_end, t, iterations)

        for i in range(n):
            if is_pso[i]:
                guide = archive[int(rng.choice(elite))]
                if failures[i] >= P.recovery_patience:
                    jump = P.recovery_scale * ZHQ_levy(rng, d)
                    flights[i] = ZHQ_reflect_unit(archive[i] + jump)
                    velocity[i] = 0.0
                    failures[i] = 0
                attraction = archive[i] - flights[i]
                step = (omega * velocity[i]
                        + P.cognitive * rng.random(d) * attraction
                        + P.social * rng.random(d) * (guide - flights[i]))
                velocity[i] = np.clip(step, -P.velocity_limit, P.velocity_limit)
                candidate = flights[i] + velocity[i]
                outside = (candidate < 0.0) | (candidate > 1.0)
                candidate = ZHQ_reflect_unit(candidate)
                velocity[i, outside] = 0.0
                candidates[i] = candidate
            else:
                candidate = ZHQ_ardo_proposal(archive, i, roles[i], order, ranks, crowd, t / iterations, rng,
                                              archive[best_index])
                candidate = ZHQ_midpoint_repair(candidate, archive[i], 0.0, 1.0)
                candidates[i] = np.clip(candidate, 0.0, 1.0)
            values[i] = objective(lower + candidates[i] * span)
            accepted = values[i] <= parent[i]
            if is_pso[i]:
                flights[i] = candidates[i]
                failures[i] = 0 if accepted else failures[i] + 1
            if accepted:
                hits[i] = values[i] <= fitness[best_index]
                archive[i], fitness[i] = candidates[i], values[i]
                if hits[i]:
                    best_index = i

        improved = values <= parent
        gain = np.maximum(parent - values, 0.0)
        reward = 0.5 * improved.astype(float) + 0.5 * np.minimum(gain / scale, 1.0)
        for j, selected in enumerate((is_pso, ~is_pso)):
            if selected.any():
                credit[j] = (1 - P.ema_rate) * credit[j] + P.ema_rate * reward[selected].mean()
        target = P.share_min + (P.share_max - P.share_min) * (
            (credit[0] + 0.01) / (credit.sum() + 0.02))
        share = (1 - P.ema_rate) * share + P.ema_rate * target
        counts = np.bincount(roles[(~is_pso) & hits], minlength=3)
        gf = float(fitness.min())
        with np.errstate(divide="ignore", invalid="ignore"):
            forward = np.tanh(np.divide(previous_best - gf, ZH_ARDO_PARAMS.progress_scale * gf))
        mobility = float(np.fmax(ZH_ARDO_PARAMS.mobility_floor, (1.0 - crowd) * forward))
        moved = ZHQ_raw_mobility(weights, counts, t / iterations, mobility)
        wp, wc = float(np.fmax(0.0, moved[0])), float(np.fmax(0.0, moved[1]))
        weights = np.array([wp, wc, max(0.0, 1.0 - wp - wc)])
        weights = weights / weights.sum()
        curve[t] = gf

    best = int(np.argmin(fitness))
    return ZHQ_OptimizationResult(float(fitness[best]), lower + archive[best] * span,
                                  curve, objective.count)
