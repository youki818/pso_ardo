
from dataclasses import asdict, dataclass

import numpy as np

from common_tools import (OptimizationResult, convergence_factor, levy, linear_schedule,
                    midpoint_repair, raw_mobility, stable_ranks)
from config import PARAMS as ARDO_PARAMS


@dataclass(frozen=True)
class CooperativeParameters:
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


PARAMETERS = CooperativeParameters()
PARAMETER_RECORD = asdict(PARAMETERS)
DIAGNOSTICS = (
    "pso_share_used", "pso_share_next", "pso_evals", "ardo_evals",
    "pso_successes", "ardo_successes", "pso_gain", "ardo_gain",
    "pso_global_hits", "ardo_global_hits", "pso_credit", "ardo_credit",
    "w_p", "w_c", "w_e", "role_p_evals", "role_c_evals", "role_e_evals",
    "role_p_successes", "role_c_successes", "role_e_successes",
    "continuous_diversity", "structural_diversity", "unique_structures", "crowding",
    "structural_trials", "structural_changes", "context_resets", "flight_recoveries",
    "cognitive_distance", "flight_archive_distance", "stagnation", "gf",
)


def reflect_unit(candidate):

    folded = np.mod(candidate, 2.0)
    return np.where(folded <= 1.0, folded, 2.0 - folded)


def role_assignment(values, weights):

    n = len(values)
    desired = (n - 3) * weights
    counts = np.floor(desired).astype(int) + 1
    rest = n - int(counts.sum())
    extra = np.argsort(-(desired - np.floor(desired)), kind="stable")[:rest]
    counts[extra] += 1
    order, ranks = stable_ranks(values)
    roles = np.empty(n, dtype=int)
    offset = 0
    for role, count in enumerate(counts):
        roles[order[offset:offset + count]] = role
        offset += count
    return roles, order, ranks


def ardo_proposal(archive, i, role, order, ranks, crowd, progress, rng, best=None):




    n, d = archive.shape
    base = archive[i]
    rank = ranks[i]
    fr = (1.0 - progress)**3 * np.exp(-5.0 * progress)
    noise = (rng.random(d) > 0.9) * rng.normal(0.0, fr, d)
    if role == 0:
        return base + PARAMETERS.levy_scale * levy(rng, d)
    if role == 1:
        better = archive[order[rng.integers(rank)]]
        worse = archive[order[rng.integers(rank - 1, n)]]
        peer = archive[rng.integers(n)]
        direction = peer - base if rng.random() > crowd else better - worse
        return base + rng.random(d) * direction + noise
    first = archive[order[rng.integers(rank)]]
    second = archive[order[rng.integers(rank)]]
    anchor = first if rng.random() > crowd else (archive[order[0]] if best is None else best)
    return anchor + rng.random(d) * (second - base) + noise


def pso_ardo(objective, initial, rng, iterations):
    P = PARAMETERS
    n, d = initial.shape
    if n < 6 or iterations < 1:
        raise ValueError("PSO-ARDO 融合要求种群数量至少为 6 且迭代次数至少为 1")
    lower, span = objective.lower, objective.upper - objective.lower
    if np.any(span <= 0):
        raise ValueError("每个决策变量的上下界必须不同")
    archive = (initial.copy() - lower) / span
    fitness = objective.population(initial)
    objective.assert_generation(n, 0)
    flights, velocity = archive.copy(), rng.uniform(-0.05, 0.05, archive.shape)
    failures = np.zeros(n, dtype=int)
    scale = max(float(np.std(fitness)), float(np.ptp(fitness)) * 0.1, 1e-12)
    weights = np.array([0.3, 0.4, 0.3])
    credit = np.full(2, 0.2)                       
    share = P.pso_share
    curve = np.empty(iterations + 1)
    curve[0] = float(fitness.min())
    diagnostic = {key: np.zeros(iterations) for key in DIAGNOSTICS}
    stagnation = 0

    for t in range(1, iterations + 1):
        previous_best = float(fitness.min())
        roles, order, ranks = role_assignment(fitness, weights)
        continuous = float(np.clip(2.0 * np.mean(np.std(archive, axis=0)), 0.0, 1.0))
        structural_diversity, unique = 0.0, 1.0 - continuous
        crowd = convergence_factor(fitness, previous_best)

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
        resets = recoveries = structural_trials = structural_changes = 0
        cognitive_distance = 0.0
        elite = order[:max(2, int(np.ceil(P.elite_fraction * n)))]
        omega = linear_schedule(P.omega_start, P.omega_end, t, iterations)

        for i in range(n):
            if is_pso[i]:
                start = 0
                guide = archive[int(rng.choice(elite))]
                if failures[i] >= P.recovery_patience:
                    jump = P.recovery_scale * levy(rng, d - start)
                    flights[i, start:] = reflect_unit(archive[i, start:] + jump)
                    velocity[i, start:] = 0.0
                    failures[i] = 0
                    recoveries += 1
                attraction = archive[i, start:] - flights[i, start:]
                cognitive_distance += float(np.linalg.norm(attraction))
                step = (omega * velocity[i, start:]
                        + P.cognitive * rng.random(d - start) * attraction
                        + P.social * rng.random(d - start) * (guide[start:] - flights[i, start:]))
                velocity[i, start:] = np.clip(step, -P.velocity_limit, P.velocity_limit)
                candidate = flights[i].copy()
                candidate[start:] += velocity[i, start:]
                outside = (candidate < 0.0) | (candidate > 1.0)
                candidate = reflect_unit(candidate)
                velocity[i, outside] = 0.0
                candidates[i] = candidate
            else:
                candidate = ardo_proposal(archive, i, roles[i], order, ranks, crowd, t / iterations, rng,
                                          best=archive[best_index])
                candidate, _ = midpoint_repair(candidate, archive[i], 0.0, 1.0)
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
        global_hits = values < previous_best
        role_attempts = np.bincount(roles[~is_pso], minlength=3)
        role_successes = np.bincount(roles[(~is_pso) & improved], minlength=3)
        for j, selected in enumerate((is_pso, ~is_pso)):
            if selected.any():
                credit[j] = (1 - P.ema_rate) * credit[j] + P.ema_rate * reward[selected].mean()
        used_share = share
        target = P.share_min + (P.share_max - P.share_min) * (
            (credit[0] + 0.01) / (credit.sum() + 0.02))
        share = (1 - P.ema_rate) * share + P.ema_rate * target
        counts = np.bincount(roles[(~is_pso) & hits], minlength=3)
        gf_now = float(fitness.min())
        with np.errstate(divide="ignore", invalid="ignore"):
            forward = np.tanh(np.divide(previous_best - gf_now, ARDO_PARAMS.progress_scale * gf_now))
        mobility = float(np.fmax(ARDO_PARAMS.mobility_floor, (1.0 - crowd) * forward))
        moved = raw_mobility(weights, counts, t / iterations, mobility)
        wp, wc = float(np.fmax(0.0, moved[0])), float(np.fmax(0.0, moved[1]))
        weights = np.array([wp, wc, max(0.0, 1.0 - wp - wc)])
        weights = weights / weights.sum()
        gf = float(fitness.min())
        stagnation = stagnation + 1 if gf >= previous_best else 0
        row = dict(
            pso_share_used=used_share, pso_share_next=share, pso_evals=pso_count,
            ardo_evals=n-pso_count, pso_successes=int(improved[is_pso].sum()),
            ardo_successes=int(improved[~is_pso].sum()), pso_gain=float(gain[is_pso].sum()),
            ardo_gain=float(gain[~is_pso].sum()), pso_global_hits=int(global_hits[is_pso].sum()),
            ardo_global_hits=int(global_hits[~is_pso].sum()), pso_credit=credit[0], ardo_credit=credit[1],
            w_p=weights[0], w_c=weights[1], w_e=weights[2],
            role_p_evals=role_attempts[0], role_c_evals=role_attempts[1], role_e_evals=role_attempts[2],
            role_p_successes=role_successes[0], role_c_successes=role_successes[1], role_e_successes=role_successes[2],
            continuous_diversity=continuous, structural_diversity=structural_diversity, unique_structures=unique,
            crowding=crowd, structural_trials=structural_trials, structural_changes=structural_changes, context_resets=resets,
            flight_recoveries=recoveries, cognitive_distance=cognitive_distance / max(1, pso_count),
            flight_archive_distance=float(np.mean(np.linalg.norm(flights - archive, axis=1))),
            stagnation=stagnation, gf=gf)
        for key in DIAGNOSTICS:
            diagnostic[key][t-1] = row[key]
        curve[t] = gf
        objective.assert_generation(n, t)

    best = int(np.argmin(fitness))
    return OptimizationResult(float(fitness[best]), lower + archive[best] * span,
                              curve, objective.count, diagnostic,
                              current=lower + flights * span,
                              personal_best=lower + archive * span, velocities=velocity * span)


