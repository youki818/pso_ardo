from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def ZHQ_schwefel_220(z):
    return np.sum(np.abs(z))


def ZHQ_trigonometric_1(z):
    # D − Σcos z = Σ2sin²(z/2)，1 − cos z − sin z = 2sin²(z/2) − sin z
    half = 2.0 * np.sin(0.5 * z)**2
    terms = np.sum(half) + np.arange(1, z.size + 1) * (half - np.sin(z))
    return np.dot(terms, terms)


def ZHQ_bohachevsky_3(z):
    # 0.3(1 − cos θ) = 0.6sin²(θ/2)，θ = 3πz_i + 4πz_{i+1}
    theta = 3.0 * np.pi * z[:-1] + 4.0 * np.pi * z[1:]
    return np.sum(z[:-1]**2 + 2.0 * z[1:]**2 + 0.6 * np.sin(0.5 * theta)**2)


def ZHQ_griewank(z):
    return float(np.sum(z**2) / 4000.0
                 - np.prod(np.cos(z / np.sqrt(np.arange(1, z.size + 1)))) + 1.0)


@dataclass(frozen=True)
class ZHQ_FunctionSpec:
    seed_number: int
    name: str
    lower: float
    upper: float
    formula: Callable
    coordinates: str = "shifted"


ZH_SUITE = "selected4"
ZH_DEFINITIONS = {
    "f1": ZHQ_FunctionSpec(307, "Rotated Schwefel 2.20", -1., 1., ZHQ_schwefel_220, coordinates="rotated"),
    "f2": ZHQ_FunctionSpec(313, "Trigonometric 1", -1., 1., ZHQ_trigonometric_1),
    "f3": ZHQ_FunctionSpec(317, "Bohachevsky 3", -15., 15., ZHQ_bohachevsky_3),
    "f4": ZHQ_FunctionSpec(319, "Rotated Griewank", -600., 600., ZHQ_griewank, coordinates="rotated"),
}


@dataclass(frozen=True)
class ZHQ_Benchmark:
    code: str
    name: str
    lower: np.ndarray
    upper: np.ndarray
    shift: np.ndarray
    optimum: np.ndarray
    formula: Callable
    coordinates: str
    rotation: Optional[np.ndarray]
    optimum_value: float = 0.0

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if self.rotation is None:
            return float(self.formula(x - self.shift))
        return float(self.formula(self.rotation @ (x - self.optimum)))


def ZHQ_make_benchmark(code: str, dimension: int, shift_seed: int, shifted: bool = True) -> ZHQ_Benchmark:
    code = code.lower()
    spec = ZH_DEFINITIONS[code]
    seed = [shift_seed, spec.seed_number]
    rng = np.random.default_rng(seed)
    lo, hi = spec.lower, spec.upper
    shift = (rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), dimension)
             if shifted else np.zeros(dimension))
    lower, upper = np.full(dimension, lo), np.full(dimension, hi)
    rotation = None
    if spec.coordinates == "rotated":
        rng = np.random.default_rng([*seed, 1])
        rotation, triangular = np.linalg.qr(rng.standard_normal((dimension, dimension)))
        rotation *= np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
        if np.linalg.slogdet(rotation)[0] < 0:
            rotation[:, -1] *= -1.0
    return ZHQ_Benchmark(code, spec.name, lower, upper, shift, shift.copy(),
                         spec.formula, spec.coordinates, rotation)
