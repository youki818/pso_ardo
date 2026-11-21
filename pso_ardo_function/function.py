from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def ZHQ_chung_reynolds(z):
    return np.dot(z, z)**2


def ZHQ_csendes(z):
    with np.errstate(divide="ignore", invalid="ignore"):
        wave = np.where(z == 0.0, 0.0, np.sin(1.0 / np.where(z == 0.0, 1.0, z)))
    return np.sum(z**6 * (2.0 + wave))


def ZHQ_quartic(z):
    return np.dot(np.arange(1, z.size + 1), z**4)


def ZHQ_schwefel_223(z):
    return np.sum(z**10)


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


ZH_SUITE = "selected6"
ZH_DEFINITIONS = {
    "f1": ZHQ_FunctionSpec(110, "Chung–Reynolds", -0.5, 0.5, ZHQ_chung_reynolds),
    "f2": ZHQ_FunctionSpec(229, "Csendes", -1., 1., ZHQ_csendes),
    "f3": ZHQ_FunctionSpec(250, "Rotated Quartic", -1.28, 1.28, ZHQ_quartic, coordinates="rotated"),
    "f4": ZHQ_FunctionSpec(251, "Rotated Csendes", -1., 1., ZHQ_csendes, coordinates="rotated"),
    "f5": ZHQ_FunctionSpec(254, "Rotated Schwefel 2.23", -10., 10., ZHQ_schwefel_223, coordinates="rotated"),
    "f6": ZHQ_FunctionSpec(7, "Griewank", -600., 600., ZHQ_griewank),
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
