from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import numpy as np


def sphere(z):
    return float(np.sum(z**2))


@lru_cache(maxsize=8)
def elliptic_weights(dimension):
    return np.geomspace(1.0, 1e6, dimension)


def elliptic(z):
    return np.dot(elliptic_weights(z.size), z**2)


def schwefel_12(z):
    return float(np.sum(np.cumsum(z)**2))


def rosenbrock(z):
    return float(np.sum(100.0 * (z[1:] - z[:-1]**2)**2 + (z[:-1] - 1.0)**2))


def ackley(z):
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.mean(z**2)))
                 - np.exp(np.mean(np.cos(2.0 * np.pi * z))) + 20.0 + np.e)


def griewank(z):
    return float(np.sum(z**2) / 4000.0
                 - np.prod(np.cos(z / np.sqrt(np.arange(1, z.size + 1)))) + 1.0)


@dataclass(frozen=True)
class FunctionSpec:
    source_number: int
    name: str
    lower: float
    upper: float
    formula: Callable
    minimum_point: Callable = np.zeros
    coordinates: str = "shifted"


DEFINITIONS = {
    "f1": FunctionSpec(1, "Sphere", -100., 100., sphere),
    "f2": FunctionSpec(2, "Rotated High-Conditioned Elliptic", -100., 100., elliptic, coordinates="rotated"),
    "f3": FunctionSpec(3, "Schwefel 1.2", -100., 100., schwefel_12),
    "f4": FunctionSpec(4, "Rosenbrock", -30., 30., rosenbrock, np.ones),
    "f5": FunctionSpec(6, "Ackley", -32., 32., ackley),
    "f6": FunctionSpec(7, "Griewank", -600., 600., griewank),
}


@dataclass(frozen=True)
class Benchmark:
    code: str
    expanded16_code: str
    name: str
    lower: np.ndarray
    upper: np.ndarray
    shift: np.ndarray
    optimum: np.ndarray
    formula: Callable
    native_optimum: np.ndarray
    coordinates: str
    rotation: Optional[np.ndarray]

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if self.coordinates == "shifted":
            return float(self.formula(x - self.shift))
        offset = x - self.optimum
        if self.rotation is not None:
            offset = self.rotation @ offset
        return float(self.formula(offset + self.native_optimum))


def make_benchmark(code: str, dimension: int, shift_seed: int, shifted: bool = True) -> Benchmark:
    code = code.lower()
    spec = DEFINITIONS[code]
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    seed = [shift_seed, spec.source_number]
    rng = np.random.default_rng(seed)
    lo, hi = spec.lower, spec.upper
    shift = (rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), dimension)
             if shifted else np.zeros(dimension))
    lower, upper = np.full(dimension, lo), np.full(dimension, hi)
    native = spec.minimum_point(dimension)
    optimum = shift + native
    if np.any((optimum < lower) | (optimum > upper)):
        raise ValueError(f"{code} optimum outside bounds")
    rotation = None
    if spec.coordinates == "rotated":
        rng = np.random.default_rng([*seed, 1])
        rotation, triangular = np.linalg.qr(rng.standard_normal((dimension, dimension)))
        rotation *= np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
        if np.linalg.slogdet(rotation)[0] < 0:
            rotation[:, -1] *= -1.0
        rotation.setflags(write=False)
    for array in (lower, upper, shift, optimum, native):
        array.setflags(write=False)
    return Benchmark(code, f"F{spec.source_number}", spec.name, lower, upper, shift,
                     optimum, spec.formula, native, spec.coordinates, rotation)
