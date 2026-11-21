
from dataclasses import dataclass


@dataclass(frozen=True)
class ZHQ_AlgorithmParameters:

    omega_max: float = 0.9
    omega_min: float = 0.4
    cognitive: float = 2.0
    social: float = 2.0
    velocity_fraction: float = 0.2
    pioneer_fraction: float = 0.3
    coordinator_fraction: float = 0.4
    levy_beta: float = 1.5
    levy_scale: float = 0.02
    similarity_sigma: float = 0.01
    perturbation_threshold: float = 0.9
    promotion: float = 0.1
    demotion: float = 0.5
    progress_scale: float = 1e-4
    contribution_epsilon: float = 1e-6
    score_floor: float = 1e-30
    mobility_floor: float = 0.1
    a_max: float = 2.0
    woa_b: float = 1.0


ZH_PARAMS = ZHQ_AlgorithmParameters()
