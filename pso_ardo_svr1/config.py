"""
算法参数配置模块。

"""


from dataclasses import dataclass


@dataclass(frozen=True)
class ZHQ_AlgorithmParameters:
    # PSO 的惯性权重和学习因子
    omega_max: float = 0.9
    omega_min: float = 0.4
    cognitive: float = 2.0
    social: float = 2.0
    velocity_fraction: float = 0.2

    # ARDO 的角色比例
    pioneer_fraction: float = 0.3
    coordinator_fraction: float = 0.4

    # Levy 飞行和随机扰动参数
    levy_beta: float = 1.5
    levy_scale: float = 0.02
    similarity_sigma: float = 0.01
    perturbation_threshold: float = 0.9

    # 角色晋升、降级和流动性参数
    promotion: float = 0.1
    demotion: float = 0.5
    progress_scale: float = 1e-4
    contribution_epsilon: float = 1e-6
    score_floor: float = 1e-30
    mobility_floor: float = 0.1

    # GWO 和 WOA 使用的算法参数
    a_max: float = 2.0
    woa_b: float = 1.0


ZH_PARAMS = ZHQ_AlgorithmParameters()
