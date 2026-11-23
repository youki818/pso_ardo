"""UCI 回归数据集上的 SVR 寻优问题（特征选择 + 超参数），供 PSO / ARDO / PSO_ARDO 当作目标函数调用。

决策向量 x = [m_1, …, m_d, ln C, ln g, ln ε]：
  * m_j ∈ [0, 1] 是第 j 个特征的选择位，m_j > 0.5 表示选中；全部未选中时保留 m_j 最大的那个特征；
  * ln C、ln g ∈ [-10, 10]，与论文 4.4.3 节的取值范围相同；ln ε ∈ [-9, 0]，ε 以归一化后的目标值为单位；
  * --no-feature-selection 去掉选择位，--fix-epsilon 把 ε 固定为 0.01（MATLAB 中的 -p 0.01）；
    两者同时使用，就是论文中只寻优 C、g 的设置。
RBF 核为 K(x, x') = exp(-g‖x − x'‖²)，与论文式 (4-9) 相同。

适应度 = 训练集上 K 折交叉验证的 MSE，按训练集归一化到 [-1, 1] 的目标值计算，与 libsvm -v 的输出同口径。
适应度里不加特征数惩罚，所以它就是一个可以直接解释的误差。
  * 交叉验证的分折固定，各算法优化的是同一个确定的函数（libsvm -v 每次调用都会重新随机分折）；
  * 缺失值填补与 min-max 归一化只在每一折的训练部分上拟合，再作用到该折的验证部分；
  * 测试集在寻优过程中从不使用，只在寻优结束后评价最终模型。
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import warnings
import zipfile

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR


ZH_ROOT = Path(__file__).resolve().parent
ZH_DATA = ZH_ROOT / "data"
ZH_AUTO_MPG_FILE = ZH_DATA / "auto-mpg.data"
ZH_MATLAB_ORDER_FILE = ZH_DATA / "matlab_split_order.csv"
ZH_SUPERCONDUCTIVITY_FILE = ZH_DATA / "superconductivity.zip"
# 由 prepare_residential_building.py 从 UCI 原始压缩包 data/residential_building.zip 原样转换而来
ZH_RESIDENTIAL_FILE = ZH_DATA / "residential_building.csv"
ZH_RESIDENTIAL_DATES = ("START_YEAR", "START_QUARTER", "COMPLETION_YEAR", "COMPLETION_QUARTER")
ZH_RESIDENTIAL_OUTPUTS = ("V-9", "V-10")
ZH_DATASET_FILES = {"auto_mpg": ZH_AUTO_MPG_FILE, "superconductivity": ZH_SUPERCONDUCTIVITY_FILE,
                    "residential_building": ZH_RESIDENTIAL_FILE}
ZH_DATASET_TITLES = {"auto_mpg": "UCI Auto MPG（UCI 编号 9）",
                     "superconductivity": "UCI Superconductivity（UCI 编号 464）",
                     "residential_building": "UCI Residential Building（UCI 编号 437）"}
ZH_AUTO_MPG_COLUMNS = ("mpg", "cylinders", "displacement", "horsepower", "weight",
                       "acceleration", "model_year", "origin")
ZH_COLUMN_TITLES = {
    "mpg": "油耗（英里/加仑）", "cylinders": "气缸数", "displacement": "排量（立方英寸）",
    "horsepower": "马力", "weight": "整备质量（磅）", "acceleration": "0–60 mph 加速时间（秒）",
    "model_year": "车型年份", "origin": "产地", "critical_temp": "临界温度（K）",
    "V-9": "实际售价（10000 IRRm）",
}
# mpg 是 UCI 定义的预测目标；displacement 用于复现同目录 MATLAB 程序（它把 H 列排量当输出、mpg 当输入）
ZH_AUTO_MPG_TARGETS = ("mpg", "displacement")
ZH_SVR_ALGORITHMS = ("PSO_ARDO", "ARDO", "PSO")
ZH_GRID, ZH_DEFAULT = "SVR_GRID", "SVR_DEFAULT"
ZH_BASELINES = (ZH_GRID, ZH_DEFAULT)


def ZHQ_normalize_algorithm(value):
    """命令行中的算法名不区分大小写，pso-ardo 与 pso_ardo 等价。"""
    key = str(value).strip().upper().replace("-", "_")
    if key not in ZH_SVR_ALGORITHMS:
        raise ValueError(f"未知算法 {value!r}，可选 {ZH_SVR_ALGORITHMS}")
    return key


@dataclass(frozen=True)
class ZHQ_SVRConfig:
    dataset: str = "auto_mpg"
    population: int = 20
    iterations: int = 100
    runs: int = 20
    seed: int = 20261111
    target: str | None = None           # Auto MPG：mpg（默认）或 displacement；Superconductivity 固定为 critical_temp
    n_samples: int = 1000               # 仅 Superconductivity：从 21263 条中无放回随机抽取的样本数
    sample_seed: int = 2026
    split: str = "matlab"               # matlab：沿用 nnn.xlsx 的打乱顺序（仅 Auto MPG）；random：按 split_seed 随机划分
    matlab_train_size: int = 319        # 与 MATLAB 程序一致：前 319 个训练，其余 79 个测试
    split_seed: int = 7
    test_fraction: float = 0.2          # 仅 split = random 时使用
    folds: int = 5                      # 与 MATLAB 程序的 -v 5 一致
    cv_seed: int = 11
    feature_selection: bool = True
    ln_c: tuple[float, float] = (-10.0, 10.0)
    ln_gamma: tuple[float, float] = (-10.0, 10.0)
    optimize_epsilon: bool = True
    epsilon: float = 0.01               # 固定 ε 时的取值，也用于网格搜索与默认参数 SVR；以 [-1, 1] 归一化目标为单位
    ln_epsilon: tuple[float, float] = (-9.0, 0.0)
    # libsvm 迭代上限；-1 表示不设上限。C 接近 e^10、g 很小时 SMO 难以收敛，100 万次迭代单次训练约 0.7 s，
    # 粒子停在该边界附近时一个任务会拖到数小时；10 万次约 0.13 s。正常区域的训练只需几千到几万次迭代，不受影响
    max_iter: int = 100_000
    baselines: bool = True
    grid_step: float = 0.5              # 网格搜索 SVR 在 ln C、ln g 上的步长
    algorithms: tuple[str, ...] = ZH_SVR_ALGORITHMS

    @property
    def evaluations(self) -> int:
        return self.population * (self.iterations + 1)

    def quick(self) -> "ZHQ_SVRConfig":
        return replace(self, population=6, iterations=5, runs=2, folds=3, grid_step=2.0,
                       n_samples=min(self.n_samples, 300))


@dataclass(frozen=True)
class ZHQ_Dataset:
    key: str
    title: str
    features: np.ndarray
    target: np.ndarray
    feature_names: tuple[str, ...]
    target_name: str
    missing: dict
    relative_metrics: bool              # 目标值远离 0 时才报告相对误差（MRE、相对误差均方）
    rows: np.ndarray | None             # 抽样时为原始文件中的行号（0 起算），不抽样时为 None


def ZHQ_read_auto_mpg(path=ZH_AUTO_MPG_FILE) -> np.ndarray:
    """读取 UCI 原始文件：每行 8 个数值字段，制表符后是带引号的车名；'?' 记为缺失。"""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="latin-1").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t", 1)[0].split()
        if len(fields) != len(ZH_AUTO_MPG_COLUMNS):
            raise ValueError(f"{Path(path).name} 第 {number} 行应有 {len(ZH_AUTO_MPG_COLUMNS)} 个数值字段，"
                             f"实际为 {len(fields)} 个")
        rows.append([np.nan if value == "?" else float(value) for value in fields])
    return np.asarray(rows, dtype=float)


@lru_cache(maxsize=1)
def ZHQ_read_superconductivity(path=ZH_SUPERCONDUCTIVITY_FILE):
    """读取 UCI 原始压缩包中的 train.csv（21263 条，81 个材料描述符 + critical_temp）。"""
    import pandas as pd
    with zipfile.ZipFile(path) as archive, archive.open("train.csv") as handle:
        return pd.read_csv(handle)


@lru_cache(maxsize=1)
def ZHQ_read_residential_building(path=ZH_RESIDENTIAL_FILE):
    """读取转换后的 Residential Building 表：372 行，4 个工程日期列 + 103 个输入 + 2 个输出（V-9 售价、V-10 建造成本）。"""
    import pandas as pd
    if not Path(path).is_file():
        raise ValueError(f"找不到 {Path(path).name}，请先运行 prepare_residential_building.py 由原始压缩包生成")
    return pd.read_csv(path)


def ZHQ_load_dataset(config: ZHQ_SVRConfig) -> ZHQ_Dataset:
    title = ZH_DATASET_TITLES.get(config.dataset)
    if config.dataset == "auto_mpg":
        target = config.target or "mpg"
        if target not in ZH_AUTO_MPG_TARGETS:
            raise ValueError(f"Auto MPG 的目标列只能是 {ZH_AUTO_MPG_TARGETS}")
        data = ZHQ_read_auto_mpg()
        column = ZH_AUTO_MPG_COLUMNS.index(target)
        if np.isnan(data[:, column]).any():
            raise ValueError(f"目标列 {target} 含缺失值")
        keep = [j for j in range(len(ZH_AUTO_MPG_COLUMNS)) if j != column]
        names = tuple(ZH_AUTO_MPG_COLUMNS[j] for j in keep)
        features = data[:, keep]
        missing = {name: int(np.isnan(features[:, k]).sum())
                   for k, name in enumerate(names) if np.isnan(features[:, k]).any()}
        return ZHQ_Dataset("auto_mpg", title, features, data[:, column], names, target,
                           missing, True, None)
    if config.dataset == "superconductivity":
        if config.target not in (None, "critical_temp"):
            raise ValueError("Superconductivity 的目标固定为 critical_temp，不能指定 --target")
        frame = ZHQ_read_superconductivity()
        if "critical_temp" not in frame.columns:
            raise ValueError("train.csv 中缺少 critical_temp 列")
        if not 10 <= config.n_samples <= len(frame):
            raise ValueError(f"n_samples 必须在 [10, {len(frame)}] 内")
        rows = np.sort(np.random.default_rng(config.sample_seed).choice(len(frame), config.n_samples,
                                                                         replace=False))
        names = tuple(column for column in frame.columns if column != "critical_temp")
        subset = frame.iloc[rows]
        features = subset[list(names)].to_numpy(dtype=float)
        target = subset["critical_temp"].to_numpy(dtype=float)
        if not (np.isfinite(features).all() and np.isfinite(target).all()):
            raise ValueError("Superconductivity 数据中出现非有限值")
        # 临界温度最小只有 0.0002 K，相对误差会被个别样本放大到失去意义，因此不报告 MRE
        return ZHQ_Dataset("superconductivity", title, features, target, names, "critical_temp",
                           {}, False, rows)
    if config.dataset == "residential_building":
        if config.target not in (None, "V-9"):
            raise ValueError("Residential Building 的目标固定为 V-9（实际售价），不能指定 --target")
        frame = ZHQ_read_residential_building()
        # 输入为 UCI 页面所说的 103 个变量：V-1～V-8 与 19 个经济变量 × 5 个时间滞后。
        # 4 个工程日期列不在 UCI 的 105 个变量之内；另一个输出 V-10（建造成本）也不能当输入
        names = tuple(column for column in frame.columns
                      if column not in ZH_RESIDENTIAL_DATES + ZH_RESIDENTIAL_OUTPUTS)
        if len(frame) != 372 or len(names) != 103:
            raise ValueError(f"Residential Building 应为 372 行、103 个输入，实际为 {len(frame)} 行、{len(names)} 个输入")
        features = frame[list(names)].to_numpy(dtype=float)
        target = frame["V-9"].to_numpy(dtype=float)
        if not (np.isfinite(features).all() and np.isfinite(target).all()) or target.min() <= 0:
            raise ValueError("Residential Building 数据中出现非有限值或非正的售价")
        return ZHQ_Dataset("residential_building", title, features, target, names, "V-9", {}, True, None)
    raise ValueError(f"未知数据集 {config.dataset!r}，可选 {tuple(ZH_DATASET_FILES)}")


def ZHQ_split_indices(n: int, config: ZHQ_SVRConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.split == "matlab":
        if config.dataset != "auto_mpg":
            raise ValueError("matlab 划分（nnn.xlsx 的打乱顺序）只适用于 Auto MPG")
        order = np.loadtxt(ZH_MATLAB_ORDER_FILE, delimiter=",", skiprows=1, dtype=int).ravel() - 1
        if order.size != n or not np.array_equal(np.sort(order), np.arange(n)):
            raise ValueError(f"{ZH_MATLAB_ORDER_FILE.name} 必须是 1..{n} 的一个排列")
        if not 1 <= config.matlab_train_size < n:
            raise ValueError("matlab_train_size 必须在 [1, 样本数) 内")
        return order[:config.matlab_train_size], order[config.matlab_train_size:]
    if config.split == "random":
        train, test = train_test_split(np.arange(n), test_size=config.test_fraction,
                                       random_state=config.split_seed)
        return train, test
    raise ValueError(f"未知划分方式 {config.split!r}")


def ZHQ_span(low, high):
    span = np.asarray(high, dtype=float) - low
    return np.where(span > 0, span, 1.0)


class ZHQ_MinMaxPreprocessor:
    """只在给定的训练行上拟合：缺失值用中位数填补，特征和目标线性映射到 [-1, 1]（同 MATLAB mapminmax）。
    逐列变换与“先选特征再变换”等价，所以可以对全部特征预先变换、寻优时再按选择位取列。"""

    def __init__(self, features: np.ndarray, target: np.ndarray):
        self.median = np.nanmedian(features, axis=0)
        filled = self.fill(features)
        self.x_low = filled.min(axis=0)
        self.x_span = ZHQ_span(self.x_low, filled.max(axis=0))
        self.y_low = float(target.min())
        self.y_span = float(ZHQ_span(self.y_low, target.max()))

    def fill(self, features: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(features), self.median, features)

    def features(self, features: np.ndarray) -> np.ndarray:
        return 2.0 * (self.fill(features) - self.x_low) / self.x_span - 1.0

    def target(self, target: np.ndarray) -> np.ndarray:
        return 2.0 * (target - self.y_low) / self.y_span - 1.0

    def inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        return (scaled + 1.0) * 0.5 * self.y_span + self.y_low


def ZHQ_regression_metrics(truth, prediction, relative: bool = True) -> dict:
    """原始量纲下的误差指标。relative 为真时再给出与论文表 4-5、MATLAB 程序同口径的相对误差：
    mre = mean(|t - p| / t)，msre = mse(|t - p| / t) = mean(((t - p) / t)²)。"""
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    residual = prediction - truth
    total = float(np.sum((truth - truth.mean())**2))
    metrics = dict(rmse=float(np.sqrt(np.mean(residual**2))), mae=float(np.mean(np.abs(residual))),
                   r2=float(1.0 - residual @ residual / total) if total > 0 else float("nan"))
    if relative:
        ratio = np.abs(residual) / np.abs(truth)
        metrics.update(mre=float(ratio.mean()), msre=float(np.mean(ratio**2)))
    return metrics


class ZHQ_SVRProblem:
    """可调用对象：problem(x) 返回决策向量 x 对应的交叉验证 MSE；lower/upper 为搜索边界。"""

    def __init__(self, config: ZHQ_SVRConfig):
        self.config = config
        self.dataset = ZHQ_load_dataset(config)
        self.case = self.dataset.key
        features, target = self.dataset.features, self.dataset.target
        self.n_features = features.shape[1]
        self.train_index, self.test_index = ZHQ_split_indices(len(target), config)
        self.train_x, self.train_y = features[self.train_index], target[self.train_index]
        self.test_x, self.test_y = features[self.test_index], target[self.test_index]
        self.preprocessor = ZHQ_MinMaxPreprocessor(self.train_x, self.train_y)
        # 把原始量纲的 MSE 换算到“训练集归一化到 [-1, 1] 的目标值”上，数值与 libsvm -v 的输出可比
        self.fitness_scale = (2.0 / self.preprocessor.y_span)**2

        if not 2 <= config.folds <= len(self.train_y):
            raise ValueError("folds 必须在 [2, 训练样本数] 内")
        self.folds, self.fold_indices = [], []
        splitter = KFold(config.folds, shuffle=True, random_state=config.cv_seed)
        for fit_rows, validation_rows in splitter.split(np.arange(len(self.train_y))):
            fold = ZHQ_MinMaxPreprocessor(self.train_x[fit_rows], self.train_y[fit_rows])
            self.folds.append((fold.features(self.train_x[fit_rows]), fold.target(self.train_y[fit_rows]),
                               fold.features(self.train_x[validation_rows]), self.train_y[validation_rows],
                               fold))
            self.fold_indices.append((self.train_index[fit_rows].copy(),
                                      self.train_index[validation_rows].copy()))

        self.mask_dims = self.n_features if config.feature_selection else 0
        bounds = [config.ln_c, config.ln_gamma] + ([config.ln_epsilon] if config.optimize_epsilon else [])
        self.continuous_names = ("ln_C", "ln_gamma", "ln_epsilon")[:len(bounds)]
        self.lower = np.concatenate([np.zeros(self.mask_dims), [low for low, _ in bounds]])
        self.upper = np.concatenate([np.ones(self.mask_dims), [high for _, high in bounds]])
        if np.any(self.lower >= self.upper):
            raise ValueError("每个搜索变量的下界都必须小于上界")
        for array in (self.lower, self.upper):
            array.setflags(write=False)
        self.dimension = self.lower.size
        self.reset_cache()

    def reset_cache(self):
        """每个算法开始前清空缓存和计数，保证各算法的用时与拟合次数可比。"""
        self.cache = {}
        self.cache_hits = self.cv_calls = self.svr_fits = self.convergence_warnings = 0

    def evaluation_stats(self) -> dict:
        return dict(cache_hits=self.cache_hits, unique_cv_evaluations=self.cv_calls,
                    svr_fits=self.svr_fits, convergence_warnings=self.convergence_warnings)

    def all_features(self) -> np.ndarray:
        return np.ones(self.n_features, dtype=bool)

    def decode(self, x) -> dict:
        x = np.asarray(x, dtype=float)
        if self.mask_dims:
            mask = x[:self.mask_dims] > 0.5
            if not mask.any():
                mask[int(np.argmax(x[:self.mask_dims]))] = True
        else:
            mask = self.all_features()
        tail = x[self.mask_dims:]
        epsilon = float(np.exp(tail[2])) if self.config.optimize_epsilon else float(self.config.epsilon)
        return dict(mask=mask, C=float(np.exp(tail[0])), gamma=float(np.exp(tail[1])), epsilon=epsilon)

    def fit(self, parameters: dict, features: np.ndarray, target: np.ndarray) -> SVR:
        model = SVR(kernel="rbf", C=parameters["C"], gamma=parameters["gamma"],
                    epsilon=parameters["epsilon"], tol=1e-3, cache_size=200,
                    max_iter=self.config.max_iter)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features, target)
        self.svr_fits += 1
        self.convergence_warnings += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        return model

    def cv_mse(self, parameters: dict) -> float:
        """K 折交叉验证：各折验证样本的平方误差汇总后求平均（同 libsvm -v 的算法）。"""
        self.cv_calls += 1
        mask, squared = parameters["mask"], 0.0
        for fit_x, fit_y, validation_x, validation_y, fold in self.folds:
            model = self.fit(parameters, fit_x[:, mask], fit_y)
            residual = fold.inverse_target(model.predict(validation_x[:, mask])) - validation_y
            squared += float(residual @ residual)
        return squared / len(self.train_y) * self.fitness_scale

    def __call__(self, x) -> float:
        x = np.asarray(x, dtype=np.float64)
        parameters = self.decode(x)
        # 选择位只通过 0.5 阈值起作用，所以缓存键用“选中的特征 + 连续参数”
        key = (np.packbits(parameters["mask"]).tobytes(), x[self.mask_dims:].tobytes())
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        value = self.cv_mse(parameters)
        self.cache[key] = value
        return value

    def evaluate(self, parameters: dict) -> dict:
        """用全部训练集重新拟合，返回训练集、测试集在原始量纲下的指标和测试集预测值。"""
        p, mask = self.preprocessor, parameters["mask"]
        train_features = p.features(self.train_x)[:, mask]
        model = self.fit(parameters, train_features, p.target(self.train_y))
        train_prediction = p.inverse_target(model.predict(train_features))
        test_prediction = p.inverse_target(model.predict(p.features(self.test_x)[:, mask]))
        relative = self.dataset.relative_metrics
        n_iter = int(np.ravel(model.n_iter_)[0])
        return dict(n_support=int(len(model.support_)), n_iter=n_iter,
                    converged=bool(self.config.max_iter < 0 or n_iter < self.config.max_iter),
                    train=ZHQ_regression_metrics(self.train_y, train_prediction, relative),
                    test=ZHQ_regression_metrics(self.test_y, test_prediction, relative),
                    test_prediction=test_prediction)


def ZHQ_grid_axis(bounds, step: float) -> np.ndarray:
    low, high = bounds
    count = int(np.floor((high - low) / step + 1e-9)) + 1
    return low + step * np.arange(count)


def ZHQ_grid_search(problem: ZHQ_SVRProblem) -> dict:
    """传统 SVR 的参数选择（论文 4.4.2 节、SVR.m）：使用全部特征、ε 固定，在 ln C × ln g 网格上逐点做交叉验证。
    外层按 C 从小到大遍历且只在严格更优时替换，适应度相同时保留较小的 C（与 SVR.m 偏好小 C 一致）。"""
    cfg = problem.config
    ln_c, ln_gamma = ZHQ_grid_axis(cfg.ln_c, cfg.grid_step), ZHQ_grid_axis(cfg.ln_gamma, cfg.grid_step)
    surface = np.empty((len(ln_c), len(ln_gamma)))
    best_value, best_index = np.inf, (0, 0)
    for i, c in enumerate(ln_c):
        for j, g in enumerate(ln_gamma):
            surface[i, j] = problem.cv_mse(dict(mask=problem.all_features(), C=float(np.exp(c)),
                                                gamma=float(np.exp(g)), epsilon=float(cfg.epsilon)))
            if surface[i, j] < best_value:
                best_value, best_index = float(surface[i, j]), (i, j)
    return dict(ln_c=ln_c, ln_gamma=ln_gamma, surface=surface, best_value=best_value, best_index=best_index)


def ZHQ_default_parameters(problem: ZHQ_SVRProblem) -> dict:
    """libsvm 的默认参数：全部特征，C = 1，g = 1 / 特征数；ε 与网格搜索相同。"""
    return dict(mask=problem.all_features(), C=1.0, gamma=1.0 / problem.n_features,
                epsilon=float(problem.config.epsilon))
