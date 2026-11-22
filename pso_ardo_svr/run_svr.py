"""UCI 回归数据集上用 PSO_ARDO、ARDO、PSO 联合寻优 SVR 的特征子集与参数（C、g、ε），
并与传统 SVR（网格搜索）、默认参数 SVR 对比。"""
import os

# 每个进程只用 1 个 BLAS/OpenMP 线程，并行交给多进程；必须在导入 numpy 之前设置（子进程会继承）
for ZH_THREAD_VARIABLE in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[ZH_THREAD_VARIABLE] = "1"

import argparse
from dataclasses import asdict, replace
import faulthandler
import hashlib
import json
from pathlib import Path

from config import ZH_PARAMS
from pso_ardo import ZH_PARAMETER_RECORD
from svr_experiment import (ZH_COMPUTE_FILES, ZHQ_code_hashes, ZHQ_execute, ZHQ_output_directory,
                            ZHQ_task_list)
from svr_problem import (ZH_AUTO_MPG_TARGETS, ZH_BASELINES, ZH_COLUMN_TITLES, ZH_DATASET_FILES,
                         ZH_DATASET_TITLES, ZH_ROOT, ZH_SVR_ALGORITHMS, ZHQ_SVRConfig, ZHQ_SVRProblem,
                         ZHQ_grid_axis, ZHQ_normalize_algorithm, ZHQ_read_superconductivity)


ZH_OVERRIDES = ("dataset", "population", "iterations", "runs", "folds", "seed", "target", "n_samples",
                "sample_seed", "split", "split_seed", "cv_seed", "epsilon", "grid_step", "max_iter")


def ZHQ_parser():
    defaults = ZHQ_SVRConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(ZH_DATASET_FILES),
                        help="auto_mpg（默认，398 个样本、7 个特征）或 superconductivity（81 个特征）")
    parser.add_argument("--quick", action="store_true",
                        help="冒烟测试：种群 6、迭代 5、重复 2 次、3 折、网格步长 2、Superconductivity 抽 300 条"
                             "（其余参数仍可单独指定）")
    parser.add_argument("--algorithms", nargs="+", type=ZHQ_normalize_algorithm, choices=ZH_SVR_ALGORITHMS,
                        default=list(ZH_SVR_ALGORITHMS), help="参与对比的群智能算法")
    parser.add_argument("--population", type=int, help=f"种群规模，默认 {defaults.population}")
    parser.add_argument("--iterations", type=int, help=f"迭代次数，默认 {defaults.iterations}")
    parser.add_argument("--runs", type=int, help=f"独立重复次数，默认 {defaults.runs}")
    parser.add_argument("--folds", type=int, help=f"交叉验证折数，默认 {defaults.folds}")
    parser.add_argument("--seed", type=int, help=f"第 1 次重复的随机种子，第 r 次为 seed + r - 1，默认 {defaults.seed}")
    parser.add_argument("--no-feature-selection", action="store_true", help="不做特征选择，使用全部特征")
    parser.add_argument("--fix-epsilon", action="store_true",
                        help="ε 不参与寻优，固定为 --epsilon；与 --no-feature-selection 一起用即论文中只寻优 C、g 的设置")
    parser.add_argument("--epsilon", type=float,
                        help=f"固定的 ε（归一化目标值单位），默认 {defaults.epsilon}；网格搜索与默认参数 SVR 也用它")
    parser.add_argument("--target", choices=ZH_AUTO_MPG_TARGETS,
                        help="仅 Auto MPG：mpg（默认，UCI 定义的任务）；displacement 复现 MATLAB 程序以排量为输出的设置")
    parser.add_argument("--n-samples", type=int,
                        help=f"仅 Superconductivity：随机抽取的样本数，默认 {defaults.n_samples}")
    parser.add_argument("--sample-seed", type=int, help=f"仅 Superconductivity：抽样种子，默认 {defaults.sample_seed}")
    parser.add_argument("--split", choices=("matlab", "random"),
                        help="Auto MPG 默认 matlab（沿用 nnn.xlsx 的划分，前 319 个训练）；Superconductivity 只能 random")
    parser.add_argument("--split-seed", type=int, help=f"随机 80/20 划分的种子，默认 {defaults.split_seed}")
    parser.add_argument("--cv-seed", type=int, help=f"交叉验证分折的种子，默认 {defaults.cv_seed}")
    parser.add_argument("--grid-step", type=float, help=f"网格搜索 SVR 的步长（ln 尺度），默认 {defaults.grid_step}")
    parser.add_argument("--no-baselines", action="store_true", help="不运行网格搜索 SVR 与默认参数 SVR")
    parser.add_argument("--max-iter", type=int, help=f"libsvm 迭代上限，-1 为不设上限，默认 {defaults.max_iter}")
    parser.add_argument("--jobs", type=int, default=1, help="并行进程数，默认 1")
    parser.add_argument("--retries", type=int, default=3,
                        help="工作进程原生崩溃后自动重跑未完成任务的最多轮数，默认 3")
    parser.add_argument("--output", type=Path, help="结果的父目录，默认取本脚本所在目录")
    parser.add_argument("--resume", type=Path,
                        help="续跑未完成的结果目录：沿用原实验的全部参数，只运行 partial/ 中缺少的任务"
                             "（此时只有 --jobs、--retries 生效）")
    parser.add_argument("--report-from", type=Path, help="从已有结果目录的 partial/ 重新生成报告和图")
    return parser


def ZHQ_resume(parser, source, jobs, retries):
    path = source / "metadata.json"
    if not path.is_file():
        parser.error(f"找不到 {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "svr_uci" or "task_keys" not in metadata:
        parser.error("该目录不是当前版本 run_svr.py 生成的 SVR 结果目录，无法续跑")
    if metadata.get("status") == "complete":
        parser.error("该结果目录已经完成；只需重出报告时请用 --report-from")
    current = ZHQ_code_hashes()
    changed = [name for name in ZH_COMPUTE_FILES if metadata["code_sha256"].get(name) != current.get(name)]
    if changed:
        parser.error(f"以下影响计算结果的文件与原实验不同，续跑会混入不同版本的结果：{', '.join(changed)}")
    config = ZHQ_SVRConfig(**{key: tuple(value) if isinstance(value, list) else value
                              for key, value in metadata["config"].items()})
    metadata = {key: value for key, value in metadata.items()
                if key not in ("error", "completed_tasks", "elapsed_seconds")}
    metadata["resumed"] = int(metadata.get("resumed", 0)) + 1
    ZHQ_execute(ZHQ_task_list(config), jobs, source, metadata, retries)


def ZHQ_main():
    faulthandler.enable()
    parser = ZHQ_parser()
    args = parser.parse_args()
    if args.jobs < 1 or args.retries < 0:
        parser.error("要求 jobs >= 1，retries >= 0")
    if args.report_from is not None:
        from svr_reporting import ZHQ_regenerate_report
        ZHQ_regenerate_report(args.report_from.resolve(), args.output)
        return
    if args.resume is not None:
        ZHQ_resume(parser, args.resume.resolve(), args.jobs, args.retries)
        return

    base = ZHQ_SVRConfig().quick() if args.quick else ZHQ_SVRConfig()
    overrides = {name: getattr(args, name) for name in ZH_OVERRIDES if getattr(args, name) is not None}
    dataset_key = overrides.get("dataset", base.dataset)
    overrides.setdefault("split", "matlab" if dataset_key == "auto_mpg" else "random")
    config = replace(base, **overrides, algorithms=tuple(args.algorithms),
                     feature_selection=not args.no_feature_selection, optimize_epsilon=not args.fix_epsilon,
                     baselines=not args.no_baselines)
    if config.population < 6 or min(config.iterations, config.runs, args.jobs) < 1 or config.folds < 2:
        parser.error("要求 population >= 6，iterations/runs/jobs >= 1，folds >= 2")
    if min(config.seed, config.split_seed, config.cv_seed, config.sample_seed) < 0:
        parser.error("seed、split-seed、cv-seed、sample-seed 必须非负")
    if not config.epsilon > 0 or not config.grid_step > 0:
        parser.error("epsilon 和 grid-step 必须为正数")
    if config.max_iter == 0 or config.max_iter < -1:
        parser.error("max-iter 必须为正整数或 -1")
    if len(set(config.algorithms)) != len(config.algorithms):
        parser.error("算法名称不能重复")
    if config.dataset != "auto_mpg" and (args.target is not None or config.split == "matlab"):
        parser.error("--target 与 matlab 划分只适用于 Auto MPG")

    try:
        problem = ZHQ_SVRProblem(config)    # 先构造一次，检查数据与划分，并记录到元数据
    except ValueError as error:
        parser.error(str(error))
    dataset = problem.dataset
    source = ZH_DATASET_FILES[config.dataset]
    grid_points = (len(ZHQ_grid_axis(config.ln_c, config.grid_step))
                   * len(ZHQ_grid_axis(config.ln_gamma, config.grid_step)))
    n_total = len(ZHQ_read_superconductivity()) if dataset.rows is not None else len(dataset.target)
    metadata = dict(
        kind="svr_uci", mode="smoke" if args.quick else "full", config=asdict(config),
        algorithms=list(config.algorithms), baselines=list(ZH_BASELINES) if config.baselines else [],
        algorithm_parameters=asdict(ZH_PARAMS), proposed_parameters=ZH_PARAMETER_RECORD,
        budget=dict(evaluations_per_run=config.evaluations, grid_points=grid_points if config.baselines else 0),
        search_space=dict(dimension=problem.dimension, feature_bits=problem.mask_dims,
                          continuous=list(problem.continuous_names), lower=problem.lower, upper=problem.upper),
        dataset=dict(key=dataset.key, title=dataset.title, source=str(source),
                     sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     n_total=n_total, n_samples=len(dataset.target), sampled_rows=dataset.rows,
                     target=dataset.target_name, target_title=ZH_COLUMN_TITLES.get(dataset.target_name,
                                                                                    dataset.target_name),
                     features=list(dataset.feature_names), missing=dataset.missing,
                     relative_metrics=dataset.relative_metrics,
                     train_index=problem.train_index, test_index=problem.test_index,
                     cv_indices=problem.fold_indices, test_target=problem.test_y,
                     fitness_scale=problem.fitness_scale))
    output = ZHQ_output_directory(args.output or ZH_ROOT,
                                  f"results_svr_{config.dataset}" + ("_smoke" if args.quick else ""))
    print(f"{ZH_DATASET_TITLES[config.dataset]}：目标 {dataset.target_name}，训练 {len(problem.train_y)} 个、"
          f"测试 {len(problem.test_y)} 个；搜索空间 {problem.dimension} 维；"
          f"每个算法每次重复评价 {config.evaluations} 次"
          + (f"；网格搜索 {grid_points} 个点" if config.baselines else ""), flush=True)
    ZHQ_execute(ZHQ_task_list(config), args.jobs, output, metadata, args.retries)


if __name__ == "__main__":
    ZHQ_main()
