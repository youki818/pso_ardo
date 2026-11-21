
from runtime import configure_threads
configure_threads()

import argparse
from dataclasses import asdict
from pathlib import Path

from function import DEFINITIONS, make_benchmark
from experiment import execute, output_directory, run_optimizers
from config import ALGORITHMS, PROPOSED, normalize_algorithm
from pso_ardo import PARAMETER_RECORD


def task(settings, code, run_index):
    problem = make_benchmark(code, settings["dimension"], settings["shift_seed"],
                             settings["shifted"])
    return run_optimizers(problem, settings["algorithms"], settings["population"],
                          settings["iterations"], settings["seed"] + run_index, run_index, code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="在所选函数上冒烟：N=8、D=6、T=8、2 次")
    parser.add_argument("--suite", choices=("selected6",), default="selected6",
                        help="本次保留的六个基准函数，编号 f1–f6")
    parser.add_argument("--algorithms", nargs="+", type=normalize_algorithm, choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--functions", nargs="+", type=str.lower, choices=tuple(DEFINITIONS), default=None)
    for name in ("population", "dimension", "iterations", "runs"):
        parser.add_argument(f"--{name}", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--shift-seed", type=int, default=731029)
    parser.add_argument("--no-shift", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, help="结果的父目录，默认取当前脚本所在目录")
    parser.add_argument("--report-from", type=Path, help="从已完成的基准结果重新生成图表和报告")
    args = parser.parse_args()
    if args.report_from is not None:
        from reporting import regenerate_report
        regenerate_report(args.report_from.resolve(), args.output)
        return

    defaults = (8, 6, 8, 2) if args.quick else (20, 30, 50, 16)
    settings = {name: getattr(args, name) if getattr(args, name) is not None else default
                for name, default in zip(("population", "dimension", "iterations", "runs"), defaults)}
    if settings["population"] < 6 or settings["dimension"] < 2 or min(settings["iterations"], settings["runs"], args.jobs) < 1:
        parser.error("population >= 6, dimension >= 2; iterations/runs/jobs >= 1")
    if len(set(args.algorithms)) != len(args.algorithms):
        parser.error("算法名称不能重复")
    if min(args.seed, args.shift_seed) < 0:
        parser.error("seed 和 shift-seed 必须非负")
    functions = args.functions or list(DEFINITIONS)
    if len(set(functions)) != len(functions):
        parser.error("函数名称不能重复")
    settings.update(suite=args.suite, algorithms=args.algorithms, seed=args.seed, shift_seed=args.shift_seed,
                    shifted=not args.no_shift, functions=functions)
    problems = [make_benchmark(code, settings["dimension"], args.shift_seed, not args.no_shift)
                for code in functions]
    metadata = dict(kind="benchmark", mode="smoke" if args.quick else "full", config=settings,
                    algorithms=args.algorithms, proposed_parameters=PARAMETER_RECORD,
                    optimizer_origin={PROPOSED: "PSO-ARDO3-share0", "ARDO": "ARDO"},
                    problems=[{k: v for k, v in asdict(p).items() if k != "formula"} for p in problems])
    prefix = f"results_function_{args.suite}" + ("_smoke" if args.quick else "")
    output = output_directory(Path(__file__).resolve().parent, prefix, args.output)
    tasks = [(settings, code, r) for code in functions for r in range(settings["runs"])]
    execute(task, tasks, args.jobs, output, metadata)


if __name__ == "__main__":
    main()


