from runtime import ZHQ_configure_threads
ZHQ_configure_threads()

import argparse
from dataclasses import asdict
from pathlib import Path

from function import ZH_DEFINITIONS, ZH_SUITE, ZHQ_make_benchmark
from experiment import ZH_OPTIMIZERS, ZHQ_execute, ZHQ_output_directory, ZHQ_run_optimizers
from pso_ardo import ZH_PARAMETER_RECORD


def ZHQ_task(settings, code, run_index):
    problem = ZHQ_make_benchmark(code, settings["dimension"], settings["shift_seed"],
                                 settings["shifted"])
    return ZHQ_run_optimizers(problem, settings["population"],
                              settings["iterations"], settings["seed"] + run_index, run_index, code)


def ZHQ_main():
    parser = argparse.ArgumentParser(description="F1–F6 基准函数优化实验")
    parser.add_argument("--functions", nargs="+", type=str.lower, choices=tuple(ZH_DEFINITIONS),
                        default=list(ZH_DEFINITIONS))
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--dimension", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--shift-seed", type=int, default=731029)
    parser.add_argument("--no-shift", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, help="结果的父目录，默认取当前脚本所在目录")
    parser.add_argument("--report-from", type=Path, help="从已完成的实验结果重新生成图表和报告")
    args = parser.parse_args()
    if args.report_from is not None:
        from reporting import ZHQ_regenerate_report
        ZHQ_regenerate_report(args.report_from.resolve(), args.output)
        return

    if args.population < 6 or args.dimension < 2 or min(args.iterations, args.runs, args.jobs) < 1:
        parser.error("population >= 6, dimension >= 2; iterations/runs/jobs >= 1")
    if len(set(args.functions)) != len(args.functions):
        parser.error("函数名称不能重复")
    if min(args.seed, args.shift_seed) < 0:
        parser.error("seed 和 shift-seed 必须非负")
    settings = dict(population=args.population, dimension=args.dimension, iterations=args.iterations,
                    runs=args.runs, suite=ZH_SUITE, algorithms=list(ZH_OPTIMIZERS), seed=args.seed,
                    shift_seed=args.shift_seed, shifted=not args.no_shift, functions=args.functions)
    problems = [ZHQ_make_benchmark(code, args.dimension, args.shift_seed, not args.no_shift)
                for code in args.functions]
    metadata = dict(kind="benchmark", config=settings, algorithms=settings["algorithms"],
                    proposed_parameters=ZH_PARAMETER_RECORD,
                    problems=[{k: v for k, v in asdict(p).items() if k != "formula"} for p in problems])
    output = ZHQ_output_directory(Path(__file__).resolve().parent, f"results_function_{ZH_SUITE}", args.output)
    tasks = [(settings, code, r) for code in args.functions for r in range(args.runs)]
    ZHQ_execute(ZHQ_task, tasks, args.jobs, output, metadata)


if __name__ == "__main__":
    ZHQ_main()
