
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from common_tools import CountedObjective
from config import normalize_algorithm
from ardo import ardo
from pso import pso
from gwo import gwo
from woa import woa
from pso_ardo import pso_ardo


def optimize(name, objective, initial, rng, iterations):
    name = normalize_algorithm(name)
    if name == "PSO_ARDO":
        return pso_ardo(objective, initial, rng, iterations)
    if name == "ARDO":
        return ardo(objective, initial, rng, iterations)
    if name == "PSO":
        return pso(objective, initial, rng, iterations)
    if name == "GWO":
        return gwo(objective, initial, rng, iterations)
    return woa(objective, initial, rng, iterations)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def code_hashes():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


def run_optimizers(problem, algorithms, population, iterations, seed, run_index, case):
    rng = np.random.default_rng(seed)
    initial = rng.uniform(problem.lower, problem.upper, (population, len(problem.lower)))
    state = copy.deepcopy(rng.bit_generator.state)
    initial_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    initial.setflags(write=False)
    budget = population * (iterations + 1)
    records = []
    for algorithm in algorithms:
        generator = np.random.default_rng()
        generator.bit_generator.state = copy.deepcopy(state)
        objective = CountedObjective(problem, problem.lower, problem.upper, budget)
        start = perf_counter()
        result = optimize(algorithm, objective, initial, generator, iterations)
        seconds = perf_counter() - start
        if result.evaluations != budget or objective.count != budget:
            raise AssertionError(f"{algorithm}：函数评价预算不一致")
        if result.curve.shape != (iterations + 1,) or not np.isfinite(result.curve).all():
            raise AssertionError(f"{algorithm}：收敛曲线无效")
        if np.any(np.diff(result.curve) > 0) or result.best_value != result.curve[-1]:
            raise AssertionError(f"{algorithm}：历史最优状态不一致")
        if hashlib.sha256(initial.tobytes()).hexdigest() != initial_hash:
            raise AssertionError(f"{algorithm}：修改了共用的初始种群")
        if (not np.isfinite(result.best_position).all()
                or np.any(result.best_position < problem.lower)
                or np.any(result.best_position > problem.upper)):
            raise AssertionError(f"{algorithm}：最优位置无效")
        record = dict(case=case, algorithm=algorithm, run=run_index, seed=seed,
                      fitness=float(result.best_value), seconds=seconds, evaluations=budget,
                      initial_hash=initial_hash, position=result.best_position, curve=result.curve,
                      diagnostics=result.diagnostics)
        records.append(record)
    return records


def output_directory(project, prefix, requested=None):
    parent = Path(requested).resolve() if requested else project
    path = parent / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "partial").mkdir(exist_ok=True)
    return path


def execute(worker, tasks, jobs, output, metadata):

    if jobs < 1:
        raise ValueError("--jobs 必须大于等于 1")
    jobs = min(jobs, len(tasks))
    metadata = dict(metadata, jobs=jobs, status="running", code_sha256=code_hashes())
    write_json(output / "metadata.json", metadata)
    collected = {}
    clock = perf_counter()
    print(f"结果目录：{output}\n任务 {len(tasks)} 组，并行进程 {jobs}；每组算法共用初始种群。", flush=True)

    def store(index, records):
        collected[index] = records
        write_json(output / "partial" / f"task_{index:05d}.json", records)
        print(f"[{len(collected)}/{len(tasks)}] {records[0]['case']}，运行 {records[0]['run'] + 1} 完成", flush=True)

    try:
        if jobs == 1:
            for index, task in enumerate(tasks):
                store(index, worker(*task))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(worker, *task): index for index, task in enumerate(tasks)}
                for future in as_completed(futures):
                    store(futures[future], future.result())
        records = [record for index in range(len(tasks)) for record in collected[index]]
        metadata.update(status="writing_report", elapsed_seconds=perf_counter() - clock)
        from reporting import write_report
        write_report(records, output, metadata)
        metadata.update(status="complete", elapsed_seconds=perf_counter() - clock)
        write_json(output / "metadata.json", metadata)
        print(f"报告：{output / 'report.md'}", flush=True)
        return records
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}",
                        completed_tasks=len(collected), elapsed_seconds=perf_counter() - clock)
        write_json(output / "metadata.json", metadata)
        raise


