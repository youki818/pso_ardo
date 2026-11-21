import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from common_tools import ZHQ_CountedObjective
from ardo import ZHQ_ardo
from pso import ZHQ_pso
from gwo import ZHQ_gwo
from woa import ZHQ_woa
from pso_ardo import ZHQ_pso_ardo


ZH_OPTIMIZERS = {
    "PSO_ARDO": ZHQ_pso_ardo,
    "ARDO": ZHQ_ardo,
    "PSO": ZHQ_pso,
    "GWO": ZHQ_gwo,
    "WOA": ZHQ_woa,
}


def ZHQ_json_ready(value):
    if isinstance(value, np.ndarray):
        return ZHQ_json_ready(value.tolist())
    if isinstance(value, np.generic):
        return ZHQ_json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): ZHQ_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ZHQ_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def ZHQ_write_json(path, value):
    path.write_text(json.dumps(ZHQ_json_ready(value), ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def ZHQ_code_hashes():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob("*.py"))}


def ZHQ_run_optimizers(problem, population, iterations, seed, run_index, case):
    rng = np.random.default_rng(seed)
    initial = rng.uniform(problem.lower, problem.upper, (population, len(problem.lower)))
    state = copy.deepcopy(rng.bit_generator.state)
    initial_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    initial.setflags(write=False)
    records = []
    for algorithm, optimize in ZH_OPTIMIZERS.items():
        generator = np.random.default_rng()
        generator.bit_generator.state = copy.deepcopy(state)
        objective = ZHQ_CountedObjective(problem, problem.lower, problem.upper)
        start = perf_counter()
        result = optimize(objective, initial, generator, iterations)
        seconds = perf_counter() - start
        records.append(dict(case=case, algorithm=algorithm, run=run_index, seed=seed,
                            fitness=float(result.best_value), seconds=seconds,
                            evaluations=result.evaluations, initial_hash=initial_hash,
                            position=result.best_position, curve=result.curve))
    return records


def ZHQ_output_directory(project, prefix, requested=None):
    parent = Path(requested).resolve() if requested else project
    path = parent / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "partial").mkdir(exist_ok=True)
    return path


def ZHQ_execute(worker, tasks, jobs, output, metadata):
    jobs = min(jobs, len(tasks))
    metadata = dict(metadata, jobs=jobs, status="running", code_sha256=ZHQ_code_hashes())
    ZHQ_write_json(output / "metadata.json", metadata)
    collected = {}
    clock = perf_counter()
    print(f"结果目录：{output}\n任务 {len(tasks)} 组，并行进程 {jobs}；每组算法共用初始种群。", flush=True)

    def store(index, records):
        collected[index] = records
        ZHQ_write_json(output / "partial" / f"task_{index:05d}.json", records)
        print(f"[{len(collected)}/{len(tasks)}] {records[0]['case']}，运行 {records[0]['run'] + 1} 完成", flush=True)

    try:
        if jobs == 1:
            for index, job in enumerate(tasks):
                store(index, worker(*job))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(worker, *job): index for index, job in enumerate(tasks)}
                for future in as_completed(futures):
                    store(futures[future], future.result())
        records = [record for index in range(len(tasks)) for record in collected[index]]
        metadata.update(status="writing_report", elapsed_seconds=perf_counter() - clock)
        from reporting import ZHQ_write_report
        ZHQ_write_report(records, output, metadata)
        metadata.update(status="complete", elapsed_seconds=perf_counter() - clock)
        ZHQ_write_json(output / "metadata.json", metadata)
        print(f"报告：{output / 'report.md'}", flush=True)
        return records
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}",
                        completed_tasks=len(collected), elapsed_seconds=perf_counter() - clock)
        ZHQ_write_json(output / "metadata.json", metadata)
        raise
