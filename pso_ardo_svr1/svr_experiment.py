"""SVR 寻优实验的调度。

每个任务是“一次重复 × 一个算法”，另有一个基线任务。同一次重复的各算法都由 seed + run 重建初始种群和
随机数状态，所以拆成独立任务后，与在同一进程里依次运行完全等价。任务一完成就写入 partial/<键>.json。
工作进程若发生原生崩溃（BrokenProcessPool），未完成的任务会在新的进程池中重跑：任务都带固定种子，
重跑结果与没有崩溃时相同。崩溃时的 Python 调用栈由 faulthandler 写入 crash_logs/。
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
import faulthandler
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np

from ardo import ZHQ_ardo
from common_tools import ZHQ_CountedObjective
from pso import ZHQ_pso
from pso_ardo import ZHQ_pso_ardo
from svr_problem import (ZH_DEFAULT, ZH_GRID, ZH_ROOT, ZHQ_SVRProblem, ZHQ_default_parameters,
                         ZHQ_grid_search)


ZH_OPTIMIZERS = {
    "PSO_ARDO": ZHQ_pso_ardo,
    "ARDO": ZHQ_ardo,
    "PSO": ZHQ_pso,
}
# 影响计算结果的文件；续跑时这些文件必须与原实验一致
ZH_COMPUTE_FILES = ("ardo.py", "common_tools.py", "config.py", "pso.py", "pso_ardo.py",
                    "svr_problem.py", "svr_experiment.py")
ZH_WORKER_STATE = {}


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
    # 在同一文件系统内替换，任务中断时只会看见上次完整的检查点。
    payload = json.dumps(ZHQ_json_ready(value), ensure_ascii=False, indent=2, allow_nan=False)
    handle = NamedTemporaryFile(dir=path.parent, suffix=".tmp", mode="w", encoding="utf-8", delete=False)
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def ZHQ_code_hashes():
    return {str(path.relative_to(ZH_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(ZH_ROOT.glob("*.py"))}


def ZHQ_record(problem, algorithm, run, seed, parameters, fitness, seconds, evaluations, stats,
               curve=None, initial_hash=None, position=None):
    """用最优参数在全部训练集上重训，整理成一条运行记录。stats 须在重训之前取得。"""
    evaluation = problem.evaluate(parameters)
    mask = np.asarray(parameters["mask"], dtype=bool)
    record = dict(case=problem.case, algorithm=algorithm, run=run, seed=seed, fitness=float(fitness),
                  selection_metric=problem.config.selection_metric,
                  seconds=float(seconds), evaluations=int(evaluations), initial_hash=initial_hash,
                  ln_C=float(np.log(parameters["C"])), ln_gamma=float(np.log(parameters["gamma"])),
                  ln_epsilon=float(np.log(parameters["epsilon"])),
                  C=parameters["C"], gamma=parameters["gamma"], epsilon=parameters["epsilon"],
                  n_selected=int(mask.sum()), n_support=evaluation["n_support"], **stats)
    for split in ("train", "test"):
        record.update({f"{split}_{name}": value for name, value in evaluation[split].items()})
    record.update(mask=mask.astype(int), position=None if position is None else np.asarray(position, dtype=float),
                  curve=None if curve is None else np.asarray(curve, dtype=float),
                  test_prediction=evaluation["test_prediction"])
    return record


def ZHQ_optimizer_task(config, run_index, algorithm):
    """第 run_index 次重复中的一个算法。初始种群和之后的随机数状态只由 seed + run_index 决定，
    因此同一次重复的各算法从同一个初始种群、同一个随机数状态出发。"""
    problem = ZHQ_SVRProblem(config)
    seed = config.seed + run_index
    rng = np.random.default_rng(seed)
    initial = rng.uniform(problem.lower, problem.upper, (config.population, problem.dimension))
    initial_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    initial.setflags(write=False)
    problem.reset_cache()
    objective = ZHQ_CountedObjective(problem, problem.lower, problem.upper)
    start = perf_counter()
    result = ZH_OPTIMIZERS[algorithm](objective, initial, rng, config.iterations)
    seconds = perf_counter() - start
    stats = problem.evaluation_stats()
    return [ZHQ_record(problem, algorithm, run_index, seed, problem.decode(result.best_position),
                       result.best_value, seconds, result.evaluations, stats, result.curve, initial_hash,
                       result.best_position)]


def ZHQ_baseline_task(config):
    """不经群智能寻优的 SVR，都使用全部特征、ε 固定：网格搜索（论文中的传统 SVR）与 libsvm 默认参数。
    两者是确定性的，只算一次。"""
    problem = ZHQ_SVRProblem(config)
    records = []

    problem.reset_cache()
    start = perf_counter()
    grid = ZHQ_grid_search(problem)
    seconds = perf_counter() - start
    stats = problem.evaluation_stats()
    i, j = grid["best_index"]
    parameters = dict(mask=problem.all_features(), C=float(np.exp(grid["ln_c"][i])),
                      gamma=float(np.exp(grid["ln_gamma"][j])), epsilon=float(config.epsilon))
    record = ZHQ_record(problem, ZH_GRID, 0, None, parameters, grid["best_value"], seconds,
                        grid["surface"].size, stats)
    record["grid"] = dict(ln_c=grid["ln_c"], ln_gamma=grid["ln_gamma"], surface=grid["surface"])
    records.append(record)

    problem.reset_cache()
    parameters = ZHQ_default_parameters(problem)
    start = perf_counter()
    fitness = problem.cv_score(parameters)
    seconds = perf_counter() - start
    stats = problem.evaluation_stats()
    records.append(ZHQ_record(problem, ZH_DEFAULT, 0, None, parameters, fitness, seconds, 1, stats))
    return records


def ZHQ_task_list(config):
    """[(键, 说明, 函数, 参数)]。键即 partial/ 中的文件名，续跑时据此跳过已完成的任务。"""
    tasks = []
    if config.baselines:
        tasks.append(("baseline", "基线：网格搜索 SVR + 默认参数 SVR", ZHQ_baseline_task, (config,)))
    for run in range(config.runs):
        for algorithm in config.algorithms:
            tasks.append((f"run{run + 1:03d}_{algorithm}", f"第 {run + 1} 次重复 · {algorithm}",
                          ZHQ_optimizer_task, (config, run, algorithm)))
    return tasks


def ZHQ_output_directory(parent, prefix, partial=True):
    path = Path(parent).resolve() / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    path.mkdir(parents=True, exist_ok=False)
    if partial:
        (path / "partial").mkdir()
    return path


def ZHQ_load_partial(output, tasks):
    keys = {key for key, *_ in tasks}
    return {path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output / "partial").glob("*.json")) if path.stem in keys}


def ZHQ_worker_init(log_dir):
    """工作进程启动时打开 faulthandler：发生访问冲突等原生崩溃时，把 Python 调用栈写入日志。"""
    handle = open(Path(log_dir) / f"worker_{os.getpid()}.log", "w", encoding="utf-8")
    ZH_WORKER_STATE["crash_log"] = handle
    faulthandler.enable(file=handle, all_threads=True)


def ZHQ_clean_crash_logs(log_dir):
    for path in log_dir.glob("worker_*.log"):
        try:
            if path.stat().st_size == 0:
                path.unlink()
        except OSError:
            pass
    try:
        log_dir.rmdir()             # 只在目录已空时成功
    except OSError:
        pass


def ZHQ_run_pool(pending, jobs, log_dir, store):
    """用一个新进程池跑完 pending；返回是否有工作进程崩溃。任务内的 Python 异常直接抛出（确定性错误，重跑无用）。"""
    broken = False
    with ProcessPoolExecutor(max_workers=min(jobs, len(pending)), initializer=ZHQ_worker_init,
                             initargs=(str(log_dir),)) as pool:
        futures = {pool.submit(worker, *arguments): key for key, _, worker, arguments in pending}
        for future in as_completed(futures):
            try:
                records = future.result()
            except BrokenProcessPool:
                broken = True
                continue
            store(futures[future], records)
    return broken


def ZHQ_execute(tasks, jobs, output, metadata, retries=3):
    """运行 tasks 中 partial/ 里还没有结果的任务，全部完成后生成报告。"""
    done = ZHQ_load_partial(output, tasks)
    jobs = max(1, min(jobs, len(tasks)))
    labels = {key: label for key, label, *_ in tasks}
    metadata = dict(metadata, jobs=jobs, retries=retries, status="running",
                    task_keys=[key for key, *_ in tasks], code_sha256=ZHQ_code_hashes())
    ZHQ_write_json(output / "metadata.json", metadata)
    log_dir = output / "crash_logs"
    log_dir.mkdir(exist_ok=True)
    crashes = 0
    clock = perf_counter()
    print(f"结果目录：{output}\n任务 {len(tasks)} 个（每个为一次重复 × 一个算法），并行进程 {jobs}；"
          "同一次重复内各算法共用初始种群。", flush=True)
    if done:
        print(f"续跑：partial/ 中已有 {len(done)} 个任务的结果，跳过这些任务。", flush=True)

    def store(key, records):
        ZHQ_write_json(output / "partial" / f"{key}.json", records)
        done[key] = records
        print(f"[{len(done)}/{len(tasks)}] {labels[key]} 完成，累计 {perf_counter() - clock:.1f} s", flush=True)

    try:
        while True:
            pending = [task for task in tasks if task[0] not in done]
            if not pending:
                break
            if jobs == 1:
                for key, _, worker, arguments in pending:
                    store(key, worker(*arguments))
                continue
            if not ZHQ_run_pool(pending, jobs, log_dir, store):
                continue
            crashes += 1
            remaining = len(tasks) - len(done)
            if crashes > retries:
                raise RuntimeError(f"工作进程已崩溃 {crashes} 次，仍有 {remaining} 个任务未完成；"
                                   f"已完成的结果保存在 partial/，可用 --resume 续跑")
            print(f"警告：有工作进程异常退出（原生崩溃，调用栈见 {log_dir}）。已完成 {len(done)}/{len(tasks)}，"
                  f"第 {crashes}/{retries} 次重试其余 {remaining} 个任务；任务带固定种子，重跑结果与不崩溃时相同。",
                  flush=True)
        ZHQ_clean_crash_logs(log_dir)
        records = [record for key, *_ in tasks for record in done[key]]
        metadata.update(status="writing_report", worker_crashes=crashes, elapsed_seconds=perf_counter() - clock)
        from svr_reporting import ZHQ_write_report
        ZHQ_write_report(records, output, metadata)
        metadata.update(status="complete", elapsed_seconds=perf_counter() - clock)
        ZHQ_write_json(output / "metadata.json", metadata)
        print(f"报告：{output / 'report.md'}", flush=True)
        return records
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}", worker_crashes=crashes,
                        completed_tasks=len(done), elapsed_seconds=perf_counter() - clock)
        ZHQ_write_json(output / "metadata.json", metadata)
        raise
