
from datetime import datetime
import json
import platform
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from experiment import ZHQ_write_json

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ZH_CURVE_COLORS = {"PSO_ARDO": "tab:purple", "ARDO": "tab:orange", "GWO": "tab:green",
               "WOA": "tab:red", "PSO": "tab:blue"}


def ZHQ_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def ZHQ_number(value):
    return "NA" if not np.isfinite(value) else f"{value:.6g}"


def ZHQ_paired_statistics(subject, opponent):

    subject, opponent = np.asarray(subject, float), np.asarray(opponent, float)
    if subject.shape != opponent.shape or subject.size == 0:
        raise ValueError("配对样本必须具有相同且非空的形状")
    difference = subject - opponent
    nonzero = difference[difference != 0]
    if not nonzero.size:
        p_value, effect = 1.0, 0.0
    else:
        ranks = rankdata(np.abs(nonzero), method="average")
        effect = float(-np.sum(np.sign(nonzero) * ranks) / ranks.sum())
        if len(subject) < 5:
            p_value = float("nan")                
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = wilcoxon(difference, alternative="two-sided", zero_method="wilcox", method="approx")
            p_value = float(result.pvalue)
    return dict(p_raw=p_value, median_improvement=float(np.median(-difference)),
                mean_improvement=float(np.mean(-difference)), rank_biserial=effect,
                wins=int(np.sum(difference < 0)), ties=int(np.sum(difference == 0)),
                losses=int(np.sum(difference > 0)), pairs=len(subject))


def ZHQ_holm_adjust(values):
    values = np.asarray(values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid], kind="stable")]
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def ZHQ_comparisons(frame, algorithms):
    rows = []
    if "PSO_ARDO" not in algorithms:
        return pd.DataFrame()
    for case, group in frame.groupby("case", sort=False):
        left = group[group.algorithm == "PSO_ARDO"].set_index(["run", "seed"]).sort_index()
        per_function = []
        for other in algorithms:
            if other == "PSO_ARDO":
                continue
            right = group[group.algorithm == other].set_index(["run", "seed"]).sort_index()
            if (not left.index.is_unique or not left.index.equals(right.index)
                    or not left.initial_hash.equals(right.initial_hash)):
                raise ValueError(f"{case}: the run seeds or initial populations do not match")
            stats = ZHQ_paired_statistics(left.fitness.to_numpy(), right.fitness.to_numpy())
            per_function.append(dict(case=case, subject="PSO_ARDO", opponent=other, **stats))
        for row, p in zip(per_function, ZHQ_holm_adjust([row["p_raw"] for row in per_function])):
            if not np.isfinite(p):
                conclusion = "未推断"
            elif p >= 0.05:
                conclusion = "不显著"
            else:
                conclusion = "显著优" if row["rank_biserial"] > 0 else "显著劣"
            rows.append(dict(row, p_holm=p, conclusion=conclusion))
    return pd.DataFrame(rows)


def ZHQ_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def ZHQ_save_curves(case, function_name, records, algorithms, population, output):

    fig, axis = plt.subplots(figsize=(8, 4.8))
    summary_rows, run_frames = [], []
    for name in algorithms:
        pool = [r for r in records if r["algorithm"] == name]
        curves = np.stack([r["curve"] for r in pool])
        iterations = np.arange(curves.shape[1])
        evaluations = (iterations + 1) * population
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1) if len(pool) > 1 else np.full(len(mean), np.nan)
        summary_rows.append(pd.DataFrame(dict(algorithm=name, iteration=iterations, evaluations=evaluations,
                                              mean_best_fitness=mean, std_best_fitness=std, runs=len(pool))))
        for r in pool:
            run_frames.append(pd.DataFrame(dict(algorithm=name, run=r["run"], seed=r["seed"],
                                                iteration=iterations, evaluations=evaluations, fitness=r["curve"])))
        axis.plot(evaluations, np.maximum(mean, 1e-30), label=name, color=ZH_CURVE_COLORS[name],
                  linewidth=2.3 if name == "PSO_ARDO" else 1.5)
    axis.set(xlabel="函数评价次数", ylabel="历史最优函数值均值",
             title=f"{case} · {function_name}", yscale="log")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / f"{case}.png", dpi=180)
    plt.close(fig)
    for rows, suffix in ((summary_rows, "mean"), (run_frames, "runs")):
        curve_frame = pd.concat(rows, ignore_index=True)
        curve_frame = curve_frame.assign(case=case, function_name=function_name)
        ZHQ_csv(curve_frame, output / f"{case}_{suffix}.csv")


def ZHQ_write_report(records, output, metadata):
    cfg, algorithms = metadata["config"], metadata["algorithms"]
    functions = {p["code"]: p["name"] for p in metadata["problems"]}
    metadata["versions"] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                pandas=pd.__version__, matplotlib=matplotlib.__version__)
    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ("position", "curve")}
                          for r in records])
    frame["function_name"] = frame["case"].map(functions)
    ZHQ_csv(frame, output / "runs.csv")
    summary = frame.groupby(["case", "function_name", "algorithm"], sort=False).agg(
        mean=("fitness", "mean"), std=("fitness", "std"), best=("fitness", "min"),
        median=("fitness", "median"), worst=("fitness", "max"), seconds=("seconds", "mean"),
        evaluations=("evaluations", "mean")
    ).reset_index()
    summary["rank"] = summary.groupby("case")["mean"].rank(method="average")
    ZHQ_csv(summary, output / "summary.csv")
    ranking = (summary.groupby("algorithm", sort=False)["rank"].mean().sort_values(kind="stable")
               .rename("average_rank").reset_index())
    tests = ZHQ_comparisons(frame, algorithms)

    budget = cfg["population"] * (1 + cfg["iterations"])
    lines = [f"# {len(functions)} 个基准函数的优化器对比", "",
             f"本次函数 {', '.join(functions)}，方法 {', '.join(algorithms)}。",
             f"维度 {cfg['dimension']}，种群 {cfg['population']}，迭代 {cfg['iterations']} 代，"
             f"每个函数重复 {cfg['runs']} 次。每个方法每次评价 {budget} 次，含初始种群。",
             "同一函数、同一次重复共用初始种群。收敛图按累计函数评价次数对齐，纵轴为历史最优函数值的运行均值，越小越好。"
             "对数图的显示下限为 1e-30，CSV 和统计检验保留原始数值。", ""]
    lines += ["## 跨函数排名", "", "在每个函数内按最终均值排名，再对各函数等权平均，名次越小越好。", "",
              ZHQ_table(["方法", "平均名次"], [[r.algorithm, ZHQ_number(r.average_rank)] for r in ranking.itertuples()]), ""]
    if len(algorithms) >= 3 and len(functions) >= 3:
        means = summary.pivot(index="case", columns="algorithm", values="mean")[algorithms].to_numpy()
        if np.all(np.ptp(means, axis=1) == 0):
            statistic, p = 0.0, 1.0
        else:
            statistic, p = friedmanchisquare(*means.T)
        ZHQ_write_json(output / "friedman.json", dict(statistic=statistic, p_value=p, blocks=list(functions)))
        lines += [f"以函数为区组的 Friedman 总体检验 p = {ZHQ_number(p)}。", ""]
    if not tests.empty:
        counts = pd.crosstab(tests.opponent, tests.conclusion).reindex(
            index=[a for a in algorithms if a != "PSO_ARDO"],
            columns=["显著优", "不显著", "显著劣", "未推断"], fill_value=0)
        lines += [ZHQ_table(["PSO_ARDO 相对对手", *counts.columns], [[name, *row] for name, row in counts.iterrows()]),
                  "", "双侧 Wilcoxon 符号秩检验按同种子配对，每个函数内对各对手用 Holm 校正（α = 0.05）。", ""]
    curves_dir = output / "curves"
    curves_dir.mkdir(exist_ok=True)
    for case, function_name in functions.items():
        selected = [r for r in records if r["case"] == case]
        ZHQ_save_curves(case, function_name, selected, algorithms, cfg["population"], curves_dir)
        rows = summary[summary.case == case].sort_values("mean", kind="stable")
        lines += [f"## {case} · {function_name}", "",
                  ZHQ_table(["方法", "最终均值", "标准差", "最好", "中位数", "最差"],
                        [[r.algorithm, *[ZHQ_number(v) for v in (r.mean, r.std, r.best, r.median, r.worst)]]
                         for r in rows.itertuples()]), "", f"![{case}](curves/{case}.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def ZHQ_regenerate_report(source, parent=None):

    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") != "benchmark" or metadata.get("status") != "complete":
        raise ValueError("请选择已完成的基准函数实验目录")
    cfg = metadata["config"]
    algorithms = metadata["algorithms"]
    records = []
    for path in sorted((source / "partial").glob("task_*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    expected = {(case, run, name) for case in cfg["functions"]
                for run in range(cfg["runs"]) for name in algorithms}
    actual = {(r["case"], r["run"], r["algorithm"]) for r in records}
    if expected != actual or len(records) != len(expected):
        raise ValueError("逐次运行记录缺失或重复，请检查原结果的 partial 目录")
    output = (parent or source.parent) / f"results_function_report_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    ZHQ_write_report(records, output, metadata)
    print(f"报告：{(output / 'report.md').resolve()}")


