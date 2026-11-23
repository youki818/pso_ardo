"""SVR 实验报告：逐次记录与汇总 CSV、配对检验、图和结果分析。

作图原则（避免误导）：
  * 收敛曲线只用一个坐标轴画全过程，同时画出各次重复的标准差带，不单独放大局部。把纵轴截到很窄的范围，
    会把远小于运行间波动的差距画成“明显分开”。最终差距用箱线图（含每次重复的散点）和配对检验来比较；
  * 收敛曲线画的是寻优目标（训练集交叉验证 MSE），不是测试误差；泛化性能另用测试集指标报告；
  * 测试样本之间没有先后顺序，预测效果用“预测值–真实值”散点图加 y = x 参考线表示，不把样本连成折线。
"""
from datetime import datetime
import json
from pathlib import Path
import platform
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import rankdata, spearmanr, wilcoxon

from svr_experiment import ZHQ_write_json
from svr_problem import ZH_BASELINES, ZH_COLUMN_TITLES, ZH_DEFAULT, ZH_GRID


ZH_PROPOSED = "PSO_ARDO"
ZH_STYLES = {
    "PSO_ARDO": ("PSO_ARDO_SVR", "#cf2426", "D"),
    "ARDO": ("ARDO_SVR", "#ec781c", "s"),
    "PSO": ("PSO_SVR", "#2673bd", "o"),
    ZH_GRID: ("SVR（网格搜索）", "#555555", "*"),
    ZH_DEFAULT: ("SVR（默认参数）", "#999999", "P"),
}
ZH_HIDDEN_FIELDS = ("position", "curve", "test_prediction", "grid", "mask")
ZH_SUMMARY_METRICS = ("fitness", "train_rmse", "train_r2", "train_mre", "train_msre",
                      "test_rmse", "test_mae", "test_r2", "test_mre", "test_msre", "n_selected",
                      "ln_C", "ln_gamma", "ln_epsilon", "n_support", "seconds", "evaluations",
                      "unique_cv_evaluations", "svr_fits", "convergence_warnings")
ZH_FIGURE_DPI = 300


def ZHQ_configure_fonts():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    chinese = [name for name in ("SimSun", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if name in installed]
    serif = ["Times New Roman"] if "Times New Roman" in installed else []
    plt.rcParams.update({"font.family": [*serif, *chinese, "DejaVu Serif"], "mathtext.fontset": "stix",
                         "axes.unicode_minus": False, "font.size": 9, "axes.labelsize": 9.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.linewidth": 0.65})


def ZHQ_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def ZHQ_number(value, digits=6):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def ZHQ_mean_std(mean, std, digits=4):
    if std is None or not np.isfinite(std):
        return ZHQ_number(mean, digits)
    return f"{ZHQ_number(mean, digits)} ± {ZHQ_number(std, 2)}"


def ZHQ_difference(value):
    """value 为相对差（正数表示 PSO_ARDO 更小），写成“低 x%”或“高 x%”。"""
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{'低' if value >= 0 else '高'} {abs(value) * 100:.2f}%"


def ZHQ_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def ZHQ_save(fig, stem):
    try:
        fig.savefig(stem.with_suffix(".png"), dpi=ZH_FIGURE_DPI, facecolor="white")
        fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    finally:
        plt.close(fig)


def ZHQ_style_axis(ax):
    ax.grid(color="#dddddd", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)


# ---------------- 统计 ----------------
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


def ZHQ_conclusion(p_value, effect):
    if not np.isfinite(p_value):
        return "未推断"
    if p_value >= 0.05:
        return "不显著"
    return "显著优" if effect > 0 else "显著劣"


def ZHQ_by_run(frame, algorithms, column):
    """{算法: 以重复编号为索引的 Series}，并核对同一次重复内各算法的初始种群确实相同。"""
    series, reference = {}, None
    for name in algorithms:
        block = frame[frame.algorithm == name].set_index("run").sort_index()
        if not block.index.is_unique:
            raise ValueError(f"{name} 的重复编号有重复")
        if reference is None:
            reference = block.initial_hash
        elif not block.index.equals(reference.index) or not block.initial_hash.equals(reference):
            raise ValueError(f"{name} 与其他算法的重复编号或初始种群不一致，无法配对")
        series[name] = block[column].astype(float)
    return series


def ZHQ_paired_tests(values, algorithms, metric, title):
    """PSO_ARDO 与每个对手按同一次重复配对做双侧 Wilcoxon 符号秩检验，同一指标内 Holm 校正。"""
    if ZH_PROPOSED not in values or len(algorithms) < 2:
        return []
    left = values[ZH_PROPOSED]
    block = []
    for other in algorithms:
        if other == ZH_PROPOSED:
            continue
        right = values[other]
        if not left.index.equals(right.index):
            raise ValueError(f"{ZH_PROPOSED} 与 {other} 的重复编号不一致")
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = (right.to_numpy() - left.to_numpy()) / np.abs(right.to_numpy())
        block.append(dict(metric=metric, metric_title=title, subject=ZH_PROPOSED, opponent=other,
                          median_relative_improvement=float(np.nanmedian(relative)),
                          **ZHQ_paired_statistics(left.to_numpy(), right.to_numpy())))
    for row, p_value in zip(block, ZHQ_holm_adjust([row["p_raw"] for row in block])):
        row.update(p_holm=p_value, conclusion=ZHQ_conclusion(p_value, row["rank_biserial"]))
    return block


def ZHQ_curve_matrix(records, name):
    pool = sorted((r for r in records if r["algorithm"] == name), key=lambda r: r["run"])
    return [r["run"] for r in pool], np.stack([np.asarray(r["curve"], dtype=float) for r in pool])


def ZHQ_checkpoints(iterations):
    return sorted({max(1, round(iterations / 4)), max(1, round(iterations / 2)), iterations})


def ZHQ_summary(frame, methods):
    rows = []
    for name in methods:
        block = frame[frame.algorithm == name]
        if block.empty:
            continue
        fitness = block.fitness.to_numpy(float)
        row = dict(algorithm=name, method=ZH_STYLES[name][0], runs=len(block),
                   fitness_best=float(fitness.min()), fitness_median=float(np.median(fitness)),
                   fitness_worst=float(fitness.max()))
        for metric in ZH_SUMMARY_METRICS:
            if metric not in block.columns:
                continue
            values = block[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def ZHQ_representative(records, name):
    """CV MSE 处于中位的那次重复，用来画代表性的预测图；基线只有一条记录。"""
    pool = sorted((r for r in records if r["algorithm"] == name), key=lambda r: (r["fitness"], r["run"]))
    return pool[(len(pool) - 1) // 2] if pool else None


# ---------------- 图 ----------------
def ZHQ_plot_convergence(records, algorithms, population, grid_record, curves_dir, stem):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    mean_frames, run_frames, runs = [], [], 0
    for name in algorithms:
        run_ids, curves = ZHQ_curve_matrix(records, name)
        runs = len(run_ids)
        iterations = np.arange(curves.shape[1])
        evaluations = (iterations + 1) * population
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1) if runs > 1 else np.zeros_like(mean)
        label, color, marker = ZH_STYLES[name]
        ax.fill_between(iterations, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
        ax.plot(iterations, mean, label=label, color=color, marker=marker,
                markevery=max(1, round((len(iterations) - 1) / 10)), markersize=3.0, markeredgewidth=0.5,
                linewidth=1.3)
        mean_frames.append(pd.DataFrame(dict(algorithm=name, method=label, iteration=iterations,
                                             evaluations=evaluations, mean_best_fitness=mean,
                                             std_best_fitness=std, runs=runs)))
        run_frames += [pd.DataFrame(dict(algorithm=name, method=label, run=run, iteration=iterations,
                                         evaluations=evaluations, best_fitness=curve))
                       for run, curve in zip(run_ids, curves)]
    if grid_record is not None:
        ax.axhline(float(grid_record["fitness"]), color=ZH_STYLES[ZH_GRID][1], linestyle="--", linewidth=0.9,
                   label=f"参考：网格搜索 SVR（{int(grid_record['evaluations'])} 次评价）")
    ax.set_xlim(0, iterations[-1])
    ax.set(xlabel="迭代次数", ylabel="历史最优适应度（训练集交叉验证 MSE）")
    top = ax.secondary_xaxis("top", functions=(lambda t: (t + 1) * population, lambda e: e / population - 1))
    top.set_xlabel("累计适应度评价次数", fontsize=8.5)
    top.tick_params(labelsize=7.5)
    ZHQ_style_axis(ax)
    ax.legend(frameon=False, fontsize=7.5, title=f"实线为 {runs} 次重复的均值，阴影为 ±1 个标准差",
              title_fontsize=7.5)
    fig.tight_layout()
    ZHQ_save(fig, stem)
    ZHQ_csv(pd.concat(mean_frames, ignore_index=True), curves_dir / "convergence_mean.csv")
    ZHQ_csv(pd.concat(run_frames, ignore_index=True), curves_dir / "convergence_runs.csv")


def ZHQ_plot_distributions(frame, algorithms, feature_selection, n_features, stem):
    panels = [("fitness", "最终交叉验证 MSE（寻优目标）"), ("test_rmse", "测试集 RMSE（原始量纲）")]
    if feature_selection:
        panels.append(("n_selected", "选中的特征数"))
    grid = frame[frame.algorithm == ZH_GRID]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.75 * len(panels) + 0.6, 3.5), squeeze=False)
    jitter = np.random.default_rng(0)
    for ax, (metric, title) in zip(axes.flat, panels):
        data = [frame.loc[frame.algorithm == name, metric].to_numpy(float) for name in algorithms]
        box = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                         medianprops=dict(color="black", linewidth=1.0))
        for k, (patch, name, values) in enumerate(zip(box["boxes"], algorithms, data), start=1):
            color = ZH_STYLES[name][1]
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
            ax.scatter(k + jitter.uniform(-0.13, 0.13, len(values)), values, s=9, color=color, alpha=0.8,
                       linewidths=0, zorder=3)
        ax.set_xticks(range(1, len(algorithms) + 1), [ZH_STYLES[name][0] for name in algorithms])
        ax.tick_params(axis="x", labelsize=7)
        if metric == "n_selected":
            ax.axhline(n_features, color=ZH_STYLES[ZH_GRID][1], linestyle="--", linewidth=0.9,
                       label=f"全部特征（{n_features}）")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(frameon=False, fontsize=7)
        elif not grid.empty:
            ax.axhline(float(grid[metric].iloc[0]), color=ZH_STYLES[ZH_GRID][1], linestyle="--", linewidth=0.9,
                       label=ZH_STYLES[ZH_GRID][0])
            ax.legend(frameon=False, fontsize=7)
        ax.set_title(title, fontsize=9)
        ZHQ_style_axis(ax)
    fig.tight_layout()
    ZHQ_save(fig, stem)


def ZHQ_plot_predictions(records, methods, truth, test_index, target_title, stem, csv_path):
    """预测值–真实值散点图：每个点是一个测试样本，越靠近 y = x 越准。不把样本连线（样本之间没有顺序）。"""
    chosen = [(name, ZHQ_representative(records, name)) for name in methods]
    chosen = [(name, record) for name, record in chosen if record is not None]
    columns = 3 if len(chosen) > 4 else 2
    rows = int(np.ceil(len(chosen) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.55 * columns + 0.3, 2.75 * rows + 0.35), squeeze=False)
    values = np.concatenate([truth] + [np.asarray(record["test_prediction"], dtype=float) for _, record in chosen])
    pad = 0.04 * float(np.ptp(values)) or 1.0
    limits = (float(values.min()) - pad, float(values.max()) + pad)
    table = {"test_row_1based": np.asarray(test_index, dtype=int) + 1, "truth": truth}
    for ax, (name, record) in zip(axes.flat, chosen):
        prediction = np.asarray(record["test_prediction"], dtype=float)
        label, color, _ = ZH_STYLES[name]
        table[label] = prediction
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=0.8, label="y = x")
        ax.scatter(truth, prediction, s=11, facecolor="none", edgecolor=color, linewidths=0.7, label="测试样本")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        note = "" if name in ZH_BASELINES else f"\n第 {record['run'] + 1} 次重复（CV MSE 居中）"
        ax.set_title(f"{label}{note}\nRMSE = {record['test_rmse']:.4g}，R² = {record['test_r2']:.4f}", fontsize=7.8)
        ax.set_xlabel(f"真实值：{target_title}", fontsize=8)
        ax.set_ylabel("预测值", fontsize=8)
        ZHQ_style_axis(ax)
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / (2.75 * rows + 0.35)))
    ZHQ_save(fig, stem)
    ZHQ_csv(pd.DataFrame(table), csv_path)


def ZHQ_feature_frequency(records, algorithms):
    return np.array([np.mean([np.asarray(r["mask"], dtype=float) for r in records if r["algorithm"] == name], axis=0)
                     for name in algorithms])


def ZHQ_plot_feature_frequency(frequency, algorithms, feature_names, stem, csv_path):
    d = len(feature_names)
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(algorithms) + 1.9))
    image = ax.imshow(frequency, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(algorithms)), [ZH_STYLES[name][0] for name in algorithms])
    if d <= 15:
        ax.set_xticks(range(d), feature_names, rotation=30, ha="right")
        for i in range(len(algorithms)):
            for j in range(d):
                ax.text(j, i, f"{frequency[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if frequency[i, j] > 0.6 else "black")
    else:
        ticks = list(range(0, d, 5))
        ax.set_xticks(ticks, [str(t + 1) for t in ticks])
        ax.set_xlabel("特征序号（序号与名称的对应见 feature_frequency.csv）")
    fig.colorbar(image, ax=ax, label="被选中的比例", fraction=0.035, pad=0.02)
    ax.set_title("各特征在全部重复中被选中的比例", fontsize=9)
    fig.tight_layout()
    ZHQ_save(fig, stem)
    table = pd.DataFrame({"feature_index": np.arange(1, d + 1), "feature": feature_names})
    for name, row in zip(algorithms, frequency):
        table[ZH_STYLES[name][0]] = row
    ZHQ_csv(table, csv_path)


def ZHQ_plot_landscape(grid_record, records, algorithms, config, stem, csv_path):
    """网格搜索得到的 ln C × ln g 适应度曲面，叠加各算法每次重复找到的最优点（只用于不做特征选择的情形）。"""
    grid = grid_record["grid"]
    ln_c, ln_gamma = np.asarray(grid["ln_c"], dtype=float), np.asarray(grid["ln_gamma"], dtype=float)
    surface = np.asarray(grid["surface"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    positive = surface[np.isfinite(surface) & (surface > 0)]
    norm = (LogNorm(vmin=positive.min(), vmax=positive.max())
            if positive.size and positive.min() < positive.max() else None)
    mesh = ax.pcolormesh(ln_gamma, ln_c, surface, shading="nearest", cmap="Greys", norm=norm)
    fig.colorbar(mesh, ax=ax, label="交叉验证 MSE（对数色标）")
    for name in algorithms:
        pool = [r for r in records if r["algorithm"] == name]
        label, color, marker = ZH_STYLES[name]
        ax.scatter([r["ln_gamma"] for r in pool], [r["ln_C"] for r in pool], s=18, marker=marker,
                   facecolor="none", edgecolor=color, linewidths=0.9, label=label)
    ax.scatter([grid_record["ln_gamma"]], [grid_record["ln_C"]], marker="*", s=110, color="black",
               edgecolor="white", linewidths=0.7, label=f"{ZH_STYLES[ZH_GRID][0]}最优点")
    title = "网格适应度曲面与各算法的最优点"
    if config.get("optimize_epsilon"):
        title += f"\n（曲面取 ε = {config['epsilon']} 的截面，算法的最优点各自带有寻优得到的 ε）"
    ax.set(xlabel="ln g", ylabel="ln C", title=title)
    ax.title.set_fontsize(8.5)
    ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.85)
    fig.tight_layout()
    ZHQ_save(fig, stem)
    rows = [dict(ln_C=c, ln_gamma=g, C=np.exp(c), gamma=np.exp(g), cv_mse=surface[i, j])
            for i, c in enumerate(ln_c) for j, g in enumerate(ln_gamma)]
    ZHQ_csv(pd.DataFrame(rows), csv_path)


# ---------------- 报告 ----------------
def ZHQ_comparison_phrase(row):
    opponent = ZH_STYLES[row["opponent"]][0]
    tally = f"胜/平/负 {row['wins']}/{row['ties']}/{row['losses']}"
    if row["conclusion"] == "未推断":
        return f"与 {opponent} 相比未做显著性推断（重复次数少于 5 次，{tally}）"
    detail = f"{tally}，Holm p = {ZHQ_number(row['p_holm'], 3)}，中位数上比对手{ZHQ_difference(row['median_relative_improvement'])}"
    if row["conclusion"] == "不显著":
        return f"与 {opponent} 无显著差异（{detail}）"
    word = "低" if row["conclusion"] == "显著优" else "高"
    return f"显著{word}于 {opponent}（{detail}）"


def ZHQ_setup_lines(cfg, dataset, algorithms, grid_record):
    n_train, n_test = len(dataset["train_index"]), len(dataset["test_index"])
    key = dataset.get("key", "auto_mpg")
    features = dataset["features"]
    if key == "auto_mpg":
        feature_text = "、".join(f"{name}（{ZH_COLUMN_TITLES.get(name, name)}）" for name in features)
    else:
        feature_text = f"{len(features)} 个材料描述符（原始列名见 metadata.json 与 feature_frequency.csv）"
    sample_text = f"{dataset['n_samples']} 个样本"
    if dataset.get("sampled_rows") is not None:
        sample_text = (f"从原始 {dataset['n_total']} 条中按 sample_seed = {cfg['sample_seed']} 无放回随机抽取 "
                       f"{dataset['n_samples']} 条（SVR 的训练开销随样本数近似平方增长）")
    missing = "；".join(f"{name} 缺失 {count} 个" for name, count in dataset.get("missing", {}).items())
    missing = f"{missing}，用（每一折）训练部分的中位数填补" if missing else "无缺失值"
    split = (f"沿用 MATLAB 程序 nnn.xlsx 的打乱顺序，前 {n_train} 个训练、其余 {n_test} 个测试"
             if cfg["split"] == "matlab" else
             f"按 split_seed = {cfg['split_seed']} 随机划分，训练 {n_train} 个、测试 {n_test} 个")
    feature_selection = cfg.get("feature_selection", False)
    parts = []
    if feature_selection:
        parts.append(f"{len(features)} 个特征选择位 m ∈ [0, 1]（> 0.5 为选中）")
    parts.append(f"ln C ∈ [{cfg['ln_c'][0]:g}, {cfg['ln_c'][1]:g}]")
    parts.append(f"ln g ∈ [{cfg['ln_gamma'][0]:g}, {cfg['ln_gamma'][1]:g}]")
    if cfg.get("optimize_epsilon"):
        parts.append(f"ln ε ∈ [{cfg['ln_epsilon'][0]:g}, {cfg['ln_epsilon'][1]:g}]")
    dimension = len(features) * feature_selection + 2 + bool(cfg.get("optimize_epsilon"))
    epsilon = "" if cfg.get("optimize_epsilon") else f"，ε 固定为 {cfg['epsilon']:g}"
    budget = cfg["population"] * (cfg["iterations"] + 1)
    lines = [f"- 数据：{dataset.get('title', 'UCI Auto MPG')}，{sample_text}；预测目标 {dataset['target']}"
             f"（{dataset['target_title']}）；输入 {len(features)} 维：{feature_text}。{missing}。",
             f"- 划分：{split}。特征与目标用训练部分拟合的 min-max 映射到 [-1, 1]；测试集不参与寻优。",
             f"- 搜索空间（{dimension} 维）：{'、'.join(parts)}{epsilon}。RBF 核 K(x, x') = exp(-g‖x − x'‖²)。",
             f"- 适应度：训练集 {cfg['folds']} 折交叉验证 MSE，按训练集归一化后的目标值计算（与 libsvm -v 同口径），"
             f"分折固定（cv_seed = {cfg['cv_seed']}），不含特征数惩罚，越小越好。",
             f"- 群智能算法：{', '.join(ZH_STYLES[name][0] for name in algorithms)}；种群 {cfg['population']}，"
             f"迭代 {cfg['iterations']} 代，独立重复 {cfg['runs']} 次，每次评价 {budget} 次（含初始种群）；"
             "同一次重复内各算法共用初始种群与随机数状态。"]
    if grid_record is not None:
        lines.append(f"- 基线（都使用全部特征、ε = {cfg['epsilon']:g}）：SVR（网格搜索）即论文中的传统 SVR，在 ln C、ln g 上"
                     f"以步长 {cfg['grid_step']:g} 遍历，共 {int(grid_record['evaluations'])} 个点；SVR（默认参数）为 libsvm "
                     f"默认的 C = 1、g = 1/{len(features)}。两者是确定性的，只计算一次。")
    return lines


def ZHQ_analysis_lines(frame, algorithms, tests, checkpoint_tests, checkpoint_means, summary, grid_record,
                       cfg, dataset, frequency):
    proposed = ZH_STYLES[ZH_PROPOSED][0]
    lines = ["## 结果分析", "",
             "以下文字由程序根据检验结果生成，只陈述数据支持的结论。", ""]
    by_metric = {}
    for row in tests + checkpoint_tests:
        by_metric.setdefault(row["metric"], []).append(row)
    if "fitness" in by_metric:
        lines.append(f"- **寻优能力（最终交叉验证 MSE）**：{proposed} "
                     + "；".join(ZHQ_comparison_phrase(row) for row in by_metric["fitness"]) + "。")
    for t in checkpoint_means.iteration[:-1]:
        rows = by_metric.get(f"fitness_iter{t}")
        if rows:
            lines.append(f"- **收敛速度（第 {t} 代，即 {cfg['population'] * (t + 1)} 次评价时的历史最优 CV MSE）**：{proposed} "
                         + "；".join(ZHQ_comparison_phrase(row) for row in rows) + "。")
    optimized = frame[frame.algorithm.isin(algorithms)]
    if "test_rmse" in by_metric:
        line = (f"- **泛化性能（测试集 RMSE）**：{proposed} "
                + "；".join(ZHQ_comparison_phrase(row) for row in by_metric["test_rmse"]) + "。")
        if len(optimized) >= 3 and optimized.fitness.nunique() > 1 and optimized.test_rmse.nunique() > 1:
            rho = float(spearmanr(optimized.fitness, optimized.test_rmse)[0])
            line += f" 全部 {len(optimized)} 次寻优结果中，CV MSE 与测试 RMSE 的 Spearman 相关系数为 {rho:.2f}。"
        fitness_better = any(r["conclusion"] == "显著优" for r in by_metric.get("fitness", []))
        test_better = [r["conclusion"] == "显著优" for r in by_metric["test_rmse"]]
        if fitness_better and not any(test_better):
            line += (" 交叉验证误差上的优势没有在测试集上表现为显著差异，因此不能据此断言 "
                     f"{proposed} 得到的模型泛化更好。")
        line += f" 测试集只有 {len(dataset['test_index'])} 个样本、一次划分，测试指标的差异含有抽样误差。"
        lines.append(line)
    if grid_record is not None and ZH_PROPOSED in algorithms:
        row = summary.set_index("algorithm").loc[ZH_PROPOSED]
        grid_fitness, grid_rmse = float(grid_record["fitness"]), float(grid_record["test_rmse"])
        line = (f"- **与传统 SVR（网格搜索）对比**：{proposed} 的平均 CV MSE 比网格搜索"
                f"{ZHQ_difference((grid_fitness - row.fitness_mean) / grid_fitness)}，平均测试 RMSE"
                f"{ZHQ_difference((grid_rmse - row.test_rmse_mean) / grid_rmse)}。网格搜索是确定性的单次结果，不做显著性检验。")
        if cfg.get("feature_selection") or cfg.get("optimize_epsilon"):
            line += (" 网格搜索只调 C、g，使用全部特征且 ε 固定，搜索空间与群智能算法不同，"
                     "所以这一差距反映的是“特征选择与参数联合寻优”整体的作用，不能单独归功于寻优算法本身。")
        else:
            line += (f" 两者搜索的是同一个 (ln C, ln g) 空间，但评价次数不同（网格 {int(grid_record['evaluations'])} 次，"
                     f"群智能算法每次 {cfg['population'] * (cfg['iterations'] + 1)} 次）。")
        lines.append(line)
    if frequency is not None:
        features = dataset["features"]
        means = "，".join(f"{ZH_STYLES[name][0]} {optimized[optimized.algorithm == name].n_selected.mean():.1f} 个"
                         for name in algorithms)
        always = [features[j] for j in range(len(features)) if np.all(frequency[:, j] >= 0.9)]
        never = [features[j] for j in range(len(features)) if np.all(frequency[:, j] <= 0.1)]

        def listing(names):
            if not names:
                return "无"
            shown = "、".join(names[:12])
            return f"{len(names)} 个：{shown}" + ("等" if len(names) > 12 else "")
        lines.append(f"- **特征选择**：平均选中特征数为 {means}（共 {len(features)} 个）。三种算法都在 ≥ 90% 的重复中选中的特征"
                     f"（{listing(always)}）；都在 ≤ 10% 的重复中选中的特征（{listing(never)}）。")
    lines += ["- **解读时注意**：最终 CV MSE 是寻优过程中从大量候选里挑出的最小值，对泛化误差是偏乐观的估计，"
              "所以评价最终模型要看测试集；以上结论只适用于本数据集、本搜索空间和本评价预算。", ""]
    return lines


def ZHQ_write_report(records, output, metadata):
    ZHQ_configure_fonts()
    cfg, dataset = metadata["config"], metadata["dataset"]
    algorithms = list(metadata["algorithms"])
    feature_selection = bool(cfg.get("feature_selection", False))
    metadata["versions"] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                pandas=pd.__version__, matplotlib=matplotlib.__version__,
                                scikit_learn=sklearn.__version__)

    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ZH_HIDDEN_FIELDS} for r in records])
    frame.insert(2, "method", frame.algorithm.map(lambda name: ZH_STYLES[name][0]))
    ZHQ_csv(frame, output / "runs.csv")
    methods = algorithms + [name for name in ZH_BASELINES if (frame.algorithm == name).any()]
    summary = ZHQ_summary(frame, methods)
    ZHQ_csv(summary, output / "summary.csv")
    relative = "test_mre_mean" in summary.columns
    if relative:
        thesis = pd.DataFrame({"算法": summary.method,
                               "训练集平均相对误差": summary.train_mre_mean, "训练集均方差": summary.train_msre_mean,
                               "预测集平均相对误差": summary.test_mre_mean, "预测集均方差": summary.test_msre_mean})
        ZHQ_csv(thesis, output / "论文表格_预测结果对比.csv")

    optimized = frame[frame.algorithm.isin(algorithms)]
    tests = []
    for metric, title in (("fitness", "最终交叉验证 MSE"), ("test_rmse", "测试集 RMSE")):
        tests += ZHQ_paired_tests(ZHQ_by_run(optimized, algorithms, metric), algorithms, metric, title)
    iterations = len(np.asarray(next(r for r in records if r["algorithm"] in algorithms)["curve"])) - 1
    checkpoints = ZHQ_checkpoints(iterations)
    checkpoint_tests, checkpoint_rows = [], []
    for t in checkpoints:
        values = {}
        for name in algorithms:
            run_ids, curves = ZHQ_curve_matrix(records, name)
            values[name] = pd.Series(curves[:, t], index=run_ids)
        checkpoint_rows.append(dict(iteration=t, evaluations=cfg["population"] * (t + 1),
                                    **{name: float(values[name].mean()) for name in algorithms}))
        if t != iterations:
            checkpoint_tests += ZHQ_paired_tests(values, algorithms, f"fitness_iter{t}", f"第 {t} 代的历史最优 CV MSE")
    checkpoint_means = pd.DataFrame(checkpoint_rows)
    if tests or checkpoint_tests:
        ZHQ_csv(pd.DataFrame(tests + checkpoint_tests), output / "paired_tests.csv")
    ZHQ_csv(checkpoint_means, output / "checkpoint_means.csv")

    figures, curves, predictions = (output / name for name in ("figures", "curves", "predictions"))
    for path in (figures, curves, predictions):
        path.mkdir(exist_ok=True)
    grid_record = next((r for r in records if r["algorithm"] == ZH_GRID), None)
    ZHQ_plot_convergence(records, algorithms, cfg["population"], grid_record, curves, figures / "01_收敛曲线")
    ZHQ_plot_distributions(frame, algorithms, feature_selection, len(dataset["features"]),
                           figures / "02_最终结果分布")
    ZHQ_plot_predictions(records, methods, np.asarray(dataset["test_target"], dtype=float), dataset["test_index"],
                         dataset["target_title"], figures / "03_测试集预测值与真实值",
                         predictions / "test_predictions.csv")
    frequency = None
    if feature_selection:
        frequency = ZHQ_feature_frequency(records, algorithms)
        ZHQ_plot_feature_frequency(frequency, algorithms, list(dataset["features"]), figures / "04_特征入选比例",
                                   output / "feature_frequency.csv")
    elif grid_record is not None:
        ZHQ_plot_landscape(grid_record, records, algorithms, cfg, figures / "04_网格适应度曲面与最优点",
                           output / "grid_surface.csv")

    lines = [f"# {dataset.get('title', 'UCI Auto MPG')}：PSO_ARDO 优化 SVR 的对比实验", "", "## 实验设置", ""]
    lines += ZHQ_setup_lines(cfg, dataset, algorithms, grid_record)

    rows = []
    for r in summary.itertuples():
        row = [r.method, ZHQ_mean_std(r.fitness_mean, r.fitness_std, 6), ZHQ_number(r.fitness_best),
               ZHQ_number(r.fitness_worst), ZHQ_mean_std(r.test_rmse_mean, r.test_rmse_std),
               ZHQ_mean_std(r.test_mae_mean, r.test_mae_std), ZHQ_mean_std(r.test_r2_mean, r.test_r2_std)]
        if relative:
            row.append(ZHQ_mean_std(r.test_mre_mean, r.test_mre_std))
        if feature_selection:
            row.append(ZHQ_mean_std(r.n_selected_mean, r.n_selected_std, 3))
        row += [ZHQ_mean_std(getattr(r, f"{name}_mean"), getattr(r, f"{name}_std"), 3)
                for name in ("ln_C", "ln_gamma", "ln_epsilon") if hasattr(r, f"{name}_mean")]
        row += [ZHQ_number(r.evaluations_mean, 5), ZHQ_number(r.seconds_mean, 3)]
        rows.append(row)
    headers = ["算法", "交叉验证 MSE", "最好", "最差", "测试 RMSE", "测试 MAE", "测试 R²"]
    headers += ["测试 MRE"] if relative else []
    headers += ["特征数"] if feature_selection else []
    headers += [label for name, label in (("ln_C", "ln C"), ("ln_gamma", "ln g"), ("ln_epsilon", "ln ε"))
                if f"{name}_mean" in summary.columns]
    headers += ["评价次数", "用时 (s)"]
    lines += ["", "## 结果汇总", "",
              "均值 ± 标准差（基线只有一次）。交叉验证 MSE 为寻优目标（训练集、归一化目标值）；RMSE、MAE 为测试集、原始量纲。", "",
              ZHQ_table(headers, rows), ""]
    if relative:
        rows = [[r.method, *[f"{v:.4f}" for v in (r.train_mre_mean, r.train_msre_mean, r.test_mre_mean, r.test_msre_mean)]]
                for r in summary.itertuples()]
        lines += ["## 预测结果对比（论文表 4-5 格式）", "",
                  "群智能算法取各次重复的均值。“均方差”沿用 MATLAB 程序与论文的口径，即相对误差的均方 mean(((t − p)/t)²)。", "",
                  ZHQ_table(["算法", "训练集平均相对误差", "训练集均方差", "预测集平均相对误差", "预测集均方差"], rows), ""]

    rows = [[int(r.iteration), int(r.evaluations), *[ZHQ_number(getattr(r, name), 6) for name in algorithms]]
            for r in checkpoint_means.itertuples()]
    lines += ["## 不同迭代处的平均历史最优 CV MSE", "",
              ZHQ_table(["迭代", "评价次数", *[ZH_STYLES[name][0] for name in algorithms]], rows), ""]
    all_tests = tests + checkpoint_tests
    if all_tests:
        rows = [[row["metric_title"], ZH_STYLES[row["opponent"]][0], f"{row['wins']}/{row['ties']}/{row['losses']}",
                 ZHQ_difference(row["median_relative_improvement"]), ZHQ_number(row["p_holm"], 3), row["conclusion"]]
                for row in all_tests]
        lines += [f"## {ZH_STYLES[ZH_PROPOSED][0]} 与对照算法的配对检验", "",
                  "按同一次重复配对（初始种群相同）的双侧 Wilcoxon 符号秩检验，同一指标内对各对手做 Holm 校正（α = 0.05）。"
                  "胜/平/负以 PSO_ARDO 的值更小为胜；“中位相对差”为各次重复 (对手 − PSO_ARDO)/对手 的中位数，"
                  "写成 PSO_ARDO 比对手低或高多少；重复次数少于 5 次时不给出 p 值。", "",
                  ZHQ_table(["指标", "对手", "胜/平/负", "中位相对差", "p（Holm）", "结论"], rows), ""]

    lines += ZHQ_analysis_lines(frame, algorithms, tests, checkpoint_tests, checkpoint_means, summary, grid_record,
                                cfg, dataset, frequency)

    lines += ["## 图", "",
              "![收敛曲线](figures/01_收敛曲线.png)", "",
              "图 1　收敛曲线。纵轴是寻优目标（训练集交叉验证 MSE），不是测试误差；实线为各次重复历史最优值的均值，"
              "阴影为 ±1 个标准差；下横轴为迭代次数，上横轴为累计评价次数；虚线为网格搜索 SVR 的 CV MSE，仅作参考。"
              "图中不截取、不放大局部纵轴，最终差距请看图 2 与配对检验。", "",
              "![最终结果分布](figures/02_最终结果分布.png)", "",
              "图 2　各次重复的最终结果。箱体为四分位范围，横线为中位数，每个点是一次重复。", "",
              "![测试集预测值与真实值](figures/03_测试集预测值与真实值.png)", "",
              "图 3　测试集预测值–真实值散点图。每个算法取 CV MSE 居中的那次重复；点越靠近虚线 y = x，预测越准。", ""]
    if feature_selection:
        lines += ["![特征入选比例](figures/04_特征入选比例.png)", "",
                  "图 4　各特征在全部重复中被选中的比例。", ""]
    elif grid_record is not None:
        lines += ["![网格适应度曲面与最优点](figures/04_网格适应度曲面与最优点.png)", "",
                  "图 4　网格搜索得到的适应度曲面（对数色标）与各算法每次重复找到的最优点。", ""]

    warned = int(frame.convergence_warnings.sum())
    lines += ["## 说明", "",
              "- 逐次结果见 runs.csv，汇总见 summary.csv，检验见 paired_tests.csv，各迭代处均值见 checkpoint_means.csv；"
              "逐代曲线数据见 curves/，全部测试集预测值见 predictions/test_predictions.csv。"]
    if warned:
        lines.append(f"- 寻优过程中共有 {warned} 次 SVR 训练达到迭代上限 max_iter = {cfg['max_iter']} 后停止"
                     "（多出现在 C 很大、g 很小的区域），这些候选参数的适应度是近似值；可用 --max-iter 调大或设为 -1 取消上限。")
    if "final_converged" in frame.columns:
        unconverged = frame[~frame.final_converged.astype(bool)]
        if len(unconverged):
            counts = "，".join(f"{ZH_STYLES[name][0]} {count} 个" for name, count in unconverged.algorithm.value_counts().items())
            lines.append(f"- 有 {len(unconverged)} 个最终模型在迭代上限内没有收敛（{counts}），它们的测试指标来自近似解，"
                         "解读时需注意；runs.csv 的 final_converged、final_n_iter 列给出了每个最终模型的情况。")
        else:
            lines.append("- 所有最终模型都在迭代上限内收敛（见 runs.csv 的 final_converged、final_n_iter 列）。")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ZHQ_regenerate_report(source, parent=None):
    """从 partial/ 的逐任务记录重新生成报告和图，不重新训练。"""
    source = Path(source)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("kind") not in ("svr_uci", "svr_auto_mpg"):
        raise ValueError("请选择 run_svr.py 生成的 SVR 实验结果目录")
    records = [record for path in sorted((source / "partial").glob("*.json"))
               for record in json.loads(path.read_text(encoding="utf-8"))]
    cfg = metadata["config"]
    expected = {(name, run) for name in metadata["algorithms"] for run in range(cfg["runs"])}
    expected |= {(name, 0) for name in metadata["baselines"]}
    actual = [(r["algorithm"], r["run"]) for r in records]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("partial 目录中的逐次记录缺失或重复，无法重新生成报告")
    output = (Path(parent).resolve() if parent else source.parent) / \
        f"results_svr_report_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(metadata, regenerated_from=str(source.resolve()))
    ZHQ_write_report(records, output, metadata)
    ZHQ_write_json(output / "metadata.json", metadata)
    print(f"报告：{(output / 'report.md').resolve()}")
