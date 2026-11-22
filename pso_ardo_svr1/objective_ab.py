"""Auto MPG 选参指标的配对汇总。仅读取各组已完成的结果。"""
import json

import numpy as np
import pandas as pd


def ZHQ_ab_groups(root, outer_folds):
    groups = {}
    for fold in [*range(1, outer_folds + 1), 0]:
        split = f"outer_{fold:02d}" if fold else "historical"
        for metric in ("mse", "mre"):
            groups[fold, metric] = root / split / metric
    return groups


def ZHQ_ab_read(directory, fold, metric):
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    if (meta.get("status") != "complete" or cfg["selection_metric"] != metric
            or cfg["outer_fold"] != fold):
        raise ValueError(f"结果组未完成或配置不符：{directory}")
    frame = pd.read_csv(directory / "runs.csv", dtype={"initial_hash": str})
    frame = frame.set_index(["algorithm", "run"], verify_integrity=True).sort_index()
    expected = {(name, run) for name in meta["algorithms"] for run in range(cfg["runs"])}
    expected.update((name, 0) for name in meta["baselines"])
    if set(frame.index) != expected or not frame.selection_metric.eq(metric).all():
        raise ValueError(f"缺少对应算法、种子或选参指标：{directory}")
    return meta, frame


def ZHQ_write_ab_report(root, groups):
    from svr_reporting import ZHQ_csv, ZHQ_table

    parts = []
    sample_sets = {}
    for fold, metric in groups:
        if metric != "mse":
            continue
        a, left = ZHQ_ab_read(groups[fold, "mse"], fold, "mse")
        b, right = ZHQ_ab_read(groups[fold, "mre"], fold, "mre")
        match_columns = ["seed", "initial_hash", "evaluations"]
        if (a["config"] != {**b["config"], "selection_metric": "mse"}
                or a["dataset"] != b["dataset"]
                or not left[match_columns].equals(right[match_columns])):
            raise ValueError(f"第 {fold} 折的 A/B 配置、数据、评价预算或配对种子不一致")
        data = a["dataset"]
        fit = set(data["train_index"])
        valid = set(data["test_index"])
        if (len(fit) != len(data["train_index"]) or len(valid) != len(data["test_index"])
                or fit & valid or not valid):
            raise ValueError(f"第 {fold} 折样本索引无效")
        sample_sets[fold] = fit, valid
        for label, frame in (("A", left), ("B", right)):
            values = frame[["test_mre", "test_msre", "test_rmse"]].copy()
            values.columns = ["mre", "msre", "rmse"]
            if not np.isfinite(values.to_numpy()).all():
                raise ValueError(f"第 {fold} 折 {label} 组含有无效预测指标")
            values = values.assign(fold=fold, group=label, n=len(valid))
            parts.append(values.reset_index())

    original_train, historical_test = sample_sets[0]
    covered = set()
    for fold, (fit, valid) in sample_sets.items():
        if fold:
            if fit | valid != original_train or valid & covered or valid & historical_test:
                raise ValueError(f"第 {fold} 外折没有正确隔离训练样本和历史测试集")
            covered.update(valid)
    if len(sample_sets) > 1 and covered != original_train:
        raise ValueError("外折验证未恰好覆盖原训练集")

    scores = pd.concat(parts, ignore_index=True)
    ZHQ_csv(scores, root / "ab_by_fold.csv")
    scores["evaluation"] = np.where(scores.fold == 0, "historical", "outer_oof")
    # 各折样本数可相差 1，RMSE 必须先还原为平方误差和。
    scores["absolute_relative_sum"] = scores.n * scores.mre
    scores["squared_relative_sum"] = scores.n * scores.msre
    scores["squared_sum"] = scores.n * scores.rmse**2
    totals = scores.groupby(["evaluation", "algorithm", "run", "group"])[
        ["n", "absolute_relative_sum", "squared_relative_sum", "squared_sum"]
    ].sum()
    pooled = pd.DataFrame(dict(mre=totals.absolute_relative_sum / totals.n,
                               msre=totals.squared_relative_sum / totals.n,
                               rmse=np.sqrt(totals.squared_sum / totals.n)))
    paired = pooled.xs("A", level="group").add_prefix("A_").join(
        pooled.xs("B", level="group").add_prefix("B_"), validate="one_to_one")
    for name in ("mre", "msre", "rmse"):
        paired[f"delta_{name}"] = paired[f"B_{name}"] - paired[f"A_{name}"]
    means = paired.groupby(["evaluation", "algorithm"]).mean()
    means["runs"] = paired.groupby(["evaluation", "algorithm"]).size()
    ZHQ_csv(means.reset_index(), root / "ab_summary.csv")
    lines = ["# Auto MPG 选参指标 A/B 对照", "",
             "A 用归一化 MSE 选参，B 用原始 mpg 上的平均相对误差（MRE）选参。两组都使用全部 7 个特征、固定 ε=0.01。",
             "主指标为 MRE。差值取 B 减 A，负值表示 B 的预测误差更低。CSV 中的 MRE 为比例。", ""]
    for name, title in (("outer_oof", "原训练集的外折验证"), ("historical", "历史留出集")):
        if name not in means.index.get_level_values("evaluation"):
            continue
        block = means.xs(name)
        rows = [[method, int(row.runs), f"{100 * row.A_mre:.4f}%", f"{100 * row.B_mre:.4f}%",
                 f"{100 * row.delta_mre:+.4f}", f"{row.A_rmse:.4f}", f"{row.B_rmse:.4f}"]
                for method, row in block.iterrows()]
        lines += [f"## {title}", "", ZHQ_table(
            ["算法", "重复数", "A 的 MRE", "B 的 MRE", "差值（百分点）", "A 的 RMSE（mpg）", "B 的 RMSE（mpg）"], rows), ""]
    lines += ["外折结果按每辆车等权合并，再对各寻优种子取均值。历史留出集单独汇报。"
              if covered else "本次仅计算历史留出集，未运行训练集内部的外折评价。",
              "各折各次重复的预测误差见 ab_by_fold.csv，MRE、MSRE、RMSE 的 A/B 均值和差值见 ab_summary.csv。"]
    (root / "ab_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"A/B 汇总：{root / 'ab_report.md'}", flush=True)
