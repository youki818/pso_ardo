

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter

from function import ZH_DEFINITIONS


ZH_ROOT = Path(__file__).resolve().parent
ZH_DISPLAY_FLOOR = 1e-30
ZH_STYLES = {
    "PSO": ("粒子群（PSO）", "#2673bd", "o"),
    "ARDO": ("自适应角色分工（ARDO）", "#ec781c", "s"),
    "GWO": ("灰狼（GWO）", "#29914a", "^"),
    "WOA": ("鲸鱼（WOA）", "#787878", "v"),
    "PSO_ARDO": ("融合算法（PSO_ARDO）", "#cf2426", "D"),
}


def ZHQ_tick_label(value, _):

    if 0.001 <= value < 1000:
        return f"{value:g}"
    power = int(np.floor(np.log10(value)))
    factor = value / 10**power
    return (rf"$10^{{{power}}}$" if np.isclose(factor, 1)
            else rf"${factor:g}\!\times\!10^{{{power}}}$")


def ZHQ_draw_panel(ax, source, code, letter, floor=ZH_DISPLAY_FLOOR):
    data = pd.read_csv(source / "curves" / f"{code}_mean.csv")
    values = data.pivot(index="iteration", columns="algorithm", values="mean_best_fitness")
    unknown = values.columns.difference(ZH_STYLES)
    if len(unknown):
        raise ValueError(f"曲线数据含未知算法名称：{unknown.tolist()}")
    styles = [name for name in ZH_STYLES if name in values]
    values = values[styles].sort_index()
    iterations = values.index.to_numpy()
    fitness = values.to_numpy()
    if (not styles or not len(iterations)
            or not np.array_equal(iterations, np.arange(len(iterations)))
            or not np.isfinite(fitness).all() or np.any(fitness < 0.0)):
        raise ValueError(f"{code.upper()} 的对数曲线需要完整、有限、非负的逐代适应度。")

    for name in styles:
        label, color, marker = ZH_STYLES[name]
        ax.plot(iterations, np.maximum(values[name], floor),
                label=label, color=color, marker=marker,
                markevery=max(1, round((len(iterations) - 1) / 8)), markersize=2.6,
                markeredgewidth=0.45, linewidth=1.1 if name == "PSO_ARDO" else 0.9)
    ax.set_xlim(iterations[0], max(1, iterations[-1]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_yscale("log")
    low, high = np.log10(np.maximum([fitness.min(), fitness.max()], floor))
    pad = max(0.04, 0.055 * (high - low))
    lower, upper = 10**(low - pad), 10**(high + pad)
    ax.set_ylim(lower, upper)

    factors = (1, 2, 5) if high - low < 2 else (1,)
    if high - low < 0.65:
        factors = (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8)
    step = max(1, int(np.ceil((high - low) / 7)))
    ticks = [factor * 10.0**power for power in range(int(np.floor(low - pad)),
                                                     int(np.ceil(high + pad)) + 1, step)
             for factor in factors if lower <= factor * 10.0**power <= upper]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(ZHQ_tick_label))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", direction="out", length=2.6, width=0.65, pad=2)
    ax.set(xlabel="迭代次数", ylabel="平均最优适应度")
    ax.grid(which="major", color="#dddddd", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(0.5, -0.27, f"({letter})  {code.upper()}", transform=ax.transAxes,
            ha="center", va="top", fontsize=10)
    return bool(np.any(fitness < floor))


def ZHQ_save_figure(source, group, output, floor=ZH_DISPLAY_FLOOR):
    rows = int(np.ceil(len(group) / 2))
    height = 3.2 * rows
    fig, axes = plt.subplots(rows, 2, figsize=(7.2, height), squeeze=False)
    try:
        fig.subplots_adjust(left=0.12, right=0.975, top=1.0 - 0.79 / height, bottom=0.85 / height,
                            wspace=0.40, hspace=0.59)
        floor_used = False
        legend = {}
        for ax, code, letter in zip(axes.flat, group, "abcdef"):
            floor_used |= ZHQ_draw_panel(ax, source, code, letter, floor)
            handles, labels = ax.get_legend_handles_labels()
            legend.update(zip(labels, handles))
        for ax in list(axes.flat)[len(group):]:
            ax.set_visible(False)
        fig.legend(legend.values(), legend.keys(), loc="upper center",
                   bbox_to_anchor=(0.52, 1.0 - 0.07 / height), ncol=3, frameon=False,
                   fontsize=8.5, handlelength=2, labelspacing=0.4, columnspacing=1.5)
        if floor_used:
            # 含 $...$ 的字符串会整体交给 mathtext 渲染，中文会变成方框，这里改用上标字符
            power = str(int(round(np.log10(floor)))).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
            fig.subplots_adjust(bottom=1.0 / height)
            fig.text(0.5, 0.06 / height,f"误差低于 10{power} 视为已达到最优值，图中显示在 10{power}",
                     ha="center", va="bottom")
        fig.savefig(output, dpi=400, facecolor="white")
        print(f"收敛曲线图已保存到：{output}")
    finally:
        plt.close(fig)


def ZHQ_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="含有 curves 的函数测试结果目录")
    parser.add_argument("--functions", nargs="+", type=str.lower, choices=tuple(ZH_DEFINITIONS),
                        help="指定要绘制的函数，默认绘制结果目录中的全部函数")
    parser.add_argument("--output", type=Path, default=ZH_ROOT, help="图片的父目录")
    parser.add_argument("--floor", type=float, default=ZH_DISPLAY_FLOOR,
                        help="对数图显示下限；CEC 约定误差低于 1e-8 视为达到最优，可设为 1e-8")
    args = parser.parse_args()
    if not 0 < args.floor < 1:
        parser.error("--floor 必须在 (0, 1) 之间")
    curves = args.results / "curves"
    available = [code for code in ZH_DEFINITIONS if (curves / f"{code}_mean.csv").is_file()]
    if not available:
        parser.error(f"未找到逐代均值 CSV 文件：{curves}")
    codes = args.functions or available
    missing = [code for code in codes if code not in available]
    if missing:
        parser.error(f"结果缺少以下函数的曲线：{', '.join(missing)}")
    if len(set(codes)) != len(codes):
        parser.error("函数名称不能重复")
    installed = {font.name for font in font_manager.fontManager.ttflist}
    chinese = [font for font in ("SimSun", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC")
               if font in installed]
    plt.rcParams.update({"font.family": ["Times New Roman", *chinese, "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.65})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output.resolve() / f"figures_convergence_function_{stamp}"
    output_dir.mkdir(parents=True)
    for start in range(0, len(codes), 6):
        group = codes[start:start + 6]
        name = "-".join(code.upper() for code in (group[0], group[-1])) if len(group) > 1 else group[0].upper()
        ZHQ_save_figure(args.results, group, output_dir / f"{name}_平均收敛曲线.png", args.floor)


if __name__ == "__main__":
    ZHQ_main()
