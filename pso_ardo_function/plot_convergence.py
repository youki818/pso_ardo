

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

from config import normalize_algorithm


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results_function_selected6"
STYLES = {
    "PSO": ("粒子群（PSO）", "#2673bd", "o"),
    "ARDO": ("自适应角色分工（ARDO）", "#ec781c", "s"),
    "GWO": ("灰狼（GWO）", "#29914a", "^"),
    "WOA": ("鲸鱼（WOA）", "#787878", "v"),
    "PSO_ARDO": ("融合算法（PSO_ARDO）", "#cf2426", "D"),
}


def tick_label(value, _):

    if value < 1000:
        return f"{value:g}"
    power = int(np.floor(np.log10(value)))
    factor = value / 10**power
    return (rf"$10^{{{power}}}$" if np.isclose(factor, 1)
            else rf"${factor:g}\!\times\!10^{{{power}}}$")


def draw_panel(ax, source, index):
    data = pd.read_csv(source / "curves" / f"f{index}_mean.csv")
    data["algorithm"] = data["algorithm"].map(normalize_algorithm)
    values = data.pivot(index="iteration", columns="algorithm", values="mean_best_fitness")
    values = values[list(STYLES)].sort_index()
    iterations = values.index.to_numpy()
    fitness = values.to_numpy()
    if (not np.array_equal(iterations, np.arange(len(iterations)))
            or not np.isfinite(fitness).all() or np.any(fitness <= 0)):
        raise ValueError(f"F{index} 的对数曲线需要完整的逐代记录和大于零的适应度。")

    for name, (label, color, marker) in STYLES.items():
        ax.plot(iterations, values[name], label=label, color=color, marker=marker,
                markevery=max(1, round((len(iterations) - 1) / 8)), markersize=2.6,
                markeredgewidth=0.45, linewidth=1.1 if name == "PSO_ARDO" else 0.9)
    ax.set_xlim(iterations[0], iterations[-1])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_yscale("log")
    low, high = np.log10([fitness.min(), fitness.max()])
    pad = max(0.04, 0.055 * (high - low))
    lower, upper = 10**(low - pad), 10**(high + pad)
    ax.set_ylim(lower, upper)

    factors = (1, 2, 5) if high - low < 2 else (1,)
    if high - low < 0.65:
        factors = (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8)
    ticks = [factor * 10.0**power for power in range(int(np.floor(low - pad)),
                                                     int(np.ceil(high + pad)) + 1)
             for factor in factors if lower <= factor * 10.0**power <= upper]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(tick_label))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", direction="out", length=2.6, width=0.65, pad=2)
    ax.set(xlabel="迭代次数", ylabel="平均最优适应度")
    ax.grid(which="major", color="#dddddd", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(0.5, -0.27, f"({chr(96 + index)})  F{index}", transform=ax.transAxes,
            ha="center", va="top", fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=SOURCE, help="含有 curves 的函数测试结果目录")
    args = parser.parse_args()
    installed = {font.name for font in font_manager.fontManager.ttflist}
    chinese = [font for font in ("SimSun", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC")
               if font in installed]
    plt.rcParams.update({"font.family": ["Times New Roman", *chinese, "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.65})
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.6))
    try:
        fig.subplots_adjust(left=0.12, right=0.975, top=0.918, bottom=0.089,
                            wspace=0.40, hspace=0.59)
        for index, ax in enumerate(axes.flat, start=1):
            draw_panel(ax, args.results, index)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.993),
                   ncol=3, frameon=False, fontsize=8.5, handlelength=2,
                   labelspacing=0.4, columnspacing=1.5)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = ROOT / f"figures_convergence_function_{stamp}"
        output_dir.mkdir()
        output = output_dir / "F1-F6_平均收敛曲线.png"
        fig.savefig(output, dpi=400, facecolor="white")
        print(f"收敛曲线图已保存到：{output}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
