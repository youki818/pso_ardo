

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MaxNLocator

from function import ZH_DEFINITIONS, ZHQ_make_benchmark


ZH_ROOT = Path(__file__).resolve().parent
ZH_SHIFT_SEED = 731029

ZH_LOCAL_HALF_WIDTH = {"f6": 15.0}


def ZHQ_coordinate_axis(problem, index, samples):

    center = problem.optimum[index]
    lower, upper = problem.lower[index], problem.upper[index]
    if problem.code in ZH_LOCAL_HALF_WIDTH:
        radius = ZH_LOCAL_HALF_WIDTH[problem.code]
        lower, upper = max(lower, center - radius), min(upper, center + radius)
    axis = np.linspace(lower, upper, samples)
    return np.unique(np.append(axis, center))


def ZHQ_sample_surface(problem, samples=121):

    x1 = ZHQ_coordinate_axis(problem, 0, samples)
    x2 = ZHQ_coordinate_axis(problem, 1, samples)
    heights = np.empty((x2.size, x1.size))
    point = problem.optimum.copy()
    for row, y in enumerate(x2):
        point[1] = y
        for col, x in enumerate(x1):
            point[0] = x
            heights[row, col] = problem(point)
    grid_x, grid_y = np.meshgrid(x1, x2)
    return grid_x, grid_y, heights


def ZHQ_draw_panel(ax, problem, letter):
    x, y, z = ZHQ_sample_surface(problem)
    low, high = float(z.min()), float(z.max())
    floor = low - 0.13 * (high - low)
    ax.plot_surface(x, y, z, cmap="viridis", rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.contour(x, y, z, levels=np.linspace(low, high, 10)[1:-1],
               zdir="z", offset=floor, cmap="viridis", linewidths=0.65)
    ax.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()),
           zlim=(floor, high), xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax.set_box_aspect((1, 1, 0.78))
    ax.view_init(elev=26, azim=-130)
    ax.set_proj_type("ortho")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ticks = MaxNLocator(nbins=4).tick_values(0.0, high)
    ax.set_zticks(ticks[(ticks >= 0) & (ticks <= high)])
    if high >= 1e4:
        ax.zaxis.set_major_formatter(FuncFormatter(
            lambda value, _: "0" if value == 0 else f"{value:.1e}"))
    ax.tick_params(axis="both", labelsize=16, pad=0)
    ax.tick_params(axis="z", labelsize=16, pad=2)
    ax.text2D(0.02, 0.96, "函数值", transform=ax.transAxes, fontsize=20)
    ax.text2D(0.5, -0.04, f"({letter})  {problem.code.upper()}",
              transform=ax.transAxes, ha="center", va="top", fontsize=22)


def ZHQ_save_figure(problems, output):
    fig = plt.figure(figsize=(8.3, 11.5), facecolor="white")
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.05, top=0.985,
                        wspace=0.32, hspace=0.24)
    try:
        for index, (problem, letter) in enumerate(zip(problems, "abcdef"), start=1):
            ZHQ_draw_panel(fig.add_subplot(3, 2, index, projection="3d"), problem, letter)
        fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
        print(f"函数图已保存到：{output}")
    finally:
        plt.close(fig)


def ZHQ_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functions", nargs="+", type=str.lower, choices=tuple(ZH_DEFINITIONS))
    parser.add_argument("--dimension", type=int, default=2,
                        help="函数维度，大于 2 时绘制前两维切片")
    parser.add_argument("--shift-seed", type=int, default=ZH_SHIFT_SEED)
    parser.add_argument("--no-shift", action="store_true")
    parser.add_argument("--output", type=Path, default=ZH_ROOT, help="图片的父目录")
    args = parser.parse_args()
    if args.dimension < 2 or args.shift_seed < 0:
        parser.error("dimension >= 2, shift-seed >= 0")
    codes = args.functions or list(ZH_DEFINITIONS)
    if len(set(codes)) != len(codes):
        parser.error("函数名称不能重复")
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese_fonts = [name for name in ("SimSun", "Microsoft YaHei", "SimHei",
                                       "Noto Sans CJK SC") if name in fonts]
    plt.rcParams.update({"font.family": chinese_fonts + ["DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.unicode_minus": False,
                         "axes.labelsize": 20, "axes.labelpad": 8})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = args.output.resolve() / f"figures_function_{stamp}"
    directory.mkdir(parents=True)
    for start in range(0, len(codes), 6):
        group = codes[start:start + 6]
        problems = [ZHQ_make_benchmark(code, args.dimension, args.shift_seed, not args.no_shift)
                    for code in group]
        name = f"{group[0]}_{group[-1]}" if len(group) > 1 else group[0]
        ZHQ_save_figure(problems, directory / f"{name}_surfaces_{args.dimension}d.png")


if __name__ == "__main__":
    ZHQ_main()
