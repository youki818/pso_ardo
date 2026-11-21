

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MaxNLocator

from function import make_benchmark


FUNCTIONS = ("f1", "f2", "f3", "f4", "f5", "f6")
SHIFT_SEED = 731029

LOCAL_HALF_WIDTH = {"f5": 5.0, "f6": 15.0}


def coordinate_axis(problem, index, samples):

    center = problem.optimum[index]
    lower, upper = problem.lower[index], problem.upper[index]
    if problem.code in LOCAL_HALF_WIDTH:
        radius = LOCAL_HALF_WIDTH[problem.code]
        lower, upper = max(lower, center - radius), min(upper, center + radius)
    axis = np.linspace(lower, upper, samples)
    return np.unique(np.append(axis, center))


def sample_surface(problem, samples=121):

    x1 = coordinate_axis(problem, 0, samples)
    x2 = coordinate_axis(problem, 1, samples)
    heights = np.empty((x2.size, x1.size))
    point = problem.optimum.copy()
    for row, y in enumerate(x2):
        point[1] = y
        for col, x in enumerate(x1):
            point[0] = x
            heights[row, col] = problem(point)
    grid_x, grid_y = np.meshgrid(x1, x2)
    return grid_x, grid_y, heights


def draw_panel(ax, problem, letter):
    x, y, z = sample_surface(problem)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(2, 30), default=2,
                        help="2 绘制二维函数，30 绘制正式测试实例的前两维切片")
    args = parser.parse_args()
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese_fonts = [name for name in ("SimSun", "Microsoft YaHei", "SimHei",
                                       "Noto Sans CJK SC") if name in fonts]
    plt.rcParams.update({"font.family": chinese_fonts + ["DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.unicode_minus": False,
                         "axes.labelsize": 20, "axes.labelpad": 8})
    fig = plt.figure(figsize=(8.3, 11.5), facecolor="white")
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.05, top=0.985,
                        wspace=0.32, hspace=0.24)
    try:
        for index, (code, letter) in enumerate(zip(FUNCTIONS, "abcdef"), start=1):
            problem = make_benchmark(code, args.dimension, SHIFT_SEED)
            draw_panel(fig.add_subplot(3, 2, index, projection="3d"), problem, letter)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = Path(__file__).resolve().parent / f"figures_function_{stamp}"
        directory.mkdir()
        output = directory / f"f1_f6_surfaces_{args.dimension}d.png"
        fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
        print(f"函数图已保存到：{output}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
