"""把 UCI Residential Building（编号 437）原始压缩包中的 Excel 表转换成 CSV，只需运行一次。

原始文件：data/residential_building.zip（从 https://archive.ics.uci.edu/dataset/437 下载，
内含 Residential-Building-Data-Set.xlsx）。读取 Excel 需要 openpyxl；转换后的
data/residential_building.csv 只用 numpy/pandas 就能读，实验环境不必安装 openpyxl。

表头有两行：第一行是变量组，第二行是变量编号。19 个经济变量在 5 个时间滞后组中重复出现，
所以 CSV 中写成 “V-11_lag1” 这样的唯一列名。数值原样写出，不做任何处理。
"""
import io
from pathlib import Path
import re
import zipfile

import pandas as pd

ZH_DATA = Path(__file__).resolve().parent / "data"
ZH_SOURCE = ZH_DATA / "residential_building.zip"
ZH_TARGET = ZH_DATA / "residential_building.csv"


def ZHQ_column_name(group, name):
    group, name = str(group), str(name).strip()
    if group.startswith("PROJECT DATES"):
        return name.replace(" ", "_")
    lag = re.search(r"TIME LAG (\d+)", group)
    return f"{name}_lag{lag.group(1)}" if lag else name


def ZHQ_main():
    with zipfile.ZipFile(ZH_SOURCE) as archive:
        raw = archive.read("Residential-Building-Data-Set.xlsx")
    frame = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=[0, 1])
    frame.columns = [ZHQ_column_name(group, name) for group, name in frame.columns]
    if frame.shape != (372, 109) or frame.columns.duplicated().any():
        raise ValueError(f"表的形状或列名与预期不符：{frame.shape}")
    frame = frame.apply(pd.to_numeric, errors="raise")
    if frame.isna().any().any():
        raise ValueError("表中有缺失值")
    frame.to_csv(ZH_TARGET, index=False, encoding="utf-8")
    print(f"已写出 {ZH_TARGET}：{frame.shape[0]} 行 × {frame.shape[1]} 列")


if __name__ == "__main__":
    ZHQ_main()
