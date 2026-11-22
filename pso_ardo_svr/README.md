# PSO_ARDO 优化 SVR：UCI 回归数据集

参照李美静《基于智能理论的纯电动汽车热管理系统研发》4.4 节（IGA 改进的 SVR）和 `IGA_SVR数据集验证` 中 MATLAB 程序的 SVR 训练流程，用 `pso_ardo_function` 中的 PSO_ARDO 融合算法优化 SVR，并与下面几种方法对比：

| 方法 | 说明 |
| --- | --- |
| PSO_ARDO_SVR | 融合算法联合寻优特征子集与 C、g、ε（本文方法） |
| ARDO_SVR | 原始 ARDO 寻优，搜索空间与预算相同 |
| PSO_SVR | 原始 PSO 寻优，搜索空间与预算相同 |
| SVR（网格搜索） | 论文中的“传统 SVR”：全部特征、ε = 0.01，在 ln C × ln g 网格上逐点做交叉验证（对应 SVR.m） |
| SVR（默认参数） | 不寻优，全部特征，libsvm 默认 C = 1、g = 1/特征数 |

## 为什么不只寻优 C、g

第一版只寻优 C、g（论文的设置，2 维），结果见 `results_svr_auto_mpg_20261122_130028_977867`。三种算法 20 次重复的平均 CV MSE 分别为 0.0212661、0.0212661、0.0212687，差别约 0.01%，测试集指标也几乎相同。

原因是二维的 C、g 寻优曲面只有一个明显的主盆地，2020 次评价足够任何算法找到最优点，结果出现“天花板效应”。这和数据集无关：只换一个 UCI 数据集、仍只调 C、g，结果大概率一样。

把搜索空间扩大为“特征选择 + C、g、ε”后，问题变成高维、分段常数、多峰的混合优化，才能检验算法之间的差别。此前 `svm/` 中 diabetes 数据集上的 13 维实验已经出现了这种差别：PSO_ARDO 的 CV 适应度显著好于 PSO、GWO、WOA，与 ARDO 无显著差异。不过 diabetes 不是 UCI 数据集。

## 实验设计

| 项目 | 设置 |
| --- | --- |
| 数据集 | **Auto MPG**（UCI 编号 9，398 个样本，7 个特征，目标 mpg），与论文对应；**Superconductivity**（UCI 编号 464，原始 21263 条、81 个材料描述符，目标临界温度）。Superconductivity 按 `--sample-seed` 从 UCI 原始压缩包的 `train.csv` 中无放回随机抽取 `--n-samples` 条（默认 1000），因为 SVR 的训练开销随样本数近似平方增长 |
| 划分 | Auto MPG 沿用 MATLAB 程序 `nnn.xlsx` 的划分（前 319 个训练，79 个测试）；Superconductivity 按 `--split-seed` 随机 80/20 划分 |
| 预处理 | 缺失值用中位数填补，特征和目标用 min-max 映射到 [-1, 1]（同 mapminmax）；都只在训练部分（交叉验证时为每一折的训练部分）上拟合 |
| 搜索空间 | 每个特征一个选择位 m ∈ [0, 1]（> 0.5 为选中，全不选时保留 m 最大的特征），加上 ln C、ln g ∈ [-10, 10]（同论文）、ln ε ∈ [-9, 0]。Auto MPG 共 10 维，Superconductivity 共 84 维 |
| 适应度 | 训练集 5 折交叉验证 MSE，按归一化目标值计算，与 libsvm `-v` 同口径；**分折固定**，**不加特征数惩罚**，所以适应度就是一个能直接解释的误差 |
| 预算 | 种群 20，迭代 100 代，独立重复 20 次；每个算法每次重复评价 20 × 101 = 2020 次。同一次重复内三种算法共用初始种群和随机数状态，所以可以配对比较 |
| 测试集 | 寻优过程中从不使用；寻优结束后用最优参数在全部训练集上重训，再在测试集上评价 |
| 统计 | PSO_ARDO 与每个对手按重复配对做双侧 Wilcoxon 符号秩检验，同一指标内 Holm 校正；检验的指标有最终 CV MSE、测试 RMSE，以及第 T/4、T/2 代的历史最优 CV MSE（反映收敛速度） |

加 `--no-feature-selection --fix-epsilon` 可以回到论文只寻优 C、g 的设置。

## 作图与分析的原则

之前那张收敛图（diabetes，5 种算法）有几处会误导读者，新代码都改掉了：

1. **(b) 图单独放大了纵轴。** 截取 0.688–0.704 这一小段后，不到 2% 的差距看起来像“完全分开”。新收敛图只用一个坐标轴画全过程，不放大局部；最终差距改用箱线图和配对检验来看。
2. **只画了均值，没有波动范围。** 那组实验里 PSO_ARDO 与 ARDO 的差异并不显著（Holm p = 0.098），图上却像是明显分开。新收敛图画出各次重复的 ±1 标准差带，箱线图上画出每一次重复的点。
3. **纵轴是混合量。** 那时的适应度是“归一化 CV RMSE + 0.01 × 特征比例”，没有直接的误差含义。新适应度就是 CV MSE，特征数单独报告。
4. **把寻优目标当成模型好坏。** 收敛曲线画的是训练集上的交叉验证误差，不是测试误差。那组实验的测试 RMSE 对所有对手都不显著，ARDO 的测试 RMSE 甚至略低。报告把“寻优能力”和“泛化性能”分开陈述。
5. **测试样本被连成折线。** MATLAB 程序的预测图按样本序号把真实值、预测值连成折线，但测试样本之间没有先后顺序。现在改为预测值–真实值散点图，加 y = x 参考线。
6. **相对误差在目标接近 0 时没有意义。** Superconductivity 的临界温度最小只有 0.0002 K，所以只报告 RMSE、MAE、R²；平均相对误差和论文表 4-5 格式的表只在 Auto MPG 上给出。

`report.md` 的“结果分析”一节由程序根据检验结果生成：

- 不显著就写“无显著差异”，同时给出胜/平/负、Holm p 和中位相对差；
- CV 误差上有优势、测试集上不显著时，会明确写出“不能据此断言泛化更好”；
- 与网格搜索对比时，会说明两者的搜索空间和评价次数都不同。

## 论文 / MATLAB 设置中需要注意的地方

- Auto MPG：MATLAB 程序取 H 列**排量**为输出，mpg 被当作输入。本程序默认按 UCI 的定义预测 mpg；`--target displacement` 可以复现 MATLAB 的设置。
- Computer Hardware（论文的第二个数据集，本程序未使用）：MATLAB 程序的输入含 PRP（真实相对性能），输出是 ERP。ERP 是原始文献用线性模型给出的估计值，所以这个设置实际是在用真实值去拟合另一个模型的预测，不是标准的回归任务。
- MATLAB 网格搜索在交叉验证时用 `-p 0.1`、最终训练用 `-p 0.01`，前后不一致；libsvm `-v` 每次随机分折，同一组参数两次评价结果不同。论文表 4-5 中 SVR 与 IGA_SVR 的差距，有一部分来自这些设置差异。

## 文件

| 文件 | 内容 |
| --- | --- |
| `pso_ardo.py` `ardo.py` `pso.py` `common_tools.py` `config.py` | 从 `../pso_ardo_function` **原样复制**的算法代码（2026-11-22 的带注释版本），没有改动 |
| `svr_problem.py` | 数据读取与抽样、划分、归一化、特征选择编码、交叉验证适应度、网格搜索、误差指标 |
| `svr_experiment.py` | 实验调度：每个任务为“一次重复 × 一个算法”，多进程并行，崩溃自动重跑，逐任务写入 `partial/` |
| `svr_reporting.py` | 汇总表、配对检验、图、结果分析 |
| `run_svr.py` | 命令行入口 |
| `data/auto-mpg.data` | UCI Auto MPG 原始数据 |
| `data/matlab_split_order.csv` | 由 `nnn.xlsx` 导出的样本打乱顺序，用来复现 MATLAB 程序的划分 |
| `data/superconductivity.zip` | UCI Superconductivity 原始压缩包（内含 `train.csv`、`unique_m.csv`），从 `pso_adro_2/data/superconductivity_raw` 复制 |

## 运行

先安装依赖：

```powershell
pip install -r "D:\VsCodeProjects1\codex_oco\pso_ardo\pso_ardo_svr\requirements.txt"
```

冒烟测试，确认流程能跑通（种群 6、迭代 5、重复 2 次；Superconductivity 只抽 300 条）：

```powershell
python "D:\VsCodeProjects1\codex_oco\pso_ardo\pso_ardo_svr\run_svr.py" --quick --jobs 4
```

```powershell
python "D:\VsCodeProjects1\codex_oco\pso_ardo\pso_ardo_svr\run_svr.py" --dataset superconductivity --quick --jobs 4
```

正式实验，Auto MPG：

```powershell
python "D:\VsCodeProjects1\codex_oco\pso_ardo\pso_ardo_svr\run_svr.py" --dataset auto_mpg --jobs 16
```

正式实验，Superconductivity：每次评价要在约 640 个样本、最多 81 个特征上训练 5 个 SVR，比 Auto MPG 慢得多。太慢时可以加 `--iterations 50`。

```powershell
python "D:\VsCodeProjects1\codex_oco\pso_ardo\pso_ardo_svr\run_svr.py" --dataset superconductivity --jobs 16
```

论文的原始设置（只寻优 C、g，ε = 0.01），作为对照：

```powershell
python "D:\VsCodeProjects1\codex_oco\pso_ardo\pso_ardo_svr\run_svr.py" --dataset auto_mpg --no-feature-selection --fix-epsilon --jobs 16
```

续跑中断的实验（把路径换成实际的结果目录）：

```powershell
python "D:\VsCodeProjects1\codex_oco\pso_ardo\pso_ardo_svr\run_svr.py" --resume "D:\VsCodeProjects1\codex_oco\pso_ardo\pso_ardo_svr\results_svr_superconductivity_xxxx" --jobs 16
```

只重出报告和图、不重新训练（也可以用在第一版的 C、g 结果上，换成新的作图方式）：

```powershell
python "D:\VsCodeProjects1\codex_oco\pso_ardo\pso_ardo_svr\run_svr.py" --report-from "D:\VsCodeProjects1\codex_oco\pso_ardo\pso_ardo_svr\results_svr_auto_mpg_20261122_130028_977867"
```

其他参数：`--population` `--iterations` `--runs` `--folds` `--seed` `--epsilon` `--n-samples` `--sample-seed` `--split` `--split-seed` `--cv-seed` `--grid-step` `--max-iter` `--no-baselines` `--algorithms` `--retries` `--output`，详见 `python run_svr.py -h`。

## 输出

每次运行生成 `results_svr_<数据集>_<时间戳>/`：

| 文件 | 内容 |
| --- | --- |
| `report.md` | 实验设置、结果汇总、（Auto MPG）论文表 4-5 格式表、各迭代处均值、配对检验、结果分析、图及图注 |
| `summary.csv` `runs.csv` | 各方法汇总；每次重复每个算法的最优参数、选中特征数、CV MSE、训练/测试指标、用时 |
| `paired_tests.csv` `checkpoint_means.csv` | 配对检验结果；第 T/4、T/2、T 代的平均历史最优 CV MSE |
| `feature_frequency.csv` | 各特征被选中的比例（做特征选择时） |
| `论文表格_预测结果对比.csv` | 仅 Auto MPG：训练集、预测集的平均相对误差与均方差 |
| `curves/` `predictions/` | 逐代收敛数据；各方法在测试集上的预测值 |
| `figures/` | 01 收敛曲线、02 最终结果分布（箱线图 + 每次重复的点）、03 测试集预测值–真实值散点图、04 特征入选比例（不做特征选择时为网格适应度曲面），png + svg |
| `partial/` `metadata.json` | 逐任务原始记录、配置、抽样行号、数据划分索引、代码哈希、软件版本 |

## 工作进程崩溃时

每个任务完成就写入 `partial/`。某个工作进程发生原生崩溃（`BrokenProcessPool`）时，程序会在新的进程池里自动重跑未完成的任务，默认最多重跑 3 轮（`--retries`）。任务都带固定种子，重跑结果与没有崩溃时完全相同。崩溃时的 Python 调用栈写在结果目录的 `crash_logs/` 里；超过重试次数时，已完成的结果仍保留，可以用 `--resume` 续跑。

本机的 RL 环境（Python 3.11.7）在 2026-11-17 到 11-22 之间出现过 6 次 `python311.dll` 写访问冲突。它们分布在不同脚本中，崩溃地址各不相同；同一时期 AIDA64、AweSun、MsiExec 也有崩溃。所以更像系统或硬件在高负载下不稳定，而不是 SVR 代码的问题。崩溃频繁时可以：

- 降低 `--jobs`；
- 检查 BIOS 中的 XMP、超频或降压设置；
- 做一次内存测试。

## 说明

- 不能事先保证 PSO_ARDO 一定胜出；报告只陈述检验结果支持的结论。
- 最终 CV MSE 是寻优过程中挑出的最小值，对泛化误差是偏乐观的估计；评价最终模型要看测试集。测试集只有一次划分，测试指标的差异含有抽样误差。
- 没有移植论文中的 IGA（改进遗传算法）。
- 如果 `pso_ardo_function` 里的算法之后有更新，需要把对应文件重新复制到本目录。
