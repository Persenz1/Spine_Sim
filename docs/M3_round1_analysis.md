# M3 构型拉力趋势分析状态

**状态：未运行正式分析。** 本文件记录分析契约，不包含构型排名。

## 实验问题

对每个完整硬件/阵列构型施加一个单元总预载 \(P=0.5,1,2\) N，经平滑斜坡共同沉降
后，在整个 +x 100 mm 拖动过程中持续保持该外载。目标是比较：

- 稳态拉力水平、分位数、峰值和路径稳定性；
- 预载改变时的拉力趋势与构型交互；
- 地形族及配对 seed 的趋势一致性；
- Neff、Gini、最大针载荷与拉力之间的关系；
- 形状方向、间距、固定角/80°→60° 梯度、轴向柔顺、针尖和针径的影响；
- 屈服、屈曲、硬限位和杆体净空约束。

当前设计包含 1344 个阵列构型、300 个严格配对地形和 3 个预载，共
1,209,600 个计划 case。数量推导见 `M3_DESIGN_AND_RUN_PLAN.md`。

## 数据完整性先于性能

正式分析第一部分必须报告：

1. 总 case 数、构型数、terrain condition 数和加载协议数；
2. 每个地形族是否恰好 100 个 condition；
3. 每个 `(configuration, terrain, protocol)` 是否恰好一次；
4. 初始化成功数、覆盖率和失败分类；
5. 完成 100 mm 路径的比例和数值失败分类。

任何不完整配对都拒绝正式分析，不能通过删除困难 seed 得到“干净”排名。

## 条件性能

只有同时满足以下条件的 case 才进入性能统计：

- `initial_preload_success=true`；
- `conditional_performance_available=true`；
- `ranking_inclusion_allowed=true`；
- 拉力统计不是 null。

初始化失败保留在覆盖率分母中，但不赋值为零拉力。推荐同时报告两张表：

- 初始化/路径覆盖率（越高越好，但不等于拉力性能）；
- 成功条件下的拉力、载荷共享和约束性能。

## 建议趋势统计

在每个预载和地形族内，以 terrain condition 做配对：

- 拉力中位数、P10/P25、稳态峰和变异系数；
- 相对基准构型的 seed 内差值，而不是不配对均值；
- 2×5 对 5×2、3×5 对 5×3 的方向差；
- 0.5→1→2 N 的单调性和斜率；
- 固定 60/70/80° 与 80°→60° 梯度差；
- Neff/Gini、最大/平均针载荷、应力/屈曲/限位的联合分布；
- 10 μm 与同 realization 5 μm 的候选内差值。

冲击峰与稳态拉力必须分开。若以后做候选筛选，应先按约束和覆盖率剔除不可行项，
再用配对 bootstrap 或分层模型估计趋势区间；不能只按单个总分排序。

## 命令

小分片/烟雾数据只做完整性审计：

```powershell
.venv\Scripts\python.exe scripts\analyze_m3_round1.py `
  <campaign_dir> --output-dir <output_dir> --allow-partial
```

不带 `--allow-partial` 时，脚本要求 1344×300×3 的严格矩阵：

```powershell
.venv\Scripts\python.exe scripts\analyze_m3_round1.py `
  <campaign_dir> --output-dir <output_dir>
```

当前逐 JSON 分析器适合 smoke 和有界分片。百万级分析应先用
`scripts/merge_m3_summaries.py` 生成 Parquet，再实现/使用列式扫描；否则内存和小文件
遍历会成为风险。

## 当前不能回答

- 哪个硬件/阵列组合拉力最好；
- 哪个预载最优；
- 梯度是否优于固定角；
- 哪个地形族最容易或最困难；
- 规模收益是否饱和。

现有一个 seed、0.1 mm 的 smoke 只证明接口与短程积分运行，不能支持上述结论。
参数标定、正式 M1 catalog、100 mm 收敛和粗糙面大阵列能量残差未关闭，因此所有
结果保持 `formal_ranking_eligible=false`。
