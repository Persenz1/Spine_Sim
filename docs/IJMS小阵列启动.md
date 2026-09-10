# IJMS 小阵列粗筛启动

本批次直接做阵列，使用新版 `spine_sim.ijms:run_case`。参数来源是
`experiments/ijms_small_array.json`，由 `src/spine_sim/small_array_campaign.py`
编译为现有 runner 可执行的 campaign。它不是让执行模型重新设计实验的提示模板。

## 固定范围

- 针尖 50/100 μm；钢针主体直径 1 mm、渐细段 2 mm；均匀角露出长度 4 mm。
- 均匀角 50/60/70/80°；梯度组为爪头 60°→爪跟 80°，爪跟露出 4 mm，其余按同导口高度补偿。
- 弹簧刚度 100/400/800/1000/2000/5000 N/m，名义弹簧行程 4 mm；无弹簧组为固定安装，露出针杆仍弯曲。名义行程不取消露出针形与导口的几何适用范围。
- 间距 4/5/6 mm；布局 2×2、3×3、4×4、5×5、5×2、2×5、8×2、2×8、5×3、3×5、8×3、3×8。
- 小阵列总预载 0.5/1/2 N；E=200 GPa、许用应力 1.2 GPa 是已声明数值情景，不能当实测材料数据。
- 混凝土 `rough_wall`、红砖 `fired_brick_standard`、砂纸 P40/P100/P240 共 5 条件，每条件 64 个合成实现，seed 从 2026091000 连续取值。
- 原始地形采用明确 `synthetic` 模式，保留生成器实际方法与来源。砂纸合成不增加独立实测样片数量；材料 profile 的 provisional/partially_validated 状态随文件保留。

主表 70 个设计，布置表 288 个设计，重复的 8 个合并后 **350 个设计**。
乘 3 预载、5 条件、64 个实现，完整粗筛 **336000 个 case**。
这是一份完整队列，不是“一晚必须完成”的预算；先持续运行一晚，按实际完成量续跑。

## 执行顺序与数据

程序先安排八个预定参考设计（两个半径各配固定安装、k=100/800/5000 N/m，
70°、4×4、间距 5 mm）在首个实现上逐材料运行，三个预载均保留 full 输出。
随后运行其余主表和布置表，按材料轮转分片，再推进下一个实现。
总计 120 个 full case，其余 335880 个为 summary；每分片最多 256 case，合计 1605 分片。
这只是顺序，未改变科学范围、网格或求解容差。

所有设计和预载共享该材料、该 seed 的同一原始墙面和固定落点。地形按需生成，
不是启动前预生成 320 张。当前小阵列最大 tip 跨度为 42 mm，单边加 8 mm
裕量并向 +x 加 10 mm 搜索区，地形域为约 **68×58 mm**，origin=(-29,-29) mm。
10 μm 网格的 float64 高度约 **316 MB/张**（约301 MiB），整库约101 GB；
这是高度数组大小，不含生成时内存、求解器和结果。
结果保留自由 Y；若越过地形域，记录终止，不能锁 Y 或重新抽取落点来补成成功。

针杆统一采用16段变截面离散；拖动距离 10 mm，基本步长 2.5 μm；接触事件可能进一步细分。
每条路径同时计算绝对持续力目标 0.5/1/2 N，以及 T/P≥1 的协议，保持距离 0.5 mm，
搜索窗口 2/5/10 mm，共12协议。失败或不完整路径保持其明确状态。
T/P、预载和平面摩擦参考应一起解释，不把简单超过0.2 N当成啮合收益。

## 命令

从仓库根目录使用项目环境。用户指定过程产物和结果统一存入
`E:\TestData\IJMS`，这也是CLI默认目录；源码和现有环境仍留在仓库。
其中`campaigns/`保存生成配置，`surfaces/`保存共享地形，`results/`保存仿真结果，`logs/`保存运行日志，`tmp/`保存临时写入缓冲。程序将TEMP/TMP指向此tmp子目录，worker继承设置。研究数据必须保留，不得把整个IJMS目录按临时文件清理。

```powershell
Set-Location D:\Code\Spine_Sim
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -B scripts/run_ijms_small.py prepare --dry-run
.\.venv\Scripts\python.exe -B scripts/run_ijms_small.py run --output-dir E:\TestData\IJMS --workers 4
```

`run` 包含按需准备，通常无需先 `prepare`。只准备第一个分片而不运行可用：

```powershell
.\.venv\Scripts\python.exe -B scripts/run_ijms_small.py prepare --output-dir E:\TestData\IJMS --max-shards 1
```

独立查看已有进度：

```powershell
.\.venv\Scripts\python.exe -B scripts/run_ijms_small.py status --output-dir E:\TestData\IJMS
```

该命令分别报告执行状态和物理终止状态、尚无结果数、最近结果时间及累计 case 耗时。
尚无结果包含正在计算及未开始，不能据此判断每个 worker 是否卡住。
运行日志还按每case约30秒间隔输出已完成求解尝试的phase、target、残差和状态；一次求解尚未返回时不会伪造进度。将stdout/stderr保存到结果目录，可以结合CPU活动与这些记录监控长case。

同一 `run` 命令是恢复入口；已完整持久化的 case，包括物理范围终止，不重跑。
执行异常或未完成落盘的 case 会按现有 runner 的恢复逻辑再次执行；先定位工程异常，
避免在原因未改变时反复重启。配置及语义版本固定在输出目录，不能修改物理后混入旧结果。
中断处尚未完成的 case 从该 case 起点重算，本入口没有每步跨进程恢复。

`--max-shards N` 仅限制这次访问的队列前缀，`--start-shard K` 用于零起点的明确任务分片。
它们不缩减正式样本表。完成前缀后恢复完整队列应去掉 `--max-shards`。
不要让两个进程同时运行同一分片；可调整 workers，不能擅改科学参数来提高吞吐量。
如通过PowerShell后台启动，使用`Start-Process -WindowStyle Hidden`，先创建`E:\TestData\IJMS\logs`，将stdout/stderr重定向到该目录。

## 给 DeepSeek 的任务

先读仓库 `AGENTS.md`、`docs/IJMS物理约束.md` 和本说明，再执行以上 dry-run 与正式启动命令。
保持固定设计和地形样本，优先用4个CPU worker，依据实际内存与CPU占用调整并发。
只负责实现错误定位、等价性能优化、执行、恢复和运行监控；保留数值失败、杆体干涉、
地形越界等原始结果，不能调容差、换地形、缩行程、改预载、降低保真度或筛掉失败来提完成率。
不得自行开始细筛或大阵列，不能自行解释材料机制或选论文赢家。
一晚后汇报完成数、各物理状态数、执行异常、用时/内存和恢复命令；按实测吞吐量估计剩余时间，
不得声称336000个case必定能在一晚完成。
