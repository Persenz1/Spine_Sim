# M3 完整设计与分片运行说明

**当前目标：** 在单元总预载 0.5、1、2 N 下完成沉降，保持该外载并沿 +x 拖拽
100 mm，比较完整硬件/阵列组合的拉力趋势。
**当前状态：** 只完成设计、解析测试和短程 smoke；未运行正式 campaign。

## 1. 设计空间

### 基础硬件

固定角布局的基础硬件是完整笛卡尔积：

- 针尖半径：50、100 μm；
- 针径：0.6、0.8 mm；
- 固定安装角：60°、70°、80°；
- 轴向设置：300、800、2000 N/m、刚性。

因此基础硬件为 \(2\times2\times3\times4=48\) 种。100 μm/0.8 mm 只标为
`primary`，50 μm 或 0.6 mm 标为 `auxiliary`；该标记不删除任何组合。

### 阵列与布局

- 形状：2×2、2×5、5×2、3×5、5×3、4×4、6×6；
- 间距：4、5、6 mm；
- 固定布局：\(48\times7\times3=1008\) 种；
- 80°→60° 梯度：角度已经由布局定义，不能再乘固定角三档。其余
  \(2\times2\times4=16\) 种尖端/针径/轴向硬件展开到 7×3，得到 336 种；
- 总阵列构型：1008+336=1344 种。

默认不加入固定 50° 或 80°→50° 梯度。

### 地形、加载与 case 数

正式输入必须是 M1 提供并通过哈希校验的：

- sandpaper/砂纸 100 seed；
- red_brick/红砖 100 seed；
- concrete/混凝土 100 seed。

所有 1344 构型必须共享完全相同的 300 个
`terrain_condition_id`。每个构型/地形再运行 0.5、1、2 N 三种加载协议：

| 部分 | 构型数 | 地形数 | 预载数 | case 数 |
|---|---:|---:|---:|---:|
| 固定角 | 1008 | 300 | 3 | 907,200 |
| 80°→60° | 336 | 300 | 3 | 302,400 |
| 合计 | 1344 | 300 | 3 | 1,209,600 |

这些是计划数量，不代表已经运行。正式完整性检查要求每个
`(array_configuration_id, terrain_condition_id, loading_protocol_id)` 恰好一次；
少一项、重复一项或跨构型地形不配对都拒绝正式分析。

## 2. 当前与后续 M1 地形

M3 不在内部合成“砂纸/红砖/混凝土”。`track_requests` 只根据 M1 catalog 中同一
recipe/region/realization 为各针的全局 y 和针尖半径取轨迹，并使用 M1 缓存。

本次使用现有
`results/m2_formal_terrains/terrain_catalog.json` 做短程接口 smoke。它只有 45 个
`defined_geometry` 条件，不能满足正式 3×100 配对门。M1 后续可替换为更真实生成器，
但应保持以下契约：

1. catalog 提供稳定的 family、seed、recipe、region、realization 和数据哈希；
2. 同 realization 能导出 10 μm 初筛及 5 μm 细筛；
3. 任意构型读取同一 terrain condition 时得到同一个上游身份；
4. M3 对 catalog 和轨迹只读，不自行改变地形。

## 3. 只生成设计，不运行

```powershell
.venv\Scripts\python.exe scripts\prepare_m3_round1_design.py `
  --output examples\m3_complete_design_manifest.json
```

该命令生成 48 种基础硬件、1344 种阵列构型和计划计数，不物化 120 万 cases，也不
启动 runner。

有正式 300 条件 catalog 后，可先只验证 catalog 并刷新 manifest：

```powershell
.venv\Scripts\python.exe scripts\prepare_m3_round1_design.py `
  --catalog <m1_terrain_catalog.json> `
  --output <m3_complete_design_manifest.json>
```

## 4. 生成有界分片

分片必须同时给出一个地形族、有界 seed 范围、一个预载和一个输出级别。例如：

```powershell
.venv\Scripts\python.exe scripts\prepare_m3_round1_design.py `
  --catalog <m1_terrain_catalog.json> `
  --terrain-family sandpaper `
  --seed-min 0 --seed-max 4 `
  --preload-n 1 `
  --output-level summary `
  --workers 1 `
  --output <m3_sandpaper_seed000_004_preload1_summary.json>
```

生成器首先强制验证完整 300 条件 catalog；它不会因为分片只取 5 个 seed 就放松
正式 catalog 门。输出配置可用下面命令运行、续跑和重试：

```powershell
.venv\Scripts\python.exe -m spine_sim.cli run-campaign <shard.json> `
  --output <fast_local_scratch> --workers <N>

.venv\Scripts\python.exe -m spine_sim.cli resume <shard.json> `
  --output <fast_local_scratch> --workers <N>

.venv\Scripts\python.exe -m spine_sim.cli retry-failed <shard.json> `
  --output <fast_local_scratch> --workers <N>
```

以上只是未来运行说明；本次任务没有执行这些正式命令。

## 5. 输出分层与存储

建议策略：

1. 所有 case 用 `summary`，保存身份、初始化覆盖、条件性能、验证和错误诊断。
2. 指定 seed、每类代表构型和异常 case 用 `aggregate_trace`。
3. 代表性构型、异常复现及最终候选才用 `full_pin_trace`。

同一 case 改变输出级别不得改变摘要数值。若 summary case 没有路径数组，
`path.npz` 不会生成；这避免每个 case 都写高频逐针数据，但每 case 目录本身仍会
产生若干小文件。因此正式百万级运行前还必须完成存储吞吐压测，优先在本地高速
scratch 上运行小分片，而不是直接写移动 SSD。

分片结束后：

```powershell
.venv\Scripts\python.exe scripts\merge_m3_summaries.py `
  <campaign_dir_1> <campaign_dir_2> ... `
  --output <archive_root>\m3_summaries
```

合并器会验证每个 `COMPLETE`、结果/路径/事件哈希，原子生成 zstd Parquet
（row group 10,000）及 manifest。它先流式检查类型/哈希，再按批写入，不把全部
summary 放入内存；跨分片重复 case ID 由临时磁盘索引拒绝。缺少 pyarrow 时明确
回退流式 JSONL。移动 SSD 建议只长期保存：

- 合并后的 Parquet/manifest；
- campaign 配置与 lineage；
- aggregate/full trace 的少量归档分片；
- 校验和与失败清单。

原始逐 case scratch 是否删除应由独立归档流程决定，本代码不会自动删除。

## 6. 分析契约

分析必须首先分别报告：

- 初始化覆盖率及各失败类别；
- 在 `ranking_inclusion_allowed=true` 条件下的性能分布。

初始化失败不能用零拉力插入性能排名。正式矩阵完整性检查：

```powershell
.venv\Scripts\python.exe scripts\analyze_m3_round1.py `
  <campaign_dir> --output-dir <analysis_dir>
```

小规模 smoke/shard 只能显式使用 `--allow-partial`，且输出不得声称正式配对完成。
百万级正式分析应直接消费合并后的 Parquet；当前 JSON 逐文件分析器尚需在规模压测
后升级为流式/列式实现。

## 7. 正式运行前门禁

- M1 正式 3 family×100 seed catalog 完整且哈希通过；
- 背板/针/材料/摩擦/接触参数完成标定；
- 100 mm 路径的时间步、接触参数和沉降阻尼收敛；
- 最终候选同 realization 的 10/5 μm 地形收敛；
- 粗糙面 4×4/6×6 动力学与能量残差收敛；
- 小分片完成速度、内存、文件系统和恢复压测。

在这些门关闭前，所有结果保持 `formal_ranking_eligible=false`。
