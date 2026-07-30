# M3 完整设计与分片运行说明

**当前目标：** 在单元总预载 0.5、1、2 N 下完成沉降，保持该外载并沿 +x 拖拽
100 mm，比较完整硬件/阵列组合的拉力趋势。
**当前状态：** 正式 M1 catalog 已完成并通过 300/300 严格配对、身份和完整数据
哈希校验；设计、解析测试、真实 runner/恢复/合并以及紧凑 summary 存储均已验证。
正式 proxy campaign 尚未启动，因为粗糙面时间步对照、代表性 10/5 μm 对照和
100 mm 吞吐门仍未关闭。

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

这些是计划数量，不代表已经运行。正式 catalog 位于
`results/m1_material_formal_300/terrain_catalog.json`，catalog ID 为
`945375fc19c5af12a3b22abb2045cd5debeefc7a017d0a2df0448f479d0442c3`。
正式完整性检查要求每个
`(array_configuration_id, terrain_condition_id, loading_protocol_id)` 恰好一次；
少一项、重复一项或跨构型地形不配对都拒绝正式分析。

加载协议还冻结确定性换落点策略：名义落点优先，其后尝试 y=±1/±2 mm 和
x=±1 mm 的全单元偏移。只有几何碰撞或地形越界触发下一候选；净空在正式保留
路径点在线检查，碰撞时立即终止当前候选，下一候选仍从沉降开始重放。该策略用于
尽量完成拖拽，不允许把多落点中的最大拉力当作排名值。

## 2. 当前与后续 M1 地形

M3 不在内部合成“砂纸/红砖/混凝土”。`track_requests` 只根据 M1 catalog 中同一
recipe/region/realization 为各针的全局 y 和针尖半径取轨迹，并使用 M1 缓存。

历史短程接口 smoke 曾使用
`results/m2_formal_terrains/terrain_catalog.json`。该旧 M2 地形库已经永久删除；
它当时只有 45 个 `defined_geometry` 条件，不能满足正式 3×100 配对门，也不能
作为当前默认输入。当前正式 M1 材料地形 catalog 已满足以下接口契约：

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

manifest 同时记录 `engineering_proxy_v1`。当前缺少可用实验标定时，基线采用：

- 钢针 \(E=200\) GPa、\(\nu=0.29\)、\(\rho=7850\) kg/m³、屈服 800 MPa；
- \(\mu_s/\mu_k=0.30/0.20\)、恢复系数 0、针模态阻尼比 0.05；
- 50 g 等效运动小车加随阵列外廓计算的 2 mm 铝背板；
- 背板竖直阻尼比 0.10，按 \(c=2\zeta\sqrt{m k_z}\) 计算；
- 接触位置修正 0.20，沉降阻尼倍数 10。

这些是可复现的经验代理值，只用于比较趋势是否稳健。manifest 还列出摩擦、针阻尼、
恢复系数、位置修正、运动质量、背板阻尼、沉降阻尼和屈服强度的 17 个
one-factor-at-a-time 情景；不得据此把绝对拉力称为已标定预测。

有正式 300 条件 catalog 后，可先只验证 catalog 并刷新 manifest：

```powershell
.venv\Scripts\python.exe scripts\prepare_m3_round1_design.py `
  --catalog <m1_terrain_catalog.json> `
  --output <m3_complete_design_manifest.json>
```

## 4. 生成有界分片

正式 worker 使用只读轨迹缓存。生成任何正式分片前必须串行执行：

```powershell
.venv\Scripts\python.exe scripts\prepare_m3_track_cache.py `
  results\m1_material_formal_300\terrain_catalog.json
```

该命令为每个地形预生成 2 个针尖半径×65 个全局 y（含换落点），总计
39,000 条轨迹，并原子写入完整 manifest。缺少任一轨迹时正式 worker 明确失败，
不会并发补写。

当前正式 catalog 的 39,000 条轨迹已全部完成；精确 track ID 集、全部数据文件、
元数据、COMPLETE 标记和零残留 writer lock 已通过清单核验。完整队列启动时还会
重新执行该文件清单门禁。

分片必须同时给出一个地形族、有界 seed 范围、一个预载和一个输出级别。例如：

```powershell
.venv\Scripts\python.exe scripts\prepare_m3_round1_design.py `
  --catalog <m1_terrain_catalog.json> `
  --terrain-family sandpaper `
  --seed-min 41001 --seed-max 41005 `
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

正式队列由 `scripts/run_m3_formal_queue.py` 逐分片物化和恢复；它只有在
`m3-preflight-v1` 报告显式允许，且轨迹缓存、时间步收敛、地形分辨率收敛和吞吐
四项 launch gate 都为 true 时才启动，并保留所有原始 scratch。物理标定仍可为
false，但这种运行只能称为 proxy，不能进入正式物理排名。

若只有有界 M1 测试 catalog，可用独立入口物化 1–3 个真实 M3 case，以验证
runner、换落点、结果完整性和续跑；它不会放松正式 shard 的 300 条件门：

```powershell
.venv\Scripts\python.exe scripts\prepare_m3_validation_campaign.py `
  <test_catalog.json> --terrain-family sandpaper --seed 41001 `
  --size 2 --drag-length-mm 0.1 `
  --output <m3_validation_campaign.json>

.venv\Scripts\python.exe -m spine_sim.cli run-campaign `
  <m3_validation_campaign.json> --output <local_validation_scratch>
```

该 campaign 固定为 `mode=small`、`formal_ranking_eligible=false`，不能与正式
分片混用。

## 5. 输出分层与存储

建议策略：

1. 所有 case 用 `summary`，保存身份、初始化覆盖、条件性能、验证和错误诊断。
2. 指定 seed、每类代表构型和异常 case 用 `aggregate_trace`。
3. 代表性构型、异常复现及最终候选才用 `full_pin_trace`。

同一 case 改变输出级别不得改变摘要数值。正式 `summary` 分片使用单个事务型
SQLite 数据库和 Parquet 索引，不再创建每 case 一个目录；`aggregate_trace` 和
`full_pin_trace` 仍用逐 case 原子目录。运行层仍采用“有界本地 scratch 分片→
校验并合并→归档列式文件”的两阶段策略，不能把 120 万 case 直接落到移动 SSD。

分片结束后：

```powershell
.venv\Scripts\python.exe scripts\merge_m3_summaries.py `
  <campaign_dir_1> <campaign_dir_2> ... `
  --output <archive_root>\m3_summaries
```

合并器会验证逐目录结果的 `COMPLETE` 和 payload 哈希，或紧凑数据库中的事务完成
状态和 payload 哈希，再原子生成 zstd Parquet
（row group 10,000）及 manifest。它先流式检查类型/哈希，再按批写入，不把全部
summary 放入内存；跨分片重复 case ID 由临时磁盘索引拒绝。缺少 pyarrow 时明确
回退流式 JSONL。移动 SSD 建议只长期保存：

- 合并后的 Parquet/manifest；
- campaign 配置与 lineage；
- aggregate/full trace 的少量归档分片；
- 校验和与失败清单。

原始逐 case scratch 是否删除应由独立归档流程决定，本代码不会自动删除。
每个分片的推荐上限是一个地形族、1–5 个 seed、一个预载，即 1344–6720 case。
只有 Parquet manifest 的 case 数、结果集哈希与源分片全部吻合且抽样解包通过后，
才可由用户批准清理该 scratch 分片；自动合并不会删除原始结果。

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

## 7. 有界收敛方案

可先生成计划：

```powershell
.venv\Scripts\python.exe scripts\prepare_m3_convergence_plan.py `
  --output examples\m3_convergence_plan.json
```

计划选择 8 个确定性哨兵，覆盖 2×2/4×4/6×6、2×5/5×2、固定/梯度、尺寸与刚柔
极值；对 0.5/1/2 N 比较 12 个单轴变体：

- 0.25 ms 参考与 0.5/1/2/5 ms 时间步；
- 20/40 次投影；
- 位置修正 0.20/0.50/1.0；
- 沉降阻尼倍数 5/10/20；
- 拖速 1/2/5 mm/s。

每个地形条件共 288 case，先用 2 mm 短路径，绝不是正式扫描。成对门限为拉力
中位数 3%、P10 5%、Neff 和最大针载荷 5%、接触比例 2 个百分点，并同时要求累计
相对能量误差不超过 \(10^{-3}\)、接触功恒等式残差不超过 \(10^{-12}\) J。
短路径通过后，再只对入围构型做 10/25/50/100 mm 路径长度外推和 10/5 μm
同 realization 对照。

`scripts/prepare_m3_convergence_campaign.py` 会把计划物化为真实、可恢复的
summary campaign，`scripts/analyze_m3_convergence.py` 执行机器门禁，并要求
两边均几何可行、可纳入条件性能、选择同一落点且各有至少 20 个统一保存位置的
稳态样本。当前 P40/seed 41001 的 2×2、1 N、0.1 mm 对照中，0.5 ms 相对
0.25 ms 的中位/P10/Neff/最大针载差为 2.09%/21.93%/1.38%/0.58%；
P240/seed 41005 的几何可行 2 mm 对照仍为
5.92%/31.43%/11.19%/23.22%，未通过门禁。

## 8. 正式运行前门禁

- M1 正式 3 family×100 seed catalog 完整且哈希通过；
- 工程代理敏感性不改变主要趋势；若无实验标定，结论必须明确限定为代理趋势；
- 100 mm 路径的时间步、接触参数和沉降阻尼收敛；
- 最终候选同 realization 的 10/5 μm 地形收敛；
- 粗糙面 4×4/6×6 动力学残差和接触稳定化注能收敛；
- 小分片完成速度、内存、文件系统和恢复压测。

在这些门关闭前，所有结果保持 `formal_ranking_eligible=false`。
