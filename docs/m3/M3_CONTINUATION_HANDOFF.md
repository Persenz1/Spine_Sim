# M3 共同背板阵列续作交接

**交接日期：** 2026-07-29

**仓库：** `D:\Code\Spine_Sim`

**分支：** `main`

**M3 版本：** `m3.5.0`

**正式 campaign：** 未启动；用户已授权下一窗口在关闭执行门禁后自主安排全量仿真

**正式排名：** `formal_ranking_eligible=false`

## 0. 2026-07-29 夜间最新交接（下一窗口优先读）

用户即将离线休息，已明确授权下一窗口：

1. 自主把 M3 修到完整流程可运行；
2. 自主安排并执行全量仿真，不必等待用户在线确认；
3. 做完后结束会话并留下完整结果、失败清单、恢复点和结论。

该授权允许在本仓库、本机 scratch 和现有正式 M1 catalog 范围内生成 campaign、
预生成轨迹、运行仿真、恢复失败 case、合并结果和生成分析报告；不等于允许删除
地形、scratch 或其他用户数据，也不等于允许把代理结果宣称为实验标定结果。

### 本次续作结论（优先于后文旧执行建议）

全量前运行层修复已经完成：

- 正式轨迹集合固定为 39,000 条，串行预生成、原子 writer lock、正式 worker
  只读缓存，缺轨迹立即失败；本次已全部生成，39,000/39,000 文件清单、元数据、
  COMPLETE 标记和残留锁检查通过，track-set hash 为
  `f9093322a084e5996180fb802c86c31b7aeebf8510f92b4a571262e7c087f8d8`；
- 正式 summary 使用事务型 SQLite 单库和 Parquet 索引，不再产生 120 万目录；
- runner、`COMPLETE`/事务完成记录、resume、payload 哈希和 Parquet 合并已用正式
  P40 条件贯通；
- 接触矩阵组装向量化、轨迹均匀网格查询和静态模态量缓存保持摘要数值不变，
  P40 2×2 0.1 mm 无换落点 wall time 从 42.5 s 降至 19.8 s；
- 杆体净空改为沉降后及每个正式保留路径点在线检查；初始或途中碰撞均立即以
  `structural_boundary` 终止当前候选并换落点，不再先跑完整段；
- 全仓 110 passed + 15 subtests；解析门 16/16。

但“测试完成没问题”的前提没有成立，因此正式 1,209,600-case proxy campaign
没有启动：

1. `m3.5.0` 已修复事件点对性能分位数的步长相关重复采样，并用同一物理位置的
   81 个稳态样本复验。P40 2×2、1 N、0.1 mm 中，0.5 ms 相对 0.25 ms 的
   中位/P10/Neff/最大针载差为 2.09%/21.93%/1.38%/0.58%；1 ms 相对
   0.25 ms 为 3.63%/31.75%/5.68%/5.81%。P240 的几何可行 2 mm 对照中，
   0.5 ms 相对 0.25 ms 仍为 5.92%/31.43%/11.19%/23.22%，所以时间步门
   仍未关闭；
2. 同 realization、同构型、同落点的 P40 10/5 μm 对照未收敛：中位拉力、
   P10、Neff、最大针载分别相差 9.01%、45.67%、5.70%、31.42%；
3. 在线净空终止后，P240 的 0.1 mm 最终基准为 2×2 2.33 s、4×4 8.15 s；
   6×6 的七个碰撞候选由 202.89 s 降至 94.76 s。为避免把固定沉降成本放大
   1000 倍，吞吐门改用 2×2、2 mm、0.5 ms 的 25.69 s 实测值，并理想化折半为
   1 ms 后再按路径外推。即使这个最小阵列、一次落点、完美并行的乐观下界，在
   4 worker 下完成 1,209,600 case 仍需约 6.16 年；按设计平均针数线性估计约
   23.31 年，尚未计入失败重试和分析开销。

机器可读证据位于 `results/m3_preflight_20260729/`。全量队列入口
`scripts/run_m3_formal_queue.py` 已实现硬门禁、分片恢复和原始 scratch 保留；
preflight 未显式允许时它会拒绝启动，并同时要求轨迹、时间步、分辨率和吞吐四项
launch gate 为 true，而不是只检查一个总开关。
一个 sandpaper/seed 41001/1 N 的 1,344-case 正式 summary 分片已成功物化但未
执行，用于证明完整 track manifest、只读缓存和紧凑存储入口可贯通。

### 已完成的最新输入

- 正式 M1 catalog 已生成：
  `D:\Code\Spine_Sim\results\m1_material_formal_300\terrain_catalog.json`；
- 砂纸、红砖、混凝土各 100 条，共 300 条全尺寸 10 μm 条件；
- 三类严格共用 seed 41001--41100；
- 砂纸 P40/P60/P100/P180/P240 各 20 条；
- 300 个唯一 `realization_id` 和 300 个唯一数据哈希；
- M3 严格 catalog 校验 300/300 通过；
- manifest、`COMPLETE`、数据 SHA-256 均一致；
- 高度与有效掩码载荷约 83.12 GiB，生成结束时 D 盘约剩余 327 GB；
- catalog ID：
  `945375fc19c5af12a3b22abb2045cd5debeefc7a017d0a2df0448f479d0442c3`。

正式地形生成报告：
`D:\Code\Spine_Sim\results\m1_material_formal_300\generation_report.json`。

### 当前工作树

工作树未提交。除本轮 M3 修复外，还新增/修改了：

- `scripts/generate_m1_material_formal_catalog.py`：可断点续跑的正式 300 地形生成器；
- `scripts/generate_m1_material_fine_refinements.py`：选定条件的嵌套 5 μm 细化入口；
- `refine_material_terrain_same_realization`：保持所有 10 μm 节点精确不变的 5 μm
  子网格细化函数；
- M3 正式 catalog 校验现强制三类使用相同 seed 集，并要求唯一
  `realization_id`。

上述 M1 细化和严格配对校验刚加入时的基线回归为
`107 passed + 15 subtests`；本次续作后的最终结果以第 0 节所列
`110 passed + 15 subtests` 为准，均无失败。

### 本次续作开始时的问题（历史记录；当前状态以第 0 节为准）

1. **全量吞吐目前不可接受。** 当前计划为
   \(1344\times300\times3=1,209,600\) case，每个正式 case 又是 100 mm、
   1 mm/s、约 100,000 个内部时间步。15 地形的 45 个 0.1 mm smoke 已耗时
   940.74 s，粗糙面 6×6 单个短程用例可达 113 s。不能直接无脑启动 120.96 万
   case；必须先完成性能剖析、轨迹预生成、合理并行度、存储压测和分阶段执行策略，
   否则运行时间和 120 万目录的文件系统开销不可控。
2. **正式 campaign 参数门仍是未闭合状态。** `build_campaign_shard` 当前写入
   `physical_calibration_completed=false`、各种 convergence flag=false，
   因而即使 case 跑完也保持 `parameter_unclosed` 和
   `formal_ranking_eligible=false`。没有实验数据时可以完成“全量代理仿真”，但
   不能伪造标定或开启正式物理排名；报告应明确区分 proxy ranking 与 calibrated
   ranking。
3. **100 mm 收敛未运行。** 必须先从少量地形、哨兵构型和预载做时间步、投影次数、
   位置修正、沉降阻尼、拖速和路径长度的有界收敛。现有计划为每个地形 288 case，
   仍需物化为真实 campaign，并采用逐级扩大路径而非一次全跑。
4. **接触稳定化注能仍偏大。** 新 15 地形短程 smoke 的最大动力学残差为
   \(9.653\times10^{-4}\) N，已接近 1 mN 门；P60 6×6 累计位置修正注能为
   \(2.21\times10^{-2}\) J。必须检查其随时间步和路径长度是否收敛，不能只看
   `path_end`。
5. **5 μm 尚未生成全尺寸证据。** 嵌套细化代码和小尺寸单测已完成，但
   `generate_m1_material_fine_refinements.py` 尚未在正式 catalog 上运行。应先对
   每类一个代表 realization 生成 5 μm，全尺寸单条约 1.19 GiB（含掩码），机器
   只有 16 GB RAM，必须逐条运行并监控内存；最终候选再扩大，不需要给全部 300 条
   都生成 5 μm。
6. **碰撞规避已在线终止，但仍是降采样几何代理。** 初始和中途正式保留点发现
   杆体碰撞时，当前候选立即停止，再按固定顺序从沉降起点尝试下一落点。P240
   6×6 七次失败候选均在首次碰撞的 19--68 μm 处终止。但净空仍是保存点上的
   24×9 圆柱代理，不是连续杆体接触自由度，保存点之间仍可能漏检。
7. **换落点会改变局部地形。** 所有构型使用相同候选顺序且只按几何可行性选择，
   但不同构型可能在同一 realization 上选择不同落点。分析必须保存/分层
   `selected_unit_origin_xy_m`，不能声称它们比较了完全相同的局部 patch。
8. **物理约束失败很多。** 15 地形 nominal smoke 中只有 15/45 同时通过屈服、
   屈曲和杆体净空；30 条被排除（24 条仅杆碰撞、5 条杆碰撞且屈服超限、1 条仅
   屈服超限）。换落点只能解决几何碰撞，不能掩盖屈服代理超限。
9. **M1 材料仍是代理。** 正式 300 catalog 在数量、配对、身份和哈希上完整，但
   红砖/混凝土及部分砂纸 profile 仍缺项目实测标定。因此
   `formal_ranking_eligible=false` 是正确状态。
10. **轨迹缓存并发风险未解决。** 多 worker 首次请求同一 terrain/y/radius 可能
    竞争写 cache。全量运行前必须串行预生成并冻结所需轨迹，正式 worker 只读。
11. **结果存储尚未做百万级压测。** 当前每 case 一个目录；120 万目录可能成为
    主要瓶颈。必须先做有界分片、峰值空间和 inode/目录操作压测。当前环境没有
    pyarrow，summary 合并会退化为 JSONL；全量前应提供 Parquet 引擎或确认可接受
    的列式替代方案。
12. **文档状态需要刷新。** `M3_OPEN_ISSUES.md`、`M3_TEST_REPORT.md` 和本文件后文
    仍有“正式 catalog 不存在/正式条件数为 0”的旧描述，应以本节最新状态为准并
    在下一窗口统一修订。
13. **最新代码尚未做最终提交。** 先检查 `git diff`，保留用户已有 M1 工作，
    完成测试和数值证据后再决定提交；用户本轮没有要求推送。

### 下一窗口建议的自主执行顺序

1. `git status --short`，完整审阅未提交 diff；
2. 重跑 M1/M3 定向测试和全仓测试；
3. 用正式 catalog 物化一个最小 M3 shard，验证 300 catalog 到
   `run_case → COMPLETE → resume → merge`；
4. 串行预生成代表地形的全部所需 y/radius 轨迹，并测量缓存体积；
5. 对 2×2/4×4/6×6 做 profiler 和 2/10/100 mm 路径外推，先解决吞吐/存储；
6. 跑少量哨兵的数值与拖速收敛，关闭或量化接触注能问题；
7. 每类生成一个嵌套 5 μm realization 并做同节点/趋势对照；
8. 确定可恢复的分片大小、worker 数、scratch 预算和 Parquet 合并方案；
9. 先跑一小批正式分片并验证分析契约，再自主扩大到全量；
10. 全量结束后合并、校验 case 数与结果集哈希，分别报告初始化覆盖、约束排除、
    proxy 条件性能和失败恢复；不得自动删除原始 scratch。

## 1. 用户最终需求

用完整硬件和阵列构型，在 0.5、1、2 N 单元总预载下：

1. 将整个单元通过一个持续外部总预载压向壁面；
2. 用平滑预载斜坡完成共同背板和全部针的动态沉降；
3. 沉降后保持同一总预载，规定背板沿 +x 拖动 100 mm；
4. 比较不同针尖、针径、安装角、轴向设置、阵列形状、方向、间距和角度布局之间的
   拉力趋势；
5. 初始化失败单独统计，不能以零拉力进入条件性能排名。

当前只开放共同背板 Z 动力学；+x 为规定运动，俯仰/横滚锁定。

## 2. 此前已完成并推送的 M3 实现提交

- `fca6344 fix(m3): harden common-backplate dynamics and campaign design`
- `ddafd91 feat(m3): add auditable proxy and convergence controls`

两个提交均已推送至 `origin/main`。它们包含：

- 共同背板与全部针的统一动力学积分；
- 原子 proposal/commit、拒绝不污染和遍历顺序不变；
- minimum-jerk 总预载斜坡及完整沉降门；
- 初始化失败分类和失败 case 排名隔离；
- 48 种基础硬件、1008 个固定布局、336 个 80°→60° 梯度布局；
- 0.5/1/2 N、100 mm 加载协议身份；
- summary/aggregate/full 三档输出；
- 原子结果、`COMPLETE`、结果/路径/事件哈希及断点续跑；
- 流式 Parquet summary 合并和重复 case 拒绝；
- backward-Euler 离散能量、接触功和位置修正注能账本；
- `engineering_proxy_v1` 经验基线与敏感性情景；
- 有界数值/速率收敛计划；
- M1 二维高度场上的圆柱杆下表面保留点在线净空检查。

## 3. 完整设计数量

基础硬件固定角笛卡尔积：

\[
2\ \text{针尖半径}
\times 2\ \text{针径}
\times 3\ \text{固定角}
\times 4\ \text{轴向设置}
=48.
\]

- 固定角：\(48\times7\) 阵列形状 \(\times3\) 间距 = 1008；
- 80°→60° 梯度：\(16\times7\times3=336\)；
- 总构型：1344；
- 正式计划：\(1344\times300\) 地形条件 \(\times3\) 预载
  = 1,209,600 cases。

不要加入固定 50° 或 80°→50° 梯度。用户已经在本次夜间交接中明确授权下一窗口
在关闭执行门禁后自主启动正式扫描，不需要再次等待用户确认；但不能跳过本文件
第 0 节列出的吞吐、收敛、缓存和存储前置检查。

## 4. 等待 M1 完成后必须核对的接口

M3 不生成砂纸、红砖或混凝土地形，只读消费 M1 catalog 和轨迹。M1 完成后首先检查：

1. catalog 是否提供稳定的 family、seed、recipe、region、realization 和数据哈希；
2. 是否有砂纸、红砖、混凝土各 100 seed，共 300 个严格配对条件；
3. 每个 condition 的完整数据哈希是否验证通过；
4. 同一 realization 是否能产生 10 μm 初筛和 5 μm 细筛；
5. 任意 M3 构型引用同一 condition 时，是否得到完全相同的上游地形身份；
6. 不同针的轨迹是否来自同一全局二维地形及各自全局 y，而不是独立随机生成；
7. 新 M1 API 是否仍能提供 M3 杆体净空所需的二维高度场、region 原点和 x/y 分辨率。

如果 M1 改变字段名或 catalog schema，应在 M3 adapter 中做显式版本兼容，不能静默
猜测或由 M3 重新合成地形。

本轮已接入 `m1-material-terrain-catalog-v1` 的 15 条增强测试 catalog（每类 5 条）：
15 个全尺寸文件的 SHA-256 已逐一重算一致，45/45 个 2×2/4×4/6×6 短程数值流程
到达 `path_end`。该 catalog 仍明确标记 `formal_300_complete=false`，且没有同
realization 的 5 μm 版本，因此只能关闭接口与短程流程问题，不能关闭正式门禁。

## 5. 当前代码与历史验证基线

此前 `m3.3.0` 专项回归在清理后复核为 `tests/test_m3_array.py` 32 passed；
本轮修复后的 `m3.5.0` 结果见 `M3_TEST_REPORT.md`。
当前 `m3.5.0` 回归和流程证据为：

- 解析验收：16/16；
- M3 专项：37 passed；
- 结果存储/续跑与 summary 合并：7 passed；
- 全仓：110 passed，另有 15 subtests；
- 真实 M1 runner case 完成，碰撞后换落点重放成功，resume 未重复计算。

以下粗糙地形数值是清理前保存的历史验证快照；对应旧 45 个 M1
`defined_geometry` 条件现已删除：

- 旧 M1 地形 0.1 mm smoke：2×2、4×4、6×6 均沉降成功并到达 `path_end`。

解析平面夹具：

- 最大动力学残差约 \(1.11\times10^{-16}\) N；
- 最大单步离散能量残差约 \(1.03\times10^{-20}\) J；
- 累计相对能量误差约 \(5.36\times10^{-16}\)；
- 接触功恒等式残差约 \(1.11\times10^{-19}\) J。

历史旧粗糙地形 6×6 短程 smoke：

- 最大动力学残差约 \(8.41\times10^{-4}\) N；
- 最大离散能量残差约 \(5.82\times10^{-8}\) J；
- 累计相对能量误差约 \(2.98\times10^{-5}\)；
- 累计接触位置修正注能约 \(5.44\times10^{-3}\) J。

因此旧的“大能量不守恒”主要是遗漏隐式积分耗散造成的账本错误。上述数值只用于
定位 M3 问题；接入新 M1 catalog 后必须重新生成当前证据，关闭粗糙面大阵列的
接触稳定化注能、动力学残差和趋势收敛。

## 6. 工程代理参数的解释

缺少实验标定时，代码使用 `engineering_proxy_v1`：

- 钢针：\(E=200\) GPa、\(\nu=0.29\)、\(\rho=7850\) kg/m³；
- 屈服强度：800 MPa；
- \(\mu_s/\mu_k=0.30/0.20\)；
- 针模态阻尼比：0.05；
- 恢复系数：0；
- 接触位置修正：0.20；
- 背板运动质量：50 g 等效小车加随阵列外廓变化的 2 mm 铝板；
- 背板竖直阻尼比：0.10；
- 沉降阻尼倍数：10。

这些值可以支持代理趋势和敏感性分析，但不能替代用户无法提供的物理标定。最终报告
必须区分：

- “代理模型下趋势稳健”；
- “绝对拉力经过实验标定”。

当前只能使用前一种表述。

## 7. 新窗口建议执行顺序

### 阶段 A：接入修订后的 M1

1. 先查看 Git 状态，识别并保留用户的 M1 修改；
2. 阅读 M1 新 catalog/schema/API 和测试；
3. 只修改必要的 M3 adapter；
4. 用一个地形条件复跑 2×2/4×4/6×6 的 0.1 mm smoke；
5. 检查地形身份、二维净空、反力、动力学残差、能量账本和位置修正注能。

### 阶段 B：有界收敛，不做正式扫描

生成计划：

```powershell
.venv\Scripts\python.exe scripts\prepare_m3_convergence_plan.py `
  --output examples\m3_convergence_plan.json
```

计划含 8 个哨兵、12 个单轴变体和 3 个预载，每个地形条件 288 cases，建议先用
2 mm 路径。依次关闭：

1. 时间步 5/2/1/0.5 ms；
2. 投影迭代 20/40；
3. 位置修正 1.0/0.5/0.2；
4. 沉降阻尼倍数 5/10/20；
5. 拖速 1/2/5 mm/s；
6. 代表性 10 μm/5 μm 同 realization；
7. 2/10/25/50/100 mm 路径长度外推。

比较时至少要求：

- 拉力中位数差不超过 3%；
- P10 差不超过 5%；
- Neff 和最大针载荷差不超过 5%；
- 接触比例差不超过 2 个百分点；
- 累计相对能量误差不超过 \(10^{-3}\)；
- 接触功恒等式残差不超过 \(10^{-12}\) J；
- 初始化和终止状态一致。

### 阶段 C：小规模趋势试验

收敛门通过后，先选少量配对 seed 和少量代表构型做 100 mm smoke，估计：

- 单 case 时间和峰值内存；
- 不同阵列规模的吞吐；
- 失败恢复率；
- summary/aggregate/full 的实际体积；
- 接触注能是否随路径累计失控。

仍不要直接运行 300-seed 正式 campaign。

### 阶段 D：正式运行前审查

只有以下条件全部满足后，才向用户申请是否启动正式分片：

- 300 条件 catalog 完整并严格配对；
- 代理敏感性不改变主要趋势；
- 时间步、接触参数、沉降阻尼和 100 mm 路径收敛；
- 入围构型的 10/5 μm 同 realization 收敛；
- 4×4/6×6 接触稳定化注能和动力学残差可接受；
- 本地 scratch、Parquet 合并、恢复和归档压测通过。

## 8. 推荐恢复命令

```powershell
git status --short
git log -3 --oneline --decorate

.venv\Scripts\python.exe -m spine_sim.array.cli validate-analytic `
  --output results\m3_validation\dynamic_analytic_validation_next.json

.venv\Scripts\python.exe -m pytest tests\test_m3_array.py -q
.venv\Scripts\python.exe -m pytest tests\test_results_runner.py -q
.venv\Scripts\python.exe -m pytest tests\test_m3_summary_merge.py -q
```

接入新 M1 后再运行：

```powershell
.venv\Scripts\python.exe -m spine_sim.array.cli smoke-existing-m1 `
  <new_m1_catalog.json> `
  --output results\m3_validation\existing_m1_smoke_after_m1_update.json `
  --drag-length-mm 0.1 --terrain-family <family> --seed <available_seed>

# 对有界测试 catalog 中的每一条 condition 运行 2x2/4x4/6x6：
.venv\Scripts\python.exe -m spine_sim.array.cli smoke-existing-m1 `
  <new_m1_catalog.json> `
  --output results\m3_validation\existing_m1_catalog_smoke.json `
  --drag-length-mm 0.1 --all-conditions --verify-data-hash `
  --placement-search
```

## 9. 存储和排名禁止项

- 不把总预载平均为逐针恒定预载；
- 不调用 M2 静态二分预载根；
- 不把逐针独立 M2 路径相加；
- 不让拒绝步提交单针历史；
- 不把初始化失败记为零承载；
- 不把冲击峰当持续拉力；
- 不把代理值描述为实验标定值；
- 不把 120 万逐 case 文件直接写到移动 SSD；
- 不删除 scratch，除非 Parquet case 数和结果集哈希均验证并得到用户授权；
- 不在本交接状态下开启正式排名。

## 10. 关键文档

- `docs/m3/M3_OPEN_ISSUES.md`
- `docs/m3/M3_DESIGN_AND_RUN_PLAN.md`
- `docs/m3/M3_DATA_DICTIONARY.md`
- `docs/m3/M3_TEST_REPORT.md`
- `docs/m3/M3_to_M4_handoff.md`
- `docs/guide/M3_提示词_共同背板阵列与自适应筛选_执行版.md`
