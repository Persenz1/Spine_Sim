# M3 共同背板阵列动力学数据字典

**生产模块版本：** `m3.2.0`
**模型等级：**
`project_model_P_common_rigid_backplate_z_dynamic_continuous_total_preload_v3`
**单位：** 状态、索引、计数和无量纲指标以外均为 SI。

## 1. 物理与接口边界

生产入口为 `DynamicCommonBackplateArray` 和
`DynamicCommonBackplateExperiment`。它们统一积分一个共同刚性背板 Z 自由度及
所有针的轴向/横向模态；不调用 M2 静态二分预载根，也不将逐针 M2 结果相加。

背板 \(x_B(t)\) 是规定的 +x 拖动输入，当前俯仰/横滚锁定。`external_total_preload_n`
是整个单元唯一外部总法向载荷，不是逐针载荷；沉降结束后保持 0.5、1 或 2 N 到
100 mm 路径终点。逐时刻接触反力可因惯性和阻尼偏离该外载。

## 2. 配置身份

### `ArrayConfiguration`

| 字段 | 含义 |
|---|---|
| `nx`, `ny` | 阵列列/行数；正式形状为 2×2、2×5、5×2、3×5、5×3、4×4、6×6 |
| `spacing_m` | 0.004、0.005、0.006 m |
| `base_spine` | 针尖、针径、安装角、轴向设置、材料、质量、阻尼和摩擦参数 |
| `angle_layout` | `fixed` 或 `gradient_80_to_60` |
| `configuration_id` | 只由完整硬件/阵列身份和模块版本生成，不含 terrain seed |

针按 `pin_index = row * nx + column` 编号，局部安装点为

\[
x_i=(column-(n_x-1)/2)s,\qquad
y_i=(row-(n_y-1)/2)s.
\]

`2×5` 与 `5×2`、`3×5` 与 `5×3` 的方向和 ID 不同。80°→60° 梯度沿阵列局部
x 方向逐列线性变化，并保持统一的竖直未加载伸出量。默认不生成固定 50° 或
80°→50°。

### terrain 与 case 身份

- M3 只消费 M1 catalog 或由 catalog recipe/region 派生的轨迹请求，不伪造地形。
- `terrain_condition_id` 标识 family+seed+realization，`terrain_data_hash` 标识实际
  地形数据。
- `loading_protocol_id` 至少包含总预载、100 mm 路径、拖速、预载斜坡、沉降阻尼和
  积分/接触协议。
- `case_id` 由完整硬件、阵列、terrain condition/data hash、加载协议及上游身份
  共同生成。

## 3. 沉降协议

`ArrayDynamicExperimentSettings` 新增或关键字段：

| 字段 | 单位/类型 | 含义 |
|---|---|---|
| `preload_ramp_profile` | string | 当前 `minimum_jerk_quintic` |
| `preload_ramp_time_s` | s | 从 0 平滑升至总预载的时间 |
| `settlement_damping_scale` | – | 仅沉降阶段作用的数值阻尼倍数；拖动阶段恢复为 1 |
| `settling_reaction_force_tolerance_n` | N | 总反力平衡绝对门 |
| `settling_reaction_force_relative_tolerance` | – | 总反力平衡相对门 |
| `settling_dynamic_residual_tolerance_n` | N | 沉降动力学残差门 |
| `settling_required_stable_steps` | count | 所有门连续满足的步数 |
| `maximum_preload_approach_m` | m | 背板最大允许接近量 |
| `dynamic_residual_tolerance_n` | N | 每个拖动 proposal 的内部残差门 |
| `coupled_projection_relaxation` | – | 多接触投影松弛系数 |

### `SettlementTracePoint`

每个沉降样本保存：

- `time_s`、`ramp_fraction`、`applied_total_preload_n`、`damping_scale`；
- `backplate_position_z_m`、`actual_approach_m`；
- 背板与全部针模态中的 `maximum_mode_speed_m_s`；
- `total_contact_reaction_z_n`、`contact_reaction_error_n`；
- `dynamic_residual_n`、`active_pin_count`、`stable_steps`。

失败分类字段为 `failure_category`/`failure_code` 和
`initialization_failure_category`/`initialization_failure_code`。分类包括
`physical_boundary`、`geometry_out_of_bounds`、`parameter_unclosed`、
`settlement_nonconvergence`、`numerical_failure`。

## 4. 动态状态与原子步骤

`ArrayDynamicState` 是不可变对象，保存背板位置/速度、全部针轴向和横向状态、接触/
弹簧状态、接触历史、累计摩擦/结构/背板阻尼耗散及接受/拒绝步数。

`propose_step(old_state, ...)` 对所有针读取同一个旧状态。只有全局 proposal
成功后，`commit_step(..., accept=true)` 才提交；拒绝时精确返回旧状态。针遍历顺序
不得改变 proposal、point 或状态。

每根针的状态覆盖 free/contact、detach/impact/recontact、stick/slide，以及轴向弹簧
lower/interior/upper hard-stop。轴向设置为 300、800、2000 N/m 或 `rigid`；刚性不
用伪大刚度表示。

## 5. 路径点与审计量

`ArrayDynamicPathPoint` 至少保存：

- 时间、路径位置、背板位置/速度/加速度和实际时间步；
- 每针安装点/球心运动、gap、支撑点、法/切向力与冲量、状态和事件；
- 每针关于安装点和单元原点的 wrench、阵列总 wrench；
- 总反力、背板惯性力和背板阻尼力；
- 活动针/有效针数、三类 \(N_\mathrm{eff}\)、最大/均值和 Gini；
- 动能、结构能、外载功、驱动功、摩擦/结构/背板阻尼耗散的增量与累计量；
- 动力学、能量、合力聚合和合矩聚合残差；
- 每针弯曲应力、屈服余量、Euler 屈曲余量和弹簧行程余量。

单针脱离后其接触力/冲量为零，但针模态继续积分，后续允许重新接触。

## 6. 三档输出

| 级别 | 所有 case | 内容 |
|---|---:|---|
| `summary` | 是 | `summary.json` 中的配置、摘要、验证、失败诊断和哈希；不写 `path.npz` |
| `aggregate_trace` | 选定 | 沉降曲线；背板状态、阵列总力/总矩、有效针数、Neff、Gini、能量和残差等降采样数组 |
| `full_pin_trace` | 少量 | aggregate 全部内容加逐针状态、力、冲量、wrench、应力/屈曲/硬限位及杆体净空代理 |

输出级别不得改变同一 case 的摘要值。`summary` 不保存空事件小文件；事件统计已经
进入摘要。

## 7. `ArrayDynamicPathSummary`

摘要字段分为四组：

1. 配置与状态：预载模式、预载、拖速、自由度、数值/模型/终止状态、失败分类。
2. 初始化：斜坡、时间、阻尼倍数、步数、连续稳定步数、实际/最大接近量、最终外载、
   反力误差、最大模态速度和动力学残差。
3. 条件性能：稳态/冲击拉力统计、反力范围、接触比例、Neff/Gini、最大/平均针载荷、
   应力/屈曲/弹簧限位、事件计数以及各类残差。
4. 门禁：时间步、接触参数、沉降阻尼、地形分辨率收敛和物理标定标志，
   `unclosed_parameter_names`、`formal_ranking_eligible`。

初始化失败时，第 3 组中的连续性能量为 null，并设置
`conditional_performance_available=false`。case 仍保留在初始化覆盖率统计中，但
不得以零承载加入性能排名。已完成路径但违反屈服、屈曲或杆体净空约束的 case 可保留
条件性能用于诊断，但 `ranking_inclusion_allowed=false`。

## 8. 完整性与合并

每个完成 case 的目录含 `summary.json`、可选 `path.npz` 和 `COMPLETE`。
`COMPLETE` 必须等于 `summary.json` 记录的结果哈希；存在路径数组时还校验
`path_sha256`，存在事件文件时还校验 `events_sha256`；summary 级别则要求两者均
不存在。续跑跳过完整且哈希正确的 case，损坏或不完整目录按失败恢复流程重算。

`scripts/merge_m3_summaries.py` 校验上述哈希后，将 summary 原子合并为 zstd
Parquet；没有 pyarrow 时可显式回退 JSONL，并另写 manifest/result-set hash。
