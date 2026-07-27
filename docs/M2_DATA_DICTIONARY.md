# M2 数据字典

**模块版本：** `m2.0.0`
**单位：** 除状态、布尔量、计数和无量纲方向/斜率外均为 SI。
**模型等级：** `project_model_P_main_plane_quasistatic_v1`

## 1. 配置

### `SpineParameters`

| 字段 | 单位/类型 | 含义 |
|---|---|---|
| `tip_radius_m` | m | 必须与 M1 track 半径完全一致 |
| `diameter_m` | m | 圆截面针杆直径 |
| `exposed_length_m` | m | 安装座到未变形球心的轴向长度 |
| `installation_angle_deg` | degree | 局部 \(+x\) 到针轴的安装角 |
| `axial_mode` | `rigid/spring` | 刚性轴向或无预压单边弹簧 |
| `spring_stiffness_n_m` | N/m or null | rigid 必须为 null |
| `spring_travel_m` | m | 默认 4 mm |
| `young_modulus_pa` | Pa | 默认 200 GPa，材料牌号尚未冻结 |
| `poisson_ratio` | - | 默认 0.29 |
| `shear_correction` | - | 默认 \(6/7\) |
| `static_friction` | - | 静摩擦系数 |
| `kinetic_friction` | - | 动摩擦系数，不能大于静摩擦 |
| `beam_enabled` | bool | 正式比较必须为 true |
| `material_assumption` | string | 显式材料假设 |
| `rod_clearance_mode` | enum | `unclosed` 或仅解析夹具用的 `disabled_analytic_fixture` |

### `ExperimentSettings`

| 字段 | 单位 | 含义 |
|---|---|---|
| `initial_center_x_m` | m | 零力球心的初始轨迹坐标 |
| `drag_length_m` | m | 固定 Z 后沿 \(+x\) 的路径长度 |
| `path_step_m` | m | 普通拖动步长 |
| `target_preload_n` | N | 只在初始阶段搜索，默认 0.5 N |
| `preload_force_tolerance_n` | N | 初始预载接纳容差，默认 \(10^{-4}\) N |
| `maximum_preload_approach_m` | m | 初始 \(-Z\) 搜索安全界 |
| `effective_normal_force_min_n` | N | 有效承载长度的独立物理阈值 |
| `free_probe_spacing_m` | m or null | FREE 段再接触探测间距；默认 M1 track 分辨率 |
| `refine_events` | bool | 是否二分宏观事件 |

`SolverSettings` 将几何/法向根容差与
`friction_residual_tolerance_n=5e-4 N` 分开。后者只处理 10 μm 离散支撑切换处
的 Coulomb 等式残余，最终候选必须在 5 μm 复核该残余是否收敛。

## 2. 状态

接触状态：

- `free`
- `first_contact_event`
- `stick`
- `slide`
- `detach_event`
- `recontact_event`

弹簧状态独立：

- `lower_stop`
- `interior`
- `hard_stop`

终态：

- `path_end`
- `terrain_bounds`
- `initial_preload_infeasible`
- `structural_boundary`
- `numerical_failure`
- `model_unclosed`

`initial_preload_infeasible` 只能由初始预载阶段产生；拖动脱离事件固定为
`detach_to_free`。

## 3. `SingleSpineState`

| 字段 | 含义 |
|---|---|
| `contact_state/spring_state` | 上一步已接纳的正交状态 |
| `has_contacted` | 区分首次接触与再次接触 |
| `anchor_center_xz_m` | STICK 切向历史锚点 |
| `last_holder_xz_m` | 能量和滑动方向历史 |
| `last_center_xz_m` | 上一步球心 |
| `last_wall_force_xz_n` | 上一步墙面对针的力 |
| `last_elastic_energy_j` | 上一步弹性能 |
| `cumulative_friction_dissipation_j` | 累积非负摩擦耗散 |
| `slide_direction` | \(-1,0,+1\) |
| `accepted_steps` | 原子提交计数 |

状态对象冻结且无隐藏可变数据。`commit=False` 返回 proposal 但
`next_state` 仍是旧对象。

## 4. 逐点数组

M0 case 输出的 `path.npz` 包含：

| 字段 | shape/单位 | 含义 |
|---|---|---|
| `path_position_m` | `[N]`, m | 固定-Z 拖动坐标 |
| `holder_xz_m` | `[N,2]`, m | 安装座位置 |
| `center_xz_m` | `[N,2]`, m | 针尖球心 |
| `support_xyz_m` | `[N,3]`, m | M1 主支撑点 |
| `gap_m` | `[N]`, m | \(c_z-H_R(c_x)\) |
| `tangent_xz/normal_xz` | `[N,2]` | M1 包络局部基 |
| `cap_gate_passed` | `[N]`, bool | 前向球冠门控 |
| `near_tie` | `[N]`, bool | M1 支撑切换敏感标记 |
| `event_refined` | `[N]`, bool | 是否为二分细化点 |
| `contact_state/spring_state/event_label` | `[N]`, string | 状态和事件 |
| `wall_on_spine_force_xz_n` | `[N,2]`, N | 墙面对针 |
| `spine_on_plate_wrench_about_holder` | `[N,6]`, N/N·m | 针对板，参考点为当前安装点 |
| `normal_force_n/tangential_force_n` | `[N]`, N | M1 包络法/切向分量 |
| `axial_force_n/transverse_force_n` | `[N]`, N | \(Q_s,V_b\) |
| `spring_compression_m` | `[N]`, m | \(\delta_s\) |
| `beam_displacement_xz_m` | `[N,2]`, m | 不含弹簧缩短的梁端位移 |
| `static_friction_margin_n` | `[N]`, N | STICK 锥余量；SLIDE 为动摩擦等式余量 |
| `spring_travel_margin_m` | `[N]`, m | 剩余单边行程 |
| `elastic_energy_j` | `[N]`, J | 梁 + 弹簧储能 |
| `holder_work_increment_j` | `[N]`, J | 安装座增量功 |
| `contact_work_increment_j` | `[N]`, J | 墙面对针的接触增量功 |
| `friction_dissipation_increment_j` | `[N]`, J | 非负滑动耗散 |
| `energy_residual_j` | `[N]`, J | 离散功—能残余 |
| `geometry_residual_m` | `[N]`, m | 接触相容残余 |
| `structure_residual_m` | `[N]`, m | 结构运动学重构残余 |
| `force_decomposition_residual_n` | `[N]`, N | 力基分解残余 |
| `root_iterations` | `[N]`, int | 当前分支根迭代 |
| `numerical_state/model_state` | `[N]`, string | 与物理状态分离 |

## 5. 路径摘要

摘要不含综合分数。稳定字段包括：

- `initial_preload_success`
- `ever_contacted`
- `ever_loaded`
- `total_contact_length_m`
- `effective_load_length_m`
- `effective_load_fraction`
- `maximum_continuous_load_length_m`
- `tangential_force_peak_n`
- `tangential_force_median_n`
- `tangential_force_p10_n`
- `tangential_force_p25_n`
- `normal_force_range_n`
- `event_counts`
- `maximum_abs_geometry_residual_m`
- `maximum_abs_energy_residual_j`
- `physical_terminal_state`
- `numerical_state`
- `model_state`
- `run_terminal_state`
- `termination_reason`

切向摘要使用绝对值表示承载幅值；逐点数组保留带符号原值。
