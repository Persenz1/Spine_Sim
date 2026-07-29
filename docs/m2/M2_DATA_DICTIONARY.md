# M2 动力学数据字典

**当前模块版本：** `m2.2.0`

**单位：** SI

**模型等级：** `project_model_P_main_plane_dynamic_constant_preload_v2`

本文描述当前 `DynamicSingleSpineExperiment` 和正式 runner 实际接受、保存的字段。
旧 fixed-Z 数据结构只在 `archive/` 中保留，不属于本接口。

## 1. 配置

### `SpineParameters`

除针尖半径、针径、露出长度、安装角、轴向弹簧、梁材料和摩擦参数外，动态实现
使用以下显式字段：

| 字段 | 单位 | 含义 |
|---|---:|---|
| `density_kg_m3` | kg/m³ | 针体密度；用于动态模态质量 |
| `axial_modal_mass_factor` | - | 轴向降阶质量系数 |
| `transverse_modal_mass_factor` | - | 横向降阶质量系数 |
| `axial_damping_ratio` | - | 轴向模态阻尼比 |
| `transverse_damping_ratio` | - | 横向模态阻尼比 |
| `yield_strength_pa` | Pa/null | 屈服后检查；未给出时阻断候选资格 |
| `rod_clearance_mode` | enum | 正式地形使用圆柱杆后检查；解析夹具可显式禁用 |

无法从材料或几何唯一推导的值必须显式写入正式 case。代理值必须同时声明
`requires_experimental_calibration=true`。

### `DynamicExperimentSettings`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `initial_center_x_m` | m | 初始球心横向位置 |
| `drag_length_m` | m | 拖动长度，必须为正 |
| `drag_speed_m_s` | m/s | 规定水平速度 |
| `constant_preload_n` | N | 持续施加到安装座的外部 \(-Z\) 力，可为 0 |
| `holder_effective_mass_kg` | kg | 随 Z 运动的安装座等效质量 |
| `holder_vertical_damping_n_s_m` | N·s/m | 安装座竖向阻尼 |
| `maximum_preload_approach_m` | m | 初始平衡搜索安全界 |
| `output_spacing_m` | m | 距离域保存间距，不是内部时间步 |
| `effective_normal_force_min_n` | N | 有效承载统计阈值 |
| `initial_preload_force_tolerance_n` | N | 初始预载平衡的显式力容差 |

### `DynamicContactSettings`

`m2.2.0` 当前只实现刚性 Moreau 接触，不接受罚刚度或切向弹簧字段。

| 字段 | 单位 | 含义 |
|---|---:|---|
| `normal_model` | enum | 当前固定 `rigid_moreau` |
| `restitution_coefficient` | - | 法向冲量恢复系数，范围 `[0,1]` |
| `position_correction` | - | 间隙位置修正系数，范围 `(0,1]` |
| `activation_tolerance_m` | m | 接触激活容差 |
| `impact_velocity_threshold_m_s` | m/s | 冲击事件速度阈值 |
| `maximum_contact_force_n` | N | 接触力安全上限 |
| `projection_iterations` | count | 每步接触/摩擦投影迭代次数 |

### `DynamicIntegratorSettings`

当前实现采用确定性固定步长。时间步收敛通过独立的 `time_step_s` 减半配对运行
审计，不存在单 case 内的自适应步长或容差字段。

| 字段 | 单位 | 含义 |
|---|---:|---|
| `method` | enum | 当前固定 `moreau_implicit_euler` |
| `time_step_s` | s | 固定内部时间步 |
| `settling_time_s` | s | 初始平衡后最短沉降时间 |
| `settling_velocity_tolerance_m_s` | m/s | 沉降速度门限 |
| `maximum_settling_time_s` | s | 沉降最长允许时间 |
| `maximum_steps` | count | 整条路径的内部步数安全上限 |

## 2. 动态状态

内部 `DynamicState` 包含：

- `time_s`；
- 三个广义位置和速度：安装座 Z、针体轴向和横向模态；
- `contact_state`、`spring_state` 和是否曾接触；
- 接受步数；
- 累积摩擦与阻尼耗散。

状态推进由配置和旧状态确定，可确定性重放。当前固定步实现没有拒绝/回退步，
摘要中的 `rejected_steps` 因而为 0；字段仍保留用于统一审计接口。

## 3. `path.npz`

正式动态 case 的 `path.npz` 由当前保存点生成，字段如下：

| 字段 | shape/单位 |
|---|---|
| `time_s`, `path_position_m` | `[N]`, s/m |
| `holder_xz_m`, `holder_velocity_xz_m_s`, `holder_acceleration_xz_m_s2` | `[N,2]` |
| `center_xz_m`, `center_velocity_xz_m_s`, `center_acceleration_xz_m_s2` | `[N,2]` |
| `support_xyz_m` | `[N,3]`, m；无接触支撑时为 NaN |
| `tangent_xz`, `normal_xz` | `[N,2]` |
| `gap_m` | `[N]`, m |
| `contact_state`, `spring_state`, `event_label` | `[N]` |
| `external_preload_n` | `[N]`, N |
| `wall_on_spine_force_xz_n` | `[N,2]`, N |
| `spine_on_plate_wrench_about_holder` | `[N,6]`, N/N·m |
| `normal_force_n`, `tangential_force_n` | `[N]`, N |
| `normal_impulse_n_s`, `tangential_impulse_n_s` | `[N]`, N·s |
| `impact_velocity_m_s` | `[N]`, m/s |
| `axial_displacement_m`, `transverse_displacement_m` | `[N]`, m |
| `axial_velocity_m_s`, `transverse_velocity_m_s` | `[N]`, m/s |
| `axial_force_n`, `transverse_force_n` | `[N]`, N |
| `spring_compression_m`, `spring_travel_margin_m` | `[N]`, m |
| `kinetic_energy_j`, `structural_energy_j` | `[N]`, J |
| `preload_work_increment_j`, `drive_work_increment_j` | `[N]`, J |
| `friction_dissipation_increment_j`, `damping_dissipation_increment_j` | `[N]`, J |
| `energy_residual_j`, `dynamic_residual_n` | `[N]`, J/N |
| `actual_time_step_s`, `nonlinear_iterations` | `[N]`, s/count |
| `bending_stress_pa`, `euler_buckling_margin_n` | `[N]`, Pa/N |
| `numerical_state`, `model_state` | `[N]` |
| `rod_clearance_m` | `[N]`, m；只在正式地形圆柱杆后检查启用时存在 |

局部 `tangential_force_n` 保留符号；承载统计使用全局拖拽方向
`abs(spine_on_plate_wrench_about_holder[:,0])`。完整 STICK/SLIDE 切换计数来自每个
接受步的流式统计，`path.npz` 只保存输出间距点和主要事件点，避免事件过采样改变
分位数。

## 4. `summary.json` 与验证

动态摘要至少包括：

- `m2_module_version=m2.2.0`、模型等级和
  `preload_mode=continuous_external_force`；
- 预载、拖速、初始预载成功与否；
- 接触/有效承载占空比及长度；
- 全局拖拽力 P10/P25/中位数、稳态峰和总峰；
- 安装座范围、冲击速度、事件计数；
- 弯曲应力、屈曲裕度和可选杆体净空；
- 最大动力学/能量残余、内部时间步和接受/拒绝步数；
- 时间步/接触参数收敛标记；
- `numerical_state`、`model_state`、终止状态与原因；
- `ranking_scope=project_model_proxy`、
  `requires_experimental_calibration=true`；
- `formal_ranking_eligible` 和单独的代理模型候选门。

单次成功运行不会自动打开正式排名。当前实现要求材料与动力学参数标定、时间步/
接触参数配对收敛及结构/净空约束另行闭合；因此现有代理 case 的
`formal_ranking_eligible` 保持 false。

旧字段 `fixed_holder_z_m` 只允许存在于 legacy fixed-Z 结果中，当前生产摘要不得
写入。
