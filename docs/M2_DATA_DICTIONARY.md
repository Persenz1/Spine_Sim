# M2 动力学数据字典

**目标模块版本：** `m2.1.0`
**单位：** SI
**模型等级：** `project_model_P_main_plane_dynamic_constant_preload_v2`

## 1. 配置

### `SpineParameters`

保留 `tip_radius_m`、`diameter_m`、`exposed_length_m`、
`installation_angle_deg`、轴向弹簧、梁材料和摩擦字段。新增/冻结要求：

| 字段 | 单位 | 含义 |
|---|---:|---|
| `density_kg_m3` | kg/m³ | 针体密度；用于动态模态质量 |
| `axial_modal_mass_factor` | - | 轴向降阶质量系数及来源 |
| `transverse_modal_mass_factor` | - | 横向降阶质量系数及来源 |
| `axial_damping_ratio` | - | 轴向模态阻尼比 |
| `transverse_damping_ratio` | - | 横向模态阻尼比 |

无法从材料/几何唯一推导的值必须显式写入配置。

### `DynamicExperimentSettings`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `initial_center_x_m` | m | 初始球心横向位置 |
| `drag_length_m` | m | 拖动长度 |
| `drag_speed_m_s` | m/s | 规定水平速度，名义 0.001 |
| `constant_preload_n` | N | 持续施加到安装座的外部 \(-Z\) 力 |
| `holder_effective_mass_kg` | kg | 随 Z 运动的安装座等效质量 |
| `holder_vertical_damping_n_s_m` | N·s/m | 安装座竖向阻尼 |
| `maximum_preload_approach_m` | m | 初始平衡搜索安全界 |
| `output_spacing_m` | m | 距离域输出间距，不是积分时间步 |
| `effective_normal_force_min_n` | N | 接触承载统计阈值 |

### `DynamicContactSettings`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `normal_model` | enum | 刚性冲量或声明的有限刚度定律 |
| `normal_stiffness_n_m` | N/m | 有限刚度模型参数 |
| `normal_damping_n_s_m` | N·s/m | 无拉伸接触阻尼 |
| `tangential_stiffness_n_m` | N/m | 若使用粘着正则化 |
| `tangential_damping_n_s_m` | N·s/m | 切向正则化阻尼 |
| `friction_regularization_velocity_m_s` | m/s | 若使用平滑 Coulomb |
| `restitution_coefficient` | - | 刚性冲量模型使用 |

不用的字段为 null，不能同时暗中启用多种接触模型。

### `DynamicIntegratorSettings`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `method` | enum | 隐式/非光滑积分器 |
| `initial_time_step_s` | s | 初始内部步长 |
| `maximum_time_step_s` | s | 最大内部步长 |
| `minimum_time_step_s` | s | 回退下限 |
| `relative_tolerance` | - | 自适应误差容差 |
| `absolute_tolerance` | mixed | 分量绝对容差 |
| `maximum_nonlinear_iterations` | count | 每步迭代上限 |

## 2. 动态状态

状态必须包含：

- `time_s`；
- 安装座与针尖位置、速度；
- 广义轴向/横向位移和速度；
- 接触状态与粘着历史；
- 弹簧状态；
- 累积摩擦与阻尼耗散；
- 接受步数和拒绝步数。

状态无隐藏可变数据，必须可序列化并确定性恢复。

## 3. `path.npz`

必须至少包含：

| 字段 | shape/单位 |
|---|---|
| `time_s`, `path_position_m` | `[N]`, s/m |
| `holder_xz_m`, `holder_velocity_xz_m_s`, `holder_acceleration_xz_m_s2` | `[N,2]` |
| `center_xz_m`, `center_velocity_xz_m_s`, `center_acceleration_xz_m_s2` | `[N,2]` |
| `axial_displacement_m`, `transverse_displacement_m` | `[N]`, m |
| `axial_velocity_m_s`, `transverse_velocity_m_s` | `[N]`, m/s |
| `external_preload_n` | `[N]`, N |
| `normal_force_n`, `tangential_force_n` | `[N]`, N |
| `spine_on_plate_wrench_about_holder` | `[N,6]`, N/N·m |
| `contact_penetration_m` 或 `normal_impulse_n_s` | `[N]` |
| `contact_state`, `spring_state`, `event_label` | `[N]` |
| `kinetic_energy_j`, `structural_energy_j`, `contact_energy_j` | `[N]`, J |
| `preload_work_increment_j`, `drive_work_increment_j` | `[N]`, J |
| `friction_dissipation_increment_j`, `damping_dissipation_increment_j` | `[N]`, J |
| `energy_residual_j`, `dynamic_residual_n` | `[N]` |
| `actual_time_step_s`, `nonlinear_iterations` | `[N]` |
| `numerical_state`, `model_state` | `[N]` |

局部 `tangential_force_n` 保留符号；承载统计使用全局拖拽方向
`abs(spine_on_plate_wrench_about_holder[:,0])`。

## 4. 摘要

必须包含：

- `preload_mode=continuous_external_force`；
- `constant_preload_n`、`drag_speed_m_s`；
- 初始平衡状态；
- 路径完成状态；
- 接触占空比；
- detach/impact/recontact/stick/slide 事件数；
- 全局拖拽力 P10/P25/中位数；
- 稳态峰与冲击峰；
- 安装座 Z、速度和加速度范围；
- 法向力和冲击速度范围；
- 弹簧行程、应力和屈曲裕度；
- 最大动力学/接触/能量残余；
- 实际最小/最大时间步与拒绝步数；
- 时间步和接触参数收敛标记；
- `numerical_state`、`model_state`、`run_terminal_state`；
- `formal_ranking_eligible`。

旧字段 `fixed_holder_z_m` 仅允许出现在 legacy 结果中。新版生产摘要不得写入该字段。
