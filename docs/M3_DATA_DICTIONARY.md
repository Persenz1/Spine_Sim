# M3 持续总预载阵列动力学数据字典

**生产模块版本：** `m3.1.0`
**单位：** 除状态、索引、计数和无量纲指标外均为 SI。
**生产模型等级：**
`project_model_P_common_rigid_backplate_z_dynamic_continuous_total_preload_v2`
**Legacy 模型等级：**
`project_model_P_common_rigid_backplate_quasistatic_v1`

## 1. 接口冻结与 Legacy 边界

M3 正式入口是 `DynamicCommonBackplateArray` 与
`DynamicCommonBackplateExperiment`。它们统一积分共同刚性背板和全部针的内部
动力学，不调用逐针 `DynamicSingleSpineExperiment.run()`。

旧路径现仅以 `LegacyCommonBackplateArray`、
`LegacyFixedZCommonBackplateExperiment`、`LegacyArrayState`、
`LegacyArrayPoseResponse` 和 `LegacyPrescribedPoseConstitutiveCore` 等显式
`Legacy*` 名称保留给迁移夹具。它们代表初始预载后的 fixed-Z / quasistatic
边界，不是正式 M3 入口，其结果不得标记为 `formal_ranking_eligible=true`。

首版生产模型只开放共同背板 Z 平动自由度。水平位置
\(x_B(t)=x_B^0+v_{\rm drag}t\) 是规定输入；俯仰和横滚锁定，因此
`backplate_rotational_dofs=locked`、`backplate_inertia_kg_m2=null`。若以后开放
俯仰/横滚，必须升级模型版本并显式提供相应惯量、阻尼、状态和残余。

## 2. 配置与几何

### `ArrayConfiguration`

| 字段 | 类型/单位 | 含义 |
|---|---|---|
| `nx`, `ny` | int | 正式构型各为 2–6；`fixture_only=true` 时允许一个方向为 1 |
| `spacing_m` | m | 4、5 或 6 mm；首版 x/y 等距 |
| `base_spine` | `SpineParameters` | M2 硬件、材料、质量、阻尼与摩擦参数 |
| `angle_layout` | enum | `fixed`、`gradient_80_to_60`、`gradient_80_to_50` |
| `fixture_only` | bool | 仅解析夹具使用；保存的项目地形 case 禁止为 true |
| `configuration_id` | string | 由完整硬件构型和 M3 版本确定生成 |

针按 row-major 编号：`pin_index = row * nx + column`。安装点相对单元中心：

\[
x_i=(column-(n_x-1)/2)s,\quad
y_i=(row-(n_y-1)/2)s.
\]

梯度只沿局部 x；每列角度线性插值，且
\(L_i\sin\alpha_i=4\sin80^\circ\) mm。阵列系统检查每根针的
`track.y_global_m == unit_origin_y + y_i`，并要求所有 track 的
`terrain_recipe_id`、`region_id` 相同。`2×5` 与 `5×2` 的
`configuration_id` 必须不同。

## 3. 动力学配置

### `ArrayDynamicExperimentSettings`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `external_total_preload_n` | N | 整个路径持续施加到背板的唯一外部总预载 |
| `initial_common_ux_m` | m | 背板初始共同 x 位移 |
| `drag_speed_m_s` | m/s | 规定共同 +x 速度 |
| `drag_length_m` | m | 动态 +x 路径 |
| `backplate_mass_kg` | kg | 共同 Z 自由度的背板质量 |
| `backplate_vertical_damping_n_s_m` | N·s/m | 共同 Z 阻尼 |
| `backplate_rotational_dofs` | enum | 首版固定为 `locked` |
| `backplate_inertia_kg_m2` | null | 首版必须为 null |
| `maximum_preload_approach_m` | m | 初始统一动态沉降安全界 |
| `output_spacing_m` | m | 输出距离间距，不是内部时间步 |
| `effective_pin_normal_force_min_n` | N | 有效承载针阈值 |
| `unclosed_parameter_names` | tuple[str] | 尚未冻结/标定的动力学参数名 |

`external_total_preload_n` 不是逐针预载，不得除以针数后送入 M2 单针。任意时刻均不
强制 \(\sum_iN_i=P_{\rm ext}\)。

### `DynamicContactSettings`

直接复用 M2 的刚性 Moreau 接触设置，并完整保存
`normal_model`、`restitution_coefficient`、`position_correction`、
`activation_tolerance_m`、`impact_velocity_threshold_m_s`、
`maximum_contact_force_n` 和 `projection_iterations`。

### `DynamicIntegratorSettings`

直接复用 M2 的积分器字段，至少保存 `method`、`time_step_s`、初始沉降时间与速度
容差、最大沉降时间和 `maximum_steps`。每点另存实际使用的内部时间步。

缺少显式冻结来源的背板质量/阻尼、针模态质量/阻尼、接触或积分参数必须列入
`unclosed_parameter_names`，并产生
`model_state=parameter_unclosed`、`formal_ranking_eligible=false`。

## 4. `ArrayDynamicState`

状态是无隐藏可变数据的不可变对象，至少包含：

- `time_s`；
- `backplate_position_z_m`、`backplate_velocity_z_m_s`；
- 所有针的轴向/横向位移和速度；
- 所有针的接触状态、弹簧状态和 `has_contacted`；
- 累积摩擦/结构阻尼/背板阻尼耗散；
- `accepted_steps`、`rejected_steps`。

`propose_step(old_state, ...)` 必须是纯 proposal：所有针从同一个
`old_state` 计算。非线性/接触迭代不得提交单针历史。只有
`commit_step(old_state, proposal, accept=true)` 才返回 proposal 状态；拒绝时必须
原样返回 `old_state`。遍历顺序只允许影响临时计算顺序，不得影响结果。

## 5. `ArrayDynamicPathPoint`

每个保存点至少包含：

- `time_s`、`path_position_m`；
- 背板位置、速度、加速度；
- `external_total_preload_n`；
- 逐针安装点、球心位置/速度/加速度；
- 逐针 gap、支撑点、法向、切向、接触状态、弹簧状态和事件；
- 逐针法向/切向力与法向/切向冲量；
- 逐针关于安装点和单元原点的 wrench，以及阵列总 wrench；
- `total_contact_reaction_z_n`、`backplate_inertia_force_z_n`、
  `backplate_damping_force_z_n`；
- 活动针数、有效承载针数、三类 \(N_{\rm eff}\)、最大/均值与 Gini；
- 总动能、结构能、外载功、驱动功、摩擦/结构阻尼/背板阻尼耗散；功与耗散同时
  保存单步增量和累计值；
- `dynamic_residual_n`、`energy_residual_j`、`actual_time_step_s` 和迭代数；
- `numerical_state`、`model_state`。

单针脱离时该针的力和冲量为零，但其状态仍保留在共同积分向量中；随后允许
IMPACT/RECONTACT。普通脱离不得产生 `no_admissible_contact_equilibrium`。

## 6. `path.npz`

主要数组：

| 字段 | shape |
|---|---|
| `time_s`, `path_position_m` | `[N]` |
| `backplate_position_xyz_m`, `backplate_velocity_xyz_m_s`, `backplate_acceleration_xyz_m_s2` | `[N,3]` |
| `external_total_preload_n` | `[N]` |
| `pin_holder_xyz_m`, `pin_center_xyz_m`, `pin_center_velocity_xyz_m_s`, `pin_center_acceleration_xyz_m_s2` | `[N,pin,3]` |
| `pin_gap_m`, `pin_normal_force_n`, `pin_tangential_force_n` | `[N,pin]` |
| `pin_normal_impulse_n_s`, `pin_tangential_impulse_n_s` | `[N,pin]` |
| `pin_wrench_about_holder`, `pin_wrench_about_unit` | `[N,pin,6]` |
| `wall_on_unit_wrench_about_origin` | `[N,6]` |
| `contact_state`, `spring_state`, `event_label` | `[N,pin]` |
| 五类 `active_*` | `[N,pin]` bool |
| `active_pin_count`, `effective_load_pin_count` | `[N]` |
| 三类 `neff_*`、`max_mean_*`、`gini_*` | `[N]` |
| `total_contact_reaction_z_n`, `backplate_inertia_force_z_n`, `backplate_damping_force_z_n` | `[N]` |
| 各能量、功和耗散增量/累计字段 | `[N]` |
| `dynamic_residual_n`, `energy_residual_j`, `actual_time_step_s` | `[N]` |
| `force_aggregation_residual_n`, `moment_aggregation_residual_nm` | `[N]` |
| `seed`, `terrain_recipe_id`, `configuration_id`, `model_level` | `[N]` |

这些字段按同一点索引共同切片形成 M3→M4 样本；禁止跨时间、seed 或接触分支拼接
力、矩和活动集。

## 7. 摘要与排名门禁

摘要必须包含：

- `preload_mode=continuous_total_external_force`；
- `external_total_preload_n`、背板质量/阻尼、锁定转动声明；
- 完整接触和积分设置；
- 初始统一动态沉降状态及路径完成状态；
- 总反力时间平均及与外载的稳态平衡误差；
- 接触占空、有效承载针、detach/impact/recontact/stick/slide 事件数；
- 稳态切向力 P10/P25/中位数/峰值；
- 与稳态统计分离的 `impact_peak_*`；
- 背板 Z/速度/加速度范围；
- 最大动力学、能量和 wrench 聚合残余；
- 实际最小/最大时间步、接受/拒绝步数；
- 时间步减半与接触参数收敛标记；
- `numerical_state`、`model_state`、`run_terminal_state`；
- `unclosed_parameter_names`、`formal_ranking_eligible`。

任何 `unclosed_parameter_names`、未通过的时间步/接触参数收敛、非 path_end 或
非 covered 模型状态都会关闭正式排名门禁。Legacy 字段
`fixed_common_uz_m`、`target_preload_n` 只能出现在显式 Legacy 结果中。
