# M3 数据字典

**模块版本：** `m3.0.0`
**单位：** 除状态、索引、计数和无量纲指标外均为 SI。
**模型等级：** `project_model_P_common_rigid_backplate_quasistatic_v1`

## 1. 配置与几何

### `ArrayConfiguration`

| 字段 | 类型/单位 | 含义 |
|---|---|---|
| `nx`, `ny` | int | 正式构型各为 2–6；`fixture_only=true` 时允许一个方向为 1 |
| `spacing_m` | m | 4、5 或 6 mm；首版 x/y 等距 |
| `base_spine` | `SpineParameters` | M2 硬件、材料与摩擦参数 |
| `angle_layout` | enum | `fixed`、`gradient_80_to_60`、`gradient_80_to_50` |
| `fixture_only` | bool | 仅解析两针夹具使用；保存的项目地形 case 禁止为 true |
| `configuration_id` | string | 由完整硬件构型和 M3 版本确定生成 |

针按 row-major 编号：`pin_index = row * nx + column`。安装点相对单元中心：

\[
x_i=(column-(n_x-1)/2)s,\quad
y_i=(row-(n_y-1)/2)s.
\]

梯度只沿局部 x；每列角度线性插值，且
\(L_i\sin\alpha_i=4\sin80^\circ\) mm。阵列系统检查每根针的
`track.y_global_m == unit_origin_y + y_i`，并要求所有 track 的
`terrain_recipe_id`、`region_id` 相同。

## 2. 共同状态与求解

### `ArrayState`

| 字段 | 含义 |
|---|---|
| `pin_states` | canonical pin 顺序下的不可变 `SingleSpineState` 元组 |
| `accepted_steps` | 已原子接纳的共同背板位姿数 |

每一轮 `solve_pose((ux,uZ), old_state, commit=False)` 都从同一个旧
`ArrayState` 计算全部针；`proposal_valid=true` 时，阵列层才可一次性使用
`proposal_state`。`commit=true` 也不会逐针调用 M2 的提交路径，而是在全部 proposal
检查通过后只切换阵列 `next_state`。

### `ArrayExperimentSettings`

| 字段 | 单位 | 含义 |
|---|---|---|
| `target_preload_n` | N | 只在起点搜索一次的总法向预载 |
| `preload_force_tolerance_n` | N | 总预载接纳容差 |
| `maximum_preload_approach_m` | m | 共同 uZ 最大接近量 |
| `drag_length_m` | m | 固定共同 uZ 后的 +x 路径 |
| `path_step_m` | m | 普通路径步长 |
| `free_probe_spacing_m` | m/null | 存在 FREE 针时的再接触探测间距；默认最细 M1 track 分辨率 |
| `effective_unit_tangential_force_min_n` | N | 有效承载长度阈值 |
| `refine_events` | bool | 是否对局部宏观事件定位并在事件共同位姿重评估全阵列 |

## 3. 单个共同位姿

### `ArrayPoseResponse`

| 字段 | shape/含义 |
|---|---|
| `common_ux_m`, `common_uz_m` | 唯一共同背板位姿 |
| `unit_origin_xyz_m` | 当前单元参考点 \(O_U\) |
| `pin_holder_xyz_m` | `[pin,3]` 当前安装点；所有 z 精确相同 |
| `pin_responses` | 同一旧阵列状态、同一共同位姿的 M2 响应 |
| `pin_wrench_about_unit` | `[pin,6]` 搬移到 \(O_U\) 后的逐针 wrench |
| `wall_on_unit_wrench_about_origin` | `[6]` 逐针 wrench 之和 |
| `active_thrust_wrench_about_origin` | `[6]` 主动推力端口；当前规定姿态试验为显式零输入 |
| `guide_reaction_wrench_about_origin` | `[6]` 导向反力端口；当前规定姿态试验为显式零输入 |
| `activity_sets` | 五类活动集 |
| `sharing` | 三类有效针数、最大/均值与 Gini |
| `event_labels` | 当前共同位姿发生事件的 `(pin_index,label)` |
| `residual` | 局部最大残余及 wrench 聚合恒等残余 |
| `proposal_state`, `next_state` | 阵列级 proposal/commit 结果 |
| `proposal_valid` | 任一针失败时为 false，且 `proposal_state` 保持旧状态 |

逐针 M2 wrench 关于当前安装点搬移：

\[
\mathbf M_i^{O_U}=\mathbf M_i^{h_i}
(\mathbf r_{h,i}-\mathbf r_{O_U})\times\mathbf F_i.
\]

## 4. 活动集

| 集合 | 机器字段 | 定义 |
|---|---|---|
| \(\mathcal I_{\rm nom}\) | `active_nominal` | 全部名义安装针 |
| \(\mathcal I_{\rm geom}\) | `active_geometric` | 有有效支撑、通过球冠门控且有闭合机会 |
| \(\mathcal I_+\) | `active_positive_normal` | M2 包络法向力大于数值力容差 |
| \(\mathcal I_{\rm adm}\) | `active_admissible` | proposal、数值、模型范围、球冠、摩擦与行程检查可接纳 |
| \(\mathcal I_{\rm load}\) | `active_target_load` | 逐针 \(|F_x|\) 大于声明的目标载荷阈值 |

三类非负权重：

- normal：M2 包络法向力；
- target tangential：逐针单元局部 \(|F_x|\)；
- resultant：逐针合力欧氏范数。

所有名义针（含零载荷针）进入分母。保存：

- `neff_normal`, `neff_target_tangential`, `neff_resultant`；
- 对应 `max_mean_*`；
- 对应 `gini_*`。

## 5. `path.npz`

主要数组：

| 字段 | shape |
|---|---|
| `path_position_m`, `common_ux_m`, `common_uz_m` | `[N]` |
| `unit_origin_xyz_m` | `[N,3]` |
| `pin_holder_xyz_m` | `[N,pin,3]` |
| `pin_center_xz_m` | `[N,pin,2]` |
| `pin_support_xyz_m` | `[N,pin,3]` |
| `pin_wrench_about_holder`, `pin_wrench_about_unit` | `[N,pin,6]` |
| `wall_on_unit_wrench_about_origin` | `[N,6]` |
| `unit_normal_force_n` | `[N]` |
| `tangential_force_positive_n`, `tangential_force_negative_n` | `[N]`，由同一带符号单元 Fx 分支拆分 |
| `unit_moment_nm` | `[N,3]` |
| `pin_normal_force_n`, `pin_tangential_force_n` | `[N,pin]` |
| `contact_state`, `spring_state`, `event_label` | `[N,pin]` |
| 五类 `active_*` | `[N,pin]` bool |
| 三类 `neff_*`、`max_mean_*`、`gini_*` | `[N]` |
| `force_aggregation_residual_n`, `moment_aggregation_residual_nm` | `[N]` |
| `seed`, `terrain_recipe_id`, `configuration_id`, `preload_n`, `model_level` | `[N]` 同状态样本元数据 |

这些字段按同一个点索引共同切片即可形成 M3→M4 样本；禁止跨点挑选力和矩。

## 6. 路径摘要

摘要不产生综合分数。稳定字段包括：

- `initial_preload_success`；
- `total_contact_length_m`、`effective_load_length_m`、
  `maximum_continuous_load_length_m`；
- 单元切向力 peak/median/p10/p25；
- `total_normal_force_range_n`；
- 三类 `neff_*_median`；
- 最大逐针合力和最大载荷集中；
- 事件计数；
- 局部几何及合力/合矩聚合残余；
- 独立 numerical/model/run terminal 状态。
