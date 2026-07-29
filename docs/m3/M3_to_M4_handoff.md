# M3 → M4 动力学交接

**来源版本：** `m3.4.0`
**模型等级：**
`project_model_P_common_rigid_backplate_z_dynamic_continuous_total_preload_v4`

## 冻结入口

M4 只读取 M3 保存的同一时刻样本，不在 M4 内重跑针单元，也不把不同时间、seed、
预载或接触分支的有利力、力矩和活动集拼接。标准数组位于 M0 case 的
`path.npz`；结构约束见 `M3_TO_M4_SAMPLE_SCHEMA.json`。

同一点索引至少共同读取：

```text
seed
terrain_recipe_id
region_id
terrain_data_sha256
configuration_id
selected_unit_origin_xy_m
time_s
path_position_m
external_total_preload_n
backplate_position_xyz_m
backplate_velocity_xyz_m_s
backplate_acceleration_xyz_m_s2
total_contact_reaction_z_n
backplate_inertia_force_z_n
backplate_damping_force_z_n
pin_normal_force_n
pin_tangential_force_n
pin_normal_impulse_n_s
pin_tangential_impulse_n_s
pin_wrench_about_unit
wall_on_unit_wrench_about_origin
active_*
contact_state
event_label
dynamic_residual_n
energy_residual_j
relative_energy_residual
cumulative_energy_residual_j
cumulative_energy_reference_j
cumulative_relative_energy_error
implicit_euler_dissipation_increment_j
normal_contact_work_increment_j
tangential_contact_work_increment_j
contact_energy_injection_increment_j
contact_work_identity_residual_j
actual_time_step_s
model_level
```

## 载荷语义

- `external_total_preload_n` 是施加于整个共同背板的持续恒定外力，不是逐针预载；
- `terrain_recipe_id`、`region_id` 和 `terrain_data_sha256` 共同冻结 M1 上游地形，
  `selected_unit_origin_xy_m` 是碰撞规避后实际重放并输出样本的落点；
- `total_contact_reaction_z_n` 不要求逐点等于外载：脱离期可更小，冲击期可显著更大；
- 首版共同背板只有 Z 动力学，x 是规定拖动运动，俯仰/横滚锁定；
- M4 不得把背板惯性力、阻尼力或外载重复加入墙面接触 wrench；
- 稳态承载样本和冲击样本必须按 `sample_class` 分开。冲击峰不得作为持续拉力包络。

## 坐标、作用对象和参考点

- M3 单元局部系：`+x` 为拖动方向，`+z` 为墙面指向背板；
- wrench 顺序是 `[Fx,Fy,Fz,Mx,My,Mz]`；
- `pin_wrench_about_holder` 已关于当前逐针安装点；
- `pin_wrench_about_unit` 已执行
  \((r_h-O_U)\times F\) 搬移；
- `wall_on_unit_wrench_about_origin` 是同一时刻全部
  `pin_wrench_about_unit` 的精确和；
- M4 不得再次把逐针安装点力臂重复计矩；
- 主动推力与导向反力是独立端口，不能冒充墙面接触能力。

## 模型门禁

当前已给出明确的 `engineering_proxy_v1` 经验基线和敏感性范围，但背板质量/阻尼、
针模态参数、摩擦/冲量接触参数和内部时间步仍没有项目实验标定来源。圆柱杆二维
净空已有路径后检查，杆体/锥段的动态接触仍未闭合。因此结果为
`model_state=parameter_unclosed`、`formal_ranking_eligible=false`，只能用于实现
验收，不能形成正式 M4 能力曲线。

正式 M4 输入仍需：

1. M1 砂纸/红砖/混凝土各 100 seed 的正式配对 catalog；
2. M3 动力学时间步减半、接触/沉降阻尼敏感性和项目参数冻结；
3. 1344 种完整构型在 0.5/1/2 N、100 mm 协议下完成配对分析；
4. 最终候选同 realization 的 10/5 μm 地形收敛；
5. 项目地形净空边界、粗糙面大阵列动力学残差和接触稳定化注能收敛；
6. 相关 case 的 numerical/model/run-terminal 门禁全部通过。

## 禁止复用

- 禁止读取任何 legacy fixed-Z 样本作为正式能力；
- 禁止跨样本拼接；
- 禁止把瞬时冲击峰当作持续承载；
- 禁止把首个单针事件当作阵列极限；
- 禁止在缺少局部 y 响应时声称完整六维被动能力。
