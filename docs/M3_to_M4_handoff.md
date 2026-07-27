# M3 → M4 交接

## 冻结入口

M4 读取 M3 保存的同状态样本，不在 M4 内实时重跑全部针，也不把不同点的最有利力、
力矩和活动集拼在一起。

标准数组位于 M0 case 的 `path.npz`。以相同点索引读取：

```text
seed
terrain_recipe_id
configuration_id
preload_n
path_position_m
unit_normal_force_n
tangential_force_positive_n
tangential_force_negative_n
unit_moment_nm
active_*
contact_state
event_label
force_aggregation_residual_n
moment_aggregation_residual_nm
model_level
```

结构约束见 `M3_TO_M4_SAMPLE_SCHEMA.json`。

## 坐标、作用对象和参考点

- M3 单元局部系：`+x` 为拖动方向，`+z` 为墙面指向背板，`y=z×x`；
- `wall_on_unit_wrench_about_origin` 的顺序是 `[Fx,Fy,Fz,Mx,My,Mz]`；
- 参考点为当前共同背板单元中心 \(O_U\)；
- 逐针输入是 M2 `spine_on_plate_wrench_about_holder`，M3 已执行
  \((r_h-O_U)\times F\) 搬移；
- M4 不得再次把逐针安装点力臂重复计矩；
- 主动推力与导向反力保存在独立端口，不能重复加入墙面接触 wrench。

## 法向与切向能力

- `unit_normal_force_n` 是同一状态下各针非负 M2 包络法向力之和；
- 正/负切向字段由同一个带符号单元 `Fx` 拆分，不是分别从不同路径点取峰值；
- 当前 M3 是局部 x-z 主平面模型，没有单元局部 y 能力；
- M4 对角加载只能在声明的轴向 ROM 边界内解释，不能虚构完整二维摩擦包络；
- 只在样本实际覆盖的法向范围内插值，禁止向上外推。

## 模型门禁

解析 M3 夹具为 `model_state=covered`。现有十个项目随机地形仍因二维杆体/锥段净空
接口未闭合而是 `parameter_unclosed`，仅可用于接口 smoke，不能形成正式 M4 能力曲线。

正式 M4 能力输入还需：

1. M2 正式轮次和用户批准参数包；
2. M3 第一轮与细筛获批并完成；
3. 项目地形净空闭合或经用户明确批准的保守边界；
4. 目标 0.5/1/2 N 法向范围的同状态样本覆盖；
5. 相关 case 的数值、模型和 run terminal 门禁通过。

## 不得复用

- 不跨 seed、路径位置、预载或事件分支拼接；
- 不使用 `active_thrust_wrench` 或 `guide_reaction_wrench` 冒充墙面能力；
- 不把十地形 smoke 当作参数排序；
- 不把首个针事件当作单元极限；
- 不在缺少局部 y 响应时声称完整六维被动能力。
