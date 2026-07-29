# M2 单刺动力学求解规格

**当前模块版本：** `m2.2.0`
**模型等级：** `project_model_P_main_plane_dynamic_constant_preload_v2`
**权威物理定义：** `M2_DYNAMIC_CONSTANT_PRELOAD_SPEC.md`

当前实现是固定步长 `moreau_implicit_euler` 加刚性 `rigid_moreau` 接触。下文中的
有限刚度、自适应步长等替代方案属于未来扩展边界，不是 `m2.2.0` 已接受的配置。

## 1. 输入和未知量

规定水平输入：

\[
x_h(t)=x_h^0+v_{\rm drag}t.
\]

动力学未知量至少包含安装座竖向位移及针体轴向/横向动态变形。推荐广义坐标：

\[
\mathbf q=[z_h,u_a,u_b]^T.
\]

正式配置必须显式给出持续外部预载、质量、阻尼、接触定律、摩擦定律和时间积分
设置。M1 `TrackGeometry` 仍以球心横坐标查询有限球针尖包络。

## 2. 运动学

\[
\mathbf c=
[x_h(t),z_h]^T+L\mathbf a-u_a\mathbf a+u_b\mathbf b,
\]

\[
\mathbf a=[\cos\alpha,-\sin\alpha]^T,\qquad
\mathbf b=[\sin\alpha,\cos\alpha]^T.
\]

接触间隙：

\[
g=c_z-H_R(c_x).
\]

实现必须计算 \(\dot g\)、相对切向速度和接触雅可比，不能用相邻输出点的力符号
猜测滑动方向。

## 3. 动力学平衡

\[
\mathbf M\ddot{\mathbf q}+
\mathbf C\dot{\mathbf q}+
\mathbf r_{\rm structure}(\mathbf q)
=\mathbf Q_{\rm preload}+\mathbf J_c^T\mathbf f_c+\mathbf Q_{\rm drive}.
\]

\(\mathbf Q_{\rm preload}\) 在整个路径中保持恒定。接触反力是方程解，不是被强制
等于预载的输入。

轴向结构保持 LOWER_STOP/INTERIOR/HARD_STOP 分段；横向恢复力的零频极限必须与
Timoshenko 端部柔度一致。动态降阶的模态质量和阻尼必须显式记录。

## 4. 接触与摩擦

`m2.2.0` 使用刚性非光滑时间步进，法向和切向冲量由 Moreau 投影求解；法向冲量
非负，并显式记录恢复系数、位置修正、激活容差、冲击速度阈值、接触力安全上限和
投影迭代数。当前配置不接受罚刚度或接触阻尼字段。

摩擦分为 STICK/SLIDE，并使用 `SpineParameters` 中显式静/动摩擦系数。若未来
加入平滑或切向弹簧正则化，其参数必须进入 case identity，且不得静默复用当前
模型版本。

`no_admissible_contact_equilibrium` 只属于旧准静态 fixed-Z 求解器，不得出现在
新版生产路径。

## 5. 初态

在 \(x_h^0\) 下求持续预载对应的静态平衡，速度置零。静态初态搜索失败只能归因于：

- 地形/几何出界；
- 弹簧行程或结构安全边界；
- 动力学参数未闭合；
- 数值搜索在回退后失败。

不能因为某个局部接触分支不存在就永久判定无法预载；必须搜索可达稳定接触。

## 6. 时间积分

`m2.2.0` 使用固定内部 `time_step_s`，输出按独立的 `output_spacing_m` 和主要事件
保存。积分器支持：

- 接触开闭；
- 冲击；
- 刚性结构/接触尺度；
- 确定性重放。

每个 case 保存积分器名称、固定实际步长、接受/拒绝步数和最大步数。时间步收敛
通过独立的 `time_step_s` 减半配对 case 完成；当前单次运行不会自适应回退，也
不会把一次成功路径标记为已完成时间步收敛。

## 7. 状态与事件

状态：

- `free`；
- `stick`；
- `slide`；
- `impact`；
- `recontact`；
- `lower_stop`；
- `interior`；
- `hard_stop`。

宏观事件：

- `first_contact`；
- `detach_to_free`；
- `impact`；
- `recontact`；
- `slip_start`；
- `stick_recovered`；
- `hard_stop`。

普通事件不终止路径。

## 8. 残余与审计

逐步至少审计：

- 动力学平衡残余；
- 单边接触/穿透余量；
- 摩擦锥、单边冲量和接触投影残余；
- 弹簧行程余量；
- 动量变化；
- 动能、结构能、接触能；
- 外部预载功和水平驱动功；
- 摩擦/阻尼耗散；
- 总能量残余。

## 9. 兼容边界

旧 `LegacyPrescribedPoseConstitutiveCore` 可暂留给 M3 迁移和解析静态夹具，但必须标记
`legacy_fixed_pose_quasistatic`，其完整路径结果永远
`formal_ranking_eligible=false`。

M3 新实现不得逐针运行独立 M2 路径；它应组装 M2 动力学单元的质量、恢复力、
接触力和状态，在共同背板自由度上统一积分。
