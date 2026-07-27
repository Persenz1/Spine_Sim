# M2 → M3 交接

## 冻结入口

M3 只能调用规定安装座位姿核心，不得调用单刺完整试验包装器：

```python
from spine_sim.contact import (
    PrescribedPoseConstitutiveCore,
    SingleSpineState,
    SpineParameters,
)

core = PrescribedPoseConstitutiveCore(parameters, track)
trial = core.solve_pose((x_h, z_h), old_state, commit=False)
```

`trial.proposal_state` 是当前针的候选状态。M3 必须从同一个旧 `ArrayState` 对所有
针试算；全部针通过后才一次性采纳各自 proposal。禁止先调用 `commit=True`
改变部分针，再求其余针。

## 力和 wrench 语义

- `wall_on_spine_force_xz_n`：墙面对针；
- `spine_on_plate_wrench_about_holder`：针作用在板上的反力与力矩；
- wrench 顺序 `[Fx,Fy,Fz,Mx,My,Mz]`；
- 参考点是该针当前安装点；
- M3 搬移到单元参考点后再求和；
- 主平面映射为全局/单元局部三维时，局部 \(y\) 分量为零，不能据此虚构二维
  切向能力。

## 状态提交

- `commit=False`：`next_state` 保持旧状态，proposal 可检查；
- `commit=True`：只适合独立单刺包装器；M3 应由阵列层原子提交 proposal；
- `proposal_valid=false` 时不得提交；
- FREE/DETACH 的力和 wrench 严格为零，但下一共同位姿必须继续查询再接触；
- `near_tie` 和 M1 模型警告不是物理失败。

## M3 必须检查

1. `numerical_state == converged`；
2. `proposal_valid`；
3. `model_state` 是否允许当前用途；
4. `cap_gate_passed`；
5. 几何、结构和力分解残余；
6. 单边法向、摩擦和弹簧行程余量；
7. 杆体净空是否已在项目地形上闭合。

## 不得复用

- 不复用 `SingleSpineExperiment` 的独立路径结果拼阵列；
- 不为每针独立搜索初始或拖动期 Z；
- 不将 `initial_preload_infeasible` 用于阵列拖动脱离；
- 不跨 seed、路径位置或状态拼接力和力矩。

## 当前未闭合接口

`TrackGeometry` 不带二维原始高度查询，因此项目随机地形上的杆体/锥段净空仍标记
为 `parameter_unclosed`。M3 正式能力样本前必须在 M1/M2 接口层补充只读净空查询
或明确批准保守几何边界。
