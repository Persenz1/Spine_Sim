# Canonical 结果数据字典

当前版本：

| 维度 | 值 |
|---|---|
| project schema | `2` |
| model schema | `canonical-single-array-1` |
| result schema | `canonical-result-2` |
| solver semantics | `single-array-event-v1` |
| geometry schema | `contact-candidate-2` |
| parameter registry | `canonical-parameters-1` |

这些版本、terrain/geometry 版本与规范化输入哈希均进入 case identity；语义变化不会覆盖旧结果。

## Campaign 根目录

| 文件 | 合同 |
|---|---|
| `manifest.json` | schema/model/result/solver 版本、campaign ID、backend、索引格式、状态计数和 result-set hash |
| `config/original.json` | 用户提供的原始 campaign |
| `config/normalized.json` | 完整规范化 `CampaignSpec`、case IDs 和 campaign ID |
| `lineage.json` | 每 case 的 module/version、config/upstream/input hash、全部语义版本与 terrain/geometry 版本 |
| `cases.parquet` | 安装 pyarrow 时的 case 索引 |
| `cases.jsonl` | pyarrow 缺失时的明确降级；manifest 记 `jsonl_fallback` |
| `paths/<case_id>/` | 单 case 原子工件 |
| `events.jsonl` | 聚合物理事件 |
| `validation.json` | campaign 级检查 |

`run-case` 从源 campaign 选择单 case 后会生成新的 campaign identity 和完整 normalized config，不复用源 campaign ID。

## 单 case 工件

- `config.json`：`BaseCaseSpec`，含全部版本和输入。
- `summary.json`：canonical metadata、物理结果、运行诊断和内容 hash。
- `path.npz`：可选命名数组。
- `trace.parquet`：canonical trace；嵌套字段以 schema metadata 标记的 `canonical-json-columns-v1` 无损编码。
- `trace.jsonl`：pyarrow 不可用时的明确降级。
- `events.jsonl`：每行一个物理事件。
- `validation.json`：残差、夹具和 schema 检查。
- `COMPLETE`：所有工件写完后最后生成，内容等于 result hash。

读取 trace 使用 `spine_sim.io.read_trace_table`；它会还原嵌套列表、mapping、空 `parameter_sources={}` 和 null，不应直接把 Arrow 内部 JSON 字符串当物理字段。

## Canonical summary 必备字段

公共字段：

- `project_schema_version/model_schema_version/result_schema_version/solver_semantics_version`
- `case_id/normalized_input_hash/terrain_version/geometry_version`
- `model_level`：仅 `single_spine_quasistatic` 或 `array_rigid_backplate_event`
- `parameter_provenance/units/frames`
- `assumptions/omissions/applicability/cannot_answer`
- `physical_state/numerical_state/model_state/run_state`
- `residuals/tolerances/per_spine`

阵列结果另须独立保存：

- `rank_status`
- `range_status`
- `equilibrium_status`
- `quasistatic_stability`
- `dynamic_stability`

不得用一个 `success/stable` 布尔量合并这些维度。

## 状态

物理八态：

`SEARCH / CONTACT / STICK / SLIP / DETACH / REBOUND / HARDSTOP / FAILED`

`CONTACT` 只存在于 trial/event 转换，不作为 accepted 常驻状态。其余维度独立：

- numerical：`NOT_RUN / CONVERGED / NONCONVERGED / INVALID_RESIDUAL`
- model：`CLOSED / PARAMETER_UNCLOSED / OUT_OF_SCOPE`
- run：`PENDING / RUNNING / COMPLETE / CANCELLED / EXECUTION_ERROR`

物理事件固定为：

`CONTACT, CONTACT_REJECT, STICK_START, SLIP_START, RESTICK, DETACH, REBOUND_START, REBOUND_COMPLETE, REENGAGE, HARDSTOP, HARDSTOP_RELEASE, MATERIAL_FAILURE`。

数值、模型和运行事件使用各自 enum，不混进 `EventType`。事件保存 `sequence/from_state/to_state/case_id/spine_id/load_parameter/path_position_m/details`。

## Wrench

任何 wrench 必须同时保存：

- `force_N[3]`
- `moment_Nm[3]`
- `frame`
- `reference_point`
- `acting_on`
- `exerted_by`
- `interaction_label`
- `sign_convention=right_hand_rule`

参考点搬移向量为“新参考点到旧参考点”，`M_new=M_old+r_new_to_old×F`。

## 阵列计数和指标

- `n_nominal`：输入刺数
- `n_geometric`：有效几何候选
- `n_contact`：事件后闭合候选
- `n_engaged`：稳定挂接数；不可评估时为 null
- `n_engaged_lower/n_engaged_upper/n_evaluable`：未知项边界与可评估数
- `n_active`：当前非零承载活动集
- `P_sum_N/P_avg_N`：局部接触法向反力和；零接触时平均值为 null
- `n_share_normal/n_share_tangent_positive`：重命名后的 inverse-Simpson
- `load_sharing_index`：理论幅值共享指标

路径指标 `J_positive/J_negative/J_net` 只跨相邻 accepted 且 valid 点积分，分子与有效长度使用同一连续段集合；missing/invalid 不补零、不跨缺口连接。
