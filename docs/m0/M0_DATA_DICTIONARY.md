# M0 结果字段数据字典

Schema 版本：`1`

## Campaign 根目录

| 文件 | 必备内容 | 说明 |
|---|---|---|
| `manifest.json` | `schema_version`, `campaign_id`, `backend`, `index_format`, `status_counts`, `result_set_hash` | 后端选择和索引格式显式记录 |
| `config/original.json` | 原始输入 | 原样语义保存，不作为 hash 的非规范文本来源 |
| `config/normalized.json` | SI/规范化配置、case ID | 供复现与统计读取 |
| `cases.parquet` | 每 case 一行 | 安装 `pyarrow` 时生成 |
| `cases.jsonl` | 与 Parquet 相同的行 | 无 `pyarrow` 时的 CPU-only 降级；`manifest.index_format=jsonl_fallback` |
| `paths/<case_id>/` | 单 case 工件 | case 隔离和恢复边界 |
| `events.jsonl` | 聚合有序事件 | M0 定义结构，不判断物理成立条件 |
| `validation.json` | campaign 级验证 | 初始为 `not_run` |
| `lineage.json` | module/config/upstream hash | 追踪 M1→M4，不构建图数据库 |

## Case 索引字段

| 字段 | 类型/单位 | 含义 |
|---|---|---|
| `case_id` | string | 规范输入与模块版本产生的确定性 ID |
| `run_state` | enum string | `complete`, `cancelled`, `execution_error` 等运行状态 |
| `result_hash` | sha256 | 排除时间、墙钟、内存等非确定性诊断后的内容 hash |
| `wall_time_s` | float, s | case 墙钟时间 |
| `peak_ram_bytes` | int, byte | Windows PeakWorkingSetSize 或 Unix `ru_maxrss` |
| `peak_python_bytes` | int, byte | `tracemalloc` 观测的 Python 分配峰值，不宣称是进程 RSS |
| `error_category` | enum/null | 配置、模型未闭合、数值、执行或取消 |
| `error_type` | string/null | 异常类名 |
| `error_message` | string/null | 诊断消息 |

`summary.json` 另记录 `peak_vram_bytes`；当前 CPU/M0 runner 写 `null`，避免伪造未测量数值。GPU M1 后端可在已知时填入实测值。

## 单 case 文件

- `config.json`：规范 case spec，含 `module`, `module_version`, `parameters`, `upstream_hash`, `tags`。
- `summary.json`：四维状态、模块指标、backend、阶段耗时、完成时间和内容 hash。
- `path.npz`：命名逐步数组；每个数组的单位必须进入字段名或模块数据字典。
- `events.jsonl`：每行一个结构化事件。
- `validation.json`：残余、范围、夹具结果和说明。
- `COMPLETE`：只在全部原子文件写完后生成，内容为结果 hash。

## Wrench

`Wrench` 禁止裸六元组，必须同时保存：

- `force_N[3]`
- `moment_Nm[3]`
- `frame`
- `reference_point`
- `acting_on`
- `exerted_by`
- `interaction_label`，例如 `spine_on_plate`, `wall_on_unit`
- `sign_convention=right_hand_rule`

参考点搬移输入向量定义为“新参考点到旧参考点”，公式为 `M_new = M_old + r_new_to_old × F`。

## 状态和事件

四维状态字段不可互换：

- `physical_state`
- `numerical_state`
- `model_state`
- `run_state`

事件类型：`contact`, `detach`, `recontact`, `slip_start`, `hard_stop`, `terrain_bounds`, `numerical_retry`, `numerical_failure`, `cancelled`, `resumed`。事件包含 `sequence`, `case_id`, 可选 `path_position_m`, `timestamp_utc`, `details`。
