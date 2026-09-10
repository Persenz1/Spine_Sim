# IJMS 批量仿真交接

DeepSeek / harness 接手时，先加载仓库 `AGENTS.md` 和 [IJMS物理约束](IJMS物理约束.md)。它们区分可自主优化的 coding 与必须保持的物理、状态和指标含义；不要将本页的跑批安排当成改变物理模型的授权。

用户已进一步固定分工：后续模型只处理代码、既定配置的批量执行与动作监控，不自行承担参数估计方案选择、研究范围设计或论文结论解释。以下配置和参数方法供执行已给定方案；示例不是后续模型任意扩展科研范围的授权。

本轮方案见 [IJMS阵列扫描规划](IJMS大规模扫描规划.md) 与 [ijms_scan_plan.json](../experiments/ijms_scan_plan.json)：圆钝尖50/100 μm、2 mm渐细段、直径1 mm主体，均匀角露出4 mm，梯度角按爪跟4 mm补偿。0.6.0已实现实际变截面弯曲/导数/应力、针形接触及无弹簧固定安装。小阵列直接使用[启动说明](IJMS小阵列启动.md)和[已锁定粗筛配置](../experiments/ijms_small_array.json)，不再从旧示例重新挑参数。阵列先小后大、上限30×30，大阵列预载/地形在小阵列后确定；当前JSON不能直接作为campaign运行。

后续任务使用 `spine_sim.ijms:run_case`，模板是 `examples/ijms_campaign.json`。入口执行首次接触、逐步建立总预载、带内部历史拖动和指标计算。不要把旧 `canonical_module` 的平壁单次平衡入口用于新版论文跑批。

runner 已按这个 callable 接入新版模型、求解器、几何与装配版本，并将对应版本写入结果 manifest。生成配置时继承模板版本，不使用旧 canonical 配置拼接新版 parameters。当前搜索距离从预载完成的拖动起点计；接近阶段定位首次接触的 Z，不输出一个尚未声明起始高度的接近行程。

模板中的材料、尺寸、摩擦、目标力和行程是未标定的数值示例。这里说明配置和运行方法；机理、离散近似及本次实际验证见新版机理说明。当前没有给出可直接外推到大型阵列或百万 case 的吞吐率。

## 1. 准备一个独立任务目录

在仓库根目录执行以下 PowerShell。`ijms_batch_20260910` 应改为本次任务名。过程配置、先导结果、临时文件都放在 `E:\Agent_Tmp_WS` 下；仓库源码和已有 `.venv` 留在原处。不要在 PDrive 建临时目录。

```powershell
Set-Location -LiteralPath 'D:\Code\Spine_Sim'
$ijmsPython = 'D:\Code\Spine_Sim\.venv\Scripts\python.exe'
$ijmsTask = 'E:\Agent_Tmp_WS\ijms_batch_20260910'
New-Item -ItemType Directory -Force -Path "$ijmsTask\config", "$ijmsTask\tmp" | Out-Null
$env:TEMP = "$ijmsTask\tmp"
$env:TMP = "$ijmsTask\tmp"
$env:PYTHONDONTWRITEBYTECODE = '1'
Copy-Item -LiteralPath 'examples\ijms_campaign.json' -Destination "$ijmsTask\config\base.json"
& $ijmsPython -B -c 'import numpy, scipy; import spine_sim.ijms'
```

若当前环境尚未安装本项目或缺依赖，用该解释器执行 `-m pip install --no-cache-dir -e .`。新版力学计算依赖 NumPy、SciPy，使用 CPU。所有运行命令显式给 `--backend cpu`；安装了 CUDA 不会把这套力学求解器变为 GPU 求解器。

先修改复制后的 `base.json`，保留仓库中的数值示例。JSON 保存为 UTF-8 无 BOM。CLI 的配置读取使用 UTF-8；不要用旧 PowerShell 的默认 UTF-16 重写 JSON。

## 2. 先导运行与正式计算预算

一个最小 case 可直接运行：

```powershell
& $ijmsPython -B -m spine_sim.cli run-case "$ijmsTask\config\base.json" `
  --backend cpu --output "$ijmsTask\pilot_results"
```

命令打印真实 `campaign_dir`。`run-case` 仅取配置的第一条 case，并强制单进程、`small` 模式；多条 case 要用 `run-campaign`。`small` 的 Python 内存跟踪会明显增加有限杆求解耗时，估计正式吞吐时将复制后的配置设为 `mode="formal"` 并用 `run-campaign`。先导只确认所选配置能执行，并记录当前机器上的耗时、内存和停止类型。不要把进程退出成功称为科学标定完成。

按已给定代表配置测预算，读取摘要的 `wall_time_s`、`peak_ram_bytes`、`stage_times_s`，结合结果目录大小估算资源。执行模型可以据此调整并行数、分片大小和并发节奏；总样本量、研究范围和保真度按已定方案，不自行扩大或降低。当前分片生成器只控制每批 case 数，不决定单条路径的成本，也不保证大阵列耗时线性增长。

`workers` 可写在 base campaign 顶层，或通过 `run-campaign/resume --workers N` 指定。优先使用少量 CPU 进程；每个 worker 会持有自己的求解状态和地形数据。不要在没有内存和耗时估计时把 worker 数直接设成机器全部逻辑核。

## 3. 设计覆盖和配对样本

`base.json` 必须恰好包含一条模板 case。生成器继承其 `callable`、module、版本、运行模式和 workers，只允许设计项覆盖 `case.parameters`。

`designs.json` 顶层是 `designs` 列表。例如以下仅演示均匀与异质弹簧的输入方法，资源公平性仍应按实际设计核对：

```json
{
  "designs": [
    {
      "name": "uniform_375",
      "parameters": {
        "array": {
          "spine": {"spring_stiffness_N_per_m": 375},
          "per_spine": []
        }
      }
    },
    {
      "name": "heterogeneous_300_450",
      "parameters": {
        "array": {
          "spine": {"spring_stiffness_N_per_m": 300},
          "per_spine": [{"index": 1, "spring_stiffness_N_per_m": 450}]
        }
      }
    }
  ]
}
```

覆盖规则是**字典递归合并，列表整体替换**。模板原有第 2 根刺的刚度覆盖不会因为改变 `array.spine` 而自动消失；均匀设计需要显式写 `per_spine: []`。改变 `per_spine`、角度矩阵、落点列表时，要提交整个目标列表。`null` 是一个值，不是删除键的指令。切换表面类型，或改用 `common_mouth_height_m` 并删除 `free_length_m` 时，直接编辑任务 base 中的完整对应对象，避免留下上一种配置的键。

阵列常用字段：

- `nx`、`ny` 和 `spacing_x_m`、`spacing_y_m` 定义无载针尖位置；编号为 `index = j * nx + i`，先沿 x，再沿 y。
- `theta_deg` 可为公共标量、每列列表或 `ny × nx` 矩阵。`per_spine` 可覆盖某根刺的角度、长度、直径、半径、弹簧和材料参数。显式 `tip_positions_xy_m` 决定针数；使用角度矩阵时，矩阵元素数仍须匹配位置列表，标量角会自动广播。
- `angle_gradient_deg={"toe":60,"heel":80}`和`heel_free_length_m=.004`用于本轮梯度长度补偿；沿+X爪头在前，爪跟在后。
- `spine.taper_length_m=.002`、`diameter_m=.001`定义渐细钢针；`mount_type`为spring/fixed，弹簧额定行程4 mm，完全縮回是范围停止。
- `common_mouth_height_m` 根据角度和尖端半径确定各刺长度，不能与显式 `free_length_m` 同时使用。安装关系改变后，孔口间距不必等于针尖间距。
- `loaded_area_m2` 是受压背板面积；改变刺数或间距时不会自动更新，固定压力试验必须同步给出真实面积。
- `solver.y_mode` 为 `free` 或 `locked`。`y_bounds_m` 是自由 Y 的模型范围；目前不是实体侧向限位接触模型。

`samples.json` 是固定样本列表，例如：

```json
[
  {"surface_seed": 17, "placement_seed": 3, "split": "train", "placement_xy_m": [0, 0]},
  {"surface_seed": 23, "placement_seed": 4, "split": "validation", "placement_xy_m": [0.0001, 0]},
  {"surface_seed": 31, "placement_seed": 5, "split": "test", "placement_xy_m": [0, 0.0001]}
]
```

生成器对每个设计复用同一张样本表，不重新抽样。adapter 用 `surface_seed` 覆盖表面种子，将 `placement_xy_m` **加到** `path.start_xy_m`。`placement_seed` 当前只作可复现记录，不自动生成落点；若要随机落点，应先按声明的可用区域生成固定坐标表。仅改变记录种子而没有改变实际表面或位置，不构成新的独立重复。

`split` 可取 `train`、`validation`、`test`。生成器拒绝完全重复的样本记录，也拒绝仅更改 split 的重复记录；它不替代对实际表面与落点独立性的判断。平面和确定性正弦表面不因 `surface_seed` 改变而出现新的随机实现；重复平面样本只能用于数值运行检查。

随机粗糙面可在任务 base 中把整个 `surface` 对象替换为以下形式，再根据研究需要设范围。数值仅演示格式：

```json
{
  "kind": "gaussian",
  "shape": [161, 321],
  "dx_m": 0.000025,
  "rms_height_m": 0.000005,
  "correlation_x_m": 0.0002,
  "correlation_y_m": 0.0001,
  "orientation_deg": 0,
  "origin_xy_m": [-0.003, -0.002],
  "seed": 17,
  "version": "gaussian-pilot-v1"
}
```

还支持 `plane`、`sinusoidal`、`heightfield` 和已有地形 API 的 `material` 输入。`heightfield` 可给 `.npy` 的绝对 `path`、`dx_m`、可选 `dy_m` 与 `origin_xy_m`，或直接给 `height_m`。有限表面必须覆盖拖动、侧移及整段针杆检查所需区域，不只覆盖球心路径。合成材料标签不是实体材料标定。

正式运行所依赖的 `.npy` 应放在保留的数据目录，不能指向交付后将删除的任务临时文件。改变实际表面来源时同步更新 case 的 `terrain_version` 与表面描述；不要随意改求解语义版本来绕过旧结果的不兼容。

## 4. 生成分片、运行和恢复

将上述文件写入 `$ijmsTask\config` 后生成分片：

```powershell
& $ijmsPython -B scripts\generate_ijms_campaign.py `
  --base "$ijmsTask\config\base.json" `
  --designs "$ijmsTask\config\designs.json" `
  --samples "$ijmsTask\config\samples.json" `
  --output-dir "$ijmsTask\config\shards" --shard-size 1000
```

输出为 `campaign_00001.json` 等。生成器逐分片写出，不物化全部组合；每个分片进入现有 runner 时仍会加载该分片的所有 case 配置。重复的有效设计参数会被拒绝。已有同名分片不会覆盖；新配置用新目录生成。

正式预算已确定后，指定一个明确保留的结果目录。下例中的目录是**正式交付结果及输入配置**，不属于任务结束要清理的临时目录：

```powershell
$ijmsResultRoot = 'E:\IJMS_Results\ijms_batch_20260910'
New-Item -ItemType Directory -Force -Path $ijmsResultRoot | Out-Null
Copy-Item -LiteralPath "$ijmsTask\config" -Destination "$ijmsResultRoot\input_configs" -Recurse
Get-ChildItem -LiteralPath "$ijmsResultRoot\input_configs\shards" -Filter 'campaign_*.json' |
  Sort-Object Name | ForEach-Object {
    & $ijmsPython -B -m spine_sim.cli run-campaign $_.FullName `
      --backend cpu --workers 2 --output $ijmsResultRoot
    if ($LASTEXITCODE -ne 0) { throw "Campaign execution failed: $($_.FullName)" }
  }
```

每个分片写入结果根目录下按配置身份生成的独立 `campaign_*` 目录。不要同时启动两个进程写同一个 campaign 目录。上例串行处理分片，每片内部并行执行 case。

中断后，使用**原分片文件、原结果根目录和相同物理配置**，把命令换成 `resume`：

```powershell
& $ijmsPython -B -m spine_sim.cli resume `
  "$ijmsResultRoot\input_configs\shards\campaign_00001.json" `
  --backend cpu --workers 2 --output $ijmsResultRoot
```

恢复粒度是 case：已完整落盘的 case 跳过，被中断而未完整保存的 case 从头运行，不从某个拖动站点继续。可对所有原分片逐个调用 `resume`。`retry-failed` 仅重跑已有执行错误记录，不会自动重跑正常返回的 `NUMERICAL_FAILURE`、模型边界或未建立承载的物理结果。

改了设计、网格、步长或目标协议后，用新配置和新输出任务进行比较。调整 worker 数可通过命令行覆盖；不要修改求解参数后把重算称为同一次断点恢复。完成后保留正式结果、输入配置及其引用的数据；只有确认它们不依赖任务临时目录后，才清理该任务在 `E:\Agent_Tmp_WS` 下的专属子目录。

## 5. 输出与状态解释

`output.level` 可为 `summary`、`trace`、`full`。`summary` 适合常规批量；`trace` 保存阵列站点序列；`full` 额外保存逐刺数据及杆内部坐标，适合代表路径和边界问题。减少输出不改变物理求解的自由度、离散段数和步长。`mode: formal` 是 runner 的批量存储/运行选项，不是高保真模型开关。

使用 `spine_sim.io.results.open_result_store(campaign_dir)` 读取结果可兼容目录式与紧凑式存储。目录式结果通常位于 `paths/<case_id>/summary.json`，轨迹为 `trace.parquet` 或 `trace.jsonl`。`formal` 且整片均为 `summary` 输出时使用 `case_summaries.sqlite3`，事件保存在摘要的 `events` 列表；`trace/full` 输出则使用独立 `events.jsonl`。不要假定所有运行都有逐 case 目录，也不要把紧凑存储的空聚合事件文件当作“没有发生事件”。配置随结果保存；无需另建重复 manifest 或来源台账。

CLI 的退出码和 `summarize` 统计的是 **runner 是否执行并保存成功**。务必继续读取每条摘要的物理 `status`：

| 字段/状态 | 含义与后续处理 |
|---|---|
| `run_state = execution_error` | 输入、依赖、代码异常或输出写入错误；看 `error`/`diagnostic_traceback` |
| `run_state = complete` | adapter 正常返回并落盘；不保证拖动完成或建立成功 |
| `status = COMPLETED`、`completed = true` | 指定拖动行程执行完；还需分别看指标完整性和建立结果 |
| `NUMERICAL_FAILURE` | 在最小步长下未满足平衡残差要求；不能当作已证明不存在物理解 |
| `NUMERICAL_STEP_BUDGET` | 达到 `solver.max_path_steps` 的路径步数上限；先定位持续缩步原因，再决定是否增加计算预算 |
| `UNRESOLVED_CONTACT_PATH` | 站间球尖路径检查未通过且已缩至最小步长；当前离散未解析出可接受的接触路径，不能越过中间接触继续 |
| `GEOMETRY_DOMAIN`、`ROD_GEOMETRY_DOMAIN` | 球尖或针杆所需表面查询超范围/输入不可用 |
| `MULTIPOINT_CONTACT_LIMIT` | 单球尖单点接触假设到边界；不能复制为独立弹簧或强选一个法向继续 |
| `ROD_COLLISION_LIMIT` | 针杆与表面发生当前模型未处理的接触；不是单尖接触可继续的路径 |
| `ELASTIC_MODEL_LIMIT` | 超出已给材料弹性许用范围；不是断裂删除或损伤结果 |
| `Y_DOMAIN_LIMIT` | 自由 Y 超过声明范围；不是实体侧向挡块反力 |

`failure` 保存停止阶段及尝试的控制参数。预载阶段失败时可能没有拖动数据，此时 `metrics`、`establishment` 为 `null`。这些 case 仍应计入任务总数并单列原因，不能悄悄从论文样本分母删除。

逐刺 `OPEN`、`STICK`、`SLIP` 是局部接触模式，`*_HARDSTOP` 表示压缩触限。滑动或触限均不自动等于阵列路径失败。末态 `final_stability` 是另一个字段：`SLIDING_NONCONSERVATIVE`、各类分支边界和 `NOT_REQUESTED` 不应改写成“已证稳定”。即使局部能量条件满足，也不代表整条路径或真实动力学已经验证。

只对 `phase = drag` 的序列计算搜索性能，用 `search_distance_m` 作为搜索坐标。预载阶段 X 不变；同 X 的事件前后状态按实际保存顺序保留，不通过去重、排序反力或跨阶段拼接制造额外面积。相邻有效拖动站间采用线性力插值；峰值不是经过独立连续极值搜索得到的真实最大值。

主要摘要口径：

- `metrics.full_length_m` 为声明行程，`effective_length_m` 和 `coverage` 表示有效区间覆盖。`J_positive`、`J_negative`、`J_net` 先从阵列总 `T` 分正负，按完整声明长度归一化；不先把逐刺正负面积相加。
- 窗口不完整时正式 J、平均/最低/峰值/分位力为 `null`；`observed_J_*` 是已观察区间对完整分母的贡献，不是把未知区间补零后的完整性能。`force_quantile` 按路径长度加权，不能改用自适应站点的普通分位数。
- `S_req_m` 为满足目标及持续距离的最早窗口起点，`confirmed_at_m` 为确认位置，必须在声明搜索上限内。目标可选 `target_force_N` 或 `target_fraction_P`，二者不能同时给；`persistence_distance_m`、`allowed_below_fraction` 与 `max_contiguous_below_m` 共同定义持续条件。
- `established = false` 表示完整搜索未满足协议；`null` 表示证据不完整。缺口后可观察到成功而更早建立位置未知，此时 `observed_S_req_m` 有值、正式 `S_req_m` 为 `null`。成功概率以独立表面/落点为重复单位，不把相邻站点当作独立样本。
- `P=sum(fz)`、`T=-sum(fx)`、`L=sum(fy)`；负 T 保留。`n_share_normal` 用 `max(P_i,0)`，`n_share_tangent_positive` 用 `max(T_i,0)`。局部 `N_i=f_i·n_i` 的和及分载另存为 `N_sum_local_N`、`n_share_local_normal`。全零权重时分载数未定义。
- `max_equilibrium_residual`、能量残差、`final_stability` 和覆盖率供定位与结果解释，不可用一个“complete”字段替代。`effective_max_step_m` 是实际最大步长：对高度场，入口还会把它限制到网格较小间距的四分之一。

可用以下 Python 读取某个结果目录的执行状态和物理状态，无需依赖存储布局：

```python
from collections import Counter
from spine_sim.io.results import open_result_store

store = open_result_store(r"E:\IJMS_Results\ijms_batch_20260910\campaign_实际ID")
execution = Counter()
physical = Counter()
for record in store.list_records():
    summary = store.load_case_summary(record.case_id)
    execution[summary["run_state"]] += 1
    physical[summary.get("status", "EXECUTION_ERROR")] += 1
print(execution)
print(physical)
```

## 6. 论文测试项目与配置对应

| 测试项目 | 配置/对照方法 | 汇总时保留 |
|---|---|---|
| 方向柔顺与传力 | `theta_deg`、`free_length_m`、弹簧刚度；相同表面和落点 | 力—位移、分载、触限、应力与建立距离 |
| 阵列规模/密度 | 改 `nx/ny/spacing_*`；固定间距扩面积与固定面积增密度分组 | 实际刺数、间距、面积、总 P 和覆盖率 |
| 预载与规模匹配 | `load.mode` 分别为 `total_force`、`pressure`、`nominal_per_spine` | `value` 单位依次为 N、Pa、N/名义刺；每条路径内恒总 P |
| 粗糙度与空间组织 | 同高度幅值，改变相关长度、方向和表面组织；固定随机种子配对 | 表面版本、参数、实现和落点；不能仅按材料标签比较 |
| 均匀/梯度/异质阵列 | 公共参数、角度矩阵与 `per_spine`；匹配资源与装配关系 | 实际每刺参数、材料/面积/刺数/预载预算 |
| 有限搜索建立 | `path.search_distance_m`、目标力、持续距离及掉落规则 | 建立与确认位置、完整性、独立重复建立概率 |
| 数值敏感性 | 代表问题改变杆 `segments`、步长、表面网格；其他输入配对 | 成本、事件/停止类型及目标指标变化 |
| 实体对照/留出条件 | 实际试样参数、同一载荷与 Y 约束；固定校准集与测试集 | 六维力旋量参考点、位移零点、校准来源、重复与样本 split |

固定名义每刺载荷只将 `value * n_nominal` 转成总 P；它不把每根刺实际反力设为同值。固定压力使用 `value * loaded_area_m2`，固定总载荷直接使用 `value`。

实体数据可使用 `research_protocols.SensorCalibration` 做校零、坐标、参考点和符号变换，再用同一建立/路径指标函数比较。该接口不自动完成传感器标定，也不从六维总力恢复逐刺力。短程搜索后的保持、指定方向加载，以及全方向极限求解尚未接入此拖动 CLI；需要确定后续协议后从已接受内部状态继续实现。

## 7. 后续任务仍需确定的研究输入

研究配置需要明确制造/设计参数与资源范围、预载和行程范围、目标力定义、持续距离和允许掉落规则、Y 约束、表面/落点分布、留出条件及计算预算。这些值不要求全部来自实体测量：用户已允许按第 8 节用计算、资料、反演或明确假设补足。尚未确认的实际装置事实应作为显式情景，不能冒充已经确认。旧报告的 5 mm、0.50 mm、10/11 站和 0.25P 不是本轮已经确认的论文协议；示例中的任何值也不应据此冻结。

后续模型可先完成配置整理、指定范围的小批运行和异常定位，再按已给预算执行大批量并监控进度、耗时、内存和退出状态。保留物理范围终止、数值失败和合法零响应的区别；按既定恢复策略处理执行异常。研究范围加密或改变物理配置由科研定稿决定，不默认追加全量回归、反复 smoke 或独立审计材料。交付保留代码、结果、最终配置、运行事实和待处理问题即可。

## 8. 没有逐项实测时，照常开展仿真

用户明确允许用公式与数值反推替代难以直接测量的输入。当前程序没有要求实测文件或 `calibrated=true` 的运行条件；`parameter_status` 是结果说明文本，不控制能否求解。已有两个完整路径示例使用数值设定参数和解析/合成表面。缺少逐项实测不应阻止仿真与模型内结论分析。

本节固定可采用的参数替代方法；后续执行模型只按已给定的来源、公式输入、拟合目标/边界和扫描表计算，不自行寻找一组“看起来合理”的材料值或设定新的研究范围。参数齐全就运行；缺少具体数值就报告缺项，继续其他完整 case。机理解释与结论定稿不属于执行模型职责。

### 参数怎样确定

| 类别 | 可直接采用的方法 | 需要说明的含义 |
|---|---|---|
| 长度、直径、半径、角度、行程、面积 | 设计尺寸、CAD、厂家规格、装配公式；无法确认时作为显式设计变量扫描 | 设计值、名义值或假设区间不冒充制造后实测尺寸 |
| 截面与弯曲刚度 | `A=πd²/4`、`I=πd⁴/64`、`EI`；相容装配长度由几何公式求出 | E 仍需要一个资料值、估计值或研究情景值；几何公式不会确定材料牌号 |
| E、弹簧刚度 | 匹配材料/型号的厂家或文献值、适用的弹簧解析公式、独立数值子模型的等效响应 | 数值等效说明子模型的材料和边界；不能将有耦合的阵列总刚度直接当每刺 ks |
| 摩擦系数、少量有效结构参数 | 相近且明确条件下的资料范围；已有整体力—位移曲线时用完整前向模型反演；没有资料时声明探索区间做敏感性扫描 | 摩擦不能仅由针几何唯一计算；仅可识别的参数组合不强行拆成全部逐刺真实参数 |
| 表面 | 参数化或已有材料生成器的合成表面、公开/已有高度数据；扫描幅值、相关长度和方向 | 结论针对声明表面族；合成“混凝土”名称不证明其代表某一真实墙面 |
| 强度许用值 | 明确材料的资料值、适用关系或显式保守/探索情景；也可省略 `allowable_stress_Pa` | 省略时仍计算弹性响应与应力，许用利用率未定义，不能据此证明该真实材料始终弹性或不会破坏 |
| 逐刺力、回缩、孔外长度、杆形、背板 Y/Z、接触状态 | 直接由相容、接触和整体平衡求解 | **这些是输出，不要求先测出来才能仿真** |
| 目标力、搜索行程、持续窗口 | 按设计目标声明，允许多个研究情景 | 属于研究协议，不是等待测出的材料常数 |

估计方法和必要来源放在现有最终配置的说明中即可；可使用 `parameter_status="theory_estimated"`、`"numerically_identified"`、`"scenario_assumed"` 等描述，程序不限定这些字符串。每条 case 仍需传入完整数值参数；当前入口不会自动替调用方查材料或拟合未知参数。

### 数值反推的两种用途

**有参考响应时做参数辨识。** 参考可以是少量整体力—位移数据、已有记录、可核实文献曲线或独立子模型结果。外层优化器反复调用 `ijms.simulate(parameters)`，拟合少量共享参数或有效参数组合，例如最小化多工况 `Σw_j||F_model(X_j;θ)-F_ref(X_j)||²`，并保留几何、加载和物理约束。权重处理各力/力矩分量单位与尺度。当前提供前向接口，尚无现成的通用参数反演 CLI；待参考数据、待拟合参数、目标、权重和边界已经给定后，后续模型可实现并执行这一 coding 功能，不自行选择这些科研条件。

一条整体曲线未必能唯一识别所有局部参数。阵列斜率同时受弹簧、弯曲、角度、接触、分载和 Y/Z 自由度影响，不能直接当每根弹簧的刚度。多组参数拟合相近时，保留相容范围或有效参数，而不是强行给出唯一实物数值。计算模型参数辨识的这类问题可参见 [Tuo 与 Wu 的校准研究](https://arxiv.org/abs/1508.07155)。

**没有任何参考响应时做情景研究或设计反求。** 可以先声明参数区间与目标，扫描规律，或求出满足目标力/行程的设计参数。它回答“什么参数组合能够达到目标”，不等于已经识别现有实物的未知参数。用同一模型生成数据再反推只检查数值自洽；独立有限元结果支持数值等效，均不能被标为实体实测。

### 可以怎样给结论

允许直接给出条件明确的机理解释、设计比较、参数匹配规律和模型预测。例如“在所考察表面族、预载及参数范围内，该布局缩短建立距离”。对影响排序的未知参数做少量有针对性的范围比较：排序稳定则据此报告稳健趋势；排序改变则报告适用条件，而不是选一个估计值掩盖反例。

实际硬件的绝对承载值和寿命仍是另一种结论，不能仅靠计算参数替代独立物理证据。能够开展少量整体实验时，可用来约束关键参数与检验预测，不要求直接测到每根针的每个量。这种以有限实验配合广泛计算研究的方式及其参数/模型不确定性，亦见 [NIST 的计算模型校准说明](https://www.nist.gov/publications/calibration-and-uncertainty-analysis-predictions-computational-models)。本项目不把补齐这些实验设为开始仿真的条件。
