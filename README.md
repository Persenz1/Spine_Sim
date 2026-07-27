# Spine Sim — M0–M3 地形、单刺与共同背板阵列

当前仓库已完成 M0 公共基础、M1 地形几何、M2 阶段 I 和 M3 阶段 I。M1 提供解析地形、全局坐标可重建
随机场、10 μm 本地 memory-map 区域、5 μm 同 realization 复核路径、文件
高度图、有限球针尖包络和 M2/M3 一维轨迹缓存。M2 提供规定安装座位姿的
单刺本构、单边弹簧、二维梁、STICK/SLIDE 历史、初始预载 + 固定-Z 拖动、
FREE/脱离/再接触、事件细化、能量审计和标准结果字段。M3 提供共同刚性背板
\((u_x,u_Z)\)、一次总预载、固定共同 \(u_Z\) 拖动、全针 proposal/原子提交、
局部事件处全阵列重评估、wrench 搬移、活动集、三类 \(N_\mathrm{eff}\)、载荷集中度、
M3→M4 同状态样本和确定性平衡覆盖设计器。整爪 M4 尚未实现。

正式 M2/M3 参数筛选尚未启动：它受 M0–M4 全链冻结 manifest、前序轮次和用户批准
门禁约束。现有十种 M1 地形只做过 M2 与 M3 smoke，不用于参数排名。

## 环境

- Python 3.11+
- NumPy 1.26+
- 可选 `pyarrow>=15`：生成 `cases.parquet`；未安装时写入显式标记的 `cases.jsonl`
- CUDA 非必需；CPU 始终可用

安装开发版本：

```powershell
python -m pip install -e .
```

若正式 campaign 要求 Parquet：

```powershell
python -m pip install -e ".[parquet]"
```

CUDA 13 GPU 环境：

```powershell
python -m pip install -e ".[test,gpu-cuda13]"
```

## CLI

```powershell
spine-sim validate-env
spine-sim run-case examples/smoke_campaign.json --output results
spine-sim run-campaign examples/smoke_campaign.json --output results --workers 2
spine-sim resume examples/smoke_campaign.json --output results
spine-sim retry-failed examples/smoke_campaign.json --output results
spine-sim summarize results/<campaign_id>
```

M1 地形命令：

```powershell
spine-terrain region-report --recipe examples/m1_defined_recipe.json
spine-terrain generate-region terrain_library examples/m1_defined_recipe.json examples/m1_debug_region.json
spine-terrain generate-track terrain_library <recipe_id> <region_id> --radius-um 50 --y-mm 0
spine-terrain delete-cache terrain_library <recipe_id> <region_id>
spine-terrain rebuild-region terrain_library <recipe_id> <region_id>
spine-terrain benchmark --output results/m1_validation/benchmark.json
spine-terrain generate-suite results/m1_gpu_suite/terrain_library examples/m1_gpu_terrain_suite.json
spine-terrain plot-region results/m1_gpu_suite/terrain_library <recipe_id> <region_id> output --overview-size-mm 10 --sphere-radius-um 100
```

M2 验收：

```powershell
spine-m2 validate-analytic
spine-m2 smoke-m1-suite results/m1_gpu_suite/suite_report.json
```

M3 验收：

```powershell
spine-m3 validate-analytic
spine-m3 smoke-m1-suite results/m1_gpu_suite/suite_report.json
```

M2 的稳定 Python 入口：

```python
from spine_sim.contact import (
    PrescribedPoseConstitutiveCore,
    SingleSpineState,
    SpineParameters,
)

core = PrescribedPoseConstitutiveCore(parameters, track)
trial = core.solve_pose((x_h, z_h), SingleSpineState(), commit=False)
```

M3 必须从同一旧阵列状态对所有针调用 `commit=False`，然后在阵列层原子采纳
proposal；不能复用独立单刺包装器的完整路径结果。

M3 的稳定 Python 入口：

```python
from spine_sim.array import (
    ArrayConfiguration,
    ArrayExperimentSettings,
    CommonBackplateArray,
    CommonBackplateExperiment,
)
```

`run-case` 默认运行配置中的第一个 case，也可用 `--case-id` 精确选择。`resume` 跳过带有合法 `COMPLETE` 标记的 case；`retry-failed` 只选择已有 `execution_error` 摘要的 case。每个 case 独立执行，单个异常不会中断其他 case。

Python API 的稳定入口包括：

```python
from spine_sim.core.config import CampaignSpec
from spine_sim.runtime.backend import discover_backend
from spine_sim.runtime.runner import CampaignRunner

campaign = CampaignSpec.from_mapping(raw_config)
runner = CampaignRunner(campaign, campaign_dir, discover_backend())
runner.initialize(raw_config)
records = runner.run(resume=True)
```

## 测试

无需 pytest：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s tests -v
```

M0 的验收证据和风险覆盖见 [M0 测试报告](docs/M0_TEST_REPORT.md)。M1 的数据
字段、缓存规则和下游接口见 [M1 数据字典](docs/M1_DATA_DICTIONARY.md)、
[本地库说明](docs/M1_TERRAIN_LIBRARY.md) 和
[M1→M2 交接](docs/M1_to_M2_handoff.md)；轻量预览命令见
[M1 地形绘图](docs/M1_TERRAIN_PLOTTING.md)。M2 的分支定义、字段和验收证据见
[求解规格](docs/M2_solver_spec.md)、[数据字典](docs/M2_DATA_DICTIONARY.md)、
[测试报告](docs/M2_TEST_REPORT.md) 和 [M2→M3 交接](docs/M2_to_M3_handoff.md)。
M3 的字段、验收和下游约束见 [M3 数据字典](docs/M3_DATA_DICTIONARY.md)、
[M3 测试报告](docs/M3_TEST_REPORT.md) 和 [M3→M4 交接](docs/M3_to_M4_handoff.md)。
