# Spine Sim — M0/M1 基础与 M2/M3 动态恒载重建

M0 公共基础和 M1 地形几何继续有效。M1 提供解析地形、全局坐标可重建随机场、
10 μm 本地 memory-map 区域、5 μm 同 realization 复核路径、文件高度图、有限球
针尖包络和一维轨迹缓存。

2026-07-28 用户确认旧 M2/M3 的“初始一次预载 + fixed-Z 拖动”边界错误。权威
模型现为：整个路径持续施加恒定外部法向预载、规定 \(+x\) 拖动速度，并对安装座/
共同背板 Z、针体变形、脱离、冲击和再接触做时域动力学求解。旧 M2/M3 代码暂作为
迁移兼容层，其结果一律不能正式排名。新基线见
`docs/M2_DYNAMIC_CONSTANT_PRELOAD_SPEC.md`。

旧 fixed-Z 边界下的所有 M2/M3 筛选与 smoke 只保留为历史诊断，正式参数筛选必须
等待动态恒载 M2/M3 重新验收。

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
    DynamicContactSettings,
    DynamicExperimentSettings,
    DynamicIntegratorSettings,
    DynamicSingleSpineExperiment,
)

result = DynamicSingleSpineExperiment(
    parameters,
    track,
    DynamicExperimentSettings(...),
    DynamicContactSettings(...),
    DynamicIntegratorSettings(...),
).run()
```

`PrescribedPoseConstitutiveCore` 与 `LegacyFixedZExperiment` 只用于旧 M3 迁移和
静态解析夹具，不能生成正式 M2 排名。

M3 当前入口仍是 legacy fixed-Z 迁移接口，必须按
`docs/M2_to_M3_handoff.md` 重建后才能重新成为生产入口：

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
[动态测试报告](docs/M2_DYNAMIC_TEST_REPORT.md)、[legacy 测试报告](docs/M2_TEST_REPORT.md)
和 [M2→M3 交接](docs/M2_to_M3_handoff.md)。
M3 的字段、验收和下游约束见 [M3 数据字典](docs/M3_DATA_DICTIONARY.md)、
[M3 测试报告](docs/M3_TEST_REPORT.md) 和 [M3→M4 交接](docs/M3_to_M4_handoff.md)。
