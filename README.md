# Spine Sim — M0 公共基础与 M1 地形几何

当前仓库已完成 M0 公共基础和 M1 地形几何。M1 提供解析地形、全局坐标可重建
随机场、10 μm 本地 memory-map 区域、5 μm 同 realization 复核路径、文件
高度图、有限球针尖包络和 M2/M3 一维轨迹缓存。接触力、摩擦、梁、阵列和
整爪物理仍不在当前实现中。

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
[M1→M2 交接](docs/M1_to_M2_handoff.md)。
