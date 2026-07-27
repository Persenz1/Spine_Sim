# Spine Sim — M0 公共基础与运行骨架

这是爪刺耦合仿真器重启后的 M0 实现。它只提供单位、坐标、配置、身份、状态、结果写入、后端发现和 case/campaign 运行能力；不包含地形、接触、摩擦、梁、阵列或整爪物理。

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

## CLI

```powershell
spine-sim validate-env
spine-sim run-case examples/smoke_campaign.json --output results
spine-sim run-campaign examples/smoke_campaign.json --output results --workers 2
spine-sim resume examples/smoke_campaign.json --output results
spine-sim retry-failed examples/smoke_campaign.json --output results
spine-sim summarize results/<campaign_id>
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

M0 的验收证据和风险覆盖见 [测试报告](docs/M0_TEST_REPORT.md)，接口字段见 [数据字典](docs/M0_DATA_DICTIONARY.md)，M1 接口见 [交接说明](docs/M0_to_M1_handoff.md)。
