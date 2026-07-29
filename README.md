# Spine Sim — M0/M1 基础与 M2/M3 动态恒载重建

M0 公共基础和 M1 地形几何继续有效。M1 提供解析地形、全局坐标可重建验证随机场、
砂纸/红砖/混凝土材料特定地形、memory-map 二维区域、文件高度图、有限球针尖
包络和一维轨迹缓存。当前尚未生成正式 M1 catalog。

2026-07-28 用户确认旧 M2/M3 的“初始一次预载 + fixed-Z 拖动”边界错误。权威
模型现为：整个路径持续施加恒定外部法向预载、规定 \(+x\) 拖动速度，并对安装座/
共同背板 Z、针体变形、脱离、冲击和再接触做时域动力学求解。当前实现版本为
M2 `m2.2.0`、M3 `m3.3.0`；fixed-Z 类只作为迁移兼容层，其结果一律不能正式排名。
物理基线见
[`docs/m2/M2_DYNAMIC_CONSTANT_PRELOAD_SPEC.md`](docs/m2/M2_DYNAMIC_CONSTANT_PRELOAD_SPEC.md)。

旧 fixed-Z 边界下的所有 M2/M3 筛选与 smoke 只保留为历史诊断。正式参数筛选必须
等待新 M1 catalog、材料/动力学标定和 M2/M3 收敛门全部闭合。

## 环境

- Python 3.11+
- NumPy 1.26+
- 可选 `pyarrow>=15`：生成 `cases.parquet`；未安装时写入显式标记的 `cases.jsonl`
- CUDA 非必需；CPU 始终可用

安装开发版本：

```powershell
python -m pip install -e .
```

开发与测试：

```powershell
python -m pip install -e ".[test]"
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

上述 `examples/` 输入配置随仓库提交，干净克隆后可直接使用。M2 代理 campaign
配置和 M3 设计/收敛计划由对应 `scripts/prepare_*.py` 生成，不作为固定输入提交。

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

材料特定随机地形（SI 制、`float32 [y,x]`、固定 seed 可复现）：

```python
from spine_sim.terrain import generate_terrain

terrain = generate_terrain(
    material="sandpaper",       # sandpaper / red_brick / concrete
    subtype="P100",
    size_x_m=0.050,
    size_y_m=0.050,
    resolution_m=5e-6,
    seed=12345,
    mode="synthetic",
)
```

砂纸支持 P40/P60/P100/P120/P180/P200/P240/P300；其中 P40、P100、P240
可使用已核验的 Hirox 实测补丁合成，P200/P300 明确是 provisional 插值/外推，
不会把其他 grit 改名。红砖和普通粗糙混凝土目前是材料特定的多尺度 provisional
模型，未声称已有总体实测验证。

```powershell
# 查看材料和子类型
spine-terrain list-materials

# 生成便携 NPZ；可选 --library 同时注册到现有 mmap/M3 读取流程
spine-terrain generate-material output/P100_seed12345.npz `
  --material sandpaper --subtype P100 `
  --size-x-mm 50 --size-y-mm 50 --resolution-um 5 `
  --seed 12345 --mode synthetic `
  --library terrain_library

# 三个 seed 的高度、PSD、相关长度、坡度、峰/坑和伪影验证
spine-terrain validate-material reports/terrain_validation/P100 `
  --material sandpaper --subtype P100 `
  --size-x-mm 3 --size-y-mm 0.6 --resolution-um 10 `
  --seed 123 --seed 456 --seed 789
```

砂纸公共数据仍由 `scripts/terrain_data_probe.py` 下载并核验，原始文件位于 Git
忽略的 `data/raw/`。新扫描通过 `scripts/calibrate_terrain_profile.py` 生成可审查
的单样本标定 JSON；脚本不会覆盖原始文件，也不会自动把单样本升级成 validated。
实现范围、数据来源、验证状态与限制见
[`docs/research/terrain/03_material_generation_implementation.md`](docs/research/terrain/03_material_generation_implementation.md)。

M2 验收：

```powershell
spine-m2 validate-analytic
spine-m2 smoke-m1-suite results/m1_gpu_suite/suite_report.json
```

第一条是当前 `m2.2.0` 解析验收。第二条只用于 `defined_geometry` 历史接口 smoke，
必须先用已提交的 `examples/m1_gpu_terrain_suite.json` 重新生成 suite；它不是当前
材料地形或正式 campaign 验收。

M3 验收：

```powershell
spine-m3 validate-analytic
spine-m3 smoke-existing-m1 <current_m1_catalog.json> `
  --drag-length-mm 0.1 --seed <available_seed>
```

解析命令只运行 `m3.3.0` 持续总外载阵列动力学验收。第二条命令只用当前 M1
catalog 的一个 terrain condition 做 2×2/4×4/6×6 短程接口 smoke；两者都不运行
旧 fixed-Z 路径，也不启动正式 campaign。完整设计与分片说明见
[`docs/m3/M3_DESIGN_AND_RUN_PLAN.md`](docs/m3/M3_DESIGN_AND_RUN_PLAN.md)。

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

`LegacyPrescribedPoseConstitutiveCore` 与 `LegacyFixedZExperiment` 只用于旧 M3 迁移和
静态解析夹具，不能生成正式 M2 排名。

M3 的稳定生产入口：

```python
from spine_sim.array import (
    ArrayConfiguration,
    ArrayDynamicExperimentSettings,
    DynamicCommonBackplateArray,
    DynamicCommonBackplateExperiment,
)
```

`LegacyCommonBackplateArray`、`LegacyArrayExperimentSettings` 与
`LegacyFixedZCommonBackplateExperiment` 只保留给迁移夹具，不能生成正式 M3
结果或排名。

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

完整测试（会收集 `unittest` 和 pytest 风格测试）：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
.\.venv\Scripts\python.exe -m pytest -q
```

只安装标准库时可运行 unittest 子集，但它不会收集仓库中的 pytest 函数式测试：

```powershell
python -m unittest discover -s tests -v
```

文档总入口见 [工程文档导航](docs/README.md)。M0 的验收证据和风险覆盖见
[M0 测试报告](docs/m0/M0_TEST_REPORT.md)。M1 的数据字段、缓存规则和下游接口见
[M1 数据字典](docs/m1/M1_DATA_DICTIONARY.md)、
[本地库说明](docs/m1/M1_TERRAIN_LIBRARY.md) 和
[M1→M2 交接](docs/m1/M1_to_M2_handoff.md)；轻量预览命令见
[M1 地形绘图](docs/m1/M1_TERRAIN_PLOTTING.md)。M2 的现行规格与历史留档分别从
[M2 文档入口](docs/m2/README.md) 和
[旧 900-case 报告](docs/m2/archive/M2_M220_900_CASE_ARCHIVE.md) 进入。
M3 的继续工作、字段、验收和下游约束见 [M3 文档入口](docs/m3/README.md)。
