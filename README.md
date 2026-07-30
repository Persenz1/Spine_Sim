# Spine Sim — M0/M1 基础与地形生成

当前仓库只保留：

- **M0 公共基础**：配置、单位与坐标系、稳定身份、运行后端、结果存储和批量运行骨架。
- **M1 地形生成**：解析地形、全局坐标可重建随机场、材料特定地形、实测数据导入、
  本地地形库、有限球针尖包络、轨迹缓存与可视化。

M2 单刺和 M3 阵列的实现、命令行入口、运行脚本、测试及专属文档已移除，等待重新设计。
保留的跨模块工程背景只用于后续重构追溯，不代表当前实现。
本地 `results/`、`output/`、`reports/` 与地形缓存未被删除。

## 环境

- Python 3.11+
- NumPy 1.26+
- 可选 `pyarrow>=15`：结果索引使用 Parquet
- 可选 `matplotlib>=3.9,<4`：地形绘图
- 可选 `cupy-cuda13x[ctk]>=14.1,<15`：CUDA 地形生成

安装开发版本：

```powershell
python -m pip install -e ".[test,plot]"
```

CUDA 13 环境：

```powershell
python -m pip install -e ".[test,plot,gpu-cuda13]"
```

## M0 命令

```powershell
spine-sim validate-env
spine-sim run-case examples/smoke_campaign.json --output results
spine-sim run-campaign examples/smoke_campaign.json --output results --workers 2
spine-sim resume examples/smoke_campaign.json --output results
spine-sim retry-failed examples/smoke_campaign.json --output results
spine-sim summarize results/<campaign_id>
```

`examples/smoke_campaign.json` 只验证 M0 运行骨架，不执行接触或阵列物理。

## M1 地形命令

```powershell
spine-terrain region-report --recipe examples/m1_defined_recipe.json
spine-terrain generate-region terrain_library examples/m1_defined_recipe.json examples/m1_debug_region.json
spine-terrain generate-track terrain_library <recipe_id> <region_id> --radius-um 50 --y-mm 0
spine-terrain delete-cache terrain_library <recipe_id> <region_id>
spine-terrain rebuild-region terrain_library <recipe_id> <region_id>
spine-terrain benchmark --output results/m1_validation/benchmark.json
spine-terrain generate-suite results/m1_gpu_suite/terrain_library examples/m1_gpu_terrain_suite.json
spine-terrain plot-region results/m1_gpu_suite/terrain_library <recipe_id> <region_id> output
```

材料特定随机地形：

```python
from spine_sim.terrain import generate_terrain

terrain = generate_terrain(
    material="sandpaper",
    subtype="P100",
    size_x_m=0.050,
    size_y_m=0.050,
    resolution_m=5e-6,
    seed=12345,
    mode="synthetic",
)
```

支持的材料族为 `sandpaper`、`red_brick` 和 `concrete`。砂纸公共数据探测、
材料标定、正式目录生成和可视化脚本保留在 `scripts/`：

```text
calibrate_terrain_profile.py
generate_m1_material_fine_refinements.py
generate_m1_material_formal_catalog.py
render_material_terrain_gallery.py
terrain_data_probe.py
```

## 测试

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
.\.venv\Scripts\python.exe -m pytest -q
```

文档入口见 [`docs/README.md`](docs/README.md)，M1 地形库规则见
[`docs/m1/M1_TERRAIN_LIBRARY.md`](docs/m1/M1_TERRAIN_LIBRARY.md)，材料地形实现状态见
[`docs/research/terrain/03_material_generation_implementation.md`](docs/research/terrain/03_material_generation_implementation.md)。
