# Spine Sim — 新版机理统一仿真程序（迁移中）

当前可运行代码仍包括：

- **M0 公共基础**：配置、单位与坐标系、稳定身份、运行后端、结果存储和批量运行骨架。
- **M1 地形生成**：解析地形、全局坐标可重建随机场、材料特定地形、实测数据导入、
  本地地形库、有限球针尖包络、轨迹缓存与可视化。
- **M3-fast 阵列筛选**：直接读取 M1 静态一维轨迹，以向量化闭式单针核和共同背板
  标量平衡完成固定步长路径、Parquet 摘要、M3-A 参数包筛选和 M3-B 几何筛选。

目标是在本仓库内按 2026-08-28 新版机理补齐单刺、阵列、任意内拉形态整爪和整机层。
程序只保留一条生产路径；旧实现与新版机理冲突时直接替换，不建立 v2 包、旧版回退或
长期兼容实现。迁移完成前，下面的 M3-fast 命令仍保持现有降阶语义，不能视为新版完整模型。
本地 `results/`、`output/`、`reports/` 与地形缓存未被删除。

当前机理规范和网页端 Pro 工程设计交接分别见
[`docs/theory/README.md`](docs/theory/README.md) 与
[`docs/handoff/2026-08-28_网页端Pro工程设计交接.md`](docs/handoff/2026-08-28_网页端Pro工程设计交接.md)。

## 环境

- Python 3.11+
- NumPy 1.26+
- 可选 `pyarrow>=15`：结果索引使用 Parquet
- 可选 `matplotlib>=3.9,<4`：地形绘图
- 可选 `cupy-cuda13x[ctk]>=14.1,<15`：CUDA 地形生成

安装开发版本：

```powershell
python -m pip install -e ".[test,plot,parquet]"
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

## M3-fast 命令

默认从 `results/m1_material_formal_300/` 中读取已生成的 M1 轨迹；不重新生成地形。

```powershell
spine-m3-fast smoke
spine-m3-fast m3a
spine-m3-fast m3b
spine-m3-fast all
spine-m3-fast full-auto --workers 6
```

默认粗筛使用 P240 的 6 个配对 seed。可用 `--material`、`--subtype` 和
`--seeds S1 S2 S3 S4 S5 S6` 指定同一地形配方的其他正式条件。结果写入
`results/m3_fast/` 下的 Parquet、manifest 和候选选择 JSON。上游目录中声明的
材料标定与 10/5 μm 收敛限制会原样保留在 M3-A manifest；当前排序只用于相对筛选，
不代表绝对承载标定。

`full-auto` 依次执行粗筛、细筛和终筛。粗筛遍历 1,344 个机械构型、3 档恒定
推力和 45 个分层地形，只保存 case 摘要；细筛保留约 96 个构型并在 150 个
地形上保存全局及逐针路径；终筛保留约 24 个构型，在全部 300 个地形上拖拽
20 mm 并保存完整路径，最后给出 6 个构型。可分别使用 `full-coarse`、
`full-fine`、`full-final` 断点续跑。运行期间可启动只读进度弹窗：

路径求解保持法向预载持续作用。接触崩开时会清除失效接触历史，在同一拖拽位置
重新压合；若当前几何分支不可达，则按确定顺序尝试当前落点附近
`0, +dx, -dx, ...` 的候选，默认最多到 `±5dx`。成功后继续完成规定拖拽距离，
并在摘要和完整路径中记录重新压合次数、落点偏移、求解尝试次数和未支撑站点，
不会再把一次脱离直接当作 case 终点。

每个落点求解尝试仍最多执行 9 次阵列评估；重新压合或更换候选落点会在同一空间
站产生额外尝试，因此同时记录 `max_station_evaluations`（单次）和
`max_station_total_evaluations`（该站合计）。纯数值未收敛不会触发接触历史清零。

```powershell
.\.venv\Scripts\pythonw.exe scripts\monitor_m3_full_scan.py `
  --output results\m3_fast\full_scan
```

## 测试

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
.\.venv\Scripts\python.exe -m pytest -q
```

文档入口见 [`docs/README.md`](docs/README.md)，M1 地形库规则见
[`docs/m1/M1_TERRAIN_LIBRARY.md`](docs/m1/M1_TERRAIN_LIBRARY.md)，材料地形实现状态见
[`docs/research/terrain/03_material_generation_implementation.md`](docs/research/terrain/03_material_generation_implementation.md)。
