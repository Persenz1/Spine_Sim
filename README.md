# Spine Sim

Spine Sim 是一套面向钩爪式爬壁机器人微刺抓附的仿真程序，当前版本为 `0.4.0`。

当前物理链：

```text
TerrainLibrary / TrackGeometry
        → ContactCandidate / CandidateCursor
        → single_spine_quasistatic
        → array_rigid_backplate_event
        → CaseOutput / ResultStore
```

已实现：

- 解析随机场、材料地形、实测高度场导入和本地地形库；
- 有限球尖包络、top-2 support、三类法向、测量不确定性和杆体 clearance；
- 八态单刺准静态求解、三维库仑摩擦、梁/悬架、单边弹簧、硬限位和事件定位；
- 刚性共同背板阵列、六自由度混合控制、活动集、事件级联、重平衡和准静态稳定性；
- versioned identity、campaign runner、恢复运行、Parquet/JSONL trace 和原子结果存储。

当前不实现整爪/整机、真实动态回弹、损伤演化、柔性背板连续弯曲或一般三维倒扣表面。完整范围见 [docs/README.md](docs/README.md)。

## 文档

- [文档总览](docs/README.md)
- [公共约定](docs/公共约定.md)
- [地形模块](docs/地形模块.md)
- [几何模块](docs/几何模块.md)
- [单刺模块](docs/单刺模块.md)
- [阵列模块](docs/阵列模块.md)
- [运行与使用](docs/运行与使用.md)
- [原始机理全文](docs/钩爪式爬壁机器人抓附机理与多尺度力学模型.md)

## 环境

- Python 3.11+
- NumPy 1.26+
- 可选 `pyarrow>=15`：Parquet case index 与 trace
- 可选 `matplotlib>=3.9,<4`：地形绘图
- 可选 `cupy-cuda13x[ctk]>=14.1,<15`：CUDA 地形生成

在已安装本项目的 Python 环境中运行：

```powershell
python -m pip install -e ".[test,plot,parquet]"
```

## 常用命令

```powershell
spine-sim validate-env --output results
spine-sim run-case examples/smoke_campaign.json --output results
spine-sim run-case examples/canonical_campaign.json --output results --backend cpu
spine-sim run-campaign campaign.json --output results --workers 2 --backend cpu
spine-sim resume campaign.json --output results
spine-sim retry-failed campaign.json --output results
spine-sim summarize results/<campaign_id>
```

运行命令支持 `--backend auto|cpu|cuda` 和 `--device-index`。CUDA 当前固定要求
`--workers 1`；worker 会在执行 case callable 前绑定所选 CuPy device。

`examples/smoke_campaign.json` 只验证通用 runner 和结果存储，不运行物理链。`examples/canonical_campaign.json` 会运行解析平墙候选、单刺和阵列承载，但不查询真实地形。项目仍需要通过 production case callable 连接地形、几何、单刺和阵列；`spine_sim.examples.canonical_module` 是解析 smoke fixture，不是已标定硬件模型。

> **生产运行边界：** 当前仓库不包含把 `TerrainLibrary` 的逐站候选连续接入单刺/阵列求解器的 production full-scan adapter 或可直接运行的完整扫描 campaign。`generate_legacy_full_scan()` 只生成历史设计点，不能单独启动物理仿真。在补齐并验证该 adapter、装配尺寸和初始间隙前，不应把 smoke fixture 的成功解释为完整仿真已经可运行。

地形入口示例：

```powershell
spine-terrain region-report --recipe examples/m1_defined_recipe.json
spine-terrain generate-region terrain_library examples/m1_defined_recipe.json examples/m1_debug_region.json
spine-terrain generate-track terrain_library <recipe_id> <region_id> --radius-um 50 --y-mm 0
spine-terrain list-materials
```

更多命令和 Python API 见 [运行与使用](docs/运行与使用.md)。

## 测试

```powershell
python -m pytest -q
```

本地 `results/`、`output/`、`reports/` 和地形缓存不提交到 Git。
