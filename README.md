# Spine Sim

Spine Sim 是一套面向钩爪式爬壁机器人微刺抓附的仿真程序，当前版本为 `0.6.0`。

IJMS 新版入口是 `spine_sim.ijms:run_case`，实现共同背板、轴向压缩弹簧、可回缩空间弯曲杆及连续曲面上的预载—拖动。新研究从 [新版机理与接口](docs/IJMS新版机理与接口.md) 和 [批量仿真交接](docs/IJMS批量仿真交接.md) 开始。`examples/ijms_campaign.json` 是可运行的未标定数值示例。

后续由 DeepSeek / harness 编码时，先读 [AGENTS.md](AGENTS.md) 和 [IJMS 物理约束](docs/IJMS物理约束.md)：可优化实现，保持物理含义；文档末尾提供 harness 任务前缀。

0.6.0已实现50/100 μm圆钝尖、2 mm渐细段与1 mm主体的变截面弯曲、固定安装、梯度角长度补偿和真实杆体检查。小阵列使用 [固定配置](experiments/ijms_small_array.json) 与 [启动说明](docs/IJMS小阵列启动.md)，支持共享材料地形、336000条粗筛队列及恢复。大阵列仍按 [扫描规划](docs/IJMS大规模扫描规划.md) 在小阵列后确定，最多30×30、总P不超过10 N。

```powershell
.venv\Scripts\python.exe -B -m spine_sim.cli run-case examples\ijms_campaign.json --backend cpu --output E:\Agent_Tmp_WS\ijms_demo
```

旧单刺/阵列求解器与解析夹具保留供对照，其物理链为：


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
- SciPy 1.14+：新版有限杆稀疏非线性求解
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

`examples/smoke_campaign.json` 只验证通用 runner 和结果存储。`examples/canonical_campaign.json` 是旧机理的解析平墙夹具。新版完整路径使用 `examples/ijms_campaign.json`；粗糙高度场使用 `examples/ijms_rough_campaign.json`。

新版直接查询连续表面，不依赖旧固定轨道候选。`generate_legacy_full_scan()` 仍只生成历史设计点。数值示例运行成功不代表实物已标定，设计比较需要声明共同预算、表面样本、搜索窗口并检查路径完成状态。

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
