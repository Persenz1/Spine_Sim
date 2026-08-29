# Spine Sim — canonical 微刺与阵列仿真器

当前生产版本为 `0.3.0`，只有一条物理链：

```text
TerrainLibrary / TrackGeometry v2
        → ContactCandidate / path cursor
        → single_spine_quasistatic
        → array_rigid_backplate_event
        → canonical summary / events / trace.parquet
```

已实现：

- machine-readable 参数 registry、逐字段 SI/degree→rad 迁移、旧 protocol 与 12 个终筛 ID provenance；
- 高度场与材料随机场、完整球尖 footprint、top-2 support、三类法向、测量上下界与姿态相关杆体 clearance；
- 八态单刺、三维摩擦锥、二维解析特例、低阶梁/悬架、单边弹簧、硬限位、条件容量、最早事件和 trial/commit；
- 任意逐刺位置/间隙的刚性背板 6D 混合控制、loader stiffness、活动集/级联、式 (9-29) 力矩感知重平衡、尺度化秩/值域与约束自由子空间稳定性；
- 原子 case/campaign、paired seeds、版本化 identity、Parquet 索引和无损 canonical trace。

旧 `m3_fast` 物理核、独立入口、全局 reseat、旧 `Neff`、monitor 和只验证旧语义的测试已删除。历史输入与结果仍由 registry 和 `docs/archive/` 追溯，不作为新版响应或排名真值。

本轮明确不实现整爪/整机、真实动态回弹、损伤演化、背板连续弯曲或 CUDA 单刺/阵列核。动态稳定性固定为 `OUT_OF_SCOPE`；缺少真实材料、表面或拓扑参数时返回 `PARAMETER_UNCLOSED/OUT_OF_SCOPE`，不补经验默认值。

规范入口：

- [当前机理主稿](docs/theory/README.md)
- [实施指导](docs/engineering/单刺与阵列统一仿真实施指导.md)
- [参数继承表](docs/engineering/工程仿真参数继承表.md)
- [本轮统一实现报告](docs/engineering/2026-08-30_单刺与阵列统一实现报告.md)
- [结果数据字典](docs/m0/M0_DATA_DICTIONARY.md)
- [地形/几何数据字典](docs/m1/M1_DATA_DICTIONARY.md)

## 环境

- Python 3.11+
- NumPy 1.26+
- 可选 `pyarrow>=15`：Parquet case index 与 trace
- 可选 `matplotlib>=3.9,<4`：地形绘图
- 可选 `cupy-cuda13x[ctk]>=14.1,<15`：仅地形生成 CUDA 后端

本机默认 Conda 环境安装：

```powershell
conda run -n codex-py312 python -m pip install -e ".[test,plot,parquet]"
```

## 命令

统一仿真入口：

```powershell
spine-sim validate-env
spine-sim run-case <campaign.json> --output results
spine-sim run-campaign <campaign.json> --output results --workers 2
spine-sim resume <campaign.json> --output results
spine-sim retry-failed <campaign.json> --output results
spine-sim summarize results/<campaign_id>
```

`CampaignSpec.callable` 指向 canonical case adapter；测试中的 `spine_sim.examples.canonical_module:run_case` 是明确标记的解析平墙 smoke catalog，不是标定硬件模型。

独立地形入口继续保留：

```powershell
spine-terrain region-report --recipe examples/m1_defined_recipe.json
spine-terrain generate-region terrain_library examples/m1_defined_recipe.json examples/m1_debug_region.json
spine-terrain generate-track terrain_library <recipe_id> <region_id> --radius-um 50 --y-mm 0
spine-terrain delete-cache terrain_library <recipe_id> <region_id>
spine-terrain rebuild-region terrain_library <recipe_id> <region_id>
spine-terrain benchmark --output results/m1_validation/benchmark.json
```

## 公共 Python API

```python
from spine_sim import (
    CandidateCursor,
    SpineAcceptedState,
    query_next_candidate,
    solve_single_spine,
    solve_array_equilibrium,
)
```

`query_next_candidate` 返回候选及同一个不可变 continuation cursor；拒绝候选只推进该刺，不重置其他刺。阵列求解器只调用 `solve_single_spine`，不复制第二套局部本构。

## 结果

每个 case 保存版本/输入哈希、单位/坐标/作用对象、参数 provenance、四维状态、假设/省略项、残差/容差、秩/值域/稳定性、逐刺几何/wrench/margins/events。`trace.parquet` 对嵌套列采用带 schema metadata 的 canonical JSON 列编码，使用 `spine_sim.io.read_trace_table` 可无损回读；没有 `pyarrow` 时明确降级为 JSONL。

## 测试

```powershell
conda run --no-capture-output -n codex-py312 python -m pytest -q
```

文档导航见 [docs/README.md](docs/README.md)。本地 `results/`、`output/`、`reports/` 与地形缓存不提交到 Git。
