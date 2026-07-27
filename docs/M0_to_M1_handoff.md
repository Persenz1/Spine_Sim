# M0 → M1 交接

## M1 可直接使用的冻结接口

以下接口在 M0 Schema `1` 内冻结：

- `spine_sim.core.units.to_si` 与 `require_range`
- `FrameMetadata`, `Wrench.rotate`, `Wrench.move_reference`
- `stable_hash`, `identity`, `terrain_recipe_id`, `region_id`, `track_id`, `lineage_hash`
- `ProjectConfig`, `BackendConfig`, `TerrainRecipeRef`, `TerrainRegionSpec`
- `StateBundle`, `Event`, `EventType` 和结构化错误类别
- `ResultStore` 的原子 JSON/NPZ、完成标记和只读摘要
- `discover_backend`
- `RunContext`, `CaseOutput`, `CampaignRunner`

字段增加允许向后兼容；已有字段的含义、单位或正号不能静默改变。算法变化必须提升 `module_version`，从而产生新身份。

## M1 的配置与 ID 责任

M1 应：

1. 用 `TerrainRecipeRef` 保存配方名称、版本、seed 和规范参数；
2. 用其 `terrain_recipe_id` 标识可重建最大地形；
3. 将区域长度和分辨率归一化为米后建立 `TerrainRegionSpec`；
4. 用 `region_id` 标识空间窗口；
5. 用 `track_id` 标识一维查询轨迹；
6. 将 terrain/region/track 记录放入下游 case 的 `upstream_hash` 或 lineage；
7. 算法变化提升 M1 `module_version`，不能复用旧 ID。

## 文件写入

M1 大型二维地形可在自己的模块目录使用 memory map；M0 不要求把地形塞入 `path.npz`。单 case 必须保留：

- SI `config.json`
- 指标与状态 `summary.json`
- 轨迹数组 `path.npz` 或稳定 Parquet
- `events.jsonl`
- `validation.json`
- 最后写入的 `COMPLETE`

M1 不得写入旧结果根目录，也不得修改完整 case。需要重算时使用新 module version/ID 或新的 campaign 目录。

## 后端

`discover_backend()` 的结果含 CPU、CUDA 可用性、provider、选中后端和设备号。CPU 始终可用；CUDA 请求但不可用会产生配置错误。M1 可用 CUDA 生成 tile，但必须把后端记录进 manifest，并保持 CPU 查询与可复现重建路径。

runner 调用模块函数时传入 `(parameters, RunContext)`；`RunContext.backend` 是父进程已发现并冻结的能力记录，M1 不应在 worker 内静默改选后端。

## M1 开始前的阻塞项

没有代码接口阻塞。正式项目治理仍有一个文档缺口：M0 提示词引用的 `04_爪刺仿真器_失败问题总结与重启约束.md` 未在仓库中提供。当前已根据权威执行版里重复声明的规则隔离旧失败语义；若后续补入 04，应先做一次差异审查。
