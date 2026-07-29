# M1 本地地形库与命令

## 目录和有效性

```text
terrain_library/
├── recipes/<terrain_recipe_id>.json
├── sources/<source_id>.json
├── regions/<terrain_recipe_id>/<region_id>/
│   ├── raw_height.npy
│   ├── valid_mask.npy          # material_hybrid
│   ├── metadata.json
│   └── COMPLETE
├── tracks/<terrain_recipe_id>/<region_id>/<radius>/<track_id>.{npz,json,complete}
├── manifests/regions/<terrain_recipe_id>/<region_id>.json
└── validation/
```

`raw_height.npy` 使用 `float32`，只保存高度；材料地形另存 bool 有效 mask。
写入先在同目录临时文件完成并 `fsync`，然后原子替换；元数据完成后最后写
`COMPLETE`。缺少完成标记的中断文件不会被读取。recipe 和 manifest 是重建依据，
region 和 track 是可删除缓存；实测分支还必须保留其来源文件和哈希。

区域运行时入口：

```python
library = TerrainLibrary("terrain_library")
height = library.open_region(recipe_id, region_id)
assert isinstance(height, np.memmap)
```

Windows 删除区域前必须释放所有 memory map；否则工具会明确拒绝删除。

## CLI

```powershell
spine-terrain region-report --recipe examples/m1_defined_recipe.json

spine-terrain generate-region terrain_library `
  examples/m1_defined_recipe.json examples/m1_debug_region.json

spine-terrain generate-track terrain_library <recipe_id> <region_id> `
  --radius-um 50 --y-mm 0

spine-terrain delete-cache terrain_library <recipe_id> <region_id>
spine-terrain rebuild-region terrain_library <recipe_id> <region_id>
spine-terrain benchmark --output results/m1_validation/benchmark.json

spine-terrain generate-suite `
  results/m1_gpu_suite/terrain_library `
  examples/m1_gpu_terrain_suite.json `
  --output results/m1_gpu_suite/suite_report.json

spine-terrain list-materials

spine-terrain generate-material output/P100_seed12345.npz `
  --material sandpaper --subtype P100 `
  --size-x-mm 50 --size-y-mm 50 --resolution-um 5 `
  --seed 12345 --mode synthetic `
  --library terrain_library

spine-terrain validate-material reports/terrain_validation/P100 `
  --material sandpaper --subtype P100 `
  --size-x-mm 3 --size-y-mm 0.6 --resolution-um 10 `
  --seed 123 --seed 456 --seed 789
```

删除命令默认同时删除该区域的轨迹，但保留 recipe 和 region manifest，因此可
恢复。`--keep-tracks` 只用于诊断；一般不应让轨迹引用一个已删除区域。

## 下游运行策略

以下是正式 campaign 的计划运行策略；当前正式 M1 catalog 为 0。运行时按 terrain
condition/seed 分组，一次只打开一个 seed 的 10 μm 最大区域；
同一 seed 的全部构型共享 OS 只读页缓存。M2/M3 先请求所需 y 和半径的轨迹，
缓存存在就只读 NPZ，不存在才从 memory map 的少数邻近行计算。完整二维包络、
二维梯度和二维支撑数组只用于夹具/调试，不是生产文件。
