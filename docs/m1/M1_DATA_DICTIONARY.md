# M1 数据字典

M1 模块版本为 `m1.0.0`；材料生成算法版本为 `material-terrain-v2`。除布尔量、
索引和无量纲斜率外，所有数值均为 SI。身份由 M0 的规范 JSON 和 SHA-256 规则
生成；算法语义变化必须提升相应算法版本。

## TerrainRecipe

| 字段 | 类型/单位 | 含义 |
|---|---|---|
| `generator_name` | enum | `defined_geometry` 或 `material_hybrid` |
| `generator_version` | string | 前者为 defined-geometry 版本，后者固定为 `material-terrain-v2` |
| `seed` | non-negative int | realization 标识，不是顺序 RNG 初态 |
| `global_origin_x_m/y_m` | m | 规范网格全局索引原点 |
| `canonical_dx_m/dy_m` | m | defined-geometry 固定 5 μm；材料 recipe 为本次输出步距的一半 |
| `production_dx_m/dy_m` | m | defined-geometry 固定 10 μm；材料 recipe 为本次输出步距，始终等于规范步距的 2 倍 |
| `target_rms_height_m` | m | recipe 身份中的 RMS；defined-geometry 用理论核标定，材料地形由生成结果计算 |
| `correlation_length_x_m/y_m` | m | defined-geometry 的高斯尺度；材料地形 recipe 中记录输出网格尺度 |
| `kernel_kind` | enum | `separable_gaussian` 或 `material_specific`，必须与生成器匹配 |
| `kernel_truncate_sigma` | dimensionless | defined-geometry 的有限核半宽 |
| `amplitude_scale` | dimensionless | defined-geometry 的固定配方幅值倍数 |
| `coordinate_convention` | string | `global_xy_nodes_origin_aligned` |
| `material` | string/null | `material_hybrid` 必填，如 `sandpaper/red_brick/concrete` |
| `subtype` | string/null | `material_hybrid` 必填，且必须属于对应材料 profile |
| `generation_mode` | enum/null | `material_hybrid` 必填：`measured` 或 `synthetic` |
| `profile_hash` | SHA-256/null | `material_hybrid` 必填；绑定完整材料 profile，变更后旧缓存不得静默复用 |
| `terrain_recipe_id` | string | 由冻结的 M0 `TerrainRecipeRef` 生成 |
| `recipe_hash` | SHA-256 | 配方、M1 版本和该生成分支 production-sampling 语义的完整哈希 |

`defined_geometry` 的白噪声由
`(seed, global_i, global_j, generator_version)` 直接寻址。滤波核先单位和
归一化，再用核的 L2 范数做总体 RMS 标定；任何局部窗口都不减自己的均值或除
自己的 RMS。该分支只用于解析、接口、缓存和 GPU 基线，不代表真实材料。

`material_hybrid` 由统一材料 API 生成。砂纸可选择实测 crop 或材料特定合成，
红砖和混凝土当前使用材料特定合成；profile、数据来源和验证等级见
[`../research/terrain/03_material_generation_implementation.md`](../research/terrain/03_material_generation_implementation.md)。
`generate_terrain()`、`generate-material` 和
`TerrainLibrary.generate_region()` 的材料分支均支持 `cpu/cuda`；CUDA metadata
必须记录实际 provider、device、runtime 和显存峰值。

## Terrain

`generate_terrain()` 返回尚未注册或已可注册的有限二维高度场：

| 字段 | 类型/单位 | 含义 |
|---|---|---|
| `height` | `float32[ny,nx]`, m | 二维单值高度场，数组顺序 `[y,x]` |
| `dx/dy` | m | x/y 节点间距 |
| `valid_mask` | `bool[ny,nx]` | 实测数据有效域；合成输出全部为 true |
| `material/subtype` | string | 材料和严格校验的子类型 |
| `seed` | non-negative int | crop 或合成 realization 身份 |
| `metadata.resolved_mode` | enum | 实际采用的 `measured` 或 `synthetic` |
| `metadata.profile_hash` | SHA-256 | 生成时使用的材料 profile |

`Terrain.to_recipe()` 产生 `material_hybrid` recipe；`register_terrain()` 将高度、
mask、元数据和完整哈希写入原 `TerrainLibrary`。`auto` 只是 API 请求模式，写入
recipe 的始终是已经解析后的 `measured` 或 `synthetic`。

## RegionSpec

| 字段 | 类型/单位 | 含义 |
|---|---|---|
| `terrain_recipe_id` | string | 上游配方 |
| `origin_x_m/y_m` | m | 左下节点全局坐标 |
| `size_x_m/y_m` | m | 首末节点之间的物理长度 |
| `resolution_x_m/y_m` | m | 等方网格步距，5 或 10 μm |
| `purpose` | enum | `module/debug/campaign/user` |
| `shape` | `(ny,nx)` | `size/spacing + 1`，边界节点均包含 |
| `region_id` | string | 由冻结的 M0 `TerrainRegionSpec` 生成 |

区域文件不保存 X/Y meshgrid。坐标由原点、步距和 shape 恢复。10 μm 区域原点
必须落在 5 μm 规范网格的偶数索引。

## TrackGeometry

| 字段 | 类型/单位 | 含义 |
|---|---|---|
| `terrain_recipe_id` | string | 地形 realization |
| `region_id` | string | 原始二维区域 |
| `track_id` | string | recipe/region/radius/y/算法版本/分辨率联合 ID |
| `radius_m` | m | 有限球代理半径 |
| `y_global_m` | m | 固定横向轨道，必须落在网格行 |
| `resolution_m` | m | x 采样间距 |
| `envelope_algorithm_version` | string | `finite-sphere-envelope-v1` |
| `x_global_m` | float64[N], m | 全局 x 节点 |
| `envelope_height_m` | float64[N], m | 球心刚好接触时的竖直高度 \(H_R\) |
| `envelope_slope_x` | float64[N] | 中心差分 \(\partial_x H_R\) |
| `support_x_m/y_m` | float64[N], m | 主最大值支撑节点 |
| `valid_mask` | bool[N] | 完整球邻域和导数均在源域内 |
| `near_tie_flag` | bool[N] | 最优和次优支撑的高度差小于声明容差 |
| `model_warning` | string tuple | 完整球代理需要球冠门控、杆体参数未闭合等 |

轨迹缓存的 NPZ 只保存一维数组，JSON 侧车保存身份、算法、生成耗时、警告和
SHA-256；`.complete` 最后写入。

## FileHeightMapSource

记录原始文件绝对路径、原始单位、原点、网格、shape 和源 SHA-256。
`sample_file_heightmap` 另行返回裁剪、去倾斜和双线性插值记录。处理结果不会被
登记成原始源，也不附加随机配方的 RMS/相关长度解释；超域抛出
`geometry_out_of_domain`。
