# M1 数据字典

M1 模块版本为 `m1.0.0`。除布尔量、索引和无量纲斜率外，所有数值均为
SI。身份由 M0 的规范 JSON 和 SHA-256 规则生成；算法语义变化必须提升版本。

## TerrainRecipe

| 字段 | 类型/单位 | 含义 |
|---|---|---|
| `generator_name` | string | 首版固定为 `defined_geometry` |
| `generator_version` | string | 随机数、核和 10/5 μm 采样规则的联合版本 |
| `seed` | non-negative int | realization 标识，不是顺序 RNG 初态 |
| `global_origin_x_m/y_m` | m | 规范网格全局索引原点 |
| `canonical_dx_m/dy_m` | m | 固定 5 μm |
| `production_dx_m/dy_m` | m | 固定 10 μm，必须等于规范步距的 2 倍 |
| `target_rms_height_m` | m | 由核权重理论标定的总体目标 RMS，不按窗口归一化 |
| `correlation_length_x_m/y_m` | m | 可分离高斯核的固定物理尺度 |
| `kernel_kind` | string | 首版固定 `separable_gaussian` |
| `kernel_truncate_sigma` | dimensionless | 有限核半宽，单位为相关长度 |
| `amplitude_scale` | dimensionless | 固定配方幅值倍数 |
| `coordinate_convention` | string | `global_xy_nodes_origin_aligned` |
| `terrain_recipe_id` | string | 由冻结的 M0 `TerrainRecipeRef` 生成 |
| `recipe_hash` | SHA-256 | 配方、M1 版本和 stride-2 节点采样规则的完整哈希 |

白噪声由 `(seed, global_i, global_j, generator_version)` 直接寻址。滤波核先单位和
归一化，再用核的 L2 范数做总体 RMS 标定；任何局部窗口都不减自己的均值或除
自己的 RMS。

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
