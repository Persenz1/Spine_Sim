# M1 → M2/M3 交接

## 冻结入口

M2/M3 的动力学接触默认消费 `TrackGeometry`：

```python
library = TerrainLibrary(root)
track = library.cache_track(
    recipe,
    region,
    radius_m=50e-6,
    y_global_m=y_i,
)
```

缓存键已经包含 terrain recipe、region、半径、全局 y、包络算法版本和分辨率。
`terrain_recipe_id` 来自 M0 `TerrainRecipeRef`，`region_id` 来自 M0
`TerrainRegionSpec`；track ID 应进入 M2 case 的 `upstream_hash`/lineage。

## 几何语义

`envelope_height_m` 是完整球代理的球心竖直高度：

\[
H_R(X,Y)=\max[h(u,v)+\sqrt{R^2-\rho^2}].
\]

它不是接触力，也不是针尖表面高度。M2 用安装座运动学得到球心后，将其与
`H_R` 比较，并同时使用：

- `envelope_slope_x` 构造切向/法向；
- `support_x_m/y_m` 形成实际支撑点；
- `valid_mask` 阻止域边缘查询；
- `near_tie_flag` 标记支撑切换敏感点；
- `forward_cap_gate` 排除完整球后半部；
- 可选 `check_rod_clearance`，参数不足则保留模型未闭合。

M1 不提供力、摩擦、弹簧、梁、状态机或“挂接成功率”。

当前 recipe 有两个合法分支：`defined_geometry` 只用于解析/接口基线；
`material_hybrid` 绑定材料、子类型、resolved generation mode 和 profile hash。
M2/M3 不得按材料名称猜测 recipe，也不得把旧 `defined_geometry` ID 映射成当前
材料地形。

## M2/M3 数据流

1. 按 seed 分组，打开一个 10 μm 最大区域。
2. 对该 seed 和硬件候选需要的 `(radius,y_i)` 生成/读取少量轨迹。
3. M2 规定安装座本构在轨迹上查询 gap、斜率和支撑；脱离后继续路径并允许
   再接触。
4. M3 的所有针共享同一 region 和背板位姿，只是各自选择对应 y 轨迹；不能
   为构型或针重新生成随机表面。
5. M4 不访问地形，只读取 M3 同 seed、同路径位置、同状态的单元力/矩样本。

## 是否必须访问二维数组

否。只要运动限制在固定 y 的首版主平面内，M2 可以完全只读
`TrackGeometry`；M3 通常也只需各针的一维轨迹。只有新轨迹首次构建时 M1
短暂 memory-map 原始二维高度的针尖邻域行。横向运动、二维调试或新的杆体
几何才需要显式二维查询。

## 开始 M2 前的门禁

- 2026-07-27 的 `defined_geometry` CPU/CUDA 基线和 2026-07-29 的材料生成测试均
  有历史通过记录，但原始地形与运行产物已清理；
- 更换 GPU、CuPy 主版本或 CUDA runtime 后需重跑 CPU/GPU 一致性；
- 当前正式 M1 catalog 为 0；M3 所需三类各 100 seed、完整哈希和 10/5 μm
  同 realization 契约尚未关闭；
- M2 必须实现球冠门控并保留杆体未闭合警告；
- 不得把 `near_tie_flag` 或 `model_warning` 自动翻译成物理失败；
- 只有 catalog 明确支持同 realization 时，才允许用 10 μm 初筛和 5 μm
  复核；不得仅复用 seed 猜测两种分辨率相同。
