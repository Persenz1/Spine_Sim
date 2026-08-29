# M1 → canonical geometry/single/array 交接

## 唯一消费入口

下游不再直接把 `TrackGeometry` 的 x slope/height 降质成旧单刺输入，而是统一经过：

```python
surface = SurfaceState(
    track=track,
    region=region,
    height_m=height,
    source_valid_mask=valid_mask,
)
path = SpinePath.from_track(track, center_z_m)
candidate, next_cursor = query_next_candidate(
    surface, path, CandidateCursor(), spine_pose
)
```

`candidate.search_cursor is next_cursor`（同一个不可变值对象）。single/array accepted state 保存它；`CONTACT_REJECT` 后继续下一 feature，不经历 detach/rebound，不增加 reengagement，也不清空其他刺历史。

## 几何语义

`envelope_height_m` 是有限球球心刚好接触的高度，不是接触力或针尖表面高度：

[
H_R(X,Y)=max_{u,v}left[h(u,v)+sqrt{R^2-ho^2}ight].
]

候选生成同时使用：

- 完整 `footprint_valid_mask`，包含非 winner 源节点；
- top-2 `support_points_m/support_feature_indices_yx`；
- surface/envelope/contact 三类法向；
- x/y 两向包络梯度；
- near-tie 和 feature switch；
- 测量上下界与 geometry uncertainty；
- forward spherical-cap gate；
- 球冠—锥段—圆柱杆的完整姿态 clearance。

support 切换只在离散事件处发生，禁止节点间线性插值。near-tie 不强压成唯一法向；杆体尺寸或原始二维高度不足时返回明确 unknown/`PARAMETER_UNCLOSED`。

## Identity 与 cache

Track v2 identity 至少绑定 recipe、region、radius、y、resolution、schema/algorithm、tie tolerance、raw height hash、valid-mask hash 和 measurement semantics hash。任何一项变化都产生新 track ID。v1 cache 必须重建，不提供兼容双读。

候选 identity/lineage 继续绑定 track 与 geometry version；case identity 另绑定 project/model/result/solver 版本及规范化输入哈希。

## 数据流

1. 同一 terrain recipe/seed 的所有构型共享同一只读二维区域。
2. 只为需要的 `(radius,y)` 缓存少量 v2 tracks。
3. 每刺拥有独立 `SpinePath/CandidateCursor`；几何驱动只负责候选遭遇。
4. `solve_single_spine` 负责接触、摩擦、悬架、容量和物理事件。
5. `solve_array_equilibrium` 只调用 canonical 单刺，负责共同 6D 背板、活动集、混合控制和同载荷重平衡。
6. 整爪/整机不在本轮范围，也不得直接绕过 canonical array 读取 terrain。

## 验证/适用性

- P40/P100/P240 单样本仍仅为 `partially_validated`；其他砂纸和砖/混凝土为 `provisional`。
- 没有下载或声称拥有新的砂纸扫描；probe/tolerance/bounds 不存在时显式 null。
- 2.5-D 单值高度场已实现；一般 mesh 为 `OUT_OF_SCOPE`。
- CPU 是单刺/阵列权威实现；CUDA 仅保留 terrain 后端。
- 10/5 μm 只有在 catalog 明确同 realization 时才能初筛/复核，不能仅凭相同 seed 推断。
