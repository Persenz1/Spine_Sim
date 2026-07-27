# M1 测试与基准报告

**日期：** 2026-07-27

**模块版本：** `m1.0.0`

**命令：**

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s tests -v
```

## 结果

- 总测试：37；
- 通过：37；
- 跳过：0；
- 失败/错误：0；
- 其中 M1 测试 21 项，M0 回归 16 项；
- CUDA 测试使用 RTX 4060 Ti、CuPy 14.1.1，并验证 GPU 生成文件仍由 CPU
  memory map 读取。

## M1 验收映射

| 验收项 | 证据 | 结果 |
|---|---|---|
| 平面完整球包络 | 常数面得到 `offset + R`、斜率 0 | 通过 |
| 斜坡解析偏移 | 离散支撑偏移和解析最大值最近节点一致，斜率一致 | 通过 |
| 单凸起半径方向 | 100 μm 的上游首次阈值接触不晚于 50 μm | 通过 |
| 同 recipe/seed/region 重复 | `np.array_equal` | 通过 |
| 小窗口=大区域裁剪 | `np.array_equal` | 通过 |
| 相邻 tile 重叠 | 50 μm 重叠逐位相等 | 通过 |
| 删除后重建 | 不同 tile 行数的 `.npy` SHA-256 相同 | 通过 |
| 10/5 μm | 10 μm 等于独立 5 μm 重建 `[::2,::2]` | 通过 |
| CPU/GPU | 小窗口逐点相等；CUDA 本地库与 CPU 容差一致 | 通过 |
| memory map | 返回 `np.memmap`、`float32`、只读 | 通过 |
| 多进程只读 | Windows spawn 两进程校验和相同 | 通过 |
| 轨迹=直接二维 | 高度、斜率、支撑和有效域一致 | 通过 |
| 网格加密 | 正弦夹具的高度、斜率和阈值事件位置收敛 | 通过 |
| 最大区域覆盖 | 对齐边界包住全部设计空间和声明余量 | 通过 |
| 中断文件 | 缺 `COMPLETE` 的 `.npy` 被拒绝 | 通过 |
| 文件高度图 | 单位、源 hash、双线性插值、超域错误 | 通过 |
| 球冠/杆体接口 | 前向点积门控和未闭合杆体警告 | 通过 |

## 性能和容量

小型 debug 区域为 2 × 1 mm、10 μm、101 × 201：

| 项目 | 结果 |
|---|---:|
| 实际 `.npy` 文件 | 81,332 B |
| CPU 生成 | 0.0351 s |
| memory-map 顺序读取 | 0.000100 s |
| 读取吞吐 | 812 MB/s |
| 50 μm 轨迹（201 点） | 0.0196 s |
| 删除后重建 | 0.0357 s |
| 重建 SHA-256 | 相同 |

最大区域宽度的单个 5 μm 工作 tile（127 × 29,593）实测：

| 项目 | 结果 |
|---|---:|
| tile 生成时间 | 2.065 s |
| tile 输出数组 | 30,066,488 B |
| 进程峰值工作集 | 258,334,720 B（约 246.4 MiB） |

该峰值包括解释器和临时数组，远小于整张 5 μm 场约 0.95 GB 的单数组；生产
生成不会同时保留整张规范场。详细最大区域和磁盘量见
`M1_BENCHMARK_REPORT.md`。

## 复现实物

机器可读小基准和 smoke 地形库位于
`results/m1_validation/benchmark.json` 与
`results/m1_validation/terrain_library/`（`results/` 已 gitignore）。

十工况 GPU 数据和机器可读报告位于 `results/m1_gpu_suite/`；提交版摘要见
`M1_GPU_TERRAIN_SUITE_REPORT.md`。
