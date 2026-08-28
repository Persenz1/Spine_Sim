# 工程文档导航

当前代码仍由 M0 公共基础、M1 地形生成和独立 M3-fast 阵列筛选构成；目标是在同一
仓库和同一生产路径内按 2026-08-28 新版机理补齐单刺、阵列、通用整爪和整机层。
运行生成的地形、case、图表和临时测试输出不放在 `docs/`，也不应提交到 Git。

## 当前入口

| 目的 | 入口 |
|---|---|
| 查看当前唯一机理规范 | [`theory/README.md`](theory/README.md) |
| 交给网页端 Pro 做统一工程设计 | [`handoff/2026-08-28_网页端Pro工程设计交接.md`](handoff/2026-08-28_网页端Pro工程设计交接.md) |
| 查看 M0 公共基础 | [`m0/README.md`](m0/README.md) |
| 查看 M1 地形实现 | [`m1/README.md`](m1/README.md) |
| 查看材料地形实现状态 | [`research/terrain/03_material_generation_implementation.md`](research/terrain/03_material_generation_implementation.md) |
| 查阅历史需求与旧仿真证据 | [`archive/README.md`](archive/README.md) |
| 回到项目运行说明 | [`../README.md`](../README.md) |

## 目录职责

- `m0/`、`m1/`：当前基础与地形实现的规格、数据字典、测试报告和未闭合问题。
- `decisions/`：架构决策记录。
- `research/`：外部数据、测量缺口和方案调研。
- `theory/`：当前权威机理原文、验算稿和逐篇证据卡。
- `handoff/`：待工程设计收敛的精简交接材料。
- `archive/`：历史需求与旧仿真证据，仅供追溯。

新版机理主稿及其验算闭环是物理实现的当前规范。旧文档和 M3-fast 代码只能提供可复用
资产与回归证据；发生冲突时必须修改唯一生产实现，不能保留第二套程序。

仓库根目录的 `examples/` 只提交稳定输入；`results/`、`output/`、`reports/` 和地形
缓存只存本地运行产物。
