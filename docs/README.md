# 工程文档导航

当前代码范围为 M0 公共基础、M1 地形生成和独立 M3-fast 阵列筛选。运行生成的
地形、case、图表和临时测试输出不放在 `docs/`，也不应提交到 Git。

## 当前入口

| 目的 | 入口 |
|---|---|
| 查看 M0 公共基础 | [`m0/README.md`](m0/README.md) |
| 查看 M1 地形实现 | [`m1/README.md`](m1/README.md) |
| 查看材料地形实现状态 | [`research/terrain/03_material_generation_implementation.md`](research/terrain/03_material_generation_implementation.md) |
| 回到项目运行说明 | [`../README.md`](../README.md) |

## 目录职责

- `m0/`、`m1/`：当前基础与地形实现的规格、数据字典、测试报告和未闭合问题。
- `decisions/`：架构决策记录。
- `research/`：外部数据、测量缺口和方案调研。
- `guide/`：保留的 M0/M1 原始需求与跨模块工程背景。

M3-fast 以用户提供的《M3 独立阵列啮合仿真快速重构执行指导》为当前权威需求；
`guide/` 中旧的 M2/M3/M4 链路描述仅供背景追溯。

仓库根目录的 `examples/` 只提交稳定输入；`results/`、`output/`、`reports/` 和地形
缓存只存本地运行产物。
