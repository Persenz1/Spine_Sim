# 工程文档导航

`docs/` 只保存可版本控制、可长期引用的工程知识。运行生成的地形、case、图表和
临时测试输出不放在这里，也不应提交到 Git。

## 从哪里开始

| 目的 | 入口 |
|---|---|
| 继续当前 M3 修正 | [`m3/M3_CONTINUATION_HANDOFF.md`](m3/M3_CONTINUATION_HANDOFF.md) |
| 查看 M3 阻断项 | [`m3/M3_OPEN_ISSUES.md`](m3/M3_OPEN_ISSUES.md) |
| 了解当前 M1 材料地形实现 | [`research/terrain/03_material_generation_implementation.md`](research/terrain/03_material_generation_implementation.md) |
| 查看旧 M2 900-case 留档 | [`m2/archive/M2_M220_900_CASE_ARCHIVE.md`](m2/archive/M2_M220_900_CASE_ARCHIVE.md) |
| 回到项目运行说明 | [`../README.md`](../README.md) |

## 目录职责

- `guide/`：用户的原始提示词、工程机理和实现指导。这里回答“为什么做、物理边界
  是什么”。
- `m0/`–`m3/`：各模块的规格、数据字典、测试报告、未闭合问题和上下游交接。
  每个目录都有自己的 `README.md`。
- `decisions/`：架构决策记录（ADR），说明重要技术选择及其理由。
- `research/`：外部数据、测量缺口和方案调研；它提供依据，不等同于已实现规格。
- 模块内的 `archive/`：已失效或被替代、但仍需追溯的历史文档。不得把其中结果
  当作当前排名依据。

## 文档状态约定

- **权威规格**：当前实现必须遵守的物理或接口定义。
- **测试报告**：某个明确版本和范围的验证证据，不自动代表现实标定。
- **未闭合问题**：继续开发前需要处理或显式接受的风险。
- **交接**：下一模块或下一轮工作的恢复入口。
- **留档**：只用于追溯、格式验证或历史比较，不用于当前正式排名。

仓库根目录的 `examples/` 只提交可复现的稳定输入。由脚本生成的 M2 代理配置和
M3 设计/收敛计划保持 Git 忽略；`results/`、`output/`、`reports/` 和地形缓存也
只存本地运行产物。

新增文档时优先放入对应模块；只有跨模块决策进入 `decisions/`，外部资料研究进入
`research/`，原始需求和机理说明进入 `guide/`。
