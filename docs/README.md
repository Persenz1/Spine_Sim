# 工程文档导航

当前代码已完成 canonical `terrain → geometry → single_spine → array` 迁移。运行生成的地形、case、图表和临时输出不放在 `docs/`，也不提交到 Git。

## 当前入口

| 目的 | 入口 |
|---|---|
| 查看唯一机理规范 | [theory/README.md](theory/README.md) |
| 查看单刺/阵列实施合同与门禁 | [engineering/单刺与阵列统一仿真实施指导.md](engineering/单刺与阵列统一仿真实施指导.md) |
| 查看原工程参数与 SI 迁移裁决 | [engineering/工程仿真参数继承表.md](engineering/工程仿真参数继承表.md) |
| 查看本轮 canonical 统一实现报告 | [engineering/2026-08-30_单刺与阵列统一实现报告.md](engineering/2026-08-30_单刺与阵列统一实现报告.md) |
| 查看 canonical 结果 schema | [m0/M0_DATA_DICTIONARY.md](m0/M0_DATA_DICTIONARY.md) |
| 查看 terrain/track/geometry schema | [m1/M1_DATA_DICTIONARY.md](m1/M1_DATA_DICTIONARY.md) |
| 查看 M1→geometry 消费合同 | [m1/M1_to_M2_handoff.md](m1/M1_to_M2_handoff.md) |
| 查看材料地形验证状态 | [research/terrain/03_material_generation_implementation.md](research/terrain/03_material_generation_implementation.md) |
| 查阅历史需求与旧仿真证据 | [archive/README.md](archive/README.md) |
| 查看最新收尾交接 | [handoff/2026-08-30_单刺与阵列统一实现_收尾交接.md](handoff/2026-08-30_单刺与阵列统一实现_收尾交接.md) |
| 回到运行说明 | [../README.md](../README.md) |

## 目录职责

- `m0/`：配置、identity、运行、结果与 canonical schema。
- `m1/`：地形、有限针尖 track、缓存和 geometry 交接。
- `decisions/`：架构决策记录。
- `research/`：外部数据、测量缺口和材料生成验证。
- `theory/`：当前权威机理原文、验算稿和逐篇证据卡。
- `engineering/`：已裁决、可直接执行的工程合同。
- `handoff/`：任务交接记录；不覆盖主稿。
- `archive/`：历史需求与旧仿真证据，仅供追溯。

新版主稿与验算闭环定义物理语义。参数 registry 保存旧构型、protocol 和 ID 的 provenance，但旧响应、排名、`m3_fast` 状态或全局 reseat 不再是可运行生产路径。
