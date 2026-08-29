# M1 文档

M1 负责地形生成、存储、有限针尖包络和向下游提供稳定的地形身份。当前实现已
加入砂纸、红砖和混凝土的材料特定生成。当前没有已生成的正式 M1 catalog。

- [`M1_TERRAIN_LIBRARY.md`](M1_TERRAIN_LIBRARY.md)：本地库与命令入口。
- [`M1_DATA_DICTIONARY.md`](M1_DATA_DICTIONARY.md)：recipe、region、track 和哈希字段。
- [`M1_10UM_5UM_RULE.md`](M1_10UM_5UM_RULE.md)：同 realization 的 10/5 μm 规则。
- [`M1_to_M2_handoff.md`](M1_to_M2_handoff.md)：当前 M1→canonical geometry/single/array 消费合同。
- [`M1_OPEN_ISSUES.md`](M1_OPEN_ISSUES.md)：剩余问题。
- [`M1_TEST_REPORT.md`](M1_TEST_REPORT.md)、[`M1_BENCHMARK_REPORT.md`](M1_BENCHMARK_REPORT.md)、
  [`M1_GPU_TERRAIN_SUITE_REPORT.md`](M1_GPU_TERRAIN_SUITE_REPORT.md)：2026-07-27
  `defined_geometry` 的历史测试与容量证据；不代表材料生成器当前验收。
- [`../research/terrain/README.md`](../research/terrain/README.md)：材料地形调研和当前
  实现记录。
