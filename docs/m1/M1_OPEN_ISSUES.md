# M1 未闭合问题与建议

1. 2026-07-27 的 `defined_geometry` 基线和 2026-07-29 的
   `material-terrain-v2` 材料生成均已在 RTX 4060 Ti/CuPy 14.1.1 上运行 CUDA
   夹具。迁移 GPU、CuPy 主版本或 CUDA runtime 时必须重新保存 provider、device
   和 CPU/GPU 重叠容差结果。实测砂纸 patch quilting 仍是 CPU 阶段，是批量
   生成的主要剩余加速点之一。
2. 杆体检查只是保守的中心线/圆柱低成本净空诊断，不是杆体分布接触。没有
   露出长度和半径时保留 `model_unclosed_rod_collision`，不得翻译成无挂接。
3. 完整球仍是局部针尖代理。M2 必须调用前向球冠门控；锥段、真实过渡圆角和
   损伤不在 M1 首版范围。
4. `defined_geometry` 仍只是项目验证随机场。新材料分支已经区分砂纸、红砖和
   混凝土，但只有 P40/P100/P240 砂纸使用单样本公共高程做了部分验证；红砖、
   混凝土及其他砂纸子型仍是 provisional，不能称为总体实测标定。
5. M3 正式设计要求砂纸、红砖、混凝土各 100 个 seed，共 300 个严格配对条件。
   当前正式 M1 catalog 为 0；旧 45 个 `defined_geometry` 条件已删除。必须先冻结
   family/subtype、seed 映射、10/5 μm 同 realization 和完整哈希，再分批生成。
6. 文档引用的 `04_爪刺仿真器_失败问题总结与重启约束.md` 仍未在仓库提供。
   若后续补入，开始 M2 大规模筛选前需做一次差异审查。
