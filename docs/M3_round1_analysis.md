# M3 第一轮分析（legacy fixed-Z，已失效）

> 2026-07-28 用户将权威边界修订为持续总外部预载和共同背板动力学。下述 fixed-Z
> 实现与 smoke 记录不得支持阵列排名；M3 必须按 `M2_to_M3_handoff.md` 重建。

**状态：未运行；m3.1.0 动态阶段 I 实现和解析验收已完成，正式轮次受门禁阻止。**

## 当前可以下的结论

- 持续恒定总外载下，共同背板 Z 与全部针内部模态统一时域积分已实现；
- 平面平均反力、不同高度载荷转移、脱离/再接触和多针同时冲击夹具通过；
- 正序、逆序和固定随机遍历产生相同全局 proposal、接纳状态和单元 wrench；
- 拒绝步保持旧 `ArrayDynamicState`，普通脱离不会终止阵列；
- 三类 \(N_\mathrm{eff}\)、最大/均值和 Gini 权重已明确；
- M3→M4 同时刻动态字段及 JSON Schema 已冻结为 `m3.1.0`；
- 内部时间步减半、动力学/能量残余和 wrench 聚合解析验收通过；
- 未运行任何旧 fixed-Z 筛选，也未对 45 个库存地形启动正式 M3 campaign。

## 正式分析尚不能回答

- 安装角是否在阵列层稳定优于其他角度；
- 80→60° 或 80→50° 梯度是否有实际收益；
- 2×5 与 5×2 在配对 seed 上的方向差异；
- 阵列规模收益何时饱和；
- 柔顺是否在正式地形上提高有效针数而不只是改变峰值；
- 哪些 18–24 个候选应进入三预载细筛。

## 门禁

- `full_chain_frozen_manifest.json` 不存在；
- M2 正式第一轮未运行；
- 用户批准的 M2 参数包不存在；
- 尚无“开始 M3 第一轮筛选”的显式批准；
- 项目随机地形杆体/锥段净空仍为 `parameter_unclosed`。

`examples/m3_round1_design_draft.json` 只验证 100/126 的确定性平衡覆盖算法。
门禁打开后必须用 4–6 个用户批准的 M2 代表包重建设计矩阵，不能直接执行 fixture
矩阵。

正式 case 数据生成后，用：

```powershell
python scripts/analyze_m3_round1.py <campaign_dir> --output-dir output/m3_round1
```

先完成数据完整性审计，再做配对 seed、Pareto、bootstrap 和 50/100 mm 排名一致性
分析。脚本在没有 case 数据时会停止，不会生成虚构图表。
