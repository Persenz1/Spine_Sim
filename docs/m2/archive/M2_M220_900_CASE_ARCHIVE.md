# M2.2.0 旧地形 900-case 留档报告

**留档日期：** 2026-07-29

**主 campaign：** `campaign_a0425310a96263ebf26c`

**处置状态：** `legacy_proxy_evidence`

**正式排名资格：** 900/900 `formal_ranking_eligible=false`

## 1. 结论

这 900 个 case 是后续 M3 设计空间曾参考的 M2.2.0 低摩擦代理基线，但它使用旧
M1 地形库，且 M2 的质量、阻尼、动态接触、摩擦和杆体接触仍未实验闭合。它可以
说明 M3 的 48 个基础硬件组合从何而来，不能作为当前 M1/M3 修正后的正式排名、
绝对拉力预测或实验结论。

原始数据已从工程目录迁移到：

`D:\Code\Spine_Sim_Data_Archive_2026-07-29\project_files`

外部归档保留完成的 M2 campaign、配置和分析的原有 `results/`、`output/` 与
`examples/` 相对路径，并附带 `inventory.csv`、关键文件 SHA-256 和处置说明。
旧地形、probe、preflight、worker benchmark、构建/pytest/字节码缓存及临时验证
输出已按用户指示永久删除。

## 2. 试验设计与运行身份

| 项目 | 值 |
|---|---|
| M2 版本 | `m2.2.0` |
| 模型 | `project_model_P_main_plane_dynamic_constant_preload_v2` |
| callable | `spine_sim.contact.case:run_case` |
| 设计 | 60 构型 × 15 配对 seed = 900 cases |
| seed | 32001–32015 |
| 针尖半径 | 50、100 μm |
| 针径 | 0.6、0.8 mm |
| 安装角 | 60°、70°、80° |
| 轴向设置 | 100、300、800、2000 N/m、刚性 |
| 摩擦 | `μs=0.30`、`μk=0.20` |
| 外部预载 | 0.5 N |
| 拖动 | 1 mm/s，100 mm |
| 积分 | Moreau implicit Euler，1 ms |
| 执行后端 | CUDA（CuPy），10 workers |
| 运行时间 | 2026-07-28 16:28–19:08（Asia/Shanghai） |

每个 case 的 `terrain_library_root` 指向旧路径
`results/m2_formal_terrains/terrain_library`。该库已经永久删除，配置保留只为
追溯；当前 M1 材料地形 catalog 不应静默替换进这些旧配置。因此已保存结果可读，
但旧 campaign 不能按原输入重跑。

## 3. 完整性

- campaign 目录：5,407 个文件，815,441,206 bytes（约 0.759 GiB）。
- 900 个 case 目录均包含 `COMPLETE`、`summary.json`、`config.json`、
  `validation.json`、`path.npz` 和 `events.jsonl`。
- `manifest.json`：900 complete，`index_format=jsonl_fallback`。
- `result_set_hash`：
  `caa38cbdc338261d229dac8434a71194fe01d1766f353304c85ee060b5312d5a`。
- 独立分析读取 900 cases、60 构型，每构型 15 个唯一 seed；
  `complete_paired_design=true`、`analysis_allowed=true`、`data_issues=[]`。
- 900/900 `numerical_state=converged`；无执行异常。
- 顶层 `validation.json` 为 `not_run`，所以完整性不等同于 campaign 级正式验证。

关键文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| campaign `manifest.json` | `008275e73266f8659d82fdccab0b7eea6bd972a0d6f27f2fd57ed2fde430b693` |
| campaign `config/normalized.json` | `a52c59c6e4540e776f84ff8d8a5f77f5ff6cbb20a092b823446994b6528e9d50` |
| campaign `cases.jsonl` | `d04bd7afc23ac504c83396234915e4e07bba53ca610ffce245d4895eafe6f203` |
| 分析 JSON | `56628c006c6a965446d841104955e8ae53b186c2a9d1581bcf0e4bd14eb7f018` |
| 分析 CSV | `0d19d89526727d05f748cef9039be63ef972164fd8471e7f9aa265481ea71b8e` |

## 4. 运行结果摘要

| 终止状态 | cases | 比例 |
|---|---:|---:|
| `path_end` | 597 | 66.3% |
| `initial_preload_infeasible` | 292 | 32.4% |
| `structural_boundary` | 11 | 1.2% |

三项结构/净空约束同时通过且到达路径末端的 case 为 240/900；在 597 个
`path_end` 中为 240/597。最常见瓶颈是杆体净空，其次是屈服。

仅作为旧代理模型的描述性结果：

- 0.6 mm 针径总体优于 0.8 mm；
- 70°的 P10 反力较高，但约束通过率低；80°更稳健；
- 100 N/m 过软，800–2000 N/m 是主要有效区；
- 角度与轴向设置存在强交互，不能分别单因素定型。

这些结论受旧地形、代理参数和初始化失败处理影响，不应外推到当前材料地形。

## 5. 实际被后续采用的部分

低/中/高三档摩擦联合分析用于缩小 M3 的基础硬件笛卡尔积：删除 100 N/m，保留
两种针尖、两种针径、三个角度，以及 300/800/2000 N/m/刚性四种轴向设置，共
`2×2×3×4=48` 个基础组合。

这是“保留诊断覆盖范围”的设计输入，不是 48 个组合已经优于其他硬件的正式证据。
M3 必须在当前 M1 catalog、共同背板动力学和新的收敛门下重新比较。

## 6. 同批迁移的其他旧数据

本次最初迁移 55 个顶层项目、39,142 个文件、14,060,605,305 bytes（约
13.095 GiB），其中包括：

- 主对象：M2.2.0 低摩擦 900 cases；
- M2.2.0 中/高摩擦各 900 cases；
- 已失效 fixed-Z 低/中/高摩擦共 2,700 cases；
- 旧 M2 正式地形库（约 10.0 GiB，随后永久删除）；
- preflight、probe、worker benchmark、日志和分析产物；
- 旧 M2 campaign 配置与一组旧基线地形预览图。

随后按用户指示永久删除 26 个旧地形/探针/测试顶层项目和可再生缓存快照，共
6,963 个文件、11,127,261,563 bytes（约 10.363 GiB）。外部归档现保留 29 个
顶层项目、32,470 个文件、2,937,541,838 bytes（约 2.736 GiB）。`inventory.csv`
的 `final_disposition` 字段记录每个最初迁移项目是 `retained` 还是
`deleted_by_user_2026-07-29`。

fixed-Z 数据只能用于格式、吞吐、恢复和旧模型诊断；M2.2.0 三档数据只能用于追溯
当时的代理设计判断。

## 7. 恢复与使用限制

若需复查旧分析，可从外部归档按原相对路径恢复相关 campaign 或直接读取其结果。
由于旧地形已永久删除，旧 campaign 不能原样复跑；不得用当前 M1 数据冒充旧
recipe/region/track。恢复已保存结果也不代表重新获得正式排名资格。

当前开发不得：

- 从旧归档向新 M1 catalog 静默映射 recipe/region/track；
- 把初始化失败作为零拉力混入当前条件性能排名；
- 把旧 M2 单针路径相加成 M3 阵列；
- 把代理趋势描述为实验标定或现实失效概率。
