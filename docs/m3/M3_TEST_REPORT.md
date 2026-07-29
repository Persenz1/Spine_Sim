# M3 测试与收敛报告

> 本报告同时包含当前代码的专项回归、新 M1 增强地形验证和清理前的历史数值
> 快照。历史旧 M1 地形及其机器可读产物已经删除；本轮生成的新证据位于
> `results/m3_validation/`，仍不具备正式排名资格。

**执行日期：** 2026-07-29
**模块版本：** `m3.4.0`
**模型等级：**
`project_model_P_common_rigid_backplate_z_dynamic_continuous_total_preload_v4`
**正式 campaign：** 未启动
**正式排名：** 未开放

## 1. 解析验收

命令：

```powershell
.venv\Scripts\python.exe -m spine_sim.array.cli validate-analytic `
  --output results\m3_validation\dynamic_analytic_validation_v4.json
```

当前结果：**16/16 通过**，耗时 24.8 s。机器可读报告为
`results/m3_validation/dynamic_analytic_validation_m3_4_0.json`。

| 门禁 | 关键结果 |
|---|---|
| 平面总反力平衡 | 0.5 N 外载下稳态平均反力 0.4999954 N |
| 平滑预载和沉降门 | minimum-jerk 斜坡；末端反力误差 \(6.47\times10^{-10}\) N；连续稳定 20 步 |
| 2×2 对称载荷 | 四针各约 0.125 N，极差 \(8.95\times10^{-15}\) N |
| 高度差载荷转移 | 两针 0.35143/0.14857 N，总和 0.5000000 N |
| 单针脱离与再接触 | DETACH、RECONTACT、IMPACT 均出现，阵列到 `path_end` |
| 同时多针冲击 | 10 个保存点包含至少两针同时 IMPACT |
| 遍历顺序不变 | 正序、逆序、固定随机顺序的 proposal/point/state 完全相同 |
| 拒绝步骤原子性 | 地形越界 proposal 无效，拒绝后状态精确等于旧状态 |
| 时间步减半 | 1/0.5 ms 平均反力差 0.000236 N，切向中位数差 \(3.68\times10^{-7}\) N |
| 动力学/能量残差 | 最大 \(1.11\times10^{-16}\) N / \(1.03\times10^{-20}\) J；累计相对能量误差 \(5.36\times10^{-16}\) |
| 方向身份 | 2×5 与 5×2 的构型 ID 和布局不同 |
| wrench 聚合 | 搬移与聚合误差均为 0 |
| 排名门禁 | 未标定参数强制 `parameter_unclosed` 和 `formal_ranking_eligible=false` |
| 完整设计 | 48 基础硬件、1008 固定布局、336 个 80°→60° 梯度 |
| 轴向设置 | 300/800/2000 N/m 按 SI 处理，刚性为独立模式 |
| 输出级别 | summary/aggregate/full 数组数 0/76/107，摘要来自同一结果 |

## 2. M3 专项回归

命令：

```powershell
.venv\Scripts\python.exe -m pytest tests\test_m3_array.py -q
```

结果：**36 passed**，耗时 114.47 s。覆盖：

- 总反力平衡、平滑斜坡、沉降速度/反力/残差/连续稳定门；
- 对称 2×2、2×5/5×2 方向身份、针遍历顺序和拒绝原子性；
- 300/800/2000 N/m 与刚性、弹簧 lower/interior/hard-stop；
- 固定 60°/70°/80° 和 80°→60° 梯度坐标变换；
- stick/slide、单针脱离后其他针继续积分及再接触；
- 力、力矩、能量、应力、屈服、屈曲和弹簧限位统计；
- 三档输出摘要不变、初始化失败不作为零承载排名；
- 相同 seed/构型确定性重复、内部时间步减半；
- 48/1008/336/1344 设计数量、正式 300 catalog 和严格配对拒绝；
- 2×2/4×4/6×6 平面解析规模 smoke。
- 工程代理参数身份、随阵列外廓缩放的背板质量/刚度/阻尼；
- 8 个哨兵、11 个单轴变体和条件性能趋势收敛门；
- 圆柱杆下表面在 M1 二维高度场上的轴向/横向净空查询。
- 弹簧针轴向 lower/interior/hard-stop 分支之间的名义模态阻尼连续性；
- M1 catalog schema、数据哈希和跨材质重号 seed 的显式拒绝；
- 确定性换落点顺序、名义落点优先和碰撞 case 的排名隔离。

## 3. 全仓回归

命令：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

当前结果：**107 passed，另有 15 个 subtests 通过**，耗时 138.56 s；0 失败。
测试期间出现 16 条来自 CuPy padding 与 NumPy 2.5 的弃用警告，不影响结果。

在完整性写入的最后一次调整后，另行复跑：

```powershell
.venv\Scripts\python.exe -m pytest `
  tests\test_results_runner.py tests\test_m3_summary_merge.py -q
```

结果完整性与流式 summary 合并合跑：**7 passed**，耗时 2.73 s；覆盖 marker、
路径/事件哈希、空事件清理、续跑、原子 Parquet/JSONL 产物和跨分片重复 case ID
拒绝。

另用真实 M1 条件执行 1 个 2×2 runner case：

- nominal 落点从路径 0 即检测到杆体碰撞，最小净空 \(-638.61\) μm；
- 第 2 个候选 `y=+1 mm` 从沉降起点重放，到达 `path_end`，净空
  \(+91.45\) μm；
- case 完整写入并标记 `complete`，resume 用时约 0.8 s 且未重复计算；
- summary 合并验证 1 个 case 的 COMPLETE marker、payload hash 和结果集哈希；
  当前环境无 Parquet 引擎，按既定协议输出 JSONL fallback。

## 4. 新 M1 十五地形短程验证

输入为
`results/m1_material_m3_test/terrain_catalog.json`：砂纸、红砖、混凝土各 5 条，
共 15 条全尺寸 10 μm 材料增强地形，约 3.57 GB。先前一轮已对 15 个数据文件
逐一重算完整 SHA-256，catalog、recipe、region、material/subtype、路径和数据哈希
身份均一致。

修复轴向阻尼分支不连续后，用生产工程代理基线（位置修正 0.20）对每个条件执行
2×2/4×4/6×6、1 N、0.1 mm 短程 smoke：

- 15/15 条件通过，45/45 用例完成沉降并到达 `path_end`；
- 45/45 数值流程通过，拒绝步为 0；最大动力学残差
  \(9.653\times10^{-4}\) N，低于 1 mN 门；
- 最大单步能量残差 \(3.575\times10^{-8}\) J，最大累计相对能量误差
  \(1.774\times10^{-5}\)，最大接触功恒等式残差
  \(5.204\times10^{-18}\) J；
- 总耗时 940.74 s，单用例 2.24--113.41 s；
- nominal 落点下 15/45 同时通过屈服、屈曲和杆体净空；30 条被物理约束排除：
  24 条仅杆体碰撞、5 条杆体碰撞且屈服代理超限、1 条仅屈服代理超限。所有屈曲
  检查均通过。约束失败不计作数值流程失败，也不能进入排名。

碰撞规避另以 P60 做了中途碰撞重放验证：

- 2×2 nominal 首次碰撞在 27 μm；依序测试候选后，`x=-1 mm` 从起点重放，
  最小净空由 \(-605.53\) μm 改善为 \(+142.30\) μm并完成；
- 4×4 nominal 首次碰撞在 18 μm；`y=+1 mm` 重放后净空
  \(+53.06\) μm并完成；
- 6×6 nominal 已有 \(+1.36\) μm 净空，无需换落点，但屈服代理仍不通过，故
  保持排名隔离。

机器可读结果为
`results/m3_validation/m1_material_15_after_fix.json` 和
`results/m3_validation/m1_material_P60_placement_search.json`。这些结果证明 15 条
增强地形的读取、沉降、短程拖拽、碰撞检测和落点重放流程可运行；不证明 100 mm
收敛，也不替代 300 条正式 catalog。

## 5. 历史旧 M1 地形短程 smoke

命令：

```powershell
.venv\Scripts\python.exe -m spine_sim.array.cli smoke-existing-m1 `
  results\m2_formal_terrains\terrain_catalog.json `
  --output results\m3_validation\existing_m1_smoke_v5_energy_injection.json `
  --drag-length-mm 0.1 --seed 32001
```

这是 2026-07-29 当时的原始命令。其旧 M2 地形输入现已永久删除；当前复跑必须
显式传入新的 M1 材料地形 catalog，不能依赖该旧路径。

结果：**2×2、4×4、6×6 均沉降成功并到达 `path_end`**，耗时 119.7 s。

| 规模 | 沉降步 | 反力误差/N | 切向力中位数/N | Neff 法向中位数 | 最大动力学残差/N | 最大能量残差/J | 累计位置修正注能/J |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2×2 | 269 | \(3.86\times10^{-11}\) | 0.22996 | 3.9759 | \(3.33\times10^{-16}\) | \(1.77\times10^{-20}\) | \(4.93\times10^{-6}\) |
| 4×4 | 269 | \(9.99\times10^{-16}\) | 0.15635 | 14.0136 | \(5.49\times10^{-6}\) | \(6.62\times10^{-11}\) | \(3.47\times10^{-6}\) |
| 6×6 | 269 | \(2.22\times10^{-16}\) | 0.08264 | 9.6566 | \(8.41\times10^{-4}\) | \(5.82\times10^{-8}\) | \(5.44\times10^{-3}\) |

该结果只证明当时旧 M1 catalog 的轨迹接口、共同沉降和短程积分能运行。限制如下：

- 路径只有 0.1 mm，不是需求中的 100 mm；
- 只用了一个 seed 和一个代理硬件；
- catalog 是已经删除的 45 个 `defined_geometry` 库存，不是 3 family×100 seed；
- 离散能量账本已经闭合；6×6 的接触位置修正注能、动力学残差和趋势指标仍需
  时间步/接触参数成对收敛；
- 所有 case 的 `formal_ranking_eligible=false`。

因此不能比较表中三个规模的正式拉力优劣。

## 6. 未执行内容

- 未运行任何 100 mm 正式扫描；
- 未物化 120 万 case；
- 未运行 300-seed campaign；
- 未将短程 smoke 用作排名；
- 未运行有界 264 case/地形条件的收敛矩阵。
