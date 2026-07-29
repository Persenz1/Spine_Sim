# M3 测试与收敛报告

**执行日期：** 2026-07-29
**模块版本：** `m3.2.0`
**模型等级：**
`project_model_P_common_rigid_backplate_z_dynamic_continuous_total_preload_v3`
**正式 campaign：** 未启动
**正式排名：** 未开放

## 1. 解析验收

命令：

```powershell
.venv\Scripts\python.exe -m spine_sim.array.cli validate-analytic `
  --output results\m3_validation\dynamic_analytic_validation_v3.json
```

结果：**16/16 通过**，耗时 22.5 s。机器可读报告：
`results/m3_validation/dynamic_analytic_validation_v3.json`。

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
| 动力学/能量残差 | 平面夹具最大 \(1.11\times10^{-16}\) N / \(1.60\times10^{-8}\) J |
| 方向身份 | 2×5 与 5×2 的构型 ID 和布局不同 |
| wrench 聚合 | 搬移与聚合误差均为 0 |
| 排名门禁 | 未标定参数强制 `parameter_unclosed` 和 `formal_ranking_eligible=false` |
| 完整设计 | 48 基础硬件、1008 固定布局、336 个 80°→60° 梯度 |
| 轴向设置 | 300/800/2000 N/m 按 SI 处理，刚性为独立模式 |
| 输出级别 | summary/aggregate/full 数组数 0/59/90，摘要来自同一结果 |

## 2. M3 专项回归

命令：

```powershell
.venv\Scripts\python.exe -m pytest tests\test_m3_array.py -q
```

结果：**29 passed**。覆盖：

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

## 3. 全仓回归

命令：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

结果：**87 passed，另有 9 个 subtests 通过**，耗时 132.25 s；0 失败。

在完整性写入的最后一次调整后，另行复跑：

```powershell
.venv\Scripts\python.exe -m pytest tests\test_results_runner.py -q
```

结果：**6 passed**，耗时 2.14 s；覆盖 marker、路径/事件哈希、空事件清理和续跑。

流式 summary 合并另行执行
`pytest tests\test_m3_summary_merge.py -q`：**1 passed**，覆盖原子
Parquet/JSONL 产物和跨分片重复 case ID 拒绝。

## 4. 现有 M1 地形短程 smoke

命令：

```powershell
.venv\Scripts\python.exe -m spine_sim.array.cli smoke-existing-m1 `
  results\m2_formal_terrains\terrain_catalog.json `
  --output results\m3_validation\existing_m1_smoke_v3.json `
  --drag-length-mm 0.1 --seed 32001
```

结果：**2×2、4×4、6×6 均沉降成功并到达 `path_end`**，耗时 117.9 s。

| 规模 | 沉降步 | 反力误差/N | 切向力中位数/N | Neff 法向中位数 | 最大动力学残差/N | 最大能量残差/J |
|---|---:|---:|---:|---:|---:|---:|
| 2×2 | 269 | \(3.86\times10^{-11}\) | 0.22996 | 3.9759 | \(3.33\times10^{-16}\) | \(9.31\times10^{-8}\) |
| 4×4 | 269 | \(9.99\times10^{-16}\) | 0.15635 | 14.0136 | \(5.49\times10^{-6}\) | \(1.16\times10^{-6}\) |
| 6×6 | 269 | \(2.22\times10^{-16}\) | 0.08264 | 9.6566 | \(8.41\times10^{-4}\) | \(1.97\times10^{-4}\) |

该结果只证明当前 M1 catalog 的轨迹接口、共同沉降和短程积分能运行。限制如下：

- 路径只有 0.1 mm，不是需求中的 100 mm；
- 只用了一个 seed 和一个代理硬件；
- catalog 是 45 个 `defined_geometry` 库存，不是 3 family×100 seed；
- 6×6 的粗糙面能量残差仍明显，需进一步收敛研究；
- 所有 case 的 `formal_ranking_eligible=false`。

因此不能比较表中三个规模的正式拉力优劣。

## 5. 未执行内容

- 未运行任何 100 mm 正式扫描；
- 未物化 120 万 case；
- 未运行 300-seed campaign；
- 未将短程 smoke 用作排名；
- 未提交或推送 Git。
