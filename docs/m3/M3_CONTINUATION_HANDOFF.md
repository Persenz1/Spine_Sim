# M3 共同背板阵列续作交接

**交接日期：** 2026-07-29

**仓库：** `D:\Code\Spine_Sim`

**分支：** `main`

**M3 版本：** `m3.3.0`

**正式 campaign：** 未启动

**正式排名：** `formal_ranking_eligible=false`

## 1. 用户最终需求

用完整硬件和阵列构型，在 0.5、1、2 N 单元总预载下：

1. 将整个单元通过一个持续外部总预载压向壁面；
2. 用平滑预载斜坡完成共同背板和全部针的动态沉降；
3. 沉降后保持同一总预载，规定背板沿 +x 拖动 100 mm；
4. 比较不同针尖、针径、安装角、轴向设置、阵列形状、方向、间距和角度布局之间的
   拉力趋势；
5. 初始化失败单独统计，不能以零拉力进入条件性能排名。

当前只开放共同背板 Z 动力学；+x 为规定运动，俯仰/横滚锁定。

## 2. 此前已完成并推送的 M3 实现提交

- `fca6344 fix(m3): harden common-backplate dynamics and campaign design`
- `ddafd91 feat(m3): add auditable proxy and convergence controls`

两个提交均已推送至 `origin/main`。它们包含：

- 共同背板与全部针的统一动力学积分；
- 原子 proposal/commit、拒绝不污染和遍历顺序不变；
- minimum-jerk 总预载斜坡及完整沉降门；
- 初始化失败分类和失败 case 排名隔离；
- 48 种基础硬件、1008 个固定布局、336 个 80°→60° 梯度布局；
- 0.5/1/2 N、100 mm 加载协议身份；
- summary/aggregate/full 三档输出；
- 原子结果、`COMPLETE`、结果/路径/事件哈希及断点续跑；
- 流式 Parquet summary 合并和重复 case 拒绝；
- backward-Euler 离散能量、接触功和位置修正注能账本；
- `engineering_proxy_v1` 经验基线与敏感性情景；
- 有界数值/速率收敛计划；
- M1 二维高度场上的圆柱杆下表面净空后检查。

## 3. 完整设计数量

基础硬件固定角笛卡尔积：

\[
2\ \text{针尖半径}
\times 2\ \text{针径}
\times 3\ \text{固定角}
\times 4\ \text{轴向设置}
=48.
\]

- 固定角：\(48\times7\) 阵列形状 \(\times3\) 间距 = 1008；
- 80°→60° 梯度：\(16\times7\times3=336\)；
- 总构型：1344；
- 正式计划：\(1344\times300\) 地形条件 \(\times3\) 预载
  = 1,209,600 cases。

不要加入固定 50° 或 80°→50° 梯度。不要启动上述正式扫描，除非用户在新窗口再次
明确授权且所有前置门已关闭。

## 4. 等待 M1 完成后必须核对的接口

M3 不生成砂纸、红砖或混凝土地形，只读消费 M1 catalog 和轨迹。M1 完成后首先检查：

1. catalog 是否提供稳定的 family、seed、recipe、region、realization 和数据哈希；
2. 是否有砂纸、红砖、混凝土各 100 seed，共 300 个严格配对条件；
3. 每个 condition 的完整数据哈希是否验证通过；
4. 同一 realization 是否能产生 10 μm 初筛和 5 μm 细筛；
5. 任意 M3 构型引用同一 condition 时，是否得到完全相同的上游地形身份；
6. 不同针的轨迹是否来自同一全局二维地形及各自全局 y，而不是独立随机生成；
7. 新 M1 API 是否仍能提供 M3 杆体净空所需的二维高度场、region 原点和 x/y 分辨率。

如果 M1 改变字段名或 catalog schema，应在 M3 adapter 中做显式版本兼容，不能静默
猜测或由 M3 重新合成地形。

## 5. 当前代码与历史验证基线

当前 `m3.3.0` 专项回归在清理后复核为 `tests/test_m3_array.py` 32 passed。
以下解析、全仓和粗糙地形数值是清理前保存的历史验证快照；对应 `results/`
机器可读产物和旧 45 个 M1 `defined_geometry` 条件现已删除：

- 解析验收：16/16；
- 结果存储/续跑：6 passed；
- summary 合并：1 passed；
- 全仓：98 passed，另有 12 subtests；
- 旧 M1 地形 0.1 mm smoke：2×2、4×4、6×6 均沉降成功并到达 `path_end`。

解析平面夹具：

- 最大动力学残差约 \(1.11\times10^{-16}\) N；
- 最大单步离散能量残差约 \(1.03\times10^{-20}\) J；
- 累计相对能量误差约 \(5.36\times10^{-16}\)；
- 接触功恒等式残差约 \(1.11\times10^{-19}\) J。

历史旧粗糙地形 6×6 短程 smoke：

- 最大动力学残差约 \(8.41\times10^{-4}\) N；
- 最大离散能量残差约 \(5.82\times10^{-8}\) J；
- 累计相对能量误差约 \(2.98\times10^{-5}\)；
- 累计接触位置修正注能约 \(5.44\times10^{-3}\) J。

因此旧的“大能量不守恒”主要是遗漏隐式积分耗散造成的账本错误。上述数值只用于
定位 M3 问题；接入新 M1 catalog 后必须重新生成当前证据，关闭粗糙面大阵列的
接触稳定化注能、动力学残差和趋势收敛。

## 6. 工程代理参数的解释

缺少实验标定时，代码使用 `engineering_proxy_v1`：

- 钢针：\(E=200\) GPa、\(\nu=0.29\)、\(\rho=7850\) kg/m³；
- 屈服强度：800 MPa；
- \(\mu_s/\mu_k=0.30/0.20\)；
- 针模态阻尼比：0.05；
- 恢复系数：0；
- 接触位置修正：0.20；
- 背板运动质量：50 g 等效小车加随阵列外廓变化的 2 mm 铝板；
- 背板竖直阻尼比：0.10；
- 沉降阻尼倍数：10。

这些值可以支持代理趋势和敏感性分析，但不能替代用户无法提供的物理标定。最终报告
必须区分：

- “代理模型下趋势稳健”；
- “绝对拉力经过实验标定”。

当前只能使用前一种表述。

## 7. 新窗口建议执行顺序

### 阶段 A：接入修订后的 M1

1. 先查看 Git 状态，识别并保留用户的 M1 修改；
2. 阅读 M1 新 catalog/schema/API 和测试；
3. 只修改必要的 M3 adapter；
4. 用一个地形条件复跑 2×2/4×4/6×6 的 0.1 mm smoke；
5. 检查地形身份、二维净空、反力、动力学残差、能量账本和位置修正注能。

### 阶段 B：有界收敛，不做正式扫描

生成计划：

```powershell
.venv\Scripts\python.exe scripts\prepare_m3_convergence_plan.py `
  --output examples\m3_convergence_plan.json
```

计划含 8 个哨兵、11 个单轴变体和 3 个预载，每个地形条件 264 cases，建议先用
2 mm 路径。依次关闭：

1. 时间步 5/2/1/0.5 ms；
2. 投影迭代 20/40；
3. 位置修正 1.0/0.5/0.2；
4. 沉降阻尼倍数 5/10/20；
5. 拖速 1/2/5 mm/s；
6. 代表性 10 μm/5 μm 同 realization；
7. 2/10/25/50/100 mm 路径长度外推。

比较时至少要求：

- 拉力中位数差不超过 3%；
- P10 差不超过 5%；
- Neff 和最大针载荷差不超过 5%；
- 接触比例差不超过 2 个百分点；
- 累计相对能量误差不超过 \(10^{-3}\)；
- 接触功恒等式残差不超过 \(10^{-12}\) J；
- 初始化和终止状态一致。

### 阶段 C：小规模趋势试验

收敛门通过后，先选少量配对 seed 和少量代表构型做 100 mm smoke，估计：

- 单 case 时间和峰值内存；
- 不同阵列规模的吞吐；
- 失败恢复率；
- summary/aggregate/full 的实际体积；
- 接触注能是否随路径累计失控。

仍不要直接运行 300-seed 正式 campaign。

### 阶段 D：正式运行前审查

只有以下条件全部满足后，才向用户申请是否启动正式分片：

- 300 条件 catalog 完整并严格配对；
- 代理敏感性不改变主要趋势；
- 时间步、接触参数、沉降阻尼和 100 mm 路径收敛；
- 入围构型的 10/5 μm 同 realization 收敛；
- 4×4/6×6 接触稳定化注能和动力学残差可接受；
- 本地 scratch、Parquet 合并、恢复和归档压测通过。

## 8. 推荐恢复命令

```powershell
git status --short
git log -3 --oneline --decorate

.venv\Scripts\python.exe -m spine_sim.array.cli validate-analytic `
  --output results\m3_validation\dynamic_analytic_validation_next.json

.venv\Scripts\python.exe -m pytest tests\test_m3_array.py -q
.venv\Scripts\python.exe -m pytest tests\test_results_runner.py -q
.venv\Scripts\python.exe -m pytest tests\test_m3_summary_merge.py -q
```

接入新 M1 后再运行：

```powershell
.venv\Scripts\python.exe -m spine_sim.array.cli smoke-existing-m1 `
  <new_m1_catalog.json> `
  --output results\m3_validation\existing_m1_smoke_after_m1_update.json `
  --drag-length-mm 0.1 --seed <available_seed>
```

## 9. 存储和排名禁止项

- 不把总预载平均为逐针恒定预载；
- 不调用 M2 静态二分预载根；
- 不把逐针独立 M2 路径相加；
- 不让拒绝步提交单针历史；
- 不把初始化失败记为零承载；
- 不把冲击峰当持续拉力；
- 不把代理值描述为实验标定值；
- 不把 120 万逐 case 文件直接写到移动 SSD；
- 不删除 scratch，除非 Parquet case 数和结果集哈希均验证并得到用户授权；
- 不在本交接状态下开启正式排名。

## 10. 关键文档

- `docs/m3/M3_OPEN_ISSUES.md`
- `docs/m3/M3_DESIGN_AND_RUN_PLAN.md`
- `docs/m3/M3_DATA_DICTIONARY.md`
- `docs/m3/M3_TEST_REPORT.md`
- `docs/m3/M3_to_M4_handoff.md`
- `docs/guide/M3_提示词_共同背板阵列与自适应筛选_执行版.md`
