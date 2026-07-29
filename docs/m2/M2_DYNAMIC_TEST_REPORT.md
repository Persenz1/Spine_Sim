# M2 持续恒载动力学测试报告

**原始执行日期：** 2026-07-28

**当前复核：** 2026-07-29

**模块版本：** `m2.2.0`
**模型等级：** `project_model_P_main_plane_dynamic_constant_preload_v2`

## 1. 验收结果

2026-07-29 清理后复核：

```powershell
spine-m2 validate-analytic `
  --output results\m2_validation\dynamic_analytic_validation_current.json
```

结果：12/12 通过。机器可读报告属于可再生 `results/` 产物，不提交。

| 门禁 | 结果 | 关键证据 |
|---|---|---|
| 平面持续预载 | 通过 | 外载 0.5 N；稳态法向反力均值 0.500149 N |
| 平面 Coulomb 拖动 | 通过 | \(\mu_kW=0.1\) N；全局拖拽力中位数 0.10000 N |
| 零预载 | 通过 | 法向力与拖拽力峰值均为 0 |
| 斜坡力和功方向 | 通过 | 完成路径；驱动功符号正确 |
| 凸起后复接触 | 通过 | 6 次脱离、6 次冲击/再接触；未终止 |
| 双凸起动态路径 | 通过 | 15 次脱离、15 次再接触；无 `model_unclosed` |
| 弹簧三段 | 通过 | LOWER_STOP/INTERIOR/HARD_STOP |
| 针径趋势 | 通过 | 0.6/0.8 mm 横向柔度比 3.12997；动态频率趋势正确 |
| 时间步减半 | 通过 | 拉力中位数差 \(1.67\times10^{-5}\) N；法向均值差 0.000311 N |
| 动力学/能量残余 | 通过 | 平面最大动力学残余 \(2.22\times10^{-16}\) N |
| 确定性重放 | 通过 | 相同输入的完整 wrench 序列逐值相同 |
| 正式排名门禁 | 通过 | 未标定参数和收敛配对未完成时强制 false |

## 2. M2 专项回归

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_m2_contact.py -q
```

结果：**9 passed**，耗时 9.55 s。此次文档清理没有重跑 M1 或全仓测试；M3 专项
复核记录在 [`../m3/M3_TEST_REPORT.md`](../m3/M3_TEST_REPORT.md)。

旧 `LegacyPrescribedPoseConstitutiveCore` 仍可编译并保持 proposal/commit 原子语义，供 M3
迁移；正式 M2 case 已切换到 `DynamicSingleSpineExperiment`。

## 3. 历史旧地形短程 smoke

以下结果生成于旧 M1 `defined_geometry` 数据仍存在时。旧地形和对应可再生
`results/` 已永久删除，因此这里只保留数值快照，不能按原输入复跑，也不能代表
当前材料地形。

使用 seed 32001、50 μm M1 track、0.8 mm 针径、70°、800 N/m、低摩擦档，在显式
声明的未标定动态参数下运行 2 mm：

- 2000 个内部时间步，496 个输出/事件点；
- 完成 2 mm 路径；
- 持续外部预载 0.5 N；
- 接触占空比 1.0；
- 全局拖拽力 P10/中位数/峰值：0.0399/0.2199/2.1722 N；
- 法向力范围：0.1881–2.3987 N；
- 最大离散动力学残余：\(4.44\times10^{-16}\) N；
- 最大单步能量残余：\(7.24\times10^{-5}\) J；
- `model_state=parameter_unclosed`；
- `formal_ranking_eligible=false`。

该 smoke 只证明 M1 随机地形接口、持续恒载时域推进和结果字段可运行。峰值及硬件
趋势不能解释为真实预测。

## 4. 正式筛选阻断项

1. 安装座等效质量未由硬件冻结；
2. 结构和执行器阻尼未标定；
3. 针尖—砖/水泥动态接触/恢复参数未标定；
4. 杆体净空、屈服、屈曲与微损伤尚未全部闭合；
5. 正式 case 的时间步减半和动态参数敏感性尚未批量执行。

因此本次提交修复物理边界和求解流程，但不授权重新运行 M2 参数筛选。
