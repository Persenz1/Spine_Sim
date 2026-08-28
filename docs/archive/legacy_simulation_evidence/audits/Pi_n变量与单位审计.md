# Pi_n 变量与单位审计

- 定义：`Pi_n = P / (N k_n sigma_g)`，无量纲。
- `P`：总法向预载，单位 N；大阵列实际档位 2/5/10 N。
- `N`：名义针数，20²/40²/60²。
- `k_n = k_s sin²(alpha)`：当前只用于固定 80°、均匀 2000 N/m 线性弹簧，
  得到 1939.693 N/m。
- `sigma_g`：每个 seed、每个规模，在 0–10 mm、201 个站点上采样有限球针尖
  `envelope_height_m`；每站对 N 根针的有效高度求标准差，再取路径中位数。
- 本轮完整柔顺 case 的 sigma_g 范围为 26.91–27.26 µm。
- 该 sigma 是有限球包络代理，尚未包含完整球冠/锥段/针杆门控。
- 30 µm 仅作为名义 RMS 敏感性基线，不作为主结果。

## 与 P/N 的公平塌缩对照

六个完整柔顺 cell 分别以 `eta = a + b log10(x)` 拟合，并做 leave-one-cell-out
交叉验证：

| 预测量 | 样本内 R² | LOOCV MAE | LOOCV RMSE |
| --- | --- | --- | --- |
| P_per_spine | 0.961 | 0.0315 | 0.0369 |
| Pi_n_finite_tip | 0.961 | 0.0315 | 0.0369 |

结论：no material separation; sigma_g varies little and k_n is fixed, so Pi_n is nearly a rescaled P/N。本轮 sigma_g 变化范围很窄且 k_n 固定，因而数据无法
辨识 sigma_g/k_n 归一化相对 P/N 的额外收益；Pi_n 的当前价值主要是给出有量纲变量
到无量纲理论曲线和反向设计关系。完整对照见
`../tables/Pi_n与P每针塌缩对照.csv`。
