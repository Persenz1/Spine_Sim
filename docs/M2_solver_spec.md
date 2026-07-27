# M2 单刺求解规格

**模块版本：** `m2.0.0`
**模型层级：** 项目模型 P + 数值算法 A
**坐标：** 首版固定在局部 \(x\)-\(Z\) 主平面；所有内部量采用 SI。
**权威接口：** M1 `TrackGeometry`，其横坐标是针尖球心坐标，不是支撑点坐标。

## 1. 输入、输出和不变量

规定姿态核心输入：

- 安装座 \(\mathbf r_h=[x_h,z_h]^T\)；
- 单刺参数 \(R_t,d,L,\alpha,E,\nu,\kappa,\mu_s,\mu_k\)；
- 轴向模式：刚性，或无预压单边弹簧 \(k_s,\delta_{\max}\)；
- M1 `TrackGeometry`；
- 上一步已接纳的 `SingleSpineState`；
- `commit` 标志。

核心不读取目标法向力、不搜索安装座位置、不持有可变隐藏状态。输出同时包含
`proposal_state` 和 `next_state`：`commit=False` 时 `next_state` 严格等于旧状态，
`commit=True` 时等于 proposal。旧状态本身是冻结数据，失败求解不能污染它。

每个响应保存墙面对针的力 `wall_on_spine_force`，以及作用反力
`spine_on_plate_wrench_about_holder`。后者的力为前者的相反数，力矩由实际支撑点
搬移到安装点。两者不得混用。

## 2. 几何查询

给定球心横坐标 \(c_x\)，在相邻有效 M1 节点间线性插值得到
\(H_R,H'_R,q_x,q_y\)。定义

\[
g(c_x,c_z)=c_z-H_R(c_x),
\quad
\boldsymbol\tau={ [1,H'_R]^T \over \sqrt{1+(H'_R)^2}},
\quad
\mathbf n={[-H'_R,1]^T \over \sqrt{1+(H'_R)^2}}.
\]

支撑高度由球几何恢复：

\[
q_z=H_R-\sqrt{R_t^2-(q_x-c_x)^2-(q_y-y_{\rm track})^2}.
\]

候选接触还必须满足：

\[
(\mathbf q-\mathbf c)\cdot\mathbf a\ge-\epsilon_{\rm cap}.
\]

任一插值端点无效、查询出域或球根为负均返回 `geometry_out_of_domain` 或
`invalid_geometry`，不能改写成脱离。`near_tie_flag` 只进入模型警告。

## 3. 结构相容

\[
\mathbf a=[\cos\alpha,-\sin\alpha]^T,\qquad
\mathbf b=[\sin\alpha,\cos\alpha]^T.
\]

\[
A={\pi d^2\over4},\quad I={\pi d^4\over64},\quad
c_a={L\over EA},\quad
c_b={L^3\over3EI}+{L\over\kappa GA},
\quad G={E\over2(1+\nu)}.
\]

墙面对针的力写为

\[
\mathbf f=-Q_s\mathbf a+V_b\mathbf b
          =\lambda_n\mathbf n+f_t\boldsymbol\tau.
\]

针尖球心必须满足

\[
\mathbf c=\mathbf r_h+L\mathbf a
           -(\delta_s+c_aQ_s)\mathbf a+c_bV_b\mathbf b.
\]

首版正式比较始终保留梁柔顺。若轴向弹簧和梁同时关闭且 STICK 反力不唯一，
返回 `indeterminate_rigid_stick`，不加罚刚度或正则化。

## 4. 单边弹簧活动集

令 \(s=-[\mathbf c-(\mathbf r_h+L\mathbf a)]\cdot\mathbf a\)。STICK 分支可手算：

| 分支 | 条件 | 解 |
|---|---|---|
| 刚性轴向 | axial mode = rigid | \(\delta_s=0,\ Q_s=s/c_a\) |
| LOWER_STOP | \(s\le0\) | \(\delta_s=0,\ Q_s=s/c_a\le0\) |
| INTERIOR | \(0<s<\delta_{\max}+c_ak_s\delta_{\max}\) | \(Q_s=s/(c_a+1/k_s),\ \delta_s=Q_s/k_s\) |
| HARD_STOP | \(s\ge\delta_{\max}+c_ak_s\delta_{\max}\) | \(\delta_s=\delta_{\max},\ Q_s=(s-\delta_{\max})/c_a\) |

边界在声明容差内归到相邻低载分支，避免数值抖动。LOWER_STOP 允许结构试算得到
\(Q_s\le0\)，但接触最终仍须通过 \(\lambda_n\ge0\)；它不是允许弹簧受拉。

SLIDE/首次闭合分支中由候选 \(\lambda_n\) 得到 \(Q_s\)，再按下式直接选段：

- \(Q_s\le0:\ \delta_s=0\)，LOWER_STOP；
- \(0<Q_s<k_s\delta_{\max}:\ \delta_s=Q_s/k_s\)，INTERIOR；
- \(Q_s\ge k_s\delta_{\max}:\ \delta_s=\delta_{\max}\)，HARD_STOP。

刚性轴向始终固定 \(\delta_s=0\)，不使用大刚度近似。

## 5. 接触分支

### 5.1 FREE

以零力结构预测

\[
\mathbf c_0=\mathbf r_h+L\mathbf a.
\]

若 \(g(\mathbf c_0)>\epsilon_g\)，或上一步为接触而 STICK 试算要求
\(\lambda_n<-\epsilon_\lambda\)，则返回零力 FREE。拖动阶段从接触转为 FREE 时记录
`detach_to_free`，路径继续。

### 5.2 FIRST_CONTACT / RECONTACT

旧状态为 FREE 且 \(g(\mathbf c_0)\le\epsilon_g\) 时建立接触候选。用
\(f_t=0\) 的无摩擦闭合方向求最小非负 \(\lambda_n\)，使
\(g(\mathbf c(\lambda_n))=0\)。包装器会把宏观跨越二分细化，因此事件点通常
\(\lambda_n\approx0\)。闭合后将当前球心设为切向锚点：

\[
(c_{x,a},c_{z,a})=(c_x,c_z).
\]

首次事件标为 `FIRST_CONTACT_EVENT`，此前曾接触则标为
`RECONTACT_EVENT`。事件状态在下一步按 STICK 历史处理。

### 5.3 STICK 试算

STICK 的几何未知量由已接纳锚点固定：

\[
\mathbf c=\mathbf c_a,\qquad g(\mathbf c_a)=0.
\]

将所需结构位移投影到 \(\mathbf a,\mathbf b\)，按第 4 节求 \(Q_s,\delta_s\)，且

\[
V_b={[\mathbf c_a-(\mathbf r_h+L\mathbf a)]\cdot\mathbf b\over c_b}.
\]

再由 \(\mathbf f=-Q_s\mathbf a+V_b\mathbf b\) 得
\(\lambda_n=\mathbf f\cdot\mathbf n,\ f_t=\mathbf f\cdot\boldsymbol\tau\)。

- \(\lambda_n<-\epsilon_\lambda\)：DETACH；
- \(|f_t|\le\mu_s\lambda_n+\epsilon_f\)：接纳 STICK；
- 否则以 \(\operatorname{sign}(v_{\rm rel,t})=-\operatorname{sign}(f_t)\)
  转入 SLIDE。

静摩擦力完全由锚点相容决定，禁止在摩擦锥内任意指定。

### 5.4 SLIDE

继续同向拖动时保持 SLIDE；停止或反向后才从上一滑动点建立 STICK 试算。滑动律：

\[
\mathbf f(\lambda_n)=\lambda_n
\left[\mathbf n-\mu_k\operatorname{sign}(v_{\rm rel,t})\boldsymbol\tau\right],
\qquad\lambda_n\ge0.
\]

对给定 \(\lambda_n\) 计算 \(Q_s,V_b,\delta_s,\mathbf c\)，求一维相容根：

\[
r_g(\lambda_n)=c_z-H_R(c_x)=0.
\]

算法先做有限载荷范围内的确定性括区扫描，再用二分求最小连续根。每次残余评估
都重新查询 M1 几何并重新选择弹簧活动段。若载荷参数化因支撑切换不能括根，
使用严格等价的球心横坐标参数化：令 \(c_z=H_R(c_x)\)，由结构相容反算
\(Q_s,V_b,\mathbf f\)，再对
\(f_t+\mu_k\operatorname{sign}(v_{\rm rel,t})\lambda_n=0\) 括根，并选取离旧球心
最近的连续解支。若零力间隙为正则返回 FREE；若有限载荷安全界内两种参数化均
无根，返回 `no_admissible_contact_equilibrium`，将模型状态与数值状态分开，
不伪造接触力。

接纳的滑动点更新“下一步若停止时”的锚点为当前球心，但连续同向滑动不会因为
\(\mu_k<\mu_s\) 每步错误回粘。

## 6. 状态切换

| 旧状态 | 条件 | 新响应 |
|---|---|---|
| FREE/DETACH | 正间隙 | FREE |
| FREE/DETACH | 闭合且首次 | FIRST_CONTACT_EVENT |
| FREE/DETACH | 闭合且曾接触 | RECONTACT_EVENT |
| FIRST/RECONTACT/STICK | 静摩擦可行 | STICK |
| FIRST/RECONTACT/STICK | 超静摩擦锥 | SLIDE + `slip_start` |
| SLIDE | 同向继续 | SLIDE |
| SLIDE | 停止/反向且静摩擦可行 | STICK |
| 任一接触 | 所需 \(\lambda_n<0\) 或零力开离 | DETACH_EVENT，proposal 为零力状态 |

接触状态与弹簧状态正交。`HARD_STOP` 事件不把接触状态改名。

## 7. proposal / commit

`solve_pose(..., old_state, commit=False)`：

1. 只读 `old_state`；
2. 计算完整响应和 `proposal_state`；
3. 所有几何、物理、数值和模型检查通过后才产生可提交 proposal；
4. `next_state is old_state`。

`commit=True` 使用完全相同的求解路径，但 `next_state=proposal_state`。事件细化
的所有试算都从同一个旧状态调用 `commit=False`；只有最终事件点提交一次。

## 8. 残余和余量

逐点至少保存：

- 几何残余 \(r_g=c_z-H_R(c_x)\)；
- 结构残余：直接运动学重构与返回球心之差；
- 力分解残余：
  \(\|\mathbf f-(\lambda_n\mathbf n+f_t\boldsymbol\tau)\|\)；
- 单边余量 \(\lambda_n\)；
- 静/动摩擦余量；
- 弹簧下限、行程和活动段余量；
- 球冠门控余量；
- 地形有效域、near-tie 和杆体碰撞模型状态；
- 根迭代数、括区和终止原因。

法向/几何根容差与离散 M1 切向上的 Coulomb 残余容差分开保存。10 μm 支撑切换
处允许的动摩擦等式残余上限为 \(5\times10^{-4}\) N，并必须在 5 μm 复核时检查
收敛；它不是有效承载阈值。有效承载阈值是独立的业务输入。

## 9. 能量

弹性能：

\[
U={1\over2}c_aQ_s^2+{1\over2}c_bV_b^2+U_s,
\]

\[
U_s=
\begin{cases}
0,&\text{刚性或 LOWER_STOP}\\
Q_s^2/(2k_s),&\text{INTERIOR}\\
k_s\delta_{\max}^2/2,&\text{HARD_STOP}.
\end{cases}
\]

逐步以梯形力计算安装座功、接触功、弹性能变化和

\[
r_E=W_{\rm holder}+W_{\rm contact}-\Delta U.
\]

滑动摩擦耗散取 \(\max(0,-W_{\rm contact})\)。事件点和大曲率处该离散残余只作
收敛审计，不作为材料损伤能。

## 10. 试验包装器

1. 由零力球心在轨迹上的首个几何闭合位置确定初始接触高度；
2. 沿 \(-Z\) 只在初始阶段括区并二分目标 \(\lambda_n=0.5\ {\rm N}\)；
3. 失败只在此处产生 `initial_preload_infeasible`；
4. 成功后冻结 \(z_h\)，沿 \(+x\) 位移控制；
5. 宏观 CONTACT/DETACH/SLIP_START/HARD_STOP 用同一旧状态二分定位；
6. FREE 段继续扫描，允许后续 RECONTACT；
7. 只在路径终点、地形出界、结构安全界、局部重试后数值失败或模型未闭合时终止。

路径摘要不生成综合分数，分别计算接触长度、有效承载长度、最大连续承载距离、
切向力峰值/中位数/10%/25% 分位、实际法向力范围、事件计数、残余和四维终态。

## 11. 夹具手算基线

- 平面：\(H_R=R_t,\ \tau=[1,0],\ n=[0,1]\)；
- 斜坡 \(H_R=mx+R_t\sqrt{1+m^2}\)：
  \(\tau=[1,m]/\sqrt{1+m^2}\)，
  \(n=[-m,1]/\sqrt{1+m^2}\)；
- 直径趋势：Euler-Bernoulli 主项
  \(c_b\propto I^{-1}\propto d^{-4}\)；
- 弹簧段：\(Q_s=0\) 和 \(Q_s=k_s\delta_{\max}\) 两处连续；
- 刚性极限：\(k_s\to\infty\) 时
  \(\delta_s\to0\)，只保留针体梁。

解析门禁必须先于任何随机地形参数排名。十种 M1 项目地形在 M0–M4 冻结前只可
用于接口、状态覆盖和数据完整性 smoke，不构成第一轮筛选结果。
