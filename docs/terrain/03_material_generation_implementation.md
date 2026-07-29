# M1 真实感材料地形生成模块实施报告

完成日期：2026-07-29

实现范围：砂纸、普通烧结红砖外表面、普通粗糙混凝土墙面的二维高度场。
明确未修改：M2、M3、M4 物理模型、现有地形存储架构、UI、GPU 和打包。

## 1. 交付结论

已提供统一入口：

```python
from spine_sim.terrain import generate_terrain

terrain = generate_terrain(
    material="sandpaper",
    subtype="P100",
    size_x_m=0.050,
    size_y_m=0.050,
    resolution_m=5e-6,
    seed=12345,
    mode="synthetic",
)
```

返回对象包含 `height`、`dx`、`dy`、`valid_mask`、`material`、`subtype`、
`seed` 和 `metadata`。高度和坐标使用 m，数组顺序为 `[y,x]`，高度为
`float32`。三类生成器均支持任意非负整数 seed；同一版本、配置和调用参数产生
逐元素完全相同的结果。

`register_terrain()` 将结果注册到原 `TerrainLibrary`，仍保存为只读 mmap
`raw_height.npy`；原 `open_region()`、二维 region 读取和有限球头 track 路径无需
修改。有效掩码作为额外文件保存，合成输出全部有效，实测输出保留原始 mask。

## 2. 真实数据与证据

### 2.1 已实际获取并用于标定

来源：Sandpaper Wind Turbine Blade Benchmark Dataset，DOI
`10.17632/hcgcnm269w.2`，CC BY 4.0，Hirox 高程 CSV。

| grit | SHA-256 | shape `[y,x]` | spacing | 单样本调平后 RMS | 状态 |
|---|---|---:|---:|---:|---|
| P40 | `919ab87e...83aeb` | 926 × 8879 | 1.08954 µm | 104.10 µm | partially_validated |
| P100 | `a085c14f...e1d18` | 930 × 9421 | 1.08954 µm | 56.93 µm | partially_validated |
| P240 | `585fc5e1...7fce46` | 933 × 9440 | 1.08954 µm | 19.46 µm | partially_validated |

完整哈希、下载 URL、许可证和原始元数据位于
`data/catalog/` 与 Git 忽略的 `data/raw/.../source_metadata.json`。单样本标定
输出位于 Git 忽略的 `reports/terrain_calibration/`，可由标定脚本重新生成。

统一预处理为：

1. 原始文件只读并验证 SHA-256；
2. 保留独立有效掩码；
3. 对这批 Hirox 数据显式采用“精确零值为仪器无效”的**暂定数据集特定假设**，
   并排除其周围 12 samples 的安全带，避免把无效边界过渡当成深坑；通用导入器
   默认把零视为真实高度且默认安全带为 0；
4. 使用三轮 MAD 剔除的稳健刚体平面拟合；
5. 只去平均值/刚体平面，不使用高阶去趋势或未声明高通；
6. 降采样前执行二维箱式抗混叠，再做双线性重采样；
7. measured 模式拒绝用插值冒充更高测量分辨率。

### 2.2 没有被误用的数据

- P200 没有实测高程；配置明确标为 P180–P240 之间的 provisional 插值模型。
- P300 没有实测高程；配置明确标为 P240 之外的 provisional 外推模型。
- P180/P240 数据没有改名为 P200，P240 没有缩放后改名为 P300。
- 水泥浆 STL/PLY 没有被标为普通混凝土墙。
- 抛光混凝土没有被标为普通粗糙墙面。
- 普通 RGB、无绝对尺度 SEM、内部孔隙 micro-CT 没有作为地形真值。

## 3. 三类合成模型

### 3.1 砂纸

P40/P100/P240 默认采用实测补丁驱动合成：

1. 真实高程按目标分辨率抗混叠重采样；
2. 从完全有效区提取有重叠的物理尺寸补丁；
3. 每个位置从 seed 决定的多个候选中选择重叠误差较低者；
4. 用 raised-cosine 权重融合接缝；
5. 不做 90° 随机旋转，保留实测轴向信息；
6. 加入限幅的非周期各向异性 PSD 残差（RMS 的 6%）；
7. 对 1/5/25/50/75/95/99% 分位数做高度分布匹配。

算法不做周期平铺，并对最近使用过的补丁原点加入重复惩罚。P60/P120/P180 和
P200/P300 在没有本地实测标定时使用“两个粒度尺度场取最大值 + 粒间谷 + 少量
不规则突出磨粒”的 provisional 结构模型，不是白噪声、单一高斯场或简单分形面。

### 3.2 红砖

目标为未严重风化、无大裂缝和大面积剥落的普通烧结红砖外表面。模型为：

`base + directional + pores + fine`

- base：两层不同相关尺度的非周期随机起伏；
- directional：长短相关长度不同的成型/挤出方向纹理，不使用规则正弦波；
- pores：Poisson 数量、对数正态直径/深度、随机长宽比和旋转、谐波扰动边界、
  可聚集的不规则凹坑；
- fine：短相关长度、幂变换和偏度修正的非高斯烧结颗粒层。

当前参数是材料特定工程先验，状态为 `provisional`，不是实测红砖总体标定。

### 3.3 混凝土

目标为普通粗糙混凝土墙面，包含砂浆、局部骨料和表面气孔，不包含大裂缝或严重
剥落。模型为：

`mortar + aggregate + void + finish + fine`

- mortar：两个相关尺度的连续非高斯砂浆背景；
- aggregate：尺寸、长宽比、高度/嵌入深度、旋转和边界粗糙度均随机的不规则骨料；
- void：与骨料使用不同尺寸/深度分布的小气孔群；
- finish：`rough_wall/cast/troweled/brushed/exposed_aggregate` 的独立配置入口；
- fine：不淹没骨料和气孔的短尺度非高斯粗糙层。

普通 `rough_wall` 已完整实现；其他 finish 为合理默认入口但仍是 provisional。

## 4. 配置与防串用

参数不在 API 或生成函数中硬编码，分别位于：

- `src/spine_sim/terrain/material_profiles/sandpaper.json`
- `src/spine_sim/terrain/material_profiles/red_brick.json`
- `src/spine_sim/terrain/material_profiles/concrete.json`

加载器核对文件 material 标签、subtype 所属、schema 和状态。把 `P100` 当作
concrete subtype 会立即报错，不会回落到另一个材料的参数。每个输出记录 profile
hash；地形库重建时 profile hash 不一致会拒绝，避免配置变化后静默生成不同地形。

## 5. 几何验证

验证代码输出：

- 高度均值、RMS、标准差、偏度、峰度和 1/5/25/50/75/95/99% 分位数；
- x/y PSD、x/y ACF 的 1/e 相关长度和各向异性比；
- x/y 坡度 RMS 与分位数；
- 局部峰/坑密度、典型峰/坑尺寸和坑深；
- 非有限值、12 倍稳健标准差异常尖峰、相对边界相关、镜像相关、
  行/列梯度异常和非周期边界证据。

三 seed（123/456/789）结果生成于 Git 忽略的
`reports/terrain_validation/*/seed_ensemble.json`。
当前代表性 ensemble 均值：

| 材料 | subtype | RMS | skewness | kurtosis | corr x / y |
|---|---|---:|---:|---:|---:|
| sandpaper | P40 | 105.17 µm | 0.870 | 3.33 | 283 / 107 µm |
| sandpaper | P100 | 54.67 µm | -0.085 | 2.32 | 257 / 93 µm |
| sandpaper | P240 | 15.13 µm | 0.249 | 3.10 | 283 / 80 µm |
| red_brick | fired_brick_standard | 188.11 µm | 0.061 | 2.88 | 653 / 500 µm |
| concrete | rough_wall | 148.04 µm | 0.097 | 3.73 | 447 / 333 µm |

砂纸图包含真实随机 crop 与合成地形的高度图、截面、CDF、方向 PSD 和坡度比较。
红砖/混凝土没有合适真实数据，图中明确写出 `REAL HEIGHT MAP NOT AVAILABLE`，只
展示合成结果，未伪造“实测对比”。P40/P100/P240 都只能称为对单一样本的形貌
拟合，不能称为砂纸总体分布验证。

## 6. M3 最低兼容

材料地形注册后：

- `open_region()` 得到只读 `float32 [y,x]` mmap；
- shape、方向、spacing 和 SI 单位与原 manifest 一致；
- 高度没有 NaN/Inf；
- 有效 mask 另存且哈希记录；
- 固定 seed 可逐元素复现；
- 50 µm 有限球头 `cache_track()` 可读取该 region 并产生有效 track。
- M3 实际使用的二维杆体间隙读取函数已分别读取砂纸、红砖和混凝土小区域，并返回
  有限的 clearance 数组。

没有修改 `src/spine_sim/contact/*` 或 `src/spine_sim/array/*`。

## 7. 运行命令

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path

# 公共砂纸下载、大小与 SHA-256 核验
.\.venv\Scripts\python.exe scripts\terrain_data_probe.py `
  download-sandpaper --file P100.csv --raw-root data\raw --max-bytes 80000000

# 新测量的单样本标定工件（不修改原始文件/生产配置）
.\.venv\Scripts\python.exe scripts\calibrate_terrain_profile.py `
  scan.npy reports\scan_fit.json `
  --material red_brick --subtype fired_brick_standard `
  --height-unit um --spacing-x-um 2 --spacing-y-um 2 `
  --source-id project_brick_batch01 --license proprietary

# 生成并注册
spine-terrain generate-material output\P100_seed12345.npz `
  --material sandpaper --subtype P100 `
  --size-x-mm 50 --size-y-mm 50 --resolution-um 5 `
  --seed 12345 --mode synthetic --library terrain_library

# 多 seed 几何验证
spine-terrain validate-material reports\terrain_validation\P100 `
  --material sandpaper --subtype P100 `
  --size-x-mm 3 --size-y-mm 0.6 --resolution-um 10 `
  --seed 123 --seed 456 --seed 789

# 测试
.\.venv\Scripts\python.exe -m pytest -q tests\test_m1_material_terrain.py
.\.venv\Scripts\python.exe -m pytest -q
```

最终回归结果：`100 passed, 15 subtests passed`。其中新增材料测试为
`9 passed, 6 subtests passed`，并包含三类材料各一次 M3 二维读取 smoke。

## 8. 主要限制与后续重标定

1. 公共砂纸每个 grit 只有一个长条扫描，本报告不声称代表厂家、批次、磨料体系或
   磨损状态总体分布。
2. Hirox 的强轴向差异可能混有扫描/拼接效应；当前保留方向而不武断旋转。
3. exact-zero 无效语义仍需数据作者确认；假设已显式写入 provenance。
4. 红砖和普通粗糙混凝土缺少带绝对尺度、表面类型与许可证均明确的真实高度样本，
   因此只能 provisional。
5. 50 mm × 50 mm、5 µm 的返回数组约 10001 × 10001，单高度约 382 MiB；补丁
   合成和验证还需要额外工作内存。地形库存储本身没有改变。
6. 单值 `z=h(x,y)` 不能表达倒扣、封闭孔洞或多值侧壁，这与本轮工程边界一致。

用户取得新扫描后，运行 `calibrate_terrain_profile.py` 得到包含来源 SHA-256、
单位、mask、调平步骤、完整 descriptors 和建议字段的 JSON。审核表面类型、方向、
零值语义和重复样本后，更新对应材料 JSON 并保留新的 profile hash；用多个 specimen
划分标定与留出验证，再重新运行 `validate-material`。只有总体差异落入真实样本间
自然差异后，才能把状态从 provisional/partially_validated 升为 validated。
