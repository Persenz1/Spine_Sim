# ADR-001：表面形貌文件读取采用混合架构

- 状态：Accepted for Phase 0–1 design；正式依赖加入仍 deferred
- 日期：2026-07-29
- 决策范围：真实表面文件进入统一二维 SI 高度场的证据层
- 不在范围：最终材料地形生成器、M2/M3 接触和求解逻辑

## Context

项目当前运行时依赖只有 NumPy。`FileHeightMapSource` 能读取规则二维 `.npy`、`.csv` 和空白分隔 `.txt`，但要求数组全部有限，且调用者预先给出单位、origin 和 spacing。它不读取常见仪器格式、点云或 mesh，也没有完整 DOI/license/instrument/preprocessing schema。

公开数据具有多种形式：

- Hirox CSV：前五行包含 calibration、height unit、X/Y size；
- Surface-Topography Challenge：原始仪器格式和规范化 NetCDF；
- Digital Metrology：OS3D；
- 混凝土：CSV、STL、PLY；
- Micro-Topo：NPY + CSV metadata。

候选软件 [ContactEngineering/SurfaceTopography](https://github.com/ContactEngineering/SurfaceTopography)
支持 35+ 格式和标准 roughness/PSD/ACF 分析。2026-07-29 核实的 PyPI 版本为
1.22.0、Python `>=3.10`、MIT。当前项目环境实际 import 失败：

```text
ModuleNotFoundError: No module named 'SurfaceTopography'
```

在本 ADR 前及 Phase 0–1 中没有把它加入正式依赖。

## Decision drivers

1. 原始文件、单位、orientation、缺失值和预处理必须可审计；
2. 不自行长期维护大量厂商二进制格式；
3. 最小公开样例即使没有可选依赖也能验证下载/哈希/基本解析；
4. 解析结果必须标准化到当前 M1 的规则二维 SI 高度场；
5. 后端选择不得泄漏进 M3 或改变 `TrackGeometry`；
6. 失败必须带完整错误，不得把“库能识别扩展名”写成“数据已获取”；
7. 大文件需要按需读取、分块或尽早拒绝，不能无界复制。

## Considered options

### Option A：直接把 SurfaceTopography 作为唯一读取层

优点：

- 广泛的仪器和标准格式；
- 统一/非统一 topography 数据模型；
- 单位、物理尺寸、mask 和统计方法已有成熟实现；
- 与 contact.engineering 和 Surface-Topography Challenge 的数据生态一致；
- 减少自写二进制解析器。

缺点：

- 增加编译/科学计算依赖及平台兼容面；
- 并非所有公开 CSV 或研究者自定义格式都能自动识别；
- 自动识别成功仍不证明轴方向、z 正方向、license 和预处理正确；
- 把唯一读取路径绑定到第三方 API，升级需要回归；
- 当前 CI/Windows 环境尚未安装和验证。

结论：适合做主要多格式后端，但不适合作为唯一证据层。

### Option B：项目自己维护全部轻量解析器

优点：

- 每一行元数据和单位转换都可见；
- 依赖小；
- 对少数明确、简单格式容易做严格错误处理；
- 可针对大 CSV 做流式/分块实现。

缺点：

- 维护 35+ 厂商/标准格式不现实；
- 容易错误解释专有格式、mask、通道、非均匀坐标或压缩；
- 统计算法、单位和 instrument metadata 会重复造轮子；
- 需要长期跟踪格式版本。

结论：只适合少量文本/数组格式和证据探针，不适合替代成熟 metrology reader。

### Option C：混合架构

使用两类 adapter，但共享同一个标准化和证据 schema：

1. 项目自带严格轻量解析器：
   - 固定格式的公开文本文件；
   - NPY；
   - 最小 metadata/哈希/下载探针；
   - 当前实现的 Hirox CSV。
2. 可选 SurfaceTopography 后端：
   - NetCDF、X3P、OS3D、仪器二进制、非均匀 topography 等；
   - 读取后的值仍须经过项目自己的 orientation/unit/provenance validation。
3. mesh/point cloud 另设明确转换器：
   - 不假装成 SurfaceTopography 已经解决的规则高度图；
   - 转成高度场时保存投影、遮挡、多值 z 和失配记录。

优点：

- 简单、关键公开格式即使没有扩展依赖也可重复验证；
- 复杂格式复用成熟实现；
- M3 只看到标准化成品；
- 便于用同一 fixture 对两个后端交叉验证。

缺点：

- 需要定义清楚 adapter 边界；
- 同一格式可能有两个读取路径，必须做一致性测试；
- 可选依赖和功能降级需要文档。

## Decision

采用 **Option C：混合架构**。

Phase 0–1 的具体决定：

- 保留 `scripts/terrain_data_probe.py` 中的严格 Hirox CSV 证据解析器；
- 不把它宣称为通用 metrology parser；
- 不在本阶段添加 SurfaceTopography 正式依赖；
- 用公开 P240 完成实际下载、SHA-256 和端到端读取；
- 把 SurfaceTopography 的安装、Windows/CI、目标格式 fixture 和依赖锁定留到下一阶段；
- 无论使用哪个后端，输出先进入统一的 provenance-preserving 标准化层，再转成现有 `float32 [y,x]` SI 高度场；
- M3 的 `TerrainLibrary`/`TrackGeometry` 调用接口不变。

下一阶段若验证通过，建议把 SurfaceTopography 放入独立 optional extra，而不是核心运行时强制依赖。具体版本范围在 CI 和目标格式测试完成后再确定，不能只因为当前 PyPI 最新版本是 1.22.0 就直接锁定。

## Normalized ingestion contract

每个 adapter 必须返回或生成以下信息：

```text
source identity:
  dataset_id, doi, landing_page, license, access_method
file identity:
  original_path, format, byte_size, sha256
measurement:
  instrument, specimen, patch, batch, finish, condition
geometry:
  axis_order, row_direction, handedness, z_positive
  shape, origin_m, spacing_m or explicit positions_m
  raw_coordinate_unit, raw_height_unit
quality:
  valid_mask, missing_ratio, saturation/outlier indicators
processing:
  ordered immutable list of crop/level/detrend/filter/fill/resample operations
output:
  float32 height_m[y,x], normalized_sha256
```

未知方向或单位必须停止标准化并产生 actionable error，不能使用默认猜测。对缺失值的填补必须生成新的 processed artifact 和 mask，不能覆盖原始数据。

## Error handling requirements

- 网络失败：记录 URL、HTTP 状态、超时和是否需要登录；
- 文件过大：在下载前比较发布方 size 与显式 `--max-bytes`；
- 哈希不匹配：删除临时文件，不覆盖已验证文件；
- 已存在但哈希错误：停止并保留文件供人工检查，不静默覆盖；
- 格式/shape/单位错误：报告文件、header 值、期望值；
- 依赖缺失：说明 optional backend 未安装，不把数据标记为不可获得；
- 授权受限：记录人工申请步骤，不自动填写表单或绕过访问。

## Testing requirements before formal dependency

1. Linux/Windows CI import；
2. 至少各一个 NetCDF、X3P、OS3D、NPY 和一个目标仪器格式 fixture；
3. 单位和 physical size round-trip；
4. mask/NaN、非均匀坐标、多通道选择；
5. 与轻量 Hirox parser 对同一标准化数组/统计的交叉检查（若格式转换可行）；
6. 大文件内存峰值和 lazy/chunk 行为；
7. 第三方升级前后 normalized hash 或容差比较；
8. M1→M3 接口回归。

## Consequences

正面：

- 复杂格式能力与小依赖核心解耦；
- 公开数据验证可以立即复现；
- 来源和处理证据不依赖第三方对象生命周期；
- 后续增加材料数据不需要修改 M3。

负面：

- 项目需要维护 adapter registry、标准化 schema 和双后端一致性测试；
- SurfaceTopography 尚未在当前环境可用；
- Hirox parser 目前仍将完整 CSV 读为 `float64`，对更大文件需要分块优化；
- 本 ADR 不解决 material calibration，也不授权最终生成器实现。
