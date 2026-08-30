# Spine Sim 文档

本目录只说明当前程序：系统怎样组成、各模块怎样实现、怎样调用、输入输出是什么，以及模型能回答到哪里。

这里不保存开发过程材料，也不进行仿真结果分析。构型优劣、实验结论、阶段报告、任务交接、提示词、修补记录和运行生成物都不属于本目录。

## 阅读顺序

第一次接触项目时，按以下顺序阅读：

1. 本页：理解系统全貌；
2. [公共约定](公共约定.md)：统一单位、坐标、状态、参数和数据含义；
3. [运行与使用](运行与使用.md)：安装、命令和调用流程；
4. 按需要阅读对应模块；
5. 遇到公式或理论边界时回看[原始机理全文](钩爪式爬壁机器人抓附机理与多尺度力学模型.md)。

## 文档清单

| 文档 | 内容 |
|---|---|
| [公共约定](公共约定.md) | 公共基础、单位、坐标、力、状态、参数、identity 和输出格式 |
| [地形模块](地形模块.md) | 地形生成、实测导入、材料 profile、地形库和有限球尖包络 |
| [几何模块](几何模块.md) | 搜索路径、候选接触、法向、门控、杆体间隙和 continuation cursor |
| [单刺模块](单刺模块.md) | 单刺柔顺、摩擦、弹簧、状态事件、容量和 trial/commit |
| [阵列模块](阵列模块.md) | 刚性背板、混合控制、活动集、事件级联、稳定性和公共指标 |
| [运行与使用](运行与使用.md) | 环境、CLI、Python API、campaign、缓存和输出读取 |
| [原始机理全文](钩爪式爬壁机器人抓附机理与多尺度力学模型.md) | 用户提供的理论原文，逐字保留，不在其中加入工程说明 |

## 系统全貌

程序的物理链为：

```text
地形配方 / 实测表面
        │
        ▼
TerrainLibrary 二维区域
        │
        ▼
有限球尖 TrackGeometry
        │
        ▼
ContactCandidate + CandidateCursor
        │
        ▼
solve_single_spine
        │
        ▼
solve_array_equilibrium
```

批量运行和存储是物理链外的一层：

```text
campaign.json
  → CampaignSpec
  → CampaignRunner
  → case callable(parameters, RunContext)
  → CaseOutput
  → ResultStore
```

通用 runner 不自动执行整条物理链。具体项目的 case callable 负责准备地形/候选/参数、调用单刺或阵列求解器，并组装输出。

## 模块与源码

| 逻辑模块 | 主要源码 | 文档归属 |
|---|---|---|
| 公共基础 | `src/spine_sim/core/` | [公共约定](公共约定.md) |
| 参数系统 | `src/spine_sim/parameters/` | [公共约定](公共约定.md) |
| 地形 | `src/spine_sim/terrain/` | [地形模块](地形模块.md) |
| 候选几何 | `src/spine_sim/geometry.py` | [几何模块](几何模块.md) |
| 单刺 | `src/spine_sim/single_spine.py` | [单刺模块](单刺模块.md) |
| 阵列 | `src/spine_sim/array.py` | [阵列模块](阵列模块.md) |
| 公共指标 | `src/spine_sim/metrics.py` | [阵列模块](阵列模块.md) |
| 执行与结果 I/O | `src/spine_sim/runtime/`、`src/spine_sim/io/`、`src/spine_sim/cli.py` | [运行与使用](运行与使用.md) |

## 当前实现范围

当前生产模型层级只有两个：

- `single_spine_quasistatic`：三维点接触、库仑摩擦、低阶线弹性梁/悬架、单边弹簧、事件定位和条件容量；
- `array_rigid_backplate_event`：刚性共同背板、六自由度混合控制、逐刺调用单刺本构、活动集、事件级联和平衡/准静态稳定性判断。

地形生成、有限针尖几何、单刺和阵列都有可运行实现。以下内容没有生产求解器：

- 整爪和整机 Newton–Euler；
- 真实动态回弹、冲击、质量和阻尼；
- 损伤后的几何演化、疲劳和磨损；
- 柔性背板连续弯曲；
- 倒悬孔洞、多值侧壁和一般三维实体接触。

缺少当前模型需要的材料或拓扑参数时，程序返回 `PARAMETER_UNCLOSED`；明确不属于当前模型的能力返回 `OUT_OF_SCOPE`。两者都不能解释为物理失效。

## 信息来源

不同内容有不同的权威来源：

1. 方程、符号和理论假设：以[原始机理全文](钩爪式爬壁机器人抓附机理与多尺度力学模型.md)为理论来源；
2. 当前程序行为、API 和边界：以源码和测试为准；
3. 候选参数、protocol、seed 和证据状态：以 [`src/spine_sim/parameters/registry.json`](../src/spine_sim/parameters/registry.json) 为机器来源；
4. 本目录负责解释以上内容，不创造第二套模型或参数真值。

原始机理包含尚未实现的整爪、整机和动态扩展。文档中“理论存在”不等于“代码已经实现”。

## 文档边界

新文档只保留长期有效的说明。以下内容不再进入 `docs/`：

- 按日期命名的实现、测试、GPU 或 benchmark 报告；
- M0/M1 等开发阶段交接；
- 开放问题清单和讨论过程；
- 历史 solver 的结果、排名和构型分析；
- 运行生成的图、表、trace 和 campaign 结果。

历史变化通过 Git 查看；运行产物写入 `results/`、`output/` 或用户指定目录。
