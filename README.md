# Spine Sim

Spine Sim is a Python simulator for microspine contact and load sharing on rough surfaces. It models individual spines and arrays attached to a common rigid backplate, following contact establishment, preload and tangential dragging. Outputs include reaction forces, spine displacement, contact and slip states, trajectories and termination reasons.

## Capabilities and model scope

- Generate synthetic height fields, import measured topography and construct finite-radius tip envelopes.
- Solve contact compatibility, nonuniform load sharing and friction history under prescribed backplate motion and loading.
- Represent axial spring mounts, finite travel and fixed mounts; compare rigid spines with condensed elastic spines.
- Run configured campaigns with parallel workers, saved results and explicit restart commands.
- Inspect progress through local browser interfaces and analyze stored trajectories.

The package contains several distinct solvers. The reduced contact solver uses rigid spines by default for spring mounts and condensed elastic spines for fixed mounts. Its smoothed tip envelope and reduced compliance do not resolve every mechanism in the general theory: elastic configurational forces, full rod collisions and tip-cap restrictions are omitted in this branch. A spatial rod formulation and older analytical fixtures are also retained for separate studies. Results from different formulations must retain their model identity.

This is a quasi-static, geometry-resolved array model, not a full robot or three-dimensional solid finite-element model. It does not simulate dynamic rebound, evolving surface damage or flexible-backplate deformation. The [mechanical theory and derivations](docs/共同背板柔顺爪刺阵列：力学理论与推导.md) supplement the module-specific documents and cover a wider range of mechanisms than any single implementation.

## Theory documentation

- [Shared conventions](docs/公共约定.md)
- [Terrain](docs/地形模块.md) and [contact geometry](docs/几何模块.md)
- [Single-spine mechanics](docs/单刺模块.md) and [array mechanics](docs/阵列模块.md)
- [Spatial rod contact formulation](docs/空间杆接触模型.md) and [reduced contact formulation](docs/降阶接触模型.md)
- [Original multiscale formulation](docs/钩爪式爬壁机器人抓附机理与多尺度力学模型.md)
- [Common-backplate compliant-array derivations](docs/共同背板柔顺爪刺阵列：力学理论与推导.md)

## Installation

Requires Python 3.11+, NumPy 1.26+ and SciPy 1.16+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,plot,parquet]"
```

Optional CUDA terrain support:

```powershell
python -m pip install -e ".[gpu-cuda13]"
```

CUDA support requires a compatible NVIDIA driver. Contact solving normally runs on the CPU. Some material recipes require measured input files in `data/raw/`; these datasets are separate from the source distribution, with source information in `data/catalog/`.

## Usage

```powershell
spine-sim --help
spine-terrain --help
spine-sim validate-env
spine-sim run-case examples/smoke_campaign.json --backend cpu --output results
```

The smoke example exercises the runner and storage only. `examples/canonical_campaign.json` is an analytical fixture for the older solver. Other examples and the configurations in `experiments/` select their own models; they are not interchangeable.

For a prepared campaign configuration:

```powershell
spine-sim run-campaign campaign.json --backend cpu --workers 2 --output results
spine-sim resume campaign.json --backend cpu --output results
spine-sim summarize results/<campaign_id>
```

Specialized campaign and analysis tools are in `scripts/`; use their `--help` to select inputs and output locations. Some retain historical filenames and machine-specific defaults, so provide paths explicitly when reusing them. Portable packages expose `START_CONFIRMATION.bat` to open the local control interface and `TEST_ENVIRONMENT.bat` to check the environment; starting a campaign remains an explicit action in that interface.

## Numerical and execution conventions

- Internal quantities use SI units. Contact force is the surface force on a spine: drag resistance is `T = -sum(f_x)`, preload is `P = sum(f_z)`, and local normal force is `N = f · n >= 0`. Individual vertical components can be negative.
- Total preload does not imply equal force per spine. Load sharing follows compatibility and equilibrium; backplate rotations remain fixed in the common-backplate formulation.
- Preserve the accepted state through approach, preload and dragging, including spring compression, contact normals, gaps and friction history. Preload placement retries start from a fresh approach and retain failed attempts; dragging does not switch placements or splice paths.
- Batch execution and restart are explicit. Opening a monitor or inspecting results must not launch a campaign. New production terrain generation uses the configured CUDA path and stops if CUDA is unavailable; existing inputs can be reused without regeneration.
- Keep material parameters, boundary conditions, tolerances and loading protocols fixed within a campaign. Stop dispatching new cases when pausing and let active cases finish. Retain failed, partial and budget-limited results with their actual termination reasons.
- Do not interpolate across unresolved trajectory gaps or treat solver convergence as physical capacity. Performance changes must preserve the intended equations and history; residual agreement alone does not establish trajectory equivalence.

## Repository layout

| Path | Contents |
|---|---|
| `src/spine_sim/` | Contact solvers, geometry, terrain, runtime and result I/O |
| `examples/` | Small input configurations and analytical fixtures |
| `experiments/` | Saved campaign definitions and design tables |
| `scripts/` | Campaign launchers, packaging and analysis tools |
| `tests/` | Numerical, boundary-condition and software tests |
| `docs/` | Module-specific mechanics and a supplementary full derivation, in Chinese |
| `monitor/dist/` | Runtime browser pages required by the launchers |
| `data/catalog/` | External input source and license information |
| `provenance/` | Input metadata referenced by the parameter registry |

Generated results, reports, terrain caches and temporary files are excluded from version control. Keep persistent datasets separate from disposable caches. The local virtual environment is reusable and is not a simulation output.

## Tests

```powershell
python -B -m pytest tests/test_balanced_contact.py -q -p no:cacheprovider
```

This checks the reduced contact solver, including fixed/free backplate conditions. Use `python -m pytest -q` for the complete test suite. Numerical tests do not constitute experimental calibration.

---

# Spine Sim 中文说明

Spine Sim 是用于粗糙表面微刺接触与阵列分载研究的 Python 仿真程序。程序计算单刺及共同刚性背板上的针阵列，从接触建立、预载到切向拖动，输出反力、针位移、接触与滑移状态、加载轨迹和终止原因。

## 功能与模型范围

- 生成合成高度场、导入实测形貌，构造有限半径球尖的可达包络。
- 在给定背板运动与载荷下，联立接触相容、非均匀分载和摩擦历史。
- 表示轴向弹簧、有限行程与固定安装，对比刚性针和凝聚弹性针。
- 按配置并行执行批次，保存结果并通过显式命令恢复。
- 通过本地浏览器页面查看进度，使用分析脚本读取已保存轨迹。

仓库保留不同层次的求解器。降阶接触求解器对弹簧安装默认使用刚性针，对固定安装使用凝聚弹性针。其平滑球尖包络与降阶柔度不包含一般理论中的全部机制：该分支省略弹性配置力、完整杆体碰撞和球尖帽域检查。空间杆模型与早期解析夹具保留用于独立对照，不同模型的结果应保留各自标识。

本程序属于准静态、几何分辨的阵列模型，不是整机模型或三维实体有限元模型；不计算动态回弹、表面损伤演化或柔性背板变形。[力学理论与推导](docs/共同背板柔顺爪刺阵列：力学理论与推导.md)涵盖的机制范围大于任一具体求解分支。

## 安装与使用

需要 Python 3.11+、NumPy 1.26+ 和 SciPy 1.16+。安装命令见上方英文部分；`test`、`plot`、`parquet` 分别提供测试、绘图和 Parquet 支持，`gpu-cuda13` 提供可选 CUDA 地形支持，需要兼容的 NVIDIA 驱动。接触求解通常使用 CPU。

部分材料配方需要 `data/raw/` 中的实测输入，原始数据不随源码分发，来源与许可见 `data/catalog/`。

```powershell
spine-sim --help
spine-terrain --help
spine-sim validate-env
spine-sim run-case examples/smoke_campaign.json --backend cpu --output results
```

`smoke_campaign.json` 只检查任务运行和结果存储；`canonical_campaign.json` 是早期求解器的解析夹具。其他示例与 `experiments/` 中的配置分别指定模型，不能直接互换。

准备好批次配置后：

```powershell
spine-sim run-campaign campaign.json --backend cpu --workers 2 --output results
spine-sim resume campaign.json --backend cpu --output results
spine-sim summarize results/<campaign_id>
```

专项运行和分析入口位于 `scripts/`，参数见各脚本的 `--help`。部分工具保留历史文件名和本机默认路径，复用时应显式指定输入输出。便携包中的 `START_CONFIRMATION.bat` 打开运行台，`TEST_ENVIRONMENT.bat` 检查环境；正式计算仍需在运行台手动启动。

## 数值与运行约定

- 内部使用 SI 单位。接触力是表面对针的作用力：抗拖力 `T = -sum(f_x)`，总预载 `P = sum(f_z)`，局部法向力 `N = f · n >= 0`。单针竖直分力允许为负。
- 总预载不意味着逐针均分载荷；实际分载由相容与平衡决定。共同背板模型的转角保持固定。
- 接近、预载和拖动连续继承已接受状态，包括弹簧压缩、接触法向、间隙和摩擦历史。预载换落点从接近阶段重新开始并保留失败记录；拖动阶段不换点、不拼接轨迹。
- 跑批与恢复均须显式启动，打开监控或读取结果不触发计算。新正式地形按配置使用 CUDA 生成，不可用时停止；已有输入可以直接复用。
- 同一批次保持材料参数、边界条件、容差和加载协议一致。暂停时停止新增派发，让在途算例自然结束。失败、部分完成和预算停止均保留真实终止原因。
- 不跨未解析区间补造连续轨迹，不把数值收敛当作物理承载能力。性能优化须保持方程和加载历史，仅残差一致不能证明完整轨迹等价。

## 目录与数据

`src/spine_sim/` 保存求解器、地形与运行框架；`examples/` 保存小型输入；`experiments/` 保存批次定义和构型表；`scripts/` 保存运行、打包和分析工具；`tests/` 保存测试；`docs/` 保留公共约定、地形、几何、单刺、阵列、空间杆及降阶接触模型的分块文档，`docs/共同背板柔顺爪刺阵列：力学理论与推导.md` 补充共同背板柔顺阵列的完整推导，原始多尺度机理文档也予以保留。

`monitor/dist/` 是运行台必需的页面文件；`data/catalog/` 保存外部输入来源；`provenance/` 保存参数注册表引用的输入来源记录。

计算结果、报告、地形缓存和临时文件不纳入版本控制。长期保存的输入数据应与可删除缓存分开；本地虚拟环境可继续复用，不属于仿真输出。

## 测试

```powershell
python -B -m pytest tests/test_balanced_contact.py -q -p no:cacheprovider
```

此命令检查降阶接触求解器及背板固定／自由边界。完整测试使用 `python -m pytest -q`。数值测试通过不代表模型已完成实验标定。
