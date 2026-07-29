# M0 测试报告

执行日期：2026-07-27
环境：Windows，Python 3.12，NumPy 2.3，PyArrow 25，CPU-only 权威路径

## 自动测试

运行命令：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s tests -v
```

覆盖与风险关系：

| 风险 | 验证 |
|---|---|
| 隐式单位、错误量纲、非法范围 | SI 数值、mm/µm/deg 转换、量纲错配和非正区域拒绝 |
| 坐标或力矩符号错误 | 90° 旋转手算、`P→O × F` 参考点搬移手算、非法旋转矩阵拒绝 |
| YAML/JSON 排版或参数次序改变身份 | 规范字典排序、同输入同 ID、版本变化新 ID |
| 状态维度混用 | 错把 `converged` 放入物理维度和缺字段均拒绝 |
| 中断污染完整结果 | 临时文件原子替换、损坏摘要识别为 incomplete、最后写 `COMPLETE` |
| 平台路径硬编码 | `pathlib` 相对路径解析测试；源代码无盘符或分隔符拼接 |
| 并行导致非确定结果 | 单 worker 与 Windows `spawn` 两 worker 的内容 hash 完全相同 |
| 单 case 异常破坏 campaign | 三 case 中一个故障，两个仍完成并保存 |
| 恢复重复覆盖 | `resume` 后完整 case 摘要 mtime 不变 |
| 无 GPU 无法运行 | 强制 CPU 的后端发现和完整 runner 测试 |
| GPU 能力未显式传递 | 可控能力探针验证 `cuda_available/selected/provider` 记录 |
| 最小模块无法保存恢复 | 确定性假模块写 summary/NPZ/validation/index 并执行恢复 |

假模块明确只测试 M0，不生成地形、不计算接触或力学。

自动测试共 16 项，全部通过。另执行 CLI `validate-env`、2-worker campaign、resume、summarize 和 Parquet 回读 smoke；2 行索引及两个完成标记均通过检查。

## 验收边界

- Windows `spawn` 已实际执行。
- Linux 使用相同显式 `spawn` 上下文和纯 `pathlib` 路径；当前机器未做 Linux 实机 CI，因此 Linux 结论是代码路径覆盖，不是实机声明。
- 已实际生成并用 PyArrow 回读 `cases.parquet`。若另一环境无 `pyarrow`，CPU-only 路径会写 `cases.jsonl` 并在 manifest 标记；正式 campaign 仍需安装 `.[parquet]`。
- 峰值 RAM 使用 Windows PeakWorkingSetSize 或 Unix `ru_maxrss`；另保留 Python 分配峰值。峰值 VRAM 未测时为 `null`。
