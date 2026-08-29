# M0 未闭合问题

1. 仓库缺少 M0 提示词引用的 `04_爪刺仿真器_失败问题总结与重启约束.md`。现有执行版已覆盖关键禁用语义；文件补齐后仍需差异审查。
2. 未在 Linux 实机执行测试；实现固定使用跨平台 `pathlib` 和 `spawn`，建议后续 CI 加 Windows/Linux 双平台。
3. `peak_ram_bytes` 已记录进程峰值 RSS/工作集，`peak_python_bytes` 另记 Python 分配峰值；`peak_vram_bytes` 未测时为 `null`。M1 若启用 CUDA，应由其后端补充可信 VRAM 采样。
4. canonical single/array 的公共 dataclass 与结果 schema 已冻结；campaign adapter 仍以映射接收项目特定参数，调用方必须提供来源和单位，不得把解析 smoke catalog 当作硬件默认配置。

这些问题不阻塞 M1 开始实现公共地形配置、ID、文件和后端接口，但第 1 项应在文件可用时尽快关闭。
