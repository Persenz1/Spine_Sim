"""确定性的 M0 冒烟模块；不实现任何地形或接触物理。

该模块供运行器、缓存、错误处理和结果文件测试使用。随机信号只验证种子复现，
绝不能作为物理仿真结果解读。
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from spine_sim.core.errors import ModelUnclosedError
from spine_sim.runtime.runner import CaseOutput, RunContext


def run_case(parameters: Mapping[str, Any], context: RunContext) -> CaseOutput:
    """按输入开关制造成功或失败案例，并返回可复现的伪结果。"""

    started = time.perf_counter()
    # 两个显式开关分别覆盖普通运行错误和“模型尚未闭合”错误路径。
    if parameters.get("fail"):
        raise RuntimeError(str(parameters.get("message", "requested fake failure")))
    if parameters.get("model_unclosed"):
        raise ModelUnclosedError("requested fake model closure failure")
    seed = int(parameters.get("seed", 0))
    samples = int(parameters.get("samples", 5))
    # 使用局部生成器，避免污染进程级 NumPy 随机状态。
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1e-3, samples)
    signal = rng.standard_normal(samples)
    # 摘要模式省略数组，验证运行器可按输出等级减少持久化体积。
    summary_only = parameters.get("output", {}).get("level") == "summary"
    return CaseOutput(
        summary={
            "physical_state": "free",
            "numerical_state": "converged",
            "model_state": "covered",
            "sample_count": samples,
            "signal_sum": float(signal.sum()),
            "selected_backend": context.backend["selected"],
        },
        arrays=(
            {}
            if summary_only
            else {"path_position_m": x, "diagnostic_signal": signal}
        ),
        events=[],
        validation={"passed": True, "note": "M0 smoke only; no physics evaluated"},
        stage_times_s={"fake_compute": time.perf_counter() - started},
    )
