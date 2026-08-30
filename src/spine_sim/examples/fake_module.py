"""Deterministic fake module. It implements no terrain or contact physics."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from spine_sim.core.errors import ModelUnclosedError
from spine_sim.runtime.runner import CaseOutput, RunContext


def run_case(parameters: Mapping[str, Any], context: RunContext) -> CaseOutput:
    started = time.perf_counter()
    if parameters.get("fail"):
        raise RuntimeError(str(parameters.get("message", "requested fake failure")))
    if parameters.get("model_unclosed"):
        raise ModelUnclosedError("requested fake model closure failure")
    seed = int(parameters.get("seed", 0))
    samples = int(parameters.get("samples", 5))
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1e-3, samples)
    signal = rng.standard_normal(samples)
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
