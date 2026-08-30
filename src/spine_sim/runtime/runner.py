"""Spawn-safe runner for independent cases and recoverable campaigns."""

from __future__ import annotations

import importlib
import multiprocessing
import platform
import time
import traceback
import tracemalloc
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from spine_sim.core.config import BaseCaseSpec, CampaignSpec
from spine_sim.core.errors import ConfigurationError, classify_exception
from spine_sim.core.identity import stable_hash
from spine_sim.core.versions import (
    MODEL_SCHEMA_VERSION,
    PARAMETER_REGISTRY_VERSION,
    PROJECT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    SOLVER_SEMANTICS_VERSION,
)
from spine_sim.io.files import utc_now
from spine_sim.io.results import (
    CaseRecord,
    CompactResultStore,
    ResultStore,
    update_manifest,
    write_campaign_index,
)
from spine_sim.runtime.backend import BackendCapabilities


@dataclass
class CaseOutput:
    summary: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    trace_rows: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    stage_times_s: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RunContext:
    case_id: str
    backend: Mapping[str, Any]
    normalized_input_hash: str
    project_schema_version: str
    model_schema_version: str
    result_schema_version: str
    solver_semantics_version: str
    terrain_version: str
    geometry_version: str
    parameter_registry_version: str


def _load_callable(
    reference: str,
) -> Callable[[Mapping[str, Any], RunContext], CaseOutput]:
    module_name, function_name = reference.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{reference} is not callable")
    return function


def _peak_rss_bytes() -> int:
    if platform.system() == "Windows":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            return int(counters.PeakWorkingSetSize) if ok else 0
        except Exception:
            return 0
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if platform.system() == "Darwin" else peak * 1024
    except (ImportError, OSError):
        return 0


def _execute(
    reference: str,
    case: BaseCaseSpec,
    backend: Mapping[str, Any],
    profile_python_memory: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    if profile_python_memory:
        tracemalloc.start()
    try:
        function = _load_callable(reference)
        context = RunContext(
            case_id=case.case_id,
            backend=backend,
            normalized_input_hash=case.normalized_input_hash,
            project_schema_version=case.project_schema_version,
            model_schema_version=case.model_schema_version,
            result_schema_version=case.result_schema_version,
            solver_semantics_version=case.solver_semantics_version,
            terrain_version=case.terrain_version,
            geometry_version=case.geometry_version,
            parameter_registry_version=case.parameter_registry_version,
        )
        output = function(case.parameters, context)
        if not isinstance(output, CaseOutput):
            raise TypeError("case callable must return CaseOutput")
        peak = tracemalloc.get_traced_memory()[1] if profile_python_memory else 0
        return {
            "ok": True,
            "case_id": case.case_id,
            "output": output,
            "wall_time_s": time.perf_counter() - started,
            "peak_ram_bytes": _peak_rss_bytes(),
            "peak_python_bytes": peak,
        }
    except BaseException as exc:
        peak = tracemalloc.get_traced_memory()[1] if profile_python_memory else 0
        return {
            "ok": False,
            "case_id": case.case_id,
            "error": classify_exception(exc),
            "traceback": traceback.format_exc(),
            "wall_time_s": time.perf_counter() - started,
            "peak_ram_bytes": _peak_rss_bytes(),
            "peak_python_bytes": peak,
        }
    finally:
        if profile_python_memory:
            tracemalloc.stop()


class CampaignRunner:
    def __init__(
        self,
        campaign: CampaignSpec,
        campaign_dir: str | Path,
        backend: BackendCapabilities,
    ):
        self.campaign = campaign
        summary_only_formal = (
            campaign.mode == "formal"
            and all(
                case.parameters.get("output", {}).get("level")
                == "summary"
                for case in campaign.cases
            )
        )
        self.store = (
            CompactResultStore(campaign_dir)
            if summary_only_formal
            else ResultStore(campaign_dir)
        )
        self._backend_record = backend.as_dict()

    def initialize(self, raw_config: Mapping[str, Any]) -> None:
        normalized = {
            "campaign": asdict(self.campaign),
            "campaign_id": self.campaign.campaign_id,
        }
        self.store.initialize(
            manifest={
                "schema_version": PROJECT_SCHEMA_VERSION,
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "solver_semantics_version": SOLVER_SEMANTICS_VERSION,
                "parameter_registry_version": PARAMETER_REGISTRY_VERSION,
                "campaign_id": self.campaign.campaign_id,
                "created_at_utc": utc_now(),
                "backend": self._backend_record,
                "index_format": "pending",
            },
            raw_config=raw_config,
            normalized_config=normalized,
            lineage={
                "campaign_id": self.campaign.campaign_id,
                "case_lineage": {
                    case.case_id: {
                        "module": case.module,
                        "module_version": case.module_version,
                        "config_hash": case.config_hash,
                        "upstream_hash": case.upstream_hash,
                        "project_schema_version": case.project_schema_version,
                        "model_schema_version": case.model_schema_version,
                        "result_schema_version": case.result_schema_version,
                        "solver_semantics_version": case.solver_semantics_version,
                        "terrain_version": case.terrain_version,
                        "geometry_version": case.geometry_version,
                        "parameter_registry_version": (
                            case.parameter_registry_version
                        ),
                        "normalized_input_hash": case.normalized_input_hash,
                    }
                    for case in self.campaign.cases
                },
            },
        )

    def run(
        self,
        *,
        resume: bool = False,
        failed_only: bool = False,
        workers: int | None = None,
    ) -> list[CaseRecord]:
        selected: list[BaseCaseSpec] = []
        for case in self.campaign.cases:
            if resume and self.store.is_complete(case.case_id):
                continue
            if failed_only:
                try:
                    if self.store.load_case_summary(case.case_id).get("run_state") != "execution_error":
                        continue
                except (FileNotFoundError, ValueError):
                    continue
            selected.append(case)

        worker_count = self.campaign.workers if workers is None else workers
        if worker_count < 1:
            raise ConfigurationError("workers must be at least one")
        backend_record = self._backend_record
        selected_by_id = {case.case_id: case for case in selected}
        profile_python_memory = self.campaign.mode != "formal"

        def persist(result: dict[str, Any]) -> None:
            case = selected_by_id[result["case_id"]]
            if result["ok"]:
                output: CaseOutput = result["output"]
                summary = dict(output.summary)
                summary.update(
                    {
                        "run_state": "complete",
                        "wall_time_s": result["wall_time_s"],
                        "peak_ram_bytes": result["peak_ram_bytes"],
                        "peak_python_bytes": result["peak_python_bytes"],
                        "peak_vram_bytes": None,
                        "stage_times_s": output.stage_times_s,
                        "backend": backend_record,
                    }
                )
                self.store.write_case(
                    case_id=case.case_id,
                    config=asdict(case),
                    summary=summary,
                    arrays=output.arrays,
                    trace_rows=output.trace_rows,
                    events=output.events,
                    validation=output.validation,
                )
            else:
                self.store.write_case(
                    case_id=case.case_id,
                    config=asdict(case),
                    summary={
                        "run_state": "execution_error",
                        "wall_time_s": result["wall_time_s"],
                        "peak_ram_bytes": result["peak_ram_bytes"],
                        "peak_python_bytes": result["peak_python_bytes"],
                        "peak_vram_bytes": None,
                        "error": result["error"],
                        "diagnostic_traceback": result["traceback"],
                        "backend": backend_record,
                    },
                )

        if worker_count == 1:
            for case in selected:
                persist(
                    _execute(
                        self.campaign.callable,
                        case,
                        backend_record,
                        profile_python_memory,
                    )
                )
        elif selected:
            # Keep only a small multiple of the worker count in flight. A formal
            # campaign can contain thousands of array-heavy results, so retaining
            # every Future until the campaign ends defeats per-case atomic writes
            # and can exhaust the coordinator process.
            context = multiprocessing.get_context("spawn")
            case_iterator = iter(selected)
            maximum_in_flight = 2 * worker_count
            with ProcessPoolExecutor(
                max_workers=worker_count, mp_context=context
            ) as executor:
                in_flight: dict[Any, BaseCaseSpec] = {}

                def submit_next() -> bool:
                    try:
                        case = next(case_iterator)
                    except StopIteration:
                        return False
                    future = executor.submit(
                        _execute,
                        self.campaign.callable,
                        case,
                        backend_record,
                        profile_python_memory,
                    )
                    in_flight[future] = case
                    return True

                for _ in range(min(maximum_in_flight, len(selected))):
                    submit_next()
                while in_flight:
                    completed, _pending = wait(
                        in_flight, return_when=FIRST_COMPLETED
                    )
                    for future in completed:
                        in_flight.pop(future)
                        persist(future.result())
                        submit_next()

        records = self.store.list_records()
        index_format = write_campaign_index(self.store.root, records)
        self.store.rebuild_event_index()
        counts: dict[str, int] = {}
        for record in records:
            counts[record.run_state] = counts.get(record.run_state, 0) + 1
        update_manifest(
            self.store.root,
            updated_at_utc=utc_now(),
            index_format=index_format,
            status_counts=counts,
            result_set_hash=stable_hash(
                [(record.case_id, record.run_state, record.result_hash) for record in records]
            ),
        )
        return records
