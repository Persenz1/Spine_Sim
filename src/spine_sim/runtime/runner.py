"""可恢复、支持 Windows ``spawn`` 多进程的 campaign runner。

runner 只负责任务隔离、资源记录和结果持久化，不解释物理参数；实际 case 逻辑由
``module.path:function`` 入口提供，并通过 :class:`RunContext` 获得冻结的版本信息。
"""

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
    """case 入口返回给 runner 的标准内存载荷。"""

    summary: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    trace_rows: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    stage_times_s: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RunContext:
    """runner 传给 case 的 identity、后端和语义版本快照。"""

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
    """解析 ``模块:函数`` 引用，并确认目标可调用。"""

    module_name, function_name = reference.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{reference} is not callable")
    return function


def _peak_rss_bytes() -> int:
    """跨平台读取进程峰值常驻内存；平台不支持时返回 0。"""

    if platform.system() == "Windows":
        # Windows 没有 resource.getrusage，直接查询当前进程的工作集峰值。
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                """与 Win32 PROCESS_MEMORY_COUNTERS 二进制布局对应。"""

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
    """在一个 worker 内执行 case，并把成功或异常统一包装为可持久化结果。"""

    started = time.perf_counter()
    if profile_python_memory:
        tracemalloc.start()
    try:
        # worker 内重新导入入口，避免向 spawn 子进程传递不可 pickle 的函数对象。
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
    """选择待运行 case、协调 worker，并维护 campaign 级索引。"""

    def __init__(
        self,
        campaign: CampaignSpec,
        campaign_dir: str | Path,
        backend: BackendCapabilities,
    ):
        """根据 campaign 模式选择逐文件或 SQLite 摘要存储。"""

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
        # 大型 formal campaign 若明确只要 summary，就避免为每个 case 建多个小文件。
        self._backend_record = backend.as_dict()

    def initialize(self, raw_config: Mapping[str, Any]) -> None:
        """首次创建 manifest、原始/规范配置和完整 case 来源链。"""

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
        """运行筛选后的 case，逐个原子提交结果，并重建 campaign 索引。"""

        selected: list[BaseCaseSpec] = []
        # resume 依据完整性复核跳过；failed_only 只重试已有执行错误，不补跑缺失 case。
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
            """把 worker 返回值转换成成功或执行错误 summary 后立即落盘。"""

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
            # 单 worker 直接在当前进程运行，便于调试且避免不必要的进程启动成本。
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
            # 只保留少量在途 Future。正式 campaign 可能有数千个大结果，若一次提交
            # 全部任务，协调进程会长期持有 Future 和返回载荷，抵消逐 case 落盘的收益。
            context = multiprocessing.get_context("spawn")
            case_iterator = iter(selected)
            maximum_in_flight = 2 * worker_count
            with ProcessPoolExecutor(
                max_workers=worker_count, mp_context=context
            ) as executor:
                in_flight: dict[Any, BaseCaseSpec] = {}

                def submit_next() -> bool:
                    """补充一个任务到有界在途队列；迭代器耗尽时返回 False。"""

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
        # case 全部持久化后再生成轻量总索引、聚合事件与结果集哈希。
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
