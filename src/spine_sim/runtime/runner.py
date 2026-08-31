"""可恢复、支持 Windows ``spawn`` 多进程的 campaign runner。

runner 只负责任务隔离、资源记录和结果持久化，不解释物理参数；实际 case 逻辑由
``module.path:function`` 入口提供，并通过 :class:`RunContext` 获得冻结的版本信息。
"""

from __future__ import annotations

import importlib
import json
import multiprocessing
import platform
import sqlite3
import time
import traceback
import tracemalloc
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from spine_sim.core.config import BaseCaseSpec, CampaignSpec
from spine_sim.core.errors import ConfigurationError, classify_exception
from spine_sim.core.identity import stable_hash
from spine_sim.core.states import Event
from spine_sim.core.versions import (
    GEOMETRY_SCHEMA_VERSION,
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
    events: list[Event | Mapping[str, Any]] = field(default_factory=list)
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


def _activate_backend_device(backend: Mapping[str, Any]) -> None:
    """在 worker 中绑定已探测的 CUDA 设备；CPU 路径不做任何操作。"""

    if backend.get("selected") != "cuda":
        return
    if backend.get("cuda_provider") != "cupy":
        raise ConfigurationError("selected CUDA backend has no supported provider")
    device_index = backend.get("device_index")
    if isinstance(device_index, bool) or not isinstance(device_index, int):
        raise ConfigurationError("selected CUDA backend has an invalid device_index")
    import cupy  # type: ignore

    cupy.cuda.Device(device_index).use()


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
        _activate_backend_device(backend)
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
    except Exception as exc:
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


def _failed_case_result(
    case_id: str,
    exc: BaseException,
    *,
    wall_time_s: float = 0.0,
    peak_ram_bytes: int = 0,
    peak_python_bytes: int = 0,
    diagnostic_traceback: str | None = None,
) -> dict[str, Any]:
    """把 worker/调度/落盘异常转换成统一的逐 case 失败载荷。"""

    return {
        "ok": False,
        "case_id": case_id,
        "error": classify_exception(exc),
        "traceback": diagnostic_traceback or traceback.format_exc(),
        "wall_time_s": wall_time_s,
        "peak_ram_bytes": peak_ram_bytes,
        "peak_python_bytes": peak_python_bytes,
    }


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
        expected_versions = {
            "project_schema_version": PROJECT_SCHEMA_VERSION,
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "solver_semantics_version": SOLVER_SEMANTICS_VERSION,
            "geometry_version": GEOMETRY_SCHEMA_VERSION,
            "parameter_registry_version": PARAMETER_REGISTRY_VERSION,
        }
        for case in campaign.cases:
            mismatches = {
                name: (getattr(case, name), expected)
                for name, expected in expected_versions.items()
                if getattr(case, name) != expected
            }
            if mismatches:
                raise ConfigurationError(
                    f"case {case.case_id} uses incompatible semantic versions: "
                    f"{mismatches}"
                )
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

    def _lineage_document(self) -> dict[str, Any]:
        """构造当前 campaign 的完整逐 case 来源链。"""

        return {
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
        }

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
                "geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
                "parameter_registry_version": PARAMETER_REGISTRY_VERSION,
                "campaign_id": self.campaign.campaign_id,
                "created_at_utc": utc_now(),
                "backend": self._backend_record,
                "index_format": "pending",
            },
            raw_config=raw_config,
            normalized_config=normalized,
            lineage=self._lineage_document(),
        )

    @staticmethod
    def _backend_signature(value: Mapping[str, Any]) -> tuple[Any, ...]:
        """只比较影响数值执行的后端字段，忽略探测备注。"""

        selected = value.get("selected")
        if selected == "cpu":
            # 未选用的 CUDA capability 可能随机器变化，不影响 CPU 数值执行。
            return (selected, value.get("platform"))
        return (
            selected,
            value.get("cuda_provider"),
            value.get("device_index"),
            value.get("platform"),
        )

    def _has_persisted_records(self, missing_component: str) -> bool:
        """安全判断是否已有 case；允许 compact 建表前的无记录半初始化。"""

        try:
            return bool(self.store.list_records())
        except sqlite3.OperationalError as exc:
            if isinstance(self.store, CompactResultStore) and "no such table" in str(
                exc
            ):
                return False
            raise ConfigurationError(
                f"campaign {missing_component} is missing and persisted cases "
                "cannot be safely inspected"
            ) from exc
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as exc:
            raise ConfigurationError(
                f"campaign {missing_component} is missing and persisted cases "
                "cannot be safely inspected"
            ) from exc

    def prepare(self, raw_config: Mapping[str, Any]) -> None:
        """初始化或严格验证可恢复 campaign，必要时修复半初始化骨架。"""

        manifest_path = self.store.root / "manifest.json"
        if not manifest_path.is_file():
            if self._has_persisted_records("manifest"):
                raise ConfigurationError(
                    "campaign manifest is missing while persisted case records "
                    "exist; refusing an unsafe initialization"
                )
            self.initialize(raw_config)
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigurationError("campaign manifest is unreadable") from exc
        if not isinstance(manifest, Mapping):
            raise ConfigurationError("campaign manifest must be a JSON object")
        expected_manifest = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "solver_semantics_version": SOLVER_SEMANTICS_VERSION,
            "geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
            "parameter_registry_version": PARAMETER_REGISTRY_VERSION,
            "campaign_id": self.campaign.campaign_id,
        }
        mismatches = {
            name: (manifest.get(name), expected)
            for name, expected in expected_manifest.items()
            if manifest.get(name) != expected
        }
        expected_storage = (
            "sqlite_transactional_summary_v1"
            if isinstance(self.store, CompactResultStore)
            else None
        )
        if manifest.get("result_storage") != expected_storage:
            mismatches["result_storage"] = (
                manifest.get("result_storage"),
                expected_storage,
            )
        if mismatches:
            raise ConfigurationError(
                f"existing campaign is incompatible with this run: {mismatches}"
            )
        stored_backend = manifest.get("backend")
        if not isinstance(stored_backend, Mapping) or self._backend_signature(
            stored_backend
        ) != self._backend_signature(self._backend_record):
            raise ConfigurationError(
                "existing campaign backend does not match the selected backend"
            )
        # capability notes 不影响执行签名；恢复运行仍复用首次记录，避免新旧 case
        # summary 和 manifest 因探测备注变化而形成混合 provenance。
        self._backend_record = dict(stored_backend)
        lineage_path = self.store.root / "lineage.json"
        if lineage_path.is_file():
            try:
                stored_lineage = json.loads(
                    lineage_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ConfigurationError("campaign lineage is unreadable") from exc
            if stable_hash(stored_lineage) != stable_hash(
                self._lineage_document()
            ):
                raise ConfigurationError(
                    "existing campaign lineage/config hashes do not match"
                )
        else:
            if self._has_persisted_records("lineage"):
                # tags 等 config 字段不一定进入 case_id。若结果已经存在却丢失
                # lineage，静默写入当前 lineage 会让旧 case 冒充当前配置。
                raise ConfigurationError(
                    "campaign lineage is missing while persisted case records "
                    "exist; refusing an unsafe resume"
                )
        if not self.store.is_initialized():
            # initialize 是幂等的：补齐缺失骨架/SQLite 表，不删除已原子提交的 case。
            self.initialize(raw_config)

    def run(
        self,
        *,
        resume: bool = False,
        failed_only: bool = False,
        workers: int | None = None,
    ) -> list[CaseRecord]:
        """运行筛选后的 case，逐个原子提交结果，并重建 campaign 索引。"""

        selected: list[BaseCaseSpec] = []
        existing_by_id = {
            record.case_id: record
            for record in self.store.list_records()
        } if (resume or failed_only) else {}
        # list_records 已做完整性复核：只有真正完整的记录仍标为 complete。
        for case in self.campaign.cases:
            existing = existing_by_id.get(case.case_id)
            if resume:
                if existing is not None and existing.run_state == "complete":
                    continue
            if failed_only:
                # retry-failed 不补跑从未出现的 case；执行失败、畸形或附件损坏
                # 都由 list_records 归一为非 complete，因而会被精确重算。
                if existing is None or existing.run_state == "complete":
                    continue
            selected.append(case)

        worker_count = self.campaign.workers if workers is None else workers
        if worker_count < 1:
            raise ConfigurationError("workers must be at least one")
        if self._backend_record.get("selected") == "cuda" and worker_count > 1:
            raise ConfigurationError(
                "CUDA campaigns currently require workers=1 because workers "
                "do not yet receive distinct device assignments"
            )
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
                try:
                    self.store.write_case(
                        case_id=case.case_id,
                        config=asdict(case),
                        summary=summary,
                        arrays=output.arrays,
                        trace_rows=output.trace_rows,
                        events=output.events,
                        validation=output.validation,
                    )
                    return
                except Exception as exc:
                    # adapter 返回值已离开 worker 后仍可能在 JSON/NPZ/Parquet 规范化阶段
                    # 失败。把它降级为该 case 的 execution_error，避免主进程中止整批。
                    result = _failed_case_result(
                        case.case_id,
                        exc,
                        wall_time_s=result["wall_time_s"],
                        peak_ram_bytes=result["peak_ram_bytes"],
                        peak_python_bytes=result["peak_python_bytes"],
                    )
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
                exhausted = False

                def submit_next() -> None:
                    """补充一个任务；提交失败也只记为该 case 的执行错误。"""

                    nonlocal exhausted

                    try:
                        case = next(case_iterator)
                    except StopIteration:
                        exhausted = True
                        return
                    try:
                        future = executor.submit(
                            _execute,
                            self.campaign.callable,
                            case,
                            backend_record,
                            profile_python_memory,
                        )
                    except Exception as exc:
                        persist(_failed_case_result(case.case_id, exc))
                    else:
                        in_flight[future] = case

                while in_flight or not exhausted:
                    while not exhausted and len(in_flight) < maximum_in_flight:
                        submit_next()
                    if not in_flight:
                        # 例如进程池已经损坏，剩余 submit 会逐 case 失败并在上面的
                        # 填充循环中被记录；耗尽迭代器后即可正常结束 campaign。
                        continue
                    completed, _pending = wait(
                        in_flight, return_when=FIRST_COMPLETED
                    )
                    for future in completed:
                        case = in_flight.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = _failed_case_result(case.case_id, exc)
                        persist(result)

        expected_case_ids = {case.case_id for case in self.campaign.cases}
        records = []
        for record in self.store.list_records():
            if record.case_id not in expected_case_ids:
                record = replace(
                    record,
                    run_state="execution_error",
                    error_category="execution",
                    error_type="ResultIntegrityError",
                    error_message=(
                        "persisted case is not declared by the current campaign"
                    ),
                )
            records.append(record)
        # case 全部持久化后再生成轻量总索引、聚合事件与结果集哈希。
        index_format = write_campaign_index(self.store.root, records)
        self.store.rebuild_event_index(expected_case_ids)
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
