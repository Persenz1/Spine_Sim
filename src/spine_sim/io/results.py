"""逐 case 原子结果存储与轻量 campaign 索引。

small/含轨迹任务使用目录式 :class:`ResultStore`；只保存摘要的大型 formal 任务使用
事务式 :class:`CompactResultStore`。两者共享结果哈希语义和读取接口。
"""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from spine_sim.core.identity import canonical_json, canonicalize, stable_hash
from spine_sim.core.states import Event
from spine_sim.io.files import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    utc_now,
)


def atomic_write_trace_table(
    directory: Path, rows: Iterable[Mapping[str, Any]]
) -> tuple[str | None, str | None, str | None]:
    """原子写入规范轨迹；有 PyArrow 时用 Parquet，否则退化为 JSONL。"""

    normalized = [canonicalize(dict(row)) for row in rows]
    # 每次写入只保留一种轨迹格式，防止读取端命中上一次运行留下的旧文件。
    for name in ("trace.parquet", "trace.jsonl"):
        stale = directory / name
        if stale.exists():
            stale.unlink()
    if not normalized:
        return None, None, None
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        target = directory / "trace.jsonl"
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in normalized
        )
        atomic_write_bytes(target, payload.encode("utf-8"))
        return target.name, "jsonl_fallback", sha256_file(target)

    columns = sorted({key for row in normalized for key in row})
    # Parquet 不能无损表达任意嵌套列和全空列；把这些列编码为规范 JSON 字符串，
    # 并在 schema metadata 中记录，读取时可精确还原原始 Python 结构。
    json_columns = {
        key
        for key in columns
        if any(isinstance(row.get(key), (dict, list)) for row in normalized)
        or all(row.get(key) is None for row in normalized)
    }
    parquet_rows = [
        {
            key: (
                canonical_json(row.get(key))
                if key in json_columns
                else row.get(key)
            )
            for key in columns
        }
        for row in normalized
    ]
    table = pa.Table.from_pylist(parquet_rows)
    metadata = dict(table.schema.metadata or {})
    metadata[b"spine_sim_trace_encoding"] = b"canonical-json-columns-v1"
    metadata[b"spine_sim_json_columns"] = json.dumps(
        sorted(json_columns), separators=(",", ":")
    ).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    target = directory / "trace.parquet"
    descriptor, temporary = tempfile.mkstemp(
        prefix=".trace.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        pq.write_table(table, str(temporary_path))
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                # Preserve the original Parquet error on Windows if Arrow still
                # owns a failed writer handle; a future atomic write uses a new
                # unique temporary name.
                pass
    return target.name, "parquet", sha256_file(target)


def read_trace_table(path: str | Path) -> list[dict[str, Any]]:
    """读取规范轨迹，并反解 Parquet 中无损保存的 JSON 列。"""

    source = Path(path)
    if source.is_dir():
        parquet = source / "trace.parquet"
        jsonl = source / "trace.jsonl"
        source = parquet if parquet.is_file() else jsonl
    if source.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line
        ]
    if source.suffix != ".parquet":
        raise ValueError("trace path must be a case directory, .parquet, or .jsonl")
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read a Parquet trace") from exc
    table = pq.read_table(source)
    metadata = table.schema.metadata or {}
    if metadata.get(b"spine_sim_trace_encoding") != b"canonical-json-columns-v1":
        raise ValueError("unsupported or missing canonical trace encoding")
    encoded_json_columns = metadata.get(b"spine_sim_json_columns")
    if encoded_json_columns is None:
        raise ValueError("missing canonical trace JSON-column metadata")
    json_columns = set(json.loads(encoded_json_columns))
    rows = table.to_pylist()
    for row in rows:
        for key in json_columns:
            if key in row:
                row[key] = json.loads(row[key])
    return rows


_RESULT_HASH_EXCLUDED_SUMMARY_FIELDS = {
    # 运行耗时、内存和完成时间不改变物理结果；排除后，同一计算可得到稳定结果哈希。
    "completed_at_utc",
    "wall_time_s",
    "peak_ram_bytes",
    "peak_python_bytes",
    "peak_vram_bytes",
    "stage_times_s",
    "diagnostic_traceback",
    "result_hash",
    "path_sha256",
}


def _result_hash(
    config: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    event_lines: list[str],
    path_sha256: str | None,
) -> str:
    """联合配置、稳定 summary、事件和附件摘要计算结果 identity。"""

    stable_summary = {
        key: value
        for key, value in document.items()
        if key not in _RESULT_HASH_EXCLUDED_SUMMARY_FIELDS
    }
    return stable_hash(
        {
            "config": config,
            "summary": stable_summary,
            "events": event_lines,
            "events_sha256": document.get("events_sha256"),
            "path_sha256": path_sha256,
            "trace_file": document.get("trace_file"),
            "trace_format": document.get("trace_format"),
            "trace_sha256": document.get("trace_sha256"),
        }
    )


@dataclass(frozen=True)
class CaseRecord:
    """campaign 索引中每个 case 的最小查询记录。"""

    case_id: str
    run_state: str
    result_hash: str
    wall_time_s: float
    peak_ram_bytes: int
    peak_python_bytes: int
    error_category: str | None = None
    error_type: str | None = None
    error_message: str | None = None


def _case_record(document: Mapping[str, Any]) -> CaseRecord:
    """从完整 summary 提取索引字段，并展开可选错误信息。"""

    if "error" in document:
        error = document["error"]
        if not isinstance(error, Mapping):
            raise TypeError("case error must be a mapping")
        error_category = str(error["category"])
        error_type = str(error["type"])
        error_message = str(error["message"])
    else:
        error_category = None
        error_type = None
        error_message = None
    return CaseRecord(
        case_id=str(document["case_id"]),
        result_hash=str(document["result_hash"]),
        run_state=str(document["run_state"]),
        wall_time_s=float(document["wall_time_s"]),
        peak_ram_bytes=int(document["peak_ram_bytes"]),
        peak_python_bytes=int(document["peak_python_bytes"]),
        error_category=error_category,
        error_type=error_type,
        error_message=error_message,
    )


def _integrity_error_record(
    case_id: str,
    message: str,
    record: CaseRecord | None = None,
) -> CaseRecord:
    """把损坏/不完整的持久化记录显式暴露为非成功状态。"""

    if record is None:
        return CaseRecord(
            case_id=case_id,
            run_state="execution_error",
            result_hash="",
            wall_time_s=0.0,
            peak_ram_bytes=0,
            peak_python_bytes=0,
            error_category="execution",
            error_type="ResultIntegrityError",
            error_message=message,
        )
    return replace(
        record,
        case_id=case_id,
        run_state="execution_error",
        error_category="execution",
        error_type="ResultIntegrityError",
        error_message=message,
    )


def write_campaign_index(
    root: str | Path, records: Iterable[CaseRecord]
) -> str:
    """按 case ID 排序写入 Parquet 索引；缺少 PyArrow 时写 JSONL。"""

    campaign_root = Path(root).resolve()
    rows = [
        asdict(record)
        for record in sorted(records, key=lambda row: row.case_id)
    ]
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        table = pa.Table.from_pylist(rows)
        target = campaign_root / "cases.parquet"
        descriptor, temporary = tempfile.mkstemp(
            prefix=".cases.", suffix=".tmp", dir=campaign_root
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            pq.write_table(table, temporary_path)
            os.replace(temporary_path, target)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        fallback = campaign_root / "cases.jsonl"
        if fallback.exists():
            fallback.unlink()
        return "parquet"
    except ImportError:
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        )
        atomic_write_bytes(
            campaign_root / "cases.jsonl", payload.encode("utf-8")
        )
        return "jsonl_fallback"


def update_manifest(root: str | Path, **fields: Any) -> None:
    """合并并原子重写 campaign manifest 的指定字段。"""

    path = Path(root).resolve() / "manifest.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    current.update(fields)
    atomic_write_json(path, current)


class ResultStore:
    """适用于数组、轨迹和事件输出的逐 case 目录存储。"""

    def __init__(self, campaign_dir: str | Path):
        """解析 campaign 根目录及 ``paths`` 子目录。"""

        self.root = Path(campaign_dir).resolve()
        self.cases_dir = self.root / "paths"

    def initialize(
        self,
        *,
        manifest: Mapping[str, Any],
        raw_config: Mapping[str, Any],
        normalized_config: Mapping[str, Any],
        lineage: Mapping[str, Any],
    ) -> None:
        """建立 campaign 目录骨架并写入配置、来源链和空聚合文件。"""

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config").mkdir(exist_ok=True)
        self.cases_dir.mkdir(exist_ok=True)
        atomic_write_json(self.root / "config" / "original.json", raw_config)
        atomic_write_json(self.root / "config" / "normalized.json", normalized_config)
        atomic_write_json(self.root / "lineage.json", lineage)
        if not (self.root / "events.jsonl").exists():
            atomic_write_bytes(self.root / "events.jsonl", b"")
        if not (self.root / "validation.json").exists():
            atomic_write_json(self.root / "validation.json", {"status": "not_run", "checks": []})
        # manifest 是初始化完成标记；最后发布，进程中断时下次调用可安全重建骨架。
        atomic_write_json(self.root / "manifest.json", manifest)

    def is_initialized(self) -> bool:
        """检查目录式 store 的初始化骨架是否完整。"""

        required = (
            self.root / "manifest.json",
            self.root / "config" / "original.json",
            self.root / "config" / "normalized.json",
            self.root / "lineage.json",
            self.root / "events.jsonl",
            self.root / "validation.json",
        )
        return self.cases_dir.is_dir() and all(path.is_file() for path in required)

    def case_dir(self, case_id: str) -> Path:
        """返回受限于 ``paths`` 下的 case 目录，拒绝路径分隔符。"""

        if not case_id or any(char in case_id for char in "\\/:"):
            raise ValueError("invalid case_id")
        return self.cases_dir / case_id

    def is_complete(self, case_id: str) -> bool:
        """复核 marker、summary、所有附件哈希及最终结果哈希。"""

        directory = self.case_dir(case_id)
        marker = directory / "COMPLETE"
        summary = directory / "summary.json"
        if not marker.is_file() or not summary.is_file():
            return False
        try:
            # COMPLETE 只是一项证据；继续校验路径数组、轨迹、事件、配置和 validation。
            document = json.loads(summary.read_text(encoding="utf-8"))
            if not isinstance(document, Mapping):
                return False
            marker_hash = marker.read_text(encoding="ascii").strip()
            if (
                document.get("case_id") != case_id
                or document.get("run_state") != "complete"
                or marker_hash != document.get("result_hash")
            ):
                return False
            path_sha256 = document.get("path_sha256")
            path_file = directory / "path.npz"
            if path_sha256 is None:
                path_valid = not path_file.exists()
            else:
                path_valid = (
                    path_file.is_file()
                    and sha256_file(path_file) == path_sha256
                )
            if not path_valid:
                return False
            trace_file_name = document.get("trace_file")
            trace_sha256 = document.get("trace_sha256")
            if trace_file_name is None:
                trace_valid = trace_sha256 is None
            else:
                trace_file = directory / str(trace_file_name)
                trace_valid = (
                    trace_file.name in {"trace.parquet", "trace.jsonl"}
                    and trace_file.is_file()
                    and trace_sha256 is not None
                    and sha256_file(trace_file) == trace_sha256
                )
            if not trace_valid:
                return False
            event_file = directory / "events.jsonl"
            if "events_sha256" not in document:
                return False
            events_sha256 = document["events_sha256"]
            if events_sha256 is None:
                if event_file.exists():
                    return False
                event_lines: list[str] = []
            else:
                if (
                    not event_file.is_file()
                    or sha256_file(event_file) != events_sha256
                ):
                    return False
                event_lines = [
                    line
                    for line in event_file.read_text(encoding="utf-8").splitlines()
                    if line
                ]
            config_file = directory / "config.json"
            if not config_file.is_file():
                return False
            config = json.loads(config_file.read_text(encoding="utf-8"))
            if not isinstance(config, Mapping):
                return False
            validation_file = directory / "validation.json"
            if (
                "validation_sha256" not in document
                or not validation_file.is_file()
            ):
                return False
            validation_document = json.loads(
                validation_file.read_text(encoding="utf-8")
            )
            if (
                not isinstance(validation_document, Mapping)
                or document["validation_sha256"]
                != stable_hash(validation_document)
            ):
                return False
            return marker_hash == _result_hash(
                config,
                document,
                event_lines=event_lines,
                path_sha256=path_sha256,
            )
        except (OSError, TypeError, ValueError):
            return False

    def write_case(
        self,
        *,
        case_id: str,
        config: Mapping[str, Any],
        summary: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray] | None = None,
        trace_rows: Iterable[Mapping[str, Any]] = (),
        events: Iterable[Event | Mapping[str, Any]] = (),
        validation: Mapping[str, Any] | None = None,
    ) -> CaseRecord:
        """写入一个 case；仅在全部内容完成后发布 ``COMPLETE`` marker。"""

        directory = self.case_dir(case_id)
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "COMPLETE"
        # 先撤销旧完成标记，避免覆盖过程中被 resume 误判为完整。
        if marker.exists():
            marker.unlink()
        atomic_write_json(directory / "config.json", config)
        path_file = directory / "path.npz"
        if arrays:
            atomic_write_npz(path_file, arrays)
            path_sha256 = sha256_file(path_file)
        else:
            path_sha256 = None
            if path_file.exists():
                path_file.unlink()
        trace_file, trace_format, trace_sha256 = atomic_write_trace_table(
            directory, trace_rows
        )
        event_lines = []
        for event in events:
            value = event.as_dict() if isinstance(event, Event) else dict(event)
            event_lines.append(
                json.dumps(
                    canonicalize(value), ensure_ascii=False, sort_keys=True
                )
            )
        event_file = directory / "events.jsonl"
        if event_lines:
            atomic_write_bytes(
                event_file,
                (("\n".join(event_lines) + "\n").encode("utf-8")),
            )
            events_sha256 = sha256_file(event_file)
        elif event_file.exists():
            event_file.unlink()
            events_sha256 = None
        else:
            events_sha256 = None
        validation_document = dict(validation or {})
        atomic_write_json(
            directory / "validation.json", validation_document
        )
        document = dict(summary)
        document["case_id"] = case_id
        document["completed_at_utc"] = utc_now()
        document["events_sha256"] = events_sha256
        document["trace_file"] = trace_file
        document["trace_format"] = trace_format
        document["trace_sha256"] = trace_sha256
        document["validation_sha256"] = stable_hash(validation_document)
        document["result_hash"] = _result_hash(
            config,
            document,
            event_lines=event_lines,
            path_sha256=path_sha256,
        )
        document["path_sha256"] = path_sha256
        record = _case_record(document)
        atomic_write_json(directory / "summary.json", document)
        if record.run_state == "complete":
            # marker 最后写入，并保存 result_hash 以绑定当前 summary 和附件集合。
            atomic_write_bytes(
                marker, (document["result_hash"] + "\n").encode("ascii")
            )
        return record

    def load_case_summary(self, case_id: str) -> dict[str, Any]:
        """读取一个 case 的 ``summary.json``。"""

        path = self.case_dir(case_id) / "summary.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def list_records(self) -> list[CaseRecord]:
        """扫描记录，并把声称完成但未通过附件复核的 case 标为损坏。"""

        records: list[CaseRecord] = []
        if not self.cases_dir.exists():
            return records
        for directory in sorted(self.cases_dir.iterdir()):
            if not directory.is_dir():
                continue
            summary_path = directory / "summary.json"
            if not summary_path.is_file():
                records.append(
                    _integrity_error_record(
                        directory.name, "case directory has no summary.json"
                    )
                )
                continue
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                if not isinstance(data, Mapping):
                    raise TypeError("case summary must be a mapping")
                record = _case_record(data)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                records.append(
                    _integrity_error_record(
                        directory.name,
                        f"case summary is unreadable or malformed: {type(exc).__name__}",
                    )
                )
                continue
            if record.case_id != directory.name:
                record = _integrity_error_record(
                    directory.name,
                    "summary case_id does not match its directory",
                    record,
                )
            elif record.run_state == "complete" and not self.is_complete(
                directory.name
            ):
                record = _integrity_error_record(
                    directory.name,
                    "complete case failed marker/attachment/hash verification",
                    record,
                )
            elif record.run_state not in {"complete", "execution_error"}:
                record = _integrity_error_record(
                    directory.name,
                    f"unsupported persisted run_state: {record.run_state!r}",
                    record,
                )
            records.append(record)
        return records

    def rebuild_event_index(
        self, allowed_case_ids: set[str] | None = None
    ) -> None:
        """只汇总完整性复核通过的 case 事件，并补上所属 case ID。"""

        lines: list[str] = []
        if self.cases_dir.exists():
            for directory in sorted(self.cases_dir.iterdir()):
                if (
                    not directory.is_dir()
                    or (
                        allowed_case_ids is not None
                        and directory.name not in allowed_case_ids
                    )
                    or not self.is_complete(directory.name)
                ):
                    continue
                source = directory / "events.jsonl"
                if source.is_file():
                    for line in source.read_text(encoding="utf-8").splitlines():
                        if line:
                            event = json.loads(line)
                            event.setdefault("case_id", directory.name)
                            lines.append(
                                json.dumps(event, ensure_ascii=False, sort_keys=True)
                            )
        atomic_write_bytes(
            self.root / "events.jsonl",
            (("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")),
        )

class CompactResultStore:
    """大型 formal 分片使用的事务式纯摘要存储。

    formal 分片可能包含数千个 case。若每个摘要都建立四五个小文件，文件系统元数据
    开销会超过内容本身。本类保留与 :class:`ResultStore` 相同的结果哈希语义，但把
    每个摘要原子提交到 SQLite；需要数组、轨迹或事件的任务仍使用目录式存储。
    """

    DATABASE_NAME = "case_summaries.sqlite3"

    def __init__(self, campaign_dir: str | Path):
        """解析 campaign 根目录和固定 SQLite 数据库路径。"""

        self.root = Path(campaign_dir).resolve()
        self.database_path = self.root / self.DATABASE_NAME

    @contextmanager
    def _connect(self) -> Iterable[sqlite3.Connection]:
        """打开 WAL/FULL 连接，并在上下文退出时提交或回滚事务。"""

        connection = sqlite3.connect(self.database_path, timeout=60.0)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(
        self,
        *,
        manifest: Mapping[str, Any],
        raw_config: Mapping[str, Any],
        normalized_config: Mapping[str, Any],
        lineage: Mapping[str, Any],
    ) -> None:
        """写入紧凑 manifest/config，并初始化摘要表。"""

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config").mkdir(exist_ok=True)
        compact_manifest = dict(manifest)
        compact_manifest["result_storage"] = (
            "sqlite_transactional_summary_v1"
        )
        atomic_write_json(
            self.root / "config" / "original.json", raw_config
        )
        campaign_document = dict(normalized_config["campaign"])
        # case 全载荷已保存在 original.json；normalized.json 只保留计数和有序集合哈希，
        # 避免在大型 formal campaign 中复制一遍巨大的配置数组。
        normalized_cases = list(campaign_document.pop("cases"))
        compact_normalized = dict(normalized_config)
        compact_normalized["campaign"] = {
            **campaign_document,
            "case_count": len(normalized_cases),
            "case_ids_hash": stable_hash(
                [
                    stable_hash(case)
                    for case in normalized_cases
                ]
            ),
            "case_payload_location": "config/original.json",
        }
        atomic_write_json(
            self.root / "config" / "normalized.json",
            compact_normalized,
        )
        atomic_write_json(self.root / "lineage.json", lineage)
        if not (self.root / "events.jsonl").exists():
            atomic_write_bytes(self.root / "events.jsonl", b"")
        if not (self.root / "validation.json").exists():
            atomic_write_json(
                self.root / "validation.json",
                {"status": "not_run", "checks": []},
            )
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS case_summary (
                    case_id TEXT PRIMARY KEY,
                    run_state TEXT NOT NULL,
                    result_hash TEXT NOT NULL,
                    wall_time_s REAL NOT NULL,
                    peak_ram_bytes INTEGER NOT NULL,
                    peak_python_bytes INTEGER NOT NULL,
                    error_category TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    summary_json TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    complete INTEGER NOT NULL CHECK (complete IN (0, 1))
                )
                """
            )
        # SQLite schema 和所有配置先闭合，最后才发布 manifest 完成标记。
        atomic_write_json(self.root / "manifest.json", compact_manifest)

    def is_initialized(self) -> bool:
        """检查紧凑 store 的文件骨架和 SQLite 摘要表是否完整。"""

        required = (
            self.root / "manifest.json",
            self.root / "config" / "original.json",
            self.root / "config" / "normalized.json",
            self.root / "lineage.json",
            self.root / "events.jsonl",
            self.root / "validation.json",
            self.database_path,
        )
        if not all(path.is_file() for path in required):
            return False
        try:
            connection = sqlite3.connect(
                f"file:{self.database_path}?mode=ro", uri=True
            )
            try:
                row = connection.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='case_summary'"
                ).fetchone()
            finally:
                connection.close()
            return row is not None
        except sqlite3.Error:
            return False

    @staticmethod
    def _payload_sha256(
        summary_json: str, validation_json: str
    ) -> str:
        """用 NUL 分隔 summary 与 validation，计算数据库载荷完整性摘要。"""

        digest = hashlib.sha256()
        digest.update(summary_json.encode("utf-8"))
        digest.update(b"\0")
        digest.update(validation_json.encode("utf-8"))
        return digest.hexdigest()

    def is_complete(self, case_id: str) -> bool:
        """在事务记录、载荷摘要和 validation 哈希均一致时才判为完整。"""

        if not self.database_path.is_file():
            return False
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT summary_json, validation_json, payload_sha256,
                           complete
                    FROM case_summary
                    WHERE case_id = ?
                    """,
                    (case_id,),
                ).fetchone()
            if row is None or int(row[3]) != 1:
                return False
            summary_json, validation_json, payload_sha256, _complete = row
            if (
                self._payload_sha256(summary_json, validation_json)
                != payload_sha256
            ):
                return False
            document = json.loads(summary_json)
            validation_document = json.loads(validation_json)
            if not isinstance(document, Mapping) or not isinstance(
                validation_document, Mapping
            ):
                return False
            return bool(
                document.get("case_id") == case_id
                and document.get("run_state") == "complete"
                and document.get("validation_sha256")
                == stable_hash(validation_document)
                and document.get("result_hash")
            )
        except (
            OSError,
            TypeError,
            ValueError,
            sqlite3.Error,
        ):
            return False

    def write_case(
        self,
        *,
        case_id: str,
        config: Mapping[str, Any],
        summary: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray] | None = None,
        trace_rows: Iterable[Mapping[str, Any]] = (),
        events: Iterable[Event | Mapping[str, Any]] = (),
        validation: Mapping[str, Any] | None = None,
    ) -> CaseRecord:
        """原子 upsert 一个纯 summary case，并拒绝该模式不支持的附件。"""

        if arrays:
            raise ValueError(
                "compact formal storage only supports summary output"
            )
        if list(trace_rows):
            raise ValueError(
                "compact formal storage only supports summary output"
            )
        event_rows = list(events)
        if event_rows:
            raise ValueError(
                "compact formal storage does not accept per-case events"
            )
        document = dict(summary)
        document["case_id"] = case_id
        document["completed_at_utc"] = utc_now()
        document["events_sha256"] = None
        document["trace_file"] = None
        document["trace_format"] = None
        document["trace_sha256"] = None
        validation_document = dict(validation or {})
        document["validation_sha256"] = stable_hash(validation_document)
        document["result_hash"] = _result_hash(
            config,
            document,
            event_lines=[],
            path_sha256=None,
        )
        document["path_sha256"] = None
        summary_json = json.dumps(
            canonicalize(document),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        validation_json = json.dumps(
            canonicalize(validation_document),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload_sha256 = self._payload_sha256(
            summary_json, validation_json
        )
        record = _case_record(document)
        with self._connect() as connection:
            # 单条 UPSERT 与事务边界共同保证中断后只会看到旧记录或完整新记录。
            connection.execute(
                """
                INSERT INTO case_summary (
                    case_id, run_state, result_hash, wall_time_s,
                    peak_ram_bytes, peak_python_bytes, error_category,
                    error_type, error_message, summary_json,
                    validation_json, payload_sha256, complete
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    run_state = excluded.run_state,
                    result_hash = excluded.result_hash,
                    wall_time_s = excluded.wall_time_s,
                    peak_ram_bytes = excluded.peak_ram_bytes,
                    peak_python_bytes = excluded.peak_python_bytes,
                    error_category = excluded.error_category,
                    error_type = excluded.error_type,
                    error_message = excluded.error_message,
                    summary_json = excluded.summary_json,
                    validation_json = excluded.validation_json,
                    payload_sha256 = excluded.payload_sha256,
                    complete = excluded.complete
                """,
                (
                    record.case_id,
                    record.run_state,
                    record.result_hash,
                    record.wall_time_s,
                    record.peak_ram_bytes,
                    record.peak_python_bytes,
                    record.error_category,
                    record.error_type,
                    record.error_message,
                    summary_json,
                    validation_json,
                    payload_sha256,
                    int(record.run_state == "complete"),
                ),
            )
        return record

    def load_case_summary(self, case_id: str) -> dict[str, Any]:
        """按 case ID 读取并解析摘要；不存在时保持文件式接口的异常语义。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary_json FROM case_summary WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(case_id)
        return json.loads(row[0])

    def list_records(self) -> list[CaseRecord]:
        """批量查询索引并复核事务载荷摘要，损坏记录不冒充成功。"""

        if not self.database_path.is_file():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT case_id, run_state, result_hash, wall_time_s,
                       peak_ram_bytes, peak_python_bytes, error_category,
                       error_type, error_message, summary_json,
                       validation_json, payload_sha256, complete
                FROM case_summary
                ORDER BY case_id
                """
            ).fetchall()
        records: list[CaseRecord] = []
        for row in rows:
            case_id = str(row[0])
            try:
                record = CaseRecord(
                    case_id=case_id,
                    run_state=str(row[1]),
                    result_hash=str(row[2]),
                    wall_time_s=float(row[3]),
                    peak_ram_bytes=int(row[4]),
                    peak_python_bytes=int(row[5]),
                    error_category=row[6],
                    error_type=row[7],
                    error_message=row[8],
                )
                summary_json = str(row[9])
                validation_json = str(row[10])
                summary = json.loads(summary_json)
                validation = json.loads(validation_json)
                intact = bool(
                    self._payload_sha256(summary_json, validation_json)
                    == row[11]
                    and isinstance(summary, Mapping)
                    and isinstance(validation, Mapping)
                    and summary.get("case_id") == record.case_id
                    and summary.get("run_state") == record.run_state
                    and summary.get("result_hash") == record.result_hash
                    and summary.get("validation_sha256")
                    == stable_hash(validation)
                    and int(row[12]) == int(record.run_state == "complete")
                    and record.run_state in {"complete", "execution_error"}
                )
            except (TypeError, ValueError):
                record = _integrity_error_record(
                    case_id,
                    "compact case contains malformed index field types",
                )
                intact = False
            if not intact:
                record = _integrity_error_record(
                    record.case_id,
                    "compact case failed transaction payload/hash verification",
                    record,
                )
            records.append(record)
        return records

    def rebuild_event_index(
        self, allowed_case_ids: set[str] | None = None
    ) -> None:
        """紧凑模式不接受逐 case 事件，因此聚合事件文件始终为空。"""

        atomic_write_bytes(self.root / "events.jsonl", b"")


def open_result_store(
    campaign_dir: str | Path,
) -> ResultStore | CompactResultStore:
    """依据 manifest 打开目录式或紧凑式结果存储。"""

    root = Path(campaign_dir).resolve()
    manifest_path = root / "manifest.json"
    result_storage = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("result manifest must be a mapping")
        result_storage = manifest.get("result_storage")
    if result_storage == "sqlite_transactional_summary_v1":
        if not (root / CompactResultStore.DATABASE_NAME).is_file():
            raise ValueError(
                "compact result storage is declared but its SQLite database "
                "is missing"
            )
        return CompactResultStore(root)
    if result_storage is not None:
        raise ValueError(f"unsupported result storage: {result_storage}")
    return ResultStore(root)
