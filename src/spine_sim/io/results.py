"""Atomic per-case writes and lightweight campaign indexes."""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from spine_sim.core.identity import canonical_json, canonicalize, stable_hash
from spine_sim.core.states import Event


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    data = json.dumps(
        canonicalize(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    atomic_write_bytes(path, data + b"\n")


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp_path = Path(temporary)
    try:
        with temp_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_trace_table(
    directory: Path, rows: Iterable[Mapping[str, Any]]
) -> tuple[str | None, str | None, str | None]:
    """Write canonical trace rows as Parquet, or JSONL when pyarrow is absent."""

    normalized = [canonicalize(dict(row)) for row in rows]
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
    """Read a canonical trace and reverse its lossless Parquet encoding."""

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
    json_columns = set(
        json.loads(metadata.get(b"spine_sim_json_columns", b"[]"))
    )
    rows = table.to_pylist()
    for row in rows:
        for key in json_columns:
            if key in row:
                row[key] = json.loads(row[key])
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    run_state: str
    result_hash: str
    wall_time_s: float
    peak_ram_bytes: int
    peak_python_bytes: int
    error_category: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class ResultStore:
    def __init__(self, campaign_dir: str | Path):
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
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config").mkdir(exist_ok=True)
        self.cases_dir.mkdir(exist_ok=True)
        atomic_write_json(self.root / "manifest.json", manifest)
        atomic_write_json(self.root / "config" / "original.json", raw_config)
        atomic_write_json(self.root / "config" / "normalized.json", normalized_config)
        atomic_write_json(self.root / "lineage.json", lineage)
        if not (self.root / "events.jsonl").exists():
            atomic_write_bytes(self.root / "events.jsonl", b"")
        if not (self.root / "validation.json").exists():
            atomic_write_json(self.root / "validation.json", {"status": "not_run", "checks": []})

    def case_dir(self, case_id: str) -> Path:
        if not case_id or any(char in case_id for char in "\\/:"):
            raise ValueError("invalid case_id")
        return self.cases_dir / case_id

    def is_complete(self, case_id: str) -> bool:
        directory = self.case_dir(case_id)
        marker = directory / "COMPLETE"
        summary = directory / "summary.json"
        if not marker.is_file() or not summary.is_file():
            return False
        try:
            document = json.loads(summary.read_text(encoding="utf-8"))
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
                return True
            events_sha256 = document["events_sha256"]
            if events_sha256 is None:
                return not event_file.exists()
            return (
                event_file.is_file()
                and sha256_file(event_file) == events_sha256
            )
        except (OSError, json.JSONDecodeError):
            return False

    def is_incomplete(self, case_id: str) -> bool:
        directory = self.case_dir(case_id)
        return directory.exists() and not self.is_complete(case_id)

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
        complete: bool,
    ) -> CaseRecord:
        directory = self.case_dir(case_id)
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "COMPLETE"
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
            event_lines.append(json.dumps(canonicalize(value), ensure_ascii=False, sort_keys=True))
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
        atomic_write_json(directory / "validation.json", validation or {})
        document = dict(summary)
        document["case_id"] = case_id
        document["completed_at_utc"] = utc_now()
        document["events_sha256"] = events_sha256
        document["trace_file"] = trace_file
        document["trace_format"] = trace_format
        document["trace_sha256"] = trace_sha256
        stable_summary = {
            key: value
            for key, value in document.items()
            if key
            not in {
                "completed_at_utc",
                "wall_time_s",
                "peak_ram_bytes",
                "peak_python_bytes",
                "peak_vram_bytes",
                "stage_times_s",
                "diagnostic_traceback",
            }
        }
        document["result_hash"] = stable_hash(
            {
                "config": config,
                "summary": stable_summary,
                "events": event_lines,
                "events_sha256": events_sha256,
                "path_sha256": path_sha256,
                "trace_file": trace_file,
                "trace_format": trace_format,
                "trace_sha256": trace_sha256,
            }
        )
        document["path_sha256"] = path_sha256
        atomic_write_json(directory / "summary.json", document)
        if complete:
            atomic_write_bytes(marker, (document["result_hash"] + "\n").encode("ascii"))
        elif marker.exists():
            marker.unlink()
        return CaseRecord(
            case_id=case_id,
            run_state=str(document.get("run_state", "execution_error")),
            result_hash=document["result_hash"],
            wall_time_s=float(document.get("wall_time_s", 0)),
            peak_ram_bytes=int(document.get("peak_ram_bytes", 0)),
            peak_python_bytes=int(document.get("peak_python_bytes", 0)),
            error_category=document.get("error", {}).get("category"),
            error_type=document.get("error", {}).get("type"),
            error_message=document.get("error", {}).get("message"),
        )

    def load_case_summary(self, case_id: str) -> dict[str, Any]:
        path = self.case_dir(case_id) / "summary.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def list_records(self) -> list[CaseRecord]:
        records: list[CaseRecord] = []
        if not self.cases_dir.exists():
            return records
        for directory in sorted(self.cases_dir.iterdir()):
            if not directory.is_dir() or not (directory / "summary.json").is_file():
                continue
            data = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            records.append(
                CaseRecord(
                    case_id=directory.name,
                    run_state=str(data.get("run_state", "execution_error")),
                    result_hash=str(data.get("result_hash", "")),
                    wall_time_s=float(data.get("wall_time_s", 0)),
                    peak_ram_bytes=int(data.get("peak_ram_bytes", 0)),
                    peak_python_bytes=int(data.get("peak_python_bytes", 0)),
                    error_category=data.get("error", {}).get("category"),
                    error_type=data.get("error", {}).get("type"),
                    error_message=data.get("error", {}).get("message"),
                )
            )
        return records

    def iter_case_summaries(
        self, *, verify_payloads: bool = False
    ) -> Iterable[dict[str, Any]]:
        if not self.cases_dir.exists():
            return
        for directory in sorted(self.cases_dir.iterdir()):
            if not directory.is_dir():
                continue
            case_id = directory.name
            if verify_payloads and not self.is_complete(case_id):
                raise RuntimeError(
                    "incomplete or hash-invalid case summary: "
                    f"{case_id}"
                )
            summary = directory / "summary.json"
            if summary.is_file():
                yield json.loads(summary.read_text(encoding="utf-8"))

    def write_campaign_index(self, records: Iterable[CaseRecord]) -> str:
        rows = [asdict(record) for record in sorted(records, key=lambda row: row.case_id)]
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            table = pa.Table.from_pylist(rows)
            target = self.root / "cases.parquet"
            descriptor, temporary = tempfile.mkstemp(prefix=".cases.", suffix=".tmp", dir=self.root)
            os.close(descriptor)
            temp_path = Path(temporary)
            try:
                pq.write_table(table, temp_path)
                os.replace(temp_path, target)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
            fallback = self.root / "cases.jsonl"
            if fallback.exists():
                fallback.unlink()
            return "parquet"
        except ImportError:
            payload = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            )
            atomic_write_bytes(self.root / "cases.jsonl", payload.encode("utf-8"))
            return "jsonl_fallback"

    def rebuild_event_index(self) -> None:
        lines: list[str] = []
        if self.cases_dir.exists():
            for directory in sorted(self.cases_dir.iterdir()):
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

    def update_manifest(self, **fields: Any) -> None:
        path = self.root / "manifest.json"
        current = json.loads(path.read_text(encoding="utf-8"))
        current.update(fields)
        atomic_write_json(path, current)


class CompactResultStore:
    """Transactional summary-only storage for large formal campaign shards.

    A formal shard can contain thousands of cases. Keeping four or five files
    in one directory per summary case turns filesystem metadata into a larger
    workload than the summaries themselves. This store retains the same
    result-hash semantics while committing each summary atomically to SQLite.
    Aggregate/full trace campaigns continue to use :class:`ResultStore`.
    """

    DATABASE_NAME = "case_summaries.sqlite3"

    def __init__(self, campaign_dir: str | Path):
        self.root = Path(campaign_dir).resolve()
        self.database_path = self.root / self.DATABASE_NAME

    @contextmanager
    def _connect(self) -> Iterable[sqlite3.Connection]:
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
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config").mkdir(exist_ok=True)
        compact_manifest = dict(manifest)
        compact_manifest["result_storage"] = (
            "sqlite_transactional_summary_v1"
        )
        atomic_write_json(self.root / "manifest.json", compact_manifest)
        atomic_write_json(
            self.root / "config" / "original.json", raw_config
        )
        campaign_document = dict(
            normalized_config.get("campaign", {})
        )
        normalized_cases = list(campaign_document.pop("cases", ()))
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

    @staticmethod
    def _payload_sha256(
        summary_json: str, validation_json: str
    ) -> str:
        digest = hashlib.sha256()
        digest.update(summary_json.encode("utf-8"))
        digest.update(b"\0")
        digest.update(validation_json.encode("utf-8"))
        return digest.hexdigest()

    def is_complete(self, case_id: str) -> bool:
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
            return (
                document.get("case_id") == case_id
                and document.get("run_state") == "complete"
                and document.get("result_hash")
            )
        except (OSError, sqlite3.Error, json.JSONDecodeError):
            return False

    def is_incomplete(self, case_id: str) -> bool:
        if not self.database_path.is_file():
            return False
        with self._connect() as connection:
            present = connection.execute(
                "SELECT 1 FROM case_summary WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        return present is not None and not self.is_complete(case_id)

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
        complete: bool,
    ) -> CaseRecord:
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
        stable_summary = {
            key: value
            for key, value in document.items()
            if key
            not in {
                "completed_at_utc",
                "wall_time_s",
                "peak_ram_bytes",
                "peak_python_bytes",
                "peak_vram_bytes",
                "stage_times_s",
                "diagnostic_traceback",
            }
        }
        document["result_hash"] = stable_hash(
            {
                "config": config,
                "summary": stable_summary,
                "events": [],
                "events_sha256": None,
                "path_sha256": None,
                "trace_file": None,
                "trace_format": None,
                "trace_sha256": None,
            }
        )
        document["path_sha256"] = None
        validation_document = dict(validation or {})
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
        error = document.get("error", {})
        with self._connect() as connection:
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
                    case_id,
                    str(document.get("run_state", "execution_error")),
                    document["result_hash"],
                    float(document.get("wall_time_s", 0)),
                    int(document.get("peak_ram_bytes", 0)),
                    int(document.get("peak_python_bytes", 0)),
                    error.get("category"),
                    error.get("type"),
                    error.get("message"),
                    summary_json,
                    validation_json,
                    payload_sha256,
                    int(bool(complete)),
                ),
            )
        return CaseRecord(
            case_id=case_id,
            run_state=str(
                document.get("run_state", "execution_error")
            ),
            result_hash=document["result_hash"],
            wall_time_s=float(document.get("wall_time_s", 0)),
            peak_ram_bytes=int(document.get("peak_ram_bytes", 0)),
            peak_python_bytes=int(
                document.get("peak_python_bytes", 0)
            ),
            error_category=error.get("category"),
            error_type=error.get("type"),
            error_message=error.get("message"),
        )

    def load_case_summary(self, case_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary_json FROM case_summary WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(case_id)
        return json.loads(row[0])

    def list_records(self) -> list[CaseRecord]:
        if not self.database_path.is_file():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT case_id, run_state, result_hash, wall_time_s,
                       peak_ram_bytes, peak_python_bytes, error_category,
                       error_type, error_message
                FROM case_summary
                ORDER BY case_id
                """
            ).fetchall()
        return [
            CaseRecord(
                case_id=row[0],
                run_state=row[1],
                result_hash=row[2],
                wall_time_s=float(row[3]),
                peak_ram_bytes=int(row[4]),
                peak_python_bytes=int(row[5]),
                error_category=row[6],
                error_type=row[7],
                error_message=row[8],
            )
            for row in rows
        ]

    def iter_case_summaries(
        self, *, verify_payloads: bool = False
    ) -> Iterable[dict[str, Any]]:
        if not self.database_path.is_file():
            return
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT case_id, summary_json, validation_json,
                       payload_sha256, complete
                FROM case_summary
                ORDER BY case_id
                """
            )
            for (
                case_id,
                summary_json,
                validation_json,
                payload_sha256,
                complete,
            ) in cursor:
                if verify_payloads and (
                    int(complete) != 1
                    or self._payload_sha256(
                        summary_json, validation_json
                    )
                    != payload_sha256
                ):
                    raise RuntimeError(
                        "incomplete or hash-invalid compact case "
                        f"summary: {case_id}"
                    )
                yield json.loads(summary_json)

    def write_campaign_index(
        self, records: Iterable[CaseRecord]
    ) -> str:
        rows = [
            asdict(record)
            for record in sorted(records, key=lambda row: row.case_id)
        ]
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            table = pa.Table.from_pylist(rows)
            target = self.root / "cases.parquet"
            descriptor, temporary = tempfile.mkstemp(
                prefix=".cases.", suffix=".tmp", dir=self.root
            )
            os.close(descriptor)
            temp_path = Path(temporary)
            try:
                pq.write_table(table, temp_path)
                os.replace(temp_path, target)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
            fallback = self.root / "cases.jsonl"
            if fallback.exists():
                fallback.unlink()
            return "parquet"
        except ImportError:
            payload = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True)
                + "\n"
                for row in rows
            )
            atomic_write_bytes(
                self.root / "cases.jsonl", payload.encode("utf-8")
            )
            return "jsonl_fallback"

    def rebuild_event_index(self) -> None:
        atomic_write_bytes(self.root / "events.jsonl", b"")

    def update_manifest(self, **fields: Any) -> None:
        path = self.root / "manifest.json"
        current = json.loads(path.read_text(encoding="utf-8"))
        current.update(fields)
        atomic_write_json(path, current)


def open_result_store(
    campaign_dir: str | Path,
) -> ResultStore | CompactResultStore:
    root = Path(campaign_dir).resolve()
    if (root / CompactResultStore.DATABASE_NAME).is_file():
        return CompactResultStore(root)
    return ResultStore(root)
