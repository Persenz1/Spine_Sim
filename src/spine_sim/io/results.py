"""Atomic per-case writes and lightweight campaign indexes."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from spine_sim.core.identity import canonicalize, stable_hash
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
            return document.get("case_id") == case_id and document.get("run_state") == "complete"
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
        events: Iterable[Event | Mapping[str, Any]] = (),
        validation: Mapping[str, Any] | None = None,
        complete: bool,
    ) -> CaseRecord:
        directory = self.case_dir(case_id)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / "config.json", config)
        if arrays:
            atomic_write_npz(directory / "path.npz", arrays)
        event_lines = []
        for event in events:
            value = event.as_dict() if isinstance(event, Event) else dict(event)
            event_lines.append(json.dumps(canonicalize(value), ensure_ascii=False, sort_keys=True))
        atomic_write_bytes(
            directory / "events.jsonl",
            (("\n".join(event_lines) + ("\n" if event_lines else "")).encode("utf-8")),
        )
        atomic_write_json(directory / "validation.json", validation or {})
        document = dict(summary)
        document["case_id"] = case_id
        document["completed_at_utc"] = utc_now()
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
            {"config": config, "summary": stable_summary, "events": event_lines}
        )
        atomic_write_json(directory / "summary.json", document)
        marker = directory / "COMPLETE"
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
