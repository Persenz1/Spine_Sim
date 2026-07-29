"""Stream bounded M3 shard summaries into one atomic columnar artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from spine_sim.io.results import ResultStore, atomic_write_json


ROW_GROUP_SIZE = 10_000


def _columnar_value(value: Any) -> Any:
    """Keep scalar columns native and encode nested fields deterministically."""

    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return value


def _columnar_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _columnar_value(value)
        for key, value in sorted(summary.items())
    }


def _summary_stream(
    campaign_dirs: Sequence[Path],
    *,
    duplicate_index: sqlite3.Connection | None,
    verify_payloads: bool,
) -> Iterator[dict[str, Any]]:
    for campaign_dir in campaign_dirs:
        store = ResultStore(campaign_dir)
        if not store.cases_dir.is_dir():
            raise RuntimeError(
                f"M3 shard has no paths directory: {campaign_dir}"
            )
        for case_dir in sorted(store.cases_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            if verify_payloads and not store.is_complete(case_id):
                raise RuntimeError(
                    "refusing summary merge: incomplete or hash-invalid "
                    f"{case_id}"
                )
            if duplicate_index is not None:
                try:
                    duplicate_index.execute(
                        "INSERT INTO seen_case(case_id) VALUES (?)",
                        (case_id,),
                    )
                except sqlite3.IntegrityError as exc:
                    raise RuntimeError(
                        f"duplicate case_id across M3 shards: {case_id}"
                    ) from exc
            yield store.load_case_summary(case_id)


def _update_result_set_hash(
    digest: Any,
    summary: Mapping[str, Any],
) -> None:
    digest.update(str(summary["case_id"]).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(summary["result_hash"]).encode("ascii"))
    digest.update(b"\n")


def _inspect_parquet_schema(
    summaries: Iterator[dict[str, Any]],
) -> tuple[dict[str, str], int, str]:
    """Infer nullable scalar types without retaining any summary rows."""

    kinds: dict[str, str] = {}
    all_keys: set[str] = set()
    digest = hashlib.sha256()
    count = 0
    for summary in summaries:
        _update_result_set_hash(digest, summary)
        count += 1
        for key, raw_value in summary.items():
            all_keys.add(key)
            value = _columnar_value(raw_value)
            if value is None:
                continue
            if isinstance(value, bool):
                kind = "bool"
            elif isinstance(value, int):
                kind = "int64"
            elif isinstance(value, float):
                kind = "float64"
            else:
                kind = "string"
            previous = kinds.get(key)
            if previous is None:
                kinds[key] = kind
            elif previous != kind:
                if {previous, kind} <= {"int64", "float64"}:
                    kinds[key] = "float64"
                else:
                    kinds[key] = "string"
    if count == 0:
        raise RuntimeError("no complete M3 summaries found")
    for key in all_keys:
        kinds.setdefault(key, "string")
    return kinds, count, digest.hexdigest()


def _write_parquet(
    summaries: Iterator[dict[str, Any]],
    *,
    target: Path,
    column_kinds: Mapping[str, str],
) -> tuple[int, str]:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temp_path = Path(temporary)
    arrow_types = {
        "bool": pa.bool_(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "string": pa.string(),
    }
    schema = pa.schema(
        [
            pa.field(key, arrow_types[column_kinds[key]])
            for key in sorted(column_kinds)
        ]
    )
    writer = pq.ParquetWriter(
        temp_path,
        schema,
        compression="zstd",
    )
    digest = hashlib.sha256()
    count = 0
    batch: list[dict[str, Any]] = []
    try:
        for summary in summaries:
            _update_result_set_hash(digest, summary)
            batch.append(_columnar_row(summary))
            count += 1
            if len(batch) < ROW_GROUP_SIZE:
                continue
            table = pa.Table.from_pylist(batch, schema=schema)
            writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
            batch.clear()
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
        writer.close()
        writer = None
        os.replace(temp_path, target)
    finally:
        if writer is not None:
            writer.close()
        if temp_path.exists():
            temp_path.unlink()
    return count, digest.hexdigest()


def _write_jsonl(
    summaries: Iterator[dict[str, Any]],
    *,
    target: Path,
) -> tuple[int, str]:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    digest = hashlib.sha256()
    count = 0
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for summary in summaries:
                _update_result_set_hash(digest, summary)
                handle.write(
                    json.dumps(
                        summary,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if count == 0:
            raise RuntimeError("no complete M3 summaries found")
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return count, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    database_descriptor, database_name = tempfile.mkstemp(
        prefix=".m3-summary-duplicates.",
        suffix=".sqlite3",
        dir=args.output.parent,
    )
    os.close(database_descriptor)
    database_path = Path(database_name)
    duplicate_index = sqlite3.connect(database_path)
    try:
        duplicate_index.execute(
            "CREATE TABLE seen_case(case_id TEXT PRIMARY KEY)"
        )
        try:
            import pyarrow  # type: ignore  # noqa: F401

            output_format = "parquet"
            target = args.output.with_suffix(".parquet")
            (
                column_kinds,
                inspected_case_count,
                inspected_result_set_hash,
            ) = _inspect_parquet_schema(
                _summary_stream(
                    args.campaign_dirs,
                    duplicate_index=duplicate_index,
                    verify_payloads=True,
                )
            )
            case_count, result_set_hash = _write_parquet(
                _summary_stream(
                    args.campaign_dirs,
                    duplicate_index=None,
                    verify_payloads=False,
                ),
                target=target,
                column_kinds=column_kinds,
            )
            if (
                case_count != inspected_case_count
                or result_set_hash != inspected_result_set_hash
            ):
                if target.exists():
                    target.unlink()
                raise RuntimeError(
                    "M3 summaries changed during the two-pass Parquet merge"
                )
        except ImportError:
            output_format = "jsonl_fallback"
            target = args.output.with_suffix(".jsonl")
            case_count, result_set_hash = _write_jsonl(
                _summary_stream(
                    args.campaign_dirs,
                    duplicate_index=duplicate_index,
                    verify_payloads=True,
                ),
                target=target,
            )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        duplicate_index.close()
        if database_path.exists():
            database_path.unlink()

    manifest = {
        "schema_version": "2",
        "format": output_format,
        "path": str(target.resolve()),
        "case_count": case_count,
        "row_group_size": (
            ROW_GROUP_SIZE if output_format == "parquet" else None
        ),
        "nested_summary_fields": (
            "canonical_json_strings_in_parquet"
            if output_format == "parquet"
            else "native_json"
        ),
        "source_shards": [
            str(path.resolve()) for path in args.campaign_dirs
        ],
        "result_set_sha256": result_set_hash,
        "result_set_hash_algorithm": (
            "sha256(case_id NUL result_hash LF in declared source order)"
        ),
        "duplicate_case_ids_rejected": True,
        "all_complete_markers_and_payload_hashes_verified": True,
        "ranking_contract": (
            "filter ranking_inclusion_allowed=true; report initialization "
            "coverage separately"
        ),
    }
    atomic_write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
