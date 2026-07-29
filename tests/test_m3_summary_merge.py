from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from spine_sim.io.results import ResultStore


def _write_summary(shard: Path, case_id: str) -> None:
    ResultStore(shard).write_case(
        case_id=case_id,
        config={"case": case_id},
        summary={
            "run_state": "complete",
            "initial_preload_success": True,
            "ranking_inclusion_allowed": False,
            "nested": {"value": case_id},
            "nullable_metric_n": None if case_id == "case_a" else 1.5,
        },
        complete=True,
    )


def test_streaming_summary_merge_and_duplicate_rejection() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        _write_summary(first, "case_a")
        _write_summary(second, "case_b")
        output = root / "merged"
        command = [
            sys.executable,
            "scripts/merge_m3_summaries.py",
            str(first),
            str(second),
            "--output",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        manifest = json.loads(
            output.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["case_count"] == 2
        assert manifest["duplicate_case_ids_rejected"] is True
        assert Path(manifest["path"]).is_file()
        if manifest["format"] == "parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(manifest["path"])
            assert str(table.schema.field("nullable_metric_n").type) == "double"
            assert table["nullable_metric_n"].to_pylist() == [None, 1.5]

        duplicate = root / "duplicate"
        _write_summary(duplicate, "case_a")
        rejected = subprocess.run(
            [
                sys.executable,
                "scripts/merge_m3_summaries.py",
                str(first),
                str(duplicate),
                "--output",
                str(root / "duplicate_merge"),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        assert rejected.returncode != 0
        assert "duplicate case_id" in (rejected.stdout + rejected.stderr)
