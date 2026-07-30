"""Run every bounded M3 proxy shard with resumable compact summary storage.

The queue refuses to start unless a machine-readable preflight report explicitly
allows the proxy launch.  It materializes only one shard configuration at a
time, retains every raw shard database, and never deletes terrain or scratch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from spine_sim.array.design import (
    TERRAIN_FAMILIES,
    TOTAL_PRELOADS_N,
    build_campaign_shard,
    validate_terrain_catalog,
)
from spine_sim.core.config import CampaignSpec
from spine_sim.io.results import (
    atomic_write_json,
    open_result_store,
    utc_now,
)

from prepare_m3_track_cache import (
    validate_complete_track_cache_manifest,
)

REQUIRED_PROXY_LAUNCH_GATES = (
    "formal_track_cache_complete",
    "time_step_convergence_passed",
    "terrain_resolution_convergence_passed",
    "throughput_acceptable",
)


def _seed_windows(seeds: list[int], size: int) -> list[tuple[int, int]]:
    return [
        (seeds[index], seeds[min(index + size - 1, len(seeds) - 1)])
        for index in range(0, len(seeds), size)
    ]


def _progress_document(
    *,
    catalog_path: Path,
    queue_root: Path,
    expected_shards: int,
    completed_shards: list[dict[str, Any]],
    active_shard: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "m3-formal-proxy-queue-v1",
        "updated_at_utc": utc_now(),
        "status": status,
        "catalog_path": str(catalog_path.resolve()),
        "queue_root": str(queue_root.resolve()),
        "expected_shard_count": expected_shards,
        "completed_shard_count": len(completed_shards),
        "completed_case_count": sum(
            int(shard["case_count"]) for shard in completed_shards
        ),
        "completed_shards": completed_shards,
        "active_shard": active_shard,
        "raw_scratch_retained": True,
        "formal_ranking_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument(
        "--queue-root",
        type=Path,
        default=Path("results/m3_formal_proxy_queue"),
    )
    parser.add_argument("--seed-window-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.seed_window_size < 1 or args.seed_window_size > 5:
        parser.error("--seed-window-size must be between 1 and 5")
    if args.workers < 1:
        parser.error("--workers must be positive")

    preflight = json.loads(
        args.preflight_report.read_text(encoding="utf-8")
    )
    if preflight.get("schema_version") != "m3-preflight-v1":
        raise SystemExit(
            "M3 formal proxy queue requires an m3-preflight-v1 report"
        )
    launch_gates = preflight.get("launch_gates", {})
    failed_launch_gates = [
        name
        for name in REQUIRED_PROXY_LAUNCH_GATES
        if launch_gates.get(name) is not True
    ]
    if (
        preflight.get("proxy_full_launch_allowed") is not True
        or failed_launch_gates
    ):
        raise SystemExit(
            "M3 formal proxy queue blocked by preflight report; "
            f"failed_gates={failed_launch_gates}; "
            f"blockers={preflight.get('proxy_full_launch_blockers', [])}"
        )

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    conditions = validate_terrain_catalog(
        catalog, require_formal_300=True
    )
    validate_complete_track_cache_manifest(
        catalog, verify_files=True
    )
    seeds = sorted({int(condition["seed"]) for condition in conditions})
    windows = _seed_windows(seeds, args.seed_window_size)
    shard_specs = [
        {
            "terrain_family": family,
            "seed_min": seed_min,
            "seed_max": seed_max,
            "preload_n": preload_n,
        }
        for family in TERRAIN_FAMILIES
        for seed_min, seed_max in windows
        for preload_n in TOTAL_PRELOADS_N
    ]

    queue_root = args.queue_root.resolve()
    config_dir = queue_root / "configs"
    scratch_root = queue_root / "shards"
    log_dir = queue_root / "logs"
    for path in (config_dir, scratch_root, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    progress_path = queue_root / "queue_progress.json"
    completed_shards: list[dict[str, Any]] = []
    if progress_path.is_file():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        completed_shards = list(
            previous.get("completed_shards", ())
        )
    completed_keys = {
        (
            shard["terrain_family"],
            int(shard["seed_min"]),
            int(shard["seed_max"]),
            float(shard["preload_n"]),
        )
        for shard in completed_shards
    }

    for shard_index, spec in enumerate(shard_specs, start=1):
        key = (
            spec["terrain_family"],
            spec["seed_min"],
            spec["seed_max"],
            spec["preload_n"],
        )
        if key in completed_keys:
            continue
        shard_name = (
            f"{spec['terrain_family']}_"
            f"{spec['seed_min']}_{spec['seed_max']}_"
            f"{spec['preload_n']:g}N"
        )
        config_path = config_dir / f"{shard_name}.json"
        campaign = build_campaign_shard(
            catalog,
            **spec,
            output_level="summary",
            workers=args.workers,
            require_formal_300=True,
        )
        parsed = CampaignSpec.from_mapping(campaign)
        atomic_write_json(config_path, campaign)
        active = {
            "shard_index": shard_index,
            "shard_name": shard_name,
            "campaign_id": parsed.campaign_id,
            "case_count": len(parsed.cases),
            **spec,
        }
        atomic_write_json(
            progress_path,
            _progress_document(
                catalog_path=args.catalog,
                queue_root=queue_root,
                expected_shards=len(shard_specs),
                completed_shards=completed_shards,
                active_shard=active,
                status="running",
            ),
        )
        log_path = log_dir / f"{shard_name}.log"
        command = [
            sys.executable,
            "-m",
            "spine_sim.cli",
            "resume",
            str(config_path),
            "--output",
            str(scratch_root),
            "--workers",
            str(args.workers),
        ]
        started = time.perf_counter()
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        campaign_dir = scratch_root / parsed.campaign_id
        records = open_result_store(campaign_dir).list_records()
        complete_count = sum(
            record.run_state == "complete" for record in records
        )
        if (
            completed.returncode != 0
            or complete_count != len(parsed.cases)
        ):
            active.update(
                {
                    "return_code": completed.returncode,
                    "complete_case_count": complete_count,
                    "elapsed_s": time.perf_counter() - started,
                    "log_path": str(log_path),
                }
            )
            atomic_write_json(
                progress_path,
                _progress_document(
                    catalog_path=args.catalog,
                    queue_root=queue_root,
                    expected_shards=len(shard_specs),
                    completed_shards=completed_shards,
                    active_shard=active,
                    status="failed",
                ),
            )
            return 2
        completed_row = {
            **active,
            "campaign_dir": str(campaign_dir),
            "elapsed_s": time.perf_counter() - started,
            "result_set_hash": json.loads(
                (campaign_dir / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )["result_set_hash"],
        }
        completed_shards.append(completed_row)
        completed_keys.add(key)
        atomic_write_json(
            progress_path,
            _progress_document(
                catalog_path=args.catalog,
                queue_root=queue_root,
                expected_shards=len(shard_specs),
                completed_shards=completed_shards,
                active_shard=None,
                status="running",
            ),
        )

    atomic_write_json(
        progress_path,
        _progress_document(
            catalog_path=args.catalog,
            queue_root=queue_root,
            expected_shards=len(shard_specs),
            completed_shards=completed_shards,
            active_shard=None,
            status="complete",
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
