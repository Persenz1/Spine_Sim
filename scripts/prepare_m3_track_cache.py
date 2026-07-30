"""Serially pre-generate and freeze every M1 track required by formal M3.

Formal workers are deliberately read-only consumers.  This command is the
single writer for the deterministic 65 y coordinates and two tip radii needed
for every selected terrain condition, including placement-search offsets.
The checkpoint report is updated after each terrain so an interrupted run can
resume without revisiting completed conditions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from spine_sim.array.design import (
    PLACEMENT_SEARCH_OFFSETS_XY_M,
    TERRAIN_FAMILIES,
    build_full_array_design,
    validate_terrain_catalog,
)
from spine_sim.array.models import ArrayConfiguration
from spine_sim.core.identity import stable_hash
from spine_sim.io.results import atomic_write_json, utc_now
from spine_sim.terrain import TerrainLibrary
from spine_sim.terrain.models import (
    ENVELOPE_ALGORITHM_VERSION,
    TrackGeometry,
)


SCHEMA_VERSION = "m3-formal-track-cache-v1"
DEFAULT_MANIFEST_NAME = "m3_formal_track_cache.json"


def required_track_coordinates() -> dict[float, tuple[float, ...]]:
    values: dict[float, set[float]] = {}
    for design in build_full_array_design():
        configuration = ArrayConfiguration.from_mapping(
            design["configuration"]
        )
        for parameters, offset in zip(
            configuration.pin_parameters,
            configuration.holder_offsets_xyz_m,
        ):
            radius_m = round(float(parameters.tip_radius_m), 12)
            for placement_offset in PLACEMENT_SEARCH_OFFSETS_XY_M:
                y_global_m = round(
                    float(offset[1]) + float(placement_offset[1]),
                    12,
                )
                values.setdefault(radius_m, set()).add(y_global_m)
    return {
        radius_m: tuple(sorted(y_values))
        for radius_m, y_values in sorted(values.items())
    }


def validate_complete_track_cache_manifest(
    catalog: dict[str, Any],
    manifest_path: Path | None = None,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    conditions = validate_terrain_catalog(
        catalog, require_formal_300=True
    )
    library_root = Path(str(catalog["library_root"])).resolve()
    path = manifest_path or (
        library_root / "manifests" / DEFAULT_MANIFEST_NAME
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    catalog_id = str(catalog["terrain_catalog_id"])
    coordinates = required_track_coordinates()
    expected_track_count = len(conditions) * sum(
        len(values) for values in coordinates.values()
    )
    expected_condition_ids = sorted(
        str(condition["terrain_condition_id"])
        for condition in conditions
    )
    expected_track_ids = sorted(
        TrackGeometry.make_id(
            terrain_recipe_id=condition["terrain_recipe_id"],
            region_id=condition["region_id"],
            radius_m=radius_m,
            y_global_m=y_global_m,
            envelope_algorithm_version=ENVELOPE_ALGORITHM_VERSION,
            resolution_m=float(
                condition["region"]["resolution_x_m"]
            ),
        )
        for condition in conditions
        for radius_m, y_values in coordinates.items()
        for y_global_m in y_values
    )
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("formal_complete") is not True
        or document.get("terrain_catalog_id") != catalog_id
        or int(document.get("condition_count", -1)) != len(conditions)
        or int(document.get("track_count", -1))
        != expected_track_count
        or sorted(document.get("completed_condition_ids", ()))
        != expected_condition_ids
        or sorted(document.get("track_ids", ()))
        != expected_track_ids
    ):
        raise ValueError(
            "formal M3 track-cache manifest is absent, incomplete, or "
            "does not match the selected M1 catalog"
        )
    if verify_files:
        library = TerrainLibrary(library_root)
        verified_track_count = 0
        for condition in conditions:
            resolution_m = float(
                condition["region"]["resolution_x_m"]
            )
            for radius_m, y_values in coordinates.items():
                for y_global_m in y_values:
                    track_id = TrackGeometry.make_id(
                        terrain_recipe_id=condition[
                            "terrain_recipe_id"
                        ],
                        region_id=condition["region_id"],
                        radius_m=radius_m,
                        y_global_m=y_global_m,
                        envelope_algorithm_version=(
                            ENVELOPE_ALGORITHM_VERSION
                        ),
                        resolution_m=resolution_m,
                    )
                    data_path = library.track_path(
                        condition["terrain_recipe_id"],
                        condition["region_id"],
                        radius_m,
                        track_id,
                    )
                    metadata_path = data_path.with_suffix(".json")
                    complete_path = data_path.with_suffix(".complete")
                    lock_path = data_path.with_suffix(".lock")
                    if (
                        not data_path.is_file()
                        or data_path.stat().st_size <= 0
                        or not metadata_path.is_file()
                        or not complete_path.is_file()
                        or lock_path.exists()
                    ):
                        raise ValueError(
                            "formal M3 track-cache file inventory is "
                            f"incomplete for track_id={track_id}"
                        )
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    marker = complete_path.read_text(
                        encoding="ascii"
                    ).strip()
                    if (
                        metadata.get("track_id") != track_id
                        or metadata.get("data_sha256") != marker
                    ):
                        raise ValueError(
                            "formal M3 track-cache metadata/COMPLETE "
                            f"mismatch for track_id={track_id}"
                        )
                    verified_track_count += 1
        document = {
            **document,
            "file_inventory_verified": True,
            "verified_track_file_count": verified_track_count,
        }
    return document


def _checkpoint(
    *,
    output_path: Path,
    catalog_id: str,
    catalog_path: Path,
    library_root: Path,
    all_condition_count: int,
    selected_condition_count: int,
    coordinates: dict[float, tuple[float, ...]],
    completed_condition_ids: list[str],
    track_ids: list[str],
    started_at_utc: str,
    elapsed_s: float,
    formal_complete: bool,
) -> dict[str, Any]:
    document = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": started_at_utc,
        "updated_at_utc": utc_now(),
        "terrain_catalog_id": catalog_id,
        "catalog_path": str(catalog_path.resolve()),
        "library_root": str(library_root),
        "all_catalog_condition_count": all_condition_count,
        "condition_count": selected_condition_count,
        "completed_condition_count": len(completed_condition_ids),
        "completed_condition_ids": sorted(completed_condition_ids),
        "required_coordinates_by_radius_m": {
            format(radius_m, ".12g"): list(values)
            for radius_m, values in coordinates.items()
        },
        "tracks_per_condition": sum(
            len(values) for values in coordinates.values()
        ),
        "track_count": len(track_ids),
        "track_ids": sorted(set(track_ids)),
        "track_set_hash": stable_hash(sorted(track_ids)),
        "elapsed_s": elapsed_s,
        "formal_complete": formal_complete,
        "worker_cache_policy": "read_only_after_serial_generation",
    }
    atomic_write_json(output_path, document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--terrain-family", choices=TERRAIN_FAMILIES)
    parser.add_argument("--seed-min", type=int)
    parser.add_argument("--seed-max", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.seed_min is None) != (args.seed_max is None):
        parser.error("--seed-min and --seed-max must be supplied together")
    if (
        args.seed_min is not None
        and args.seed_max is not None
        and args.seed_max < args.seed_min
    ):
        parser.error("--seed-max cannot be less than --seed-min")

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    all_conditions = validate_terrain_catalog(
        catalog, require_formal_300=True
    )
    selected = [
        condition
        for condition in all_conditions
        if (
            args.terrain_family is None
            or condition["terrain_family"] == args.terrain_family
        )
        and (
            args.seed_min is None
            or args.seed_min <= condition["seed"] <= args.seed_max
        )
    ]
    if not selected:
        parser.error("track-cache selection contains no terrain conditions")
    library_root = Path(str(catalog["library_root"])).resolve()
    output_path = args.output or (
        library_root / "manifests" / DEFAULT_MANIFEST_NAME
    )
    coordinates = required_track_coordinates()
    tracks_per_condition = sum(
        len(values) for values in coordinates.values()
    )
    full_selection = len(selected) == len(all_conditions)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "condition_count": len(selected),
                    "tracks_per_condition": tracks_per_condition,
                    "track_count": len(selected) * tracks_per_condition,
                    "formal_complete_when_finished": full_selection,
                    "output": str(output_path.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    completed_condition_ids: list[str] = []
    track_ids: list[str] = []
    started_at_utc = utc_now()
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            previous.get("schema_version") == SCHEMA_VERSION
            and previous.get("terrain_catalog_id")
            == str(catalog["terrain_catalog_id"])
        ):
            completed_condition_ids = list(
                previous.get("completed_condition_ids", ())
            )
            track_ids = list(previous.get("track_ids", ()))
            started_at_utc = str(
                previous.get("created_at_utc", started_at_utc)
            )
    completed_set = set(completed_condition_ids)
    library = TerrainLibrary(library_root)
    started = time.perf_counter()
    for index, condition in enumerate(selected, start=1):
        condition_id = str(condition["terrain_condition_id"])
        if condition_id in completed_set:
            print(
                f"[{index}/{len(selected)}] already complete "
                f"{condition['terrain_family']}/seed={condition['seed']}",
                flush=True,
            )
            continue
        condition_started = time.perf_counter()
        recipe = library.load_recipe(condition["terrain_recipe_id"])
        region = library.load_region_spec(
            condition["terrain_recipe_id"], condition["region_id"]
        )
        condition_track_ids: list[str] = []
        for radius_m, y_values in coordinates.items():
            for y_global_m in y_values:
                track = library.cache_track(
                    recipe,
                    region,
                    radius_m=radius_m,
                    y_global_m=y_global_m,
                )
                condition_track_ids.append(track.track_id)
        track_ids.extend(condition_track_ids)
        completed_condition_ids.append(condition_id)
        completed_set.add(condition_id)
        _checkpoint(
            output_path=output_path,
            catalog_id=str(catalog["terrain_catalog_id"]),
            catalog_path=args.catalog,
            library_root=library_root,
            all_condition_count=len(all_conditions),
            selected_condition_count=len(selected),
            coordinates=coordinates,
            completed_condition_ids=completed_condition_ids,
            track_ids=track_ids,
            started_at_utc=started_at_utc,
            elapsed_s=time.perf_counter() - started,
            formal_complete=False,
        )
        print(
            f"[{index}/{len(selected)}] generated "
            f"{len(condition_track_ids)} tracks for "
            f"{condition['terrain_family']}/seed={condition['seed']} "
            f"in {time.perf_counter() - condition_started:.2f}s",
            flush=True,
        )

    formal_complete = (
        full_selection
        and len(completed_set) == len(all_conditions)
        and len(set(track_ids))
        == len(all_conditions) * tracks_per_condition
    )
    document = _checkpoint(
        output_path=output_path,
        catalog_id=str(catalog["terrain_catalog_id"]),
        catalog_path=args.catalog,
        library_root=library_root,
        all_condition_count=len(all_conditions),
        selected_condition_count=len(selected),
        coordinates=coordinates,
        completed_condition_ids=completed_condition_ids,
        track_ids=track_ids,
        started_at_utc=started_at_utc,
        elapsed_s=time.perf_counter() - started,
        formal_complete=formal_complete,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in document.items()
                if key != "track_ids"
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if (formal_complete or not full_selection) else 2


if __name__ == "__main__":
    raise SystemExit(main())
