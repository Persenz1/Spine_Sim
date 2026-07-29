"""Generate the formal full-size 3-family x 100-condition M1 catalog.

The batch is deterministic and resumable.  Sandpaper subtypes cycle through
P40/P60/P100/P180/P240 in a frozen order while all three material families use
the same 100 seeds, preserving the paired M3 condition design.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from spine_sim.core.identity import identity, stable_hash
from spine_sim.io.results import atomic_write_json, utc_now
from spine_sim.terrain import (
    M1_MODULE_VERSION,
    TerrainLibrary,
    generate_terrain,
    register_terrain,
)


FAMILIES = ("sandpaper", "red_brick", "concrete")
SANDPAPER_SUBTYPES = ("P40", "P60", "P100", "P180", "P240")
FAMILY_SUBTYPE = {
    "red_brick": "fired_brick_standard",
    "concrete": "rough_wall",
}
SEED_START = 41001
SEED_COUNT = 100
SIZE_X_M = 147.960e-3
SIZE_Y_M = 40.200e-3
ORIGIN_X_M = -21.980e-3
ORIGIN_Y_M = -20.100e-3
RESOLUTION_M = 10e-6
SCHEMA_VERSION = "m1-material-formal-batch-v1"


def formal_condition_design() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    index = 0
    for family in FAMILIES:
        for seed_offset in range(SEED_COUNT):
            seed = SEED_START + seed_offset
            subtype = (
                SANDPAPER_SUBTYPES[
                    seed_offset % len(SANDPAPER_SUBTYPES)
                ]
                if family == "sandpaper"
                else FAMILY_SUBTYPE[family]
            )
            index += 1
            rows.append(
                {
                    "index": index,
                    "terrain_family": family,
                    "material": family,
                    "subtype": subtype,
                    "seed": seed,
                    "name": f"{family}_{subtype}_seed_{seed}",
                }
            )
    return tuple(rows)


def _safe_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or resolved.name in {"", ".", ".."}:
        raise ValueError("output must be a specific directory")
    return resolved


def _release_cuda_memory() -> None:
    try:
        import cupy as cp  # type: ignore
    except ImportError:
        return
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _realization_id(recipe) -> str:
    return identity(
        "material_realization",
        {
            "generator_version": recipe.generator_version,
            "material": recipe.material,
            "subtype": recipe.subtype,
            "seed": recipe.seed,
            "generation_mode": recipe.generation_mode,
            "profile_hash": recipe.profile_hash,
        },
        module_version=M1_MODULE_VERSION,
    )


def _load_checkpoint(
    report_path: Path,
    *,
    design_hash: str,
    overwrite: bool,
) -> dict[int, dict[str, Any]]:
    if overwrite or not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("existing formal M1 report has an incompatible schema")
    if report.get("design_hash") != design_hash:
        raise ValueError("existing formal M1 report has a different design")
    completed: dict[int, dict[str, Any]] = {}
    for condition in report.get("conditions", []):
        index = int(condition["index"])
        if index in completed:
            raise ValueError(f"duplicate checkpoint condition index {index}")
        completed[index] = dict(condition)
    return completed


def _verify_checkpoint_condition(
    library: TerrainLibrary,
    expected: dict[str, Any],
    condition: dict[str, Any],
) -> None:
    for key in ("index", "terrain_family", "material", "subtype", "seed"):
        if condition.get(key) != expected[key]:
            raise ValueError(
                f"checkpoint condition {expected['index']} changed {key}"
            )
    mapped = library.open_region(
        str(condition["terrain_recipe_id"]),
        str(condition["region_id"]),
        verify_hash=True,
    )
    try:
        if list(mapped.shape) != list(condition["shape_yx"]):
            raise ValueError(
                f"checkpoint condition {expected['index']} shape changed"
            )
    finally:
        mapped._mmap.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/m1_material_formal_300"),
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate all 300 conditions and replace matching regions",
    )
    args = parser.parse_args(argv)

    output = _safe_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    library_root = output / "terrain_library"
    catalog_path = output / "terrain_catalog.json"
    report_path = output / "generation_report.json"
    design = formal_condition_design()
    design_hash = stable_hash(
        {
            "conditions": design,
            "size_x_m": SIZE_X_M,
            "size_y_m": SIZE_Y_M,
            "origin_x_m": ORIGIN_X_M,
            "origin_y_m": ORIGIN_Y_M,
            "resolution_m": RESOLUTION_M,
        }
    )
    completed = _load_checkpoint(
        report_path,
        design_hash=design_hash,
        overwrite=args.overwrite,
    )
    conditions: list[dict[str, Any]] = []
    library = TerrainLibrary(library_root)
    batch_started = time.perf_counter()

    def write_report(
        status: str,
        *,
        failure: BaseException | None = None,
    ) -> None:
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "updated_at_utc": utc_now(),
            "requested_backend": args.backend,
            "design_hash": design_hash,
            "library_root": str(library_root.resolve()),
            "catalog_path": str(catalog_path.resolve()),
            "condition_count": len(design),
            "completed_condition_count": len(conditions),
            "size_x_m": SIZE_X_M,
            "size_y_m": SIZE_Y_M,
            "origin_x_m": ORIGIN_X_M,
            "origin_y_m": ORIGIN_Y_M,
            "resolution_m": RESOLUTION_M,
            "shape_yx": [
                round(SIZE_Y_M / RESOLUTION_M) + 1,
                round(SIZE_X_M / RESOLUTION_M) + 1,
            ],
            "batch_wall_time_s": time.perf_counter() - batch_started,
            "conditions": conditions,
        }
        if failure is not None:
            report["failure"] = {
                "type": type(failure).__name__,
                "message": str(failure),
            }
        atomic_write_json(report_path, report)

    write_report("running")
    try:
        for expected in design:
            index = int(expected["index"])
            if index in completed and not args.overwrite:
                condition = completed[index]
                _verify_checkpoint_condition(library, expected, condition)
                conditions.append(condition)
                write_report("running")
                print(
                    json.dumps(
                        {
                            "completed": f"{index}/{len(design)}",
                            "name": condition["name"],
                            "status": "verified_checkpoint",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue

            generation_started = time.perf_counter()
            terrain = generate_terrain(
                material=str(expected["material"]),
                subtype=str(expected["subtype"]),
                size_x_m=SIZE_X_M,
                size_y_m=SIZE_Y_M,
                resolution_m=RESOLUTION_M,
                seed=int(expected["seed"]),
                mode="synthetic",
                backend=args.backend,
            )
            generation_time_s = time.perf_counter() - generation_started
            registration_started = time.perf_counter()
            recipe, region, metadata = register_terrain(
                library_root,
                terrain,
                origin_x_m=ORIGIN_X_M,
                origin_y_m=ORIGIN_Y_M,
                purpose="campaign",
                overwrite=args.overwrite,
            )
            registration_time_s = time.perf_counter() - registration_started
            mapped = library.open_region(
                recipe.terrain_recipe_id,
                region.region_id,
                verify_hash=True,
            )
            try:
                rms_height_um = float(
                    np.std(mapped, dtype=np.float64) * 1e6
                )
            finally:
                mapped._mmap.close()
            condition = {
                **expected,
                "realization_id": _realization_id(recipe),
                "resolution_level": "screening_10um",
                "profile_status": terrain.metadata["profile_status"],
                "terrain_recipe_id": recipe.terrain_recipe_id,
                "recipe_hash": recipe.recipe_hash,
                "recipe": recipe.normalized(),
                "region_id": region.region_id,
                "region": region.normalized(),
                "shape_yx": list(terrain.height.shape),
                "data_path": str(
                    library.region_data_path(
                        recipe.terrain_recipe_id,
                        region.region_id,
                    )
                ),
                "data_sha256": metadata["data_sha256"],
                "full_sha256_verified": True,
                "valid_mask_sha256": metadata["valid_mask_sha256"],
                "valid_fraction": metadata["valid_fraction"],
                "file_size_bytes": metadata["file_size_bytes"],
                "rms_height_um": rms_height_um,
                "generation_backend": metadata["generation_backend"],
                "backend": metadata["backend"],
                "generation_time_s": generation_time_s,
                "registration_time_s": registration_time_s,
            }
            conditions.append(condition)
            write_report("running")
            print(
                json.dumps(
                    {
                        "completed": f"{index}/{len(design)}",
                        "name": condition["name"],
                        "generation_backend": condition[
                            "generation_backend"
                        ],
                        "generation_time_s": generation_time_s,
                        "registration_time_s": registration_time_s,
                        "rms_height_um": rms_height_um,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del mapped
            del terrain
            gc.collect()
            _release_cuda_memory()
    except BaseException as exc:
        write_report("interrupted", failure=exc)
        raise

    catalog_id = stable_hash(
        [
            {
                "terrain_family": condition["terrain_family"],
                "subtype": condition["subtype"],
                "seed": condition["seed"],
                "realization_id": condition["realization_id"],
                "terrain_recipe_id": condition["terrain_recipe_id"],
                "region_id": condition["region_id"],
                "data_sha256": condition["data_sha256"],
            }
            for condition in conditions
        ]
    )
    catalog = {
        "schema_version": "m1-material-terrain-catalog-v1",
        "m1_module_version": M1_MODULE_VERSION,
        "terrain_catalog_id": catalog_id,
        "catalog_scope": "formal_full_size_3_family_x_100_condition_M3",
        "status": "complete",
        "created_at_utc": utc_now(),
        "library_root": str(library_root.resolve()),
        "condition_count": len(conditions),
        "all_full_hashes_verified": True,
        "requested_backend": args.backend,
        "design_hash": design_hash,
        "seed_start": SEED_START,
        "seed_count_per_family": SEED_COUNT,
        "paired_seed_policy": "same_100_seeds_in_each_material_family",
        "sandpaper_subtype_policy": (
            "cyclic_P40_P60_P100_P180_P240_20_each"
        ),
        "size_x_m": SIZE_X_M,
        "size_y_m": SIZE_Y_M,
        "origin_x_m": ORIGIN_X_M,
        "origin_y_m": ORIGIN_Y_M,
        "resolution_m": RESOLUTION_M,
        "shape_yx": [
            round(SIZE_Y_M / RESOLUTION_M) + 1,
            round(SIZE_X_M / RESOLUTION_M) + 1,
        ],
        "formal_300_complete": len(conditions) == 300,
        "formal_ranking_eligible": False,
        "limitations": [
            "Red-brick and concrete profiles remain provisional pending "
            "project measurement calibration.",
            "Material-hybrid 10/5 um same-realization convergence must be "
            "closed for final M3 candidates.",
            "M1 terrain completeness does not close M3 numerical or physical "
            "calibration gates.",
        ],
        "conditions": conditions,
    }
    atomic_write_json(catalog_path, catalog)
    write_report("complete")
    print(
        json.dumps(
            {
                "catalog": str(catalog_path),
                "report": str(report_path),
                "terrain_catalog_id": catalog_id,
                "condition_count": len(conditions),
                "batch_wall_time_s": time.perf_counter() - batch_started,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
