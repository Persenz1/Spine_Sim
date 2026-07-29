"""Generate the full-size 15-condition M1 material library for M3 tests."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from spine_sim.core.identity import stable_hash
from spine_sim.io.results import atomic_write_json, utc_now
from spine_sim.terrain import (
    M1_MODULE_VERSION,
    TerrainLibrary,
    generate_terrain,
    register_terrain,
)


CONDITIONS = (
    ("sandpaper", "P40", 41001),
    ("sandpaper", "P60", 41002),
    ("sandpaper", "P100", 41003),
    ("sandpaper", "P180", 41004),
    ("sandpaper", "P240", 41005),
    ("red_brick", "fired_brick_standard", 41001),
    ("red_brick", "fired_brick_standard", 41002),
    ("red_brick", "fired_brick_standard", 41003),
    ("red_brick", "fired_brick_standard", 41004),
    ("red_brick", "fired_brick_standard", 41005),
    ("concrete", "rough_wall", 41001),
    ("concrete", "rough_wall", 41002),
    ("concrete", "rough_wall", 41003),
    ("concrete", "rough_wall", 41004),
    ("concrete", "rough_wall", 41005),
)

SIZE_X_M = 147.960e-3
SIZE_Y_M = 40.200e-3
ORIGIN_X_M = -21.980e-3
ORIGIN_Y_M = -20.100e-3
RESOLUTION_M = 10e-6


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/m1_material_m3_test"),
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    output = _safe_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    library_root = output / "terrain_library"
    catalog_path = output / "terrain_catalog.json"
    report_path = output / "generation_report.json"
    conditions: list[dict[str, Any]] = []
    batch_started = time.perf_counter()

    def write_report(
        status: str,
        *,
        failure: BaseException | None = None,
    ) -> None:
        report: dict[str, Any] = {
            "schema_version": "m1-material-m3-test-batch-v1",
            "status": status,
            "updated_at_utc": utc_now(),
            "requested_backend": args.backend,
            "library_root": str(library_root.resolve()),
            "catalog_path": str(catalog_path.resolve()),
            "condition_count": len(CONDITIONS),
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
        for index, (material, subtype, seed) in enumerate(CONDITIONS, start=1):
            generation_started = time.perf_counter()
            terrain = generate_terrain(
                material=material,
                subtype=subtype,
                size_x_m=SIZE_X_M,
                size_y_m=SIZE_Y_M,
                resolution_m=RESOLUTION_M,
                seed=seed,
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
            mapped = TerrainLibrary(library_root).open_region(
                recipe.terrain_recipe_id,
                region.region_id,
                verify_hash=True,
            )
            rms_height_um = float(
                np.std(mapped, dtype=np.float64) * 1e6
            )
            mapped._mmap.close()
            del mapped
            condition = {
                "index": index,
                "name": f"{material}_{subtype}_seed_{seed}",
                "terrain_family": material,
                "material": material,
                "subtype": subtype,
                "seed": seed,
                "profile_status": terrain.metadata["profile_status"],
                "terrain_recipe_id": recipe.terrain_recipe_id,
                "recipe_hash": recipe.recipe_hash,
                "recipe": recipe.normalized(),
                "region_id": region.region_id,
                "region": region.normalized(),
                "shape_yx": list(terrain.height.shape),
                "data_path": str(
                    TerrainLibrary(library_root).region_data_path(
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
                        "completed": f"{index}/{len(CONDITIONS)}",
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
        "catalog_scope": "full_size_15_condition_M3_improvement_test",
        "status": "complete",
        "created_at_utc": utc_now(),
        "library_root": str(library_root.resolve()),
        "condition_count": len(conditions),
        "all_full_hashes_verified": True,
        "requested_backend": args.backend,
        "size_x_m": SIZE_X_M,
        "size_y_m": SIZE_Y_M,
        "origin_x_m": ORIGIN_X_M,
        "origin_y_m": ORIGIN_Y_M,
        "resolution_m": RESOLUTION_M,
        "shape_yx": [
            round(SIZE_Y_M / RESOLUTION_M) + 1,
            round(SIZE_X_M / RESOLUTION_M) + 1,
        ],
        "formal_300_complete": False,
        "formal_ranking_eligible": False,
        "limitations": [
            "This is a 15-condition M3 implementation-test catalog, not the "
            "required 3-family x 100-condition formal catalog.",
            "Red-brick and concrete profiles remain provisional pending "
            "project measurement calibration.",
            "Material-hybrid 10/5 um same-realization convergence is not yet "
            "closed.",
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
