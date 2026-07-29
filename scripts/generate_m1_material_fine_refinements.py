"""Generate nested 5 um refinements for selected formal M1 conditions."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from spine_sim.core.identity import stable_hash
from spine_sim.io.results import atomic_write_json, utc_now
from spine_sim.terrain import (
    M1_MODULE_VERSION,
    Terrain,
    TerrainLibrary,
    generate_terrain,
    refine_material_terrain_same_realization,
    register_terrain,
)


FINE_RESOLUTION_M = 5e-6


def _streaming_rms_um(height: np.ndarray, block_rows: int = 128) -> float:
    count = 0
    total = 0.0
    total_square = 0.0
    for start in range(0, height.shape[0], block_rows):
        block = np.asarray(
            height[start : start + block_rows],
            dtype=np.float64,
        )
        count += block.size
        total += float(np.sum(block, dtype=np.float64))
        total_square += float(
            np.sum(block * block, dtype=np.float64)
        )
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    return variance**0.5 * 1e6


def _verify_nested_nodes(
    coarse: np.ndarray,
    fine: np.ndarray,
    *,
    block_rows: int = 128,
) -> None:
    if fine.shape != (
        2 * coarse.shape[0] - 1,
        2 * coarse.shape[1] - 1,
    ):
        raise ValueError("registered fine terrain has the wrong nested shape")
    for start in range(0, coarse.shape[0], block_rows):
        stop = min(start + block_rows, coarse.shape[0])
        if not np.array_equal(
            np.asarray(fine[2 * start : 2 * stop : 2, ::2]),
            np.asarray(coarse[start:stop]),
        ):
            raise ValueError(
                "registered fine terrain does not preserve coarse nodes"
            )


def _selected_conditions(
    catalog: dict[str, Any],
    *,
    families: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    selected = [
        dict(condition)
        for condition in catalog["conditions"]
        if condition["terrain_family"] in families
        and int(condition["seed"]) in seeds
    ]
    expected = len(families) * len(seeds)
    if len(selected) != expected:
        raise ValueError(
            f"expected {expected} selected formal conditions, got "
            f"{len(selected)}"
        )
    selected.sort(
        key=lambda item: (
            families.index(item["terrain_family"]),
            seeds.index(int(item["seed"])),
        )
    )
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument(
        "--terrain-family",
        action="append",
        choices=("sandpaper", "red_brick", "concrete"),
        dest="families",
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    catalog_path = args.catalog.resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if (
        catalog.get("status") != "complete"
        or catalog.get("formal_300_complete") is not True
        or len(catalog.get("conditions", [])) != 300
    ):
        raise ValueError("fine refinement requires the complete formal catalog")
    families = tuple(
        dict.fromkeys(
            args.families
            or ("sandpaper", "red_brick", "concrete")
        )
    )
    seeds = tuple(dict.fromkeys(args.seeds or (41_001,)))
    parents = _selected_conditions(
        catalog,
        families=families,
        seeds=seeds,
    )
    library = TerrainLibrary(catalog["library_root"])
    output_path = (
        args.output.resolve()
        if args.output is not None
        else catalog_path.with_name("fine_refinement_catalog.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for index, parent in enumerate(parents, start=1):
        recipe_id = str(parent["terrain_recipe_id"])
        region_id = str(parent["region_id"])
        recipe = library.load_recipe(recipe_id)
        region = library.load_region_spec(recipe_id, region_id)
        manifest = json.loads(
            library.region_manifest_path(recipe_id, region_id).read_text(
                encoding="utf-8"
            )
        )
        if manifest.get("data_sha256") != parent["data_sha256"]:
            raise ValueError("parent formal terrain hash changed")
        coarse_height = library.open_region(
            recipe_id,
            region_id,
            verify_hash=True,
        )
        coarse_mask = np.load(
            library.region_dir(recipe_id, region_id) / "valid_mask.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        coarse = Terrain(
            height=coarse_height,
            dx=region.resolution_x_m,
            dy=region.resolution_y_m,
            valid_mask=coarse_mask,
            material=str(parent["material"]),
            subtype=str(parent["subtype"]),
            seed=int(parent["seed"]),
            metadata=manifest["material_metadata"],
        )
        fine_detail = generate_terrain(
            material=coarse.material,
            subtype=coarse.subtype,
            size_x_m=region.size_x_m,
            size_y_m=region.size_y_m,
            resolution_m=FINE_RESOLUTION_M,
            seed=coarse.seed,
            mode="synthetic",
            backend=args.backend,
        )
        fine = refine_material_terrain_same_realization(
            coarse,
            fine_detail,
        )
        del fine_detail
        gc.collect()
        fine_recipe, fine_region, fine_metadata = register_terrain(
            catalog["library_root"],
            fine,
            origin_x_m=region.origin_x_m,
            origin_y_m=region.origin_y_m,
            purpose="campaign",
            overwrite=args.overwrite,
        )
        registered_fine = library.open_region(
            fine_recipe.terrain_recipe_id,
            fine_region.region_id,
            verify_hash=True,
        )
        try:
            _verify_nested_nodes(coarse_height, registered_fine)
            fine_rms_height_um = _streaming_rms_um(registered_fine)
        finally:
            registered_fine._mmap.close()
        row = {
            "index": index,
            "name": parent["name"] + "_nested_5um",
            "terrain_family": parent["terrain_family"],
            "material": parent["material"],
            "subtype": parent["subtype"],
            "seed": parent["seed"],
            "realization_id": parent["realization_id"],
            "resolution_level": "fine_5um",
            "parent_resolution_level": parent["resolution_level"],
            "parent_terrain_recipe_id": recipe_id,
            "parent_region_id": region_id,
            "parent_data_sha256": parent["data_sha256"],
            "terrain_recipe_id": fine_recipe.terrain_recipe_id,
            "recipe_hash": fine_recipe.recipe_hash,
            "recipe": fine_recipe.normalized(),
            "region_id": fine_region.region_id,
            "region": fine_region.normalized(),
            "shape_yx": list(fine.height.shape),
            "data_path": str(
                library.region_data_path(
                    fine_recipe.terrain_recipe_id,
                    fine_region.region_id,
                )
            ),
            "data_sha256": fine_metadata["data_sha256"],
            "full_sha256_verified": True,
            "valid_mask_sha256": fine_metadata["valid_mask_sha256"],
            "valid_fraction": fine_metadata["valid_fraction"],
            "file_size_bytes": fine_metadata["file_size_bytes"],
            "rms_height_um": fine_rms_height_um,
            "generation_backend": fine_metadata["generation_backend"],
            "backend": fine_metadata["backend"],
            "refinement_algorithm": (
                fine.metadata["same_realization_refinement"]["algorithm"]
            ),
            "coarse_node_identity_exact": True,
        }
        rows.append(row)
        atomic_write_json(
            output_path,
            {
                "schema_version": (
                    "m1-material-fine-refinement-catalog-v1"
                ),
                "m1_module_version": M1_MODULE_VERSION,
                "status": "running",
                "created_at_utc": utc_now(),
                "parent_catalog_path": str(catalog_path),
                "parent_terrain_catalog_id": catalog[
                    "terrain_catalog_id"
                ],
                "library_root": str(
                    Path(catalog["library_root"]).resolve()
                ),
                "condition_count": len(rows),
                "all_full_hashes_verified": True,
                "all_coarse_nodes_exact": True,
                "conditions": rows,
            },
        )
        print(
            json.dumps(
                {
                    "completed": f"{index}/{len(parents)}",
                    "name": row["name"],
                    "file_size_bytes": row["file_size_bytes"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        coarse_height._mmap.close()
        del coarse_mask
        del coarse
        del fine
        gc.collect()

    result = {
        "schema_version": "m1-material-fine-refinement-catalog-v1",
        "m1_module_version": M1_MODULE_VERSION,
        "fine_refinement_catalog_id": stable_hash(
            [
                {
                    "realization_id": row["realization_id"],
                    "parent_data_sha256": row["parent_data_sha256"],
                    "data_sha256": row["data_sha256"],
                }
                for row in rows
            ]
        ),
        "status": "complete",
        "created_at_utc": utc_now(),
        "parent_catalog_path": str(catalog_path),
        "parent_terrain_catalog_id": catalog["terrain_catalog_id"],
        "library_root": str(Path(catalog["library_root"]).resolve()),
        "condition_count": len(rows),
        "all_full_hashes_verified": True,
        "all_coarse_nodes_exact": True,
        "conditions": rows,
    }
    atomic_write_json(output_path, result)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "condition_count": len(rows),
                "all_coarse_nodes_exact": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
