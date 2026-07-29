"""Command-line tools for M1 region, track, cache and benchmark workflows."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from spine_sim.io.results import atomic_write_json

from .api import generate_terrain, register_terrain, save_terrain
from .errors import GeometryOutOfDomainError, TerrainConfigurationError
from .formal import generate_formal_terrain_batch
from .library import TerrainLibrary
from .models import (
    RegionSpec,
    TerrainRecipe,
    compute_campaign_region,
)
from .plotting import render_terrain_views
from .profiles import available_profiles
from .suite import generate_terrain_suite, load_suite
from .validation import (
    compare_topographies,
    render_comparison,
    summarize_seed_ensemble,
    write_validation_json,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_recipe(path: Path | None) -> TerrainRecipe:
    if path is None:
        return TerrainRecipe()
    document = _read_json(path)
    return TerrainRecipe.from_mapping(document.get("recipe", document))


def _load_region(path: Path) -> RegionSpec:
    document = _read_json(path)
    return RegionSpec.from_mapping(document.get("region", document))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spine-terrain")
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("region-report")
    report.add_argument("--recipe", type=Path)
    report.add_argument("--output", type=Path)

    generate = sub.add_parser("generate-region")
    generate.add_argument("library", type=Path)
    generate.add_argument("recipe", type=Path)
    generate.add_argument("region", type=Path)
    generate.add_argument("--tile-rows", type=int, default=64)
    generate.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    generate.add_argument("--overwrite", action="store_true")

    track = sub.add_parser("generate-track")
    track.add_argument("library", type=Path)
    track.add_argument("terrain_recipe_id")
    track.add_argument("region_id")
    track.add_argument("--radius-um", type=float, required=True, choices=(50.0, 100.0))
    track.add_argument("--y-mm", type=float, required=True)
    track.add_argument("--overwrite", action="store_true")

    delete = sub.add_parser("delete-cache")
    delete.add_argument("library", type=Path)
    delete.add_argument("terrain_recipe_id")
    delete.add_argument("region_id")
    delete.add_argument("--keep-tracks", action="store_true")

    rebuild = sub.add_parser("rebuild-region")
    rebuild.add_argument("library", type=Path)
    rebuild.add_argument("terrain_recipe_id")
    rebuild.add_argument("region_id")
    rebuild.add_argument("--tile-rows", type=int, default=64)
    rebuild.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--library", type=Path)
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--tile-rows", type=int, default=32)

    suite = sub.add_parser("generate-suite")
    suite.add_argument("library", type=Path)
    suite.add_argument("suite", type=Path)
    suite.add_argument("--output", type=Path)
    suite.add_argument("--tile-rows", type=int, default=64)
    suite.add_argument("--overwrite", action="store_true")

    formal = sub.add_parser("generate-formal-batch")
    formal.add_argument("library", type=Path)
    formal.add_argument("batch", type=Path)
    formal.add_argument("output", type=Path)
    formal.add_argument("--tile-rows", type=int, default=256)
    formal.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    formal.add_argument("--overwrite", action="store_true")

    plot = sub.add_parser("plot-region")
    plot.add_argument("library", type=Path)
    plot.add_argument("terrain_recipe_id")
    plot.add_argument("region_id")
    plot.add_argument("output", type=Path)
    plot.add_argument("--center-x-mm", type=float)
    plot.add_argument("--center-y-mm", type=float)
    plot.add_argument("--overview-size-mm", type=float, default=10.0)
    plot.add_argument("--sphere-radius-um", type=float, default=100.0)
    plot.add_argument("--overview-max-points", type=int, default=1201)
    plot.add_argument("--surface-max-points", type=int, default=181)
    plot.add_argument("--sphere-transparency", type=float, default=0.0)
    plot.add_argument("--dpi", type=int, default=180)
    plot.add_argument("--prefix", default="terrain")

    sub.add_parser("list-materials")

    material = sub.add_parser("generate-material")
    material.add_argument("output", type=Path)
    material.add_argument(
        "--material", required=True, choices=("sandpaper", "red_brick", "concrete")
    )
    material.add_argument("--subtype")
    material.add_argument("--size-x-mm", type=float, required=True)
    material.add_argument("--size-y-mm", type=float, required=True)
    material.add_argument("--resolution-um", type=float, required=True)
    material.add_argument("--seed", type=int, required=True)
    material.add_argument(
        "--mode", choices=("measured", "synthetic", "auto"), default="synthetic"
    )
    material.add_argument(
        "--backend", choices=("cpu", "cuda"), default="cpu"
    )
    material.add_argument("--measured-path", type=Path)
    material.add_argument("--library", type=Path)
    material.add_argument("--origin-x-mm", type=float, default=0.0)
    material.add_argument("--origin-y-mm", type=float, default=0.0)
    material.add_argument("--overwrite", action="store_true")

    validate_material = sub.add_parser("validate-material")
    validate_material.add_argument("output", type=Path)
    validate_material.add_argument(
        "--material", required=True, choices=("sandpaper", "red_brick", "concrete")
    )
    validate_material.add_argument("--subtype")
    validate_material.add_argument("--size-x-mm", type=float, default=2.0)
    validate_material.add_argument("--size-y-mm", type=float, default=1.0)
    validate_material.add_argument("--resolution-um", type=float, default=10.0)
    validate_material.add_argument(
        "--seed", action="append", type=int, dest="seeds"
    )
    validate_material.add_argument("--measured-path", type=Path)
    validate_material.add_argument(
        "--backend", choices=("cpu", "cuda"), default="cpu"
    )
    return parser


def _benchmark(library_root: Path, *, tile_rows: int) -> dict[str, Any]:
    library = TerrainLibrary(library_root)
    recipe = TerrainRecipe(seed=20260727)
    region = RegionSpec(
        terrain_recipe_id=recipe.terrain_recipe_id,
        origin_x_m=-1e-3,
        origin_y_m=-0.5e-3,
        size_x_m=2e-3,
        size_y_m=1e-3,
        purpose="debug",
    )
    generated = library.generate_region(
        recipe, region, tile_rows=tile_rows, overwrite=True
    )
    mapped = library.open_region(
        recipe.terrain_recipe_id, region.region_id, verify_hash=True
    )
    read_started = time.perf_counter()
    checksum = float(np.sum(mapped, dtype=np.float64))
    read_seconds = time.perf_counter() - read_started
    throughput = mapped.nbytes / max(read_seconds, np.finfo(float).eps)
    mmap_type = type(mapped).__name__
    mapped._mmap.close()
    del mapped
    track_started = time.perf_counter()
    track = library.cache_track(
        recipe,
        region,
        radius_m=50e-6,
        y_global_m=0.0,
        overwrite=True,
    )
    track_seconds = time.perf_counter() - track_started
    original_hash = generated["data_sha256"]
    deletion = library.delete_region_cache(
        recipe.terrain_recipe_id, region.region_id, include_tracks=False
    )
    rebuilt = library.rebuild_region(
        recipe.terrain_recipe_id,
        region.region_id,
        tile_rows=tile_rows,
    )
    campaign = compute_campaign_region(recipe)
    return {
        "benchmark_kind": "small_debug_region_plus_campaign_size_projection",
        "recipe_id": recipe.terrain_recipe_id,
        "region_id": region.region_id,
        "region_shape": list(region.shape),
        "region_file_size_bytes": generated["file_size_bytes"],
        "generation_time_s": generated["generation_time_s"],
        "estimated_tile_peak_bytes": generated["estimated_tile_peak_bytes"],
        "mmap_type": mmap_type,
        "mmap_read_time_s": read_seconds,
        "mmap_read_throughput_bytes_s": throughput,
        "mmap_checksum": checksum,
        "track_generation_time_s": track_seconds,
        "track_sample_count": int(track.x_global_m.size),
        "track_valid_count": int(np.count_nonzero(track.valid_mask)),
        "cache_delete": deletion,
        "rebuild_time_s": rebuilt["generation_time_s"],
        "rebuild_hash_equal": rebuilt["data_sha256"] == original_hash,
        "campaign_region": campaign.as_dict(),
        "campaign_seed_payload_bytes": campaign.region.expected_npy_payload_bytes,
        "projected_disk_bytes": {
            "30_seeds": 30 * campaign.region.expected_npy_payload_bytes,
            "50_seeds": 50 * campaign.region.expected_npy_payload_bytes,
            "80_seeds": 80 * campaign.region.expected_npy_payload_bytes,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-materials":
        print(json.dumps(available_profiles(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "generate-material":
        terrain = generate_terrain(
            material=args.material,
            subtype=args.subtype,
            size_x_m=args.size_x_mm * 1e-3,
            size_y_m=args.size_y_mm * 1e-3,
            resolution_m=args.resolution_um * 1e-6,
            seed=args.seed,
            mode=args.mode,
            measured_path=args.measured_path,
            backend=args.backend,
        )
        artifact = save_terrain(args.output, terrain)
        result: dict[str, Any] = {
            "artifact": str(artifact),
            "material": terrain.material,
            "subtype": terrain.subtype,
            "profile_status": terrain.metadata["profile_status"],
            "resolved_mode": terrain.resolved_mode,
            "generation_backend": terrain.metadata["generation_backend"],
            "shape_yx": list(terrain.height.shape),
            "dtype": str(terrain.height.dtype),
        }
        if args.library is not None:
            recipe, region, metadata = register_terrain(
                args.library,
                terrain,
                origin_x_m=args.origin_x_mm * 1e-3,
                origin_y_m=args.origin_y_mm * 1e-3,
                overwrite=args.overwrite,
            )
            result["library"] = {
                "root": str(args.library.resolve()),
                "terrain_recipe_id": recipe.terrain_recipe_id,
                "region_id": region.region_id,
                "data_sha256": metadata["data_sha256"],
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-material":
        seeds = args.seeds or [1101, 1102, 1103]
        terrains = [
            generate_terrain(
                material=args.material,
                subtype=args.subtype,
                size_x_m=args.size_x_mm * 1e-3,
                size_y_m=args.size_y_mm * 1e-3,
                resolution_m=args.resolution_um * 1e-6,
                seed=seed,
                mode="synthetic",
                measured_path=args.measured_path,
                backend=args.backend,
            )
            for seed in seeds
        ]
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        ensemble_path = write_validation_json(
            output / "seed_ensemble.json", summarize_seed_ensemble(terrains)
        )
        reference = None
        try:
            reference = generate_terrain(
                material=args.material,
                subtype=args.subtype,
                size_x_m=args.size_x_mm * 1e-3,
                size_y_m=args.size_y_mm * 1e-3,
                resolution_m=args.resolution_um * 1e-6,
                seed=seeds[0] + 10_000,
                mode="measured",
                measured_path=args.measured_path,
            )
        except (FileNotFoundError, GeometryOutOfDomainError, TerrainConfigurationError):
            reference = None
        figure_path = render_comparison(
            terrains[0],
            output / "geometry_comparison.png",
            reference_height_m=None if reference is None else reference.height,
            reference_dx_m=None if reference is None else reference.dx,
            reference_dy_m=None if reference is None else reference.dy,
            reference_valid_mask=(
                None if reference is None else reference.valid_mask
            ),
            reference_label="measured crop (single source)",
        )
        comparison_path = None
        if reference is not None:
            comparison_path = write_validation_json(
                output / "measured_vs_synthetic.json",
                compare_topographies(
                    reference.height,
                    terrains[0].height,
                    reference_dx_m=reference.dx,
                    reference_dy_m=reference.dy,
                    synthetic_dx_m=terrains[0].dx,
                    synthetic_dy_m=terrains[0].dy,
                    reference_valid_mask=reference.valid_mask,
                    synthetic_valid_mask=terrains[0].valid_mask,
                ),
            )
        print(
            json.dumps(
                {
                    "ensemble": str(ensemble_path),
                    "figure": str(figure_path),
                    "comparison": (
                        None if comparison_path is None else str(comparison_path)
                    ),
                    "reference_available": reference is not None,
                    "profile_status": terrains[0].metadata["profile_status"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "region-report":
        result = compute_campaign_region(_load_recipe(args.recipe)).as_dict()
        if args.output:
            atomic_write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "generate-region":
        result = TerrainLibrary(args.library).generate_region(
            _load_recipe(args.recipe),
            _load_region(args.region),
            tile_rows=args.tile_rows,
            backend=args.backend,
            overwrite=args.overwrite,
        )
    elif args.command == "generate-track":
        library = TerrainLibrary(args.library)
        recipe = library.load_recipe(args.terrain_recipe_id)
        region = library.load_region_spec(args.terrain_recipe_id, args.region_id)
        track = library.cache_track(
            recipe,
            region,
            radius_m=args.radius_um * 1e-6,
            y_global_m=args.y_mm * 1e-3,
            overwrite=args.overwrite,
        )
        result = {
            "track_id": track.track_id,
            "sample_count": int(track.x_global_m.size),
            "valid_count": int(np.count_nonzero(track.valid_mask)),
            "model_warning": list(track.model_warning),
        }
    elif args.command == "delete-cache":
        result = TerrainLibrary(args.library).delete_region_cache(
            args.terrain_recipe_id,
            args.region_id,
            include_tracks=not args.keep_tracks,
        )
    elif args.command == "rebuild-region":
        result = TerrainLibrary(args.library).rebuild_region(
            args.terrain_recipe_id,
            args.region_id,
            tile_rows=args.tile_rows,
            backend=args.backend,
        )
    elif args.command == "benchmark":
        if args.library:
            result = _benchmark(args.library, tile_rows=args.tile_rows)
        else:
            with tempfile.TemporaryDirectory() as temporary:
                result = _benchmark(Path(temporary), tile_rows=args.tile_rows)
        if args.output:
            atomic_write_json(args.output, result)
    elif args.command == "generate-suite":
        result = generate_terrain_suite(
            args.library,
            load_suite(args.suite),
            tile_rows=args.tile_rows,
            overwrite=args.overwrite,
        )
        if args.output:
            atomic_write_json(args.output, result)
    elif args.command == "generate-formal-batch":
        result = generate_formal_terrain_batch(
            args.library,
            _read_json(args.batch),
            output_path=args.output,
            tile_rows=args.tile_rows,
            backend=args.backend,
            overwrite=args.overwrite,
        )
    elif args.command == "plot-region":
        result = render_terrain_views(
            args.library,
            args.terrain_recipe_id,
            args.region_id,
            args.output,
            center_x_m=(
                None if args.center_x_mm is None else args.center_x_mm * 1e-3
            ),
            center_y_m=(
                None if args.center_y_mm is None else args.center_y_mm * 1e-3
            ),
            overview_size_m=args.overview_size_mm * 1e-3,
            sphere_radius_m=args.sphere_radius_um * 1e-6,
            overview_maximum_axis_points=args.overview_max_points,
            surface_maximum_axis_points=args.surface_max_points,
            sphere_transparency=args.sphere_transparency,
            dpi=args.dpi,
            prefix=args.prefix,
        )
    else:
        raise AssertionError(args.command)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
