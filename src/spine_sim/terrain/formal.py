"""Reproducible formal-terrain batches built on the M0/M1 contracts."""

from __future__ import annotations

import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Literal, Mapping

from spine_sim.core.config import BackendConfig
from spine_sim.io.results import atomic_write_json, utc_now
from spine_sim.runtime.backend import discover_backend

from .errors import TerrainConfigurationError
from .library import TerrainLibrary
from .models import (
    M1_MODULE_VERSION,
    CampaignDesignSpace,
    TerrainRecipe,
    compute_campaign_region,
)


def _safe_name(value: Any, *, field_name: str) -> str:
    name = str(value).strip()
    if (
        not name
        or name in {".", ".."}
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in name
        )
    ):
        raise TerrainConfigurationError(f"{field_name} is empty or unsafe")
    return name


def _design_space(value: Mapping[str, Any] | None) -> CampaignDesignSpace:
    if value is None:
        return CampaignDesignSpace()
    allowed = {item.name for item in fields(CampaignDesignSpace)}
    extra = set(value) - allowed
    if extra:
        raise TerrainConfigurationError(
            f"formal terrain design_space contains unknown fields: {sorted(extra)}"
        )
    normalized = dict(value)
    for name in ("fixed_angles_deg", "gradient_angles_deg"):
        if name in normalized:
            normalized[name] = tuple(float(item) for item in normalized[name])
    return CampaignDesignSpace(**normalized)


def generate_formal_terrain_batch(
    library_root: str | Path,
    batch: Mapping[str, Any],
    *,
    output_path: str | Path,
    tile_rows: int = 256,
    backend: Literal["cpu", "cuda"] = "cuda",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate a resumable batch of full 2-D terrains and finite-tip tracks."""

    allowed = {
        "schema_version",
        "batch_name",
        "description",
        "base_recipe",
        "seed_values",
        "round1_seed_count",
        "design_space",
        "track_radii_m",
        "track_y_global_m",
    }
    extra = set(batch) - allowed
    if extra:
        raise TerrainConfigurationError(
            f"formal terrain batch contains unknown fields: {sorted(extra)}"
        )
    if str(batch.get("schema_version")) != "1":
        raise TerrainConfigurationError(
            "formal terrain batch schema_version must be '1'"
        )
    batch_name = _safe_name(batch.get("batch_name"), field_name="batch_name")
    seeds = tuple(int(seed) for seed in batch.get("seed_values", ()))
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise TerrainConfigurationError(
            "seed_values must be a non-empty list of unique non-negative integers"
        )
    round1_seed_count = int(batch.get("round1_seed_count", len(seeds)))
    if round1_seed_count < 1 or round1_seed_count > len(seeds):
        raise TerrainConfigurationError(
            "round1_seed_count must lie between one and the total seed count"
        )
    radii = tuple(float(value) for value in batch.get("track_radii_m", (50e-6, 100e-6)))
    track_y_values = tuple(
        float(value) for value in batch.get("track_y_global_m", (0.0,))
    )
    if not radii or any(value <= 0 for value in radii):
        raise TerrainConfigurationError("track_radii_m must contain positive values")
    if not track_y_values:
        raise TerrainConfigurationError("track_y_global_m cannot be empty")
    if tile_rows < 1:
        raise TerrainConfigurationError("tile_rows must be positive")
    if backend not in {"cpu", "cuda"}:
        raise TerrainConfigurationError("backend must be cpu or cuda")

    base_recipe = dict(batch.get("base_recipe", {}))
    base_recipe.pop("seed", None)
    design = _design_space(batch.get("design_space"))
    library = TerrainLibrary(Path(library_root).resolve())
    target = Path(output_path).resolve()
    backend_record = discover_backend(BackendConfig(preference=backend)).as_dict()
    started = time.perf_counter()
    conditions: list[dict[str, Any]] = []

    def report(status: str, *, failure: BaseException | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": "1",
            "m1_module_version": M1_MODULE_VERSION,
            "batch_name": batch_name,
            "description": str(batch.get("description", "")),
            "status": status,
            "created_at_utc": utc_now(),
            "generation_backend": backend,
            "backend": backend_record,
            "tile_rows": tile_rows,
            "library_root": str(library.root),
            "seed_count": len(seeds),
            "seed_values": list(seeds),
            "round1_seed_count": round1_seed_count,
            "round1_seed_values": list(seeds[:round1_seed_count]),
            "round2_seed_count": len(seeds) - round1_seed_count,
            "round2_reserve_seed_values": list(seeds[round1_seed_count:]),
            "completed_seed_count": len(conditions),
            "design_space": asdict(design),
            "track_radii_m": list(radii),
            "track_y_global_m": list(track_y_values),
            "conditions": conditions,
            "total_file_size_bytes": sum(
                int(item["file_size_bytes"]) for item in conditions
            ),
            "unique_data_hash_count": len(
                {str(item["data_sha256"]) for item in conditions}
            ),
            "all_full_hashes_verified": all(
                bool(item["full_sha256_verified"]) for item in conditions
            ),
            "batch_wall_time_s": time.perf_counter() - started,
            "m0_contract": {
                "recipe_record": "m0_terrain_recipe_ref",
                "region_record": "m0_region_spec",
                "binary_storage_owner": "M1 TerrainLibrary",
            },
        }
        if failure is not None:
            result["failure"] = {
                "type": type(failure).__name__,
                "message": str(failure),
            }
        atomic_write_json(target, result)
        return result

    report("running")
    try:
        for index, seed in enumerate(seeds, start=1):
            recipe = TerrainRecipe.from_mapping({**base_recipe, "seed": seed})
            campaign = compute_campaign_region(recipe, design)
            metadata = library.generate_region(
                recipe,
                campaign.region,
                tile_rows=tile_rows,
                backend=backend,
                overwrite=overwrite,
            )
            mapped = library.open_region(
                recipe.terrain_recipe_id,
                campaign.region.region_id,
                verify_hash=True,
            )
            mapped._mmap.close()
            del mapped
            tracks = [
                library.cache_track(
                    recipe,
                    campaign.region,
                    radius_m=radius_m,
                    y_global_m=y_global_m,
                    overwrite=overwrite,
                )
                for radius_m in radii
                for y_global_m in track_y_values
            ]
            conditions.append(
                {
                    "index": index,
                    "name": f"formal_seed_{seed}",
                    "description": "Independent paired baseline-terrain realization.",
                    "seed": seed,
                    "screening_round": (
                        "round1" if index <= round1_seed_count else "round2_reserve"
                    ),
                    "terrain_recipe_id": recipe.terrain_recipe_id,
                    "recipe_hash": recipe.recipe_hash,
                    "recipe": recipe.normalized(),
                    "m0_terrain_recipe_ref": asdict(recipe.to_m0_ref()),
                    "region_id": campaign.region.region_id,
                    "region": campaign.region.normalized(),
                    "m0_region_spec": asdict(campaign.region.to_m0_spec()),
                    "region_shape": list(campaign.region.shape),
                    "data_path": str(
                        library.region_data_path(
                            recipe.terrain_recipe_id, campaign.region.region_id
                        )
                    ),
                    "data_sha256": metadata["data_sha256"],
                    "file_size_bytes": metadata["file_size_bytes"],
                    "generation_time_s": metadata["generation_time_s"],
                    "generation_backend": metadata["generation_backend"],
                    "gpu_memory_pool_peak_cached_bytes": metadata.get(
                        "gpu_memory_pool_peak_cached_bytes"
                    ),
                    "full_sha256_verified": True,
                    "track_ids": [track.track_id for track in tracks],
                    "tracks": [
                        {
                            "track_id": track.track_id,
                            "radius_m": track.radius_m,
                            "y_global_m": track.y_global_m,
                            "sample_count": int(track.x_global_m.size),
                            "valid_count": int(track.valid_mask.sum()),
                        }
                        for track in tracks
                    ],
                }
            )
            report("running")
    except BaseException as exc:
        report("interrupted", failure=exc)
        raise
    return report("complete")
