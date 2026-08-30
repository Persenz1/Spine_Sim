"""GPU 地形套件生成，以及 CPU/GPU 重叠区域一致性验证。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.io.files import atomic_write_json, utc_now
from spine_sim.runtime.backend import BackendConfig, discover_backend

from .errors import TerrainConfigurationError
from .library import TerrainLibrary
from .models import (
    M1_MODULE_VERSION,
    RegionSpec,
    TerrainRecipe,
    compute_campaign_region,
)
from .random_field import generate_defined_geometry


def _cuda_record() -> dict[str, Any]:
    """记录实际 CUDA/CuPy/设备版本与显存能力。"""

    capabilities = discover_backend(BackendConfig(preference="cuda"))
    try:
        import cupy as cp  # type: ignore
    except ImportError as exc:
        raise TerrainConfigurationError(
            "GPU suite requires CuPy in the active environment"
        ) from exc
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    raw_device_name = properties["name"]
    device_name = (
        raw_device_name.decode()
        if isinstance(raw_device_name, bytes)
        else str(raw_device_name)
    )
    return {
        "backend": capabilities.as_dict(),
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "device_id": int(device.id),
        "device_name": device_name,
        "compute_capability": device.compute_capability,
        "total_global_memory_bytes": int(properties["totalGlobalMem"]),
    }


def _overlap_check(
    recipe: TerrainRecipe,
    *,
    size_x_m: float,
    size_y_m: float,
) -> dict[str, Any]:
    """在小型同域区域比较 CPU/GPU defined geometry 的误差。"""

    region = RegionSpec(
        terrain_recipe_id=recipe.terrain_recipe_id,
        origin_x_m=0.0,
        origin_y_m=0.0,
        size_x_m=size_x_m,
        size_y_m=size_y_m,
        resolution_x_m=recipe.production_dx_m,
        resolution_y_m=recipe.production_dy_m,
        purpose="debug",
    )
    cpu_started = time.perf_counter()
    cpu = generate_defined_geometry(recipe, region, backend="cpu")
    cpu_time = time.perf_counter() - cpu_started
    gpu_started = time.perf_counter()
    gpu = generate_defined_geometry(recipe, region, backend="cuda")
    gpu_time = time.perf_counter() - gpu_started
    absolute = np.abs(cpu.astype(np.float64) - gpu.astype(np.float64))
    denominator = np.maximum(np.abs(cpu.astype(np.float64)), 1e-12)
    result = {
        "region_id": region.region_id,
        "shape": list(region.shape),
        "cpu_time_s": cpu_time,
        "gpu_time_s": gpu_time,
        "max_abs_error_m": float(absolute.max(initial=0.0)),
        "max_relative_error": float(
            np.max(absolute / denominator, initial=0.0)
        ),
        "exact_float32_equal": bool(np.array_equal(cpu, gpu)),
        "tolerance_abs_m": 1e-10,
        "tolerance_relative": 2e-6,
    }
    result["passed"] = bool(
        np.allclose(cpu, gpu, rtol=result["tolerance_relative"], atol=result["tolerance_abs_m"])
    )
    return result


def _region_statistics(
    library: TerrainLibrary,
    recipe: TerrainRecipe,
    region: RegionSpec,
) -> dict[str, Any]:
    """分块扫描只读 memmap，计算完整 region 的基础统计量。"""

    height = library.open_region(
        recipe.terrain_recipe_id,
        region.region_id,
        verify_hash=True,
    )
    try:
        count = int(height.size)
        total = 0.0
        total_squared = 0.0
        minimum = float("inf")
        maximum = float("-inf")
        # 分块累加一、二阶矩，避免把完整 campaign region 转成 float64 副本。
        for row_start in range(0, height.shape[0], 64):
            block = np.asarray(
                height[row_start : row_start + 64], dtype=np.float64
            )
            total += float(block.sum())
            total_squared += float(np.square(block).sum())
            minimum = min(minimum, float(block.min()))
            maximum = max(maximum, float(block.max()))
        mean = total / count
        mean_square = total_squared / count
        return {
            "sample_count": count,
            "mean_m": mean,
            "rms_about_zero_m": float(np.sqrt(mean_square)),
            "std_m": float(np.sqrt(max(0.0, mean_square - mean * mean))),
            "min_m": minimum,
            "max_m": maximum,
            "memory_map_type": type(height).__name__,
            "memory_map_read_only": not bool(height.flags.writeable),
            "full_sha256_verified": True,
        }
    finally:
        height._mmap.close()


def generate_terrain_suite(
    library_root: str | Path,
    suite: Mapping[str, Any],
    *,
    tile_rows: int = 64,
    overwrite: bool = False,
) -> dict[str, Any]:
    """在 CUDA 上生成一组版本化的完整 campaign region。"""

    allowed = {
        "schema_version",
        "suite_name",
        "description",
        "base_recipe",
        "conditions",
        "overlap_region_size_m",
    }
    extra = set(suite) - allowed
    if extra:
        raise TerrainConfigurationError(
            f"terrain suite contains unknown fields: {sorted(extra)}"
        )
    if str(suite.get("schema_version")) != "1":
        raise TerrainConfigurationError("terrain suite schema_version must be '1'")
    suite_name = str(suite.get("suite_name", "")).strip()
    if (
        not suite_name
        or suite_name in {".", ".."}
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in suite_name
        )
    ):
        raise TerrainConfigurationError("suite_name is empty or unsafe")
    conditions = list(suite.get("conditions", ()))
    if len(conditions) != 10:
        raise TerrainConfigurationError(
            "this validation suite must define exactly 10 terrain conditions"
        )
    names = [str(condition.get("name", "")) for condition in conditions]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise TerrainConfigurationError(
            "suite condition names must be non-empty and unique"
        )
    base_recipe = dict(suite.get("base_recipe", {}))
    overlap_size = suite.get(
        "overlap_region_size_m", {"x": 0.0005, "y": 0.0004}
    )
    size_x_m = float(overlap_size["x"])
    size_y_m = float(overlap_size["y"])
    library = TerrainLibrary(library_root)
    cuda = _cuda_record()
    results: list[dict[str, Any]] = []
    suite_started = time.perf_counter()

    # 每个条件先通过小域 CPU/GPU 对照，再生成全域、复核统计并缓存两种球尖 track。
    for condition in conditions:
        allowed_condition = {"name", "description", "recipe_overrides"}
        condition_extra = set(condition) - allowed_condition
        if condition_extra:
            raise TerrainConfigurationError(
                f"condition {condition['name']} contains unknown fields: "
                f"{sorted(condition_extra)}"
            )
        recipe_mapping = dict(base_recipe)
        recipe_mapping.update(condition.get("recipe_overrides", {}))
        recipe = TerrainRecipe.from_mapping(recipe_mapping)
        campaign = compute_campaign_region(recipe)
        overlap = _overlap_check(
            recipe, size_x_m=size_x_m, size_y_m=size_y_m
        )
        if not overlap["passed"]:
            raise TerrainConfigurationError(
                f"CPU/GPU overlap failed for condition {condition['name']}"
            )
        metadata = library.generate_region(
            recipe,
            campaign.region,
            tile_rows=tile_rows,
            backend="cuda",
            overwrite=overwrite,
        )
        statistics = _region_statistics(
            library, recipe, campaign.region
        )
        tracks = [
            library.cache_track(
                recipe,
                campaign.region,
                radius_m=radius_m,
                y_global_m=0.0,
                overwrite=overwrite,
            )
            for radius_m in (50e-6, 100e-6)
        ]
        results.append(
            {
                "name": condition["name"],
                "description": condition.get("description", ""),
                "terrain_recipe_id": recipe.terrain_recipe_id,
                "recipe_hash": recipe.recipe_hash,
                "recipe": recipe.normalized(),
                "region_id": campaign.region.region_id,
                "region_shape": list(campaign.region.shape),
                "region_size_m": [
                    campaign.region.size_x_m,
                    campaign.region.size_y_m,
                ],
                "data_path": str(
                    library.region_data_path(
                        recipe.terrain_recipe_id, campaign.region.region_id
                    )
                ),
                "data_sha256": metadata["data_sha256"],
                "file_size_bytes": metadata["file_size_bytes"],
                "generation_time_s": metadata["generation_time_s"],
                "gpu_memory_pool_peak_cached_bytes": metadata.get(
                    "gpu_memory_pool_peak_cached_bytes"
                ),
                "cpu_gpu_overlap": overlap,
                "full_region_statistics": statistics,
                "track_ids": [track.track_id for track in tracks],
            }
        )

    report = {
        "schema_version": "1",
        "m1_module_version": M1_MODULE_VERSION,
        "suite_name": suite_name,
        "description": suite.get("description", ""),
        "created_at_utc": utc_now(),
        "generation_backend": "cuda",
        "cuda": cuda,
        "tile_rows": tile_rows,
        "condition_count": len(results),
        "conditions": results,
        "total_file_size_bytes": sum(
            item["file_size_bytes"] for item in results
        ),
        "total_generation_time_s": sum(
            item["generation_time_s"] for item in results
        ),
        "suite_wall_time_s": time.perf_counter() - suite_started,
        "all_overlap_checks_passed": all(
            item["cpu_gpu_overlap"]["passed"] for item in results
        ),
        "all_full_hashes_verified": all(
            item["full_region_statistics"]["full_sha256_verified"]
            for item in results
        ),
        "unique_data_hash_count": len(
            {item["data_sha256"] for item in results}
        ),
    }
    validation_path = (
        library.root / "validation" / f"{suite_name}.json"
    )
    atomic_write_json(validation_path, report)
    return report


def load_suite(path: str | Path) -> dict[str, Any]:
    """读取 suite JSON 配置或已生成报告。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))
