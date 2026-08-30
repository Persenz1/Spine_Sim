"""Unified API for measured and material-specific synthetic terrain."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from .errors import GeometryOutOfDomainError, TerrainConfigurationError
from .material_generators import (
    _load_profile_surface,
    generate_concrete,
    generate_red_brick,
    generate_sandpaper,
    resolve_profile_source,
)
from .measured import load_measured_surface, random_measured_crop
from .models import MATERIAL_TERRAIN_VERSION, RegionSpec, TerrainRecipe
from .profiles import load_material_profile


TerrainMode = Literal["measured", "synthetic", "auto"]
TerrainBackend = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class Terrain:
    """A finite SI height field returned by :func:`generate_terrain`."""

    height: NDArray[np.float32]
    dx: float
    dy: float
    valid_mask: NDArray[np.bool_]
    material: str
    subtype: str
    seed: int
    metadata: Mapping[str, Any]
    measurement_probe: Mapping[str, Any] | None = None
    measurement_tolerance_m: float | None = None
    determinate_mask: NDArray[np.bool_] | None = None
    geometry_uncertain_mask: NDArray[np.bool_] | None = None
    geometry_lower_bound_m: NDArray[np.float32] | None = None
    geometry_upper_bound_m: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        height = np.asarray(self.height)
        mask = np.asarray(self.valid_mask)
        if height.ndim != 2 or min(height.shape) < 2:
            raise TerrainConfigurationError("terrain.height must be a 2-D grid")
        if height.dtype != np.float32:
            raise TerrainConfigurationError("terrain.height must use float32")
        if mask.shape != height.shape or mask.dtype != np.bool_:
            raise TerrainConfigurationError(
                "terrain.valid_mask must be bool with the height shape"
            )
        if not np.all(np.isfinite(height)):
            raise TerrainConfigurationError("terrain.height contains NaN or Inf")
        if not math.isfinite(self.dx) or self.dx <= 0:
            raise TerrainConfigurationError("terrain.dx must be positive and finite")
        if not math.isfinite(self.dy) or self.dy <= 0:
            raise TerrainConfigurationError("terrain.dy must be positive and finite")
        if not self.material or not self.subtype:
            raise TerrainConfigurationError("terrain material/subtype cannot be empty")
        if self.seed < 0:
            raise TerrainConfigurationError("terrain seed must be non-negative")
        if self.measurement_tolerance_m is not None and (
            not math.isfinite(self.measurement_tolerance_m)
            or self.measurement_tolerance_m < 0.0
        ):
            raise TerrainConfigurationError(
                "measurement_tolerance_m must be finite and non-negative"
            )
        for name in ("determinate_mask", "geometry_uncertain_mask"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value)
                if array.shape != height.shape or array.dtype != np.bool_:
                    raise TerrainConfigurationError(
                        f"terrain.{name} must be bool with the height shape"
                    )
        if (self.geometry_lower_bound_m is None) != (
            self.geometry_upper_bound_m is None
        ):
            raise TerrainConfigurationError(
                "terrain geometry bounds must both be present or both be None"
            )
        if self.geometry_lower_bound_m is not None:
            lower = np.asarray(self.geometry_lower_bound_m)
            upper = np.asarray(self.geometry_upper_bound_m)
            if (
                lower.shape != height.shape
                or upper.shape != height.shape
                or lower.dtype != np.float32
                or upper.dtype != np.float32
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower[mask] > upper[mask])
            ):
                raise TerrainConfigurationError(
                    "terrain geometry bounds must be finite float32 arrays with lower<=upper"
                )

    @property
    def size_x_m(self) -> float:
        return (self.height.shape[1] - 1) * self.dx

    @property
    def size_y_m(self) -> float:
        return (self.height.shape[0] - 1) * self.dy

    @property
    def resolved_mode(self) -> str:
        return str(self.metadata["resolved_mode"])

    def to_recipe(self) -> TerrainRecipe:
        """Build the recipe identity used by the existing terrain library."""

        return TerrainRecipe(
            generator_name="material_hybrid",
            generator_version=MATERIAL_TERRAIN_VERSION,
            seed=self.seed,
            canonical_dx_m=0.5 * self.dx,
            canonical_dy_m=0.5 * self.dy,
            production_dx_m=self.dx,
            production_dy_m=self.dy,
            target_rms_height_m=float(
                np.std(self.height, dtype=np.float64)
            ),
            correlation_length_x_m=self.dx,
            correlation_length_y_m=self.dy,
            kernel_kind="material_specific",
            material=self.material,
            subtype=self.subtype,
            generation_mode=self.resolved_mode,
            profile_hash=str(self.metadata["profile_hash"]),
        )


def _grid_shape(
    size_x_m: float, size_y_m: float, resolution_m: float
) -> tuple[int, int]:
    for name, value in (
        ("size_x_m", size_x_m),
        ("size_y_m", size_y_m),
        ("resolution_m", resolution_m),
    ):
        if not math.isfinite(value) or value <= 0:
            raise TerrainConfigurationError(f"{name} must be positive and finite")
    intervals_x_float = size_x_m / resolution_m
    intervals_y_float = size_y_m / resolution_m
    intervals_x = int(round(intervals_x_float))
    intervals_y = int(round(intervals_y_float))
    if not math.isclose(
        intervals_x_float, intervals_x, rel_tol=0.0, abs_tol=1e-9
    ):
        raise TerrainConfigurationError(
            "size_x_m must be an integer multiple of resolution_m"
        )
    if not math.isclose(
        intervals_y_float, intervals_y, rel_tol=0.0, abs_tol=1e-9
    ):
        raise TerrainConfigurationError(
            "size_y_m must be an integer multiple of resolution_m"
        )
    if intervals_x < 1 or intervals_y < 1:
        raise TerrainConfigurationError(
            "terrain must contain at least one interval per axis"
        )
    return intervals_y + 1, intervals_x + 1


def _measured(
    profile: Mapping[str, Any],
    *,
    size_x_m: float,
    size_y_m: float,
    resolution_m: float,
    rng: np.random.Generator,
    measured_path: str | Path | None,
    measured_options: Mapping[str, Any] | None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, Any]]:
    source_path, source_record = resolve_profile_source(profile, measured_path)
    if source_path is None:
        raise FileNotFoundError(
            f"no measured source is available for "
            f"{profile['material']}/{profile['subtype']}; pass measured_path"
        )
    if source_record is not None:
        surface = _load_profile_surface(source_path, source_record)
    else:
        surface = load_measured_surface(
            source_path, **dict(measured_options or {})
        )
    height, mask, crop_record = random_measured_crop(
        surface,
        size_x_m=size_x_m,
        size_y_m=size_y_m,
        resolution_m=resolution_m,
        rng=rng,
    )
    return height, mask, {
        "method": "random_measured_crop",
        "source": surface.metadata["source"],
        "source_preprocessing": surface.metadata["preprocessing"],
        "measurement_semantics": surface.metadata["measurement_semantics"],
        "surface_model": surface.metadata["surface_model"],
        "general_mesh_scope": surface.metadata["general_mesh_scope"],
        "crop": crop_record,
    }


def generate_terrain(
    *,
    material: str,
    subtype: str | None = None,
    size_x_m: float,
    size_y_m: float,
    resolution_m: float,
    seed: int,
    mode: TerrainMode = "synthetic",
    measured_path: str | Path | None = None,
    measured_options: Mapping[str, Any] | None = None,
    backend: TerrainBackend = "cpu",
) -> Terrain:
    """Generate a reproducible red-brick, concrete, or sandpaper height field.

    The array is node-centred, indexed ``[y, x]``, stored as ``float32``, and
    measured in metres.  ``mode='auto'`` uses a measured crop only when a
    suitable source exists and the requested extent/resolution is supported;
    otherwise it records the reason and falls back to synthetic generation.
    """

    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise TerrainConfigurationError("seed must be a non-negative integer")
    if mode not in {"measured", "synthetic", "auto"}:
        raise TerrainConfigurationError(
            "mode must be 'measured', 'synthetic', or 'auto'"
        )
    if backend not in {"cpu", "cuda"}:
        raise TerrainConfigurationError(
            "backend must be 'cpu' or 'cuda'"
        )
    profile = load_material_profile(material, subtype)
    shape = _grid_shape(size_x_m, size_y_m, resolution_m)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    fallback_reason: str | None = None

    if mode in {"measured", "auto"}:
        try:
            height, mask, generation_record = _measured(
                profile,
                size_x_m=size_x_m,
                size_y_m=size_y_m,
                resolution_m=resolution_m,
                rng=rng,
                measured_path=measured_path,
                measured_options=measured_options,
            )
            resolved_mode = "measured"
        except (FileNotFoundError, GeometryOutOfDomainError, TerrainConfigurationError) as exc:
            if mode == "measured":
                raise
            fallback_reason = f"{type(exc).__name__}: {exc}"
            # Reinitialize so auto fallback is bit-identical to explicit synthetic.
            rng = np.random.Generator(np.random.PCG64(int(seed)))
            resolved_mode = "synthetic"
    else:
        resolved_mode = "synthetic"

    cupy_module: Any | None = None
    if resolved_mode == "synthetic" and backend == "cuda":
        try:
            import cupy as cp  # type: ignore
        except ImportError as exc:
            raise TerrainConfigurationError(
                "CUDA material generation requested but CuPy is not installed"
            ) from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise TerrainConfigurationError(
                "CUDA material generation requested but no CUDA device is available"
            )
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cupy_module = cp

    if resolved_mode == "synthetic":
        if material == "sandpaper":
            height, generation_record = generate_sandpaper(
                profile,
                shape=shape,
                resolution_m=resolution_m,
                rng=rng,
                source_path=measured_path,
                backend=backend,
            )
        elif material == "red_brick":
            height, generation_record = generate_red_brick(
                profile,
                shape=shape,
                resolution_m=resolution_m,
                rng=rng,
                backend=backend,
            )
        elif material == "concrete":
            height, generation_record = generate_concrete(
                profile,
                shape=shape,
                resolution_m=resolution_m,
                rng=rng,
                backend=backend,
            )
        else:  # load_material_profile has already rejected this path
            raise AssertionError(material)
        mask = np.ones(shape, dtype=np.bool_)

    resolved_backend = (
        backend if resolved_mode == "synthetic" else "cpu"
    )
    backend_record: dict[str, Any] = {
        "requested": backend,
        "resolved": resolved_backend,
    }
    if cupy_module is not None:
        cupy_module.cuda.Device().synchronize()
        properties = cupy_module.cuda.runtime.getDeviceProperties(0)
        device_name = properties["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode(errors="replace")
        pool = cupy_module.get_default_memory_pool()
        pinned_pool = cupy_module.get_default_pinned_memory_pool()
        backend_record.update(
            {
                "provider": "cupy",
                "cupy_version": cupy_module.__version__,
                "device_index": 0,
                "device_name": str(device_name),
                "cuda_runtime_version": int(
                    cupy_module.cuda.runtime.runtimeGetVersion()
                ),
                "cuda_driver_version": int(
                    cupy_module.cuda.runtime.driverGetVersion()
                ),
                "gpu_memory_pool_used_bytes": int(pool.used_bytes()),
                "gpu_memory_pool_peak_cached_bytes": int(pool.total_bytes()),
                "gpu_pinned_pool_cached_blocks": int(
                    pinned_pool.n_free_blocks()
                ),
                "implementation": (
                    "cuda_tiled_correlated_fields_and_feature_stamping_"
                    "with_cpu_measured_quilting"
                ),
            }
        )
    else:
        backend_record.update(
            {
                "provider": "numpy",
                "implementation": (
                    "measured_crop_cpu"
                    if resolved_mode == "measured"
                    else "numpy_material_generator"
                ),
            }
        )

    if height.shape != shape or mask.shape != shape:
        raise TerrainConfigurationError(
            f"generator returned shape {height.shape}, expected {shape}"
        )
    if resolved_mode == "measured":
        measurement_semantics = dict(generation_record["measurement_semantics"])
        geometry_uncertain_mask = mask.copy()
    else:
        measurement_semantics = {
            "status": "not_applicable",
            "probe": None,
            "measurement_tolerance_m": None,
            "determinate_mask": None,
            "bounds": None,
        }
        geometry_uncertain_mask = np.zeros(shape, dtype=np.bool_)
    metadata: dict[str, Any] = {
        "schema_version": "material-terrain-output-v1",
        "generator_version": MATERIAL_TERRAIN_VERSION,
        "material": material,
        "subtype": profile["subtype"],
        "seed": int(seed),
        "requested_mode": mode,
        "resolved_mode": resolved_mode,
        "generation_backend": backend_record,
        "profile_status": profile["status"],
        "parameter_basis": profile["parameter_basis"],
        "profile_hash": profile["profile_hash"],
        "profile_generation_parameters": profile["generation"],
        "source_data": profile.get("source_data", []),
        "limitations": profile.get("limitations", []),
        "grid": {
            "array_order": "y_x",
            "coordinate_unit": "m",
            "height_unit": "m",
            "node_centered": True,
            "shape_yx": list(shape),
            "spacing_x_m": resolution_m,
            "spacing_y_m": resolution_m,
            "size_x_m": size_x_m,
            "size_y_m": size_y_m,
            "dtype": "float32",
        },
        "generation": generation_record,
        "measurement_semantics": measurement_semantics,
        "surface_model": "single_valued_height_field_2_5d",
        "general_mesh_scope": "OUT_OF_SCOPE",
    }
    if fallback_reason is not None:
        metadata["auto_fallback_reason"] = fallback_reason
    return Terrain(
        height=height,
        dx=float(resolution_m),
        dy=float(resolution_m),
        valid_mask=mask,
        material=material,
        subtype=str(profile["subtype"]),
        seed=int(seed),
        metadata=metadata,
        measurement_probe=None,
        measurement_tolerance_m=None,
        determinate_mask=None,
        geometry_uncertain_mask=geometry_uncertain_mask,
        geometry_lower_bound_m=None,
        geometry_upper_bound_m=None,
    )


def refine_material_terrain_same_realization(
    coarse: Terrain,
    fine_detail: Terrain,
) -> Terrain:
    """Create a nested 2x refinement while preserving every coarse node.

    ``fine_detail`` supplies only the sub-grid residual of the same
    material/profile/seed.  The returned 5 um-style field is exactly equal to
    ``coarse`` at all stride-2 nodes, so 10/5 um comparisons cannot drift onto
    a different large-scale realization.
    """

    if any(
        value is not None
        for value in (
            coarse.geometry_lower_bound_m,
            coarse.geometry_upper_bound_m,
            fine_detail.geometry_lower_bound_m,
            fine_detail.geometry_upper_bound_m,
        )
    ):
        raise TerrainConfigurationError(
            "same-realization refinement of measurement geometry bounds is not supported"
        )

    if (
        coarse.material != fine_detail.material
        or coarse.subtype != fine_detail.subtype
        or coarse.seed != fine_detail.seed
    ):
        raise TerrainConfigurationError(
            "same-realization refinement requires matching material, "
            "subtype and seed"
        )
    if coarse.resolved_mode != fine_detail.resolved_mode:
        raise TerrainConfigurationError(
            "same-realization refinement requires matching generation mode"
        )
    if coarse.metadata.get("profile_hash") != fine_detail.metadata.get(
        "profile_hash"
    ):
        raise TerrainConfigurationError(
            "same-realization refinement requires one material profile"
        )
    if not (
        math.isclose(
            coarse.dx,
            2.0 * fine_detail.dx,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            coarse.dy,
            2.0 * fine_detail.dy,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise TerrainConfigurationError(
            "fine-detail spacing must be exactly half the coarse spacing"
        )
    expected_shape = (
        2 * coarse.height.shape[0] - 1,
        2 * coarse.height.shape[1] - 1,
    )
    if fine_detail.height.shape != expected_shape:
        raise TerrainConfigurationError(
            "fine-detail shape must contain the same extent at 2x resolution"
        )

    coarse_height = coarse.height
    detail_height = fine_detail.height
    refined = np.empty_like(detail_height)
    refined[::2, ::2] = coarse_height
    refined[1::2, ::2] = (
        0.5 * (coarse_height[:-1, :] + coarse_height[1:, :])
        + detail_height[1::2, ::2]
        - 0.5
        * (
            detail_height[:-2:2, ::2]
            + detail_height[2::2, ::2]
        )
    )
    refined[::2, 1::2] = (
        0.5 * (coarse_height[:, :-1] + coarse_height[:, 1:])
        + detail_height[::2, 1::2]
        - 0.5
        * (
            detail_height[::2, :-2:2]
            + detail_height[::2, 2::2]
        )
    )
    refined[1::2, 1::2] = (
        0.25
        * (
            coarse_height[:-1, :-1]
            + coarse_height[1:, :-1]
            + coarse_height[:-1, 1:]
            + coarse_height[1:, 1:]
        )
        + detail_height[1::2, 1::2]
        - 0.25
        * (
            detail_height[:-2:2, :-2:2]
            + detail_height[2::2, :-2:2]
            + detail_height[:-2:2, 2::2]
            + detail_height[2::2, 2::2]
        )
    )

    coarse_mask = coarse.valid_mask
    refined_mask = fine_detail.valid_mask.copy()
    refined_mask[::2, ::2] &= coarse_mask
    refined_mask[1::2, ::2] &= coarse_mask[:-1, :] & coarse_mask[1:, :]
    refined_mask[::2, 1::2] &= coarse_mask[:, :-1] & coarse_mask[:, 1:]
    refined_mask[1::2, 1::2] &= (
        coarse_mask[:-1, :-1]
        & coarse_mask[1:, :-1]
        & coarse_mask[:-1, 1:]
        & coarse_mask[1:, 1:]
    )
    coarse_uncertain = (
        np.zeros_like(coarse_mask)
        if coarse.geometry_uncertain_mask is None
        else coarse.geometry_uncertain_mask
    )
    detail_uncertain = (
        np.zeros_like(fine_detail.valid_mask)
        if fine_detail.geometry_uncertain_mask is None
        else fine_detail.geometry_uncertain_mask
    )
    refined_uncertain = detail_uncertain.copy()
    refined_uncertain[::2, ::2] |= coarse_uncertain
    refined_uncertain[1::2, ::2] |= (
        coarse_uncertain[:-1, :] | coarse_uncertain[1:, :]
    )
    refined_uncertain[::2, 1::2] |= (
        coarse_uncertain[:, :-1] | coarse_uncertain[:, 1:]
    )
    refined_uncertain[1::2, 1::2] |= (
        coarse_uncertain[:-1, :-1]
        | coarse_uncertain[1:, :-1]
        | coarse_uncertain[:-1, 1:]
        | coarse_uncertain[1:, 1:]
    )

    metadata = dict(fine_detail.metadata)
    metadata["same_realization_refinement"] = {
        "algorithm": (
            "coarse_exact_nodes_plus_fine_model_subgrid_residual_v1"
        ),
        "coarse_spacing_x_m": coarse.dx,
        "coarse_spacing_y_m": coarse.dy,
        "fine_spacing_x_m": fine_detail.dx,
        "fine_spacing_y_m": fine_detail.dy,
        "coarse_shape_yx": list(coarse.height.shape),
        "fine_shape_yx": list(fine_detail.height.shape),
        "coarse_node_identity_exact": True,
        "material": coarse.material,
        "subtype": coarse.subtype,
        "seed": coarse.seed,
        "profile_hash": coarse.metadata.get("profile_hash"),
    }
    return Terrain(
        height=refined,
        dx=fine_detail.dx,
        dy=fine_detail.dy,
        valid_mask=refined_mask,
        material=coarse.material,
        subtype=coarse.subtype,
        seed=coarse.seed,
        metadata=metadata,
        measurement_probe=fine_detail.measurement_probe,
        measurement_tolerance_m=fine_detail.measurement_tolerance_m,
        determinate_mask=None,
        geometry_uncertain_mask=refined_uncertain,
        geometry_lower_bound_m=None,
        geometry_upper_bound_m=None,
    )


def save_terrain(path: str | Path, terrain: Terrain) -> Path:
    """Atomically save a portable NPZ artifact containing height, mask and metadata."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        metadata = dict(terrain.metadata)
        metadata["measurement_semantics"] = {
            "status": (
                "unknown_probe"
                if terrain.resolved_mode == "measured"
                and terrain.measurement_probe is None
                else (
                    "known_probe"
                    if terrain.resolved_mode == "measured"
                    else "not_applicable"
                )
            ),
            "probe": (
                None
                if terrain.measurement_probe is None
                else dict(terrain.measurement_probe)
            ),
            "measurement_tolerance_m": terrain.measurement_tolerance_m,
            "determinate_mask": terrain.determinate_mask is not None,
            "bounds": terrain.geometry_lower_bound_m is not None,
        }
        arrays: dict[str, NDArray[Any]] = {
            "height": terrain.height,
            "valid_mask": terrain.valid_mask,
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, ensure_ascii=False)
            ),
        }
        if terrain.determinate_mask is not None:
            arrays["determinate_mask"] = terrain.determinate_mask
        if terrain.geometry_uncertain_mask is not None:
            arrays["geometry_uncertain_mask"] = terrain.geometry_uncertain_mask
        if terrain.geometry_lower_bound_m is not None:
            arrays["geometry_lower_bound_m"] = terrain.geometry_lower_bound_m
            arrays["geometry_upper_bound_m"] = terrain.geometry_upper_bound_m
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_terrain(path: str | Path) -> Terrain:
    """Load and validate an NPZ artifact created by :func:`save_terrain`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        height = archive["height"]
        valid_mask = archive["valid_mask"]
        metadata = json.loads(str(archive["metadata_json"].item()))
        determinate_mask = (
            archive["determinate_mask"]
            if "determinate_mask" in archive.files
            else None
        )
        geometry_uncertain_mask = (
            archive["geometry_uncertain_mask"]
            if "geometry_uncertain_mask" in archive.files
            else None
        )
        geometry_lower_bound_m = (
            archive["geometry_lower_bound_m"]
            if "geometry_lower_bound_m" in archive.files
            else None
        )
        geometry_upper_bound_m = (
            archive["geometry_upper_bound_m"]
            if "geometry_upper_bound_m" in archive.files
            else None
        )
    grid = metadata["grid"]
    measurement = metadata.get("measurement_semantics")
    required_measurement_fields = {
        "status",
        "probe",
        "measurement_tolerance_m",
        "determinate_mask",
        "bounds",
    }
    if not isinstance(measurement, Mapping) or not required_measurement_fields <= (
        measurement.keys()
    ):
        raise TerrainConfigurationError(
            "terrain artifact is missing canonical measurement_semantics"
        )
    return Terrain(
        height=height,
        dx=float(grid["spacing_x_m"]),
        dy=float(grid["spacing_y_m"]),
        valid_mask=valid_mask,
        material=str(metadata["material"]),
        subtype=str(metadata["subtype"]),
        seed=int(metadata["seed"]),
        metadata=metadata,
        measurement_probe=measurement["probe"],
        measurement_tolerance_m=measurement["measurement_tolerance_m"],
        determinate_mask=determinate_mask,
        geometry_uncertain_mask=geometry_uncertain_mask,
        geometry_lower_bound_m=geometry_lower_bound_m,
        geometry_upper_bound_m=geometry_upper_bound_m,
    )


def register_terrain(
    library_root: str | Path,
    terrain: Terrain,
    *,
    origin_x_m: float = 0.0,
    origin_y_m: float = 0.0,
    purpose: str = "user",
    overwrite: bool = False,
) -> tuple[TerrainRecipe, RegionSpec, dict[str, Any]]:
    """Register generated output in the existing mmap terrain-library contract."""

    from .library import TerrainLibrary

    recipe = terrain.to_recipe()
    region = RegionSpec(
        terrain_recipe_id=recipe.terrain_recipe_id,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        size_x_m=terrain.size_x_m,
        size_y_m=terrain.size_y_m,
        resolution_x_m=terrain.dx,
        resolution_y_m=terrain.dy,
        purpose=purpose,
    )
    metadata = TerrainLibrary(library_root).register_material_region(
        recipe, region, terrain, overwrite=overwrite
    )
    return recipe, region, metadata
