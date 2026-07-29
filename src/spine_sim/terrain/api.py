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
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(self.height, dtype=np.float64)
                            - float(np.mean(self.height, dtype=np.float64))
                        )
                        ** 2
                    )
                )
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

    if resolved_mode == "synthetic":
        if material == "sandpaper":
            height, generation_record = generate_sandpaper(
                profile,
                shape=shape,
                resolution_m=resolution_m,
                rng=rng,
                source_path=measured_path,
            )
        elif material == "red_brick":
            height, generation_record = generate_red_brick(
                profile,
                shape=shape,
                resolution_m=resolution_m,
                rng=rng,
            )
        elif material == "concrete":
            height, generation_record = generate_concrete(
                profile,
                shape=shape,
                resolution_m=resolution_m,
                rng=rng,
            )
        else:  # load_material_profile has already rejected this path
            raise AssertionError(material)
        mask = np.ones(shape, dtype=np.bool_)

    height = np.asarray(height, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.bool_)
    if height.shape != shape or mask.shape != shape:
        raise TerrainConfigurationError(
            f"generator returned shape {height.shape}, expected {shape}"
        )
    if not np.all(np.isfinite(height)):
        raise TerrainConfigurationError("generator produced NaN or Inf")
    metadata: dict[str, Any] = {
        "schema_version": "material-terrain-output-v1",
        "generator_version": MATERIAL_TERRAIN_VERSION,
        "material": material,
        "subtype": profile["subtype"],
        "seed": int(seed),
        "requested_mode": mode,
        "resolved_mode": resolved_mode,
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
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                height=terrain.height,
                valid_mask=terrain.valid_mask,
                metadata_json=np.asarray(
                    json.dumps(terrain.metadata, sort_keys=True, ensure_ascii=False)
                ),
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_terrain(path: str | Path) -> Terrain:
    """Load and validate an NPZ artifact created by :func:`save_terrain`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        height = np.asarray(archive["height"], dtype=np.float32)
        valid_mask = np.asarray(archive["valid_mask"], dtype=np.bool_)
        metadata = json.loads(str(archive["metadata_json"].item()))
    grid = metadata["grid"]
    return Terrain(
        height=height,
        dx=float(grid["spacing_x_m"]),
        dy=float(grid["spacing_y_m"]),
        valid_mask=valid_mask,
        material=str(metadata["material"]),
        subtype=str(metadata["subtype"]),
        seed=int(metadata["seed"]),
        metadata=metadata,
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
