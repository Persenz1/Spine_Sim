"""Versioned M1 data models and campaign-region construction."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from spine_sim.core.identity import (
    stable_hash,
    track_id as make_track_id,
)
from spine_sim.core.config import TerrainRecipeRef, TerrainRegionSpec

from .errors import TerrainConfigurationError


M1_MODULE_VERSION = "m1.0.0"
DEFINED_GEOMETRY_VERSION = "defined-geometry-v1-canonical5um-stride2-nodal"
MATERIAL_TERRAIN_VERSION = "material-terrain-v2"
ENVELOPE_ALGORITHM_VERSION = "finite-sphere-envelope-v1"
CANONICAL_SPACING_M = 5e-6
PRODUCTION_SPACING_M = 10e-6
_ALIGNMENT_ATOL = 1e-9


def _identity_float(value: float) -> float:
    """Remove sub-femtometre binary arithmetic noise before exact hashing."""

    return round(float(value), 15)


def _normalize_float_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _identity_float(item) if isinstance(item, float) else item
        for key, item in value.items()
    }


def _require_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise TerrainConfigurationError(f"{name} must be finite")
    return value


def _require_positive(name: str, value: float) -> float:
    _require_finite(name, value)
    if value <= 0:
        raise TerrainConfigurationError(f"{name} must be positive")
    return value


def _grid_intervals(size_m: float, spacing_m: float, *, name: str) -> int:
    intervals_float = size_m / spacing_m
    intervals = int(round(intervals_float))
    if not math.isclose(intervals_float, intervals, rel_tol=0.0, abs_tol=_ALIGNMENT_ATOL):
        raise TerrainConfigurationError(
            f"{name}={size_m!r} must be an integer multiple of spacing {spacing_m!r}"
        )
    if intervals < 1:
        raise TerrainConfigurationError(f"{name} must contain at least one grid interval")
    return intervals


def _aligned_index(coordinate_m: float, origin_m: float, spacing_m: float, *, name: str) -> int:
    index_float = (coordinate_m - origin_m) / spacing_m
    index = int(round(index_float))
    if not math.isclose(index_float, index, rel_tol=0.0, abs_tol=_ALIGNMENT_ATOL):
        raise TerrainConfigurationError(
            f"{name}={coordinate_m!r} is not aligned to the global grid"
        )
    return index


@dataclass(frozen=True)
class TerrainRecipe:
    """A coordinate-addressable synthetic terrain realization."""

    generator_name: str = "defined_geometry"
    generator_version: str = DEFINED_GEOMETRY_VERSION
    seed: int = 0
    global_origin_x_m: float = 0.0
    global_origin_y_m: float = 0.0
    canonical_dx_m: float = CANONICAL_SPACING_M
    canonical_dy_m: float = CANONICAL_SPACING_M
    production_dx_m: float = PRODUCTION_SPACING_M
    production_dy_m: float = PRODUCTION_SPACING_M
    target_rms_height_m: float = 30e-6
    correlation_length_x_m: float = 50e-6
    correlation_length_y_m: float = 50e-6
    kernel_kind: str = "separable_gaussian"
    kernel_truncate_sigma: float = 3.0
    amplitude_scale: float = 1.0
    coordinate_convention: str = "global_xy_nodes_origin_aligned"
    material: str | None = None
    subtype: str | None = None
    generation_mode: str | None = None
    profile_hash: str | None = None

    def __post_init__(self) -> None:
        if self.generator_name not in {"defined_geometry", "material_hybrid"}:
            raise TerrainConfigurationError(
                "generator_name must be 'defined_geometry' or 'material_hybrid'"
            )
        if not self.generator_version:
            raise TerrainConfigurationError("generator_version cannot be empty")
        if self.seed < 0:
            raise TerrainConfigurationError("seed must be non-negative")
        for name in (
            "canonical_dx_m",
            "canonical_dy_m",
            "production_dx_m",
            "production_dy_m",
            "correlation_length_x_m",
            "correlation_length_y_m",
            "kernel_truncate_sigma",
            "amplitude_scale",
        ):
            _require_positive(name, getattr(self, name))
        _require_finite("target_rms_height_m", self.target_rms_height_m)
        if self.target_rms_height_m < 0:
            raise TerrainConfigurationError("target_rms_height_m must be non-negative")
        if self.generator_name == "defined_geometry":
            if self.kernel_kind != "separable_gaussian":
                raise TerrainConfigurationError(
                    "defined_geometry requires separable_gaussian"
                )
            if any(
                value is not None
                for value in (
                    self.material,
                    self.subtype,
                    self.generation_mode,
                    self.profile_hash,
                )
            ):
                raise TerrainConfigurationError(
                    "defined_geometry cannot carry material-profile fields"
                )
        else:
            if self.generator_version != MATERIAL_TERRAIN_VERSION:
                raise TerrainConfigurationError(
                    f"material_hybrid requires {MATERIAL_TERRAIN_VERSION!r}"
                )
            if self.kernel_kind != "material_specific":
                raise TerrainConfigurationError(
                    "material_hybrid requires kernel_kind='material_specific'"
                )
            if not all(
                isinstance(value, str) and value
                for value in (
                    self.material,
                    self.subtype,
                    self.generation_mode,
                    self.profile_hash,
                )
            ):
                raise TerrainConfigurationError(
                    "material_hybrid requires material, subtype, generation_mode "
                    "and profile_hash"
                )
            if self.generation_mode not in {"measured", "synthetic"}:
                raise TerrainConfigurationError(
                    "material_hybrid generation_mode must be measured or synthetic"
                )
        if not math.isclose(
            self.production_dx_m, 2.0 * self.canonical_dx_m, rel_tol=0.0, abs_tol=1e-15
        ) or not math.isclose(
            self.production_dy_m, 2.0 * self.canonical_dy_m, rel_tol=0.0, abs_tol=1e-15
        ):
            raise TerrainConfigurationError(
                "production spacing must be exactly stride-2 of the canonical grid"
            )
        if self.coordinate_convention != "global_xy_nodes_origin_aligned":
            raise TerrainConfigurationError("unsupported coordinate convention")

    def normalized(self) -> dict[str, Any]:
        normalized = asdict(self)
        # Preserve all pre-material recipe identities exactly.
        if self.generator_name == "defined_geometry":
            for name in ("material", "subtype", "generation_mode", "profile_hash"):
                normalized.pop(name)
        return _normalize_float_fields(normalized)

    def to_m0_ref(self) -> TerrainRecipeRef:
        """Represent this full recipe through M0's frozen recipe-reference schema."""

        parameters = self.normalized()
        parameters.pop("generator_name")
        parameters.pop("generator_version")
        parameters.pop("seed")
        parameters["production_sampling"] = self.production_sampling
        return TerrainRecipeRef(
            recipe_name=self.generator_name,
            recipe_version=self.generator_version,
            seed=self.seed,
            parameters=parameters,
        )

    @property
    def terrain_recipe_id(self) -> str:
        return self.to_m0_ref().terrain_recipe_id

    @property
    def recipe_hash(self) -> str:
        return stable_hash(
            {
                "module_version": M1_MODULE_VERSION,
                "recipe": self.normalized(),
                "production_sampling": self.production_sampling,
            }
        )

    @property
    def kernel_definition(self) -> dict[str, Any]:
        if self.generator_name == "material_hybrid":
            return {
                "kind": "material_specific",
                "material": self.material,
                "subtype": self.subtype,
                "profile_hash": self.profile_hash,
            }
        return {
            "kind": self.kernel_kind,
            "truncate_sigma": self.kernel_truncate_sigma,
            "normalization": "unit_sum_then_theoretical_l2_rms_calibration",
            "window_normalization": False,
        }

    @property
    def production_sampling(self) -> str:
        if self.generator_name == "material_hybrid":
            return "material_output_grid_stride2_from_identity_grid_nodal"
        return "canonical_even_indices_stride2_nodal"

    def canonical_indices(
        self, x_m: float, y_m: float
    ) -> tuple[int, int]:
        return (
            _aligned_index(
                x_m, self.global_origin_x_m, self.canonical_dx_m, name="x_m"
            ),
            _aligned_index(
                y_m, self.global_origin_y_m, self.canonical_dy_m, name="y_m"
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TerrainRecipe":
        allowed = set(cls.__dataclass_fields__)
        extra = set(value) - allowed
        if extra:
            raise TerrainConfigurationError(
                f"TerrainRecipe contains unknown fields: {sorted(extra)}"
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class RegionSpec:
    """Inclusive, node-centred rectangular terrain cache region."""

    terrain_recipe_id: str
    origin_x_m: float
    origin_y_m: float
    size_x_m: float
    size_y_m: float
    resolution_x_m: float = PRODUCTION_SPACING_M
    resolution_y_m: float = PRODUCTION_SPACING_M
    purpose: str = "campaign"

    def __post_init__(self) -> None:
        if not self.terrain_recipe_id:
            raise TerrainConfigurationError("terrain_recipe_id cannot be empty")
        _require_finite("origin_x_m", self.origin_x_m)
        _require_finite("origin_y_m", self.origin_y_m)
        _require_positive("size_x_m", self.size_x_m)
        _require_positive("size_y_m", self.size_y_m)
        _require_positive("resolution_x_m", self.resolution_x_m)
        _require_positive("resolution_y_m", self.resolution_y_m)
        _grid_intervals(self.size_x_m, self.resolution_x_m, name="size_x_m")
        _grid_intervals(self.size_y_m, self.resolution_y_m, name="size_y_m")
        if self.purpose not in {"module", "debug", "campaign", "user"}:
            raise TerrainConfigurationError(
                "purpose must be module, debug, campaign or user"
            )
        if not math.isclose(
            self.resolution_x_m,
            self.resolution_y_m,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise TerrainConfigurationError(
                "M0 TerrainRegionSpec supports one isotropic resolution"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (
            _grid_intervals(
                self.size_y_m, self.resolution_y_m, name="size_y_m"
            )
            + 1,
            _grid_intervals(
                self.size_x_m, self.resolution_x_m, name="size_x_m"
            )
            + 1,
        )

    @property
    def x_max_m(self) -> float:
        return self.origin_x_m + self.size_x_m

    @property
    def y_max_m(self) -> float:
        return self.origin_y_m + self.size_y_m

    def normalized(self) -> dict[str, Any]:
        return _normalize_float_fields(asdict(self))

    def to_m0_spec(self) -> TerrainRegionSpec:
        """Return the frozen M0 region record used to define the region ID."""

        # Grid arithmetic can produce values such as -0.021980000000000003.
        # Quantize well below the 5 um grid before passing values to M0's exact
        # canonical hash so equivalent JSON and computed regions share one ID.
        return TerrainRegionSpec(
            terrain_recipe_id=self.terrain_recipe_id,
            origin_x_m=_identity_float(self.origin_x_m),
            origin_y_m=_identity_float(self.origin_y_m),
            size_x_m=_identity_float(self.size_x_m),
            size_y_m=_identity_float(self.size_y_m),
            resolution_m=_identity_float(self.resolution_x_m),
        )

    @property
    def region_id(self) -> str:
        return self.to_m0_spec().region_id

    @property
    def expected_npy_payload_bytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * np.dtype(np.float32).itemsize

    def validate_against(self, recipe: TerrainRecipe) -> None:
        if self.terrain_recipe_id != recipe.terrain_recipe_id:
            raise TerrainConfigurationError("region references a different terrain recipe")
        start_x, start_y = recipe.canonical_indices(
            self.origin_x_m, self.origin_y_m
        )
        if math.isclose(
            self.resolution_x_m, recipe.production_dx_m, rel_tol=0.0, abs_tol=1e-15
        ) and start_x % 2:
            raise TerrainConfigurationError(
                "10 um region origin must lie on an even canonical x index"
            )
        if math.isclose(
            self.resolution_y_m, recipe.production_dy_m, rel_tol=0.0, abs_tol=1e-15
        ) and start_y % 2:
            raise TerrainConfigurationError(
                "10 um region origin must lie on an even canonical y index"
            )
        valid_x = (
            math.isclose(
                self.resolution_x_m,
                recipe.canonical_dx_m,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or math.isclose(
                self.resolution_x_m,
                recipe.production_dx_m,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        valid_y = (
            math.isclose(
                self.resolution_y_m,
                recipe.canonical_dy_m,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or math.isclose(
                self.resolution_y_m,
                recipe.production_dy_m,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        if not (valid_x and valid_y):
            raise TerrainConfigurationError(
                "region resolution must be the recipe canonical or production spacing"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RegionSpec":
        allowed = set(cls.__dataclass_fields__)
        extra = set(value) - allowed
        if extra:
            raise TerrainConfigurationError(
                f"RegionSpec contains unknown fields: {sorted(extra)}"
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class TrackGeometry:
    """One-dimensional finite-tip geometry consumed by M2 and M3."""

    terrain_recipe_id: str
    region_id: str
    track_id: str
    radius_m: float
    y_global_m: float
    resolution_m: float
    envelope_algorithm_version: str
    x_global_m: NDArray[np.float64]
    envelope_height_m: NDArray[np.float64]
    envelope_slope_x: NDArray[np.float64]
    support_x_m: NDArray[np.float64]
    support_y_m: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    near_tie_flag: NDArray[np.bool_]
    model_warning: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.terrain_recipe_id or not self.region_id or not self.track_id:
            raise TerrainConfigurationError("track IDs cannot be empty")
        _require_positive("radius_m", self.radius_m)
        _require_positive("resolution_m", self.resolution_m)
        arrays = (
            self.x_global_m,
            self.envelope_height_m,
            self.envelope_slope_x,
            self.support_x_m,
            self.support_y_m,
            self.valid_mask,
            self.near_tie_flag,
        )
        shapes = {np.asarray(array).shape for array in arrays}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise TerrainConfigurationError(
                "all TrackGeometry arrays must have the same one-dimensional shape"
            )
        if np.asarray(self.valid_mask).dtype != np.bool_:
            raise TerrainConfigurationError("valid_mask must be boolean")
        if np.asarray(self.near_tie_flag).dtype != np.bool_:
            raise TerrainConfigurationError("near_tie_flag must be boolean")

    @staticmethod
    def make_id(
        *,
        terrain_recipe_id: str,
        region_id: str,
        radius_m: float,
        y_global_m: float,
        envelope_algorithm_version: str,
        resolution_m: float,
    ) -> str:
        return make_track_id(
            {
                "terrain_recipe_id": terrain_recipe_id,
                "region_id": region_id,
                "radius_m": _identity_float(radius_m),
                "y_global_m": _identity_float(y_global_m),
                "envelope_algorithm_version": envelope_algorithm_version,
                "resolution_m": _identity_float(resolution_m),
            },
            module_version=M1_MODULE_VERSION,
        )


@dataclass(frozen=True)
class CampaignDesignSpace:
    """Inputs needed to derive, rather than hard-code, the maximum terrain region."""

    drag_length_m: float = 0.100
    max_array_nx: int = 6
    max_array_ny: int = 6
    max_spacing_x_m: float = 0.006
    max_spacing_y_m: float = 0.006
    fixed_angles_deg: tuple[float, ...] = (50.0, 60.0, 70.0, 80.0)
    gradient_angles_deg: tuple[float, ...] = (50.0, 60.0, 80.0)
    fixed_exposed_length_m: float = 0.004
    gradient_reference_length_m: float = 0.004
    gradient_reference_angle_deg: float = 80.0
    max_tip_radius_m: float = 100e-6
    max_spring_travel_m: float = 0.004
    beam_displacement_allowance_m: float = 0.0005
    interpolation_event_margin_m: float = 0.0001
    rod_clearance_margin_m: float = 0.00425
    lateral_compliance_margin_m: float = 0.0005
    resolution_m: float = PRODUCTION_SPACING_M

    def __post_init__(self) -> None:
        for name in (
            "drag_length_m",
            "max_spacing_x_m",
            "max_spacing_y_m",
            "fixed_exposed_length_m",
            "gradient_reference_length_m",
            "max_tip_radius_m",
            "max_spring_travel_m",
            "beam_displacement_allowance_m",
            "interpolation_event_margin_m",
            "rod_clearance_margin_m",
            "lateral_compliance_margin_m",
            "resolution_m",
        ):
            _require_positive(name, getattr(self, name))
        if self.max_array_nx < 1 or self.max_array_ny < 1:
            raise TerrainConfigurationError("array dimensions must be positive")
        angles = self.fixed_angles_deg + self.gradient_angles_deg
        if not angles or any(angle <= 0 or angle >= 90 for angle in angles):
            raise TerrainConfigurationError("installation angles must lie in (0, 90) degrees")


@dataclass(frozen=True)
class CampaignRegionReport:
    region: RegionSpec
    raw_bounds_m: Mapping[str, float]
    aligned_bounds_m: Mapping[str, float]
    margins_m: Mapping[str, float]
    projection_range_m: tuple[float, float]
    installation_span_m: tuple[float, float]
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "region": self.region.normalized(),
            "region_id": self.region.region_id,
            "shape": list(self.region.shape),
            "npy_payload_bytes": self.region.expected_npy_payload_bytes,
            "raw_bounds_m": dict(self.raw_bounds_m),
            "aligned_bounds_m": dict(self.aligned_bounds_m),
            "margins_m": dict(self.margins_m),
            "projection_range_m": list(self.projection_range_m),
            "installation_span_m": list(self.installation_span_m),
            "assumptions": list(self.assumptions),
        }


def _align_floor(value: float, origin: float, spacing: float) -> float:
    return origin + math.floor((value - origin) / spacing + 1e-12) * spacing


def _align_ceil(value: float, origin: float, spacing: float) -> float:
    return origin + math.ceil((value - origin) / spacing - 1e-12) * spacing


def compute_campaign_region(
    recipe: TerrainRecipe,
    design: CampaignDesignSpace | None = None,
) -> CampaignRegionReport:
    """Compute and outward-align the maximum campaign terrain bounds."""

    design = design or CampaignDesignSpace()
    fixed_projections = [
        design.fixed_exposed_length_m * math.cos(math.radians(angle))
        for angle in design.fixed_angles_deg
    ]
    reference_height = design.gradient_reference_length_m * math.sin(
        math.radians(design.gradient_reference_angle_deg)
    )
    gradient_projections = [
        reference_height / math.sin(math.radians(angle)) * math.cos(math.radians(angle))
        for angle in design.gradient_angles_deg
    ]
    projections = fixed_projections + gradient_projections
    projection_min = min(projections)
    projection_max = max(projections)

    span_x = (design.max_array_nx - 1) * design.max_spacing_x_m
    span_y = (design.max_array_ny - 1) * design.max_spacing_y_m
    install_x_min = -0.5 * span_x
    install_x_max = 0.5 * span_x
    install_y_min = -0.5 * span_y
    install_y_max = 0.5 * span_y

    max_horizontal_spring = max(
        design.max_spring_travel_m * math.cos(math.radians(angle))
        for angle in design.fixed_angles_deg + design.gradient_angles_deg
    )
    filter_halo_x = (
        recipe.kernel_truncate_sigma * recipe.correlation_length_x_m
    )
    filter_halo_y = (
        recipe.kernel_truncate_sigma * recipe.correlation_length_y_m
    )
    x_margin_parts = {
        "tip_radius": design.max_tip_radius_m,
        "spring_horizontal_projection": max_horizontal_spring,
        "beam_displacement": design.beam_displacement_allowance_m,
        "interpolation_and_event_refinement": design.interpolation_event_margin_m,
        "rod_clearance": design.rod_clearance_margin_m,
        "random_filter_halo": filter_halo_x,
    }
    y_margin_parts = {
        "tip_radius": design.max_tip_radius_m,
        "lateral_compliance": design.lateral_compliance_margin_m,
        "interpolation_and_event_refinement": design.interpolation_event_margin_m,
        "rod_clearance": design.rod_clearance_margin_m,
        "random_filter_halo": filter_halo_y,
    }
    margin_x = sum(x_margin_parts.values())
    margin_y = sum(y_margin_parts.values())

    raw_x_min = install_x_min + projection_min - margin_x
    raw_x_max = (
        design.drag_length_m + install_x_max + projection_max + margin_x
    )
    raw_y_min = install_y_min - margin_y
    raw_y_max = install_y_max + margin_y
    x_min = _align_floor(
        raw_x_min, recipe.global_origin_x_m, design.resolution_m
    )
    x_max = _align_ceil(
        raw_x_max, recipe.global_origin_x_m, design.resolution_m
    )
    y_min = _align_floor(
        raw_y_min, recipe.global_origin_y_m, design.resolution_m
    )
    y_max = _align_ceil(
        raw_y_max, recipe.global_origin_y_m, design.resolution_m
    )
    region = RegionSpec(
        terrain_recipe_id=recipe.terrain_recipe_id,
        origin_x_m=x_min,
        origin_y_m=y_min,
        size_x_m=x_max - x_min,
        size_y_m=y_max - y_min,
        resolution_x_m=design.resolution_m,
        resolution_y_m=design.resolution_m,
        purpose="campaign",
    )
    region.validate_against(recipe)
    margins = {
        **{f"x_{key}": value for key, value in x_margin_parts.items()},
        **{f"y_{key}": value for key, value in y_margin_parts.items()},
        "x_total": margin_x,
        "y_total": margin_y,
    }
    return CampaignRegionReport(
        region=region,
        raw_bounds_m={
            "x_min": raw_x_min,
            "x_max": raw_x_max,
            "y_min": raw_y_min,
            "y_max": raw_y_max,
        },
        aligned_bounds_m={
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        margins_m=margins,
        projection_range_m=(projection_min, projection_max),
        installation_span_m=(span_x, span_y),
        assumptions=(
            "array installation coordinates are centred on the campaign origin",
            "drag is along local/global +x for the maximum-region calculation",
            "rod clearance is a declared geometric reserve, not distributed rod contact",
            "random-filter halo is reported even though coordinate-addressable generation "
            "reconstructs it outside the saved region",
        ),
    )
