"""CPU-authoritative finite-sphere envelope, tracks and geometry gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import GeometryOutOfDomainError, TerrainConfigurationError
from .models import (
    ENVELOPE_ALGORITHM_VERSION,
    RegionSpec,
    TrackGeometry,
)


@dataclass(frozen=True)
class SphereEnvelope2D:
    envelope_height_m: NDArray[np.float64]
    envelope_slope_x: NDArray[np.float64]
    envelope_slope_y: NDArray[np.float64] | None
    support_x_m: NDArray[np.float64]
    support_y_m: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    near_tie_flag: NDArray[np.bool_]


@dataclass(frozen=True)
class RodClearanceResult:
    collision: bool | None
    minimum_clearance_m: float | None
    sample_count: int
    model_warning: tuple[str, ...]


def _offsets(
    radius_m: float, dx_m: float, dy_m: float
) -> Iterable[tuple[int, int, float]]:
    if radius_m <= 0 or not math.isfinite(radius_m):
        raise TerrainConfigurationError("radius_m must be finite and positive")
    max_x = int(math.floor(radius_m / dx_m + 1e-12))
    max_y = int(math.floor(radius_m / dy_m + 1e-12))
    radius_squared = radius_m * radius_m
    for offset_y in range(-max_y, max_y + 1):
        physical_y = offset_y * dy_m
        for offset_x in range(-max_x, max_x + 1):
            physical_x = offset_x * dx_m
            rho_squared = physical_x * physical_x + physical_y * physical_y
            if rho_squared <= radius_squared * (1.0 + 1e-12):
                cap = math.sqrt(max(0.0, radius_squared - rho_squared))
                yield offset_y, offset_x, cap


def _slices(offset: int, length: int) -> tuple[slice, slice]:
    if offset >= 0:
        return slice(0, length - offset), slice(offset, length)
    return slice(-offset, length), slice(0, length + offset)


def _update_candidates(
    candidate: NDArray[np.float64],
    best: NDArray[np.float64],
    second: NDArray[np.float64],
    support_x_index: NDArray[np.int32],
    support_y_index: NDArray[np.int32],
    *,
    source_x_indices: NDArray[np.int32],
    source_y_indices: NDArray[np.int32],
) -> None:
    greater = candidate > best
    previous_best = best.copy()
    second[:] = np.where(
        greater, previous_best, np.maximum(second, candidate)
    )
    best[:] = np.where(greater, candidate, previous_best)
    support_x_index[:] = np.where(greater, source_x_indices, support_x_index)
    support_y_index[:] = np.where(greater, source_y_indices, support_y_index)


def compute_sphere_envelope_2d(
    height_m: ArrayLike,
    region: RegionSpec,
    *,
    radius_m: float,
    near_tie_tolerance_m: float = 1e-10,
    compute_slope_y: bool = True,
) -> SphereEnvelope2D:
    """Compute a full 2-D finite-sphere envelope for fixtures and debugging."""

    height = np.asarray(height_m)
    if height.ndim != 2 or height.shape != region.shape:
        raise TerrainConfigurationError("height shape must match RegionSpec")
    if not np.issubdtype(height.dtype, np.floating):
        raise TerrainConfigurationError("height must use a floating dtype")
    if not np.all(np.isfinite(height)):
        raise TerrainConfigurationError("height contains non-finite values")
    if near_tie_tolerance_m < 0:
        raise TerrainConfigurationError("near_tie_tolerance_m must be non-negative")
    ny, nx = height.shape
    best = np.full((ny, nx), -np.inf, dtype=np.float64)
    second = np.full((ny, nx), -np.inf, dtype=np.float64)
    support_x_index = np.full((ny, nx), -1, dtype=np.int32)
    support_y_index = np.full((ny, nx), -1, dtype=np.int32)
    max_offset_x = 0
    max_offset_y = 0

    for offset_y, offset_x, cap in _offsets(
        radius_m, region.resolution_x_m, region.resolution_y_m
    ):
        max_offset_x = max(max_offset_x, abs(offset_x))
        max_offset_y = max(max_offset_y, abs(offset_y))
        target_y, source_y = _slices(offset_y, ny)
        target_x, source_x = _slices(offset_x, nx)
        candidate = (
            np.asarray(height[source_y, source_x], dtype=np.float64) + cap
        )
        best_view = best[target_y, target_x]
        second_view = second[target_y, target_x]
        support_x_view = support_x_index[target_y, target_x]
        support_y_view = support_y_index[target_y, target_x]
        source_x_grid = np.broadcast_to(
            np.arange(source_x.start, source_x.stop, dtype=np.int32)[None, :],
            candidate.shape,
        )
        source_y_grid = np.broadcast_to(
            np.arange(source_y.start, source_y.stop, dtype=np.int32)[:, None],
            candidate.shape,
        )
        _update_candidates(
            candidate,
            best_view,
            second_view,
            support_x_view,
            support_y_view,
            source_x_indices=source_x_grid,
            source_y_indices=source_y_grid,
        )

    slope_y, slope_x = np.gradient(
        best, region.resolution_y_m, region.resolution_x_m, edge_order=2
    )
    geometry_valid = np.ones((ny, nx), dtype=np.bool_)
    if max_offset_y:
        geometry_valid[:max_offset_y, :] = False
        geometry_valid[-max_offset_y:, :] = False
    if max_offset_x:
        geometry_valid[:, :max_offset_x] = False
        geometry_valid[:, -max_offset_x:] = False
    derivative_valid = geometry_valid.copy()
    derivative_valid[:, :1] = False
    derivative_valid[:, -1:] = False
    derivative_valid[:1, :] = False
    derivative_valid[-1:, :] = False
    valid = derivative_valid & np.isfinite(best)
    slope_x = np.where(valid, slope_x, np.nan)
    slope_y_output = np.where(valid, slope_y, np.nan) if compute_slope_y else None
    support_x = (
        region.origin_x_m
        + support_x_index.astype(np.float64) * region.resolution_x_m
    )
    support_y = (
        region.origin_y_m
        + support_y_index.astype(np.float64) * region.resolution_y_m
    )
    support_x = np.where(support_x_index >= 0, support_x, np.nan)
    support_y = np.where(support_y_index >= 0, support_y, np.nan)
    near_tie = (
        np.isfinite(second)
        & ((best - second) <= near_tie_tolerance_m)
        & valid
    )
    return SphereEnvelope2D(
        envelope_height_m=best,
        envelope_slope_x=slope_x,
        envelope_slope_y=slope_y_output,
        support_x_m=support_x,
        support_y_m=support_y,
        valid_mask=valid,
        near_tie_flag=near_tie,
    )


def compute_track_geometry(
    height_m: ArrayLike,
    region: RegionSpec,
    *,
    radius_m: float,
    y_global_m: float,
    near_tie_tolerance_m: float = 1e-10,
) -> TrackGeometry:
    """Compute one fixed-y track without materializing a 2-D envelope."""

    height = np.asanyarray(height_m)
    if height.ndim != 2 or height.shape != region.shape:
        raise TerrainConfigurationError("height shape must match RegionSpec")
    if not np.issubdtype(height.dtype, np.floating):
        raise TerrainConfigurationError("height must use a floating dtype")
    y_index_float = (
        y_global_m - region.origin_y_m
    ) / region.resolution_y_m
    y_index = int(round(y_index_float))
    if not math.isclose(
        y_index_float, y_index, rel_tol=0.0, abs_tol=1e-9
    ):
        raise TerrainConfigurationError("y_global_m must lie on a terrain grid row")
    ny, nx = height.shape
    if not 0 <= y_index < ny:
        raise GeometryOutOfDomainError("geometry_out_of_domain: track y is outside region")
    if near_tie_tolerance_m < 0:
        raise TerrainConfigurationError("near_tie_tolerance_m must be non-negative")

    best = np.full(nx, -np.inf, dtype=np.float64)
    second = np.full(nx, -np.inf, dtype=np.float64)
    support_x_index = np.full(nx, -1, dtype=np.int32)
    support_y_index = np.full(nx, -1, dtype=np.int32)
    max_offset_x = 0
    max_offset_y = 0
    for offset_y, offset_x, cap in _offsets(
        radius_m, region.resolution_x_m, region.resolution_y_m
    ):
        source_y = y_index + offset_y
        max_offset_x = max(max_offset_x, abs(offset_x))
        max_offset_y = max(max_offset_y, abs(offset_y))
        if not 0 <= source_y < ny:
            continue
        target_x, source_x = _slices(offset_x, nx)
        candidate = (
            np.asarray(height[source_y, source_x], dtype=np.float64) + cap
        )
        if not np.all(np.isfinite(candidate)):
            raise TerrainConfigurationError(
                "track source rows contain non-finite terrain heights"
            )
        best_view = best[target_x]
        second_view = second[target_x]
        support_x_view = support_x_index[target_x]
        support_y_view = support_y_index[target_x]
        source_x_indices = np.arange(
            source_x.start, source_x.stop, dtype=np.int32
        )
        source_y_indices = np.full(
            candidate.shape, source_y, dtype=np.int32
        )
        _update_candidates(
            candidate,
            best_view,
            second_view,
            support_x_view,
            support_y_view,
            source_x_indices=source_x_indices,
            source_y_indices=source_y_indices,
        )

    x_global = (
        region.origin_x_m
        + np.arange(nx, dtype=np.float64) * region.resolution_x_m
    )
    slope = np.gradient(best, region.resolution_x_m, edge_order=2)
    geometry_valid = np.ones(nx, dtype=np.bool_)
    if max_offset_x:
        geometry_valid[:max_offset_x] = False
        geometry_valid[-max_offset_x:] = False
    if y_index < max_offset_y or y_index >= ny - max_offset_y:
        geometry_valid[:] = False
    derivative_valid = geometry_valid.copy()
    derivative_valid[:1] = False
    derivative_valid[-1:] = False
    valid = derivative_valid & np.isfinite(best)
    slope = np.where(valid, slope, np.nan)
    support_x = (
        region.origin_x_m
        + support_x_index.astype(np.float64) * region.resolution_x_m
    )
    support_y = (
        region.origin_y_m
        + support_y_index.astype(np.float64) * region.resolution_y_m
    )
    support_x = np.where(support_x_index >= 0, support_x, np.nan)
    support_y = np.where(support_y_index >= 0, support_y, np.nan)
    near_tie = (
        np.isfinite(second)
        & ((best - second) <= near_tie_tolerance_m)
        & valid
    )
    track_id = TrackGeometry.make_id(
        terrain_recipe_id=region.terrain_recipe_id,
        region_id=region.region_id,
        radius_m=radius_m,
        y_global_m=y_global_m,
        envelope_algorithm_version=ENVELOPE_ALGORITHM_VERSION,
        resolution_m=region.resolution_x_m,
    )
    return TrackGeometry(
        terrain_recipe_id=region.terrain_recipe_id,
        region_id=region.region_id,
        track_id=track_id,
        radius_m=radius_m,
        y_global_m=y_global_m,
        resolution_m=region.resolution_x_m,
        envelope_algorithm_version=ENVELOPE_ALGORITHM_VERSION,
        x_global_m=x_global,
        envelope_height_m=best,
        envelope_slope_x=slope,
        support_x_m=support_x,
        support_y_m=support_y,
        valid_mask=valid,
        near_tie_flag=near_tie,
        model_warning=(
            "full_sphere_proxy_requires_forward_cap_gate_in_m2",
            "model_unclosed_rod_collision_until_optional_clearance_check",
        ),
    )


def forward_cap_gate(
    support_xyz_m: ArrayLike,
    sphere_center_xyz_m: ArrayLike,
    tip_axis: ArrayLike,
    *,
    tolerance_m: float = 0.0,
) -> NDArray[np.bool_] | np.bool_:
    """Apply ``(q-c) dot a >= 0`` to scalar or batched support coordinates."""

    support = np.asarray(support_xyz_m, dtype=np.float64)
    center = np.asarray(sphere_center_xyz_m, dtype=np.float64)
    axis = np.asarray(tip_axis, dtype=np.float64)
    if support.shape[-1:] != (3,) or center.shape[-1:] != (3,) or axis.shape != (3,):
        raise TerrainConfigurationError(
            "support/center must end in length 3 and tip_axis must be a 3-vector"
        )
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm <= 0:
        raise TerrainConfigurationError("tip_axis must be finite and non-zero")
    if tolerance_m < 0:
        raise TerrainConfigurationError("tolerance_m must be non-negative")
    unit_axis = axis / norm
    return np.sum((support - center) * unit_axis, axis=-1) >= -tolerance_m


def _bilinear_height(
    height: NDArray[np.floating],
    region: RegionSpec,
    x_m: float,
    y_m: float,
) -> float:
    x_float = (x_m - region.origin_x_m) / region.resolution_x_m
    y_float = (y_m - region.origin_y_m) / region.resolution_y_m
    nx = height.shape[1]
    ny = height.shape[0]
    if x_float < 0 or y_float < 0 or x_float > nx - 1 or y_float > ny - 1:
        raise GeometryOutOfDomainError(
            "geometry_out_of_domain: rod-clearance sample left terrain region"
        )
    x0 = min(int(math.floor(x_float)), nx - 2)
    y0 = min(int(math.floor(y_float)), ny - 2)
    tx = x_float - x0
    ty = y_float - y0
    values = np.asarray(height[y0 : y0 + 2, x0 : x0 + 2], dtype=np.float64)
    return float(
        (1 - ty) * ((1 - tx) * values[0, 0] + tx * values[0, 1])
        + ty * ((1 - tx) * values[1, 0] + tx * values[1, 1])
    )


def check_rod_clearance(
    height_m: ArrayLike,
    region: RegionSpec,
    *,
    sphere_center_xyz_m: ArrayLike,
    tip_axis: ArrayLike,
    exposed_rod_length_m: float | None,
    rod_radius_m: float | None,
    sample_count: int = 32,
) -> RodClearanceResult:
    """Low-cost conservative centreline/cylinder clearance diagnostic."""

    if exposed_rod_length_m is None or rod_radius_m is None:
        return RodClearanceResult(
            collision=None,
            minimum_clearance_m=None,
            sample_count=0,
            model_warning=("model_unclosed_rod_collision",),
        )
    if exposed_rod_length_m <= 0 or rod_radius_m <= 0 or sample_count < 2:
        raise TerrainConfigurationError(
            "rod length/radius must be positive and sample_count at least 2"
        )
    height = np.asanyarray(height_m)
    if height.ndim != 2 or height.shape != region.shape:
        raise TerrainConfigurationError("height shape must match RegionSpec")
    center = np.asarray(sphere_center_xyz_m, dtype=np.float64)
    axis = np.asarray(tip_axis, dtype=np.float64)
    if center.shape != (3,) or axis.shape != (3,) or not np.all(np.isfinite(center)):
        raise TerrainConfigurationError("center and axis must be finite 3-vectors")
    axis_norm = np.linalg.norm(axis)
    if not np.isfinite(axis_norm) or axis_norm <= 0:
        raise TerrainConfigurationError("tip_axis must be finite and non-zero")
    axis = axis / axis_norm
    distances = np.linspace(0.0, exposed_rod_length_m, sample_count)
    points = center[None, :] - distances[:, None] * axis[None, :]
    clearances = np.empty(sample_count, dtype=np.float64)
    for index, point in enumerate(points):
        terrain_height = _bilinear_height(
            height, region, float(point[0]), float(point[1])
        )
        clearances[index] = point[2] - rod_radius_m - terrain_height
    minimum = float(clearances.min())
    return RodClearanceResult(
        collision=minimum < 0.0,
        minimum_clearance_m=minimum,
        sample_count=sample_count,
        model_warning=(
            "low_cost_conservative_cylinder_clearance_not_distributed_contact",
        ),
    )
