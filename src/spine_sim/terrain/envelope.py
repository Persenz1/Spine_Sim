"""CPU-authoritative finite-sphere envelope, tracks and geometry gates."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import GeometryOutOfDomainError, TerrainConfigurationError
from .models import (
    ENVELOPE_ALGORITHM_VERSION,
    RegionSpec,
    TRACK_SCHEMA_VERSION,
    TrackGeometry,
)
from spine_sim.core.identity import stable_hash


@dataclass(frozen=True)
class SphereEnvelope2D:
    envelope_height_m: NDArray[np.float64]
    envelope_slope_x: NDArray[np.float64]
    envelope_slope_y: NDArray[np.float64] | None
    support_x_m: NDArray[np.float64]
    support_y_m: NDArray[np.float64]
    support_points_m: NDArray[np.float64]
    support_feature_indices_yx: NDArray[np.int64]
    support_value_gap_m: NDArray[np.float64]
    surface_normals: NDArray[np.float64]
    envelope_normals: NDArray[np.float64]
    contact_normals: NDArray[np.float64]
    footprint_valid_mask: NDArray[np.bool_]
    valid_mask: NDArray[np.bool_]
    near_tie_flag: NDArray[np.bool_]
    feature_switch_flag: NDArray[np.bool_]
    geometry_uncertain_mask: NDArray[np.bool_]
    envelope_height_lower_m: NDArray[np.float64] | None
    envelope_height_upper_m: NDArray[np.float64] | None


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
    second_support_x_index: NDArray[np.int32],
    second_support_y_index: NDArray[np.int32],
    *,
    source_x_indices: NDArray[np.int32],
    source_y_indices: NDArray[np.int32],
) -> None:
    greater = candidate > best
    second_greater = (~greater) & (candidate > second)
    previous_best = best.copy()
    previous_support_x = support_x_index.copy()
    previous_support_y = support_y_index.copy()
    second[:] = np.where(
        greater,
        previous_best,
        np.where(second_greater, candidate, second),
    )
    second_support_x_index[:] = np.where(
        greater,
        previous_support_x,
        np.where(
            second_greater,
            source_x_indices,
            second_support_x_index,
        ),
    )
    second_support_y_index[:] = np.where(
        greater,
        previous_support_y,
        np.where(
            second_greater,
            source_y_indices,
            second_support_y_index,
        ),
    )
    best[:] = np.where(greater, candidate, previous_best)
    support_x_index[:] = np.where(greater, source_x_indices, support_x_index)
    support_y_index[:] = np.where(greater, source_y_indices, support_y_index)


def array_sha256(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _source_geometry_inputs(
    height_m: ArrayLike,
    region: RegionSpec,
    *,
    source_valid_mask: ArrayLike | None,
    source_uncertain_mask: ArrayLike | None,
    height_lower_bound_m: ArrayLike | None,
    height_upper_bound_m: ArrayLike | None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
]:
    height = np.asarray(height_m)
    if height.ndim != 2 or height.shape != region.shape:
        raise TerrainConfigurationError("height shape must match RegionSpec")
    if not np.issubdtype(height.dtype, np.floating):
        raise TerrainConfigurationError("height must use a floating dtype")
    if not np.all(np.isfinite(height)):
        raise TerrainConfigurationError("height contains non-finite values")
    valid = (
        np.ones(height.shape, dtype=np.bool_)
        if source_valid_mask is None
        else np.asarray(source_valid_mask)
    )
    if valid.shape != height.shape or valid.dtype != np.bool_:
        raise TerrainConfigurationError(
            "source_valid_mask must be boolean with the height shape"
        )
    uncertain = (
        np.zeros(height.shape, dtype=np.bool_)
        if source_uncertain_mask is None
        else np.asarray(source_uncertain_mask)
    )
    if uncertain.shape != height.shape or uncertain.dtype != np.bool_:
        raise TerrainConfigurationError(
            "source_uncertain_mask must be boolean with the height shape"
        )
    if (height_lower_bound_m is None) != (height_upper_bound_m is None):
        raise TerrainConfigurationError(
            "source height bounds must both be present or both be None"
        )
    lower = None
    upper = None
    if height_lower_bound_m is not None:
        lower = np.asarray(height_lower_bound_m)
        upper = np.asarray(height_upper_bound_m)
        if (
            lower.shape != height.shape
            or upper.shape != height.shape
            or not np.issubdtype(lower.dtype, np.floating)
            or not np.issubdtype(upper.dtype, np.floating)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
        ):
            raise TerrainConfigurationError(
                "source height bounds must be finite floating arrays with the height shape"
            )
        if np.any(lower[valid] > upper[valid]):
            raise TerrainConfigurationError(
                "source lower height bound cannot exceed upper bound"
            )
    return height, valid, uncertain, lower, upper


def _normalized_height_normals(
    height: NDArray[np.floating],
    valid: NDArray[np.bool_],
    region: RegionSpec,
) -> NDArray[np.float64]:
    edge_order = 2 if min(height.shape) >= 3 else 1
    slope_y, slope_x = np.gradient(
        np.asarray(height, dtype=np.float64),
        region.resolution_y_m,
        region.resolution_x_m,
        edge_order=edge_order,
    )
    normals = np.stack((-slope_x, -slope_y, np.ones_like(slope_x)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    normal_valid = np.zeros_like(valid)
    if min(valid.shape) >= 3:
        normal_valid[1:-1, 1:-1] = (
            valid[1:-1, 1:-1]
            & valid[1:-1, :-2]
            & valid[1:-1, 2:]
            & valid[:-2, 1:-1]
            & valid[2:, 1:-1]
        )
    normals[~normal_valid] = np.nan
    return normals


def _normal_from_slopes(
    slope_x: NDArray[np.float64],
    slope_y: NDArray[np.float64],
    valid: NDArray[np.bool_],
    near_tie: NDArray[np.bool_],
) -> NDArray[np.float64]:
    normals = np.stack((-slope_x, -slope_y, np.ones_like(slope_x)), axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    np.divide(normals, norm, out=normals, where=norm > 0)
    normals[~valid | near_tie] = np.nan
    return normals


def _feature_switch(
    feature_indices_yx: NDArray[np.int64],
    near_tie: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    primary = feature_indices_yx[..., 0, :]
    switch = near_tie.copy()
    if primary.ndim == 2:
        changed = np.any(primary[1:] != primary[:-1], axis=-1)
        usable = np.all(primary[1:] >= 0, axis=-1) & np.all(
            primary[:-1] >= 0, axis=-1
        )
        changed &= usable
        switch[1:] |= changed
        switch[:-1] |= changed
        return switch
    changed_x = np.any(primary[:, 1:] != primary[:, :-1], axis=-1)
    usable_x = np.all(primary[:, 1:] >= 0, axis=-1) & np.all(
        primary[:, :-1] >= 0, axis=-1
    )
    changed_x &= usable_x
    switch[:, 1:] |= changed_x
    switch[:, :-1] |= changed_x
    changed_y = np.any(primary[1:, :] != primary[:-1, :], axis=-1)
    usable_y = np.all(primary[1:, :] >= 0, axis=-1) & np.all(
        primary[:-1, :] >= 0, axis=-1
    )
    changed_y &= usable_y
    switch[1:, :] |= changed_y
    switch[:-1, :] |= changed_y
    return switch


def _support_geometry(
    best: NDArray[np.float64],
    second: NDArray[np.float64],
    support_x_index: NDArray[np.int32],
    support_y_index: NDArray[np.int32],
    second_support_x_index: NDArray[np.int32],
    second_support_y_index: NDArray[np.int32],
    height: NDArray[np.floating],
    surface_normal_grid: NDArray[np.float64],
    region: RegionSpec,
    target_x_m: NDArray[np.float64],
    target_y_m: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    shape = best.shape
    support_points = np.full(shape + (2, 3), np.nan, dtype=np.float64)
    feature_indices = np.full(shape + (2, 2), -1, dtype=np.int64)
    surface_normals = np.full(shape + (2, 3), np.nan, dtype=np.float64)
    contact_normals = np.full(shape + (2, 3), np.nan, dtype=np.float64)
    for rank, (x_index, y_index) in enumerate(
        (
            (support_x_index, support_y_index),
            (second_support_x_index, second_support_y_index),
        )
    ):
        present = (x_index >= 0) & (y_index >= 0)
        safe_x = np.where(present, x_index, 0)
        safe_y = np.where(present, y_index, 0)
        feature_indices[..., rank, 0] = np.where(present, y_index, -1)
        feature_indices[..., rank, 1] = np.where(present, x_index, -1)
        support_points[..., rank, 0] = np.where(
            present,
            region.origin_x_m
            + x_index.astype(np.float64) * region.resolution_x_m,
            np.nan,
        )
        support_points[..., rank, 1] = np.where(
            present,
            region.origin_y_m
            + y_index.astype(np.float64) * region.resolution_y_m,
            np.nan,
        )
        support_points[..., rank, 2] = np.where(
            present,
            np.asarray(height[safe_y, safe_x], dtype=np.float64),
            np.nan,
        )
        surface_normals[..., rank, :] = np.where(
            present[..., None], surface_normal_grid[safe_y, safe_x], np.nan
        )
        center = np.stack(
            (
                np.broadcast_to(target_x_m, shape),
                np.broadcast_to(target_y_m, shape),
                best,
            ),
            axis=-1,
        )
        radial = center - support_points[..., rank, :]
        radial_norm = np.linalg.norm(radial, axis=-1, keepdims=True)
        np.divide(
            radial,
            radial_norm,
            out=contact_normals[..., rank, :],
            where=radial_norm > 0,
        )
        contact_normals[..., rank, :][~present] = np.nan
    support_gap = np.where(
        np.isfinite(second), best - second, np.inf
    )
    return support_points, feature_indices, support_gap, surface_normals, contact_normals


def compute_sphere_envelope_2d(
    height_m: ArrayLike,
    region: RegionSpec,
    *,
    radius_m: float,
    near_tie_tolerance_m: float = 1e-10,
    compute_slope_y: bool = True,
    source_valid_mask: ArrayLike | None = None,
    source_uncertain_mask: ArrayLike | None = None,
    height_lower_bound_m: ArrayLike | None = None,
    height_upper_bound_m: ArrayLike | None = None,
) -> SphereEnvelope2D:
    """Compute a full 2-D finite-sphere envelope for fixtures and debugging."""

    height, source_valid, source_uncertain, lower, upper = _source_geometry_inputs(
        height_m,
        region,
        source_valid_mask=source_valid_mask,
        source_uncertain_mask=source_uncertain_mask,
        height_lower_bound_m=height_lower_bound_m,
        height_upper_bound_m=height_upper_bound_m,
    )
    if near_tie_tolerance_m < 0:
        raise TerrainConfigurationError("near_tie_tolerance_m must be non-negative")
    ny, nx = height.shape
    best = np.full((ny, nx), -np.inf, dtype=np.float64)
    second = np.full((ny, nx), -np.inf, dtype=np.float64)
    support_x_index = np.full((ny, nx), -1, dtype=np.int32)
    support_y_index = np.full((ny, nx), -1, dtype=np.int32)
    second_support_x_index = np.full((ny, nx), -1, dtype=np.int32)
    second_support_y_index = np.full((ny, nx), -1, dtype=np.int32)
    footprint_valid = np.ones((ny, nx), dtype=np.bool_)
    footprint_uncertain = np.zeros((ny, nx), dtype=np.bool_)
    best_lower = None if lower is None else np.full((ny, nx), -np.inf)
    best_upper = None if upper is None else np.full((ny, nx), -np.inf)
    max_offset_x = 0
    max_offset_y = 0

    for offset_y, offset_x, cap in _offsets(
        radius_m, region.resolution_x_m, region.resolution_y_m
    ):
        max_offset_x = max(max_offset_x, abs(offset_x))
        max_offset_y = max(max_offset_y, abs(offset_y))
        target_y, source_y = _slices(offset_y, ny)
        target_x, source_x = _slices(offset_x, nx)
        source_is_valid = source_valid[source_y, source_x]
        candidate = np.where(
            source_is_valid,
            np.asarray(height[source_y, source_x], dtype=np.float64) + cap,
            -np.inf,
        )
        best_view = best[target_y, target_x]
        second_view = second[target_y, target_x]
        support_x_view = support_x_index[target_y, target_x]
        support_y_view = support_y_index[target_y, target_x]
        second_support_x_view = second_support_x_index[target_y, target_x]
        second_support_y_view = second_support_y_index[target_y, target_x]
        footprint_valid[target_y, target_x] &= source_is_valid
        footprint_uncertain[target_y, target_x] |= source_uncertain[
            source_y, source_x
        ]
        if best_lower is not None and best_upper is not None:
            best_lower[target_y, target_x] = np.maximum(
                best_lower[target_y, target_x],
                np.where(
                    source_is_valid,
                    np.asarray(lower[source_y, source_x], dtype=np.float64)
                    + cap,
                    -np.inf,
                ),
            )
            best_upper[target_y, target_x] = np.maximum(
                best_upper[target_y, target_x],
                np.where(
                    source_is_valid,
                    np.asarray(upper[source_y, source_x], dtype=np.float64)
                    + cap,
                    -np.inf,
                ),
            )
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
            second_support_x_view,
            second_support_y_view,
            source_x_indices=source_x_grid,
            source_y_indices=source_y_grid,
        )

    best[~np.isfinite(best)] = np.nan
    edge_order = 2 if min(best.shape) >= 3 else 1
    slope_y, slope_x = np.gradient(
        best,
        region.resolution_y_m,
        region.resolution_x_m,
        edge_order=edge_order,
    )
    geometry_valid = np.ones((ny, nx), dtype=np.bool_)
    if max_offset_y:
        geometry_valid[:max_offset_y, :] = False
        geometry_valid[-max_offset_y:, :] = False
    if max_offset_x:
        geometry_valid[:, :max_offset_x] = False
        geometry_valid[:, -max_offset_x:] = False
    footprint_valid &= geometry_valid
    derivative_valid = np.zeros_like(footprint_valid)
    if min(footprint_valid.shape) >= 3:
        derivative_valid[1:-1, 1:-1] = (
            footprint_valid[1:-1, 1:-1]
            & footprint_valid[1:-1, :-2]
            & footprint_valid[1:-1, 2:]
            & footprint_valid[:-2, 1:-1]
            & footprint_valid[2:, 1:-1]
        )
    valid = derivative_valid & np.isfinite(best)
    slope_x = np.where(valid, slope_x, np.nan)
    slope_y_output = np.where(valid, slope_y, np.nan) if compute_slope_y else None
    near_tie = (
        np.isfinite(second)
        & ((best - second) <= near_tie_tolerance_m)
        & footprint_valid
    )
    x_global = region.origin_x_m + np.arange(nx) * region.resolution_x_m
    y_global = region.origin_y_m + np.arange(ny) * region.resolution_y_m
    surface_normal_grid = _normalized_height_normals(
        height, source_valid, region
    )
    (
        support_points,
        feature_indices,
        support_gap,
        surface_normals,
        contact_normals,
    ) = _support_geometry(
        best,
        second,
        support_x_index,
        support_y_index,
        second_support_x_index,
        second_support_y_index,
        height,
        surface_normal_grid,
        region,
        x_global[None, :],
        y_global[:, None],
    )
    envelope_normals = _normal_from_slopes(slope_x, slope_y, valid, near_tie)
    point_uncertain = footprint_uncertain | ~footprint_valid
    if best_lower is not None and best_upper is not None:
        best_lower[~np.isfinite(best_lower)] = np.nan
        best_upper[~np.isfinite(best_upper)] = np.nan
        point_uncertain |= (
            np.isfinite(best_lower)
            & np.isfinite(best_upper)
            & (best_upper > best_lower)
        )
    geometry_uncertain = point_uncertain | ~valid
    if min(point_uncertain.shape) >= 3:
        geometry_uncertain[1:-1, 1:-1] |= (
            point_uncertain[1:-1, :-2]
            | point_uncertain[1:-1, 2:]
            | point_uncertain[:-2, 1:-1]
            | point_uncertain[2:, 1:-1]
        )
    return SphereEnvelope2D(
        envelope_height_m=best,
        envelope_slope_x=slope_x,
        envelope_slope_y=slope_y_output,
        support_x_m=support_points[..., 0, 0],
        support_y_m=support_points[..., 0, 1],
        support_points_m=support_points,
        support_feature_indices_yx=feature_indices,
        support_value_gap_m=support_gap,
        surface_normals=surface_normals,
        envelope_normals=envelope_normals,
        contact_normals=contact_normals,
        footprint_valid_mask=footprint_valid,
        valid_mask=valid,
        near_tie_flag=near_tie,
        feature_switch_flag=_feature_switch(feature_indices, near_tie),
        geometry_uncertain_mask=geometry_uncertain,
        envelope_height_lower_m=best_lower,
        envelope_height_upper_m=best_upper,
    )


@dataclass(frozen=True)
class _TrackSlice:
    best: NDArray[np.float64]
    second: NDArray[np.float64]
    support_x_index: NDArray[np.int32]
    support_y_index: NDArray[np.int32]
    second_support_x_index: NDArray[np.int32]
    second_support_y_index: NDArray[np.int32]
    footprint_valid: NDArray[np.bool_]
    footprint_uncertain: NDArray[np.bool_]
    lower: NDArray[np.float64] | None
    upper: NDArray[np.float64] | None


def _track_slice_uncertainty(value: _TrackSlice) -> NDArray[np.bool_]:
    uncertain = value.footprint_uncertain | ~value.footprint_valid
    if value.lower is not None and value.upper is not None:
        uncertain |= (
            np.isfinite(value.lower)
            & np.isfinite(value.upper)
            & (value.upper > value.lower)
        )
    return uncertain


def _compute_track_slice(
    height: NDArray[np.floating],
    source_valid: NDArray[np.bool_],
    source_uncertain: NDArray[np.bool_],
    lower: NDArray[np.floating] | None,
    upper: NDArray[np.floating] | None,
    region: RegionSpec,
    *,
    radius_m: float,
    y_index: int,
) -> _TrackSlice:
    ny, nx = height.shape
    best = np.full(nx, -np.inf, dtype=np.float64)
    second = np.full(nx, -np.inf, dtype=np.float64)
    support_x_index = np.full(nx, -1, dtype=np.int32)
    support_y_index = np.full(nx, -1, dtype=np.int32)
    second_support_x_index = np.full(nx, -1, dtype=np.int32)
    second_support_y_index = np.full(nx, -1, dtype=np.int32)
    footprint_valid = np.ones(nx, dtype=np.bool_)
    footprint_uncertain = np.zeros(nx, dtype=np.bool_)
    best_lower = None if lower is None else np.full(nx, -np.inf)
    best_upper = None if upper is None else np.full(nx, -np.inf)
    max_offset_x = 0
    max_offset_y = 0
    if not 0 <= y_index < ny:
        footprint_valid.fill(False)
        return _TrackSlice(
            best,
            second,
            support_x_index,
            support_y_index,
            second_support_x_index,
            second_support_y_index,
            footprint_valid,
            np.ones(nx, dtype=np.bool_),
            best_lower,
            best_upper,
        )
    for offset_y, offset_x, cap in _offsets(
        radius_m, region.resolution_x_m, region.resolution_y_m
    ):
        source_y = y_index + offset_y
        max_offset_x = max(max_offset_x, abs(offset_x))
        max_offset_y = max(max_offset_y, abs(offset_y))
        if not 0 <= source_y < ny:
            continue
        target_x, source_x = _slices(offset_x, nx)
        source_is_valid = source_valid[source_y, source_x]
        candidate = np.where(
            source_is_valid,
            np.asarray(height[source_y, source_x], dtype=np.float64) + cap,
            -np.inf,
        )
        source_x_indices = np.arange(
            source_x.start, source_x.stop, dtype=np.int32
        )
        source_y_indices = np.full(candidate.shape, source_y, dtype=np.int32)
        _update_candidates(
            candidate,
            best[target_x],
            second[target_x],
            support_x_index[target_x],
            support_y_index[target_x],
            second_support_x_index[target_x],
            second_support_y_index[target_x],
            source_x_indices=source_x_indices,
            source_y_indices=source_y_indices,
        )
        footprint_valid[target_x] &= source_is_valid
        footprint_uncertain[target_x] |= source_uncertain[source_y, source_x]
        if best_lower is not None and best_upper is not None:
            best_lower[target_x] = np.maximum(
                best_lower[target_x],
                np.where(
                    source_is_valid,
                    np.asarray(lower[source_y, source_x], dtype=np.float64)
                    + cap,
                    -np.inf,
                ),
            )
            best_upper[target_x] = np.maximum(
                best_upper[target_x],
                np.where(
                    source_is_valid,
                    np.asarray(upper[source_y, source_x], dtype=np.float64)
                    + cap,
                    -np.inf,
                ),
            )
    if max_offset_x:
        footprint_valid[:max_offset_x] = False
        footprint_valid[-max_offset_x:] = False
    if y_index < max_offset_y or y_index >= ny - max_offset_y:
        footprint_valid[:] = False
    best[~np.isfinite(best)] = np.nan
    if best_lower is not None and best_upper is not None:
        best_lower[~np.isfinite(best_lower)] = np.nan
        best_upper[~np.isfinite(best_upper)] = np.nan
    return _TrackSlice(
        best,
        second,
        support_x_index,
        support_y_index,
        second_support_x_index,
        second_support_y_index,
        footprint_valid,
        footprint_uncertain,
        best_lower,
        best_upper,
    )


def compute_track_geometry(
    height_m: ArrayLike,
    region: RegionSpec,
    *,
    radius_m: float,
    y_global_m: float,
    near_tie_tolerance_m: float = 1e-10,
    source_valid_mask: ArrayLike | None = None,
    source_uncertain_mask: ArrayLike | None = None,
    height_lower_bound_m: ArrayLike | None = None,
    height_upper_bound_m: ArrayLike | None = None,
    source_data_sha256: str | None = None,
    source_valid_mask_sha256: str | None = None,
    measurement_semantics_hash: str | None = None,
) -> TrackGeometry:
    """Compute one fixed-y track without materializing a 2-D envelope."""

    height, source_valid, source_uncertain, lower, upper = _source_geometry_inputs(
        height_m,
        region,
        source_valid_mask=source_valid_mask,
        source_uncertain_mask=source_uncertain_mask,
        height_lower_bound_m=height_lower_bound_m,
        height_upper_bound_m=height_upper_bound_m,
    )
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

    center = _compute_track_slice(
        height,
        source_valid,
        source_uncertain,
        lower,
        upper,
        region,
        radius_m=radius_m,
        y_index=y_index,
    )
    below = _compute_track_slice(
        height,
        source_valid,
        source_uncertain,
        lower,
        upper,
        region,
        radius_m=radius_m,
        y_index=y_index - 1,
    )
    above = _compute_track_slice(
        height,
        source_valid,
        source_uncertain,
        lower,
        upper,
        region,
        radius_m=radius_m,
        y_index=y_index + 1,
    )

    x_global = (
        region.origin_x_m
        + np.arange(nx, dtype=np.float64) * region.resolution_x_m
    )
    edge_order = 2 if nx >= 3 else 1
    slope_x = np.gradient(
        center.best, region.resolution_x_m, edge_order=edge_order
    )
    slope_y = (above.best - below.best) / (2.0 * region.resolution_y_m)
    derivative_valid = np.zeros(nx, dtype=np.bool_)
    if nx >= 3:
        derivative_valid[1:-1] = (
            center.footprint_valid[1:-1]
            & center.footprint_valid[:-2]
            & center.footprint_valid[2:]
            & below.footprint_valid[1:-1]
            & above.footprint_valid[1:-1]
        )
    valid = derivative_valid & np.isfinite(center.best)
    slope_x = np.where(valid, slope_x, np.nan)
    slope_y = np.where(valid, slope_y, np.nan)
    near_tie = (
        np.isfinite(center.second)
        & ((center.best - center.second) <= near_tie_tolerance_m)
        & center.footprint_valid
    )
    surface_normal_grid = _normalized_height_normals(
        height, source_valid, region
    )
    (
        support_points,
        feature_indices,
        support_gap,
        surface_normals,
        contact_normals,
    ) = _support_geometry(
        center.best,
        center.second,
        center.support_x_index,
        center.support_y_index,
        center.second_support_x_index,
        center.second_support_y_index,
        height,
        surface_normal_grid,
        region,
        x_global,
        np.asarray(y_global_m),
    )
    envelope_normals = _normal_from_slopes(slope_x, slope_y, valid, near_tie)
    center_uncertain = _track_slice_uncertainty(center)
    below_uncertain = _track_slice_uncertainty(below)
    above_uncertain = _track_slice_uncertainty(above)
    geometry_uncertain = center_uncertain | ~valid
    if nx >= 3:
        geometry_uncertain[1:-1] |= (
            center_uncertain[:-2]
            | center_uncertain[2:]
            | below_uncertain[1:-1]
            | above_uncertain[1:-1]
        )
    data_digest = source_data_sha256 or array_sha256(
        np.asarray(height)
    )
    mask_digest = source_valid_mask_sha256 or array_sha256(source_valid)
    measurement_digest = measurement_semantics_hash or stable_hash(
        {
            "source_uncertain_mask_sha256": array_sha256(source_uncertain),
            "has_height_bounds": lower is not None,
            "height_lower_sha256": (
                None if lower is None else array_sha256(np.asarray(lower))
            ),
            "height_upper_sha256": (
                None if upper is None else array_sha256(np.asarray(upper))
            ),
        }
    )
    track_id = TrackGeometry.make_id(
        terrain_recipe_id=region.terrain_recipe_id,
        region_id=region.region_id,
        radius_m=radius_m,
        y_global_m=y_global_m,
        track_schema_version=TRACK_SCHEMA_VERSION,
        envelope_algorithm_version=ENVELOPE_ALGORITHM_VERSION,
        near_tie_tolerance_m=near_tie_tolerance_m,
        resolution_m=region.resolution_x_m,
        source_data_sha256=data_digest,
        source_valid_mask_sha256=mask_digest,
        measurement_semantics_hash=measurement_digest,
    )
    return TrackGeometry(
        terrain_recipe_id=region.terrain_recipe_id,
        region_id=region.region_id,
        track_id=track_id,
        radius_m=radius_m,
        y_global_m=y_global_m,
        resolution_m=region.resolution_x_m,
        track_schema_version=TRACK_SCHEMA_VERSION,
        envelope_algorithm_version=ENVELOPE_ALGORITHM_VERSION,
        near_tie_tolerance_m=near_tie_tolerance_m,
        source_data_sha256=data_digest,
        source_valid_mask_sha256=mask_digest,
        measurement_semantics_hash=measurement_digest,
        x_global_m=x_global,
        envelope_height_m=center.best,
        envelope_slope_x=slope_x,
        envelope_slope_y=slope_y,
        support_x_m=support_points[..., 0, 0],
        support_y_m=support_points[..., 0, 1],
        support_points_m=support_points,
        support_feature_indices_yx=feature_indices,
        support_value_gap_m=support_gap,
        surface_normals=surface_normals,
        envelope_normals=envelope_normals,
        contact_normals=contact_normals,
        footprint_valid_mask=center.footprint_valid,
        valid_mask=valid,
        near_tie_flag=near_tie,
        feature_switch_flag=_feature_switch(feature_indices, near_tie),
        geometry_uncertain_mask=geometry_uncertain,
        envelope_height_lower_m=center.lower,
        envelope_height_upper_m=center.upper,
        model_warning=(
            "full_sphere_proxy_requires_forward_cap_gate_in_m2",
            "model_unclosed_rod_collision_until_optional_clearance_check",
            *(
                ("measured_geometry_uncertain",)
                if np.any(geometry_uncertain)
                else ()
            ),
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


def check_segmented_tip_rod_clearance(
    height_m: ArrayLike,
    region: RegionSpec,
    *,
    sphere_center_xyz_m: ArrayLike,
    tip_axis: ArrayLike,
    tip_radius_m: float,
    spherical_cap_axial_length_m: float | None,
    cone_length_m: float | None,
    exposed_rod_length_m: float | None,
    rod_radius_m: float | None,
    source_valid_mask: ArrayLike | None = None,
    axial_sample_count: int = 32,
    perimeter_sample_count: int = 16,
) -> RodClearanceResult:
    """Sample the actual rear sphere-cap, cone and cylindrical rod surface."""

    dimensions = (
        spherical_cap_axial_length_m,
        cone_length_m,
        exposed_rod_length_m,
        rod_radius_m,
    )
    if any(value is None for value in dimensions):
        return RodClearanceResult(
            collision=None,
            minimum_clearance_m=None,
            sample_count=0,
            model_warning=("model_unclosed_segmented_tip_rod_geometry",),
        )
    cap_length = float(spherical_cap_axial_length_m)
    cone_length = float(cone_length_m)
    rod_length = float(exposed_rod_length_m)
    rod_radius = float(rod_radius_m)
    if (
        not math.isfinite(tip_radius_m)
        or tip_radius_m <= 0.0
        or not 0.0 < cap_length <= tip_radius_m
        or cone_length <= 0.0
        or rod_length <= 0.0
        or rod_radius <= 0.0
        or axial_sample_count < 3
        or perimeter_sample_count < 8
    ):
        raise TerrainConfigurationError(
            "segmented tip/rod dimensions and sampling counts are invalid"
        )
    height = np.asarray(height_m)
    if height.shape != region.shape or not np.issubdtype(height.dtype, np.floating):
        raise TerrainConfigurationError("height shape/dtype must match RegionSpec")
    valid_mask = (
        np.ones(region.shape, dtype=np.bool_)
        if source_valid_mask is None
        else np.asarray(source_valid_mask)
    )
    if valid_mask.shape != region.shape or valid_mask.dtype != np.bool_:
        raise TerrainConfigurationError(
            "source_valid_mask must be boolean with the height shape"
        )
    center = np.asarray(sphere_center_xyz_m, dtype=np.float64)
    axis = np.asarray(tip_axis, dtype=np.float64)
    if center.shape != (3,) or axis.shape != (3,) or not np.all(np.isfinite(center)):
        raise TerrainConfigurationError("center and axis must be finite 3-vectors")
    axis_norm = float(np.linalg.norm(axis))
    if not math.isfinite(axis_norm) or axis_norm <= 0.0:
        raise TerrainConfigurationError("tip_axis must be finite and non-zero")
    axis /= axis_norm
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(axis, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)

    total_length = cap_length + cone_length + rod_length
    axial = np.linspace(0.0, total_length, axial_sample_count)
    cap_join_radius = math.sqrt(
        max(0.0, tip_radius_m * tip_radius_m - cap_length * cap_length)
    )
    radii = np.empty_like(axial)
    cap_mask = axial <= cap_length
    cone_mask = (axial > cap_length) & (axial <= cap_length + cone_length)
    radii[cap_mask] = np.sqrt(
        np.maximum(0.0, tip_radius_m * tip_radius_m - axial[cap_mask] ** 2)
    )
    cone_fraction = (axial[cone_mask] - cap_length) / cone_length
    radii[cone_mask] = cap_join_radius + cone_fraction * (
        rod_radius - cap_join_radius
    )
    radii[~(cap_mask | cone_mask)] = rod_radius

    angles = np.linspace(0.0, 2.0 * np.pi, perimeter_sample_count, endpoint=False)
    radial_directions = (
        np.cos(angles)[:, None] * basis_u[None, :]
        + np.sin(angles)[:, None] * basis_v[None, :]
    )
    radial_fractions = (0.0, 0.5, 1.0)
    minimum = math.inf
    evaluated = 0
    for distance, radius in zip(axial, radii, strict=True):
        axis_point = center - float(distance) * axis
        for fraction in radial_fractions:
            if fraction == 0.0:
                points = axis_point[None, :]
            else:
                points = (
                    axis_point[None, :]
                    + float(fraction * radius) * radial_directions
                )
            for point in points:
                x_float = (
                    float(point[0]) - region.origin_x_m
                ) / region.resolution_x_m
                y_float = (
                    float(point[1]) - region.origin_y_m
                ) / region.resolution_y_m
                if (
                    x_float < 0.0
                    or y_float < 0.0
                    or x_float > height.shape[1] - 1
                    or y_float > height.shape[0] - 1
                ):
                    raise GeometryOutOfDomainError(
                        "geometry_out_of_domain: segmented clearance left terrain region"
                    )
                x0 = min(int(math.floor(x_float)), height.shape[1] - 2)
                y0 = min(int(math.floor(y_float)), height.shape[0] - 2)
                if not np.all(valid_mask[y0 : y0 + 2, x0 : x0 + 2]):
                    return RodClearanceResult(
                        collision=None,
                        minimum_clearance_m=None,
                        sample_count=evaluated,
                        model_warning=(
                            "segmented_tip_rod_clearance_crosses_unknown_terrain",
                        ),
                    )
                clearance = float(point[2]) - _bilinear_height(
                    height,
                    region,
                    float(point[0]),
                    float(point[1]),
                )
                minimum = min(minimum, clearance)
                evaluated += 1
    return RodClearanceResult(
        collision=minimum < 0.0,
        minimum_clearance_m=minimum,
        sample_count=evaluated,
        model_warning=("sampled_segmented_axisymmetric_tip_rod_clearance",),
    )
