"""Canonical array counts, load sharing, and path-integral metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class SpineMetricInput:
    geometric: bool
    signed_gap_m: float | None
    engagement: bool | None
    active: bool
    normal_force_N: float | None = None
    tangent_resistance_N: float | None = None


@dataclass(frozen=True)
class ArrayCounts:
    n_nominal: int
    n_geometric: int
    n_contact: int
    n_engaged: int | None
    n_engaged_lower: int
    n_engaged_upper: int
    n_active: int
    n_evaluable: int
    P_sum_N: float | None
    P_avg_N: float | None
    n_share_normal: float | None
    n_share_tangent_positive: float | None
    load_sharing_index: float | None


@dataclass(frozen=True)
class PathResistanceMetrics:
    J_positive: float
    J_negative: float
    J_net: float
    effective_length_m: float
    full_length_m: float
    coverage: float
    interval_count: int


def _inverse_simpson(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    total = float(np.sum(array))
    denominator = float(np.dot(array, array))
    if total <= 0.0 or denominator <= 0.0:
        return None
    return total * total / denominator


def compute_array_counts(
    spines: Iterable[SpineMetricInput], *, gap_tolerance_m: float
) -> ArrayCounts:
    """Evaluate physical counts without coercing unknown engagement to false."""

    if not np.isfinite(gap_tolerance_m) or gap_tolerance_m < 0.0:
        raise ValueError("gap_tolerance_m must be finite and non-negative")
    items = tuple(spines)
    contacts = tuple(
        item
        for item in items
        if item.geometric
        and item.signed_gap_m is not None
        and np.isfinite(item.signed_gap_m)
        and item.signed_gap_m <= gap_tolerance_m
    )
    engagement_values = tuple(item.engagement for item in items)
    n_evaluable = sum(value is not None for value in engagement_values)
    n_engaged_lower = sum(value is True for value in engagement_values)
    unknown = sum(value is None for value in engagement_values)
    n_engaged = n_engaged_lower if unknown == 0 else None

    normal_values: list[float] = []
    normal_known = True
    for item in contacts:
        value = item.normal_force_N
        if value is None or not np.isfinite(value):
            normal_known = False
            continue
        if value < 0.0:
            raise ValueError("contact normal force must be non-negative")
        normal_values.append(float(value))
    if normal_known:
        P_sum = float(sum(normal_values))
        P_avg = P_sum / len(contacts) if contacts else None
        n_share_normal = _inverse_simpson(normal_values)
    else:
        P_sum = None
        P_avg = None
        n_share_normal = None

    tangent_positive = [
        float(item.tangent_resistance_N)
        for item in items
        if item.tangent_resistance_N is not None
        and np.isfinite(item.tangent_resistance_N)
        and item.tangent_resistance_N > 0.0
    ]
    active_magnitudes = [
        abs(float(item.tangent_resistance_N))
        for item in items
        if item.active
        and item.tangent_resistance_N is not None
        and np.isfinite(item.tangent_resistance_N)
    ]
    if active_magnitudes and float(np.mean(active_magnitudes)) > 0.0:
        sharing_index = float(
            np.max(active_magnitudes) / np.mean(active_magnitudes)
        )
    else:
        sharing_index = None

    return ArrayCounts(
        n_nominal=len(items),
        n_geometric=sum(item.geometric for item in items),
        n_contact=len(contacts),
        n_engaged=n_engaged,
        n_engaged_lower=n_engaged_lower,
        n_engaged_upper=n_engaged_lower + unknown,
        n_active=sum(item.active for item in items),
        n_evaluable=n_evaluable,
        P_sum_N=P_sum,
        P_avg_N=P_avg,
        n_share_normal=n_share_normal,
        n_share_tangent_positive=_inverse_simpson(tangent_positive),
        load_sharing_index=sharing_index,
    )


def integrate_path_resistance(
    path_position_m: ArrayLike,
    resistance_force_N: ArrayLike,
    *,
    external_normal_preload_N: float,
    accepted: ArrayLike,
    valid: ArrayLike,
) -> PathResistanceMetrics:
    """Integrate J+/J-/Jnet only across adjacent accepted valid samples."""

    x = np.asarray(path_position_m, dtype=float)
    force = np.asarray(resistance_force_N, dtype=float)
    accepted_mask = np.asarray(accepted, dtype=bool)
    valid_mask = np.asarray(valid, dtype=bool)
    if x.ndim != 1 or not (
        x.shape == force.shape == accepted_mask.shape == valid_mask.shape
    ):
        raise ValueError("path, force, accepted, and valid must be equal 1-D arrays")
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("path must contain at least two finite positions")
    dx = np.diff(x)
    if np.any(dx <= 0.0):
        raise ValueError("path positions must be strictly increasing")
    if (
        not np.isfinite(external_normal_preload_N)
        or external_normal_preload_N <= 0.0
    ):
        raise ValueError("external_normal_preload_N must be finite and positive")

    point_valid = accepted_mask & valid_mask & np.isfinite(force)
    interval_valid = point_valid[:-1] & point_valid[1:]
    effective_length = float(np.sum(dx[interval_valid]))
    full_length = float(x[-1] - x[0])
    if effective_length <= 0.0:
        raise ValueError("no accepted valid path interval is available")
    normalized = force / external_normal_preload_N
    positive = np.maximum(normalized, 0.0)
    negative = np.maximum(-normalized, 0.0)
    positive_integral = float(
        np.sum(
            0.5
            * (positive[:-1][interval_valid] + positive[1:][interval_valid])
            * dx[interval_valid]
        )
    )
    negative_integral = float(
        np.sum(
            0.5
            * (negative[:-1][interval_valid] + negative[1:][interval_valid])
            * dx[interval_valid]
        )
    )
    J_positive = positive_integral / effective_length
    J_negative = negative_integral / effective_length
    return PathResistanceMetrics(
        J_positive=J_positive,
        J_negative=J_negative,
        J_net=J_positive - J_negative,
        effective_length_m=effective_length,
        full_length_m=full_length,
        coverage=effective_length / full_length,
        interval_count=int(np.count_nonzero(interval_valid)),
    )
