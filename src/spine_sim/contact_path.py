"""Geometric reachability checks between accepted physical configurations.

The center segment is an approximation to the actual sphere-center trajectory.
Its swept capsule detects intervening obstacles; reducing the physical load/path
increment improves that approximation. Optimizer iterates are never supplied as
trajectory points. Passing this check establishes geometric clearance of this
segment, not existence, stability, or uniqueness of the mechanical continuation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .continuous_geometry import ContinuousSurface


@dataclass(frozen=True)
class SpherePathCheck:
    admissible: bool
    requires_refinement: bool
    geometry_event: bool
    status: str
    # Positive clearance is a separation distance. A negative value is a
    # collision certificate, not a deep-penetration solution.
    clearance_m: float | None


def check_sphere_path(
    surface: ContinuousSurface,
    previous_center_m: ArrayLike,
    center_m: ArrayLike,
    radius_m: float,
    *,
    previous_normal: ArrayLike | None = None,
    normal: ArrayLike | None = None,
    previous_feature: Any = None,
    feature: Any = None,
    penetration_tolerance_m: float = 1e-8,
    sharp_normal_angle_rad: float = 0.05,
) -> SpherePathCheck:
    """Check a sphere swept along one proposed physical configuration segment.

    The nominal capsule first checks the finite data domain. If it collides, a
    radius reduced by the allowed penetration tolerance tests whether that
    collision exceeds the tolerance. This second query is necessary because a
    capsule routine can return immediately on a tiny endpoint penetration while
    a much larger obstacle lies in the middle of the segment.

    ``geometry_event`` flags a changed mesh feature with a substantial normal
    change. It is independent of ``admissible``: the caller locates that event in
    its physical path parameter and may accept it once its event tolerance is
    reached. ``requires_refinement`` concerns only failed segment clearance.
    """
    if not 0 <= penetration_tolerance_m < radius_m:
        raise ValueError("sphere path tolerance must be non-negative and smaller than its radius")
    changed = previous_feature is not None and feature is not None and previous_feature != feature
    geometry_event = False
    if changed and previous_normal is not None and normal is not None:
        cosine = float(np.asarray(previous_normal) @ np.asarray(normal))
        geometry_event = bool(np.arccos(np.clip(cosine, -1., 1.)) > sharp_normal_angle_rad)
    points = np.array((previous_center_m, center_m), dtype=float)
    nominal = surface.query_centerline(points, radius_m)
    if nominal.status in {"out_of_domain", "invalid_surface"}:
        return SpherePathCheck(False, True, geometry_event, nominal.status, None)
    if nominal.status == "clear":
        return SpherePathCheck(True, False, geometry_event, "clear", nominal.gap_m)
    reduced = surface.query_centerline(points, radius_m-penetration_tolerance_m)
    if reduced.status in {"out_of_domain", "invalid_surface"}:
        return SpherePathCheck(False, True, geometry_event, reduced.status, None)
    clearance = reduced.gap_m-penetration_tolerance_m
    admissible = reduced.status == "clear"
    return SpherePathCheck(admissible, not admissible, geometry_event,
                           "clear_within_tolerance" if admissible else "intervening_collision",
                           clearance)
