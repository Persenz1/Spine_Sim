"""Actual shaft/taper clearance, including narrow obstacles between rod nodes."""

import numpy as np
import pytest

from spine_sim.continuous_geometry import HeightFieldSurface, PlaneSurface
from spine_sim.guided_rod import GuidedRod, GuidedRodParameters
from spine_sim.tapered_geometry import query_tapered_rod_clearance


def needle(radius=50e-6, angle=50):
    parameters = GuidedRodParameters(4e-3, 1e-3, 200e9, 1000, 4e-3, radius,
                                      taper_length_m=2e-3)
    direction = np.array((np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))))
    tip = np.array((0, 0, radius))
    return GuidedRod(parameters, tip-4e-3*direction, direction)


@pytest.mark.parametrize("radius", [50e-6, 100e-6])
def test_real_taper_clears_plane_where_max_radius_capsule_falsely_penetrates(radius):
    rod = needle(radius)
    state = rod.zero_state()
    plane = PlaneSurface()
    old = plane.query_centerline(rod.evaluate(state, derivatives=False).centerline_m, 0.5e-3)
    assert old.status == "collision"
    result = query_tapered_rod_clearance(plane, rod, state, np.zeros(3), 2e-9)
    assert result.status == "clear"
    assert result.gap_m > 0
    mesh = HeightFieldSurface(np.zeros((41, 101)), 0.1e-3, 0.1e-3, (-6e-3, -2e-3))
    assert query_tapered_rod_clearance(mesh, rod, state, np.zeros(3), 2e-9).status == "clear"


def test_actual_taper_penetration_is_retained():
    rod = needle()
    result = query_tapered_rod_clearance(PlaneSurface(), rod, rod.zero_state(),
                                        np.array((0, 0, -0.15e-3)), 2e-9)
    assert result.status == "collision"
    assert result.gap_m < -2e-9


def test_narrow_heightfield_peak_between_clear_rod_nodes_is_detected():
    spacing = 50e-6
    height = np.zeros((41, 121))
    height[:, 29] = 0.5e-3
    mesh = HeightFieldSurface(height, spacing, spacing, (0, -1e-3))
    parameters = GuidedRodParameters(4e-3, 0.1e-3, 200e9, 1000, 4e-3, 50e-6,
                                      taper_length_m=2e-3, segments=1)
    rod = GuidedRod(parameters, (0.2e-3, 0, 0.3e-3), (1, 0, 0))
    state = rod.zero_state()
    assert all(mesh.query_sphere(point, 50e-6).gap_m > 0
               for point in rod.evaluate(state, derivatives=False).centerline_m)
    result = query_tapered_rod_clearance(mesh, rod, state, np.zeros(3), 2e-9)
    assert result.status == "collision"
    assert result.gap_m < 0


def test_actual_footprint_outside_domain_and_missing_vertices_remain_explicit():
    rod = needle()
    state = rod.zero_state()
    mesh = HeightFieldSurface(np.zeros((41, 101)), 0.1e-3, 0.1e-3, (-6e-3, -2e-3))
    assert query_tapered_rod_clearance(mesh, rod, state, np.array((0, 3e-3, 0)),
                                      2e-9).status == "out_of_domain"
    mask = np.ones((41, 101), dtype=bool)
    mask[20, 60] = False  # The ball-centre cross-section lies over this missing vertex.
    missing = HeightFieldSurface(np.zeros_like(mask, dtype=float), 0.1e-3, 0.1e-3,
                                 (-6e-3, -2e-3), valid_mask=mask)
    assert query_tapered_rod_clearance(missing, rod, state, np.zeros(3),
                                      2e-9).status == "invalid_surface"
