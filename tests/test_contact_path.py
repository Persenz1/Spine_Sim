import numpy as np
import pytest

from spine_sim.contact_path import check_sphere_path
from spine_sim.continuous_geometry import HeightFieldSurface, PlaneSurface


def test_flat_contact_slide_is_admissible_and_feature_label_alone_is_not_event():
    result = check_sphere_path(PlaneSurface(), (0, 0, 0.1), (1, 0.2, 0.1), 0.1,
                               previous_feature="face:1", feature="face:2",
                               previous_normal=(0, 0, 1), normal=(0, 0, 1))
    assert result.admissible
    assert not result.requires_refinement
    assert not result.geometry_event
    assert result.clearance_m == pytest.approx(0)


def test_intervening_peak_blocks_a_path_whose_endpoints_are_clear():
    height = np.zeros((7, 7))
    height[:, 3] = 1
    surface = HeightFieldSurface(height, 1, 1)
    start, end = (1.2, 3.2, 0.7), (4.8, 3.2, 0.7)
    assert surface.query_sphere(start, 0.1).gap_m > 0
    assert surface.query_sphere(end, 0.1).gap_m > 0
    result = check_sphere_path(surface, start, end, 0.1)
    assert not result.admissible
    assert result.requires_refinement
    assert result.status == "intervening_collision"


def test_tiny_endpoint_penetration_does_not_hide_a_large_intervening_obstacle():
    height = np.zeros((7, 7))
    height[:, 3] = 1
    surface = HeightFieldSurface(height, 1, 1)
    result = check_sphere_path(surface, (1.2, 3.2, 0.1-1e-10), (4.8, 3.2, 0.1), 0.1)
    assert not result.admissible
    assert result.clearance_m < -1e-8


def test_curved_contact_chord_becomes_admissible_when_step_is_reduced():
    x, _ = np.meshgrid(np.arange(-3, 4, dtype=float), np.arange(-3, 4, dtype=float))
    surface = HeightFieldSurface(-np.abs(x), 1, 1, (-3, -3))
    radius, tolerance = 0.5, 1e-5
    def center(angle):
        return (radius*np.sin(angle), 0.13, radius*np.cos(angle))
    long = check_sphere_path(surface, center(-0.1), center(0.1), radius,
                             penetration_tolerance_m=tolerance)
    short = check_sphere_path(surface, center(-0.001), center(0.001), radius,
                              penetration_tolerance_m=tolerance)
    assert not long.admissible
    assert short.admissible


def test_sharp_feature_event_is_independent_of_path_clearance():
    result = check_sphere_path(PlaneSurface(), (0, 0, 1), (0.1, 0, 1), 0.1,
                               previous_feature="face:1", feature="face:2",
                               previous_normal=(0, 0, 1), normal=(0.6, 0, 0.8))
    assert result.admissible
    assert result.geometry_event
    assert not result.requires_refinement


def test_nominal_sphere_domain_is_checked_before_tolerance_reduction():
    surface = HeightFieldSurface(np.zeros((5, 5)), 1, 1)
    result = check_sphere_path(surface, (0.1-1e-10, 1, 0.2), (0.2, 1, 0.2), 0.1)
    assert not result.admissible
    assert result.status == "out_of_domain"
