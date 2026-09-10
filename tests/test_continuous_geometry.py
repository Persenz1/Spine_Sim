"""Physical geometry regressions for continuous guided-spine positions."""

import numpy as np
import pytest

from spine_sim.continuous_geometry import HeightFieldSurface, PlaneSurface


def test_plane_uses_unit_normal_distance_and_retains_penetration_sign():
    surface = PlaneSurface((0, 0, 0.2), (-0.3, 0.4, 1.0))
    center = np.array((0.7, -0.2, 1.2))
    contact = surface.query_sphere(center, 0.1).selected
    expected = (center-surface.point_m) @ surface.normal-0.1
    assert contact.gap_m == pytest.approx(expected)
    np.testing.assert_allclose(center-contact.contact_point_m, (expected+0.1)*contact.normal)
    assert surface.query_sphere(surface.point_m, 0.1).gap_m == pytest.approx(-0.1)


def test_arbitrary_xy_on_sloping_heightfield_matches_plane_not_vertical_gap():
    x, y = np.meshgrid(np.linspace(-2, 2, 9), np.linspace(-2, 2, 9))
    mesh = HeightFieldSurface(0.3*x-0.2*y+0.1, 0.5, 0.5, (-2, -2))
    plane = PlaneSurface((0, 0, 0.1), (-0.3, 0.2, 1))
    for center in ((0.117, -0.233, 0.8), (-0.172, 0.319, 0.5), (0.0, 0.0, 0.4)):
        observed = mesh.query_sphere(center, 0.1).selected
        expected = plane.query_sphere(center, 0.1).selected
        assert observed is not None
        assert observed.gap_m == pytest.approx(expected.gap_m, abs=1e-12)
        np.testing.assert_allclose(observed.contact_point_m, expected.contact_point_m, atol=1e-12)
        np.testing.assert_allclose(observed.normal, expected.normal, atol=1e-12)
        np.testing.assert_allclose(observed.normal_jacobian, 0, atol=1e-12)
    center = np.array((0.117, -0.233, 0.8))
    numerical_gradient = np.array([
        (mesh.query_sphere(center+1e-6*direction, 0.1).gap_m
         -mesh.query_sphere(center-1e-6*direction, 0.1).gap_m)/2e-6
        for direction in np.eye(3)
    ])
    np.testing.assert_allclose(numerical_gradient, mesh.query_sphere(center, 0.1).selected.normal, atol=1e-9)


def test_flat_mesh_seams_and_vertices_do_not_create_multiple_contacts():
    mesh = HeightFieldSurface(np.zeros((7, 7)), 0.5, 0.5)
    for center in ((1, 1, 0.2), (1.2, 1.2, 0.2), (1.2000001, 1.2, 0.2)):
        result = mesh.query_sphere(center, 0.1, tie_tolerance_m=1e-8)
        assert result.status == "ok"
        np.testing.assert_allclose(result.selected.normal, (0, 0, 1), atol=1e-12)
        np.testing.assert_allclose(result.selected.normal_jacobian, 0, atol=1e-12)


def test_valley_reports_two_supports_and_no_invented_unique_normal():
    x, _ = np.meshgrid(np.arange(-3, 4, dtype=float), np.arange(-3, 4, dtype=float))
    mesh = HeightFieldSurface(np.abs(x), 1, 1, (-3, -3))
    result = mesh.query_sphere((0, 0.13, np.sqrt(2)), 1)
    assert result.status == "multiple_contacts"
    assert result.selected is None
    assert len(result.contacts) == 2
    for contact in result.contacts:
        assert contact.gap_m == pytest.approx(0, abs=1e-12)
        assert contact.normal_jacobian is None
    assert result.contacts[0].normal[0]*result.contacts[1].normal[0] < 0


def test_ridge_normal_curvature_matches_center_perturbation():
    x, _ = np.meshgrid(np.arange(-3, 4, dtype=float), np.arange(-3, 4, dtype=float))
    mesh = HeightFieldSurface(-np.abs(x), 1, 1, (-3, -3))
    center = np.array((0.2, 0.13, 0.8))
    result = mesh.query_sphere(center, 0.1)
    assert result.status == "ok"
    assert result.selected.feature_id.startswith("edge:")
    numerical = np.column_stack([
        (mesh.query_sphere(center+1e-6*d, 0.1).selected.normal
         -mesh.query_sphere(center-1e-6*d, 0.1).selected.normal)/2e-6
        for d in np.eye(3)
    ])
    np.testing.assert_allclose(result.selected.normal_jacobian, numerical, atol=1e-8)


def test_capsule_detects_obstacle_between_clear_endpoints():
    height = np.zeros((7, 7))
    height[:, 3] = 1.0
    mesh = HeightFieldSurface(height, 1, 1)
    endpoints = np.array(((1.2, 3.2, 0.7), (4.8, 3.2, 0.7)))
    assert all(mesh.query_sphere(point, 0.1).gap_m > 0 for point in endpoints)
    result = mesh.query_centerline(endpoints, 0.1)
    assert result.status == "collision"
    assert result.gap_m < 0
    np.testing.assert_allclose(result.centerline_point_m, result.surface_point_m, atol=1e-12)


def test_capsule_clearance_on_sloped_plane_matches_heightfield():
    x, y = np.meshgrid(np.arange(-3, 4, dtype=float), np.arange(-3, 4, dtype=float))
    mesh = HeightFieldSurface(0.3*x-0.2*y, 1, 1, (-3, -3))
    plane = PlaneSurface(normal=(-0.3, 0.2, 1))
    points = np.array(((-1, -1, 0.8), (0.1, 0.2, 0.4), (1, 1, 1)))
    observed, expected = mesh.query_centerline(points, 0.1), plane.query_centerline(points, 0.1)
    assert observed.status == "clear"
    assert observed.gap_m == pytest.approx(expected.gap_m, abs=1e-12)


def test_missing_surface_and_finite_domain_are_explicit():
    heights = np.zeros((5, 5))
    mesh = HeightFieldSurface(heights, 1, 1)
    assert mesh.query_sphere((0.1, 1, 0.5), 0.2).status == "out_of_domain"
    assert mesh.query_centerline(((1, 1, 1), (4.1, 1, 1)), 0.1).status == "out_of_domain"
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    incomplete = HeightFieldSurface(heights, 1, 1, valid_mask=mask)
    assert incomplete.query_sphere((2, 2, 0.5), 0.2).status == "invalid_surface"


def test_sphere_center_below_surface_has_negative_signed_distance():
    mesh = HeightFieldSurface(np.zeros((5, 5)), 1, 1)
    contact = mesh.query_sphere((2.1, 2.2, -0.3), 0.1).selected
    assert contact.gap_m == pytest.approx(-0.4)
    np.testing.assert_allclose(contact.normal, (0, 0, 1), atol=1e-12)
