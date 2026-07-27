from __future__ import annotations

import unittest

import numpy as np

from spine_sim.terrain.analytic import evaluate_analytic
from spine_sim.terrain.envelope import (
    check_rod_clearance,
    compute_sphere_envelope_2d,
    compute_track_geometry,
    forward_cap_gate,
)
from spine_sim.terrain.models import RegionSpec


def fixture_region(dx: float = 10e-6) -> RegionSpec:
    return RegionSpec(
        terrain_recipe_id="terrain_recipe_fixture",
        origin_x_m=-0.5e-3,
        origin_y_m=-0.25e-3,
        size_x_m=1e-3,
        size_y_m=0.5e-3,
        resolution_x_m=dx,
        resolution_y_m=dx,
        purpose="debug",
    )


def axes(region: RegionSpec) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = region.shape
    x = region.origin_x_m + np.arange(nx) * region.resolution_x_m
    y = region.origin_y_m + np.arange(ny) * region.resolution_y_m
    return x, y


class EnvelopeTests(unittest.TestCase):
    def test_plane_envelope_is_constant(self) -> None:
        region = fixture_region()
        x, y = axes(region)
        height = evaluate_analytic("plane", x, y, {"offset_m": 7e-6})
        result = compute_sphere_envelope_2d(height, region, radius_m=50e-6)
        np.testing.assert_allclose(
            result.envelope_height_m[result.valid_mask], 57e-6, atol=1e-15
        )
        np.testing.assert_allclose(
            result.envelope_slope_x[result.valid_mask], 0.0, atol=1e-12
        )

    def test_slope_support_offset_matches_discrete_analytic_maximum(self) -> None:
        region = fixture_region()
        x, y = axes(region)
        slope_x = 0.2
        height = evaluate_analytic("slope", x, y, {"slope_x": slope_x})
        radius = 100e-6
        result = compute_sphere_envelope_2d(height, region, radius_m=radius)
        row = region.shape[0] // 2
        valid_columns = np.flatnonzero(result.valid_mask[row])
        support_offset = (
            result.support_x_m[row, valid_columns] - x[valid_columns]
        )
        continuous_offset = radius * slope_x / np.sqrt(1.0 + slope_x**2)
        expected_offset = (
            round(continuous_offset / region.resolution_x_m)
            * region.resolution_x_m
        )
        np.testing.assert_allclose(support_offset, expected_offset, atol=1e-15)
        np.testing.assert_allclose(
            result.envelope_slope_x[row, valid_columns], slope_x, atol=1e-12
        )
        expected_lift = (
            slope_x * expected_offset
            + np.sqrt(radius**2 - expected_offset**2)
        )
        np.testing.assert_allclose(
            result.envelope_height_m[row, valid_columns]
            - slope_x * x[valid_columns],
            expected_lift,
            atol=1e-15,
        )

    def test_larger_radius_reaches_smooth_bump_no_later(self) -> None:
        region = fixture_region(dx=5e-6)
        x, y = axes(region)
        height = evaluate_analytic(
            "smooth_bump",
            x,
            y,
            {
                "amplitude_m": 100e-6,
                "sigma_x_m": 40e-6,
                "sigma_y_m": 40e-6,
            },
        )
        small = compute_track_geometry(
            height, region, radius_m=50e-6, y_global_m=0.0
        )
        large = compute_track_geometry(
            height, region, radius_m=100e-6, y_global_m=0.0
        )
        threshold = 2e-6
        small_effective = small.envelope_height_m - small.radius_m
        large_effective = large.envelope_height_m - large.radius_m
        small_first = x[np.flatnonzero((small_effective > threshold) & small.valid_mask)[0]]
        large_first = x[np.flatnonzero((large_effective > threshold) & large.valid_mask)[0]]
        self.assertLessEqual(large_first, small_first)

    def test_track_matches_direct_2d_row(self) -> None:
        region = fixture_region()
        x, y = axes(region)
        height = evaluate_analytic(
            "double_bump",
            x,
            y,
            {"center_1_x_m": -150e-6, "center_2_x_m": 170e-6},
        )
        full = compute_sphere_envelope_2d(height, region, radius_m=50e-6)
        track = compute_track_geometry(
            height, region, radius_m=50e-6, y_global_m=0.0
        )
        row = int(round((0.0 - region.origin_y_m) / region.resolution_y_m))
        np.testing.assert_array_equal(track.valid_mask, full.valid_mask[row])
        np.testing.assert_allclose(
            track.envelope_height_m, full.envelope_height_m[row], atol=0.0
        )
        np.testing.assert_allclose(
            track.envelope_slope_x[track.valid_mask],
            full.envelope_slope_x[row, track.valid_mask],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            track.support_x_m, full.support_x_m[row], atol=0.0
        )
        np.testing.assert_allclose(
            track.support_y_m, full.support_y_m[row], atol=0.0
        )

    def test_grid_refinement_converges_for_smooth_sine(self) -> None:
        coarse_region = fixture_region(10e-6)
        fine_region = fixture_region(5e-6)
        coarse_x, coarse_y = axes(coarse_region)
        fine_x, fine_y = axes(fine_region)
        parameters = {"amplitude_m": 20e-6, "wavelength_m": 300e-6}
        coarse_height = evaluate_analytic(
            "sine_1d", coarse_x, coarse_y, parameters
        )
        fine_height = evaluate_analytic("sine_1d", fine_x, fine_y, parameters)
        coarse = compute_track_geometry(
            coarse_height, coarse_region, radius_m=50e-6, y_global_m=0.0
        )
        fine = compute_track_geometry(
            fine_height, fine_region, radius_m=50e-6, y_global_m=0.0
        )
        common = coarse.valid_mask & fine.valid_mask[::2]
        height_error = np.max(
            np.abs(coarse.envelope_height_m[common] - fine.envelope_height_m[::2][common])
        )
        slope_error = np.max(
            np.abs(coarse.envelope_slope_x[common] - fine.envelope_slope_x[::2][common])
        )
        self.assertLess(height_error, 1e-6)
        self.assertLess(slope_error, 0.03)

        def upward_event_position(track, threshold_m: float) -> float:
            effective = track.envelope_height_m - track.radius_m
            for index in range(effective.size - 1):
                if (
                    track.valid_mask[index]
                    and track.valid_mask[index + 1]
                    and effective[index] < threshold_m <= effective[index + 1]
                ):
                    fraction = (threshold_m - effective[index]) / (
                        effective[index + 1] - effective[index]
                    )
                    return float(
                        track.x_global_m[index]
                        + fraction
                        * (
                            track.x_global_m[index + 1]
                            - track.x_global_m[index]
                        )
                    )
            raise AssertionError("fixture did not contain the expected upward event")

        coarse_event = upward_event_position(coarse, 5e-6)
        fine_event = upward_event_position(fine, 5e-6)
        self.assertLess(abs(coarse_event - fine_event), 10e-6)


class GateTests(unittest.TestCase):
    def test_forward_cap_gate(self) -> None:
        center = np.array([0.0, 0.0, 1.0])
        supports = np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        np.testing.assert_array_equal(
            forward_cap_gate(supports, center, np.array([1.0, 0.0, 0.0])),
            np.array([True, False]),
        )

    def test_rod_clearance_reports_unclosed_or_low_cost_result(self) -> None:
        region = fixture_region()
        height = np.zeros(region.shape)
        unclosed = check_rod_clearance(
            height,
            region,
            sphere_center_xyz_m=(0.0, 0.0, 1e-3),
            tip_axis=(1.0, 0.0, -1.0),
            exposed_rod_length_m=None,
            rod_radius_m=None,
        )
        self.assertIsNone(unclosed.collision)
        self.assertIn("model_unclosed_rod_collision", unclosed.model_warning)
        checked = check_rod_clearance(
            height,
            region,
            sphere_center_xyz_m=(0.2e-3, 0.0, 1e-3),
            tip_axis=(1.0, 0.0, -1.0),
            exposed_rod_length_m=0.2e-3,
            rod_radius_m=50e-6,
        )
        self.assertFalse(checked.collision)
        self.assertGreater(checked.minimum_clearance_m or 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
