from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from spine_sim.terrain.analytic import evaluate_analytic
from spine_sim.terrain.models import RegionSpec, TerrainRecipe
from spine_sim.terrain.random_field import generate_defined_geometry


def region(
    recipe: TerrainRecipe,
    *,
    origin_x_m: float,
    origin_y_m: float,
    size_x_m: float,
    size_y_m: float,
    resolution_m: float,
) -> RegionSpec:
    return RegionSpec(
        terrain_recipe_id=recipe.terrain_recipe_id,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        size_x_m=size_x_m,
        size_y_m=size_y_m,
        resolution_x_m=resolution_m,
        resolution_y_m=resolution_m,
        purpose="debug",
    )


class AnalyticTerrainTests(unittest.TestCase):
    def test_required_fixtures_are_finite_on_arbitrary_grid(self) -> None:
        x = np.linspace(-1e-3, 2e-3, 47)
        y = np.linspace(-0.5e-3, 0.7e-3, 23)
        fixtures = (
            "plane",
            "slope",
            "smooth_bump",
            "double_bump",
            "sine_1d",
            "sine_2d",
            "cross_slope",
        )
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                height = evaluate_analytic(fixture, x, y)
                self.assertEqual(height.shape, (y.size, x.size))
                self.assertTrue(np.all(np.isfinite(height)))

    def test_slope_and_sine_have_expected_coordinate_semantics(self) -> None:
        x = np.array([0.0, 1e-3])
        y = np.array([0.0, 2e-3])
        slope = evaluate_analytic(
            "slope", x, y, {"offset_m": 1e-6, "slope_x": 0.1, "slope_y": -0.2}
        )
        np.testing.assert_allclose(
            slope,
            np.array([[1e-6, 101e-6], [-399e-6, -299e-6]]),
            atol=1e-15,
        )
        sine = evaluate_analytic(
            "sine_1d",
            np.array([0.0, 0.25e-3]),
            y,
            {"amplitude_m": 2e-6, "wavelength_m": 1e-3},
        )
        np.testing.assert_allclose(sine[:, 1], 2e-6, atol=1e-15)


class DefinedGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe = TerrainRecipe(
            seed=1234,
            target_rms_height_m=20e-6,
            correlation_length_x_m=20e-6,
            correlation_length_y_m=25e-6,
            kernel_truncate_sigma=2.5,
        )

    def test_repeat_and_small_window_equal_large_crop_exactly(self) -> None:
        large = region(
            self.recipe,
            origin_x_m=-200e-6,
            origin_y_m=-150e-6,
            size_x_m=600e-6,
            size_y_m=400e-6,
            resolution_m=5e-6,
        )
        small = region(
            self.recipe,
            origin_x_m=-50e-6,
            origin_y_m=-50e-6,
            size_x_m=200e-6,
            size_y_m=150e-6,
            resolution_m=5e-6,
        )
        full_one = generate_defined_geometry(self.recipe, large)
        full_two = generate_defined_geometry(self.recipe, large)
        local = generate_defined_geometry(self.recipe, small)
        self.assertTrue(np.array_equal(full_one, full_two))
        row = int(round((small.origin_y_m - large.origin_y_m) / 5e-6))
        column = int(round((small.origin_x_m - large.origin_x_m) / 5e-6))
        crop = full_one[
            row : row + small.shape[0], column : column + small.shape[1]
        ]
        self.assertTrue(np.array_equal(local, crop))

    def test_adjacent_tiles_have_identical_overlap(self) -> None:
        left = region(
            self.recipe,
            origin_x_m=-200e-6,
            origin_y_m=-100e-6,
            size_x_m=300e-6,
            size_y_m=250e-6,
            resolution_m=5e-6,
        )
        right = region(
            self.recipe,
            origin_x_m=50e-6,
            origin_y_m=-100e-6,
            size_x_m=300e-6,
            size_y_m=250e-6,
            resolution_m=5e-6,
        )
        left_height = generate_defined_geometry(self.recipe, left)
        right_height = generate_defined_geometry(self.recipe, right)
        np.testing.assert_array_equal(left_height[:, -11:], right_height[:, :11])

    def test_production_nodes_are_canonical_even_indices(self) -> None:
        canonical = region(
            self.recipe,
            origin_x_m=-200e-6,
            origin_y_m=-100e-6,
            size_x_m=600e-6,
            size_y_m=400e-6,
            resolution_m=5e-6,
        )
        production = region(
            self.recipe,
            origin_x_m=-200e-6,
            origin_y_m=-100e-6,
            size_x_m=600e-6,
            size_y_m=400e-6,
            resolution_m=10e-6,
        )
        canonical_height = generate_defined_geometry(self.recipe, canonical)
        production_height = generate_defined_geometry(self.recipe, production)
        np.testing.assert_array_equal(production_height, canonical_height[::2, ::2])

    @unittest.skipUnless(importlib.util.find_spec("cupy"), "CuPy/GPU is unavailable")
    def test_cpu_gpu_overlap_is_within_declared_tolerance(self) -> None:
        target = region(
            self.recipe,
            origin_x_m=-100e-6,
            origin_y_m=-100e-6,
            size_x_m=200e-6,
            size_y_m=200e-6,
            resolution_m=5e-6,
        )
        cpu = generate_defined_geometry(self.recipe, target, backend="cpu")
        gpu = generate_defined_geometry(self.recipe, target, backend="cuda")
        np.testing.assert_allclose(cpu, gpu, rtol=2e-6, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
