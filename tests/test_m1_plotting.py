from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spine_sim.terrain.analytic import evaluate_analytic
from spine_sim.terrain.library import TerrainLibrary
from spine_sim.terrain.models import RegionSpec, TerrainRecipe
from spine_sim.terrain.plotting import (
    extract_centered_patch,
    place_sphere_on_patch,
    render_terrain_views,
    select_groove_center,
)


class TerrainPlotGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.region = RegionSpec(
            terrain_recipe_id="terrain_recipe_plot_fixture",
            origin_x_m=-0.3e-3,
            origin_y_m=-0.3e-3,
            size_x_m=0.6e-3,
            size_y_m=0.6e-3,
            resolution_x_m=10e-6,
            resolution_y_m=10e-6,
            purpose="debug",
        )
        ny, nx = self.region.shape
        self.x = self.region.origin_x_m + np.arange(nx) * 10e-6
        self.y = self.region.origin_y_m + np.arange(ny) * 10e-6
        self.height = evaluate_analytic(
            "smooth_bump",
            self.x,
            self.y,
            {
                "amplitude_m": 40e-6,
                "sigma_x_m": 70e-6,
                "sigma_y_m": 50e-6,
            },
        )

    def test_patch_is_bounded_copy_centred_on_grid(self) -> None:
        patch = extract_centered_patch(
            self.height,
            self.region,
            center_x_m=3e-6,
            center_y_m=-4e-6,
            window_size_x_m=0.2e-3,
            window_size_y_m=0.2e-3,
        )
        self.assertEqual(patch.shape, (21, 21))
        self.assertEqual(patch.source_shape, (21, 21))
        self.assertAlmostEqual(patch.center_x_m, 0.0)
        self.assertAlmostEqual(patch.center_y_m, 0.0)
        self.height[...] = -1.0
        self.assertGreater(float(np.max(patch.height_m)), 0.0)

    def test_patch_sampling_keeps_full_extent(self) -> None:
        patch = extract_centered_patch(
            self.height,
            self.region,
            window_size_x_m=0.4e-3,
            maximum_axis_points=11,
        )
        self.assertEqual(patch.shape, (11, 11))
        self.assertEqual(patch.source_shape, (41, 41))
        self.assertAlmostEqual(patch.size_x_m, 0.4e-3)
        self.assertAlmostEqual(patch.size_y_m, 0.4e-3)

    def test_sphere_placement_touches_without_penetration(self) -> None:
        patch = extract_centered_patch(
            self.height, self.region, window_size_x_m=0.4e-3
        )
        placement = place_sphere_on_patch(patch, radius_m=100e-6)
        center = np.asarray(placement.center_xyz_m)
        support = np.asarray(placement.support_xyz_m)
        self.assertAlmostEqual(
            float(np.linalg.norm(center - support)), 100e-6, places=13
        )
        self.assertGreaterEqual(placement.minimum_clearance_m, -1e-15)
        self.assertLessEqual(placement.minimum_clearance_m, 1e-15)

    def test_groove_selector_finds_broad_depression(self) -> None:
        patch = extract_centered_patch(
            -self.height, self.region, window_size_x_m=0.6e-3
        )
        groove = select_groove_center(patch, sphere_radius_m=100e-6)
        self.assertLess(abs(groove.center_x_m), 30e-6)
        self.assertLess(abs(groove.center_y_m), 30e-6)
        self.assertGreater(groove.depth_score_m, 0.0)
        self.assertGreater(
            groove.surrounding_mean_height_m,
            groove.inner_mean_height_m,
        )


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib") is not None,
    "optional plotting dependency is not installed",
)
class TerrainRenderTests(unittest.TestCase):
    def test_render_writes_three_pngs_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = TerrainLibrary(root / "library")
            recipe = TerrainRecipe(
                seed=17,
                target_rms_height_m=20e-6,
                correlation_length_x_m=30e-6,
                correlation_length_y_m=30e-6,
            )
            region = RegionSpec(
                terrain_recipe_id=recipe.terrain_recipe_id,
                origin_x_m=-0.3e-3,
                origin_y_m=-0.3e-3,
                size_x_m=0.6e-3,
                size_y_m=0.6e-3,
                purpose="debug",
            )
            library.generate_region(recipe, region, tile_rows=16)
            result = render_terrain_views(
                library.root,
                recipe.terrain_recipe_id,
                region.region_id,
                root / "output",
                overview_size_m=0.5e-3,
                overview_maximum_axis_points=51,
                surface_maximum_axis_points=41,
                dpi=80,
                prefix="fixture",
            )
            self.assertEqual(len(result["files"]), 3)
            for path in result["files"].values():
                generated = Path(path)
                self.assertTrue(generated.is_file())
                self.assertGreater(generated.stat().st_size, 1000)
            self.assertTrue(Path(result["metadata_path"]).is_file())
            metadata = json.loads(
                Path(result["metadata_path"]).read_text(encoding="utf-8")
            )
            self.assertIn("groove_selection", metadata)
            self.assertLessEqual(
                metadata["rendering"]["oblique_vertical_exaggeration"],
                1.5,
            )


if __name__ == "__main__":
    unittest.main()
