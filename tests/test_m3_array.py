import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spine_sim.array import (
    AngleLayout,
    ArrayConfiguration,
    ArrayExperimentSettings,
    CommonBackplateArray,
    CommonBackplateExperiment,
    build_candidate_pool,
    select_balanced_candidates,
)
from spine_sim.array.case import _arrays, run_case
from spine_sim.array.validation import (
    _plane_track,
    _tracks_for_configuration,
    run_analytic_validation,
)
from spine_sim.contact.validation import _fixture_parameters
from spine_sim.runtime.runner import RunContext
from spine_sim.terrain import RegionSpec, TerrainLibrary, TerrainRecipe


class M3GeometryAndAtomicityTests(unittest.TestCase):
    def setUp(self):
        self.parameters = _fixture_parameters()
        self.configuration = ArrayConfiguration(
            2,
            2,
            5e-3,
            self.parameters,
        )
        self.system = CommonBackplateArray(
            self.configuration,
            _tracks_for_configuration(self.configuration),
            unit_origin_xy_m=(0.0, 0.0),
        )

    def test_production_geometry_rejects_singleton_dimension(self):
        with self.assertRaises(ValueError):
            ArrayConfiguration(2, 1, 4e-3, self.parameters)
        fixture = ArrayConfiguration(
            2,
            1,
            4e-3,
            self.parameters,
            fixture_only=True,
        )
        self.assertEqual(fixture.pin_count, 2)

    def test_gradient_lengths_preserve_vertical_reach(self):
        for layout in (
            AngleLayout.GRADIENT_80_TO_60,
            AngleLayout.GRADIENT_80_TO_50,
        ):
            configuration = ArrayConfiguration(
                5,
                2,
                5e-3,
                self.parameters,
                angle_layout=layout,
            )
            reach = [
                pin.exposed_length_m
                * np.sin(np.radians(pin.installation_angle_deg))
                for pin in configuration.pin_parameters[: configuration.nx]
            ]
            self.assertLess(np.ptp(reach), 1e-15)

    def test_tracks_must_share_global_recipe_and_match_pin_y(self):
        tracks = list(_tracks_for_configuration(self.configuration))
        tracks[0] = replace(tracks[0], terrain_recipe_id="different_recipe")
        with self.assertRaises(ValueError):
            CommonBackplateArray(
                self.configuration,
                tracks,
                unit_origin_xy_m=(0.0, 0.0),
            )
        wrong_y = list(_tracks_for_configuration(self.configuration))
        wrong_y[0] = _plane_track(y_m=0.0)
        with self.assertRaises(ValueError):
            CommonBackplateArray(
                self.configuration,
                wrong_y,
                unit_origin_xy_m=(0.0, 0.0),
            )

    def test_preload_and_atomic_traversal_order(self):
        result = CommonBackplateExperiment(
            self.system,
            ArrayExperimentSettings(
                drag_length_m=0.1e-3,
                path_step_m=50e-6,
            ),
        ).run()
        self.assertTrue(result.summary.initial_preload_success)
        preload = result.points[1].response
        self.assertAlmostEqual(preload.total_normal_force_n, 1.0, delta=1e-4)
        self.assertAlmostEqual(preload.sharing.neff_normal, 4.0, places=12)
        old = preload.next_state
        forward = self.system.solve_pose(
            (25e-6, preload.common_uz_m),
            old,
            traversal_order=(0, 1, 2, 3),
        )
        reverse = self.system.solve_pose(
            (25e-6, preload.common_uz_m),
            old,
            traversal_order=(3, 2, 1, 0),
        )
        self.assertEqual(old, preload.next_state)
        self.assertEqual(forward.proposal_state, reverse.proposal_state)
        self.assertEqual(
            forward.wall_on_unit_wrench_about_origin,
            reverse.wall_on_unit_wrench_about_origin,
        )

    def test_same_state_arrays_include_m3_to_m4_contract(self):
        result = CommonBackplateExperiment(
            self.system,
            ArrayExperimentSettings(
                drag_length_m=0.1e-3,
                path_step_m=50e-6,
            ),
        ).run()
        arrays = _arrays(result)
        required = {
            "path_position_m",
            "common_ux_m",
            "common_uz_m",
            "pin_holder_xyz_m",
            "pin_wrench_about_unit",
            "wall_on_unit_wrench_about_origin",
            "unit_normal_force_n",
            "tangential_force_positive_n",
            "tangential_force_negative_n",
            "unit_moment_nm",
            "active_positive_normal",
            "contact_state",
            "event_label",
            "neff_normal",
            "force_aggregation_residual_n",
        }
        self.assertTrue(required.issubset(arrays))
        point_count = len(result.points)
        self.assertEqual(
            arrays["pin_wrench_about_unit"].shape,
            (point_count, 4, 6),
        )
        self.assertEqual(
            arrays["wall_on_unit_wrench_about_origin"].shape,
            (point_count, 6),
        )
        self.assertTrue(
            np.allclose(
                arrays["common_uz_m"][1:],
                result.fixed_common_uz_m,
            )
        )

    def test_m0_case_adapter_loads_all_global_y_tracks(self):
        with TemporaryDirectory() as directory:
            recipe = TerrainRecipe(seed=17, target_rms_height_m=0.0)
            region = RegionSpec(
                terrain_recipe_id=recipe.terrain_recipe_id,
                origin_x_m=-0.006,
                origin_y_m=-0.003,
                size_x_m=0.012,
                size_y_m=0.006,
                purpose="module",
            )
            library = TerrainLibrary(directory)
            library.generate_region(recipe, region)
            project_parameters = replace(
                self.parameters,
                rod_clearance_mode="unclosed",
            )
            configuration = ArrayConfiguration(
                2,
                2,
                4e-3,
                project_parameters,
            )
            tracks_by_y = {}
            for offset in configuration.holder_offsets_xyz_m:
                tracks_by_y.setdefault(
                    offset[1],
                    library.cache_track(
                        recipe,
                        region,
                        radius_m=project_parameters.tip_radius_m,
                        y_global_m=offset[1],
                    ),
                )
            tracks = [
                tracks_by_y[offset[1]]
                for offset in configuration.holder_offsets_xyz_m
            ]
            output = run_case(
                {
                    "terrain_library_root": str(Path(directory)),
                    "terrain_recipe_id": recipe.terrain_recipe_id,
                    "region_id": region.region_id,
                    "tracks": [
                        {
                            "track_id": track.track_id,
                            "radius_m": track.radius_m,
                        }
                        for track in tracks
                    ],
                    "configuration": configuration.as_dict(),
                    "unit_origin_xy_m": [0.0, 0.0],
                    "experiment": {
                        "drag_length_m": 0.1e-3,
                        "path_step_m": 50e-6,
                        "target_preload_n": 1.0,
                    },
                },
                RunContext(
                    case_id="case_m3_adapter",
                    backend={"selected": "cpu"},
                ),
            )
            self.assertTrue(output.summary["initial_preload_success"])
            self.assertEqual(output.summary["run_terminal_state"], "path_end")
            self.assertEqual(output.summary["seed"], 17)
            self.assertIn("configuration_id", output.arrays)
            self.assertIn("model_level", output.arrays)
            self.assertFalse(output.validation["formal_ranking_eligible"])


class M3ScreeningDesignTests(unittest.TestCase):
    def test_balanced_design_is_deterministic_and_distinguishes_orientation(self):
        base = _fixture_parameters().as_dict()
        alternate = dict(base)
        alternate["tip_radius_m"] = 100e-6
        packs = [
            {"parameter_pack_id": "pack_a", "spine": base},
            {"parameter_pack_id": "pack_b", "spine": alternate},
        ]
        pool = build_candidate_pool(packs)
        first = select_balanced_candidates(pool, 100)
        second = select_balanced_candidates(pool, 100)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        shapes = {(row["nx"], row["ny"]) for row in first}
        self.assertIn((2, 5), shapes)
        self.assertIn((5, 2), shapes)
        layouts = {row["angle_layout"] for row in first}
        self.assertIn("fixed", layouts)
        self.assertIn("gradient_80_to_60", layouts)
        self.assertIn("gradient_80_to_50", layouts)


class M3ValidationTests(unittest.TestCase):
    def test_all_analytic_gates_pass(self):
        report = run_analytic_validation()
        self.assertEqual(report["gate_count"], 14)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["formal_m3_round1_allowed"])


if __name__ == "__main__":
    unittest.main()
