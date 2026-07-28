import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import spine_sim.array as array_api
from spine_sim.array import (
    AngleLayout,
    ArrayConfiguration,
    DynamicCommonBackplateArray,
    DynamicCommonBackplateExperiment,
    LegacyArrayExperimentSettings,
    LegacyCommonBackplateArray,
    LegacyFixedZCommonBackplateExperiment,
    build_candidate_pool,
    select_balanced_candidates,
)
from spine_sim.array.case import _arrays, run_case
from spine_sim.array.dynamic_validation import (
    _drop_profile,
    _integrator,
    _run,
    _settings,
    _tracks,
    run_dynamic_analytic_validation,
)
from spine_sim.contact import DynamicContactSettings, EventLabel
from spine_sim.contact.validation import _fixture_parameters
from spine_sim.runtime.runner import RunContext
from spine_sim.terrain import RegionSpec, TerrainLibrary, TerrainRecipe


class M3PublicBoundaryAndGeometryTests(unittest.TestCase):
    def setUp(self):
        self.parameters = _fixture_parameters()

    def test_production_entry_is_dynamic_and_legacy_is_explicit(self):
        self.assertIs(
            array_api.DynamicCommonBackplateArray,
            DynamicCommonBackplateArray,
        )
        self.assertIs(
            array_api.DynamicCommonBackplateExperiment,
            DynamicCommonBackplateExperiment,
        )
        self.assertFalse(hasattr(array_api, "CommonBackplateArray"))
        self.assertFalse(hasattr(array_api, "CommonBackplateExperiment"))
        self.assertIsNotNone(LegacyCommonBackplateArray)
        self.assertIsNotNone(LegacyFixedZCommonBackplateExperiment)
        self.assertIsNotNone(LegacyArrayExperimentSettings)

    def test_production_geometry_rejects_singleton_dimension(self):
        with self.assertRaises(ValueError):
            ArrayConfiguration(2, 1, 4e-3, self.parameters)
        fixture = ArrayConfiguration(
            2, 1, 4e-3, self.parameters, fixture_only=True
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

    def test_tracks_share_recipe_region_and_match_global_y(self):
        configuration = ArrayConfiguration(
            1, 2, 4e-3, self.parameters, fixture_only=True
        )
        tracks = list(_tracks(configuration))
        tracks[0] = replace(tracks[0], terrain_recipe_id="other")
        with self.assertRaises(Exception):
            DynamicCommonBackplateArray(
                configuration,
                tracks,
                unit_origin_xy_m=(0.0, 0.0),
            )
        tracks = list(_tracks(configuration))
        tracks[0] = replace(tracks[0], y_global_m=0.0)
        with self.assertRaises(Exception):
            DynamicCommonBackplateArray(
                configuration,
                tracks,
                unit_origin_xy_m=(0.0, 0.0),
            )

    def test_2x5_and_5x2_are_distinct(self):
        two_by_five = ArrayConfiguration(2, 5, 5e-3, self.parameters)
        five_by_two = ArrayConfiguration(5, 2, 5e-3, self.parameters)
        self.assertNotEqual(
            two_by_five.configuration_id,
            five_by_two.configuration_id,
        )
        self.assertNotEqual(
            two_by_five.holder_offsets_xyz_m,
            five_by_two.holder_offsets_xyz_m,
        )


class M3JointDynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = _fixture_parameters(
            axial_damping_ratio=0.20,
            transverse_damping_ratio=0.20,
        )
        cls.configuration = ArrayConfiguration(
            1, 2, 4e-3, cls.parameters, fixture_only=True
        )
        cls.plane_tracks = _tracks(cls.configuration)
        cls.plane_system, cls.plane = _run(
            cls.configuration,
            cls.plane_tracks,
            drag_length_m=0.30e-3,
        )

    def test_plane_mean_total_reaction_balances_one_external_preload(self):
        values = [
            point.total_contact_reaction_z_n
            for point in self.plane.points[len(self.plane.points) // 2 :]
        ]
        self.assertAlmostEqual(np.mean(values), 0.5, delta=0.025)
        self.assertTrue(
            any(
                abs(point.total_contact_reaction_z_n - 0.5) > 1e-3
                for point in self.plane.points
            ),
            "instantaneous reaction must not be constrained to preload",
        )
        self.assertEqual(
            self.plane.summary.preload_mode,
            "continuous_total_external_force",
        )

    def test_different_heights_transfer_load_without_equal_preassignment(self):
        tracks = _tracks(
            self.configuration,
            vertical_offsets_m=(80e-6, 0.0),
            suffix="test_height_transfer",
        )
        _system, result = _run(
            self.configuration, tracks, drag_length_m=0.08e-3
        )
        forces = [
            pin.normal_force_n for pin in result.points[0].pin_responses
        ]
        self.assertAlmostEqual(sum(forces), 0.5, delta=0.01)
        self.assertGreater(abs(forces[0] - forces[1]), 0.02)

    def test_single_pin_detaches_recontacts_and_array_reaches_path_end(self):
        tracks = _tracks(
            self.configuration,
            profiles=(_drop_profile, None),
            suffix="test_single_detach",
        )
        _system, result = _run(
            self.configuration, tracks, drag_length_m=0.55e-3
        )
        labels = [
            label
            for point in result.points
            for index, label in point.event_labels
            if index == 0
        ]
        self.assertIn(EventLabel.DETACH_TO_FREE.value, labels)
        self.assertIn(EventLabel.RECONTACT.value, labels)
        self.assertEqual(result.summary.run_terminal_state.value, "path_end")
        self.assertNotIn(
            "no_admissible_contact_equilibrium",
            result.summary.termination_reason,
        )
        for point in result.points:
            for pin in point.pin_responses:
                if pin.contact_state.value in {"free", "detach_event"}:
                    self.assertEqual(pin.normal_force_n, 0.0)
                    self.assertEqual(pin.tangential_force_n, 0.0)

    def test_coupled_solver_handles_simultaneous_multi_pin_impact(self):
        tracks = _tracks(
            self.configuration,
            profiles=(_drop_profile, _drop_profile),
            suffix="test_simultaneous",
        )
        _system, result = _run(
            self.configuration, tracks, drag_length_m=0.55e-3
        )
        simultaneous = [
            point
            for point in result.points
            if sum(
                label == EventLabel.IMPACT.value
                for _index, label in point.event_labels
            )
            >= 2
        ]
        self.assertTrue(simultaneous)
        self.assertGreater(
            result.summary.total_normal_force_range_n[1],
            result.summary.external_total_preload_n,
        )

    def test_traversal_order_and_rejected_commit_are_atomic(self):
        experiment = DynamicCommonBackplateExperiment(
            self.plane_system,
            _settings(drag_length_m=0.1e-3),
            _integrator(1e-3),
        )
        old, _point = experiment._settle()
        settings = _settings(drag_length_m=0.1e-3)
        forward = self.plane_system.propose_step(
            old,
            settings,
            common_ux_m=1e-6,
            drag_speed_m_s=1e-3,
            dt=1e-3,
            traversal_order=(0, 1),
        )
        reverse = self.plane_system.propose_step(
            old,
            settings,
            common_ux_m=1e-6,
            drag_speed_m_s=1e-3,
            dt=1e-3,
            traversal_order=(1, 0),
        )
        self.assertEqual(forward.proposal_state, reverse.proposal_state)
        self.assertEqual(forward.point, reverse.point)
        self.assertEqual(
            self.plane_system.commit_step(old, forward, accept=False),
            old,
        )
        self.assertEqual(
            self.plane_system.commit_step(old, forward, accept=True),
            forward.proposal_state,
        )
        invalid = self.plane_system.propose_step(
            old,
            settings,
            common_ux_m=0.1,
            drag_speed_m_s=1e-3,
            dt=1e-3,
        )
        self.assertFalse(invalid.proposal_valid)
        self.assertEqual(invalid.proposal_state, old)
        self.assertEqual(
            self.plane_system.commit_step(old, invalid, accept=False),
            old,
        )

    def test_dynamic_arrays_wrench_identity_energy_and_parameter_gate(self):
        arrays = _arrays(self.plane)
        required = {
            "time_s",
            "path_position_m",
            "backplate_position_xyz_m",
            "backplate_velocity_xyz_m_s",
            "backplate_acceleration_xyz_m_s2",
            "external_total_preload_n",
            "pin_center_xyz_m",
            "pin_center_velocity_xyz_m_s",
            "pin_normal_force_n",
            "pin_normal_impulse_n_s",
            "pin_wrench_about_holder",
            "pin_wrench_about_unit",
            "wall_on_unit_wrench_about_origin",
            "total_contact_reaction_z_n",
            "backplate_inertia_force_z_n",
            "backplate_damping_force_z_n",
            "active_pin_count",
            "effective_load_pin_count",
            "neff_normal",
            "gini_normal",
            "kinetic_energy_j",
            "structural_energy_j",
            "preload_work_increment_j",
            "drive_work_increment_j",
            "cumulative_preload_work_j",
            "cumulative_drive_work_j",
            "friction_dissipation_increment_j",
            "cumulative_friction_dissipation_j",
            "dynamic_residual_n",
            "energy_residual_j",
            "actual_time_step_s",
        }
        self.assertTrue(required.issubset(arrays))
        self.assertNotIn("fixed_common_uz_m", arrays)
        self.assertNotIn("target_preload_n", arrays)
        self.assertTrue(
            np.allclose(
                arrays["pin_wrench_about_unit"].sum(axis=1),
                arrays["wall_on_unit_wrench_about_origin"],
                atol=1e-15,
                rtol=0.0,
            )
        )
        backplate_balance = (
            arrays["backplate_inertia_force_z_n"][1:]
            + arrays["backplate_damping_force_z_n"][1:]
            + arrays["external_total_preload_n"][1:]
            - arrays["total_contact_reaction_z_n"][1:]
        )
        self.assertLess(np.max(np.abs(backplate_balance)), 1e-8)
        self.assertLess(
            self.plane.summary.maximum_abs_dynamic_residual_n, 1e-8
        )
        self.assertLess(
            self.plane.summary.maximum_abs_energy_residual_j, 1e-5
        )
        self.assertEqual(
            self.plane.summary.model_state.value, "parameter_unclosed"
        )
        self.assertFalse(self.plane.summary.formal_ranking_eligible)

    def test_internal_time_step_halving_preserves_steady_response(self):
        _system, half = _run(
            self.configuration,
            self.plane_tracks,
            drag_length_m=0.30e-3,
            time_step_s=0.5e-3,
        )
        self.assertAlmostEqual(
            self.plane.summary.total_contact_reaction_time_mean_n,
            half.summary.total_contact_reaction_time_mean_n,
            delta=0.02,
        )
        self.assertAlmostEqual(
            self.plane.summary.tangential_force_median_n,
            half.summary.tangential_force_median_n,
            delta=0.02,
        )


class M3CaseAdapterTests(unittest.TestCase):
    def test_case_adapter_saves_explicit_dynamic_parameters_and_closes_ranking(self):
        parameters = _fixture_parameters()
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
                parameters,
                rod_clearance_mode="unclosed",
                yield_strength_pa=None,
            )
            configuration = ArrayConfiguration(
                2, 2, 4e-3, project_parameters
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
            experiment = _settings(drag_length_m=0.08e-3)
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
                    "experiment": asdict(experiment),
                    "contact": asdict(DynamicContactSettings()),
                    "integrator": asdict(_integrator(1e-3)),
                },
                RunContext(
                    case_id="case_m3_dynamic_adapter",
                    backend={"selected": "cpu"},
                ),
            )
            self.assertTrue(output.summary["initial_preload_success"])
            self.assertEqual(output.summary["run_terminal_state"], "path_end")
            self.assertIn("backplate_mass_kg", output.summary["experiment"])
            self.assertIn("time_step_s", output.summary["integrator"])
            self.assertIn("normal_model", output.summary["contact"])
            self.assertIn("external_total_preload_n", output.arrays)
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


class M3ValidationTests(unittest.TestCase):
    def test_all_dynamic_analytic_gates_pass_without_formal_campaign(self):
        report = run_dynamic_analytic_validation()
        self.assertEqual(report["gate_count"], 11)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["formal_m3_round1_allowed"])
        self.assertIn("legacy fixed-Z validation not executed", report["validation_scope"])


if __name__ == "__main__":
    unittest.main()
