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
    build_base_hardware,
    build_engineering_proxy_scenarios,
    build_convergence_sentinels,
    build_convergence_variants,
    build_full_array_design,
    compare_trend_summaries,
    estimate_backplate_dynamics,
    select_balanced_candidates,
    validate_paired_cases,
    validate_terrain_catalog,
)
from spine_sim.array.case import (
    _RodClearanceOutOfBoundsError,
    _arrays,
    _placement_search_offsets,
    _proxy_array_rod_clearance,
    run_case,
)
from spine_sim.array.design import _loading_protocol
from spine_sim.array.dynamic_validation import (
    _drop_profile,
    _existing_m1_smoke_contact_settings,
    _existing_m1_smoke_integrator,
    _integrator,
    _run,
    _select_existing_m1_condition,
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
    def test_axial_modal_damping_is_continuous_across_spring_stop(self):
        parameters = _fixture_parameters(
            axial_mode="spring",
            spring_stiffness_n_m=800.0,
        )
        configuration = ArrayConfiguration(
            2,
            2,
            4e-3,
            parameters,
        )
        system = DynamicCommonBackplateArray(
            configuration,
            _tracks(configuration),
            unit_origin_xy_m=(0.0, 0.0),
        )
        state = system.initial_state(
            _settings(drag_length_m=0.1e-3)
        )
        q, _velocity = system._pack_state(state)
        axial_dof = system._pin_dofs(0)[0]
        lower = q.copy()
        interior = q.copy()
        lower[axial_dof] = -1e-10
        interior[axial_dof] = 1e-10
        damping_lower = system._structure(
            lower,
            _settings(drag_length_m=0.1e-3),
        )[0][axial_dof]
        damping_interior = system._structure(
            interior,
            _settings(drag_length_m=0.1e-3),
        )[0][axial_dof]
        self.assertEqual(damping_lower, damping_interior)

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
            "implicit_euler_dissipation_increment_j",
            "normal_contact_work_increment_j",
            "tangential_contact_work_increment_j",
            "generalized_contact_work_increment_j",
            "contact_work_identity_residual_j",
            "contact_energy_injection_increment_j",
            "dynamic_residual_n",
            "energy_residual_j",
            "relative_energy_residual",
            "cumulative_energy_residual_j",
            "cumulative_energy_reference_j",
            "cumulative_relative_energy_error",
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
            self.plane.summary.maximum_abs_energy_residual_j, 1e-15
        )
        self.assertLess(
            self.plane.summary.maximum_abs_contact_work_identity_residual_j,
            1e-15,
        )
        self.assertLess(
            self.plane.summary.cumulative_relative_energy_error,
            1e-12,
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

    def test_preload_ramp_and_all_settling_gates_are_audited(self):
        trace = self.plane.settlement_trace
        applied = np.asarray(
            [point.applied_total_preload_n for point in trace]
        )
        self.assertGreater(len(trace), 20)
        self.assertTrue(np.all(np.diff(applied) >= -1e-15))
        self.assertLess(applied[0], 1e-3 * applied[-1])
        self.assertAlmostEqual(applied[-1], 0.5, places=12)
        self.assertLess(
            max(point.total_contact_reaction_z_n for point in trace),
            1.1 * applied[-1],
        )
        summary = self.plane.summary
        self.assertEqual(
            summary.settlement_ramp_profile,
            "minimum_jerk_quintic",
        )
        self.assertGreaterEqual(
            summary.settlement_stable_steps,
            summary.settlement_required_stable_steps,
        )
        self.assertLessEqual(
            summary.settlement_final_reaction_error_n,
            max(
                _settings(
                    drag_length_m=0.1e-3
                ).settling_reaction_force_tolerance_n,
                0.5
                * _settings(
                    drag_length_m=0.1e-3
                ).settling_reaction_force_relative_tolerance,
            ),
        )
        self.assertLessEqual(
            summary.settlement_final_maximum_mode_speed_m_s,
            _integrator(1e-3).settling_velocity_tolerance_m_s,
        )
        self.assertLessEqual(
            abs(summary.settlement_final_dynamic_residual_n),
            _settings(
                drag_length_m=0.1e-3
            ).settling_dynamic_residual_tolerance_n,
        )

    def test_failed_settlement_is_not_a_zero_load_ranking_row(self):
        impossible_force_gate = replace(
            _settings(drag_length_m=0.02e-3),
            settling_reaction_force_tolerance_n=0.0,
            settling_reaction_force_relative_tolerance=0.0,
        )
        short_integrator = replace(
            _integrator(1e-3),
            maximum_settling_time_s=0.10,
        )
        result = DynamicCommonBackplateExperiment(
            self.plane_system,
            impossible_force_gate,
            short_integrator,
        ).run()
        self.assertFalse(result.summary.initial_preload_success)
        self.assertFalse(result.summary.conditional_performance_available)
        self.assertEqual(
            result.summary.initialization_failure_category,
            "settlement_nonconvergence",
        )
        self.assertIsNone(result.summary.tangential_force_median_n)
        self.assertIsNone(result.summary.total_contact_reaction_time_mean_n)
        self.assertFalse(result.summary.formal_ranking_eligible)

    def test_output_levels_do_not_change_summary_values(self):
        before = asdict(self.plane.summary)
        summary_arrays = _arrays(self.plane, "summary")
        aggregate = _arrays(self.plane, "aggregate_trace")
        full = _arrays(self.plane, "full_pin_trace")
        self.assertEqual(summary_arrays, {})
        self.assertIn("settlement_ramp_fraction", aggregate)
        self.assertIn("wall_on_unit_wrench_about_origin", aggregate)
        self.assertNotIn("pin_normal_force_n", aggregate)
        self.assertIn("pin_normal_force_n", full)
        self.assertIn("pin_bending_stress_pa", full)
        self.assertEqual(before, asdict(self.plane.summary))

    def test_symmetric_2x2_plane_load_distribution(self):
        configuration = ArrayConfiguration(
            2, 2, 4e-3, self.parameters
        )
        _system, result = _run(
            configuration,
            _tracks(configuration),
            drag_length_m=0.02e-3,
        )
        initial_loads = np.asarray(
            [
                pin.normal_force_n
                for pin in result.points[0].pin_responses
            ]
        )
        self.assertLess(np.ptp(initial_loads), 1e-12)
        self.assertAlmostEqual(np.sum(initial_loads), 0.5, delta=1e-5)

    def test_rigid_and_three_required_spring_stiffnesses(self):
        axial_forces = []
        for stiffness in (300.0, 800.0, 2000.0):
            parameters = replace(
                self.parameters,
                axial_mode="spring",
                spring_stiffness_n_m=stiffness,
            )
            configuration = ArrayConfiguration(
                1, 2, 4e-3, parameters, fixture_only=True
            )
            system, result = _run(
                configuration,
                _tracks(configuration),
                drag_length_m=0.005e-3,
            )
            axial_forces.append(system.units[0].axial_response(10e-6)[0])
            self.assertEqual(
                result.summary.run_terminal_state.value,
                "path_end",
            )
        rigid_parameters = replace(
            self.parameters,
            axial_mode="rigid",
            spring_stiffness_n_m=None,
        )
        rigid_configuration = ArrayConfiguration(
            1, 2, 4e-3, rigid_parameters, fixture_only=True
        )
        rigid_system, rigid = _run(
            rigid_configuration,
            _tracks(rigid_configuration),
            drag_length_m=0.005e-3,
        )
        rigid_force = rigid_system.units[0].axial_response(10e-6)[0]
        self.assertTrue(np.all(np.diff(axial_forces) > 0.0))
        self.assertGreater(rigid_force, axial_forces[-1])
        self.assertEqual(rigid.summary.run_terminal_state.value, "path_end")

    def test_spring_lower_interior_hard_stop_and_slide_transitions(self):
        unit = self.plane_system.units[0]
        lower = unit.axial_response(-1e-6)
        interior = unit.axial_response(10e-6)
        hard = unit.axial_response(
            self.parameters.spring_travel_m
            + self.parameters.axial_compliance_m_n
            * float(self.parameters.spring_stiffness_n_m)
            * self.parameters.spring_travel_m
            + 1e-6
        )
        self.assertEqual(lower[3].value, "lower_stop")
        self.assertEqual(interior[3].value, "interior")
        self.assertEqual(hard[3].value, "hard_stop")
        self.assertGreaterEqual(hard[4], self.parameters.spring_travel_m)
        self.assertGreater(
            self.plane.summary.event_counts[EventLabel.SLIP_START.value],
            0,
        )
        self.assertTrue(
            any(
                pin.contact_state.value == "slide"
                for point in self.plane.points
                for pin in point.pin_responses
            )
        )

    def test_failure_categories_are_not_collapsed(self):
        geometry_result = DynamicCommonBackplateExperiment(
            self.plane_system,
            replace(
                _settings(drag_length_m=0.01e-3),
                initial_common_ux_m=0.1,
            ),
            _integrator(1e-3),
        ).run()
        self.assertEqual(
            geometry_result.summary.failure_category,
            "geometry_out_of_bounds",
        )
        physical_result = DynamicCommonBackplateExperiment(
            self.plane_system,
            replace(
                _settings(drag_length_m=0.01e-3),
                maximum_preload_approach_m=1e-9,
            ),
            _integrator(1e-3),
        ).run()
        self.assertEqual(
            physical_result.summary.failure_category,
            "physical_boundary",
        )
        self.assertTrue(physical_result.settlement_trace)
        self.assertGreater(
            physical_result.summary.settlement_actual_approach_m,
            physical_result.summary.settlement_maximum_approach_m,
        )
        numerical_result = DynamicCommonBackplateExperiment(
            self.plane_system,
            _settings(drag_length_m=0.01e-3),
            replace(_integrator(1e-3), maximum_steps=1),
        ).run()
        self.assertTrue(numerical_result.summary.initial_preload_success)
        self.assertEqual(
            numerical_result.summary.failure_category,
            "numerical_failure",
        )
        self.assertEqual(
            self.plane.summary.model_state.value,
            "parameter_unclosed",
        )

    def test_fixed_angles_and_default_gradient_transform(self):
        for angle in (60.0, 70.0, 80.0):
            configuration = ArrayConfiguration(
                2,
                2,
                4e-3,
                replace(self.parameters, installation_angle_deg=angle),
            )
            self.assertEqual(configuration.column_angles_deg, (angle, angle))
            for pin in configuration.pin_parameters:
                self.assertAlmostEqual(np.linalg.norm(pin.axis_xz), 1.0)
                self.assertGreater(pin.axis_xz[0], 0.0)
                self.assertLess(pin.axis_xz[1], 0.0)
        gradient = ArrayConfiguration(
            5,
            2,
            4e-3,
            replace(self.parameters, installation_angle_deg=80.0),
            angle_layout=AngleLayout.GRADIENT_80_TO_60,
        )
        self.assertTrue(
            np.allclose(
                gradient.column_angles_deg,
                np.linspace(80.0, 60.0, 5),
            )
        )

    def test_repeated_configuration_run_is_deterministic(self):
        _system, repeated = _run(
            self.configuration,
            self.plane_tracks,
            drag_length_m=0.30e-3,
        )
        self.assertEqual(self.plane.points, repeated.points)
        self.assertEqual(self.plane.summary, repeated.summary)

    def test_representative_2x2_4x4_6x6_smoke(self):
        for size in (2, 4, 6):
            configuration = ArrayConfiguration(
                size, size, 4e-3, self.parameters
            )
            _system, result = _run(
                configuration,
                _tracks(configuration),
                drag_length_m=0.005e-3,
            )
            self.assertEqual(
                result.summary.run_terminal_state.value,
                "path_end",
            )
            self.assertLess(
                result.summary.maximum_abs_dynamic_residual_n,
                1e-8,
            )


class M3CaseAdapterTests(unittest.TestCase):
    def test_two_dimensional_rod_clearance_on_flat_region(self):
        parameters = replace(
            _fixture_parameters(),
            rod_clearance_mode="proxy_cylindrical_shank_postcheck",
            yield_strength_pa=800e6,
        )
        with TemporaryDirectory() as directory:
            recipe = TerrainRecipe(seed=19, target_rms_height_m=0.0)
            region = RegionSpec(
                terrain_recipe_id=recipe.terrain_recipe_id,
                origin_x_m=-0.008,
                origin_y_m=-0.005,
                size_x_m=0.016,
                size_y_m=0.010,
                purpose="module",
            )
            library = TerrainLibrary(directory)
            library.generate_region(recipe, region)
            configuration = ArrayConfiguration(
                2,
                2,
                4e-3,
                parameters,
            )
            tracks = tuple(
                library.cache_track(
                    recipe,
                    region,
                    radius_m=parameters.tip_radius_m,
                    y_global_m=offset[1],
                )
                for offset in configuration.holder_offsets_xyz_m
            )
            system = DynamicCommonBackplateArray(
                configuration,
                tracks,
                unit_origin_xy_m=(0.0, 0.0),
                contact=DynamicContactSettings(
                    projection_iterations=20,
                ),
            )
            result = DynamicCommonBackplateExperiment(
                system,
                _settings(drag_length_m=0.02e-3),
                _integrator(1e-3),
            ).run()
            clearance = _proxy_array_rod_clearance(
                library=library,
                result=result,
                axial_sample_count=12,
                lateral_sample_count=7,
            )
            self.assertEqual(
                clearance.shape,
                (len(result.points), configuration.pin_count),
            )
            self.assertTrue(np.all(np.isfinite(clearance)))
            self.assertGreater(float(np.min(clearance)), 0.0)
            outside_result = replace(
                result,
                points=tuple(
                    replace(
                        point,
                        pin_responses=tuple(
                            replace(
                                response,
                                center_xyz_m=(
                                    response.center_xyz_m[0] + 0.1,
                                    response.center_xyz_m[1],
                                    response.center_xyz_m[2],
                                ),
                            )
                            for response in point.pin_responses
                        ),
                    )
                    for point in result.points
                ),
            )
            with self.assertRaises(_RodClearanceOutOfBoundsError):
                _proxy_array_rod_clearance(
                    library=library,
                    result=outside_result,
                    axial_sample_count=12,
                    lateral_sample_count=7,
                )

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
            region_metadata = library.generate_region(recipe, region)
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
                    "terrain_data_sha256": region_metadata["data_sha256"],
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
            self.assertEqual(
                output.summary["terrain_data_sha256"],
                region_metadata["data_sha256"],
            )
            self.assertIn("external_total_preload_n", output.arrays)
            self.assertTrue(
                np.all(
                    output.arrays["terrain_data_sha256"]
                    == region_metadata["data_sha256"]
                )
            )
            self.assertTrue(
                np.allclose(
                    output.arrays["selected_unit_origin_xy_m"],
                    np.zeros((len(output.arrays["time_s"]), 2)),
                )
            )
            self.assertFalse(output.validation["formal_ranking_eligible"])
            with self.assertRaisesRegex(ValueError, "terrain_data_sha256"):
                run_case(
                    {
                        "terrain_library_root": str(Path(directory)),
                        "terrain_recipe_id": recipe.terrain_recipe_id,
                        "region_id": region.region_id,
                        "terrain_data_sha256": "0" * 64,
                        "configuration": {},
                        "unit_origin_xy_m": [0.0, 0.0],
                        "experiment": {},
                    },
                    RunContext(
                        case_id="case_m3_bad_terrain_hash",
                        backend={"selected": "cpu"},
                    ),
                )


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

    def test_complete_cartesian_design_counts_and_levels(self):
        hardware = build_base_hardware()
        design = build_full_array_design()
        self.assertEqual(len(hardware), 48)
        self.assertEqual(
            sum(row["angle_layout"] == "fixed" for row in design),
            1008,
        )
        self.assertEqual(
            sum(
                row["angle_layout"] == "gradient_80_to_60"
                for row in design
            ),
            336,
        )
        self.assertFalse(
            any(row["fixed_angle_deg"] == 50.0 for row in design)
        )
        self.assertFalse(
            any(
                row["angle_layout"] == "gradient_80_to_50"
                for row in design
            )
        )

    def test_output_level_is_not_part_of_loading_protocol_identity(self):
        first = _loading_protocol(
            1.0,
            output_spacing_m=50e-6,
            time_step_s=1e-3,
        )
        second = _loading_protocol(
            1.0,
            output_spacing_m=50e-6,
            time_step_s=1e-3,
        )
        self.assertEqual(
            first["loading_protocol_id"],
            second["loading_protocol_id"],
        )
        self.assertNotIn("output_level", first)
        self.assertEqual(
            first["placement_search"]["selection_rule"],
            "first_collision_free",
        )

    def test_placement_search_policy_is_deterministic_and_nominal_first(self):
        offsets = _placement_search_offsets({"enabled": True})
        self.assertEqual(offsets[0], (0.0, 0.0))
        self.assertGreater(len(offsets), 1)
        self.assertEqual(offsets, _placement_search_offsets({"enabled": True}))
        with self.assertRaisesRegex(ValueError, "nominal"):
            _placement_search_offsets(
                {
                    "enabled": True,
                    "offsets_xy_m": [[0.0, 1e-3], [0.0, 0.0]],
                }
            )

    def test_engineering_proxy_is_explicit_and_geometry_scaled(self):
        scenarios = build_engineering_proxy_scenarios()
        self.assertEqual(len(scenarios), 17)
        self.assertEqual(
            len({scenario.scenario_id for scenario in scenarios}),
            len(scenarios),
        )
        parameters = _fixture_parameters()
        small = ArrayConfiguration(2, 2, 4e-3, parameters)
        large = ArrayConfiguration(6, 6, 6e-3, parameters)
        small_dynamics = estimate_backplate_dynamics(small)
        large_dynamics = estimate_backplate_dynamics(large)
        self.assertGreater(
            large_dynamics["backplate_mass_kg"],
            small_dynamics["backplate_mass_kg"],
        )
        self.assertGreater(
            large_dynamics["nominal_vertical_stiffness_n_m"],
            small_dynamics["nominal_vertical_stiffness_n_m"],
        )
        self.assertGreater(
            large_dynamics["backplate_vertical_damping_n_s_m"],
            0.0,
        )

    def test_bounded_convergence_plan_and_trend_gate(self):
        self.assertEqual(len(build_convergence_sentinels()), 8)
        self.assertEqual(len(build_convergence_variants()), 11)
        reference = {
            "run_terminal_state": "path_end",
            "initial_preload_success": True,
            "tangential_force_median_n": 1.0,
            "tangential_force_p10_n": 0.8,
            "neff_resultant_median": 4.0,
            "maximum_pin_normal_force_n": 0.3,
            "contact_fraction": 0.9,
            "cumulative_relative_energy_error": 1e-8,
            "maximum_abs_contact_work_identity_residual_j": 1e-16,
        }
        close = dict(reference)
        close["tangential_force_median_n"] = 1.02
        close["neff_resultant_median"] = 4.1
        self.assertTrue(compare_trend_summaries(reference, close)["passed"])
        failed = dict(reference)
        failed["tangential_force_median_n"] = 1.2
        self.assertFalse(
            compare_trend_summaries(reference, failed)["passed"]
        )

    def test_formal_catalog_requires_three_families_by_100(self):
        conditions = []
        for family_index, family in enumerate(
            ("sandpaper", "red_brick", "concrete")
        ):
            for index in range(100):
                seed = 41_001 + index
                conditions.append(
                    {
                        "terrain_family": family,
                        "seed": seed,
                        "realization_id": (
                            f"realization_{family}_{seed}"
                        ),
                        "terrain_recipe_id": f"recipe_{seed}",
                        "region_id": f"region_{family}_{seed}",
                        "data_sha256": (
                            f"{family_index + 1:02x}{seed:062x}"
                        ),
                        "full_sha256_verified": True,
                    }
                )
        catalog = {
            "schema_version": "m1-material-terrain-catalog-v1",
            "status": "complete",
            "all_full_hashes_verified": True,
            "formal_300_complete": True,
            "m1_module_version": "m1-test",
            "conditions": conditions,
        }
        self.assertEqual(len(validate_terrain_catalog(catalog)), 300)
        incomplete = dict(catalog)
        incomplete["conditions"] = conditions[:-1]
        with self.assertRaises(ValueError):
            validate_terrain_catalog(incomplete)
        missing_schema = dict(catalog)
        missing_schema.pop("schema_version")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_terrain_catalog(missing_schema)
        unknown_schema = dict(catalog)
        unknown_schema["schema_version"] = "future-unknown-v9"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_terrain_catalog(unknown_schema)
        malformed_hash = dict(catalog)
        malformed_conditions = [dict(item) for item in conditions]
        malformed_conditions[0]["data_sha256"] = "not-a-sha256"
        malformed_hash["conditions"] = malformed_conditions
        with self.assertRaisesRegex(ValueError, "data_sha256"):
            validate_terrain_catalog(malformed_hash)
        unpaired = dict(catalog)
        unpaired_conditions = [dict(item) for item in conditions]
        unpaired_conditions[-1]["seed"] = 999_999
        unpaired["conditions"] = unpaired_conditions
        with self.assertRaisesRegex(ValueError, "paired seeds"):
            validate_terrain_catalog(unpaired)

    def test_existing_m1_smoke_selection_never_guesses_across_families(self):
        conditions = [
            {
                "name": f"{family}_condition",
                "terrain_family": family,
                "seed": 41001,
                "terrain_recipe_id": f"recipe_{family}",
                "region_id": f"region_{family}",
                "data_sha256": f"{index + 1:064x}",
                "full_sha256_verified": True,
            }
            for index, family in enumerate(
                ("sandpaper", "red_brick", "concrete")
            )
        ]
        catalog = {
            "schema_version": "m1-material-terrain-catalog-v1",
            "status": "complete",
            "all_full_hashes_verified": True,
            "conditions": conditions,
        }
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _select_existing_m1_condition(catalog, seed=41001)
        selected = _select_existing_m1_condition(
            catalog,
            seed=41001,
            terrain_family="red_brick",
        )
        self.assertEqual(selected["name"], "red_brick_condition")

    def test_existing_m1_smoke_uses_production_proxy_contact_baseline(self):
        contact = _existing_m1_smoke_contact_settings()
        integrator = _existing_m1_smoke_integrator()
        self.assertEqual(contact.position_correction, 0.20)
        self.assertEqual(contact.projection_iterations, 20)
        self.assertEqual(integrator.time_step_s, 1e-3)
        self.assertEqual(integrator.maximum_settling_time_s, 2.0)

    def test_missing_paired_condition_is_rejected(self):
        designs = build_full_array_design()[:2]
        terrain_ids = ("terrain_a", "terrain_b")
        protocol_ids = ("preload_1",)
        cases = [
            {
                "parameters": {
                    "configuration": design["configuration"],
                    "terrain_condition_id": terrain_id,
                    "loading_protocol_id": "preload_1",
                }
            }
            for design in designs
            for terrain_id in terrain_ids
        ]
        validate_paired_cases(
            cases,
            configuration_ids=[
                row["array_configuration_id"] for row in designs
            ],
            terrain_condition_ids=terrain_ids,
            loading_protocol_ids=protocol_ids,
        )
        with self.assertRaises(ValueError):
            validate_paired_cases(
                cases[:-1],
                configuration_ids=[
                    row["array_configuration_id"] for row in designs
                ],
                terrain_condition_ids=terrain_ids,
                loading_protocol_ids=protocol_ids,
            )


class M3ValidationTests(unittest.TestCase):
    def test_all_dynamic_analytic_gates_pass_without_formal_campaign(self):
        report = run_dynamic_analytic_validation()
        self.assertEqual(report["gate_count"], 16)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["formal_m3_round1_allowed"])
        self.assertIn("legacy fixed-Z validation not executed", report["validation_scope"])


if __name__ == "__main__":
    unittest.main()
