import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spine_sim.contact import (
    ContactState,
    DynamicContactSettings,
    DynamicExperimentSettings,
    DynamicIntegratorSettings,
    DynamicSingleSpineExperiment,
    DynamicSingleSpineUnit,
    LegacyFixedZExperiment,
    LegacyPrescribedPoseConstitutiveCore,
    SingleSpineState,
    SpineParameters,
)
from spine_sim.contact.case import _arrays, run_case
from spine_sim.contact.validation import (
    _fixture_parameters,
    _fixture_track,
    run_analytic_validation,
)
from spine_sim.runtime.runner import RunContext
from spine_sim.terrain import RegionSpec, TerrainLibrary, TerrainRecipe


def _dynamic_settings(
    *,
    initial_x_m: float = -0.001,
    drag_length_m: float = 0.0002,
    preload_n: float = 0.5,
) -> DynamicExperimentSettings:
    return DynamicExperimentSettings(
        initial_center_x_m=initial_x_m,
        drag_length_m=drag_length_m,
        drag_speed_m_s=0.001,
        constant_preload_n=preload_n,
        holder_effective_mass_kg=0.05,
        holder_vertical_damping_n_s_m=1.0,
        maximum_preload_approach_m=0.008,
        output_spacing_m=10e-6,
        effective_normal_force_min_n=0.05,
    )


class M2ModelAndCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = _fixture_track("plane")

    def test_beam_compliance_and_modal_frequency_have_expected_diameter_trend(self):
        smaller = SpineParameters(diameter_m=0.6e-3)
        larger = SpineParameters(diameter_m=0.8e-3)
        ratio = (
            smaller.transverse_compliance_m_n
            / larger.transverse_compliance_m_n
        )
        self.assertGreater(ratio, 2.8)
        self.assertLess(ratio, 3.3)
        omega_smaller = np.sqrt(
            (1.0 / smaller.transverse_compliance_m_n)
            / smaller.transverse_modal_mass_kg
        )
        omega_larger = np.sqrt(
            (1.0 / larger.transverse_compliance_m_n)
            / larger.transverse_modal_mass_kg
        )
        self.assertLess(omega_smaller, omega_larger)

    def test_dynamic_axial_law_retains_spring_segments(self):
        unit = DynamicSingleSpineUnit(
            _fixture_parameters(),
            self.track,
            DynamicContactSettings(),
        )
        self.assertEqual(unit.axial_response(-1e-6)[3].value, "lower_stop")
        self.assertEqual(unit.axial_response(0.2e-3)[3].value, "interior")
        self.assertEqual(unit.axial_response(5e-3)[3].value, "hard_stop")

    def test_legacy_prescribed_pose_core_remains_atomic_for_m3_migration(self):
        core = LegacyPrescribedPoseConstitutiveCore(
            _fixture_parameters(),
            self.track,
        )
        old = SingleSpineState()
        proposal = core.solve_pose((0.0, 0.01), old, commit=False)
        self.assertIs(proposal.next_state, old)
        committed = core.solve_pose((0.0, 0.01), old, commit=True)
        self.assertEqual(committed.next_state, committed.proposal_state)
        self.assertEqual(committed.contact_state, ContactState.FREE)
        self.assertTrue(LegacyFixedZExperiment)


class M2DynamicExperimentTests(unittest.TestCase):
    def test_plane_drag_keeps_external_preload_and_allows_holder_z_motion(self):
        result = DynamicSingleSpineExperiment(
            _fixture_parameters(),
            _fixture_track("plane"),
            _dynamic_settings(drag_length_m=0.0005),
            DynamicContactSettings(),
            DynamicIntegratorSettings(time_step_s=1e-3),
        ).run()
        self.assertTrue(result.summary.initial_preload_success)
        self.assertEqual(result.summary.preload_mode, "continuous_external_force")
        self.assertEqual(result.summary.run_terminal_state.value, "path_end")
        self.assertTrue(
            all(
                point.external_preload_n == 0.5
                for point in result.points
            )
        )
        steady_normal = np.asarray(
            [point.normal_force_n for point in result.points[5:]]
        )
        self.assertAlmostEqual(float(np.mean(steady_normal)), 0.5, delta=0.02)
        self.assertAlmostEqual(
            result.summary.global_pull_force_median_n,
            0.1,
            delta=0.01,
        )
        z = np.asarray([point.holder_xz_m[1] for point in result.points])
        self.assertGreater(float(np.ptp(z)), 0.0)

    def test_bump_detaches_impacts_and_recontacts_without_model_unclosed(self):
        track = _fixture_track(
            "smooth_bump",
            {
                "amplitude_m": 400e-6,
                "center_x_m": -0.001,
                "sigma_x_m": 150e-6,
                "sigma_y_m": 150e-6,
            },
        )
        result = DynamicSingleSpineExperiment(
            _fixture_parameters(),
            track,
            _dynamic_settings(drag_length_m=0.001),
            DynamicContactSettings(),
            DynamicIntegratorSettings(time_step_s=1e-3),
        ).run()
        self.assertEqual(result.summary.run_terminal_state.value, "path_end")
        self.assertGreaterEqual(
            result.summary.event_counts["detach_to_free"], 1
        )
        self.assertGreaterEqual(result.summary.event_counts["recontact"], 1)
        self.assertGreaterEqual(result.summary.event_counts["impact"], 1)
        self.assertNotIn(
            "no_admissible_contact_equilibrium",
            result.summary.termination_reason,
        )

    def test_zero_preload_does_not_create_contact_force(self):
        result = DynamicSingleSpineExperiment(
            _fixture_parameters(),
            _fixture_track("plane"),
            _dynamic_settings(preload_n=0.0),
            DynamicContactSettings(),
            DynamicIntegratorSettings(time_step_s=1e-3),
        ).run()
        self.assertEqual(result.summary.global_pull_force_peak_n, 0.0)
        self.assertTrue(
            all(point.normal_force_n == 0.0 for point in result.points)
        )

    def test_result_arrays_are_dynamic_and_have_no_fixed_z_field(self):
        result = DynamicSingleSpineExperiment(
            _fixture_parameters(),
            _fixture_track("plane"),
            _dynamic_settings(),
            DynamicContactSettings(),
            DynamicIntegratorSettings(time_step_s=1e-3),
        ).run()
        arrays = _arrays(result)
        required = {
            "time_s",
            "path_position_m",
            "holder_xz_m",
            "holder_velocity_xz_m_s",
            "holder_acceleration_xz_m_s2",
            "center_xz_m",
            "center_velocity_xz_m_s",
            "external_preload_n",
            "normal_impulse_n_s",
            "spine_on_plate_wrench_about_holder",
            "dynamic_residual_n",
            "kinetic_energy_j",
            "structural_energy_j",
        }
        self.assertTrue(required.issubset(arrays))
        self.assertEqual(
            arrays["spine_on_plate_wrench_about_holder"].shape[1],
            6,
        )
        self.assertNotIn("fixed_holder_z_m", arrays)

    def test_m0_case_adapter_requires_explicit_dynamic_parameters(self):
        with TemporaryDirectory() as directory:
            recipe = TerrainRecipe(seed=7, target_rms_height_m=0.0)
            region = RegionSpec(
                terrain_recipe_id=recipe.terrain_recipe_id,
                origin_x_m=-0.005,
                origin_y_m=-0.0002,
                size_x_m=0.010,
                size_y_m=0.0004,
                purpose="module",
            )
            library = TerrainLibrary(directory)
            library.generate_region(recipe, region)
            track = library.cache_track(
                recipe,
                region,
                radius_m=50e-6,
                y_global_m=0.0,
            )
            parameters = _fixture_parameters(
                rod_clearance_mode="unclosed",
                yield_strength_pa=None,
            ).as_dict()
            output = run_case(
                {
                    "terrain_library_root": str(Path(directory)),
                    "terrain_recipe_id": recipe.terrain_recipe_id,
                    "region_id": region.region_id,
                    "track_id": track.track_id,
                    "radius_m": track.radius_m,
                    "spine": parameters,
                    "experiment": {
                        "initial_center_x_m": -0.001,
                        "drag_length_m": 0.0001,
                        "drag_speed_m_s": 0.001,
                        "constant_preload_n": 0.5,
                        "holder_effective_mass_kg": 0.05,
                        "holder_vertical_damping_n_s_m": 1.0,
                        "maximum_preload_approach_m": 0.008,
                        "output_spacing_m": 10e-6,
                    },
                    "dynamic_contact": {
                        "normal_model": "rigid_moreau",
                        "restitution_coefficient": 0.0,
                    },
                    "dynamic_integrator": {
                        "method": "moreau_implicit_euler",
                        "time_step_s": 0.001,
                    },
                },
                RunContext(
                    case_id="case_m2_dynamic_adapter",
                    backend={"selected": "cpu"},
                ),
            )
            self.assertEqual(
                output.summary["preload_mode"],
                "continuous_external_force",
            )
            self.assertEqual(output.summary["run_terminal_state"], "path_end")
            self.assertIn("time_s", output.arrays)
            self.assertFalse(output.validation["formal_ranking_eligible"])

    def test_all_dynamic_analytic_gates_pass(self):
        report = run_analytic_validation()
        self.assertEqual(report["gate_count"], 12)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["formal_random_screening_allowed"])


if __name__ == "__main__":
    unittest.main()
