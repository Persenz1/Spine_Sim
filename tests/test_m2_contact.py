import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spine_sim.contact import (
    AxialMode,
    ContactState,
    ExperimentSettings,
    PrescribedPoseConstitutiveCore,
    SingleSpineExperiment,
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


class M2ModelAndCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = _fixture_track("plane")

    def test_beam_compliance_has_expected_diameter_trend(self):
        smaller = SpineParameters(diameter_m=0.6e-3)
        larger = SpineParameters(diameter_m=0.8e-3)
        ratio = (
            smaller.transverse_compliance_m_n
            / larger.transverse_compliance_m_n
        )
        self.assertGreater(ratio, 2.8)
        self.assertLess(ratio, 3.3)

    def test_rigid_mode_is_not_a_large_spring(self):
        rigid = SpineParameters(
            axial_mode=AxialMode.RIGID,
            spring_stiffness_n_m=None,
        )
        self.assertEqual(rigid.axial_mode, AxialMode.RIGID)
        self.assertIsNone(rigid.spring_stiffness_n_m)

    def test_free_proposal_and_commit_are_atomic(self):
        core = PrescribedPoseConstitutiveCore(
            _fixture_parameters(),
            self.track,
        )
        old = SingleSpineState()
        proposal = core.solve_pose((0.0, 0.01), old, commit=False)
        self.assertIs(proposal.next_state, old)
        self.assertEqual(old, SingleSpineState())
        committed = core.solve_pose((0.0, 0.01), old, commit=True)
        self.assertEqual(committed.next_state, committed.proposal_state)
        self.assertEqual(committed.contact_state, ContactState.FREE)

    def test_first_contact_uses_forward_cap_and_zero_force(self):
        parameters = _fixture_parameters()
        core = PrescribedPoseConstitutiveCore(parameters, self.track)
        center_x = -0.001
        holder_x = center_x - parameters.exposed_length_m * core._a[0]
        holder_z = (
            self.track.radius_m
            - parameters.exposed_length_m * core._a[1]
        )
        response = core.solve_pose((holder_x, holder_z), commit=True)
        self.assertEqual(
            response.contact_state,
            ContactState.FIRST_CONTACT_EVENT,
        )
        self.assertTrue(response.cap_gate_passed)
        self.assertAlmostEqual(response.normal_force_n, 0.0, places=12)
        self.assertIsNotNone(response.proposal_state.anchor_center_xz_m)


class M2ExperimentAndValidationTests(unittest.TestCase):
    def test_fixed_z_plane_drag_is_not_constant_force_control(self):
        track = _fixture_track("plane")
        core = PrescribedPoseConstitutiveCore(_fixture_parameters(), track)
        result = SingleSpineExperiment(
            core,
            ExperimentSettings(
                initial_center_x_m=-0.001,
                drag_length_m=0.001,
                path_step_m=50e-6,
            ),
        ).run()
        self.assertTrue(result.summary.initial_preload_success)
        self.assertAlmostEqual(
            result.points[1].response.normal_force_n,
            0.5,
            delta=1e-4,
        )
        self.assertAlmostEqual(
            result.fixed_holder_z_m,
            result.points[-1].response.holder_xz_m[1],
            places=15,
        )
        minimum, maximum = result.summary.normal_force_range_n
        self.assertGreater(maximum - minimum, 1e-3)
        post_drag_slide = [
            point.response
            for point in result.points
            if point.path_position_m > 0
            and point.response.contact_state is ContactState.SLIDE
        ]
        self.assertTrue(post_drag_slide)
        self.assertTrue(
            all(response.tangential_force_n <= 0 for response in post_drag_slide)
        )

    def test_result_arrays_include_m2_to_m3_contract(self):
        track = _fixture_track("plane")
        result = SingleSpineExperiment(
            PrescribedPoseConstitutiveCore(_fixture_parameters(), track),
            ExperimentSettings(
                initial_center_x_m=-0.001,
                drag_length_m=0.0001,
                path_step_m=50e-6,
            ),
        ).run()
        arrays = _arrays(result)
        required = {
            "holder_xz_m",
            "center_xz_m",
            "support_xyz_m",
            "gap_m",
            "contact_state",
            "spring_state",
            "spine_on_plate_wrench_about_holder",
            "axial_force_n",
            "transverse_force_n",
            "spring_compression_m",
            "beam_displacement_xz_m",
            "geometry_residual_m",
            "energy_residual_j",
        }
        self.assertTrue(required.issubset(arrays))
        self.assertEqual(
            arrays["spine_on_plate_wrench_about_holder"].shape[1],
            6,
        )
        self.assertEqual(arrays["holder_xz_m"].dtype, np.float64)

    def test_m0_case_adapter_reads_saved_m1_track(self):
        with TemporaryDirectory() as directory:
            recipe = TerrainRecipe(
                seed=7,
                target_rms_height_m=0.0,
            )
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
                        "path_step_m": 50e-6,
                    },
                },
                RunContext(
                    case_id="case_m2_adapter",
                    backend={"selected": "cpu"},
                ),
            )
            self.assertTrue(output.summary["initial_preload_success"])
            self.assertEqual(output.summary["run_terminal_state"], "path_end")
            self.assertIn("spine_on_plate_wrench_about_holder", output.arrays)
            self.assertFalse(output.validation["formal_ranking_eligible"])

    def test_all_analytic_gates_pass(self):
        report = run_analytic_validation()
        self.assertEqual(report["gate_count"], 14)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["formal_random_screening_allowed"])


if __name__ == "__main__":
    unittest.main()
