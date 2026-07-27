"""Reproducible M2 analytic gates and M1 ten-terrain smoke validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.io.results import atomic_write_json, utc_now
from spine_sim.terrain import (
    RegionSpec,
    TerrainLibrary,
    compute_track_geometry,
    evaluate_analytic,
)

from .experiment import ExperimentSettings, SingleSpineExperiment
from .models import (
    AxialMode,
    ContactState,
    EventLabel,
    M2_MODULE_VERSION,
    SingleSpineState,
    SpineParameters,
    SpringState,
)
from .solver import PrescribedPoseConstitutiveCore


def _fixture_track(
    kind: str,
    parameters: Mapping[str, float] | None = None,
    *,
    radius_m: float = 50e-6,
    dx_m: float = 10e-6,
):
    region = RegionSpec(
        terrain_recipe_id=f"terrain_recipe_m2_{kind}",
        origin_x_m=-0.003,
        origin_y_m=-0.0004,
        size_x_m=0.008,
        size_y_m=0.0008,
        resolution_x_m=dx_m,
        resolution_y_m=dx_m,
        purpose="module",
    )
    x = region.origin_x_m + np.arange(region.shape[1]) * dx_m
    y = region.origin_y_m + np.arange(region.shape[0]) * dx_m
    height = evaluate_analytic(kind, x, y, parameters)
    return compute_track_geometry(
        height,
        region,
        radius_m=radius_m,
        y_global_m=0.0,
    )


def _fixture_parameters(**overrides: Any) -> SpineParameters:
    values: dict[str, Any] = {
        "tip_radius_m": 50e-6,
        "diameter_m": 0.8e-3,
        "exposed_length_m": 4e-3,
        "installation_angle_deg": 70.0,
        "axial_mode": AxialMode.SPRING,
        "spring_stiffness_n_m": 2000.0,
        "spring_travel_m": 4e-3,
        "static_friction": 0.30,
        "kinetic_friction": 0.20,
        "rod_clearance_mode": "disabled_analytic_fixture",
    }
    values.update(overrides)
    return SpineParameters(**values)


def _run_fixture(
    kind: str,
    terrain_parameters: Mapping[str, float] | None,
    *,
    initial_x_m: float,
    drag_length_m: float,
    path_step_m: float,
    spine_parameters: SpineParameters | None = None,
):
    track = _fixture_track(kind, terrain_parameters)
    parameters = spine_parameters or _fixture_parameters()
    core = PrescribedPoseConstitutiveCore(parameters, track)
    settings = ExperimentSettings(
        initial_center_x_m=initial_x_m,
        drag_length_m=drag_length_m,
        path_step_m=path_step_m,
    )
    return SingleSpineExperiment(core, settings).run()


def _gate(
    name: str,
    passed: bool,
    evidence: Mapping[str, Any],
    *,
    requirement: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "requirement": requirement,
        "evidence": dict(evidence),
    }


def run_analytic_validation(output_path: str | Path | None = None) -> dict[str, Any]:
    """Run all M2 stage-I analytic and state-machine acceptance gates."""

    gates: list[dict[str, Any]] = []
    plane = _run_fixture(
        "plane",
        None,
        initial_x_m=-0.001,
        drag_length_m=0.001,
        path_step_m=50e-6,
    )
    last_plane = plane.points[-1].response
    gates.append(
        _gate(
            "plane",
            plane.summary.initial_preload_success
            and plane.summary.run_terminal_state.value == "path_end"
            and plane.summary.event_counts["detach_to_free"] == 0
            and abs(last_plane.normal_xz[0]) < 1e-12
            and abs(last_plane.normal_xz[1] - 1.0) < 1e-12,
            {
                "preload_success": plane.summary.initial_preload_success,
                "terminal": plane.summary.run_terminal_state.value,
                "normal_xz": list(last_plane.normal_xz),
                "normal_force_range_n": list(plane.summary.normal_force_range_n),
                "detach_count": plane.summary.event_counts["detach_to_free"],
            },
            requirement="plane normal, force direction and fixed-Z continuation",
        )
    )

    slope_value = 0.15
    slope = _run_fixture(
        "slope",
        {"slope_x": slope_value},
        initial_x_m=-0.001,
        drag_length_m=0.0005,
        path_step_m=25e-6,
    )
    slope_response = slope.points[-1].response
    expected_normal = np.array([-slope_value, 1.0]) / math.hypot(
        1.0, slope_value
    )
    gates.append(
        _gate(
            "slope",
            slope.summary.initial_preload_success
            and np.linalg.norm(
                np.asarray(slope_response.normal_xz) - expected_normal
            )
            < 2e-3
            and slope_response.normal_force_n >= 0.0,
            {
                "normal_xz": list(slope_response.normal_xz),
                "expected_normal_xz": expected_normal.tolist(),
                "normal_force_n": slope_response.normal_force_n,
            },
            requirement="analytic tangent/normal and unilateral force on a slope",
        )
    )

    bump_parameters = {
        "amplitude_m": 400e-6,
        "center_x_m": -0.001,
        "sigma_x_m": 150e-6,
        "sigma_y_m": 150e-6,
    }
    single_bump = _run_fixture(
        "smooth_bump",
        bump_parameters,
        initial_x_m=-0.001,
        drag_length_m=0.001,
        path_step_m=10e-6,
    )
    gates.append(
        _gate(
            "single_bump",
            single_bump.summary.initial_preload_success
            and single_bump.summary.event_counts["detach_to_free"] >= 1
            and single_bump.summary.run_terminal_state.value == "path_end",
            {
                "events": dict(single_bump.summary.event_counts),
                "terminal": single_bump.summary.run_terminal_state.value,
                "contact_length_m": single_bump.summary.total_contact_length_m,
            },
            requirement="contact-load-detach without terminating the prescribed path",
        )
    )

    double_parameters = {
        "amplitude_1_m": 400e-6,
        "amplitude_2_m": 400e-6,
        "center_1_x_m": -0.001,
        "center_2_x_m": 0.0005,
        "sigma_x_m": 150e-6,
        "sigma_y_m": 150e-6,
    }
    double_bump = _run_fixture(
        "double_bump",
        double_parameters,
        initial_x_m=-0.001,
        drag_length_m=0.002,
        path_step_m=10e-6,
    )
    gates.append(
        _gate(
            "double_bump",
            double_bump.summary.event_counts["detach_to_free"] >= 1
            and double_bump.summary.event_counts["recontact"] >= 1
            and double_bump.summary.run_terminal_state.value == "path_end",
            {
                "events": dict(double_bump.summary.event_counts),
                "terminal": double_bump.summary.run_terminal_state.value,
            },
            requirement="detach, continue through FREE, then recontact",
        )
    )

    sine = _run_fixture(
        "sine_1d",
        {"amplitude_m": 80e-6, "wavelength_m": 600e-6},
        initial_x_m=-0.0012,
        drag_length_m=0.0018,
        path_step_m=20e-6,
    )
    gates.append(
        _gate(
            "sine",
            sine.summary.initial_preload_success
            and sine.summary.run_terminal_state.value == "path_end"
            and sine.summary.numerical_state.value == "converged",
            {
                "events": dict(sine.summary.event_counts),
                "normal_force_range_n": list(sine.summary.normal_force_range_n),
                "max_geometry_residual_m": sine.summary.maximum_abs_geometry_residual_m,
            },
            requirement="bounded periodic response with converged state transitions",
        )
    )

    plane_track = _fixture_track("plane")

    def response_for_stiffness(stiffness: float | None):
        if stiffness is None:
            parameters = _fixture_parameters(
                axial_mode=AxialMode.RIGID,
                spring_stiffness_n_m=None,
                static_friction=10.0,
                kinetic_friction=0.2,
            )
        else:
            parameters = _fixture_parameters(
                spring_stiffness_n_m=stiffness,
                static_friction=10.0,
                kinetic_friction=0.2,
            )
        core = PrescribedPoseConstitutiveCore(parameters, plane_track)
        center_x = -0.001
        holder_x = center_x - parameters.exposed_length_m * core._a[0]
        contact_z = (
            plane_track.radius_m
            - parameters.exposed_length_m * core._a[1]
        )
        first = core.solve_pose((holder_x, contact_z), commit=True)
        return core.solve_pose(
            (holder_x, contact_z - 20e-9),
            first.next_state,
            commit=False,
        )

    stiffnesses = (1e5, 1e7, 1e9)
    stiffness_responses = [response_for_stiffness(value) for value in stiffnesses]
    rigid_response = response_for_stiffness(None)
    distances = [
        abs(response.normal_force_n - rigid_response.normal_force_n)
        for response in stiffness_responses
    ]
    gates.append(
        _gate(
            "rigid_axial_limit",
            distances[2] < distances[1] < distances[0]
            and stiffness_responses[2].spring_compression_m
            < stiffness_responses[1].spring_compression_m
            < stiffness_responses[0].spring_compression_m,
            {
                "spring_stiffness_n_m": list(stiffnesses),
                "normal_force_n": [
                    response.normal_force_n for response in stiffness_responses
                ],
                "rigid_normal_force_n": rigid_response.normal_force_n,
                "distance_to_rigid_n": distances,
                "spring_compression_m": [
                    response.spring_compression_m
                    for response in stiffness_responses
                ],
            },
            requirement="increasing spring stiffness continuously approaches beam-only rigid axial mode",
        )
    )

    p06 = _fixture_parameters(diameter_m=0.6e-3)
    p08 = _fixture_parameters(diameter_m=0.8e-3)
    compliance_ratio = (
        p06.transverse_compliance_m_n / p08.transverse_compliance_m_n
    )
    pure_bending_ratio = (0.8 / 0.6) ** 4
    gates.append(
        _gate(
            "diameter_d_minus_4",
            compliance_ratio > 2.8
            and abs(compliance_ratio - pure_bending_ratio) / pure_bending_ratio
            < 0.12,
            {
                "c_b_0p6_m_n": p06.transverse_compliance_m_n,
                "c_b_0p8_m_n": p08.transverse_compliance_m_n,
                "ratio": compliance_ratio,
                "pure_bending_d_minus_4_ratio": pure_bending_ratio,
            },
            requirement="transverse compliance follows the expected d^-4 dominated trend",
        )
    )

    gates.append(
        _gate(
            "free_after_detach_continues",
            single_bump.summary.event_counts["detach_to_free"] >= 1
            and single_bump.summary.run_terminal_state.value == "path_end",
            {
                "detach_count": single_bump.summary.event_counts["detach_to_free"],
                "terminal": single_bump.summary.run_terminal_state.value,
            },
            requirement="FREE is a physical path state, not a case failure",
        )
    )
    gates.append(
        _gate(
            "recontact_after_detach",
            double_bump.summary.event_counts["detach_to_free"] >= 1
            and double_bump.summary.event_counts["recontact"] >= 1,
            {"events": dict(double_bump.summary.event_counts)},
            requirement="a later asperity can close geometry after a FREE interval",
        )
    )

    plane_slide = [
        point.response
        for point in plane.points
        if point.path_position_m > 0
        and point.response.contact_state is ContactState.SLIDE
    ]
    gates.append(
        _gate(
            "stick_slide_direction_and_work",
            bool(plane_slide)
            and all(response.tangential_force_n <= 1e-9 for response in plane_slide)
            and all(
                response.friction_dissipation_increment_j >= -1e-18
                for response in plane_slide
            ),
            {
                "post_drag_slide_count": len(plane_slide),
                "tangential_force_range_n": (
                    [
                        min(response.tangential_force_n for response in plane_slide),
                        max(response.tangential_force_n for response in plane_slide),
                    ]
                    if plane_slide
                    else []
                ),
                "total_friction_dissipation_j": (
                    plane.points[-1]
                    .response.proposal_state.cumulative_friction_dissipation_j
                ),
            },
            requirement="kinetic friction opposes +x sliding and does non-negative dissipative work",
        )
    )

    free_state = PrescribedPoseConstitutiveCore(
        _fixture_parameters(), plane_track
    ).solve_pose((-0.001, 0.01), commit=False)
    interior = plane.points[1].response
    hard_parameters = _fixture_parameters(
        spring_stiffness_n_m=100.0,
        static_friction=10.0,
    )
    hard_core = PrescribedPoseConstitutiveCore(hard_parameters, plane_track)
    hard_center_x = -0.001
    hard_holder_x = (
        hard_center_x - hard_parameters.exposed_length_m * hard_core._a[0]
    )
    hard_contact_z = (
        plane_track.radius_m
        - hard_parameters.exposed_length_m * hard_core._a[1]
    )
    hard_first = hard_core.solve_pose(
        (hard_holder_x, hard_contact_z),
        commit=True,
    )
    hard_shortening = (
        hard_parameters.spring_travel_m
        + hard_parameters.axial_compliance_m_n
        * float(hard_parameters.spring_stiffness_n_m)
        * hard_parameters.spring_travel_m
        + hard_parameters.axial_compliance_m_n * 0.1
    )
    hard_holder = np.asarray(hard_first.holder_xz_m) + (
        hard_shortening * hard_core._a
    )
    hard = hard_core.solve_pose(
        (float(hard_holder[0]), float(hard_holder[1])),
        hard_first.next_state,
        commit=False,
    )
    gates.append(
        _gate(
            "spring_three_segments",
            free_state.spring_state is SpringState.LOWER_STOP
            and interior.spring_state is SpringState.INTERIOR
            and hard.spring_state is SpringState.HARD_STOP,
            {
                "lower_state": free_state.spring_state.value,
                "interior_state": interior.spring_state.value,
                "hard_state": hard.spring_state.value,
                "hard_compression_m": hard.spring_compression_m,
            },
            requirement="LOWER_STOP, INTERIOR and HARD_STOP are independent contact-compatible states",
        )
    )

    proposal_core = PrescribedPoseConstitutiveCore(
        _fixture_parameters(), plane_track
    )
    x0 = -0.001
    holder_x0 = x0 - proposal_core.parameters.exposed_length_m * proposal_core._a[0]
    holder_z0 = (
        plane_track.radius_m
        - proposal_core.parameters.exposed_length_m * proposal_core._a[1]
    )
    immutable_old = SingleSpineState()
    proposal = proposal_core.solve_pose(
        (holder_x0, holder_z0),
        immutable_old,
        commit=False,
    )
    committed = proposal_core.solve_pose(
        (holder_x0, holder_z0),
        immutable_old,
        commit=True,
    )
    gates.append(
        _gate(
            "proposal_commit_atomicity",
            proposal.next_state is immutable_old
            and proposal.proposal_state != immutable_old
            and committed.next_state == committed.proposal_state
            and immutable_old == SingleSpineState(),
            {
                "proposal_next_is_old": proposal.next_state is immutable_old,
                "committed_steps": committed.next_state.accepted_steps,
                "old_steps": immutable_old.accepted_steps,
            },
            requirement="commit=False has no side effects and commit=True accepts exactly one proposal",
        )
    )

    double_coarse = _run_fixture(
        "double_bump",
        double_parameters,
        initial_x_m=-0.001,
        drag_length_m=0.002,
        path_step_m=20e-6,
    )
    double_fine = double_bump
    contact_delta = abs(
        double_coarse.summary.total_contact_length_m
        - double_fine.summary.total_contact_length_m
    )
    gates.append(
        _gate(
            "path_step_halving",
            contact_delta <= 30e-6
            and double_coarse.summary.event_counts["recontact"]
            == double_fine.summary.event_counts["recontact"],
            {
                "coarse_step_m": 20e-6,
                "fine_step_m": 10e-6,
                "contact_length_delta_m": contact_delta,
                "coarse_events": dict(double_coarse.summary.event_counts),
                "fine_events": dict(double_fine.summary.event_counts),
            },
            requirement="halving the path step preserves macro events and robust path metrics",
        )
    )

    fixture_results = (plane, slope, single_bump, double_bump, sine)
    max_geometry = max(
        result.summary.maximum_abs_geometry_residual_m
        for result in fixture_results
    )
    max_energy = max(
        result.summary.maximum_abs_energy_residual_j
        for result in fixture_results
    )
    max_force = max(
        abs(component)
        for result in fixture_results
        for point in result.points
        for component in point.response.wall_on_spine_force_xz_n
    )
    maximum_slide_friction_residual = max(
        (
            abs(
                abs(point.response.tangential_force_n)
                - result.parameters.kinetic_friction
                * point.response.normal_force_n
            )
            for result in fixture_results
            for point in result.points
            if point.response.contact_state is ContactState.SLIDE
        ),
        default=0.0,
    )
    gates.append(
        _gate(
            "residual_and_peak_audit",
            max_geometry <= 5e-9
            and max_energy <= 2e-5
            and max_force < 20.0
            and maximum_slide_friction_residual
            <= max(
                PrescribedPoseConstitutiveCore(
                    _fixture_parameters(), plane_track
                ).settings.friction_residual_tolerance_n,
                1e-12,
            ),
            {
                "maximum_abs_geometry_residual_m": max_geometry,
                "maximum_abs_energy_residual_j": max_energy,
                "maximum_abs_force_component_n": max_force,
                "maximum_slide_friction_residual_n": (
                    maximum_slide_friction_residual
                ),
            },
            requirement="no unexplained force peak, contact residual or energy residual",
        )
    )

    report = {
        "schema_version": "1",
        "m2_module_version": M2_MODULE_VERSION,
        "created_at_utc": utc_now(),
        "gate_count": len(gates),
        "passed_count": sum(gate["passed"] for gate in gates),
        "all_passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "formal_random_screening_allowed": False,
        "formal_random_screening_blockers": [
            "M4 and the M0-M4 full chain are not implemented/frozen",
            "full_chain_frozen_manifest.json does not exist",
            "user approval to start M2 round 1 has not been recorded",
        ],
    }
    if output_path is not None:
        atomic_write_json(Path(output_path), report)
    return report


def run_m1_suite_smoke(
    suite_report_path: str | Path,
    *,
    drag_length_m: float = 1e-3,
    path_step_m: float = 50e-6,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Exercise M2 fields on all ten saved M1 conditions without ranking them."""

    report_path = Path(suite_report_path)
    source = json.loads(report_path.read_text(encoding="utf-8"))
    library_root = report_path.parent / "terrain_library"
    library = TerrainLibrary(library_root)
    conditions: list[dict[str, Any]] = []
    for condition in source["conditions"]:
        recipe_id = condition["terrain_recipe_id"]
        region_id = condition["region_id"]
        metadata_candidates = list(
            (
                library_root
                / "tracks"
                / recipe_id
                / region_id
                / "50um"
            ).glob("*.json")
        )
        zero_y_candidates = []
        for candidate in metadata_candidates:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
            if math.isclose(
                float(metadata["y_global_m"]),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                zero_y_candidates.append((candidate, metadata))
        if len(zero_y_candidates) != 1:
            conditions.append(
                {
                    "name": condition["name"],
                    "passed": False,
                    "reason": "expected exactly one saved y=0, 50um track",
                }
            )
            continue
        _metadata_path, metadata = zero_y_candidates[0]
        track = library.load_track(
            recipe_id,
            region_id,
            float(metadata["radius_m"]),
            metadata["track_id"],
        )
        valid_indices = np.flatnonzero(track.valid_mask)
        parameters = _fixture_parameters(
            rod_clearance_mode="unclosed",
        )
        core = PrescribedPoseConstitutiveCore(parameters, track)
        attempt_fractions = (
            0.50,
            0.60,
            0.40,
            0.70,
            0.30,
            0.55,
            0.65,
            0.80,
        )
        attempts: list[dict[str, Any]] = []
        result = None
        initial_x = math.nan
        for fraction in attempt_fractions:
            center_index = valid_indices[
                int(round(fraction * (len(valid_indices) - 1)))
            ]
            candidate_x = float(track.x_global_m[center_index])
            candidate_result = SingleSpineExperiment(
                core,
                ExperimentSettings(
                    initial_center_x_m=candidate_x,
                    drag_length_m=drag_length_m,
                    path_step_m=path_step_m,
                ),
            ).run()
            attempts.append(
                {
                    "valid_track_fraction": fraction,
                    "initial_center_x_m": candidate_x,
                    "preload_success": (
                        candidate_result.summary.initial_preload_success
                    ),
                    "terminal": (
                        candidate_result.summary.run_terminal_state.value
                    ),
                }
            )
            result = candidate_result
            initial_x = candidate_x
            if (
                candidate_result.summary.initial_preload_success
                and candidate_result.summary.numerical_state.value == "converged"
                and candidate_result.summary.run_terminal_state.value == "path_end"
            ):
                break
        assert result is not None
        numerical_failure = result.summary.numerical_state.value == "nonconverged"
        condition_passed = (
            result.summary.initial_preload_success
            and not numerical_failure
            and result.summary.run_terminal_state.value == "path_end"
        )
        conditions.append(
            {
                "name": condition["name"],
                "description": condition["description"],
                "terrain_recipe_id": recipe_id,
                "region_id": region_id,
                "track_id": track.track_id,
                "initial_center_x_m": initial_x,
                "start_search_attempts": attempts,
                "passed": condition_passed,
                "initial_preload_success": result.summary.initial_preload_success,
                "run_terminal_state": result.summary.run_terminal_state.value,
                "termination_reason": result.summary.termination_reason,
                "numerical_state": result.summary.numerical_state.value,
                "model_state": result.summary.model_state.value,
                "point_count": len(result.points),
                "event_counts": dict(result.summary.event_counts),
                "normal_force_range_n": list(result.summary.normal_force_range_n),
                "maximum_abs_geometry_residual_m": (
                    result.summary.maximum_abs_geometry_residual_m
                ),
                "formal_ranking_eligible": False,
            }
        )
    output = {
        "schema_version": "1",
        "m2_module_version": M2_MODULE_VERSION,
        "created_at_utc": utc_now(),
        "source_suite": str(report_path),
        "condition_count": len(conditions),
        "passed_count": sum(item["passed"] for item in conditions),
        "all_passed": all(item["passed"] for item in conditions),
        "purpose": "interface_state_and_field_smoke_only",
        "formal_ranking_allowed": False,
        "conditions": conditions,
    }
    if output_path is not None:
        atomic_write_json(Path(output_path), output)
    return output
