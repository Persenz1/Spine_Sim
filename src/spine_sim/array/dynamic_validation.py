"""Analytic and existing-M1 gates for m3.4.0 common-backplate dynamics."""

from __future__ import annotations

import math
import random
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from spine_sim.contact import (
    DynamicContactSettings,
    DynamicIntegratorSettings,
    EventLabel,
)
from spine_sim.contact.validation import _fixture_parameters
from spine_sim.io.results import atomic_write_json, utc_now
from spine_sim.terrain import TerrainLibrary, TrackGeometry

from .dynamics import (
    ArrayDynamicExperimentSettings,
    DynamicCommonBackplateArray,
    DynamicCommonBackplateExperiment,
)
from .case import (
    _RodClearanceOutOfBoundsError,
    _arrays,
    _proxy_array_rod_clearance,
)
from .design import (
    PLACEMENT_SEARCH_OFFSETS_XY_M,
    build_base_hardware,
    build_full_array_design,
    validate_terrain_catalog,
)
from .models import M3_MODEL_LEVEL, M3_MODULE_VERSION, ArrayConfiguration
from .proxy_parameters import (
    EngineeringProxyScenario,
    estimate_backplate_dynamics,
)


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


def _track(
    *,
    y_m: float,
    profile: Callable[[np.ndarray], np.ndarray] | None = None,
    vertical_offset_m: float = 0.0,
    track_suffix: str = "plane",
    radius_m: float = 50e-6,
) -> TrackGeometry:
    resolution = 10e-6
    x = np.arange(-0.020, 0.020 + 0.5 * resolution, resolution)
    terrain_height = (
        np.zeros_like(x)
        if profile is None
        else np.asarray(profile(x), dtype=np.float64)
    )
    envelope = radius_m + vertical_offset_m + terrain_height
    slope = np.gradient(envelope, resolution)
    return TrackGeometry(
        terrain_recipe_id="terrain_recipe_m3_dynamic_analytic",
        region_id="region_m3_dynamic_analytic",
        track_id=f"track_m3_dynamic_{track_suffix}_{y_m:+.6f}",
        radius_m=radius_m,
        y_global_m=float(y_m),
        resolution_m=resolution,
        envelope_algorithm_version="m3-dynamic-analytic-v1",
        x_global_m=x,
        envelope_height_m=envelope,
        envelope_slope_x=slope,
        support_x_m=x.copy(),
        support_y_m=np.full_like(x, y_m),
        valid_mask=np.ones(x.size, dtype=np.bool_),
        near_tie_flag=np.zeros(x.size, dtype=np.bool_),
    )


def _tracks(
    configuration: ArrayConfiguration,
    *,
    profiles: Sequence[Callable[[np.ndarray], np.ndarray] | None] | None = None,
    vertical_offsets_m: Sequence[float] | None = None,
    suffix: str = "plane",
) -> tuple[TrackGeometry, ...]:
    count = configuration.pin_count
    profile_values = tuple(profiles or (None,) * count)
    offset_values = tuple(vertical_offsets_m or (0.0,) * count)
    if len(profile_values) != count or len(offset_values) != count:
        raise ValueError("one analytic profile and height offset are required per pin")
    return tuple(
        _track(
            y_m=offset[1],
            profile=profile_values[index],
            vertical_offset_m=offset_values[index],
            track_suffix=f"{suffix}_{index}",
            radius_m=configuration.pin_parameters[index].tip_radius_m,
        )
        for index, offset in enumerate(configuration.holder_offsets_xyz_m)
    )


def _settings(
    *,
    drag_length_m: float,
    preload_n: float = 0.5,
) -> ArrayDynamicExperimentSettings:
    return ArrayDynamicExperimentSettings(
        drag_length_m=drag_length_m,
        external_total_preload_n=preload_n,
        initial_common_ux_m=0.0,
        drag_speed_m_s=1e-3,
        backplate_mass_kg=0.05,
        backplate_vertical_damping_n_s_m=1.0,
        backplate_rotational_dofs="locked",
        backplate_inertia_kg_m2=None,
        maximum_preload_approach_m=8e-3,
        output_spacing_m=20e-6,
        effective_pin_normal_force_min_n=0.02,
        unclosed_parameter_names=("analytic_fixture_parameters_unfrozen",),
        time_step_convergence_checked=False,
        contact_parameter_convergence_checked=False,
    )


def _integrator(time_step_s: float) -> DynamicIntegratorSettings:
    return DynamicIntegratorSettings(
        time_step_s=time_step_s,
        settling_time_s=0.05,
        settling_velocity_tolerance_m_s=2e-5,
        maximum_settling_time_s=1.0,
        maximum_steps=100_000,
    )


def _run(
    configuration: ArrayConfiguration,
    tracks: Sequence[TrackGeometry],
    *,
    drag_length_m: float,
    preload_n: float = 0.5,
    time_step_s: float = 1e-3,
):
    system = DynamicCommonBackplateArray(
        configuration,
        tracks,
        unit_origin_xy_m=(0.0, 0.0),
        contact=DynamicContactSettings(
            normal_model="rigid_moreau",
            restitution_coefficient=0.0,
            position_correction=1.0,
            activation_tolerance_m=2e-9,
            impact_velocity_threshold_m_s=1e-5,
            maximum_contact_force_n=250.0,
            projection_iterations=20,
        ),
    )
    result = DynamicCommonBackplateExperiment(
        system,
        _settings(drag_length_m=drag_length_m, preload_n=preload_n),
        _integrator(time_step_s),
    ).run()
    return system, result


def _drop_profile(x: np.ndarray) -> np.ndarray:
    start = 1.55e-3
    width = 0.06e-3
    depth = 0.20e-3
    fraction = np.clip((x - start) / width, 0.0, 1.0)
    return -0.5 * depth * (1.0 - np.cos(math.pi * fraction))


def _steady_mean(result) -> float:
    points = result.points
    start = max(1, len(points) // 2)
    values = [
        point.total_contact_reaction_z_n
        for point in points[start:]
        if not any(
            label == EventLabel.IMPACT.value
            for _index, label in point.event_labels
        )
    ]
    return float(np.mean(values))


def run_dynamic_analytic_validation(
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run only the production dynamic M3 gates; no legacy screen is executed."""

    parameters = _fixture_parameters(
        axial_damping_ratio=0.20,
        transverse_damping_ratio=0.20,
    )
    pair = ArrayConfiguration(
        1, 2, 4e-3, parameters, fixture_only=True
    )
    plane_tracks = _tracks(pair)
    plane_system, plane = _run(
        pair, plane_tracks, drag_length_m=0.30e-3
    )
    plane_mean = _steady_mean(plane)
    gates: list[dict[str, Any]] = [
        _gate(
            "plane_multi_pin_mean_total_reaction",
            plane.summary.run_terminal_state.value == "path_end"
            and abs(plane_mean - 0.5) < 0.025,
            {
                "external_total_preload_n": 0.5,
                "steady_total_reaction_mean_n": plane_mean,
                "instantaneous_reaction_range_n": list(
                    plane.summary.total_normal_force_range_n
                ),
            },
            requirement=(
                "time-mean total normal reaction balances one constant external "
                "array preload; instantaneous equality is not imposed"
            ),
        )
    ]
    settlement_applied = np.asarray(
        [
            point.applied_total_preload_n
            for point in plane.settlement_trace
        ],
        dtype=np.float64,
    )
    settlement_peak_reaction = max(
        point.total_contact_reaction_z_n
        for point in plane.settlement_trace
    )
    gates.append(
        _gate(
            "smooth_total_preload_ramp_and_settlement_gates",
            settlement_applied[0] < 1e-3 * settlement_applied[-1]
            and np.all(np.diff(settlement_applied) >= -1e-15)
            and abs(settlement_applied[-1] - 0.5) <= 1e-12
            and settlement_peak_reaction < 0.55
            and plane.summary.settlement_stable_steps
            >= plane.summary.settlement_required_stable_steps
            and plane.summary.settlement_final_reaction_error_n
            <= max(
                plane.experiment.settling_reaction_force_tolerance_n,
                0.5
                * plane.experiment.settling_reaction_force_relative_tolerance,
            )
            and plane.summary.settlement_final_maximum_mode_speed_m_s
            <= plane.integrator.settling_velocity_tolerance_m_s
            and abs(plane.summary.settlement_final_dynamic_residual_n)
            <= plane.experiment.settling_dynamic_residual_tolerance_n,
            {
                "profile": plane.summary.settlement_ramp_profile,
                "settlement_steps": plane.summary.settlement_steps,
                "first_applied_preload_n": settlement_applied[0],
                "final_applied_preload_n": settlement_applied[-1],
                "maximum_reaction_n": settlement_peak_reaction,
                "final_reaction_error_n": (
                    plane.summary.settlement_final_reaction_error_n
                ),
                "final_maximum_mode_speed_m_s": (
                    plane.summary.settlement_final_maximum_mode_speed_m_s
                ),
                "final_dynamic_residual_n": (
                    plane.summary.settlement_final_dynamic_residual_n
                ),
                "stable_steps": plane.summary.settlement_stable_steps,
            },
            requirement=(
                "minimum-jerk total preload is monotone and settlement requires "
                "velocity, reaction balance, dynamic residual, contact and a "
                "continuous stable window"
            ),
        )
    )

    square = ArrayConfiguration(2, 2, 4e-3, parameters)
    _square_system, square_result = _run(
        square,
        _tracks(square),
        drag_length_m=0.005e-3,
    )
    square_initial_loads = np.asarray(
        [
            pin.normal_force_n
            for pin in square_result.points[0].pin_responses
        ],
        dtype=np.float64,
    )
    gates.append(
        _gate(
            "symmetric_2x2_load_distribution",
            square_result.summary.run_terminal_state.value == "path_end"
            and np.ptp(square_initial_loads) <= 1e-12
            and abs(np.sum(square_initial_loads) - 0.5) <= 1e-5,
            {
                "initial_pin_normal_force_n": square_initial_loads.tolist(),
                "load_range_n": float(np.ptp(square_initial_loads)),
            },
            requirement=(
                "a symmetric plane and symmetric 2x2 array distribute the "
                "settled total preload symmetrically"
            ),
        )
    )

    height_tracks = _tracks(
        pair,
        vertical_offsets_m=(80e-6, 0.0),
        suffix="height_transfer",
    )
    _height_system, height = _run(
        pair, height_tracks, drag_length_m=0.08e-3
    )
    initial_loads = [
        pin.normal_force_n for pin in height.points[0].pin_responses
    ]
    gates.append(
        _gate(
            "different_initial_heights_transfer_load",
            height.summary.run_terminal_state.value == "path_end"
            and abs(sum(initial_loads) - 0.5) < 0.01
            and abs(initial_loads[0] - initial_loads[1]) > 0.02,
            {
                "pin_initial_normal_force_n": initial_loads,
                "sum_n": sum(initial_loads),
            },
            requirement=(
                "height mismatch produces dynamic load transfer instead of a "
                "preassigned per-pin share"
            ),
        )
    )

    detach_tracks = _tracks(
        pair,
        profiles=(_drop_profile, None),
        suffix="single_detach",
    )
    _detach_system, detach = _run(
        pair, detach_tracks, drag_length_m=0.55e-3
    )
    detach_events = [
        (index, label)
        for point in detach.points
        for index, label in point.event_labels
    ]
    pin0_labels = [label for index, label in detach_events if index == 0]
    gates.append(
        _gate(
            "single_pin_detach_and_recontact",
            EventLabel.DETACH_TO_FREE.value in pin0_labels
            and EventLabel.RECONTACT.value in pin0_labels
            and detach.summary.run_terminal_state.value == "path_end",
            {
                "pin_0_event_counts": {
                    label: pin0_labels.count(label)
                    for label in sorted(set(pin0_labels))
                },
                "pin_0_first_events": pin0_labels[:20],
                "terminal": detach.summary.run_terminal_state.value,
                "termination_reason": detach.summary.termination_reason,
            },
            requirement=(
                "a detached pin has zero force, remains in the array integration, "
                "and can reimpact without no_admissible_contact_equilibrium"
            ),
        )
    )

    impact_tracks = _tracks(
        pair,
        profiles=(_drop_profile, _drop_profile),
        suffix="simultaneous_impact",
    )
    _impact_system, impact = _run(
        pair, impact_tracks, drag_length_m=0.55e-3
    )
    simultaneous = [
        point
        for point in impact.points
        if sum(
            label == EventLabel.IMPACT.value
            for _index, label in point.event_labels
        )
        >= 2
    ]
    gates.append(
        _gate(
            "simultaneous_multi_pin_impact",
            bool(simultaneous)
            and impact.summary.run_terminal_state.value == "path_end",
            {
                "simultaneous_impact_point_count": len(simultaneous),
                "impact_peak_reaction_n": impact.summary.total_normal_force_range_n[1],
                "external_total_preload_n": 0.5,
            },
            requirement=(
                "coupled contact solve accepts simultaneous impacts and permits "
                "impact reaction above external preload"
            ),
        )
    )

    settled_state, _settled_point = DynamicCommonBackplateExperiment(
        plane_system,
        _settings(drag_length_m=0.1e-3),
        _integrator(1e-3),
    )._settle()
    forward = plane_system.propose_step(
        settled_state,
        _settings(drag_length_m=0.1e-3),
        common_ux_m=1e-6,
        drag_speed_m_s=1e-3,
        dt=1e-3,
        traversal_order=(0, 1),
    )
    reverse = plane_system.propose_step(
        settled_state,
        _settings(drag_length_m=0.1e-3),
        common_ux_m=1e-6,
        drag_speed_m_s=1e-3,
        dt=1e-3,
        traversal_order=(1, 0),
    )
    random_order = [0, 1]
    random.Random(8128).shuffle(random_order)
    shuffled = plane_system.propose_step(
        settled_state,
        _settings(drag_length_m=0.1e-3),
        common_ux_m=1e-6,
        drag_speed_m_s=1e-3,
        dt=1e-3,
        traversal_order=random_order,
    )
    traversal_equal = (
        forward.proposal_state == reverse.proposal_state
        and forward.proposal_state == shuffled.proposal_state
        and forward.point == reverse.point
        and forward.point == shuffled.point
    )
    gates.append(
        _gate(
            "pin_traversal_order_invariance",
            traversal_equal,
            {"orders": [[0, 1], [1, 0], random_order]},
            requirement=(
                "forward, reverse and seeded-random evaluation produce the same "
                "global proposal and accepted state"
            ),
        )
    )

    invalid = plane_system.propose_step(
        settled_state,
        _settings(drag_length_m=0.1e-3),
        common_ux_m=0.1,
        drag_speed_m_s=1e-3,
        dt=1e-3,
    )
    rejected_state = plane_system.commit_step(
        settled_state, invalid, accept=False
    )
    gates.append(
        _gate(
            "atomic_rejected_step_no_pollution",
            not invalid.proposal_valid
            and invalid.proposal_state == settled_state
            and rejected_state == settled_state
            and settled_state.accepted_steps
            == invalid.source_state.accepted_steps,
            {
                "proposal_valid": invalid.proposal_valid,
                "rejection_reason": invalid.rejection_reason,
                "old_accepted_steps": settled_state.accepted_steps,
                "rejected_state_equal_old": rejected_state == settled_state,
            },
            requirement=(
                "rejecting a fully evaluated global proposal returns the exact old "
                "ArrayDynamicState"
            ),
        )
    )

    _half_system, half = _run(
        pair, plane_tracks, drag_length_m=0.30e-3, time_step_s=0.5e-3
    )
    full_mean = _steady_mean(plane)
    half_mean = _steady_mean(half)
    gates.append(
        _gate(
            "internal_time_step_halving",
            half.summary.run_terminal_state.value == "path_end"
            and abs(full_mean - half_mean) < 0.02
            and abs(
                plane.summary.tangential_force_median_n
                - half.summary.tangential_force_median_n
            )
            < 0.02,
            {
                "dt_s": 1e-3,
                "half_dt_s": 0.5e-3,
                "total_reaction_mean_difference_n": abs(
                    full_mean - half_mean
                ),
                "steady_tangential_median_difference_n": abs(
                    plane.summary.tangential_force_median_n
                    - half.summary.tangential_force_median_n
                ),
            },
            requirement=(
                "halving the internal time step preserves steady load and pull "
                "statistics within declared fixture tolerances"
            ),
        )
    )

    gates.append(
        _gate(
            "dynamic_and_energy_residuals",
            plane.summary.maximum_abs_dynamic_residual_n < 1e-8
            and plane.summary.maximum_abs_energy_residual_j < 1e-15
            and plane.summary.cumulative_relative_energy_error < 1e-12
            and plane.summary.maximum_abs_contact_work_identity_residual_j
            < 1e-15,
            {
                "maximum_abs_dynamic_residual_n": (
                    plane.summary.maximum_abs_dynamic_residual_n
                ),
                "maximum_abs_energy_residual_j": (
                    plane.summary.maximum_abs_energy_residual_j
                ),
                "cumulative_relative_energy_error": (
                    plane.summary.cumulative_relative_energy_error
                ),
                "maximum_abs_contact_work_identity_residual_j": (
                    plane.summary.maximum_abs_contact_work_identity_residual_j
                ),
            },
            requirement=(
                "joint equation, backward-Euler algorithmic dissipation and "
                "prescribed-drive/contact work identities are audited"
            ),
        )
    )

    two_by_five = ArrayConfiguration(2, 5, 5e-3, parameters)
    five_by_two = ArrayConfiguration(5, 2, 5e-3, parameters)
    gates.append(
        _gate(
            "array_orientation_identity",
            two_by_five.configuration_id != five_by_two.configuration_id
            and two_by_five.holder_offsets_xyz_m
            != five_by_two.holder_offsets_xyz_m,
            {
                "2x5_configuration_id": two_by_five.configuration_id,
                "5x2_configuration_id": five_by_two.configuration_id,
            },
            requirement="2x5 and 5x2 remain distinct hardware orientations",
        )
    )

    wrench_point = plane.points[-1]
    pin_wrench = np.asarray(
        [
            pin.spine_on_plate_wrench_about_unit
            for pin in wrench_point.pin_responses
        ],
        dtype=np.float64,
    )
    total_wrench = np.asarray(
        wrench_point.wall_on_unit_wrench_about_origin,
        dtype=np.float64,
    )
    aggregate_error = float(np.max(np.abs(np.sum(pin_wrench, axis=0) - total_wrench)))
    shift_errors: list[float] = []
    origin = np.asarray(wrench_point.backplate_position_xyz_m)
    for pin in wrench_point.pin_responses:
        holder_wrench = np.asarray(pin.spine_on_plate_wrench_about_holder)
        shifted = holder_wrench.copy()
        shifted[3:] += np.cross(
            np.asarray(pin.holder_xyz_m) - origin,
            shifted[:3],
        )
        shift_errors.append(
            float(
                np.max(
                    np.abs(
                        shifted
                        - np.asarray(pin.spine_on_plate_wrench_about_unit)
                    )
                )
            )
        )
    gates.append(
        _gate(
            "wrench_shift_and_aggregation_identity",
            aggregate_error <= 1e-15 and max(shift_errors) <= 1e-15,
            {
                "maximum_shift_error": max(shift_errors),
                "aggregation_error": aggregate_error,
            },
            requirement=(
                "every holder wrench is shifted to the common origin before the "
                "exact array sum"
            ),
        )
    )

    gates.append(
        _gate(
            "unfrozen_parameter_ranking_gate",
            plane.summary.model_state.value == "parameter_unclosed"
            and not plane.summary.formal_ranking_eligible
            and bool(plane.summary.unclosed_parameter_names),
            {
                "model_state": plane.summary.model_state.value,
                "formal_ranking_eligible": plane.summary.formal_ranking_eligible,
                "unclosed_parameter_names": list(
                    plane.summary.unclosed_parameter_names
                ),
            },
            requirement=(
                "unfrozen physical/numerical parameters force parameter_unclosed "
                "and keep formal ranking disabled"
            ),
        )
    )

    base_hardware = build_base_hardware()
    full_design = build_full_array_design()
    gates.append(
        _gate(
            "complete_cartesian_array_design",
            len(base_hardware) == 48
            and sum(
                row["angle_layout"] == "fixed" for row in full_design
            )
            == 1008
            and sum(
                row["angle_layout"] == "gradient_80_to_60"
                for row in full_design
            )
            == 336
            and not any(
                row["fixed_angle_deg"] == 50.0 for row in full_design
            ),
            {
                "base_hardware_count": len(base_hardware),
                "fixed_array_count": sum(
                    row["angle_layout"] == "fixed"
                    for row in full_design
                ),
                "gradient_array_count": sum(
                    row["angle_layout"] == "gradient_80_to_60"
                    for row in full_design
                ),
            },
            requirement=(
                "the formal plan preserves all 48 base combinations, seven "
                "oriented shapes, three spacings and the 80-to-60 gradient"
            ),
        )
    )

    stiffness_forces: list[float] = []
    for stiffness in (300.0, 800.0, 2000.0):
        spring_parameters = replace(
            parameters,
            axial_mode="spring",
            spring_stiffness_n_m=stiffness,
        )
        spring_pair = ArrayConfiguration(
            1, 2, 4e-3, spring_parameters, fixture_only=True
        )
        spring_system = DynamicCommonBackplateArray(
            spring_pair,
            _tracks(spring_pair),
            unit_origin_xy_m=(0.0, 0.0),
        )
        stiffness_forces.append(
            spring_system.units[0].axial_response(10e-6)[0]
        )
    rigid_parameters = replace(
        parameters,
        axial_mode="rigid",
        spring_stiffness_n_m=None,
    )
    rigid_pair = ArrayConfiguration(
        1, 2, 4e-3, rigid_parameters, fixture_only=True
    )
    rigid_system = DynamicCommonBackplateArray(
        rigid_pair,
        _tracks(rigid_pair),
        unit_origin_xy_m=(0.0, 0.0),
    )
    rigid_force = rigid_system.units[0].axial_response(10e-6)[0]
    gates.append(
        _gate(
            "required_axial_stiffness_units_and_rigid_limit",
            np.all(np.diff(stiffness_forces) > 0.0)
            and rigid_force > stiffness_forces[-1],
            {
                "spring_stiffness_n_m": [300.0, 800.0, 2000.0],
                "force_at_10um_shortening_n": stiffness_forces,
                "rigid_force_at_10um_shortening_n": rigid_force,
            },
            requirement=(
                "300/800/2000 N/m are treated as SI stiffness and increase "
                "continuously toward the axial rigid installation"
            ),
        )
    )

    summary_only = _arrays(plane, "summary")
    aggregate = _arrays(plane, "aggregate_trace")
    full = _arrays(plane, "full_pin_trace")
    gates.append(
        _gate(
            "output_level_summary_invariance",
            not summary_only
            and "pin_normal_force_n" not in aggregate
            and "pin_normal_force_n" in full
            and "settlement_ramp_fraction" in aggregate,
            {
                "summary_array_count": len(summary_only),
                "aggregate_array_count": len(aggregate),
                "full_array_count": len(full),
                "summary_hash_basis": "same in-memory experiment result",
            },
            requirement=(
                "summary, aggregate_trace and full_pin_trace change only "
                "serialization detail, never simulated summary values"
            ),
        )
    )

    report = {
        "generated_at_utc": utc_now(),
        "m3_module_version": M3_MODULE_VERSION,
        "model_level": M3_MODEL_LEVEL,
        "validation_scope": (
            "production continuous-total-preload dynamic array only; "
            "legacy fixed-Z validation not executed"
        ),
        "gate_count": len(gates),
        "passed_count": sum(bool(gate["passed"]) for gate in gates),
        "all_passed": all(bool(gate["passed"]) for gate in gates),
        "formal_m3_round1_allowed": False,
        "formal_m3_round1_blockers": [
            "dynamic parameters are not project calibrated/frozen",
            "the required 3-family x 100-seed paired M1 catalog is absent",
            "100 mm path time-step/contact/settlement-damping convergence is open",
            "5 um final terrain-resolution convergence is open",
            "rough-terrain 6x6 contact-stabilization convergence is open",
        ],
        "gates": gates,
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report


def _select_existing_m1_condition(
    catalog: Mapping[str, Any],
    *,
    seed: int | None = None,
    terrain_family: str | None = None,
    condition_name: str | None = None,
) -> dict[str, Any]:
    """Select exactly one verified condition without guessing across families."""

    conditions = validate_terrain_catalog(
        catalog,
        require_formal_300=False,
    )
    if seed is not None:
        conditions = [
            condition
            for condition in conditions
            if int(condition["seed"]) == seed
        ]
    if terrain_family is not None:
        normalized_family = terrain_family.strip().lower().replace("-", "_")
        conditions = [
            condition
            for condition in conditions
            if condition["terrain_family"] == normalized_family
        ]
    if condition_name is not None:
        conditions = [
            condition
            for condition in conditions
            if str(condition.get("name", "")) == condition_name
        ]
    if not conditions:
        raise ValueError("the selected existing M1 catalog condition is absent")
    if len(conditions) != 1:
        choices = ", ".join(
            (
                f"{condition.get('name', '<unnamed>')}"
                f"[{condition['terrain_family']}/seed={condition['seed']}]"
            )
            for condition in conditions
        )
        raise ValueError(
            "existing M1 smoke selection is ambiguous; specify "
            "--condition-name or both --terrain-family and --seed. "
            f"Matching conditions: {choices}"
        )
    return conditions[0]


def _existing_m1_smoke_contact_settings() -> DynamicContactSettings:
    scenario = EngineeringProxyScenario("baseline")
    return DynamicContactSettings(
        normal_model="rigid_moreau",
        restitution_coefficient=scenario.restitution_coefficient,
        position_correction=scenario.contact_position_correction,
        activation_tolerance_m=2e-9,
        impact_velocity_threshold_m_s=1e-5,
        maximum_contact_force_n=250.0,
        projection_iterations=20,
    )


def _existing_m1_smoke_integrator() -> DynamicIntegratorSettings:
    return DynamicIntegratorSettings(
        time_step_s=1e-3,
        settling_time_s=0.25,
        settling_velocity_tolerance_m_s=2e-5,
        maximum_settling_time_s=2.0,
        maximum_steps=200_000,
    )


def _verify_existing_m1_condition_identity(
    *,
    library: TerrainLibrary,
    condition: Mapping[str, Any],
    verify_data_hash: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    recipe_id = str(condition["terrain_recipe_id"])
    region_id = str(condition["region_id"])
    recipe = library.load_recipe(recipe_id)
    region = library.load_region_spec(recipe_id, region_id)
    metadata = json.loads(
        library.region_manifest_path(recipe_id, region_id).read_text(
            encoding="utf-8"
        )
    )
    checks = {
        "catalog_full_sha256_verified": (
            condition.get("full_sha256_verified") is True
        ),
        "recipe_seed_matches_catalog": (
            recipe.seed == int(condition["seed"])
        ),
        "recipe_hash_matches_catalog": (
            not condition.get("recipe_hash")
            or recipe.recipe_hash == condition["recipe_hash"]
        ),
        "manifest_recipe_hash_matches": (
            metadata.get("recipe_hash") == recipe.recipe_hash
        ),
        "manifest_region_identity_matches": (
            metadata.get("terrain_recipe_id") == recipe_id
            and metadata.get("region_id") == region_id
        ),
        "manifest_data_sha256_matches_catalog": (
            metadata.get("data_sha256") == condition["data_sha256"]
        ),
        "material_family_matches_catalog": (
            recipe.generator_name != "material_hybrid"
            or recipe.material == condition["terrain_family"]
        ),
        "material_subtype_matches_catalog": (
            recipe.generator_name != "material_hybrid"
            or not condition.get("subtype")
            or recipe.subtype == condition["subtype"]
        ),
    }
    if condition.get("data_path"):
        checks["data_path_matches_library"] = (
            Path(str(condition["data_path"])).resolve()
            == library.region_data_path(recipe_id, region_id).resolve()
        )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(
            "existing M1 condition identity validation failed: "
            + ", ".join(failed)
        )
    if verify_data_hash:
        height = library.open_region(
            recipe_id,
            region_id,
            verify_hash=True,
        )
        if getattr(height, "_mmap", None) is not None:
            height._mmap.close()
    checks["data_file_sha256_recomputed"] = bool(verify_data_hash)
    return recipe, region, checks


def _run_existing_m1_condition_smoke(
    catalog_path: Path,
    catalog: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    drag_length_m: float,
    verify_data_hash: bool,
    placement_search: bool,
) -> dict[str, Any]:
    """Run one explicitly selected M1 realization; never a campaign."""

    condition_started = time.perf_counter()
    library = TerrainLibrary(catalog["library_root"])
    recipe, region, identity_checks = _verify_existing_m1_condition_identity(
        library=library,
        condition=condition,
        verify_data_hash=verify_data_hash,
    )
    proxy_scenario = EngineeringProxyScenario("baseline")
    parameters = _fixture_parameters(
        tip_radius_m=100e-6,
        diameter_m=0.8e-3,
        installation_angle_deg=70.0,
        spring_stiffness_n_m=800.0,
        rod_clearance_mode="proxy_cylindrical_shank_postcheck",
        static_friction=proxy_scenario.static_friction,
        kinetic_friction=proxy_scenario.kinetic_friction,
        axial_damping_ratio=proxy_scenario.pin_modal_damping_ratio,
        transverse_damping_ratio=proxy_scenario.pin_modal_damping_ratio,
        yield_strength_pa=proxy_scenario.yield_strength_pa,
    )
    rows: list[dict[str, Any]] = []
    placement_offsets = (
        PLACEMENT_SEARCH_OFFSETS_XY_M
        if placement_search
        else ((0.0, 0.0),)
    )
    for size in (2, 4, 6):
        size_started = time.perf_counter()
        configuration = ArrayConfiguration(
            size,
            size,
            4e-3,
            parameters,
        )
        backplate = estimate_backplate_dynamics(
            configuration,
            proxy_scenario,
        )
        settings = replace(
            _settings(
                drag_length_m=drag_length_m,
                preload_n=1.0,
            ),
            backplate_mass_kg=float(backplate["backplate_mass_kg"]),
            backplate_vertical_damping_n_s_m=float(
                backplate["backplate_vertical_damping_n_s_m"]
            ),
            settlement_damping_scale=(
                proxy_scenario.settlement_damping_scale
            ),
            output_spacing_m=min(20e-6, drag_length_m),
            unclosed_parameter_names=(
                "existing_M1_smoke_not_formal_300_catalog",
                "dynamic_parameters_not_experimentally_calibrated",
            ),
        )
        attempts: list[dict[str, Any]] = []
        selected = None
        best_clearance = -math.inf
        geometry_retry_triggered = False
        for attempt_index, placement_offset in enumerate(
            placement_offsets
        ):
            attempt_started = time.perf_counter()
            tracks_by_y: dict[float, TrackGeometry] = {}
            for holder_offset in configuration.holder_offsets_xyz_m:
                y_global_m = holder_offset[1] + placement_offset[1]
                if y_global_m not in tracks_by_y:
                    tracks_by_y[y_global_m] = library.cache_track(
                        recipe,
                        region,
                        radius_m=parameters.tip_radius_m,
                        y_global_m=y_global_m,
                    )
            tracks = tuple(
                tracks_by_y[
                    holder_offset[1] + placement_offset[1]
                ]
                for holder_offset in configuration.holder_offsets_xyz_m
            )
            system = DynamicCommonBackplateArray(
                configuration,
                tracks,
                unit_origin_xy_m=placement_offset,
                contact=_existing_m1_smoke_contact_settings(),
            )
            candidate_result = DynamicCommonBackplateExperiment(
                system,
                settings,
                _existing_m1_smoke_integrator(),
            ).run()
            candidate_minimum_clearance = None
            candidate_collision = None
            first_collision_path_position_m = None
            rod_clearance_failure_code = None
            if (
                candidate_result.summary.initial_preload_success
                and candidate_result.points
            ):
                try:
                    candidate_clearance = _proxy_array_rod_clearance(
                        library=library,
                        result=candidate_result,
                    )
                except _RodClearanceOutOfBoundsError:
                    candidate_collision = True
                    rod_clearance_failure_code = (
                        "rod_clearance_samples_out_of_bounds"
                    )
                else:
                    candidate_minimum_clearance = float(
                        np.min(candidate_clearance)
                    )
                    candidate_collision = (
                        candidate_minimum_clearance < 0.0
                    )
                    if candidate_collision:
                        first_collision_index = int(
                            np.flatnonzero(
                                np.any(candidate_clearance < 0.0, axis=1)
                            )[0]
                        )
                        first_collision_path_position_m = float(
                            candidate_result.points[
                                first_collision_index
                            ].path_position_m
                        )
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "offset_xy_m": list(placement_offset),
                    "terminal": (
                        candidate_result.summary.run_terminal_state.value
                    ),
                    "initial_preload_success": (
                        candidate_result.summary.initial_preload_success
                    ),
                    "minimum_rod_clearance_m": (
                        candidate_minimum_clearance
                    ),
                    "rod_collision_detected": candidate_collision,
                    "first_collision_path_position_m": (
                        first_collision_path_position_m
                    ),
                    "rod_clearance_failure_code": (
                        rod_clearance_failure_code
                    ),
                    "elapsed_s": time.perf_counter() - attempt_started,
                    "selected": False,
                }
            )
            candidate = (
                candidate_result,
                candidate_minimum_clearance,
                candidate_collision,
                attempt_index,
                placement_offset,
            )
            if selected is None:
                selected = candidate
            if (
                candidate_minimum_clearance is not None
                and candidate_result.summary.run_terminal_state.value
                == "path_end"
                and candidate_result.summary.initial_preload_success
                and candidate_minimum_clearance > best_clearance
            ):
                best_clearance = candidate_minimum_clearance
                selected = candidate
            if (
                candidate_result.summary.run_terminal_state.value
                == "path_end"
                and candidate_result.summary.initial_preload_success
                and candidate_collision is False
            ):
                selected = candidate
                break
            retryable = (
                candidate_collision is True
                or candidate_result.summary.initialization_failure_category
                == "geometry_out_of_bounds"
                or candidate_result.summary.run_terminal_state.value
                == "terrain_bounds"
            )
            if retryable:
                geometry_retry_triggered = True
                continue
            if attempt_index == 0 or not geometry_retry_triggered:
                selected = candidate
                break
        assert selected is not None
        (
            result,
            minimum_rod_clearance_m,
            rod_collision_detected,
            selected_attempt_index,
            selected_offset_xy_m,
        ) = selected
        attempts[selected_attempt_index]["selected"] = True
        yield_ok = (
            result.summary.yield_violation_pin_step_count == 0
            and result.summary.maximum_bending_stress_pa is not None
            and result.summary.maximum_bending_stress_pa
            <= proxy_scenario.yield_strength_pa
        )
        buckling_ok = (
            result.summary.buckling_violation_pin_step_count == 0
            and result.summary.minimum_euler_buckling_margin_n is not None
            and result.summary.minimum_euler_buckling_margin_n >= 0.0
        )
        numerical_flow_passed = bool(
            result.summary.run_terminal_state.value == "path_end"
            and result.summary.initial_preload_success
            and result.summary.maximum_abs_dynamic_residual_n is not None
            and result.summary.maximum_abs_dynamic_residual_n
            <= settings.dynamic_residual_tolerance_n
            and result.summary.maximum_abs_energy_residual_j is not None
            and math.isfinite(result.summary.maximum_abs_energy_residual_j)
            and result.summary.cumulative_relative_energy_error is not None
            and result.summary.cumulative_relative_energy_error <= 1e-3
            and result.summary.maximum_abs_contact_work_identity_residual_j
            is not None
            and result.summary.maximum_abs_contact_work_identity_residual_j
            <= 1e-12
            and result.summary.maximum_force_aggregation_residual_n
            is not None
            and result.summary.maximum_force_aggregation_residual_n <= 1e-12
            and result.summary.maximum_moment_aggregation_residual_nm
            is not None
            and result.summary.maximum_moment_aggregation_residual_nm <= 1e-15
        )
        constraints_ok = bool(
            rod_collision_detected is False and yield_ok and buckling_ok
        )
        rows.append(
            {
                "size": f"{size}x{size}",
                "pin_count": configuration.pin_count,
                "configuration_id": configuration.configuration_id,
                "terminal": result.summary.run_terminal_state.value,
                "initial_preload_success": (
                    result.summary.initial_preload_success
                ),
                "initialization_failure_category": (
                    result.summary.initialization_failure_category
                ),
                "initialization_failure_code": (
                    result.summary.initialization_failure_code
                ),
                "termination_reason": result.summary.termination_reason,
                "numerical_state": result.summary.numerical_state.value,
                "model_state": result.summary.model_state.value,
                "accepted_steps": result.summary.accepted_steps,
                "rejected_steps": result.summary.rejected_steps,
                "dynamic_residual_tolerance_n": (
                    settings.dynamic_residual_tolerance_n
                ),
                "settlement_steps": result.summary.settlement_steps,
                "settlement_reaction_error_n": (
                    result.summary.settlement_final_reaction_error_n
                ),
                "settlement_maximum_mode_speed_m_s": (
                    result.summary.settlement_final_maximum_mode_speed_m_s
                ),
                "settlement_dynamic_residual_n": (
                    result.summary.settlement_final_dynamic_residual_n
                ),
                "settlement_stable_steps": (
                    result.summary.settlement_stable_steps
                ),
                "settlement_required_stable_steps": (
                    result.summary.settlement_required_stable_steps
                ),
                "tangential_force_median_n": (
                    result.summary.tangential_force_median_n
                ),
                "tangential_force_p10_n": (
                    result.summary.tangential_force_p10_n
                ),
                "contact_fraction": result.summary.contact_fraction,
                "neff_normal_median": result.summary.neff_normal_median,
                "neff_resultant_median": (
                    result.summary.neff_resultant_median
                ),
                "maximum_pin_normal_force_n": (
                    result.summary.maximum_pin_normal_force_n
                ),
                "maximum_bending_stress_pa": (
                    result.summary.maximum_bending_stress_pa
                ),
                "minimum_euler_buckling_margin_n": (
                    result.summary.minimum_euler_buckling_margin_n
                ),
                "yield_ok": yield_ok,
                "buckling_ok": buckling_ok,
                "minimum_rod_clearance_m": minimum_rod_clearance_m,
                "rod_collision_detected": rod_collision_detected,
                "rod_clearance_failure_code": attempts[
                    selected_attempt_index
                ]["rod_clearance_failure_code"],
                "placement_search_enabled": placement_search,
                "placement_attempt_count": len(attempts),
                "placement_attempts": attempts,
                "selected_placement_attempt_index": (
                    selected_attempt_index
                ),
                "selected_offset_xy_m": list(selected_offset_xy_m),
                "placement_relocated": selected_attempt_index != 0,
                "elapsed_s": time.perf_counter() - size_started,
                "constraints_ok": constraints_ok,
                "maximum_abs_dynamic_residual_n": (
                    result.summary.maximum_abs_dynamic_residual_n
                ),
                "maximum_abs_energy_residual_j": (
                    result.summary.maximum_abs_energy_residual_j
                ),
                "maximum_relative_energy_residual": (
                    result.summary.maximum_relative_energy_residual
                ),
                "cumulative_relative_energy_error": (
                    result.summary.cumulative_relative_energy_error
                ),
                "cumulative_implicit_euler_dissipation_j": (
                    result.summary.cumulative_implicit_euler_dissipation_j
                ),
                "cumulative_contact_energy_injection_j": (
                    result.summary.cumulative_contact_energy_injection_j
                ),
                "maximum_abs_contact_work_identity_residual_j": (
                    result.summary.maximum_abs_contact_work_identity_residual_j
                ),
                "maximum_force_aggregation_residual_n": (
                    result.summary.maximum_force_aggregation_residual_n
                ),
                "maximum_moment_aggregation_residual_nm": (
                    result.summary.maximum_moment_aggregation_residual_nm
                ),
                "numerical_flow_passed": numerical_flow_passed,
                "ranking_inclusion_allowed": bool(
                    numerical_flow_passed and constraints_ok
                ),
                "passed": numerical_flow_passed,
                "formal_ranking_eligible": False,
            }
        )
    report = {
        "generated_at_utc": utc_now(),
        "m3_module_version": M3_MODULE_VERSION,
        "model_level": M3_MODEL_LEVEL,
        "scope": "one_existing_M1_condition_short_path_interface_smoke",
        "catalog_path": str(catalog_path.resolve()),
        "terrain_catalog_id": catalog.get("terrain_catalog_id"),
        "terrain_condition_id": condition["terrain_condition_id"],
        "condition_name": condition.get("name"),
        "terrain_family": condition["terrain_family"],
        "subtype": condition.get("subtype"),
        "terrain_recipe_id": condition["terrain_recipe_id"],
        "recipe_hash": recipe.recipe_hash,
        "region_id": condition["region_id"],
        "terrain_data_sha256": condition["data_sha256"],
        "seed": int(condition["seed"]),
        "drag_length_m": drag_length_m,
        "external_total_preload_n": 1.0,
        "elapsed_s": time.perf_counter() - condition_started,
        "identity_checks": identity_checks,
        "engineering_proxy_scenario": proxy_scenario.as_dict(),
        "contact": {
            "position_correction": (
                proxy_scenario.contact_position_correction
            ),
            "projection_iterations": 20,
        },
        "placement_search_enabled": placement_search,
        "placement_search_offsets_xy_m": [
            list(offset) for offset in placement_offsets
        ],
        "sizes": rows,
        "all_numerical_flows_passed": all(
            row["numerical_flow_passed"] for row in rows
        ),
        "all_physical_constraints_passed": all(
            row["constraints_ok"] for row in rows
        ),
        "constraint_excluded_sizes": [
            row["size"] for row in rows if not row["constraints_ok"]
        ],
        "all_passed": all(row["passed"] for row in rows),
        "formal_ranking_eligible": False,
        "limitations": [
            "the selected existing M1 condition is an interface-smoke input",
            "short smoke path is not the required 100 mm trend experiment",
            "this is not the planned 3-family x 100-seed paired catalog",
            "large-array contact-stabilization injection convergence is open",
        ],
    }
    return report


def run_existing_m1_terrain_smoke(
    catalog_path: Path,
    *,
    output_path: Path | None = None,
    drag_length_m: float = 0.1e-3,
    seed: int | None = None,
    terrain_family: str | None = None,
    condition_name: str | None = None,
    verify_data_hash: bool = False,
    placement_search: bool = False,
) -> dict[str, Any]:
    """Run one explicitly selected current M1 realization on three arrays."""

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    condition = _select_existing_m1_condition(
        catalog,
        seed=seed,
        terrain_family=terrain_family,
        condition_name=condition_name,
    )
    report = _run_existing_m1_condition_smoke(
        catalog_path,
        catalog,
        condition,
        drag_length_m=drag_length_m,
        verify_data_hash=verify_data_hash,
        placement_search=placement_search,
    )
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report


def run_existing_m1_catalog_smoke(
    catalog_path: Path,
    *,
    output_path: Path | None = None,
    drag_length_m: float = 0.1e-3,
    verify_data_hash: bool = False,
    placement_search: bool = False,
) -> dict[str, Any]:
    """Run every verified condition in a bounded M1 catalog exactly once."""

    catalog_started = time.perf_counter()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    conditions = validate_terrain_catalog(
        catalog,
        require_formal_300=False,
    )
    condition_reports = [
        _run_existing_m1_condition_smoke(
            catalog_path,
            catalog,
            condition,
            drag_length_m=drag_length_m,
            verify_data_hash=verify_data_hash,
            placement_search=placement_search,
        )
        for condition in conditions
    ]
    report = {
        "generated_at_utc": utc_now(),
        "m3_module_version": M3_MODULE_VERSION,
        "model_level": M3_MODEL_LEVEL,
        "scope": "all_existing_M1_conditions_short_path_interface_smoke",
        "catalog_path": str(catalog_path.resolve()),
        "terrain_catalog_id": catalog.get("terrain_catalog_id"),
        "condition_count": len(condition_reports),
        "passed_condition_count": sum(
            bool(item["all_passed"]) for item in condition_reports
        ),
        "failed_condition_names": [
            item["condition_name"]
            for item in condition_reports
            if not item["all_passed"]
        ],
        "constraint_excluded_condition_names": [
            item["condition_name"]
            for item in condition_reports
            if not item["all_physical_constraints_passed"]
        ],
        "drag_length_m": drag_length_m,
        "external_total_preload_n": 1.0,
        "elapsed_s": time.perf_counter() - catalog_started,
        "data_files_sha256_recomputed": bool(verify_data_hash),
        "placement_search_enabled": placement_search,
        "conditions": condition_reports,
        "all_passed": all(
            bool(item["all_passed"]) for item in condition_reports
        ),
        "formal_ranking_eligible": False,
        "formal_campaign_started": False,
        "limitations": [
            "this bounded catalog smoke is not the 3-family x 100-seed campaign",
            "the short path does not establish 100 mm trend convergence",
            "material-hybrid 10/5 um same-realization convergence remains an M1 blocker",
            "engineering proxy parameters are not experimental calibration",
        ],
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report
