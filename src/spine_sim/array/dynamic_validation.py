"""Analytic acceptance gates for m3.1.0 joint common-backplate dynamics."""

from __future__ import annotations

import math
import random
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
from spine_sim.terrain import TrackGeometry

from .dynamics import (
    ArrayDynamicExperimentSettings,
    DynamicCommonBackplateArray,
    DynamicCommonBackplateExperiment,
)
from .models import M3_MODEL_LEVEL, M3_MODULE_VERSION, ArrayConfiguration


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
            projection_iterations=6,
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
            and plane.summary.maximum_abs_energy_residual_j < 1e-5,
            {
                "maximum_abs_dynamic_residual_n": (
                    plane.summary.maximum_abs_dynamic_residual_n
                ),
                "maximum_abs_energy_residual_j": (
                    plane.summary.maximum_abs_energy_residual_j
                ),
            },
            requirement="joint equation and discrete energy residuals are audited",
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
            "M2 approved parameter packs are absent",
            "full_chain_frozen_manifest.json is absent",
            "explicit approval '开始 M3 第一轮筛选' is absent",
            "45 generated terrains are inventory only; the second 30 are not approved",
        ],
        "gates": gates,
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report
