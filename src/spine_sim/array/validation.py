"""Legacy m3.0.0 fixed-Z validation retained as migration evidence only."""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.contact import AxialMode, SpineParameters
from spine_sim.contact.validation import _fixture_parameters, _fixture_track
from spine_sim.io.results import atomic_write_json, utc_now
from spine_sim.terrain import TerrainLibrary, TrackGeometry

from .experiment import (
    LegacyArrayExperimentSettings as ArrayExperimentSettings,
    LegacyFixedZCommonBackplateExperiment as CommonBackplateExperiment,
)
from .models import (
    M3_LEGACY_MODULE_VERSION as M3_MODULE_VERSION,
    AngleLayout,
    ArrayConfiguration,
)
from .solver import LegacyCommonBackplateArray as CommonBackplateArray


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


def _plane_track(
    *,
    y_m: float,
    height_m: float = 0.0,
    x_min_m: float = -0.060,
    x_max_m: float = 0.070,
    dx_m: float = 10e-6,
    recipe_id: str = "terrain_recipe_m3_plane_fixture",
    region_id: str = "region_m3_plane_fixture",
    radius_m: float = 50e-6,
) -> TrackGeometry:
    x = np.arange(x_min_m, x_max_m + 0.5 * dx_m, dx_m, dtype=np.float64)
    valid = np.ones(x.size, dtype=np.bool_)
    valid[[0, -1]] = False
    track_id = TrackGeometry.make_id(
        terrain_recipe_id=recipe_id,
        region_id=region_id,
        radius_m=radius_m,
        y_global_m=y_m,
        envelope_algorithm_version="m3-analytic-plane-v1",
        resolution_m=dx_m,
    )
    return TrackGeometry(
        terrain_recipe_id=recipe_id,
        region_id=region_id,
        track_id=track_id,
        radius_m=radius_m,
        y_global_m=y_m,
        resolution_m=dx_m,
        envelope_algorithm_version="m3-analytic-plane-v1",
        x_global_m=x,
        envelope_height_m=np.full(x.size, height_m + radius_m),
        envelope_slope_x=np.zeros(x.size),
        support_x_m=x.copy(),
        support_y_m=np.full(x.size, y_m),
        valid_mask=valid,
        near_tie_flag=np.zeros(x.size, dtype=np.bool_),
        model_warning=(),
    )


def _tracks_for_configuration(
    configuration: ArrayConfiguration,
    *,
    height_by_pin_m: tuple[float, ...] | None = None,
) -> tuple[TrackGeometry, ...]:
    heights = height_by_pin_m or (0.0,) * configuration.pin_count
    return tuple(
        _plane_track(y_m=offset[1], height_m=height)
        for offset, height in zip(configuration.holder_offsets_xyz_m, heights)
    )


def _run_plane(
    configuration: ArrayConfiguration,
    *,
    height_by_pin_m: tuple[float, ...] | None = None,
    target_preload_n: float = 1.0,
    drag_length_m: float = 0.2e-3,
    path_step_m: float = 50e-6,
):
    system = CommonBackplateArray(
        configuration,
        _tracks_for_configuration(
            configuration,
            height_by_pin_m=height_by_pin_m,
        ),
        unit_origin_xy_m=(0.0, 0.0),
    )
    return CommonBackplateExperiment(
        system,
        ArrayExperimentSettings(
            drag_length_m=drag_length_m,
            path_step_m=path_step_m,
            target_preload_n=target_preload_n,
        ),
    ).run()


def _shift_fixture_track(
    track: TrackGeometry,
    *,
    y_m: float,
    recipe_id: str,
    region_id: str,
) -> TrackGeometry:
    return replace(
        track,
        terrain_recipe_id=recipe_id,
        region_id=region_id,
        track_id=TrackGeometry.make_id(
            terrain_recipe_id=recipe_id,
            region_id=region_id,
            radius_m=track.radius_m,
            y_global_m=y_m,
            envelope_algorithm_version=track.envelope_algorithm_version,
            resolution_m=track.resolution_m,
        ),
        y_global_m=y_m,
        support_y_m=np.full_like(track.support_y_m, y_m),
    )


def run_legacy_analytic_validation(
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    fixture_parameters = _fixture_parameters()

    two = ArrayConfiguration(
        2, 1, 4e-3, fixture_parameters, fixture_only=True
    )
    equal = _run_plane(two, target_preload_n=0.5)
    preload_forces = np.asarray(
        [
            response.normal_force_n
            for response in equal.points[1].response.pin_responses
        ]
    )
    gates.append(
        _gate(
            "two_pin_equal_height_equal_load",
            equal.summary.initial_preload_success
            and np.ptp(preload_forces) <= 1e-10,
            {
                "preload_pin_normal_force_n": preload_forces.tolist(),
                "neff_normal": equal.points[1].response.sharing.neff_normal,
            },
            requirement="equal-height, equal-parameter pins share preload equally",
        )
    )

    height_difference = (50e-6, 0.0)
    stiff_parameters = replace(
        fixture_parameters,
        spring_stiffness_n_m=10_000.0,
    )
    stiff_two = ArrayConfiguration(
        2, 1, 4e-3, stiff_parameters, fixture_only=True
    )
    spring_result = _run_plane(
        two,
        height_by_pin_m=height_difference,
        target_preload_n=0.5,
    )
    early_result = _run_plane(
        two,
        height_by_pin_m=height_difference,
        target_preload_n=0.05,
    )
    stiff_result = _run_plane(
        stiff_two,
        height_by_pin_m=height_difference,
        target_preload_n=0.5,
    )
    spring_low = spring_result.points[1].response.pin_responses[1].normal_force_n
    stiff_low = stiff_result.points[1].response.pin_responses[1].normal_force_n
    gates.append(
        _gate(
            "height_order_and_compliance",
            early_result.points[1].response.pin_responses[0].normal_force_n > 0.0
            and early_result.points[1].response.pin_responses[1].normal_force_n
            <= 1e-12
            and spring_low >= stiff_low - 1e-8
            and spring_result.points[1].response.sharing.neff_normal
            >= stiff_result.points[1].response.sharing.neff_normal - 1e-8,
            {
                "spring_low_pin_force_n": spring_low,
                "stiff_low_pin_force_n": stiff_low,
                "early_high_pin_force_n": (
                    early_result.points[1].response.pin_responses[0].normal_force_n
                ),
                "early_low_pin_force_n": (
                    early_result.points[1].response.pin_responses[1].normal_force_n
                ),
                "spring_neff_normal": (
                    spring_result.points[1].response.sharing.neff_normal
                ),
                "stiff_neff_normal": (
                    stiff_result.points[1].response.sharing.neff_normal
                ),
            },
            requirement=(
                "the high point engages first and added axial compliance does not "
                "reduce low-point participation at equal total preload"
            ),
        )
    )

    square = ArrayConfiguration(2, 2, 5e-3, fixture_parameters)
    square_result = _run_plane(square)
    square_forces = np.asarray(
        [
            response.normal_force_n
            for response in square_result.points[1].response.pin_responses
        ]
    )
    gates.append(
        _gate(
            "two_by_two_symmetry",
            square_result.summary.initial_preload_success
            and np.ptp(square_forces) <= 1e-10
            and square_result.points[1].response.sharing.neff_normal > 3.999999,
            {
                "pin_normal_force_n": square_forces.tolist(),
                "unit_wrench": list(
                    square_result.points[1].response.wall_on_unit_wrench_about_origin
                ),
            },
            requirement="a symmetric 2x2 plane fixture has equal load and zero imbalance",
        )
    )

    result_2x5 = _run_plane(
        ArrayConfiguration(2, 5, 5e-3, fixture_parameters),
        drag_length_m=0.1e-3,
    )
    result_5x2 = _run_plane(
        ArrayConfiguration(5, 2, 5e-3, fixture_parameters),
        drag_length_m=0.1e-3,
    )
    wrench_2x5 = np.asarray(
        result_2x5.points[1].response.wall_on_unit_wrench_about_origin
    )
    wrench_5x2 = np.asarray(
        result_5x2.points[1].response.wall_on_unit_wrench_about_origin
    )
    gates.append(
        _gate(
            "transpose_relation",
            result_2x5.configuration.configuration_id
            != result_5x2.configuration.configuration_id
            and np.allclose(wrench_2x5, wrench_5x2, rtol=0.0, atol=1e-10),
            {
                "configuration_2x5": result_2x5.configuration.configuration_id,
                "configuration_5x2": result_5x2.configuration.configuration_id,
                "preload_wrench_2x5": wrench_2x5.tolist(),
                "preload_wrench_5x2": wrench_5x2.tolist(),
            },
            requirement="2x5 and 5x2 remain distinct and obey plane-fixture transpose symmetry",
        )
    )

    system = CommonBackplateArray(
        square,
        _tracks_for_configuration(square),
        unit_origin_xy_m=(0.0, 0.0),
    )
    preload_state = square_result.points[1].response.next_state
    fixed_uz = square_result.fixed_common_uz_m
    assert fixed_uz is not None
    forward = system.solve_pose((25e-6, fixed_uz), preload_state)
    reverse = system.solve_pose(
        (25e-6, fixed_uz),
        preload_state,
        traversal_order=tuple(reversed(range(square.pin_count))),
    )
    shuffled_order = list(range(square.pin_count))
    random.Random(31001).shuffle(shuffled_order)
    shuffled = system.solve_pose(
        (25e-6, fixed_uz),
        preload_state,
        traversal_order=shuffled_order,
    )
    traversal_equal = (
        forward.proposal_valid == reverse.proposal_valid == shuffled.proposal_valid
        and forward.proposal_state == reverse.proposal_state == shuffled.proposal_state
        and np.array_equal(
            forward.wall_on_unit_wrench_about_origin,
            reverse.wall_on_unit_wrench_about_origin,
        )
        and np.array_equal(
            forward.wall_on_unit_wrench_about_origin,
            shuffled.wall_on_unit_wrench_about_origin,
        )
    )
    gates.append(
        _gate(
            "atomic_traversal_order",
            traversal_equal,
            {
                "forward_wrench": list(forward.wall_on_unit_wrench_about_origin),
                "reverse_wrench": list(reverse.wall_on_unit_wrench_about_origin),
                "shuffled_order": shuffled_order,
                "old_state_unchanged": preload_state == square_result.points[1].response.next_state,
            },
            requirement="forward, reverse and shuffled proposal traversal are identical",
        )
    )

    bump_parameters = {
        "amplitude_m": 400e-6,
        "center_x_m": -0.001,
        "sigma_x_m": 150e-6,
        "sigma_y_m": 150e-6,
    }
    bump_base = _fixture_track("smooth_bump", bump_parameters)
    plane_base = _fixture_track("plane")
    event_config = ArrayConfiguration(
        2, 1, 4e-3, fixture_parameters, fixture_only=True
    )
    event_tracks = (
        _shift_fixture_track(
            bump_base,
            y_m=event_config.holder_offsets_xyz_m[0][1],
            recipe_id="terrain_recipe_m3_event_fixture",
            region_id="region_m3_event_fixture",
        ),
        _shift_fixture_track(
            plane_base,
            y_m=event_config.holder_offsets_xyz_m[1][1],
            recipe_id="terrain_recipe_m3_event_fixture",
            region_id="region_m3_event_fixture",
        ),
    )
    first_offset = event_config.holder_offsets_xyz_m[0][0]
    first_parameters = event_config.pin_parameters[0]
    origin_x = (
        -0.001
        - first_offset
        - first_parameters.exposed_length_m
        * math.cos(math.radians(first_parameters.installation_angle_deg))
    )
    event_system = CommonBackplateArray(
        event_config,
        event_tracks,
        unit_origin_xy_m=(origin_x, 0.0),
    )
    event_result = CommonBackplateExperiment(
        event_system,
        ArrayExperimentSettings(
            drag_length_m=1e-3,
            path_step_m=20e-6,
            target_preload_n=0.5,
        ),
    ).run()
    detach_points = [
        point
        for point in event_result.points
        if any(label == "detach_to_free" for _index, label in point.response.event_labels)
    ]
    non_event_pin_recomputed = False
    if detach_points:
        detach = detach_points[0]
        non_event_pin_recomputed = math.isclose(
            detach.response.pin_responses[1].holder_xz_m[0],
            event_system.pin_holder_xz_m(
                1,
                detach.path_position_m,
                detach.response.common_uz_m,
            )[0],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    gates.append(
        _gate(
            "event_full_array_reevaluation",
            bool(detach_points)
            and non_event_pin_recomputed
            and (
                len(
                    {
                        state.accepted_steps
                        for state in detach_points[0].response.next_state.pin_states
                    }
                )
                == 1
            )
            and event_result.summary.run_terminal_state.value == "path_end",
            {
                "event_counts": dict(event_result.summary.event_counts),
                "detach_path_position_m": (
                    detach_points[0].path_position_m if detach_points else None
                ),
                "non_event_pin_recomputed_at_same_pose": non_event_pin_recomputed,
                "all_pin_commit_counts": (
                    [
                        state.accepted_steps
                        for state in detach_points[0].response.next_state.pin_states
                    ]
                    if detach_points
                    else []
                ),
            },
            requirement="a local detach commits an all-pin response at the same backplate pose",
        )
    )

    def direct_total_force(parameters: SpineParameters) -> float:
        configuration = ArrayConfiguration(
            2, 1, 4e-3, parameters, fixture_only=True
        )
        local_system = CommonBackplateArray(
            configuration,
            _tracks_for_configuration(configuration),
            unit_origin_xy_m=(0.0, 0.0),
        )
        free = CommonBackplateExperiment(
            local_system,
            ArrayExperimentSettings(drag_length_m=0.1e-3, path_step_m=50e-6),
        )._initial_free_pose()
        response = local_system.solve_pose(
            (0.0, free.common_uz_m - 25e-6),
            free.next_state,
        )
        return response.total_normal_force_n

    rigid_parameters = replace(
        fixture_parameters,
        axial_mode=AxialMode.RIGID,
        spring_stiffness_n_m=None,
    )
    rigid_force = direct_total_force(rigid_parameters)
    limit_forces = [
        direct_total_force(
            replace(
                fixture_parameters,
                axial_mode=AxialMode.SPRING,
                spring_stiffness_n_m=stiffness,
            )
        )
        for stiffness in (1e5, 1e7, 1e9)
    ]
    limit_errors = [abs(value - rigid_force) for value in limit_forces]
    gates.append(
        _gate(
            "rigid_axial_limit",
            limit_errors[2] < limit_errors[1] < limit_errors[0],
            {
                "rigid_total_normal_force_n": rigid_force,
                "spring_total_normal_force_n": limit_forces,
                "absolute_errors_n": limit_errors,
            },
            requirement="the array aggregation preserves M2's ks-to-infinity rigid limit",
        )
    )

    gates.append(
        _gate(
            "wrench_and_residual",
            square_result.summary.maximum_force_aggregation_residual_n <= 1e-15
            and square_result.summary.maximum_moment_aggregation_residual_nm <= 1e-18
            and square_result.summary.maximum_abs_local_geometry_residual_m <= 2.1e-9,
            {
                "force_aggregation_residual_n": (
                    square_result.summary.maximum_force_aggregation_residual_n
                ),
                "moment_aggregation_residual_nm": (
                    square_result.summary.maximum_moment_aggregation_residual_nm
                ),
                "local_geometry_residual_m": (
                    square_result.summary.maximum_abs_local_geometry_residual_m
                ),
            },
            requirement="force/moment aggregation identity and local residual bounds",
        )
    )

    common_z_exact = all(
        len({holder[2] for holder in point.response.pin_holder_xyz_m}) == 1
        for point in square_result.points
    )
    gates.append(
        _gate(
            "common_pose_no_independent_normal_roots",
            common_z_exact,
            {
                "all_pin_holder_z_exactly_common": common_z_exact,
                "preload_search_count": 1,
                "drag_z_frozen": square_result.fixed_common_uz_m,
            },
            requirement="M3 exposes one common uZ and never a per-pin normal root",
        )
    )

    coarse = _run_plane(
        square,
        drag_length_m=0.4e-3,
        path_step_m=50e-6,
    )
    fine = _run_plane(
        square,
        drag_length_m=0.4e-3,
        path_step_m=25e-6,
    )
    gates.append(
        _gate(
            "path_step_halving",
            tuple(coarse.summary.event_counts) == tuple(fine.summary.event_counts)
            and abs(
                coarse.summary.tangential_force_median_n
                - fine.summary.tangential_force_median_n
            )
            <= 0.03,
            {
                "coarse_event_counts": dict(coarse.summary.event_counts),
                "fine_event_counts": dict(fine.summary.event_counts),
                "coarse_tangential_median_n": coarse.summary.tangential_force_median_n,
                "fine_tangential_median_n": fine.summary.tangential_force_median_n,
            },
            requirement="halving the prescribed path step preserves event order and robust metrics",
        )
    )

    out_of_bounds = system.solve_pose(
        (1.0, fixed_uz),
        preload_state,
        commit=False,
    )
    gates.append(
        _gate(
            "terrain_bounds_not_silent",
            not out_of_bounds.proposal_valid
            and "geometry_out_of_domain" in out_of_bounds.residual.termination_reason
            and out_of_bounds.proposal_state == preload_state
            and out_of_bounds.next_state == preload_state,
            {
                "proposal_valid": out_of_bounds.proposal_valid,
                "termination_reason": out_of_bounds.residual.termination_reason,
                "failed_proposal_preserved_old_state": (
                    out_of_bounds.proposal_state == preload_state
                    and out_of_bounds.next_state == preload_state
                ),
            },
            requirement="array extent/path overflow is explicit rather than silently truncated",
        )
    )

    for layout, final_angle in (
        (AngleLayout.GRADIENT_80_TO_60, 60.0),
        (AngleLayout.GRADIENT_80_TO_50, 50.0),
    ):
        gradient = ArrayConfiguration(
            5,
            2,
            5e-3,
            fixture_parameters,
            angle_layout=layout,
        )
        vertical_reach = np.asarray(
            [
                parameters.exposed_length_m
                * math.sin(math.radians(parameters.installation_angle_deg))
                for parameters in gradient.pin_parameters[: gradient.nx]
            ]
        )
        gates.append(
            _gate(
                layout.value,
                math.isclose(
                    gradient.column_angles_deg[-1],
                    final_angle,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and np.ptp(vertical_reach) <= 1e-15,
                {
                    "column_angles_deg": list(gradient.column_angles_deg),
                    "vertical_reach_m": vertical_reach.tolist(),
                },
                requirement="gradient angles run along x and preserve unloaded tip height",
            )
        )

    large = _run_plane(
        ArrayConfiguration(6, 6, 6e-3, fixture_parameters),
        target_preload_n=0.1,
        drag_length_m=0.05e-3,
        path_step_m=50e-6,
    )
    gates.append(
        _gate(
            "large_array",
            large.summary.initial_preload_success
            and large.points[1].response.sharing.neff_normal > 35.999
            and large.summary.run_terminal_state.value == "path_end",
            {
                "pin_count": large.configuration.pin_count,
                "preload_neff_normal": large.points[1].response.sharing.neff_normal,
                "terminal": large.summary.run_terminal_state.value,
            },
            requirement="the reference implementation handles a 6x6 array without special cases",
        )
    )

    report = {
        "schema_version": "1",
        "m3_module_version": M3_MODULE_VERSION,
        "created_at_utc": utc_now(),
        "gate_count": len(gates),
        "passed_count": sum(gate["passed"] for gate in gates),
        "all_passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "formal_m3_round1_allowed": False,
        "formal_m3_round1_blockers": [
            "full_chain_frozen_manifest.json does not exist",
            "M2 formal round one has not run",
            "M2 selected parameter pack proposal is unapproved and empty",
            "explicit approval '开始 M3 第一轮筛选' has not been recorded",
        ],
    }
    if output_path is not None:
        atomic_write_json(Path(output_path), report)
    return report


def _valid_origin_interval(
    system: CommonBackplateArray,
    *,
    drag_length_m: float,
) -> tuple[float, float]:
    lower = -math.inf
    upper = math.inf
    for core, parameters, offset in zip(
        system.cores,
        system.pin_parameters,
        system.holder_offsets_xyz_m,
    ):
        valid_min, valid_max = core.geometry.valid_x_range_m
        axis_shift = parameters.exposed_length_m * core._a[0]
        lower = max(lower, valid_min - offset[0] - axis_shift)
        upper = min(
            upper,
            valid_max - offset[0] - axis_shift - drag_length_m,
        )
    if lower >= upper:
        raise ValueError("array and drag path do not fit the common M1 track domain")
    return lower, upper


def run_legacy_m1_suite_smoke(
    suite_report_path: str | Path,
    *,
    drag_length_m: float = 1e-3,
    path_step_m: float = 50e-6,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one fixed 2x2 Stage-I M3 fixture on all ten existing terrain conditions."""

    report_path = Path(suite_report_path)
    source = json.loads(report_path.read_text(encoding="utf-8"))
    library_root = report_path.parent / "terrain_library"
    library = TerrainLibrary(library_root)
    parameters = replace(
        _fixture_parameters(),
        rod_clearance_mode="unclosed",
    )
    configuration = ArrayConfiguration(2, 2, 5e-3, parameters)
    origin_y = 0.0
    conditions: list[dict[str, Any]] = []
    for condition in source["conditions"]:
        recipe_id = condition["terrain_recipe_id"]
        region_id = condition["region_id"]
        recipe = library.load_recipe(recipe_id)
        region = library.load_region_spec(recipe_id, region_id)
        track_by_y: dict[float, TrackGeometry] = {}
        for offset in configuration.holder_offsets_xyz_m:
            y_m = origin_y + offset[1]
            if y_m not in track_by_y:
                track_by_y[y_m] = library.cache_track(
                    recipe,
                    region,
                    radius_m=parameters.tip_radius_m,
                    y_global_m=y_m,
                )
        tracks = tuple(
            track_by_y[origin_y + offset[1]]
            for offset in configuration.holder_offsets_xyz_m
        )
        template = CommonBackplateArray(
            configuration,
            tracks,
            unit_origin_xy_m=(0.0, origin_y),
        )
        lower, upper = _valid_origin_interval(
            template,
            drag_length_m=drag_length_m,
        )
        attempt_fractions = (
            0.50,
            0.60,
            0.40,
            0.70,
            0.30,
            0.55,
            0.65,
            0.80,
            0.20,
        )
        attempts: list[dict[str, Any]] = []
        result = None
        selected_origin_x = math.nan
        traversal_invariant = False
        for fraction in attempt_fractions:
            origin_x = lower + fraction * (upper - lower)
            system = CommonBackplateArray(
                configuration,
                tracks,
                unit_origin_xy_m=(origin_x, origin_y),
            )
            candidate = CommonBackplateExperiment(
                system,
                ArrayExperimentSettings(
                    drag_length_m=drag_length_m,
                    path_step_m=path_step_m,
                    target_preload_n=1.0,
                ),
            ).run()
            attempts.append(
                {
                    "valid_common_origin_fraction": fraction,
                    "unit_origin_x_m": origin_x,
                    "preload_success": candidate.summary.initial_preload_success,
                    "terminal": candidate.summary.run_terminal_state.value,
                    "termination_reason": candidate.summary.termination_reason,
                }
            )
            result = candidate
            selected_origin_x = origin_x
            if (
                candidate.summary.initial_preload_success
                and candidate.summary.numerical_state.value == "converged"
                and candidate.summary.run_terminal_state.value == "path_end"
            ):
                preload = candidate.points[1].response
                check_ux = min(10e-6, drag_length_m)
                forward = system.solve_pose(
                    (check_ux, preload.common_uz_m),
                    preload.next_state,
                )
                reverse = system.solve_pose(
                    (check_ux, preload.common_uz_m),
                    preload.next_state,
                    traversal_order=tuple(
                        reversed(range(configuration.pin_count))
                    ),
                )
                traversal_invariant = (
                    forward.proposal_valid == reverse.proposal_valid
                    and forward.proposal_state == reverse.proposal_state
                    and np.array_equal(
                        forward.wall_on_unit_wrench_about_origin,
                        reverse.wall_on_unit_wrench_about_origin,
                    )
                )
                if traversal_invariant:
                    break
        assert result is not None
        passed = (
            result.summary.initial_preload_success
            and result.summary.numerical_state.value == "converged"
            and result.summary.run_terminal_state.value == "path_end"
            and traversal_invariant
            and result.summary.maximum_force_aggregation_residual_n <= 1e-12
            and result.summary.maximum_moment_aggregation_residual_nm <= 1e-15
        )
        conditions.append(
            {
                "name": condition["name"],
                "description": condition["description"],
                "terrain_recipe_id": recipe_id,
                "region_id": region_id,
                "configuration_id": configuration.configuration_id,
                "track_ids": [track.track_id for track in tracks],
                "track_y_global_m": [track.y_global_m for track in tracks],
                "unit_origin_x_m": selected_origin_x,
                "start_search_attempts": attempts,
                "passed": passed,
                "initial_preload_success": result.summary.initial_preload_success,
                "run_terminal_state": result.summary.run_terminal_state.value,
                "termination_reason": result.summary.termination_reason,
                "numerical_state": result.summary.numerical_state.value,
                "model_state": result.summary.model_state.value,
                "point_count": len(result.points),
                "event_counts": dict(result.summary.event_counts),
                "normal_force_range_n": list(
                    result.summary.total_normal_force_range_n
                ),
                "neff_normal_median": result.summary.neff_normal_median,
                "neff_resultant_median": result.summary.neff_resultant_median,
                "maximum_resultant_load_concentration": (
                    result.summary.maximum_resultant_load_concentration
                ),
                "maximum_abs_local_geometry_residual_m": (
                    result.summary.maximum_abs_local_geometry_residual_m
                ),
                "force_aggregation_residual_n": (
                    result.summary.maximum_force_aggregation_residual_n
                ),
                "moment_aggregation_residual_nm": (
                    result.summary.maximum_moment_aggregation_residual_nm
                ),
                "traversal_order_invariant": traversal_invariant,
                "formal_ranking_eligible": False,
            }
        )
    output = {
        "schema_version": "1",
        "m3_module_version": M3_MODULE_VERSION,
        "created_at_utc": utc_now(),
        "source_suite": str(report_path),
        "condition_count": len(conditions),
        "passed_count": sum(item["passed"] for item in conditions),
        "all_passed": all(item["passed"] for item in conditions),
        "purpose": "m3_stage_i_common_backplate_interface_state_and_field_smoke",
        "tested_configuration": configuration.as_dict(),
        "tested_preload_n": 1.0,
        "tested_drag_length_m": drag_length_m,
        "formal_ranking_allowed": False,
        "formal_ranking_blockers": [
            "random-terrain rod/cone clearance remains parameter_unclosed",
            "formal M2 parameter packs do not exist",
            "full-chain frozen manifest and M3 round-one approval are absent",
            "one deterministic start per condition is an interface smoke, not paired-seed screening",
        ],
        "conditions": conditions,
    }
    if output_path is not None:
        atomic_write_json(Path(output_path), output)
    return output
