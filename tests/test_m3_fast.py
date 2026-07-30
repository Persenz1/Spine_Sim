from __future__ import annotations

import math

import numpy as np
import spine_sim.m3_fast.solver as solver_module

from spine_sim.m3_fast.campaign import (
    _select_m3b_configurations,
    generate_full_scan_designs,
    generate_m3a_packages,
    generate_m3b_geometries,
)
from spine_sim.m3_fast.model import (
    FREE,
    HARD_STOP,
    INTERIOR,
    LOWER_STOP,
    RIGID,
    SLIDE,
    STICK,
    build_spine_batch,
    evaluate_spines,
    make_contact_state,
    make_model_workspace,
)
from spine_sim.m3_fast.solver import (
    MAX_STATION_EVALUATIONS,
    STATION_NUMERICAL_FAILURE,
    STATION_OK,
    STATION_PRELOAD_UNREACHABLE,
    STATION_RECONTACT_REQUIRED,
    STATION_SUPPORT_LOST,
    STATION_TRACK_INVALID,
    PathSettings,
    PathTrace,
    StationWorkspace,
    simulate_path,
    solve_station,
)
from spine_sim.m3_fast.terrain import TrackBank


def _flat_track_bank(y_values_m: np.ndarray) -> TrackBank:
    dx_m = 1e-4
    x_global_m = np.arange(-0.01, 0.015 + 0.5 * dx_m, dx_m)
    shape = (y_values_m.size, x_global_m.size)
    height_m = np.zeros(shape, dtype=np.float64)
    slope_x = np.zeros(shape, dtype=np.float64)
    arc_length_m = np.broadcast_to(
        x_global_m - x_global_m[0], shape
    ).copy()
    return TrackBank(
        x_global_m=x_global_m,
        y_values_m=y_values_m,
        envelope_height_m=height_m,
        envelope_slope_x=slope_x,
        arc_length_m=arc_length_m,
        valid_mask=np.ones(shape, dtype=np.bool_),
        terrain_id="synthetic-flat-test",
        seed=1,
        radius_m=50e-6,
        resolution_m=dx_m,
        terrain_recipe_id="synthetic-recipe",
        region_id="synthetic-region",
        realization_id="synthetic-realization",
        track_ids=tuple(f"track-{index}" for index in range(y_values_m.size)),
    )


def _test_batch():
    return build_spine_batch(
        2,
        2,
        0.005,
        fixed_angle_deg=70.0,
        tip_radius_m=50e-6,
        diameter_m=0.8e-3,
        spring_stiffness_N_per_m=500.0,
        static_friction=0.0,
        kinetic_friction=0.0,
    )


def test_campaign_design_has_80_packages_and_28_unique_geometries() -> None:
    packages = generate_m3a_packages()
    geometries = generate_m3b_geometries()

    assert len(packages) == len({item.package_id for item in packages}) == 80
    assert (
        len(geometries)
        == len({item.geometry_id for item in geometries})
        == 28
    )


def test_full_scan_has_1344_designs_and_plus_x_angle_gradient() -> None:
    designs = generate_full_scan_designs()

    assert len(designs) == len({item.design_id for item in designs}) == 1344
    assert {
        item.geometry.array_shape for item in designs
    } == {"2x2", "2x5", "5x2", "3x5", "5x3", "4x4", "6x6"}
    gradient = next(
        item
        for item in designs
        if item.geometry.array_shape == "5x2"
        and item.geometry.angle_pattern == "60_to_80"
    )
    batch = build_spine_batch(
        gradient.geometry.nx,
        gradient.geometry.ny,
        gradient.geometry.spacing_m,
        angle_pattern=gradient.geometry.angle_pattern,
        tip_radius_m=gradient.package.tip_radius_m,
        diameter_m=gradient.package.diameter_m,
        spring_stiffness_N_per_m=(
            gradient.package.spring_stiffness_N_per_m
        ),
    )
    columns = np.unique(batch.mount_x_m)
    column_angles = [
        np.rad2deg(
            np.unique(batch.angle_rad[batch.mount_x_m == value]).item()
        )
        for value in columns
    ]
    assert np.allclose(column_angles, np.linspace(60.0, 80.0, 5))


def test_vectorized_model_produces_finite_full_array_trial() -> None:
    batch = _test_batch()
    state = make_contact_state(batch)
    workspace = make_model_workspace(batch)
    state_before = (
        state.mode.copy(),
        state.u_t_history_m.copy(),
        state.spring_branch.copy(),
    )
    backplate_z_m = float(-batch.tip_z_offset_m[0] - 10e-6)

    total_fz_N, total_kz_N_per_m = evaluate_spines(
        batch,
        state,
        workspace,
        backplate_z_m=backplate_z_m,
        envelope_height_m=np.zeros(batch.spine_count),
        envelope_slope_x=np.zeros(batch.spine_count),
        delta_arc_m=np.zeros(batch.spine_count),
        valid_mask=np.ones(batch.spine_count, dtype=np.bool_),
    )

    assert total_fz_N > 0.0
    assert total_kz_N_per_m > 0.0
    for values in (
        workspace.force_x_N,
        workspace.force_z_N,
        workspace.lambda_n_N,
        workspace.tangent_force_N,
        workspace.u_t_history_m,
        workspace.vertical_stiffness_N_per_m,
    ):
        assert values.shape == (batch.spine_count,)
        assert np.all(np.isfinite(values))
    assert set(np.unique(workspace.mode)) <= {FREE, STICK, SLIDE}
    assert set(np.unique(workspace.spring_branch)) <= {
        RIGID,
        INTERIOR,
        LOWER_STOP,
        HARD_STOP,
    }
    assert np.all(workspace.mode != FREE)
    assert np.array_equal(state.mode, state_before[0])
    assert np.array_equal(state.u_t_history_m, state_before[1])
    assert np.array_equal(state.spring_branch, state_before[2])


def test_free_spines_do_not_accumulate_detached_arc_length() -> None:
    batch = _test_batch()
    state = make_contact_state(batch)
    workspace = make_model_workspace(batch)

    evaluate_spines(
        batch,
        state,
        workspace,
        backplate_z_m=float(-batch.tip_z_offset_m[0] - 10e-6),
        envelope_height_m=np.zeros(batch.spine_count),
        envelope_slope_x=np.zeros(batch.spine_count),
        delta_arc_m=np.full(batch.spine_count, 1e-3),
        valid_mask=np.ones(batch.spine_count, dtype=np.bool_),
    )

    assert np.all(workspace.tangent_trial_m == 0.0)


def test_common_backplate_path_is_finite_and_never_uses_dense_solve(
    monkeypatch,
) -> None:
    def reject_dense_solve(*_args, **_kwargs):
        raise AssertionError("M3-fast formal path called np.linalg.solve")

    monkeypatch.setattr(np.linalg, "solve", reject_dense_solve)
    batch = _test_batch()
    bank = _flat_track_bank(np.unique(batch.y_m))
    metrics, diagnostics = simulate_path(
        batch,
        bank,
        bank.rows_for_y(batch.y_m),
        settings := PathSettings(
            preload_N=1.0,
            path_length_m=0.001,
            dx_m=0.0001,
            backplate_travel_m=0.006,
        ),
        trace=(
            trace := PathTrace.allocate(
                batch,
                settings,
                include_spines=True,
            )
        ),
    )

    assert metrics["case_status"] == "complete"
    assert metrics["completion_ratio"] == 1.0
    for name in (
        "Fx_q10",
        "Fx_median",
        "Fx_peak_qs",
        "contact_ratio",
        "Neff_q10",
        "Neff_median",
        "max_load_share_q90",
        "slide_ratio",
        "hard_stop_ratio",
    ):
        assert math.isfinite(metrics[name])
    assert diagnostics["station_count_completed"] == 10
    assert (
        1
        <= diagnostics["max_station_evaluations"]
        <= MAX_STATION_EVALUATIONS
        == 9
    )
    assert trace.path_x_m.shape == (11,)
    assert np.all(trace.accepted)
    assert np.all(np.isfinite(trace.force_x_N))
    assert trace.spine_force_x_N is not None
    assert trace.spine_force_x_N.shape == (11, batch.spine_count)


def test_rejected_station_does_not_commit_candidate_state() -> None:
    batch = _test_batch()
    state = make_contact_state(batch)
    model_workspace = make_model_workspace(batch)
    station_workspace = StationWorkspace.allocate()
    zeros = np.zeros(batch.spine_count)
    valid = np.ones(batch.spine_count, dtype=np.bool_)

    status, accepted_z_m, evaluations, _ = solve_station(
        batch,
        state,
        model_workspace,
        station_workspace,
        previous_z_m=math.nan,
        envelope_height_m=zeros,
        envelope_slope_x=zeros,
        delta_arc_m=zeros,
        valid_mask=valid,
        preload_N=1.0,
        backplate_travel_m=0.006,
    )
    assert status == STATION_OK
    assert evaluations <= MAX_STATION_EVALUATIONS
    accepted_state = (
        state.mode.copy(),
        state.u_t_history_m.copy(),
        state.spring_branch.copy(),
    )

    status, returned_z_m, evaluations, _ = solve_station(
        batch,
        state,
        model_workspace,
        station_workspace,
        previous_z_m=accepted_z_m,
        envelope_height_m=zeros,
        envelope_slope_x=zeros,
        delta_arc_m=np.full(batch.spine_count, 1e-4),
        valid_mask=valid,
        preload_N=1e9,
        backplate_travel_m=1e-6,
    )

    assert status == STATION_PRELOAD_UNREACHABLE
    assert returned_z_m == accepted_z_m
    assert evaluations <= MAX_STATION_EVALUATIONS
    assert np.array_equal(state.mode, accepted_state[0])
    assert np.array_equal(state.u_t_history_m, accepted_state[1])
    assert np.array_equal(state.spring_branch, accepted_state[2])


def _accept_scripted_station(
    batch,
    previous_state,
    workspace,
    preload_N: float,
) -> None:
    previous_state.mode.fill(STICK)
    previous_state.u_t_history_m.fill(2e-5)
    previous_state.spring_branch.fill(INTERIOR)
    workspace.mode.fill(STICK)
    workspace.u_t_history_m.fill(2e-5)
    workspace.spring_branch.fill(INTERIOR)
    workspace.force_x_N.fill(-0.1 / batch.spine_count)
    workspace.force_z_N.fill(preload_N / batch.spine_count)
    workspace.lambda_n_N.fill(preload_N / batch.spine_count)
    workspace.tangent_force_N.fill(-0.1 / batch.spine_count)
    workspace.spring_axial_load_N.fill(0.0)


def test_detach_represses_from_fresh_state_and_finishes_path(
    monkeypatch,
) -> None:
    batch = _test_batch()
    bank = _flat_track_bank(np.unique(batch.y_m))
    calls: list[dict[str, np.ndarray]] = []

    def scripted_solve(
        current_batch,
        previous_state,
        workspace,
        _station_workspace,
        *,
        delta_arc_m,
        preload_N,
        previous_z_m,
        **_kwargs,
    ):
        calls.append(
            {
                "mode": previous_state.mode.copy(),
                "history": previous_state.u_t_history_m.copy(),
                "delta_arc": np.asarray(delta_arc_m).copy(),
            }
        )
        if len(calls) == 2:
            return (
                STATION_RECONTACT_REQUIRED,
                previous_z_m,
                1,
                0.2,
            )
        _accept_scripted_station(
            current_batch, previous_state, workspace, preload_N
        )
        return STATION_OK, -1e-3, 1, 0.0

    monkeypatch.setattr(solver_module, "solve_station", scripted_solve)
    settings = PathSettings(
        preload_N=1.0,
        path_length_m=0.0002,
        dx_m=0.0001,
        relanding_search_steps=1,
    )
    trace = PathTrace.allocate(batch, settings, include_spines=True)
    metrics, diagnostics = simulate_path(
        batch,
        bank,
        bank.rows_for_y(batch.y_m),
        settings,
        trace=trace,
    )

    assert len(calls) == 4
    assert np.all(calls[2]["mode"] == FREE)
    assert np.all(calls[2]["history"] == 0.0)
    assert np.all(calls[2]["delta_arc"] == 0.0)
    assert np.allclose(calls[3]["delta_arc"], settings.dx_m)
    assert metrics["case_status"] == "complete"
    assert metrics["completion_ratio"] == 1.0
    assert metrics["path_end_attempted"] is True
    assert metrics["recontact_count"] == 1
    assert metrics["detach_count"] == 1
    assert diagnostics["station_count_attempted"] == 2
    assert diagnostics["station_count_completed"] == 2
    assert trace.recontacted.tolist() == [False, True, False]


def test_unreachable_landing_tries_neighbor_before_continuing(
    monkeypatch,
) -> None:
    batch = _test_batch()
    bank = _flat_track_bank(np.unique(batch.y_m))
    call_count = 0

    def scripted_solve(
        current_batch,
        previous_state,
        workspace,
        _station_workspace,
        *,
        preload_N,
        previous_z_m,
        **_kwargs,
    ):
        nonlocal call_count
        call_count += 1
        if call_count in {2, 3}:
            return (
                STATION_PRELOAD_UNREACHABLE,
                previous_z_m,
                2,
                0.5,
            )
        _accept_scripted_station(
            current_batch, previous_state, workspace, preload_N
        )
        return STATION_OK, -1e-3, 1, 0.0

    monkeypatch.setattr(solver_module, "solve_station", scripted_solve)
    settings = PathSettings(
        preload_N=1.0,
        path_length_m=0.0001,
        dx_m=0.0001,
        relanding_search_steps=2,
    )
    trace = PathTrace.allocate(batch, settings, include_spines=False)
    metrics, _ = simulate_path(
        batch,
        bank,
        bank.rows_for_y(batch.y_m),
        settings,
        trace=trace,
    )

    assert call_count == 4
    assert metrics["case_status"] == "complete"
    assert metrics["recontact_count"] == 1
    assert metrics["landing_change_count"] == 1
    assert np.isclose(trace.landing_offset_m[1], settings.dx_m)


def test_invalid_station_is_recorded_but_does_not_abort_traversal(
    monkeypatch,
) -> None:
    batch = _test_batch()
    bank = _flat_track_bank(np.unique(batch.y_m))
    call_count = 0

    def scripted_solve(
        current_batch,
        previous_state,
        workspace,
        _station_workspace,
        *,
        preload_N,
        previous_z_m,
        **_kwargs,
    ):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            return STATION_TRACK_INVALID, previous_z_m, 0, math.nan
        _accept_scripted_station(
            current_batch, previous_state, workspace, preload_N
        )
        return STATION_OK, -1e-3, 1, 0.0

    monkeypatch.setattr(solver_module, "solve_station", scripted_solve)
    settings = PathSettings(
        preload_N=1.0,
        path_length_m=0.0001,
        dx_m=0.0001,
        relanding_search_steps=1,
    )
    trace = PathTrace.allocate(batch, settings, include_spines=False)
    metrics, diagnostics = simulate_path(
        batch,
        bank,
        bank.rows_for_y(batch.y_m),
        settings,
        trace=trace,
    )

    assert metrics["case_status"] == "completed_with_gaps"
    assert metrics["traversal_attempt_ratio"] == 1.0
    assert metrics["unsupported_station_count"] == 1
    assert metrics["track_invalid_station_count"] == 1
    assert diagnostics["station_count_attempted"] == 1
    assert trace.station_status[-1] == STATION_TRACK_INVALID


def test_numerical_failure_does_not_clear_contact_history(
    monkeypatch,
) -> None:
    batch = _test_batch()
    bank = _flat_track_bank(np.unique(batch.y_m))
    calls: list[dict[str, np.ndarray]] = []

    def scripted_solve(
        current_batch,
        previous_state,
        workspace,
        _station_workspace,
        *,
        delta_arc_m,
        preload_N,
        previous_z_m,
        **_kwargs,
    ):
        calls.append(
            {
                "mode": previous_state.mode.copy(),
                "history": previous_state.u_t_history_m.copy(),
                "delta_arc": np.asarray(delta_arc_m).copy(),
            }
        )
        if len(calls) == 2:
            return (
                STATION_NUMERICAL_FAILURE,
                previous_z_m,
                MAX_STATION_EVALUATIONS,
                0.02,
            )
        _accept_scripted_station(
            current_batch, previous_state, workspace, preload_N
        )
        return STATION_OK, -1e-3, 1, 0.0

    monkeypatch.setattr(solver_module, "solve_station", scripted_solve)
    settings = PathSettings(
        preload_N=1.0,
        path_length_m=0.0002,
        dx_m=0.0001,
        relanding_search_steps=2,
    )
    metrics, diagnostics = simulate_path(
        batch,
        bank,
        bank.rows_for_y(batch.y_m),
        settings,
    )

    assert len(calls) == 3
    assert np.all(calls[2]["mode"] == STICK)
    assert np.all(calls[2]["history"] == 2e-5)
    assert np.allclose(calls[2]["delta_arc"], 2 * settings.dx_m)
    assert metrics["case_status"] == "completed_with_gaps"
    assert metrics["numerical_failure_station_count"] == 1
    assert metrics["recontact_count"] == 0
    assert metrics["detach_count"] == 0
    assert diagnostics["station_count_attempted"] == 2


def test_candidate_coverage_never_reintroduces_ineligible_cases() -> None:
    eligible = {
        "configuration_id": "working",
        "eligible": True,
        "array_shape": "6x6",
        "spacing": 0.006,
        "angle_pattern": "fixed",
        "spring_family": "stiff",
    }
    rejected = {
        "configuration_id": "failed-boundary",
        "eligible": False,
        "array_shape": "2x2",
        "spacing": 0.005,
        "angle_pattern": "80_to_50",
        "spring_family": "compliant",
    }

    selected, unavailable = _select_m3b_configurations(
        [eligible, rejected]
    )

    assert selected == [eligible]
    assert rejected not in selected
    assert {"field": "array_shape", "value": "2x2"} in unavailable
    assert {"field": "angle_pattern", "value": "80_to_50"} in unavailable
