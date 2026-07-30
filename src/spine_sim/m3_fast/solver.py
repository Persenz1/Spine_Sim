"""Common-backplate scalar equilibrium and fixed-step M3 path solver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .model import (
    FREE,
    HARD_STOP,
    INTERIOR,
    SLIDE,
    STICK,
    ContactState,
    ModelWorkspace,
    SpineBatch,
    evaluate_spines,
    make_contact_state,
    make_model_workspace,
)
from .terrain import TrackBank, interpolate_tracks


STATION_OK = 0
STATION_SUPPORT_LOST = 1
STATION_TRACK_INVALID = 2

MAX_NEWTON_CORRECTIONS = 3
MAX_NEWTON_EVALUATIONS = MAX_NEWTON_CORRECTIONS + 1
MAX_INTERVAL_EVALUATIONS = 6
MAX_STATION_EVALUATIONS = 9


@dataclass(frozen=True)
class PathSettings:
    """Fixed spatial path and common-backplate settings for one case."""

    preload_N: float = 1.0
    path_length_m: float = 0.010
    dx_m: float = 0.0001
    path_start_x_m: float = 0.0
    backplate_travel_m: float = 0.006
    contact_clearance_m: float = 1e-12

    def __post_init__(self) -> None:
        for name, value in (
            ("preload_N", self.preload_N),
            ("path_length_m", self.path_length_m),
            ("dx_m", self.dx_m),
            ("backplate_travel_m", self.backplate_travel_m),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.path_start_x_m):
            raise ValueError("path_start_x_m must be finite")
        if (
            not math.isfinite(self.contact_clearance_m)
            or self.contact_clearance_m < 0.0
        ):
            raise ValueError("contact_clearance_m must be finite and non-negative")
        station_count = self.path_length_m / self.dx_m
        if not math.isclose(
            station_count,
            round(station_count),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("path_length_m must be an integer multiple of dx_m")

    @property
    def station_count(self) -> int:
        return int(round(self.path_length_m / self.dx_m))


@dataclass
class PathTrace:
    """Preallocated accepted path history for fine and final screening."""

    path_x_m: NDArray[np.float64]
    accepted: NDArray[np.bool_]
    station_status: NDArray[np.int8]
    root_evaluations: NDArray[np.int16]
    backplate_z_m: NDArray[np.float64]
    force_x_N: NDArray[np.float32]
    force_z_N: NDArray[np.float32]
    force_residual_N: NDArray[np.float32]
    contact_count: NDArray[np.int16]
    neff: NDArray[np.float32]
    max_load_share: NDArray[np.float32]
    stick_ratio: NDArray[np.float32]
    slide_ratio: NDArray[np.float32]
    hard_stop_ratio: NDArray[np.float32]
    spine_force_x_N: NDArray[np.float32] | None
    spine_force_z_N: NDArray[np.float32] | None
    spine_lambda_n_N: NDArray[np.float32] | None
    spine_tangent_force_N: NDArray[np.float32] | None
    spine_mode: NDArray[np.int8] | None
    spine_u_t_history_m: NDArray[np.float32] | None
    spine_spring_branch: NDArray[np.int8] | None
    spine_spring_load_N: NDArray[np.float32] | None
    spine_spring_displacement_m: NDArray[np.float32] | None

    @classmethod
    def allocate(
        cls,
        batch: SpineBatch,
        settings: PathSettings,
        *,
        include_spines: bool,
    ) -> "PathTrace":
        station_count = settings.station_count + 1
        path_x_m = settings.path_start_x_m + (
            np.arange(station_count, dtype=np.float64) * settings.dx_m
        )
        spine_shape = (station_count, batch.spine_count)

        def spine_float() -> NDArray[np.float32] | None:
            if not include_spines:
                return None
            values = np.empty(spine_shape, dtype=np.float32)
            values.fill(np.nan)
            return values

        def spine_int() -> NDArray[np.int8] | None:
            if not include_spines:
                return None
            return np.full(spine_shape, -1, dtype=np.int8)

        float_values = [
            np.full(station_count, np.nan, dtype=np.float32)
            for _ in range(8)
        ]
        return cls(
            path_x_m=path_x_m,
            accepted=np.zeros(station_count, dtype=np.bool_),
            station_status=np.full(station_count, -1, dtype=np.int8),
            root_evaluations=np.zeros(station_count, dtype=np.int16),
            backplate_z_m=np.full(station_count, np.nan, dtype=np.float64),
            force_x_N=float_values[0],
            force_z_N=float_values[1],
            force_residual_N=float_values[2],
            contact_count=np.zeros(station_count, dtype=np.int16),
            neff=float_values[3],
            max_load_share=float_values[4],
            stick_ratio=float_values[5],
            slide_ratio=float_values[6],
            hard_stop_ratio=float_values[7],
            spine_force_x_N=spine_float(),
            spine_force_z_N=spine_float(),
            spine_lambda_n_N=spine_float(),
            spine_tangent_force_N=spine_float(),
            spine_mode=spine_int(),
            spine_u_t_history_m=spine_float(),
            spine_spring_branch=spine_int(),
            spine_spring_load_N=spine_float(),
            spine_spring_displacement_m=spine_float(),
        )

    @property
    def includes_spines(self) -> bool:
        return self.spine_force_x_N is not None


@dataclass
class StationWorkspace:
    """Fixed-size scalar root workspace, allocated once per case."""

    z_samples_m: NDArray[np.float64]
    residual_samples_N: NDArray[np.float64]

    @classmethod
    def allocate(cls) -> "StationWorkspace":
        return cls(
            z_samples_m=np.empty(MAX_STATION_EVALUATIONS, dtype=np.float64),
            residual_samples_N=np.empty(
                MAX_STATION_EVALUATIONS, dtype=np.float64
            ),
        )


def _commit_state(state: ContactState, workspace: ModelWorkspace) -> None:
    np.copyto(state.mode, workspace.mode)
    np.copyto(state.u_t_history_m, workspace.u_t_history_m)
    np.copyto(state.spring_branch, workspace.spring_branch)


def _force_tolerance_N(preload_N: float) -> float:
    return max(0.01, 0.01 * preload_N)


def _record_trace_station(
    trace: PathTrace,
    index: int,
    *,
    status: int,
    evaluations: int,
    residual_N: float,
    backplate_z_m: float,
    batch: SpineBatch,
    workspace: ModelWorkspace | None,
) -> None:
    trace.station_status[index] = status
    trace.root_evaluations[index] = evaluations
    if status != STATION_OK or workspace is None:
        return
    trace.accepted[index] = True
    trace.backplate_z_m[index] = backplate_z_m
    trace.force_x_N[index] = -float(np.sum(workspace.force_x_N))
    trace.force_z_N[index] = float(np.sum(workspace.force_z_N))
    trace.force_residual_N[index] = residual_N
    contact = workspace.mode != FREE
    contact_count = int(np.count_nonzero(contact))
    trace.contact_count[index] = contact_count
    if contact_count:
        trace.stick_ratio[index] = (
            np.count_nonzero(contact & (workspace.mode == STICK))
            / contact_count
        )
        trace.slide_ratio[index] = (
            np.count_nonzero(contact & (workspace.mode == SLIDE))
            / contact_count
        )
        trace.hard_stop_ratio[index] = (
            np.count_nonzero(
                contact & (workspace.spring_branch == HARD_STOP)
            )
            / contact_count
        )
    else:
        trace.stick_ratio[index] = 0.0
        trace.slide_ratio[index] = 0.0
        trace.hard_stop_ratio[index] = 0.0
    positive_support_N = np.maximum(workspace.force_z_N, 0.0)
    support_sum_N = float(np.sum(positive_support_N))
    support_square_sum_N2 = float(
        np.dot(positive_support_N, positive_support_N)
    )
    if support_sum_N > 0.0 and support_square_sum_N2 > 0.0:
        trace.neff[index] = (
            support_sum_N * support_sum_N / support_square_sum_N2
        )
        trace.max_load_share[index] = (
            float(np.max(positive_support_N)) / support_sum_N
        )
    else:
        trace.neff[index] = 0.0
        trace.max_load_share[index] = 1.0

    if not trace.includes_spines:
        return
    assert trace.spine_force_x_N is not None
    assert trace.spine_force_z_N is not None
    assert trace.spine_lambda_n_N is not None
    assert trace.spine_tangent_force_N is not None
    assert trace.spine_mode is not None
    assert trace.spine_u_t_history_m is not None
    assert trace.spine_spring_branch is not None
    assert trace.spine_spring_load_N is not None
    assert trace.spine_spring_displacement_m is not None
    trace.spine_force_x_N[index] = workspace.force_x_N
    trace.spine_force_z_N[index] = workspace.force_z_N
    trace.spine_lambda_n_N[index] = workspace.lambda_n_N
    trace.spine_tangent_force_N[index] = workspace.tangent_force_N
    trace.spine_mode[index] = workspace.mode
    trace.spine_u_t_history_m[index] = workspace.u_t_history_m
    trace.spine_spring_branch[index] = workspace.spring_branch
    trace.spine_spring_load_N[index] = workspace.spring_axial_load_N
    spring_displacement_m = trace.spine_spring_displacement_m[index]
    spring_displacement_m.fill(0.0)
    interior = workspace.spring_branch == INTERIOR
    np.divide(
        workspace.spring_axial_load_N,
        batch.spring_stiffness_N_per_m,
        out=spring_displacement_m,
        where=interior,
    )
    hard = workspace.spring_branch == HARD_STOP
    spring_displacement_m[hard] = batch.spring_delta_max_m[hard]


def solve_station(
    batch: SpineBatch,
    previous_state: ContactState,
    model_workspace: ModelWorkspace,
    station_workspace: StationWorkspace,
    *,
    previous_z_m: float,
    envelope_height_m: NDArray[np.float64],
    envelope_slope_x: NDArray[np.float64],
    delta_arc_m: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    preload_N: float,
    backplate_travel_m: float,
    contact_clearance_m: float = 1e-12,
) -> tuple[int, float, int, float]:
    """Solve one common backplate height without mutating rejected state.

    The initial evaluation is followed by at most three local stiffness
    corrections.  If necessary, up to six interval evaluations are allowed,
    subject to the shared nine-evaluation hard limit.  The accepted trial
    already contains the final all-spine response, so it is committed directly
    rather than evaluated a tenth time.
    """

    spine_count = batch.spine_count
    if (
        np.asarray(envelope_height_m).shape != (spine_count,)
        or np.asarray(envelope_slope_x).shape != (spine_count,)
        or np.asarray(delta_arc_m).shape != (spine_count,)
        or np.asarray(valid_mask).shape != (spine_count,)
    ):
        raise ValueError("station terrain arrays must match the spine count")
    if not np.all(valid_mask):
        return STATION_TRACK_INVALID, previous_z_m, 0, math.nan
    if not math.isfinite(preload_N) or preload_N <= 0.0:
        raise ValueError("preload_N must be positive and finite")
    if not math.isfinite(backplate_travel_m) or backplate_travel_m <= 0.0:
        raise ValueError("backplate_travel_m must be positive and finite")

    # z_c^0 = Z + tip_z_offset.  The upper endpoint is deliberately just
    # beyond every zero-force contact threshold, hence R(z_max) = -preload.
    contact_threshold_m = envelope_height_m - batch.tip_z_offset_m
    z_max_m = (
        float(np.max(contact_threshold_m)) + float(contact_clearance_m)
    )
    z_min_m = z_max_m - backplate_travel_m
    upper_residual_N = -preload_N
    tolerance_N = _force_tolerance_N(preload_N)
    evaluation_count = 0
    best_abs_residual_N = math.inf

    if math.isfinite(previous_z_m):
        z_m = float(
            np.clip(
                previous_z_m,
                z_min_m,
                np.nextafter(z_max_m, z_min_m),
            )
        )
    else:
        z_m = z_max_m - min(0.05 * backplate_travel_m, 0.0003)

    # For +Z away from the wall, K_Z=-d(sum(f_z))/dZ is positive.  Therefore
    # Newton's correction is Z <- Z + (sum(f_z)-P)/K_Z.
    for correction_index in range(MAX_NEWTON_EVALUATIONS):
        total_fz_N, total_kz_N_per_m = evaluate_spines(
            batch,
            previous_state,
            model_workspace,
            backplate_z_m=z_m,
            envelope_height_m=envelope_height_m,
            envelope_slope_x=envelope_slope_x,
            delta_arc_m=delta_arc_m,
            valid_mask=valid_mask,
        )
        residual_N = total_fz_N - preload_N
        station_workspace.z_samples_m[evaluation_count] = z_m
        station_workspace.residual_samples_N[evaluation_count] = residual_N
        evaluation_count += 1
        best_abs_residual_N = min(best_abs_residual_N, abs(residual_N))
        if math.isfinite(residual_N) and abs(residual_N) <= tolerance_N:
            _commit_state(previous_state, model_workspace)
            return STATION_OK, z_m, evaluation_count, residual_N
        if (
            not math.isfinite(residual_N)
            or not math.isfinite(total_kz_N_per_m)
            or total_kz_N_per_m <= 0.0
        ):
            break
        correction_m = residual_N / total_kz_N_per_m
        candidate_z_m = z_m + correction_m
        # The mandated one-shot spring correction can create a large,
        # non-monotone branch jump when the old accepted branch is still the
        # nearby physical continuation.  Probe the opposite side of the local
        # stiffness step once, then resume ordinary scalar corrections.  This
        # stays inside the three-correction budget and does not alter state.
        if (
            correction_index == 0
            and math.isfinite(previous_z_m)
            and abs(residual_N) > 10.0 * preload_N
            and np.any(
                model_workspace.spring_branch
                != previous_state.spring_branch
            )
        ):
            opposite_z_m = z_m - correction_m
            if z_min_m < opposite_z_m < z_max_m:
                candidate_z_m = opposite_z_m
        if (
            not math.isfinite(candidate_z_m)
            or candidate_z_m <= z_min_m
            or candidate_z_m >= z_max_m
            or math.isclose(
                candidate_z_m,
                z_m,
                rel_tol=0.0,
                abs_tol=np.finfo(np.float64).eps
                * max(1.0, abs(z_m)),
            )
        ):
            break
        z_m = candidate_z_m

    # Reuse the tightest positive-residual Newton sample as the lower end.
    lower_z_m = z_min_m
    lower_residual_N = math.nan
    for index in range(evaluation_count):
        sample_residual_N = station_workspace.residual_samples_N[index]
        sample_z_m = station_workspace.z_samples_m[index]
        if (
            math.isfinite(sample_residual_N)
            and sample_residual_N > 0.0
            and (
                not math.isfinite(lower_residual_N)
                or sample_z_m > lower_z_m
            )
        ):
            lower_z_m = sample_z_m
            lower_residual_N = sample_residual_N

    interval_evaluations = 0
    if not math.isfinite(lower_residual_N):
        # The no-contact upper residual is known analytically; only the fully
        # compressed endpoint needs a full-array evaluation to establish a
        # bracket.
        total_fz_N, _ = evaluate_spines(
            batch,
            previous_state,
            model_workspace,
            backplate_z_m=z_min_m,
            envelope_height_m=envelope_height_m,
            envelope_slope_x=envelope_slope_x,
            delta_arc_m=delta_arc_m,
            valid_mask=valid_mask,
        )
        lower_residual_N = total_fz_N - preload_N
        station_workspace.z_samples_m[evaluation_count] = z_min_m
        station_workspace.residual_samples_N[evaluation_count] = lower_residual_N
        evaluation_count += 1
        interval_evaluations += 1
        best_abs_residual_N = min(
            best_abs_residual_N, abs(lower_residual_N)
        )
        if (
            math.isfinite(lower_residual_N)
            and abs(lower_residual_N) <= tolerance_N
        ):
            _commit_state(previous_state, model_workspace)
            return (
                STATION_OK,
                z_min_m,
                evaluation_count,
                lower_residual_N,
            )
        if not math.isfinite(lower_residual_N) or lower_residual_N < 0.0:
            return (
                STATION_SUPPORT_LOST,
                previous_z_m,
                evaluation_count,
                best_abs_residual_N,
            )

    # A negative Newton sample above the lower endpoint gives a tighter upper
    # bound; otherwise the analytically contact-free endpoint is used.
    upper_z_m = z_max_m
    for index in range(evaluation_count):
        sample_residual_N = station_workspace.residual_samples_N[index]
        sample_z_m = station_workspace.z_samples_m[index]
        if (
            math.isfinite(sample_residual_N)
            and sample_residual_N < 0.0
            and lower_z_m < sample_z_m < upper_z_m
        ):
            upper_z_m = sample_z_m
            upper_residual_N = sample_residual_N

    while (
        interval_evaluations < MAX_INTERVAL_EVALUATIONS
        and evaluation_count < MAX_STATION_EVALUATIONS
    ):
        z_m = 0.5 * (lower_z_m + upper_z_m)
        total_fz_N, _ = evaluate_spines(
            batch,
            previous_state,
            model_workspace,
            backplate_z_m=z_m,
            envelope_height_m=envelope_height_m,
            envelope_slope_x=envelope_slope_x,
            delta_arc_m=delta_arc_m,
            valid_mask=valid_mask,
        )
        residual_N = total_fz_N - preload_N
        station_workspace.z_samples_m[evaluation_count] = z_m
        station_workspace.residual_samples_N[evaluation_count] = residual_N
        evaluation_count += 1
        interval_evaluations += 1
        best_abs_residual_N = min(best_abs_residual_N, abs(residual_N))
        if math.isfinite(residual_N) and abs(residual_N) <= tolerance_N:
            _commit_state(previous_state, model_workspace)
            return STATION_OK, z_m, evaluation_count, residual_N
        if not math.isfinite(residual_N):
            break
        if residual_N > 0.0:
            lower_z_m = z_m
            lower_residual_N = residual_N
        else:
            upper_z_m = z_m
            upper_residual_N = residual_N

    return (
        STATION_SUPPORT_LOST,
        previous_z_m,
        evaluation_count,
        best_abs_residual_N,
    )


def _empty_metrics(case_status: str, support_position_m: float) -> dict[str, Any]:
    return {
        "completion_ratio": 0.0,
        "Fx_q10": math.nan,
        "Fx_median": math.nan,
        "Fx_peak_qs": math.nan,
        "contact_ratio": 0.0,
        "Neff_q10": math.nan,
        "Neff_median": math.nan,
        "max_load_share_q90": math.nan,
        "slide_ratio": 0.0,
        "hard_stop_ratio": 0.0,
        "support_loss_position": support_position_m,
        "case_status": case_status,
    }


def simulate_path(
    batch: SpineBatch,
    track_bank: TrackBank,
    track_rows: NDArray[np.intp],
    settings: PathSettings,
    *,
    trace: PathTrace | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one fixed-step case and return its compact metrics and diagnostics."""

    spine_count = batch.spine_count
    rows = np.asarray(track_rows, dtype=np.intp)
    if rows.shape != (spine_count,):
        raise ValueError("track_rows must contain one row per spine")
    if trace is not None:
        expected_station_count = settings.station_count + 1
        if trace.path_x_m.shape != (expected_station_count,):
            raise ValueError("trace station count does not match path settings")
        if trace.includes_spines:
            assert trace.spine_force_x_N is not None
            if trace.spine_force_x_N.shape != (
                expected_station_count,
                spine_count,
            ):
                raise ValueError(
                    "trace spine shape does not match the current batch"
                )

    state = make_contact_state(batch)
    model_workspace = make_model_workspace(batch)
    station_workspace = StationWorkspace.allocate()
    height_m = np.empty(spine_count, dtype=np.float64)
    slope_x = np.empty(spine_count, dtype=np.float64)
    arc_m = np.empty(spine_count, dtype=np.float64)
    previous_arc_m = np.empty(spine_count, dtype=np.float64)
    delta_arc_m = np.empty(spine_count, dtype=np.float64)
    valid = np.empty(spine_count, dtype=np.bool_)
    query_x_m = np.empty(spine_count, dtype=np.float64)

    station_count = settings.station_count
    resistance_N = np.empty(station_count, dtype=np.float64)
    neff = np.empty(station_count, dtype=np.float64)
    max_load_share = np.empty(station_count, dtype=np.float64)
    root_evaluations = np.empty(station_count + 1, dtype=np.int16)
    total_contact_samples = 0
    total_slide_samples = 0
    total_hard_stop_samples = 0
    completed_stations = 0

    # Establish initial preload at x0 without adding it to path statistics.
    np.add(
        batch.tip_x_offset_m,
        settings.path_start_x_m,
        out=query_x_m,
    )
    interpolate_tracks(
        track_bank,
        rows,
        query_x_m,
        out_height=height_m,
        out_slope=slope_x,
        out_arc_length=arc_m,
        out_valid=valid,
    )
    delta_arc_m.fill(0.0)
    initial_status, previous_z_m, evaluations, initial_residual_N = solve_station(
        batch,
        state,
        model_workspace,
        station_workspace,
        previous_z_m=math.nan,
        envelope_height_m=height_m,
        envelope_slope_x=slope_x,
        delta_arc_m=delta_arc_m,
        valid_mask=valid,
        preload_N=settings.preload_N,
        backplate_travel_m=settings.backplate_travel_m,
        contact_clearance_m=settings.contact_clearance_m,
    )
    root_evaluations[0] = evaluations
    if trace is not None:
        _record_trace_station(
            trace,
            0,
            status=initial_status,
            evaluations=evaluations,
            residual_N=initial_residual_N,
            backplate_z_m=previous_z_m,
            batch=batch,
            workspace=(
                model_workspace if initial_status == STATION_OK else None
            ),
        )
    if initial_status != STATION_OK:
        status_name = (
            "track_invalid"
            if initial_status == STATION_TRACK_INVALID
            else "support_lost"
        )
        metrics = _empty_metrics(status_name, settings.path_start_x_m)
        diagnostics = {
            "initial_station_status": initial_status,
            "max_station_evaluations": int(evaluations),
            "station_count_requested": station_count,
            "station_count_completed": 0,
        }
        return metrics, diagnostics
    np.copyto(previous_arc_m, arc_m)

    failure_status = STATION_OK
    support_loss_position_m: float | None = None
    for station in range(1, station_count + 1):
        path_x_m = settings.path_start_x_m + station * settings.dx_m
        np.add(batch.tip_x_offset_m, path_x_m, out=query_x_m)
        interpolate_tracks(
            track_bank,
            rows,
            query_x_m,
            out_height=height_m,
            out_slope=slope_x,
            out_arc_length=arc_m,
            out_valid=valid,
        )
        np.subtract(arc_m, previous_arc_m, out=delta_arc_m)
        (
            station_status,
            solved_z_m,
            evaluations,
            station_residual_N,
        ) = solve_station(
            batch,
            state,
            model_workspace,
            station_workspace,
            previous_z_m=previous_z_m,
            envelope_height_m=height_m,
            envelope_slope_x=slope_x,
            delta_arc_m=delta_arc_m,
            valid_mask=valid,
            preload_N=settings.preload_N,
            backplate_travel_m=settings.backplate_travel_m,
            contact_clearance_m=settings.contact_clearance_m,
        )
        root_evaluations[station] = evaluations
        if trace is not None:
            _record_trace_station(
                trace,
                station,
                status=station_status,
                evaluations=evaluations,
                residual_N=station_residual_N,
                backplate_z_m=solved_z_m,
                batch=batch,
                workspace=(
                    model_workspace
                    if station_status == STATION_OK
                    else None
                ),
            )
        if station_status != STATION_OK:
            failure_status = station_status
            support_loss_position_m = path_x_m
            break

        previous_z_m = solved_z_m
        np.copyto(previous_arc_m, arc_m)
        sample_index = completed_stations
        resistance_N[sample_index] = -float(
            np.sum(model_workspace.force_x_N)
        )
        contact_mask = model_workspace.mode != FREE
        contact_count = int(np.count_nonzero(contact_mask))
        slide_count = int(
            np.count_nonzero(contact_mask & (model_workspace.mode == SLIDE))
        )
        hard_stop_count = int(
            np.count_nonzero(
                contact_mask & (model_workspace.spring_branch == HARD_STOP)
            )
        )
        positive_support_N = np.maximum(model_workspace.force_z_N, 0.0)
        support_sum_N = float(np.sum(positive_support_N))
        support_square_sum_N2 = float(
            np.dot(positive_support_N, positive_support_N)
        )
        if support_sum_N > 0.0 and support_square_sum_N2 > 0.0:
            neff[sample_index] = (
                support_sum_N * support_sum_N / support_square_sum_N2
            )
            max_load_share[sample_index] = (
                float(np.max(positive_support_N)) / support_sum_N
            )
        else:
            neff[sample_index] = 0.0
            max_load_share[sample_index] = 1.0
        total_contact_samples += contact_count
        total_slide_samples += slide_count
        total_hard_stop_samples += hard_stop_count
        completed_stations += 1

    if completed_stations:
        accepted_resistance = resistance_N[:completed_stations]
        accepted_neff = neff[:completed_stations]
        accepted_share = max_load_share[:completed_stations]
        metrics: dict[str, Any] = {
            "completion_ratio": completed_stations / station_count,
            "Fx_q10": float(np.quantile(accepted_resistance, 0.10)),
            "Fx_median": float(np.median(accepted_resistance)),
            "Fx_peak_qs": float(np.max(accepted_resistance)),
            "contact_ratio": total_contact_samples
            / (completed_stations * spine_count),
            "Neff_q10": float(np.quantile(accepted_neff, 0.10)),
            "Neff_median": float(np.median(accepted_neff)),
            "max_load_share_q90": float(
                np.quantile(accepted_share, 0.90)
            ),
            "slide_ratio": (
                total_slide_samples / total_contact_samples
                if total_contact_samples
                else 0.0
            ),
            "hard_stop_ratio": (
                total_hard_stop_samples / total_contact_samples
                if total_contact_samples
                else 0.0
            ),
            "support_loss_position": support_loss_position_m,
            "case_status": (
                "complete"
                if failure_status == STATION_OK
                else (
                    "track_invalid"
                    if failure_status == STATION_TRACK_INVALID
                    else "support_lost"
                )
            ),
        }
    else:
        metrics = _empty_metrics(
            (
                "track_invalid"
                if failure_status == STATION_TRACK_INVALID
                else "support_lost"
            ),
            0.0 if support_loss_position_m is None else support_loss_position_m,
        )

    attempted_count = min(
        completed_stations + 2,
        station_count + 1,
    )
    diagnostics = {
        "initial_station_status": initial_status,
        "max_station_evaluations": int(
            np.max(root_evaluations[:attempted_count])
        ),
        "station_count_requested": station_count,
        "station_count_completed": completed_stations,
        "final_backplate_z_m": (
            float(previous_z_m) if math.isfinite(previous_z_m) else None
        ),
    }
    return metrics, diagnostics


__all__ = [
    "MAX_STATION_EVALUATIONS",
    "PathSettings",
    "PathTrace",
    "STATION_OK",
    "STATION_SUPPORT_LOST",
    "STATION_TRACK_INVALID",
    "StationWorkspace",
    "simulate_path",
    "solve_station",
]
