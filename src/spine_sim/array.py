"""Canonical O(N) rigid-backplate array equilibrium and event trial solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core.errors import ConfigurationError
from .core.frames import Wrench
from .core.states import (
    ContinuationAction,
    Event,
    ModelState,
    NumericalState,
    PhysicalState,
)
from .geometry import ContactCandidate
from .metrics import ArrayCounts, SpineMetricInput, compute_array_counts
from .single_spine import (
    BaseMotion,
    FrictionParameters,
    SingleSpineResult,
    SingleSpineTolerances,
    SingleSpineTrial,
    SpineAcceptedState,
    SpineGeometry,
    SpineMaterial,
    SuspensionParameters,
    commit_single_spine_trial,
    solve_single_spine,
)


Vector6 = NDArray[np.float64]
Matrix6 = NDArray[np.float64]


class ControlMode(StrEnum):
    PRESCRIBED_POSE = "prescribed_pose"
    REQUIRED_WRENCH = "required_wrench"


class RankStatus(StrEnum):
    FULL_RANK = "full_rank"
    RANK_DEFICIENT = "rank_deficient"


class RangeStatus(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"


class EquilibriumStatus(StrEnum):
    SOLVED = "solved"
    RANK_DEFICIENT = "rank_deficient"
    RANGE_INCOMPATIBLE = "range_incompatible"
    NONCONVERGED = "nonconverged"
    MODEL_LIMIT = "model_limit"


class QuasistaticStability(StrEnum):
    STABLE_CONSERVATIVE = "stable_conservative"
    MARGINAL_CONSERVATIVE = "marginal_conservative"
    UNSTABLE_CONSERVATIVE = "unstable_conservative"
    NO_FREE_MODE = "no_free_mode"
    DIRECTIONALLY_ADMISSIBLE_QUASISTATIC = (
        "directionally_admissible_quasistatic"
    )
    NOT_EVALUATED = "not_evaluated"


def _vector(value: ArrayLike, length: int, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ConfigurationError(f"{name} must be a finite {length}-vector")
    return result


def _matrix(value: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size, size) or not np.all(np.isfinite(result)):
        raise ConfigurationError(f"{name} must be a finite {size}x{size} matrix")
    return result


def _tuple_vector(value: ArrayLike) -> tuple[float, ...]:
    return tuple(float(item) for item in np.asarray(value, dtype=np.float64))


def _tuple_matrix(value: ArrayLike) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(item) for item in row)
        for row in np.asarray(value, dtype=np.float64)
    )


def _skew(value: ArrayLike) -> NDArray[np.float64]:
    x, y, z = _vector(value, 3, "skew vector")
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ArrayTolerances:
    scaled_residual: float = 1e-9
    rank_absolute: float = 1e-12
    rank_relative: float = 1e-10
    range_residual: float = 1e-9
    stability_absolute: float = 1e-10
    stability_relative: float = 1e-8
    gap_m: float = 1e-10
    maximum_iterations: int = 30

    def __post_init__(self) -> None:
        for name in (
            "scaled_residual",
            "rank_absolute",
            "rank_relative",
            "range_residual",
            "stability_absolute",
            "stability_relative",
            "gap_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ConfigurationError(f"{name} must be finite and positive")
        if self.maximum_iterations < 1:
            raise ConfigurationError("maximum_iterations must be positive")


@dataclass(frozen=True)
class MixedControl:
    """Each component selects exactly one pose or wrench control contract."""

    modes: tuple[ControlMode, ...]
    prescribed_q_C: tuple[float | None, ...]
    required_wrench: tuple[float | None, ...]
    loader_stiffness: tuple[tuple[float, ...], ...]
    initial_q_C: tuple[float, ...]
    q_rate_C: tuple[float, ...]
    F_ref_N: float
    L_ref_m: float
    frame: str = "wall"
    reference_point: str = "array_reference"
    reference_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    backplate_object: str = "rigid_backplate"
    resistance_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    equality_matrix: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if len(self.modes) != 6:
            raise ConfigurationError("mixed control requires six component modes")
        if len(self.prescribed_q_C) != 6 or len(self.required_wrench) != 6:
            raise ConfigurationError("mixed control values must contain six components")
        for index, mode in enumerate(self.modes):
            pose = self.prescribed_q_C[index]
            wrench = self.required_wrench[index]
            if mode is ControlMode.PRESCRIBED_POSE:
                if pose is None or wrench is not None:
                    raise ConfigurationError(
                        "a prescribed-pose component requires pose only"
                    )
            elif mode is ControlMode.REQUIRED_WRENCH:
                if wrench is None or pose is not None:
                    raise ConfigurationError(
                        "a required-wrench component requires wrench only"
                    )
            else:
                raise ConfigurationError(f"unsupported control mode: {mode}")
        _vector(self.initial_q_C, 6, "initial_q_C")
        _vector(self.q_rate_C, 6, "q_rate_C")
        loader = _matrix(self.loader_stiffness, 6, "loader_stiffness")
        if not np.allclose(loader, loader.T, atol=1e-12):
            raise ConfigurationError("loader_stiffness must be symmetric")
        scale = max(1.0, float(np.linalg.norm(loader, ord=2)))
        if float(np.linalg.eigvalsh(loader).min()) < -1e-12 * scale:
            raise ConfigurationError("loader_stiffness must be positive semidefinite")
        if not math.isfinite(self.F_ref_N) or self.F_ref_N <= 0.0:
            raise ConfigurationError("F_ref_N must be finite and positive")
        if not math.isfinite(self.L_ref_m) or self.L_ref_m <= 0.0:
            raise ConfigurationError("L_ref_m must be finite and positive")
        _vector(self.reference_position_m, 3, "reference_position_m")
        direction = _vector(self.resistance_direction, 3, "resistance_direction")
        if float(np.linalg.norm(direction)) <= 0.0:
            raise ConfigurationError("resistance_direction must be non-zero")
        for row in self.equality_matrix:
            _vector(row, 6, "equality_matrix row")
        for name in ("frame", "reference_point", "backplate_object"):
            if not getattr(self, name):
                raise ConfigurationError(f"{name} cannot be empty")


@dataclass(frozen=True)
class SpineInstance:
    geometry: SpineGeometry
    material: SpineMaterial
    friction: FrictionParameters
    suspension: SuspensionParameters
    tolerances: SingleSpineTolerances
    initial_gap_m: float
    candidate: ContactCandidate | None
    stable_engagement: bool | None = None
    continuation_candidates: tuple[ContactCandidate, ...] = ()
    search_distance_increment_m: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_gap_m) or self.initial_gap_m < 0.0:
            raise ConfigurationError("initial_gap_m must be finite and non-negative")
        if (
            not math.isfinite(self.search_distance_increment_m)
            or self.search_distance_increment_m < 0.0
        ):
            raise ConfigurationError(
                "search_distance_increment_m must be finite and non-negative"
            )
        candidates = tuple(
            item
            for item in (self.candidate, *self.continuation_candidates)
            if item is not None
        )
        ids = [item.candidate_id for item in candidates]
        indices = [
            int(getattr(item, "candidate_index", index))
            for index, item in enumerate(candidates)
        ]
        if len(ids) != len(set(ids)) or indices != sorted(set(indices)):
            raise ConfigurationError(
                "candidate continuation must have unique IDs and increasing indices"
            )

    @property
    def spine_id(self) -> str:
        return self.geometry.spine_id


@dataclass(frozen=True)
class ArrayAcceptedState:
    q_C: tuple[float, ...]
    spine_states: tuple[SpineAcceptedState, ...]
    load_parameter: float
    revision: int = 0

    @classmethod
    def initial(
        cls, spines: Sequence[SpineInstance], *, load_parameter: float = 0.0
    ) -> "ArrayAcceptedState":
        return cls(
            q_C=(0.0,) * 6,
            spine_states=tuple(
                SpineAcceptedState.initial(spine.spine_id, load_parameter=load_parameter)
                for spine in spines
            ),
            load_parameter=float(load_parameter),
            revision=0,
        )


@dataclass(frozen=True)
class PerSpineArrayResult:
    spine_id: str
    initial_gap_m: float
    terrain_signed_gap_m: float | None
    closure_threshold_m: float | None
    signed_gap_m: float | None
    loading_displacement_m: float | None
    generalized_wrench: Wrench
    single_result: SingleSpineResult


@dataclass(frozen=True)
class EquilibriumDiagnostics:
    iterations: int
    scaled_residual_norm: float | None
    scaled_rank: int
    free_dof_count: int
    singular_values: tuple[float, ...]
    range_residual_norm: float | None
    minimum_stability_eigenvalue: float | None
    stability_threshold: float | None
    assembled_spine_count: int
    admissible_free_mode_count: int
    largest_dense_matrix_shape: tuple[int, int] = (6, 6)


@dataclass(frozen=True)
class ArrayResult:
    q_C: tuple[float, ...]
    physical_backplate_pose: tuple[float, ...]
    total_wrench: Wrench
    tangent: tuple[tuple[float, ...], ...]
    per_spine: tuple[PerSpineArrayResult, ...]
    counts: ArrayCounts
    rank_status: RankStatus
    range_status: RangeStatus
    equilibrium_status: EquilibriumStatus
    quasistatic_stability: QuasistaticStability
    dynamic_stability: ModelState
    model_state: ModelState
    numerical_state: NumericalState
    diagnostics: EquilibriumDiagnostics
    events: tuple[Event, ...]
    released_wrenches: tuple[Mapping[str, Any], ...]
    rebalance_predictions: tuple[Mapping[str, Any], ...]
    assumptions: tuple[str, ...]
    omissions: tuple[str, ...]


@dataclass(frozen=True)
class ArrayTrial:
    base_revision: int
    proposed_state: ArrayAcceptedState
    result: ArrayResult
    committable: bool


@dataclass(frozen=True)
class _Assembly:
    support_wrench: Vector6
    tangent: Matrix6
    trials: tuple[SingleSpineTrial, ...]
    per_spine: tuple[PerSpineArrayResult, ...]
    counts: ArrayCounts
    released_wrenches: tuple[Mapping[str, Any], ...]


def _candidate_for_state(
    spine: SpineInstance, accepted: SpineAcceptedState
) -> ContactCandidate | None:
    candidates = tuple(
        item
        for item in (spine.candidate, *spine.continuation_candidates)
        if item is not None
    )
    if not candidates or accepted.physical_state is PhysicalState.FAILED:
        return None
    if accepted.physical_state in {
        PhysicalState.CONTACT,
        PhysicalState.STICK,
        PhysicalState.SLIP,
        PhysicalState.HARDSTOP,
    }:
        return next(
            (
                item
                for item in candidates
                if item.candidate_id == accepted.candidate_id
            ),
            None,
        )
    if accepted.physical_state in {
        PhysicalState.DETACH,
        PhysicalState.REBOUND,
    }:
        return None
    cursor = accepted.search_cursor
    if cursor is None:
        return candidates[0]
    if bool(getattr(cursor, "exhausted", False)):
        return None
    cursor_index = getattr(cursor, "candidate_index", cursor)
    if isinstance(cursor_index, int):
        return next(
            (
                item
                for index, item in enumerate(candidates)
                if int(getattr(item, "candidate_index", index))
                >= cursor_index
            ),
            None,
        )
    return next(
        (item for item in candidates if item.candidate_id == cursor_index),
        None,
    )


def _control_vectors(
    control: MixedControl,
    accepted: ArrayAcceptedState,
) -> tuple[Vector6, Vector6, NDArray[np.int64], NDArray[np.int64]]:
    q = _vector(
        accepted.q_C if accepted.revision > 0 else control.initial_q_C,
        6,
        "accepted/initial q_C",
    ).copy()
    required = np.zeros(6, dtype=np.float64)
    prescribed: list[int] = []
    free: list[int] = []
    for index, mode in enumerate(control.modes):
        if mode is ControlMode.PRESCRIBED_POSE:
            q[index] = float(control.prescribed_q_C[index])
            prescribed.append(index)
        else:
            required[index] = float(control.required_wrench[index])
            free.append(index)
    return (
        q,
        required,
        np.asarray(prescribed, dtype=np.int64),
        np.asarray(free, dtype=np.int64),
    )


def _scaled_rank_range(
    matrix: Matrix6,
    residual: Vector6,
    basis: NDArray[np.float64],
    control: MixedControl,
    tolerances: ArrayTolerances,
) -> tuple[RankStatus, RangeStatus, int, NDArray[np.float64], float, NDArray[np.float64]]:
    mode_count = basis.shape[1]
    if mode_count == 0:
        return (
            RankStatus.FULL_RANK,
            RangeStatus.COMPATIBLE,
            0,
            np.empty(0),
            0.0,
            np.empty((0, 0)),
        )
    row_scale = np.array(
        [1.0 / control.F_ref_N] * 3
        + [1.0 / (control.F_ref_N * control.L_ref_m)] * 3
    )
    coordinate_scale = np.array([control.L_ref_m] * 3 + [1.0] * 3)
    scaled_full = (
        row_scale[:, None]
        * matrix
        * coordinate_scale[None, :]
    )
    scaled = basis.T @ scaled_full @ basis
    rhs = -basis.T @ (row_scale * residual)
    u, singular, _vt = np.linalg.svd(scaled, full_matrices=True)
    leading = float(singular[0]) if singular.size else 0.0
    threshold = tolerances.rank_absolute + tolerances.rank_relative * leading
    rank = int(np.count_nonzero(singular > threshold))
    if rank:
        projected = u[:, :rank] @ (u[:, :rank].T @ rhs)
    else:
        projected = np.zeros_like(rhs)
    range_residual = float(np.linalg.norm(rhs - projected))
    return (
        RankStatus.FULL_RANK
        if rank == mode_count
        else RankStatus.RANK_DEFICIENT,
        RangeStatus.COMPATIBLE
        if range_residual <= tolerances.range_residual
        else RangeStatus.INCOMPATIBLE,
        rank,
        singular,
        range_residual,
        scaled,
    )


def _metric_input(
    result: SingleSpineResult,
    candidate: ContactCandidate | None,
    stable_engagement: bool | None,
    resistance_direction: NDArray[np.float64],
    force_tolerance_N: float,
) -> SpineMetricInput:
    geometric = bool(candidate is not None and candidate.valid)
    signed_gap = None if candidate is None else float(candidate.signed_gap_m)
    if result.physical_state in {
        PhysicalState.SEARCH,
        PhysicalState.DETACH,
        PhysicalState.REBOUND,
        PhysicalState.FAILED,
    }:
        engagement: bool | None = False
    elif candidate is not None and (
        candidate.near_tie
        or result.model_state is ModelState.PARAMETER_UNCLOSED
    ):
        engagement: bool | None = None
    elif result.physical_state in {
        PhysicalState.STICK,
        PhysicalState.SLIP,
        PhysicalState.HARDSTOP,
    }:
        # Friction feasibility is resolved by the single-spine constitutive
        # state.  The separate local safe-slope/directional predicate is an
        # explicit input because it depends on the queried path direction; an
        # absent assessment is epistemically unknown, never coerced to false.
        engagement = stable_engagement
    else:
        engagement = None
    force = np.asarray(result.wall_force_N, dtype=np.float64)
    active = (
        result.physical_state
        in {PhysicalState.STICK, PhysicalState.SLIP, PhysicalState.HARDSTOP}
        and float(np.linalg.norm(force)) > force_tolerance_N
    )
    return SpineMetricInput(
        geometric=geometric,
        signed_gap_m=signed_gap,
        engagement=engagement,
        active=active,
        normal_force_N=(result.normal_force_N if geometric else None),
        tangent_resistance_N=float(np.dot(force, resistance_direction)),
    )


def _assemble(
    spines: Sequence[SpineInstance],
    states: Sequence[SpineAcceptedState],
    q_C: Vector6,
    control: MixedControl,
    *,
    load_parameter: float,
) -> _Assembly:
    total = np.zeros(6, dtype=np.float64)
    tangent = np.zeros((6, 6), dtype=np.float64)
    trials: list[SingleSpineTrial] = []
    per_spine: list[PerSpineArrayResult] = []
    metrics: list[SpineMetricInput] = []
    released: list[Mapping[str, Any]] = []
    reference_position = _vector(
        control.reference_position_m, 3, "reference_position_m"
    )
    q_rate = _vector(control.q_rate_C, 6, "q_rate_C")
    direction = _vector(
        control.resistance_direction, 3, "resistance_direction"
    )
    direction /= np.linalg.norm(direction)
    for spine, accepted in zip(spines, states, strict=True):
        candidate = _candidate_for_state(spine, accepted)
        terrain_signed_gap: float | None = None
        closure_threshold: float | None = None
        loading_displacement: float | None = None
        if candidate is None or candidate.selected_normal is None:
            point = _vector(spine.geometry.root_position_m, 3, "root_position_m")
            B = np.hstack((np.eye(3), -_skew(point - reference_position)))
            dynamic_candidate = candidate
            relative = B @ q_C
            signed_gap = None if candidate is None else candidate.signed_gap_m
        else:
            normal = _vector(candidate.selected_normal, 3, "selected_normal")
            normal /= np.linalg.norm(normal)
            point = (
                _vector(candidate.sphere_center_m, 3, "sphere_center_m")
                - spine.geometry.tip_radius_m * normal
            )
            B = np.hstack((np.eye(3), -_skew(point - reference_position)))
            point_displacement = B @ q_C
            terrain_signed_gap = float(candidate.signed_gap_m)
            closure_threshold = spine.initial_gap_m + terrain_signed_gap
            loading_displacement = (
                float(np.dot(normal, point_displacement)) - closure_threshold
            )
            relative = point_displacement - closure_threshold * normal
            open_gap = -loading_displacement
            # Once the unilateral contact is closed, elastic deformation is
            # carried by BaseMotion and the geometric gap remains zero.  Passing
            # overclosure as a negative terrain gap would make the canonical
            # single-spine solver reject a perfectly valid loaded contact.
            signed_gap = max(open_gap, 0.0)
            dynamic_candidate = replace(candidate, signed_gap_m=signed_gap)
        motion = BaseMotion(
            relative_displacement_m=tuple(float(value) for value in relative),
            relative_tangential_velocity_m_per_s=tuple(
                float(value) for value in B @ q_rate
            ),
            load_parameter=float(load_parameter),
            search_distance_increment_m=spine.search_distance_increment_m,
        )
        trial = solve_single_spine(
            spine.geometry,
            spine.material,
            spine.friction,
            spine.suspension,
            accepted,
            motion,
            dynamic_candidate,
            tolerances=spine.tolerances,
        )
        trials.append(trial)
        single = trial.result
        moved = single.root_wrench.move_reference(
            np.asarray(spine.geometry.root_position_m) - reference_position,
            new_reference_point=control.reference_point,
        )
        total += moved.vector
        local_tangent = np.asarray(single.local_tangent_N_per_m, dtype=np.float64)
        tangent += B.T @ local_tangent @ B
        per_spine.append(
            PerSpineArrayResult(
                spine_id=spine.spine_id,
                initial_gap_m=spine.initial_gap_m,
                terrain_signed_gap_m=terrain_signed_gap,
                closure_threshold_m=closure_threshold,
                signed_gap_m=(None if signed_gap is None else float(signed_gap)),
                loading_displacement_m=loading_displacement,
                generalized_wrench=moved,
                single_result=single,
            )
        )
        metrics.append(
            _metric_input(
                single,
                dynamic_candidate,
                spine.stable_engagement,
                direction,
                spine.tolerances.force_N,
            )
        )
        for event in single.events:
            failure = event.details.get("failure")
            force_before = event.details.get("force_before_N")
            if (
                event.event_type.value == "material_failure"
                and isinstance(failure, Mapping)
                and failure.get("continuation_action")
                in {
                    ContinuationAction.PERMANENT_REMOVE,
                    ContinuationAction.PERMANENT_REMOVE.value,
                }
                and force_before is not None
            ):
                released_force = _vector(force_before, 3, "released force")
                released.append(
                    {
                        "spine_id": spine.spine_id,
                        "force_N": _tuple_vector(released_force),
                        "moment_Nm": _tuple_vector(
                            np.cross(point - reference_position, released_force)
                        ),
                        "reference_point": control.reference_point,
                    }
                )
    counts = compute_array_counts(metrics, gap_tolerance_m=min(
        spine.tolerances.gap_m for spine in spines
    ))
    return _Assembly(
        total,
        tangent,
        tuple(trials),
        tuple(per_spine),
        counts,
        tuple(released),
    )


def _predict_contact_seed(
    spines: Sequence[SpineInstance],
    states: Sequence[SpineAcceptedState],
    q_C: Vector6,
    admissible_basis: NDArray[np.float64],
    control: MixedControl,
    residual: Vector6,
    attempted: set[tuple[int, str]],
) -> tuple[Vector6, tuple[int, str]] | None:
    """Select one load-compatible open contact and seed it in the free subspace."""

    if admissible_basis.shape[1] == 0:
        return None
    reference_position = _vector(
        control.reference_position_m, 3, "reference_position_m"
    )
    row_scale = np.array(
        [1.0 / control.F_ref_N] * 3
        + [1.0 / (control.F_ref_N * control.L_ref_m)] * 3
    )
    coordinate_scale = np.array([control.L_ref_m] * 3 + [1.0] * 3)
    displacement_basis = coordinate_scale[:, None] * admissible_basis
    desired_mode = -admissible_basis.T @ (row_scale * residual)
    best_score = 0.0
    best: tuple[Vector6, tuple[int, str]] | None = None
    probe = max(
        1e-9,
        16.0 * max(spine.tolerances.gap_m for spine in spines),
    )
    for index, (spine, state) in enumerate(zip(spines, states, strict=True)):
        if state.physical_state is not PhysicalState.SEARCH:
            continue
        candidate = _candidate_for_state(spine, state)
        if (
            candidate is None
            or not candidate.valid
            or candidate.selected_normal is None
        ):
            continue
        key = (index, candidate.candidate_id)
        if key in attempted:
            continue
        normal = _vector(candidate.selected_normal, 3, "selected_normal")
        normal /= np.linalg.norm(normal)
        point = (
            _vector(candidate.sphere_center_m, 3, "sphere_center_m")
            - spine.geometry.tip_radius_m * normal
        )
        B = np.hstack((np.eye(3), -_skew(point - reference_position)))
        closure_row = normal @ B
        closure_mode = closure_row @ displacement_basis
        closure_norm_squared = float(np.dot(closure_mode, closure_mode))
        if closure_norm_squared <= 0.0:
            continue
        target = (
            spine.initial_gap_m
            + float(candidate.signed_gap_m)
            + probe
            - float(np.dot(closure_row, q_C))
        )
        if target <= 0.0:
            continue
        score = float(np.dot(closure_mode, desired_mode)) / math.sqrt(
            closure_norm_squared
        )
        if score <= best_score:
            continue
        delta = (target / closure_norm_squared) * closure_mode
        seeded = q_C + displacement_basis @ delta
        if np.all(np.isfinite(seeded)) and not np.allclose(
            seeded, q_C, atol=0.0, rtol=0.0
        ):
            best_score = score
            best = seeded, key
    return best


def _active_set_signature(
    spines: Sequence[SpineInstance], states: Sequence[SpineAcceptedState]
) -> tuple[tuple[str, str | None, str, str | None, str | None], ...]:
    signature: list[
        tuple[str, str | None, str, str | None, str | None]
    ] = []
    for spine, state in zip(spines, states, strict=True):
        candidate = _candidate_for_state(spine, state)
        signature.append(
            (
                state.physical_state.value,
                (
                    None
                    if state.contact_submode is None
                    else state.contact_submode.value
                ),
                state.spring_branch.value,
                state.candidate_id,
                None if candidate is None else candidate.candidate_id,
            )
        )
    return tuple(signature)


def _trial_event_location(trial: SingleSpineTrial) -> tuple[float, float]:
    locations: list[tuple[float, float]] = []
    for event in trial.result.events:
        event_load = (
            float(event.load_parameter)
            if event.load_parameter is not None
            and math.isfinite(float(event.load_parameter))
            else float(trial.result.evaluated_motion.load_parameter)
        )
        event_fraction = float(event.details.get("event_fraction", 1.0))
        if not math.isfinite(event_fraction):
            event_fraction = 1.0
        locations.append((event_load, event_fraction))
    if locations:
        return min(locations)
    if trial.result.failure is not None:
        return (float(trial.result.evaluated_motion.load_parameter), 1.0)
    return (math.inf, math.inf)


def _stops_at_model_limit(trial: SingleSpineTrial) -> bool:
    failure = trial.result.failure
    return bool(
        failure is not None
        and failure.continuation_action
        is ContinuationAction.STOP_MODEL_LIMIT
    )


def _select_earliest_event_indices(
    spines: Sequence[SpineInstance],
    trials: Sequence[SingleSpineTrial],
    q_references: NDArray[np.float64],
    target_q_C: Vector6,
    step_start_q_C: Vector6,
    coordinate_scale: Vector6,
) -> set[int]:
    eventful = tuple(
        (index, trial, _trial_event_location(trial))
        for index, trial in enumerate(trials)
        if (trial.result.events and trial.committable)
        or _stops_at_model_limit(trial)
    )
    if not eventful:
        return set()
    event_tolerance = max(
        spine.tolerances.event_fraction for spine in spines
    )
    earliest_load = min(item[2][0] for item in eventful)
    at_earliest_load = tuple(
        item
        for item in eventful
        if abs(item[2][0] - earliest_load)
        <= event_tolerance * max(1.0, abs(earliest_load))
    )
    if len(at_earliest_load) == 1:
        return {at_earliest_load[0][0]}
    event_positions: list[tuple[int, NDArray[np.float64]]] = []
    for index, _trial, location in at_earliest_load:
        fraction = location[1] if math.isfinite(location[1]) else 1.0
        fraction = float(np.clip(fraction, 0.0, 1.0))
        event_positions.append(
            (
                index,
                q_references[index]
                + fraction * (target_q_C - q_references[index]),
            )
        )
    scaled_positions = [
        (index, (position - step_start_q_C) / coordinate_scale)
        for index, position in event_positions
    ]
    position_tolerance = event_tolerance * max(
        1.0,
        *(float(np.linalg.norm(value)) for _index, value in scaled_positions),
    )
    first_position = scaled_positions[0][1]
    if all(
        float(np.linalg.norm(value - first_position)) <= position_tolerance
        for _index, value in scaled_positions[1:]
    ):
        return {index for index, _value in scaled_positions}
    step_delta = (target_q_C - step_start_q_C) / coordinate_scale
    step_norm_squared = float(np.dot(step_delta, step_delta))
    if step_norm_squared <= np.finfo(np.float64).eps:
        raise ConfigurationError(
            "same-load event positions are not comparable at zero global step"
        )
    progress: list[tuple[int, float]] = []
    for index, position in scaled_positions:
        value = float(np.dot(position, step_delta) / step_norm_squared)
        off_path = position - value * step_delta
        if float(np.linalg.norm(off_path)) > position_tolerance:
            raise ConfigurationError(
                "same-load event positions are not comparable on the global secant"
            )
        progress.append((index, value))
    earliest_progress = min(value for _index, value in progress)
    return {
        index
        for index, value in progress
        if abs(value - earliest_progress) <= event_tolerance
    }


def _nullspace_basis(
    free: NDArray[np.int64], control: MixedControl, tolerances: ArrayTolerances
) -> NDArray[np.float64]:
    if free.size == 0:
        return np.empty((6, 0), dtype=np.float64)
    selector = np.eye(6, dtype=np.float64)[:, free]
    if not control.equality_matrix:
        return selector
    equality = np.asarray(control.equality_matrix, dtype=np.float64)
    coordinate_scale = np.diag([control.L_ref_m] * 3 + [1.0] * 3)
    scaled = equality @ coordinate_scale @ selector
    _u, singular, vt = np.linalg.svd(scaled, full_matrices=True)
    leading = float(singular[0]) if singular.size else 0.0
    threshold = tolerances.rank_absolute + tolerances.rank_relative * leading
    rank = int(np.count_nonzero(singular > threshold))
    return selector @ vt[rank:].T


def _enforce_equality_constraints(
    q_C: Vector6,
    free: NDArray[np.int64],
    control: MixedControl,
    tolerances: ArrayTolerances,
) -> Vector6:
    """Project the initial free coordinates onto homogeneous equalities."""

    if not control.equality_matrix:
        return q_C
    equality = np.asarray(control.equality_matrix, dtype=np.float64)
    residual = equality @ q_C
    row_norm = np.linalg.norm(equality, axis=1)
    nonzero = row_norm > 0.0
    if not np.any(nonzero):
        return q_C
    normalized_residual = residual[nonzero] / row_norm[nonzero]
    if float(np.linalg.norm(normalized_residual)) <= tolerances.range_residual:
        return q_C
    if free.size == 0:
        raise ConfigurationError(
            "prescribed pose is incompatible with equality_matrix"
        )
    selector = np.eye(6, dtype=np.float64)[:, free]
    coordinate_scale = np.diag([control.L_ref_m] * 3 + [1.0] * 3)
    operator = equality @ coordinate_scale @ selector
    operator = operator[nonzero] / row_norm[nonzero, None]
    correction, _residuals, _rank, _singular = np.linalg.lstsq(
        operator, -normalized_residual, rcond=None
    )
    projected = q_C + coordinate_scale @ selector @ correction
    final = equality @ projected
    final_normalized = final[nonzero] / row_norm[nonzero]
    if float(np.linalg.norm(final_normalized)) > tolerances.range_residual:
        raise ConfigurationError(
            "prescribed pose/free DOFs cannot satisfy equality_matrix"
        )
    return projected


def _stability(
    tangent: Matrix6,
    assembly: _Assembly,
    spines: Sequence[SpineInstance],
    free: NDArray[np.int64],
    control: MixedControl,
    tolerances: ArrayTolerances,
    *,
    equilibrium_solved: bool,
) -> tuple[QuasistaticStability, float | None, float | None]:
    if not equilibrium_solved:
        return QuasistaticStability.NOT_EVALUATED, None, None
    slip_or_nonconservative = any(
        item.single_result.physical_state is PhysicalState.SLIP
        or (
            item.single_result.physical_state is PhysicalState.HARDSTOP
            and item.single_result.contact_submode is PhysicalState.SLIP
        )
        for item in assembly.per_spine
    ) or not np.allclose(tangent, tangent.T, atol=1e-10, rtol=1e-8)
    if slip_or_nonconservative:
        admissible = True
        for item, trial, spine in zip(
            assembly.per_spine, assembly.trials, spines, strict=True
        ):
            single = item.single_result
            residuals = single.complementarity_residuals
            local_tolerance = spine.tolerances
            admissible &= (
                trial.committable
                and single.numerical_state is NumericalState.CONVERGED
                and single.normal_force_N >= -local_tolerance.force_N
                and residuals.get("penetration_m", 0.0)
                <= local_tolerance.gap_m
                and residuals.get("negative_normal_N", 0.0)
                <= local_tolerance.force_N
                and residuals.get("gap_force_Nm", 0.0)
                <= local_tolerance.gap_m * local_tolerance.force_N
                and residuals.get("static_friction_cone_N", 0.0)
                <= local_tolerance.friction_N
                and residuals.get("kinetic_slip_cone_N", 0.0)
                <= local_tolerance.friction_N
                and residuals.get("spring", 0.0)
                <= local_tolerance.spring_N
                and residuals.get("balance_m", 0.0)
                <= local_tolerance.gap_m
            )
            if single.physical_state is PhysicalState.SLIP or (
                single.physical_state is PhysicalState.HARDSTOP
                and single.contact_submode is PhysicalState.SLIP
            ):
                velocity = np.asarray(
                    single.evaluated_motion.relative_tangential_velocity_m_per_s,
                    dtype=np.float64,
                )
                dissipation = float(
                    np.dot(single.tangential_force_N, velocity)
                )
                admissible &= dissipation <= local_tolerance.force_N * max(
                    local_tolerance.velocity_m_per_s,
                    float(np.linalg.norm(velocity)),
                )
        return (
            QuasistaticStability.DIRECTIONALLY_ADMISSIBLE_QUASISTATIC
            if admissible
            else QuasistaticStability.NOT_EVALUATED,
            None,
            None,
        )
    return evaluate_conservative_stability(
        tangent, free, control, tolerances
    )


def evaluate_conservative_stability(
    tangent: ArrayLike,
    free_dofs: ArrayLike,
    control: MixedControl,
    tolerances: ArrayTolerances = ArrayTolerances(),
) -> tuple[QuasistaticStability, float | None, float | None]:
    """Evaluate the scaled equality-constrained conservative free subspace."""

    tangent_matrix = _matrix(tangent, 6, "tangent")
    free = np.asarray(free_dofs, dtype=np.int64)
    if free.ndim != 1 or np.any(free < 0) or np.any(free >= 6):
        raise ConfigurationError("free_dofs must be a 1-D subset of 0..5")
    if np.unique(free).size != free.size:
        raise ConfigurationError("free_dofs cannot contain duplicates")
    basis = _nullspace_basis(free, control, tolerances)
    if basis.shape[1] == 0:
        return QuasistaticStability.NO_FREE_MODE, None, None
    D_q = np.diag([control.L_ref_m] * 3 + [1.0] * 3)
    K_hat = D_q.T @ tangent_matrix @ D_q / (
        control.F_ref_N * control.L_ref_m
    )
    reduced = basis.T @ (0.5 * (K_hat + K_hat.T)) @ basis
    eigenvalues = np.linalg.eigvalsh(reduced)
    minimum = float(eigenvalues.min())
    threshold = tolerances.stability_absolute + tolerances.stability_relative * float(
        np.linalg.norm(reduced, ord=2)
    )
    if minimum > threshold:
        status = QuasistaticStability.STABLE_CONSERVATIVE
    elif minimum < -threshold:
        status = QuasistaticStability.UNSTABLE_CONSERVATIVE
    else:
        status = QuasistaticStability.MARGINAL_CONSERVATIVE
    return status, minimum, threshold


def solve_array_equilibrium(
    spines: Sequence[SpineInstance],
    accepted: ArrayAcceptedState,
    control: MixedControl,
    *,
    load_parameter: float,
    tolerances: ArrayTolerances = ArrayTolerances(),
) -> ArrayTrial:
    """Solve one 6-D array trial using only canonical single-spine trials."""

    if not spines:
        raise ConfigurationError("array must contain at least one spine")
    if len(spines) != len(accepted.spine_states):
        raise ConfigurationError("array state count does not match spine count")
    ids = [spine.spine_id for spine in spines]
    state_ids = [state.spine_id for state in accepted.spine_states]
    if ids != state_ids or len(ids) != len(set(ids)):
        raise ConfigurationError("array spine IDs/order must be unique and stable")
    step_start_q = _vector(
        accepted.q_C if accepted.revision > 0 else control.initial_q_C,
        6,
        "accepted/initial q_C",
    ).copy()
    q_C, required, _prescribed, free = _control_vectors(control, accepted)
    q_C = _enforce_equality_constraints(q_C, free, control, tolerances)
    admissible_basis = _nullspace_basis(free, control, tolerances)
    loader = _matrix(control.loader_stiffness, 6, "loader_stiffness")
    row_scale = np.array(
        [1.0 / control.F_ref_N] * 3
        + [1.0 / (control.F_ref_N * control.L_ref_m)] * 3
    )
    coordinate_scale = np.array([control.L_ref_m] * 3 + [1.0] * 3)
    rank_status = RankStatus.FULL_RANK
    range_status = RangeStatus.COMPATIBLE
    equilibrium_status = EquilibriumStatus.NONCONVERGED
    rank = 0
    singular = np.empty(0)
    range_residual = math.inf
    scaled_residual = math.inf
    assembly: _Assembly | None = None
    iterations = 0
    contact_seed_attempts: dict[
        tuple[
            tuple[str, str | None, str, str | None, str | None],
            ...,
        ],
        set[tuple[int, str]],
    ] = {}
    working_states = list(accepted.spine_states)
    working_q_references = np.repeat(
        step_start_q[None, :], len(spines), axis=0
    )
    failed_indices: set[int] = set()
    cascade_events: list[Event] = []
    last_event_results: dict[int, SingleSpineResult] = {}
    cascade_released: list[Mapping[str, Any]] = []
    rebalance_predictions: list[Mapping[str, Any]] = []
    pending_release: Vector6 | None = None
    pending_spine_ids: tuple[str, ...] = ()
    terminal_events: tuple[Event, ...] = ()
    last_evaluated_q_C = q_C.copy()
    iteration_exhausted = False

    def assemble_model_limit_boundary(
        source: _Assembly,
        selected_indices: set[int],
        target_q_C: Vector6,
    ) -> tuple[Vector6, _Assembly, tuple[Event, ...]]:
        terminal_index = next(
            index
            for index in sorted(selected_indices)
            if _stops_at_model_limit(source.trials[index])
        )
        events = tuple(
            event
            for index in sorted(selected_indices)
            for event in source.trials[index].result.events
        )
        terminal_trial = source.trials[terminal_index]
        fraction = _trial_event_location(terminal_trial)[1]
        if not math.isfinite(fraction):
            fraction = 1.0
        fraction = float(np.clip(fraction, 0.0, 1.0))
        start_q_C = working_q_references[terminal_index]
        boundary_q_C = start_q_C + fraction * (
            target_q_C - start_q_C
        )
        boundary_q_C = _enforce_equality_constraints(
            boundary_q_C, free, control, tolerances
        )
        boundary_load = float(
            terminal_trial.result.evaluated_motion.load_parameter
        )
        boundary = _assemble(
            spines,
            working_states,
            boundary_q_C,
            control,
            load_parameter=boundary_load,
        )
        return boundary_q_C, boundary, events

    for iterations in range(1, tolerances.maximum_iterations + 1):
        last_evaluated_q_C = q_C.copy()
        assembly = _assemble(
            spines,
            working_states,
            q_C,
            control,
            load_parameter=load_parameter,
        )
        if pending_release is not None:
            postfailure_tangent = assembly.tangent + loader
            (
                prediction_rank,
                prediction_range,
                _prediction_rank_value,
                _prediction_singular,
                prediction_range_residual,
                prediction_matrix,
            ) = _scaled_rank_range(
                postfailure_tangent,
                -pending_release,
                admissible_basis,
                control,
                tolerances,
            )
            predicted_delta: Vector6 | None = None
            if (
                prediction_rank is RankStatus.FULL_RANK
                and prediction_range is RangeStatus.COMPATIBLE
                and admissible_basis.shape[1] > 0
            ):
                reduced_release = admissible_basis.T @ (
                    row_scale * pending_release
                )
                delta_scaled = np.linalg.solve(
                    prediction_matrix, reduced_release
                )
                predicted_delta = coordinate_scale * (
                    admissible_basis @ delta_scaled
                )
                q_C += predicted_delta
            elif admissible_basis.shape[1] == 0:
                predicted_delta = np.zeros(6, dtype=np.float64)
            rebalance_predictions.append(
                {
                    "equation": "(K_L+K_R)delta_q_C=f_R_gen_minus",
                    "load_parameter": float(load_parameter),
                    "failed_spine_ids": pending_spine_ids,
                    "released_force_N": _tuple_vector(pending_release[:3]),
                    "released_moment_Nm": _tuple_vector(pending_release[3:]),
                    "rank_status": prediction_rank.value,
                    "range_status": prediction_range.value,
                    "range_residual_norm": prediction_range_residual,
                    "delta_q_C": (
                        None
                        if predicted_delta is None
                        else _tuple_vector(predicted_delta)
                    ),
                }
            )
            pending_release = None
            pending_spine_ids = ()
            if predicted_delta is not None and np.any(predicted_delta != 0.0):
                continue
        selected_event_indices = _select_earliest_event_indices(
            spines,
            assembly.trials,
            working_q_references,
            q_C,
            step_start_q,
            coordinate_scale,
        )
        if any(
            _stops_at_model_limit(assembly.trials[index])
            for index in selected_event_indices
        ):
            q_C, assembly, terminal_events = (
                assemble_model_limit_boundary(
                    assembly, selected_event_indices, q_C
                )
            )
            equilibrium_status = EquilibriumStatus.MODEL_LIMIT
            break
        newly_failed: list[int] = []
        eventful_indices: list[int] = []
        for index, trial in enumerate(assembly.trials):
            failure = trial.result.failure
            if index in selected_event_indices:
                event_fraction = _trial_event_location(trial)[1]
                if not math.isfinite(event_fraction):
                    event_fraction = 1.0
                event_fraction = float(
                    np.clip(event_fraction, 0.0, 1.0)
                )
                working_q_references[index] += event_fraction * (
                    q_C - working_q_references[index]
                )
                working_states[index] = trial.proposed_state
                cascade_events.extend(trial.result.events)
                last_event_results[index] = trial.result
                eventful_indices.append(index)
            if (
                index in selected_event_indices
                and index not in failed_indices
                and failure is not None
                and failure.continuation_action
                is ContinuationAction.PERMANENT_REMOVE
            ):
                failed_indices.add(index)
                newly_failed.append(index)
        if newly_failed:
            new_ids = tuple(spines[index].spine_id for index in newly_failed)
            releases = tuple(
                item
                for item in assembly.released_wrenches
                if item.get("spine_id") in new_ids
            )
            cascade_released.extend(releases)
            pending_release = np.zeros(6, dtype=np.float64)
            for item in releases:
                pending_release[:3] += _vector(
                    item["force_N"], 3, "released force"
                )
                pending_release[3:] += _vector(
                    item["moment_Nm"], 3, "released moment"
                )
            pending_spine_ids = new_ids
        if eventful_indices:
            continue
        residual = assembly.support_wrench + loader @ q_C - required
        scaled_residual = (
            0.0
            if admissible_basis.shape[1] == 0
            else float(
                np.linalg.norm(
                    admissible_basis.T @ (row_scale * residual)
                )
            )
        )
        total_tangent = assembly.tangent + loader
        (
            rank_status,
            range_status,
            rank,
            singular,
            range_residual,
            scaled_matrix,
        ) = _scaled_rank_range(
            total_tangent,
            residual,
            admissible_basis,
            control,
            tolerances,
        )
        if admissible_basis.shape[1] == 0:
            equilibrium_status = EquilibriumStatus.SOLVED
            break
        seed_signature = _active_set_signature(spines, working_states)
        if rank_status is RankStatus.RANK_DEFICIENT:
            attempted = contact_seed_attempts.setdefault(seed_signature, set())
            prediction = _predict_contact_seed(
                spines,
                working_states,
                q_C,
                admissible_basis,
                control,
                residual,
                attempted,
            )
            if prediction is not None:
                seeded, key = prediction
                attempted.add(key)
                q_C = _enforce_equality_constraints(
                    seeded, free, control, tolerances
                )
                continue
        if range_status is RangeStatus.INCOMPATIBLE:
            equilibrium_status = EquilibriumStatus.RANGE_INCOMPATIBLE
            break
        if rank_status is RankStatus.RANK_DEFICIENT:
            equilibrium_status = EquilibriumStatus.RANK_DEFICIENT
            break
        if scaled_residual <= tolerances.scaled_residual:
            equilibrium_status = EquilibriumStatus.SOLVED
            break
        delta_scaled = np.linalg.solve(
            scaled_matrix,
            -admissible_basis.T @ (row_scale * residual),
        )
        q_C += coordinate_scale * (admissible_basis @ delta_scaled)
    else:
        iteration_exhausted = True
        # A Newton update on the final permitted iteration changes q_C after
        # the assembly was evaluated. Reassemble for a coherent diagnostic
        # snapshot, but retain the last event-free iterate if the new pose
        # exposes an event that the exhausted loop cannot process.
        final_assembly = _assemble(
            spines,
            working_states,
            q_C,
            control,
            load_parameter=load_parameter,
        )
        final_selected_indices = _select_earliest_event_indices(
            spines,
            final_assembly.trials,
            working_q_references,
            q_C,
            step_start_q,
            coordinate_scale,
        )
        if any(
            _stops_at_model_limit(final_assembly.trials[index])
            for index in final_selected_indices
        ):
            q_C, assembly, terminal_events = (
                assemble_model_limit_boundary(
                    final_assembly, final_selected_indices, q_C
                )
            )
            equilibrium_status = EquilibriumStatus.MODEL_LIMIT
        elif any(
            trial.result.events or trial.result.failure is not None
            for trial in final_assembly.trials
        ):
            q_C = last_evaluated_q_C
        else:
            assembly = final_assembly
    assert assembly is not None
    residual = assembly.support_wrench + loader @ q_C - required
    scaled_residual = (
        0.0
        if admissible_basis.shape[1] == 0
        else float(
            np.linalg.norm(
                admissible_basis.T @ (row_scale * residual)
            )
        )
    )
    (
        rank_status,
        range_status,
        rank,
        singular,
        range_residual,
        _scaled_matrix,
    ) = _scaled_rank_range(
        assembly.tangent + loader,
        residual,
        admissible_basis,
        control,
        tolerances,
    )
    total_tangent = assembly.tangent + loader
    final_per_spine = tuple(
        replace(item, single_result=last_event_results[index])
        if (
            index in last_event_results
            and last_event_results[index].physical_state
            is item.single_result.physical_state
            and last_event_results[index].evaluated_motion
            == item.single_result.evaluated_motion
        )
        else item
        for index, item in enumerate(assembly.per_spine)
    )
    equilibrium_solved = equilibrium_status is EquilibriumStatus.SOLVED
    local_trials_solved = all(
        trial.committable
        and trial.result.numerical_state is NumericalState.CONVERGED
        for trial in assembly.trials
    )
    stability, minimum_eigenvalue, stability_threshold = _stability(
        total_tangent,
        assembly,
        spines,
        free,
        control,
        tolerances,
        equilibrium_solved=equilibrium_solved and local_trials_solved,
    )
    model_state = ModelState.CLOSED
    if equilibrium_status is EquilibriumStatus.MODEL_LIMIT:
        model_state = ModelState.OUT_OF_SCOPE
    elif any(
        trial.result.model_state is ModelState.PARAMETER_UNCLOSED
        for trial in assembly.trials
    ):
        model_state = ModelState.PARAMETER_UNCLOSED
    local_numerical_states = tuple(
        trial.result.numerical_state for trial in assembly.trials
    )
    if (
        equilibrium_solved
        and all(trial.committable for trial in assembly.trials)
        and all(
            state is NumericalState.CONVERGED
            for state in local_numerical_states
        )
    ):
        numerical_state = NumericalState.CONVERGED
    elif NumericalState.INVALID_RESIDUAL in local_numerical_states:
        numerical_state = NumericalState.INVALID_RESIDUAL
    else:
        numerical_state = NumericalState.NONCONVERGED
    total_wrench = Wrench(
        force_N=_tuple_vector(assembly.support_wrench[:3]),
        moment_Nm=_tuple_vector(assembly.support_wrench[3:]),
        frame=control.frame,
        reference_point=control.reference_point,
        acting_on=control.backplate_object,
        exerted_by="spine_array",
    )
    diagnostics = EquilibriumDiagnostics(
        iterations=iterations,
        scaled_residual_norm=(
            scaled_residual if math.isfinite(scaled_residual) else None
        ),
        scaled_rank=rank,
        free_dof_count=int(free.size),
        singular_values=_tuple_vector(singular),
        range_residual_norm=(
            range_residual if math.isfinite(range_residual) else None
        ),
        minimum_stability_eigenvalue=minimum_eigenvalue,
        stability_threshold=stability_threshold,
        assembled_spine_count=len(spines),
        admissible_free_mode_count=admissible_basis.shape[1],
    )
    if equilibrium_status is EquilibriumStatus.MODEL_LIMIT:
        final_events = terminal_events
    elif iteration_exhausted:
        final_events = ()
    else:
        final_events = tuple(
            event
            for index, trial in enumerate(assembly.trials)
            if index not in failed_indices
            for event in trial.result.events
        )
    events = tuple(cascade_events) + final_events
    result = ArrayResult(
        q_C=_tuple_vector(q_C),
        physical_backplate_pose=_tuple_vector(-q_C),
        total_wrench=total_wrench,
        tangent=_tuple_matrix(total_tangent),
        per_spine=final_per_spine,
        counts=assembly.counts,
        rank_status=rank_status,
        range_status=range_status,
        equilibrium_status=equilibrium_status,
        quasistatic_stability=stability,
        dynamic_stability=ModelState.OUT_OF_SCOPE,
        model_state=model_state,
        numerical_state=numerical_state,
        diagnostics=diagnostics,
        events=events,
        released_wrenches=tuple(cascade_released),
        rebalance_predictions=tuple(rebalance_predictions),
        assumptions=(
            "rigid_backplate",
            "quasistatic",
            "fixed_wall_physical_pose_is_negative_q_C",
            "point_contact",
        ),
        omissions=(
            "mass_and_damping",
            "dynamic_stability",
            "continuous_backplate_bending",
            "geometric_stiffness_unless_in_local_tangent",
        ),
    )
    acceptable_stability = stability in {
        QuasistaticStability.STABLE_CONSERVATIVE,
        QuasistaticStability.NO_FREE_MODE,
        QuasistaticStability.DIRECTIONALLY_ADMISSIBLE_QUASISTATIC,
    }
    committable = (
        equilibrium_solved
        and acceptable_stability
        and all(trial.committable for trial in assembly.trials)
        and equilibrium_status is not EquilibriumStatus.MODEL_LIMIT
    )
    if committable:
        proposed_states = tuple(
            commit_single_spine_trial(state, trial)
            for state, trial in zip(
                working_states, assembly.trials, strict=True
            )
        )
        proposed = ArrayAcceptedState(
            q_C=_tuple_vector(q_C),
            spine_states=proposed_states,
            load_parameter=float(load_parameter),
            revision=accepted.revision + 1,
        )
    else:
        proposed = accepted
    return ArrayTrial(accepted.revision, proposed, result, committable)


def commit_array_trial(
    accepted: ArrayAcceptedState, trial: ArrayTrial
) -> ArrayAcceptedState:
    if trial.base_revision != accepted.revision:
        raise ConfigurationError("stale array trial cannot be committed")
    if not trial.committable:
        raise ConfigurationError("array trial is not a completed admissible equilibrium")
    return trial.proposed_state


__all__ = [
    "ArrayAcceptedState",
    "ArrayResult",
    "ArrayTolerances",
    "ArrayTrial",
    "ControlMode",
    "EquilibriumDiagnostics",
    "EquilibriumStatus",
    "MixedControl",
    "PerSpineArrayResult",
    "QuasistaticStability",
    "RangeStatus",
    "RankStatus",
    "SpineInstance",
    "commit_array_trial",
    "evaluate_conservative_stability",
    "solve_array_equilibrium",
]
