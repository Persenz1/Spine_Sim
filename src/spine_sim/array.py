"""刚性共同背板阵列的 O(N) 装配、六自由度平衡与事件 trial 求解器。

每根刺仍由唯一的标准单刺求解器计算；阵列层只负责刚性运动映射、广义 wrench/
切线装配、混合控制、活动集事件级联、失效后重平衡和准静态稳定性。
"""

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
    EventType,
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
    solve_single_spine,
)


Vector6 = NDArray[np.float64]
Matrix6 = NDArray[np.float64]


class ControlMode(StrEnum):
    """每个广义分量选择位姿控制或 wrench 控制。"""

    PRESCRIBED_POSE = "prescribed_pose"
    REQUIRED_WRENCH = "required_wrench"


class RankStatus(StrEnum):
    """自由子空间内缩放切线矩阵的秩状态。"""

    FULL_RANK = "full_rank"
    RANK_DEFICIENT = "rank_deficient"


class RangeStatus(StrEnum):
    """当前残差是否位于缩放切线的可解值域。"""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"


class EquilibriumStatus(StrEnum):
    """阵列平衡求解的终止原因。"""

    SOLVED = "solved"
    RANK_DEFICIENT = "rank_deficient"
    RANGE_INCOMPATIBLE = "range_incompatible"
    NONCONVERGED = "nonconverged"
    MODEL_LIMIT = "model_limit"


class QuasistaticStability(StrEnum):
    """保守自由模态或有滑移方向时的准静态稳定性结论。"""

    STABLE_CONSERVATIVE = "stable_conservative"
    MARGINAL_CONSERVATIVE = "marginal_conservative"
    UNSTABLE_CONSERVATIVE = "unstable_conservative"
    NO_FREE_MODE = "no_free_mode"
    DIRECTIONALLY_ADMISSIBLE_QUASISTATIC = (
        "directionally_admissible_quasistatic"
    )
    NOT_EVALUATED = "not_evaluated"


def _vector(value: ArrayLike, length: int, name: str) -> NDArray[np.float64]:
    """校验指定长度的有限向量。"""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ConfigurationError(f"{name} must be a finite {length}-vector")
    return result


def _matrix(value: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    """校验指定阶数的有限方阵。"""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size, size) or not np.all(np.isfinite(result)):
        raise ConfigurationError(f"{name} must be a finite {size}x{size} matrix")
    return result


def _tuple_vector(value: ArrayLike) -> tuple[float, ...]:
    """把 NumPy 向量转换为冻结结果可保存的标量元组。"""

    return tuple(float(item) for item in np.asarray(value, dtype=np.float64))


def _tuple_matrix(value: ArrayLike) -> tuple[tuple[float, ...], ...]:
    """把 NumPy 矩阵转换为嵌套元组。"""

    return tuple(
        tuple(float(item) for item in row)
        for row in np.asarray(value, dtype=np.float64)
    )


def _skew(value: NDArray[np.float64]) -> NDArray[np.float64]:
    """返回满足 ``skew(r) @ v = r × v`` 的反对称矩阵。"""

    x, y, z = value
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ArrayTolerances:
    """平衡残差、秩/值域、稳定性、间隙和迭代次数容差。"""

    scaled_residual: float = 1e-9
    rank_absolute: float = 1e-12
    rank_relative: float = 1e-10
    range_residual: float = 1e-9
    stability_absolute: float = 1e-10
    stability_relative: float = 1e-8
    gap_m: float = 1e-10
    maximum_iterations: int = 30

    def __post_init__(self) -> None:
        """校验所有数值容差为正，并至少允许一次迭代。"""

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
    """六个广义分量各自且仅选择一个位姿或 wrench 控制契约。

    ``q_C`` 表示接触约束相对背板的加载坐标；固定墙面时真实背板位姿为 ``-q_C``。
    ``equality_matrix`` 可进一步约束 wrench 控制留下的自由子空间。
    """

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
        """检查六维控制互斥性、加载器半正定性、尺度和语义标签。"""

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
    """阵列中一根刺的全部本构输入、初始间隙和候选 continuation。"""

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
        """校验间隙、搜索增量及候选 ID/序号的严格有序性。"""

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
        indices = [item.candidate_index for item in candidates]
        if len(ids) != len(set(ids)) or indices != sorted(set(indices)):
            raise ConfigurationError(
                "candidate continuation must have unique IDs and increasing indices"
            )

    @property
    def spine_id(self) -> str:
        """代理几何对象中的稳定单刺 ID。"""

        return self.geometry.spine_id


@dataclass(frozen=True)
class ArrayAcceptedState:
    """共同 ``q_C``、逐刺 accepted 状态、载荷参数和阵列 revision。"""

    q_C: tuple[float, ...]
    spine_states: tuple[SpineAcceptedState, ...]
    load_parameter: float
    revision: int = 0

    def __post_init__(self) -> None:
        """统一六维坐标并检查逐刺 ID 唯一、载荷与 revision 有效。"""

        q_C = _vector(self.q_C, 6, "q_C")
        if not self.spine_states:
            raise ConfigurationError("array state must contain at least one spine")
        state_ids = [state.spine_id for state in self.spine_states]
        if len(state_ids) != len(set(state_ids)):
            raise ConfigurationError("array state spine IDs must be unique")
        load_parameter = float(self.load_parameter)
        if not math.isfinite(load_parameter):
            raise ConfigurationError("load_parameter must be finite")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, (int, np.integer))
            or self.revision < 0
        ):
            raise ConfigurationError("revision must be a non-negative integer")
        object.__setattr__(self, "q_C", _tuple_vector(q_C))
        object.__setattr__(self, "spine_states", tuple(self.spine_states))
        object.__setattr__(self, "load_parameter", load_parameter)
        object.__setattr__(self, "revision", int(self.revision))

    @classmethod
    def initial(
        cls, spines: Sequence[SpineInstance], *, load_parameter: float = 0.0
    ) -> "ArrayAcceptedState":
        """为给定有序刺集合创建零位姿、revision 0 的阵列状态。"""

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
    """单刺结果及其从几何间隙到阵列加载位移的映射诊断。"""

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
    """缩放残差、秩、值域、稳定性和装配规模诊断。"""

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
    """一个阵列 trial 的平衡、逐刺、事件、稳定性和模型边界输出。"""

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
    """基于某一阵列 revision 的不可变平衡提案。"""

    base_revision: int
    proposed_state: ArrayAcceptedState
    result: ArrayResult
    committable: bool


@dataclass(frozen=True)
class _Assembly:
    """某个 ``q_C`` 上 O(N) 累加得到的内部装配快照。"""

    support_wrench: Vector6
    tangent: Matrix6
    trials: tuple[SingleSpineTrial, ...]
    per_spine: tuple[PerSpineArrayResult, ...]
    counts: ArrayCounts
    released_wrenches: tuple[Mapping[str, Any], ...]


def _candidate_for_state(
    spine: SpineInstance, accepted: SpineAcceptedState
) -> ContactCandidate | None:
    """按单刺 resident state 和 continuation cursor 选择当前候选。"""

    candidates = tuple(
        item
        for item in (spine.candidate, *spine.continuation_candidates)
        if item is not None
    )
    if accepted.physical_state is PhysicalState.FAILED:
        return None
    if accepted.physical_state in {
        PhysicalState.STICK,
        PhysicalState.SLIP,
        PhysicalState.HARDSTOP,
    }:
        # 已承载接触必须继续使用同一 candidate_id，不能在 Newton 中跳换 feature。
        candidate = next(
            (
                item
                for item in candidates
                if item.candidate_id == accepted.candidate_id
            ),
            None,
        )
        if candidate is None:
            raise ConfigurationError(
                "active spine candidate_id is missing from its candidate sequence"
            )
        return candidate
    if accepted.physical_state in {
        PhysicalState.DETACH,
        PhysicalState.REBOUND,
    }:
        return None
    if not candidates:
        return None
    cursor = accepted.search_cursor
    if cursor is None:
        return candidates[0]
    if cursor.exhausted:
        return None
    return next(
        (
            item
            for item in candidates
            if item.candidate_index >= cursor.candidate_index
        ),
        None,
    )


def _control_vectors(
    control: MixedControl,
    initial_q_C: Vector6,
) -> tuple[Vector6, Vector6, NDArray[np.int64]]:
    """应用位姿控制值，并返回 required wrench 与自由分量索引。"""

    q = initial_q_C.copy()
    required = np.zeros(6, dtype=np.float64)
    free: list[int] = []
    for index, mode in enumerate(control.modes):
        if mode is ControlMode.PRESCRIBED_POSE:
            q[index] = float(control.prescribed_q_C[index])
        else:
            required[index] = float(control.required_wrench[index])
            free.append(index)
    return (
        q,
        required,
        np.asarray(free, dtype=np.int64),
    )


def _scaled_rank_range(
    matrix: Matrix6,
    residual: Vector6,
    basis: NDArray[np.float64],
    row_scale: Vector6,
    coordinate_scale: Vector6,
    tolerances: ArrayTolerances,
) -> tuple[RankStatus, RangeStatus, int, NDArray[np.float64], float, NDArray[np.float64]]:
    """在等式约束自由基中评估无量纲切线秩和残差值域相容性。"""

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
    # 行尺度把力/力矩化为 F_ref 基准，列尺度把平移/转动化为 L_ref 基准。
    scaled_full = (
        row_scale[:, None]
        * matrix
        * coordinate_scale[None, :]
    )
    scaled = basis.T @ scaled_full @ basis
    rhs = -basis.T @ (row_scale * residual)
    u, singular, _vt = np.linalg.svd(scaled, full_matrices=True)
    # 残差在列空间外的分量即值域不相容量；它与“矩阵欠秩”是两个独立结论。
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
    """把单刺物理状态转换为计数与载荷分担指标所需的最小输入。"""

    geometric = candidate is not None and candidate.valid
    signed_gap = None if candidate is None else candidate.signed_gap_m
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
        # 摩擦可行性已由单刺本构解析；局部安全坡度/方向性仍取决于具体查询路径，
        # 因而由 stable_engagement 显式输入。缺失评估保持 unknown，不能强制转成 false。
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
    reference_position: NDArray[np.float64],
    q_rate: Vector6,
    resistance_direction: NDArray[np.float64],
    *,
    load_parameter: float,
) -> _Assembly:
    """在给定六维 ``q_C`` 上调用每根单刺，并 O(N) 装配总 wrench 和 6×6 切线。"""

    total = np.zeros(6, dtype=np.float64)
    tangent = np.zeros((6, 6), dtype=np.float64)
    trials: list[SingleSpineTrial] = []
    per_spine: list[PerSpineArrayResult] = []
    metrics: list[SpineMetricInput] = []
    released: list[Mapping[str, Any]] = []
    for spine, accepted in zip(spines, states, strict=True):
        # B=[I,-skew(r)] 把背板小位移/小转角映射到当前接触点的三维相对位移。
        candidate = _candidate_for_state(spine, accepted)
        terrain_signed_gap: float | None = None
        closure_threshold: float | None = None
        loading_displacement: float | None = None
        if candidate is None or candidate.selected_normal is None:
            point = np.asarray(
                spine.geometry.root_position_m, dtype=np.float64
            )
            B = np.hstack((np.eye(3), -_skew(point - reference_position)))
            dynamic_candidate = candidate
            relative = B @ q_C
            signed_gap = None if candidate is None else candidate.signed_gap_m
        else:
            normal = candidate.selected_normal / np.linalg.norm(
                candidate.selected_normal
            )
            point = (
                candidate.sphere_center_m
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
            # 单边接触闭合后，继续加载由 BaseMotion 的弹性变形承担，几何 gap 保持 0。
            # 若把 overclosure 作为负地形间隙传入，单刺层会误拒绝本来合法的承载接触。
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
        # 虚功一致映射：局部切线 K_i 贡献为 Bᵀ K_i B，因此总装配仅随刺数线性增长。
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
                resistance_direction,
                spine.tolerances.force_N,
            )
        )
        failure = single.failure
        if (
            failure is not None
            and failure.continuation_action
            is ContinuationAction.PERMANENT_REMOVE
        ):
            # 永久断开后记录失效瞬间释放的广义 wrench，供下一轮重平衡预测使用。
            force_before = next(
                (
                    event.details.get("force_before_N")
                    for event in single.events
                    if event.event_type is EventType.MATERIAL_FAILURE
                ),
                None,
            )
            if force_before is None:
                raise ConfigurationError(
                    "permanent failure event is missing force_before_N"
                )
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
    reference_position: NDArray[np.float64],
    row_scale: Vector6,
    coordinate_scale: Vector6,
    residual: Vector6,
    attempted: set[tuple[int, str]],
) -> tuple[Vector6, tuple[int, str]] | None:
    """选择与残差方向相容的开放接触，在自由子空间内生成闭合 seed。"""

    if admissible_basis.shape[1] == 0:
        return None
    displacement_basis = coordinate_scale[:, None] * admissible_basis
    # desired_mode 是减少当前广义残差所需的自由模态方向。
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
        normal = candidate.selected_normal / np.linalg.norm(
            candidate.selected_normal
        )
        point = (
            candidate.sphere_center_m
            - spine.geometry.tip_radius_m * normal
        )
        B = np.hstack((np.eye(3), -_skew(point - reference_position)))
        # closure_row 给出该接触法向闭合量对 q_C 的线性响应。
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
    """冻结当前逐刺物理/弹簧/候选组合，用于限制同一活动集的 seed 尝试。"""

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
    """返回单刺 trial 最早事件的 ``(load_parameter, step_fraction)``。"""

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
    """该单刺 trial 是否触及不可继续模拟的材料模型边界。"""

    failure = trial.result.failure
    return (
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
    """在所有单刺 trial 中选出全局最早事件；同点事件允许同时提交。"""

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
    # 先比较载荷参数；同载荷事件再放回缩放后的全局 secant 上比较几何进度。
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
    free: NDArray[np.int64],
    control: MixedControl,
    tolerances: ArrayTolerances,
    coordinate_scale: Vector6,
) -> NDArray[np.float64]:
    """构造自由 DOF 中同时满足齐次 equality_matrix 的正交基。"""

    if free.size == 0:
        return np.empty((6, 0), dtype=np.float64)
    selector = np.eye(6, dtype=np.float64)[:, free]
    if not control.equality_matrix:
        return selector
    equality = np.asarray(control.equality_matrix, dtype=np.float64)
    scaled = equality @ np.diag(coordinate_scale) @ selector
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
    coordinate_scale: Vector6,
) -> Vector6:
    """仅调整自由分量，把初始 ``q_C`` 投影到齐次等式约束。"""

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
    D_q = np.diag(coordinate_scale)
    operator = equality @ D_q @ selector
    operator = operator[nonzero] / row_norm[nonzero, None]
    # 在无量纲自由坐标中求最小范数修正，随后再次验证约束残差。
    correction, _residuals, _rank, _singular = np.linalg.lstsq(
        operator, -normalized_residual, rcond=None
    )
    projected = q_C + D_q @ selector @ correction
    final = equality @ projected
    final_normalized = final[nonzero] / row_norm[nonzero]
    if float(np.linalg.norm(final_normalized)) > tolerances.range_residual:
        raise ConfigurationError(
            "prescribed pose/free DOFs cannot satisfy equality_matrix"
        )
    return projected


def _conservative_stability(
    tangent: Matrix6,
    admissible_basis: NDArray[np.float64],
    control: MixedControl,
    tolerances: ArrayTolerances,
    coordinate_scale: Vector6,
) -> tuple[QuasistaticStability, float | None, float | None]:
    """在 admissible 自由子空间中用对称切线最小特征值判定保守稳定性。"""

    if admissible_basis.shape[1] == 0:
        return QuasistaticStability.NO_FREE_MODE, None, None
    D_q = np.diag(coordinate_scale)
    K_hat = D_q.T @ tangent @ D_q / (
        control.F_ref_N * control.L_ref_m
    )
    # 只对无量纲切线的对称部分做能量判据，并排除控制/等式约束禁止的模态。
    reduced = admissible_basis.T @ (0.5 * (K_hat + K_hat.T)) @ admissible_basis
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


def _stability(
    tangent: Matrix6,
    assembly: _Assembly,
    spines: Sequence[SpineInstance],
    admissible_basis: NDArray[np.float64],
    control: MixedControl,
    tolerances: ArrayTolerances,
    coordinate_scale: Vector6,
    *,
    equilibrium_solved: bool,
) -> tuple[QuasistaticStability, float | None, float | None]:
    """保守接触走特征值判据；滑移/非对称切线只检查耗散方向可接受性。"""

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
        # 非保守切线不能用势能 Hessian 下结论，只验证摩擦力未沿速度方向供能。
        admissible = True
        for item, spine in zip(assembly.per_spine, spines, strict=True):
            single = item.single_result
            local_tolerance = spine.tolerances
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
    return _conservative_stability(
        tangent,
        admissible_basis,
        control,
        tolerances,
        coordinate_scale,
    )


def evaluate_conservative_stability(
    tangent: ArrayLike,
    free_dofs: ArrayLike,
    control: MixedControl,
    tolerances: ArrayTolerances = ArrayTolerances(),
) -> tuple[QuasistaticStability, float | None, float | None]:
    """公开评估缩放后、受 equality 约束的保守自由子空间稳定性。"""

    tangent_matrix = _matrix(tangent, 6, "tangent")
    free = np.asarray(free_dofs, dtype=np.int64)
    if free.ndim != 1 or np.any(free < 0) or np.any(free >= 6):
        raise ConfigurationError("free_dofs must be a 1-D subset of 0..5")
    if np.unique(free).size != free.size:
        raise ConfigurationError("free_dofs cannot contain duplicates")
    coordinate_scale = np.array([control.L_ref_m] * 3 + [1.0] * 3)
    basis = _nullspace_basis(
        free, control, tolerances, coordinate_scale
    )
    return _conservative_stability(
        tangent_matrix,
        basis,
        control,
        tolerances,
        coordinate_scale,
    )


def solve_array_equilibrium(
    spines: Sequence[SpineInstance],
    accepted: ArrayAcceptedState,
    control: MixedControl,
    *,
    load_parameter: float,
    tolerances: ArrayTolerances = ArrayTolerances(),
) -> ArrayTrial:
    """仅通过标准单刺 trial 求解一个六自由度阵列平衡 trial。"""

    # 阶段 1：验证刺序稳定，建立有量纲控制向量和无量纲缩放/约束自由基。
    if not spines:
        raise ConfigurationError("array must contain at least one spine")
    if len(spines) != len(accepted.spine_states):
        raise ConfigurationError("array state count does not match spine count")
    ids = [spine.spine_id for spine in spines]
    state_ids = [state.spine_id for state in accepted.spine_states]
    if ids != state_ids or len(ids) != len(set(ids)):
        raise ConfigurationError("array spine IDs/order must be unique and stable")
    load_parameter = float(load_parameter)
    if not math.isfinite(load_parameter):
        raise ConfigurationError("load_parameter must be finite")
    loader = np.asarray(control.loader_stiffness, dtype=np.float64)
    reference_position = np.asarray(
        control.reference_position_m, dtype=np.float64
    )
    q_rate = np.asarray(control.q_rate_C, dtype=np.float64)
    resistance_direction = np.asarray(
        control.resistance_direction, dtype=np.float64
    )
    resistance_direction /= np.linalg.norm(resistance_direction)
    step_start_q = np.asarray(
        accepted.q_C if accepted.revision > 0 else control.initial_q_C,
        dtype=np.float64,
    ).copy()
    q_C, required, free = _control_vectors(
        control, step_start_q
    )
    # 力行以 F_ref 缩放，力矩行以 F_ref*L_ref 缩放；平移坐标则以 L_ref 缩放。
    row_scale = np.array(
        [1.0 / control.F_ref_N] * 3
        + [1.0 / (control.F_ref_N * control.L_ref_m)] * 3
    )
    coordinate_scale = np.array([control.L_ref_m] * 3 + [1.0] * 3)
    q_C = _enforce_equality_constraints(
        q_C, free, control, tolerances, coordinate_scale
    )
    admissible_basis = _nullspace_basis(
        free, control, tolerances, coordinate_scale
    )
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
        """把全局步截断到最早 STOP_MODEL_LIMIT 事件并重装配边界状态。"""

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
            boundary_q_C,
            free,
            control,
            tolerances,
            coordinate_scale,
        )
        boundary_load = float(
            terminal_trial.result.evaluated_motion.load_parameter
        )
        boundary = _assemble(
            spines,
            working_states,
            boundary_q_C,
            control,
            reference_position,
            q_rate,
            resistance_direction,
            load_parameter=boundary_load,
        )
        return boundary_q_C, boundary, events

    # 阶段 2：Newton/活动集循环。每轮先完整装配，再处理释放、事件，最后才做平衡更新。
    for iterations in range(1, tolerances.maximum_iterations + 1):
        last_evaluated_q_C = q_C.copy()
        assembly = _assemble(
            spines,
            working_states,
            q_C,
            control,
            reference_position,
            q_rate,
            resistance_direction,
            load_parameter=load_parameter,
        )
        if pending_release is not None:
            # 永久失效释放 f_R^- 后，用 (K_L+K_R)δq=f_R^- 预测下一重平衡起点。
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
                row_scale,
                coordinate_scale,
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
            # 非永久材料失效没有后续损伤本构，只能在首次模型边界处终止。
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
                # 只提交全局最早事件；其余刺的 trial 仍基于旧状态，下一轮重新计算。
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
            # 汇总同一事件位置永久断开的刺，其释放 wrench 在下一轮统一重分配。
            new_ids = tuple(spines[index].spine_id for index in newly_failed)
            releases = tuple(
                item
                for item in assembly.released_wrenches
                if item["spine_id"] in new_ids
            )
            cascade_released.extend(releases)
            pending_release = np.zeros(6, dtype=np.float64)
            for item in releases:
                pending_release[:3] += item["force_N"]
                pending_release[3:] += item["moment_Nm"]
            pending_spine_ids = new_ids
        if eventful_indices:
            continue
        # 无事件后才评估平衡：支撑 wrench + 加载器反力 - 目标 wrench = 0。
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
            row_scale,
            coordinate_scale,
            tolerances,
        )
        if admissible_basis.shape[1] == 0:
            equilibrium_status = EquilibriumStatus.SOLVED
            break
        seed_signature = _active_set_signature(spines, working_states)
        if rank_status is RankStatus.RANK_DEFICIENT:
            # 欠秩可能只是所有可承载接触尚未闭合，先尝试一个残差相容的接触 seed。
            attempted = contact_seed_attempts.setdefault(seed_signature, set())
            prediction = _predict_contact_seed(
                spines,
                working_states,
                q_C,
                admissible_basis,
                reference_position,
                row_scale,
                coordinate_scale,
                residual,
                attempted,
            )
            if prediction is not None:
                seeded, key = prediction
                attempted.add(key)
                q_C = _enforce_equality_constraints(
                    seeded,
                    free,
                    control,
                    tolerances,
                    coordinate_scale,
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
        # 在 admissible 自由模态中解缩放 Newton 步，再映射回有量纲 q_C。
        delta_scaled = np.linalg.solve(
            scaled_matrix,
            -admissible_basis.T @ (row_scale * residual),
        )
        q_C += coordinate_scale * (admissible_basis @ delta_scaled)
    else:
        iteration_exhausted = True
        # 最后一次允许的 Newton 更新发生在装配之后；重新装配以获得一致诊断快照。
        # 若新位姿暴露了已无迭代预算处理的事件，则退回最后一个无事件迭代点。
        final_assembly = _assemble(
            spines,
            working_states,
            q_C,
            control,
            reference_position,
            q_rate,
            resistance_direction,
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
        elif any(trial.result.events for trial in final_assembly.trials):
            q_C = last_evaluated_q_C
        else:
            assembly = final_assembly
    assert assembly is not None
    # 阶段 3：在最终装配点重新计算统一的残差、秩和值域诊断，避免使用中间迭代缓存。
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
        row_scale,
        coordinate_scale,
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
    # 阵列平衡成立还不够；每个单刺 trial 也必须数值收敛且可提交。
    local_numerical_states = tuple(
        trial.result.numerical_state for trial in assembly.trials
    )
    local_trials_solved = all(
        trial.committable
        and numerical_state is NumericalState.CONVERGED
        for trial, numerical_state in zip(
            assembly.trials, local_numerical_states, strict=True
        )
    )
    solution_accepted = equilibrium_solved and local_trials_solved
    stability, minimum_eigenvalue, stability_threshold = _stability(
        total_tangent,
        assembly,
        spines,
        admissible_basis,
        control,
        tolerances,
        coordinate_scale,
        equilibrium_solved=solution_accepted,
    )
    # 模型、数值、平衡和稳定性是不同维度，分别汇总后再共同决定 committable。
    model_state = ModelState.CLOSED
    if equilibrium_status is EquilibriumStatus.MODEL_LIMIT:
        model_state = ModelState.OUT_OF_SCOPE
    elif any(
        trial.result.model_state is ModelState.PARAMETER_UNCLOSED
        for trial in assembly.trials
    ):
        model_state = ModelState.PARAMETER_UNCLOSED
    if solution_accepted:
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
        # q_C 是约束相对背板的加载坐标；固定墙面时真实背板位姿符号相反。
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
    committable = solution_accepted and acceptable_stability
    # 不可提交 trial 返回原 accepted state，调用方仍可读取完整失败诊断而不会污染状态。
    if committable:
        proposed_states = tuple(
            trial.proposed_state for trial in assembly.trials
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
    """仅提交基于当前 revision 且达到可接受平衡/稳定性的阵列 trial。"""

    # 与单刺相同，revision 防止旧 trial 覆盖更新后的阵列状态。
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
