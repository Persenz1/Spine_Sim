"""单根微刺的标准准静态接触—悬架求解器。

输入是一个已经由几何层确定的候选和相对加载位移；本模块联合求解梁柔顺、三维
Coulomb 摩擦、单边弹簧、容量与事件，并以不可变 trial/commit 更新八态状态机。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

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
    SpringBranch,
)
from .geometry import CandidateCursor, ContactCandidate


Vector3 = NDArray[np.float64]
Matrix3 = NDArray[np.float64]


def _vector3(value: ArrayLike, name: str) -> Vector3:
    """校验有限三维向量。"""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ConfigurationError(f"{name} must be a finite 3-vector")
    return result


def _unit3(value: ArrayLike, name: str) -> Vector3:
    """校验并归一化非零三维向量。"""

    result = _vector3(value, name)
    norm = float(np.linalg.norm(result))
    if norm <= 0.0:
        raise ConfigurationError(f"{name} must be non-zero")
    return result / norm


def _tuple3(value: ArrayLike) -> tuple[float, float, float]:
    """把 NumPy 三维向量转成冻结 dataclass 可保存的标量元组。"""

    vector = np.asarray(value, dtype=np.float64)
    return tuple(float(item) for item in vector)


def _matrix3(value: ArrayLike, name: str) -> Matrix3:
    """校验有限 3×3 矩阵。"""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ConfigurationError(f"{name} must be a finite 3x3 matrix")
    return result


def _tuple_matrix3(value: ArrayLike) -> tuple[tuple[float, float, float], ...]:
    """把 NumPy 3×3 矩阵转成嵌套元组。"""

    matrix = np.asarray(value, dtype=np.float64)
    return tuple(tuple(float(item) for item in row) for row in matrix)


def _positive(value: float, name: str) -> float:
    """校验有限正标量并返回其 float 表示。"""

    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigurationError(f"{name} must be positive and finite")
    return result


def _optional_positive(value: float | None, name: str) -> float | None:
    """对可选参数应用正数校验。"""

    return None if value is None else _positive(value, name)


@dataclass(frozen=True)
class SpineGeometry:
    """单刺根部、轴向和圆杆/球尖尺寸，以及输出 wrench 的语义标签。"""

    spine_id: str
    root_position_m: tuple[float, float, float]
    axis_root_to_tip: tuple[float, float, float]
    length_m: float
    diameter_m: float
    tip_radius_m: float
    frame: str
    root_reference: str
    backplate_object: str

    def __post_init__(self) -> None:
        """校验标识、坐标向量和所有正尺寸。"""

        if not all(
            (self.spine_id, self.frame, self.root_reference, self.backplate_object)
        ):
            raise ConfigurationError("spine and wrench labels cannot be empty")
        _vector3(self.root_position_m, "root_position_m")
        _unit3(self.axis_root_to_tip, "axis_root_to_tip")
        _positive(self.length_m, "length_m")
        _positive(self.diameter_m, "diameter_m")
        _positive(self.tip_radius_m, "tip_radius_m")


@dataclass(frozen=True)
class SpineMaterial:
    """针杆弹性参数及可选的轴体、表面和断裂容量参数。"""

    young_modulus_Pa: float
    poisson_ratio: float
    shear_correction: float
    shaft_allowable_stress_Pa: float | None = None
    surface_young_modulus_Pa: float | None = None
    surface_poisson_ratio: float | None = None
    surface_allowable_tensile_stress_Pa: float | None = None
    fracture_toughness_Pa_sqrt_m: float | None = None
    crack_half_length_m: float | None = None
    fracture_area_m2: float | None = None
    fracture_geometry_factor: float | None = None
    surface_capacity_present: bool = False
    fracture_topology_present: bool = False
    shaft_failure_is_catastrophic_disconnect: bool = False
    parameter_sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验弹性常数、泊松比和所有已给出的容量参数。"""

        _positive(self.young_modulus_Pa, "young_modulus_Pa")
        if not -1.0 < float(self.poisson_ratio) < 0.5:
            raise ConfigurationError("poisson_ratio must lie in (-1, 0.5)")
        _positive(self.shear_correction, "shear_correction")
        for name in (
            "shaft_allowable_stress_Pa",
            "surface_young_modulus_Pa",
            "surface_allowable_tensile_stress_Pa",
            "fracture_toughness_Pa_sqrt_m",
            "crack_half_length_m",
            "fracture_area_m2",
            "fracture_geometry_factor",
        ):
            _optional_positive(getattr(self, name), name)
        if self.surface_poisson_ratio is not None and not (
            -1.0 < float(self.surface_poisson_ratio) < 0.5
        ):
            raise ConfigurationError(
                "surface_poisson_ratio must lie in (-1, 0.5)"
            )


@dataclass(frozen=True)
class FrictionParameters:
    """三维各向同性 Coulomb 静/动摩擦系数及其来源。"""

    static_coefficient: float
    kinetic_coefficient: float
    parameter_source: str

    def __post_init__(self) -> None:
        """要求 ``0 <= μ_k <= μ_s`` 且来源标签非空。"""

        static = float(self.static_coefficient)
        kinetic = float(self.kinetic_coefficient)
        if (
            not math.isfinite(static)
            or not math.isfinite(kinetic)
            or kinetic < 0.0
            or static < kinetic
        ):
            raise ConfigurationError(
                "friction coefficients require 0 <= kinetic <= static"
            )
        if not self.parameter_source:
            raise ConfigurationError("friction parameter_source cannot be empty")


@dataclass(frozen=True)
class SuspensionParameters:
    """附加线性柔顺、可选单边轴向弹簧和准静态回弹距离。"""

    additional_compliance_m_per_N: (
        tuple[tuple[float, float, float], ...] | None
    )
    axial_spring_stiffness_N_per_m: float | None
    axial_spring_travel_m: float | None
    rebound_recovery_distance_m: float | None
    parameter_sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """检查柔度半正定性及弹簧刚度—行程的成对配置。"""

        if self.additional_compliance_m_per_N is not None:
            compliance = _matrix3(
                self.additional_compliance_m_per_N,
                "additional_compliance_m_per_N",
            )
            if not np.allclose(compliance, compliance.T, atol=1e-13):
                raise ConfigurationError("additional compliance must be symmetric")
            scale = max(1.0, float(np.linalg.norm(compliance, ord=2)))
            if float(np.linalg.eigvalsh(compliance).min()) < -1e-12 * scale:
                raise ConfigurationError(
                    "additional compliance must be positive semidefinite"
                )
        stiffness = self.axial_spring_stiffness_N_per_m
        travel = self.axial_spring_travel_m
        if (stiffness is None) != (travel is None):
            raise ConfigurationError(
                "axial spring stiffness and travel must both be set or both be None"
            )
        _optional_positive(stiffness, "axial_spring_stiffness_N_per_m")
        _optional_positive(travel, "axial_spring_travel_m")
        _optional_positive(
            self.rebound_recovery_distance_m,
            "rebound_recovery_distance_m",
        )


@dataclass(frozen=True)
class SingleSpineTolerances:
    """几何、力、摩擦、弹簧、速度、容量和事件定位的独立容差。"""

    gap_m: float
    force_N: float
    friction_N: float
    spring_N: float
    velocity_m_per_s: float
    capacity_relative: float
    event_fraction: float

    def __post_init__(self) -> None:
        """所有容差都必须为有限正数。"""

        for name in (
            "gap_m",
            "force_N",
            "friction_N",
            "spring_N",
            "velocity_m_per_s",
            "capacity_relative",
            "event_fraction",
        ):
            _positive(getattr(self, name), name)


@dataclass(frozen=True)
class BaseMotion:
    """接触约束相对刺根/加载坐标系的位移、切向速度和载荷参数。

    ``relative_displacement_m`` 是本构加载变形，并不是刺根的绝对空间位姿。
    """

    relative_displacement_m: tuple[float, float, float]
    relative_tangential_velocity_m_per_s: tuple[float, float, float]
    load_parameter: float
    search_distance_increment_m: float = 0.0

    def __post_init__(self) -> None:
        """校验两个三维运动量、载荷参数和非负搜索增量。"""

        _vector3(self.relative_displacement_m, "relative_displacement_m")
        _vector3(
            self.relative_tangential_velocity_m_per_s,
            "relative_tangential_velocity_m_per_s",
        )
        if not math.isfinite(float(self.load_parameter)):
            raise ConfigurationError("load_parameter must be finite")
        if (
            not math.isfinite(float(self.search_distance_increment_m))
            or self.search_distance_increment_m < 0.0
        ):
            raise ConfigurationError(
                "search_distance_increment_m must be finite and non-negative"
            )


@dataclass(frozen=True)
class SpineAcceptedState:
    """最近一次已提交的单刺状态与 continuation 信息。

    CONTACT 只允许存在于 trial 事件链中，不能成为 resident state；HARDSTOP 则通过
    ``contact_submode`` 继续保存其内部 STICK/SLIP 模式。
    """

    spine_id: str
    physical_state: PhysicalState
    contact_submode: PhysicalState | None
    spring_branch: SpringBranch
    candidate_id: str | None
    contact_point_m: tuple[float, float, float] | None
    contact_normal: tuple[float, float, float] | None
    relative_displacement_m: tuple[float, float, float]
    elastic_displacement_m: tuple[float, float, float]
    slip_displacement_m: tuple[float, float, float]
    search_cursor: CandidateCursor | None
    completed_detach_cycles: int
    reengagement_count: int
    rebound_distance_m: float
    damage_history: tuple[str, ...]
    last_load_parameter: float
    event_sequence: int
    revision: int

    def __post_init__(self) -> None:
        """校验状态、接触几何、游标、计数和 revision 的组合一致性。"""

        if not self.spine_id:
            raise ConfigurationError("spine_id cannot be empty")
        if self.physical_state is PhysicalState.CONTACT:
            raise ConfigurationError("CONTACT is a trial-only state")
        if self.physical_state is PhysicalState.HARDSTOP:
            if self.contact_submode not in {
                PhysicalState.STICK,
                PhysicalState.SLIP,
            }:
                raise ConfigurationError(
                    "HARDSTOP must preserve STICK or SLIP contact_submode"
                )
            if self.spring_branch is not SpringBranch.HARDSTOP:
                raise ConfigurationError(
                    "physical HARDSTOP requires spring HARDSTOP branch"
                )
        elif self.contact_submode is not None:
            raise ConfigurationError(
                "contact_submode is stored only in physical HARDSTOP"
            )
        if self.physical_state in {
            PhysicalState.STICK,
            PhysicalState.SLIP,
            PhysicalState.HARDSTOP,
        } and (
            self.candidate_id is None
            or self.contact_point_m is None
            or self.contact_normal is None
        ):
            raise ConfigurationError(
                "a load-bearing accepted state requires contact geometry"
            )
        for name in (
            "relative_displacement_m",
            "elastic_displacement_m",
            "slip_displacement_m",
        ):
            _vector3(getattr(self, name), name)
        if self.contact_point_m is not None:
            _vector3(self.contact_point_m, "contact_point_m")
        if self.contact_normal is not None:
            _unit3(self.contact_normal, "contact_normal")
        if self.search_cursor is not None and not isinstance(
            self.search_cursor, CandidateCursor
        ):
            raise ConfigurationError(
                "search_cursor must be a CandidateCursor or None"
            )
        for name in (
            "completed_detach_cycles",
            "reengagement_count",
            "event_sequence",
            "revision",
        ):
            if getattr(self, name) < 0:
                raise ConfigurationError(f"{name} must be non-negative")
        if self.rebound_distance_m < 0.0 or not math.isfinite(
            self.rebound_distance_m
        ):
            raise ConfigurationError(
                "rebound_distance_m must be finite and non-negative"
            )
        if not math.isfinite(self.last_load_parameter):
            raise ConfigurationError("last_load_parameter must be finite")

    @classmethod
    def initial(
        cls, spine_id: str, *, load_parameter: float = 0.0
    ) -> "SpineAcceptedState":
        """创建 revision 0 的 SEARCH 状态。"""

        return cls(
            spine_id=spine_id,
            physical_state=PhysicalState.SEARCH,
            contact_submode=None,
            spring_branch=SpringBranch.RIGID,
            candidate_id=None,
            contact_point_m=None,
            contact_normal=None,
            relative_displacement_m=(0.0, 0.0, 0.0),
            elastic_displacement_m=(0.0, 0.0, 0.0),
            slip_displacement_m=(0.0, 0.0, 0.0),
            search_cursor=None,
            completed_detach_cycles=0,
            reengagement_count=0,
            rebound_distance_m=0.0,
            damage_history=(),
            last_load_parameter=float(load_parameter),
            event_sequence=0,
            revision=0,
        )


@dataclass(frozen=True)
class WedgeResult:
    """二维楔形解析核的力、极限比和接触状态。"""

    normal_force_N: float
    signed_friction_demand_N: float
    friction_margin_N: float
    forward_limit_ratio: float
    reverse_limit_ratio: float
    positive_direction_self_lock: bool
    physical_state: PhysicalState
    slip_direction_sign: int


def solve_wedge_2d(
    tangential_load_N: float,
    compressive_load_N: float,
    slope_angle_rad: float,
    friction_coefficient: float,
) -> WedgeResult:
    """计算含 ``P<0`` 剥离情形的有符号二维楔形闭式解。"""

    values = (
        tangential_load_N,
        compressive_load_N,
        slope_angle_rad,
        friction_coefficient,
    )
    if not all(math.isfinite(float(item)) for item in values):
        raise ConfigurationError("wedge inputs must be finite")
    if tangential_load_N < 0.0 or friction_coefficient < 0.0:
        raise ConfigurationError("T and friction coefficient must be non-negative")
    sine = math.sin(slope_angle_rad)
    cosine = math.cos(slope_angle_rad)
    # 将外部切向/压紧载荷投影到局部法向和切向，再用 μN-|Fτ| 判断静摩擦。
    normal = tangential_load_N * sine + compressive_load_N * cosine
    demand = tangential_load_N * cosine - compressive_load_N * sine
    margin = friction_coefficient * normal - abs(demand)
    denominator = cosine - friction_coefficient * sine
    roundoff = 16.0 * np.finfo(np.float64).eps * max(
        1.0, abs(cosine), abs(friction_coefficient * sine)
    )
    # 正向极限分母接近零或变负时，理论极限比发散，对应正向自锁。
    self_lock = denominator <= roundoff
    forward = (
        math.inf
        if self_lock
        else (sine + friction_coefficient * cosine) / denominator
    )
    reverse_denominator = cosine + friction_coefficient * sine
    reverse = (
        (sine - friction_coefficient * cosine) / reverse_denominator
        if reverse_denominator > 0.0
        and sine > friction_coefficient * cosine
        else 0.0
    )
    if normal < 0.0:
        state = PhysicalState.DETACH
    elif margin >= 0.0:
        state = PhysicalState.STICK
    else:
        state = PhysicalState.SLIP
    sign = 1 if demand > 0.0 else -1 if demand < 0.0 else 0
    return WedgeResult(
        normal_force_N=normal,
        signed_friction_demand_N=demand,
        friction_margin_N=margin,
        forward_limit_ratio=forward,
        reverse_limit_ratio=reverse,
        positive_direction_self_lock=self_lock,
        physical_state=state,
        slip_direction_sign=sign,
    )


@dataclass(frozen=True)
class SpringSolution:
    """单边弹簧活动分支、位移、限位反力和互补残差。"""

    branch: SpringBranch
    displacement_m: float
    lower_stop_reaction_N: float
    hardstop_reaction_N: float
    complementarity_residual: float


def solve_unilateral_spring(
    axial_load_N: float,
    stiffness_N_per_m: float | None,
    travel_m: float | None,
    *,
    force_tolerance_N: float,
) -> SpringSolution:
    """求解单边弹簧的下限位、内部和硬限位互补分支。"""

    if not math.isfinite(float(axial_load_N)):
        raise ConfigurationError("axial_load_N must be finite")
    _positive(force_tolerance_N, "force_tolerance_N")
    if stiffness_N_per_m is None:
        if travel_m is not None:
            raise ConfigurationError("rigid spring branch cannot have travel")
        return SpringSolution(SpringBranch.RIGID, 0.0, 0.0, 0.0, 0.0)
    stiffness = _positive(stiffness_N_per_m, "stiffness_N_per_m")
    travel = _positive(travel_m, "travel_m") if travel_m is not None else None
    if travel is None:
        raise ConfigurationError("finite spring stiffness requires travel")
    upper_load = stiffness * travel
    # 轴向载荷不足以压缩弹簧时停在下限位；负载由下限位反力承担。
    if axial_load_N <= force_tolerance_N:
        reaction = max(-axial_load_N, 0.0)
        residual = max(axial_load_N, 0.0)
        return SpringSolution(
            SpringBranch.LOWER_STOP, 0.0, reaction, 0.0, residual
        )
    # 超过 k*u_max 后位移固定，超出部分转为硬限位反力。
    if axial_load_N >= upper_load - force_tolerance_N:
        reaction = max(axial_load_N - upper_load, 0.0)
        residual = max(upper_load - axial_load_N, 0.0)
        return SpringSolution(
            SpringBranch.HARDSTOP, travel, 0.0, reaction, residual
        )
    return SpringSolution(
        SpringBranch.INTERIOR,
        axial_load_N / stiffness,
        0.0,
        0.0,
        0.0,
    )


@dataclass(frozen=True)
class CapacityAssessment:
    """一项容量判据的闭合状态、需求、容量、利用率和失效语义。"""

    name: str
    model_state: ModelState
    demand: float | None
    capacity: float | None
    utilization: float | None
    margin: float | None
    failure_object: str | None
    failure_mode: str | None
    parameter_sources: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FailurePayload:
    """控制性失效判据及阵列层应采用的 continuation 动作。"""

    failure_object: str
    failure_mode: str
    criterion: str
    demand: float
    capacity: float
    margin: float
    parameter_sources: Mapping[str, str]
    continuation_action: ContinuationAction


@dataclass(frozen=True)
class SingleSpineResult:
    """一个加载站的单刺力学、状态、容量、残差和事件输出。"""

    wall_force_N: tuple[float, float, float]
    root_wrench: Wrench
    local_tangent_N_per_m: tuple[tuple[float, float, float], ...]
    physical_state: PhysicalState
    contact_submode: PhysicalState | None
    spring_branch: SpringBranch
    spring_displacement_m: float
    normal_force_N: float
    tangential_force_N: tuple[float, float, float]
    elastic_displacement_m: tuple[float, float, float]
    slip_displacement_m: tuple[float, float, float]
    model_state: ModelState
    numerical_state: NumericalState
    margins: Mapping[str, float | None]
    capacity_assessments: Mapping[str, CapacityAssessment]
    complementarity_residuals: Mapping[str, float]
    diagnostics: Mapping[str, float | None]
    events: tuple[Event, ...]
    failure: FailurePayload | None
    evaluated_motion: BaseMotion
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class SingleSpineTrial:
    """基于某一 accepted revision 的不可变求解提案。"""

    spine_id: str
    base_revision: int
    proposed_state: SpineAcceptedState
    result: SingleSpineResult
    committable: bool


@dataclass(frozen=True)
class _MechanicalResponse:
    """给定接触/弹簧活动分支下的内部力学解。"""

    force_N: Vector3
    tangent_N_per_m: Matrix3
    normal_force_N: float
    tangential_force_N: Vector3
    static_required_margin_N: float
    current_friction_margin_N: float
    elastic_displacement_m: Vector3
    slip_displacement_m: Vector3
    contact_mode: PhysicalState
    spring: SpringSolution
    balance_residual_m: float


def _beam_compliance(
    geometry: SpineGeometry,
    material: SpineMaterial,
    suspension: SuspensionParameters,
    axis: Vector3,
) -> Matrix3:
    """组装圆杆轴向、Euler–Bernoulli 弯曲、剪切和附加悬架柔度。"""

    diameter = float(geometry.diameter_m)
    length = float(geometry.length_m)
    area = math.pi * diameter**2 / 4.0
    second_moment = math.pi * diameter**4 / 64.0
    shear_modulus = material.young_modulus_Pa / (
        2.0 * (1.0 + material.poisson_ratio)
    )
    # c_a=L/(EA)；横向柔度同时包含弯曲 L³/(3EI) 和 Timoshenko 剪切项。
    axial = length / (material.young_modulus_Pa * area)
    transverse = length**3 / (
        3.0 * material.young_modulus_Pa * second_moment
    ) + length / (
        material.shear_correction * shear_modulus * area
    )
    # 轴向投影 eeᵀ 与横向投影 I-eeᵀ 把两个标量柔度映射回全局坐标。
    projector = np.outer(axis, axis)
    compliance = axial * projector + transverse * (np.eye(3) - projector)
    if suspension.additional_compliance_m_per_N is not None:
        compliance = compliance + np.asarray(
            suspension.additional_compliance_m_per_N, dtype=np.float64
        )
    if float(np.linalg.eigvalsh(compliance).min()) <= 0.0:
        raise ConfigurationError("combined beam/suspension compliance must be positive definite")
    return compliance


def spine_linear_compliance(
    geometry: SpineGeometry,
    material: SpineMaterial,
    suspension: SuspensionParameters,
) -> Matrix3:
    """返回单刺梁与线性悬架在其声明坐标系中的三维柔度矩阵。"""

    axis = _unit3(geometry.axis_root_to_tip, "axis_root_to_tip")
    return _beam_compliance(geometry, material, suspension, axis)


def _branch_compliance(
    base_compliance: Matrix3,
    compression_direction: Vector3,
    branch: SpringBranch,
    suspension: SuspensionParameters,
) -> tuple[Matrix3, Vector3]:
    """把指定单边弹簧分支转换为等效柔度和仿射位移偏置。"""

    offset = np.zeros(3, dtype=np.float64)
    if branch is SpringBranch.RIGID or branch is SpringBranch.LOWER_STOP:
        return base_compliance, offset
    stiffness = suspension.axial_spring_stiffness_N_per_m
    travel = suspension.axial_spring_travel_m
    if stiffness is None or travel is None:
        raise ConfigurationError("non-rigid spring branch requires stiffness and travel")
    if branch is SpringBranch.INTERIOR:
        # 内部分支增加沿压缩方向的 1/k 柔度。
        return (
            base_compliance
            + np.outer(compression_direction, compression_direction) / stiffness,
            offset,
        )
    if branch is SpringBranch.HARDSTOP:
        # 硬限位后不再增加柔度，只保留固定行程形成的仿射位移。
        offset = travel * compression_direction
        return base_compliance, offset
    raise AssertionError(f"unhandled spring branch {branch}")


def _friction_components(
    force_N: Vector3, normal: Vector3, friction: FrictionParameters
) -> tuple[float, Vector3, float]:
    """分解法向/切向力并计算静摩擦锥裕度 ``μ_s Fn-|Ft|``。"""

    normal_force = float(np.dot(force_N, normal))
    tangential = force_N - normal_force * normal
    margin = friction.static_coefficient * normal_force - float(
        np.linalg.norm(tangential)
    )
    return normal_force, tangential, margin


def _solve_contact_for_branch(
    relative_displacement_m: Vector3,
    tangential_velocity_m_per_s: Vector3,
    normal: Vector3,
    compliance: Matrix3,
    affine_offset_m: Vector3,
    friction: FrictionParameters,
    previous_mode: PhysicalState,
    tolerances: SingleSpineTolerances,
    *,
    force_slip: bool = False,
) -> tuple[
    Vector3,
    Matrix3,
    float,
    Vector3,
    float,
    float,
    Vector3,
    Vector3,
    PhysicalState,
]:
    """在固定弹簧分支上求 STICK 或 SLIP 接触力及一致切线。"""

    effective_displacement = relative_displacement_m - affine_offset_m
    # 先假定完全粘着，由位移反解所需力；摩擦裕度决定该假设是否可行。
    stiffness = np.linalg.inv(compliance)
    required_force = stiffness @ effective_displacement
    required_normal, required_tangent, required_margin = _friction_components(
        required_force, normal, friction
    )
    velocity = tangential_velocity_m_per_s - normal * float(
        np.dot(tangential_velocity_m_per_s, normal)
    )
    velocity_norm = float(np.linalg.norm(velocity))
    should_slip = force_slip or required_margin < -tolerances.friction_N
    # 已在滑移且仍有可辨切向速度时保持滑移，避免摩擦边界附近数值抖动。
    if previous_mode is PhysicalState.SLIP and velocity_norm > tolerances.velocity_m_per_s:
        should_slip = True
    if (
        previous_mode is PhysicalState.SLIP
        and velocity_norm <= tolerances.velocity_m_per_s
        and required_margin <= tolerances.friction_N
    ):
        should_slip = True
    if not should_slip:
        # STICK：全部相对位移由弹性变形承担，切线就是组合刚度矩阵。
        elastic = compliance @ required_force + affine_offset_m
        return (
            required_force,
            stiffness,
            required_normal,
            required_tangent,
            required_margin,
            required_margin,
            elastic,
            relative_displacement_m - elastic,
            PhysicalState.STICK,
        )

    if velocity_norm > tolerances.velocity_m_per_s:
        # 这里的速度是“墙面接触约束相对刺根”的速度；墙面对刺/背板的摩擦力
        # 与它同向，因而与刺根相对墙面的真实速度反向。
        slip_direction = velocity / velocity_norm
    else:
        tangent_norm = float(np.linalg.norm(required_tangent))
        if tangent_norm <= tolerances.force_N:
            return (
                required_force,
                stiffness,
                required_normal,
                required_tangent,
                required_margin,
                required_margin,
                compliance @ required_force + affine_offset_m,
                np.zeros(3, dtype=np.float64),
                PhysicalState.STICK,
            )
        slip_direction = required_tangent / tangent_norm
    cone_direction = normal + friction.kinetic_coefficient * slip_direction
    # F=Fn(n+μ_k v̂_constraint/root)，再由法向位移相容条件求唯一 Fn。
    denominator = float(normal @ compliance @ cone_direction)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise np.linalg.LinAlgError("sliding contact has no positive normal compliance")
    normal_force = float(normal @ effective_displacement) / denominator
    force = normal_force * cone_direction
    tangent = force - normal_force * normal
    consistent_tangent = np.outer(cone_direction, normal) / denominator
    elastic = compliance @ force + affine_offset_m
    slip = relative_displacement_m - elastic
    current_margin = friction.static_coefficient * normal_force - float(
        np.linalg.norm(tangent)
    )
    return (
        force,
        consistent_tangent,
        normal_force,
        tangent,
        required_margin,
        current_margin,
        elastic,
        slip,
        PhysicalState.SLIP,
    )


def _spring_consistent(
    branch: SpringBranch,
    axial_load_N: float,
    suspension: SuspensionParameters,
    tolerances: SingleSpineTolerances,
) -> SpringSolution | None:
    """复算单边弹簧并仅接受与当前假设分支一致的解。"""

    solution = solve_unilateral_spring(
        axial_load_N,
        suspension.axial_spring_stiffness_N_per_m,
        suspension.axial_spring_travel_m,
        force_tolerance_N=tolerances.spring_N,
    )
    if solution.branch is branch:
        return solution
    return None


def _mechanical_response(
    displacement: Vector3,
    velocity: Vector3,
    normal: Vector3,
    base_compliance: Matrix3,
    compression_direction: Vector3,
    friction: FrictionParameters,
    suspension: SuspensionParameters,
    accepted: SpineAcceptedState,
    tolerances: SingleSpineTolerances,
    *,
    force_slip: bool = False,
) -> _MechanicalResponse | None:
    """枚举少量弹簧活动分支，返回首个同时满足接触和弹簧相容性的解。"""

    if suspension.axial_spring_stiffness_N_per_m is None:
        branches = (SpringBranch.RIGID,)
    else:
        # 优先尝试上一步分支以保持 continuation，再按完整活动集补试其他分支。
        ordered = (
            accepted.spring_branch,
            SpringBranch.LOWER_STOP,
            SpringBranch.INTERIOR,
            SpringBranch.HARDSTOP,
        )
        branches = tuple(
            branch
            for index, branch in enumerate(ordered)
            if branch is not SpringBranch.RIGID
            and branch not in ordered[:index]
        )
    previous_mode = (
        accepted.contact_submode
        if accepted.physical_state is PhysicalState.HARDSTOP
        else accepted.physical_state
    )
    for branch in branches:
        # 每个候选分支只需一次 3×3 接触解；分支不相容就继续而不是修改 accepted。
        compliance, offset = _branch_compliance(
            base_compliance, compression_direction, branch, suspension
        )
        try:
            (
                force,
                tangent_matrix,
                normal_force,
                tangential_force,
                required_margin,
                current_margin,
                elastic,
                slip,
                contact_mode,
            ) = _solve_contact_for_branch(
                displacement,
                velocity,
                normal,
                compliance,
                offset,
                friction,
                previous_mode,
                tolerances,
                force_slip=force_slip,
            )
        except np.linalg.LinAlgError:
            continue
        axial_load = float(np.dot(force, compression_direction))
        spring = _spring_consistent(
            branch, axial_load, suspension, tolerances
        )
        if spring is None:
            continue
        balance_residual = float(
            np.linalg.norm(compliance @ force + offset + slip - displacement)
        )
        return _MechanicalResponse(
            force_N=force,
            tangent_N_per_m=tangent_matrix,
            normal_force_N=normal_force,
            tangential_force_N=tangential_force,
            static_required_margin_N=required_margin,
            current_friction_margin_N=current_margin,
            elastic_displacement_m=elastic,
            slip_displacement_m=slip,
            contact_mode=contact_mode,
            spring=spring,
            balance_residual_m=balance_residual,
        )
    return None


def _equivalent_radius_m(
    candidate: ContactCandidate, tip_radius_m: float
) -> float | None:
    """由球尖和局部表面曲率计算 Hertz 等效半径；曲率未知则不闭合。"""

    surface_radius = candidate.curvature_radius_m
    if surface_radius is None:
        return None
    return 1.0 / (1.0 / tip_radius_m + 1.0 / surface_radius)


def _assessment_unclosed(
    name: str,
    failure_object: str,
    failure_mode: str,
    sources: Mapping[str, str],
) -> CapacityAssessment:
    """构造“判据存在但参数或来源不足”的容量结果。"""

    return CapacityAssessment(
        name=name,
        model_state=ModelState.PARAMETER_UNCLOSED,
        demand=None,
        capacity=None,
        utilization=None,
        margin=None,
        failure_object=failure_object,
        failure_mode=failure_mode,
        parameter_sources=sources,
    )


def _assessment_out_of_scope(
    name: str, failure_object: str, failure_mode: str
) -> CapacityAssessment:
    """构造当前模型明确不处理的容量结果。"""

    return CapacityAssessment(
        name=name,
        model_state=ModelState.OUT_OF_SCOPE,
        demand=None,
        capacity=None,
        utilization=None,
        margin=None,
        failure_object=failure_object,
        failure_mode=failure_mode,
    )


def _has_sources(material: SpineMaterial, names: tuple[str, ...]) -> bool:
    """确认指定材料参数均带有非空来源记录。"""

    return all(bool(material.parameter_sources.get(name)) for name in names)


def _capacity_assessments(
    geometry: SpineGeometry,
    material: SpineMaterial,
    candidate: ContactCandidate,
    axis: Vector3,
    root_position_m: Vector3,
    force_N: Vector3,
    normal_force_N: float,
    tangential_force_N: Vector3,
    contact_point_m: Vector3,
) -> tuple[dict[str, CapacityAssessment], dict[str, float | None]]:
    """评估轴体强度、Hertz/表面拉应力和 II 型断裂容量。"""

    assessments: dict[str, CapacityAssessment] = {}
    diagnostics: dict[str, float | None] = {
        "shaft_axial_force_N": None,
        "shaft_shear_force_N": None,
        "shaft_bending_moment_Nm": None,
        "shaft_torsion_moment_Nm": None,
        "shaft_von_mises_upper_Pa": None,
        "hertz_contact_radius_m": None,
        "hertz_center_pressure_Pa": None,
        "surface_edge_tension_Pa": None,
        "fracture_force_capacity_N": None,
    }
    # 先把接触力搬到刺根，分解轴力、剪力、弯矩和扭矩。
    lever = contact_point_m - root_position_m
    moment = np.cross(lever, force_N)
    axial_force = float(np.dot(force_N, axis))
    shear_force = float(np.linalg.norm(force_N - axial_force * axis))
    torsion_moment = abs(float(np.dot(moment, axis)))
    bending_moment = float(
        np.linalg.norm(moment - np.dot(moment, axis) * axis)
    )
    diameter = geometry.diameter_m
    area = math.pi * diameter**2 / 4.0
    second_moment = math.pi * diameter**4 / 64.0
    polar_moment = math.pi * diameter**4 / 32.0
    # 轴向正应力与最外纤维弯曲应力保守相加，剪切与扭转再进入 von Mises 上界。
    normal_stress = abs(axial_force) / area + bending_moment * (
        diameter / 2.0
    ) / second_moment
    transverse_shear = 4.0 * shear_force / (3.0 * area)
    torsion_shear = torsion_moment * (diameter / 2.0) / polar_moment
    von_mises_upper = math.sqrt(
        normal_stress**2 + 3.0 * (transverse_shear + torsion_shear) ** 2
    )
    diagnostics.update(
        {
            "shaft_axial_force_N": axial_force,
            "shaft_shear_force_N": shear_force,
            "shaft_bending_moment_Nm": bending_moment,
            "shaft_torsion_moment_Nm": torsion_moment,
            "shaft_von_mises_upper_Pa": von_mises_upper,
        }
    )
    shaft_capacity = material.shaft_allowable_stress_Pa
    if shaft_capacity is None or not _has_sources(
        material, ("shaft_allowable_stress_Pa",)
    ):
        assessments["shaft"] = _assessment_unclosed(
            "shaft",
            "spine_shaft",
            "shaft_strength",
            material.parameter_sources,
        )
    else:
        utilization = von_mises_upper / shaft_capacity
        assessments["shaft"] = CapacityAssessment(
            name="shaft",
            model_state=ModelState.CLOSED,
            demand=von_mises_upper,
            capacity=shaft_capacity,
            utilization=utilization,
            margin=1.0 - utilization,
            failure_object="spine_shaft",
            failure_mode=(
                "catastrophic_disconnect"
                if material.shaft_failure_is_catastrophic_disconnect
                else "elastic_limit_or_yield"
            ),
            parameter_sources=material.parameter_sources,
        )

    if not material.surface_capacity_present:
        # 没有声明表面容量拓扑时是 OUT_OF_SCOPE，而不是把未知容量当作无限大。
        assessments["hertz"] = _assessment_out_of_scope(
            "hertz", "tip_surface_contact", "hertz_contact"
        )
        assessments["surface"] = _assessment_out_of_scope(
            "surface", "surface", "edge_tension"
        )
    else:
        equivalent_radius = _equivalent_radius_m(
            candidate, geometry.tip_radius_m
        )
        needed = (
            material.surface_young_modulus_Pa,
            material.surface_poisson_ratio,
            material.surface_allowable_tensile_stress_Pa,
            equivalent_radius,
        )
        if any(value is None for value in needed) or not _has_sources(
            material,
            (
                "surface_young_modulus_Pa",
                "surface_poisson_ratio",
                "surface_allowable_tensile_stress_Pa",
            ),
        ):
            assessments["hertz"] = _assessment_unclosed(
                "hertz",
                "tip_surface_contact",
                "hertz_contact",
                material.parameter_sources,
            )
            assessments["surface"] = _assessment_unclosed(
                "surface",
                "surface",
                "edge_tension",
                material.parameter_sources,
            )
        else:
            surface_E = float(material.surface_young_modulus_Pa)
            surface_nu = float(material.surface_poisson_ratio)
            surface_allowable = float(
                material.surface_allowable_tensile_stress_Pa
            )
            radius = float(equivalent_radius)
            # 两弹性体的约化模量，以及球—局部曲面 Hertz 接触半径/中心压力。
            inverse_modulus = (
                (1.0 - material.poisson_ratio**2)
                / material.young_modulus_Pa
                + (1.0 - surface_nu**2) / surface_E
            )
            equivalent_modulus = 1.0 / inverse_modulus
            if normal_force_N > 0.0:
                hertz_radius = (
                    3.0 * normal_force_N * radius
                    / (4.0 * equivalent_modulus)
                ) ** (1.0 / 3.0)
                center_pressure = 3.0 * normal_force_N / (
                    2.0 * math.pi * hertz_radius**2
                )
            else:
                hertz_radius = 0.0
                center_pressure = 0.0
            edge_tension = (1.0 - 2.0 * surface_nu) * center_pressure / 3.0
            diagnostics.update(
                {
                    "hertz_contact_radius_m": hertz_radius,
                    "hertz_center_pressure_Pa": center_pressure,
                    "surface_edge_tension_Pa": edge_tension,
                }
            )
            assessments["hertz"] = CapacityAssessment(
                name="hertz",
                model_state=ModelState.CLOSED,
                demand=center_pressure,
                capacity=None,
                utilization=None,
                margin=None,
                failure_object="tip_surface_contact",
                failure_mode="hertz_contact",
                parameter_sources=material.parameter_sources,
            )
            utilization = edge_tension / surface_allowable
            assessments["surface"] = CapacityAssessment(
                name="surface",
                model_state=ModelState.CLOSED,
                demand=edge_tension,
                capacity=surface_allowable,
                utilization=utilization,
                margin=1.0 - utilization,
                failure_object="surface",
                failure_mode="edge_tension",
                parameter_sources=material.parameter_sources,
            )

    if not material.fracture_topology_present:
        # 只有明确声明裂纹/断裂拓扑后才启用断裂韧度判据。
        assessments["fracture"] = _assessment_out_of_scope(
            "fracture", "asperity", "mode_ii_fracture"
        )
    else:
        fracture_values = (
            material.fracture_toughness_Pa_sqrt_m,
            material.crack_half_length_m,
            material.fracture_area_m2,
            material.fracture_geometry_factor,
        )
        if any(value is None for value in fracture_values) or not _has_sources(
            material,
            (
                "fracture_toughness_Pa_sqrt_m",
                "crack_half_length_m",
                "fracture_area_m2",
                "fracture_geometry_factor",
            ),
        ):
            assessments["fracture"] = _assessment_unclosed(
                "fracture",
                "asperity",
                "mode_ii_fracture",
                material.parameter_sources,
            )
        else:
            # K_II=Y*τ*sqrt(pi*a)，再乘有效断裂面积得到切向力容量。
            critical_shear = float(material.fracture_toughness_Pa_sqrt_m) / (
                float(material.fracture_geometry_factor)
                * math.sqrt(math.pi * float(material.crack_half_length_m))
            )
            fracture_capacity = critical_shear * float(material.fracture_area_m2)
            fracture_demand = float(np.linalg.norm(tangential_force_N))
            utilization = fracture_demand / fracture_capacity
            diagnostics["fracture_force_capacity_N"] = fracture_capacity
            assessments["fracture"] = CapacityAssessment(
                name="fracture",
                model_state=ModelState.CLOSED,
                demand=fracture_demand,
                capacity=fracture_capacity,
                utilization=utilization,
                margin=1.0 - utilization,
                failure_object="asperity",
                failure_mode="mode_ii_fracture",
                parameter_sources=material.parameter_sources,
            )
    return assessments, diagnostics


def _aggregate_model_state(
    assessments: Mapping[str, CapacityAssessment]
) -> ModelState:
    """只要任一已启用判据参数未闭合，整体模型状态就保持未闭合。"""

    if any(
        assessment.model_state is ModelState.PARAMETER_UNCLOSED
        for assessment in assessments.values()
    ):
        return ModelState.PARAMETER_UNCLOSED
    return ModelState.CLOSED


def _failure_from_assessments(
    assessments: Mapping[str, CapacityAssessment],
    material: SpineMaterial,
    tolerances: SingleSpineTolerances,
) -> FailurePayload | None:
    """选择最小容量裕度作为控制失效，并决定永久移除或停止于模型边界。"""

    reached = [
        assessment
        for assessment in assessments.values()
        if assessment.model_state is ModelState.CLOSED
        and assessment.margin is not None
        and assessment.margin <= tolerances.capacity_relative
        and assessment.demand is not None
        and assessment.capacity is not None
    ]
    if not reached:
        return None
    controlling = min(reached, key=lambda item: float(item.margin))
    # 当前只有声明为灾难性断开的轴体失效能继续为“永久移除”；其他损伤演化未建模。
    permanent = (
        controlling.name == "shaft"
        and material.shaft_failure_is_catastrophic_disconnect
        and controlling.failure_mode == "catastrophic_disconnect"
    )
    return FailurePayload(
        failure_object=controlling.failure_object,
        failure_mode=controlling.failure_mode,
        criterion=f"{controlling.name}_utilization>=1",
        demand=float(controlling.demand),
        capacity=float(controlling.capacity),
        margin=float(controlling.margin),
        parameter_sources=controlling.parameter_sources,
        continuation_action=(
            ContinuationAction.PERMANENT_REMOVE
            if permanent
            else ContinuationAction.STOP_MODEL_LIMIT
        ),
    )


def _new_event(
    accepted: SpineAcceptedState,
    offset: int,
    event_type: EventType,
    from_state: PhysicalState,
    to_state: PhysicalState | None,
    motion: BaseMotion,
    details: Mapping[str, Any] | None = None,
) -> Event:
    """基于 accepted 序号创建下一条已校验物理事件。"""

    return Event(
        event_type=event_type,
        sequence=accepted.event_sequence + offset,
        from_state=from_state,
        to_state=to_state,
        spine_id=accepted.spine_id,
        load_parameter=motion.load_parameter,
        details={} if details is None else details,
    )


def _contact_point(
    geometry: SpineGeometry,
    candidate: ContactCandidate,
    normal: Vector3,
) -> Vector3:
    """从球心沿所选外法向回退一个尖端半径得到点接触位置。"""

    return candidate.sphere_center_m - geometry.tip_radius_m * normal


def _root_wrench(
    geometry: SpineGeometry,
    root_position_m: Vector3,
    contact_point_m: Vector3,
    force_N: Vector3,
) -> Wrench:
    """把墙面对刺的接触力等效为刺传给背板的根部 wrench。"""

    moment = np.cross(contact_point_m - root_position_m, force_N)
    return Wrench(
        force_N=_tuple3(force_N),
        moment_Nm=_tuple3(moment),
        frame=geometry.frame,
        reference_point=geometry.root_reference,
        acting_on=geometry.backplate_object,
        exerted_by=geometry.spine_id,
    )


def _zero_wrench(geometry: SpineGeometry) -> Wrench:
    """构造标签完整的零根部 wrench。"""

    return Wrench(
        force_N=(0.0, 0.0, 0.0),
        moment_Nm=(0.0, 0.0, 0.0),
        frame=geometry.frame,
        reference_point=geometry.root_reference,
        acting_on=geometry.backplate_object,
        exerted_by=geometry.spine_id,
    )


def _zero_result(
    geometry: SpineGeometry,
    motion: BaseMotion,
    physical_state: PhysicalState,
    spring_branch: SpringBranch,
    model_state: ModelState,
    numerical_state: NumericalState,
    events: tuple[Event, ...] = (),
    *,
    assumptions: tuple[str, ...] = (),
) -> SingleSpineResult:
    """为 SEARCH/DETACH/REBOUND/FAILED 等无承载状态构造零力结果。"""

    return SingleSpineResult(
        wall_force_N=(0.0, 0.0, 0.0),
        root_wrench=_zero_wrench(geometry),
        local_tangent_N_per_m=_tuple_matrix3(np.zeros((3, 3))),
        physical_state=physical_state,
        contact_submode=None,
        spring_branch=spring_branch,
        spring_displacement_m=0.0,
        normal_force_N=0.0,
        tangential_force_N=(0.0, 0.0, 0.0),
        elastic_displacement_m=(0.0, 0.0, 0.0),
        slip_displacement_m=(0.0, 0.0, 0.0),
        model_state=model_state,
        numerical_state=numerical_state,
        margins={},
        capacity_assessments={},
        complementarity_residuals={},
        diagnostics={},
        events=events,
        failure=None,
        evaluated_motion=motion,
        assumptions=assumptions,
    )


def _candidate_gate(
    candidate: ContactCandidate,
) -> tuple[Vector3 | None, str | None, ModelState]:
    """按 near-tie、法向、forward-cap、杆体碰撞和 valid 顺序门控候选。"""

    if candidate.near_tie:
        return None, "near_tie_requires_resolved_normal_model", ModelState.PARAMETER_UNCLOSED
    selected_normal = candidate.selected_normal
    if selected_normal is None:
        return None, "contact_normal_unclosed", ModelState.PARAMETER_UNCLOSED
    normal_norm = float(np.linalg.norm(selected_normal))
    if normal_norm <= 0.0:
        return None, "contact_normal_invalid", ModelState.CLOSED
    normal = selected_normal / normal_norm
    forward = candidate.forward_cap_valid
    if forward is None:
        return None, "forward_cap_unclosed", ModelState.PARAMETER_UNCLOSED
    if not forward:
        return None, "forward_cap_rejected", ModelState.CLOSED
    collision = candidate.rod_clearance.collision
    if collision is None:
        return None, "rod_clearance_unclosed", ModelState.PARAMETER_UNCLOSED
    if collision:
        return None, "rod_collision", ModelState.CLOSED
    if not candidate.valid:
        return None, "candidate_invalid", ModelState.CLOSED
    return normal, None, ModelState.CLOSED


def _rejected_search_trial(
    geometry: SpineGeometry,
    accepted: SpineAcceptedState,
    motion: BaseMotion,
    reason: str,
    model_state: ModelState,
    search_cursor: CandidateCursor | None,
) -> SingleSpineTrial:
    """把 SEARCH 中被拒候选转成可提交 trial，并推进其 continuation cursor。"""

    event = _new_event(
        accepted,
        1,
        EventType.CONTACT_REJECT,
        PhysicalState.CONTACT,
        PhysicalState.SEARCH,
        motion,
        {"reason": reason, "trial_proposal_committed": False},
    )
    proposed = replace(
        accepted,
        physical_state=PhysicalState.SEARCH,
        contact_submode=None,
        candidate_id=None,
        contact_point_m=None,
        contact_normal=None,
        relative_displacement_m=motion.relative_displacement_m,
        elastic_displacement_m=(0.0, 0.0, 0.0),
        slip_displacement_m=(0.0, 0.0, 0.0),
        search_cursor=search_cursor,
        last_load_parameter=motion.load_parameter,
        event_sequence=event.sequence,
        revision=accepted.revision + 1,
    )
    result = _zero_result(
        geometry,
        motion,
        PhysicalState.SEARCH,
        accepted.spring_branch,
        model_state,
        NumericalState.CONVERGED,
        (event,),
    )
    return SingleSpineTrial(
        geometry.spine_id, accepted.revision, proposed, result, True
    )


def _advance_detach_or_rebound(
    geometry: SpineGeometry,
    suspension: SuspensionParameters,
    accepted: SpineAcceptedState,
    motion: BaseMotion,
) -> SingleSpineTrial:
    """按准静态恢复距离推进 DETACH→REBOUND→SEARCH。"""

    if accepted.physical_state is PhysicalState.DETACH:
        # DETACH 只驻留一个已提交站，下一次调用显式进入 REBOUND。
        event = _new_event(
            accepted,
            1,
            EventType.REBOUND_START,
            PhysicalState.DETACH,
            PhysicalState.REBOUND,
            motion,
        )
        proposed = replace(
            accepted,
            physical_state=PhysicalState.REBOUND,
            contact_submode=None,
            candidate_id=None,
            contact_point_m=None,
            contact_normal=None,
            rebound_distance_m=0.0,
            last_load_parameter=motion.load_parameter,
            event_sequence=event.sequence,
            revision=accepted.revision + 1,
        )
        result = _zero_result(
            geometry,
            motion,
            PhysicalState.REBOUND,
            accepted.spring_branch,
            ModelState.CLOSED,
            NumericalState.CONVERGED,
            (event,),
            assumptions=("quasistatic_rebound_distance_rule",),
        )
        return SingleSpineTrial(
            geometry.spine_id, accepted.revision, proposed, result, True
        )

    distance = accepted.rebound_distance_m + motion.search_distance_increment_m
    threshold = suspension.rebound_recovery_distance_m
    if threshold is None:
        # 未提供恢复距离时不能臆造重挂条件；保持 REBOUND 并标记参数未闭合。
        proposed = replace(
            accepted,
            rebound_distance_m=distance,
            last_load_parameter=motion.load_parameter,
            revision=accepted.revision + 1,
        )
        result = _zero_result(
            geometry,
            motion,
            PhysicalState.REBOUND,
            accepted.spring_branch,
            ModelState.PARAMETER_UNCLOSED,
            NumericalState.CONVERGED,
            assumptions=("dynamic_rebound_out_of_scope",),
        )
        return SingleSpineTrial(
            geometry.spine_id, accepted.revision, proposed, result, True
        )
    if distance < threshold:
        proposed = replace(
            accepted,
            rebound_distance_m=distance,
            last_load_parameter=motion.load_parameter,
            revision=accepted.revision + 1,
        )
        result = _zero_result(
            geometry,
            motion,
            PhysicalState.REBOUND,
            accepted.spring_branch,
            ModelState.CLOSED,
            NumericalState.CONVERGED,
            assumptions=("quasistatic_rebound_distance_rule",),
        )
        return SingleSpineTrial(
            geometry.spine_id, accepted.revision, proposed, result, True
        )
    event = _new_event(
        accepted,
        1,
        EventType.REBOUND_COMPLETE,
        PhysicalState.REBOUND,
        PhysicalState.SEARCH,
        motion,
        {"recovery_distance_m": threshold},
    )
    proposed = replace(
        accepted,
        physical_state=PhysicalState.SEARCH,
        contact_submode=None,
        candidate_id=None,
        contact_point_m=None,
        contact_normal=None,
        rebound_distance_m=0.0,
        completed_detach_cycles=accepted.completed_detach_cycles + 1,
        last_load_parameter=motion.load_parameter,
        event_sequence=event.sequence,
        revision=accepted.revision + 1,
    )
    result = _zero_result(
        geometry,
        motion,
        PhysicalState.SEARCH,
        accepted.spring_branch,
        ModelState.CLOSED,
        NumericalState.CONVERGED,
        (event,),
        assumptions=("quasistatic_rebound_distance_rule",),
    )
    return SingleSpineTrial(
        geometry.spine_id, accepted.revision, proposed, result, True
    )


def _detachment_trial(
    geometry: SpineGeometry,
    accepted: SpineAcceptedState,
    motion: BaseMotion,
    reason: str,
) -> SingleSpineTrial:
    """清空接触几何和承载变形，构造进入 DETACH 的可提交 trial。"""

    event = _new_event(
        accepted,
        1,
        EventType.DETACH,
        accepted.physical_state,
        PhysicalState.DETACH,
        motion,
        {"reason": reason},
    )
    proposed = replace(
        accepted,
        physical_state=PhysicalState.DETACH,
        contact_submode=None,
        candidate_id=None,
        contact_point_m=None,
        contact_normal=None,
        relative_displacement_m=motion.relative_displacement_m,
        elastic_displacement_m=(0.0, 0.0, 0.0),
        slip_displacement_m=(0.0, 0.0, 0.0),
        last_load_parameter=motion.load_parameter,
        event_sequence=event.sequence,
        revision=accepted.revision + 1,
    )
    result = _zero_result(
        geometry,
        motion,
        PhysicalState.DETACH,
        accepted.spring_branch,
        ModelState.CLOSED,
        NumericalState.CONVERGED,
        (event,),
    )
    return SingleSpineTrial(
        geometry.spine_id, accepted.revision, proposed, result, True
    )


def _interpolate_motion(
    accepted: SpineAcceptedState,
    target: BaseMotion,
    start_displacement_m: Vector3,
    target_displacement_m: Vector3,
    fraction: float,
) -> BaseMotion:
    """按同一 fraction 插值位移、载荷参数与本步搜索距离。"""

    displacement = start_displacement_m + fraction * (
        target_displacement_m - start_displacement_m
    )
    load_parameter = accepted.last_load_parameter + fraction * (
        target.load_parameter - accepted.last_load_parameter
    )
    return BaseMotion(
        relative_displacement_m=_tuple3(displacement),
        relative_tangential_velocity_m_per_s=(
            target.relative_tangential_velocity_m_per_s
        ),
        load_parameter=load_parameter,
        search_distance_increment_m=(
            fraction * target.search_distance_increment_m
        ),
    )


def _event_values(
    geometry: SpineGeometry,
    material: SpineMaterial,
    suspension: SuspensionParameters,
    candidate: ContactCandidate,
    axis: Vector3,
    root_position_m: Vector3,
    compression_direction: Vector3,
    contact_point: Vector3,
    tolerances: SingleSpineTolerances,
    mechanical: _MechanicalResponse,
) -> dict[str, float]:
    """把各事件条件写成“正值=尚未触发，非正=到达边界”的标量函数。"""

    axial_load = float(np.dot(mechanical.force_N, compression_direction))
    values = {
        "normal": mechanical.normal_force_N - tolerances.force_N,
        "friction": mechanical.static_required_margin_N,
    }
    stiffness = suspension.axial_spring_stiffness_N_per_m
    travel = suspension.axial_spring_travel_m
    if stiffness is not None and travel is not None:
        values["hardstop"] = stiffness * travel - axial_load
        values["hardstop_release"] = axial_load - stiffness * travel
    assessments, _ = _capacity_assessments(
        geometry,
        material,
        candidate,
        axis,
        root_position_m,
        mechanical.force_N,
        mechanical.normal_force_N,
        mechanical.tangential_force_N,
        contact_point,
    )
    for name, assessment in assessments.items():
        if assessment.margin is not None:
            values[f"capacity:{name}"] = (
                assessment.margin - tolerances.capacity_relative
            )
    return values


def _locate_earliest_event(
    geometry: SpineGeometry,
    material: SpineMaterial,
    friction: FrictionParameters,
    suspension: SuspensionParameters,
    accepted: SpineAcceptedState,
    target: BaseMotion,
    candidate: ContactCandidate,
    normal: Vector3,
    axis: Vector3,
    root_position_m: Vector3,
    compression_direction: Vector3,
    contact_point: Vector3,
    base_compliance: Matrix3,
    start_displacement_m: Vector3,
    target_displacement_m: Vector3,
    tangential_velocity_m_per_s: Vector3,
    target_mechanical: _MechanicalResponse | None,
    tolerances: SingleSpineTolerances,
) -> tuple[str, float, BaseMotion] | None:
    """在 accepted→target 步内二分定位最早的接触、摩擦、限位或容量事件。"""

    if target_mechanical is None:
        return None
    start_mechanical = _mechanical_response(
        start_displacement_m,
        tangential_velocity_m_per_s,
        normal,
        base_compliance,
        compression_direction,
        friction,
        suspension,
        accepted,
        tolerances,
    )
    if start_mechanical is None:
        return None
    start_values = _event_values(
        geometry,
        material,
        suspension,
        candidate,
        axis,
        root_position_m,
        compression_direction,
        contact_point,
        tolerances,
        start_mechanical,
    )
    end_values = _event_values(
        geometry,
        material,
        suspension,
        candidate,
        axis,
        root_position_m,
        compression_direction,
        contact_point,
        tolerances,
        target_mechanical,
    )
    # 只跟踪与上一步 resident state 有关的边界，避免把无效反向条件当成新事件。
    active_keys: list[str] = ["normal"]
    previous_mode = (
        accepted.contact_submode
        if accepted.physical_state is PhysicalState.HARDSTOP
        else accepted.physical_state
    )
    if previous_mode is PhysicalState.STICK:
        active_keys.append("friction")
    if (
        accepted.physical_state is PhysicalState.HARDSTOP
        and accepted.spring_branch is SpringBranch.HARDSTOP
    ):
        active_keys.append("hardstop_release")
    else:
        active_keys.append("hardstop")
    active_keys.extend(
        key for key in start_values if key.startswith("capacity:")
    )
    crossings: list[tuple[float, str]] = []
    for key in active_keys:
        if key not in start_values or key not in end_values:
            continue
        start_value = start_values[key]
        end_value = end_values[key]
        if not (math.isfinite(start_value) and math.isfinite(end_value)):
            continue
        if start_value <= 0.0 or end_value > 0.0:
            continue
        # 已确认条件从正值跨到非正值，用 fraction 区间二分到事件容差。
        low = 0.0
        high = 1.0
        while high - low > tolerances.event_fraction:
            middle = 0.5 * (low + high)
            middle_displacement = start_displacement_m + middle * (
                target_displacement_m - start_displacement_m
            )
            middle_mechanical = _mechanical_response(
                middle_displacement,
                tangential_velocity_m_per_s,
                normal,
                base_compliance,
                compression_direction,
                friction,
                suspension,
                accepted,
                tolerances,
            )
            middle_values = (
                {}
                if middle_mechanical is None
                else _event_values(
                    geometry,
                    material,
                    suspension,
                    candidate,
                    axis,
                    root_position_m,
                    compression_direction,
                    contact_point,
                    tolerances,
                    middle_mechanical,
                )
            )
            middle_value = middle_values.get(key, math.nan)
            if not math.isfinite(middle_value) or middle_value <= 0.0:
                high = middle
            else:
                low = middle
        crossings.append((high, key))
    if not crossings:
        return None
    # 多个条件同一步跨越时按最小 fraction 截断，后续事件留给下一 trial 处理。
    fraction, key = min(crossings)
    return key, fraction, _interpolate_motion(
        accepted,
        target,
        start_displacement_m,
        target_displacement_m,
        fraction,
    )


def solve_single_spine(
    geometry: SpineGeometry,
    material: SpineMaterial,
    friction: FrictionParameters,
    suspension: SuspensionParameters,
    accepted: SpineAcceptedState,
    motion: BaseMotion,
    candidate: ContactCandidate | None,
    *,
    tolerances: SingleSpineTolerances,
) -> SingleSpineTrial:
    """由 accepted 状态构造一个不可变 trial；只有调用方显式 commit 才生效。"""

    # 阶段 1：验证状态归属并处理 FAILED、DETACH、REBOUND 等无需接触求解的 resident state。
    if accepted.spine_id != geometry.spine_id:
        raise ConfigurationError("accepted state and geometry spine_id do not match")
    axis = np.asarray(geometry.axis_root_to_tip, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    root_position = np.asarray(geometry.root_position_m, dtype=np.float64)
    target_displacement = np.asarray(
        motion.relative_displacement_m, dtype=np.float64
    )
    tangential_velocity = np.asarray(
        motion.relative_tangential_velocity_m_per_s,
        dtype=np.float64,
    )
    if accepted.physical_state is PhysicalState.FAILED:
        result = _zero_result(
            geometry,
            motion,
            PhysicalState.FAILED,
            accepted.spring_branch,
            ModelState.CLOSED,
            NumericalState.CONVERGED,
        )
        return SingleSpineTrial(
            geometry.spine_id, accepted.revision, accepted, result, True
        )
    if accepted.physical_state in {
        PhysicalState.DETACH,
        PhysicalState.REBOUND,
    }:
        return _advance_detach_or_rebound(
            geometry, suspension, accepted, motion
        )

    if candidate is None:
        # SEARCH 无候选时继续空载搜索；已有承载态失去候选则必须先进入 DETACH。
        if accepted.physical_state is PhysicalState.SEARCH:
            proposed = replace(
                accepted,
                relative_displacement_m=motion.relative_displacement_m,
                last_load_parameter=motion.load_parameter,
                revision=accepted.revision + 1,
            )
            result = _zero_result(
                geometry,
                motion,
                PhysicalState.SEARCH,
                accepted.spring_branch,
                ModelState.CLOSED,
                NumericalState.CONVERGED,
            )
            return SingleSpineTrial(
                geometry.spine_id, accepted.revision, proposed, result, True
            )
        return _detachment_trial(
            geometry, accepted, motion, "contact_candidate_missing"
        )

    # 阶段 2：检查 candidate identity、gap 和几何门控。活动接触不得直接切换 feature。
    candidate_id = candidate.candidate_id
    gap = candidate.signed_gap_m
    if (
        accepted.physical_state is not PhysicalState.SEARCH
        and accepted.candidate_id is not None
        and accepted.candidate_id != candidate_id
    ):
        raise ConfigurationError(
            "an active contact must detach before switching candidate_id"
        )
    if gap > tolerances.gap_m:
        # 正间隙表示尚未接触；负间隙超容差则表示离散候选发生不允许的穿透。
        if accepted.physical_state is PhysicalState.SEARCH:
            proposed = replace(
                accepted,
                relative_displacement_m=motion.relative_displacement_m,
                last_load_parameter=motion.load_parameter,
                revision=accepted.revision + 1,
            )
            result = _zero_result(
                geometry,
                motion,
                PhysicalState.SEARCH,
                accepted.spring_branch,
                ModelState.CLOSED,
                NumericalState.CONVERGED,
            )
            return SingleSpineTrial(
                geometry.spine_id, accepted.revision, proposed, result, True
            )
        return _detachment_trial(geometry, accepted, motion, "positive_gap")
    if gap < -tolerances.gap_m:
        if accepted.physical_state is PhysicalState.SEARCH:
            return _rejected_search_trial(
                geometry,
                accepted,
                motion,
                "candidate_penetration",
                ModelState.CLOSED,
                candidate.search_cursor,
            )
        return _detachment_trial(
            geometry, accepted, motion, "candidate_penetration"
        )
    normal, gate_reason, gate_model_state = _candidate_gate(candidate)
    if normal is None:
        if accepted.physical_state is PhysicalState.SEARCH:
            return _rejected_search_trial(
                geometry,
                accepted,
                motion,
                gate_reason,
                gate_model_state,
                candidate.search_cursor,
            )
        return _detachment_trial(
            geometry, accepted, motion, gate_reason
        )

    # 阶段 3：建立梁/悬架柔度，在目标位移上求活动集，并定位步内最早事件。
    compression_direction = -axis
    base_compliance = spine_linear_compliance(
        geometry, material, suspension
    )
    start_displacement = np.asarray(
        accepted.relative_displacement_m, dtype=np.float64
    )
    contact_point = _contact_point(geometry, candidate, normal)
    located_event: tuple[str, float] | None = None
    mechanical = _mechanical_response(
        target_displacement,
        tangential_velocity,
        normal,
        base_compliance,
        compression_direction,
        friction,
        suspension,
        accepted,
        tolerances,
    )
    located = _locate_earliest_event(
        geometry,
        material,
        friction,
        suspension,
        accepted,
        motion,
        candidate,
        normal,
        axis,
        root_position,
        compression_direction,
        contact_point,
        base_compliance,
        start_displacement,
        target_displacement,
        tangential_velocity,
        mechanical,
        tolerances,
    )
    if located is not None:
        event_key, event_fraction, event_motion = located
        if event_key in {"hardstop", "hardstop_release"}:
            # 限位边界处两侧切线不同，向事件后轻推一个数值小量以选择正确 resident 分支。
            post_fraction = min(
                1.0,
                event_fraction + 64.0 * tolerances.event_fraction,
            )
            evaluated_displacement = start_displacement + post_fraction * (
                target_displacement - start_displacement
            )
            motion = _interpolate_motion(
                accepted,
                motion,
                start_displacement,
                target_displacement,
                post_fraction,
            )
        else:
            evaluated_displacement = start_displacement + event_fraction * (
                target_displacement - start_displacement
            )
            motion = event_motion
        located_event = (event_key, event_fraction)
        mechanical = _mechanical_response(
            evaluated_displacement,
            tangential_velocity,
            normal,
            base_compliance,
            compression_direction,
            friction,
            suspension,
            accepted,
            tolerances,
            force_slip=event_key == "friction",
        )
    if mechanical is None:
        # 没有任何接触—弹簧分支同时相容时，trial 不可提交且 accepted 保持不变。
        result = _zero_result(
            geometry,
            motion,
            accepted.physical_state,
            accepted.spring_branch,
            ModelState.CLOSED,
            NumericalState.NONCONVERGED,
            assumptions=("spring_contact_active_set_not_closed",),
        )
        return SingleSpineTrial(
            geometry.spine_id, accepted.revision, accepted, result, False
        )
    if mechanical.normal_force_N <= tolerances.force_N:
        # 单边接触不能提供拉向表面的负法向反力。
        if accepted.physical_state is PhysicalState.SEARCH:
            if mechanical.normal_force_N >= -tolerances.force_N:
                # 候选恰好闭合但还没有反力时不是几何拒绝：力控阵列需要
                # 保留它，才能在秩不足后生成越过力容差的 contact seed。
                # 此处不产生 CONTACT_REJECT，也不推进 continuation cursor。
                proposed = replace(
                    accepted,
                    relative_displacement_m=motion.relative_displacement_m,
                    last_load_parameter=motion.load_parameter,
                    revision=accepted.revision + 1,
                )
                result = _zero_result(
                    geometry,
                    motion,
                    PhysicalState.SEARCH,
                    accepted.spring_branch,
                    ModelState.CLOSED,
                    NumericalState.CONVERGED,
                    assumptions=("zero_reaction_candidate_retained",),
                )
                return SingleSpineTrial(
                    geometry.spine_id,
                    accepted.revision,
                    proposed,
                    result,
                    True,
                )
            return _rejected_search_trial(
                geometry,
                accepted,
                motion,
                "nonpositive_normal_reaction",
                ModelState.CLOSED,
                candidate.search_cursor,
            )
        return _detachment_trial(
            geometry, accepted, motion, "nonpositive_normal_reaction"
        )

    events: list[Event] = []

    def append_event(
        event_type: EventType,
        from_state: PhysicalState,
        to_state: PhysicalState | None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """按本 trial 内顺序追加事件，并附带步内定位信息。"""

        merged_details = dict(details or {})
        if located_event is not None:
            merged_details.update(
                {
                    "located_event": located_event[0],
                    "event_fraction": located_event[1],
                }
            )
        events.append(
            _new_event(
                accepted,
                len(events) + 1,
                event_type,
                from_state,
                to_state,
                motion,
                merged_details,
            )
        )

    # 阶段 4：将连续力学解映射为合法的离散事件链和最终 resident state。
    current = accepted.physical_state
    reengaged = False
    if current is PhysicalState.SEARCH:
        # 新候选先产生瞬时 CONTACT，再在同一 trial 进入 STICK 或 SLIP。
        contact_event = (
            EventType.REENGAGE
            if accepted.completed_detach_cycles > 0
            else EventType.CONTACT
        )
        append_event(
            contact_event,
            PhysicalState.SEARCH,
            PhysicalState.CONTACT,
            {"candidate_id": candidate_id, "trial_proposal": True},
        )
        current = PhysicalState.CONTACT
        target = mechanical.contact_mode
        append_event(
            EventType.STICK_START
            if target is PhysicalState.STICK
            else EventType.SLIP_START,
            PhysicalState.CONTACT,
            target,
        )
        current = target
        reengaged = contact_event is EventType.REENGAGE
    elif current is PhysicalState.HARDSTOP:
        # HARDSTOP 是外层 resident state；内部摩擦模式通过 contact_submode 独立演化。
        if mechanical.spring.branch is not SpringBranch.HARDSTOP:
            append_event(
                EventType.HARDSTOP_RELEASE,
                PhysicalState.HARDSTOP,
                PhysicalState.CONTACT,
            )
            current = PhysicalState.CONTACT
            target = mechanical.contact_mode
            append_event(
                EventType.STICK_START
                if target is PhysicalState.STICK
                else EventType.SLIP_START,
                PhysicalState.CONTACT,
                target,
            )
            current = target
        elif (
            accepted.contact_submode is PhysicalState.STICK
            and mechanical.contact_mode is PhysicalState.SLIP
        ):
            append_event(
                EventType.SLIP_START,
                PhysicalState.STICK,
                PhysicalState.SLIP,
                {"resident_state": PhysicalState.HARDSTOP.value},
            )
        elif (
            accepted.contact_submode is PhysicalState.SLIP
            and mechanical.contact_mode is PhysicalState.STICK
        ):
            append_event(
                EventType.RESTICK,
                PhysicalState.SLIP,
                PhysicalState.STICK,
                {"resident_state": PhysicalState.HARDSTOP.value},
            )
    elif current is PhysicalState.STICK and mechanical.contact_mode is PhysicalState.SLIP:
        append_event(
            EventType.SLIP_START,
            PhysicalState.STICK,
            PhysicalState.SLIP,
        )
        current = PhysicalState.SLIP
    elif current is PhysicalState.SLIP and mechanical.contact_mode is PhysicalState.STICK:
        append_event(
            EventType.RESTICK,
            PhysicalState.SLIP,
            PhysicalState.STICK,
        )
        current = PhysicalState.STICK

    contact_submode: PhysicalState | None = None
    if mechanical.spring.branch is SpringBranch.HARDSTOP:
        # 首次进入上限位时把当前 STICK/SLIP 保存为硬限位子模态。
        contact_submode = mechanical.contact_mode
        if current is not PhysicalState.HARDSTOP:
            append_event(
                EventType.HARDSTOP,
                current,
                PhysicalState.HARDSTOP,
                {"contact_submode": contact_submode.value},
            )
            current = PhysicalState.HARDSTOP

    # 阶段 5：评估材料容量。永久断开将当前输出力清零，其他失效停在模型边界。
    force = mechanical.force_N
    tangential_force = mechanical.tangential_force_N
    assessments, diagnostics = _capacity_assessments(
        geometry,
        material,
        candidate,
        axis,
        root_position,
        force,
        mechanical.normal_force_N,
        tangential_force,
        contact_point,
    )
    model_state = _aggregate_model_state(assessments)
    failure = _failure_from_assessments(
        assessments, material, tolerances
    )
    force_before_failure = force.copy()
    tangent_matrix = mechanical.tangent_N_per_m
    if failure is not None:
        to_state = (
            PhysicalState.FAILED
            if failure.continuation_action
            is ContinuationAction.PERMANENT_REMOVE
            else None
        )
        append_event(
            EventType.MATERIAL_FAILURE,
            current,
            to_state,
            {
                "failure": asdict(failure),
                "force_before_N": list(_tuple3(force_before_failure)),
            },
        )
        if to_state is PhysicalState.FAILED:
            # 失效事件保留 force_before_N 供追溯，但提交后的断开刺不再向背板传力。
            current = PhysicalState.FAILED
            contact_submode = None
            force = np.zeros(3, dtype=np.float64)
            tangential_force = np.zeros(3, dtype=np.float64)
            tangent_matrix = np.zeros((3, 3), dtype=np.float64)
        else:
            model_state = ModelState.OUT_OF_SCOPE

    # 阶段 6：整理边界裕度和互补残差；这些诊断决定数值状态及是否可提交。
    axial_load = float(np.dot(force_before_failure, compression_direction))
    stiffness = suspension.axial_spring_stiffness_N_per_m
    travel = suspension.axial_spring_travel_m
    upper_margin = (
        None
        if stiffness is None or travel is None
        else stiffness * travel - axial_load
    )
    margins: dict[str, float | None] = {
        "normal_N": mechanical.normal_force_N,
        "friction_N": mechanical.current_friction_margin_N,
        "spring_lower_N": axial_load if stiffness is not None else None,
        "spring_upper_N": upper_margin,
        "travel_m": (
            None
            if travel is None
            else travel - mechanical.spring.displacement_m
        ),
    }
    margins.update(
        {
            name: assessment.margin
            for name, assessment in assessments.items()
        }
    )
    tangential_force_norm = float(
        np.linalg.norm(mechanical.tangential_force_N)
    )
    complementarity = {
        "penetration_m": max(-gap, 0.0),
        "negative_normal_N": max(-mechanical.normal_force_N, 0.0),
        "gap_force_Nm": abs(gap * mechanical.normal_force_N),
        "static_friction_cone_N": max(
            tangential_force_norm
            - friction.static_coefficient * mechanical.normal_force_N,
            0.0,
        ),
        "kinetic_slip_cone_N": (
            abs(
                tangential_force_norm
                - friction.kinetic_coefficient * mechanical.normal_force_N
            )
            if mechanical.contact_mode is PhysicalState.SLIP
            else 0.0
        ),
        "spring": mechanical.spring.complementarity_residual,
        "balance_m": mechanical.balance_residual_m,
    }
    residuals_closed = (
        complementarity["penetration_m"] <= tolerances.gap_m
        and complementarity["negative_normal_N"] <= tolerances.force_N
        and complementarity["gap_force_Nm"]
        <= tolerances.gap_m * tolerances.force_N
        and complementarity["static_friction_cone_N"]
        <= tolerances.friction_N
        and complementarity["kinetic_slip_cone_N"]
        <= tolerances.friction_N
        and complementarity["spring"] <= tolerances.spring_N
        and complementarity["balance_m"] <= tolerances.gap_m
    )
    numerical_state = (
        NumericalState.CONVERGED
        if residuals_closed
        else NumericalState.INVALID_RESIDUAL
    )
    stopped_at_model_limit = (
        failure is not None
        and failure.continuation_action is ContinuationAction.STOP_MODEL_LIMIT
    )
    committable = (
        numerical_state is NumericalState.CONVERGED
        and not stopped_at_model_limit
    )
    # 阶段 7：分别构造只读 result 和 proposed state；求解函数本身不修改 accepted。
    result_wrench = _root_wrench(
        geometry, root_position, contact_point, force
    )
    elastic_displacement = _tuple3(mechanical.elastic_displacement_m)
    slip_displacement = _tuple3(mechanical.slip_displacement_m)
    result = SingleSpineResult(
        wall_force_N=_tuple3(force),
        root_wrench=result_wrench,
        local_tangent_N_per_m=_tuple_matrix3(tangent_matrix),
        physical_state=current,
        contact_submode=contact_submode,
        spring_branch=mechanical.spring.branch,
        spring_displacement_m=mechanical.spring.displacement_m,
        normal_force_N=(
            0.0
            if current is PhysicalState.FAILED
            else mechanical.normal_force_N
        ),
        tangential_force_N=_tuple3(tangential_force),
        elastic_displacement_m=elastic_displacement,
        slip_displacement_m=slip_displacement,
        model_state=model_state,
        numerical_state=numerical_state,
        margins=margins,
        capacity_assessments=assessments,
        complementarity_residuals=complementarity,
        diagnostics=diagnostics,
        events=tuple(events),
        failure=failure,
        evaluated_motion=motion,
        assumptions=(
            "quasistatic",
            "single_point_contact",
            "linear_elastic_low_order_beam",
            "dynamic_stability_out_of_scope",
        ),
    )
    proposed = replace(
        accepted,
        physical_state=current,
        contact_submode=contact_submode,
        spring_branch=mechanical.spring.branch,
        candidate_id=(None if current is PhysicalState.FAILED else candidate_id),
        contact_point_m=(
            None if current is PhysicalState.FAILED else _tuple3(contact_point)
        ),
        contact_normal=(
            None if current is PhysicalState.FAILED else _tuple3(normal)
        ),
        relative_displacement_m=motion.relative_displacement_m,
        elastic_displacement_m=elastic_displacement,
        slip_displacement_m=slip_displacement,
        search_cursor=candidate.search_cursor,
        reengagement_count=(
            accepted.reengagement_count + 1
            if reengaged
            else accepted.reengagement_count
        ),
        damage_history=(
            accepted.damage_history
            if failure is None
            else accepted.damage_history
            + (f"{failure.failure_object}:{failure.failure_mode}",)
        ),
        last_load_parameter=motion.load_parameter,
        event_sequence=(
            events[-1].sequence if events else accepted.event_sequence
        ),
        revision=accepted.revision + 1,
    )
    return SingleSpineTrial(
        spine_id=geometry.spine_id,
        base_revision=accepted.revision,
        proposed_state=proposed,
        result=result,
        committable=committable,
    )


def commit_single_spine_trial(
    accepted: SpineAcceptedState, trial: SingleSpineTrial
) -> SpineAcceptedState:
    """仅当 spine、base revision 和数值状态全部匹配时提交 proposed state。"""

    if trial.spine_id != accepted.spine_id:
        raise ConfigurationError("trial and accepted state spine_id do not match")
    # revision 检查阻止基于旧 accepted 状态计算的并发/缓存 trial 覆盖较新状态。
    if trial.base_revision != accepted.revision:
        raise ConfigurationError("stale single-spine trial cannot be committed")
    if not trial.committable:
        raise ConfigurationError("noncommittable single-spine trial cannot be committed")
    if trial.result.numerical_state is not NumericalState.CONVERGED:
        raise ConfigurationError("nonconverged single-spine trial cannot be committed")
    return trial.proposed_state


__all__ = [
    "BaseMotion",
    "CapacityAssessment",
    "FailurePayload",
    "FrictionParameters",
    "SingleSpineResult",
    "SingleSpineTolerances",
    "SingleSpineTrial",
    "SpineAcceptedState",
    "SpineGeometry",
    "SpineMaterial",
    "SpringSolution",
    "SuspensionParameters",
    "WedgeResult",
    "commit_single_spine_trial",
    "solve_single_spine",
    "solve_unilateral_spring",
    "solve_wedge_2d",
    "spine_linear_compliance",
]
