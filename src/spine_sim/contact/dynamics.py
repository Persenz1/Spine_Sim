"""Continuous-preload time-domain dynamics for one flexible spine.

The production path uses a deterministic Moreau-style impulse step.  The
horizontal holder motion is prescribed, while holder Z and the axial and
transverse spine modes remain dynamic.  Contact impulses enforce the M1
finite-tip envelope without a hidden penalty stiffness.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from spine_sim.core.states import ModelState, NumericalState
from spine_sim.terrain.models import TrackGeometry

from .errors import ContactConfigurationError, ContactGeometryError
from .geometry import TrackInterpolator
from .models import (
    ContactState,
    EventLabel,
    M2_MODEL_LEVEL,
    PathTerminalState,
    SolverSettings,
    SpineParameters,
    SpringState,
)
from .solver import LegacyPrescribedPoseConstitutiveCore


@dataclass(frozen=True)
class DynamicContactSettings:
    """Rigid nonsmooth contact and Coulomb impulse parameters."""

    normal_model: str = "rigid_moreau"
    restitution_coefficient: float = 0.0
    position_correction: float = 1.0
    activation_tolerance_m: float = 2e-9
    impact_velocity_threshold_m_s: float = 1e-5
    maximum_contact_force_n: float = 250.0
    projection_iterations: int = 6

    def __post_init__(self) -> None:
        if self.normal_model != "rigid_moreau":
            raise ContactConfigurationError(
                "DynamicContactSettings currently supports rigid_moreau only"
            )
        if not 0.0 <= self.restitution_coefficient <= 1.0:
            raise ContactConfigurationError(
                "restitution_coefficient must lie in [0, 1]"
            )
        if not 0.0 < self.position_correction <= 1.0:
            raise ContactConfigurationError(
                "position_correction must lie in (0, 1]"
            )
        for name in (
            "activation_tolerance_m",
            "impact_velocity_threshold_m_s",
            "maximum_contact_force_n",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContactConfigurationError(
                    f"{name} must be finite and positive"
                )
        if self.projection_iterations < 1:
            raise ContactConfigurationError(
                "projection_iterations must be positive"
            )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "DynamicContactSettings":
        return cls(**dict(value or {}))


@dataclass(frozen=True)
class DynamicIntegratorSettings:
    """Fixed-step nonsmooth integration controls.

    A fixed internal step makes deterministic replay and step-halving audits
    exact.  Output spacing is independent and belongs to the experiment.
    """

    method: str = "moreau_implicit_euler"
    time_step_s: float = 1e-3
    settling_time_s: float = 0.25
    settling_velocity_tolerance_m_s: float = 2e-5
    maximum_settling_time_s: float = 2.0
    maximum_steps: int = 2_000_000

    def __post_init__(self) -> None:
        if self.method != "moreau_implicit_euler":
            raise ContactConfigurationError(
                "DynamicIntegratorSettings supports moreau_implicit_euler only"
            )
        for name in (
            "time_step_s",
            "settling_time_s",
            "settling_velocity_tolerance_m_s",
            "maximum_settling_time_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContactConfigurationError(
                    f"{name} must be finite and positive"
                )
        if self.maximum_settling_time_s < self.settling_time_s:
            raise ContactConfigurationError(
                "maximum_settling_time_s cannot be less than settling_time_s"
            )
        if self.maximum_steps < 1:
            raise ContactConfigurationError("maximum_steps must be positive")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "DynamicIntegratorSettings":
        return cls(**dict(value or {}))


@dataclass(frozen=True)
class DynamicExperimentSettings:
    initial_center_x_m: float
    drag_length_m: float
    drag_speed_m_s: float = 1e-3
    constant_preload_n: float = 0.5
    holder_effective_mass_kg: float = 0.05
    holder_vertical_damping_n_s_m: float = 1.0
    maximum_preload_approach_m: float = 8e-3
    output_spacing_m: float = 10e-6
    effective_normal_force_min_n: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            "drag_length_m",
            "drag_speed_m_s",
            "holder_effective_mass_kg",
            "maximum_preload_approach_m",
            "output_spacing_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContactConfigurationError(
                    f"{name} must be finite and positive"
                )
        if not math.isfinite(self.initial_center_x_m):
            raise ContactConfigurationError(
                "initial_center_x_m must be finite"
            )
        if (
            not math.isfinite(self.constant_preload_n)
            or self.constant_preload_n < 0.0
        ):
            raise ContactConfigurationError(
                "constant_preload_n must be finite and non-negative"
            )
        if (
            not math.isfinite(self.holder_vertical_damping_n_s_m)
            or self.holder_vertical_damping_n_s_m < 0.0
        ):
            raise ContactConfigurationError(
                "holder_vertical_damping_n_s_m must be finite and non-negative"
            )
        if (
            not math.isfinite(self.effective_normal_force_min_n)
            or self.effective_normal_force_min_n < 0.0
        ):
            raise ContactConfigurationError(
                "effective_normal_force_min_n must be finite and non-negative"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DynamicExperimentSettings":
        return cls(**dict(value))


@dataclass(frozen=True)
class DynamicState:
    time_s: float
    generalized_position: tuple[float, float, float]
    generalized_velocity: tuple[float, float, float]
    contact_state: ContactState
    spring_state: SpringState
    has_contacted: bool
    accepted_steps: int
    cumulative_friction_dissipation_j: float = 0.0
    cumulative_damping_dissipation_j: float = 0.0


@dataclass(frozen=True)
class DynamicPathPoint:
    time_s: float
    path_position_m: float
    holder_xz_m: tuple[float, float]
    holder_velocity_xz_m_s: tuple[float, float]
    holder_acceleration_xz_m_s2: tuple[float, float]
    center_xz_m: tuple[float, float]
    center_velocity_xz_m_s: tuple[float, float]
    center_acceleration_xz_m_s2: tuple[float, float]
    support_xyz_m: tuple[float, float, float] | None
    tangent_xz: tuple[float, float] | None
    normal_xz: tuple[float, float] | None
    gap_m: float
    contact_state: ContactState
    spring_state: SpringState
    event_label: EventLabel
    event_labels: tuple[str, ...]
    external_preload_n: float
    wall_on_spine_force_xz_n: tuple[float, float]
    spine_on_plate_wrench_about_holder: tuple[
        float, float, float, float, float, float
    ]
    normal_force_n: float
    tangential_force_n: float
    normal_impulse_n_s: float
    tangential_impulse_n_s: float
    impact_velocity_m_s: float
    axial_displacement_m: float
    transverse_displacement_m: float
    axial_velocity_m_s: float
    transverse_velocity_m_s: float
    axial_force_n: float
    transverse_force_n: float
    spring_compression_m: float
    spring_travel_margin_m: float
    kinetic_energy_j: float
    structural_energy_j: float
    preload_work_increment_j: float
    drive_work_increment_j: float
    friction_dissipation_increment_j: float
    damping_dissipation_increment_j: float
    energy_residual_j: float
    dynamic_residual_n: float
    actual_time_step_s: float
    nonlinear_iterations: int
    bending_stress_pa: float
    euler_buckling_margin_n: float
    numerical_state: NumericalState
    model_state: ModelState


@dataclass(frozen=True)
class DynamicPathSummary:
    preload_mode: str
    constant_preload_n: float
    drag_speed_m_s: float
    initial_preload_success: bool
    ever_contacted: bool
    ever_loaded: bool
    contact_fraction: float
    effective_load_fraction: float
    global_pull_force_peak_n: float
    global_pull_force_steady_peak_n: float
    global_pull_force_median_n: float
    global_pull_force_p10_n: float
    global_pull_force_p25_n: float
    normal_force_range_n: tuple[float, float]
    holder_z_range_m: tuple[float, float]
    holder_speed_peak_m_s: float
    holder_acceleration_peak_m_s2: float
    impact_velocity_peak_m_s: float
    event_counts: Mapping[str, int]
    maximum_bending_stress_pa: float
    minimum_euler_buckling_margin_n: float
    maximum_abs_dynamic_residual_n: float
    maximum_abs_energy_residual_j: float
    internal_time_step_s: float
    accepted_steps: int
    rejected_steps: int
    time_step_convergence_checked: bool
    contact_parameter_convergence_checked: bool
    numerical_state: NumericalState
    model_state: ModelState
    run_terminal_state: PathTerminalState
    termination_reason: str
    formal_ranking_eligible: bool


@dataclass(frozen=True)
class DynamicSingleSpineResult:
    parameters: SpineParameters
    experiment: DynamicExperimentSettings
    contact: DynamicContactSettings
    integrator: DynamicIntegratorSettings
    track_id: str
    points: tuple[DynamicPathPoint, ...]
    summary: DynamicPathSummary
    assumptions: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class _StepData:
    state: DynamicState
    point: DynamicPathPoint
    active: bool


class DynamicSingleSpineUnit:
    """Pure mechanics helper reused by the joint common-backplate M3 solver."""

    def __init__(
        self,
        parameters: SpineParameters,
        track: TrackGeometry,
        contact: DynamicContactSettings,
    ) -> None:
        if not parameters.beam_enabled:
            raise ContactConfigurationError(
                "dynamic M2 requires beam_enabled=true"
            )
        self.parameters = parameters
        self.track = track
        self.geometry = TrackInterpolator(track, parameters)
        self.contact = contact
        self.a = parameters.axis_xz
        self.b = parameters.transverse_xz
        self.modes_mass = np.array(
            [
                math.nan,
                parameters.axial_modal_mass_kg,
                parameters.transverse_modal_mass_kg,
            ],
            dtype=np.float64,
        )
        if min(self.modes_mass[1:]) <= 0.0:
            raise ContactConfigurationError(
                "dynamic modal masses must be positive"
            )

    def axial_response(
        self, shortening_m: float
    ) -> tuple[float, float, float, SpringState, float]:
        """Return force, tangent, energy, spring state and compression."""

        p = self.parameters
        ca = p.axial_compliance_m_n
        if ca <= 0.0:
            raise ContactConfigurationError(
                "dynamic axial mode requires positive beam axial compliance"
            )
        if p.axial_mode.value == "rigid":
            force = shortening_m / ca
            return (
                force,
                1.0 / ca,
                0.5 * force * shortening_m,
                SpringState.LOWER_STOP,
                0.0,
            )
        stiffness = float(p.spring_stiffness_n_m)
        travel = p.spring_travel_m
        if shortening_m <= 0.0:
            force = shortening_m / ca
            return (
                force,
                1.0 / ca,
                0.5 * shortening_m * force,
                SpringState.LOWER_STOP,
                0.0,
            )
        effective_compliance = ca + 1.0 / stiffness
        interior_limit = travel + ca * stiffness * travel
        if shortening_m < interior_limit:
            tangent = 1.0 / effective_compliance
            force = tangent * shortening_m
            compression = force / stiffness
            return (
                force,
                tangent,
                0.5 * force * shortening_m,
                SpringState.INTERIOR,
                compression,
            )
        beam_shortening = shortening_m - travel
        force = beam_shortening / ca
        energy = (
            0.5 * stiffness * travel**2
            + 0.5 * beam_shortening * force
        )
        return (
            force,
            1.0 / ca,
            energy,
            SpringState.HARD_STOP,
            travel,
        )

    @property
    def transverse_stiffness_n_m(self) -> float:
        compliance = self.parameters.transverse_compliance_m_n
        if compliance <= 0.0:
            raise ContactConfigurationError(
                "dynamic transverse mode requires positive compliance"
            )
        return 1.0 / compliance

    @property
    def euler_buckling_load_n(self) -> float:
        # Cantilever effective length is 2L.
        p = self.parameters
        return (
            math.pi**2
            * p.young_modulus_pa
            * p.second_moment_m4
            / (4.0 * p.exposed_length_m**2)
        )

    def kinematics(
        self,
        holder_x_m: float,
        drag_speed_m_s: float,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        holder = np.array([holder_x_m, q[0]], dtype=np.float64)
        center = (
            holder
            + self.parameters.exposed_length_m * self.a
            - q[1] * self.a
            + q[2] * self.b
        )
        center_velocity = (
            np.array([drag_speed_m_s, v[0]], dtype=np.float64)
            - v[1] * self.a
            + v[2] * self.b
        )
        return center, center_velocity

    def generalized_contact_jacobian(
        self, direction_xz: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.array(
            [
                direction_xz[1],
                float((-self.a) @ direction_xz),
                float(self.b @ direction_xz),
            ],
            dtype=np.float64,
        )


class DynamicSingleSpineExperiment:
    """Integrate one spine under a continuous external normal preload."""

    def __init__(
        self,
        parameters: SpineParameters,
        track: TrackGeometry,
        experiment: DynamicExperimentSettings,
        contact: DynamicContactSettings | None = None,
        integrator: DynamicIntegratorSettings | None = None,
    ) -> None:
        self.parameters = parameters
        self.track = track
        self.experiment = experiment
        self.contact = contact or DynamicContactSettings()
        self.integrator = integrator or DynamicIntegratorSettings()
        self.unit = DynamicSingleSpineUnit(
            parameters,
            track,
            self.contact,
        )
        self._mass = self.unit.modes_mass.copy()
        self._mass[0] = experiment.holder_effective_mass_kg

    def _model_state(self) -> ModelState:
        if (
            self.parameters.rod_clearance_mode == "unclosed"
            or self.parameters.yield_strength_pa is None
        ):
            return ModelState.PARAMETER_UNCLOSED
        return ModelState.COVERED

    def _damping_and_tangent(
        self, q: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, SpringState, float]:
        axial_force, axial_tangent, axial_energy, spring_state, compression = (
            self.unit.axial_response(float(q[1]))
        )
        transverse_stiffness = self.unit.transverse_stiffness_n_m
        damping = np.array(
            [
                self.experiment.holder_vertical_damping_n_s_m,
                2.0
                * self.parameters.axial_damping_ratio
                * math.sqrt(axial_tangent * self._mass[1]),
                2.0
                * self.parameters.transverse_damping_ratio
                * math.sqrt(transverse_stiffness * self._mass[2]),
            ],
            dtype=np.float64,
        )
        tangent = np.array(
            [0.0, axial_tangent, transverse_stiffness],
            dtype=np.float64,
        )
        return (
            damping,
            tangent,
            axial_force,
            axial_energy,
            spring_state,
            compression,
        )

    def _free_implicit_step(
        self,
        q0: NDArray[np.float64],
        v0: NDArray[np.float64],
        dt: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
        v = v0.copy()
        iterations = 0
        for iterations in range(1, 21):
            q = q0 + dt * v
            (
                damping,
                tangent,
                axial_force,
                _axial_energy,
                _spring_state,
                _compression,
            ) = self._damping_and_tangent(q)
            restoring = np.array(
                [
                    0.0,
                    axial_force,
                    self.unit.transverse_stiffness_n_m * q[2],
                ],
                dtype=np.float64,
            )
            external = np.array(
                [-self.experiment.constant_preload_n, 0.0, 0.0],
                dtype=np.float64,
            )
            residual = (
                self._mass * (v - v0) / dt
                + damping * v
                + restoring
                - external
            )
            diagonal = self._mass / dt + damping + dt * tangent
            update = residual / diagonal
            v -= update
            if float(np.max(np.abs(update))) <= 1e-11:
                break
        q = q0 + dt * v
        damping, tangent, *_ = self._damping_and_tangent(q)
        effective_inverse = 1.0 / (
            self._mass + dt * damping + dt**2 * tangent
        )
        return q, v, effective_inverse, iterations

    def _contact_project(
        self,
        q0: NDArray[np.float64],
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        effective_inverse: NDArray[np.float64],
        *,
        holder_x_m: float,
        drag_speed_m_s: float,
        dt: float,
        was_active: bool,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        Any,
        float,
        float,
        NDArray[np.float64],
        float,
        bool,
        bool,
        int,
    ]:
        q_free = q.copy()
        v_free = v.copy()
        normal_impulse = 0.0
        tangential_impulse = 0.0
        impact_velocity = 0.0
        active = False
        sticking = False
        geometry = None
        generalized_impulse = np.zeros(3, dtype=np.float64)
        iterations = 0

        for iterations in range(1, self.contact.projection_iterations + 1):
            center_free, velocity_free = self.unit.kinematics(
                holder_x_m,
                drag_speed_m_s,
                q_free,
                v_free,
            )
            query_x = (
                float(center_free[0])
                if geometry is None
                else float(
                    self.unit.kinematics(
                        holder_x_m,
                        drag_speed_m_s,
                        q,
                        v,
                    )[0][0]
                )
            )
            geometry = self.unit.geometry.query(query_x)
            gap = float(
                center_free[1] - geometry.envelope_height_m
            )
            normal = np.asarray(geometry.normal_xz, dtype=np.float64)
            tangent = np.asarray(geometry.tangent_xz, dtype=np.float64)
            jn = self.unit.generalized_contact_jacobian(normal)
            jt = self.unit.generalized_contact_jacobian(tangent)
            normal_velocity = float(velocity_free @ normal)
            tangential_velocity = float(velocity_free @ tangent)

            should_activate = (
                gap <= self.contact.activation_tolerance_m
                and (was_active or normal_velocity < 0.0 or gap < 0.0)
            )
            if not should_activate:
                break
            active = True
            if not was_active and normal_velocity < 0.0:
                impact_velocity = max(impact_velocity, -normal_velocity)

            wnn = float(np.sum(jn * effective_inverse * jn))
            wtt = float(np.sum(jt * effective_inverse * jt))
            wnt = float(np.sum(jn * effective_inverse * jt))
            if wnn <= 0.0 or wtt <= 0.0:
                raise ContactConfigurationError(
                    "contact effective mass must be positive"
                )
            normal_rhs = max(
                0.0,
                -self.contact.position_correction
                * gap
                / dt,
                -(1.0 + self.contact.restitution_coefficient)
                * normal_velocity
                if not was_active
                else -normal_velocity,
            )
            matrix = np.array(
                [[wnn, wnt], [wnt, wtt]],
                dtype=np.float64,
            )
            rhs = np.array(
                [normal_rhs, -tangential_velocity],
                dtype=np.float64,
            )
            try:
                stick_impulse = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError as exc:
                raise ContactConfigurationError(
                    "contact Delassus matrix is singular"
                ) from exc
            candidate_normal = float(stick_impulse[0])
            candidate_tangent = float(stick_impulse[1])
            if (
                candidate_normal >= 0.0
                and abs(candidate_tangent)
                <= self.parameters.static_friction
                * candidate_normal
                + 1e-18
            ):
                normal_impulse = candidate_normal
                tangential_impulse = candidate_tangent
                sticking = True
            else:
                direction = (
                    math.copysign(1.0, tangential_velocity)
                    if abs(tangential_velocity) > 1e-15
                    else math.copysign(1.0, drag_speed_m_s)
                )
                denominator = (
                    wnn
                    - self.parameters.kinetic_friction
                    * direction
                    * wnt
                )
                if denominator <= 0.0:
                    raise ContactConfigurationError(
                        "sliding contact effective mass is non-positive"
                    )
                normal_impulse = max(0.0, normal_rhs / denominator)
                tangential_impulse = (
                    -self.parameters.kinetic_friction
                    * normal_impulse
                    * direction
                )
                sticking = False
            v = v_free + effective_inverse * (
                jn * normal_impulse + jt * tangential_impulse
            )
            generalized_impulse = (
                jn * normal_impulse + jt * tangential_impulse
            )
            q = q0 + dt * v

            center_after, _ = self.unit.kinematics(
                holder_x_m,
                drag_speed_m_s,
                q,
                v,
            )
            geometry_after = self.unit.geometry.query(float(center_after[0]))
            gap_after = float(
                center_after[1] - geometry_after.envelope_height_m
            )
            if abs(gap_after) <= self.contact.activation_tolerance_m:
                geometry = geometry_after
                break
            geometry = geometry_after

        return (
            q,
            v,
            geometry,
            normal_impulse,
            tangential_impulse,
            generalized_impulse,
            impact_velocity,
            active,
            sticking,
            iterations,
        )

    def _step(
        self,
        state: DynamicState,
        *,
        holder_x_m: float,
        drag_speed_m_s: float,
        dt: float,
    ) -> _StepData:
        q0 = np.asarray(state.generalized_position, dtype=np.float64)
        v0 = np.asarray(state.generalized_velocity, dtype=np.float64)
        q, v, effective_inverse, nonlinear_iterations = (
            self._free_implicit_step(q0, v0, dt)
        )
        was_active = state.contact_state not in {
            ContactState.FREE,
            ContactState.DETACH_EVENT,
        }
        (
            q,
            v,
            geometry,
            normal_impulse,
            tangential_impulse,
            generalized_contact_impulse,
            impact_velocity,
            active,
            sticking,
            projection_iterations,
        ) = self._contact_project(
            q0,
            q,
            v,
            effective_inverse,
            holder_x_m=holder_x_m,
            drag_speed_m_s=drag_speed_m_s,
            dt=dt,
            was_active=was_active,
        )

        center, center_velocity = self.unit.kinematics(
            holder_x_m,
            drag_speed_m_s,
            q,
            v,
        )
        acceleration = (v - v0) / dt
        center_acceleration = (
            np.array([0.0, acceleration[0]], dtype=np.float64)
            - acceleration[1] * self.unit.a
            + acceleration[2] * self.unit.b
        )
        if geometry is None:
            geometry = self.unit.geometry.query(float(center[0]))
        gap = float(center[1] - geometry.envelope_height_m)
        normal = np.asarray(geometry.normal_xz, dtype=np.float64)
        tangent = np.asarray(geometry.tangent_xz, dtype=np.float64)
        normal_force = normal_impulse / dt
        tangential_force = tangential_impulse / dt
        wall_force = normal_force * normal + tangential_force * tangent
        if float(np.linalg.norm(wall_force)) > self.contact.maximum_contact_force_n:
            raise RuntimeError(
                "structural_boundary: dynamic contact force exceeds safety limit"
            )

        (
            damping,
            _tangent_stiffness,
            axial_force,
            axial_energy,
            spring_state,
            compression,
        ) = self._damping_and_tangent(q)
        transverse_force = self.unit.transverse_stiffness_n_m * q[2]
        structural_energy = (
            axial_energy
            + 0.5 * self.unit.transverse_stiffness_n_m * q[2] ** 2
        )
        kinetic_energy = 0.5 * float(np.sum(self._mass * v**2))
        damping_dissipation = float(np.sum(damping * v**2)) * dt
        tangential_speed = float(center_velocity @ tangent)
        friction_dissipation = max(
            0.0,
            -tangential_force * tangential_speed * dt,
        )
        preload_work = (
            -self.experiment.constant_preload_n * (q[0] - q0[0])
        )
        internal_tip_force = (
            axial_force * self.unit.a
            - transverse_force * self.unit.b
        )
        drive_force_x = float(internal_tip_force[0])
        drive_work = drive_force_x * drag_speed_m_s * dt

        jn = self.unit.generalized_contact_jacobian(normal)
        jt = self.unit.generalized_contact_jacobian(tangent)
        restoring = np.array(
            [0.0, axial_force, transverse_force],
            dtype=np.float64,
        )
        external = np.array(
            [-self.experiment.constant_preload_n, 0.0, 0.0],
            dtype=np.float64,
        )
        generalized_contact = generalized_contact_impulse / dt
        dynamic_residual = float(
            np.max(
                np.abs(
                    self._mass * acceleration
                    + damping * v
                    + restoring
                    - external
                    - generalized_contact
                )
            )
        )

        if active:
            if not was_active:
                contact_state = (
                    ContactState.IMPACT_EVENT
                    if impact_velocity
                    >= self.contact.impact_velocity_threshold_m_s
                    else ContactState.RECONTACT_EVENT
                )
            else:
                contact_state = (
                    ContactState.STICK if sticking else ContactState.SLIDE
                )
        else:
            contact_state = (
                ContactState.DETACH_EVENT
                if was_active
                else ContactState.FREE
            )

        event_labels: list[str] = []
        event_label = EventLabel.NONE
        if active and not was_active:
            event_labels.append(EventLabel.RECONTACT.value)
            event_label = EventLabel.RECONTACT
            if impact_velocity >= self.contact.impact_velocity_threshold_m_s:
                event_labels.append(EventLabel.IMPACT.value)
                event_label = EventLabel.IMPACT
        elif not active and was_active:
            event_labels.append(EventLabel.DETACH_TO_FREE.value)
            event_label = EventLabel.DETACH_TO_FREE
        elif active and not sticking and state.contact_state is ContactState.STICK:
            event_labels.append(EventLabel.SLIP_START.value)
            event_label = EventLabel.SLIP_START
        elif active and sticking and state.contact_state is ContactState.SLIDE:
            event_labels.append(EventLabel.STICK_RECOVERED.value)
            event_label = EventLabel.STICK_RECOVERED
        if (
            spring_state is SpringState.HARD_STOP
            and state.spring_state is not SpringState.HARD_STOP
        ):
            event_labels.append(EventLabel.HARD_STOP.value)
            if event_label is EventLabel.NONE:
                event_label = EventLabel.HARD_STOP

        plate_force = -wall_force
        support = geometry.support_xyz_m if active else None
        if support is None:
            moment = np.zeros(3, dtype=np.float64)
        else:
            holder_xyz = np.array([holder_x_m, 0.0, q[0]], dtype=np.float64)
            lever = np.asarray(support, dtype=np.float64) - holder_xyz
            plate_force_xyz = np.array(
                [plate_force[0], 0.0, plate_force[1]],
                dtype=np.float64,
            )
            moment = np.cross(lever, plate_force_xyz)
        bending_stress = (
            32.0
            * abs(transverse_force)
            * self.parameters.exposed_length_m
            / (math.pi * self.parameters.diameter_m**3)
        )
        buckling_margin = self.unit.euler_buckling_load_n - max(
            axial_force, 0.0
        )
        model_state = self._model_state()

        previous_energy = 0.5 * float(np.sum(self._mass * v0**2))
        previous_axial = self.unit.axial_response(float(q0[1]))[2]
        previous_structure = (
            previous_axial
            + 0.5 * self.unit.transverse_stiffness_n_m * q0[2] ** 2
        )
        energy_residual = (
            kinetic_energy
            + structural_energy
            - previous_energy
            - previous_structure
            - preload_work
            - drive_work
            + damping_dissipation
            + friction_dissipation
        )

        new_state = DynamicState(
            time_s=state.time_s + dt,
            generalized_position=tuple(float(value) for value in q),
            generalized_velocity=tuple(float(value) for value in v),
            contact_state=contact_state,
            spring_state=spring_state,
            has_contacted=state.has_contacted or active,
            accepted_steps=state.accepted_steps + 1,
            cumulative_friction_dissipation_j=(
                state.cumulative_friction_dissipation_j
                + friction_dissipation
            ),
            cumulative_damping_dissipation_j=(
                state.cumulative_damping_dissipation_j
                + damping_dissipation
            ),
        )
        point = DynamicPathPoint(
            time_s=new_state.time_s,
            path_position_m=0.0,
            holder_xz_m=(float(holder_x_m), float(q[0])),
            holder_velocity_xz_m_s=(float(drag_speed_m_s), float(v[0])),
            holder_acceleration_xz_m_s2=(0.0, float(acceleration[0])),
            center_xz_m=tuple(float(value) for value in center),
            center_velocity_xz_m_s=tuple(
                float(value) for value in center_velocity
            ),
            center_acceleration_xz_m_s2=tuple(
                float(value) for value in center_acceleration
            ),
            support_xyz_m=support,
            tangent_xz=geometry.tangent_xz if active else None,
            normal_xz=geometry.normal_xz if active else None,
            gap_m=gap,
            contact_state=contact_state,
            spring_state=spring_state,
            event_label=event_label,
            event_labels=tuple(event_labels),
            external_preload_n=self.experiment.constant_preload_n,
            wall_on_spine_force_xz_n=tuple(
                float(value) for value in wall_force
            ),
            spine_on_plate_wrench_about_holder=(
                float(plate_force[0]),
                0.0,
                float(plate_force[1]),
                float(moment[0]),
                float(moment[1]),
                float(moment[2]),
            ),
            normal_force_n=float(normal_force),
            tangential_force_n=float(tangential_force),
            normal_impulse_n_s=float(normal_impulse),
            tangential_impulse_n_s=float(tangential_impulse),
            impact_velocity_m_s=float(impact_velocity),
            axial_displacement_m=float(q[1]),
            transverse_displacement_m=float(q[2]),
            axial_velocity_m_s=float(v[1]),
            transverse_velocity_m_s=float(v[2]),
            axial_force_n=float(axial_force),
            transverse_force_n=float(transverse_force),
            spring_compression_m=float(compression),
            spring_travel_margin_m=float(
                self.parameters.spring_travel_m - compression
            ),
            kinetic_energy_j=float(kinetic_energy),
            structural_energy_j=float(structural_energy),
            preload_work_increment_j=float(preload_work),
            drive_work_increment_j=float(drive_work),
            friction_dissipation_increment_j=float(friction_dissipation),
            damping_dissipation_increment_j=float(damping_dissipation),
            energy_residual_j=float(energy_residual),
            dynamic_residual_n=float(dynamic_residual),
            actual_time_step_s=float(dt),
            nonlinear_iterations=int(
                nonlinear_iterations + projection_iterations
            ),
            bending_stress_pa=float(bending_stress),
            euler_buckling_margin_n=float(buckling_margin),
            numerical_state=NumericalState.CONVERGED,
            model_state=model_state,
        )
        return _StepData(new_state, point, active)

    def _initial_equilibrium(self) -> tuple[DynamicState, float]:
        initial_geometry = self.unit.geometry.query(
            self.experiment.initial_center_x_m
        )
        holder_x = (
            self.experiment.initial_center_x_m
            - self.parameters.exposed_length_m * self.unit.a[0]
        )
        holder_z = (
            initial_geometry.envelope_height_m
            - self.parameters.exposed_length_m * self.unit.a[1]
        )
        if self.experiment.constant_preload_n == 0.0:
            return (
                DynamicState(
                    time_s=0.0,
                    generalized_position=(float(holder_z), 0.0, 0.0),
                    generalized_velocity=(0.0, 0.0, 0.0),
                    contact_state=ContactState.FREE,
                    spring_state=SpringState.LOWER_STOP,
                    has_contacted=False,
                    accepted_steps=0,
                ),
                float(holder_x),
            )
        core = LegacyPrescribedPoseConstitutiveCore(
            self.parameters,
            self.track,
            SolverSettings(),
        )
        first = core.solve_pose(
            (float(holder_x), float(holder_z)),
            commit=True,
        )
        if not first.proposal_valid:
            raise RuntimeError(
                "initial_preload_infeasible: no initial contact state"
            )
        initial_state = first.next_state
        target = self.experiment.constant_preload_n
        approaches = np.geomspace(
            1e-10,
            self.experiment.maximum_preload_approach_m,
            96,
        )
        lower = 0.0
        upper: float | None = None
        for approach in approaches:
            candidate = core.solve_pose(
                (float(holder_x), float(holder_z - approach)),
                initial_state,
                commit=False,
            )
            if (
                candidate.proposal_valid
                and candidate.wall_on_spine_force_xz_n[1] >= target
            ):
                upper = float(approach)
                break
            lower = float(approach)
        if upper is None:
            raise RuntimeError(
                "initial_preload_infeasible: external preload equilibrium "
                "not bracketed"
            )
        best = None
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            candidate = core.solve_pose(
                (float(holder_x), float(holder_z - middle)),
                initial_state,
                commit=False,
            )
            if not candidate.proposal_valid:
                lower = middle
                continue
            best = candidate
            vertical_force = candidate.wall_on_spine_force_xz_n[1]
            if abs(vertical_force - target) <= 1e-7:
                break
            if vertical_force < target:
                lower = middle
            else:
                upper = middle
        if best is None:
            raise RuntimeError(
                "initial_preload_infeasible: external preload root failed"
            )
        committed = core.solve_pose(
            best.holder_xz_m,
            initial_state,
            commit=True,
        )
        if (
            not committed.proposal_valid
            or abs(committed.wall_on_spine_force_xz_n[1] - target)
            > 1e-5
        ):
            raise RuntimeError(
                "initial_preload_infeasible: external preload root rejected"
            )
        axial_displacement = (
            committed.spring_compression_m
            + self.parameters.axial_compliance_m_n
            * committed.axial_force_n
        )
        transverse_displacement = (
            self.parameters.transverse_compliance_m_n
            * committed.transverse_force_n
        )
        return (
            DynamicState(
                time_s=0.0,
                generalized_position=(
                    float(committed.holder_xz_m[1]),
                    float(axial_displacement),
                    float(transverse_displacement),
                ),
                generalized_velocity=(0.0, 0.0, 0.0),
                contact_state=ContactState.STICK,
                spring_state=committed.spring_state,
                has_contacted=True,
                accepted_steps=0,
            ),
            float(committed.holder_xz_m[0]),
        )

    @staticmethod
    def _with_path_position(
        point: DynamicPathPoint, path_position_m: float
    ) -> DynamicPathPoint:
        values = asdict(point)
        values["path_position_m"] = float(path_position_m)
        for name, enum_type in (
            ("contact_state", ContactState),
            ("spring_state", SpringState),
            ("event_label", EventLabel),
            ("numerical_state", NumericalState),
            ("model_state", ModelState),
        ):
            values[name] = enum_type(values[name])
        return DynamicPathPoint(**values)

    def run(self) -> DynamicSingleSpineResult:
        try:
            state, holder_x0 = self._initial_equilibrium()
            settled = self._step(
                state,
                holder_x_m=holder_x0,
                drag_speed_m_s=0.0,
                dt=self.integrator.time_step_s,
            )
        except (ContactGeometryError, RuntimeError) as exc:
            return self._failed_result(str(exc))

        # Reset physical time and cumulative work after initial equilibrium.
        state = DynamicState(
            time_s=0.0,
            generalized_position=settled.state.generalized_position,
            generalized_velocity=(0.0, 0.0, 0.0),
            contact_state=(
                ContactState.STICK
                if settled.active
                else ContactState.FREE
            ),
            spring_state=settled.state.spring_state,
            has_contacted=settled.active,
            accepted_steps=0,
        )
        initial_point = self._with_path_position(settled.point, 0.0)
        initial_values = asdict(initial_point)
        initial_values["time_s"] = 0.0
        initial_values["holder_velocity_xz_m_s"] = (0.0, 0.0)
        initial_values["holder_acceleration_xz_m_s2"] = (0.0, 0.0)
        initial_values["center_velocity_xz_m_s"] = (0.0, 0.0)
        initial_values["center_acceleration_xz_m_s2"] = (0.0, 0.0)
        initial_values["event_label"] = (
            EventLabel.FIRST_CONTACT if settled.active else EventLabel.NONE
        )
        initial_values["event_labels"] = (
            (EventLabel.FIRST_CONTACT.value,) if settled.active else ()
        )
        initial_values["contact_state"] = (
            ContactState.FIRST_CONTACT_EVENT
            if settled.active
            else ContactState.FREE
        )
        initial_values["numerical_state"] = NumericalState.CONVERGED
        initial_values["model_state"] = self._model_state()
        initial_point = DynamicPathPoint(**initial_values)

        points: list[DynamicPathPoint] = [initial_point]
        dt_nominal = self.integrator.time_step_s
        total_time = (
            self.experiment.drag_length_m
            / self.experiment.drag_speed_m_s
        )
        output_time = (
            self.experiment.output_spacing_m
            / self.experiment.drag_speed_m_s
        )
        next_output = output_time
        steps = int(math.ceil(total_time / dt_nominal))
        if steps > self.integrator.maximum_steps:
            return self._failed_result(
                "numerical_failure: maximum_steps would be exceeded"
            )
        termination_reason = "path_end"
        terminal = PathTerminalState.PATH_END
        for _ in range(steps):
            dt = min(dt_nominal, total_time - state.time_s)
            if dt <= 1e-15:
                break
            next_time = state.time_s + dt
            holder_x = (
                holder_x0
                + self.experiment.drag_speed_m_s * next_time
            )
            try:
                step = self._step(
                    state,
                    holder_x_m=holder_x,
                    drag_speed_m_s=self.experiment.drag_speed_m_s,
                    dt=dt,
                )
            except ContactGeometryError as exc:
                termination_reason = str(exc)
                terminal = PathTerminalState.TERRAIN_BOUNDS
                break
            except RuntimeError as exc:
                termination_reason = str(exc)
                terminal = PathTerminalState.STRUCTURAL_BOUNDARY
                break
            state = step.state
            path_position = (
                self.experiment.drag_speed_m_s * state.time_s
            )
            has_event = bool(step.point.event_labels)
            if (
                state.time_s + 1e-12 >= next_output
                or has_event
                or state.time_s + 1e-12 >= total_time
            ):
                points.append(
                    self._with_path_position(
                        step.point,
                        path_position,
                    )
                )
                while next_output <= state.time_s + 1e-12:
                    next_output += output_time

        summary = self._summarize(
            points,
            terminal=terminal,
            reason=termination_reason,
            accepted_steps=state.accepted_steps,
        )
        return DynamicSingleSpineResult(
            parameters=self.parameters,
            experiment=self.experiment,
            contact=self.contact,
            integrator=self.integrator,
            track_id=self.track.track_id,
            points=tuple(points),
            summary=summary,
            assumptions=(
                M2_MODEL_LEVEL,
                "continuous_external_preload_not_constant_contact_reaction",
                "prescribed_horizontal_speed",
                "rigid_moreau_contact_impulses",
                "first_axial_and_transverse_dynamic_modes",
                "no_penetration_damage_or_wear",
                "initial_static_equilibrium_under_external_preload",
            ),
        )

    def _failed_result(self, reason: str) -> DynamicSingleSpineResult:
        terminal = (
            PathTerminalState.INITIAL_PRELOAD_INFEASIBLE
            if "initial_preload_infeasible" in reason
            else PathTerminalState.NUMERICAL_FAILURE
        )
        summary = DynamicPathSummary(
            preload_mode="continuous_external_force",
            constant_preload_n=self.experiment.constant_preload_n,
            drag_speed_m_s=self.experiment.drag_speed_m_s,
            initial_preload_success=False,
            ever_contacted=False,
            ever_loaded=False,
            contact_fraction=0.0,
            effective_load_fraction=0.0,
            global_pull_force_peak_n=0.0,
            global_pull_force_steady_peak_n=0.0,
            global_pull_force_median_n=0.0,
            global_pull_force_p10_n=0.0,
            global_pull_force_p25_n=0.0,
            normal_force_range_n=(0.0, 0.0),
            holder_z_range_m=(math.nan, math.nan),
            holder_speed_peak_m_s=0.0,
            holder_acceleration_peak_m_s2=0.0,
            impact_velocity_peak_m_s=0.0,
            event_counts={name.value: 0 for name in EventLabel},
            maximum_bending_stress_pa=0.0,
            minimum_euler_buckling_margin_n=math.nan,
            maximum_abs_dynamic_residual_n=math.inf,
            maximum_abs_energy_residual_j=math.inf,
            internal_time_step_s=self.integrator.time_step_s,
            accepted_steps=0,
            rejected_steps=0,
            time_step_convergence_checked=False,
            contact_parameter_convergence_checked=False,
            numerical_state=NumericalState.NONCONVERGED,
            model_state=self._model_state(),
            run_terminal_state=terminal,
            termination_reason=reason,
            formal_ranking_eligible=False,
        )
        return DynamicSingleSpineResult(
            parameters=self.parameters,
            experiment=self.experiment,
            contact=self.contact,
            integrator=self.integrator,
            track_id=self.track.track_id,
            points=(),
            summary=summary,
            assumptions=(M2_MODEL_LEVEL,),
        )

    def _summarize(
        self,
        points: list[DynamicPathPoint],
        *,
        terminal: PathTerminalState,
        reason: str,
        accepted_steps: int,
    ) -> DynamicPathSummary:
        normal = np.asarray(
            [point.normal_force_n for point in points],
            dtype=np.float64,
        )
        pull = np.abs(
            np.asarray(
                [
                    point.spine_on_plate_wrench_about_holder[0]
                    for point in points
                ],
                dtype=np.float64,
            )
        )
        active = normal > 0.0
        effective = normal >= self.experiment.effective_normal_force_min_n
        impact = np.asarray(
            [point.impact_velocity_m_s > 0.0 for point in points],
            dtype=np.bool_,
        )
        steady_pull = pull[~impact]
        event_counts = {name.value: 0 for name in EventLabel}
        for point in points:
            for label in point.event_labels:
                event_counts[label] = event_counts.get(label, 0) + 1
        holder_z = np.asarray(
            [point.holder_xz_m[1] for point in points],
            dtype=np.float64,
        )
        holder_speed = np.abs(
            np.asarray(
                [point.holder_velocity_xz_m_s[1] for point in points],
                dtype=np.float64,
            )
        )
        holder_acceleration = np.abs(
            np.asarray(
                [
                    point.holder_acceleration_xz_m_s2[1]
                    for point in points
                ],
                dtype=np.float64,
            )
        )
        model_state = self._model_state()
        numerical_state = (
            NumericalState.CONVERGED
            if terminal is PathTerminalState.PATH_END
            else NumericalState.NONCONVERGED
        )
        formal_eligible = (
            terminal is PathTerminalState.PATH_END
            and numerical_state is NumericalState.CONVERGED
            and model_state is ModelState.COVERED
            and False  # convergence pair is a separate required production run
        )
        return DynamicPathSummary(
            preload_mode="continuous_external_force",
            constant_preload_n=self.experiment.constant_preload_n,
            drag_speed_m_s=self.experiment.drag_speed_m_s,
            initial_preload_success=True,
            ever_contacted=bool(np.any(active)),
            ever_loaded=bool(np.any(effective)),
            contact_fraction=float(np.mean(active)),
            effective_load_fraction=float(np.mean(effective)),
            global_pull_force_peak_n=float(np.max(pull)),
            global_pull_force_steady_peak_n=(
                float(np.max(steady_pull)) if steady_pull.size else 0.0
            ),
            global_pull_force_median_n=float(np.median(pull)),
            global_pull_force_p10_n=float(np.quantile(pull, 0.10)),
            global_pull_force_p25_n=float(np.quantile(pull, 0.25)),
            normal_force_range_n=(
                float(np.min(normal)),
                float(np.max(normal)),
            ),
            holder_z_range_m=(
                float(np.min(holder_z)),
                float(np.max(holder_z)),
            ),
            holder_speed_peak_m_s=float(np.max(holder_speed)),
            holder_acceleration_peak_m_s2=float(
                np.max(holder_acceleration)
            ),
            impact_velocity_peak_m_s=float(
                max(point.impact_velocity_m_s for point in points)
            ),
            event_counts=event_counts,
            maximum_bending_stress_pa=float(
                max(point.bending_stress_pa for point in points)
            ),
            minimum_euler_buckling_margin_n=float(
                min(point.euler_buckling_margin_n for point in points)
            ),
            maximum_abs_dynamic_residual_n=float(
                max(abs(point.dynamic_residual_n) for point in points)
            ),
            maximum_abs_energy_residual_j=float(
                max(abs(point.energy_residual_j) for point in points)
            ),
            internal_time_step_s=self.integrator.time_step_s,
            accepted_steps=accepted_steps,
            rejected_steps=0,
            time_step_convergence_checked=False,
            contact_parameter_convergence_checked=False,
            numerical_state=numerical_state,
            model_state=model_state,
            run_terminal_state=terminal,
            termination_reason=reason,
            formal_ranking_eligible=formal_eligible,
        )
