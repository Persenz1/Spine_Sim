"""Versioned data models for M2 single-spine contact."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from spine_sim.core.states import ModelState, NumericalState

from .errors import ContactConfigurationError


M2_MODULE_VERSION = "m2.1.0"
M2_MODEL_LEVEL = "project_model_P_main_plane_dynamic_constant_preload_v2"
M2_LEGACY_MODEL_LEVEL = "legacy_fixed_pose_quasistatic_v1"


class AxialMode(StrEnum):
    RIGID = "rigid"
    SPRING = "spring"


class ContactState(StrEnum):
    FREE = "free"
    FIRST_CONTACT_EVENT = "first_contact_event"
    STICK = "stick"
    SLIDE = "slide"
    DETACH_EVENT = "detach_event"
    RECONTACT_EVENT = "recontact_event"
    IMPACT_EVENT = "impact_event"


class SpringState(StrEnum):
    LOWER_STOP = "lower_stop"
    INTERIOR = "interior"
    HARD_STOP = "hard_stop"


class PathTerminalState(StrEnum):
    PATH_END = "path_end"
    TERRAIN_BOUNDS = "terrain_bounds"
    INITIAL_PRELOAD_INFEASIBLE = "initial_preload_infeasible"
    STRUCTURAL_BOUNDARY = "structural_boundary"
    NUMERICAL_FAILURE = "numerical_failure"
    MODEL_UNCLOSED = "model_unclosed"


class EventLabel(StrEnum):
    NONE = "none"
    FIRST_CONTACT = "first_contact"
    DETACH_TO_FREE = "detach_to_free"
    RECONTACT = "recontact"
    SLIP_START = "slip_start"
    HARD_STOP = "hard_stop"
    IMPACT = "impact"
    STICK_RECOVERED = "stick_recovered"


@dataclass(frozen=True)
class SpineParameters:
    """Single-spine hardware and friction parameters, all in SI."""

    tip_radius_m: float = 50e-6
    diameter_m: float = 0.8e-3
    exposed_length_m: float = 4e-3
    installation_angle_deg: float = 70.0
    axial_mode: AxialMode = AxialMode.SPRING
    spring_stiffness_n_m: float | None = 800.0
    spring_travel_m: float = 4e-3
    young_modulus_pa: float = 200e9
    poisson_ratio: float = 0.29
    shear_correction: float = 6.0 / 7.0
    static_friction: float = 0.45
    kinetic_friction: float = 0.35
    beam_enabled: bool = True
    material_assumption: str = "unfrozen_high_carbon_steel_proxy_E200GPa_nu0.29"
    rod_clearance_mode: str = "unclosed"
    density_kg_m3: float = 7850.0
    axial_modal_mass_factor: float = 1.0 / 3.0
    transverse_modal_mass_factor: float = 0.236
    axial_damping_ratio: float = 0.05
    transverse_damping_ratio: float = 0.05
    yield_strength_pa: float | None = None

    def __post_init__(self) -> None:
        try:
            axial_mode = AxialMode(self.axial_mode)
        except ValueError as exc:
            raise ContactConfigurationError("axial_mode must be rigid or spring") from exc
        object.__setattr__(self, "axial_mode", axial_mode)
        positive = (
            "tip_radius_m",
            "diameter_m",
            "exposed_length_m",
            "spring_travel_m",
            "young_modulus_pa",
            "shear_correction",
            "density_kg_m3",
            "axial_modal_mass_factor",
            "transverse_modal_mass_factor",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ContactConfigurationError(f"{name} must be finite and positive")
        if not 0.0 < self.installation_angle_deg < 90.0:
            raise ContactConfigurationError(
                "installation_angle_deg must lie in (0, 90)"
            )
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ContactConfigurationError("poisson_ratio must lie in (-1, 0.5)")
        if (
            not math.isfinite(self.static_friction)
            or not math.isfinite(self.kinetic_friction)
            or self.static_friction < 0
            or self.kinetic_friction < 0
            or self.kinetic_friction > self.static_friction
        ):
            raise ContactConfigurationError(
                "friction coefficients require 0 <= kinetic <= static"
            )
        if axial_mode is AxialMode.SPRING:
            if (
                self.spring_stiffness_n_m is None
                or not math.isfinite(self.spring_stiffness_n_m)
                or self.spring_stiffness_n_m <= 0
            ):
                raise ContactConfigurationError(
                    "spring mode requires positive spring_stiffness_n_m"
                )
        elif self.spring_stiffness_n_m is not None:
            raise ContactConfigurationError(
                "rigid axial mode requires spring_stiffness_n_m=None"
            )
        if self.rod_clearance_mode not in {"unclosed", "disabled_analytic_fixture"}:
            raise ContactConfigurationError(
                "rod_clearance_mode must be unclosed or disabled_analytic_fixture"
            )
        for name in ("axial_damping_ratio", "transverse_damping_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ContactConfigurationError(
                    f"{name} must be finite and non-negative"
                )
        if self.yield_strength_pa is not None and (
            not math.isfinite(self.yield_strength_pa)
            or self.yield_strength_pa <= 0.0
        ):
            raise ContactConfigurationError(
                "yield_strength_pa must be null or finite and positive"
            )

    @property
    def angle_rad(self) -> float:
        return math.radians(self.installation_angle_deg)

    @property
    def axis_xz(self) -> NDArray[np.float64]:
        return np.array(
            [math.cos(self.angle_rad), -math.sin(self.angle_rad)],
            dtype=np.float64,
        )

    @property
    def transverse_xz(self) -> NDArray[np.float64]:
        return np.array(
            [math.sin(self.angle_rad), math.cos(self.angle_rad)],
            dtype=np.float64,
        )

    @property
    def area_m2(self) -> float:
        return math.pi * self.diameter_m**2 / 4.0

    @property
    def second_moment_m4(self) -> float:
        return math.pi * self.diameter_m**4 / 64.0

    @property
    def shear_modulus_pa(self) -> float:
        return self.young_modulus_pa / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def axial_compliance_m_n(self) -> float:
        if not self.beam_enabled:
            return 0.0
        return self.exposed_length_m / (self.young_modulus_pa * self.area_m2)

    @property
    def transverse_compliance_m_n(self) -> float:
        if not self.beam_enabled:
            return 0.0
        bending = self.exposed_length_m**3 / (
            3.0 * self.young_modulus_pa * self.second_moment_m4
        )
        shear = self.exposed_length_m / (
            self.shear_correction * self.shear_modulus_pa * self.area_m2
        )
        return bending + shear

    @property
    def spine_mass_kg(self) -> float:
        return self.density_kg_m3 * self.area_m2 * self.exposed_length_m

    @property
    def axial_modal_mass_kg(self) -> float:
        return self.axial_modal_mass_factor * self.spine_mass_kg

    @property
    def transverse_modal_mass_kg(self) -> float:
        return self.transverse_modal_mass_factor * self.spine_mass_kg

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["axial_mode"] = self.axial_mode.value
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SpineParameters":
        allowed = set(cls.__dataclass_fields__)
        extra = set(value) - allowed
        if extra:
            raise ContactConfigurationError(
                f"SpineParameters contains unknown fields: {sorted(extra)}"
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class SolverSettings:
    gap_tolerance_m: float = 1e-10
    force_tolerance_n: float = 1e-8
    friction_residual_tolerance_n: float = 5e-4
    residual_tolerance_m: float = 2e-9
    cap_tolerance_m: float = 2e-9
    event_tolerance_m: float = 1e-8
    max_contact_force_n: float = 100.0
    root_max_iterations: int = 80
    root_scan_points: int = 96
    event_max_iterations: int = 50
    stick_motion_tolerance_m: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "gap_tolerance_m",
            "force_tolerance_n",
            "friction_residual_tolerance_n",
            "residual_tolerance_m",
            "cap_tolerance_m",
            "event_tolerance_m",
            "max_contact_force_n",
            "stick_motion_tolerance_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ContactConfigurationError(f"{name} must be finite and positive")
        if self.root_max_iterations < 1 or self.root_scan_points < 3:
            raise ContactConfigurationError("root iteration limits are too small")
        if self.event_max_iterations < 1:
            raise ContactConfigurationError("event_max_iterations must be positive")


@dataclass(frozen=True)
class GeometrySample:
    center_x_m: float
    envelope_height_m: float
    envelope_slope_x: float
    support_xyz_m: tuple[float, float, float]
    tangent_xz: tuple[float, float]
    normal_xz: tuple[float, float]
    valid: bool
    near_tie: bool
    cap_margin_m: float


@dataclass(frozen=True)
class ResidualAudit:
    geometry_m: float = 0.0
    structure_m: float = 0.0
    force_decomposition_n: float = 0.0
    energy_j: float = 0.0
    unilateral_margin_n: float = 0.0
    friction_margin_n: float = 0.0
    spring_lower_margin_m: float = 0.0
    spring_travel_margin_m: float = 0.0
    cap_margin_m: float = 0.0
    root_iterations: int = 0
    root_bracket_n: tuple[float, float] | None = None
    termination_reason: str = "converged"


@dataclass(frozen=True)
class SingleSpineState:
    contact_state: ContactState = ContactState.FREE
    spring_state: SpringState = SpringState.LOWER_STOP
    has_contacted: bool = False
    anchor_center_xz_m: tuple[float, float] | None = None
    last_holder_xz_m: tuple[float, float] | None = None
    last_center_xz_m: tuple[float, float] | None = None
    last_wall_force_xz_n: tuple[float, float] = (0.0, 0.0)
    last_elastic_energy_j: float = 0.0
    cumulative_friction_dissipation_j: float = 0.0
    slide_direction: int = 0
    accepted_steps: int = 0

    def with_contact_state(self, state: ContactState) -> "SingleSpineState":
        return replace(self, contact_state=state)


@dataclass(frozen=True)
class ConstitutiveResponse:
    holder_xz_m: tuple[float, float]
    center_xz_m: tuple[float, float]
    support_xyz_m: tuple[float, float, float] | None
    gap_m: float
    tangent_xz: tuple[float, float] | None
    normal_xz: tuple[float, float] | None
    cap_gate_passed: bool
    near_tie: bool
    contact_state: ContactState
    spring_state: SpringState
    event_label: EventLabel
    wall_on_spine_force_xz_n: tuple[float, float]
    spine_on_plate_wrench_about_holder: tuple[
        float, float, float, float, float, float
    ]
    normal_force_n: float
    tangential_force_n: float
    axial_force_n: float
    transverse_force_n: float
    spring_compression_m: float
    beam_displacement_xz_m: tuple[float, float]
    static_friction_margin_n: float
    spring_travel_margin_m: float
    elastic_energy_j: float
    holder_work_increment_j: float
    contact_work_increment_j: float
    friction_dissipation_increment_j: float
    energy_residual_j: float
    numerical_state: NumericalState
    model_state: ModelState
    model_warnings: tuple[str, ...]
    residual: ResidualAudit
    proposal_state: SingleSpineState
    next_state: SingleSpineState
    proposal_valid: bool


@dataclass(frozen=True)
class PathPoint:
    path_position_m: float
    response: ConstitutiveResponse
    event_refined: bool = False


@dataclass(frozen=True)
class PathSummary:
    initial_preload_success: bool
    ever_contacted: bool
    ever_loaded: bool
    total_contact_length_m: float
    effective_load_length_m: float
    effective_load_fraction: float
    maximum_continuous_load_length_m: float
    tangential_force_peak_n: float
    tangential_force_median_n: float
    tangential_force_p10_n: float
    tangential_force_p25_n: float
    normal_force_range_n: tuple[float, float]
    event_counts: Mapping[str, int]
    maximum_abs_geometry_residual_m: float
    maximum_abs_energy_residual_j: float
    physical_terminal_state: str
    numerical_state: NumericalState
    model_state: ModelState
    run_terminal_state: PathTerminalState
    termination_reason: str


@dataclass(frozen=True)
class SingleSpineExperimentResult:
    parameters: SpineParameters
    track_id: str
    fixed_holder_z_m: float | None
    points: tuple[PathPoint, ...]
    summary: PathSummary
    assumptions: tuple[str, ...] = field(default_factory=tuple)
