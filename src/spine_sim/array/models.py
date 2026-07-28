"""Versioned immutable data models for M3 common-backplate arrays."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping

import numpy as np

from spine_sim.contact import (
    ConstitutiveResponse,
    ContactState,
    EventLabel,
    PathTerminalState,
    SingleSpineState,
    SpineParameters,
    SpringState,
)
from spine_sim.core.identity import identity
from spine_sim.core.states import ModelState, NumericalState


M3_MODULE_VERSION = "m3.1.0"
M3_MODEL_LEVEL = (
    "project_model_P_common_rigid_backplate_z_dynamic_"
    "continuous_total_preload_v2"
)
M3_LEGACY_MODULE_VERSION = "m3.0.0"
M3_LEGACY_MODEL_LEVEL = "project_model_P_common_rigid_backplate_quasistatic_v1"


class AngleLayout(StrEnum):
    FIXED = "fixed"
    GRADIENT_80_TO_60 = "gradient_80_to_60"
    GRADIENT_80_TO_50 = "gradient_80_to_50"


@dataclass(frozen=True)
class ArrayConfiguration:
    """Hardware geometry; the unit reference point is the grid centre."""

    nx: int
    ny: int
    spacing_m: float
    base_spine: SpineParameters
    angle_layout: AngleLayout = AngleLayout.FIXED
    fixture_only: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.nx, bool) or isinstance(self.ny, bool):
            raise ValueError("nx and ny must be integer counts")
        allowed = {1, 2, 3, 4, 5, 6} if self.fixture_only else {2, 3, 4, 5, 6}
        if self.nx not in allowed or self.ny not in allowed:
            raise ValueError(
                "production nx and ny must be 2..6; singleton dimensions are fixture-only"
            )
        if self.pin_count < 2:
            raise ValueError("an array fixture still requires at least two pins")
        if not any(
            math.isclose(self.spacing_m, value, rel_tol=0.0, abs_tol=1e-12)
            for value in (4e-3, 5e-3, 6e-3)
        ):
            raise ValueError("spacing_m must be 4, 5 or 6 mm")
        object.__setattr__(self, "angle_layout", AngleLayout(self.angle_layout))
        if not isinstance(self.base_spine, SpineParameters):
            raise ValueError("base_spine must be a SpineParameters instance")

    @property
    def pin_count(self) -> int:
        return self.nx * self.ny

    @property
    def configuration_id(self) -> str:
        return identity(
            "configuration",
            self.as_dict(),
            module_version=M3_MODULE_VERSION,
        )

    @property
    def column_angles_deg(self) -> tuple[float, ...]:
        if self.angle_layout is AngleLayout.FIXED:
            return (self.base_spine.installation_angle_deg,) * self.nx
        end = (
            60.0
            if self.angle_layout is AngleLayout.GRADIENT_80_TO_60
            else 50.0
        )
        return tuple(float(value) for value in np.linspace(80.0, end, self.nx))

    @property
    def pin_parameters(self) -> tuple[SpineParameters, ...]:
        output: list[SpineParameters] = []
        reference_height = 4e-3 * math.sin(math.radians(80.0))
        for _row in range(self.ny):
            for angle_deg in self.column_angles_deg:
                length = (
                    self.base_spine.exposed_length_m
                    if self.angle_layout is AngleLayout.FIXED
                    else reference_height / math.sin(math.radians(angle_deg))
                )
                output.append(
                    replace(
                        self.base_spine,
                        installation_angle_deg=angle_deg,
                        exposed_length_m=length,
                    )
                )
        return tuple(output)

    @property
    def holder_offsets_xyz_m(
        self,
    ) -> tuple[tuple[float, float, float], ...]:
        offsets: list[tuple[float, float, float]] = []
        for row in range(self.ny):
            y = (row - 0.5 * (self.ny - 1)) * self.spacing_m
            for column in range(self.nx):
                x = (column - 0.5 * (self.nx - 1)) * self.spacing_m
                offsets.append((float(x), float(y), 0.0))
        return tuple(offsets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "spacing_m": self.spacing_m,
            "base_spine": self.base_spine.as_dict(),
            "angle_layout": self.angle_layout.value,
            "fixture_only": self.fixture_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArrayConfiguration":
        allowed = {
            "nx",
            "ny",
            "spacing_m",
            "base_spine",
            "angle_layout",
            "fixture_only",
        }
        extra = set(value) - allowed
        if extra:
            raise ValueError(
                f"ArrayConfiguration contains unknown fields: {sorted(extra)}"
            )
        return cls(
            nx=int(value["nx"]),
            ny=int(value["ny"]),
            spacing_m=float(value["spacing_m"]),
            base_spine=SpineParameters.from_mapping(value["base_spine"]),
            angle_layout=AngleLayout(value.get("angle_layout", "fixed")),
            fixture_only=bool(value.get("fixture_only", False)),
        )


@dataclass(frozen=True)
class LegacyArrayState:
    pin_states: tuple[SingleSpineState, ...]
    accepted_steps: int = 0

    @classmethod
    def empty(cls, pin_count: int) -> "LegacyArrayState":
        if pin_count < 1:
            raise ValueError("pin_count must be positive")
        return cls(tuple(SingleSpineState() for _ in range(pin_count)))


@dataclass(frozen=True)
class ActivitySets:
    nominal: tuple[int, ...]
    geometric: tuple[int, ...]
    positive_normal: tuple[int, ...]
    admissible: tuple[int, ...]
    target_load: tuple[int, ...]


@dataclass(frozen=True)
class LoadSharingMetrics:
    neff_normal: float
    neff_target_tangential: float
    neff_resultant: float
    max_mean_normal: float
    max_mean_target_tangential: float
    max_mean_resultant: float
    gini_normal: float
    gini_target_tangential: float
    gini_resultant: float
    weight_definition: str = (
        "normal=nonnegative pin contact-normal force; "
        "target_tangential=absolute pin Fx in unit-local +x; "
        "resultant=Euclidean pin force; zeros included over all nominal pins"
    )


@dataclass(frozen=True)
class LegacyArrayResidualAudit:
    force_aggregation_n: float
    moment_aggregation_nm: float
    maximum_local_geometry_m: float
    maximum_local_structure_m: float
    maximum_local_force_decomposition_n: float
    termination_reason: str = "converged"


@dataclass(frozen=True)
class LegacyArrayPoseResponse:
    common_ux_m: float
    common_uz_m: float
    unit_origin_xyz_m: tuple[float, float, float]
    pin_holder_xyz_m: tuple[tuple[float, float, float], ...]
    pin_responses: tuple[ConstitutiveResponse, ...]
    pin_wrench_about_unit: tuple[
        tuple[float, float, float, float, float, float], ...
    ]
    wall_on_unit_wrench_about_origin: tuple[
        float, float, float, float, float, float
    ]
    active_thrust_wrench_about_origin: tuple[
        float, float, float, float, float, float
    ]
    guide_reaction_wrench_about_origin: tuple[
        float, float, float, float, float, float
    ]
    activity_sets: ActivitySets
    sharing: LoadSharingMetrics
    residual: LegacyArrayResidualAudit
    numerical_state: NumericalState
    model_state: ModelState
    event_labels: tuple[tuple[int, str], ...]
    proposal_state: LegacyArrayState
    next_state: LegacyArrayState
    proposal_valid: bool

    @property
    def total_normal_force_n(self) -> float:
        return float(sum(response.normal_force_n for response in self.pin_responses))

    @property
    def tangential_force_positive_n(self) -> float:
        return max(0.0, self.wall_on_unit_wrench_about_origin[0])

    @property
    def tangential_force_negative_n(self) -> float:
        return max(0.0, -self.wall_on_unit_wrench_about_origin[0])


@dataclass(frozen=True)
class LegacyArrayPathPoint:
    path_position_m: float
    response: LegacyArrayPoseResponse
    event_refined: bool = False


@dataclass(frozen=True)
class LegacyArrayPathSummary:
    initial_preload_success: bool
    total_contact_length_m: float
    effective_load_length_m: float
    effective_load_fraction: float
    maximum_continuous_load_length_m: float
    tangential_force_peak_n: float
    tangential_force_median_n: float
    tangential_force_p10_n: float
    tangential_force_p25_n: float
    total_normal_force_range_n: tuple[float, float]
    neff_normal_median: float
    neff_target_tangential_median: float
    neff_resultant_median: float
    maximum_pin_resultant_force_n: float
    maximum_resultant_load_concentration: float
    event_counts: Mapping[str, int]
    maximum_abs_local_geometry_residual_m: float
    maximum_force_aggregation_residual_n: float
    maximum_moment_aggregation_residual_nm: float
    numerical_state: NumericalState
    model_state: ModelState
    run_terminal_state: PathTerminalState
    termination_reason: str


@dataclass(frozen=True)
class LegacyArrayExperimentResult:
    configuration: ArrayConfiguration
    terrain_recipe_id: str
    region_id: str
    track_ids: tuple[str, ...]
    fixed_common_uz_m: float | None
    target_preload_n: float
    points: tuple[LegacyArrayPathPoint, ...]
    summary: LegacyArrayPathSummary
    assumptions: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ArrayDynamicState:
    """Complete immutable state for one jointly integrated array."""

    time_s: float
    backplate_position_z_m: float
    backplate_velocity_z_m_s: float
    pin_axial_displacement_m: tuple[float, ...]
    pin_transverse_displacement_m: tuple[float, ...]
    pin_axial_velocity_m_s: tuple[float, ...]
    pin_transverse_velocity_m_s: tuple[float, ...]
    pin_contact_state: tuple[ContactState, ...]
    pin_spring_state: tuple[SpringState, ...]
    pin_has_contacted: tuple[bool, ...]
    accepted_steps: int = 0
    rejected_steps: int = 0
    cumulative_preload_work_j: float = 0.0
    cumulative_drive_work_j: float = 0.0
    cumulative_friction_dissipation_j: float = 0.0
    cumulative_structural_damping_dissipation_j: float = 0.0
    cumulative_backplate_damping_dissipation_j: float = 0.0

    @classmethod
    def initial(cls, pin_count: int, backplate_position_z_m: float) -> "ArrayDynamicState":
        if pin_count < 1:
            raise ValueError("pin_count must be positive")
        zeros = (0.0,) * pin_count
        return cls(
            time_s=0.0,
            backplate_position_z_m=float(backplate_position_z_m),
            backplate_velocity_z_m_s=0.0,
            pin_axial_displacement_m=zeros,
            pin_transverse_displacement_m=zeros,
            pin_axial_velocity_m_s=zeros,
            pin_transverse_velocity_m_s=zeros,
            pin_contact_state=(ContactState.FREE,) * pin_count,
            pin_spring_state=(SpringState.LOWER_STOP,) * pin_count,
            pin_has_contacted=(False,) * pin_count,
        )


@dataclass(frozen=True)
class PinDynamicResponse:
    """One pin's response from a single accepted global array step."""

    holder_xyz_m: tuple[float, float, float]
    center_xyz_m: tuple[float, float, float]
    center_velocity_xyz_m_s: tuple[float, float, float]
    center_acceleration_xyz_m_s2: tuple[float, float, float]
    support_xyz_m: tuple[float, float, float] | None
    tangent_xz: tuple[float, float] | None
    normal_xz: tuple[float, float] | None
    gap_m: float
    contact_state: ContactState
    spring_state: SpringState
    event_label: EventLabel
    event_labels: tuple[str, ...]
    wall_on_spine_force_xz_n: tuple[float, float]
    spine_on_plate_wrench_about_holder: tuple[
        float, float, float, float, float, float
    ]
    spine_on_plate_wrench_about_unit: tuple[
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
    bending_stress_pa: float
    euler_buckling_margin_n: float


@dataclass(frozen=True)
class ArrayDynamicPathPoint:
    time_s: float
    path_position_m: float
    backplate_position_xyz_m: tuple[float, float, float]
    backplate_velocity_xyz_m_s: tuple[float, float, float]
    backplate_acceleration_xyz_m_s2: tuple[float, float, float]
    external_total_preload_n: float
    pin_responses: tuple[PinDynamicResponse, ...]
    wall_on_unit_wrench_about_origin: tuple[
        float, float, float, float, float, float
    ]
    active_thrust_wrench_about_origin: tuple[
        float, float, float, float, float, float
    ]
    guide_reaction_wrench_about_origin: tuple[
        float, float, float, float, float, float
    ]
    activity_sets: ActivitySets
    sharing: LoadSharingMetrics
    active_pin_count: int
    effective_load_pin_count: int
    total_contact_reaction_z_n: float
    backplate_inertia_force_z_n: float
    backplate_damping_force_z_n: float
    kinetic_energy_j: float
    structural_energy_j: float
    preload_work_increment_j: float
    drive_work_increment_j: float
    cumulative_preload_work_j: float
    cumulative_drive_work_j: float
    friction_dissipation_increment_j: float
    cumulative_friction_dissipation_j: float
    structural_damping_dissipation_increment_j: float
    cumulative_structural_damping_dissipation_j: float
    backplate_damping_dissipation_increment_j: float
    cumulative_backplate_damping_dissipation_j: float
    dynamic_residual_n: float
    energy_residual_j: float
    force_aggregation_residual_n: float
    moment_aggregation_residual_nm: float
    actual_time_step_s: float
    nonlinear_iterations: int
    numerical_state: NumericalState
    model_state: ModelState
    event_labels: tuple[tuple[int, str], ...]

    @property
    def total_normal_force_n(self) -> float:
        return float(sum(pin.normal_force_n for pin in self.pin_responses))

    @property
    def tangential_force_positive_n(self) -> float:
        return max(0.0, self.wall_on_unit_wrench_about_origin[0])

    @property
    def tangential_force_negative_n(self) -> float:
        return max(0.0, -self.wall_on_unit_wrench_about_origin[0])


@dataclass(frozen=True)
class ArrayDynamicStepProposal:
    """Pure global-step proposal; committing is an explicit separate operation."""

    source_state: ArrayDynamicState
    proposal_state: ArrayDynamicState
    point: ArrayDynamicPathPoint | None
    proposal_valid: bool
    rejection_reason: str


@dataclass(frozen=True)
class ArrayDynamicPathSummary:
    preload_mode: str
    external_total_preload_n: float
    drag_speed_m_s: float
    backplate_rotational_dofs: str
    initial_preload_success: bool
    total_contact_reaction_time_mean_n: float
    steady_normal_balance_error_n: float
    contact_fraction: float
    effective_load_fraction: float
    tangential_force_peak_n: float
    tangential_force_steady_peak_n: float
    tangential_force_impact_peak_n: float
    tangential_force_median_n: float
    tangential_force_p10_n: float
    tangential_force_p25_n: float
    total_normal_force_range_n: tuple[float, float]
    backplate_z_range_m: tuple[float, float]
    backplate_speed_peak_m_s: float
    backplate_acceleration_peak_m_s2: float
    impact_velocity_peak_m_s: float
    neff_normal_median: float
    neff_target_tangential_median: float
    neff_resultant_median: float
    maximum_normal_load_concentration: float
    maximum_gini_normal: float
    event_counts: Mapping[str, int]
    maximum_abs_dynamic_residual_n: float
    maximum_abs_energy_residual_j: float
    maximum_force_aggregation_residual_n: float
    maximum_moment_aggregation_residual_nm: float
    minimum_actual_time_step_s: float
    maximum_actual_time_step_s: float
    accepted_steps: int
    rejected_steps: int
    time_step_convergence_checked: bool
    contact_parameter_convergence_checked: bool
    unclosed_parameter_names: tuple[str, ...]
    numerical_state: NumericalState
    model_state: ModelState
    run_terminal_state: PathTerminalState
    termination_reason: str
    formal_ranking_eligible: bool


@dataclass(frozen=True)
class ArrayDynamicExperimentResult:
    configuration: ArrayConfiguration
    terrain_recipe_id: str
    region_id: str
    track_ids: tuple[str, ...]
    experiment: Any
    contact: Any
    integrator: Any
    points: tuple[ArrayDynamicPathPoint, ...]
    summary: ArrayDynamicPathSummary
    assumptions: tuple[str, ...] = field(default_factory=tuple)
