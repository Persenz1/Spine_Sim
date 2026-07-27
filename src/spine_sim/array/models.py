"""Versioned immutable data models for M3 common-backplate arrays."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping

import numpy as np

from spine_sim.contact import (
    ConstitutiveResponse,
    PathTerminalState,
    SingleSpineState,
    SpineParameters,
)
from spine_sim.core.identity import identity
from spine_sim.core.states import ModelState, NumericalState


M3_MODULE_VERSION = "m3.0.0"
M3_MODEL_LEVEL = "project_model_P_common_rigid_backplate_quasistatic_v1"


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
class ArrayState:
    pin_states: tuple[SingleSpineState, ...]
    accepted_steps: int = 0

    @classmethod
    def empty(cls, pin_count: int) -> "ArrayState":
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
        "normal=M2 nonnegative envelope-normal force; "
        "target_tangential=absolute pin Fx in unit-local +x; "
        "resultant=Euclidean pin force; zeros included over all nominal pins"
    )


@dataclass(frozen=True)
class ArrayResidualAudit:
    force_aggregation_n: float
    moment_aggregation_nm: float
    maximum_local_geometry_m: float
    maximum_local_structure_m: float
    maximum_local_force_decomposition_n: float
    termination_reason: str = "converged"


@dataclass(frozen=True)
class ArrayPoseResponse:
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
    residual: ArrayResidualAudit
    numerical_state: NumericalState
    model_state: ModelState
    event_labels: tuple[tuple[int, str], ...]
    proposal_state: ArrayState
    next_state: ArrayState
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
class ArrayPathPoint:
    path_position_m: float
    response: ArrayPoseResponse
    event_refined: bool = False


@dataclass(frozen=True)
class ArrayPathSummary:
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
class ArrayExperimentResult:
    configuration: ArrayConfiguration
    terrain_recipe_id: str
    region_id: str
    track_ids: tuple[str, ...]
    fixed_common_uz_m: float | None
    target_preload_n: float
    points: tuple[ArrayPathPoint, ...]
    summary: ArrayPathSummary
    assumptions: tuple[str, ...] = field(default_factory=tuple)
