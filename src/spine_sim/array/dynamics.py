"""Joint time-domain dynamics for a rigid common backplate and all spines.

The only externally prescribed generalized motion is common horizontal
translation.  Backplate Z and every axial/transverse spine mode are integrated
in one vector.  Contact impulses are solved as one coupled array problem, so no
per-spine experiment is run and no instantaneous force-balancing root is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from spine_sim.contact import (
    ContactState,
    DynamicContactSettings,
    DynamicIntegratorSettings,
    DynamicSingleSpineUnit,
    EventLabel,
    PathTerminalState,
    SpringState,
)
from spine_sim.contact.errors import ContactConfigurationError, ContactGeometryError
from spine_sim.core.states import ModelState, NumericalState
from spine_sim.terrain import TrackGeometry

from .models import (
    M3_MODEL_LEVEL,
    ActivitySets,
    ArrayConfiguration,
    ArrayDynamicExperimentResult,
    ArrayDynamicPathPoint,
    ArrayDynamicPathSummary,
    ArrayDynamicState,
    ArrayDynamicStepProposal,
    LoadSharingMetrics,
    PinDynamicResponse,
    SettlementTracePoint,
)


_ACTIVE_CONTACT_STATES = {
    ContactState.FIRST_CONTACT_EVENT,
    ContactState.RECONTACT_EVENT,
    ContactState.IMPACT_EVENT,
    ContactState.STICK,
    ContactState.SLIDE,
}


def _neff(weights: NDArray[np.float64]) -> float:
    total = float(np.sum(weights))
    square_sum = float(weights @ weights)
    return total * total / square_sum if total > 0.0 and square_sum > 0.0 else 0.0


def _max_mean(weights: NDArray[np.float64]) -> float:
    mean = float(np.mean(weights))
    return float(np.max(weights)) / mean if mean > 0.0 else 0.0


def _gini(weights: NDArray[np.float64]) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    ordered = np.sort(weights)
    count = ordered.size
    coefficients = 2.0 * np.arange(1, count + 1) - count - 1.0
    return float(coefficients @ ordered / (count * total))


@dataclass(frozen=True)
class ArrayDynamicExperimentSettings:
    """Frozen loading protocol and common-backplate physical parameters."""

    drag_length_m: float
    external_total_preload_n: float = 1.0
    initial_common_ux_m: float = 0.0
    drag_speed_m_s: float = 1e-3
    backplate_mass_kg: float = 0.10
    backplate_vertical_damping_n_s_m: float = 2.0
    backplate_rotational_dofs: str = "locked"
    backplate_inertia_kg_m2: None = None
    maximum_preload_approach_m: float = 8e-3
    preload_ramp_time_s: float = 0.25
    preload_ramp_profile: str = "minimum_jerk_quintic"
    settlement_damping_scale: float = 10.0
    settling_reaction_force_tolerance_n: float = 0.01
    settling_reaction_force_relative_tolerance: float = 0.02
    settling_dynamic_residual_tolerance_n: float = 1e-8
    settling_stable_steps: int = 20
    dynamic_residual_tolerance_n: float = 1e-3
    coupled_projection_relaxation: float = 0.8
    output_spacing_m: float = 10e-6
    effective_pin_normal_force_min_n: float = 0.05
    unclosed_parameter_names: tuple[str, ...] = (
        "array_dynamic_parameters_not_project_calibrated",
    )
    time_step_convergence_checked: bool = False
    contact_parameter_convergence_checked: bool = False
    settlement_damping_convergence_checked: bool = False
    terrain_resolution_convergence_checked: bool = False
    physical_calibration_completed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "drag_length_m",
            "drag_speed_m_s",
            "backplate_mass_kg",
            "maximum_preload_approach_m",
            "preload_ramp_time_s",
            "settlement_damping_scale",
            "output_spacing_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContactConfigurationError(f"{name} must be finite and positive")
        for name in (
            "external_total_preload_n",
            "backplate_vertical_damping_n_s_m",
            "effective_pin_normal_force_min_n",
            "settling_reaction_force_tolerance_n",
            "settling_reaction_force_relative_tolerance",
            "settling_dynamic_residual_tolerance_n",
            "dynamic_residual_tolerance_n",
            "coupled_projection_relaxation",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ContactConfigurationError(
                    f"{name} must be finite and non-negative"
                )
        if not math.isfinite(self.initial_common_ux_m):
            raise ContactConfigurationError("initial_common_ux_m must be finite")
        if self.preload_ramp_profile != "minimum_jerk_quintic":
            raise ContactConfigurationError(
                "preload_ramp_profile currently supports minimum_jerk_quintic only"
            )
        if not 0.0 < self.coupled_projection_relaxation <= 1.0:
            raise ContactConfigurationError(
                "coupled_projection_relaxation must lie in (0, 1]"
            )
        if self.settling_stable_steps < 1:
            raise ContactConfigurationError(
                "settling_stable_steps must be positive"
            )
        if self.backplate_rotational_dofs != "locked":
            raise ContactConfigurationError(
                "m3.3.0 opens common backplate Z only; rotations must be locked"
            )
        if self.backplate_inertia_kg_m2 is not None:
            raise ContactConfigurationError(
                "backplate inertia must be null while pitch/roll are locked"
            )
        names = tuple(str(name) for name in self.unclosed_parameter_names)
        if any(not name for name in names):
            raise ContactConfigurationError(
                "unclosed_parameter_names cannot contain empty names"
            )
        object.__setattr__(self, "unclosed_parameter_names", names)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "ArrayDynamicExperimentSettings":
        allowed = set(cls.__dataclass_fields__)
        extra = set(value) - allowed
        if extra:
            raise ContactConfigurationError(
                f"ArrayDynamicExperimentSettings contains unknown fields: {sorted(extra)}"
            )
        data = dict(value)
        if "unclosed_parameter_names" in data:
            data["unclosed_parameter_names"] = tuple(
                data["unclosed_parameter_names"]
            )
        return cls(**data)


@dataclass(frozen=True)
class _ContactGeometry:
    pin_index: int
    geometry: Any
    gap_m: float
    normal: NDArray[np.float64]
    tangent: NDArray[np.float64]
    normal_jacobian: NDArray[np.float64]
    tangent_jacobian: NDArray[np.float64]
    normal_velocity_m_s: float
    tangential_velocity_m_s: float
    was_active: bool


class DynamicCommonBackplateArray:
    """Pure array-level mechanics used by the production M3 experiment."""

    def __init__(
        self,
        configuration: ArrayConfiguration,
        tracks: Sequence[TrackGeometry],
        *,
        unit_origin_xy_m: tuple[float, float],
        contact: DynamicContactSettings | None = None,
        target_load_threshold_n: float = 1e-6,
    ) -> None:
        self.configuration = configuration
        self.tracks = tuple(tracks)
        self.unit_origin_xy_m = (
            float(unit_origin_xy_m[0]),
            float(unit_origin_xy_m[1]),
        )
        self.contact = contact or DynamicContactSettings(
            projection_iterations=20
        )
        if (
            not math.isfinite(target_load_threshold_n)
            or target_load_threshold_n < 0.0
        ):
            raise ContactConfigurationError(
                "target_load_threshold_n must be finite and non-negative"
            )
        self.target_load_threshold_n = float(target_load_threshold_n)
        if len(self.tracks) != configuration.pin_count:
            raise ContactConfigurationError(
                "one TrackGeometry entry is required for every pin"
            )
        recipe_ids = {track.terrain_recipe_id for track in self.tracks}
        region_ids = {track.region_id for track in self.tracks}
        if len(recipe_ids) != 1 or len(region_ids) != 1:
            raise ContactConfigurationError(
                "all pins must share one terrain recipe and one region"
            )
        self.terrain_recipe_id = next(iter(recipe_ids))
        self.region_id = next(iter(region_ids))
        self.pin_parameters = configuration.pin_parameters
        self.holder_offsets_xyz_m = configuration.holder_offsets_xyz_m
        expected_y = np.asarray(
            [
                self.unit_origin_xy_m[1] + offset[1]
                for offset in self.holder_offsets_xyz_m
            ],
            dtype=np.float64,
        )
        actual_y = np.asarray(
            [track.y_global_m for track in self.tracks],
            dtype=np.float64,
        )
        if not np.allclose(expected_y, actual_y, rtol=0.0, atol=1e-12):
            raise ContactConfigurationError(
                "each pin track y_global_m must match its global holder y coordinate"
            )
        for parameters, track in zip(self.pin_parameters, self.tracks):
            if not math.isclose(
                parameters.tip_radius_m,
                track.radius_m,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ContactConfigurationError(
                    "every pin tip radius must match its M1 track"
                )
        self.units = tuple(
            DynamicSingleSpineUnit(parameters, track, self.contact)
            for parameters, track in zip(self.pin_parameters, self.tracks)
        )

    @property
    def pin_count(self) -> int:
        return self.configuration.pin_count

    @property
    def dof_count(self) -> int:
        return 1 + 2 * self.pin_count

    @staticmethod
    def _pin_dofs(pin_index: int) -> tuple[int, int]:
        return 1 + 2 * pin_index, 2 + 2 * pin_index

    def _holder_x_m(self, pin_index: int, common_ux_m: float) -> float:
        return (
            self.unit_origin_xy_m[0]
            + self.holder_offsets_xyz_m[pin_index][0]
            + common_ux_m
        )

    def _pack_state(
        self, state: ArrayDynamicState
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if len(state.pin_axial_displacement_m) != self.pin_count:
            raise ContactConfigurationError(
                "ArrayDynamicState pin count does not match configuration"
            )
        q = np.empty(self.dof_count, dtype=np.float64)
        v = np.empty(self.dof_count, dtype=np.float64)
        q[0] = state.backplate_position_z_m
        v[0] = state.backplate_velocity_z_m_s
        for index in range(self.pin_count):
            axial, transverse = self._pin_dofs(index)
            q[axial] = state.pin_axial_displacement_m[index]
            q[transverse] = state.pin_transverse_displacement_m[index]
            v[axial] = state.pin_axial_velocity_m_s[index]
            v[transverse] = state.pin_transverse_velocity_m_s[index]
        return q, v

    def _mass(self, backplate_mass_kg: float) -> NDArray[np.float64]:
        mass = np.empty(self.dof_count, dtype=np.float64)
        mass[0] = backplate_mass_kg
        for index, unit in enumerate(self.units):
            axial, transverse = self._pin_dofs(index)
            mass[axial] = unit.modes_mass[1]
            mass[transverse] = unit.modes_mass[2]
        return mass

    def _structure(
        self,
        q: NDArray[np.float64],
        settings: ArrayDynamicExperimentSettings,
        damping_scale: float = 1.0,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        tuple[SpringState, ...],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        mass = self._mass(settings.backplate_mass_kg)
        damping = np.zeros(self.dof_count, dtype=np.float64)
        tangent = np.zeros(self.dof_count, dtype=np.float64)
        restoring = np.zeros(self.dof_count, dtype=np.float64)
        spring_states: list[SpringState] = []
        compression = np.zeros(self.pin_count, dtype=np.float64)
        energy = np.zeros(self.pin_count, dtype=np.float64)
        damping[0] = settings.backplate_vertical_damping_n_s_m
        for index, (unit, parameters) in enumerate(
            zip(self.units, self.pin_parameters)
        ):
            axial, transverse = self._pin_dofs(index)
            (
                axial_force,
                axial_tangent,
                axial_energy,
                spring_state,
                spring_compression,
            ) = unit.axial_response(float(q[axial]))
            transverse_stiffness = unit.transverse_stiffness_n_m
            restoring[axial] = axial_force
            restoring[transverse] = transverse_stiffness * q[transverse]
            tangent[axial] = axial_tangent
            tangent[transverse] = transverse_stiffness
            damping[axial] = (
                2.0
                * parameters.axial_damping_ratio
                * math.sqrt(axial_tangent * mass[axial])
            )
            damping[transverse] = (
                2.0
                * parameters.transverse_damping_ratio
                * math.sqrt(transverse_stiffness * mass[transverse])
            )
            spring_states.append(spring_state)
            compression[index] = spring_compression
            energy[index] = (
                axial_energy
                + 0.5 * transverse_stiffness * q[transverse] ** 2
            )
        damping *= damping_scale
        return (
            damping,
            tangent,
            restoring,
            tuple(spring_states),
            compression,
            energy,
        )

    def _free_implicit_step(
        self,
        state: ArrayDynamicState,
        settings: ArrayDynamicExperimentSettings,
        dt: float,
        applied_external_total_preload_n: float,
        damping_scale: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
    ]:
        q0, v0 = self._pack_state(state)
        mass = self._mass(settings.backplate_mass_kg)
        v = np.empty(self.dof_count, dtype=np.float64)
        plate_damping = (
            settings.backplate_vertical_damping_n_s_m * damping_scale
        )
        v[0] = (
            mass[0] * v0[0] / dt - applied_external_total_preload_n
        ) / (mass[0] / dt + plate_damping)

        def linear_mode_velocity(
            *,
            position: float,
            velocity: float,
            mode_mass: float,
            stiffness: float,
            damping_ratio: float,
            offset_force: float = 0.0,
        ) -> tuple[float, float]:
            damping = (
                2.0
                * damping_ratio
                * math.sqrt(stiffness * mode_mass)
                * damping_scale
            )
            solved_velocity = (
                mode_mass * velocity / dt
                - stiffness * position
                - offset_force
            ) / (
                mode_mass / dt
                + damping
                + stiffness * dt
            )
            return solved_velocity, position + dt * solved_velocity

        for index, (unit, parameters) in enumerate(
            zip(self.units, self.pin_parameters)
        ):
            axial, transverse = self._pin_dofs(index)
            compliance = parameters.axial_compliance_m_n
            beam_stiffness = 1.0 / compliance
            candidates: list[tuple[float, float]] = []

            def add_candidate(
                stiffness: float,
                offset_force: float,
                lower: float,
                upper: float,
            ) -> None:
                velocity, position = linear_mode_velocity(
                    position=float(q0[axial]),
                    velocity=float(v0[axial]),
                    mode_mass=float(mass[axial]),
                    stiffness=stiffness,
                    damping_ratio=parameters.axial_damping_ratio,
                    offset_force=offset_force,
                )
                violation = max(
                    lower - position,
                    0.0,
                    position - upper,
                )
                candidates.append((violation, velocity))

            if parameters.axial_mode.value == "rigid":
                add_candidate(
                    beam_stiffness,
                    0.0,
                    -math.inf,
                    math.inf,
                )
            else:
                spring_stiffness = float(
                    parameters.spring_stiffness_n_m
                )
                interior_stiffness = 1.0 / (
                    compliance + 1.0 / spring_stiffness
                )
                hard_start = (
                    parameters.spring_travel_m
                    + compliance
                    * spring_stiffness
                    * parameters.spring_travel_m
                )
                add_candidate(
                    beam_stiffness,
                    0.0,
                    -math.inf,
                    0.0,
                )
                add_candidate(
                    interior_stiffness,
                    0.0,
                    0.0,
                    hard_start,
                )
                add_candidate(
                    beam_stiffness,
                    -parameters.spring_travel_m / compliance,
                    hard_start,
                    math.inf,
                )
            v[axial] = min(candidates, key=lambda item: item[0])[1]
            transverse_stiffness = unit.transverse_stiffness_n_m
            v[transverse], _position = linear_mode_velocity(
                position=float(q0[transverse]),
                velocity=float(v0[transverse]),
                mode_mass=float(mass[transverse]),
                stiffness=transverse_stiffness,
                damping_ratio=parameters.transverse_damping_ratio,
            )
        q = q0 + dt * v
        damping, tangent, *_ = self._structure(
            q,
            settings,
            damping_scale,
        )
        effective_inverse = 1.0 / (mass + dt * damping + dt**2 * tangent)
        return q, v, q0, v0, effective_inverse, 1

    def _pin_kinematics(
        self,
        pin_index: int,
        common_ux_m: float,
        drag_speed_m_s: float,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        axial, transverse = self._pin_dofs(pin_index)
        local_q = np.array([q[0], q[axial], q[transverse]], dtype=np.float64)
        local_v = np.array([v[0], v[axial], v[transverse]], dtype=np.float64)
        return self.units[pin_index].kinematics(
            self._holder_x_m(pin_index, common_ux_m),
            drag_speed_m_s,
            local_q,
            local_v,
        )

    def _global_jacobian(
        self,
        pin_index: int,
        direction_xz: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        local = self.units[pin_index].generalized_contact_jacobian(direction_xz)
        result = np.zeros(self.dof_count, dtype=np.float64)
        axial, transverse = self._pin_dofs(pin_index)
        result[0] = local[0]
        result[axial] = local[1]
        result[transverse] = local[2]
        return result

    def _contact_geometries(
        self,
        state: ArrayDynamicState,
        common_ux_m: float,
        drag_speed_m_s: float,
        q_free: NDArray[np.float64],
        v_free: NDArray[np.float64],
        query_q: NDArray[np.float64],
        traversal_order: Sequence[int],
    ) -> tuple[_ContactGeometry, ...]:
        output: list[_ContactGeometry | None] = [None] * self.pin_count
        for index in traversal_order:
            center_free, velocity_free = self._pin_kinematics(
                index,
                common_ux_m,
                drag_speed_m_s,
                q_free,
                v_free,
            )
            center_query, _ = self._pin_kinematics(
                index,
                common_ux_m,
                drag_speed_m_s,
                query_q,
                v_free,
            )
            geometry = self.units[index].geometry.query(float(center_query[0]))
            normal = np.asarray(geometry.normal_xz, dtype=np.float64)
            tangent = np.asarray(geometry.tangent_xz, dtype=np.float64)
            output[index] = _ContactGeometry(
                pin_index=index,
                geometry=geometry,
                gap_m=float(
                    center_query[1] - geometry.envelope_height_m
                ),
                normal=normal,
                tangent=tangent,
                normal_jacobian=self._global_jacobian(index, normal),
                tangent_jacobian=self._global_jacobian(index, tangent),
                normal_velocity_m_s=float(velocity_free @ normal),
                tangential_velocity_m_s=float(velocity_free @ tangent),
                was_active=state.pin_contact_state[index]
                in _ACTIVE_CONTACT_STATES,
            )
        assert all(item is not None for item in output)
        return tuple(item for item in output if item is not None)

    def _solve_coupled_impulses(
        self,
        contacts: Sequence[_ContactGeometry],
        effective_inverse: NDArray[np.float64],
        dt: float,
        drag_speed_m_s: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[bool, ...]]:
        count = len(contacts)
        pn = np.zeros(count, dtype=np.float64)
        pt = np.zeros(count, dtype=np.float64)
        sticking = np.zeros(count, dtype=np.bool_)
        if count == 0:
            return pn, pt, tuple(bool(value) for value in sticking)

        active = list(range(count))
        sliding: dict[int, float] = {}
        normal_rhs_all = np.asarray(
            [
                max(
                    0.0,
                    -self.contact.position_correction * item.gap_m / dt,
                    (
                        -(1.0 + self.contact.restitution_coefficient)
                        * item.normal_velocity_m_s
                        if not item.was_active
                        else -item.normal_velocity_m_s
                    ),
                )
                for item in contacts
            ],
            dtype=np.float64,
        )
        tangent_rhs_all = np.asarray(
            [-item.tangential_velocity_m_s for item in contacts],
            dtype=np.float64,
        )

        for _ in range(2 * count + 4):
            if not active:
                break
            stick = [index for index in active if index not in sliding]
            row_vectors = [contacts[index].normal_jacobian for index in active]
            row_vectors.extend(contacts[index].tangent_jacobian for index in stick)
            column_vectors: list[NDArray[np.float64]] = []
            for index in active:
                column = contacts[index].normal_jacobian.copy()
                if index in sliding:
                    mu = self.pin_parameters[contacts[index].pin_index].kinetic_friction
                    column -= (
                        mu
                        * sliding[index]
                        * contacts[index].tangent_jacobian
                    )
                column_vectors.append(column)
            column_vectors.extend(
                contacts[index].tangent_jacobian for index in stick
            )
            matrix = np.asarray(
                [
                    [
                        float(row @ (effective_inverse * column))
                        for column in column_vectors
                    ]
                    for row in row_vectors
                ],
                dtype=np.float64,
            )
            rhs = np.concatenate(
                (
                    normal_rhs_all[active],
                    tangent_rhs_all[stick],
                )
            )
            try:
                solution = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=1e-12)

            negative = [
                index
                for position, index in enumerate(active)
                if solution[position] < -1e-14
            ]
            if negative:
                active = [index for index in active if index not in negative]
                for index in negative:
                    sliding.pop(index, None)
                continue

            candidate_pn = {
                index: max(0.0, float(solution[position]))
                for position, index in enumerate(active)
            }
            candidate_pt = {
                index: float(solution[len(active) + position])
                for position, index in enumerate(stick)
            }
            violated: list[int] = []
            for index in stick:
                parameters = self.pin_parameters[contacts[index].pin_index]
                if (
                    abs(candidate_pt[index])
                    > parameters.static_friction * candidate_pn[index] + 1e-14
                ):
                    violated.append(index)
            if violated:
                for index in violated:
                    speed = contacts[index].tangential_velocity_m_s
                    direction = (
                        math.copysign(1.0, speed)
                        if abs(speed) > 1e-15
                        else math.copysign(1.0, drag_speed_m_s or 1.0)
                    )
                    sliding[index] = direction
                continue

            for index in active:
                pn[index] = candidate_pn[index]
                if index in sliding:
                    parameters = self.pin_parameters[contacts[index].pin_index]
                    pt[index] = (
                        -parameters.kinetic_friction
                        * sliding[index]
                        * pn[index]
                    )
                else:
                    pt[index] = candidate_pt[index]
                    sticking[index] = True
            break
        return pn, pt, tuple(bool(value) for value in sticking)

    def _unclosed_parameter_names(
        self, settings: ArrayDynamicExperimentSettings
    ) -> tuple[str, ...]:
        names = set(settings.unclosed_parameter_names)
        if not settings.physical_calibration_completed:
            names.add("m3.physical_calibration_completed")
        if not settings.terrain_resolution_convergence_checked:
            names.add("m3.terrain_resolution_convergence_checked")
        if not settings.settlement_damping_convergence_checked:
            names.add("m3.settlement_damping_scale_not_converged")
        for parameters in self.pin_parameters:
            if parameters.rod_clearance_mode == "unclosed":
                names.add("base_spine.rod_clearance_mode")
            if parameters.yield_strength_pa is None:
                names.add("base_spine.yield_strength_pa")
            if "unfrozen" in parameters.material_assumption.lower():
                names.add("base_spine.material_assumption")
        return tuple(sorted(names))

    def _refine_dynamic_equilibrium(
        self,
        *,
        settings: ArrayDynamicExperimentSettings,
        q0: NDArray[np.float64],
        v0: NDArray[np.float64],
        free_q: NDArray[np.float64],
        free_v: NDArray[np.float64],
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        contacts: Sequence[_ContactGeometry],
        normal_impulses: NDArray[np.float64],
        tangential_impulses: NDArray[np.float64],
        sticking: Sequence[bool],
        applied_preload_n: float,
        damping_scale: float,
        dt: float,
        residual_tolerance_n: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
    ]:
        """KKT correction that preserves active contact velocity constraints."""

        q_refined = q.copy()
        v_refined = v.copy()
        pn = normal_impulses.copy()
        pt = tangential_impulses.copy()
        mass = self._mass(settings.backplate_mass_kg)
        external = np.zeros(self.dof_count, dtype=np.float64)
        external[0] = -applied_preload_n
        maximum_iterations = 20
        for iteration in range(1, maximum_iterations + 1):
            active = [
                index
                for index, impulse in enumerate(pn)
                if impulse > 1e-16
            ]
            active_pin_indices = {
                contacts[index].pin_index for index in active
            }

            def linear_velocity(
                *,
                position: float,
                velocity: float,
                mode_mass: float,
                stiffness: float,
                damping_ratio: float,
                offset_force: float = 0.0,
            ) -> tuple[float, float]:
                damping_value = (
                    2.0
                    * damping_ratio
                    * math.sqrt(stiffness * mode_mass)
                    * damping_scale
                )
                solved = (
                    mode_mass * velocity / dt
                    - stiffness * position
                    - offset_force
                ) / (
                    mode_mass / dt
                    + damping_value
                    + stiffness * dt
                )
                return solved, position + dt * solved

            # Contact-free pin modes are diagonal.  Re-solve their nonsmooth
            # axial branches exactly after every common-plate/contact update.
            for pin_index, (unit, parameters) in enumerate(
                zip(self.units, self.pin_parameters)
            ):
                if pin_index in active_pin_indices:
                    continue
                axial, transverse = self._pin_dofs(pin_index)
                compliance = parameters.axial_compliance_m_n
                beam_stiffness = 1.0 / compliance
                candidates: list[tuple[float, float]] = []

                def add_candidate(
                    stiffness: float,
                    offset_force: float,
                    lower: float,
                    upper: float,
                ) -> None:
                    solved, position = linear_velocity(
                        position=float(q0[axial]),
                        velocity=float(v0[axial]),
                        mode_mass=float(mass[axial]),
                        stiffness=stiffness,
                        damping_ratio=parameters.axial_damping_ratio,
                        offset_force=offset_force,
                    )
                    candidates.append(
                        (
                            max(
                                lower - position,
                                0.0,
                                position - upper,
                            ),
                            solved,
                        )
                    )

                if parameters.axial_mode.value == "rigid":
                    add_candidate(
                        beam_stiffness,
                        0.0,
                        -math.inf,
                        math.inf,
                    )
                else:
                    spring_stiffness = float(
                        parameters.spring_stiffness_n_m
                    )
                    interior_stiffness = 1.0 / (
                        compliance + 1.0 / spring_stiffness
                    )
                    hard_start = (
                        parameters.spring_travel_m
                        + compliance
                        * spring_stiffness
                        * parameters.spring_travel_m
                    )
                    add_candidate(
                        beam_stiffness,
                        0.0,
                        -math.inf,
                        0.0,
                    )
                    add_candidate(
                        interior_stiffness,
                        0.0,
                        0.0,
                        hard_start,
                    )
                    add_candidate(
                        beam_stiffness,
                        -parameters.spring_travel_m / compliance,
                        hard_start,
                        math.inf,
                    )
                v_refined[axial] = min(
                    candidates,
                    key=lambda item: item[0],
                )[1]
                v_refined[transverse], _position = linear_velocity(
                    position=float(q0[transverse]),
                    velocity=float(v0[transverse]),
                    mode_mass=float(mass[transverse]),
                    stiffness=unit.transverse_stiffness_n_m,
                    damping_ratio=parameters.transverse_damping_ratio,
                )
                q_refined[axial] = q0[axial] + dt * v_refined[axial]
                q_refined[transverse] = (
                    q0[transverse] + dt * v_refined[transverse]
                )
            damping, tangent, restoring, *_ = self._structure(
                q_refined,
                settings,
                damping_scale,
            )
            generalized_contact = np.zeros(
                self.dof_count,
                dtype=np.float64,
            )
            for index in active:
                generalized_contact += (
                    contacts[index].normal_jacobian * pn[index]
                    + contacts[index].tangent_jacobian * pt[index]
                ) / dt
            residual = (
                mass * (v_refined - v0) / dt
                + damping * v_refined
                + restoring
                - external
                - generalized_contact
            )
            if (
                float(np.max(np.abs(residual)))
                <= residual_tolerance_n
            ):
                return q_refined, v_refined, pn, pt, iteration
            correction_residual = residual.copy()
            for pin_index in range(self.pin_count):
                if pin_index in active_pin_indices:
                    continue
                axial, transverse = self._pin_dofs(pin_index)
                correction_residual[axial] = 0.0
                correction_residual[transverse] = 0.0
            velocity_tangent = mass / dt + damping + dt * tangent
            effective_inverse = 1.0 / (
                mass + dt * damping + dt**2 * tangent
            )
            if not active:
                velocity_update = -correction_residual / velocity_tangent
                v_refined += velocity_update
                q_refined = q0 + dt * v_refined
                continue

            stick = [index for index in active if sticking[index]]
            rows = [contacts[index].normal_jacobian for index in active]
            rows.extend(contacts[index].tangent_jacobian for index in stick)
            columns: list[NDArray[np.float64]] = []
            slide_ratios: dict[int, float] = {}
            for index in active:
                column = contacts[index].normal_jacobian.copy()
                if not sticking[index]:
                    ratio = (
                        pt[index] / pn[index]
                        if pn[index] > 1e-16
                        else 0.0
                    )
                    slide_ratios[index] = ratio
                    column += ratio * contacts[index].tangent_jacobian
                columns.append(column)
            columns.extend(
                contacts[index].tangent_jacobian for index in stick
            )
            matrix = np.asarray(
                [
                    [
                        float(row @ (effective_inverse * column))
                        for column in columns
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )
            rhs = dt * np.asarray(
                [
                    float(row @ (effective_inverse * correction_residual))
                    for row in rows
                ],
                dtype=np.float64,
            )
            try:
                impulse_update = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                impulse_update, *_ = np.linalg.lstsq(
                    matrix,
                    rhs,
                    rcond=1e-12,
                )
            generalized_impulse_update = np.zeros(
                self.dof_count,
                dtype=np.float64,
            )
            invalid_active_index: int | None = None
            for position, index in enumerate(active):
                delta_normal = float(impulse_update[position])
                proposed_normal = pn[index] + delta_normal
                if proposed_normal < -1e-14:
                    invalid_active_index = index
                    break
                pn[index] = max(0.0, proposed_normal)
                if not sticking[index]:
                    pt[index] = slide_ratios[index] * pn[index]
                generalized_impulse_update += (
                    columns[position] * delta_normal
                )
            if invalid_active_index is not None:
                pn[invalid_active_index] = 0.0
                pt[invalid_active_index] = 0.0
                freed_pin = contacts[invalid_active_index].pin_index
                freed_axial, freed_transverse = self._pin_dofs(
                    freed_pin
                )
                for dof in (freed_axial, freed_transverse):
                    q_refined[dof] = free_q[dof]
                    v_refined[dof] = free_v[dof]
                continue
            else:
                for position, index in enumerate(stick):
                    delta_tangent = float(
                        impulse_update[len(active) + position]
                    )
                    pt[index] += delta_tangent
                    generalized_impulse_update += (
                        contacts[index].tangent_jacobian
                        * delta_tangent
                    )
                velocity_update = (
                    -dt * effective_inverse * correction_residual
                    + effective_inverse * generalized_impulse_update
                )
                v_refined += velocity_update
                q_refined = q0 + dt * v_refined
                continue
        return q_refined, v_refined, pn, pt, maximum_iterations

    def _model_state(
        self, settings: ArrayDynamicExperimentSettings
    ) -> ModelState:
        if self._unclosed_parameter_names(settings):
            return ModelState.PARAMETER_UNCLOSED
        return ModelState.COVERED

    def initial_state(
        self,
        settings: ArrayDynamicExperimentSettings,
    ) -> ArrayDynamicState:
        required_z: list[float] = []
        for index, unit in enumerate(self.units):
            holder_x = self._holder_x_m(index, settings.initial_common_ux_m)
            undeformed_center_x = (
                holder_x
                + self.pin_parameters[index].exposed_length_m * unit.a[0]
            )
            geometry = unit.geometry.query(float(undeformed_center_x))
            required_z.append(
                float(
                    geometry.envelope_height_m
                    - self.pin_parameters[index].exposed_length_m * unit.a[1]
                )
            )
        # Highest required holder Z leaves every undeformed tip nonpenetrating.
        return ArrayDynamicState.initial(self.pin_count, max(required_z))

    def propose_step(
        self,
        state: ArrayDynamicState,
        settings: ArrayDynamicExperimentSettings,
        *,
        common_ux_m: float,
        drag_speed_m_s: float,
        dt: float,
        applied_external_total_preload_n: float | None = None,
        damping_scale: float = 1.0,
        enforce_dynamic_residual_gate: bool = True,
        traversal_order: Sequence[int] | None = None,
    ) -> ArrayDynamicStepProposal:
        """Build one global proposal without mutating any state or pin history."""

        if not math.isfinite(dt) or dt <= 0.0:
            raise ContactConfigurationError("dt must be finite and positive")
        applied_preload = (
            settings.external_total_preload_n
            if applied_external_total_preload_n is None
            else float(applied_external_total_preload_n)
        )
        if (
            not math.isfinite(applied_preload)
            or applied_preload < 0.0
            or applied_preload > settings.external_total_preload_n + 1e-12
        ):
            raise ContactConfigurationError(
                "applied_external_total_preload_n must be finite and lie "
                "between zero and the configured total preload"
            )
        if not math.isfinite(damping_scale) or damping_scale <= 0.0:
            raise ContactConfigurationError(
                "damping_scale must be finite and positive"
            )
        if traversal_order is None:
            order = tuple(range(self.pin_count))
        else:
            order = tuple(int(index) for index in traversal_order)
            if sorted(order) != list(range(self.pin_count)):
                raise ContactConfigurationError(
                    "traversal_order must be a permutation of pin indices"
                )
        try:
            (
                q_free,
                v_free,
                q0,
                v0,
                effective_inverse,
                nonlinear_iterations,
            ) = self._free_implicit_step(
                state,
                settings,
                dt,
                applied_preload,
                damping_scale,
            )
            q = q_free.copy()
            v = v_free.copy()
            contacts: tuple[_ContactGeometry, ...] = ()
            pn = np.zeros(0, dtype=np.float64)
            pt = np.zeros(0, dtype=np.float64)
            sticking: tuple[bool, ...] = ()
            projection_iterations = 0
            for projection_iterations in range(
                1, self.contact.projection_iterations + 1
            ):
                if projection_iterations == 1:
                    base_q = q_free
                    base_v = v_free
                    projection_effective_inverse = effective_inverse
                else:
                    mass = self._mass(settings.backplate_mass_kg)
                    damping, tangent, restoring, *_ = self._structure(
                        q,
                        settings,
                        damping_scale,
                    )
                    external = np.zeros(
                        self.dof_count,
                        dtype=np.float64,
                    )
                    external[0] = -applied_preload
                    no_contact_residual = (
                        mass * (v - v0) / dt
                        + damping * v
                        + restoring
                        - external
                    )
                    velocity_tangent = (
                        mass / dt + damping + dt * tangent
                    )
                    base_v = v - no_contact_residual / velocity_tangent
                    base_q = q0 + dt * base_v
                    projection_effective_inverse = (
                        1.0
                        / (
                            mass
                            + dt * damping
                            + dt**2 * tangent
                        )
                    )
                all_geometry = self._contact_geometries(
                    state,
                    common_ux_m,
                    drag_speed_m_s,
                    base_q,
                    base_v,
                    q,
                    order,
                )
                contacts = tuple(
                    item
                    for item in all_geometry
                    if item.gap_m <= self.contact.activation_tolerance_m
                    and (
                        item.was_active
                        or item.normal_velocity_m_s < 0.0
                        or item.gap_m < 0.0
                    )
                )
                pn, pt, sticking = self._solve_coupled_impulses(
                    contacts,
                    projection_effective_inverse,
                    dt,
                    drag_speed_m_s,
                )
                generalized_impulse = np.zeros(self.dof_count, dtype=np.float64)
                for item, normal_impulse, tangent_impulse in zip(
                    contacts, pn, pt
                ):
                    generalized_impulse += (
                        item.normal_jacobian * normal_impulse
                        + item.tangent_jacobian * tangent_impulse
                    )
                v_new = (
                    base_v
                    + projection_effective_inverse * generalized_impulse
                )
                q_candidate = q0 + dt * v_new
                if float(np.max(np.abs(q_candidate - q))) <= 1e-12:
                    q = q_candidate
                    v = v_new
                    break
                q = (
                    q
                    + settings.coupled_projection_relaxation
                    * (q_candidate - q)
                )
                v = (q - q0) / dt

            (
                q,
                v,
                pn,
                pt,
                refinement_iterations,
            ) = self._refine_dynamic_equilibrium(
                settings=settings,
                q0=q0,
                v0=v0,
                free_q=q_free,
                free_v=v_free,
                q=q,
                v=v,
                contacts=contacts,
                normal_impulses=pn,
                tangential_impulses=pt,
                sticking=sticking,
                applied_preload_n=applied_preload,
                damping_scale=damping_scale,
                dt=dt,
                residual_tolerance_n=(
                    settings.dynamic_residual_tolerance_n
                    if enforce_dynamic_residual_gate
                    else settings.settling_dynamic_residual_tolerance_n
                ),
            )
            point, new_state = self._assemble_step(
                state,
                settings,
                common_ux_m=common_ux_m,
                drag_speed_m_s=drag_speed_m_s,
                applied_external_total_preload_n=applied_preload,
                damping_scale=damping_scale,
                enforce_dynamic_residual_gate=(
                    enforce_dynamic_residual_gate
                ),
                dt=dt,
                q0=q0,
                v0=v0,
                q=q,
                v=v,
                contacts=contacts,
                normal_impulses=pn,
                tangential_impulses=pt,
                sticking=sticking,
                nonlinear_iterations=nonlinear_iterations
                + projection_iterations
                + refinement_iterations,
            )
        except ContactGeometryError as exc:
            return ArrayDynamicStepProposal(
                source_state=state,
                proposal_state=state,
                point=None,
                proposal_valid=False,
                rejection_reason=f"terrain_bounds:{exc}",
            )
        except (ContactConfigurationError, RuntimeError) as exc:
            return ArrayDynamicStepProposal(
                source_state=state,
                proposal_state=state,
                point=None,
                proposal_valid=False,
                rejection_reason=str(exc),
            )
        return ArrayDynamicStepProposal(
            source_state=state,
            proposal_state=new_state,
            point=point,
            proposal_valid=True,
            rejection_reason="",
        )

    @staticmethod
    def commit_step(
        old_state: ArrayDynamicState,
        proposal: ArrayDynamicStepProposal,
        *,
        accept: bool,
    ) -> ArrayDynamicState:
        if proposal.source_state != old_state:
            raise ContactConfigurationError(
                "proposal source does not match the supplied old state"
            )
        if not accept or not proposal.proposal_valid:
            return old_state
        return proposal.proposal_state

    def _assemble_step(
        self,
        state: ArrayDynamicState,
        settings: ArrayDynamicExperimentSettings,
        *,
        common_ux_m: float,
        drag_speed_m_s: float,
        applied_external_total_preload_n: float,
        damping_scale: float,
        enforce_dynamic_residual_gate: bool,
        dt: float,
        q0: NDArray[np.float64],
        v0: NDArray[np.float64],
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        contacts: Sequence[_ContactGeometry],
        normal_impulses: NDArray[np.float64],
        tangential_impulses: NDArray[np.float64],
        sticking: Sequence[bool],
        nonlinear_iterations: int,
    ) -> tuple[ArrayDynamicPathPoint, ArrayDynamicState]:
        acceleration = (v - v0) / dt
        mass = self._mass(settings.backplate_mass_kg)
        (
            damping,
            _tangent,
            restoring,
            spring_states,
            compression,
            structural_energy_by_pin,
        ) = self._structure(q, settings, damping_scale)
        (
            _damping0,
            _tangent0,
            _restoring0,
            _spring0,
            _compression0,
            structural_energy0_by_pin,
        ) = self._structure(q0, settings, damping_scale)
        contact_by_pin = {
            item.pin_index: (item, normal_impulse, tangent_impulse, stick)
            for item, normal_impulse, tangent_impulse, stick in zip(
                contacts,
                normal_impulses,
                tangential_impulses,
                sticking,
            )
            if normal_impulse > 1e-16
        }
        generalized_contact = np.zeros(self.dof_count, dtype=np.float64)
        for item, normal_impulse, tangent_impulse, _stick in contact_by_pin.values():
            generalized_contact += (
                item.normal_jacobian * normal_impulse
                + item.tangent_jacobian * tangent_impulse
            ) / dt

        external = np.zeros(self.dof_count, dtype=np.float64)
        external[0] = -applied_external_total_preload_n
        residual_vector = (
            mass * acceleration
            + damping * v
            + restoring
            - external
            - generalized_contact
        )
        dynamic_residual = float(np.max(np.abs(residual_vector)))

        unit_origin = np.array(
            [
                self.unit_origin_xy_m[0] + common_ux_m,
                self.unit_origin_xy_m[1],
                q[0],
            ],
            dtype=np.float64,
        )
        pin_responses: list[PinDynamicResponse] = []
        pin_wrenches: list[NDArray[np.float64]] = []
        event_labels: list[tuple[int, str]] = []
        friction_dissipation = 0.0
        drive_work = 0.0
        normal_contact_work = 0.0
        tangential_contact_work = 0.0
        contact_energy_injection = 0.0
        active_indices: list[int] = []
        geometric_indices: list[int] = []
        admissible_indices: list[int] = []
        has_contacted: list[bool] = []
        next_contact_states: list[ContactState] = []

        for index, (unit, parameters, spring_state) in enumerate(
            zip(self.units, self.pin_parameters, spring_states)
        ):
            axial, transverse = self._pin_dofs(index)
            center_xz, center_velocity_xz = self._pin_kinematics(
                index, common_ux_m, drag_speed_m_s, q, v
            )
            center_acceleration_xz = (
                np.array([0.0, acceleration[0]], dtype=np.float64)
                - acceleration[axial] * unit.a
                + acceleration[transverse] * unit.b
            )
            geometry = unit.geometry.query(float(center_xz[0]))
            gap = float(center_xz[1] - geometry.envelope_height_m)
            if (
                index in contact_by_pin
                or gap <= self.contact.activation_tolerance_m
                or float(
                    center_velocity_xz
                    @ np.asarray(geometry.normal_xz, dtype=np.float64)
                )
                < 0.0
            ):
                geometric_indices.append(index)
            admissible_indices.append(index)
            was_active = state.pin_contact_state[index] in _ACTIVE_CONTACT_STATES
            if index in contact_by_pin:
                item, normal_impulse, tangent_impulse, stick = contact_by_pin[index]
                active = True
                normal = item.normal
                tangent = item.tangent
                impact_velocity = (
                    max(0.0, -item.normal_velocity_m_s)
                    if not was_active
                    else 0.0
                )
                normal_force = normal_impulse / dt
                tangential_force = tangent_impulse / dt
                wall_force = normal_force * normal + tangential_force * tangent
                if (
                    float(np.linalg.norm(wall_force))
                    > self.contact.maximum_contact_force_n
                ):
                    raise RuntimeError(
                        "structural_boundary: dynamic contact force exceeds safety limit"
                    )
                active_indices.append(index)
                if not was_active:
                    contact_state = (
                        ContactState.IMPACT_EVENT
                        if impact_velocity
                        >= self.contact.impact_velocity_threshold_m_s
                        else ContactState.RECONTACT_EVENT
                    )
                    labels = [EventLabel.RECONTACT.value]
                    event_label = EventLabel.RECONTACT
                    if (
                        impact_velocity
                        >= self.contact.impact_velocity_threshold_m_s
                    ):
                        labels.append(EventLabel.IMPACT.value)
                        event_label = EventLabel.IMPACT
                else:
                    contact_state = (
                        ContactState.STICK if stick else ContactState.SLIDE
                    )
                    labels = []
                    event_label = EventLabel.NONE
                    if (
                        not stick
                        and state.pin_contact_state[index] is ContactState.STICK
                    ):
                        labels.append(EventLabel.SLIP_START.value)
                        event_label = EventLabel.SLIP_START
                    elif (
                        stick
                        and state.pin_contact_state[index] is ContactState.SLIDE
                    ):
                        labels.append(EventLabel.STICK_RECOVERED.value)
                        event_label = EventLabel.STICK_RECOVERED
                support = geometry.support_xyz_m
                tangent_value = geometry.tangent_xz
                normal_value = geometry.normal_xz
                tangential_speed = float(center_velocity_xz @ tangent)
                normal_speed = float(center_velocity_xz @ normal)
                pin_normal_work = normal_impulse * normal_speed
                pin_tangential_work = tangent_impulse * tangential_speed
                normal_contact_work += pin_normal_work
                tangential_contact_work += pin_tangential_work
                friction_dissipation += max(0.0, -pin_tangential_work)
                contact_energy_injection += max(
                    0.0,
                    pin_normal_work + pin_tangential_work,
                )
                drive_work += -float(wall_force[0]) * drag_speed_m_s * dt
            else:
                active = False
                normal_impulse = 0.0
                tangent_impulse = 0.0
                impact_velocity = 0.0
                normal_force = 0.0
                tangential_force = 0.0
                wall_force = np.zeros(2, dtype=np.float64)
                contact_state = (
                    ContactState.DETACH_EVENT
                    if was_active
                    else ContactState.FREE
                )
                labels = (
                    [EventLabel.DETACH_TO_FREE.value] if was_active else []
                )
                event_label = (
                    EventLabel.DETACH_TO_FREE if was_active else EventLabel.NONE
                )
                support = None
                tangent_value = None
                normal_value = None
            if (
                spring_state is SpringState.HARD_STOP
                and state.pin_spring_state[index] is not SpringState.HARD_STOP
            ):
                labels.append(EventLabel.HARD_STOP.value)
                if event_label is EventLabel.NONE:
                    event_label = EventLabel.HARD_STOP
            for label in labels:
                event_labels.append((index, label))

            holder = np.array(
                [
                    self._holder_x_m(index, common_ux_m),
                    self.unit_origin_xy_m[1]
                    + self.holder_offsets_xyz_m[index][1],
                    q[0],
                ],
                dtype=np.float64,
            )
            plate_force_xyz = np.array(
                [-wall_force[0], 0.0, -wall_force[1]],
                dtype=np.float64,
            )
            if support is None:
                holder_moment = np.zeros(3, dtype=np.float64)
            else:
                holder_moment = np.cross(
                    np.asarray(support, dtype=np.float64) - holder,
                    plate_force_xyz,
                )
            holder_wrench = np.concatenate((plate_force_xyz, holder_moment))
            unit_wrench = np.concatenate(
                (
                    plate_force_xyz,
                    holder_moment
                    + np.cross(holder - unit_origin, plate_force_xyz),
                )
            )
            pin_wrenches.append(unit_wrench)
            axial_force = restoring[axial]
            transverse_force = restoring[transverse]
            bending_stress = (
                32.0
                * abs(transverse_force)
                * parameters.exposed_length_m
                / (math.pi * parameters.diameter_m**3)
            )
            buckling_margin = unit.euler_buckling_load_n - max(
                axial_force, 0.0
            )
            pin_responses.append(
                PinDynamicResponse(
                    holder_xyz_m=tuple(float(value) for value in holder),
                    center_xyz_m=(
                        float(center_xz[0]),
                        float(holder[1]),
                        float(center_xz[1]),
                    ),
                    center_velocity_xyz_m_s=(
                        float(center_velocity_xz[0]),
                        0.0,
                        float(center_velocity_xz[1]),
                    ),
                    center_acceleration_xyz_m_s2=(
                        float(center_acceleration_xz[0]),
                        0.0,
                        float(center_acceleration_xz[1]),
                    ),
                    support_xyz_m=support,
                    tangent_xz=tangent_value,
                    normal_xz=normal_value,
                    gap_m=gap,
                    contact_state=contact_state,
                    spring_state=spring_state,
                    event_label=event_label,
                    event_labels=tuple(labels),
                    wall_on_spine_force_xz_n=tuple(
                        float(value) for value in wall_force
                    ),
                    spine_on_plate_wrench_about_holder=tuple(
                        float(value) for value in holder_wrench
                    ),
                    spine_on_plate_wrench_about_unit=tuple(
                        float(value) for value in unit_wrench
                    ),
                    normal_force_n=float(normal_force),
                    tangential_force_n=float(tangential_force),
                    normal_impulse_n_s=float(normal_impulse),
                    tangential_impulse_n_s=float(tangent_impulse),
                    impact_velocity_m_s=float(impact_velocity),
                    axial_displacement_m=float(q[axial]),
                    transverse_displacement_m=float(q[transverse]),
                    axial_velocity_m_s=float(v[axial]),
                    transverse_velocity_m_s=float(v[transverse]),
                    axial_force_n=float(axial_force),
                    transverse_force_n=float(transverse_force),
                    spring_compression_m=float(compression[index]),
                    spring_travel_margin_m=float(
                        parameters.spring_travel_m - compression[index]
                    ),
                    bending_stress_pa=float(bending_stress),
                    euler_buckling_margin_n=float(buckling_margin),
                )
            )
            next_contact_states.append(contact_state)
            has_contacted.append(state.pin_has_contacted[index] or active)

        pin_wrench_array = np.asarray(pin_wrenches, dtype=np.float64)
        unit_wrench = np.sum(pin_wrench_array, axis=0)
        force_residual = float(
            np.linalg.norm(
                unit_wrench[:3] - np.sum(pin_wrench_array[:, :3], axis=0)
            )
        )
        moment_residual = float(
            np.linalg.norm(
                unit_wrench[3:] - np.sum(pin_wrench_array[:, 3:], axis=0)
            )
        )
        normal_weights = np.asarray(
            [pin.normal_force_n for pin in pin_responses], dtype=np.float64
        )
        tangential_weights = np.abs(pin_wrench_array[:, 0])
        resultant_weights = np.linalg.norm(pin_wrench_array[:, :3], axis=1)
        sharing = LoadSharingMetrics(
            neff_normal=_neff(normal_weights),
            neff_target_tangential=_neff(tangential_weights),
            neff_resultant=_neff(resultant_weights),
            max_mean_normal=_max_mean(normal_weights),
            max_mean_target_tangential=_max_mean(tangential_weights),
            max_mean_resultant=_max_mean(resultant_weights),
            gini_normal=_gini(normal_weights),
            gini_target_tangential=_gini(tangential_weights),
            gini_resultant=_gini(resultant_weights),
        )
        target_indices = tuple(
            int(index)
            for index in np.flatnonzero(
                tangential_weights > self.target_load_threshold_n
            )
        )
        activity = ActivitySets(
            nominal=tuple(range(self.pin_count)),
            geometric=tuple(geometric_indices),
            positive_normal=tuple(active_indices),
            admissible=tuple(admissible_indices),
            target_load=target_indices,
        )

        kinetic_energy = 0.5 * float(np.sum(mass * v**2))
        kinetic_energy0 = 0.5 * float(np.sum(mass * v0**2))
        structural_energy = float(np.sum(structural_energy_by_pin))
        structural_energy0 = float(np.sum(structural_energy0_by_pin))
        kinetic_energy_change = kinetic_energy - kinetic_energy0
        structural_energy_change = (
            structural_energy - structural_energy0
        )
        displacement_increment = q - q0
        preload_work = (
            -applied_external_total_preload_n * displacement_increment[0]
        )
        structural_damping_dissipation = float(
            np.sum(damping[1:] * v[1:] ** 2) * dt
        )
        backplate_damping_dissipation = float(damping[0] * v[0] ** 2 * dt)
        kinetic_backward_euler_dissipation = 0.5 * float(
            np.sum(mass * (v - v0) ** 2)
        )
        potential_backward_euler_dissipation = float(
            restoring @ displacement_increment
            - structural_energy_change
        )
        implicit_euler_dissipation = (
            kinetic_backward_euler_dissipation
            + potential_backward_euler_dissipation
        )
        generalized_contact_work = float(
            generalized_contact @ displacement_increment
        )
        contact_work_identity_residual = float(
            generalized_contact_work
            - (
                drive_work
                + normal_contact_work
                + tangential_contact_work
            )
        )
        energy_residual = (
            kinetic_energy_change
            + structural_energy_change
            + implicit_euler_dissipation
            + structural_damping_dissipation
            + backplate_damping_dissipation
            - preload_work
            - generalized_contact_work
        )
        energy_reference = (
            abs(kinetic_energy_change)
            + abs(structural_energy_change)
            + abs(implicit_euler_dissipation)
            + structural_damping_dissipation
            + backplate_damping_dissipation
            + abs(preload_work)
            + abs(generalized_contact_work)
        )
        relative_energy_residual = (
            abs(energy_residual) / max(energy_reference, 1e-15)
        )
        total_contact_reaction_z = float(
            sum(pin.wall_on_spine_force_xz_n[1] for pin in pin_responses)
        )
        model_state = self._model_state(settings)
        new_state = ArrayDynamicState(
            time_s=state.time_s + dt,
            backplate_position_z_m=float(q[0]),
            backplate_velocity_z_m_s=float(v[0]),
            pin_axial_displacement_m=tuple(
                float(q[self._pin_dofs(index)[0]])
                for index in range(self.pin_count)
            ),
            pin_transverse_displacement_m=tuple(
                float(q[self._pin_dofs(index)[1]])
                for index in range(self.pin_count)
            ),
            pin_axial_velocity_m_s=tuple(
                float(v[self._pin_dofs(index)[0]])
                for index in range(self.pin_count)
            ),
            pin_transverse_velocity_m_s=tuple(
                float(v[self._pin_dofs(index)[1]])
                for index in range(self.pin_count)
            ),
            pin_contact_state=tuple(next_contact_states),
            pin_spring_state=spring_states,
            pin_has_contacted=tuple(has_contacted),
            accepted_steps=state.accepted_steps + 1,
            rejected_steps=state.rejected_steps,
            cumulative_preload_work_j=(
                state.cumulative_preload_work_j + preload_work
            ),
            cumulative_drive_work_j=(
                state.cumulative_drive_work_j + drive_work
            ),
            cumulative_friction_dissipation_j=(
                state.cumulative_friction_dissipation_j
                + friction_dissipation
            ),
            cumulative_structural_damping_dissipation_j=(
                state.cumulative_structural_damping_dissipation_j
                + structural_damping_dissipation
            ),
            cumulative_backplate_damping_dissipation_j=(
                state.cumulative_backplate_damping_dissipation_j
                + backplate_damping_dissipation
            ),
            cumulative_implicit_euler_dissipation_j=(
                state.cumulative_implicit_euler_dissipation_j
                + implicit_euler_dissipation
            ),
            cumulative_normal_contact_work_j=(
                state.cumulative_normal_contact_work_j
                + normal_contact_work
            ),
            cumulative_tangential_contact_work_j=(
                state.cumulative_tangential_contact_work_j
                + tangential_contact_work
            ),
            cumulative_contact_energy_injection_j=(
                state.cumulative_contact_energy_injection_j
                + contact_energy_injection
            ),
            cumulative_energy_residual_j=(
                state.cumulative_energy_residual_j + energy_residual
            ),
            cumulative_energy_reference_j=(
                state.cumulative_energy_reference_j + energy_reference
            ),
            maximum_abs_energy_residual_j=max(
                state.maximum_abs_energy_residual_j,
                abs(energy_residual),
            ),
            maximum_relative_energy_residual=max(
                state.maximum_relative_energy_residual,
                relative_energy_residual,
            ),
            maximum_abs_contact_work_identity_residual_j=max(
                state.maximum_abs_contact_work_identity_residual_j,
                abs(contact_work_identity_residual),
            ),
        )
        cumulative_relative_energy_error = abs(
            new_state.cumulative_energy_residual_j
        ) / max(new_state.cumulative_energy_reference_j, 1e-15)
        point = ArrayDynamicPathPoint(
            time_s=new_state.time_s,
            path_position_m=0.0,
            backplate_position_xyz_m=tuple(float(value) for value in unit_origin),
            backplate_velocity_xyz_m_s=(
                float(drag_speed_m_s),
                0.0,
                float(v[0]),
            ),
            backplate_acceleration_xyz_m_s2=(0.0, 0.0, float(acceleration[0])),
            external_total_preload_n=applied_external_total_preload_n,
            pin_responses=tuple(pin_responses),
            wall_on_unit_wrench_about_origin=tuple(
                float(value) for value in unit_wrench
            ),
            active_thrust_wrench_about_origin=(0.0,) * 6,
            guide_reaction_wrench_about_origin=(0.0,) * 6,
            activity_sets=activity,
            sharing=sharing,
            active_pin_count=len(active_indices),
            effective_load_pin_count=int(
                np.count_nonzero(
                    normal_weights
                    >= settings.effective_pin_normal_force_min_n
                )
            ),
            total_contact_reaction_z_n=total_contact_reaction_z,
            backplate_inertia_force_z_n=float(mass[0] * acceleration[0]),
            backplate_damping_force_z_n=float(damping[0] * v[0]),
            kinetic_energy_j=float(kinetic_energy),
            structural_energy_j=float(structural_energy),
            preload_work_increment_j=float(preload_work),
            drive_work_increment_j=float(drive_work),
            cumulative_preload_work_j=float(
                state.cumulative_preload_work_j + preload_work
            ),
            cumulative_drive_work_j=float(
                state.cumulative_drive_work_j + drive_work
            ),
            friction_dissipation_increment_j=float(friction_dissipation),
            cumulative_friction_dissipation_j=float(
                state.cumulative_friction_dissipation_j
                + friction_dissipation
            ),
            structural_damping_dissipation_increment_j=float(
                structural_damping_dissipation
            ),
            cumulative_structural_damping_dissipation_j=float(
                state.cumulative_structural_damping_dissipation_j
                + structural_damping_dissipation
            ),
            backplate_damping_dissipation_increment_j=float(
                backplate_damping_dissipation
            ),
            cumulative_backplate_damping_dissipation_j=float(
                state.cumulative_backplate_damping_dissipation_j
                + backplate_damping_dissipation
            ),
            implicit_euler_dissipation_increment_j=float(
                implicit_euler_dissipation
            ),
            cumulative_implicit_euler_dissipation_j=float(
                new_state.cumulative_implicit_euler_dissipation_j
            ),
            normal_contact_work_increment_j=float(normal_contact_work),
            cumulative_normal_contact_work_j=float(
                new_state.cumulative_normal_contact_work_j
            ),
            tangential_contact_work_increment_j=float(
                tangential_contact_work
            ),
            cumulative_tangential_contact_work_j=float(
                new_state.cumulative_tangential_contact_work_j
            ),
            generalized_contact_work_increment_j=float(
                generalized_contact_work
            ),
            contact_work_identity_residual_j=float(
                contact_work_identity_residual
            ),
            contact_energy_injection_increment_j=float(
                contact_energy_injection
            ),
            cumulative_contact_energy_injection_j=float(
                new_state.cumulative_contact_energy_injection_j
            ),
            dynamic_residual_n=float(dynamic_residual),
            energy_residual_j=float(energy_residual),
            relative_energy_residual=float(relative_energy_residual),
            cumulative_energy_residual_j=float(
                new_state.cumulative_energy_residual_j
            ),
            cumulative_energy_reference_j=float(
                new_state.cumulative_energy_reference_j
            ),
            cumulative_relative_energy_error=float(
                cumulative_relative_energy_error
            ),
            running_maximum_abs_energy_residual_j=float(
                new_state.maximum_abs_energy_residual_j
            ),
            running_maximum_relative_energy_residual=float(
                new_state.maximum_relative_energy_residual
            ),
            running_maximum_abs_contact_work_identity_residual_j=float(
                new_state.maximum_abs_contact_work_identity_residual_j
            ),
            force_aggregation_residual_n=force_residual,
            moment_aggregation_residual_nm=moment_residual,
            actual_time_step_s=float(dt),
            nonlinear_iterations=int(nonlinear_iterations),
            numerical_state=NumericalState.CONVERGED,
            model_state=model_state,
            event_labels=tuple(event_labels),
        )
        if enforce_dynamic_residual_gate and (
            not math.isfinite(point.dynamic_residual_n)
            or abs(point.dynamic_residual_n)
            > settings.dynamic_residual_tolerance_n
        ):
            raise RuntimeError(
                "numerical_failure: dynamic residual gate exceeded: "
                f"{point.dynamic_residual_n:.9g} N > "
                f"{settings.dynamic_residual_tolerance_n:.9g} N"
            )
        return point, new_state


class _SettlementFailure(RuntimeError):
    def __init__(
        self,
        *,
        category: str,
        code: str,
        message: str,
        trace: Sequence[SettlementTracePoint],
    ) -> None:
        super().__init__(message)
        self.category = category
        self.code = code
        self.trace = tuple(trace)


class DynamicCommonBackplateExperiment:
    """Integrate common backplate Z and all pin modes under one total preload."""

    def __init__(
        self,
        system: DynamicCommonBackplateArray,
        settings: ArrayDynamicExperimentSettings,
        integrator: DynamicIntegratorSettings | None = None,
    ) -> None:
        self.system = system
        self.settings = settings
        self.integrator = integrator or DynamicIntegratorSettings()

    @staticmethod
    def _minimum_jerk_ramp_fraction(time_s: float, duration_s: float) -> float:
        ratio = min(1.0, max(0.0, time_s / duration_s))
        return ratio**3 * (10.0 - 15.0 * ratio + 6.0 * ratio**2)

    def _settle_with_trace(
        self,
    ) -> tuple[
        ArrayDynamicState,
        ArrayDynamicPathPoint,
        tuple[SettlementTracePoint, ...],
    ]:
        state = self.system.initial_state(self.settings)
        initial_z = state.backplate_position_z_m
        dt = self.integrator.time_step_s
        minimum_settling_time = max(
            self.integrator.settling_time_s,
            self.settings.preload_ramp_time_s,
        )
        minimum_steps = int(math.ceil(minimum_settling_time / dt))
        maximum_steps = int(
            math.ceil(self.integrator.maximum_settling_time_s / dt)
        )
        stable_steps = 0
        last_point: ArrayDynamicPathPoint | None = None
        trace: list[SettlementTracePoint] = []
        force_tolerance = max(
            self.settings.settling_reaction_force_tolerance_n,
            self.settings.settling_reaction_force_relative_tolerance
            * self.settings.external_total_preload_n,
        )
        for step_index in range(maximum_steps):
            next_time = state.time_s + dt
            ramp_fraction = self._minimum_jerk_ramp_fraction(
                next_time,
                self.settings.preload_ramp_time_s,
            )
            applied_preload = (
                self.settings.external_total_preload_n * ramp_fraction
            )
            proposal = self.system.propose_step(
                state,
                self.settings,
                common_ux_m=self.settings.initial_common_ux_m,
                drag_speed_m_s=0.0,
                dt=dt,
                applied_external_total_preload_n=applied_preload,
                damping_scale=self.settings.settlement_damping_scale,
                enforce_dynamic_residual_gate=False,
            )
            if not proposal.proposal_valid or proposal.point is None:
                reason = proposal.rejection_reason
                if reason.startswith("terrain_bounds:"):
                    category = "geometry_out_of_bounds"
                    code = "settlement_terrain_bounds"
                elif reason.startswith("structural_boundary:"):
                    category = "physical_boundary"
                    code = "settlement_structural_boundary"
                else:
                    category = "numerical_failure"
                    code = "settlement_step_rejected"
                raise _SettlementFailure(
                    category=category,
                    code=code,
                    message=reason,
                    trace=trace,
                )
            next_state = self.system.commit_step(state, proposal, accept=True)
            actual_approach = (
                initial_z - next_state.backplate_position_z_m
            )
            state = next_state
            last_point = proposal.point
            speeds = [
                abs(state.backplate_velocity_z_m_s),
                *(abs(value) for value in state.pin_axial_velocity_m_s),
                *(abs(value) for value in state.pin_transverse_velocity_m_s),
            ]
            maximum_speed = max(speeds)
            reaction_error = abs(
                last_point.total_contact_reaction_z_n - applied_preload
            )
            ramp_complete = ramp_fraction >= 1.0 - 1e-12
            velocity_gate = (
                maximum_speed
                <= self.integrator.settling_velocity_tolerance_m_s
            )
            force_gate = reaction_error <= force_tolerance
            residual_gate = (
                abs(last_point.dynamic_residual_n)
                <= self.settings.settling_dynamic_residual_tolerance_n
            )
            contact_gate = (
                self.settings.external_total_preload_n == 0.0
                or last_point.total_contact_reaction_z_n > 0.0
            )
            if (
                ramp_complete
                and velocity_gate
                and force_gate
                and residual_gate
                and contact_gate
            ):
                stable_steps += 1
            else:
                stable_steps = 0
            trace.append(
                SettlementTracePoint(
                    time_s=float(state.time_s),
                    ramp_fraction=float(ramp_fraction),
                    applied_total_preload_n=float(applied_preload),
                    damping_scale=float(
                        self.settings.settlement_damping_scale
                    ),
                    backplate_position_z_m=float(
                        state.backplate_position_z_m
                    ),
                    actual_approach_m=float(actual_approach),
                    maximum_mode_speed_m_s=float(maximum_speed),
                    total_contact_reaction_z_n=float(
                        last_point.total_contact_reaction_z_n
                    ),
                    contact_reaction_error_n=float(reaction_error),
                    dynamic_residual_n=float(last_point.dynamic_residual_n),
                    active_pin_count=int(last_point.active_pin_count),
                    stable_steps=int(stable_steps),
                )
            )
            if actual_approach > self.settings.maximum_preload_approach_m:
                raise _SettlementFailure(
                    category="physical_boundary",
                    code="maximum_preload_approach_exceeded",
                    message=(
                        "settlement maximum approach exceeded: "
                        f"{actual_approach:.9g} m > "
                        f"{self.settings.maximum_preload_approach_m:.9g} m"
                    ),
                    trace=trace,
                )
            if (
                step_index + 1 >= minimum_steps
                and stable_steps >= self.settings.settling_stable_steps
            ):
                break
        else:
            failed_gates: list[str] = []
            if trace:
                final = trace[-1]
                if (
                    final.maximum_mode_speed_m_s
                    > self.integrator.settling_velocity_tolerance_m_s
                ):
                    failed_gates.append("velocity")
                if final.contact_reaction_error_n > force_tolerance:
                    failed_gates.append("reaction_force_balance")
                if (
                    abs(final.dynamic_residual_n)
                    > self.settings.settling_dynamic_residual_tolerance_n
                ):
                    failed_gates.append("dynamic_residual")
                if (
                    self.settings.external_total_preload_n > 0.0
                    and final.active_pin_count == 0
                ):
                    failed_gates.append("positive_contact")
            raise _SettlementFailure(
                category="settlement_nonconvergence",
                code="settlement_stability_window_not_reached",
                message=(
                    "unified dynamic settling did not converge; failed_gates="
                    + ",".join(failed_gates or ["stable_window"])
                ),
                trace=trace,
            )
        assert last_point is not None
        return state, last_point, tuple(trace)

    def _settle(self) -> tuple[ArrayDynamicState, ArrayDynamicPathPoint]:
        """Compatibility fixture hook; production runs retain the audit trace."""

        state, point, _trace = self._settle_with_trace()
        return state, point

    def run(self) -> ArrayDynamicExperimentResult:
        try:
            (
                settled_state,
                settled_point,
                settlement_trace,
            ) = self._settle_with_trace()
        except ContactGeometryError as exc:
            return self._failed_result(
                reason=str(exc),
                failure_category="geometry_out_of_bounds",
                failure_code="settlement_terrain_bounds",
            )
        except _SettlementFailure as exc:
            return self._failed_result(
                reason=str(exc),
                failure_category=exc.category,
                failure_code=exc.code,
                settlement_trace=exc.trace,
            )

        state = replace(
            settled_state,
            time_s=0.0,
            accepted_steps=0,
            rejected_steps=0,
            backplate_velocity_z_m_s=0.0,
            pin_axial_velocity_m_s=(0.0,) * self.system.pin_count,
            pin_transverse_velocity_m_s=(0.0,) * self.system.pin_count,
            cumulative_preload_work_j=0.0,
            cumulative_drive_work_j=0.0,
            cumulative_friction_dissipation_j=0.0,
            cumulative_structural_damping_dissipation_j=0.0,
            cumulative_backplate_damping_dissipation_j=0.0,
            cumulative_implicit_euler_dissipation_j=0.0,
            cumulative_normal_contact_work_j=0.0,
            cumulative_tangential_contact_work_j=0.0,
            cumulative_contact_energy_injection_j=0.0,
            cumulative_energy_residual_j=0.0,
            cumulative_energy_reference_j=0.0,
            maximum_abs_energy_residual_j=0.0,
            maximum_relative_energy_residual=0.0,
            maximum_abs_contact_work_identity_residual_j=0.0,
        )
        initial_pin_responses: list[PinDynamicResponse] = []
        initial_events: list[tuple[int, str]] = []
        for index, pin in enumerate(settled_point.pin_responses):
            active = pin.normal_force_n > 0.0
            labels = (EventLabel.FIRST_CONTACT.value,) if active else ()
            if active:
                initial_events.append((index, EventLabel.FIRST_CONTACT.value))
            initial_pin_responses.append(
                replace(
                    pin,
                    center_velocity_xyz_m_s=(0.0, 0.0, 0.0),
                    center_acceleration_xyz_m_s2=(0.0, 0.0, 0.0),
                    contact_state=(
                        ContactState.FIRST_CONTACT_EVENT
                        if active
                        else ContactState.FREE
                    ),
                    event_label=(
                        EventLabel.FIRST_CONTACT if active else EventLabel.NONE
                    ),
                    event_labels=labels,
                    normal_impulse_n_s=0.0,
                    tangential_impulse_n_s=0.0,
                )
            )
        initial_point = replace(
            settled_point,
            time_s=0.0,
            path_position_m=0.0,
            backplate_velocity_xyz_m_s=(0.0, 0.0, 0.0),
            backplate_acceleration_xyz_m_s2=(0.0, 0.0, 0.0),
            pin_responses=tuple(initial_pin_responses),
            preload_work_increment_j=0.0,
            drive_work_increment_j=0.0,
            cumulative_preload_work_j=0.0,
            cumulative_drive_work_j=0.0,
            friction_dissipation_increment_j=0.0,
            cumulative_friction_dissipation_j=0.0,
            structural_damping_dissipation_increment_j=0.0,
            cumulative_structural_damping_dissipation_j=0.0,
            backplate_damping_dissipation_increment_j=0.0,
            cumulative_backplate_damping_dissipation_j=0.0,
            implicit_euler_dissipation_increment_j=0.0,
            cumulative_implicit_euler_dissipation_j=0.0,
            normal_contact_work_increment_j=0.0,
            cumulative_normal_contact_work_j=0.0,
            tangential_contact_work_increment_j=0.0,
            cumulative_tangential_contact_work_j=0.0,
            generalized_contact_work_increment_j=0.0,
            contact_work_identity_residual_j=0.0,
            contact_energy_injection_increment_j=0.0,
            cumulative_contact_energy_injection_j=0.0,
            dynamic_residual_n=abs(settled_point.dynamic_residual_n),
            energy_residual_j=0.0,
            relative_energy_residual=0.0,
            cumulative_energy_residual_j=0.0,
            cumulative_energy_reference_j=0.0,
            cumulative_relative_energy_error=0.0,
            running_maximum_abs_energy_residual_j=0.0,
            running_maximum_relative_energy_residual=0.0,
            running_maximum_abs_contact_work_identity_residual_j=0.0,
            actual_time_step_s=self.integrator.time_step_s,
            event_labels=tuple(initial_events),
        )
        points: list[ArrayDynamicPathPoint] = [initial_point]

        total_time = self.settings.drag_length_m / self.settings.drag_speed_m_s
        output_time = self.settings.output_spacing_m / self.settings.drag_speed_m_s
        next_output = output_time
        dt_nominal = self.integrator.time_step_s
        steps = int(math.ceil(total_time / dt_nominal))
        if steps > self.integrator.maximum_steps:
            return self._failed_result(
                reason="drag maximum_steps would be exceeded",
                failure_category="numerical_failure",
                failure_code="drag_maximum_steps_exceeded",
                settlement_trace=settlement_trace,
                initial_preload_success=True,
            )
        terminal = PathTerminalState.PATH_END
        reason = "path_end"
        rejected_steps = 0
        for _ in range(steps):
            dt = min(dt_nominal, total_time - state.time_s)
            if dt <= 1e-15:
                break
            next_time = state.time_s + dt
            common_ux = (
                self.settings.initial_common_ux_m
                + self.settings.drag_speed_m_s * next_time
            )
            proposal = self.system.propose_step(
                state,
                self.settings,
                common_ux_m=common_ux,
                drag_speed_m_s=self.settings.drag_speed_m_s,
                dt=dt,
            )
            if not proposal.proposal_valid or proposal.point is None:
                rejected_steps += 1
                unchanged = self.system.commit_step(state, proposal, accept=False)
                assert unchanged == state
                reason = proposal.rejection_reason
                if reason.startswith("terrain_bounds:"):
                    terminal = PathTerminalState.TERRAIN_BOUNDS
                elif reason.startswith("structural_boundary:"):
                    terminal = PathTerminalState.STRUCTURAL_BOUNDARY
                else:
                    terminal = PathTerminalState.NUMERICAL_FAILURE
                break
            state = self.system.commit_step(state, proposal, accept=True)
            path_position = self.settings.drag_speed_m_s * state.time_s
            point = replace(proposal.point, path_position_m=path_position)
            if (
                state.time_s + 1e-12 >= next_output
                or point.event_labels
                or state.time_s + 1e-12 >= total_time
            ):
                points.append(point)
                while next_output <= state.time_s + 1e-12:
                    next_output += output_time

        state = replace(state, rejected_steps=rejected_steps)
        summary = self._summarize(
            points,
            settlement_trace=settlement_trace,
            terminal=terminal,
            reason=reason,
            accepted_steps=state.accepted_steps,
            rejected_steps=rejected_steps,
        )
        return ArrayDynamicExperimentResult(
            configuration=self.system.configuration,
            terrain_recipe_id=self.system.terrain_recipe_id,
            region_id=self.system.region_id,
            track_ids=tuple(track.track_id for track in self.system.tracks),
            experiment=self.settings,
            contact=self.system.contact,
            integrator=self.integrator,
            settlement_trace=settlement_trace,
            points=tuple(points),
            summary=summary,
            assumptions=(
                M3_MODEL_LEVEL,
                "continuous_total_external_preload_not_per_pin_preload",
                "joint_backplate_and_all_pin_time_integration",
                "prescribed_common_horizontal_speed",
                "common_backplate_z_only_pitch_and_roll_locked",
                "rigid_moreau_coupled_array_contact_impulses",
                "atomic_array_proposal_commit",
                "no_penetration_damage_or_wear",
            ),
        )

    def _failed_result(
        self,
        *,
        reason: str,
        failure_category: str,
        failure_code: str,
        settlement_trace: Sequence[SettlementTracePoint] = (),
        initial_preload_success: bool = False,
    ) -> ArrayDynamicExperimentResult:
        terminal_by_category = {
            "geometry_out_of_bounds": PathTerminalState.TERRAIN_BOUNDS,
            "physical_boundary": PathTerminalState.STRUCTURAL_BOUNDARY,
            "parameter_unclosed": PathTerminalState.MODEL_UNCLOSED,
            "settlement_nonconvergence": (
                PathTerminalState.INITIAL_PRELOAD_INFEASIBLE
            ),
            "numerical_failure": PathTerminalState.NUMERICAL_FAILURE,
        }
        terminal = terminal_by_category.get(
            failure_category,
            PathTerminalState.NUMERICAL_FAILURE,
        )
        model_state = self.system._model_state(self.settings)
        final_settlement = settlement_trace[-1] if settlement_trace else None
        summary = ArrayDynamicPathSummary(
            preload_mode="continuous_total_external_force",
            external_total_preload_n=self.settings.external_total_preload_n,
            drag_speed_m_s=self.settings.drag_speed_m_s,
            backplate_rotational_dofs=self.settings.backplate_rotational_dofs,
            initial_preload_success=initial_preload_success,
            conditional_performance_available=False,
            failure_category=failure_category,
            failure_code=failure_code,
            initialization_failure_category=(
                None if initial_preload_success else failure_category
            ),
            initialization_failure_code=(
                None if initial_preload_success else failure_code
            ),
            settlement_ramp_profile=self.settings.preload_ramp_profile,
            settlement_ramp_time_s=self.settings.preload_ramp_time_s,
            settlement_damping_scale=(
                self.settings.settlement_damping_scale
            ),
            settlement_steps=len(settlement_trace),
            settlement_stable_steps=(
                final_settlement.stable_steps if final_settlement else 0
            ),
            settlement_required_stable_steps=(
                self.settings.settling_stable_steps
            ),
            settlement_actual_approach_m=(
                final_settlement.actual_approach_m
                if final_settlement
                else None
            ),
            settlement_maximum_approach_m=(
                self.settings.maximum_preload_approach_m
            ),
            settlement_final_applied_preload_n=(
                final_settlement.applied_total_preload_n
                if final_settlement
                else None
            ),
            settlement_final_reaction_error_n=(
                final_settlement.contact_reaction_error_n
                if final_settlement
                else None
            ),
            settlement_final_maximum_mode_speed_m_s=(
                final_settlement.maximum_mode_speed_m_s
                if final_settlement
                else None
            ),
            settlement_final_dynamic_residual_n=(
                final_settlement.dynamic_residual_n
                if final_settlement
                else None
            ),
            total_contact_reaction_time_mean_n=None,
            steady_normal_balance_error_n=None,
            contact_fraction=0.0,
            effective_load_fraction=0.0,
            tangential_force_peak_n=None,
            tangential_force_steady_peak_n=None,
            tangential_force_impact_peak_n=None,
            tangential_force_median_n=None,
            tangential_force_p10_n=None,
            tangential_force_p25_n=None,
            total_normal_force_range_n=None,
            backplate_z_range_m=None,
            backplate_speed_peak_m_s=None,
            backplate_acceleration_peak_m_s2=None,
            impact_velocity_peak_m_s=None,
            neff_normal_median=None,
            neff_target_tangential_median=None,
            neff_resultant_median=None,
            maximum_normal_load_concentration=None,
            maximum_gini_normal=None,
            maximum_pin_normal_force_n=None,
            mean_pin_normal_force_n=None,
            mean_active_pin_normal_force_n=None,
            maximum_bending_stress_pa=None,
            minimum_yield_margin_pa=None,
            minimum_euler_buckling_margin_n=None,
            minimum_spring_travel_margin_m=None,
            yield_violation_pin_step_count=0,
            buckling_violation_pin_step_count=0,
            hard_stop_pin_step_count=0,
            event_counts={label.value: 0 for label in EventLabel},
            maximum_abs_dynamic_residual_n=None,
            maximum_abs_energy_residual_j=None,
            maximum_relative_energy_residual=None,
            cumulative_energy_residual_j=None,
            cumulative_energy_reference_j=None,
            cumulative_relative_energy_error=None,
            cumulative_implicit_euler_dissipation_j=None,
            cumulative_normal_contact_work_j=None,
            cumulative_tangential_contact_work_j=None,
            cumulative_contact_energy_injection_j=None,
            maximum_abs_contact_work_identity_residual_j=None,
            maximum_force_aggregation_residual_n=None,
            maximum_moment_aggregation_residual_nm=None,
            minimum_actual_time_step_s=None,
            maximum_actual_time_step_s=None,
            accepted_steps=0,
            rejected_steps=0,
            time_step_convergence_checked=self.settings.time_step_convergence_checked,
            contact_parameter_convergence_checked=(
                self.settings.contact_parameter_convergence_checked
            ),
            settlement_damping_convergence_checked=(
                self.settings.settlement_damping_convergence_checked
            ),
            terrain_resolution_convergence_checked=(
                self.settings.terrain_resolution_convergence_checked
            ),
            physical_calibration_completed=(
                self.settings.physical_calibration_completed
            ),
            unclosed_parameter_names=self.system._unclosed_parameter_names(
                self.settings
            ),
            numerical_state=NumericalState.NONCONVERGED,
            model_state=model_state,
            run_terminal_state=terminal,
            termination_reason=reason,
            formal_ranking_eligible=False,
        )
        return ArrayDynamicExperimentResult(
            configuration=self.system.configuration,
            terrain_recipe_id=self.system.terrain_recipe_id,
            region_id=self.system.region_id,
            track_ids=tuple(track.track_id for track in self.system.tracks),
            experiment=self.settings,
            contact=self.system.contact,
            integrator=self.integrator,
            settlement_trace=tuple(settlement_trace),
            points=(),
            summary=summary,
            assumptions=(M3_MODEL_LEVEL,),
        )

    def _summarize(
        self,
        points: Sequence[ArrayDynamicPathPoint],
        *,
        settlement_trace: Sequence[SettlementTracePoint],
        terminal: PathTerminalState,
        reason: str,
        accepted_steps: int,
        rejected_steps: int,
    ) -> ArrayDynamicPathSummary:
        total_normal = np.asarray(
            [point.total_contact_reaction_z_n for point in points],
            dtype=np.float64,
        )
        pull = np.abs(
            np.asarray(
                [point.wall_on_unit_wrench_about_origin[0] for point in points],
                dtype=np.float64,
            )
        )
        impact_mask = np.asarray(
            [
                any(
                    label == EventLabel.IMPACT.value
                    for _index, label in point.event_labels
                )
                for point in points
            ],
            dtype=np.bool_,
        )
        time_values = np.asarray(
            [point.time_s for point in points], dtype=np.float64
        )
        steady_start = 0.2 * float(time_values[-1])
        event_mask = np.asarray(
            [bool(point.event_labels) for point in points],
            dtype=np.bool_,
        )
        steady_mask = (
            ~impact_mask
            & ~event_mask
            & (time_values + 1e-15 >= steady_start)
        )
        if not np.any(steady_mask):
            steady_mask = ~impact_mask
        steady_pull = pull[steady_mask]
        steady_normal = total_normal[steady_mask]
        steady_time = time_values[steady_mask]
        if steady_time.size > 1 and steady_time[-1] > steady_time[0]:
            steady_normal_mean = float(
                np.sum(
                    0.5
                    * (steady_normal[:-1] + steady_normal[1:])
                    * np.diff(steady_time)
                )
                / (steady_time[-1] - steady_time[0])
            )
        else:
            steady_normal_mean = float(np.mean(steady_normal))
        event_counts = {label.value: 0 for label in EventLabel}
        for point in points:
            for _index, label in point.event_labels:
                event_counts[label] = event_counts.get(label, 0) + 1
        active = np.asarray(
            [point.active_pin_count > 0 for point in points], dtype=np.bool_
        )
        effective = np.asarray(
            [point.effective_load_pin_count > 0 for point in points],
            dtype=np.bool_,
        )
        model_state = (
            ModelState.PARAMETER_UNCLOSED
            if any(
                point.model_state is ModelState.PARAMETER_UNCLOSED
                for point in points
            )
            else self.system._model_state(self.settings)
        )
        numerical_state = (
            NumericalState.CONVERGED
            if terminal is PathTerminalState.PATH_END
            else NumericalState.NONCONVERGED
        )
        formal_eligible = (
            terminal is PathTerminalState.PATH_END
            and numerical_state is NumericalState.CONVERGED
            and model_state is ModelState.COVERED
            and not self.system._unclosed_parameter_names(self.settings)
            and self.settings.time_step_convergence_checked
            and self.settings.contact_parameter_convergence_checked
            and self.settings.settlement_damping_convergence_checked
            and self.settings.terrain_resolution_convergence_checked
            and self.settings.physical_calibration_completed
        )
        dt_values = np.asarray(
            [point.actual_time_step_s for point in points], dtype=np.float64
        )
        final_settlement = settlement_trace[-1]
        pin_samples = [
            (pin_index, pin)
            for point in points
            for pin_index, pin in enumerate(point.pin_responses)
        ]
        maximum_bending_stress = max(
            pin.bending_stress_pa for _index, pin in pin_samples
        )
        yield_margins = [
            float(self.system.pin_parameters[index].yield_strength_pa)
            - pin.bending_stress_pa
            for index, pin in pin_samples
            if self.system.pin_parameters[index].yield_strength_pa is not None
        ]
        yield_violations = sum(
            self.system.pin_parameters[index].yield_strength_pa is not None
            and pin.bending_stress_pa
            > float(self.system.pin_parameters[index].yield_strength_pa)
            for index, pin in pin_samples
        )
        buckling_violations = sum(
            pin.euler_buckling_margin_n < 0.0
            for _index, pin in pin_samples
        )
        hard_stop_samples = sum(
            pin.spring_state is SpringState.HARD_STOP
            for _index, pin in pin_samples
        )
        pin_normal_forces = [
            pin.normal_force_n for _index, pin in pin_samples
        ]
        positive_pin_normal_forces = [
            value for value in pin_normal_forces if value > 0.0
        ]
        return ArrayDynamicPathSummary(
            preload_mode="continuous_total_external_force",
            external_total_preload_n=self.settings.external_total_preload_n,
            drag_speed_m_s=self.settings.drag_speed_m_s,
            backplate_rotational_dofs=self.settings.backplate_rotational_dofs,
            initial_preload_success=True,
            conditional_performance_available=True,
            failure_category=(
                None
                if terminal is PathTerminalState.PATH_END
                else (
                    "geometry_out_of_bounds"
                    if terminal is PathTerminalState.TERRAIN_BOUNDS
                    else (
                        "physical_boundary"
                        if terminal is PathTerminalState.STRUCTURAL_BOUNDARY
                        else "numerical_failure"
                    )
                )
            ),
            failure_code=(
                None
                if terminal is PathTerminalState.PATH_END
                else terminal.value
            ),
            initialization_failure_category=None,
            initialization_failure_code=None,
            settlement_ramp_profile=self.settings.preload_ramp_profile,
            settlement_ramp_time_s=self.settings.preload_ramp_time_s,
            settlement_damping_scale=(
                self.settings.settlement_damping_scale
            ),
            settlement_steps=len(settlement_trace),
            settlement_stable_steps=final_settlement.stable_steps,
            settlement_required_stable_steps=(
                self.settings.settling_stable_steps
            ),
            settlement_actual_approach_m=(
                final_settlement.actual_approach_m
            ),
            settlement_maximum_approach_m=(
                self.settings.maximum_preload_approach_m
            ),
            settlement_final_applied_preload_n=(
                final_settlement.applied_total_preload_n
            ),
            settlement_final_reaction_error_n=(
                final_settlement.contact_reaction_error_n
            ),
            settlement_final_maximum_mode_speed_m_s=(
                final_settlement.maximum_mode_speed_m_s
            ),
            settlement_final_dynamic_residual_n=(
                final_settlement.dynamic_residual_n
            ),
            total_contact_reaction_time_mean_n=steady_normal_mean,
            steady_normal_balance_error_n=float(
                abs(steady_normal_mean - self.settings.external_total_preload_n)
            ),
            contact_fraction=float(np.mean(active)),
            effective_load_fraction=float(np.mean(effective)),
            tangential_force_peak_n=float(np.max(pull)),
            tangential_force_steady_peak_n=(
                float(np.max(steady_pull)) if steady_pull.size else 0.0
            ),
            tangential_force_impact_peak_n=(
                float(np.max(pull[impact_mask])) if np.any(impact_mask) else 0.0
            ),
            tangential_force_median_n=float(np.median(steady_pull)),
            tangential_force_p10_n=float(np.quantile(steady_pull, 0.10)),
            tangential_force_p25_n=float(np.quantile(steady_pull, 0.25)),
            total_normal_force_range_n=(
                float(np.min(total_normal)),
                float(np.max(total_normal)),
            ),
            backplate_z_range_m=(
                float(min(point.backplate_position_xyz_m[2] for point in points)),
                float(max(point.backplate_position_xyz_m[2] for point in points)),
            ),
            backplate_speed_peak_m_s=float(
                max(abs(point.backplate_velocity_xyz_m_s[2]) for point in points)
            ),
            backplate_acceleration_peak_m_s2=float(
                max(
                    abs(point.backplate_acceleration_xyz_m_s2[2])
                    for point in points
                )
            ),
            impact_velocity_peak_m_s=float(
                max(
                    pin.impact_velocity_m_s
                    for point in points
                    for pin in point.pin_responses
                )
            ),
            neff_normal_median=float(
                np.median([point.sharing.neff_normal for point in points])
            ),
            neff_target_tangential_median=float(
                np.median(
                    [
                        point.sharing.neff_target_tangential
                        for point in points
                    ]
                )
            ),
            neff_resultant_median=float(
                np.median([point.sharing.neff_resultant for point in points])
            ),
            maximum_normal_load_concentration=float(
                max(point.sharing.max_mean_normal for point in points)
            ),
            maximum_gini_normal=float(
                max(point.sharing.gini_normal for point in points)
            ),
            maximum_pin_normal_force_n=float(max(pin_normal_forces)),
            mean_pin_normal_force_n=float(np.mean(pin_normal_forces)),
            mean_active_pin_normal_force_n=(
                float(np.mean(positive_pin_normal_forces))
                if positive_pin_normal_forces
                else 0.0
            ),
            maximum_bending_stress_pa=float(maximum_bending_stress),
            minimum_yield_margin_pa=(
                float(min(yield_margins)) if yield_margins else None
            ),
            minimum_euler_buckling_margin_n=float(
                min(
                    pin.euler_buckling_margin_n
                    for _index, pin in pin_samples
                )
            ),
            minimum_spring_travel_margin_m=float(
                min(
                    pin.spring_travel_margin_m
                    for _index, pin in pin_samples
                )
            ),
            yield_violation_pin_step_count=int(yield_violations),
            buckling_violation_pin_step_count=int(buckling_violations),
            hard_stop_pin_step_count=int(hard_stop_samples),
            event_counts=event_counts,
            maximum_abs_dynamic_residual_n=float(
                max(abs(point.dynamic_residual_n) for point in points)
            ),
            maximum_abs_energy_residual_j=float(
                points[-1].running_maximum_abs_energy_residual_j
            ),
            maximum_relative_energy_residual=float(
                points[-1].running_maximum_relative_energy_residual
            ),
            cumulative_energy_residual_j=float(
                points[-1].cumulative_energy_residual_j
            ),
            cumulative_energy_reference_j=float(
                points[-1].cumulative_energy_reference_j
            ),
            cumulative_relative_energy_error=float(
                points[-1].cumulative_relative_energy_error
            ),
            cumulative_implicit_euler_dissipation_j=float(
                points[-1].cumulative_implicit_euler_dissipation_j
            ),
            cumulative_normal_contact_work_j=float(
                points[-1].cumulative_normal_contact_work_j
            ),
            cumulative_tangential_contact_work_j=float(
                points[-1].cumulative_tangential_contact_work_j
            ),
            cumulative_contact_energy_injection_j=float(
                points[-1].cumulative_contact_energy_injection_j
            ),
            maximum_abs_contact_work_identity_residual_j=float(
                points[
                    -1
                ].running_maximum_abs_contact_work_identity_residual_j
            ),
            maximum_force_aggregation_residual_n=float(
                max(point.force_aggregation_residual_n for point in points)
            ),
            maximum_moment_aggregation_residual_nm=float(
                max(point.moment_aggregation_residual_nm for point in points)
            ),
            minimum_actual_time_step_s=float(np.min(dt_values)),
            maximum_actual_time_step_s=float(np.max(dt_values)),
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            time_step_convergence_checked=self.settings.time_step_convergence_checked,
            contact_parameter_convergence_checked=(
                self.settings.contact_parameter_convergence_checked
            ),
            settlement_damping_convergence_checked=(
                self.settings.settlement_damping_convergence_checked
            ),
            terrain_resolution_convergence_checked=(
                self.settings.terrain_resolution_convergence_checked
            ),
            physical_calibration_completed=(
                self.settings.physical_calibration_completed
            ),
            unclosed_parameter_names=self.system._unclosed_parameter_names(
                self.settings
            ),
            numerical_state=numerical_state,
            model_state=model_state,
            run_terminal_state=terminal,
            termination_reason=reason,
            formal_ranking_eligible=formal_eligible,
        )
