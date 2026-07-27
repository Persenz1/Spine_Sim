"""CPU-authoritative prescribed-pose solver for a rigid common backplate."""

from __future__ import annotations

import math
from typing import Sequence, cast

import numpy as np

from spine_sim.contact import (
    ConstitutiveResponse,
    ContactState,
    EventLabel,
    PrescribedPoseConstitutiveCore,
    SolverSettings,
)
from spine_sim.core.states import ModelState, NumericalState
from spine_sim.terrain import TrackGeometry

from .models import (
    ActivitySets,
    ArrayConfiguration,
    ArrayPoseResponse,
    ArrayResidualAudit,
    ArrayState,
    LoadSharingMetrics,
)


_ACTIVE_CONTACT_STATES = {
    ContactState.FIRST_CONTACT_EVENT,
    ContactState.RECONTACT_EVENT,
    ContactState.STICK,
    ContactState.SLIDE,
}
_FRICTION_STATES = {ContactState.STICK, ContactState.SLIDE}


def _neff(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    square_sum = float(weights @ weights)
    return total * total / square_sum if total > 0.0 and square_sum > 0.0 else 0.0


def _max_mean(weights: np.ndarray) -> float:
    mean = float(np.mean(weights))
    return float(np.max(weights)) / mean if mean > 0.0 else 0.0


def _gini(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    ordered = np.sort(weights)
    count = ordered.size
    coefficients = 2.0 * np.arange(1, count + 1) - count - 1.0
    return float(coefficients @ ordered / (count * total))


class CommonBackplateArray:
    """Pure M3 array solve; every pin proposal sees the same old ArrayState."""

    def __init__(
        self,
        configuration: ArrayConfiguration,
        tracks: Sequence[TrackGeometry],
        *,
        unit_origin_xy_m: tuple[float, float],
        solver_settings: SolverSettings | None = None,
        target_load_threshold_n: float = 1e-6,
    ) -> None:
        self.configuration = configuration
        self.tracks = tuple(tracks)
        self.unit_origin_xy_m = (
            float(unit_origin_xy_m[0]),
            float(unit_origin_xy_m[1]),
        )
        self.solver_settings = solver_settings or SolverSettings()
        if (
            not math.isfinite(target_load_threshold_n)
            or target_load_threshold_n < 0.0
        ):
            raise ValueError("target_load_threshold_n must be finite and nonnegative")
        self.target_load_threshold_n = float(target_load_threshold_n)
        if len(self.tracks) != configuration.pin_count:
            raise ValueError("one TrackGeometry entry is required for every pin")
        recipe_ids = {track.terrain_recipe_id for track in self.tracks}
        region_ids = {track.region_id for track in self.tracks}
        if len(recipe_ids) != 1 or len(region_ids) != 1:
            raise ValueError("all pins must share one terrain recipe and one region")
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
        actual_y = np.asarray([track.y_global_m for track in self.tracks])
        if not np.allclose(expected_y, actual_y, rtol=0.0, atol=1e-12):
            raise ValueError(
                "each pin track y_global_m must match its global holder y coordinate"
            )
        for parameters, track in zip(self.pin_parameters, self.tracks):
            if not math.isclose(
                parameters.tip_radius_m,
                track.radius_m,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("every pin tip radius must match its M1 track")
        self.cores = tuple(
            PrescribedPoseConstitutiveCore(parameters, track, self.solver_settings)
            for parameters, track in zip(self.pin_parameters, self.tracks)
        )

    @property
    def empty_state(self) -> ArrayState:
        return ArrayState.empty(self.configuration.pin_count)

    def pin_holder_xz_m(
        self, pin_index: int, common_ux_m: float, common_uz_m: float
    ) -> tuple[float, float]:
        offset = self.holder_offsets_xyz_m[pin_index]
        return (
            self.unit_origin_xy_m[0] + offset[0] + common_ux_m,
            common_uz_m,
        )

    def solve_pose(
        self,
        common_pose_m: tuple[float, float],
        old_state: ArrayState | None = None,
        *,
        commit: bool = False,
        traversal_order: Sequence[int] | None = None,
    ) -> ArrayPoseResponse:
        """Compute all proposals first and only then expose an atomic array commit."""

        common_ux_m, common_uz_m = map(float, common_pose_m)
        if not math.isfinite(common_ux_m) or not math.isfinite(common_uz_m):
            raise ValueError("common pose must be finite")
        old = old_state or self.empty_state
        if len(old.pin_states) != self.configuration.pin_count:
            raise ValueError("ArrayState pin count does not match the configuration")
        if traversal_order is None:
            order = tuple(range(self.configuration.pin_count))
        else:
            order = tuple(int(index) for index in traversal_order)
            if sorted(order) != list(range(self.configuration.pin_count)):
                raise ValueError("traversal_order must be a permutation of pin indices")

        responses: list[ConstitutiveResponse | None] = [
            None
        ] * self.configuration.pin_count
        for index in order:
            responses[index] = self.cores[index].solve_pose(
                self.pin_holder_xz_m(index, common_ux_m, common_uz_m),
                old.pin_states[index],
                commit=False,
            )
        assert all(response is not None for response in responses)
        pin_responses = tuple(
            cast(ConstitutiveResponse, response) for response in responses
        )

        invalid_reasons: list[str] = []
        for index, response in enumerate(pin_responses):
            if not response.proposal_valid:
                invalid_reasons.append(
                    f"pin_{index}:{response.residual.termination_reason}"
                )
            elif response.numerical_state is not NumericalState.CONVERGED:
                invalid_reasons.append(
                    f"pin_{index}:numerical_{response.numerical_state.value}"
                )
            elif response.model_state is ModelState.OUT_OF_SCOPE:
                invalid_reasons.append(f"pin_{index}:model_out_of_scope")
            elif (
                response.contact_state in _ACTIVE_CONTACT_STATES
                and not response.cap_gate_passed
            ):
                invalid_reasons.append(f"pin_{index}:cap_gate_rejected_active_contact")
            elif response.normal_force_n < -self.solver_settings.force_tolerance_n:
                invalid_reasons.append(f"pin_{index}:negative_normal_force")
            elif (
                response.contact_state in _FRICTION_STATES
                and response.static_friction_margin_n
                < -self.solver_settings.friction_residual_tolerance_n
            ):
                invalid_reasons.append(f"pin_{index}:friction_margin")
            elif response.spring_travel_margin_m < -self.solver_settings.gap_tolerance_m:
                invalid_reasons.append(f"pin_{index}:spring_travel")

        pin_holder_xyz: list[tuple[float, float, float]] = []
        pin_wrench_unit: list[tuple[float, float, float, float, float, float]] = []
        origin = np.array(
            [
                self.unit_origin_xy_m[0] + common_ux_m,
                self.unit_origin_xy_m[1],
                common_uz_m,
            ],
            dtype=np.float64,
        )
        for index, (offset, response) in enumerate(
            zip(self.holder_offsets_xyz_m, pin_responses)
        ):
            holder = np.array(
                [
                    self.unit_origin_xy_m[0] + offset[0] + common_ux_m,
                    self.unit_origin_xy_m[1] + offset[1],
                    common_uz_m,
                ],
                dtype=np.float64,
            )
            pin_holder_xyz.append(tuple(float(value) for value in holder))
            wrench = np.asarray(
                response.spine_on_plate_wrench_about_holder,
                dtype=np.float64,
            )
            force = wrench[:3]
            moment = wrench[3:] + np.cross(holder - origin, force)
            shifted = np.concatenate((force, moment))
            pin_wrench_unit.append(tuple(float(value) for value in shifted))

        pin_wrench_array = np.asarray(pin_wrench_unit, dtype=np.float64)
        unit_wrench = np.sum(pin_wrench_array, axis=0)
        force_residual = float(
            np.linalg.norm(unit_wrench[:3] - np.sum(pin_wrench_array[:, :3], axis=0))
        )
        moment_residual = float(
            np.linalg.norm(unit_wrench[3:] - np.sum(pin_wrench_array[:, 3:], axis=0))
        )

        normal_weights = np.asarray(
            [max(0.0, response.normal_force_n) for response in pin_responses],
            dtype=np.float64,
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
        nominal = tuple(range(self.configuration.pin_count))
        geometric = tuple(
            index
            for index, response in enumerate(pin_responses)
            if response.support_xyz_m is not None
            and response.cap_gate_passed
            and (
                response.gap_m <= self.solver_settings.gap_tolerance_m
                or response.contact_state in _ACTIVE_CONTACT_STATES
            )
        )
        positive = tuple(
            index
            for index, response in enumerate(pin_responses)
            if response.normal_force_n > self.solver_settings.force_tolerance_n
        )
        admissible = tuple(
            index
            for index, response in enumerate(pin_responses)
            if response.proposal_valid
            and response.numerical_state is NumericalState.CONVERGED
            and response.model_state is not ModelState.OUT_OF_SCOPE
            and (
                response.contact_state not in _FRICTION_STATES
                or (
                    response.cap_gate_passed
                    and response.static_friction_margin_n
                    >= -self.solver_settings.friction_residual_tolerance_n
                )
            )
        )
        target_load = tuple(
            int(index)
            for index in np.flatnonzero(
                tangential_weights > self.target_load_threshold_n
            )
        )
        activity = ActivitySets(
            nominal=nominal,
            geometric=geometric,
            positive_normal=positive,
            admissible=admissible,
            target_load=target_load,
        )
        events = tuple(
            (index, response.event_label.value)
            for index, response in enumerate(pin_responses)
            if response.event_label is not EventLabel.NONE
        )

        numerical_state = NumericalState.CONVERGED
        if any(
            response.numerical_state is NumericalState.NONCONVERGED
            for response in pin_responses
        ):
            numerical_state = NumericalState.NONCONVERGED
        elif any(
            response.numerical_state is not NumericalState.CONVERGED
            for response in pin_responses
        ):
            numerical_state = NumericalState.NOT_RUN
        model_state = ModelState.COVERED
        if any(
            response.model_state is ModelState.OUT_OF_SCOPE
            for response in pin_responses
        ):
            model_state = ModelState.OUT_OF_SCOPE
        elif any(
            response.model_state is ModelState.PARAMETER_UNCLOSED
            for response in pin_responses
        ):
            model_state = ModelState.PARAMETER_UNCLOSED

        proposal_valid = not invalid_reasons
        proposal = (
            ArrayState(
                tuple(response.proposal_state for response in pin_responses),
                accepted_steps=old.accepted_steps + 1,
            )
            if proposal_valid
            else old
        )
        local_geometry = [
            abs(response.residual.geometry_m)
            for response in pin_responses
            if math.isfinite(response.residual.geometry_m)
            and response.contact_state in _ACTIVE_CONTACT_STATES
        ]
        local_structure = [
            abs(response.residual.structure_m)
            for response in pin_responses
            if math.isfinite(response.residual.structure_m)
        ]
        local_force = [
            abs(response.residual.force_decomposition_n)
            for response in pin_responses
            if math.isfinite(response.residual.force_decomposition_n)
        ]
        residual = ArrayResidualAudit(
            force_aggregation_n=force_residual,
            moment_aggregation_nm=moment_residual,
            maximum_local_geometry_m=max(local_geometry, default=0.0),
            maximum_local_structure_m=max(local_structure, default=0.0),
            maximum_local_force_decomposition_n=max(local_force, default=0.0),
            termination_reason=(
                "converged" if proposal_valid else ";".join(invalid_reasons)
            ),
        )
        zero_wrench = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return ArrayPoseResponse(
            common_ux_m=common_ux_m,
            common_uz_m=common_uz_m,
            unit_origin_xyz_m=tuple(float(value) for value in origin),
            pin_holder_xyz_m=tuple(pin_holder_xyz),
            pin_responses=pin_responses,
            pin_wrench_about_unit=tuple(pin_wrench_unit),
            wall_on_unit_wrench_about_origin=tuple(
                float(value) for value in unit_wrench
            ),
            active_thrust_wrench_about_origin=zero_wrench,
            guide_reaction_wrench_about_origin=zero_wrench,
            activity_sets=activity,
            sharing=sharing,
            residual=residual,
            numerical_state=numerical_state,
            model_state=model_state,
            event_labels=events,
            proposal_state=proposal,
            next_state=proposal if commit and proposal_valid else old,
            proposal_valid=proposal_valid,
        )
