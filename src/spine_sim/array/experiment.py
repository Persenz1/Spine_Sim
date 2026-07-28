"""Legacy initial-preload plus fixed-common-Z array migration fixture."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spine_sim.contact import ContactState, EventLabel, PathTerminalState
from spine_sim.core.states import ModelState, NumericalState

from .models import (
    LegacyArrayExperimentResult,
    LegacyArrayPathPoint,
    LegacyArrayPathSummary,
    LegacyArrayPoseResponse,
    LegacyArrayState,
)
from .solver import LegacyCommonBackplateArray


@dataclass(frozen=True)
class LegacyArrayExperimentSettings:
    drag_length_m: float
    path_step_m: float
    target_preload_n: float = 1.0
    preload_force_tolerance_n: float = 1e-4
    maximum_preload_approach_m: float = 8e-3
    effective_unit_tangential_force_min_n: float = 0.05
    free_probe_spacing_m: float | None = None
    refine_events: bool = True

    def __post_init__(self) -> None:
        for name in (
            "drag_length_m",
            "path_step_m",
            "target_preload_n",
            "preload_force_tolerance_n",
            "maximum_preload_approach_m",
            "effective_unit_tangential_force_min_n",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.drag_length_m <= 0.0 or self.path_step_m <= 0.0:
            raise ValueError("drag length and path step must be positive")
        if (
            self.target_preload_n <= 0.0
            or self.preload_force_tolerance_n <= 0.0
            or self.maximum_preload_approach_m <= 0.0
        ):
            raise ValueError("preload settings must be positive")
        if self.effective_unit_tangential_force_min_n < 0.0:
            raise ValueError("effective force threshold cannot be negative")
        if self.free_probe_spacing_m is not None and self.free_probe_spacing_m <= 0:
            raise ValueError("free_probe_spacing_m must be positive")


class LegacyFixedZCommonBackplateExperiment:
    def __init__(
        self,
        system: LegacyCommonBackplateArray,
        settings: LegacyArrayExperimentSettings,
    ) -> None:
        self.system = system
        self.settings = settings

    def run(self) -> LegacyArrayExperimentResult:
        first_free = self._initial_free_pose()
        if not first_free.proposal_valid:
            return self._failed_result(
                PathTerminalState.MODEL_UNCLOSED,
                first_free.residual.termination_reason,
                points=(),
            )
        preload = self._establish_preload(first_free.next_state, first_free.common_uz_m)
        first_point = LegacyArrayPathPoint(0.0, first_free, event_refined=True)
        if preload is None:
            return self._failed_result(
                PathTerminalState.INITIAL_PRELOAD_INFEASIBLE,
                "initial_total_preload_infeasible",
                points=(first_point,),
            )
        points: list[LegacyArrayPathPoint] = [
            first_point,
            LegacyArrayPathPoint(0.0, preload, event_refined=True),
        ]
        state = preload.next_state
        fixed_uz = preload.common_uz_m
        current_ux = 0.0
        terminal = PathTerminalState.PATH_END
        reason = "path_end"

        while current_ux < self.settings.drag_length_m - 1e-15:
            target_ux = min(
                current_ux + self.settings.path_step_m,
                self.settings.drag_length_m,
            )
            free_present = any(
                pin.contact_state in {ContactState.FREE, ContactState.DETACH_EVENT}
                for pin in state.pin_states
            )
            probe_spacing = self.settings.free_probe_spacing_m or min(
                track.resolution_m for track in self.system.tracks
            )
            subdivisions = (
                max(1, int(math.ceil((target_ux - current_ux) / probe_spacing)))
                if free_present
                else 1
            )
            probe_targets = np.linspace(
                current_ux,
                target_ux,
                subdivisions + 1,
                dtype=np.float64,
            )[1:]
            for probe_ux in probe_targets:
                segment, state = self._advance_segment(
                    state,
                    current_ux,
                    float(probe_ux),
                    fixed_uz,
                )
                for response, refined in segment:
                    points.append(
                        LegacyArrayPathPoint(
                            path_position_m=response.common_ux_m,
                            response=response,
                            event_refined=refined,
                        )
                    )
                    if not response.proposal_valid:
                        if response.numerical_state is NumericalState.NONCONVERGED:
                            terminal = PathTerminalState.NUMERICAL_FAILURE
                        elif "geometry_out_of_domain" in response.residual.termination_reason:
                            terminal = PathTerminalState.TERRAIN_BOUNDS
                        else:
                            terminal = PathTerminalState.MODEL_UNCLOSED
                        reason = response.residual.termination_reason
                        return self._result(
                            fixed_uz,
                            tuple(points),
                            terminal,
                            reason,
                        )
                current_ux = float(probe_ux)
        return self._result(fixed_uz, tuple(points), terminal, reason)

    def _initial_free_pose(self) -> LegacyArrayPoseResponse:
        contact_heights: list[float] = []
        for index, (core, parameters) in enumerate(
            zip(self.system.cores, self.system.pin_parameters)
        ):
            holder_x, _ = self.system.pin_holder_xz_m(index, 0.0, 0.0)
            unloaded_center_x = holder_x + parameters.exposed_length_m * core._a[0]
            geometry = core.geometry.query(unloaded_center_x)
            contact_heights.append(
                geometry.envelope_height_m
                - parameters.exposed_length_m * core._a[1]
            )
        free_uz = max(contact_heights) + max(
            10.0 * self.system.solver_settings.gap_tolerance_m,
            1e-8,
        )
        return self.system.solve_pose(
            (0.0, free_uz),
            self.system.empty_state,
            commit=True,
        )

    def _establish_preload(
        self,
        initial_state: LegacyArrayState,
        free_uz_m: float,
    ) -> LegacyArrayPoseResponse | None:
        target = self.settings.target_preload_n
        approaches = np.geomspace(
            max(self.system.solver_settings.gap_tolerance_m, 1e-10),
            self.settings.maximum_preload_approach_m,
            96,
        )
        lower_approach = 0.0
        upper_approach: float | None = None
        for approach in approaches:
            candidate = self.system.solve_pose(
                (0.0, free_uz_m - float(approach)),
                initial_state,
                commit=False,
            )
            if not candidate.proposal_valid:
                lower_approach = float(approach)
                continue
            if candidate.total_normal_force_n >= target:
                upper_approach = float(approach)
                break
            lower_approach = float(approach)
        if upper_approach is None:
            return None
        best: LegacyArrayPoseResponse | None = None
        for _ in range(self.system.solver_settings.root_max_iterations):
            middle = 0.5 * (lower_approach + upper_approach)
            candidate = self.system.solve_pose(
                (0.0, free_uz_m - middle),
                initial_state,
                commit=False,
            )
            if not candidate.proposal_valid:
                lower_approach = middle
                continue
            best = candidate
            error = candidate.total_normal_force_n - target
            if abs(error) <= self.system.solver_settings.force_tolerance_n:
                break
            if error < 0.0:
                lower_approach = middle
            else:
                upper_approach = middle
        if best is None:
            return None
        committed = self.system.solve_pose(
            (best.common_ux_m, best.common_uz_m),
            initial_state,
            commit=True,
        )
        if (
            not committed.proposal_valid
            or abs(committed.total_normal_force_n - target)
            > self.settings.preload_force_tolerance_n
        ):
            return None
        return committed

    def _advance_segment(
        self,
        state: LegacyArrayState,
        start_ux: float,
        end_ux: float,
        fixed_uz: float,
        *,
        depth: int = 0,
    ) -> tuple[list[tuple[LegacyArrayPoseResponse, bool]], LegacyArrayState]:
        trial = self.system.solve_pose(
            (end_ux, fixed_uz),
            state,
            commit=False,
        )
        if not trial.proposal_valid:
            return [(trial, False)], state
        events = [
            (index, response.event_label)
            for index, response in enumerate(trial.pin_responses)
            if response.event_label is not EventLabel.NONE
        ]
        if (
            not self.settings.refine_events
            or not events
            or depth >= 12
            or end_ux - start_ux <= self.system.solver_settings.event_tolerance_m
        ):
            committed = self.system.solve_pose(
                (end_ux, fixed_uz),
                state,
                commit=True,
            )
            return [(committed, False)], committed.next_state

        candidates = [
            self._locate_pin_event(
                index,
                event,
                state,
                start_ux,
                end_ux,
                fixed_uz,
            )
            for index, event in events
        ]
        event_ux = min(candidates)
        committed_event = self.system.solve_pose(
            (event_ux, fixed_uz),
            state,
            commit=True,
        )
        output: list[tuple[LegacyArrayPoseResponse, bool]] = [
            (committed_event, True)
        ]
        if end_ux - event_ux <= self.system.solver_settings.event_tolerance_m:
            return output, committed_event.next_state
        remainder, final_state = self._advance_segment(
            committed_event.next_state,
            event_ux,
            end_ux,
            fixed_uz,
            depth=depth + 1,
        )
        output.extend(remainder)
        return output, final_state

    def _locate_pin_event(
        self,
        pin_index: int,
        event: EventLabel,
        state: LegacyArrayState,
        start_ux: float,
        end_ux: float,
        fixed_uz: float,
    ) -> float:
        lower = start_ux
        upper = end_ux
        core = self.system.cores[pin_index]
        old_pin = state.pin_states[pin_index]
        for _ in range(self.system.solver_settings.event_max_iterations):
            if upper - lower <= self.system.solver_settings.event_tolerance_m:
                break
            middle = 0.5 * (lower + upper)
            trial = core.solve_pose(
                self.system.pin_holder_xz_m(pin_index, middle, fixed_uz),
                old_pin,
                commit=False,
            )
            if self._event_reached(trial, event):
                upper = middle
            else:
                lower = middle
        return upper

    @staticmethod
    def _event_reached(response, event: EventLabel) -> bool:
        active = {
            ContactState.FIRST_CONTACT_EVENT,
            ContactState.RECONTACT_EVENT,
            ContactState.STICK,
            ContactState.SLIDE,
        }
        if event is EventLabel.DETACH_TO_FREE:
            return response.contact_state in {
                ContactState.FREE,
                ContactState.DETACH_EVENT,
            }
        if event in {EventLabel.FIRST_CONTACT, EventLabel.RECONTACT}:
            return response.contact_state in active
        if event is EventLabel.SLIP_START:
            return response.contact_state is ContactState.SLIDE
        if event is EventLabel.HARD_STOP:
            return response.spring_state.value == "hard_stop"
        return False

    def _result(
        self,
        fixed_uz: float,
        points: tuple[LegacyArrayPathPoint, ...],
        terminal: PathTerminalState,
        reason: str,
    ) -> LegacyArrayExperimentResult:
        return LegacyArrayExperimentResult(
            configuration=self.system.configuration,
            terrain_recipe_id=self.system.terrain_recipe_id,
            region_id=self.system.region_id,
            track_ids=tuple(track.track_id for track in self.system.tracks),
            fixed_common_uz_m=fixed_uz,
            target_preload_n=self.settings.target_preload_n,
            points=points,
            summary=self._summarize(points, terminal, reason),
            assumptions=(
                "one total preload search followed by frozen common uZ",
                "all pins share the same rigid-backplate (ux,uZ) pose",
                "every accepted array pose is an atomic all-pin commit",
                "active thrust and guide reaction wrenches are stored separately as zero inputs",
                "quasistatic independent-pin M2 constitutive laws coupled by common geometry",
            ),
        )

    def _failed_result(
        self,
        terminal: PathTerminalState,
        reason: str,
        *,
        points: tuple[LegacyArrayPathPoint, ...],
    ) -> LegacyArrayExperimentResult:
        return LegacyArrayExperimentResult(
            configuration=self.system.configuration,
            terrain_recipe_id=self.system.terrain_recipe_id,
            region_id=self.system.region_id,
            track_ids=tuple(track.track_id for track in self.system.tracks),
            fixed_common_uz_m=None,
            target_preload_n=self.settings.target_preload_n,
            points=points,
            summary=self._summarize(points, terminal, reason),
            assumptions=("initial total preload failed before fixed-uZ drag",),
        )

    def _summarize(
        self,
        points: tuple[LegacyArrayPathPoint, ...],
        terminal: PathTerminalState,
        reason: str,
    ) -> LegacyArrayPathSummary:
        if not points:
            return LegacyArrayPathSummary(
                initial_preload_success=False,
                total_contact_length_m=0.0,
                effective_load_length_m=0.0,
                effective_load_fraction=0.0,
                maximum_continuous_load_length_m=0.0,
                tangential_force_peak_n=0.0,
                tangential_force_median_n=0.0,
                tangential_force_p10_n=0.0,
                tangential_force_p25_n=0.0,
                total_normal_force_range_n=(0.0, 0.0),
                neff_normal_median=0.0,
                neff_target_tangential_median=0.0,
                neff_resultant_median=0.0,
                maximum_pin_resultant_force_n=0.0,
                maximum_resultant_load_concentration=0.0,
                event_counts={},
                maximum_abs_local_geometry_residual_m=0.0,
                maximum_force_aggregation_residual_n=0.0,
                maximum_moment_aggregation_residual_nm=0.0,
                numerical_state=NumericalState.NOT_RUN,
                model_state=ModelState.OUT_OF_SCOPE,
                run_terminal_state=terminal,
                termination_reason=reason,
            )
        contact_length = 0.0
        effective_length = 0.0
        continuous = 0.0
        maximum_continuous = 0.0
        for previous, current in zip(points, points[1:]):
            length = max(0.0, current.path_position_m - previous.path_position_m)
            loaded = bool(previous.response.activity_sets.positive_normal)
            tangential = abs(
                previous.response.wall_on_unit_wrench_about_origin[0]
            )
            effective = (
                loaded
                and tangential
                >= self.settings.effective_unit_tangential_force_min_n
            )
            if loaded:
                contact_length += length
            if effective:
                effective_length += length
                continuous += length
                maximum_continuous = max(maximum_continuous, continuous)
            elif length > 0.0:
                continuous = 0.0

        path_points = points[1:] if len(points) > 1 else points
        tangential = np.asarray(
            [
                abs(point.response.wall_on_unit_wrench_about_origin[0])
                for point in path_points
            ],
            dtype=np.float64,
        )
        normal = np.asarray(
            [point.response.total_normal_force_n for point in path_points],
            dtype=np.float64,
        )
        neff_normal = np.asarray(
            [point.response.sharing.neff_normal for point in path_points]
        )
        neff_tangent = np.asarray(
            [
                point.response.sharing.neff_target_tangential
                for point in path_points
            ]
        )
        neff_resultant = np.asarray(
            [point.response.sharing.neff_resultant for point in path_points]
        )
        maximum_pin_force = max(
            (
                float(np.linalg.norm(np.asarray(wrench[:3])))
                for point in points
                for wrench in point.response.pin_wrench_about_unit
            ),
            default=0.0,
        )
        event_counts: dict[str, int] = {}
        for point in points:
            for _index, label in point.response.event_labels:
                event_counts[label] = event_counts.get(label, 0) + 1
        numerical = NumericalState.CONVERGED
        if any(
            point.response.numerical_state is NumericalState.NONCONVERGED
            for point in points
        ):
            numerical = NumericalState.NONCONVERGED
        elif any(
            point.response.numerical_state is not NumericalState.CONVERGED
            for point in points
        ):
            numerical = NumericalState.NOT_RUN
        model = ModelState.COVERED
        if any(
            point.response.model_state is ModelState.OUT_OF_SCOPE for point in points
        ):
            model = ModelState.OUT_OF_SCOPE
        elif any(
            point.response.model_state is ModelState.PARAMETER_UNCLOSED
            for point in points
        ):
            model = ModelState.PARAMETER_UNCLOSED
        preload_success = (
            len(points) >= 2
            and abs(points[1].response.total_normal_force_n - self.settings.target_preload_n)
            <= self.settings.preload_force_tolerance_n
        )
        return LegacyArrayPathSummary(
            initial_preload_success=preload_success,
            total_contact_length_m=contact_length,
            effective_load_length_m=effective_length,
            effective_load_fraction=(
                effective_length / self.settings.drag_length_m
                if self.settings.drag_length_m > 0.0
                else 0.0
            ),
            maximum_continuous_load_length_m=maximum_continuous,
            tangential_force_peak_n=float(np.max(tangential)),
            tangential_force_median_n=float(np.median(tangential)),
            tangential_force_p10_n=float(np.quantile(tangential, 0.10)),
            tangential_force_p25_n=float(np.quantile(tangential, 0.25)),
            total_normal_force_range_n=(
                float(np.min(normal)),
                float(np.max(normal)),
            ),
            neff_normal_median=float(np.median(neff_normal)),
            neff_target_tangential_median=float(np.median(neff_tangent)),
            neff_resultant_median=float(np.median(neff_resultant)),
            maximum_pin_resultant_force_n=maximum_pin_force,
            maximum_resultant_load_concentration=max(
                point.response.sharing.max_mean_resultant for point in points
            ),
            event_counts=event_counts,
            maximum_abs_local_geometry_residual_m=max(
                point.response.residual.maximum_local_geometry_m for point in points
            ),
            maximum_force_aggregation_residual_n=max(
                point.response.residual.force_aggregation_n for point in points
            ),
            maximum_moment_aggregation_residual_nm=max(
                point.response.residual.moment_aggregation_nm for point in points
            ),
            numerical_state=numerical,
            model_state=model,
            run_terminal_state=terminal,
            termination_reason=reason,
        )
