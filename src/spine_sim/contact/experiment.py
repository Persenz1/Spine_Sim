"""Initial-preload plus fixed-Z drag wrapper for one spine."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spine_sim.core.states import ModelState, NumericalState

from .errors import ContactConfigurationError
from .models import (
    ContactState,
    EventLabel,
    PathPoint,
    PathSummary,
    PathTerminalState,
    SingleSpineExperimentResult,
    SingleSpineState,
)
from .solver import PrescribedPoseConstitutiveCore, _ACTIVE_STATES


@dataclass(frozen=True)
class ExperimentSettings:
    initial_center_x_m: float
    drag_length_m: float
    path_step_m: float
    target_preload_n: float = 0.5
    preload_force_tolerance_n: float = 1e-4
    maximum_preload_approach_m: float = 8e-3
    effective_normal_force_min_n: float = 0.05
    free_probe_spacing_m: float | None = None
    refine_events: bool = True

    def __post_init__(self) -> None:
        finite = (
            "initial_center_x_m",
            "drag_length_m",
            "path_step_m",
            "target_preload_n",
            "preload_force_tolerance_n",
            "maximum_preload_approach_m",
            "effective_normal_force_min_n",
        )
        for name in finite:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ContactConfigurationError(f"{name} must be finite")
        if self.drag_length_m <= 0 or self.path_step_m <= 0:
            raise ContactConfigurationError("drag length and path step must be positive")
        if (
            self.target_preload_n <= 0
            or self.preload_force_tolerance_n <= 0
            or self.maximum_preload_approach_m <= 0
        ):
            raise ContactConfigurationError(
                "target preload, preload tolerance and maximum approach must be positive"
            )
        if self.effective_normal_force_min_n < 0:
            raise ContactConfigurationError(
                "effective_normal_force_min_n cannot be negative"
            )
        if self.free_probe_spacing_m is not None and self.free_probe_spacing_m <= 0:
            raise ContactConfigurationError("free_probe_spacing_m must be positive")


class SingleSpineExperiment:
    """Build preload once, freeze holder Z, then drag through FREE/recontact states."""

    def __init__(
        self,
        core: PrescribedPoseConstitutiveCore,
        settings: ExperimentSettings,
    ) -> None:
        self.core = core
        self.settings = settings

    def run(self) -> SingleSpineExperimentResult:
        first_contact = self._first_contact()
        if not first_contact.proposal_valid:
            return self._failed_result(
                PathTerminalState.MODEL_UNCLOSED,
                first_contact.residual.termination_reason,
                points=(),
            )
        preload = self._establish_preload(first_contact)
        if preload is None:
            return self._failed_result(
                PathTerminalState.INITIAL_PRELOAD_INFEASIBLE,
                "initial_preload_infeasible",
                points=(
                    PathPoint(0.0, first_contact, event_refined=True),
                ),
            )
        first_point = PathPoint(0.0, first_contact, event_refined=True)
        preload_point = PathPoint(0.0, preload, event_refined=True)
        points: list[PathPoint] = [first_point, preload_point]
        state = preload.next_state
        fixed_z = preload.holder_xz_m[1]
        start_holder_x = preload.holder_xz_m[0]
        current_x = start_holder_x
        terminal = PathTerminalState.PATH_END
        termination_reason = "path_end"

        while current_x < start_holder_x + self.settings.drag_length_m - 1e-15:
            target_x = min(
                current_x + self.settings.path_step_m,
                start_holder_x + self.settings.drag_length_m,
            )
            probe_spacing = (
                self.settings.free_probe_spacing_m
                or self.core.track.resolution_m
            )
            if state.contact_state in {ContactState.FREE, ContactState.DETACH_EVENT}:
                subdivisions = max(1, int(math.ceil((target_x - current_x) / probe_spacing)))
            else:
                subdivisions = 1
            probe_targets = np.linspace(
                current_x,
                target_x,
                subdivisions + 1,
                dtype=np.float64,
            )[1:]
            for probe_x in probe_targets:
                segment_points, state = self._advance_segment(
                    state,
                    (current_x, fixed_z),
                    (float(probe_x), fixed_z),
                )
                for response, refined in segment_points:
                    points.append(
                        PathPoint(
                            path_position_m=response.holder_xz_m[0] - start_holder_x,
                            response=response,
                            event_refined=refined,
                        )
                    )
                    if not response.proposal_valid:
                        if response.numerical_state is NumericalState.NONCONVERGED:
                            terminal = PathTerminalState.NUMERICAL_FAILURE
                        elif "geometry_out_of_domain" in response.residual.termination_reason:
                            terminal = PathTerminalState.TERRAIN_BOUNDS
                        elif "structural_boundary" in response.residual.termination_reason:
                            terminal = PathTerminalState.STRUCTURAL_BOUNDARY
                        else:
                            terminal = PathTerminalState.MODEL_UNCLOSED
                        termination_reason = response.residual.termination_reason
                        return self._result(
                            fixed_z,
                            tuple(points),
                            terminal,
                            termination_reason,
                        )
                current_x = float(probe_x)
        return self._result(
            fixed_z,
            tuple(points),
            terminal,
            termination_reason,
        )

    def _first_contact(self):
        parameters = self.core.parameters
        holder_x = (
            self.settings.initial_center_x_m
            - parameters.exposed_length_m * self.core._a[0]
        )
        geometry = self.core.geometry.query(self.settings.initial_center_x_m)
        holder_z = (
            geometry.envelope_height_m
            - parameters.exposed_length_m * self.core._a[1]
        )
        return self.core.solve_pose(
            (holder_x, holder_z),
            SingleSpineState(),
            commit=True,
        )

    def _establish_preload(self, contact_response):
        initial_state = contact_response.next_state
        holder_x, contact_z = contact_response.holder_xz_m
        target = self.settings.target_preload_n
        approach_values = np.geomspace(
            max(self.core.settings.gap_tolerance_m, 1e-10),
            self.settings.maximum_preload_approach_m,
            72,
        )
        lower_approach = 0.0
        upper_approach = None
        for approach in approach_values:
            candidate = self.core.solve_pose(
                (holder_x, contact_z - float(approach)),
                initial_state,
                commit=False,
            )
            if not candidate.proposal_valid:
                continue
            if candidate.normal_force_n >= target:
                upper_approach = float(approach)
                break
            lower_approach = float(approach)
        if upper_approach is None:
            return None
        best = None
        for _ in range(self.core.settings.root_max_iterations):
            middle = 0.5 * (lower_approach + upper_approach)
            candidate = self.core.solve_pose(
                (holder_x, contact_z - middle),
                initial_state,
                commit=False,
            )
            if not candidate.proposal_valid:
                lower_approach = middle
                continue
            best = candidate
            force_error = candidate.normal_force_n - target
            if (
                abs(force_error) <= self.core.settings.force_tolerance_n
            ):
                break
            if force_error < 0:
                lower_approach = middle
            else:
                upper_approach = middle
        if best is None:
            return None
        committed = self.core.solve_pose(
            best.holder_xz_m,
            initial_state,
            commit=True,
        )
        if (
            not committed.proposal_valid
            or abs(committed.normal_force_n - target)
            > self.settings.preload_force_tolerance_n
        ):
            return None
        return committed

    def _advance_segment(
        self,
        state: SingleSpineState,
        start_holder: tuple[float, float],
        end_holder: tuple[float, float],
        *,
        depth: int = 0,
    ) -> tuple[list[tuple[object, bool]], SingleSpineState]:
        trial = self.core.solve_pose(end_holder, state, commit=False)
        if not trial.proposal_valid:
            return [(trial, False)], state
        if (
            not self.settings.refine_events
            or trial.event_label is EventLabel.NONE
            or depth >= 8
            or abs(end_holder[0] - start_holder[0])
            <= self.core.settings.event_tolerance_m
        ):
            committed = self.core.solve_pose(end_holder, state, commit=True)
            return [(committed, False)], committed.next_state

        event = trial.event_label
        event_holder = self._locate_event(state, start_holder, end_holder, event)
        committed_event = self.core.solve_pose(event_holder, state, commit=True)
        output: list[tuple[object, bool]] = [(committed_event, True)]
        if abs(end_holder[0] - event_holder[0]) <= self.core.settings.event_tolerance_m:
            return output, committed_event.next_state
        remainder, final_state = self._advance_segment(
            committed_event.next_state,
            event_holder,
            end_holder,
            depth=depth + 1,
        )
        output.extend(remainder)
        return output, final_state

    def _locate_event(self, state, start_holder, end_holder, event):
        lower = np.asarray(start_holder, dtype=np.float64)
        upper = np.asarray(end_holder, dtype=np.float64)
        for _ in range(self.core.settings.event_max_iterations):
            if np.linalg.norm(upper - lower) <= self.core.settings.event_tolerance_m:
                break
            middle = 0.5 * (lower + upper)
            trial = self.core.solve_pose(
                (float(middle[0]), float(middle[1])),
                state,
                commit=False,
            )
            if self._event_reached(trial, event):
                upper = middle
            else:
                lower = middle
        return float(upper[0]), float(upper[1])

    @staticmethod
    def _event_reached(response, event: EventLabel) -> bool:
        if event is EventLabel.DETACH_TO_FREE:
            return response.contact_state in {
                ContactState.FREE,
                ContactState.DETACH_EVENT,
            }
        if event in {EventLabel.FIRST_CONTACT, EventLabel.RECONTACT}:
            return response.contact_state in _ACTIVE_STATES
        if event is EventLabel.SLIP_START:
            return response.contact_state is ContactState.SLIDE
        if event is EventLabel.HARD_STOP:
            return response.spring_state.value == "hard_stop"
        return False

    def _result(
        self,
        fixed_z_m: float,
        points: tuple[PathPoint, ...],
        terminal: PathTerminalState,
        reason: str,
    ) -> SingleSpineExperimentResult:
        return SingleSpineExperimentResult(
            parameters=self.core.parameters,
            track_id=self.core.track.track_id,
            fixed_holder_z_m=fixed_z_m,
            points=points,
            summary=self._summarize(points, terminal, reason),
            assumptions=(
                "initial preload is established once and holder Z is then fixed",
                "quasistatic main-plane response; physical speed is metadata only",
                "full-sphere M1 proxy is filtered by the forward-cap gate",
                self.core.parameters.material_assumption,
            ),
        )

    def _failed_result(
        self,
        terminal: PathTerminalState,
        reason: str,
        *,
        points: tuple[PathPoint, ...],
    ) -> SingleSpineExperimentResult:
        return SingleSpineExperimentResult(
            parameters=self.core.parameters,
            track_id=self.core.track.track_id,
            fixed_holder_z_m=None,
            points=points,
            summary=self._summarize(points, terminal, reason),
            assumptions=(
                "initial preload failed before fixed-Z drag",
                self.core.parameters.material_assumption,
            ),
        )

    def _summarize(
        self,
        points: tuple[PathPoint, ...],
        terminal: PathTerminalState,
        reason: str,
    ) -> PathSummary:
        if not points:
            return PathSummary(
                initial_preload_success=False,
                ever_contacted=False,
                ever_loaded=False,
                total_contact_length_m=0.0,
                effective_load_length_m=0.0,
                effective_load_fraction=0.0,
                maximum_continuous_load_length_m=0.0,
                tangential_force_peak_n=0.0,
                tangential_force_median_n=0.0,
                tangential_force_p10_n=0.0,
                tangential_force_p25_n=0.0,
                normal_force_range_n=(0.0, 0.0),
                event_counts={label.value: 0 for label in EventLabel if label is not EventLabel.NONE},
                maximum_abs_geometry_residual_m=0.0,
                maximum_abs_energy_residual_j=0.0,
                physical_terminal_state="not_started",
                numerical_state=NumericalState.NOT_RUN,
                model_state=ModelState.OUT_OF_SCOPE,
                run_terminal_state=terminal,
                termination_reason=reason,
            )
        contact_length = 0.0
        effective_length = 0.0
        maximum_continuous = 0.0
        continuous = 0.0
        for previous, current in zip(points, points[1:]):
            length = max(0.0, current.path_position_m - previous.path_position_m)
            previous_active = previous.response.contact_state in _ACTIVE_STATES
            previous_effective = (
                previous_active
                and previous.response.normal_force_n
                >= self.settings.effective_normal_force_min_n
            )
            if previous_active:
                contact_length += length
            if previous_effective:
                effective_length += length
                continuous += length
                maximum_continuous = max(maximum_continuous, continuous)
            elif length > 0:
                continuous = 0.0
        active_responses = [
            point.response
            for point in points
            if point.response.contact_state in _ACTIVE_STATES
            and point.response.normal_force_n > self.core.settings.force_tolerance_n
        ]
        tangential = np.asarray(
            [abs(response.tangential_force_n) for response in active_responses],
            dtype=np.float64,
        )
        normal = np.asarray(
            [response.normal_force_n for response in active_responses],
            dtype=np.float64,
        )
        event_counts = {
            label.value: sum(
                point.response.event_label is label for point in points
            )
            for label in EventLabel
            if label is not EventLabel.NONE
        }
        numerical = NumericalState.CONVERGED
        if any(
            point.response.numerical_state is NumericalState.NONCONVERGED
            for point in points
        ):
            numerical = NumericalState.NONCONVERGED
        elif any(
            point.response.numerical_state is NumericalState.NOT_RUN
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
        path_length = self.settings.drag_length_m
        return PathSummary(
            initial_preload_success=(
                len(points) >= 2
                and abs(
                    points[1].response.normal_force_n
                    - self.settings.target_preload_n
                )
                <= self.settings.preload_force_tolerance_n
            ),
            ever_contacted=any(
                point.response.contact_state in _ACTIVE_STATES for point in points
            ),
            ever_loaded=bool(normal.size),
            total_contact_length_m=contact_length,
            effective_load_length_m=effective_length,
            effective_load_fraction=(
                effective_length / path_length if path_length > 0 else 0.0
            ),
            maximum_continuous_load_length_m=maximum_continuous,
            tangential_force_peak_n=float(tangential.max()) if tangential.size else 0.0,
            tangential_force_median_n=(
                float(np.median(tangential)) if tangential.size else 0.0
            ),
            tangential_force_p10_n=(
                float(np.quantile(tangential, 0.10)) if tangential.size else 0.0
            ),
            tangential_force_p25_n=(
                float(np.quantile(tangential, 0.25)) if tangential.size else 0.0
            ),
            normal_force_range_n=(
                (float(normal.min()), float(normal.max()))
                if normal.size
                else (0.0, 0.0)
            ),
            event_counts=event_counts,
            maximum_abs_geometry_residual_m=max(
                abs(point.response.residual.geometry_m)
                for point in points
                if math.isfinite(point.response.residual.geometry_m)
            ),
            maximum_abs_energy_residual_j=max(
                abs(point.response.energy_residual_j) for point in points
            ),
            physical_terminal_state=points[-1].response.contact_state.value,
            numerical_state=numerical,
            model_state=model,
            run_terminal_state=terminal,
            termination_reason=reason,
        )
