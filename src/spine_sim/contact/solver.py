"""CPU-authoritative prescribed-holder-pose constitutive core for M2."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from spine_sim.core.states import ModelState, NumericalState
from spine_sim.terrain.models import TrackGeometry

from .errors import ContactConfigurationError, ContactGeometryError
from .geometry import TrackInterpolator
from .models import (
    AxialMode,
    ConstitutiveResponse,
    ContactState,
    EventLabel,
    ResidualAudit,
    SingleSpineState,
    SolverSettings,
    SpineParameters,
    SpringState,
)


_ACTIVE_STATES = {
    ContactState.FIRST_CONTACT_EVENT,
    ContactState.RECONTACT_EVENT,
    ContactState.STICK,
    ContactState.SLIDE,
}


class _StructuralIndeterminacy(RuntimeError):
    pass


class _StructuralBoundary(RuntimeError):
    pass


class PrescribedPoseConstitutiveCore:
    """Pure single-step contact solve at a prescribed installation-seat pose."""

    def __init__(
        self,
        parameters: SpineParameters,
        track: TrackGeometry,
        settings: SolverSettings | None = None,
    ) -> None:
        self.parameters = parameters
        self.track = track
        self.settings = settings or SolverSettings()
        self.geometry = TrackInterpolator(track, parameters)
        self._a = parameters.axis_xz
        self._b = parameters.transverse_xz
        self._ca = parameters.axial_compliance_m_n
        self._cb = parameters.transverse_compliance_m_n

    def solve_pose(
        self,
        holder_xz_m: tuple[float, float],
        old_state: SingleSpineState | None = None,
        *,
        commit: bool = False,
    ) -> ConstitutiveResponse:
        """Solve one pose without mutating ``old_state`` or the M1 track."""

        old = old_state or SingleSpineState()
        holder = np.asarray(holder_xz_m, dtype=np.float64)
        if holder.shape != (2,) or not np.all(np.isfinite(holder)):
            raise ContactConfigurationError("holder_xz_m must be a finite 2-vector")
        base = holder + self.parameters.exposed_length_m * self._a
        try:
            base_geometry = self.geometry.query(float(base[0]))
        except ContactGeometryError as exc:
            return self._invalid_response(
                holder,
                base,
                old,
                commit=commit,
                numerical_state=NumericalState.NOT_RUN,
                model_state=ModelState.OUT_OF_SCOPE,
                reason=str(exc),
            )
        base_gap = float(base[1] - base_geometry.envelope_height_m)

        try:
            if old.contact_state in {ContactState.FREE, ContactState.DETACH_EVENT}:
                if base_gap > self.settings.gap_tolerance_m:
                    return self._free_response(
                        holder,
                        base,
                        base_geometry,
                        base_gap,
                        old,
                        commit=commit,
                    )
                return self._new_contact_response(
                    holder,
                    base,
                    base_geometry,
                    old,
                    commit=commit,
                )

            if old.contact_state is ContactState.SLIDE and self._continues_sliding(
                holder, old
            ):
                if base_gap > self.settings.gap_tolerance_m:
                    return self._free_response(
                        holder,
                        base,
                        base_geometry,
                        base_gap,
                        old,
                        commit=commit,
                        detached=True,
                    )
                return self._slide_response(
                    holder,
                    base,
                    old,
                    slide_direction=old.slide_direction,
                    commit=commit,
                )

            return self._stick_trial_response(holder, base, old, commit=commit)
        except ContactGeometryError as exc:
            return self._invalid_response(
                holder,
                base,
                old,
                commit=commit,
                numerical_state=NumericalState.NOT_RUN,
                model_state=ModelState.OUT_OF_SCOPE,
                reason=str(exc),
            )
        except _StructuralIndeterminacy as exc:
            return self._invalid_response(
                holder,
                base,
                old,
                commit=commit,
                numerical_state=NumericalState.CONVERGED,
                model_state=ModelState.OUT_OF_SCOPE,
                reason=str(exc),
            )
        except _StructuralBoundary as exc:
            return self._invalid_response(
                holder,
                base,
                old,
                commit=commit,
                numerical_state=NumericalState.CONVERGED,
                model_state=ModelState.OUT_OF_SCOPE,
                reason=str(exc),
            )

    def _continues_sliding(
        self, holder: NDArray[np.float64], old: SingleSpineState
    ) -> bool:
        if (
            old.slide_direction == 0
            or old.last_holder_xz_m is None
            or old.last_center_xz_m is None
        ):
            return False
        geometry = self.geometry.query(old.last_center_xz_m[0])
        displacement = holder - np.asarray(old.last_holder_xz_m, dtype=np.float64)
        tangent = np.asarray(geometry.tangent_xz, dtype=np.float64)
        tangential_motion = float(displacement @ tangent)
        if abs(tangential_motion) <= self.settings.stick_motion_tolerance_m:
            return False
        return int(math.copysign(1, tangential_motion)) == old.slide_direction

    def _new_contact_response(
        self,
        holder: NDArray[np.float64],
        base: NDArray[np.float64],
        base_geometry,
        old: SingleSpineState,
        *,
        commit: bool,
    ) -> ConstitutiveResponse:
        solved = self._solve_contact_ray(
            holder,
            base,
            friction_coefficient=0.0,
            slide_direction=0,
            initial_center_x=base[0],
        )
        if solved is None:
            return self._invalid_response(
                holder,
                base,
                old,
                commit=commit,
                numerical_state=NumericalState.CONVERGED,
                model_state=ModelState.OUT_OF_SCOPE,
                reason="no_admissible_contact_equilibrium",
            )
        (
            center,
            geometry,
            force,
            normal_force,
            tangential_force,
            axial_force,
            transverse_force,
            spring_compression,
            spring_state,
            iterations,
            bracket,
        ) = solved
        if geometry.cap_margin_m < -self.settings.cap_tolerance_m:
            return self._free_response(
                holder,
                base,
                base_geometry,
                float(base[1] - base_geometry.envelope_height_m),
                old,
                commit=commit,
                extra_warning="forward_cap_gate_rejected",
            )
        event = (
            EventLabel.RECONTACT if old.has_contacted else EventLabel.FIRST_CONTACT
        )
        state = (
            ContactState.RECONTACT_EVENT
            if old.has_contacted
            else ContactState.FIRST_CONTACT_EVENT
        )
        return self._contact_response(
            holder,
            center,
            geometry,
            force,
            normal_force,
            tangential_force,
            axial_force,
            transverse_force,
            spring_compression,
            spring_state,
            state,
            event,
            old,
            slide_direction=0,
            root_iterations=iterations,
            root_bracket=bracket,
            commit=commit,
        )

    def _stick_trial_response(
        self,
        holder: NDArray[np.float64],
        base: NDArray[np.float64],
        old: SingleSpineState,
        *,
        commit: bool,
    ) -> ConstitutiveResponse:
        if old.anchor_center_xz_m is None:
            raise _StructuralIndeterminacy(
                "indeterminate_rigid_stick: active contact is missing a contact anchor"
            )
        center = np.asarray(old.anchor_center_xz_m, dtype=np.float64)
        geometry = self.geometry.query(float(center[0]))
        geometry_residual = float(center[1] - geometry.envelope_height_m)
        if abs(geometry_residual) > self.settings.residual_tolerance_m * 5.0:
            raise ContactGeometryError(
                "invalid_geometry: stored STICK anchor no longer lies on the M1 envelope"
            )
        required = center - base
        shortening = -float(required @ self._a)
        (
            axial_force,
            spring_compression,
            spring_state,
            _spring_energy,
        ) = self._axial_from_shortening(shortening)
        if self._cb <= 0.0:
            raise _StructuralIndeterminacy(
                "indeterminate_rigid_stick: transverse beam compliance is disabled"
            )
        transverse_force = float(required @ self._b) / self._cb
        force = -axial_force * self._a + transverse_force * self._b
        if float(np.linalg.norm(force)) > self.settings.max_contact_force_n:
            raise _StructuralBoundary(
                "structural_boundary: STICK force exceeds the configured safety limit"
            )
        normal = np.asarray(geometry.normal_xz, dtype=np.float64)
        tangent = np.asarray(geometry.tangent_xz, dtype=np.float64)
        normal_force = float(force @ normal)
        tangential_force = float(force @ tangent)
        if normal_force < -self.settings.force_tolerance_n:
            zero_force_center = base
            zero_geometry = self.geometry.query(float(zero_force_center[0]))
            zero_gap = float(
                zero_force_center[1] - zero_geometry.envelope_height_m
            )
            if zero_gap < -self.settings.gap_tolerance_m:
                direction = self._relative_motion_direction(
                    holder,
                    old,
                    tangent,
                    fallback_force=tangential_force,
                )
                return self._slide_response(
                    holder,
                    base,
                    old,
                    slide_direction=direction,
                    commit=commit,
                    slip_start=True,
                )
            return self._free_response(
                holder,
                zero_force_center,
                zero_geometry,
                zero_gap,
                old,
                commit=commit,
                detached=True,
            )
        friction_limit = self.parameters.static_friction * max(normal_force, 0.0)
        if abs(tangential_force) <= friction_limit + self.settings.force_tolerance_n:
            event = EventLabel.NONE
            if (
                spring_state is SpringState.HARD_STOP
                and old.spring_state is not SpringState.HARD_STOP
            ):
                event = EventLabel.HARD_STOP
            return self._contact_response(
                holder,
                center,
                geometry,
                force,
                max(0.0, normal_force),
                tangential_force,
                axial_force,
                transverse_force,
                spring_compression,
                spring_state,
                ContactState.STICK,
                event,
                old,
                slide_direction=0,
                root_iterations=0,
                root_bracket=None,
                commit=commit,
            )
        direction = -int(math.copysign(1, tangential_force))
        return self._slide_response(
            holder,
            base,
            old,
            slide_direction=direction,
            commit=commit,
            slip_start=True,
        )

    def _relative_motion_direction(
        self,
        holder: NDArray[np.float64],
        old: SingleSpineState,
        tangent: NDArray[np.float64],
        *,
        fallback_force: float,
    ) -> int:
        if old.last_holder_xz_m is not None:
            displacement = holder - np.asarray(
                old.last_holder_xz_m, dtype=np.float64
            )
            motion = float(displacement @ tangent)
            if abs(motion) > self.settings.stick_motion_tolerance_m:
                return int(math.copysign(1, motion))
        if abs(fallback_force) > self.settings.force_tolerance_n:
            return -int(math.copysign(1, fallback_force))
        return old.slide_direction if old.slide_direction in {-1, 1} else 1

    def _slide_response(
        self,
        holder: NDArray[np.float64],
        base: NDArray[np.float64],
        old: SingleSpineState,
        *,
        slide_direction: int,
        commit: bool,
        slip_start: bool = False,
    ) -> ConstitutiveResponse:
        if slide_direction not in {-1, 1}:
            raise ContactConfigurationError("SLIDE requires direction -1 or +1")
        initial_x = (
            old.last_center_xz_m[0]
            if old.last_center_xz_m is not None
            else float(base[0])
        )
        solved = self._solve_contact_ray(
            holder,
            base,
            friction_coefficient=self.parameters.kinetic_friction,
            slide_direction=slide_direction,
            initial_center_x=initial_x,
        )
        if solved is None:
            base_geometry = self.geometry.query(float(base[0]))
            base_gap = float(base[1] - base_geometry.envelope_height_m)
            if base_gap > -self.settings.gap_tolerance_m:
                return self._free_response(
                    holder,
                    base,
                    base_geometry,
                    base_gap,
                    old,
                    commit=commit,
                    detached=True,
                )
            return self._invalid_response(
                holder,
                base,
                old,
                commit=commit,
                numerical_state=NumericalState.CONVERGED,
                model_state=ModelState.OUT_OF_SCOPE,
                reason="no_admissible_contact_equilibrium",
            )
        (
            center,
            geometry,
            force,
            normal_force,
            tangential_force,
            axial_force,
            transverse_force,
            spring_compression,
            spring_state,
            iterations,
            bracket,
        ) = solved
        if geometry.cap_margin_m < -self.settings.cap_tolerance_m:
            base_geometry = self.geometry.query(float(base[0]))
            return self._free_response(
                holder,
                base,
                base_geometry,
                float(base[1] - base_geometry.envelope_height_m),
                old,
                commit=commit,
                detached=True,
                extra_warning="forward_cap_gate_rejected",
            )
        event = EventLabel.SLIP_START if slip_start else EventLabel.NONE
        if (
            event is EventLabel.NONE
            and spring_state is SpringState.HARD_STOP
            and old.spring_state is not SpringState.HARD_STOP
        ):
            event = EventLabel.HARD_STOP
        return self._contact_response(
            holder,
            center,
            geometry,
            force,
            normal_force,
            tangential_force,
            axial_force,
            transverse_force,
            spring_compression,
            spring_state,
            ContactState.SLIDE,
            event,
            old,
            slide_direction=slide_direction,
            root_iterations=iterations,
            root_bracket=bracket,
            commit=commit,
        )

    def _solve_contact_ray(
        self,
        holder: NDArray[np.float64],
        base: NDArray[np.float64],
        *,
        friction_coefficient: float,
        slide_direction: int,
        initial_center_x: float,
    ):
        settings = self.settings

        def evaluate(normal_force: float):
            geometry = self.geometry.query(float(initial_center_x))
            center = base.copy()
            force = np.zeros(2, dtype=np.float64)
            axial_force = 0.0
            transverse_force = 0.0
            compression = 0.0
            spring_state = SpringState.LOWER_STOP
            for _ in range(40):
                normal = np.asarray(geometry.normal_xz, dtype=np.float64)
                tangent = np.asarray(geometry.tangent_xz, dtype=np.float64)
                force = normal_force * (
                    normal
                    - friction_coefficient * float(slide_direction) * tangent
                )
                axial_force = -float(force @ self._a)
                transverse_force = float(force @ self._b)
                (
                    compression,
                    spring_state,
                    _energy,
                    _lower_margin,
                    _travel_margin,
                ) = self._spring_from_force(axial_force)
                center = self._kinematic_center(
                    holder,
                    axial_force,
                    transverse_force,
                    compression,
                )
                updated = self.geometry.query(float(center[0]))
                if (
                    abs(updated.envelope_height_m - geometry.envelope_height_m)
                    <= settings.residual_tolerance_m * 0.1
                    and abs(updated.envelope_slope_x - geometry.envelope_slope_x)
                    <= 1e-8
                ):
                    geometry = updated
                    break
                geometry = updated
            gap = float(center[1] - geometry.envelope_height_m)
            return (
                gap,
                center,
                geometry,
                force,
                axial_force,
                transverse_force,
                compression,
                spring_state,
            )

        first = evaluate(0.0)
        if first[0] >= -settings.gap_tolerance_m:
            if first[0] <= settings.gap_tolerance_m:
                return self._ray_result(0.0, first, iterations=0, bracket=(0.0, 0.0))
            return None

        minimum = max(settings.force_tolerance_n, 1e-9)
        scan_forces = np.geomspace(
            minimum,
            settings.max_contact_force_n,
            settings.root_scan_points,
        )
        lower_force = 0.0
        lower_eval = first
        bracket: tuple[float, float] | None = None
        upper_eval = None
        for candidate_force in scan_forces:
            candidate = evaluate(float(candidate_force))
            if candidate[0] >= 0.0:
                bracket = (lower_force, float(candidate_force))
                upper_eval = candidate
                break
            lower_force = float(candidate_force)
            lower_eval = candidate
        if bracket is None or upper_eval is None:
            return self._solve_contact_by_center(
                holder,
                base,
                friction_coefficient=friction_coefficient,
                slide_direction=slide_direction,
                initial_center_x=initial_center_x,
            )

        lo, hi = bracket
        best = upper_eval
        iterations = 0
        for iterations in range(1, settings.root_max_iterations + 1):
            middle = 0.5 * (lo + hi)
            candidate = evaluate(middle)
            best = candidate
            if (
                abs(candidate[0]) <= settings.residual_tolerance_m
            ):
                break
            if candidate[0] < 0.0:
                lo = middle
                lower_eval = candidate
            else:
                hi = middle
                upper_eval = candidate
        normal_force = 0.5 * (lo + hi)
        best = evaluate(normal_force)
        if abs(best[0]) > max(
            settings.residual_tolerance_m * 2.0,
            settings.gap_tolerance_m * 2.0,
        ):
            return self._solve_contact_by_center(
                holder,
                base,
                friction_coefficient=friction_coefficient,
                slide_direction=slide_direction,
                initial_center_x=initial_center_x,
            )
        result = self._ray_result(
            normal_force,
            best,
            iterations=iterations,
            bracket=bracket,
        )
        friction_residual = abs(
            result[4]
            + friction_coefficient
            * float(slide_direction)
            * result[3]
        )
        if friction_residual > max(
            self.settings.friction_residual_tolerance_n,
            self.settings.force_tolerance_n * 10.0,
        ):
            return self._solve_contact_by_center(
                holder,
                base,
                friction_coefficient=friction_coefficient,
                slide_direction=slide_direction,
                initial_center_x=initial_center_x,
            )
        return result

    def _solve_contact_by_center(
        self,
        holder: NDArray[np.float64],
        base: NDArray[np.float64],
        *,
        friction_coefficient: float,
        slide_direction: int,
        initial_center_x: float,
    ):
        """Fallback scalar solve using contact-center x as the continuation variable."""

        if self._cb <= 0.0:
            return None
        valid_min, valid_max = self.geometry.valid_x_range_m
        maximum_shift = (
            self.parameters.spring_travel_m
            + (self._ca + self._cb) * self.settings.max_contact_force_n
            + 4.0 * self.track.resolution_m
        )
        lower = max(valid_min, float(base[0] - maximum_shift))
        upper = min(valid_max, float(base[0] + maximum_shift))
        if upper <= lower:
            return None
        spacing = self.track.resolution_m
        count = max(3, int(math.ceil((upper - lower) / spacing)) + 1)
        centers = np.linspace(lower, upper, count, dtype=np.float64)
        if lower <= initial_center_x <= upper:
            centers = np.unique(
                np.concatenate(
                    (centers, np.asarray([initial_center_x], dtype=np.float64))
                )
            )

        def evaluate(center_x: float):
            geometry = self.geometry.query(float(center_x))
            center = np.array(
                [center_x, geometry.envelope_height_m],
                dtype=np.float64,
            )
            required = center - base
            shortening = -float(required @ self._a)
            try:
                (
                    axial_force,
                    compression,
                    spring_state,
                    _spring_energy,
                ) = self._axial_from_shortening(shortening)
            except (_StructuralBoundary, _StructuralIndeterminacy):
                return None
            transverse_force = float(required @ self._b) / self._cb
            force = -axial_force * self._a + transverse_force * self._b
            if (
                not np.all(np.isfinite(force))
                or float(np.linalg.norm(force)) > self.settings.max_contact_force_n
            ):
                return None
            normal_force = float(
                force @ np.asarray(geometry.normal_xz, dtype=np.float64)
            )
            tangential_force = float(
                force @ np.asarray(geometry.tangent_xz, dtype=np.float64)
            )
            if normal_force < -self.settings.force_tolerance_n:
                return None
            residual = tangential_force + (
                friction_coefficient * float(slide_direction) * normal_force
            )
            return (
                residual,
                center,
                geometry,
                force,
                max(0.0, normal_force),
                tangential_force,
                axial_force,
                transverse_force,
                compression,
                spring_state,
            )

        roots: list[tuple[float, tuple, int, tuple[float, float]]] = []
        previous_x: float | None = None
        previous = None
        for center_x in centers:
            current = evaluate(float(center_x))
            if current is None:
                previous_x = None
                previous = None
                continue
            if abs(current[0]) <= self.settings.force_tolerance_n:
                roots.append(
                    (
                        float(center_x),
                        current,
                        0,
                        (float(center_x), float(center_x)),
                    )
                )
            if (
                previous is not None
                and previous_x is not None
                and previous[0] * current[0] < 0.0
            ):
                lo = previous_x
                hi = float(center_x)
                lo_value = previous
                best = current
                iterations = 0
                valid_bracket = True
                for iterations in range(1, self.settings.root_max_iterations + 1):
                    middle = 0.5 * (lo + hi)
                    candidate = evaluate(middle)
                    if candidate is None:
                        valid_bracket = False
                        break
                    best = candidate
                    if (
                        abs(candidate[0]) <= self.settings.force_tolerance_n
                        or hi - lo <= self.settings.residual_tolerance_m
                    ):
                        break
                    if lo_value[0] * candidate[0] <= 0.0:
                        hi = middle
                    else:
                        lo = middle
                        lo_value = candidate
                if valid_bracket and abs(best[0]) <= max(
                    self.settings.friction_residual_tolerance_n,
                    self.settings.force_tolerance_n * 10.0,
                ):
                    root_x = float(best[1][0])
                    roots.append(
                        (root_x, best, iterations, (previous_x, float(center_x)))
                    )
            previous_x = float(center_x)
            previous = current
        if not roots:
            return None
        root_x, solved, iterations, bracket = min(
            roots,
            key=lambda item: abs(item[0] - initial_center_x),
        )
        (
            _residual,
            center,
            geometry,
            force,
            normal_force,
            tangential_force,
            axial_force,
            transverse_force,
            compression,
            spring_state,
        ) = solved
        return (
            center,
            geometry,
            force,
            normal_force,
            tangential_force,
            axial_force,
            transverse_force,
            compression,
            spring_state,
            iterations,
            bracket,
        )

    def _ray_result(self, normal_force, evaluated, *, iterations, bracket):
        (
            _gap,
            center,
            geometry,
            force,
            axial_force,
            transverse_force,
            compression,
            spring_state,
        ) = evaluated
        tangent = np.asarray(geometry.tangent_xz, dtype=np.float64)
        actual_normal = float(
            force @ np.asarray(geometry.normal_xz, dtype=np.float64)
        )
        tangential = float(force @ tangent)
        return (
            center,
            geometry,
            force,
            actual_normal if normal_force != 0.0 else 0.0,
            tangential,
            axial_force,
            transverse_force,
            compression,
            spring_state,
            iterations,
            bracket,
        )

    def _axial_from_shortening(
        self, shortening_m: float
    ) -> tuple[float, float, SpringState, float]:
        if self.parameters.axial_mode is AxialMode.RIGID:
            if self._ca <= 0.0:
                raise _StructuralIndeterminacy(
                    "indeterminate_rigid_stick: axial spring and beam are disabled"
                )
            force = shortening_m / self._ca
            return force, 0.0, SpringState.LOWER_STOP, 0.0

        stiffness = float(self.parameters.spring_stiffness_n_m)
        travel = self.parameters.spring_travel_m
        if shortening_m <= 0.0:
            if self._ca <= 0.0 and shortening_m < -self.settings.residual_tolerance_m:
                raise _StructuralBoundary(
                    "structural_boundary: rigid axial member cannot lengthen at LOWER_STOP"
                )
            force = shortening_m / self._ca if self._ca > 0.0 else 0.0
            return force, 0.0, SpringState.LOWER_STOP, 0.0
        interior_limit = travel + self._ca * stiffness * travel
        if shortening_m < interior_limit:
            force = shortening_m / (self._ca + 1.0 / stiffness)
            compression = force / stiffness
            return (
                force,
                compression,
                SpringState.INTERIOR,
                0.5 * force * compression,
            )
        if self._ca <= 0.0:
            if math.isclose(
                shortening_m,
                travel,
                rel_tol=0.0,
                abs_tol=self.settings.residual_tolerance_m,
            ):
                force = stiffness * travel
                return (
                    force,
                    travel,
                    SpringState.HARD_STOP,
                    0.5 * stiffness * travel**2,
                )
            raise _StructuralBoundary(
                "structural_boundary: displacement exceeds a rigid HARD_STOP"
            )
        force = (shortening_m - travel) / self._ca
        return (
            force,
            travel,
            SpringState.HARD_STOP,
            0.5 * stiffness * travel**2,
        )

    def _spring_from_force(
        self, axial_force_n: float
    ) -> tuple[float, SpringState, float, float, float]:
        if self.parameters.axial_mode is AxialMode.RIGID:
            return (
                0.0,
                SpringState.LOWER_STOP,
                0.0,
                0.0,
                self.parameters.spring_travel_m,
            )
        stiffness = float(self.parameters.spring_stiffness_n_m)
        travel = self.parameters.spring_travel_m
        if axial_force_n <= 0.0:
            return 0.0, SpringState.LOWER_STOP, 0.0, -axial_force_n, travel
        hard_force = stiffness * travel
        if axial_force_n < hard_force:
            compression = axial_force_n / stiffness
            return (
                compression,
                SpringState.INTERIOR,
                0.5 * axial_force_n * compression,
                compression,
                travel - compression,
            )
        return (
            travel,
            SpringState.HARD_STOP,
            0.5 * stiffness * travel**2,
            travel,
            0.0,
        )

    def _kinematic_center(
        self,
        holder: NDArray[np.float64],
        axial_force_n: float,
        transverse_force_n: float,
        compression_m: float,
    ) -> NDArray[np.float64]:
        return (
            holder
            + self.parameters.exposed_length_m * self._a
            - (compression_m + self._ca * axial_force_n) * self._a
            + self._cb * transverse_force_n * self._b
        )

    def _default_model_state(self) -> ModelState:
        if self.parameters.rod_clearance_mode == "unclosed":
            return ModelState.PARAMETER_UNCLOSED
        return ModelState.COVERED

    def _model_warnings(self, *, extra: str | None = None, near_tie=False):
        warnings = list(self.track.model_warning)
        if self.parameters.rod_clearance_mode == "disabled_analytic_fixture":
            warnings.append("rod_clearance_disabled_for_analytic_fixture")
        if near_tie:
            warnings.append("m1_near_tie_support")
        if extra:
            warnings.append(extra)
        return tuple(dict.fromkeys(warnings))

    def _energy_terms(
        self,
        old: SingleSpineState,
        holder: NDArray[np.float64],
        center: NDArray[np.float64],
        force: NDArray[np.float64],
        elastic_energy: float,
        *,
        sliding: bool,
    ) -> tuple[float, float, float, float]:
        if (
            old.last_holder_xz_m is None
            or old.last_center_xz_m is None
        ):
            return 0.0, 0.0, 0.0, 0.0
        previous_force = np.asarray(old.last_wall_force_xz_n, dtype=np.float64)
        average_force = 0.5 * (previous_force + force)
        holder_displacement = holder - np.asarray(
            old.last_holder_xz_m, dtype=np.float64
        )
        center_displacement = center - np.asarray(
            old.last_center_xz_m, dtype=np.float64
        )
        holder_work = float((-average_force) @ holder_displacement)
        contact_work = float(average_force @ center_displacement)
        elastic_change = elastic_energy - old.last_elastic_energy_j
        residual = holder_work + contact_work - elastic_change
        dissipation = max(0.0, -contact_work) if sliding else 0.0
        return holder_work, contact_work, dissipation, residual

    def _contact_response(
        self,
        holder,
        center,
        geometry,
        force,
        normal_force,
        tangential_force,
        axial_force,
        transverse_force,
        spring_compression,
        spring_state,
        contact_state,
        event_label,
        old,
        *,
        slide_direction,
        root_iterations,
        root_bracket,
        commit,
    ) -> ConstitutiveResponse:
        force = np.asarray(force, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
        beam_displacement = (
            -self._ca * axial_force * self._a
            + self._cb * transverse_force * self._b
        )
        reconstructed = self._kinematic_center(
            np.asarray(holder, dtype=np.float64),
            axial_force,
            transverse_force,
            spring_compression,
        )
        geometry_residual = float(
            center[1] - geometry.envelope_height_m
        )
        structure_residual = float(np.linalg.norm(reconstructed - center))
        normal = np.asarray(geometry.normal_xz, dtype=np.float64)
        tangent = np.asarray(geometry.tangent_xz, dtype=np.float64)
        decomposed = normal_force * normal + tangential_force * tangent
        force_residual = float(np.linalg.norm(force - decomposed))
        (
            _compression,
            _spring_state,
            spring_energy,
            lower_margin,
            travel_margin,
        ) = self._spring_from_force(axial_force)
        elastic_energy = (
            0.5 * self._ca * axial_force**2
            + 0.5 * self._cb * transverse_force**2
            + spring_energy
        )
        holder_work, contact_work, dissipation, energy_residual = self._energy_terms(
            old,
            np.asarray(holder, dtype=np.float64),
            center,
            force,
            elastic_energy,
            sliding=contact_state is ContactState.SLIDE,
        )
        friction_margin = (
            self.parameters.static_friction * normal_force
            - abs(tangential_force)
            if contact_state is ContactState.STICK
            else self.settings.force_tolerance_n
            + self.settings.friction_residual_tolerance_n
            - abs(
                abs(tangential_force)
                - self.parameters.kinetic_friction * normal_force
            )
        )
        support = geometry.support_xyz_m
        plate_force = np.array([-force[0], 0.0, -force[1]], dtype=np.float64)
        lever = np.array(
            [
                support[0] - holder[0],
                0.0,
                support[2] - holder[1],
            ],
            dtype=np.float64,
        )
        moment = np.cross(lever, plate_force)
        warnings = self._model_warnings(
            near_tie=geometry.near_tie,
        )
        model_state = self._default_model_state()
        proposal = SingleSpineState(
            contact_state=contact_state,
            spring_state=spring_state,
            has_contacted=True,
            anchor_center_xz_m=(float(center[0]), float(center[1])),
            last_holder_xz_m=(float(holder[0]), float(holder[1])),
            last_center_xz_m=(float(center[0]), float(center[1])),
            last_wall_force_xz_n=(float(force[0]), float(force[1])),
            last_elastic_energy_j=float(elastic_energy),
            cumulative_friction_dissipation_j=(
                old.cumulative_friction_dissipation_j + dissipation
            ),
            slide_direction=slide_direction,
            accepted_steps=old.accepted_steps + 1,
        )
        residual = ResidualAudit(
            geometry_m=geometry_residual,
            structure_m=structure_residual,
            force_decomposition_n=force_residual,
            energy_j=energy_residual,
            unilateral_margin_n=normal_force,
            friction_margin_n=friction_margin,
            spring_lower_margin_m=lower_margin,
            spring_travel_margin_m=travel_margin,
            cap_margin_m=geometry.cap_margin_m,
            root_iterations=root_iterations,
            root_bracket_n=root_bracket,
            termination_reason="converged",
        )
        return ConstitutiveResponse(
            holder_xz_m=(float(holder[0]), float(holder[1])),
            center_xz_m=(float(center[0]), float(center[1])),
            support_xyz_m=support,
            gap_m=geometry_residual,
            tangent_xz=geometry.tangent_xz,
            normal_xz=geometry.normal_xz,
            cap_gate_passed=(
                geometry.cap_margin_m >= -self.settings.cap_tolerance_m
            ),
            near_tie=geometry.near_tie,
            contact_state=contact_state,
            spring_state=spring_state,
            event_label=event_label,
            wall_on_spine_force_xz_n=(float(force[0]), float(force[1])),
            spine_on_plate_wrench_about_holder=(
                float(plate_force[0]),
                float(plate_force[1]),
                float(plate_force[2]),
                float(moment[0]),
                float(moment[1]),
                float(moment[2]),
            ),
            normal_force_n=float(normal_force),
            tangential_force_n=float(tangential_force),
            axial_force_n=float(axial_force),
            transverse_force_n=float(transverse_force),
            spring_compression_m=float(spring_compression),
            beam_displacement_xz_m=(
                float(beam_displacement[0]),
                float(beam_displacement[1]),
            ),
            static_friction_margin_n=float(friction_margin),
            spring_travel_margin_m=float(travel_margin),
            elastic_energy_j=float(elastic_energy),
            holder_work_increment_j=holder_work,
            contact_work_increment_j=contact_work,
            friction_dissipation_increment_j=dissipation,
            energy_residual_j=energy_residual,
            numerical_state=NumericalState.CONVERGED,
            model_state=model_state,
            model_warnings=warnings,
            residual=residual,
            proposal_state=proposal,
            next_state=proposal if commit else old,
            proposal_valid=True,
        )

    def _free_response(
        self,
        holder,
        center,
        geometry,
        gap,
        old,
        *,
        commit,
        detached=False,
        extra_warning=None,
    ) -> ConstitutiveResponse:
        center = np.asarray(center, dtype=np.float64)
        holder = np.asarray(holder, dtype=np.float64)
        force = np.zeros(2, dtype=np.float64)
        holder_work, contact_work, _diss, energy_residual = self._energy_terms(
            old,
            holder,
            center,
            force,
            0.0,
            sliding=False,
        )
        event = EventLabel.DETACH_TO_FREE if detached else EventLabel.NONE
        state = ContactState.DETACH_EVENT if detached else ContactState.FREE
        warnings = self._model_warnings(
            extra=extra_warning,
            near_tie=geometry.near_tie,
        )
        proposal = SingleSpineState(
            contact_state=state,
            spring_state=SpringState.LOWER_STOP,
            has_contacted=old.has_contacted,
            anchor_center_xz_m=None,
            last_holder_xz_m=(float(holder[0]), float(holder[1])),
            last_center_xz_m=(float(center[0]), float(center[1])),
            last_wall_force_xz_n=(0.0, 0.0),
            last_elastic_energy_j=0.0,
            cumulative_friction_dissipation_j=old.cumulative_friction_dissipation_j,
            slide_direction=0,
            accepted_steps=old.accepted_steps + 1,
        )
        residual = ResidualAudit(
            geometry_m=min(0.0, float(gap)),
            structure_m=0.0,
            force_decomposition_n=0.0,
            energy_j=energy_residual,
            unilateral_margin_n=0.0,
            friction_margin_n=0.0,
            spring_lower_margin_m=0.0,
            spring_travel_margin_m=self.parameters.spring_travel_m,
            cap_margin_m=geometry.cap_margin_m,
            root_iterations=0,
            root_bracket_n=None,
            termination_reason="detach_to_free" if detached else "free",
        )
        return ConstitutiveResponse(
            holder_xz_m=(float(holder[0]), float(holder[1])),
            center_xz_m=(float(center[0]), float(center[1])),
            support_xyz_m=geometry.support_xyz_m,
            gap_m=float(gap),
            tangent_xz=geometry.tangent_xz,
            normal_xz=geometry.normal_xz,
            cap_gate_passed=(
                geometry.cap_margin_m >= -self.settings.cap_tolerance_m
            ),
            near_tie=geometry.near_tie,
            contact_state=state,
            spring_state=SpringState.LOWER_STOP,
            event_label=event,
            wall_on_spine_force_xz_n=(0.0, 0.0),
            spine_on_plate_wrench_about_holder=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            normal_force_n=0.0,
            tangential_force_n=0.0,
            axial_force_n=0.0,
            transverse_force_n=0.0,
            spring_compression_m=0.0,
            beam_displacement_xz_m=(0.0, 0.0),
            static_friction_margin_n=0.0,
            spring_travel_margin_m=self.parameters.spring_travel_m,
            elastic_energy_j=0.0,
            holder_work_increment_j=holder_work,
            contact_work_increment_j=contact_work,
            friction_dissipation_increment_j=0.0,
            energy_residual_j=energy_residual,
            numerical_state=NumericalState.CONVERGED,
            model_state=self._default_model_state(),
            model_warnings=warnings,
            residual=residual,
            proposal_state=proposal,
            next_state=proposal if commit else old,
            proposal_valid=True,
        )

    def _invalid_response(
        self,
        holder,
        center,
        old,
        *,
        commit,
        numerical_state,
        model_state,
        reason,
    ) -> ConstitutiveResponse:
        holder = np.asarray(holder, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
        residual = ResidualAudit(
            termination_reason=reason,
        )
        return ConstitutiveResponse(
            holder_xz_m=(float(holder[0]), float(holder[1])),
            center_xz_m=(float(center[0]), float(center[1])),
            support_xyz_m=None,
            gap_m=math.nan,
            tangent_xz=None,
            normal_xz=None,
            cap_gate_passed=False,
            near_tie=False,
            contact_state=old.contact_state,
            spring_state=old.spring_state,
            event_label=EventLabel.NONE,
            wall_on_spine_force_xz_n=(0.0, 0.0),
            spine_on_plate_wrench_about_holder=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            normal_force_n=0.0,
            tangential_force_n=0.0,
            axial_force_n=0.0,
            transverse_force_n=0.0,
            spring_compression_m=0.0,
            beam_displacement_xz_m=(0.0, 0.0),
            static_friction_margin_n=0.0,
            spring_travel_margin_m=self.parameters.spring_travel_m,
            elastic_energy_j=old.last_elastic_energy_j,
            holder_work_increment_j=0.0,
            contact_work_increment_j=0.0,
            friction_dissipation_increment_j=0.0,
            energy_residual_j=0.0,
            numerical_state=numerical_state,
            model_state=model_state,
            model_warnings=self._model_warnings(extra=reason),
            residual=residual,
            proposal_state=old,
            next_state=old,
            proposal_valid=False,
        )
