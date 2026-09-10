"""Energy-consistent, retracting, inextensible rod for the IJMS model.

The rod has a fixed guide tangent and two bending coordinates at each of
``segments`` tangent nodes. Adjacent tangents are joined by a great-circle arc;
each arc has its exact length and bending energy. Thus this is a spatial rod
discretization, not a cantilever stiffness updated with the current length.
Its accuracy is controlled by segment refinement.

The dimensionless state is ``[s / l0, beta_1, gamma_1, ...]``. The spring stores
``k*s**2/2``. A tapered needle integrates its actual section rigidity over
each exposed material interval; the taper does not stretch during retraction.
Differentiating
the SAME energy and kinematics retains the moving-boundary configurational
force. The lower compression bound does not generate an anti-pullout reaction
unless ``lower_stop`` is explicitly enabled by the assembly definition.

Lengths are metres, forces newtons, moments newton-metres, angles radians.
Contact forces are wall-on-rod. A sphere contact supplies the centre moment
``-radius * cross(normal, force)``; it is included exactly once in virtual work.
The tip frame is transported without material twist from the fixed guide.
Axial material strain and torsional material strain are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _skew(v: FloatArray) -> FloatArray:
    x, y, z = v
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def rotation_increment(rotation: ArrayLike, previous_rotation: ArrayLike) -> FloatArray:
    """Spatial logarithmic rotation increment (radians), including near pi."""
    relative = np.asarray(rotation, dtype=float) @ np.asarray(previous_rotation, dtype=float).T
    axial = np.array((relative[2, 1] - relative[1, 2],
                      relative[0, 2] - relative[2, 0],
                      relative[1, 0] - relative[0, 1])) / 2.0
    sine = float(np.linalg.norm(axial))
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arctan2(sine, cosine))
    if sine > 1e-8:
        return axial * (angle / sine)
    if cosine > 0.0:
        return axial
    # A symmetric eigenproblem avoids dividing by sin(pi).
    _, vectors = np.linalg.eigh((relative + relative.T) / 2.0)
    axis = vectors[:, -1]
    if float(axis @ axial) < 0.0:
        axis = -axis
    return angle * axis


@dataclass(frozen=True)
class GuidedRodParameters:
    free_length_m: float
    diameter_m: float
    young_modulus_Pa: float
    spring_stiffness_N_per_m: float
    max_compression_m: float
    tip_radius_m: float
    segments: int = 3
    lower_stop: bool = False
    allowable_stress_Pa: float | None = None
    taper_length_m: float = 0.0
    mount_type: str = "spring"

    def __post_init__(self) -> None:
        if min(self.free_length_m, self.diameter_m, self.young_modulus_Pa,
               self.tip_radius_m) <= 0:
            raise ValueError("Rod length, diameter, modulus and radius must be positive")
        if self.mount_type not in ("spring", "fixed"):
            raise ValueError("mount_type must be 'spring' or 'fixed'")
        if self.mount_type == "spring" and self.spring_stiffness_N_per_m <= 0:
            raise ValueError("A spring installation requires positive spring stiffness")
        if self.max_compression_m < 0:
            raise ValueError("Nominal compression travel must be nonnegative")
        if self.taper_length_m < 0 or (self.taper_length_m > 0
                                      and self.tip_radius_m > self.diameter_m / 2):
            raise ValueError("Taper length must be nonnegative and its tip no wider than the shaft")
        if self.segments < 1:
            raise ValueError("At least one rod segment is required")
        if self.allowable_stress_Pa is not None and self.allowable_stress_Pa <= 0:
            raise ValueError("Allowable stress must be positive when supplied")

    @property
    def area_m2(self) -> float:
        return float(np.pi * self.diameter_m**2 / 4.0)

    @property
    def second_moment_m4(self) -> float:
        return float(np.pi * self.diameter_m**4 / 64.0)

    @property
    def bending_rigidity_Nm2(self) -> float:
        """Main shaft rigidity; a tapered rod does not use this along its whole length."""
        return self.young_modulus_Pa * self.second_moment_m4

    @property
    def compression_limit_m(self) -> float:
        """Available compression domain, without changing the rated spring stroke.

        The exposed-length endpoint is open: l=0 is a geometry limit, not a
        load-carrying hard stop. Fixed installations have no compression DOF.
        """
        return 0.0 if self.mount_type == "fixed" else min(self.max_compression_m, self.free_length_m)

    @property
    def geometry_limited_compression(self) -> bool:
        return self.mount_type == "spring" and self.max_compression_m >= self.free_length_m

    def section_radius_m(self, distance_from_tip_m: ArrayLike) -> FloatArray:
        """Section radius at material coordinate u, measured from ball centre toward guide."""
        u = np.asarray(distance_from_tip_m, dtype=float)
        if self.taper_length_m == 0:
            return np.full_like(u, self.diameter_m / 2)
        fraction = np.clip(u / self.taper_length_m, 0.0, 1.0)
        return self.tip_radius_m + (self.diameter_m / 2 - self.tip_radius_m) * fraction

    def section_area_m2(self, distance_from_tip_m: ArrayLike) -> FloatArray:
        return np.pi * self.section_radius_m(distance_from_tip_m)**2

    def section_second_moment_m4(self, distance_from_tip_m: ArrayLike) -> FloatArray:
        return np.pi * self.section_radius_m(distance_from_tip_m)**4 / 4

    def integrated_second_moment_m5(self, lower_m: ArrayLike, upper_m: ArrayLike) -> FloatArray:
        """Exact integral of I(u) on exposed material intervals (nonnegative u)."""
        lo, hi = np.asarray(lower_m, dtype=float), np.asarray(upper_m, dtype=float)
        if self.taper_length_m == 0 or self.tip_radius_m == self.diameter_m / 2:
            return self.second_moment_m4 * (hi - lo)
        slope = (self.diameter_m / 2 - self.tip_radius_m) / self.taper_length_m
        tlo, thi = np.minimum(lo, self.taper_length_m), np.minimum(hi, self.taper_length_m)
        alo, ahi = self.tip_radius_m + slope * tlo, self.tip_radius_m + slope * thi
        # Factoring the difference of fifth powers avoids cancellation in short intervals.
        fifth_difference = (ahi - alo) * sum(ahi**j * alo**(4-j) for j in range(5))
        taper_integral = np.pi * fifth_difference / (20 * slope)
        shaft_integral = self.second_moment_m4 * (np.maximum(hi-self.taper_length_m, 0)
                                                  - np.maximum(lo-self.taper_length_m, 0))
        return taper_integral + shaft_integral


@dataclass(frozen=True)
class RodEvaluation:
    center_m: FloatArray
    tip_rotation: FloatArray
    energy_J: float
    spring_energy_J: float
    bending_energy_J: float
    compression_m: float
    exposed_length_m: float
    center_jacobian_m: FloatArray | None
    rotation_jacobian: FloatArray | None
    energy_gradient_J: FloatArray | None
    centerline_m: FloatArray
    tangents: FloatArray


@dataclass(frozen=True)
class RodStress:
    guide_moment_Nm: FloatArray
    guide_axial_force_N: float
    guide_bending_moment_Nm: float
    guide_torsional_moment_Nm: float
    max_section_von_mises_upper_Pa: float
    utilization: float | None
    model_limit: bool
    # Values are upper bounds over the cross-section at these sampled positions;
    # they do not certify an exact continuous spatial maximum between samples.
    section_position_m: FloatArray
    section_von_mises_upper_Pa: FloatArray


@dataclass(frozen=True)
class AxialBoundary:
    branch: str
    residual_N: float
    upper_reaction_N: float
    lower_reaction_N: float
    admissible: bool


class GuidedRod:
    """Rod kinematics and virtual-work residual for a global contact solver.

    ``evaluate`` deliberately does not clip trial compression to a stop. The
    outer solver enforces bounds and complementarity. Derivatives use central
    differences for angular coordinates and exact compression derivatives;
    they are derivatives of this discretized energy and geometry throughout.
    """

    derivative_step = 2e-5

    def __init__(self, parameters: GuidedRodParameters,
                 guide_position_m: ArrayLike, axis: ArrayLike):
        self.parameters = parameters
        self.guide_position_m = np.asarray(guide_position_m, dtype=float).copy()
        direction = np.asarray(axis, dtype=float)
        self.axis = direction / np.linalg.norm(direction)
        # A deterministic right-handed guide frame; no preferred bending plane.
        reference = np.eye(3)[int(np.argmin(np.abs(self.axis)))]
        transverse = reference - self.axis * float(reference @ self.axis)
        transverse /= np.linalg.norm(transverse)
        self.guide_rotation = np.column_stack((self.axis, transverse,
                                              np.cross(self.axis, transverse)))

    @property
    def dimension(self) -> int:
        return 1 + 2 * self.parameters.segments

    @property
    def compression_bounds(self) -> tuple[float, float]:
        return (0.0, self.parameters.compression_limit_m / self.parameters.free_length_m)

    @property
    def segment_fractions(self) -> FloatArray:
        """Guide-to-tip node positions; tapered rods resolve their thin end more finely."""
        fractions = np.linspace(0.0, 1.0, self.parameters.segments + 1)
        return fractions if self.parameters.taper_length_m == 0 else 1.0 - (1.0-fractions)**2

    def segment_boundaries_m(self, state: ArrayLike) -> FloatArray:
        """Arc lengths from the guide to all tangent nodes, including the ball centre."""
        return self.parameters.free_length_m * (1.0 - np.asarray(state)[0]) * self.segment_fractions

    def zero_state(self) -> FloatArray:
        return np.zeros(self.dimension)

    def _kinematics(self, state: FloatArray) -> tuple[FloatArray, FloatArray, float, FloatArray, FloatArray]:
        p = self.parameters
        length = p.free_length_m * (1.0 - state[0])
        if length <= 0.0:
            raise ValueError("Rod trial has no exposed length")
        angular = state[1:].reshape(p.segments, 2)
        magnitude = np.linalg.norm(angular, axis=1)
        transverse = angular @ self.guide_rotation[:, 1:].T
        tangent = (np.cos(magnitude)[:, None] * self.axis
                   + np.sinc(magnitude / np.pi)[:, None] * transverse)
        tangent = np.vstack((self.axis, tangent))
        boundaries = self.segment_boundaries_m(state)
        segment_lengths = np.diff(boundaries)
        section_integrals = p.integrated_second_moment_m5(length-boundaries[1:], length-boundaries[:-1])
        centerline = [self.guide_position_m.copy()]
        rotation = self.guide_rotation.copy()
        energy = 0.0
        for left, right, h, integral in zip(tangent[:-1], tangent[1:], segment_lengths, section_integrals):
            cross = np.cross(left, right)
            cosine = float(np.clip(left @ right, -1.0, 1.0))
            if cosine < -1.0 + 1e-9:
                raise ValueError("Adjacent rod tangents are antipodal; refine the rod discretization")
            angle = float(np.arctan2(np.linalg.norm(cross), cosine))
            factor = (0.5 + angle**2 / 24.0 + angle**4 / 240.0
                      if angle < 1e-4 else np.tan(angle / 2.0) / angle)
            centerline.append(centerline[-1] + h * factor * (left + right))
            cross_matrix = _skew(cross)
            rotation = (np.eye(3) + cross_matrix
                        + cross_matrix @ cross_matrix / (1.0 + cosine)) @ rotation
            energy += p.young_modulus_Pa * integral * angle**2 / (2.0 * h**2)
        return centerline[-1], rotation, float(energy), np.asarray(centerline), tangent

    def _angular_kinematics_batch(self, state: FloatArray, angular_states: FloatArray
                                  ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """The same circular arcs for angle trials sharing one compression.

        All central-difference perturbations are evaluated together. Material
        intervals are shared, while each trial retains its own tangents,
        ordered frame transport, centre and section-integrated energy.
        """
        p = self.parameters
        count = len(angular_states)
        angular = angular_states.reshape(count, p.segments, 2)
        magnitude = np.linalg.norm(angular, axis=2)
        transverse = angular @ self.guide_rotation[:, 1:].T
        tangents = np.empty((count, p.segments+1, 3))
        tangents[:, 0] = self.axis
        tangents[:, 1:] = (np.cos(magnitude)[:, :, None] * self.axis
                          + np.sinc(magnitude/np.pi)[:, :, None] * transverse)
        left, right = tangents[:, :-1], tangents[:, 1:]
        crosses = np.cross(left, right)
        cosines = np.clip(np.matmul(left[..., None, :], right[..., :, None])[..., 0, 0], -1., 1.)
        if np.any(cosines < -1.+1e-9):
            raise ValueError("Adjacent rod tangents are antipodal; refine the rod discretization")
        angles = np.arctan2(np.linalg.norm(crosses, axis=2), cosines)
        factors = .5 + angles**2/24. + angles**4/240.
        finite = angles >= 1e-4
        factors[finite] = np.tan(angles[finite]/2.)/angles[finite]

        skew = np.zeros((count, p.segments, 3, 3))
        skew[:, :, 0, 1], skew[:, :, 0, 2] = -crosses[:, :, 2], crosses[:, :, 1]
        skew[:, :, 1, 0], skew[:, :, 1, 2] = crosses[:, :, 2], -crosses[:, :, 0]
        skew[:, :, 2, 0], skew[:, :, 2, 1] = -crosses[:, :, 1], crosses[:, :, 0]
        rotations = np.eye(3) + skew + (skew @ skew)/(1.+cosines[:, :, None, None])
        length = p.free_length_m*(1.-state[0])
        boundaries = self.segment_boundaries_m(state)
        segment_lengths = np.diff(boundaries)
        section_integrals = p.integrated_second_moment_m5(length-boundaries[1:], length-boundaries[:-1])
        centers = np.broadcast_to(self.guide_position_m, (count, 3)).copy()
        rotation = np.broadcast_to(self.guide_rotation, (count, 3, 3)).copy()
        energy = np.zeros(count)
        for j, (h, integral) in enumerate(zip(segment_lengths, section_integrals)):
            centers += h*factors[:, j, None]*(left[:, j]+right[:, j])
            rotation = rotations[:, j] @ rotation
            energy += p.young_modulus_Pa*integral*angles[:, j]**2/(2.*h**2)
        return centers, rotation, energy

    def evaluate(self, state: ArrayLike, derivatives: bool = True) -> RodEvaluation:
        x = np.asarray(state, dtype=float)
        p = self.parameters
        center, rotation, bending, centerline, tangent = self._kinematics(x)
        compression = p.free_length_m * x[0]
        length = p.free_length_m - compression
        stiffness = p.spring_stiffness_N_per_m if p.mount_type == "spring" else 0.0
        spring = 0.5 * stiffness * compression**2
        jacobian = rotation_jacobian = gradient = None
        if derivatives:
            jacobian = np.zeros((3, self.dimension))
            rotation_jacobian = np.zeros((3, self.dimension))
            gradient = np.zeros(self.dimension)
            jacobian[:, 0] = -p.free_length_m * (center - self.guide_position_m) / length
            boundaries = self.segment_boundaries_m(x)
            hi, lo = length-boundaries[:-1], length-boundaries[1:]
            intervals = p.integrated_second_moment_m5(lo, hi)
            angles = np.arctan2(np.linalg.norm(np.cross(tangent[:-1], tangent[1:]), axis=1),
                                np.clip(np.einsum("ij,ij->i", tangent[:-1], tangent[1:]), -1., 1.))
            # Nodes scale with the exposed domain. I(u) remains attached to the
            # material: d/dl ∫[lo(l),hi(l)] I(u)du includes both moving endpoints.
            boundary_derivative = (p.section_second_moment_m4(hi) * hi
                                   - p.section_second_moment_m4(lo) * lo) / length
            dbending_dlength = np.sum(p.young_modulus_Pa * angles**2 / (2*np.diff(boundaries)**2)
                                      * (boundary_derivative - 2*intervals/length))
            gradient[0] = p.free_length_m * (stiffness * compression - dbending_dlength)
            n = self.dimension-1
            steps = self.derivative_step*np.maximum(1., np.abs(x[1:]))
            angle_trials = np.tile(x[1:], (2*n, 1))
            indices = np.arange(n)
            angle_trials[indices, indices] += steps
            angle_trials[n+indices, indices] -= steps
            centers, rotations, energies = self._angular_kinematics_batch(x, angle_trials)
            jacobian[:, 1:] = ((centers[:n]-centers[n:])/(2.*steps[:, None])).T
            spin = ((rotations[:n]-rotations[n:])/(2.*steps[:, None, None])) @ rotation.T
            rotation_jacobian[:, 1:] = np.stack((spin[:, 2, 1]-spin[:, 1, 2],
                                                 spin[:, 0, 2]-spin[:, 2, 0],
                                                 spin[:, 1, 0]-spin[:, 0, 1]))/2.
            gradient[1:] = (energies[:n]-energies[n:])/(2.*steps)
        return RodEvaluation(center, rotation, spring + bending, spring, bending,
                             float(compression), float(length), jacobian,
                             rotation_jacobian, gradient, centerline, tangent)

    def generalized_residual(self, state: ArrayLike, force_N: ArrayLike,
                             moment_at_center_Nm: ArrayLike = (0.0, 0.0, 0.0),
                             evaluation: RodEvaluation | None = None) -> FloatArray:
        """Internal minus external generalized forces, in joules/radian (J).

        The axial physical-force residual is ``result[0] / l0``. At an upper
        stop it is balanced by a nonnegative compression-stop reaction. At the
        free-extension boundary it must remain zero when there is no shoulder.
        """
        e = self.evaluate(state) if evaluation is None else evaluation
        return (e.energy_gradient_J - e.center_jacobian_m.T @ np.asarray(force_N)
                - e.rotation_jacobian.T @ np.asarray(moment_at_center_Nm))

    def residual_tangent(self, state: ArrayLike, force_N: ArrayLike,
                         moment_at_center_Nm: ArrayLike = (0.0, 0.0, 0.0)) -> FloatArray:
        """Fixed applied-wrench Jacobian of the complete virtual-work residual.

        Contact-normal, friction and constraint derivatives belong to the
        global solver. A fixed spatial torque is a follower loading here, so
        this matrix need not be symmetric and is not a stability certificate.
        """
        x = np.asarray(state, dtype=float)
        columns = []
        for j in range(self.dimension):
            h = 5e-5 * max(1.0, abs(x[j]))
            plus, minus = x.copy(), x.copy()
            plus[j] += h
            minus[j] -= h
            columns.append((self.generalized_residual(plus, force_N, moment_at_center_Nm)
                            - self.generalized_residual(minus, force_N, moment_at_center_Nm)) / (2.0 * h))
        return np.column_stack(columns)

    def axial_boundary(self, state: ArrayLike, residual: ArrayLike,
                       force_tolerance_N: float = 1e-8,
                       compression_tolerance_m: float = 1e-10) -> AxialBoundary:
        p = self.parameters
        s = float(np.asarray(state)[0] * p.free_length_m)
        residual_N = float(np.asarray(residual)[0] / p.free_length_m)
        if p.mount_type == "fixed":
            return AxialBoundary("FIXED", residual_N, max(-residual_N, 0.0),
                                 max(residual_N, 0.0), abs(s) <= compression_tolerance_m)
        if p.geometry_limited_compression and s >= p.free_length_m - compression_tolerance_m:
            return AxialBoundary("EXPOSED_LENGTH_LIMIT", residual_N, 0.0, 0.0, False)
        upper = s >= p.max_compression_m - compression_tolerance_m
        lower = s <= compression_tolerance_m
        within = -compression_tolerance_m <= s <= p.max_compression_m + compression_tolerance_m
        if upper and residual_N <= force_tolerance_N:
            return AxialBoundary("HARDSTOP", residual_N, max(-residual_N, 0.0), 0.0, within)
        if lower and p.lower_stop and residual_N >= -force_tolerance_N:
            return AxialBoundary("LOWER_STOP", residual_N, 0.0, max(residual_N, 0.0), within)
        return AxialBoundary("FREE_EXTENSION" if lower else "INTERIOR", residual_N,
                             0.0, 0.0, within and abs(residual_N) <= force_tolerance_N)

    def sample_centerline(self, state: ArrayLike,
                          samples_per_segment: int = 4) -> tuple[FloatArray, FloatArray]:
        """Sample exact circular segments for shaft clearance and stress queries."""
        boundaries = self.segment_boundaries_m(state)
        distances = np.concatenate(([0.0], *(np.linspace(left, right, samples_per_segment+1)[1:]
                                            for left, right in zip(boundaries[:-1], boundaries[1:]))))
        return self.centerline_at(state, distances)

    def centerline_at(self, state: ArrayLike,
                      distances_from_guide_m: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Evaluate position and unit tangent at arbitrary exposed arc lengths.

        Input distances run from zero at the guide to l at the sphere centre;
        the corresponding section material coordinate is u=l-distance.
        """
        x = np.asarray(state, dtype=float)
        _, _, _, nodes, tangents = self._kinematics(x)
        boundaries = self.segment_boundaries_m(x)
        distances = np.atleast_1d(np.asarray(distances_from_guide_m, dtype=float))
        indices = np.clip(np.searchsorted(boundaries, distances, side="right")-1,
                          0, self.parameters.segments-1)
        points, directions = np.empty((len(distances), 3)), np.empty((len(distances), 3))
        for i, (origin, left, right) in enumerate(zip(nodes[:-1], tangents[:-1], tangents[1:])):
            selected = indices == i
            if not np.any(selected):
                continue
            h = boundaries[i+1] - boundaries[i]
            u = (distances[selected] - boundaries[i]) / h
            cross = np.cross(left, right)
            sine = float(np.linalg.norm(cross))
            angle = float(np.arctan2(sine, np.clip(left @ right, -1.0, 1.0)))
            if angle < 1e-10:
                points[selected] = origin + h * u[:, None] * left
                directions[selected] = left
            else:
                bend_direction = np.cross(cross / sine, left)
                points[selected] = origin + h / angle * (np.sin(u*angle)[:, None] * left
                                                         + 2*np.sin(u*angle/2)[:, None]**2 * bend_direction)
                directions[selected] = (np.cos(u*angle)[:, None] * left
                                        + np.sin(u*angle)[:, None] * bend_direction)
        return points, directions

    def clearance_centerline(self, state: ArrayLike, subdivisions: int = 4,
                             tolerance_m: float | None = None) -> tuple[FloatArray, float]:
        """Polyline and maximum arc/chord deviation for conservative clearance.

        Query capsules of radius ``diameter/2 + returned_sagitta`` around these
        chords to contain the actual circular-arc rod. Inflation can report a
        collision before the curved rod actually touches; refine subdivisions
        (or request a smaller tolerance) before interpreting such a boundary.
        This envelope describes the exposed rod centreline, including its end;
        a rod/sphere assembly must account for any shaft hidden inside the tip.
        """
        x = np.asarray(state, dtype=float)
        _, _, _, _, tangents = self._kinematics(x)
        lengths = np.diff(self.segment_boundaries_m(x))
        angles = np.array([np.arctan2(np.linalg.norm(np.cross(left, right)),
                                     np.clip(left @ right, -1., 1.))
                           for left, right in zip(tangents[:-1], tangents[1:])])
        curved = angles > 1e-10
        arc_radius = lengths[curved] / angles[curved]
        count = max(int(subdivisions), 1)
        if tolerance_m is not None:
            if tolerance_m <= 0:
                raise ValueError("Clearance approximation tolerance must be positive")
            half_angle = 2 * np.arcsin(np.sqrt(np.minimum(tolerance_m / (2 * arc_radius), 1.)))
            if len(half_angle):
                count = max(count, int(np.ceil(np.max(angles[curved] / (2 * half_angle)))))
        error = (float(np.max(2 * arc_radius * np.sin(angles[curved] / (4 * count))**2))
                 if np.any(curved) else 0.)
        points, _ = self.sample_centerline(x, count)
        return points, error

    def stress(self, state: ArrayLike, force_N: ArrayLike,
               moment_at_center_Nm: ArrayLike = (0.0, 0.0, 0.0),
               evaluation: RodEvaluation | None = None,
               samples_per_segment: int = 4) -> RodStress:
        """Circular-section stress bounds using current geometry and all moments."""
        e = self.evaluate(state, derivatives=False) if evaluation is None else evaluation
        p = self.parameters
        force = np.asarray(force_N, dtype=float)
        moment = np.asarray(moment_at_center_Nm, dtype=float)
        points, tangents = self.sample_centerline(state, samples_per_segment)
        moments = np.cross(e.center_m - points, force) + moment
        axial = tangents @ force
        transverse_force = force - axial[:, None] * tangents
        torsion = np.einsum("ij,ij->i", moments, tangents)
        bending_moments = moments - torsion[:, None] * tangents
        boundaries = self.segment_boundaries_m(state)
        distances = np.concatenate(([0.0], *(np.linspace(left, right, samples_per_segment+1)[1:]
                                            for left, right in zip(boundaries[:-1], boundaries[1:]))))
        material_coordinate = e.exposed_length_m - distances
        radii = p.section_radius_m(material_coordinate)
        area = p.section_area_m2(material_coordinate)
        inertia = p.section_second_moment_m4(material_coordinate)
        normal_stress = (np.abs(axial) / area
                         + np.linalg.norm(bending_moments, axis=1) * radii / inertia)
        shear_stress = (4.0 * np.linalg.norm(transverse_force, axis=1) / (3.0 * area)
                        + np.abs(torsion) * radii / (2.0 * inertia))
        vm = np.sqrt(normal_stress**2 + 3.0 * shear_stress**2)
        maximum = float(np.max(vm))
        utilization = None if p.allowable_stress_Pa is None else maximum / p.allowable_stress_Pa
        return RodStress(moments[0], float(axial[0]), float(np.linalg.norm(bending_moments[0])),
                         abs(float(torsion[0])), maximum, utilization,
                         utilization is not None and utilization >= 1.0, points, vm)


def sphere_contact_moment(radius_m: float, normal: ArrayLike, force_N: ArrayLike) -> FloatArray:
    """Translate a single point-contact force to the sphere centre once."""
    return -radius_m * np.cross(np.asarray(normal, dtype=float), np.asarray(force_N, dtype=float))


def linear_reference_compliance(parameters: GuidedRodParameters, axis: ArrayLike,
                                exposed_length_m: float | None = None) -> FloatArray:
    """Fixed-length small-deflection reference, without the sphere-end moment.

    The transverse compliance is ∫u²/[E I(u)]du, with u measured from the tip.
    The axial term is 1/k on the interior spring branch and zero for a fixed
    mount. This is not the tangent of the finite rod under preload.
    """
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    length = parameters.free_length_m if exposed_length_m is None else exposed_length_m
    axial = np.outer(a, a)
    p = parameters
    if p.taper_length_m == 0 or p.tip_radius_m == p.diameter_m / 2:
        transverse_compliance = length**3 / (3.0 * p.bending_rigidity_Nm2)
    else:
        tip_length = min(length, p.taper_length_m)
        slope = (p.diameter_m / 2-p.tip_radius_m) / p.taper_length_m
        # ∫u²/(r+b*u)^4 du = u³/[3*r*(r+b*u)^3].
        # This form also remains stable for a very short exposed tip interval.
        taper_compliance = (4.0 / (np.pi*p.young_modulus_Pa) * tip_length**3
                            / (3*p.tip_radius_m*(p.tip_radius_m+slope*tip_length)**3))
        shaft_compliance = (length**3-tip_length**3) / (3*p.bending_rigidity_Nm2)
        transverse_compliance = taper_compliance + shaft_compliance
    axial_compliance = 1.0 / p.spring_stiffness_N_per_m if p.mount_type == "spring" else 0.0
    return axial_compliance * axial + transverse_compliance * (np.eye(3)-axial)
