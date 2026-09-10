"""Energy-consistent, retracting, inextensible rod for the IJMS model.

The rod has a fixed guide tangent and two bending coordinates at each of
``segments`` tangent nodes. Adjacent tangents are joined by a great-circle arc;
each arc has its exact length and bending energy. Thus this is a spatial rod
discretization, not a cantilever stiffness updated with the current length.
Its accuracy is controlled by segment refinement.

The dimensionless state is ``[s / l0, beta_1, gamma_1, ...]``. The spring stores
``k*s**2/2`` and the bending energy scales with ``1/(l0-s)``. Differentiating
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

    def __post_init__(self) -> None:
        if min(self.free_length_m, self.diameter_m, self.young_modulus_Pa,
               self.spring_stiffness_N_per_m, self.tip_radius_m) <= 0:
            raise ValueError("Rod length, diameter, modulus, spring stiffness and radius must be positive")
        if not 0 <= self.max_compression_m < self.free_length_m:
            raise ValueError("Compression travel must satisfy 0 <= smax < l0")
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
        return self.young_modulus_Pa * self.second_moment_m4


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
        return (0.0, self.parameters.max_compression_m / self.parameters.free_length_m)

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
        h = length / p.segments
        centerline = [self.guide_position_m.copy()]
        rotation = self.guide_rotation.copy()
        curvature_sum = 0.0
        for left, right in zip(tangent[:-1], tangent[1:]):
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
            curvature_sum += angle**2
        energy = p.bending_rigidity_Nm2 * curvature_sum / (2.0 * h)
        return centerline[-1], rotation, float(energy), np.asarray(centerline), tangent

    def evaluate(self, state: ArrayLike, derivatives: bool = True) -> RodEvaluation:
        x = np.asarray(state, dtype=float)
        p = self.parameters
        center, rotation, bending, centerline, tangent = self._kinematics(x)
        compression = p.free_length_m * x[0]
        length = p.free_length_m - compression
        spring = 0.5 * p.spring_stiffness_N_per_m * compression**2
        jacobian = rotation_jacobian = gradient = None
        if derivatives:
            jacobian = np.zeros((3, self.dimension))
            rotation_jacobian = np.zeros((3, self.dimension))
            gradient = np.zeros(self.dimension)
            jacobian[:, 0] = -p.free_length_m * (center - self.guide_position_m) / length
            gradient[0] = (p.spring_stiffness_N_per_m * compression * p.free_length_m
                           + bending * p.free_length_m / length)
            for j in range(1, self.dimension):
                h = self.derivative_step * max(1.0, abs(x[j]))
                plus, minus = x.copy(), x.copy()
                plus[j] += h
                minus[j] -= h
                cp, rp, up, _, _ = self._kinematics(plus)
                cm, rm, um, _, _ = self._kinematics(minus)
                jacobian[:, j] = (cp - cm) / (2.0 * h)
                spin = ((rp - rm) / (2.0 * h)) @ rotation.T
                rotation_jacobian[:, j] = np.array((spin[2, 1] - spin[1, 2],
                                                    spin[0, 2] - spin[2, 0],
                                                    spin[1, 0] - spin[0, 1])) / 2.0
                gradient[j] = (up - um) / (2.0 * h)
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
        x = np.asarray(state, dtype=float)
        _, _, _, nodes, tangents = self._kinematics(x)
        h = self.parameters.free_length_m * (1.0 - x[0]) / self.parameters.segments
        points = [nodes[0]]
        directions = [tangents[0]]
        for origin, left, right in zip(nodes[:-1], tangents[:-1], tangents[1:]):
            cross = np.cross(left, right)
            sine = float(np.linalg.norm(cross))
            angle = float(np.arctan2(sine, np.clip(left @ right, -1.0, 1.0)))
            if angle < 1e-10:
                for j in range(1, samples_per_segment + 1):
                    u = j / samples_per_segment
                    points.append(origin + h * u * left)
                    directions.append(left)
            else:
                bend_direction = np.cross(cross / sine, left)
                for j in range(1, samples_per_segment + 1):
                    u = j / samples_per_segment
                    points.append(origin + h / angle * (np.sin(u * angle) * left
                                                         + 2 * np.sin(u * angle / 2.0)**2 * bend_direction))
                    directions.append(np.cos(u * angle) * left + np.sin(u * angle) * bend_direction)
        return np.asarray(points), np.asarray(directions)

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
        h = self.parameters.free_length_m * (1.0 - x[0]) / self.parameters.segments
        angles = np.array([np.arctan2(np.linalg.norm(np.cross(left, right)),
                                     np.clip(left @ right, -1., 1.))
                           for left, right in zip(tangents[:-1], tangents[1:])])
        curved = angles > 1e-10
        arc_radius = h / angles[curved]
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
        normal_stress = (np.abs(axial) / p.area_m2
                         + np.linalg.norm(bending_moments, axis=1) * p.diameter_m
                         / (2.0 * p.second_moment_m4))
        shear_stress = (4.0 * np.linalg.norm(transverse_force, axis=1) / (3.0 * p.area_m2)
                        + np.abs(torsion) * p.diameter_m / (4.0 * p.second_moment_m4))
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

    This reference is valid on the interior spring branch. It is not the
    tangent of the finite rod under preload and is never used as that tangent.
    """
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    length = parameters.free_length_m if exposed_length_m is None else exposed_length_m
    axial = np.outer(a, a)
    transverse_compliance = length**3 / (3.0 * parameters.bending_rigidity_Nm2)
    return axial / parameters.spring_stiffness_N_per_m + transverse_compliance * (np.eye(3) - axial)
