"""Common-backplate continuation with finite guided rods and unilateral contact.

This is the IJMS mechanism entry point.  The older six-DOF small-displacement
solver remains available for its analytical fixtures.  Here rotations are exact
constraints and the only shared unknowns are Y and Z. Internal rod coordinates
are retained, so a hard stop never requires an artificial large stiffness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial.transform import Rotation

from .guided_rod import GuidedRod

THEORY_VERSION = "ijms-common-backplate-2026-09-10"
SOLVER_VERSION = "guided-rod-incremental-contact-1"


@dataclass(frozen=True)
class PathSettings:
    y_mode: str = "free"
    y_locked_m: float = 0.0
    y_bounds_m: tuple[float, float] | None = None
    friction_static: float = 0.5
    friction_kinetic: float = 0.4
    max_nfev: int = 80
    residual_tolerance: float = 2e-6
    slip_tolerance_m: float = 1e-9
    contact_tolerance_m: float = 2e-9
    minimum_step_m: float = 1e-8
    event_tolerance_m: float = 1e-7
    check_body_clearance: bool = True
    swept_contact_tolerance_m: float = 1e-8
    max_path_steps: int = 5000

    def __post_init__(self):
        if self.y_mode not in {"free", "locked"}:
            raise ValueError("y_mode must be free or locked")
        if not 0 <= self.friction_kinetic <= self.friction_static:
            raise ValueError("require 0 <= kinetic friction <= static friction")


@dataclass
class PathState:
    position_m: np.ndarray
    rod_coordinates: tuple[np.ndarray, ...]
    forces_N: np.ndarray
    centers_m: np.ndarray
    rotations: tuple[np.ndarray, ...]
    normals: np.ndarray
    modes: tuple[str, ...]
    preload_N: float
    energy_J: float = 0.0
    features: tuple[Any, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        """Serializable continuation state, including friction/rotation history."""
        return dict(position_m=self.position_m.tolist(),
                    rod_coordinates=[u.tolist() for u in self.rod_coordinates],
                    forces_N=self.forces_N.tolist(), centers_m=self.centers_m.tolist(),
                    rotations=[r.tolist() for r in self.rotations], normals=self.normals.tolist(),
                    modes=list(self.modes), preload_N=self.preload_N, energy_J=self.energy_J,
                    features=list(self.features))

    @classmethod
    def from_snapshot(cls, value: dict[str, Any]) -> "PathState":
        """Restore with the same assembly, surface and solver semantics."""
        return cls(np.asarray(value["position_m"], float),
                   tuple(np.asarray(u, float) for u in value["rod_coordinates"]),
                   np.asarray(value["forces_N"], float), np.asarray(value["centers_m"], float),
                   tuple(np.asarray(r, float) for r in value["rotations"]),
                   np.asarray(value["normals"], float), tuple(value["modes"]),
                   value["preload_N"], value["energy_J"], tuple(value["features"]))


@dataclass
class EquilibriumTrial:
    state: PathState | None
    status: str
    residual: float
    iterations: int
    details: dict[str, Any] = field(default_factory=dict)


def _rotation_increment(current, previous):
    return Rotation.from_matrix(current @ previous.T).as_rotvec()


def _disk_projection(vector, radius):
    norm = np.linalg.norm(vector)
    return vector * min(1.0, max(0.0, radius) / max(norm, 1e-30))


class GuidedArray:
    """Sparse augmented equilibrium; accepted states are never changed by trials."""

    def __init__(self, rods: list[GuidedRod], surface, settings: PathSettings):
        self.rods, self.surface, self.settings = rods, surface, settings
        self.length_scale = float(np.median([r.parameters.free_length_m for r in rods]))
        self.nd = [1 + 2 * r.parameters.segments for r in rods]
        self.offsets = np.cumsum([0] + [d + 3 for d in self.nd])
        self.ng = 2 if settings.y_mode == "free" else 1
        # Columns belonging to different rods have disjoint local rows, except
        # force columns in the shared balance rows. Those shared derivatives are
        # filled explicitly below, allowing one colour per local coordinate.
        self.max_local_size = max(d + 3 for d in self.nd)

    def _jacobian(self, function, unknown):
        size = len(unknown)
        jac = lil_matrix((size, size), dtype=float)
        h = 2e-5
        for k in range(self.max_local_size):
            columns = [int(self.offsets[i] + k) for i, nd in enumerate(self.nd) if k < nd+3]
            plus, minus = unknown.copy(), unknown.copy()
            plus[columns] += h
            minus[columns] -= h
            change = (function(plus) - function(minus)) / (2*h)
            for i, nd in enumerate(self.nd):
                if k < nd + 3:
                    a, b = self.offsets[i:i+2]
                    jac[a:b, a+k] = change[a:b, None]
        for col in range(size-self.ng, size):
            plus, minus = unknown.copy(), unknown.copy()
            plus[col] += h
            minus[col] -= h
            jac[:, col] = ((function(plus)-function(minus))/(2*h))[:, None]
        # Each rod's normalised force contributes linearly to the total load.
        for i in range(len(self.nd)):
            b = self.offsets[i+1]
            jac[-1, b-1] = self._balance_scale
            if self.ng == 2:
                jac[-2, b-2] = self._balance_scale
        return csr_matrix(jac)

    def unloaded(self, position_m):
        coordinates = tuple(np.zeros(n) for n in self.nd)
        evaluations = [r.evaluate(x, derivatives=False) for r, x in zip(self.rods, coordinates)]
        centers = np.array([v.center_m for v in evaluations]) + position_m
        queries = [self.surface.query_sphere(c, r.parameters.tip_radius_m) for c, r in zip(centers, self.rods)]
        normals = np.array([query.contacts[0].normal if query.contacts else [0., 0., 1.] for query in queries])
        features = tuple(str(query.contacts[0].feature_id) if query.contacts else None for query in queries)
        return PathState(np.asarray(position_m, float), coordinates,
                         np.zeros((len(self.rods), 3)), centers,
                         tuple(v.tip_rotation for v in evaluations),
                         normals, tuple("OPEN" for _ in self.rods), 0., features=features)

    def solve(self, previous: PathState, x_m: float, preload_N: float) -> EquilibriumTrial:
        cfg, length = self.settings, self.length_scale
        force_scale = max(preload_N / len(self.rods), 1e-4)
        self._balance_scale = force_scale / max(preload_N, force_scale)
        initial = np.zeros(int(self.offsets[-1]) + self.ng)
        lower, upper = np.full(initial.size, -np.inf), np.full(initial.size, np.inf)
        for i, (rod, nd) in enumerate(zip(self.rods, self.nd)):
            a, b = self.offsets[i:i+2]
            initial[a:a+nd] = previous.rod_coordinates[i]
            initial[b-3:b] = previous.forces_N[i] / force_scale
            if previous.preload_N == 0 and preload_N > 0:
                initial[b-1] = preload_N / len(self.rods) / force_scale
                initial[a] = min(0.5 * rod.parameters.max_compression_m,
                                 preload_N / len(self.rods) * max(0., -rod.axis[2])
                                 / rod.parameters.spring_stiffness_N_per_m) / rod.parameters.free_length_m
            lower[a] = 0.
            upper[a] = rod.parameters.max_compression_m / rod.parameters.free_length_m
            # The upper stop is imposed exactly. A zero-travel rod is specified
            # as a very short positive travel only if that is the actual device.
            if upper[a] <= 0:
                raise ValueError("guided spring max_compression_m must be positive")
        initial[-1] = previous.position_m[2] / length
        if self.ng == 2:
            initial[-2] = previous.position_m[1] / length
        initial = np.maximum(lower, np.minimum(upper, initial))
        mu = np.array([cfg.friction_kinetic if mode.startswith("SLIP") else cfg.friction_static
                       for mode in previous.modes])
        cache: dict[str, Any] = {}

        def assemble(unknown):
            q = np.array([x_m, cfg.y_locked_m, unknown[-1] * length])
            if self.ng == 2:
                q[1] = unknown[-2] * length
            residual = np.zeros_like(unknown)
            rows, total_force = [], np.zeros(3)
            domain = None
            for i, (rod, nd) in enumerate(zip(self.rods, self.nd)):
                a, b = self.offsets[i:i+2]
                coordinates = unknown[a:a+nd]
                force = unknown[b-3:b] * force_scale
                total_force += force
                evaluation = rod.evaluate(coordinates, derivatives=True)
                center = evaluation.center_m + q
                query = self.surface.query_sphere(center, rod.parameters.tip_radius_m)
                contact = query.selected or (query.contacts[0] if query.contacts else None)
                if contact is None:
                    domain = query.status
                    residual[a:b] = 1e3
                    rows.append(None)
                    continue
                normal = np.asarray(contact.normal)
                gap = contact.gap_m
                normal_force = float(force @ normal)
                tangential_force = force - normal_force * normal
                moment = -rod.parameters.tip_radius_m * np.cross(normal, force)
                local = (np.asarray(evaluation.energy_gradient_J)
                         - evaluation.center_jacobian_m.T @ force
                         - evaluation.rotation_jacobian.T @ moment) / (force_scale * length)
                # R_s = -lambda_h*l0 at upper stop, +lambda_0*l0 at
                # an explicitly present shoulder. No lower reaction otherwise.
                xmax = upper[a]
                if rod.parameters.lower_stop:
                    local[0] = min(local[0], coordinates[0])
                local[0] = max(local[0], coordinates[0] - xmax)
                residual[a:a+nd] = local
                rotation_step = _rotation_increment(evaluation.tip_rotation, previous.rotations[i])
                mean_normal = normal + previous.normals[i]
                mean_normal /= max(np.linalg.norm(mean_normal), 1e-30)
                motion = (center - previous.centers_m[i]
                          + np.cross(rotation_step, -rod.parameters.tip_radius_m * mean_normal))
                tangent_motion = motion - float(motion @ normal) * normal
                trial_tangent = tangential_force - (force_scale / length) * tangent_motion
                disk = _disk_projection(trial_tangent, mu[i] * normal_force)
                # Vector equation includes a normal complementarity component;
                # it contains exactly three independent contact equations.
                residual[b-3:b] = ((tangential_force - disk) / force_scale
                                   + normal * min(gap / length, normal_force / force_scale))
                rows.append(dict(coordinates=coordinates.copy(), force=force, evaluation=evaluation,
                                 center=center, contact=contact, query=query,
                                 tangent_motion=tangent_motion, normal_force=normal_force,
                                 tangential_force=tangential_force, moment=moment))
            residual[-1] = (total_force[2] - preload_N) / max(preload_N, force_scale)
            if self.ng == 2:
                residual[-2] = total_force[1] / max(preload_N, force_scale)
            cache.update(q=q, rows=rows, residual=residual, domain=domain)
            return residual

        # Continue kinetic sliding. Return to the static cone only when material
        # motion arrests inside the kinetic disk (e.g. elastic unloading).
        result = None
        for _ in range(3):
            result = least_squares(assemble, initial, bounds=(lower, upper),
                                   jac=lambda x: self._jacobian(assemble, x), x_scale="jac",
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10,
                                   max_nfev=cfg.max_nfev, diff_step=2e-5)
            residual = assemble(result.x)
            new_mu = mu.copy()
            for i, row in enumerate(cache["rows"]):
                if row is not None and row["normal_force"] > force_scale * cfg.residual_tolerance:
                    if np.linalg.norm(row["tangent_motion"]) > cfg.slip_tolerance_m:
                        new_mu[i] = cfg.friction_kinetic
                    elif (mu[i] == cfg.friction_kinetic and
                          np.linalg.norm(row["tangential_force"]) < cfg.friction_kinetic*row["normal_force"]
                          - force_scale*cfg.residual_tolerance):
                        new_mu[i] = cfg.friction_static
            if np.array_equal(new_mu, mu):
                break
            mu, initial = new_mu, result.x
        norm = float(np.max(np.abs(residual)))
        if cache["domain"]:
            return EquilibriumTrial(None, "GEOMETRY_DOMAIN", norm, result.nfev,
                                    {"geometry_status": cache["domain"]})
        for i, row in enumerate(cache["rows"]):
            if (row["query"].selected is None and
                    abs(row["contact"].gap_m) <= cfg.contact_tolerance_m):
                return EquilibriumTrial(None, "MULTIPOINT_CONTACT_LIMIT", norm, result.nfev,
                                        {"spine_index": i})
        if norm > cfg.residual_tolerance:
            return EquilibriumTrial(None, "NUMERICAL_FAILURE", norm, result.nfev,
                                    {"optimizer_message": result.message})
        rows = cache["rows"]
        q = cache["q"]
        modes, energy, dissipation, per_spine = [], 0., 0., []
        status = "ACCEPTED"
        for rod, row in zip(self.rods, rows):
            evaluation, contact = row["evaluation"], row["contact"]
            if row["query"].selected is None and contact.gap_m <= cfg.contact_tolerance_m:
                status = "MULTIPOINT_CONTACT_LIMIT"
            n = row["normal_force"]
            if n <= force_scale * cfg.residual_tolerance:
                mode = "OPEN"
            elif np.linalg.norm(row["tangent_motion"]) > cfg.slip_tolerance_m:
                mode = "SLIP"
            else:
                mode = "STICK"
            energy += evaluation.energy_J
            dissipation += max(0., -float(row["tangential_force"] @ row["tangent_motion"]))
            compression = row["coordinates"][0] * rod.parameters.free_length_m
            hard_stop = compression >= rod.parameters.max_compression_m - cfg.contact_tolerance_m
            stress = rod.stress(row["coordinates"], row["force"], row["moment"], evaluation=evaluation)
            raw_residual = rod.generalized_residual(row["coordinates"], row["force"], row["moment"], evaluation=evaluation)
            boundary = rod.axial_boundary(row["coordinates"], raw_residual,
                                          compression_tolerance_m=cfg.contact_tolerance_m)
            if stress.model_limit:
                status = "ELASTIC_MODEL_LIMIT"
            body_gap = None
            if cfg.check_body_clearance:
                centerline, sagitta = rod.clearance_centerline(row["coordinates"],
                                                               tolerance_m=cfg.contact_tolerance_m/4)
                clearance = self.surface.query_centerline(centerline + q, rod.parameters.diameter_m/2+sagitta)
                body_gap = clearance.gap_m
                if clearance.status in {"out_of_domain", "invalid_surface"}:
                    status = "ROD_GEOMETRY_DOMAIN"
                elif body_gap is not None and body_gap < -cfg.contact_tolerance_m:
                    status = "ROD_COLLISION_LIMIT"
            modes.append(mode + ("_HARDSTOP" if hard_stop else ""))
            per_spine.append(dict(force_N=row["force"].tolist(), P_N=float(row["force"][2]),
                                  T_N=float(-row["force"][0]), N_N=float(n),
                                  compression_m=float(compression),
                                  exposed_length_m=float(rod.parameters.free_length_m-compression),
                                  gap_m=float(contact.gap_m), mode=modes[-1],
                                  body_gap_m=body_gap,
                                  max_stress_upper_Pa=stress.max_section_von_mises_upper_Pa,
                                  stress_utilization=stress.utilization,
                                  guide_moment_Nm=stress.guide_moment_Nm.tolist(),
                                  travel_utilization=float(compression/rod.parameters.max_compression_m),
                                  spring_branch=boundary.branch,
                                  upper_stop_reaction_N=boundary.upper_reaction_N,
                                  lower_stop_reaction_N=boundary.lower_reaction_N,
                                  spring_energy_J=evaluation.spring_energy_J,
                                  bending_energy_J=evaluation.bending_energy_J,
                                  feature_id=str(contact.feature_id),
                                  center_m=row["center"].tolist(),
                                  contact_point_m=np.asarray(contact.contact_point_m).tolist(),
                                  normal=np.asarray(contact.normal).tolist(),
                                  tangent_increment_m=row["tangent_motion"].tolist()))
        if cfg.y_bounds_m and not cfg.y_bounds_m[0] <= q[1] <= cfg.y_bounds_m[1]:
            status = "Y_DOMAIN_LIMIT"
        total = sum((row["force"] for row in rows), np.zeros(3))
        moments = sum((np.cross(np.asarray(row["contact"].contact_point_m) - q, row["force"])
                       for row in rows), np.zeros(3))
        work = (-0.5 * (total[0] + previous.forces_N[:, 0].sum()) * (x_m - previous.position_m[0])
                -0.5 * (preload_N + previous.preload_N) * (q[2] - previous.position_m[2]))
        diagnostics = dict(residual=norm, nfev=result.nfev, per_spine=per_spine,
                           total_force_N=total.tolist(), total_moment_Nm=moments.tolist(),
                           incremental_external_work_J=float(work),
                           incremental_friction_dissipation_J=float(dissipation),
                           incremental_energy_residual_J=float(work-(energy-previous.energy_J)-dissipation),
                           quasistatic_stability="NOT_EVALUATED", dynamic_stability="OUT_OF_SCOPE")
        state = PathState(q.copy(), tuple(row["coordinates"] for row in rows),
                          np.array([row["force"] for row in rows]),
                          np.array([row["center"] for row in rows]),
                          tuple(row["evaluation"].tip_rotation for row in rows),
                          np.array([row["contact"].normal for row in rows]), tuple(modes),
                          preload_N, float(energy),
                          tuple(str(row["contact"].feature_id) for row in rows), diagnostics)
        return EquilibriumTrial(state, status, norm, result.nfev)
