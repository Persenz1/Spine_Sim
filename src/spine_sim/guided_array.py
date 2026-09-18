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
from scipy.optimize import least_squares, brentq
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial.transform import Rotation

from .guided_rod import GuidedRod

THEORY_VERSION = "ijms-tapered-common-backplate-2026-09-10"
SOLVER_VERSION = "guided-rod-incremental-contact-3"


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
    friction_solver: str = "projection"
    multipoint_contact: bool = False

    def __post_init__(self):
        if self.y_mode not in {"free", "locked"}:
            raise ValueError("y_mode must be free or locked")
        if not 0 <= self.friction_kinetic <= self.friction_static:
            raise ValueError("require 0 <= kinetic friction <= static friction")
        if self.friction_solver not in {"projection", "active_set"}:
            raise ValueError("friction_solver must be projection or active_set")


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
    contact_history: tuple[tuple[dict[str, Any], ...], ...] = ()

    def snapshot(self) -> dict[str, Any]:
        """Serializable continuation state, including friction/rotation history."""
        return dict(position_m=self.position_m.tolist(),
                    rod_coordinates=[u.tolist() for u in self.rod_coordinates],
                    forces_N=self.forces_N.tolist(), centers_m=self.centers_m.tolist(),
                    rotations=[r.tolist() for r in self.rotations], normals=self.normals.tolist(),
                    modes=list(self.modes), preload_N=self.preload_N, energy_J=self.energy_J,
                    features=list(self.features), contact_history=self.contact_history)

    @classmethod
    def from_snapshot(cls, value: dict[str, Any]) -> "PathState":
        """Restore with the same assembly, surface and solver semantics."""
        return cls(np.asarray(value["position_m"], float),
                   tuple(np.asarray(u, float) for u in value["rod_coordinates"]),
                   np.asarray(value["forces_N"], float), np.asarray(value["centers_m"], float),
                   tuple(np.asarray(r, float) for r in value["rotations"]),
                   np.asarray(value["normals"], float), tuple(value["modes"]),
                   value["preload_N"], value["energy_J"], tuple(value["features"]),
                   contact_history=tuple(tuple(row) for row in value.get("contact_history", ())))


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
        self.nd = [r.dimension for r in rods]
        self.contact_width = 4 if settings.friction_solver == "active_set" else 3
        self.offsets = np.cumsum([0] + [d + self.contact_width for d in self.nd])
        self.ng = 2 if settings.y_mode == "free" else 1
        # Columns belonging to different rods have disjoint local rows, except
        # force columns in the shared balance rows. Those shared derivatives are
        # filled explicitly below, allowing one colour per local coordinate.
        self.max_local_size = max(d + self.contact_width for d in self.nd)

    def _jacobian(self, function, unknown, active_rods=None):
        size = len(unknown)
        jac = lil_matrix((size, size), dtype=float)
        # The former perturbation moved a 4 mm guide by 80 nm and crossed
        # nearby contact/friction branches. Resolve those derivatives locally.
        h = 2e-7
        for k in range(self.max_local_size):
            columns = [int(self.offsets[i] + k) for i, nd in enumerate(self.nd)
                       if k < nd+self.contact_width and (active_rods is None or i in active_rods)]
            if not columns:
                continue
            plus, minus = unknown.copy(), unknown.copy()
            plus[columns] += h
            minus[columns] -= h
            lower, upper = self._current_bounds
            plus[columns] = np.minimum(plus[columns], upper[columns])
            minus[columns] = np.maximum(minus[columns], lower[columns])
            change = function(plus) - function(minus)
            for i, nd in enumerate(self.nd):
                if k < nd + self.contact_width and (active_rods is None or i in active_rods):
                    a, b = self.offsets[i:i+2]
                    delta = plus[a+k]-minus[a+k]
                    jac[a:b, a+k] = change[a:b, None]/delta
        for col in range(size-self.ng, size):
            plus, minus = unknown.copy(), unknown.copy()
            plus[col] += h
            minus[col] -= h
            jac[:, col] = ((function(plus)-function(minus))/(2*h))[:, None]
        # Each rod's normalised force contributes linearly to the total load.
        for i in range(len(self.nd)):
            b = self.offsets[i]+self.nd[i]+3
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

    def solve(self, previous: PathState, x_m: float, preload_N: float, *, progress=None) -> EquilibriumTrial:
        needs_multi = self.settings.multipoint_contact and any(len(row) > 1 for row in previous.contact_history)
        if self.settings.multipoint_contact and not needs_multi:
            for i, rod in enumerate(self.rods):
                if previous.preload_N > 0 and previous.modes[i] == "OPEN":
                    continue
                center = previous.centers_m[i]+np.array([x_m-previous.position_m[0], 0., 0.])
                query = self.surface.query_sphere(center, rod.parameters.tip_radius_m,
                                                  tie_tolerance_m=20*self.settings.contact_tolerance_m)
                if len(query.contacts) > 1 and (previous.modes[i] != "OPEN"
                                               or query.gap_m <= self.settings.contact_tolerance_m):
                    needs_multi = True
                    break
                if (query.contacts and previous.modes[i] != "OPEN"
                        and str(query.contacts[0].feature_id) != previous.features[i]
                        and np.linalg.norm(query.contacts[0].normal-previous.normals[i]) > 1e-6):
                    needs_multi = True
                    break
        if needs_multi:
            from .multipoint_contact import solve_multicontact
            return solve_multicontact(self, previous, x_m, preload_N, progress=progress)
        trial = self._solve_single(previous, x_m, preload_N, progress=progress)
        if self.settings.multipoint_contact and trial.status in {"MULTIPOINT_CONTACT_LIMIT", "NUMERICAL_FAILURE"}:
            from .multipoint_contact import solve_multicontact
            return solve_multicontact(self, previous, x_m, preload_N, progress=progress)
        return trial

    def _solve_single(self, previous: PathState, x_m: float, preload_N: float, *, progress=None) -> EquilibriumTrial:
        cfg, length = self.settings, self.length_scale
        force_scale = max(preload_N / len(self.rods), 1e-4)
        self._balance_scale = force_scale / max(preload_N, force_scale)
        initial = np.zeros(int(self.offsets[-1]) + self.ng)
        lower, upper = np.full(initial.size, -np.inf), np.full(initial.size, np.inf)
        first_load = previous.preload_N == 0 and preload_N > 0
        seed_load = np.zeros(len(self.rods))
        if first_load:
            touching = np.array([
                self.surface.query_sphere(center, rod.parameters.tip_radius_m).gap_m <= min(cfg.contact_tolerance_m, 1e-10)
                for center, rod in zip(previous.centers_m, self.rods)
            ])
            # Seed the first contact branch, not a fictitious loaded contact at
            # every open tip. All forces remain unknowns in the coupled solve.
            if np.any(touching):
                seed_load[touching] = preload_N / np.count_nonzero(touching)
        for i, (rod, nd) in enumerate(zip(self.rods, self.nd)):
            a, b = self.offsets[i:i+2]
            force_end = a+nd+3
            initial[a:a+nd] = previous.rod_coordinates[i]
            initial[force_end-3:force_end] = previous.forces_N[i] / force_scale
            if self.contact_width == 4:
                lower[b-1] = 0.
            if first_load:
                initial[force_end-1] = seed_load[i] / force_scale
                if rod.parameters.mount_type == "spring":
                    initial[a] = min(0.5 * rod.parameters.compression_limit_m,
                                     seed_load[i] * max(0., -rod.axis[2])
                                     / rod.parameters.spring_stiffness_N_per_m) / rod.parameters.free_length_m
            if rod.parameters.mount_type == "fixed":
                # This dummy coordinate has the exact equation x_s=0. It is
                # never a compliant mount or an artificial stiff spring.
                initial[a] = 0.
            else:
                lower[a] = 0.
                upper[a] = min(rod.parameters.max_compression_m,
                               rod.parameters.free_length_m-cfg.contact_tolerance_m) / rod.parameters.free_length_m
        initial[-1] = previous.position_m[2] / length
        if self.ng == 2:
            initial[-2] = previous.position_m[1] / length
        initial = np.maximum(lower, np.minimum(upper, initial))
        self._current_bounds = (lower, upper)
        mu = np.array([cfg.friction_kinetic if mode.startswith("SLIP") else cfg.friction_static
                       for mode in previous.modes])
        cache: dict[str, Any] = {}
        slipping = np.zeros(len(self.rods), dtype=bool)
        evaluation_keys, evaluations = [None]*len(self.rods), [None]*len(self.rods)
        query_keys, queries = [None]*len(self.rods), [None]*len(self.rods)

        def assemble(unknown, *, verify_friction=False):
            q = np.array([x_m, cfg.y_locked_m, unknown[-1] * length])
            if self.ng == 2:
                q[1] = unknown[-2] * length
            residual = np.zeros_like(unknown)
            rows, total_force = [], np.zeros(3)
            domain = None
            for i, (rod, nd) in enumerate(zip(self.rods, self.nd)):
                a, b = self.offsets[i:i+2]
                force_end = a+nd+3
                coordinates = unknown[a:a+nd].copy()
                fixed_mount = rod.parameters.mount_type == "fixed"
                if fixed_mount:
                    coordinates[0] = 0.
                force = unknown[force_end-3:force_end] * force_scale
                total_force += force
                evaluation_key = coordinates.tobytes()
                if evaluation_keys[i] != evaluation_key:
                    evaluations[i] = rod.evaluate(coordinates, derivatives=True)
                    evaluation_keys[i] = evaluation_key
                evaluation = evaluations[i]
                center = evaluation.center_m + q
                query_key = center.tobytes()
                if query_keys[i] != query_key:
                    queries[i] = self.surface.query_sphere(center, rod.parameters.tip_radius_m)
                    query_keys[i] = query_key
                query = queries[i]
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
                if fixed_mount:
                    local[0] = unknown[a]
                else:
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
                friction_residual = (tangential_force-disk)/force_scale
                if cfg.friction_solver == "active_set" and not verify_friction:
                    if (normal_force <= force_scale*cfg.residual_tolerance or cfg.friction_static == 0.
                            or (slipping[i] and cfg.friction_kinetic == 0.)):
                        friction_residual = tangential_force/force_scale
                        residual[b-1] = unknown[b-1]
                    elif not slipping[i]:
                        # Solve the sticking branch first, then test its force
                        # against the appropriate static/continuing-kinetic cone.
                        stick_length = min(length, cfg.slip_tolerance_m/cfg.residual_tolerance)
                        friction_residual = tangent_motion/stick_length
                        residual[b-1] = unknown[b-1]
                    else:
                        # A nonnegative slip multiplier keeps the actual
                        # material motion opposite friction. Unlike a frozen
                        # direction, this retains the free-Y coupling.
                        slip_length = min(length, cfg.slip_tolerance_m/cfg.residual_tolerance)
                        friction_residual = tangent_motion/slip_length+unknown[b-1]*tangential_force/force_scale
                        residual[b-1] = (np.linalg.norm(tangential_force)-cfg.friction_kinetic*normal_force)/force_scale
                # Vector equation includes a normal complementarity component;
                # it contains exactly three independent contact equations.
                residual[force_end-3:force_end] = (friction_residual
                                   + normal * min(gap / length, normal_force / force_scale))
                rows.append(dict(coordinates=coordinates.copy(), force=force, evaluation=evaluation,
                                 center=center, contact=contact, query=query,
                                 tangent_motion=tangent_motion, normal_force=normal_force,
                                 tangential_force=tangential_force, moment=moment,
                                 sliding=(bool(slipping[i]) if cfg.friction_solver == "active_set"
                                          else np.linalg.norm(tangent_motion) > cfg.slip_tolerance_m)))
            residual[-1] = (total_force[2] - preload_N) / max(preload_N, force_scale)
            if self.ng == 2:
                residual[-2] = total_force[1] / max(preload_N, force_scale)
            cache.update(q=q, rows=rows, residual=residual, domain=domain)
            return residual

        # Continue kinetic sliding. Return to the static cone only when material
        # motion arrests inside the kinetic disk (e.g. elastic unloading).
        result = None
        # Small arrays have strongly differing shaft/tip stiffness scales. A
        # direct trust-region solve avoids inaccurate inner iterative steps;
        # the same residual and physical tolerances apply to both choices.
        active_rods = set(np.flatnonzero(touching)) if first_load else {
            i for i, mode in enumerate(previous.modes) if mode != "OPEN"}
        # An unloaded, separated rod has exactly zero force and the straight,
        # uncompressed elastic equilibrium. Eliminate only those current-state
        # unknowns; keep the full previous history and every gap constraint.
        for i in range(len(self.rods)):
            if i not in active_rods:
                initial[self.offsets[i]:self.offsets[i+1]] = 0.
        assemble(initial)
        if (cfg.friction_solver == "active_set" and preload_N > 0
                and all(row is not None and row["contact"].gap_m > cfg.contact_tolerance_m
                        for row in cache["rows"])):
            # With every tip open, the force-controlled equation is flat in Z.
            # Move only the Newton initial guess down to first contact. The
            # previous physical state, spring history and target P are intact.
            centers = [row["center"].copy() for row in cache["rows"]]
            if hasattr(self.surface, "height_at"):
                lower_shift = max(self.surface.height_at(*center[:2])-center[2] for center in centers)
            else:
                lower_shift = max(-row["contact"].gap_m/row["contact"].normal[2] for row in cache["rows"])
            def first_gap(shift):
                gaps = []
                for center, rod in zip(centers, self.rods):
                    point = center+np.array([0., 0., shift])
                    if (hasattr(self.surface, "height_at") and
                            point[2] <= self.surface.height_at(*point[:2])+1e-14):
                        # A sphere centre on/inside the wall has negative gap,
                        # even at a corner with no unique normal. Only a
                        # bracketing sign is needed below the wall surface.
                        return -rod.parameters.tip_radius_m
                    gaps.append(self.surface.query_sphere(point, rod.parameters.tip_radius_m).gap_m)
                return min(gaps)
            shift = brentq(first_gap, lower_shift, 0., xtol=1e-13)
            initial[-1] += shift/length
            assemble(initial)
            contact_indices = [i for i, row in enumerate(cache["rows"])
                               if row["contact"].gap_m <= cfg.contact_tolerance_m]
            for i in range(len(self.rods)):
                b = self.offsets[i]+self.nd[i]+3
                initial[b-3:b] = 0.
                if i in contact_indices:
                    initial[b-1] = preload_N/len(contact_indices)/force_scale
            assemble(initial)
            if progress is not None:
                progress(dict(status="CONTACT_SEARCH", normal_shift_m=float(shift)))
        active_rods.update(i for i, row in enumerate(cache["rows"])
                           if row is not None and row["contact"].gap_m <= cfg.contact_tolerance_m)
        def stop_when_solved(intermediate_result):
            target = min(cfg.residual_tolerance, 0.5*cfg.contact_tolerance_m/length)
            if np.max(np.abs(intermediate_result.fun)) <= target:
                if cfg.friction_solver == "active_set":
                    assemble(expand(intermediate_result.x))
                    for row in cache["rows"]:
                        if row is not None and row["normal_force"] > force_scale*cfg.residual_tolerance:
                            speed = np.linalg.norm(row["tangent_motion"])
                            if speed > cfg.slip_tolerance_m:
                                error = np.linalg.norm(row["tangential_force"]+cfg.friction_kinetic*row["normal_force"]
                                                       *row["tangent_motion"]/speed)/force_scale
                                if error > cfg.residual_tolerance:
                                    return
                raise StopIteration
        for _ in range(12 if cfg.friction_solver == "active_set" else 3):
            for _activation in range(len(self.rods)+1):
                columns = np.array([j for i in sorted(active_rods)
                                    for j in range(self.offsets[i], self.offsets[i+1])]
                                   + list(range(initial.size-self.ng, initial.size)), dtype=int)
                def expand(reduced):
                    full = np.zeros_like(initial)
                    full[columns] = reduced
                    return full
                def reduced_residual(reduced):
                    return assemble(expand(reduced))
                direct_linear_solve = len(columns) <= 1024
                def jacobian(reduced):
                    x = expand(reduced)
                    value = self._jacobian(assemble, x, active_rods)[:, columns]
                    if progress is not None:
                        progress(dict(status="ITERATING", residual=float(np.max(np.abs(assemble(x)))),
                                      active_spines=len(active_rods), total_spines=len(self.rods),
                                      sliding_spines=int(np.count_nonzero(slipping))))
                    return value.toarray() if direct_linear_solve else value
                result = least_squares(reduced_residual, initial[columns], bounds=(lower[columns], upper[columns]),
                                       jac=jacobian, x_scale="jac",
                                       tr_solver="exact" if direct_linear_solve else "lsmr",
                                       tr_options={} if direct_linear_solve else {"atol": 1e-10, "btol": 1e-10},
                                       ftol=1e-10, xtol=1e-10, gtol=1e-14,
                                       max_nfev=cfg.max_nfev, diff_step=2e-5, callback=stop_when_solved)
                result.x = expand(result.x)
                assemble(result.x)
                newly_touching = {i for i, row in enumerate(cache["rows"])
                                  if i not in active_rods and row is not None
                                  and row["contact"].gap_m <= cfg.contact_tolerance_m}
                if not newly_touching:
                    break
                active_rods.update(newly_touching)
                initial = result.x
            residual = assemble(result.x)
            if cfg.friction_solver == "active_set":
                changed = False
                for i, row in enumerate(cache["rows"]):
                    if row is None or row["normal_force"] <= force_scale*cfg.residual_tolerance:
                        continue
                    n = row["normal_force"]
                    ft = row["tangential_force"]
                    motion = row["tangent_motion"]
                    speed = np.linalg.norm(motion)
                    limit = cfg.friction_kinetic if previous.modes[i].startswith("SLIP") else cfg.friction_static
                    if not slipping[i] and np.linalg.norm(ft) > limit*n+force_scale*cfg.residual_tolerance:
                        slipping[i] = True
                        changed = True
                    # Slip direction is an unknown in these same equations.
                    # A remaining direction error is a failed solve, not a new
                    # friction branch: return it for path refinement instead of
                    # repeating an unchanged least-squares problem up to 12 times.
                    mu[i] = cfg.friction_kinetic if speed > cfg.slip_tolerance_m else cfg.friction_static
                initial = result.x
                if changed:
                    continue
                break
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
        if cfg.friction_solver == "active_set":
            residual = assemble(result.x, verify_friction=True)
        norm = float(np.max(np.abs(residual)))
        if cfg.friction_solver == "active_set":
            # Verify actual slip direction, not merely a small natural-map
            # residual obtained with a tiny material displacement.
            for row in cache["rows"]:
                if row is not None and row["normal_force"] > force_scale*cfg.residual_tolerance:
                    speed = np.linalg.norm(row["tangent_motion"])
                    if speed > cfg.slip_tolerance_m:
                        error = np.linalg.norm(row["tangential_force"]+cfg.friction_kinetic*row["normal_force"]
                                               *row["tangent_motion"]/speed)/force_scale
                        norm = max(norm, float(error))
        if cache["domain"]:
            return EquilibriumTrial(None, "GEOMETRY_DOMAIN", norm, result.nfev,
                                    {"geometry_status": cache["domain"]})
        for i, row in enumerate(cache["rows"]):
            rod = self.rods[i]
            if (rod.parameters.mount_type == "spring"
                    and rod.parameters.max_compression_m >= rod.parameters.free_length_m
                    and row["evaluation"].exposed_length_m <= 2*cfg.contact_tolerance_m):
                return EquilibriumTrial(None, "EXPOSED_LENGTH_LIMIT", norm, result.nfev,
                                        {"spine_index": i, "nominal_spring_stroke_m": rod.parameters.max_compression_m,
                                         "exposed_length_m": row["evaluation"].exposed_length_m})
            if (row["query"].selected is None and
                    abs(row["contact"].gap_m) <= cfg.contact_tolerance_m):
                return EquilibriumTrial(None, "MULTIPOINT_CONTACT_LIMIT", norm, result.nfev,
                                        {"spine_index": i,
                                         "center_m": row["center"].tolist(),
                                         "contacts": [dict(point_m=np.asarray(c.contact_point_m).tolist(),
                                                           normal=np.asarray(c.normal).tolist(),
                                                           gap_m=float(c.gap_m), feature_id=str(c.feature_id))
                                                      for c in row["query"].contacts]})
        if norm > cfg.residual_tolerance:
            return EquilibriumTrial(None, "NUMERICAL_FAILURE", norm, result.nfev,
                                    {"optimizer_message": result.message})
        return self._state_from_rows(previous, x_m, preload_N, cache["rows"], cache["q"], norm, result.nfev)

    def _state_from_rows(self, previous, x_m, preload_N, rows, q, norm, nfev):
        cfg = self.settings
        force_scale = max(preload_N / len(self.rods), 1e-4)
        modes, energy, dissipation, per_spine = [], 0., 0., []
        status = "ACCEPTED"
        for rod, row in zip(self.rods, rows):
            evaluation, contact = row["evaluation"], row["contact"]
            if (rod.parameters.mount_type == "spring"
                    and rod.parameters.max_compression_m >= rod.parameters.free_length_m
                    and evaluation.exposed_length_m <= 2*cfg.contact_tolerance_m):
                return EquilibriumTrial(None, "EXPOSED_LENGTH_LIMIT", norm, nfev,
                                        {"exposed_length_m": evaluation.exposed_length_m})
            if not cfg.multipoint_contact and row["query"].selected is None and contact.gap_m <= cfg.contact_tolerance_m:
                status = "MULTIPOINT_CONTACT_LIMIT"
            supports = row.get("contacts", [row])
            loaded = [c for c in supports if c["normal_force"] > force_scale*cfg.residual_tolerance]
            n = sum(c["normal_force"] for c in supports)
            if n <= force_scale * cfg.residual_tolerance:
                mode = "OPEN"
            elif any(c.get("sliding", np.linalg.norm(c["tangent_motion"]) > cfg.slip_tolerance_m) for c in loaded):
                mode = "SLIP"
            else:
                mode = "STICK"
            energy += evaluation.energy_J
            dissipation += sum(max(0., -float(c["tangential_force"] @ c["tangent_motion"])) for c in supports)
            compression = row["coordinates"][0] * rod.parameters.free_length_m
            fixed_mount = rod.parameters.mount_type == "fixed"
            hard_stop = (not fixed_mount and rod.parameters.max_compression_m < rod.parameters.free_length_m
                         and compression >= rod.parameters.max_compression_m - cfg.contact_tolerance_m)
            stress = rod.stress(row["coordinates"], row["force"], row["moment"], evaluation=evaluation)
            raw_residual = rod.generalized_residual(row["coordinates"], row["force"], row["moment"], evaluation=evaluation)
            boundary = rod.axial_boundary(row["coordinates"], raw_residual,
                                          compression_tolerance_m=cfg.contact_tolerance_m)
            if stress.model_limit:
                status = "ELASTIC_MODEL_LIMIT"
            if (rod.parameters.taper_length_m > 0 and any(
                    c["contact"].gap_m <= cfg.contact_tolerance_m
                    and float(-np.asarray(c["contact"].normal) @ evaluation.tip_rotation[:, 0]) < -1e-10
                    for c in loaded)):
                # The rear sphere is buried in the taper. Reaching it is a
                # contact outside this tip-cap model, not an available hook.
                status = "TIP_CONTACT_DOMAIN"
            body_gap = None
            if cfg.check_body_clearance:
                if rod.parameters.taper_length_m > 0:
                    from .tapered_geometry import query_tapered_rod_clearance
                    clearance = query_tapered_rod_clearance(
                        self.surface, rod, row["coordinates"], q, cfg.contact_tolerance_m)
                else:
                    centerline, sagitta = rod.clearance_centerline(row["coordinates"],
                                                                   tolerance_m=cfg.contact_tolerance_m/4)
                    clearance = self.surface.query_centerline(centerline + q, rod.parameters.diameter_m/2+sagitta)
                body_gap = clearance.gap_m
                if clearance.status in {"out_of_domain", "invalid_surface"}:
                    status = "ROD_GEOMETRY_DOMAIN"
                elif clearance.status == "indeterminate":
                    status = "ROD_GEOMETRY_UNRESOLVED"
                elif body_gap is not None and body_gap < -cfg.contact_tolerance_m:
                    status = "ROD_COLLISION_LIMIT"
            modes.append(mode + ("_HARDSTOP" if hard_stop else ""))
            per_spine.append(dict(force_N=row["force"].tolist(), P_N=float(row["force"][2]),
                                  T_N=float(-row["force"][0]), N_N=float(n),
                                  compression_m=float(compression),
                                  exposed_length_m=float(rod.parameters.free_length_m-compression),
                                  gap_m=float(min(c["contact"].gap_m for c in supports)), mode=modes[-1],
                                  contacts=[dict(force_N=c["force"].tolist(), N_N=float(c["normal_force"]),
                                                 normal=np.asarray(c["contact"].normal).tolist(),
                                                 point_m=np.asarray(c["contact"].contact_point_m).tolist(),
                                                 gap_m=float(c["contact"].gap_m),
                                                 feature_id=str(c["contact"].feature_id),
                                                 tangent_increment_m=c["tangent_motion"].tolist(),
                                                 mode=("OPEN" if c["normal_force"] <= force_scale*cfg.residual_tolerance else "SLIP" if
                                                       c.get("sliding", np.linalg.norm(c["tangent_motion"]) > cfg.slip_tolerance_m) else "STICK"))
                                            for c in supports],
                                  body_gap_m=body_gap,
                                  max_stress_upper_Pa=stress.max_section_von_mises_upper_Pa,
                                  stress_utilization=stress.utilization,
                                  guide_moment_Nm=stress.guide_moment_Nm.tolist(),
                                  travel_utilization=(0. if fixed_mount else float(compression/rod.parameters.max_compression_m)),
                                  mount_type=rod.parameters.mount_type,
                                  nominal_spring_stroke_m=(None if fixed_mount else rod.parameters.max_compression_m),
                                  fixed_mount_axial_residual_N=(float(raw_residual[0]/rod.parameters.free_length_m)
                                                               if fixed_mount else None),
                                  fixed_mount_reaction_N=((-row["force"]).tolist() if fixed_mount else None),
                                  fixed_mount_axial_reaction_N=(float(-row["force"] @ rod.axis)
                                                               if fixed_mount else None),
                                  spring_branch=boundary.branch,
                                  upper_stop_reaction_N=(0. if fixed_mount else boundary.upper_reaction_N),
                                  lower_stop_reaction_N=(0. if fixed_mount else boundary.lower_reaction_N),
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
        moments = sum((np.cross(row["center"] - q, row["force"]) + row["moment"]
                       for row in rows), np.zeros(3))
        work = (-0.5 * (total[0] + previous.forces_N[:, 0].sum()) * (x_m - previous.position_m[0])
                -0.5 * (preload_N + previous.preload_N) * (q[2] - previous.position_m[2]))
        diagnostics = dict(residual=norm, nfev=nfev, per_spine=per_spine,
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
                          tuple("|".join(sorted(c["feature_id"] for c in r["contacts"] if c["mode"] != "OPEN"))
                                or str(row["contact"].feature_id) for r, row in zip(per_spine, rows)), diagnostics,
                          tuple(tuple(c for c in r["contacts"] if c["mode"] != "OPEN"
                                      or c["gap_m"] <= cfg.contact_tolerance_m) for r in per_spine))
        return EquilibriumTrial(state, status, norm, nfev)
