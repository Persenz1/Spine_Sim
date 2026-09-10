"""Allowed-mode incremental energy check for a smooth, sticking array branch.

This checks fixed X and fixed guide rotations, free Y/Z and internal rod modes.
It does not use the sign of the total drag-force slope. Geometry is queried at
perturbed current centres, so moving-normal terms, preload and the changing rod
length enter the same virtual-work derivative.

No-slip rolling constraints need not be holonomic. A positive result therefore
reports a *local constrained incremental energy* sufficient condition on the
specified smooth sticking branch, not an unconditional dynamic or finite-path
stability result. Sliding and active-set boundaries receive explicit statuses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .guided_rod import sphere_contact_moment


def _skew(v):
    x, y, z = v
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def _nullspace(matrix, columns):
    if not len(matrix):
        return np.eye(columns)
    _, singular, vh = np.linalg.svd(np.asarray(matrix), full_matrices=True)
    rank = int(np.sum(singular > max(float(singular[0]), 1.) * 1e-9)) if len(singular) else 0
    return vh[rank:].T


class _BranchChanged(Exception):
    pass


def assess_stick_stability(array, state) -> dict[str, Any]:
    """Condense small per-rod energy blocks onto the permitted Y/Z subspace.

    ``array`` supplies rods, surface, settings, length_scale and ng; ``state``
    is an accepted ``guided_array.PathState``. Results are JSON-ready. The
    stiffness units refer to dimensionless local states and Y/Z divided by the
    array length scale; eigenvalue signs, rather than their absolute values,
    are the physically relevant outcome.

    The calculation is optional because it differentiates every local block.
    Memory scales with local rod dimension, rather than total array size.
    """
    cfg, length, ng = array.settings, array.length_scale, array.ng
    base = dict(dynamic_stability="OUT_OF_SCOPE", method="constrained_incremental_energy",
                scope="smooth sticking branch; fixed X and guide rotations; allowed Y/Z and rod modes",
                local_matrix_units="J per dimensionless coordinate squared")
    if any(mode.startswith("SLIP") for mode in state.modes):
        return dict(base, status="SLIDING_NONCONSERVATIVE", sufficient_condition=False)
    directions = np.eye(3)[:, [1, 2] if ng == 2 else [2]]
    global_schur = np.zeros((ng, ng))
    global_constraints = []
    minimum_internal = float("inf")
    internal_count = 0
    max_asymmetry = 0.
    one_sided_free_bound = False
    force_tolerance = max(state.preload_N / len(array.rods), 1e-4) * cfg.residual_tolerance * 10
    energy_scale = max(state.preload_N * length, 1e-8)
    energy_tolerance = energy_scale * 2e-6

    for i, rod in enumerate(array.rods):
        coordinates = state.rod_coordinates[i]
        force = state.forces_N[i]
        nd = rod.dimension
        p = rod.parameters
        evaluation = rod.evaluate(coordinates)
        contact = array.surface.query_sphere(evaluation.center_m + state.position_m,
                                             p.tip_radius_m).selected
        if contact is None or contact.normal_jacobian is None:
            return dict(base, status="GEOMETRY_BRANCH_BOUNDARY", sufficient_condition=False, spine_index=i)
        normal = np.asarray(contact.normal)
        normal_force = float(force @ normal)
        active = state.modes[i].startswith("STICK") and normal_force > force_tolerance
        if active and (cfg.friction_static * normal_force - np.linalg.norm(force - normal_force * normal)
                       <= force_tolerance):
            return dict(base, status="FRICTION_BRANCH_BOUNDARY", sufficient_condition=False, spine_index=i)
        if not active and np.linalg.norm(force) > force_tolerance:
            return dict(base, status="CONTACT_BRANCH_BOUNDARY", sufficient_condition=False, spine_index=i)
        if not active and abs(contact.gap_m) <= cfg.contact_tolerance_m:
            return dict(base, status="CONTACT_BRANCH_BOUNDARY", sufficient_condition=False, spine_index=i)

        def local_residual(z):
            x = z[:nd]
            q = state.position_m + length * directions @ z[nd:]
            e = rod.evaluate(x)
            if active:
                candidate = array.surface.query_sphere(e.center_m + q, p.tip_radius_m).selected
                if (candidate is None or candidate.normal_jacobian is None
                        or candidate.feature_id != contact.feature_id):
                    raise _BranchChanged
                moment = sphere_contact_moment(p.tip_radius_m, candidate.normal, force)
            else:
                moment = np.zeros(3)
            r = rod.generalized_residual(x, force, moment, evaluation=e)
            # Constant total P gives a linear P*Z potential. Its Hessian is zero;
            # the constant force term here is sufficient for local derivatives.
            return np.concatenate((r, -length * directions.T @ force))

        z = np.concatenate((coordinates, np.zeros(ng)))
        jacobian = np.empty((nd + ng, nd + ng))
        try:
            for j in range(nd + ng):
                h = 5e-5 * max(1., abs(z[j]))
                plus, minus = z.copy(), z.copy()
                plus[j] += h
                minus[j] -= h
                jacobian[:, j] = (local_residual(plus) - local_residual(minus)) / (2 * h)
        except _BranchChanged:
            return dict(base, status="GEOMETRY_BRANCH_BOUNDARY", sufficient_condition=False, spine_index=i)
        max_asymmetry = max(max_asymmetry, float(np.linalg.norm(jacobian - jacobian.T)
                                                / max(np.linalg.norm(jacobian), energy_tolerance)))
        hessian = (jacobian + jacobian.T) / 2
        if active:
            point_jacobian = (evaluation.center_jacobian_m
                              + p.tip_radius_m * _skew(normal) @ evaluation.rotation_jacobian)
            constraint = np.column_stack((point_jacobian / length, directions))
        else:
            constraint = np.empty((0, nd + ng))
        local = rod.generalized_residual(coordinates, force,
                                         sphere_contact_moment(p.tip_radius_m, normal, force), evaluation)
        boundary = rod.axial_boundary(coordinates, local, force_tolerance,
                                     compression_tolerance_m=cfg.contact_tolerance_m)
        compression_tolerance = cfg.contact_tolerance_m
        at_upper = p.max_compression_m - evaluation.compression_m <= compression_tolerance
        at_lower = evaluation.compression_m <= compression_tolerance
        fixed_mount = p.mount_type == "fixed"
        strict_stop = (fixed_mount or (at_upper and boundary.upper_reaction_N > force_tolerance)
                       or (at_lower and p.lower_stop and boundary.lower_reaction_N > force_tolerance))
        if strict_stop:
            stop = np.zeros(nd + ng)
            stop[0] = 1
            constraint = np.vstack((constraint, stop))
        elif at_upper or (at_lower and p.lower_stop):
            return dict(base, status="STOP_BRANCH_BOUNDARY", sufficient_condition=False, spine_index=i)
        elif at_lower:
            one_sided_free_bound = True

        cx, cq = constraint[:, :nd], constraint[:, nd:]
        if len(cx):
            u, singular, vh = np.linalg.svd(cx, full_matrices=True)
            rank = int(np.sum(singular > max(float(singular[0]), 1.) * 1e-9)) if len(singular) else 0
            v = vh[rank:].T
            w = -(vh[:rank].T / singular[:rank]) @ u[:, :rank].T @ cq
            global_constraints.extend(u[:, rank:].T @ cq)
        else:
            v, w = np.eye(nd), np.zeros((nd, ng))
        a = v.T @ hessian[:nd, :nd] @ v
        d = v.T @ (hessian[:nd, :nd] @ w + hessian[:nd, nd:])
        c = (w.T @ hessian[:nd, :nd] @ w + w.T @ hessian[:nd, nd:]
             + hessian[nd:, :nd] @ w + hessian[nd:, nd:])
        if a.size:
            eigenvalues = np.linalg.eigvalsh(a)
            minimum_internal = min(minimum_internal, float(eigenvalues[0]))
            internal_count += len(eigenvalues)
            if eigenvalues[0] <= energy_tolerance:
                return dict(base, status="NOT_STRICTLY_POSITIVE", sufficient_condition=False,
                            spine_index=i, minimum_internal_eigenvalue_J=float(eigenvalues[0]),
                            one_sided_bounds_enlarged=one_sided_free_bound,
                            interpretation="No positive-energy certificate; this alone does not prove dynamic instability")
            c = c - d.T @ np.linalg.solve(a, d)
        global_schur += c

    allowed = _nullspace(global_constraints, ng)
    condensed = allowed.T @ global_schur @ allowed
    global_eigenvalues = np.linalg.eigvalsh((condensed + condensed.T) / 2)
    positive = not len(global_eigenvalues) or bool(global_eigenvalues[0] > energy_tolerance)
    return dict(base, status=("CONSTRAINED_INCREMENTAL_ENERGY_POSITIVE" if positive
                              else "NOT_STRICTLY_POSITIVE"), sufficient_condition=positive,
                internal_modes_checked=internal_count, global_modes_checked=allowed.shape[1],
                minimum_internal_eigenvalue_J=None if minimum_internal == float("inf") else minimum_internal,
                condensed_global_eigenvalues_J=global_eigenvalues.tolist(),
                maximum_local_jacobian_asymmetry=max_asymmetry,
                one_sided_bounds_enlarged=one_sided_free_bound)
