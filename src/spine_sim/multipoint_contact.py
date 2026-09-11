"""Several unilateral Coulomb supports acting on one finite rod.

Contact forces and sphere-centre moments are summed before evaluating the one
rod's virtual-work equation. Candidate distance only discovers nearby branches;
every force-bearing support still satisfies the original gap tolerance.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment, brentq

from .guided_array import EquilibriumTrial, _rotation_increment
from .continuous_geometry import SphereQuery


def _match(anchors, contacts):
    if not anchors or not contacts:
        return {}
    cost = np.linalg.norm(np.asarray(anchors)[:, None, :]
                          - np.asarray([c.contact_point_m for c in contacts])[None, :, :], axis=2)
    a, b = linear_sum_assignment(cost)
    return dict(zip(a.tolist(), b.tolist()))


def _frame(normal):
    reference = np.array([1., 0., 0.]) if abs(normal[0]) < .9 else np.array([0., 1., 0.])
    tangent = reference-(reference@normal)*normal
    tangent /= np.linalg.norm(tangent)
    return np.column_stack((normal, tangent, np.cross(normal, tangent)))


def solve_multicontact(model, previous, x_m, preload_N, *, progress=None):
    cfg, rods, surface = model.settings, model.rods, model.surface
    length = model.length_scale
    scale = max(preload_N/len(rods), 1e-4)
    slip_length = min(length, cfg.slip_tolerance_m/cfg.residual_tolerance)
    gap_length = min(length, cfg.contact_tolerance_m/(4*cfg.residual_tolerance))
    margin = max(20*cfg.contact_tolerance_m, 2*abs(x_m-previous.position_m[0]))
    q0 = previous.position_m.copy(); q0[0] = x_m
    coordinates0 = [u.copy() for u in previous.rod_coordinates]
    centers0 = [r.evaluate(u, derivatives=False).center_m+q0 for r, u in zip(rods, coordinates0)]

    def query(center, rod):
        return surface.query_sphere(center, rod.parameters.tip_radius_m, tie_tolerance_m=margin)

    queries0 = [query(c, r) for c, r in zip(centers0, rods)]
    if any(not q.contacts for q in queries0):
        return EquilibriumTrial(None, "GEOMETRY_DOMAIN", float('inf'), 0)
    if preload_N > 0 and all(q.gap_m > cfg.contact_tolerance_m for q in queries0):
        if hasattr(surface, 'height_at'):
            lower = max(surface.height_at(*c[:2])-c[2] for c in centers0)
        else:
            lower = max(-q.gap_m/q.contacts[0].normal[2] for q in queries0)
        def gap(shift):
            values = []
            for center, rod in zip(centers0, rods):
                point = center + [0., 0., shift]
                if hasattr(surface, 'height_at') and point[2] <= surface.height_at(*point[:2])+1e-14:
                    return -rod.parameters.tip_radius_m
                values.append(query(point, rod).gap_m)
            return min(values)
        shift = brentq(gap, lower, 0., xtol=1e-13)
        q0[2] += shift
        centers0 = [c + [0., 0., shift] for c in centers0]
        queries0 = [query(c, r) for c, r in zip(centers0, rods)]
        if progress:
            progress(dict(status='CONTACT_SEARCH', normal_shift_m=float(shift)))
    candidates = [list(q.contacts) for q in queries0]
    seed_rows = None
    total_evaluations = 0

    # A previously open rod or a new nearby support can join during equilibrium.
    # Rebuild only the contact unknowns; the accepted physical history stays fixed.
    for activation in range(len(rods)+8):
        anchors = [[c.contact_point_m for c in cs] for cs in candidates]
        nc = [len(cs) for cs in candidates]
        offsets = np.cumsum([0]+[r.dimension+4*n for r, n in zip(rods, nc)])
        ng = model.ng
        size = int(offsets[-1])+ng
        initial = np.zeros(size)
        lower, upper = np.full(size, -np.inf), np.full(size, np.inf)
        old = []
        for i, rod in enumerate(rods):
            a, nd = offsets[i], rod.dimension
            initial[a:a+nd] = coordinates0[i]
            if rod.parameters.mount_type == 'fixed':
                initial[a] = 0.
            else:
                lower[a] = 0.
                upper[a] = min(rod.parameters.max_compression_m,
                               rod.parameters.free_length_m-cfg.contact_tolerance_m)/rod.parameters.free_length_m
            history = list(previous.contact_history[i]) if previous.contact_history else []
            if not history and previous.modes[i] != 'OPEN':
                history = [dict(point_m=(previous.centers_m[i]-rod.parameters.tip_radius_m*previous.normals[i]).tolist(),
                                force_N=previous.forces_N[i].tolist(), normal=previous.normals[i].tolist(),
                                mode=previous.modes[i])]
            history_map = _match([h['point_m'] for h in history], candidates[i])
            center_guess = rod.evaluate(coordinates0[i], derivatives=False).center_m+q0
            # A distinct, newly encountered support starts a new contact episode.
            # Mesh-feature changes along a continuous nearby support retain it.
            tracking_distance = 4*np.linalg.norm(center_guess-previous.centers_m[i])+20*cfg.contact_tolerance_m
            history_map = {k:j for k,j in history_map.items()
                           if np.linalg.norm(np.asarray(history[k]['point_m'])-candidates[i][j].contact_point_m)
                           <= tracking_distance}
            inverse_history = {j:k for k,j in history_map.items()}
            rows_i = seed_rows[i]['contacts'] if seed_rows else None
            seed_map = _match([c['contact'].contact_point_m for c in rows_i], candidates[i]) if rows_i else {}
            inverse_seed = {j:k for k,j in seed_map.items()}
            local_old = []
            for j, contact in enumerate(candidates[i]):
                b = a+nd+4*j
                lower[b] = 0.
                lower[b+3] = 0.
                h = history[inverse_history[j]] if j in inverse_history else dict(
                    normal=contact.normal, mode='OPEN', force_N=[0., 0., 0.])
                local_old.append(h)
                seed_force = rows_i[inverse_seed[j]]['force'] if j in inverse_seed else h['force_N']
                initial[b:b+3] = _frame(contact.normal).T@np.asarray(seed_force)/scale
                if h['mode'] == 'OPEN' and contact.gap_m <= cfg.contact_tolerance_m and preload_N > 0:
                    # Start a newly closed inequality in the interior. A zero
                    # pressure at its bound can otherwise prevent the coupled
                    # trust-region step from transferring any load to it.
                    initial[b] = max(initial[b], .01*preload_N/len(rods)/scale)
            old.append(local_old)
        initial[-1] = q0[2]/length
        if ng == 2:
            initial[-2] = q0[1]/length
        initial = np.clip(initial, lower, upper)
        active = {i for i in range(len(rods)) if previous.modes[i] != 'OPEN'
                  or min(c.gap_m for c in candidates[i]) <= cfg.contact_tolerance_m}
        if previous.preload_N > 0:
            for i in active:
                rod=rods[i]
                minimum=min(c.gap_m for c in candidates[i])
                nearest=[j for j,c in enumerate(candidates[i]) if c.gap_m <= minimum+1e-10]
                carrying=[j for j in range(nc[i]) if initial[offsets[i]+rod.dimension+4*j] > cfg.residual_tolerance]
                if previous.modes[i].startswith('SLIP') and carrying and any(j not in nearest for j in carrying):
                    # Crossing the surface envelope can release the old support
                    # abruptly. Seed the nearer branch instead of trapping the
                    # iteration with positive pressures on both branches. All
                    # gap constraints and contact unknowns remain in the solve.
                    for j,c in enumerate(candidates[i]):
                        b=offsets[i]+rod.dimension+4*j
                        initial[b:b+3]=(_frame(c.normal).T@previous.forces_N[i]/len(nearest)/scale
                                        if j in nearest else 0.)
                        initial[b]=max(0.,initial[b])
        if preload_N > 0 and not any(np.linalg.norm(f) > scale*cfg.residual_tolerance for f in previous.forces_N):
            touching = [(i,j) for i in active for j,c in enumerate(candidates[i])
                        if c.gap_m <= cfg.contact_tolerance_m]
            vertical = sum(candidates[i][j].normal[2] for i,j in touching)
            if vertical > 0:
                for i,j in touching:
                    initial[offsets[i]+rods[i].dimension+4*j] = preload_N/vertical/scale
        for i in range(len(rods)):
            if i not in active:
                initial[offsets[i]:offsets[i+1]] = 0.
        sliding = [np.zeros(n, dtype=bool) for n in nc]
        for i, rod in enumerate(rods):
            guess_center = rod.evaluate(initial[offsets[i]:offsets[i]+rod.dimension], derivatives=False).center_m+q0
            for j,c in enumerate(candidates[i]):
                b=offsets[i]+rod.dimension+4*j
                motion=guess_center-previous.centers_m[i]
                vt=motion-(motion@c.normal)*c.normal
                previous_force=np.asarray(old[i][j]['force_N'])
                ft=previous_force-(previous_force@c.normal)*c.normal
                if (cfg.friction_kinetic > 0 and np.linalg.norm(ft) > scale*cfg.residual_tolerance
                        and old[i][j]['mode'].startswith('SLIP') and ft@vt < 0):
                    sliding[i][j]=True
                elif (cfg.friction_kinetic > 0 and old[i][j]['mode'] == 'OPEN'
                      and initial[b] > cfg.residual_tolerance and np.linalg.norm(vt) > cfg.slip_tolerance_m):
                    sliding[i][j]=True
                    kinetic = -cfg.friction_kinetic*initial[b]*vt/np.linalg.norm(vt)
                    initial[b+1:b+3] = _frame(c.normal)[:,1:].T@kinetic
        cache = {}
        slip_directions = []
        for i, rod in enumerate(rods):
            directions=[]
            for j,c in enumerate(candidates[i]):
                b=offsets[i]+rod.dimension+4*j
                ft=_frame(c.normal)[:,1:]@initial[b+1:b+3]
                directions.append(-ft/max(np.linalg.norm(ft),1e-30))
            slip_directions.append(directions)
        eval_cache, query_cache = [{} for _ in rods], [{} for _ in rods]
        reference_queries = [SphereQuery('ok' if len(cs)==1 else 'multiple_contacts', tuple(cs)) for cs in candidates]

        def assemble(u, *, exact_geometry=False):
            q = np.array([x_m, cfg.y_locked_m if ng == 1 else u[-2]*length, u[-1]*length])
            residual = np.zeros(size+len(rods))
            rows, unseen, total = [], [], np.zeros(3)
            for i, rod in enumerate(rods):
                a, nd = offsets[i], rod.dimension
                coordinates = u[a:a+nd].copy()
                if rod.parameters.mount_type == 'fixed':
                    coordinates[0] = 0.
                key = coordinates.tobytes()
                if key not in eval_cache[i]:
                    eval_cache[i].clear()
                    eval_cache[i][key] = rod.evaluate(coordinates, derivatives=True)
                evaluation = eval_cache[i][key]
                center = evaluation.center_m+q
                key = center.tobytes()
                reference = reference_queries[i]
                still_open = (i not in active and reference.gap_m
                              -np.linalg.norm(center-reference.contacts[0].center_m) > cfg.contact_tolerance_m)
                if still_open and not exact_geometry:
                    # Distance to a fixed surface is 1-Lipschitz. This bound
                    # proves that an eliminated, zero-force rod stays open.
                    # Requery all geometry before accepting a physical state.
                    sphere = reference
                else:
                    if key not in query_cache[i]:
                        query_cache[i].clear()
                        query_cache[i][key] = query(center, rod)
                    sphere = query_cache[i][key]
                if not sphere.contacts:
                    residual[a:offsets[i+1]] = 1e3
                    rows.append(None); unseen.append([])
                    continue
                match = _match(anchors[i], sphere.contacts)
                extra = [c for j,c in enumerate(sphere.contacts) if j not in match.values()]
                unseen.append(extra)
                residual[size+i] = min([0.]+[c.gap_m/gap_length for c in extra])
                force, moment, supports = np.zeros(3), np.zeros(3), []
                rotation = _rotation_increment(evaluation.tip_rotation, previous.rotations[i])
                for j in range(nc[i]):
                    b = a+nd+4*j
                    if j not in match:
                        residual[b:b+4] = u[b:b+4]
                        continue
                    c = sphere.contacts[match[j]]
                    n = c.normal
                    frame = _frame(n)
                    f = frame@u[b:b+3]*scale
                    N = float(u[b]*scale)
                    ft = f-N*n
                    mean_n = n+np.asarray(old[i][j]['normal'])
                    mean_n /= max(np.linalg.norm(mean_n), 1e-30)
                    motion = center-previous.centers_m[i]+np.cross(rotation, -rod.parameters.tip_radius_m*mean_n)
                    vt = motion-(motion@n)*n
                    if (N <= scale*cfg.residual_tolerance or cfg.friction_static == 0.
                            or (sliding[i][j] and cfg.friction_kinetic == 0.)):
                        friction = ft/scale
                        residual[b+3] = u[b+3]
                    elif sliding[i][j]:
                        speed=np.linalg.norm(vt)
                        direction=vt/speed if speed>1e-12 else slip_directions[i][j]
                        friction = (ft+cfg.friction_kinetic*N*direction)/scale
                        residual[b+3] = u[b+3]
                    else:
                        friction = vt/slip_length
                        limit = cfg.friction_kinetic if old[i][j]['mode'].startswith('SLIP') else cfg.friction_static
                        # Static contact forces can be indeterminate. Find a
                        # distribution inside every cone before declaring slip;
                        # an arbitrary unconstrained self-stress is not failure.
                        residual[b+3] = u[b+3]+max(0., np.linalg.norm(ft)-limit*N)/scale
                    residual[b] = min(c.gap_m/gap_length, N/scale)
                    residual[b+1:b+3] = frame[:,1:].T@friction
                    m = -rod.parameters.tip_radius_m*np.cross(n, f)
                    force += f; moment += m
                    supports.append(dict(contact=c, force=f, normal_force=N, tangential_force=ft,
                                         tangent_motion=vt, slot=j, sliding=bool(sliding[i][j])))
                total += force
                local = (evaluation.energy_gradient_J-evaluation.center_jacobian_m.T@force
                         -evaluation.rotation_jacobian.T@moment)/(scale*length)
                if rod.parameters.mount_type == 'fixed':
                    local[0] = u[a]
                else:
                    if rod.parameters.lower_stop:
                        local[0] = min(local[0], coordinates[0])
                    local[0] = max(local[0], coordinates[0]-upper[a])
                residual[a:a+nd] = local
                primary = max(supports, key=lambda c:c['normal_force']) if supports else dict(
                    contact=sphere.contacts[0], force=np.zeros(3), normal_force=0.,
                    tangential_force=np.zeros(3), tangent_motion=np.zeros(3))
                rows.append(dict(primary, coordinates=coordinates, evaluation=evaluation, center=center,
                                 query=sphere, contacts=supports or [primary], force=force, moment=moment))
            residual[size-1] = (total[2]-preload_N)/max(preload_N, scale)
            if ng == 2:
                residual[size-2] = total[1]/max(preload_N, scale)
            cache.update(rows=rows, q=q, unseen=unseen)
            return residual

        columns = np.array([j for i in sorted(active) for j in range(offsets[i], offsets[i+1])]
                           +list(range(size-ng, size)), dtype=int)
        def expand(v):
            u = np.zeros(size); u[columns] = v
            return u
        def jacobian(v):
            u = expand(v)
            jac = np.zeros((size+len(rods), size))
            h = 2e-7
            for k in range(max(np.diff(offsets))):
                group = [offsets[i]+k for i in sorted(active) if k < offsets[i+1]-offsets[i]]
                if not group:
                    continue
                plus, minus = u.copy(), u.copy()
                plus[group] = np.minimum(plus[group]+h, upper[group])
                minus[group] = np.maximum(minus[group]-h, lower[group])
                plus_residual = assemble(plus)
                plus_forces = [row['force'].copy() if row is not None else np.zeros(3) for row in cache['rows']]
                change = plus_residual-assemble(minus)
                for i in sorted(active):
                    if k < offsets[i+1]-offsets[i]:
                        j = offsets[i]+k; delta=plus[j]-minus[j]
                        jac[offsets[i]:offsets[i+1], j] = change[offsets[i]:offsets[i+1]]/delta
                        jac[size+i, j] = change[size+i]/delta
                        minus_force = cache['rows'][i]['force'] if cache['rows'][i] is not None else np.zeros(3)
                        df = (plus_forces[i]-minus_force)/delta/max(preload_N, scale)
                        jac[size-1, j] = df[2]
                        if ng == 2:
                            jac[size-2, j] = df[1]
            for j in range(size-ng, size):
                plus, minus = u.copy(), u.copy(); plus[j]+=h; minus[j]-=h
                jac[:, j] = (assemble(plus)-assemble(minus))/(2*h)
            if progress:
                progress(dict(status='ITERATING', residual=float(np.max(np.abs(assemble(u)))),
                              active_spines=len(active), total_spines=len(rods),
                              contact_candidates=sum(nc[i] for i in active)))
            return jac[:, columns]

        def physical_norm(residual):
            # Gap weighting conditions Newton's problem; it must not silently
            # replace the declared physical gap and equilibrium tolerances.
            physical=residual.copy()
            extra_error=0.
            for i,row in enumerate(cache['rows']):
                if row is None:
                    return float('inf')
                physical[size+i] *= gap_length/length
                if row['query'].gap_m < -cfg.contact_tolerance_m:
                    extra_error=max(extra_error, 2*cfg.residual_tolerance)
                for c in row['contacts']:
                    N, ft, vt = c['normal_force'], c['tangential_force'], c['tangent_motion']
                    j=c.get('slot')
                    if j is not None:
                        physical[offsets[i]+rods[i].dimension+4*j]=min(c['contact'].gap_m/length,N/scale)
                    if N > scale*cfg.residual_tolerance:
                        speed=np.linalg.norm(vt)
                        error=(np.linalg.norm(ft+cfg.friction_kinetic*N*vt/speed)/scale if speed > cfg.slip_tolerance_m
                               else (np.linalg.norm(ft)-cfg.friction_static*N)/scale)
                        extra_error=max(extra_error,float(error))
                        if abs(c['contact'].gap_m) > cfg.contact_tolerance_m:
                            extra_error=max(extra_error,2*cfg.residual_tolerance)
            return max(float(np.max(np.abs(physical))),extra_error)

        def stop(intermediate_result):
            residual=assemble(expand(intermediate_result.x))
            if physical_norm(residual) <= cfg.residual_tolerance:
                raise StopIteration
        for branch in range(2*sum(nc)+3):
            # A free neutral translation may have only roundoff derivatives.
            # Do not turn those into an enormous trust-region coordinate scale.
            variable_scale = 1/np.maximum(np.linalg.norm(jacobian(initial[columns]), axis=0), 1.)
            result = least_squares(lambda v:assemble(expand(v)), initial[columns], jac=jacobian,
                                   bounds=(lower[columns], upper[columns]), x_scale=variable_scale,
                                   ftol=1e-10, xtol=1e-10, gtol=1e-14, max_nfev=cfg.max_nfev, callback=stop)
            total_evaluations += result.nfev
            initial = expand(result.x)
            residual = assemble(initial, exact_geometry=True)
            changed = False
            for i, row in enumerate(cache['rows']):
                if row is None:
                    continue
                for c in row['contacts']:
                    j = c.get('slot')
                    if j is None or c['normal_force'] <= scale*cfg.residual_tolerance:
                        continue
                    if sliding[i][j] and np.linalg.norm(c['tangent_motion']) <= cfg.slip_tolerance_m:
                        sliding[i][j]=False
                        initial[offsets[i]+rods[i].dimension+4*j+3]=0.
                        changed=True
                        continue
                    limit = cfg.friction_kinetic if old[i][j]['mode'].startswith('SLIP') else cfg.friction_static
                    magnitude = np.linalg.norm(c['tangential_force'])
                    at_limit = magnitude >= limit*c['normal_force']-scale*cfg.residual_tolerance
                    must_slide = (magnitude > limit*c['normal_force']+scale*cfg.residual_tolerance
                                  or (at_limit and np.linalg.norm(c['tangent_motion']) > cfg.slip_tolerance_m))
                    if not sliding[i][j] and must_slide:
                        sliding[i][j] = True; changed = True
                        b = offsets[i]+rods[i].dimension+4*j
                        if magnitude > 0:
                            initial[b+1:b+3] *= cfg.friction_kinetic*c['normal_force']/magnitude
                            slip_directions[i][j]=-c['tangential_force']/magnitude
                        initial[b+3] = 0.
            if not changed:
                break
        if changed:
            return EquilibriumTrial(None, 'NUMERICAL_FAILURE', float(np.max(np.abs(residual))), total_evaluations,
                                    dict(optimizer_message='friction branches did not settle'))
        rows = cache['rows']
        if any(row is None for row in rows):
            return EquilibriumTrial(None, 'GEOMETRY_DOMAIN', float(np.max(np.abs(residual))), total_evaluations)
        newly_active = any(i not in active and row['query'].gap_m <= cfg.contact_tolerance_m for i,row in enumerate(rows))
        new_contacts = any(c.gap_m <= cfg.contact_tolerance_m for extra in cache['unseen'] for c in extra)
        if newly_active or new_contacts:
            q0 = cache['q'].copy(); coordinates0 = [r['coordinates'].copy() for r in rows]
            candidates = [list(r['query'].contacts) for r in rows]
            seed_rows = rows
            continue
        norm = physical_norm(residual)
        if norm > cfg.residual_tolerance:
            return EquilibriumTrial(None, 'NUMERICAL_FAILURE', norm, total_evaluations,
                                    dict(optimizer_message=result.message, contact_candidates=nc,
                                         residual_index=int(np.argmax(np.abs(residual))),
                                         contacts=[[dict(gap=c['contact'].gap_m, N=c['normal_force'],
                                                        speed=float(np.linalg.norm(c['tangent_motion'])))
                                                    for c in row['contacts']] for row in rows]))
        return model._state_from_rows(previous, x_m, preload_N, rows, cache['q'], norm, total_evaluations)
    return EquilibriumTrial(None, 'NUMERICAL_FAILURE', float('inf'), total_evaluations,
                            dict(optimizer_message='contact candidate activation did not settle'))
