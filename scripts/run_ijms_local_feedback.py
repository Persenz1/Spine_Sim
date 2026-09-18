"""Replay selected accepted increments with four backplate boundary conditions.

Prepare all 50 original CUDA envelopes on disk before any contact solves.
The original trajectories and source archive remain read-only.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import csv
import json
import sys
import time
import zipfile
from copy import deepcopy
from pathlib import Path

PROGRAM = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROGRAM / 'src'), str(PROGRAM / 'scripts')]
import numpy as np

DEFAULT_ROOT = Path(r'E:\TestData\IJMS\final_最终确认')
DEFAULT_TABLE = Path(r'E:\Agent Tmp WS\IJMS_最后补算交接_20260918\仿真说明与输入表\附件\共同背板局部试验_正式基态输入表.csv')
DEFAULT_OUT = Path(r'E:\TestData\IJMS\local_feedback_20260918')


def read_table(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def prepare(root, rows, out):
    from ijms_confirmation_surface import generate, digest
    folder = out / 'surfaces'
    folder.mkdir(parents=True, exist_ok=True)
    identities = sorted({(r['material'], int(r['sample'])) for r in rows})
    for i, (material, sample) in enumerate(identities, 1):
        surface_id = f'{material}-s{sample:04d}'
        meta = json.loads((root / 'campaigns/surfaces' / (surface_id + '.json')).read_text('utf-8'))
        path = folder / (surface_id + '.npy')
        if path.exists():
            if digest(np.load(path, mmap_mode='r')) != meta['envelope_sha256']:
                raise ValueError('Saved envelope differs: ' + surface_id)
            print(f'SURFACE {i}/{len(identities)} {surface_id} reused', flush=True)
            continue
        params = meta['generation_parameters']
        job = dict(material_family=params['material'], material=params['subtype'],
                   seed=params['seed'], surface_id=surface_id, sample=sample)
        started = time.perf_counter()
        height, actual = generate(job, (params['size_x_m'], params['size_y_m']))
        for key in ('raw_sha256', 'envelope_sha256'):
            if actual[key] != meta[key]:
                raise ValueError(f'{surface_id}: regenerated {key} differs from original')
        np.save(path, height)
        (folder / (surface_id + '.json')).write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding='utf-8')
        del height
        print(f'SURFACE {i}/{len(identities)} {surface_id} saved, original raw/envelope match, {time.perf_counter()-started:.1f}s', flush=True)


def restored(state):
    return {k: np.asarray(v, dtype=bool if k == 'slip' else None) if isinstance(v, list) else v
            for k, v in state.items()}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def axial_status(array, state):
    s = state['compression']
    per_needle = np.full(array.n, 'IN_RANGE', dtype=object)
    per_needle[s < -1e-6] = 'AXIAL_EXTENSION_UNSUPPORTED'
    per_needle[s >= array.length] = 'EXPOSED_LENGTH_LIMIT'
    per_needle[np.isfinite(array.compression_stop) & (s >= array.compression_stop)] = 'COMPRESSION_STOP'
    status = ('AXIAL_EXTENSION_UNSUPPORTED' if np.any(s < -1e-6) else
              'EXPOSED_LENGTH_LIMIT' if np.any(s >= array.length) else 'IN_RANGE')
    return status, per_needle


def compare_states(state, other, p, n):
    active = state['normal_force'] > p / n * 1e-3
    other_active = other['normal_force'] > p / n * 1e-3
    return dict(T_error_N=float(-(state['force'][:, 0] - other['force'][:, 0]).sum()),
                Y_error_m=float(state['position'][1] - other['position'][1]),
                Z_error_m=float(state['position'][2] - other['position'][2]),
                force_max_error_N=float(np.max(np.abs(state['force'] - other['force']))),
                normal_max_error=float(np.max(np.abs(state['normal'] - other['normal']))),
                compression_max_error_m=float(np.max(np.abs(state['compression'] - other['compression']))),
                active_changed=bool(np.any(active != other_active)),
                slip_changed=bool(np.any(state['slip'] != other['slip'])))


def decompose(u, v, array, p):
    lu, lv = u['normal_force'], v['normal_force']
    nu, nv = u['normal'], v['normal']
    au, av = lu > p / array.n * 1e-3, lv > p / array.n * 1e-3
    common = au & av
    load = -(lu - lv) * (nu[:, 0] + nv[:, 0]) / 2 * common
    direction = -(lu + lv) / 2 * (nu[:, 0] - nv[:, 0]) * common
    support = -lu * nu[:, 0] * (au & ~av) + lv * nv[:, 0] * (av & ~au)
    tiny = -lu * nu[:, 0] * ~au + lv * nv[:, 0] * ~av
    tangent_u = u['force'] - lu[:, None] * nu
    tangent_v = v['force'] - lv[:, None] * nv
    tangent = -(tangent_u[:, 0] - tangent_v[:, 0])
    total = -(u['force'][:, 0] - v['force'][:, 0])
    board = np.broadcast_to(u['position'] - v['position'], (array.n, 3)).copy()
    compression = -(u['compression'] - v['compression'])[:, None] * array.axes
    center = u['center'] - v['center']
    detail = dict(normal_load_N=load, normal_direction_N=direction, support_set_N=support,
                  subthreshold_N=tiny, tangent_N=tangent, delta_T_N=total,
                  board_translation_m=board, compression_translation_m=compression,
                  actual_center_delta_m=center,
                  normal_angle_deg=np.degrees(np.arccos(np.clip(np.sum(nu * nv, axis=1), -1, 1))),
                  active_u=au, active_v=av)
    aggregate = {k: float(np.sum(detail[k])) for k in
                 ('normal_load_N', 'normal_direction_N', 'support_set_N', 'subthreshold_N', 'tangent_N', 'delta_T_N')}
    aggregate['force_closure_error_N'] = float(np.max(np.abs(total - load - direction - support - tiny - tangent)))
    aggregate['position_closure_error_m'] = float(np.max(np.abs(center - board - compression)))
    return aggregate, detail


def solve_base(row, trace, surface, campaign):
    from spine_sim.balanced_contact import CondensedArray
    from spine_sim.ijms import build_rods
    before = restored(trace['rows'][int(row['row_previous'])]['accepted_state'])
    recorded = restored(trace['rows'][int(row['row_target'])]['accepted_state'])
    settings = trace['numerical_settings']
    friction = campaign['base']['solver']
    array = CondensedArray(build_rods(trace['array']), surface, rigid=trace['rigid'],
                           mu_static=friction['friction_static'], mu_kinetic=friction['friction_kinetic'],
                           friction_transition_m=settings['friction_transition_um'] * 1e-6,
                           tolerance=settings['relative_solver_tolerance'])
    p = float(row['P_reference_N'])
    x = recorded['position'][0]
    if not (trace['seed'] == int(row['seed']) and trace['case_id'] == row['case_id']):
        raise ValueError('Input table/trace identity mismatch: ' + row['test_id'])
    variants = {'A': {}, 'B': dict(fixed_y=before['position'][1], fixed_z=before['position'][2]),
                'C': dict(fixed_y=before['position'][1]), 'D': dict(fixed_z=before['position'][2])}
    states = {}
    records = []
    details = dict(input=row, before=before, recorded_after=recorded, variants={})
    for variant, bc in variants.items():
        start = time.perf_counter()
        info = dict(row, variant=variant, fixed_y_m=bc.get('fixed_y'), fixed_z_m=bc.get('fixed_z'),
                    accepted=False, valid=False, status='', elapsed_s=0.)
        try:
            state = array.solve(x, p, deepcopy(before), recover=False, **bc)
        except ValueError as exc:
            if str(exc) != 'SURFACE_DOMAIN_LIMIT':
                raise
            info.update(status='SURFACE_DOMAIN_LIMIT', elapsed_s=time.perf_counter()-start)
            records.append(info)
            details['variants'][variant] = dict(summary=info, error=str(exc))
            continue
        info['elapsed_s'] = time.perf_counter() - start
        status, per_needle = axial_status(array, state)
        active = state['normal_force'] > p / array.n * 1e-3
        total = state['force'].sum(0)
        loads = np.maximum(state['force'][:, 2], 0)
        delta = state['position'] - before['position']
        stop = np.zeros(array.n)
        has_stop = np.isfinite(array.compression_stop)
        axial = np.sum(state['force'] * array.axes, axis=1)
        stop[has_stop] = np.maximum(0, -axial[has_stop] - state['compression'][has_stop] / array.spring[has_stop])
        height, _ = surface.query(state['center'][:, :2])
        movement2 = np.sum(state['movement']**2, axis=1)
        state.update(surface_query_xy_m=state['center'][:, :2], surface_height_m=height,
                     mu=array.mu_kinetic + (array.mu_static-array.mu_kinetic) * np.exp(-movement2 / array.friction_transition_m**2),
                     axial_status=per_needle, compression_stop_reactions_N=stop)
        info.update(accepted=state['accepted'], valid=bool(state['accepted'] and status == 'IN_RANGE'),
                    status=status if state['accepted'] else 'NUMERICAL_FAILURE', axial_range=status,
                    T_N=float(-total[0]), Py_N=float(total[1]), Pactual_N=float(total[2]),
                    Y_m=float(state['position'][1]), Z_m=float(state['position'][2]),
                    deltaY_m=float(delta[1]), deltaZ_m=float(delta[2]),
                    clamp_Ry_N=float(-total[1]) if 'fixed_y' in bc else None,
                    clamp_Rz_N=float(p-total[2]) if 'fixed_z' in bc else None,
                    total_external_Z_N=float(-total[2]) if 'fixed_z' in bc else -p,
                    active=int(active.sum()), neff_Z=float(loads.sum()**2 / (loads @ loads)) if np.any(loads) else None,
                    residual=state['residual'], nfev=state['nfev'], fallback=state['linear_solver_fallback'],
                    active_changed=bool(np.any(active != (before['normal_force'] > p / array.n * 1e-3))),
                    slip_changed=bool(np.any(state['slip'] != before['slip'])),
                    axial_balance_max_N=float(np.max(np.abs(state['compression']/array.spring + stop + axial))),
                    y_boundary_residual=float(total[1]/p if 'fixed_y' not in bc else (state['position'][1]-bc['fixed_y'])/1e-5),
                    z_boundary_residual=float((total[2]-p)/p if 'fixed_z' not in bc else (state['position'][2]-bc['fixed_z'])/1e-5))
        if variant == 'A':
            info.update({'replay_' + k: v for k, v in compare_states(state, recorded, p, array.n).items()})
        elif 'A' in states:
            cmp = compare_states(state, states['A'], p, array.n)
            info.update(active_differs_A=cmp['active_changed'], slip_differs_A=cmp['slip_changed'])
        states[variant] = state
        records.append(info)
        details['variants'][variant] = dict(summary=info, state=state)
    effect = {k: row[k] for k in ('test_id', 'material', 'sample', 'case_id', 'selection')}
    valid = {r['variant']: r for r in records if r['valid']}
    for label in variants:
        rec = next(r for r in records if r['variant'] == label)
        effect.update({f'{label}_status': rec['status'], f'{label}_T_N': rec.get('T_N'), f'{label}_Pactual_N': rec.get('Pactual_N')})
    effect['all_valid'] = len(valid) == 4
    details['decompositions'] = {}
    for tag, u, v in [('YZ', 'A', 'B'), ('Z', 'C', 'B'), ('Y', 'D', 'B'), ('A_minus_C', 'A', 'C')]:
        effect['deltaT_' + tag + '_N'] = None
        if u in valid and v in valid:
            aggregate, detail = decompose(states[u], states[v], array, p)
            effect.update({'deltaT_' + tag + '_N': aggregate['delta_T_N'],
                           'deltaT_' + tag + '_perP0': aggregate['delta_T_N'] / p})
            effect.update({tag + '_' + k: value for k, value in aggregate.items() if k != 'delta_T_N'})
            details['decompositions'][tag] = detail
    effect['deltaT_interaction_N'] = None
    if len(valid) == 4:
        interaction = valid['A']['T_N'] - valid['C']['T_N'] - valid['D']['T_N'] + valid['B']['T_N']
        effect.update(deltaT_interaction_N=interaction, deltaT_interaction_perP0=interaction/p,
                      interaction_identity_error_N=effect['deltaT_YZ_N'] - effect['deltaT_Z_N'] - effect['deltaT_Y_N'] - interaction)
    for rec in records:
        if rec['variant'] == 'A':
            effect.update({k: v for k, v in rec.items() if k.startswith('replay_')})
    details['effect'] = effect
    return details


def run(root, rows, out):
    from spine_sim.balanced_contact import EnvelopeSurface
    from ijms_compact import unpack_trace
    import scipy
    for material, sample in {(r['material'], int(r['sample'])) for r in rows}:
        path = out / 'surfaces' / f'{material}-s{sample:04d}.npy'
        if not path.exists():
            raise FileNotFoundError(f'Prepare all surfaces before solving: {path}')
    campaign = json.loads((root / 'campaigns/scan.json').read_text('utf-8'))
    (out / 'states').mkdir(exist_ok=True)
    (out / 'execution.json').write_text(json.dumps(dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        solver_baseline='returned c0eaf52, fixed Y/Z boundary extension', recover=False, max_nfev=80,
        source_root=str(root), table='base_states.csv', terrain_mode='all envelopes generated and saved before solving'), indent=2), encoding='utf-8')
    surface = trace = None
    last_surface = last_trace = None
    all_records, all_effects = [], []
    started = time.perf_counter()
    for i, row in enumerate(sorted(rows, key=lambda r: (r['material'], int(r['sample']), r['case_id'], r['selection'])), 1):
        dest = out / 'states' / (row['test_id'] + '.json')
        if dest.exists():
            detail = json.loads(dest.read_text('utf-8'))
        else:
            surface_id = f"{row['material']}-s{int(row['sample']):04d}"
            if surface_id != last_surface:
                meta = json.loads((root / 'campaigns/surfaces' / (surface_id + '.json')).read_text('utf-8'))
                height = np.load(out / 'surfaces' / (surface_id + '.npy'), mmap_mode='r')
                processing = meta['processing']
                surface = EnvelopeSurface(height, processing['envelope_grid_m'], processing['envelope_grid_m'], processing['origin_xy_m'])
                del height
                last_surface = surface_id
            identity = (row['archive'], row['member'])
            if identity != last_trace:
                with zipfile.ZipFile(root / 'compact' / row['archive']) as z:
                    trace = unpack_trace(z.read(row['member'] + '.zip'))
                last_trace = identity
            detail = solve_base(row, trace, surface, campaign)
            dest.write_text(json.dumps(detail, ensure_ascii=False, allow_nan=False, default=json_default), encoding='utf-8')
        all_records.extend(d['summary'] for d in detail['variants'].values())
        all_effects.append(detail['effect'])
        if i % 10 == 0 or i == len(rows):
            write_csv(out / 'local_states.csv', all_records)
            write_csv(out / 'local_effects.csv', all_effects)
            print(f'BASE {i}/{len(rows)} valid targets {sum(r["valid"] for r in all_records)}/{len(all_records)} elapsed {time.perf_counter()-started:.1f}s', flush=True)
    print('DONE local boundary replay', len(all_records), 'targets', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--table', type=Path, default=DEFAULT_TABLE)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUT)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='First N base states for initial replay checks')
    args = ap.parse_args()
    rows = read_table(args.table)
    args.output.mkdir(parents=True, exist_ok=True)
    import shutil
    saved_table = args.output / 'base_states.csv'
    if args.table.resolve() != saved_table.resolve():
        shutil.copy2(args.table, saved_table)
    if args.prepare_only:
        prepare(args.root, rows, args.output)
    else:
        run(args.root, rows[:args.limit] if args.limit else rows, args.output)


if __name__ == '__main__':
    main()
