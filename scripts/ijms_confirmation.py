"""Frozen final-confirmation queue and observation-only v6 recording."""
import csv,json,time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import numpy as np
from spine_sim.balanced_contact import CondensedArray,MODEL_VERSION
from spine_sim.ijms import build_rods
from spine_sim.placement_retry import run_with_placements
from run_ijms_balanced_batch import summarize
from ijms_compact import pack_trace

PROGRAM=Path(__file__).resolve().parents[1]
BATCH='ijms-final-confirmation-v6-20260916'
MATERIALS=[('sandpaper','P240'),('sandpaper','P100'),('sandpaper','P40'),
           ('red_brick','fired_brick_standard'),('concrete','rough_wall')]
PLANNED_CASES=90000
SURFACE=dict(size_x_m=.068,size_y_m=.058,resolution_m=1e-5,origin_xy_m=[-.029,-.029],
             envelope_grid_m=20e-6,smoothing_sigma_m=10e-6,tip_radius_m=100e-6,
             mode='synthetic',backend='cuda',seed_start=2026100000,material_seed_stride=10000)


@lru_cache(maxsize=1)
def inputs():
    base=json.loads((PROGRAM/'experiments/ijms_small_array_ritz.json').read_text('utf-8'))
    settings=json.loads((PROGRAM/'experiments/ijms_fine_execution.json').read_text('utf-8'))
    records=list(csv.DictReader((PROGRAM/'experiments/ijms_confirmation_designs.csv').open(encoding='utf-8-sig')))
    designs={}
    for r in records:
        spine=deepcopy(base['reference_spine'])
        spine.update(free_length_m=float(r['free_length_mm'])/1000,diameter_m=float(r['body_diameter_mm'])/1000,
            taper_length_m=float(r['taper_length_mm'])/1000,tip_radius_m=float(r['radius_um'])*1e-6,
            mount_type='spring',spring_stiffness_N_per_m=float(r['spring_stiffness_N_per_m']),
            max_compression_m=float(r['nominal_spring_travel_mm'])/1000)
        nx,ny=int(r['nx']),int(r['ny']);px=float(r['pitch_x_mm'])/1000;py=float(r['pitch_y_mm'])/1000
        array=dict(nx=nx,ny=ny,spacing_x_m=px,spacing_y_m=py,spine=spine,
            theta_deg=float(r['angle_deg']),yaw_deg=float(r['yaw_deg']),
            tip_positions_xy_m=[[(i-(nx-1)/2)*px,(j-(ny-1)/2)*py] for j in range(ny) for i in range(nx)],
            loaded_area_m2=nx*ny*px*py)
        designs[r['design_id']]=dict(array=array,confirmation_id=r['confirmation_id'])
    return settings,base,records,designs


def jobs():
    _,_,records,_=inputs()
    for sample in range(1000):
        for mi,(family,material) in enumerate(MATERIALS):
            for r in records:
                yield dict(key=f'{BATCH}/{material}/s{sample:04d}/{r["case_id"]}.json.gz',
                    batch=BATCH,domain='confirmation',sample=sample,material=material,material_family=family,
                    seed=SURFACE['seed_start']+mi*SURFACE['material_seed_stride']+sample,
                    design=r['design_id'],confirmation_id=r['confirmation_id'],case_id=r['case_id'],
                    preload=float(r['preload_N']),detailed=sample<20,role='final_confirmation',
                    surface_id=f'{material}-s{sample:04d}')


def state_copy(state):
    return {key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in state.items()}


def export_summary(connection,path):
    from itertools import chain
    cursor=connection.execute('SELECT summary_json FROM cases ORDER BY case_key')
    first=cursor.fetchone();temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
        if first:
            row=json.loads(first[0]);writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
            for value in chain([first],cursor):writer.writerow(json.loads(value[0]))
    temporary.replace(path)


class Recorder:
    """Copy accepted solver values; never change the state or solver choices."""
    def __init__(self,surface,detailed):
        self.surface=surface;self.detailed=detailed;self.events=[]

    def __call__(self,event,state,row,placement_index):
        if row is not None:
            row.update(requested_preload_N=state['requested_preload_N'],
                contact_normals=state['normal'].tolist(),gaps_m=state['gap'].tolist(),
                tangential_movement_m=state['movement'].tolist(),slip_flags=state['slip'].tolist())
            if self.detailed:
                row['accepted_state']=state_copy(state)
                row['surface_query_xy_m']=state['center'][:,:2].tolist()
                row['surface_height_m']=self.surface.query(state['center'][:,:2])[0].tolist()
        elif self.detailed:
            self.events.append(dict(event=event,placement_index=placement_index,state=state_copy(state),
                surface_query_xy_m=state['center'][:,:2].tolist(),
                surface_height_m=self.surface.query(state['center'][:,:2])[0].tolist()))


def solve(job,surface,progress=None):
    settings,base,_,designs=inputs();design=designs[job['design']]
    array=CondensedArray(build_rods(design['array']),surface,rigid=True,
        mu_static=base['solver']['friction_static'],mu_kinetic=base['solver']['friction_kinetic'],
        friction_transition_m=settings['friction_transition_um']*1e-6,tolerance=settings['relative_solver_tolerance'])
    recorder=Recorder(surface,job['detailed']);distance=job.get('test_distance_mm',10.)/1000
    started=time.perf_counter();cpu=time.process_time()
    result=run_with_placements(array,np.zeros(2),job['preload'],distance,settings['path_step_um']*1e-6,
        offsets_m=settings['preload_relocation']['offsets_m'],attempt_cpu_budget_s=settings['preload_attempt_cpu_s'],
        total_cpu_budget_s=settings['preload_total_cpu_s'],progress=progress,state_observer=recorder)
    continuations=[]
    while result['status']=='BALANCED_BUDGET_LIMIT' and result.get('phase')=='drag':
        previous=result;index=result['selected_placement_index']
        result=array.run(np.array(result['selected_start_xy_m']),job['preload'],distance,settings['path_step_um']*1e-6,
            cpu_budget_s=settings['drag_chunk_cpu_s'],resume=previous,progress=progress,
            state_observer=lambda event,state,row:recorder(event,state,row,index))
        for key in ('placement_attempts','selected_placement_index','selected_start_xy_m','placement_search_status'):
            result[key]=previous[key]
        continuations.append(dict(previous_status=previous['status'],status=result['status'],
                                  last_x_m=float(result['checkpoint']['position'][0])))
        if np.array_equal(result['checkpoint']['position'],previous['checkpoint']['position']) and result['status']=='BALANCED_BUDGET_LIMIT':
            result['reason']='NO_ACCEPTED_PROGRESS_WITHIN_CPU_CHUNK';break
    result.update(model=MODEL_VERSION,configuration=dict(design=job['design'],material=job['material'],preload=job['preload']),
        batch=BATCH,case_id=job['case_id'],confirmation_id=job['confirmation_id'],array=design['array'],
        numerical_settings=settings,rigid=True,solve_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu,
        continuations=continuations,sample_index=job['sample'],surface_id=job['surface_id'],seed=job['seed'],
        surface_set='confirmation',role=job['role'],detailed=job['detailed'],
        execution=dict(solver_baseline='c0eaf52',linear_solver='baseline_dense_with_existing_fallback',
            balance_row_scaling='none',surface_query='baseline_fitpack',storage='compact-stream-v1',
            distance_m=distance,test='test_distance_mm' in job),
        geometry=dict(needle_ids=list(range(array.n)),guide_positions_m=array.guides.tolist(),axes=array.axes.tolist(),
            free_lengths_m=array.length.tolist(),tip_radii_m=array.radius.tolist(),
            coordinates='global center=guide+backplate_position+(free_length-compression)*axis; x_m,Y_m relative to selected_start; Z_m absolute'),
        approximations=['rigid_needle','spring_axial_force_without_configurational_term','smooth_single_support_envelope_20um',
                        'no_body_or_cap_checks','static_kinetic_transition_1um','quasistatic_rearrangements_without_transient_resolution'])
    if job['detailed']:result['detailed_events']=recorder.events
    summary=summarize(dict(result,trace_file=job['key']))
    summary.update(batch=BATCH,case_id=job['case_id'],confirmation_id=job['confirmation_id'],sample_index=job['sample'],
                   surface_set='confirmation',role=job['role'],detailed=job['detailed'],seed=job['seed'])
    tick=time.perf_counter();packed=pack_trace(result,job['key']);summary['pack_s']=time.perf_counter()-tick
    summary['trace_bytes']=len(packed)
    return summary,packed
