"""Frozen 206-design CPU campaign. Surface generation is a separate local action."""
import csv, gzip, json, os, time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import numpy as np
from spine_sim.balanced_contact import CondensedArray, EnvelopeSurface, MODEL_VERSION
from spine_sim.ijms import build_rods
from spine_sim.placement_retry import run_with_placements
from run_ijms_balanced_batch import summarize

PROGRAM=Path(__file__).resolve().parents[1]
PACKAGE=PROGRAM.parent
SUFFIX='-r100-envelope20um-sigma10um.npy'

@lru_cache(maxsize=1)
def inputs():
    base=json.loads((PROGRAM/'experiments/ijms_small_array_ritz.json').read_text(encoding='utf-8'))
    settings=json.loads((PROGRAM/'experiments/ijms_fine_execution.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((PROGRAM/'experiments/ijms_fine_designs.csv').open(encoding='utf-8-sig')))
    designs={}
    for row in rows:
        nx,ny=int(row['nx']),int(row['ny']);pitch=float(row['pitch_x_mm'])/1000
        spine=deepcopy(base['reference_spine'])
        spine.update(tip_radius_m=float(row['radius_um'])*1e-6,mount_type=row['mount'],
            spring_stiffness_N_per_m=float(row['spring_stiffness_N_per_m'] or 0),
            max_compression_m=.004 if row['mount']=='spring' else 0)
        array=dict(nx=nx,ny=ny,spacing_x_m=pitch,spacing_y_m=pitch,spine=spine,
            tip_positions_xy_m=[[(i-(nx-1)/2)*pitch,(j-(ny-1)/2)*pitch] for j in range(ny) for i in range(nx)],loaded_area_m2=nx*ny*pitch**2)
        if row['angle_deg']:array['theta_deg']=float(row['angle_deg'])
        else:
            array['angle_gradient_deg']=dict(toe=float(row['toe_angle_deg']),heel=float(row['heel_angle_deg']))
            array['heel_free_length_m']=spine.pop('free_length_m')
        designs[row['design_id']]=dict(array=array,record=row)
    return settings,base,designs

def job_table(stage,root):
    settings,base,designs=inputs()
    for domain in (['small','large'] if stage=='all' else [stage]):
        for sample in range(100 if domain=='small' else 20):
            # All work for a surface is adjacent to reuse the worker spline and OS cache.
            for material in base['materials']:
                for name,design in designs.items():
                    r=design['record'];large=int(r['needle_count'])>25
                    if domain=='small' and large:continue
                    if domain=='large' and not large and r['additional_large_surface_reference']!='True':continue
                    preloads=r['primary_preloads_N'] if domain=='small' or large else r['additional_large_preloads_N']
                    for preload in map(float,preloads.split(';')):
                        key=f"{domain}/s{sample:03d}/{material['subtype']}/{name}-P{preload:g}-v6.json.gz"
                        yield dict(key=key,root=str(root),domain=domain,sample=sample,design=name,
                            material=material['subtype'],material_family=material['material'],preload=preload,
                            role='screening' if domain=='small' else 'validation' if large else 'validation_reference')

@lru_cache(maxsize=1)
def field(path,origin):
    # Read-only mappings let Windows share the three coefficient arrays across
    # all workers. The interpolation and its derivative coefficients are unchanged.
    import msvcrt
    from scipy.interpolate import RectBivariateSpline
    source=Path(path)
    cache=PACKAGE/'data'/'spline_cache'/source.parent.name/source.stem
    cache.mkdir(parents=True,exist_ok=True)
    index=cache/'knots.npz'
    with (cache/'build.lock').open('a+b') as lock:
        lock.seek(0,2)
        if not lock.tell():lock.write(b'0');lock.flush()
        lock.seek(0)
        # LK_LOCK retries only for ten seconds; a full large-field build can take longer.
        while True:
            try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(.1)
        try:
            if not index.exists():
                original=EnvelopeSurface(np.load(source,mmap_mode='r'),20e-6,20e-6,origin)
                info=dict(x=original.x,y=original.y)
                for name in ('spline','slope_x','slope_y'):
                    obj=getattr(original,name);tx,ty,c=obj.tck
                    np.save(cache/(name+'.npy'),c)
                    info[name+'_tx']=tx;info[name+'_ty']=ty
                    info[name+'_degrees']=obj.degrees
                temporary=cache/'knots.tmp.npz';np.savez(temporary,**info);os.replace(temporary,index)
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    result=EnvelopeSurface.__new__(EnvelopeSurface)
    with np.load(index) as info:
        result.x=info['x'];result.y=info['y']
        for name in ('spline','slope_x','slope_y'):
            coeff=np.load(cache/(name+'.npy'),mmap_mode='r')
            tck=(info[name+'_tx'],info[name+'_ty'],coeff,*map(int,info[name+'_degrees']))
            setattr(result,name,RectBivariateSpline._from_tck(tck))
    return result

def save(path,result):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.gz.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as f:json.dump(result,f,ensure_ascii=False,allow_nan=False)
    os.replace(tmp,path)

def export_csv(connection,path):
    fields=['design','material','preload_N','sample_index','surface_set','role','rigid','model','status','solve_s',
        'placement_attempt_count','selected_placement_index','selected_start_xy_m','placement_search_status',
        'coverage_mm','resolved_distance_mm','mean_T_resolved_N','p90_T_resolved_N','recoveries',
        'unresolved_rearrangements','max_force_balance_error_N','max_penetration_m','trace_file','error']
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for (value,) in connection.execute('SELECT summary_json FROM cases ORDER BY case_key'):
            writer.writerow(json.loads(value))
    os.replace(temporary,path)

def worker(job):
    settings,base,designs=inputs();target=Path(job['root'])/'results'/job['key']
    if target.exists():
        with gzip.open(target,'rt',encoding='utf-8') as f:result=json.load(f)
        return dict(summarize(dict(result,trace_file=job['key'])),sample_index=job['sample'],surface_set=job['domain'],role=job['role'])
    t=time.perf_counter();design=designs[job['design']];rigid=design['record']['mount']=='spring'
    surface=PACKAGE/'data/surfaces'/job['domain']/f"{job['material_family']}-{job['material']}-s{job['sample']:03d}{SUFFIX}"
    origin=(-.029,-.029) if job['domain']=='small' else (-.065,-.065)
    array=CondensedArray(build_rods(design['array']),field(str(surface),origin),rigid=rigid,
        mu_static=base['solver']['friction_static'],mu_kinetic=base['solver']['friction_kinetic'],
        friction_transition_m=settings['friction_transition_um']*1e-6,tolerance=settings['relative_solver_tolerance'])
    prepared=time.perf_counter()-t;tick=time.process_time();checkpoint=target.with_suffix('.checkpoint.gz')
    if checkpoint.exists():
        with gzip.open(checkpoint,'rt',encoding='utf-8') as f:result=json.load(f)
    else:
        result=run_with_placements(array,np.array([0.,0.]),job['preload'],.01,settings['path_step_um']*1e-6,
            offsets_m=settings['preload_relocation']['offsets_m'],
            attempt_cpu_budget_s=settings['preload_attempt_cpu_s'],total_cpu_budget_s=settings['preload_total_cpu_s'])
    continuations=list(result.get('continuations',[]))
    while result['status']=='BALANCED_BUDGET_LIMIT' and result.get('phase')=='drag':
        result['continuations']=continuations;save(checkpoint,result)
        previous=result;start=np.array(result['selected_start_xy_m'])
        result=array.run(start,job['preload'],.01,settings['path_step_um']*1e-6,
            cpu_budget_s=settings['drag_chunk_cpu_s'],resume=previous)
        for k in ('placement_attempts','selected_placement_index','selected_start_xy_m','placement_search_status'):
            result[k]=previous[k]
        continuations.append(dict(previous_status=previous['status'],status=result['status'],last_x_m=float(result['checkpoint']['position'][0])))
        if np.array_equal(result['checkpoint']['position'],previous['checkpoint']['position']) and result['status']=='BALANCED_BUDGET_LIMIT':
            result['reason']='NO_ACCEPTED_PROGRESS_WITHIN_CPU_CHUNK';break
    result.update(model=MODEL_VERSION,configuration=dict(design=job['design'],material=job['material'],preload=job['preload']),
        numerical_settings=settings,array=design['array'],sample_index=job['sample'],surface_set=job['domain'],role=job['role'],
        rigid=rigid,solve_s=time.perf_counter()-t-prepared,preparation_s=prepared,cpu_s=time.process_time()-tick,
        continuations=continuations,approximations=['rigid_needle' if rigid else 'small_rotation_linear_beam',
            'spring_axial_force_without_configurational_term','smooth_single_support_envelope_20um','no_body_or_cap_checks',
            'static_kinetic_transition_1um','reported_quasistatic_rearrangements_without_transient_resolution'])
    save(target,result)
    if checkpoint.exists():checkpoint.unlink()
    summary=summarize(dict(result,trace_file=job['key']))
    return dict(summary,sample_index=job['sample'],surface_set=job['domain'],role=job['role'])
