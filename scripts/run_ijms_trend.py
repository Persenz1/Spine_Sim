"""Bounded trend campaign; shared track envelopes amortize geometry work."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,csv,json,time,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import numpy as np
from spine_sim.core.identity import stable_hash
from spine_sim.ijms import build_rods
from spine_sim.small_array_campaign import read_config,design_table,prepare_surface,Shard,freeze_config
from spine_sim.trend_array import MODEL_VERSION,sampled_envelope,solve_trend


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def geometry_id(design):
    a=design['array']
    return stable_hash(dict(xy=a['tip_positions_xy_m'],r=a['spine']['tip_radius_m']))[:16]


def run_group(job):
    root=Path(job['root']);settings=job['settings'];base=job['base']
    surface=job['surface'];designs=job['designs'];tag=job['tag']
    started=time.perf_counter()
    x=np.linspace(0,base['path']['search_distance_m'],int(np.ceil(base['path']['search_distance_m']/settings['path_step_m']))+1)
    envelope_path=root/'tmp'/f'envelope-{tag}.npz'
    if envelope_path.exists():
        with np.load(envelope_path) as saved:
            envelope=saved['envelope']
    else:
        a=designs[0]['array'];xy=np.asarray(a['tip_positions_xy_m'],float)
        track=np.broadcast_to(xy,(len(x),*xy.shape)).copy()
        track[:,:,0]+=x[:,None]+base['path']['start_xy_m'][0]+base['surface']['placement_xy_m'][0]
        track[:,:,1]+=base['path']['start_xy_m'][1]+base['surface']['placement_xy_m'][1]
        height=np.load(surface['path'],mmap_mode='r',allow_pickle=False)
        envelope=sampled_envelope(height,surface['dx_m'],surface['dy_m'],surface['origin_xy_m'],track,
                                  a['spine']['tip_radius_m'],settings['envelope_spacing_m'])
        np.savez_compressed(envelope_path,envelope=envelope)
    records=[];curves={name:[] for name in ('T_N','Z_m','loads_N','active_count','max_compression_m','range_flag')}
    for design in designs:
        rods=build_rods(design['array']);a=design['array']
        for preload in base['preloads_N']:
            result=solve_trend(x,envelope,rods,preload,base['solver']['friction_kinetic'])
            case_key=stable_hash(dict(model=MODEL_VERSION,design=design['name'],preload=preload,
                                     material=job['material'],sample=settings['sample_index'],settings=settings))[:20]
            records.append(dict(case_id=case_key,design=design['name'],material=job['material'],preload_N=preload,
                radius_um=a['spine']['tip_radius_m']*1e6,angle_deg=a.get('theta_deg','gradient60-80'),
                mount=a['spine']['mount_type'],stiffness_N_m=a['spine']['spring_stiffness_N_per_m'],
                nx=a['nx'],ny=a['ny'],spacing_mm=a['spacing_x_m']*1e3,
                T_mean_N=float(np.trapezoid(result['T_N'],x)/x[-1]),T_p90_N=float(np.quantile(result['T_N'],.9)),
                T_peak_N=float(np.max(result['T_N'])),mean_active_spines=float(np.mean(result['active_count'])),
                max_compression_mm=float(np.max(result['max_compression_m'])*1e3),
                stress_estimate_max_MPa=float(np.max(result['max_stress_estimate_Pa'])/1e6),
                range_flag_fraction=float(np.mean(result['range_flag'])),
                normal_balance_error_N=result['normal_balance_error_N'],
                status='TREND_RANGE_FLAGGED' if np.any(result['range_flag']) else 'TREND_COMPLETE',
                geometry_checks='not_evaluated',model=MODEL_VERSION,sample_index=settings['sample_index'],
                trace_file=f'{tag}.npz',trace_index=len(records)))
            for key in curves:
                curves[key].append(result[key])
    np.savez_compressed(root/'results'/f'{tag}.npz',x_m=x,
        **{key:np.asarray(value,dtype=np.float32) for key,value in curves.items()})
    write_json(root/'results'/f'{tag}.json',records)
    return dict(records=records,seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('experiments/ijms_trend.json'))
    parser.add_argument('--phase',choices=['reference','all'],default='reference')
    parser.add_argument('--workers',type=int)
    args=parser.parse_args();settings=read_config(args.config)
    root=Path(settings['output_dir']);base=read_config(args.config.parent/settings['base_config'])
    base['name']=settings['name'];base['surface']['realizations_per_material']=1
    base['trend_approximation']=dict(settings,model=MODEL_VERSION)
    for name in ('results','logs','tmp','surfaces','campaigns'):
        (root/name).mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(root/'tmp')
    freeze_config(root,base)
    # Immutable same-wall inputs, shared on the same NTFS volume.
    for source in Path('E:/TestData/IJMS/multipoint_ritz/surfaces').glob('*'):
        target=root/'surfaces'/source.name
        if source.is_file() and not target.exists():
            if source.suffix=='.npy': os.link(source,target)
            else: shutil.copyfile(source,target)
    designs=design_table(base)
    selected=[d for d in designs if args.phase=='all' or d['full_reference']]
    groups={}
    for design in selected: groups.setdefault(geometry_id(design),[]).append(design)
    jobs=[]
    for material_index,material in enumerate(base['materials']):
        surface=prepare_surface(root,base,Shard(settings['sample_index'],material_index,0,True,()),terrain_backend='cuda')
        for geometry,items in groups.items():
            tag=f"s{settings['sample_index']:03d}-{material['subtype']}-{geometry}"
            jobs.append(dict(root=str(root),settings=settings,base=base,surface=surface,designs=items,
                             material=material['subtype'],tag=tag))
    started=time.perf_counter();records=[];errors=[]
    state=dict(kind='trend',model=MODEL_VERSION,phase=args.phase,planned_cases=len(selected)*15,
               full_single_realization_cases=len(designs)*15,completed_cases=0,failed_jobs=0,status='RUNNING',
               path_mm=10.,path_step_um=settings['path_step_m']*1e6,workers=args.workers or settings['workers'],
               note='趋势近似：线性法向分载、名义轨迹、连续滑动；不用于精确峰值、粘滑或强度结论。')
    def progress():
        state.update(completed_cases=len(records),failed_jobs=len(errors),elapsed_s=time.perf_counter()-started,
                     range_flagged_cases=sum(r['status']=='TREND_RANGE_FLAGGED' for r in records))
        write_json(root/'trend_status.json',state)
    progress()
    with ProcessPoolExecutor(max_workers=state['workers']) as pool:
        futures={pool.submit(run_group,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                result=future.result();records.extend(result['records'])
                print(json.dumps(dict(group=job['tag'],cases=len(result['records']),seconds=result['seconds'])),flush=True)
            except Exception as exc:
                errors.append(dict(group=job['tag'],error=f'{type(exc).__name__}: {exc}'))
                print(json.dumps(errors[-1]),flush=True)
            progress()
    records.sort(key=lambda r:(r['design'],r['material'],r['preload_N']))
    write_json(root/'results'/'summary.json',records)
    with (root/'results'/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        if records:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write_json(root/'logs'/'errors.json',errors)
    state['status']='COMPLETED' if not errors else 'COMPLETED_WITH_ERRORS';progress()
    print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
