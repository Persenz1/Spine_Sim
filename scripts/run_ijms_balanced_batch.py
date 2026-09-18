"""Resume the fixed reference/single-realization design table with bounded workers."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,csv,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import numpy as np
from spine_sim.small_array_campaign import read_config,design_table,prepare_surface,Shard
from spine_sim.balanced_contact import MODEL_VERSION
from run_ijms_balanced_contact import main as run_one,prepare_envelope


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    # Windows readers can briefly prevent replacement of the dashboard file.
    for attempt in range(20):
        try:
            os.replace(tmp,path)
            return
        except PermissionError:
            if attempt==19:raise
            time.sleep(0.05)


def summarize(result):
    rows=result['rows'];config=result['configuration'];drag=[r for r in rows if r['phase']=='drag']
    preload=[r for r in rows if r['phase']=='preload']
    path=preload[-1:]+drag
    x=np.array([r['x_m'] for r in path]);t=np.array([r['T_N'] for r in path])
    dx=np.diff(x);valid=np.array([not r.get('unresolved_rearrangement',r.get('reconfigured',False)) for r in path[1:]],dtype=bool)
    width=dx[valid];values=((t[1:]+t[:-1])/2)[valid]
    distance=float(width.sum());order=np.argsort(values)
    p90=float(values[order][np.searchsorted(np.cumsum(width[order]),.9*distance)]) if distance>0 else None
    return dict(design=config['design'],material=config['material'],preload_N=config['preload'],
        rigid=result['rigid'],model=result['model'],status=result['status'],solve_s=result['solve_s'],
        placement_attempt_count=len(result.get('placement_attempts',[])),
        selected_placement_index=result.get('selected_placement_index'),
        selected_start_xy_m=result.get('selected_start_xy_m'),
        placement_search_status=result.get('placement_search_status','LEGACY_SINGLE_PLACEMENT'),
        coverage_mm=float(x[-1]*1000) if len(x) else 0.,resolved_distance_mm=distance*1000,
        mean_T_resolved_N=float(np.dot(values,width)/distance) if distance>0 else None,p90_T_resolved_N=p90,
        recoveries=sum(r.get('reconfigured',False) for r in rows),
        unresolved_rearrangements=sum(r.get('unresolved_rearrangement',r.get('reconfigured',False)) for r in rows),
        max_force_balance_error_N=max((r.get('force_balance_error_N',0) for r in rows),default=0),
        max_penetration_m=max((r.get('max_penetration_m',0) for r in rows),default=0),
        trace_file=result['trace_file'])


def worker(job):
    return summarize(run_one(job))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['reference','all'],default='reference')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--rerun-failed',action='store_true')
    args=parser.parse_args();config_path=Path('experiments/ijms_balanced_contact.json')
    settings=read_config(config_path);base=read_config(config_path.parent/settings['base_config'])
    root=Path(settings['output_dir']);(root/'campaigns').mkdir(parents=True,exist_ok=True)
    write_json(root/'campaigns'/f'{MODEL_VERSION}.json',dict(settings=settings,base=base))
    selected=[d for d in design_table(base) if args.phase=='all' or d['full_reference']]
    started=time.perf_counter();records=[];jobs=[]
    state=dict(model=MODEL_VERSION,phase=args.phase,planned_cases=len(selected)*len(base['materials'])*len(base['preloads_N']),
               completed_cases=0,workers=args.workers,status='PREPARING',elapsed_s=0,counts={},errors=[])
    def progress():
        state.update(completed_cases=len(records),elapsed_s=time.perf_counter()-started,
                     counts=dict(Counter(r['status'] for r in records)))
        write_json(root/'batch_status.json',state)
        if len(records)%10==0 or state['status']!='RUNNING':write_json(root/'batch_summary.json',records)
    progress()
    for mi,material in enumerate(base['materials']):
        surface=prepare_surface(Path('E:/TestData/IJMS/trend'),base,Shard(settings['sample_index'],mi,0,True,()),terrain_backend='cuda')
        for radius in sorted({d['array']['spine']['tip_radius_m'] for d in selected}):
            prepare_envelope(surface,radius,root,settings['surface_envelope_grid_um'],settings['surface_smoothing_sigma_um'])
        for design in selected:
            rigid=settings['needle_model'][design['array']['spine']['mount_type']]=='rigid'
            for preload in base['preloads_N']:
                name=f"{design['name']}-{material['subtype']}-P{preload:g}-{'rigid' if rigid else 'elastic'}-10mm-{settings['path_step_um']:g}um-v{MODEL_VERSION.rsplit('-',1)[-1]}.json"
                path=root/'results'/name
                if path.exists():
                    result=json.loads(path.read_text(encoding='utf-8'))
                    previous_settings=result.get('numerical_settings',{})
                    same_sampling=all(previous_settings.get(k)==settings.get(k) for k in ['sample_index','preload_relocation'])
                    if result['model']==MODEL_VERSION and same_sampling and (result['status']=='BALANCED_COMPLETE' or not args.rerun_failed):
                        records.append(summarize(dict(result,trace_file=name)));continue
                jobs.append(['--design',design['name'],'--material',material['subtype'],'--preload',str(preload)])
    state['status']='RUNNING';progress()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(worker,job):job for job in jobs}
        for future in as_completed(pending):
            try:records.append(future.result())
            except Exception as exc:
                job=pending[future];state['errors'].append(dict(job=job,error=f'{type(exc).__name__}: {exc}'))
                records.append(dict(design=job[1],material=job[3],preload_N=float(job[5]),status='EXECUTION_ERROR'))
            progress()
    state['status']='COMPLETED';progress()
    fields=list(dict.fromkeys(k for row in records for k in row))
    with (root/'results'/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)
    print(json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
