"""Authorized full-factorial scan with bounded submission and a resumable ledger."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,csv,json,sqlite3,time,shutil,ctypes
from collections import Counter,defaultdict,deque
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from run_ijms_balanced_contact import campaign_inputs,prepare_envelope,cached_envelope,result_name,read_result,main as run_one
from run_ijms_balanced_batch import summarize,write_json
from spine_sim.small_array_campaign import prepare_surface,Shard
from spine_sim.balanced_contact import MODEL_VERSION
from spine_sim.wave_control import begin_wave,end_wave


def available_memory():
    class MemoryStatus(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
                   ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended']]
    status=MemoryStatus();status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):raise OSError('GlobalMemoryStatusEx failed')
    return status.avail_phys


def ordered_designs(designs):
    groups=defaultdict(deque)
    for d in designs.values():
        p=d['array']['spine'];groups[(p['tip_radius_m'],p['mount_type'],p['spring_stiffness_N_per_m'])].append(d)
    ordered=[]
    while any(groups.values()):
        for group in groups.values():
            if group:ordered.append(group.popleft())
    return ordered


def job_table(config_path,settings,base,designs):
    for sample in settings['sample_indices']:
        for first in range(0,len(designs),32):
            for material in base['materials']:
                for design in designs[first:first+32]:
                    rigid=settings['needle_model'][design['array']['spine']['mount_type']]=='rigid'
                    for preload in base['preloads_N']:
                        name=result_name(design['name'],material['subtype'],preload,rigid,10,settings['path_step_um'],sample,True)
                        args=['--config',str(config_path),'--sample-index',str(sample),'--design',design['name'],
                              '--material',material['subtype'],'--preload',str(preload)]
                        yield dict(key=name,args=args,root=settings['output_dir'],sample=sample,design=design['name'],
                                   material=material['subtype'],preload=preload)


def worker(job):
    path=Path(job['root'])/'results'/job['key']
    result=dict(read_result(path),trace_file=job['key']) if path.exists() else run_one(job['args'])
    return dict(summarize(result),sample_index=job['sample'])


def export_csv(connection,path):
    fields=['design','material','preload_N','sample_index','rigid','model','status','solve_s',
            'placement_attempt_count','selected_placement_index','selected_start_xy_m','placement_search_status',
            'coverage_mm','resolved_distance_mm','mean_T_resolved_N','p90_T_resolved_N','recoveries',
            'unresolved_rearrangements','max_force_balance_error_N','max_penetration_m','trace_file','error']
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for (value,) in connection.execute('SELECT summary_json FROM cases ORDER BY case_key'):
            writer.writerow(json.loads(value))
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('experiments/ijms_coarse_151200.json'))
    parser.add_argument('--workers',type=int)
    parser.add_argument('--wave-hours',type=float,default=8)
    parser.add_argument('--start-next-wave',action='store_true',help='Use only after the user requests the next wave')
    args=parser.parse_args();config_path=args.config.resolve()
    settings,base,design_map=campaign_inputs(str(config_path));designs=ordered_designs(design_map)
    root=Path(settings['output_dir'])
    for name in ['campaigns','results','logs','surfaces','tmp']:(root/name).mkdir(parents=True,exist_ok=True)
    import msvcrt
    runner_lock=(root/'logs'/'runner.lock').open('a+b')
    runner_lock.seek(0,2)
    if runner_lock.tell()==0:runner_lock.write(b'0');runner_lock.flush()
    runner_lock.seek(0)
    msvcrt.locking(runner_lock.fileno(),msvcrt.LK_NBLCK,1)
    os.environ['TEMP']=os.environ['TMP']=str(root/'tmp')
    frozen=dict(settings=settings,base=base,model=MODEL_VERSION)
    frozen_path=root/'campaigns'/'scan.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding='utf-8'))!=frozen:
        raise ValueError('Existing production directory belongs to a different configuration')
    write_json(frozen_path,frozen)
    db=sqlite3.connect(root/'results'/'case_index.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS cases(case_key TEXT PRIMARY KEY,status TEXT NOT NULL,summary_json TEXT NOT NULL)')
    done={key for (key,) in db.execute('SELECT case_key FROM cases')}
    counts=Counter(dict(db.execute('SELECT status,count(*) FROM cases GROUP BY status')))
    started=time.perf_counter();initial_count=len(done);workers=args.workers or settings['workers']
    planned=len(designs)*len(base['materials'])*len(base['preloads_N'])*len(settings['sample_indices'])
    if len(done)>=planned:
        export_csv(db,root/'results'/'summary.csv');db.close();print('All cases are already recorded.');return
    wave=begin_wave(db,args.wave_hours,workers,len(done),start_next=args.start_next_wave)
    state=dict(model=MODEL_VERSION,phase='all',planned_cases=planned,completed_cases=len(done),workers=workers,
               status='PREPARING',counts=dict(counts),elapsed_s=0,process_id=os.getpid(),
               prepared_surfaces=0,total_surfaces=len(base['materials'])*len(settings['sample_indices']),
               wave_number=wave['wave_id'],wave_started_at=wave['started_at'],wave_deadline_at=wave['deadline_at'])
    def progress():
        elapsed=time.perf_counter()-started;finished=len(done)-initial_count
        rate=finished/elapsed if finished>=32 and elapsed>=60 else None
        state.update(completed_cases=len(done),counts=dict(counts),elapsed_s=elapsed,
                     wave_elapsed_s=max(0,time.time()-wave['started_at']),
                     wave_remaining_s=max(0,wave['deadline_at']-time.time()),
                     wave_completed_cases=len(done)-wave['initial_completed'],
                     cases_per_hour=rate*3600 if rate else None,
                     eta_seconds=(planned-len(done))/rate if rate else None)
        write_json(root/'batch_status.json',state)
    progress()
    # All required inputs are self-contained in this production directory.
    for sample in settings['sample_indices']:
        for mi,material in enumerate(base['materials']):
            surface=prepare_surface(root,base,Shard(sample,mi,0,True,()),terrain_backend='cuda')
            for radius in base['tip_radii_m']:
                prepare_envelope(surface,radius,root,settings['surface_envelope_grid_um'],settings['surface_smoothing_sigma_um'])
            state['prepared_surfaces']+=1;progress()
    cached_envelope.cache_clear()
    jobs=(job for job in job_table(config_path,settings,base,designs) if job['key'] not in done)
    state['status']='RUNNING';progress()
    stop_reason=None
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={}
            def check_stop():
                nonlocal stop_reason
                if stop_reason:return
                free_disk=shutil.disk_usage(root).free;free_memory=available_memory()
                state.update(disk_free_bytes=free_disk,memory_available_bytes=free_memory)
                control=root/'control.json'
                if time.time()>=wave['deadline_at']:stop_reason='TIME_LIMIT'
                elif free_disk<20*2**30:stop_reason='LOW_DISK'
                elif free_memory<2*2**30:stop_reason='LOW_MEMORY'
                elif control.exists() and json.loads(control.read_text(encoding='utf-8')).get('stop_after_current'):
                    stop_reason='USER_REQUEST'
                if stop_reason:state.update(status='DRAINING',pause_reason=stop_reason)
            def fill():
                check_stop()
                while stop_reason is None and len(pending)<workers*2:
                    job=next(jobs,None)
                    if job is None:break
                    pending[pool.submit(worker,job)]=job
            fill()
            while pending:
                check_stop()
                ready,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for future in ready:
                    job=pending.pop(future)
                    try:row=future.result()
                    except Exception as exc:
                        row=dict(design=job['design'],material=job['material'],preload_N=job['preload'],
                                 sample_index=job['sample'],status='EXECUTION_ERROR',trace_file=job['key'],
                                 error=f'{type(exc).__name__}: {exc}')
                    db.execute('INSERT INTO cases VALUES(?,?,?)',(job['key'],row['status'],json.dumps(row,ensure_ascii=False,allow_nan=False)))
                    db.commit();done.add(job['key']);counts[row['status']]+=1
                    if len(done)==100 or len(done)%5000==0:export_csv(db,root/'results'/'summary.csv')
                progress();fill()
        state['status']='COMPLETED' if len(done)==planned else 'PAUSED'
        state['pause_reason']=stop_reason
        end_wave(db,wave,state['status'],stop_reason,len(done))
        progress();export_csv(db,root/'results'/'summary.csv')
    except BaseException:
        state['status']='INTERRUPTED';end_wave(db,wave,'INTERRUPTED','PROCESS_INTERRUPTED',len(done));progress();raise
    finally:db.close()
    print(json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
