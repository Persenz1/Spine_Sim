"""Offline Windows launcher for the frozen IJMS queue; no terrain generation."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse,ctypes,json,multiprocessing,sqlite3,subprocess,sys,time,threading,webbrowser
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from run_ijms_production import available_memory,ordered_designs,job_table,worker,export_csv
from run_ijms_balanced_contact import campaign_inputs
from run_ijms_balanced_batch import write_json
from spine_sim.balanced_contact import MODEL_VERSION
import shutil

PROGRAM=Path(__file__).resolve().parents[1]
PACKAGE=PROGRAM.parent
DATA=PACKAGE/'data'/'production_151200'
PAGE=PROGRAM/'monitor'/'dist'/'portable.html'
STATUS=PACKAGE/'data'/'server_status.json'
CONTROL=PACKAGE/'data'/'server_control.json'

def runner_active():
    import msvcrt
    path=PACKAGE/'data'/'server_runner.lock'
    if not path.exists():return False
    with path.open('r+b') as lock:
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:return True
        msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    return False

def hardware():
    kernel=ctypes.windll.kernel32
    groups=[kernel.GetActiveProcessorCount(i) for i in range(kernel.GetActiveProcessorGroupCount())]
    cpus=sum(groups)
    memory=available_memory()
    # Two envelope caches plus solver working arrays per process; leave OS headroom.
    workers=max(1,min(cpus,int((memory-3*2**30)//(0.75*2**30))))
    return dict(cpus=cpus,groups=groups,available_memory=memory,
                disk_free=shutil.disk_usage(DATA).free,auto_workers=workers)

def initialize_worker(counter,groups):
    # Explicit processor groups also cover older Windows Server systems with >64 CPUs.
    with counter.get_lock():
        index=counter.value;counter.value+=1
    slot=index%sum(groups);group=0
    while slot>=groups[group]:slot-=groups[group];group+=1
    class Affinity(ctypes.Structure):
        _fields_=[('mask',ctypes.c_size_t),('group',ctypes.c_ushort),('reserved',ctypes.c_ushort*3)]
    affinity=Affinity((1<<groups[group])-1,group)
    kernel=ctypes.windll.kernel32
    kernel.GetCurrentThread.restype=ctypes.c_void_p
    kernel.SetThreadGroupAffinity.argtypes=[ctypes.c_void_p,ctypes.POINTER(Affinity),ctypes.c_void_p]
    if not kernel.SetThreadGroupAffinity(kernel.GetCurrentThread(),ctypes.byref(affinity),None):
        raise ctypes.WinError()

def inputs(output):
    frozen=json.loads((DATA/'campaigns'/'scan.json').read_text(encoding='utf-8'))
    if frozen['model']!=MODEL_VERSION:raise ValueError('Model version differs from frozen campaign')
    base_path=PROGRAM/'experiments'/'ijms_small_array_ritz.json'
    base=json.loads(base_path.read_text(encoding='utf-8'))
    base['design_mode']=frozen['settings']['design_mode']
    if base!=frozen['base']:raise ValueError('Base configuration differs from frozen campaign')
    settings=dict(frozen['settings'],base_config=str(base_path),output_dir=str(output),
                  surface_root=str(DATA),envelope_root=str(DATA))
    # Fail before dispatch if any packaged input is missing; never synthesize on CPU/GPU.
    for sample in settings['sample_indices']:
        for material in base['materials']:
            stem=f"{material['material']}-{material['subtype']}-s{sample:03d}"
            names=[stem+'.npy',stem+'.json']
            names += [f"{stem}-r{radius*1e6:.0f}-envelope{settings['surface_envelope_grid_um']:g}um-sigma{settings['surface_smoothing_sigma_um']:g}um.npy" for radius in base['tip_radii_m']]
            for name in names:
                if not (DATA/'surfaces'/name).is_file():raise FileNotFoundError('Packaged terrain missing: '+name)
    output.mkdir(parents=True,exist_ok=True)
    config=output/'campaigns'/'server_config.json';config.parent.mkdir(parents=True,exist_ok=True)
    write_json(config,settings)
    return config

def ledger_counts(root=DATA):
    path=root/'results'/'case_index.sqlite3'
    if not path.exists():return {}
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=2) as db:
        return dict(db.execute('SELECT status,count(*) FROM cases GROUP BY status'))

def run(count,requested_workers,test=False):
    import msvcrt
    lock_path=PACKAGE/'data'/'server_runner.lock'
    with lock_path.open('a+b') as lock:
        lock.seek(0,2)
        if not lock.tell():lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        output=PACKAGE/'data'/'tests'/time.strftime('%Y%m%d-%H%M%S') if test else DATA
        config=inputs(output)
        settings,base,design_map=campaign_inputs(str(config))
        designs=ordered_designs(design_map)
        for name in ('results','logs','tmp'):(output/name).mkdir(parents=True,exist_ok=True)
        os.environ['TEMP']=os.environ['TMP']=str(output/'tmp')
        db=sqlite3.connect(output/'results'/'case_index.sqlite3')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS cases(case_key TEXT PRIMARY KEY,status TEXT NOT NULL,summary_json TEXT NOT NULL)')
        done={key for key, in db.execute('SELECT case_key FROM cases')}
        counts=Counter(dict(db.execute('SELECT status,count(*) FROM cases GROUP BY status')))
        total=len(designs)*len(base['materials'])*len(base['preloads_N'])*len(settings['sample_indices'])
        initial=len(done);target=min(total,initial+count) if count else total
        hw=hardware();workers=min(requested_workers or hw['auto_workers'],hw['cpus'],max(1,target-initial))
        workers=min(workers,hw['auto_workers'])
        state=dict(status='RUNNING',mode='test' if test else 'production',process_id=os.getpid(),
                   planned_cases=total,target_cases=target,initial_cases=initial,workers=workers,
                   logical_cpus=hw['cpus'],output_dir=str(output),started_at=time.time())
        start=time.monotonic();stop=None;submitted=0
        pending={}
        def progress():
            elapsed=time.monotonic()-start;n=len(done)-initial
            rate=n/elapsed if n and elapsed>=5 else None
            state.update(completed_cases=len(done),counts=dict(counts),session_done=n,
                         session_target=target-initial,elapsed_s=elapsed,in_flight=len(pending),
                         cases_per_hour=rate*3600 if rate else None,
                         eta_seconds=(target-len(done))/rate if rate else None,
                         disk_free_bytes=shutil.disk_usage(output).free,
                         memory_available_bytes=available_memory(),updated_at=time.time())
            write_json(STATUS,state)
        jobs=(j for j in job_table(config,settings,base,designs) if j['key'] not in done)
        ctx=multiprocessing.get_context('spawn');counter=ctx.Value('i',0)
        progress()
        try:
            with ExitStack() as stack:
                # Windows ProcessPoolExecutor has a 61-worker limit per pool, not per program.
                sizes=[min(61,workers-i) for i in range(0,workers,61)]
                pools=[stack.enter_context(ProcessPoolExecutor(max_workers=n,mp_context=ctx,
                        initializer=initialize_worker,initargs=(counter,hw['groups']))) for n in sizes]
                pool_jobs=[0]*len(pools)
                while True:
                    if stop is None:
                        if CONTROL.exists() and json.loads(CONTROL.read_text()).get('stop_after_current'):stop='USER_REQUEST'
                        elif available_memory()<2*2**30:stop='LOW_MEMORY'
                        elif shutil.disk_usage(output).free<20*2**30:stop='LOW_DISK'
                    if stop:state.update(status='DRAINING',pause_reason=stop)
                    while stop is None and len(pending)<workers*2 and submitted<target-initial:
                        job=next(jobs,None)
                        if job is None:break
                        pi=min(range(len(pools)),key=lambda i:pool_jobs[i]/sizes[i])
                        future=pools[pi].submit(worker,job);pending[future]=(job,pi)
                        pool_jobs[pi]+=1;submitted+=1
                    if not pending:break
                    ready,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                    for future in ready:
                        job,pi=pending.pop(future);pool_jobs[pi]-=1
                        try:row=future.result()
                        except BrokenProcessPool:
                            stop='WORKER_CRASH'
                            continue
                        except Exception as exc:
                            row=dict(design=job['design'],material=job['material'],preload_N=job['preload'],
                                     sample_index=job['sample'],trace_file=job['key'],status='EXECUTION_ERROR',
                                     error=f'{type(exc).__name__}: {exc}')
                        db.execute('INSERT INTO cases VALUES(?,?,?)',(job['key'],row['status'],json.dumps(row,ensure_ascii=False,allow_nan=False)))
                        db.commit();done.add(job['key']);counts[row['status']]+=1
                    progress()
            state.update(status='PAUSED' if stop else 'COMPLETED',pause_reason=stop)
            progress();export_csv(db,output/'results'/'summary.csv')
        except BaseException as exc:
            state.update(status='INTERRUPTED',error=f'{type(exc).__name__}: {exc}')
            progress();raise
        finally:db.close()

def serve(port,test_default=False,open_browser=True):
    child=None;handle=None
    def snapshot():
        state=json.loads(STATUS.read_text(encoding='utf-8')) if STATUS.exists() else {}
        active=(child is not None and child.poll() is None) or runner_active()
        if state.get('status') in ('RUNNING','DRAINING','STARTING') and not active:
            state.update(status='INTERRUPTED',error='Runner exited; see data/launcher.stderr.log')
        return dict(state=state,running=active,hardware=hardware(),counts=ledger_counts(),
                    package=str(PACKAGE),test_default=test_default)
    start_lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def reply(self,value,status=200,html=False):
            data=value if html else json.dumps(value,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            try:
                if self.path=='/':self.reply(PAGE.read_bytes(),html=True)
                elif self.path=='/api/status':self.reply(snapshot())
                else:self.send_error(404)
            except Exception as exc:self.reply(dict(error=str(exc)),500)
        def do_POST(self):
            nonlocal child,handle
            # Mutations are local same-origin requests only.
            if self.headers.get('Origin')!=f'http://127.0.0.1:{port}':self.send_error(403);return
            try:
                request=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
                with start_lock:
                    if self.path=='/api/stop':
                        write_json(CONTROL,dict(stop_after_current=True));self.reply(dict(ok=True));return
                    if self.path!='/api/start':self.send_error(404);return
                    if (child is not None and child.poll() is None) or runner_active():raise ValueError('已有任务运行，请先等待自然收尾')
                    count=int(request.get('count',0));workers=int(request.get('workers',0));test=bool(request.get('test'))
                    if count<0 or count>151200 or workers<0:raise ValueError('case数须在0至151200之间，worker数不能为负')
                    if test and not 1<=count<=32:raise ValueError('测试请选择1至32例')
                    if CONTROL.exists():CONTROL.unlink()
                    if handle:handle.close()
                    handle=(PACKAGE/'data'/'launcher.stderr.log').open('a',encoding='utf-8')
                    write_json(STATUS,dict(status='STARTING'))
                    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--run','--count',str(count),'--workers',str(workers)]
                    if test:command.append('--test')
                    child=subprocess.Popen(command,cwd=PROGRAM,stdout=handle,stderr=handle,creationflags=subprocess.CREATE_NO_WINDOW)
                    self.reply(dict(ok=True))
            except Exception as exc:self.reply(dict(error=str(exc)),400)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    url=f'http://127.0.0.1:{port}'
    print('IJMS: '+url+'  (keep this window open)',flush=True)
    if open_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:
        if child and child.poll() is None:
            write_json(CONTROL,dict(stop_after_current=True));print('Waiting for dispatched cases...',flush=True);child.wait()
    finally:server.server_close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--test',action='store_true')
    parser.add_argument('--count',type=int,default=0);parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--port',type=int,default=8766);parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    if args.run:run(args.count,args.workers,args.test)
    else:serve(args.port,args.test,not args.no_browser)
