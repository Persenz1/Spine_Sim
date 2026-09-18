"""Manual final-confirmation launcher: baseline solver, CUDA RAM pipeline, results only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUPY_CACHE_IN_MEMORY']='1'
import argparse,gc,json,math,multiprocessing,shutil,sys,time
from collections import Counter,deque
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait,FIRST_COMPLETED
from itertools import groupby
from pathlib import Path
PROGRAM=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROGRAM/'src'),str(PROGRAM/'scripts')]
from ijms_confirmation import BATCH,PLANNED_CASES,SURFACE,MATERIALS,inputs,jobs,solve,export_summary
from ijms_confirmation_surface import prepare,snapshot
from ijms_stream_surface import attach_surface
from ijms_stream_storage import BatchWriter
from ijms_stream_hardware import hardware,CpuUsage
from run_ijms_balanced_batch import write_json

LIVE=None


def initialize(progress):
    global LIVE
    LIVE=progress


def solve_shared(job,descriptor):
    surface,handles=attach_surface(descriptor)
    def report(row):
        slot=job['slot']*5
        LIVE[slot]=1 if row['phase']=='preload' else 2;LIVE[slot+1]=row['x_m']*1000
        LIVE[slot+2]=time.monotonic();LIVE[slot+3]=row['P_N'];LIVE[slot+4]=row.get('placement_index',0)
    try:return solve(job,surface,report)
    finally:
        surface=None;gc.collect()
        for handle in handles:handle.close()


def run(args):
    import msvcrt
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'runner.lock').open('a+b') as lock:
        lock.seek(0,2)
        if not lock.tell():lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        run_queue(args)


def run_queue(args):
    results=args.output/'results';configs=results/'campaigns';configs.mkdir(parents=True,exist_ok=True)
    settings,base,records,designs=inputs()
    frozen=dict(batch=BATCH,settings=settings,base=base,records=records,designs=designs,surface=SURFACE,
                solver='baseline-c0eaf52-dense',recording_version=1,test=args.test,
                distance_mm=args.test_distance_mm if args.test else 10.)
    config=configs/'scan.json'
    if config.exists() and json.loads(config.read_text('utf-8'))!=frozen:raise ValueError('Use a separate output for another configuration')
    write_json(config,frozen);snapshot(results)
    metas=configs/'surfaces';metas.mkdir(exist_ok=True)
    writer=BatchWriter(results)
    done={r[0] for r in writer.db.execute('SELECT case_key FROM cases')};initial=len(done)
    selected=[j for j in jobs() if j['key'] not in done and (not args.test_case or j['case_id']==args.test_case)
              and (not args.test_material or j['material']==args.test_material)]
    if args.count:selected=selected[:args.count]
    if args.test:
        for job in selected:job['test_distance_mm']=args.test_distance_mm
    groups=iter([list(group) for _,group in groupby(selected,key=lambda j:j['surface_id'])])
    hw=hardware();workers=max(1,min(args.workers or hw['auto_workers'],hw['auto_workers']))
    # Multiple executors cover dual-socket Windows hosts above the 61-process limit.
    npools=math.ceil(workers/61);sizes=[workers//npools+(i<workers%npools) for i in range(npools)]
    ctx=multiprocessing.get_context('spawn');live=ctx.RawArray('d',workers*5)
    pools=[ProcessPoolExecutor(max_workers=n,mp_context=ctx,initializer=initialize,initargs=(live,)) for n in sizes]
    producer=ThreadPoolExecutor(max_workers=1)
    # A few slow cases must not pin all terrain slots and starve idle workers.
    # At most one field per in-flight case plus two ready fields; ready jobs
    # provide a tighter admission bound during normal throughput.
    window=workers+2
    owners={};ready=deque();pending={};prepared=None;preparing=None;exhausted=False;stop=None
    counts=Counter(dict(writer.db.execute('SELECT status,count(*) FROM cases GROUP BY status')))
    started=time.monotonic();cpu=CpuUsage();cpu.sample();last=started;generation_wait=0.
    state=dict(status='RUNNING',mode='test' if args.test else 'production',planned_cases=PLANNED_CASES,
        workers=workers,max_workers=workers,adaptive_workers=False,session_target=len(selected),output_dir=str(args.output),
        prepared_surfaces=0,generation_s=0.,peak_resident_surfaces=0,peak_in_flight=0,surface_window=window)
    def progress():
        nonlocal last,generation_wait
        now=time.monotonic()
        if prepared is not None and not pending and not ready:generation_wait+=now-last
        last=now;writer.flush_due();persisted=writer.db.execute('SELECT count(*) FROM cases').fetchone()[0]
        n=persisted-initial;elapsed=now-started
        durable_counts=counts-Counter(entry[1]['status'] for entry in writer.entries)
        state.update(completed_cases=persisted,session_done=n,elapsed_s=elapsed,counts=dict(durable_counts),in_flight=len(pending),
            buffered_cases=len(writer.entries),resident_surfaces=len(owners),generation_active=prepared is not None,
            generation_wait_s=generation_wait,phase='solving' if pending else 'preparing_surface',
            cpu_utilization_percent=cpu.sample(),cases_per_hour=n/elapsed*3600 if n else None,
            eta_seconds=(len(selected)-n)*elapsed/n if n else None,
            peak_resident_surfaces=max(state['peak_resident_surfaces'],len(owners)),
            peak_in_flight=max(state['peak_in_flight'],len(pending)),updated_at=time.time())
        state['in_flight_progress']=[dict(design=j['case_id'],surface_id=j['surface_id'],preload_N=j['preload'],
            phase={0:'initializing',1:'preload',2:'drag'}[int(live[j['slot']*5])],x_mm=live[j['slot']*5+1],
            seconds_since_progress=now-live[j['slot']*5+2]) for j in pending.values()]
        write_json(args.output/'status.json',state)
    try:
        progress()
        while pending or ready or prepared is not None or not exhausted:
            if (args.output/'stop.json').exists():stop='USER_REQUEST'
            elif hardware()['available_memory']<hw['reserve_bytes']:stop='LOW_MEMORY'
            elif shutil.disk_usage(args.output).free<2*2**30:stop='LOW_DISK'
            if stop:
                state.update(status='DRAINING',pause_reason=stop);ready.clear();exhausted=True
            if prepared is not None and prepared.done():
                owner,metadata=prepared.result();prepared=None;surface_id=preparing[0]['surface_id']
                owners[surface_id]=dict(owner=owner,remaining=len(preparing))
                state['peak_resident_surfaces']=max(state['peak_resident_surfaces'],len(owners))
                path=metas/(surface_id+'.json')
                if path.exists():
                    old=json.loads(path.read_text('utf-8'))
                    if any(old[k]!=metadata[k] for k in ('raw_sha256','envelope_sha256')):
                        raise ValueError('Regenerated surface differs: '+surface_id)
                else:write_json(path,metadata)
                state['prepared_surfaces']+=1;state['generation_s']+=metadata['prepare_s']
                if not stop:ready.extend(preparing)
                preparing=None
            if not stop and prepared is None and not exhausted and len(owners)<window and len(ready)<max(workers,18):
                preparing=next(groups,None)
                if preparing is None:exhausted=True
                else:prepared=producer.submit(prepare,preparing[0])
            while not stop and ready and len(pending)<workers:
                job=ready.popleft();occupied={j['slot'] for j in pending.values()}
                slot=next(i for i in range(workers) if i not in occupied);job['slot']=slot
                for k in range(5):live[slot*5+k]=0
                live[slot*5+2]=time.monotonic()
                future=pools[slot%npools].submit(solve_shared,job,owners[job['surface_id']]['owner'].descriptor)
                pending[future]=job
                state['peak_in_flight']=max(state['peak_in_flight'],len(pending))
            if pending:
                finished,_=wait(pending,timeout=.25,return_when=FIRST_COMPLETED)
                for future in finished:
                    job=pending.pop(future);row,trace=future.result()
                    writer.add(job['key'],row,trace);counts[row['status']]+=1
                    entry=owners[job['surface_id']];entry['remaining']-=1
                    if not entry['remaining']:
                        entry['owner'].close();del owners[job['surface_id']]
            elif prepared is not None:
                time.sleep(.1)
            if time.monotonic()-last>=1:progress()
        writer.flush();state.update(status='PAUSED' if stop else 'COMPLETED',pause_reason=stop);progress()
        print(json.dumps(state,ensure_ascii=False),flush=True)
    except BaseException as exc:
        writer.flush();state.update(status='INTERRUPTED',error=f'{type(exc).__name__}: {exc}');progress();raise
    finally:
        producer.shutdown(wait=True)
        for pool in pools:pool.shutdown(wait=True)
        if prepared is not None:
            try:prepared.result()[0].close()
            except Exception:pass
        for item in owners.values():item['owner'].close()
        export_summary(writer.db,results/'summary.csv');writer.close()


def serve(args):
    import subprocess,threading,webbrowser
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    child=None;handle=None;active=args.output;mutex=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def reply(self,value,status=200,html=False):
            data=value if html else json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            try:
                if self.path=='/':return self.reply((PROGRAM/'monitor/dist/confirmation.html').read_bytes(),html=True)
                if self.path!='/api/status':return self.send_error(404)
                path=active/'status.json';state=json.loads(path.read_text('utf-8')) if path.exists() else {}
                running=child is not None and child.poll() is None
                if not running and state.get('status') in ('RUNNING','DRAINING','STARTING'):state.update(status='INTERRUPTED',error='Runner exited; inspect launcher.log')
                hw=hardware();hw['disk_free']=shutil.disk_usage(args.output).free
                self.reply(dict(state=state,running=running,hardware=hw,counts=state.get('counts',{}),
                                planned_cases=PLANNED_CASES,package=str(active),test_default=True))
            except Exception as exc:self.reply(dict(error=str(exc)),500)
        def do_POST(self):
            nonlocal child,handle,active
            if self.headers.get('Origin')!=f'http://127.0.0.1:{args.port}':return self.send_error(403)
            try:
                value=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
                with mutex:
                    if self.path=='/api/stop':
                        write_json(active/'stop.json',dict(stop_after_current=True));return self.reply(dict(ok=True))
                    if self.path!='/api/start':return self.send_error(404)
                    if child is not None and child.poll() is None:raise ValueError('已有任务运行')
                    count=int(value['count']);workers=int(value['workers']);test=bool(value['test'])
                    if not 0<=count<=PLANNED_CASES or workers<0:raise ValueError('数量或并发无效')
                    if test and not 1<=count<=90:raise ValueError('独立测试请选择1～90例')
                    active=args.output/'tests'/time.strftime('%Y%m%d-%H%M%S') if test else args.output
                    active.mkdir(parents=True,exist_ok=True);(active/'stop.json').unlink(missing_ok=True)
                    if handle:handle.close()
                    handle=(active/'launcher.log').open('a',encoding='utf-8')
                    command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--run','--output',str(active),
                             '--count',str(count),'--workers',str(workers)]
                    if test:command+=['--test','--test-distance-mm','0.1']
                    write_json(active/'status.json',dict(status='STARTING'))
                    child=subprocess.Popen(command,cwd=PROGRAM,stdout=handle,stderr=handle,creationflags=subprocess.CREATE_NO_WINDOW)
                    self.reply(dict(ok=True))
            except Exception as exc:self.reply(dict(error=str(exc)),400)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    url=f'http://127.0.0.1:{args.port}';print(url,flush=True)
    if not args.no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:
        if child and child.poll() is None:write_json(active/'stop.json',dict(stop_after_current=True));child.wait()
    finally:
        server.server_close()
        if handle:handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default=PROGRAM.parent/'data/final_confirmation' if PROGRAM.name=='program' else Path('E:/TestData/IJMS/final_confirmation')
    parser.add_argument('--output',type=Path,default=default);parser.add_argument('--run',action='store_true')
    parser.add_argument('--test',action='store_true');parser.add_argument('--test-case',choices=[r['case_id'] for r in inputs()[2]])
    parser.add_argument('--test-material',choices=[m[1] for m in MATERIALS])
    parser.add_argument('--test-distance-mm',type=float,default=10.)
    parser.add_argument('--count',type=int,default=0);parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--port',type=int,default=8769);parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    if not 0<=args.count<=PLANNED_CASES or args.workers<0:parser.error('Invalid count/workers')
    if args.test and args.run and not 1<=args.count<=90:parser.error('Independent tests require count 1..90')
    if not 0<args.test_distance_mm<=10:parser.error('Distance must be in (0,10] mm')
    if not args.test and (args.test_case or args.test_material or args.test_distance_mm!=10):parser.error('Test selectors require --test')
    if args.test and args.output==default:args.output=default/'tests'/time.strftime('%Y%m%d-%H%M%S')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if args.run:run(args)
    else:serve(args)


if __name__=='__main__':main()
