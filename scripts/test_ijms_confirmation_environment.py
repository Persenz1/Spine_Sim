"""Independent dependency/replay test; never starts the confirmation queue."""
import os,sys,json,time,argparse,gc
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUPY_CACHE_IN_MEMORY']='1'
PROGRAM=Path(__file__).resolve().parents[1];sys.path[:0]=[str(PROGRAM/'src'),str(PROGRAM/'scripts')]
import numpy as np
from ijms_confirmation import jobs,inputs,Recorder
from ijms_confirmation_surface import generate,environment,prepare
from ijms_stream_surface import attach_surface
from spine_sim.balanced_contact import CondensedArray
from spine_sim.ijms import build_rods
from ijms_compact import pack_trace,unpack_trace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--compare-full-path',action='store_true');args=parser.parse_args()
    print(json.dumps(environment()),flush=True)
    if not args.compare_full_path:
        selected={}
        for job in jobs():
            selected.setdefault(job['material'],job)
            if len(selected)==5:break
        for job in selected.values():
            a,meta=generate(job,(.004,.004));b,other=generate(job,(.004,.004))
            assert meta['raw_sha256']==other['raw_sha256'];np.testing.assert_array_equal(a,b)
            print(json.dumps(dict(material=job['material'],regeneration_identical=True)),flush=True)
        original=dict(rows=[dict(gaps_m=[-0.,1e-7],slip_flags=[False,True])])
        assert unpack_trace(pack_trace(original,'probe'),verify=True)==original
        print('Environment and five-material regeneration checks passed; no production cases started.',flush=True)
        return
    job=next(j for j in jobs() if j['case_id']=='F7_P1');owner,meta=prepare(job)
    surface,handles=attach_surface(owner.descriptor);config=inputs()[3][job['design']]['array']
    results=[];timings=[];recorder=Recorder(surface,True)
    for observer in (None,lambda event,state,row:recorder(event,state,row,0)):
        array=CondensedArray(build_rods(config),surface,rigid=True);tick=time.perf_counter()
        result=array.run(np.zeros(2),1.,distance=.01,cpu_budget_s=600,state_observer=observer)
        timings.append(time.perf_counter()-tick);results.append(result)
    old,new=results
    assert old['status']==new['status'];assert len(old['rows'])==len(new['rows'])
    for a,b in zip(old['rows'],new['rows']):assert all(b[k]==v for k,v in a.items())
    assert old['checkpoint']==new['checkpoint']
    packed=pack_trace(new,'F7_P1');assert unpack_trace(packed,verify=True)==new
    print(json.dumps(dict(case='F7_P1',material=job['material'],full_shape=meta['envelope_shape'],
        generation_s=meta['prepare_s'],status=new['status'],rows=len(new['rows']),
        last_x_mm=new['rows'][-1]['x_m']*1000 if new['rows'] else None,baseline_s=timings[0],recorded_s=timings[1],
        existing_fields_and_checkpoint_identical=True,compressed_bytes=len(packed))),flush=True)
    del recorder,array,surface;gc.collect()
    for handle in handles:handle.close()
    owner.close()


if __name__=='__main__':main()
