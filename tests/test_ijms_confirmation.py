import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ijms_confirmation import inputs,jobs,Recorder,SURFACE
from ijms_compact import pack_trace,unpack_trace
from ijms_stream_storage import BatchWriter
from ijms_stream_surface import SharedSurface,attach_surface
from spine_sim.balanced_contact import CondensedArray,EnvelopeSurface
from spine_sim.ijms import build_rods


def test_queue_and_fixed_detailed_subset():
    settings,base,records,designs=inputs();queue=list(jobs())
    assert len(designs)==11 and len(records)==18
    assert len(queue)==len({j['key'] for j in queue})==90000
    assert len({j['surface_id'] for j in queue})==5000
    assert len({j['seed'] for j in queue})==5000
    assert min(j['seed'] for j in queue)>2026091319
    assert sum(j['detailed'] for j in queue)==1800
    for layout in ('5x2','2x5'):
        d=designs[f'r100_a80_spring800_{layout}_p5']['array']
        assert d['spine']['spring_stiffness_N_per_m']==800 and d['yaw_deg']==0


def test_recording_and_shared_surface_preserve_accepted_history():
    _,_,_,designs=inputs();config=designs['r100_a80_spring800_5x2_p5']['array']
    x=np.arange(137)*.0005-.029;y=np.arange(117)*.0005-.029
    height=2e-5*np.sin(x[None,:]*100)+np.zeros((len(y),1))
    surface=EnvelopeSurface(height,.0005,.0005,(-.029,-.029));owner=SharedSurface(surface)
    shared,handles=attach_surface(owner.descriptor)
    points=np.array([[0.,0.],[.0123,-.01],[x[-1],y[-1]]])
    for old,new in zip(surface.query(points),shared.query(points)):np.testing.assert_array_equal(old,new)
    baseline=CondensedArray(build_rods(config),surface,rigid=True).run(np.zeros(2),1.,distance=.0001,cpu_budget_s=600)
    recorder=Recorder(shared,True)
    recorded=CondensedArray(build_rods(config),shared,rigid=True).run(np.zeros(2),1.,distance=.0001,cpu_budget_s=600,
        state_observer=lambda event,state,row:recorder(event,state,row,0))
    assert baseline['status']==recorded['status']=='BALANCED_COMPLETE'
    assert len(baseline['rows'])==len(recorded['rows'])
    for old,new in zip(baseline['rows'],recorded['rows']):
        assert all(new[k]==v for k,v in old.items())
        state=new['accepted_state'];normal=np.array(new['contact_normals']);force=np.array(new['forces_N'])
        assert normal.shape==force.shape==(10,3)
        np.testing.assert_allclose(np.linalg.norm(normal,axis=1),1,atol=1e-14)
        np.testing.assert_array_equal(np.sum(force*normal,axis=1),new['normal_forces_N'])
        np.testing.assert_array_equal(state['movement'],new['tangential_movement_m'])
        center=np.array(state['center']);height=np.array(new['surface_height_m'])
        np.testing.assert_array_equal((center[:,2]-config['spine']['tip_radius_m']-height)*normal[:,2],new['gaps_m'])
        np.testing.assert_allclose(np.sum(np.array(new['tangential_movement_m'])*normal,axis=1),0,atol=1e-18)
        a=CondensedArray(build_rods(config),surface,rigid=True)
        rebuilt=a.guides+np.array(state['position'])+(a.length-np.array(new['compressions_m']))[:,None]*a.axes
        np.testing.assert_array_equal(rebuilt,center)
        assert new['requested_preload_N']==state['requested_preload_N']
    recorded['detailed_events']=recorder.events
    assert unpack_trace(pack_trace(recorded,'test'),verify=True)==recorded
    assert recorder.events[0]['event']=='initial'
    del recorder,shared
    for h in handles:h.close()
    owner.close()


def test_compact_batch_recovers_after_index_loss(tmp_path):
    import sqlite3
    from ijms_compact import TraceReader
    result=dict(status='BALANCED_NUMERICAL_FAILURE',rows=[dict(slip_flags=[False,True],gaps_m=[-0.,1e-7],
        contact_normals=[[0.,0.,1.],[0.,0.,1.]],tangential_movement_m=[[1e-6,0.,0.],[0.,0.,0.]])])
    writer=BatchWriter(tmp_path);writer.add('case',dict(status=result['status']),pack_trace(result,'case'));writer.close()
    with sqlite3.connect(tmp_path/'compact/index.sqlite3') as db:
        for name in ('cases','traces','batches'):db.execute('DELETE FROM '+name)
    writer=BatchWriter(tmp_path);writer.close()
    with TraceReader(tmp_path) as reader:assert reader.read('case',verify=True)==result


def test_small_domain_worker_budget_uses_all_32_server_threads(monkeypatch):
    import ijms_stream_hardware as module
    monkeypatch.setattr(module,'memory_status',lambda:(128*2**30,110*2**30))
    monkeypatch.setattr(module,'physical_cores',lambda:16)
    monkeypatch.setattr(module.os,'cpu_count',lambda:32)
    assert module.hardware()['auto_workers']==32


def test_pipeline_prepares_next_terrain_while_previous_cases_are_slow(tmp_path,monkeypatch):
    import time
    from types import SimpleNamespace
    from concurrent.futures import ThreadPoolExecutor
    import run_ijms_confirmation as runner
    events=[]
    queue=list(jobs())[:54]
    def prepare(job):
        events.append(('prepare',job['surface_id']))
        return SimpleNamespace(descriptor={},close=lambda:None),dict(prepare_s=0.,raw_sha256='raw',envelope_sha256='envelope')
    def solve(job,descriptor):
        if job['case_id']=='F1_P1':time.sleep(.35)
        else:time.sleep(.002)
        events.append(('done',job['surface_id'],job['case_id']))
        return dict(status='BALANCED_COMPLETE'),pack_trace(dict(rows=[],status='BALANCED_COMPLETE'),job['key'])
    monkeypatch.setattr(runner,'jobs',lambda:iter(queue))
    monkeypatch.setattr(runner,'snapshot',lambda _:None)
    monkeypatch.setattr(runner,'prepare',prepare);monkeypatch.setattr(runner,'solve_shared',solve)
    monkeypatch.setattr(runner,'ProcessPoolExecutor',lambda max_workers,**kwargs:ThreadPoolExecutor(max_workers=max_workers))
    monkeypatch.setattr(runner,'hardware',lambda:dict(auto_workers=4,available_memory=110*2**30,reserve_bytes=8*2**30))
    runner.run_queue(SimpleNamespace(output=tmp_path,test=True,test_distance_mm=.1,test_case=None,test_material=None,count=54,workers=4))
    first=queue[0]['surface_id'];third=queue[36]['surface_id']
    assert events.index(('prepare',third))<events.index(('done',first,'F1_P1'))
    state=json.loads((tmp_path/'status.json').read_text('utf-8'))
    assert state['session_done']==54 and state['peak_resident_surfaces']>=3 and state['peak_in_flight']==4
