from types import SimpleNamespace
import numpy as np
from spine_sim.placement_retry import run_with_placements


class ArrayStub:
    guides=np.zeros((1,3));length=np.array([.004]);axes=np.array([[0.,0.,-1.]])
    radius=np.array([.00005]);surface=SimpleNamespace(x=np.array([-.1,.1]),y=np.array([-.1,.1]))
    def __init__(self,results):self.results=iter(results);self.starts=[]
    def run(self,start,*args,**kwargs):
        self.starts.append(start.copy());return next(self.results)


def run(array,offsets=((0,0),(.001,0),(0,.001))):
    return run_with_placements(array,[0,0],1,.01,25e-6,offsets_m=offsets,
                               attempt_cpu_budget_s=45,total_cpu_budget_s=120)


def test_preload_retry_retains_failure_and_only_selected_trace():
    failed=dict(status='BALANCED_NUMERICAL_FAILURE',phase='preload',rows=[{'old':True}])
    success=dict(status='BALANCED_COMPLETE',rows=[{'new':True}])
    a=ArrayStub([failed,success]);result=run(a)
    assert len(a.starts)==2
    assert a.starts[1].tolist()==[.001,0.]
    assert result['rows']==success['rows']
    assert result['placement_attempts'][0]['failed_result']==failed
    assert result['selected_placement_index']==1
    assert result['selected_start_xy_m']==[.001,0.]


def test_drag_failure_does_not_change_placement():
    failed=dict(status='BALANCED_NUMERICAL_FAILURE',phase='drag',rows=[{'x_m':.003}])
    a=ArrayStub([failed]);result=run(a)
    assert len(a.starts)==1
    assert result['status']==failed['status']
    assert result['selected_placement_index']==0


def test_exhausted_preload_attempts_are_not_a_success():
    failed=dict(status='BALANCED_BUDGET_LIMIT',phase='preload',rows=[])
    a=ArrayStub([failed,failed,failed]);result=run(a)
    assert len(a.starts)==3
    assert result['placement_search_status']=='EXHAUSTED'
    assert result['selected_placement_index'] is None
    assert result['status']=='BALANCED_BUDGET_LIMIT'


def test_candidate_requires_the_entire_nominal_drag_path_in_domain():
    a=ArrayStub([dict(status='BALANCED_COMPLETE',rows=[])])
    result=run(a,offsets=((.095,0),(0,0)))
    assert len(a.starts)==1
    assert result['placement_attempts'][0]['status']=='PLACEMENT_OUTSIDE_DOMAIN'
    assert result['selected_placement_index']==1
