import numpy as np
import pytest
from spine_sim.balanced_contact import EnvelopeSurface,CondensedArray,compliance_table
from spine_sim.guided_rod import GuidedRod,GuidedRodParameters


def test_cached_surface_derivatives_match_plane():
    x=np.linspace(-.01,.01,41);y=x.copy()
    h=.2*x[None,:]-.1*y[:,None]
    surface=EnvelopeSurface(h,.0005,.0005,(-.01,-.01))
    z,n=surface.query(np.array([[.003,.004],[-.001,.002]]))
    assert z==pytest.approx([.0002,-.0004])
    assert n==pytest.approx(np.broadcast_to(np.array([-.2,.1,1])/np.sqrt(1.05),(2,3)))


def test_condensed_uniform_beam_force_and_moment_compliance():
    p=GuidedRodParameters(.004,.0002,200e9,100.,.003,5e-5)
    lengths,coeff=compliance_table(p)
    assert coeff[-1]==pytest.approx(np.array([p.free_length_m**3/3,p.free_length_m**2/2,p.free_length_m])/p.bending_rigidity_Nm2,rel=1e-12)


@pytest.mark.parametrize('rigid',[True,False])
def test_flat_drag_maintains_preload_and_coulomb_sliding(rigid):
    p=GuidedRodParameters(.004,.0002,200e9,100.,.003,5e-5)
    a=np.array([.5,0.,-np.sqrt(.75)])
    rods=[GuidedRod(p,np.array([0.,y,p.tip_radius_m])-p.free_length_m*a,a) for y in [-.001,.001]]
    surface=EnvelopeSurface(np.zeros((41,41)),.0005,.0005,(-.01,-.01))
    result=CondensedArray(rods,surface,rigid=rigid).run(np.zeros(2),.02,distance=.0005,step=25e-6)
    assert result['status']=='BALANCED_COMPLETE'
    assert result['rows'][-1]['T_N']==pytest.approx(.4*.02,abs=2e-5)
    assert all(abs(r['P_N']-.02)<2e-5 for r in result['rows'] if r['phase']=='drag')
    assert abs(result['rows'][-1]['Y_m'])<1e-8


@pytest.mark.parametrize('rigid',[True,False])
def test_energy_seed_refines_to_force_and_contact_equilibrium(rigid):
    p=GuidedRodParameters(.004,.0002,200e9,100.,.003,5e-5,
                          mount_type='spring' if rigid else 'fixed')
    a=np.array([.5,0.,-np.sqrt(.75)])
    rods=[GuidedRod(p,np.array([0.,y,p.tip_radius_m])-p.free_length_m*a,a) for y in [-.001,.001]]
    surface=EnvelopeSurface(np.zeros((41,41)),.0005,.0005,(-.01,-.01))
    array=CondensedArray(rods,surface,rigid=rigid);old=array.initial(np.zeros(2))
    force,position=array.energy_seed(0.,.02,old)
    state=array.solve(0.,.02,old,guess=np.r_[force.ravel()/.01,position[1:]/1e-5])
    assert state['accepted']
    assert state['force'][:,2].sum()==pytest.approx(.02,abs=2e-5)
    assert abs(state['force'][:,1].sum())<2e-5
    assert state['gap'].min()>-1e-8
    assert state['normal_force'].min()>-1e-5


def test_short_final_interval_is_bisected_after_solver_failure():
    p=GuidedRodParameters(.004,.0002,200e9,100.,.003,5e-5)
    a=np.array([.5,0.,-np.sqrt(.75)])
    rod=GuidedRod(p,np.array([0.,0.,p.tip_radius_m])-p.free_length_m*a,a)
    surface=EnvelopeSurface(np.zeros((41,41)),.0005,.0005,(-.01,-.01))
    array=CondensedArray([rod],surface,rigid=True);original=array.solve;attempts=[]
    def solve(x,preload,previous,**kwargs):
        state=original(x,preload,previous,**kwargs)
        if x>25e-6+1e-12:
            dx=x-previous['position'][0];attempts.append(dx)
            if dx>.3e-6:state['accepted']=False
        return state
    array.solve=solve
    result=array.run(np.zeros(2),.02,distance=25.5e-6,step=25e-6)
    assert result['status']=='BALANCED_COMPLETE'
    assert attempts[:2]==pytest.approx([.5e-6,.25e-6])
    assert result['rows'][-1]['x_m']==pytest.approx(25.5e-6)


def stopped_flat_array():
    p=GuidedRodParameters(.004,.0002,200e9,100.,.0001,5e-5)
    axis=np.array([.5,0.,-np.sqrt(.75)])
    rod=GuidedRod(p,np.array([0.,0.,p.tip_radius_m])-p.free_length_m*axis,axis)
    surface=EnvelopeSurface(np.zeros((41,41)),.0005,.0005,(-.01,-.01))
    return CondensedArray([rod],surface,rigid=True)


def test_compression_stop_transmits_extra_force_and_releases_on_unloading():
    array=stopped_flat_array()
    result=array.run(np.zeros(2),.03,distance=.0001)
    assert result['status']=='BALANCED_COMPLETE'
    row=result['rows'][-1]
    assert row['max_compression_m']==pytest.approx(.0001)
    assert row['T_N']==pytest.approx(.4*.03,abs=2e-5)
    assert row['P_N']==pytest.approx(.03,abs=2e-5)
    assert row['compression_stop_reactions_N'][0]>.02
    previous={k:np.asarray(v) if isinstance(v,list) else v for k,v in result['checkpoint'].items()}
    unloaded=array.solve(previous['position'][0]+25e-6,.001,previous)
    assert unloaded['accepted']
    assert 0<unloaded['compression'][0]<.0001
    assert -unloaded['force'][0]@array.axes[0]==pytest.approx(100*unloaded['compression'][0])


def test_svd_failure_uses_same_equations_with_lsmr(monkeypatch):
    import spine_sim.balanced_contact as module
    original=module.least_squares
    calls=[]
    def solve(*args,**kwargs):
        calls.append(kwargs.get('tr_solver'))
        if len(calls)==1:raise np.linalg.LinAlgError('SVD did not converge for slice = 0.')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'least_squares',solve)
    array=stopped_flat_array()
    state=array.solve(0.,.001,array.initial(np.zeros(2)))
    assert calls==[None,'lsmr']
    assert state['linear_solver_fallback'] and state['accepted']
    assert state['force'][:,2].sum()==pytest.approx(.001,abs=1e-6)
    assert state['gap'].min()>-1e-8


def test_budget_resume_preserves_preload_and_drag_history(monkeypatch):
    import spine_sim.balanced_contact as module
    array=stopped_flat_array();clock=[0.]
    monkeypatch.setattr(module.time,'process_time',lambda:clock[0])
    def interrupt(row):
        if row['phase']=='drag' and row['x_m']>=50e-6:clock[0]=2.
    stopped=array.run(np.zeros(2),.001,distance=.0002,cpu_budget_s=1,progress=interrupt)
    assert stopped['status']=='BALANCED_BUDGET_LIMIT' and stopped['phase']=='drag'
    continued=array.run(np.zeros(2),.001,distance=.0002,cpu_budget_s=1,resume=stopped)
    reference=array.run(np.zeros(2),.001,distance=.0002,cpu_budget_s=1)
    assert continued['status']=='BALANCED_COMPLETE'
    assert continued['rows'][:len(stopped['rows'])]==stopped['rows']
    for field in ['x_m','T_N','P_N','Y_m','Z_m']:
        assert [r[field] for r in continued['rows']]==pytest.approx([r[field] for r in reference['rows']])


@pytest.mark.parametrize('reason',['EXPOSED_LENGTH_LIMIT','AXIAL_EXTENSION_UNSUPPORTED'])
def test_continuation_does_not_bypass_physical_range_stop(reason):
    with pytest.raises(ValueError,match='can continue'):
        stopped_flat_array().run(np.zeros(2),.001,resume={
            'status':'BALANCED_AXIAL_RANGE','phase':'drag','reason':reason})


@pytest.mark.parametrize('fixed_axes',[(),('y',),('z',),('y','z')])
def test_local_fixed_coordinates_jacobian_and_boundary_balance(monkeypatch,fixed_axes):
    import spine_sim.balanced_contact as module
    array=stopped_flat_array()
    result=array.run(np.zeros(2),.001,distance=50e-6)
    previous={k:np.asarray(v) if isinstance(v,list) else v for k,v in result['checkpoint'].items()}
    original=module.least_squares
    def checked(fun,guess,**kwargs):
        analytic=kwargs['jac'](guess)[-2:]
        numeric=np.zeros_like(analytic);h=1e-5
        for j in range(len(guess)):
            plus=guess.copy();minus=guess.copy();plus[j]+=h;minus[j]-=h
            numeric[:,j]=(fun(plus)[-2:]-fun(minus)[-2:])/(2*h)
        np.testing.assert_allclose(analytic,numeric,atol=1e-8,rtol=1e-8)
        return original(fun,guess,**kwargs)
    monkeypatch.setattr(module,'least_squares',checked)
    bc={f'fixed_{name}':previous['position'][1 if name=='y' else 2] for name in fixed_axes}
    state=array.solve(previous['position'][0]+25e-6,.001,previous,**bc)
    assert state['accepted']
    for name,index,target in [('y',1,0.),('z',2,.001)]:
        if name in fixed_axes:
            assert state['position'][index]==pytest.approx(previous['position'][index],abs=1e-8)
        else:
            assert state['force'][:,index].sum()==pytest.approx(target,abs=1e-6)
    assert np.min(state['gap'])>=-1e-8


def test_fixed_coordinates_reject_unimplemented_recovery():
    array=stopped_flat_array()
    with pytest.raises(ValueError,match='recovery'):
        array.solve(0.,.001,array.initial(np.zeros(2)),fixed_y=0.,recover=True)
