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
