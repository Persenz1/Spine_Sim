import numpy as np
import pytest
from spine_sim.trend_array import distribute_preload,sampled_envelope,section_coefficients,solve_trend
from spine_sim.guided_rod import GuidedRod,GuidedRodParameters


def test_linear_compliance_matches_uniform_cantilever():
    p=GuidedRodParameters(.004,.0002,200e9,100.,.002,.0001)
    bending,_,_=section_coefficients(p)
    assert bending==pytest.approx(p.free_length_m**3/(3*p.bending_rigidity_Nm2),rel=1e-12)


def test_preload_active_set_and_downward_following():
    h=np.array([[0.,0.],[0.,-.001],[-.001,-.002]])
    z,f=distribute_preload(h,np.array([100.,300.]),.02)
    assert f[0]==pytest.approx([.005,.015])
    assert f[1]==pytest.approx([.02,0.])
    assert f.sum(axis=1)==pytest.approx([.02]*3,abs=1e-12)
    assert z[2]-z[1]==pytest.approx(-.001)


def test_flat_envelope_and_sliding_resistance():
    x=np.linspace(0.,.001,21)
    xy=np.stack((x,np.zeros_like(x)),axis=-1)[:,None,:]
    h=sampled_envelope(np.zeros((301,401)),1e-5,1e-5,(-.001,-.001),xy,5e-5,1e-5)
    assert h==pytest.approx(np.zeros((21,1)),abs=1e-15)
    p=GuidedRodParameters(.004,.0002,200e9,100.,.002,5e-5)
    rod=GuidedRod(p,np.array([0.,0.,.004]),np.array([.5,0.,-np.sqrt(.75)]))
    result=solve_trend(x,h,[rod],.02,.4)
    assert result['T_N']==pytest.approx(np.full(21,.008),abs=1e-12)
    assert result['normal_balance_error_N']<1e-12
