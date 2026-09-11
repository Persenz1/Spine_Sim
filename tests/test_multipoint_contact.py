"""Force sharing at distinct supports, without copying the rod/spring."""
import numpy as np
import pytest
import json
from pathlib import Path
from dataclasses import asdict

from spine_sim.continuous_geometry import HeightFieldSurface
from spine_sim.guided_array import GuidedArray, PathSettings, PathState
from spine_sim.guided_rod import GuidedRod, GuidedRodParameters


def groove_model(*, along_x=False, friction=0.):
    x = np.linspace(-.002, .002, 81)
    height = np.broadcast_to(.5*np.abs(x), (81,81)).copy()
    if along_x:
        height = height.T.copy()
    surface = HeightFieldSurface(height, 50e-6, 50e-6, origin_xy_m=(-.002,-.002))
    p = GuidedRodParameters(free_length_m=.004, diameter_m=.0002, young_modulus_Pa=200e9,
                           spring_stiffness_N_per_m=100., max_compression_m=.002,
                           tip_radius_m=.00015, segments=2)
    rod = GuidedRod(p, np.array([0.,0.,.004+p.tip_radius_m]), np.array([0.,0.,-1.]))
    cfg = PathSettings(multipoint_contact=True, friction_solver='active_set',
                       friction_static=friction, friction_kinetic=friction)
    model = GuidedArray([rod], surface, cfg)
    previous = model.unloaded([0.,0.,p.tip_radius_m*(np.sqrt(1.25)-1)])
    return model, previous


def test_v_groove_two_supports_share_one_spring_and_unload():
    model, previous = groove_model()
    cfg = model.settings
    trial = model.solve(previous, 0., .02)
    assert trial.status == 'ACCEPTED', (trial.status, trial.residual, trial.details)
    row = trial.state.diagnostics['per_spine'][0]
    contacts = row['contacts']
    assert len(contacts) == 2
    assert sorted(c['force_N'][0] for c in contacts) == pytest.approx([-.005,.005], abs=2e-8)
    assert [c['force_N'][2] for c in contacts] == pytest.approx([.01,.01], abs=2e-8)
    assert row['compression_m'] == pytest.approx(.0002, abs=2e-9)
    assert max(abs(c['gap_m']) for c in contacts) < cfg.contact_tolerance_m
    restored = PathState.from_snapshot(trial.state.snapshot())
    assert len(restored.contact_history[0]) == 2
    unloaded = model.solve(restored, 0., .005)
    assert unloaded.status == 'ACCEPTED', (unloaded.status, unloaded.residual)
    assert unloaded.state.diagnostics['per_spine'][0]['compression_m'] == pytest.approx(50e-6, abs=2e-9)


def test_two_supports_keep_material_slip_law_and_history():
    model, previous = groove_model(along_x=True, friction=.4)
    preload = model.solve(previous, 0., .02)
    assert preload.status == 'ACCEPTED', (preload.status, preload.residual)
    state = preload.state
    for x in [2e-6, 10e-6, 20e-6]:
        trial = model.solve(state, x, .02)
        assert trial.status == 'ACCEPTED', (trial.status, trial.residual, trial.details)
        state = trial.state
    contacts = state.diagnostics['per_spine'][0]['contacts']
    loaded = [c for c in contacts if c['N_N'] > 1e-6]
    assert len(loaded) == 2
    assert state.forces_N[0,2] == pytest.approx(.02, abs=2e-8)
    assert state.forces_N[0,1] == pytest.approx(0., abs=2e-8)
    for c in loaded:
        vt = np.asarray(c['tangent_increment_m'])
        ft = np.asarray(c['force_N'])-c['N_N']*np.asarray(c['normal'])
        assert c['mode'] == 'SLIP'
        assert ft == pytest.approx(-.4*c['N_N']*vt/np.linalg.norm(vt), abs=4e-8)
    restored = PathState.from_snapshot(state.snapshot())
    replay = model.solve(restored, 22e-6, .02)
    original = model.solve(state, 22e-6, .02)
    assert replay.status == original.status == 'ACCEPTED'
    assert replay.state.forces_N == pytest.approx(original.state.forces_N, abs=1e-10)
    reversal = model.solve(replay.state, 21.9e-6, .02)
    assert reversal.status == 'ACCEPTED', (reversal.status, reversal.residual, reversal.details)
    assert all(c['mode'] == 'STICK' for c in reversal.state.contact_history[0] if c['N_N'] > 1e-6)


def test_support_can_release_without_losing_total_preload():
    model, previous = groove_model()
    preload = model.solve(previous, 0., .02)
    assert preload.status == 'ACCEPTED'
    state = preload.state
    for x in [5e-6, 10e-6, 20e-6, 40e-6]:
        trial = model.solve(state, x, .02)
        assert trial.status == 'ACCEPTED', (trial.status, trial.residual, trial.details)
        state = trial.state
    row = state.diagnostics['per_spine'][0]
    assert len([c for c in row['contacts'] if c['N_N'] > 1e-6]) == 1
    assert state.forces_N[0,2] == pytest.approx(.02, abs=2e-8)
    assert row['gap_m'] >= -model.settings.contact_tolerance_m


def test_indeterminate_stick_forces_find_feasible_cones_before_slipping():
    from spine_sim.ijms import build_rods
    fixture = json.loads((Path(__file__).parent/'fixtures/multipoint_descent_transition.json').read_text())
    reference, _ = groove_model()
    rods = build_rods(dict(nx=1, ny=1, theta_deg=90., tip_positions_xy_m=[[-.0004,0.]],
                           spine=asdict(reference.rods[0].parameters)))
    x = np.arange(161)*50e-6-.004
    height = np.broadcast_to(np.where(x < 0., 0., -100e-6), (81,161)).copy()
    surface = HeightFieldSurface(height, 50e-6, 50e-6, origin_xy_m=(-.004,-.002))
    model = GuidedArray(rods, surface, PathSettings(multipoint_contact=True, friction_solver='active_set'))
    trial = model.solve(PathState.from_snapshot(fixture['previous']), fixture['target_x_m'], .02)
    assert trial.status == 'ACCEPTED', (trial.status, trial.residual, trial.details)
    assert trial.state.forces_N[0,2] == pytest.approx(.02, abs=2e-8)
    loaded = [c for c in trial.state.contact_history[0] if c['mode'] != 'OPEN']
    assert len(loaded) == 2
    for c in loaded:
        assert c['mode'] == 'STICK'
        ft = np.asarray(c['force_N'])-c['N_N']*np.asarray(c['normal'])
        assert np.linalg.norm(ft) <= .5*c['N_N']+4e-8
        assert abs(c['gap_m']) < model.settings.contact_tolerance_m
