"""Short physical paths covering array friction, redistribution and stops."""

import numpy as np
import pytest

from spine_sim.continuous_geometry import PlaneSurface, HeightFieldSurface
from spine_sim.guided_array import GuidedArray, PathSettings
from spine_sim.guided_rod import GuidedRod, GuidedRodParameters


def rod(*, spring=100., travel=.002, axis=(0., 0., -1.), offset=(0., 0.)):
    p = GuidedRodParameters(
        free_length_m=.004, diameter_m=.0002, young_modulus_Pa=200e9,
        spring_stiffness_N_per_m=spring, max_compression_m=travel,
        tip_radius_m=.00015, segments=2,
    )
    direction = np.asarray(axis)
    center = np.array((*offset, p.tip_radius_m))
    return GuidedRod(p, center - p.free_length_m * direction, direction)


def accepted(array, previous, x, load):
    trial = array.solve(previous, x, load)
    assert trial.status == "ACCEPTED", (trial.status, trial.residual, trial.details)
    assert trial.state is not None
    return trial.state


@pytest.mark.parametrize("friction_solver", ["projection", "active_set"])
def test_plane_slide_retains_kinetic_friction_and_reverse_increment_resticks(friction_solver):
    array = GuidedArray([rod()], PlaneSurface(), PathSettings(friction_solver=friction_solver))
    preload = .02
    state = accepted(array, array.unloaded(np.zeros(3)), 0., preload)
    assert state.modes == ("STICK",)
    for x in (.00010, .00011, .00012):
        state = accepted(array, state, x, preload)
        assert state.modes == ("SLIP",)
        assert -state.forces_N[0, 0] == pytest.approx(.4 * preload, rel=2e-5)
        assert state.forces_N[0, 2] == pytest.approx(preload, rel=2e-6)
        # Check original virtual work; a clipped diagnostic alone cannot prove
        # that friction opposes the actual contact material-point motion.
        row = state.diagnostics["per_spine"][0]
        normal = np.asarray(row["normal"])
        force = state.forces_N[0]
        tangent_force = force - (force @ normal) * normal
        material_increment = np.asarray(row["tangent_increment_m"])
        assert np.linalg.norm(material_increment) > 0
        assert tangent_force @ material_increment == pytest.approx(
            -.4 * preload * np.linalg.norm(material_increment), rel=2e-5, abs=1e-12)
    state = accepted(array, state, .000118, preload)
    assert state.modes == ("STICK",)
    assert abs(state.forces_N[0, 0]) < .4 * preload


@pytest.mark.parametrize("friction_solver", ["projection", "active_set"])
def test_unequal_springs_share_total_preload_with_free_lateral_balance(friction_solver):
    angle = np.deg2rad(20.)
    axis = (0., np.sin(angle), -np.cos(angle))
    array = GuidedArray([rod(spring=100., axis=axis, offset=(-.001, 0)),
                         rod(spring=300., axis=axis, offset=(.001, 0))],
                        PlaneSurface(), PathSettings(friction_solver=friction_solver))
    preload = .04
    state = accepted(array, array.unloaded(np.zeros(3)), 0., preload)
    assert state.forces_N[:, 2].sum() == pytest.approx(preload, abs=1e-8)
    assert state.forces_N[:, 1].sum() == pytest.approx(0., abs=1e-8)
    assert abs(state.forces_N[0, 2] - state.forces_N[1, 2]) > .005
    assert abs(state.position_m[1]) > 1e-6
    assert np.all(state.forces_N[:, 2] > 0)


@pytest.mark.parametrize("friction_solver", ["projection", "active_set"])
def test_upper_stop_carries_excess_load_then_releases_on_unloading(friction_solver):
    array = GuidedArray([rod(travel=50e-6)], PlaneSurface(), PathSettings(friction_solver=friction_solver))
    state = accepted(array, array.unloaded(np.zeros(3)), 0., .02)
    row = state.diagnostics["per_spine"][0]
    assert state.modes == ("STICK_HARDSTOP",)
    assert row["compression_m"] == pytest.approx(50e-6, abs=2e-9)
    assert row["upper_stop_reaction_N"] == pytest.approx(.015, abs=1e-7)
    state = accepted(array, state, 0., .002)
    row = state.diagnostics["per_spine"][0]
    assert state.modes == ("STICK",)
    assert row["compression_m"] == pytest.approx(20e-6, abs=2e-9)
    assert row["upper_stop_reaction_N"] == pytest.approx(0.)


@pytest.mark.parametrize("friction_solver", ["projection", "active_set"])
def test_separated_rod_is_unloaded_then_joins_load_balance(friction_solver):
    first, second = rod(), rod(offset=(.001, 0.))
    second.guide_position_m[2] += 50e-6
    array = GuidedArray([first, second], PlaneSurface(), PathSettings(friction_solver=friction_solver))
    state = accepted(array, array.unloaded(np.zeros(3)), 0., .002)
    assert state.modes == ("STICK", "OPEN")
    assert np.array_equal(state.rod_coordinates[1], second.zero_state())
    assert np.array_equal(state.forces_N[1], np.zeros(3))
    state = accepted(array, state, 0., .01)
    assert state.modes == ("STICK", "STICK")
    assert state.forces_N[:, 2] == pytest.approx([.0075, .0025], abs=2e-7)
    assert state.position_m[2] == pytest.approx(-75e-6, abs=2e-9)


def test_constant_preload_finds_lower_surface_after_support_is_lost():
    x = np.arange(161)*50e-6-.004
    height = np.broadcast_to(np.where(x < 0., 0., -100e-6), (81, 161)).copy()
    from dataclasses import asdict
    from spine_sim.ijms import simulate
    parameters = dict(
        array=dict(nx=1, ny=1, tip_positions_xy_m=[[-.0004, 0.]], theta_deg=90.,
                   spine=asdict(rod().parameters)),
        surface=dict(kind="heightfield", height_m=height, dx_m=50e-6, dy_m=50e-6,
                     origin_xy_m=[-.004, -.002]),
        load=dict(mode="total_force", value=.02),
        path=dict(start_xy_m=[0., 0.], search_distance_m=.0008, max_step_m=10e-6, preload_steps=1),
        solver=dict(friction_solver="active_set", multipoint_contact=True, stability="off"),
        metrics=dict(force_quantile=.1), output=dict(level="full"))
    output = simulate(parameters)
    summary = output.summary
    assert summary["status"] == "COMPLETED", summary["failure"]
    assert summary["final_total_force_N"][2] == pytest.approx(.02, rel=2e-6)
    assert summary["final_position_m"][2] < summary["preload_state"]["position_m"][2]-90e-6
    assert abs(output.trace_rows[-1]["per_spine"][0]["gap_m"]) < 2e-9
