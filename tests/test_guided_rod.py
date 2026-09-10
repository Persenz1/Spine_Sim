"""Physical reference checks for the retracting finite rod."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import root

from spine_sim.guided_rod import (
    GuidedRod,
    GuidedRodParameters,
    linear_reference_compliance,
    rotation_increment,
    sphere_contact_moment,
)


def parameters(**changes):
    return replace(GuidedRodParameters(
        free_length_m=0.004, diameter_m=0.00012, young_modulus_Pa=200e9,
        spring_stiffness_N_per_m=100.0, max_compression_m=0.002,
        tip_radius_m=40e-6, segments=3,
    ), **changes)


def equilibrium(rod, force, moment=(0.0, 0.0, 0.0), initial=None):
    solved = root(lambda x: rod.generalized_residual(x, force, moment) / 1e-3,
                  rod.zero_state() if initial is None else initial, tol=1e-8)
    assert np.linalg.norm(solved.fun, ord=np.inf) < 2e-7, solved.message
    return solved.x, rod.evaluate(solved.x)


def test_axial_compression_stops_do_not_create_implicit_pullout_support():
    rod = GuidedRod(parameters(), (0, 0, 0), (1, 0, 0))
    x = rod.zero_state()
    x[0] = 0.1
    e = rod.evaluate(x)
    assert e.center_m == pytest.approx((0.0036, 0, 0))
    assert e.energy_J == pytest.approx(0.5 * 100 * 0.0004**2)
    assert rod.generalized_residual(x, (-0.04, 0, 0)) == pytest.approx(np.zeros(rod.dimension))

    x[0] = 0
    pull = rod.generalized_residual(x, (0.01, 0, 0))
    assert not rod.axial_boundary(x, pull).admissible
    assert rod.axial_boundary(x, pull).lower_reaction_N == 0
    shoulder = GuidedRod(parameters(lower_stop=True), (0, 0, 0), (1, 0, 0))
    assert shoulder.axial_boundary(x, pull).lower_reaction_N == pytest.approx(0.01)

    x[0] = rod.compression_bounds[1]
    excess = rod.generalized_residual(x, (-0.3, 0, 0))
    stop = rod.axial_boundary(x, excess)
    assert stop.admissible and stop.branch == "HARDSTOP"
    assert stop.upper_reaction_N == pytest.approx(0.1)
    assert not rod.axial_boundary(x, rod.generalized_residual(x, (-0.1, 0, 0))).admissible


def test_two_direction_small_bending_converges_to_cantilever_reference():
    force = np.array((-0.0001, 0.00007, 0.00004))
    errors = []
    for segments in (3, 6):
        rod = GuidedRod(parameters(segments=segments), (0, 0, 0), (1, 0, 0))
        _, e = equilibrium(rod, force)
        reference = linear_reference_compliance(rod.parameters, rod.axis, e.exposed_length_m) @ force
        actual = e.center_m - rod.parameters.free_length_m * rod.axis
        errors.append(np.linalg.norm(actual[1:] - reference[1:]) / np.linalg.norm(reference[1:]))
        assert actual[1] / actual[2] == pytest.approx(force[1] / force[2], rel=1e-6)
    assert errors[1] < 0.008
    assert errors[1] < errors[0] / 3


def test_finite_circular_arc_energy_and_end_rotation_are_exact():
    rod = GuidedRod(parameters(segments=3), (0.001, 0.002, 0.003), (1, 0, 0))
    x = rod.zero_state()
    x[0] = 0.2
    angle = 0.6
    x[1::2] = angle * np.arange(1, 4) / 3
    e = rod.evaluate(x)
    length = e.exposed_length_m
    assert e.center_m - rod.guide_position_m == pytest.approx(
        (length * np.sin(angle) / angle, length * (1 - np.cos(angle)) / angle, 0), abs=1e-14)
    assert e.bending_energy_J == pytest.approx(rod.parameters.bending_rigidity_Nm2 * angle**2 / (2 * length))
    assert rotation_increment(e.tip_rotation, rod.guide_rotation) == pytest.approx((0, 0, angle), abs=1e-12)
    points, directions = rod.sample_centerline(x, 6)
    assert points[-1] == pytest.approx(e.center_m)
    assert np.linalg.norm(directions, axis=1) == pytest.approx(np.ones(len(directions)))
    coarse, coarse_error = rod.clearance_centerline(x, subdivisions=2)
    refined, refined_error = rod.clearance_centerline(x, tolerance_m=1e-8)
    assert refined_error <= 1e-8 and len(refined) > len(coarse)
    assert coarse_error == pytest.approx(length / angle * (1 - np.cos(angle / 12)))
    # Retraction's derivative includes the changing bending domain.
    expected = (100 * e.compression_m + e.bending_energy_J / length) * 0.004
    assert e.energy_gradient_J[0] == pytest.approx(expected)


def test_moving_boundary_configuration_force_matches_planar_continuum_on_refinement():
    force = np.array((-0.06, 0.07, 0.0))
    discrepancies = []
    for segments in (3, 6):
        rod = GuidedRod(parameters(segments=segments), (0, 0, 0), (1, 0, 0))
        x, e = equilibrium(rod, force)
        moment = np.cross(e.center_m, force)
        configurational_force = np.linalg.norm(moment)**2 / (2 * rod.parameters.bending_rigidity_Nm2)
        spring_force = 100 * e.compression_m
        discrepancies.append(abs(-force[0] - spring_force - configurational_force))
        assert e.compression_m > 0
        assert abs(-force[0] - spring_force) > 1e-3  # Omitting configuration force would fail.
    assert discrepancies[1] < 2e-4
    assert discrepancies[1] < discrepancies[0] / 3


def test_tip_contact_moment_changes_bending_and_is_counted_once_in_stress():
    rod = GuidedRod(parameters(segments=4), (0, 0, 0), (1, 0, 0))
    force = np.array((-0.001, 0.0004, 0.0))
    moment = sphere_contact_moment(rod.parameters.tip_radius_m, (0, 1, 0), force)
    _, without = equilibrium(rod, force)
    x, e = equilibrium(rod, force, moment)
    assert abs(e.center_m[1] - without.center_m[1]) > 1e-8
    stress = rod.stress(x, force, moment, e)
    contact = e.center_m - rod.parameters.tip_radius_m * np.array((0, 1, 0))
    assert stress.guide_moment_Nm == pytest.approx(np.cross(contact - rod.guide_position_m, force))
    assert stress.utilization is None and not stress.model_limit
    limited = GuidedRod(replace(rod.parameters, allowable_stress_Pa=1), (0, 0, 0), (1, 0, 0))
    assert limited.stress(x, force, moment).model_limit


def test_energy_and_virtual_work_are_invariant_under_rigid_frame_rotation():
    rod = GuidedRod(parameters(), (0, 0, 0), (1, 0, 0))
    x = np.array((0.2, 0.07, -0.1, 0.15, -0.12, 0.3, -0.05))
    e = rod.evaluate(x)
    # A cyclic permutation preserves this deterministic guide-frame choice.
    Q = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
    rotated = GuidedRod(parameters(), (0, 0, 0), Q @ rod.axis)
    # Transfer tangent directions through the guide frames, without assuming
    # that a deterministic transverse-frame convention is rotation invariant.
    local_map = rotated.guide_rotation[:, 1:].T @ Q @ rod.guide_rotation[:, 1:]
    xr = x.copy()
    xr[1:] = (x[1:].reshape(-1, 2) @ local_map.T).ravel()
    er = rotated.evaluate(xr)
    assert er.energy_J == pytest.approx(e.energy_J, rel=1e-12)
    assert er.center_m == pytest.approx(Q @ e.center_m, abs=1e-14)
    force, torque = np.array((-.02, .01, .005)), np.array((1e-6, 2e-6, -3e-6))
    residual = rod.generalized_residual(x, force, torque)
    transformed = rotated.generalized_residual(xr, Q @ force, Q @ torque)
    assert transformed[0] == pytest.approx(residual[0], rel=1e-10)
    assert transformed[1:].reshape(-1, 2) == pytest.approx(
        residual[1:].reshape(-1, 2) @ local_map.T, rel=1e-7, abs=1e-11)
