"""Physical checks for the force/moment Ritz subspace."""
import numpy as np
import pytest

from spine_sim.guided_rod import GuidedRod, GuidedRodParameters
from spine_sim.modal_rod import ModalGuidedRod


def pair():
    p = GuidedRodParameters(free_length_m=.004, diameter_m=.001, taper_length_m=.002,
                            young_modulus_Pa=200e9, spring_stiffness_N_per_m=800.,
                            max_compression_m=.004, tip_radius_m=50e-6, segments=16)
    axis = np.array([.5, 0., -np.sqrt(.75)])
    return GuidedRod(p, np.zeros(3), axis), ModalGuidedRod(p, np.zeros(3), axis)


def test_ritz_retains_discrete_small_deflection_end_wrench_compliance():
    full, reduced = pair()
    p = full.parameters
    boundaries = full.segment_boundaries_m(full.zero_state())
    length = p.free_length_m
    weights = p.young_modulus_Pa*p.integrated_second_moment_m5(
        length-boundaries[1:], length-boundaries[:-1])/np.diff(boundaries)**2
    difference = np.eye(p.segments)-np.eye(p.segments, k=-1)
    stiffness = np.kron(difference.T @ np.diag(weights) @ difference, np.eye(2))
    e = full.evaluate(full.zero_state())
    wrench_jacobian = np.vstack((e.center_jacobian_m[:, 1:], e.rotation_jacobian[:, 1:]))
    basis = np.kron(reduced.bending_basis(0.), np.eye(2))
    expected = wrench_jacobian @ np.linalg.solve(stiffness, wrench_jacobian.T)
    actual = wrench_jacobian @ basis @ np.linalg.solve(basis.T @ stiffness @ basis,
                                                      basis.T @ wrench_jacobian.T)
    assert actual == pytest.approx(expected, rel=2e-6, abs=1e-11)


def test_retraction_derivative_includes_changing_modal_basis():
    _, rod = pair()
    state = np.array([.31, .02, -.01, .03, .008])
    e = rod.evaluate(state)
    for j in range(rod.dimension):
        step = 1e-6
        plus, minus = state.copy(), state.copy()
        plus[j] += step
        minus[j] -= step
        ep, em = rod.evaluate(plus, False), rod.evaluate(minus, False)
        assert e.energy_gradient_J[j] == pytest.approx((ep.energy_J-em.energy_J)/(2*step), rel=2e-5, abs=1e-9)
        assert e.center_jacobian_m[:, j] == pytest.approx((ep.center_m-em.center_m)/(2*step), rel=2e-5, abs=1e-10)
