"""Actual tapered-section energy, material retraction and fixed-mount references."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from spine_sim.guided_rod import GuidedRod, GuidedRodParameters, linear_reference_compliance


def parameters(**changes):
    return replace(GuidedRodParameters(
        free_length_m=4e-3, diameter_m=1e-3, young_modulus_Pa=200e9,
        spring_stiffness_N_per_m=800., max_compression_m=4e-3,
        tip_radius_m=50e-6, taper_length_m=2e-3, segments=8,
    ), **changes)


@pytest.mark.parametrize("length", [4e-3, 1e-3])
def test_tapered_energy_uses_actual_exposed_material_and_its_compression_derivative(length):
    p = parameters()
    rod = GuidedRod(p, (0., 0., 0.), (1., 0., 0.))
    x = rod.zero_state()
    x[0] = 1-length/p.free_length_m
    angle = .4
    x[1::2] = angle * rod.segment_fractions[1:]
    e = rod.evaluate(x)
    inertia_integral = quad(lambda u: np.pi*float(p.section_radius_m(u))**4/4,
                            0., length, points=[min(length, p.taper_length_m)], epsabs=1e-24)[0]
    assert e.bending_energy_J == pytest.approx(p.young_modulus_Pa*inertia_integral*angle**2/(2*length**2))
    assert e.center_m == pytest.approx((length*np.sin(angle)/angle, length*(1-np.cos(angle))/angle, 0.))
    step = 2e-7
    plus, minus = x.copy(), x.copy()
    plus[0] += step
    minus[0] -= step
    numerical = (rod.evaluate(plus, False).energy_J-rod.evaluate(minus, False).energy_J)/(2*step)
    assert e.energy_gradient_J[0] == pytest.approx(numerical, rel=2e-7, abs=1e-10)
    # Pulling a taper through the ideal guide exposes its original material,
    # never a stretched version of the initial 2 mm taper.
    assert p.section_radius_m(length) == pytest.approx(275e-6 if length == 1e-3 else 500e-6)


def test_local_stress_uses_tip_and_shaft_sections_not_one_effective_diameter():
    rod = GuidedRod(parameters(mount_type="fixed", spring_stiffness_N_per_m=0.),
                    (0., 0., 0.), (1., 0., 0.))
    x = rod.zero_state()
    stress = rod.stress(x, (1., 0., 0.))
    assert stress.section_von_mises_upper_Pa[0] == pytest.approx(1/(np.pi*(.5e-3)**2))
    assert stress.section_von_mises_upper_Pa[-1] == pytest.approx(1/(np.pi*(50e-6)**2))
    moment_stress = rod.stress(x, (0., 0., 0.), (0., 0., 1e-6))
    assert moment_stress.section_von_mises_upper_Pa[-1] == pytest.approx(4e-6/(np.pi*(50e-6)**3))


def test_actual_taper_compliance_and_fixed_mount_have_correct_small_deflection_reference():
    p = parameters()
    C = linear_reference_compliance(p, (1., 0., 0.))
    exact = quad(lambda u: u*u/(p.young_modulus_Pa*float(p.section_second_moment_m4(u))),
                 0., p.free_length_m, points=[p.taper_length_m], epsabs=1e-14)[0]
    assert C == pytest.approx(np.diag((1/p.spring_stiffness_N_per_m, exact, exact)))
    shaft = p.free_length_m**3/(3*p.bending_rigidity_Nm2)
    thin = p.free_length_m**3/(3*p.young_modulus_Pa*np.pi*p.tip_radius_m**4/4)
    assert shaft < exact < thin
    fixed = replace(p, mount_type="fixed", spring_stiffness_N_per_m=0.)
    assert linear_reference_compliance(fixed, (1., 0., 0.)) == pytest.approx(np.diag((0., exact, exact)))

    # Independent linear solution of this constant-curvature discretization:
    # moment in each cell is its mean tip lever arm times transverse force.
    errors = []
    for count in (4, 8, 16):
        rod = GuidedRod(replace(p, segments=count), (0., 0., 0.), (1., 0., 0.))
        boundaries = rod.segment_boundaries_m(rod.zero_state())
        lo, hi = p.free_length_m-boundaries[1:], p.free_length_m-boundaries[:-1]
        h = hi-lo
        discrete = np.sum(((lo+hi)/2)**2*h*h/(p.young_modulus_Pa*p.integrated_second_moment_m5(lo, hi)))
        errors.append(abs(discrete-exact)/exact)
    assert errors[2] < errors[1] < errors[0]
    assert errors[1] < .06


def test_rated_stroke_does_not_create_a_hard_stop_at_zero_exposed_length():
    p = parameters()
    assert p.max_compression_m == 4e-3
    assert p.compression_limit_m == 4e-3 and p.geometry_limited_compression
    rod = GuidedRod(p, (0., 0., 0.), (1., 0., 0.))
    x = rod.zero_state()
    x[0] = 1.
    boundary = rod.axial_boundary(x, np.zeros(rod.dimension))
    assert boundary.branch == "EXPOSED_LENGTH_LIMIT" and not boundary.admissible
    assert boundary.upper_reaction_N == 0
    with pytest.raises(ValueError, match="no exposed length"):
        rod.evaluate(x)

    fixed = GuidedRod(replace(p, mount_type="fixed", spring_stiffness_N_per_m=0.),
                      (0., 0., 0.), (1., 0., 0.))
    x = fixed.zero_state()
    e = fixed.evaluate(x)
    assert e.spring_energy_J == 0.
    boundary = fixed.axial_boundary(x, fixed.generalized_residual(x, (-1., 0., 0.)))
    assert boundary.branch == "FIXED" and boundary.admissible
    assert e.center_jacobian_m[:, 0] == pytest.approx((-p.free_length_m, 0., 0.))


def test_arbitrary_arc_length_samples_follow_tapered_nonuniform_nodes():
    rod = GuidedRod(parameters(), (0., 0., 0.), (1., 0., 0.))
    x = rod.zero_state()
    angle = .7
    x[1::2] = angle*rod.segment_fractions[1:]
    distances = np.array([0., .13e-3, 1.99e-3, 2.001e-3, 4e-3])
    points, tangents = rod.centerline_at(x, distances)
    local_angles = angle*distances/rod.parameters.free_length_m
    radius = rod.parameters.free_length_m/angle
    assert points == pytest.approx(np.column_stack((radius*np.sin(local_angles),
                                                    radius*(1-np.cos(local_angles)), np.zeros(len(distances)))))
    assert tangents == pytest.approx(np.column_stack((np.cos(local_angles), np.sin(local_angles),
                                                      np.zeros(len(distances)))))
