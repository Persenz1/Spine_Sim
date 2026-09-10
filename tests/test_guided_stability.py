from dataclasses import replace

import numpy as np

from spine_sim.continuous_geometry import PlaneSurface
from spine_sim.guided_array import GuidedArray, PathSettings, PathState
from spine_sim.guided_rod import GuidedRod, GuidedRodParameters
from spine_sim.guided_stability import assess_stick_stability


def test_plane_compressed_rod_checks_allowed_internal_and_yz_modes():
    parameters = GuidedRodParameters(
        free_length_m=.004, diameter_m=.00012, young_modulus_Pa=200e9,
        spring_stiffness_N_per_m=100, max_compression_m=.002,
        tip_radius_m=40e-6, segments=3,
    )
    rod = GuidedRod(parameters, (0, 0, .00404), (0, 0, -1))
    array = GuidedArray([rod], PlaneSurface(), PathSettings(check_body_clearance=False))
    force = .02
    coordinates = rod.zero_state()
    coordinates[0] = force / parameters.spring_stiffness_N_per_m / parameters.free_length_m
    e = rod.evaluate(coordinates)
    q = np.array((0, 0, -e.compression_m))
    state = PathState(q, (coordinates,), np.array(((0, 0, force),)),
                      np.array((e.center_m + q,)), (e.tip_rotation,),
                      np.array(((0, 0, 1),)), ("STICK",), force,
                      energy_J=e.energy_J, features=("plane",))
    result = assess_stick_stability(array, state)
    assert result["status"] == "CONSTRAINED_INCREMENTAL_ENERGY_POSITIVE", result
    assert result["internal_modes_checked"] > 0
    assert result["global_modes_checked"] == 2
    assert result["minimum_internal_eigenvalue_J"] > 0
    assert min(result["condensed_global_eigenvalues_J"]) > 0
    assert result["dynamic_stability"] == "OUT_OF_SCOPE"
    slipping = replace(state, modes=("SLIP",))
    assert assess_stick_stability(array, slipping)["status"] == "SLIDING_NONCONSERVATIVE"
