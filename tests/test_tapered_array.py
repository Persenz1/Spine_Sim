"""Small array integration paths for real needle sections and mount boundaries."""

import numpy as np
import pytest

from spine_sim.continuous_geometry import PlaneSurface
from spine_sim.guided_array import GuidedArray, PathSettings
from spine_sim.ijms import build_rods


@pytest.mark.parametrize("mount_type", ["fixed", "spring"])
def test_two_by_two_tapered_array_keeps_preload_and_its_installation_boundary(mount_type):
    rods = build_rods(dict(
        nx=2, ny=2, spacing_x_m=4e-3, spacing_y_m=4e-3, theta_deg=60.,
        spine=dict(free_length_m=4e-3, diameter_m=1e-3, young_modulus_Pa=200e9,
                   tip_radius_m=50e-6, taper_length_m=2e-3, segments=4,
                   mount_type=mount_type, spring_stiffness_N_per_m=800., max_compression_m=4e-3),
    ))
    settings = PathSettings()
    model = GuidedArray(rods, PlaneSurface(), settings)
    state = model.unloaded(np.zeros(3))
    for x in (0., 10e-6):
        trial = model.solve(state, x, .5)
        assert trial.status == "ACCEPTED", (trial.status, trial.residual, trial.details)
        state = trial.state
        assert state is not None
        assert state.forces_N[:, 2].sum() == pytest.approx(.5, abs=.5*settings.residual_tolerance)
        assert state.forces_N[:, 1].sum() == pytest.approx(0., abs=.5*settings.residual_tolerance)
        for rod, coordinates, row in zip(rods, state.rod_coordinates, state.diagnostics["per_spine"]):
            assert rod.parameters.max_compression_m == 4e-3
            assert row["body_gap_m"] >= -settings.contact_tolerance_m
            assert row["exposed_length_m"] > 0.
            if mount_type == "fixed":
                assert coordinates[0] == 0.
                assert row["compression_m"] == 0.
                assert row["spring_energy_J"] == 0.
                assert row["spring_branch"] == "FIXED"
                assert row["fixed_mount_reaction_N"] == pytest.approx(-np.asarray(row["force_N"]))
                assert row["fixed_mount_axial_reaction_N"] == pytest.approx(-np.asarray(row["force_N"]) @ rod.axis)
            else:
                assert 0. < row["compression_m"] < rod.parameters.free_length_m
                assert row["spring_energy_J"] == pytest.approx(.5*800.*row["compression_m"]**2)
                assert row["nominal_spring_stroke_m"] == 4e-3
                assert row["spring_branch"] == "INTERIOR"
                assert row["upper_stop_reaction_N"] == 0.
