"""New assembly and summary semantics without a research simulation batch."""
import numpy as np
import pytest

from spine_sim.ijms import build_rods, build_surface, evaluate_protocols, _weighted_path_statistics


def test_gradient_uses_four_mm_heel_and_keeps_fixed_mounts_coplanar():
    config = dict(nx=3, ny=2, spacing_x_m=.005, spacing_y_m=.005,
                  angle_gradient_deg={"toe": 60, "heel": 80}, heel_free_length_m=.004,
                  spine=dict(diameter_m=.001, taper_length_m=.002, tip_radius_m=50e-6,
                             young_modulus_Pa=200e9, mount_type="fixed",
                             spring_stiffness_N_per_m=0., max_compression_m=0., segments=4))
    rods = build_rods(config)
    height = 50e-6 + .004*np.sin(np.deg2rad(80))
    assert rods[0].parameters.free_length_m == pytest.approx(.004)
    assert rods[2].parameters.free_length_m == pytest.approx(.004548632170413031)
    for rod in rods:
        assert rod.guide_position_m[2] == pytest.approx(height)
        assert rod.evaluate(rod.zero_state()).center_m[2] == pytest.approx(50e-6)


def test_multiwindow_summary_preserves_confirmed_prefix_and_unknown_tail():
    x, force = [0., .0005, .001], [.8, .8, .8]
    config = {"protocols": [
        dict(name="absolute", target_force_N=.5, persistence_distance_m=.0005,
             search_window_m=[0., .001]),
        dict(name="relative", target_fraction_P=1., persistence_distance_m=.0005,
             search_window_m=[0., .002]),
    ]}
    protocols, windows = evaluate_protocols(x, force, preload=1., distance=.002, metric_config=config)
    assert protocols["absolute"]["established"] is True
    assert protocols["absolute"]["confirmed_at_m"] == pytest.approx(.0005)
    assert protocols["relative"]["established"] is None
    assert windows["0:0.001"]["complete"]
    assert windows["0:0.002"]["J_positive"] is None
    assert windows["0:0.002"]["coverage"] == pytest.approx(.5)


def test_path_statistics_are_distance_weighted_and_keep_missing_coverage():
    rows = [dict(search_distance_m=x, counts={"n_active": count})
            for x, count in [(0., 0), (.0009, 2), (.001, 2)]]
    result = _weighted_path_statistics(rows, {"full": {"window_m": [0., .001]},
                                               "long": {"window_m": [0., .002]}})
    assert result["full"]["n_active"]["mean"] == pytest.approx(1.1)
    assert result["long"]["n_active"]["mean"] is None
    assert result["long"]["n_active"]["coverage"] == pytest.approx(.5)


def test_saved_raw_height_reuses_surface_and_preserves_material_source(tmp_path):
    import json
    path = tmp_path / "surface.npy"
    metadata = tmp_path / "surface.json"
    np.save(path, np.zeros((3, 4), dtype=np.float64))
    metadata.write_text(json.dumps({"material": "sandpaper", "resolved_mode": "synthetic",
                                    "generation": {"measured_patch_used": True}}), encoding="utf-8")
    config = dict(kind="heightfield", path=str(path), metadata_path=str(metadata), dx_m=1e-5)
    first = build_surface(config)
    assert build_surface(config) is first
    assert first.metadata["resolved_mode"] == "synthetic"
    assert first.metadata["generation"]["measured_patch_used"]
    assert first.query_sphere([1e-5, 1e-5, 1e-6], 1e-6).gap_m == pytest.approx(0.)
