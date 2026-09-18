"""Prescribed design pairing and interrupted preparation reuse."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from spine_sim import small_array_campaign as scan


def test_designs_preserve_user_parameters_and_deduplicate_references():
    config = scan.read_config()
    designs = scan.design_table(config)
    description = scan.describe(config)
    assert len(designs) == 350
    assert description["cases"] == 336000
    assert description["full_output_cases"] == 120
    assert description["surfaces"] == 320
    assert description["shards"] == 1605
    assert description["domain"]["size_x_m"] == pytest.approx(.068)
    assert description["domain"]["size_y_m"] == pytest.approx(.058)
    assert sum(item["full_reference"] for item in designs) == 8
    for item in designs:
        array, spine = item["array"], item["array"]["spine"]
        assert np.asarray(array["tip_positions_xy_m"]).mean(axis=0) == pytest.approx([0., 0.], abs=1e-17)
        assert spine["tip_radius_m"] in (50e-6, 100e-6)
        assert spine["diameter_m"] == .001
        assert spine["taper_length_m"] == .002
        if "angle_gradient_deg" in array:
            assert "free_length_m" not in spine
            assert array["heel_free_length_m"] == .004
        else:
            assert array["theta_deg"] in (50, 60, 70, 80)
            assert spine["free_length_m"] == .004
        if spine["mount_type"] == "fixed":
            assert spine["spring_stiffness_N_per_m"] == 0.
            assert spine["max_compression_m"] == 0.
        else:
            assert spine["max_compression_m"] == .004
            assert spine["spring_stiffness_N_per_m"] in (100, 400, 800, 1000, 2000, 5000)


def test_all_loads_and_designs_share_surface_realization_with_unique_cases():
    config = scan.read_config()
    designs = scan.design_table(config)
    schedule = iter(scan.shard_schedule(config, designs))
    first_five = [next(schedule) for _ in range(5)]
    assert [s.material_index for s in first_five] == list(range(5))
    assert all(s.sample_index == 0 and s.full for s in first_five)
    surface = dict(kind="heightfield", path="shared.npy", dx_m=1e-5, dy_m=1e-5)
    campaign = scan.build_campaign(config, designs, first_five[0], surface)
    assert len(campaign.cases) == 24
    assert len({c.case_id for c in campaign.cases}) == 24
    assert {c.parameters["load"]["value"] for c in campaign.cases} == {.5, 1., 2.}
    assert len({json.dumps(c.parameters["surface"], sort_keys=True) for c in campaign.cases}) == 1
    assert len({json.dumps(c.parameters["sample"], sort_keys=True) for c in campaign.cases}) == 1
    protocols = campaign.cases[0].parameters["metrics"]["protocols"]
    assert len(protocols) == 12
    assert all(p["persistence_distance_m"] == .0005 for p in protocols)
    assert {tuple(p["search_window_m"]) for p in protocols} == {(0., .002), (0., .005), (0., .01)}


def test_lazy_surface_reuse_and_scientific_config_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "TEMP_ROOT", tmp_path / "temporary")
    config = scan.read_config()
    calls = []

    def generate(**parameters):
        calls.append(parameters)
        return SimpleNamespace(height=np.arange(6, dtype=np.float32).reshape(2, 3),
                               metadata={"resolved_mode": "synthetic", "generation": {"source": "fixture"}})

    monkeypatch.setattr(scan, "generate_terrain", generate)
    output = tmp_path / "results"
    scan.freeze_config(output, config)
    scan.freeze_config(output, config)
    shard = next(scan.shard_schedule(config, scan.design_table(config)))
    first = scan.prepare_surface(output, config, shard)
    second = scan.prepare_surface(output, config, shard)
    assert first == second and len(calls) == 1
    assert calls[0]["mode"] == "synthetic"
    stored = np.load(first["path"], mmap_mode="r")
    assert stored.dtype == np.float64
    np.testing.assert_array_equal(stored, np.arange(6).reshape(2, 3))
    del stored
    metadata = json.loads(Path(first["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["generation"]["source"] == "fixture"
    changed = deepcopy(config)
    changed["preloads_N"] = [1., 2., 3.]
    with pytest.raises(ValueError, match="different scientific configuration"):
        scan.freeze_config(output, changed)


def test_status_counts_persisted_full_and_compact_physical_stops(tmp_path):
    file_case = tmp_path / "results" / "full" / "paths" / "case-a" / "summary.json"
    file_case.parent.mkdir(parents=True)
    file_case.write_text(json.dumps(dict(run_state="complete", status="BODY_CONTACT_LIMIT", wall_time_s=3.)), encoding="utf-8")
    database = tmp_path / "results" / "summary" / "case_summaries.sqlite3"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE case_summary (summary_json TEXT)")
        connection.execute("INSERT INTO case_summary VALUES (?)", (json.dumps(dict(run_state="execution_error", wall_time_s=2.)),))
    report = scan.status(tmp_path, scan.read_config())
    assert report["recorded_cases"] == 2
    assert report["not_recorded_cases"] == 335998
    assert report["execution_states"] == {"complete": 1, "execution_error": 1}
    assert report["physics_statuses"] == {"BODY_CONTACT_LIMIT": 1, "NO_PHYSICS_RESULT": 1}
    assert report["summed_case_wall_time_s"] == 5.
