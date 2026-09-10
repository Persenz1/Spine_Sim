"""Production assembly and actual compact runner integration."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from spine_sim.core.config import CampaignSpec
from spine_sim.guided_array import PathState
from spine_sim.ijms import build_rods
from spine_sim.io.results import CompactResultStore
from spine_sim.runtime.backend import BackendConfig, discover_backend
from spine_sim.runtime.runner import CampaignRunner


def template():
    return json.loads((Path(__file__).parents[1]/"examples/ijms_campaign.json").read_text(encoding="utf-8"))


def test_heterogeneous_coplanar_tips_and_common_mouth():
    config = deepcopy(template()["cases"][0]["parameters"]["array"])
    config["theta_deg"] = [35., 75.]
    config["per_spine"] = [{"index":1, "tip_radius_m":0.00015}]
    config["common_mouth_height_m"] = 0.003
    del config["spine"]["free_length_m"]
    rods = build_rods(config)
    for rod in rods:
        assert abs(rod.guide_position_m[2]-0.003) < 1e-15
        assert abs(rod.evaluate(rod.zero_state()).center_m[2]-rod.parameters.tip_radius_m) < 1e-15
    config = deepcopy(template()["cases"][0]["parameters"]["array"])
    config.pop("nx")
    config.pop("ny")
    config["tip_positions_xy_m"] = [[0.,0.], [.001,.002]]
    assert len(build_rods(config)) == 2


def test_formal_runner_persists_new_physics_summary_and_resumes(tmp_path):
    raw = template()
    raw["mode"] = "formal"
    p = raw["cases"][0]["parameters"]
    p["array"]["nx"] = 1
    p["array"]["per_spine"] = []
    p["array"]["spine"]["segments"] = 1
    p["path"].update(search_distance_m=1e-6, max_step_m=1e-6, preload_steps=1)
    p["solver"]["stability"] = "off"
    p["metrics"]["persistence_distance_m"] = 0.5e-6
    p["output"]["level"] = "summary"
    campaign = CampaignSpec.from_mapping(raw)
    runner = CampaignRunner(campaign, tmp_path/"formal", discover_backend(BackendConfig(preference="cpu")))
    assert isinstance(runner.store, CompactResultStore)
    runner.prepare(raw)
    records = runner.run(workers=1)
    assert records[0].run_state == "complete"
    summary = runner.store.load_case_summary(records[0].case_id)
    assert summary["status"] == "COMPLETED"
    assert summary["metrics"]["complete"]
    assert summary["events"][0]["kind"] == "FIRST_CONTACT"
    assert np.isclose(summary["final_total_force_N"][2], p["load"]["value"], atol=1e-8)
    manifest = json.loads((tmp_path/"formal/manifest.json").read_text())
    assert manifest["solver_semantics_version"] == "guided-rod-incremental-contact-1"
    resumed = CampaignRunner(campaign, tmp_path/"formal", discover_backend(BackendConfig(preference="cpu")))
    resumed.prepare(raw)
    assert resumed.run(resume=True)[0].result_hash == records[0].result_hash


def test_continuation_snapshot_preserves_material_history():
    state = PathState(np.array([1.,2.,3.]), (np.array([.1,.2,.3]),),
                      np.array([[.1,0.,.2]]), np.array([[1.,2.,3.1]]),
                      (np.eye(3),), np.array([[0.,0.,1.]]), ("SLIP",), .2,
                      .002, ("ridge",))
    restored = PathState.from_snapshot(json.loads(json.dumps(state.snapshot())))
    assert restored.modes == state.modes
    assert np.array_equal(restored.rotations[0], state.rotations[0])
    assert np.array_equal(restored.rod_coordinates[0], state.rod_coordinates[0])
