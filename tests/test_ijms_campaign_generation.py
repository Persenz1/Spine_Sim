from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

from spine_sim.core.config import BaseCaseSpec, CampaignSpec


def test_paired_campaign_shards_preserve_samples_versions_and_nested_defaults(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_ijms_campaign.py"
    spec = importlib.util.spec_from_file_location("generate_ijms_campaign", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = BaseCaseSpec(
        "guided", "test-version", {"spring": {"k": 100.0, "travel": 0.01}, "surface": {"seed": 1}},
    )
    base = CampaignSpec("paired", "campaign-test", "guided:run_case", (template,))
    designs = [{"name": name, "parameters": {"spring": {"k": stiffness}}}
               for name, stiffness in (("soft", 100.0), ("stiff", 200.0))]
    samples = [{"surface_seed": seed, "placement_seed": 7, "split": "test", "placement_xy_m": [0.0, 0.01]}
               for seed in (17, 23)]
    counts = module.write_campaign_shards(asdict(base), designs, samples, output_directory=tmp_path, shard_size=3)
    assert counts == (4, 2)
    shards = [CampaignSpec.from_mapping(json.loads(path.read_text(encoding="utf-8")))
              for path in sorted(tmp_path.glob("campaign_*.json"))]
    assert [len(shard.cases) for shard in shards] == [3, 1]
    cases = [case for shard in shards for case in shard.cases]
    assert cases[0].parameters["sample"] == cases[2].parameters["sample"] == samples[0]
    assert cases[1].parameters["sample"] == cases[3].parameters["sample"] == samples[1]
    assert all(case.module_version == "test-version" for case in cases)
    assert all(case.parameters["spring"]["travel"] == 0.01 for case in cases)
    assert len({case.case_id for case in cases}) == 4
    assert template.parameters["spring"]["k"] == 100.0
