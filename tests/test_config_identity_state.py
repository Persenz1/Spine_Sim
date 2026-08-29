from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spine_sim.core.config import (
    BaseCaseSpec,
    CampaignSpec,
    ProjectConfig,
    TerrainRegionSpec,
)
from spine_sim.core.errors import ConfigurationError
from spine_sim.core.identity import identity, stable_hash, track_id
from spine_sim.core.states import StateBundle
from spine_sim.core.versions import PROJECT_SCHEMA_VERSION


class IdentityTests(unittest.TestCase):
    def test_order_and_format_do_not_change_hash(self) -> None:
        left = {"b": [2, 3], "a": 1.0}
        right = {"a": 1.0, "b": [2, 3]}
        self.assertEqual(stable_hash(left), stable_hash(right))
        self.assertEqual(identity("case", left), identity("case", right))

    def test_version_changes_identity(self) -> None:
        self.assertNotEqual(
            identity("case", {"x": 1}, module_version="1"),
            identity("case", {"x": 1}, module_version="2"),
        )
        self.assertEqual(track_id({"region": "r", "x": [0, 1]}), track_id({"x": [0, 1], "region": "r"}))


class ConfigTests(unittest.TestCase):
    def test_project_relative_path_is_resolved_portably(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = ProjectConfig.from_mapping(
                {
                    "schema_version": PROJECT_SCHEMA_VERSION,
                    "module_version": "m0",
                    "results_root": "results/data",
                },
                base_dir=base,
            )
            self.assertEqual(config.results_root, (base / "results" / "data").resolve())

    def test_region_units_and_range(self) -> None:
        region = TerrainRegionSpec.from_mapping(
            {
                "terrain_recipe_id": "terrain_recipe_x",
                "origin_x": {"value": 0, "unit": "mm"},
                "origin_y": 0.0,
                "size_x": {"value": 10, "unit": "mm"},
                "size_y": {"value": 5, "unit": "mm"},
                "resolution": {"value": 10, "unit": "um"},
            }
        )
        self.assertAlmostEqual(region.size_x_m, 0.01)
        self.assertAlmostEqual(region.resolution_m, 1e-5)
        with self.assertRaises(ConfigurationError):
            TerrainRegionSpec.from_mapping(
                {
                    "terrain_recipe_id": "x",
                    "origin_x": 0,
                    "origin_y": 0,
                    "size_x": 0,
                    "size_y": 1,
                    "resolution": 1,
                }
            )

    def test_case_and_campaign_identity_are_deterministic(self) -> None:
        raw = {
            "module": "m0",
            "module_version": "1",
            "parameters": {"seed": 3, "radius_m": 5e-5},
        }
        one = BaseCaseSpec.from_mapping(raw)
        two = BaseCaseSpec.from_mapping(
            {**raw, "parameters": {"radius_m": 5e-5, "seed": 3}}
        )
        self.assertEqual(one.case_id, two.case_id)
        self.assertEqual(one.normalized_input_hash, two.normalized_input_hash)
        changed_semantics = BaseCaseSpec.from_mapping(
            {**raw, "solver_semantics_version": "different"}
        )
        self.assertNotEqual(one.case_id, changed_semantics.case_id)
        campaign = CampaignSpec("x", "1", "spine_sim.examples.fake_module:run_case", (one,))
        self.assertTrue(campaign.campaign_id.startswith("campaign_"))


class StateTests(unittest.TestCase):
    def test_dimensions_cannot_be_mixed_or_omitted(self) -> None:
        with self.assertRaises(ConfigurationError):
            StateBundle.from_mapping(
                {
                    "physical_state": "converged",
                    "numerical_state": "free",
                    "model_state": "covered",
                    "run_state": "complete",
                }
            )
        with self.assertRaises(ConfigurationError):
            StateBundle.from_mapping({"physical_state": "free"})


if __name__ == "__main__":
    unittest.main()
