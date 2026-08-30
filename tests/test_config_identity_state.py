from __future__ import annotations

import unittest

from spine_sim import BackendConfig as PublicBackendConfig
from spine_sim.core.config import BaseCaseSpec, CampaignSpec
from spine_sim.core.errors import ConfigurationError
from spine_sim.core.identity import identity, stable_hash
from spine_sim.core.versions import PARAMETER_REGISTRY_VERSION
from spine_sim.runtime.backend import BackendConfig


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


class ConfigTests(unittest.TestCase):
    def test_backend_config_is_runtime_owned_and_validated(self) -> None:
        self.assertIs(PublicBackendConfig, BackendConfig)
        self.assertEqual(
            BackendConfig.from_mapping({"preference": "cpu"}).preference,
            "cpu",
        )
        with self.assertRaises(ConfigurationError):
            BackendConfig.from_mapping({"unknown": True})

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
        self.assertEqual(
            one.parameter_registry_version,
            PARAMETER_REGISTRY_VERSION,
        )
        changed_registry = BaseCaseSpec.from_mapping(
            {**raw, "parameter_registry_version": "different"}
        )
        self.assertNotEqual(one.case_id, changed_registry.case_id)
        campaign = CampaignSpec("x", "1", "spine_sim.examples.fake_module:run_case", (one,))
        self.assertTrue(campaign.campaign_id.startswith("campaign_"))


if __name__ == "__main__":
    unittest.main()
