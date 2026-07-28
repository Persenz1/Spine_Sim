"""Clone the M2 round-one baseline into paired medium/high friction campaigns.

The baseline ordering, terrain membership, hardware grid, time integration, and
all non-friction proxy parameters remain unchanged.  Only the Coulomb friction
pair and explicit screening metadata differ, so every tier contains the same
60 hardware configurations on the same 15 round-one seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from spine_sim.core.config import CampaignSpec


FRICTION_TIERS = {
    "medium": (0.60, 0.40),
    "high": (0.90, 0.60),
}


def build_tier(
    baseline: dict[str, Any],
    *,
    tier: str,
    static_friction: float,
    kinetic_friction: float,
    workers: int,
) -> dict[str, Any]:
    if len(baseline.get("cases", [])) != 900:
        raise ValueError("the source baseline must contain exactly 900 cases")
    if not 0.0 <= kinetic_friction <= static_friction:
        raise ValueError("friction coefficients require 0 <= kinetic <= static")

    campaign = copy.deepcopy(baseline)
    campaign["name"] = f"m2_dynamic_round1_proxy_{tier}_friction"
    campaign["workers"] = workers
    for case in campaign["cases"]:
        parameters = case["parameters"]
        spine = parameters["spine"]
        policy = parameters["screening_policy"]
        spine["static_friction"] = static_friction
        spine["kinetic_friction"] = kinetic_friction
        policy["parameter_set_id"] = f"m2_proxy_v1_friction_{tier}"
        policy["case_role"] = "friction_tier_full"
        policy["friction_tier"] = tier
        policy["friction_pair"] = {
            "static": static_friction,
            "kinetic": kinetic_friction,
        }
        case["tags"] = [
            tag
            for tag in case.get("tags", [])
            if not str(tag).startswith("friction_")
        ]
        case["tags"].append(f"friction_{tier}")

    parsed = CampaignSpec.from_mapping(campaign)
    if len(parsed.cases) != 900:
        raise AssertionError(f"expected 900 {tier} cases, got {len(parsed.cases)}")
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("examples/m2_dynamic_round1_proxy_baseline.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("examples"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    outputs: list[dict[str, Any]] = []
    for tier, (static_friction, kinetic_friction) in FRICTION_TIERS.items():
        campaign = build_tier(
            baseline,
            tier=tier,
            static_friction=static_friction,
            kinetic_friction=kinetic_friction,
            workers=args.workers,
        )
        output = (
            args.output_dir
            / f"m2_dynamic_round1_proxy_{tier}_friction.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(campaign, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        parsed = CampaignSpec.from_mapping(campaign)
        outputs.append(
            {
                "tier": tier,
                "static_friction": static_friction,
                "kinetic_friction": kinetic_friction,
                "output": str(output.resolve()),
                "campaign_id": parsed.campaign_id,
                "case_count": len(parsed.cases),
                "workers": parsed.workers,
            }
        )

    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
