"""Compare one paired coarse/fine M3 campaign at the same realization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spine_sim.array.convergence import compare_trend_summaries
from spine_sim.io.results import atomic_write_json, open_result_store


def _single_summary(campaign_dir: Path) -> dict:
    rows = list(
        open_result_store(campaign_dir).iter_case_summaries(
            verify_payloads=True
        )
    )
    if len(rows) != 1:
        raise ValueError(
            "resolution-pair analysis requires one case per campaign"
        )
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse_campaign_dir", type=Path)
    parser.add_argument("fine_campaign_dir", type=Path)
    parser.add_argument("--realization-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    coarse = _single_summary(args.coarse_campaign_dir)
    fine = _single_summary(args.fine_campaign_dir)
    identity_checks = {
        "same_realization": (
            coarse.get("terrain_realization_id")
            == fine.get("terrain_realization_id")
        ),
        "coarse_realization_matches_requested": (
            coarse.get("terrain_realization_id")
            == args.realization_id
        ),
        "fine_realization_matches_requested": (
            fine.get("terrain_realization_id")
            == args.realization_id
        ),
        "same_terrain_family": (
            coarse.get("terrain_family")
            == fine.get("terrain_family")
        ),
        "same_seed": coarse.get("seed") == fine.get("seed"),
        "same_configuration": (
            coarse.get("configuration_id")
            == fine.get("configuration_id")
        ),
        "same_loading_protocol": (
            coarse.get("loading_protocol_id")
            == fine.get("loading_protocol_id")
        ),
        "same_selected_origin": (
            coarse.get("selected_unit_origin_xy_m")
            == fine.get("selected_unit_origin_xy_m")
        ),
        "different_terrain_payload": (
            coarse.get("terrain_data_sha256")
            != fine.get("terrain_data_sha256")
        ),
    }
    comparison = compare_trend_summaries(coarse, fine)
    passed = bool(comparison["passed"]) and all(
        identity_checks.values()
    )
    report = {
        "schema_version": "m3-resolution-pair-report-v1",
        "realization_id": args.realization_id,
        "coarse_campaign_dir": str(
            args.coarse_campaign_dir.resolve()
        ),
        "fine_campaign_dir": str(args.fine_campaign_dir.resolve()),
        "coarse_case_id": coarse["case_id"],
        "fine_case_id": fine["case_id"],
        "identity_checks": identity_checks,
        "comparison": comparison,
        "passed": passed,
        "formal_ranking_eligible": False,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
