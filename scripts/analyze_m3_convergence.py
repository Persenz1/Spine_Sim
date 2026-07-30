"""Evaluate a completed compact/regular M3 convergence campaign."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from spine_sim.array.convergence import compare_trend_summaries
from spine_sim.io.results import atomic_write_json, open_result_store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/m3_convergence_report.json"),
    )
    args = parser.parse_args()

    original_config_path = (
        args.campaign_dir / "config" / "original.json"
    )
    original_config = json.loads(
        original_config_path.read_text(encoding="utf-8")
    )
    expected_cases = original_config.get(
        "cases",
        original_config.get("campaign", {}).get("cases", ()),
    )
    expected_summary_count = len(expected_cases)
    summaries = list(
        open_result_store(args.campaign_dir).iter_case_summaries(
            verify_payloads=True
        )
    )
    grouped: dict[
        tuple[str, float], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    invalid_rows: list[dict[str, Any]] = []
    for summary in summaries:
        metadata = summary.get("convergence_case")
        if not isinstance(metadata, dict):
            invalid_rows.append(
                {
                    "case_id": summary.get("case_id"),
                    "reason": "missing_convergence_case_metadata",
                }
            )
            continue
        key = (
            str(metadata["sentinel_configuration_id"]),
            float(metadata["preload_n"]),
        )
        grouped[key][str(metadata["variant_name"])] = summary

    comparisons: list[dict[str, Any]] = []
    group_passes: list[bool] = []
    for (
        sentinel_configuration_id,
        preload_n,
    ), variants in sorted(grouped.items()):
        reference = variants.get("reference")
        if reference is None:
            comparisons.append(
                {
                    "sentinel_configuration_id": (
                        sentinel_configuration_id
                    ),
                    "preload_n": preload_n,
                    "passed": False,
                    "failure_reason": "missing_reference",
                }
            )
            group_passes.append(False)
            continue
        for variant_name, candidate in sorted(variants.items()):
            if variant_name == "reference":
                continue
            result = compare_trend_summaries(reference, candidate)
            comparisons.append(
                {
                    "sentinel_configuration_id": (
                        sentinel_configuration_id
                    ),
                    "preload_n": preload_n,
                    "reference_case_id": reference["case_id"],
                    "candidate_case_id": candidate["case_id"],
                    "variant_name": variant_name,
                    **result,
                }
            )
            group_passes.append(bool(result["passed"]))

    expected_comparison_count = sum(
        max(len(variants) - 1, 0) for variants in grouped.values()
    )
    complete = (
        not invalid_rows
        and len(summaries) == expected_summary_count
        and len(comparisons) == expected_comparison_count
        and expected_comparison_count > 0
    )
    all_passed = complete and all(group_passes)
    report = {
        "schema_version": "m3-convergence-report-v1",
        "campaign_dir": str(args.campaign_dir.resolve()),
        "expected_summary_count": expected_summary_count,
        "summary_count": len(summaries),
        "group_count": len(grouped),
        "comparison_count": len(comparisons),
        "complete": complete,
        "all_passed": all_passed,
        "invalid_rows": invalid_rows,
        "comparisons": comparisons,
        "interpretation": (
            "proxy numerical/rate convergence only; physical calibration "
            "and formal ranking remain closed"
        ),
    }
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "comparisons"
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
