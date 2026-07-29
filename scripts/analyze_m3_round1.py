"""M3 paired-coverage audit; failed initialization never becomes zero load."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spine_sim.io.results import ResultStore, atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "campaign_dirs",
        type=Path,
        nargs="+",
        help="one or more bounded M3 shard campaign directories",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/m3_round1"))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="audit a smoke/shard without claiming formal paired analysis",
    )
    args = parser.parse_args()

    summaries = []
    for campaign_dir in args.campaign_dirs:
        store = ResultStore(campaign_dir)
        if store.cases_dir.is_dir():
            for case_dir in sorted(store.cases_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                if not store.is_complete(case_dir.name):
                    raise SystemExit(
                        "M3 analysis rejected an incomplete or hash-invalid "
                        f"case: {case_dir.name}"
                    )
                summaries.append(store.load_case_summary(case_dir.name))
    if not summaries:
        raise SystemExit(
            "No M3 round-one case summaries found. The formal gated campaign has not run."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initialization_success = [
        item
        for item in summaries
        if item.get("initial_preload_success") is True
    ]
    performance = [
        item
        for item in summaries
        if item.get("ranking_inclusion_allowed") is True
        and item.get("tangential_force_median_n") is not None
    ]
    configuration_ids = {
        item.get("configuration_id") for item in summaries
    }
    terrain_ids = {
        item.get("terrain_condition_id") for item in summaries
    }
    protocol_ids = {
        item.get("loading_protocol_id") for item in summaries
    }
    pairing_keys = [
        (
            item.get("configuration_id"),
            item.get("terrain_condition_id"),
            item.get("loading_protocol_id"),
        )
        for item in summaries
    ]
    family_condition_counts = {
        family: len(
            {
                item.get("terrain_condition_id")
                for item in summaries
                if item.get("terrain_family") == family
            }
        )
        for family in ("sandpaper", "red_brick", "concrete")
    }
    formal_pairing_complete = (
        len(summaries) == 1_209_600
        and len(configuration_ids) == 1344
        and len(terrain_ids) == 300
        and len(protocol_ids) == 3
        and len(set(pairing_keys)) == len(pairing_keys)
        and family_condition_counts
        == {"sandpaper": 100, "red_brick": 100, "concrete": 100}
    )
    audit = {
        "case_count": len(summaries),
        "configuration_count": len(configuration_ids),
        "terrain_condition_count": len(terrain_ids),
        "loading_protocol_count": len(protocol_ids),
        "family_condition_counts": family_condition_counts,
        "formal_pairing_complete": formal_pairing_complete,
        "initialization_success_count": len(initialization_success),
        "initialization_coverage": (
            len(initialization_success) / len(summaries)
        ),
        "conditional_performance_case_count": len(performance),
        "initialization_failure_category_counts": {
            category: sum(
                item.get("initialization_failure_category") == category
                for item in summaries
            )
            for category in sorted(
                {
                    str(item.get("initialization_failure_category"))
                    for item in summaries
                    if item.get("initialization_failure_category") is not None
                }
            )
        },
        "run_state_counts": {
            state: sum(item.get("run_state") == state for item in summaries)
            for state in sorted({str(item.get("run_state")) for item in summaries})
        },
        "formal_eligible_count": sum(
            item.get("formal_ranking_eligible") is True
            for item in summaries
        ),
        "ranking_contract": (
            "only ranking_inclusion_allowed=true cases enter conditional "
            "performance; initialization failures remain coverage failures"
        ),
    }
    atomic_write_json(args.output_dir / "data_integrity_audit.json", audit)
    if not formal_pairing_complete and not args.allow_partial:
        raise SystemExit(
            "formal M3 analysis rejected: the complete 1344×300×3 paired "
            "matrix is absent; use --allow-partial only for smoke/shard audit"
        )
    if not performance:
        raise SystemExit(
            "no successfully settled path-end cases are available for performance analysis"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Install the project 'plot' extra to render the one-shot figure."
        ) from exc
    labels = [str(item.get("configuration_id", "unknown"))[-8:] for item in performance]
    medians = np.asarray(
        [float(item["tangential_force_median_n"]) for item in performance]
    )
    sharing = np.asarray(
        [float(item["neff_resultant_median"]) for item in performance]
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    scatter = axis.scatter(sharing, medians, c=np.arange(len(medians)), s=25)
    axis.set_xlabel("median effective pin count (resultant weights)")
    axis.set_ylabel("median unit tangential force [N]")
    axis.set_title("M3 round-one paired-case overview")
    for label, x_value, y_value in zip(labels, sharing, medians):
        axis.annotate(label, (x_value, y_value), fontsize=6, alpha=0.7)
    figure.colorbar(scatter, ax=axis, label="case index")
    figure.tight_layout()
    figure.savefig(args.output_dir / "m3_round1_overview.png", dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
