"""One-shot M3 round-one audit/plot script; refuses to invent absent campaign data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spine_sim.io.results import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/m3_round1"))
    args = parser.parse_args()

    case_root = args.campaign_dir / "paths"
    summaries = []
    if case_root.is_dir():
        for path in sorted(case_root.glob("*/summary.json")):
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        raise SystemExit(
            "No M3 round-one case summaries found. The formal gated campaign has not run."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "case_count": len(summaries),
        "configuration_count": len(
            {item.get("configuration_id") for item in summaries}
        ),
        "run_state_counts": {
            state: sum(item.get("run_state") == state for item in summaries)
            for state in sorted({str(item.get("run_state")) for item in summaries})
        },
        "formal_eligible_count": sum(
            item.get("model_state") == "covered"
            and item.get("run_terminal_state") == "path_end"
            for item in summaries
        ),
    }
    atomic_write_json(args.output_dir / "data_integrity_audit.json", audit)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Install the project 'plot' extra to render the one-shot figure."
        ) from exc
    labels = [str(item.get("configuration_id", "unknown"))[-8:] for item in summaries]
    medians = np.asarray(
        [float(item.get("tangential_force_median_n", 0.0)) for item in summaries]
    )
    sharing = np.asarray(
        [float(item.get("neff_resultant_median", 0.0)) for item in summaries]
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
