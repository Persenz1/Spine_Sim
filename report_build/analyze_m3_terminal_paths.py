from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analyze_m3_fine_paths import (
    aggregate_records,
    extract_case_metrics,
    group_aggregate,
    load_design_metadata,
    summarize_designs,
    write_csv,
    write_parquet,
)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


PAIR_DEFINITIONS = (
    (
        "刚度300→800",
        "strict_stiffness",
        "m3_full_design_679df696f2811efa84f4",
        "m3_full_design_5b14632013cae856bf1f",
    ),
    (
        "刚度800→2000",
        "strict_stiffness",
        "m3_full_design_5b14632013cae856bf1f",
        "m3_full_design_373480a2081564f5a91e",
    ),
    (
        "刚度300→2000",
        "strict_stiffness",
        "m3_full_design_679df696f2811efa84f4",
        "m3_full_design_373480a2081564f5a91e",
    ),
    (
        "方向2×5→5×2",
        "strict_orientation",
        "m3_full_design_ddb66bd524680d765f5c",
        "m3_full_design_7c8a201bbcbd0989e2b1",
    ),
    (
        "刚性快速A1→高承载A2",
        "engineering_contrast",
        "m3_full_design_b1417f0abdd7ab61b2b1",
        "m3_full_design_4f0c3f82336e077e90bc",
    ),
    (
        "柔顺300 A4→800 A5",
        "engineering_contrast",
        "m3_full_design_84bc8db4f9902f13b75c",
        "m3_full_design_10af3f726401a9407f30",
    ),
)

PAIR_METRICS = (
    ("establish_success", "稳定起载成功率", "higher"),
    ("establish25_penalized_mm", "惩罚后起载距离/mm", "lower"),
    ("duty25", "承载占空比", "higher"),
    (
        "positive_force_ratio_mean_clipped2",
        "稳健正向累计承载/Fx÷P",
        "higher",
    ),
    ("net_force_ratio_mean", "净承载/Fx÷P", "higher"),
    ("bearing_neff_fraction_median", "承载Neff/N", "higher"),
)

SPRING_COLORS = {
    "300": "#2A9D8F",
    "800": "#457B9D",
    "2000": "#1D3557",
    "rigid": "#E76F51",
}

ROLE_SHORT = {
    "优势候选A1": "A1",
    "优势候选A2": "A2",
    "优势候选A3": "A3",
    "优势候选A4": "A4",
    "优势候选A5": "A5",
    "机理对照C1": "C1",
    "机理对照C2": "C2",
    "刚度对照C3": "K300",
    "刚度对照C4": "K800",
    "刚度对照C5": "K2000",
    "方向对照C6": "D2x5",
    "方向对照C7": "D5x2",
}


def finite_quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    return float(np.quantile(array, q)) if array.size else math.nan


def add_terminal_case_fields(records: list[dict[str, Any]]) -> None:
    for record in records:
        established = math.isfinite(float(record["establish25_mm"]))
        record["establish_success"] = float(established)
        record["establish25_penalized_mm"] = (
            float(record["establish25_mm"]) if established else 10.5
        )


def status_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def q(name: str, probability: float) -> float:
        return finite_quantile(
            (float(record[name]) for record in records), probability
        )

    return {
        "completion_ratio_median": q("completion_ratio", 0.50),
        "completion_ratio_q10": q("completion_ratio", 0.10),
        "gap_free_case_rate": float(
            np.mean(
                [
                    float(record["support_gap_fraction"]) == 0.0
                    for record in records
                ]
            )
        ),
        "numerical_failure_case_rate": float(
            np.mean(
                [
                    float(record["numerical_failure_station_fraction"]) > 0.0
                    for record in records
                ]
            )
        ),
        "numerical_failure_station_fraction_median": q(
            "numerical_failure_station_fraction", 0.50
        ),
        "numerical_failure_station_fraction_q90": q(
            "numerical_failure_station_fraction", 0.90
        ),
        "preload_unreachable_case_rate": float(
            np.mean(
                [
                    float(record["preload_unreachable_station_fraction"]) > 0.0
                    for record in records
                ]
            )
        ),
        "preload_unreachable_station_fraction_median": q(
            "preload_unreachable_station_fraction", 0.50
        ),
        "recontact_case_rate": float(
            np.mean(
                [
                    float(record["recontact_required_station_fraction"]) > 0.0
                    for record in records
                ]
            )
        ),
        "recontact_required_station_fraction_median": q(
            "recontact_required_station_fraction", 0.50
        ),
        "recontacted_station_fraction_median": q(
            "recontact_station_fraction", 0.50
        ),
        "alternate_landing_station_fraction_median": q(
            "alternate_landing_station_fraction", 0.50
        ),
    }


def build_design_rows(
    records: list[dict[str, Any]],
    roles: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = summarize_designs(records)
    by_design: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_design[str(record["design_id"])].append(record)
    for row in rows:
        design_id = str(row["design_id"])
        row.update(roles[design_id])
        group = by_design[design_id]
        row.update(status_summary(group))
        quarter_successes = [
            float(record["establish25_0p25mm_window_mm"])
            for record in group
            if math.isfinite(
                float(record["establish25_0p25mm_window_mm"])
            )
        ]
        row["establish25_0p25mm_window_success_rate"] = (
            len(quarter_successes) / len(group)
        )
        row["establish25_0p25mm_window_median_mm_successes"] = (
            float(np.median(quarter_successes))
            if quarter_successes
            else math.nan
        )
    role_order = {role["role"]: index for index, role in enumerate(roles.values())}
    rows.sort(key=lambda row: role_order[str(row["role"])])
    return rows


def build_strata_rows(
    records: list[dict[str, Any]],
    roles: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = group_aggregate(
        records,
        ("design_id", "terrain_stratum", "preload_N"),
        "terminal_design_terrain_preload",
    )
    by_group: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["design_id"]),
            str(record["terrain_stratum"]),
            float(record["preload_N"]),
        )
        by_group[key].append(record)
    for row in rows:
        design_id = str(row["design_id"])
        row.update(roles[design_id])
        group = by_group[
            (
                design_id,
                str(row["terrain_stratum"]),
                float(row["preload_N"]),
            )
        ]
        row.update(status_summary(group))
    return rows


def add_worst_strata(
    design_rows: list[dict[str, Any]],
    strata_rows: list[dict[str, Any]],
) -> None:
    by_design: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strata_rows:
        by_design[str(row["design_id"])].append(row)
    for row in design_rows:
        strata = by_design[str(row["design_id"])]
        row["worst_stratum_establish25_success_rate"] = min(
            float(item["establish25_success_rate"]) for item in strata
        )
        row["worst_stratum_duty25_median"] = min(
            float(item["duty25_median"]) for item in strata
        )
        row["worst_stratum_net_force_ratio_median"] = min(
            float(item["net_force_ratio_mean_median"]) for item in strata
        )
        row["positive_net_stratum_rate"] = float(
            np.mean(
                [
                    float(item["net_force_ratio_mean_median"]) > 0.0
                    for item in strata
                ]
            )
        )


def design_curve_rows(
    curve_data: dict[str, Any],
    roles: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    x_mm = np.asarray(curve_data["path_x_mm"], dtype=np.float64)
    output: list[dict[str, Any]] = []
    for (design_id, preload), chunks in sorted(
        curve_data["design_preload_curves"].items()
    ):
        array = np.stack(chunks).astype(np.float64)
        quantiles = np.quantile(array, (0.10, 0.25, 0.50, 0.75, 0.90), axis=0)
        for index, distance in enumerate(x_mm):
            output.append(
                {
                    "design_id": design_id,
                    "role": roles[design_id]["role"],
                    "mechanism": roles[design_id]["mechanism"],
                    "preload_N": preload,
                    "distance_mm": float(distance),
                    "force_ratio_q10": float(quantiles[0, index]),
                    "force_ratio_q25": float(quantiles[1, index]),
                    "force_ratio_median": float(quantiles[2, index]),
                    "force_ratio_q75": float(quantiles[3, index]),
                    "force_ratio_q90": float(quantiles[4, index]),
                    "bearing25_station_fraction": float(
                        np.mean(array[:, index] >= 0.25)
                    ),
                }
            )
    return output


def cluster_means(
    records: list[dict[str, Any]],
    metric: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = float(record[metric])
        if math.isfinite(value):
            grouped[str(record["terrain_id"])].append(value)
    return {
        terrain_id: float(np.mean(values))
        for terrain_id, values in grouped.items()
    }


def bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    seed: int,
    replicates: int = 10000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = differences.size
    chunk = 500
    values: list[np.ndarray] = []
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        indices = rng.integers(0, n, size=(count, n))
        values.append(np.mean(differences[indices], axis=1))
    bootstrap = np.concatenate(values)
    return (
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def paired_comparisons(
    records: list[dict[str, Any]],
    roles: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    by_design: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_design[str(record["design_id"])].append(record)
    output: list[dict[str, Any]] = []
    for pair_index, (name, kind, design_a, design_b) in enumerate(
        PAIR_DEFINITIONS
    ):
        for metric_index, (metric, label, better) in enumerate(PAIR_METRICS):
            means_a = cluster_means(by_design[design_a], metric)
            means_b = cluster_means(by_design[design_b], metric)
            common = sorted(set(means_a) & set(means_b))
            a = np.asarray([means_a[key] for key in common], dtype=np.float64)
            b = np.asarray([means_b[key] for key in common], dtype=np.float64)
            difference = b - a
            low, high = bootstrap_mean_ci(
                difference,
                seed=20260730 + 100 * pair_index + metric_index,
            )
            output.append(
                {
                    "pair_name": name,
                    "comparison_kind": kind,
                    "design_A": design_a,
                    "role_A": roles[design_a]["role"],
                    "design_B": design_b,
                    "role_B": roles[design_b]["role"],
                    "metric": metric,
                    "metric_label": label,
                    "better_direction": better,
                    "terrain_pair_count": len(common),
                    "mean_A": float(np.mean(a)),
                    "mean_B": float(np.mean(b)),
                    "mean_difference_B_minus_A": float(np.mean(difference)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_B_better_fraction": float(
                        np.mean(b > a) if better == "higher" else np.mean(b < a)
                    ),
                }
            )
    return output


def plot_metric_panels(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [ROLE_SHORT[str(row["role"])] for row in rows]
    colors = [SPRING_COLORS[str(row["spring_label"])] for row in rows]
    panels = (
        ("establish25_success_rate", "Stable-bearing success", (0.0, 1.05)),
        (
            "establish25_median_mm_successes",
            "Establishment distance (mm; lower is better)",
            (0.0, None),
        ),
        ("duty25_median", "Bearing duty ratio", (0.0, 0.65)),
        (
            "positive_force_ratio_mean_clipped2_median",
            "Cumulative positive resistance (Fx/P)",
            (0.0, 0.40),
        ),
        ("net_force_ratio_mean_median", "Net resistance (Fx/P)", (-0.5, 0.45)),
        ("bearing_neff_fraction_median", "Effective sharing (Neff/N)", (0.0, 0.85)),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))
    x = np.arange(len(rows))
    for axis, (field, title, limits) in zip(axes.flat, panels):
        values = [float(row[field]) for row in rows]
        axis.bar(x, values, color=colors, alpha=0.9)
        axis.axhline(0.0, color="#555555", lw=0.8)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=55, ha="right")
        if limits[0] is not None:
            axis.set_ylim(bottom=limits[0])
        if limits[1] is not None:
            axis.set_ylim(top=limits[1])
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle("M3 terminal screening: 12-design mechanism comparison", y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_tradeoff(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 6.4))
    for row in rows:
        role = str(row["role"])
        axis.scatter(
            float(row["establish25_median_mm_successes"]),
            float(row["positive_force_ratio_mean_clipped2_median"]),
            s=45 + 180 * float(row["establish25_success_rate"]),
            color=SPRING_COLORS[str(row["spring_label"])],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.85,
        )
        axis.annotate(
            ROLE_SHORT[role],
            (
                float(row["establish25_median_mm_successes"]),
                float(row["positive_force_ratio_mean_clipped2_median"]),
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    for spring, color in SPRING_COLORS.items():
        axis.scatter([], [], color=color, label=spring)
    axis.set_xlabel("Median stable-bearing distance among successes (mm)")
    axis.set_ylabel("Median cumulative positive resistance (clipped Fx/P)")
    axis.set_title("Terminal trade-off: shorter search, stronger resistance, larger = more reliable")
    axis.grid(alpha=0.18)
    axis.legend(title="Axial support (N/m)", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def curve_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, float], list[dict[str, Any]]]:
    output: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[(str(row["design_id"]), float(row["preload_N"]))].append(row)
    for values in output.values():
        values.sort(key=lambda row: float(row["distance_mm"]))
    return output


def plot_stiffness_curves(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    lookup = curve_lookup(rows)
    designs = (
        ("m3_full_design_679df696f2811efa84f4", "300", "#2A9D8F"),
        ("m3_full_design_5b14632013cae856bf1f", "800", "#457B9D"),
        ("m3_full_design_373480a2081564f5a91e", "2000", "#1D3557"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), sharey=True)
    for axis, preload in zip(axes, (0.5, 1.0, 2.0)):
        for design_id, label, color in designs:
            curve = lookup[(design_id, preload)]
            axis.plot(
                [float(row["distance_mm"]) for row in curve],
                [float(row["force_ratio_median"]) for row in curve],
                color=color,
                lw=2,
                label=label,
            )
        axis.axhline(0.25, color="#555555", ls="--", lw=1)
        axis.axhline(0.0, color="#888888", lw=0.8)
        axis.set_title(f"Preload = {preload:g} N")
        axis.set_xlabel("Drag distance (mm)")
        axis.set_xlim(0.0, 10.0)
        axis.set_ylim(-1.0, 0.75)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Median normalized resistance, Fx/P")
    axes[-1].legend(title="Spring (N/m)", frameon=False)
    figure.suptitle("Strictly matched 2×2 stiffness triplet", y=1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_candidate_curves(
    rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    lookup = curve_lookup(rows)
    candidates = [
        row for row in design_rows if str(row["role"]).startswith("优势候选")
    ]
    candidate_colors = {
        "优势候选A1": "#E76F51",
        "优势候选A2": "#F4A261",
        "优势候选A3": "#1D3557",
        "优势候选A4": "#2A9D8F",
        "优势候选A5": "#457B9D",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), sharey=True)
    for axis, preload in zip(axes, (0.5, 1.0, 2.0)):
        for row in candidates:
            design_id = str(row["design_id"])
            curve = lookup[(design_id, preload)]
            axis.plot(
                [float(item["distance_mm"]) for item in curve],
                [float(item["force_ratio_median"]) for item in curve],
                lw=2,
                color=candidate_colors[str(row["role"])],
                label=ROLE_SHORT[str(row["role"])],
            )
        axis.axhline(0.25, color="#555555", ls="--", lw=1)
        axis.axhline(0.0, color="#888888", lw=0.8)
        axis.set_title(f"Preload = {preload:g} N")
        axis.set_xlabel("Drag distance (mm)")
        axis.set_xlim(0.0, 10.0)
        axis.set_ylim(-0.8, 0.75)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Median normalized resistance, Fx/P")
    axes[-1].legend(
        title="Candidate",
        loc="lower right",
        frameon=False,
        fontsize=9,
    )
    figure.suptitle("Five advantage candidates: median force-distance paths", y=1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze M3 terminal full paths.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.source_root / "fine" / "selected_designs.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    roles = {
        str(design_id): {
            "role": str(value["role"]),
            "mechanism": str(value["mechanism"]),
            "selection_reason": str(value["reason"]),
        }
        for design_id, value in selection["roles"].items()
    }
    metadata = load_design_metadata(selection_path)
    records, curve_data = extract_case_metrics(
        args.source_root / "final" / "paths",
        metadata,
    )
    add_terminal_case_fields(records)
    design_rows = build_design_rows(records, roles)
    strata_rows = build_strata_rows(records, roles)
    add_worst_strata(design_rows, strata_rows)
    design_preload_rows = group_aggregate(
        records,
        ("design_id", "preload_N"),
        "terminal_design_preload",
    )
    design_terrain_rows = group_aggregate(
        records,
        ("design_id", "terrain_stratum"),
        "terminal_design_terrain",
    )
    terrain_preload_rows = group_aggregate(
        records,
        ("terrain_stratum", "preload_N"),
        "terminal_terrain_preload",
    )
    for grouped_rows in (design_preload_rows, design_terrain_rows):
        for row in grouped_rows:
            row.update(roles[str(row["design_id"])])
    curve_rows = design_curve_rows(curve_data, roles)
    paired_rows = paired_comparisons(records, roles)
    status_counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        status_counts[(str(record["design_id"]), str(record["case_status"]))] += 1
    status_rows = [
        {
            "design_id": design_id,
            **roles[design_id],
            "case_status": status,
            "case_count": count,
            "case_fraction": count / 900.0,
        }
        for (design_id, status), count in sorted(status_counts.items())
    ]

    write_parquet(args.output_dir / "terminal_case_metrics.parquet", records)
    write_csv(args.output_dir / "terminal_design_metrics.csv", design_rows)
    write_parquet(args.output_dir / "terminal_design_metrics.parquet", design_rows)
    write_csv(
        args.output_dir / "terminal_design_terrain_preload_metrics.csv",
        strata_rows,
    )
    write_csv(
        args.output_dir / "terminal_design_preload_metrics.csv",
        design_preload_rows,
    )
    write_csv(
        args.output_dir / "terminal_design_terrain_metrics.csv",
        design_terrain_rows,
    )
    write_csv(
        args.output_dir / "terminal_terrain_preload_metrics.csv",
        terrain_preload_rows,
    )
    write_csv(args.output_dir / "terminal_case_status_counts.csv", status_rows)
    write_csv(args.output_dir / "terminal_paired_comparisons.csv", paired_rows)
    write_csv(args.output_dir / "terminal_force_distance_curves.csv", curve_rows)

    plot_metric_panels(
        design_rows,
        args.figure_dir / "终筛图1_12构型综合指标.png",
    )
    plot_tradeoff(
        design_rows,
        args.figure_dir / "终筛图2_搜索距离与承载权衡.png",
    )
    plot_stiffness_curves(
        curve_rows,
        args.figure_dir / "终筛图3_严格匹配刚度三联力距曲线.png",
    )
    plot_candidate_curves(
        curve_rows,
        design_rows,
        args.figure_dir / "终筛图4_五个优势候选力距曲线.png",
    )

    program_selection_path = (
        args.source_root / "final" / "selected_designs.json"
    )
    program_selection = json.loads(
        program_selection_path.read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": "m3-terminal-bearing-analysis-v1",
        "source_root": str(args.source_root.resolve()),
        "case_count": len(records),
        "design_count": len(design_rows),
        "terrain_condition_count": len({record["terrain_id"] for record in records}),
        "preloads_N": sorted({record["preload_N"] for record in records}),
        "path_length_mm": sorted({record["path_length_mm"] for record in records}),
        "dx_mm": sorted({record["dx_mm"] for record in records}),
        "establishment_rule": records[0]["establishment_definition"],
        "unsupported_station_treatment": "zero_force",
        "force_sign": records[0]["force_sign_convention"],
        "program_selected_design_ids_not_used_as_final_judgment": (
            program_selection.get("selected_design_ids", [])
        ),
    }
    (args.output_dir / "terminal_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
