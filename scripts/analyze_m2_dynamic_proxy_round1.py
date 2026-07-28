"""Analyze the 15-seed M2 dynamic proxy-model baseline without auto-running round two."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FACTORS = (
    "tip_radius_m",
    "diameter_m",
    "axial_mode",
    "spring_stiffness_n_m",
    "installation_angle_deg",
)
OBJECTIVES = (
    "pull_p10_n",
    "effective_load_fraction",
    "maximum_continuous_load_length_m",
    "constraint_pass",
)


def _hardware_key(parameters: dict[str, Any]) -> str:
    return json.dumps(
        {name: parameters.get(name) for name in FACTORS},
        sort_keys=True,
        separators=(",", ":"),
    )


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def load_rows(campaign_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for summary_path in sorted((campaign_dir / "paths").glob("*/summary.json")):
        case_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
        validation = json.loads(
            (case_dir / "validation.json").read_text(encoding="utf-8")
        )
        parameters = summary.get("parameters")
        case_parameters = config.get("parameters", {})
        if not isinstance(parameters, dict):
            issues.append(f"{case_dir.name}: missing normalized parameters")
            continue
        screening = case_parameters.get("screening_policy", {})
        if screening.get("ranking_scope") != "project_model_proxy":
            issues.append(f"{case_dir.name}: unexpected ranking scope")
            continue
        terminal = summary.get("run_terminal_state")
        path_end = terminal == "path_end"
        numerical_valid = summary.get("numerical_state") == "converged"
        checks = validation.get("constraint_checks", {})
        constraint_pass = bool(
            path_end
            and checks.get("yield_ok")
            and checks.get("buckling_ok")
            and checks.get("rod_clearance_ok")
        )
        yield_strength = parameters.get("yield_strength_pa")
        stress = summary.get("maximum_bending_stress_pa")
        stress_utilization = (
            _finite(stress) / _finite(yield_strength, default=np.nan)
            if _finite(yield_strength, default=np.nan) > 0.0
            else np.inf
        )
        row = {
            "case_id": case_dir.name,
            "seed": int(screening["seed"]),
            "terrain_recipe_id": case_parameters["terrain_recipe_id"],
            "hardware_key": _hardware_key(parameters),
            "parameters": parameters,
            "terminal": terminal,
            "path_end": path_end,
            "numerical_valid": numerical_valid,
            "constraint_pass": float(constraint_pass),
            "project_model_baseline_eligible": bool(
                validation.get("project_model_baseline_eligible", False)
            ),
            "pull_p10_n": _finite(summary.get("global_pull_force_p10_n")),
            "pull_p25_n": _finite(summary.get("global_pull_force_p25_n")),
            "pull_median_n": _finite(summary.get("global_pull_force_median_n")),
            "steady_peak_n": _finite(
                summary.get("global_pull_force_steady_peak_n")
            ),
            "contact_fraction": _finite(summary.get("contact_fraction")),
            "effective_load_fraction": _finite(
                summary.get("effective_load_fraction")
            ),
            "maximum_continuous_load_length_m": _finite(
                summary.get("maximum_continuous_load_length_m")
            ),
            "stress_utilization": stress_utilization,
            "buckling_margin_n": _finite(
                summary.get("minimum_euler_buckling_margin_n"),
                default=-np.inf,
            ),
            "minimum_rod_clearance_m": _finite(
                summary.get("minimum_rod_clearance_m"),
                default=-np.inf,
            ),
            "impact_velocity_peak_m_s": _finite(
                summary.get("impact_velocity_peak_m_s")
            ),
            "energy_residual_peak_j": _finite(
                summary.get("maximum_abs_energy_residual_j")
            ),
            "detach_count": int(
                summary.get("event_counts", {}).get("detach_to_free", 0)
            ),
            "impact_count": int(
                summary.get("event_counts", {}).get("impact", 0)
            ),
            "hard_stop_count": int(
                summary.get("event_counts", {}).get("hard_stop", 0)
            ),
        }
        rows.append(row)
    return rows, issues


def _describe(values: list[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if not finite.size:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def summarize_configurations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["hardware_key"]].append(row)
    summaries: list[dict[str, Any]] = []
    metrics = (
        "pull_p10_n",
        "pull_p25_n",
        "pull_median_n",
        "steady_peak_n",
        "contact_fraction",
        "effective_load_fraction",
        "maximum_continuous_load_length_m",
        "stress_utilization",
        "buckling_margin_n",
        "minimum_rod_clearance_m",
        "impact_velocity_peak_m_s",
        "energy_residual_peak_j",
        "detach_count",
        "impact_count",
        "hard_stop_count",
    )
    for key, group in sorted(groups.items()):
        terminal_counts = Counter(row["terminal"] for row in group)
        record: dict[str, Any] = {
            "hardware_key": key,
            "parameters": group[0]["parameters"],
            "seed_count": len({row["seed"] for row in group}),
            "terminal_counts": dict(sorted(terminal_counts.items())),
            "path_end_fraction": float(np.mean([row["path_end"] for row in group])),
            "numerical_valid_fraction": float(
                np.mean([row["numerical_valid"] for row in group])
            ),
            "constraint_pass_fraction": float(
                np.mean([row["constraint_pass"] for row in group])
            ),
            "project_model_baseline_eligible_fraction": float(
                np.mean(
                    [row["project_model_baseline_eligible"] for row in group]
                )
            ),
            "metrics": {
                metric: _describe([float(row[metric]) for row in group])
                for metric in metrics
            },
        }
        summaries.append(record)
    return summaries


def _dominates(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.all(first >= second) and np.any(first > second))


def bootstrap_pareto(
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, Any]:
    seeds = sorted({row["seed"] for row in rows})
    hardware = sorted({row["hardware_key"] for row in rows})
    lookup = {(row["hardware_key"], row["seed"]): row for row in rows}
    counts = Counter()
    rng = np.random.default_rng(random_seed)
    for _ in range(replicates):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        vectors: dict[str, np.ndarray] = {}
        for key in hardware:
            sample_rows = [lookup[(key, int(seed))] for seed in sampled]
            vectors[key] = np.asarray(
                [
                    np.median([row["pull_p10_n"] for row in sample_rows]),
                    np.median(
                        [row["effective_load_fraction"] for row in sample_rows]
                    ),
                    np.median(
                        [
                            row["maximum_continuous_load_length_m"]
                            for row in sample_rows
                        ]
                    ),
                    np.mean([row["constraint_pass"] for row in sample_rows]),
                ],
                dtype=np.float64,
            )
        for key, vector in vectors.items():
            if not any(
                other_key != key and _dominates(other, vector)
                for other_key, other in vectors.items()
            ):
                counts[key] += 1
    return {
        "replicates": replicates,
        "random_seed": random_seed,
        "objectives": list(OBJECTIVES),
        "membership_probability": {
            key: counts[key] / replicates for key in hardware
        },
    }


def factor_marginals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for factor in FACTORS:
        levels: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row["parameters"].get(factor)
            if factor == "spring_stiffness_n_m" and row["parameters"].get(
                "axial_mode"
            ) == "rigid":
                value = "rigid"
            levels[str(value)].append(row)
        output[factor] = {
            level: {
                metric: _describe([float(row[metric]) for row in group])
                for metric in (
                    "pull_p10_n",
                    "effective_load_fraction",
                    "maximum_continuous_load_length_m",
                    "constraint_pass",
                    "stress_utilization",
                )
            }
            for level, group in sorted(levels.items())
        }
    return output


def write_csv(path: Path, configurations: list[dict[str, Any]]) -> None:
    fields = [
        *FACTORS,
        "seed_count",
        "path_end_fraction",
        "constraint_pass_fraction",
        "pull_p10_median_n",
        "pull_p25_median_n",
        "pull_median_median_n",
        "effective_load_fraction_median",
        "maximum_continuous_load_length_median_m",
        "stress_utilization_p90",
        "minimum_rod_clearance_minimum_m",
        "impact_velocity_p90_m_s",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in configurations:
            parameters = record["parameters"]
            metrics = record["metrics"]
            writer.writerow(
                {
                    **{factor: parameters.get(factor) for factor in FACTORS},
                    "seed_count": record["seed_count"],
                    "path_end_fraction": record["path_end_fraction"],
                    "constraint_pass_fraction": record[
                        "constraint_pass_fraction"
                    ],
                    "pull_p10_median_n": metrics["pull_p10_n"].get(
                        "median", 0.0
                    ),
                    "pull_p25_median_n": metrics["pull_p25_n"].get(
                        "median", 0.0
                    ),
                    "pull_median_median_n": metrics["pull_median_n"].get(
                        "median", 0.0
                    ),
                    "effective_load_fraction_median": metrics[
                        "effective_load_fraction"
                    ].get("median", 0.0),
                    "maximum_continuous_load_length_median_m": metrics[
                        "maximum_continuous_load_length_m"
                    ].get("median", 0.0),
                    "stress_utilization_p90": metrics[
                        "stress_utilization"
                    ].get("p90"),
                    "minimum_rod_clearance_minimum_m": metrics[
                        "minimum_rod_clearance_m"
                    ].get("minimum"),
                    "impact_velocity_p90_m_s": metrics[
                        "impact_velocity_peak_m_s"
                    ].get("p90"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/m2_dynamic_round1_proxy_analysis.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260728)
    args = parser.parse_args()
    rows, issues = load_rows(args.campaign_dir.resolve())
    terminal_counts = Counter(row["terminal"] for row in rows)
    configurations = summarize_configurations(rows)
    complete_design = (
        len(rows) == 900
        and len(configurations) == 60
        and all(record["seed_count"] == 15 for record in configurations)
    )
    report = {
        "schema_version": "1",
        "campaign_dir": str(args.campaign_dir.resolve()),
        "ranking_scope": "project_model_proxy",
        "requires_experimental_calibration": True,
        "automatic_second_round": False,
        "case_count": len(rows),
        "configuration_count": len(configurations),
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "data_issues": issues,
        "complete_paired_design": complete_design,
        "analysis_allowed": complete_design and not issues,
        "configuration_summaries": configurations,
        "factor_marginals": factor_marginals(rows),
        "bootstrap_pareto": (
            bootstrap_pareto(
                rows,
                replicates=args.bootstrap_replicates,
                random_seed=args.random_seed,
            )
            if complete_design and not issues
            else None
        ),
        "decision": (
            "review_proxy_robustness_before_any_round2"
            if complete_design and not issues
            else "current_cannot_conclude"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output.with_suffix(".csv"), configurations)
    args.output.with_suffix(".md").write_text(
        "\n".join(
            [
                "# M2 持续恒载代理模型第一轮分析",
                "",
                f"- case：{len(rows)}/900",
                f"- 构型：{len(configurations)}/60",
                f"- 完整配对设计：{complete_design}",
                f"- 分析允许：{report['analysis_allowed']}",
                f"- 终止状态：`{dict(sorted(terminal_counts.items()))}`",
                "- 排名范围：项目代理模型内相对比较；仍需现实标定",
                "- 自动第二轮：否",
                "",
                "完整数值见同名 JSON，构型摘要见同名 CSV。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_count": len(rows),
                "configuration_count": len(configurations),
                "complete_paired_design": complete_design,
                "analysis_allowed": report["analysis_allowed"],
                "terminal_counts": dict(sorted(terminal_counts.items())),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["analysis_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
