"""One-shot paired analysis for an approved, completed M2 round-1 campaign.

This script reads M0 case summaries only. It never launches simulations and never
materializes a second-round campaign.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METRICS = (
    "effective_load_fraction",
    "maximum_continuous_load_length_m",
    "tangential_force_p10_n",
    "tangential_force_p25_n",
    "tangential_force_median_n",
    "tangential_force_peak_n",
)
FACTORS = (
    "tip_radius_m",
    "diameter_m",
    "axial_mode",
    "spring_stiffness_n_m",
    "installation_angle_deg",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hardware_key(parameters: dict[str, Any]) -> str:
    return _canonical({name: parameters.get(name) for name in FACTORS})


def _load_rows(campaign_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for summary_path in sorted((campaign_dir / "paths").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        case_id = summary_path.parent.name
        if summary.get("run_state") != "complete":
            issues.append(f"{case_id}: run_state={summary.get('run_state')}")
            continue
        parameters = summary.get("parameters")
        if not isinstance(parameters, dict):
            issues.append(f"{case_id}: missing normalized M2 parameters")
            continue
        config_path = summary_path.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        case_parameters = config.get("parameters", {})
        terrain_id = case_parameters.get("terrain_recipe_id")
        if not terrain_id:
            issues.append(f"{case_id}: missing terrain_recipe_id pairing key")
            continue
        missing_metrics = [name for name in METRICS if name not in summary]
        if missing_metrics:
            issues.append(f"{case_id}: missing metrics {missing_metrics}")
            continue
        row = {
            "case_id": case_id,
            "terrain_recipe_id": terrain_id,
            "hardware_key": _hardware_key(parameters),
            "parameters": parameters,
            "formal_ranking_eligible": bool(
                json.loads(
                    (summary_path.parent / "validation.json").read_text(
                        encoding="utf-8"
                    )
                ).get("formal_ranking_eligible", False)
            ),
            "hard_stop_count": int(summary.get("event_counts", {}).get("hard_stop", 0)),
            "numerical_state": summary.get("numerical_state"),
            "model_state": summary.get("model_state"),
        }
        row.update({name: float(summary[name]) for name in METRICS})
        rows.append(row)
    return rows, issues


def _describe(values: Iterable[float]) -> dict[str, float | int]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {"count": 0}
    return {
        "count": int(data.size),
        "median": float(np.median(data)),
        "iqr": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
        "p10": float(np.quantile(data, 0.10)),
        "p25": float(np.quantile(data, 0.25)),
    }


def _group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["hardware_key"]].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        output.append(
            {
                "hardware_key": key,
                "parameters": group[0]["parameters"],
                "paired_terrain_count": len(
                    {row["terrain_recipe_id"] for row in group}
                ),
                "metrics": {
                    metric: _describe(row[metric] for row in group)
                    for metric in METRICS
                },
                "hard_stop_count": _describe(
                    row["hard_stop_count"] for row in group
                ),
            }
        )
    return output


def _factor_value(row: dict[str, Any], factor: str) -> str:
    value = row["parameters"].get(factor)
    if factor == "spring_stiffness_n_m" and row["parameters"].get("axial_mode") == "rigid":
        return "rigid"
    return str(value)


def _main_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for factor in FACTORS:
        levels: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            levels[_factor_value(row, factor)].append(row)
        output[factor] = {
            level: {
                metric: _describe(row[metric] for row in group)
                for metric in METRICS
            }
            for level, group in sorted(levels.items())
        }
    return output


def _paired_level_differences(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for factor in FACTORS:
        by_pair: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            by_pair[row["terrain_recipe_id"]][_factor_value(row, factor)].append(row)
        levels = sorted(
            {_factor_value(row, factor) for row in rows}
        )
        comparisons: dict[str, Any] = {}
        for first_index, first in enumerate(levels):
            for second in levels[first_index + 1 :]:
                metric_differences: dict[str, list[float]] = defaultdict(list)
                for seed_groups in by_pair.values():
                    if first not in seed_groups or second not in seed_groups:
                        continue
                    for metric in METRICS:
                        first_value = float(
                            np.median([row[metric] for row in seed_groups[first]])
                        )
                        second_value = float(
                            np.median([row[metric] for row in seed_groups[second]])
                        )
                        metric_differences[metric].append(second_value - first_value)
                comparisons[f"{second}_minus_{first}"] = {
                    metric: _describe(values)
                    for metric, values in metric_differences.items()
                }
        output[factor] = comparisons
    return output


def _dominates(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.all(first >= second) and np.any(first > second))


def _bootstrap_pareto(
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, Any]:
    seeds = sorted({row["terrain_recipe_id"] for row in rows})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["hardware_key"], row["terrain_recipe_id"])].append(row)
    hardware = sorted({row["hardware_key"] for row in rows})
    counts = {key: 0 for key in hardware}
    top_counts = {key: 0 for key in hardware}
    rng = np.random.default_rng(random_seed)
    for _ in range(replicates):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        objectives: dict[str, np.ndarray] = {}
        for key in hardware:
            samples = []
            for seed in sampled:
                candidates = grouped.get((key, str(seed)), ())
                if not candidates:
                    continue
                samples.append(
                    [
                        np.median(
                            [row["tangential_force_p10_n"] for row in candidates]
                        ),
                        np.median(
                            [row["effective_load_fraction"] for row in candidates]
                        ),
                        np.median(
                            [
                                row["maximum_continuous_load_length_m"]
                                for row in candidates
                            ]
                        ),
                        -np.median(
                            [row["hard_stop_count"] for row in candidates]
                        ),
                    ]
                )
            if samples:
                objectives[key] = np.median(
                    np.asarray(samples, dtype=np.float64), axis=0
                )
        pareto = []
        for key, objective in objectives.items():
            if not any(
                other_key != key and _dominates(other, objective)
                for other_key, other in objectives.items()
            ):
                pareto.append(key)
                counts[key] += 1
        if objectives:
            scales = np.ptp(
                np.asarray(list(objectives.values()), dtype=np.float64), axis=0
            )
            scales = np.where(scales > 0, scales, 1.0)
            matrix = np.asarray(list(objectives.values()), dtype=np.float64)
            minima = matrix.min(axis=0)
            normalized = {
                key: (value - minima) / scales for key, value in objectives.items()
            }
            compromise = max(
                normalized,
                key=lambda key: float(np.min(normalized[key])),
            )
            top_counts[compromise] += 1
    return {
        "replicates": replicates,
        "random_seed": random_seed,
        "pareto_membership_probability": {
            key: counts[key] / replicates for key in hardware
        },
        "maximin_compromise_probability": {
            key: top_counts[key] / replicates for key in hardware
        },
    }


def _interaction_table(
    rows: list[dict[str, Any]],
    first_factor: str,
    second_factor: str,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                _factor_value(row, first_factor),
                _factor_value(row, second_factor),
            )
        ].append(row)
    return {
        f"{first}|{second}": {
            metric: _describe(row[metric] for row in group)
            for metric in METRICS
        }
        for (first, second), group in sorted(groups.items())
    }


def analyze(
    campaign_dir: Path,
    *,
    bootstrap_replicates: int,
    random_seed: int,
) -> dict[str, Any]:
    rows, issues = _load_rows(campaign_dir)
    eligibility_issues = [
        f"{row['case_id']}: formal_ranking_eligible=false"
        for row in rows
        if not row["formal_ranking_eligible"]
    ]
    numerical_issues = [
        f"{row['case_id']}: numerical={row['numerical_state']} model={row['model_state']}"
        for row in rows
        if row["numerical_state"] != "converged" or row["model_state"] != "covered"
    ]
    complete_rows = [
        row
        for row in rows
        if row["formal_ranking_eligible"]
        and row["numerical_state"] == "converged"
        and row["model_state"] == "covered"
    ]
    report: dict[str, Any] = {
        "schema_version": "1",
        "campaign_dir": str(campaign_dir),
        "loaded_case_count": len(rows),
        "eligible_case_count": len(complete_rows),
        "data_issues": issues,
        "eligibility_issues": eligibility_issues,
        "numerical_or_model_issues": numerical_issues,
        "analysis_allowed": bool(complete_rows)
        and not issues
        and not eligibility_issues
        and not numerical_issues,
        "automatic_second_round": False,
    }
    if not report["analysis_allowed"]:
        report["decision"] = "current_cannot_conclude"
        return report
    report.update(
        {
            "configuration_summaries": _group_summaries(complete_rows),
            "main_effects": _main_effects(complete_rows),
            "paired_level_differences": _paired_level_differences(complete_rows),
            "interactions": {
                "tip_radius_x_spring_stiffness": _interaction_table(
                    complete_rows, "tip_radius_m", "spring_stiffness_n_m"
                ),
                "spring_stiffness_x_angle": _interaction_table(
                    complete_rows,
                    "spring_stiffness_n_m",
                    "installation_angle_deg",
                ),
                "diameter_x_spring_stiffness": _interaction_table(
                    complete_rows, "diameter_m", "spring_stiffness_n_m"
                ),
            },
            "bootstrap_pareto": _bootstrap_pareto(
                complete_rows,
                replicates=bootstrap_replicates,
                random_seed=random_seed,
            ),
            "decision": "review_required_before_second_round",
        }
    )
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# M2 第一轮自动分析摘要",
        "",
        f"- 读取 case：{report['loaded_case_count']}",
        f"- 可分析 case：{report['eligible_case_count']}",
        f"- 分析允许：{report['analysis_allowed']}",
        f"- 决策：`{report['decision']}`",
        "- 自动第二轮：否",
        "",
    ]
    if not report["analysis_allowed"]:
        lines.extend(
            [
                "## 阻断项",
                "",
                *[
                    f"- {issue}"
                    for issue in (
                        report["data_issues"]
                        + report["eligibility_issues"]
                        + report["numerical_or_model_issues"]
                    )
                ],
            ]
        )
    else:
        lines.extend(
            [
                "## 产物",
                "",
                "- 中位数、IQR、10%/25% 分位",
                "- 同 terrain realization 配对差值",
                "- 半径、刚度、针径、角度主效应",
                "- 三组冻结交互",
                "- bootstrap Pareto 与 maximin compromise 稳定性",
                "",
                "请审阅相邻 JSON 的完整数值后，再决定是否补 seed、加密刚度或加入 50°。",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/m2_round1_analysis.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260727)
    args = parser.parse_args()
    report = analyze(
        args.campaign_dir.resolve(),
        bootstrap_replicates=args.bootstrap_replicates,
        random_seed=args.random_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output.with_suffix(".md"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["analysis_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
