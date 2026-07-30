from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


THRESHOLDS = (0.10, 0.25, 0.50)
HOLD_WINDOW_MM = 0.5
HOLD_WINDOW_STATIONS = 6
HOLD_REQUIRED_STATIONS = 5
POSITIVE_FORCE_CLIP_RATIO = 2.0


def finite_quantile(values: Iterable[float], probability: float) -> float:
    array = np.asarray(
        [value for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if array.size == 0:
        return math.nan
    return float(np.quantile(array, probability))


def first_crossing(x_mm: np.ndarray, values: np.ndarray, threshold: float) -> np.ndarray:
    hit = values >= threshold
    any_hit = np.any(hit, axis=1)
    first = np.argmax(hit, axis=1)
    return np.where(any_hit, x_mm[first], np.nan)


def sustained_establishment(
    x_mm: np.ndarray,
    values: np.ndarray,
    threshold: float,
    *,
    window_stations: int = HOLD_WINDOW_STATIONS,
    required_stations: int = HOLD_REQUIRED_STATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    hit = values >= threshold
    cumulative = np.pad(
        np.cumsum(hit, axis=1),
        ((0, 0), (1, 0)),
        constant_values=0,
    )
    window_count = (
        cumulative[:, window_stations:]
        - cumulative[:, :-window_stations]
    )
    qualifying = window_count >= required_stations
    success = np.any(qualifying, axis=1)
    first = np.argmax(qualifying, axis=1)
    distance_mm = np.where(success, x_mm[first], np.nan)
    return distance_mm, first


def run_diagnostics(
    hit: np.ndarray,
    first_index: int,
    dx_mm: float,
) -> tuple[float, float, float]:
    tail = hit[first_index:]
    all_below_runs: list[float] = []
    recovered_runs: list[float] = []
    terminal_below_mm = 0.0
    index = 0
    while index < tail.size:
        if tail[index]:
            index += 1
            continue
        end = index
        while end < tail.size and not tail[end]:
            end += 1
        length_mm = (end - index) * dx_mm
        all_below_runs.append(length_mm)
        if end < tail.size:
            recovered_runs.append(length_mm)
        else:
            terminal_below_mm = length_mm
        index = end
    maximum_below_mm = max(all_below_runs, default=0.0)
    recovery_q90_mm = (
        float(np.quantile(recovered_runs, 0.90)) if recovered_runs else 0.0
    )
    return maximum_below_mm, recovery_q90_mm, terminal_below_mm


def load_design_metadata(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for design in payload["selected_designs"]:
        package = design["package"]
        geometry = design["geometry"]
        stiffness = package["spring_stiffness_N_per_m"]
        spring = (
            "rigid"
            if package["spring_family"] == "rigid"
            else str(int(stiffness))
        )
        angle = (
            "60_to_80"
            if geometry["angle_pattern"] != "fixed"
            else f"fixed_{int(package['fixed_angle_deg'])}"
        )
        result[design["design_id"]] = {
            "array_shape": geometry["array_shape"],
            "nx": int(geometry["nx"]),
            "ny": int(geometry["ny"]),
            "spine_count": int(geometry["nx"] * geometry["ny"]),
            "spacing_mm": float(geometry["spacing_m"] * 1000.0),
            "angle_pattern": geometry["angle_pattern"],
            "angle_label": angle,
            "fixed_angle_deg": float(package["fixed_angle_deg"]),
            "tip_radius_um": float(package["tip_radius_m"] * 1e6),
            "diameter_mm": float(package["diameter_m"] * 1000.0),
            "spring_stiffness_N_per_m": (
                None if stiffness is None else float(stiffness)
            ),
            "spring_label": spring,
            "spring_family": package["spring_family"],
        }
    return result


def extract_case_metrics(
    path_dir: Path,
    design_metadata: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    curve_accumulator: dict[tuple[str, float], list[np.ndarray]] = defaultdict(list)
    design_curve_accumulator: dict[
        tuple[str, float], list[np.ndarray]
    ] = defaultdict(list)
    common_x_mm: np.ndarray | None = None

    for path in sorted(path_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            path_x_mm = archive["path_x_m"].astype(np.float64) * 1000.0
            common_x_mm = path_x_mm
            dx_mm = float(np.median(np.diff(path_x_mm)))
            case_ids = archive["case_id"].astype(str)
            design_ids = archive["design_id"].astype(str)
            preloads = archive["preload_N"].astype(np.float64)
            accepted = archive["accepted"].copy()
            station_status = archive["station_status"].copy()
            case_status = archive["case_status"].astype(str)
            completion_ratio = archive["completion_ratio"].astype(np.float64)
            force_x_N = archive["force_x_N"].astype(np.float64)
            normalized_force = np.where(
                accepted,
                force_x_N / preloads[:, None],
                0.0,
            )
            neff = archive["neff"].astype(np.float64)
            max_share = archive["max_load_share"].astype(np.float64)
            spine_counts = archive["spine_count"].astype(np.float64)
            recontacted = archive["recontacted"].copy()
            landing_offset_m = archive["landing_offset_m"].astype(np.float64)
            terrain_id = str(archive["terrain_id"].item())
            terrain = str(archive["terrain_stratum"].item())

        hold_window_stations = int(round(HOLD_WINDOW_MM / dx_mm)) + 1
        hold_required_stations = hold_window_stations - 1
        sensitivity_window_mm = 0.25
        sensitivity_window_stations = (
            int(math.ceil(sensitivity_window_mm / dx_mm - 1e-12)) + 1
        )
        sensitivity_required_stations = sensitivity_window_stations - 1
        establishment: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        onset: dict[float, np.ndarray] = {}
        for threshold in THRESHOLDS:
            onset[threshold] = first_crossing(
                path_x_mm,
                normalized_force,
                threshold,
            )
            establishment[threshold] = sustained_establishment(
                path_x_mm,
                normalized_force,
                threshold,
                window_stations=hold_window_stations,
                required_stations=hold_required_stations,
            )
        establish25_quarter_window, _ = sustained_establishment(
            path_x_mm,
            normalized_force,
            0.25,
            window_stations=sensitivity_window_stations,
            required_stations=sensitivity_required_stations,
        )
        actual_force = np.where(accepted, force_x_N, 0.0)
        absolute_establish25, _ = sustained_establishment(
            path_x_mm,
            actual_force,
            0.25,
            window_stations=hold_window_stations,
            required_stations=hold_required_stations,
        )
        absolute_establish50, _ = sustained_establishment(
            path_x_mm,
            actual_force,
            0.50,
            window_stations=hold_window_stations,
            required_stations=hold_required_stations,
        )

        early_mask = path_x_mm <= 1.0 + 1e-12
        middle_mask = (path_x_mm >= 4.0 - 1e-12) & (
            path_x_mm <= 6.0 + 1e-12
        )
        late_mask = path_x_mm >= 8.0 - 1e-12

        for index, design_id in enumerate(design_ids):
            metadata = design_metadata[design_id]
            force_ratio = normalized_force[index]
            bearing25 = force_ratio >= 0.25
            hold25_mm, hold25_index = establishment[0.25]
            if math.isfinite(float(hold25_mm[index])):
                (
                    maximum_below25_mm,
                    recovery_q90_mm,
                    terminal_below25_mm,
                ) = run_diagnostics(
                    bearing25,
                    int(hold25_index[index]),
                    dx_mm,
                )
            else:
                maximum_below25_mm = math.nan
                recovery_q90_mm = math.nan
                terminal_below25_mm = math.nan

            bearing_mask = bearing25 & accepted[index]
            if np.any(bearing_mask):
                bearing_neff_fraction = float(
                    np.median(
                        neff[index, bearing_mask] / spine_counts[index]
                    )
                )
                bearing_max_load_share = float(
                    np.median(max_share[index, bearing_mask])
                )
            else:
                bearing_neff_fraction = math.nan
                bearing_max_load_share = math.nan

            clipped_positive = np.clip(
                force_ratio,
                0.0,
                POSITIVE_FORCE_CLIP_RATIO,
            )
            raw_positive = np.maximum(force_ratio, 0.0)
            accepted_force = force_ratio[accepted[index]]
            gap = ~accepted[index]
            gap_runs: list[int] = []
            run_start: int | None = None
            for station, is_gap in enumerate(gap):
                if is_gap and run_start is None:
                    run_start = station
                elif not is_gap and run_start is not None:
                    gap_runs.append(station - run_start)
                    run_start = None
            if run_start is not None:
                gap_runs.append(gap.size - run_start)

            record: dict[str, Any] = {
                "case_id": str(case_ids[index]),
                "design_id": design_id,
                "terrain_id": terrain_id,
                "terrain_stratum": terrain,
                "preload_N": float(preloads[index]),
                "path_length_mm": float(path_x_mm[-1]),
                "dx_mm": dx_mm,
                "case_status": str(case_status[index]),
                "completion_ratio": float(completion_ratio[index]),
                "force_sign_convention": (
                    "positive=resistance_to_positive_x_drag"
                ),
                "establishment_definition": (
                    f"first {HOLD_WINDOW_MM:g} mm window with "
                    f">={hold_required_stations}/{hold_window_stations} "
                    "stations at Fx/P >= threshold"
                ),
                "onset10_mm": float(onset[0.10][index]),
                "onset25_mm": float(onset[0.25][index]),
                "onset50_mm": float(onset[0.50][index]),
                "establish10_mm": float(establishment[0.10][0][index]),
                "establish25_mm": float(establishment[0.25][0][index]),
                "establish25_0p25mm_window_mm": float(
                    establish25_quarter_window[index]
                ),
                "establish50_mm": float(establishment[0.50][0][index]),
                "establish_0p25N_mm": float(absolute_establish25[index]),
                "establish_0p50N_mm": float(absolute_establish50[index]),
                "duty10": float(np.mean(force_ratio >= 0.10)),
                "duty25": float(np.mean(bearing25)),
                "duty50": float(np.mean(force_ratio >= 0.50)),
                "positive_force_ratio_mean_raw": float(
                    np.mean(raw_positive)
                ),
                "positive_force_ratio_mean_clipped2": float(
                    np.mean(clipped_positive)
                ),
                "net_force_ratio_mean": float(np.mean(force_ratio)),
                "positive_force_N_mean_raw": float(
                    np.mean(raw_positive) * preloads[index]
                ),
                "positive_force_N_mean_clipped_2P": float(
                    np.mean(clipped_positive) * preloads[index]
                ),
                "net_force_N_mean": float(
                    np.mean(force_ratio) * preloads[index]
                ),
                "positive_resisting_work_mJ_raw": float(
                    np.trapezoid(
                        raw_positive * preloads[index],
                        path_x_mm / 1000.0,
                    )
                    * 1000.0
                ),
                "positive_resisting_work_mJ_clipped_2P": float(
                    np.trapezoid(
                        clipped_positive * preloads[index],
                        path_x_mm / 1000.0,
                    )
                    * 1000.0
                ),
                "net_work_mJ": float(
                    np.trapezoid(
                        force_ratio * preloads[index],
                        path_x_mm / 1000.0,
                    )
                    * 1000.0
                ),
                "force_ratio_q10_all_stations": float(
                    np.quantile(force_ratio, 0.10)
                ),
                "force_ratio_median_all_stations": float(
                    np.median(force_ratio)
                ),
                "force_ratio_q90_all_stations": float(
                    np.quantile(force_ratio, 0.90)
                ),
                "force_ratio_q99_all_stations": float(
                    np.quantile(force_ratio, 0.99)
                ),
                "force_ratio_peak_all_stations": float(
                    np.max(force_ratio)
                ),
                "accepted_force_ratio_median": (
                    float(np.median(accepted_force))
                    if accepted_force.size
                    else math.nan
                ),
                "early_0_1mm_net_force_ratio": float(
                    np.mean(force_ratio[early_mask])
                ),
                "middle_4_6mm_net_force_ratio": float(
                    np.mean(force_ratio[middle_mask])
                ),
                "late_8_10mm_net_force_ratio": float(
                    np.mean(force_ratio[late_mask])
                ),
                "early_0_1mm_positive_clipped2_ratio": float(
                    np.mean(clipped_positive[early_mask])
                ),
                "late_8_10mm_positive_clipped2_ratio": float(
                    np.mean(clipped_positive[late_mask])
                ),
                "maximum_below25_after_establish_mm": maximum_below25_mm,
                "recovery_q90_after_establish_mm": recovery_q90_mm,
                "terminal_below25_mm": terminal_below25_mm,
                "bearing_neff_fraction_median": bearing_neff_fraction,
                "bearing_max_load_share_median": bearing_max_load_share,
                "support_gap_fraction": float(np.mean(gap)),
                "recontact_required_station_fraction": float(
                    np.mean(station_status[index] == 1)
                ),
                "track_invalid_station_fraction": float(
                    np.mean(station_status[index] == 2)
                ),
                "numerical_failure_station_fraction": float(
                    np.mean(station_status[index] == 3)
                ),
                "preload_unreachable_station_fraction": float(
                    np.mean(station_status[index] == 4)
                ),
                "maximum_support_gap_mm": (
                    max(gap_runs, default=0) * dx_mm
                ),
                "recontact_station_fraction": float(
                    np.mean(recontacted[index])
                ),
                "alternate_landing_station_fraction": float(
                    np.mean(np.abs(landing_offset_m[index]) > 1e-15)
                ),
            }
            record.update(metadata)
            records.append(record)
            curve_accumulator[
                (metadata["spring_label"], float(preloads[index]))
            ].append(force_ratio.astype(np.float32))
            design_curve_accumulator[
                (design_id, float(preloads[index]))
            ].append(force_ratio.astype(np.float32))

    if common_x_mm is None:
        raise RuntimeError("no fine path NPZ files found")
    return records, {
        "path_x_mm": common_x_mm,
        "spring_preload_curves": curve_accumulator,
        "design_preload_curves": design_curve_accumulator,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def q(name: str, probability: float) -> float:
        return finite_quantile(
            (float(record[name]) for record in records),
            probability,
        )

    establish_success = [
        record
        for record in records
        if math.isfinite(float(record["establish25_mm"]))
    ]

    def q_success(name: str, probability: float) -> float:
        return finite_quantile(
            (float(record[name]) for record in establish_success),
            probability,
        )

    return {
        "case_count": len(records),
        "case_any_gap_rate": (
            sum(float(record["support_gap_fraction"]) > 0 for record in records)
            / len(records)
        ),
        "station_gap_fraction_median": q("support_gap_fraction", 0.50),
        "station_gap_fraction_q90": q("support_gap_fraction", 0.90),
        "onset10_median_mm": q("onset10_mm", 0.50),
        "onset25_median_mm": q("onset25_mm", 0.50),
        "onset50_median_mm": q("onset50_mm", 0.50),
        "establish10_success_rate": (
            sum(math.isfinite(float(record["establish10_mm"])) for record in records)
            / len(records)
        ),
        "establish25_success_rate": len(establish_success) / len(records),
        "establish50_success_rate": (
            sum(math.isfinite(float(record["establish50_mm"])) for record in records)
            / len(records)
        ),
        "establish_0p25N_success_rate": (
            sum(
                math.isfinite(float(record["establish_0p25N_mm"]))
                for record in records
            )
            / len(records)
        ),
        "establish_0p50N_success_rate": (
            sum(
                math.isfinite(float(record["establish_0p50N_mm"]))
                for record in records
            )
            / len(records)
        ),
        "establish_0p25N_median_mm_successes": finite_quantile(
            (
                float(record["establish_0p25N_mm"])
                for record in records
            ),
            0.50,
        ),
        "establish25_median_mm_successes": q_success(
            "establish25_mm", 0.50
        ),
        "establish25_q90_mm_successes": q_success(
            "establish25_mm", 0.90
        ),
        "duty10_median": q("duty10", 0.50),
        "duty25_median": q("duty25", 0.50),
        "duty25_q10": q("duty25", 0.10),
        "duty50_median": q("duty50", 0.50),
        "positive_force_ratio_mean_raw_median": q(
            "positive_force_ratio_mean_raw", 0.50
        ),
        "positive_force_ratio_mean_clipped2_median": q(
            "positive_force_ratio_mean_clipped2", 0.50
        ),
        "positive_force_ratio_mean_clipped2_q10": q(
            "positive_force_ratio_mean_clipped2", 0.10
        ),
        "net_force_ratio_mean_median": q("net_force_ratio_mean", 0.50),
        "net_force_ratio_mean_q10": q("net_force_ratio_mean", 0.10),
        "positive_force_N_mean_raw_median": q(
            "positive_force_N_mean_raw", 0.50
        ),
        "positive_force_N_mean_clipped_2P_median": q(
            "positive_force_N_mean_clipped_2P", 0.50
        ),
        "net_force_N_mean_median": q("net_force_N_mean", 0.50),
        "positive_resisting_work_mJ_raw_median": q(
            "positive_resisting_work_mJ_raw", 0.50
        ),
        "positive_resisting_work_mJ_clipped_2P_median": q(
            "positive_resisting_work_mJ_clipped_2P", 0.50
        ),
        "net_work_mJ_median": q("net_work_mJ", 0.50),
        "early_0_1mm_net_force_ratio_median": q(
            "early_0_1mm_net_force_ratio", 0.50
        ),
        "late_8_10mm_net_force_ratio_median": q(
            "late_8_10mm_net_force_ratio", 0.50
        ),
        "late_minus_early_net_force_ratio_median": (
            q("late_8_10mm_net_force_ratio", 0.50)
            - q("early_0_1mm_net_force_ratio", 0.50)
        ),
        "maximum_below25_median_mm_successes": q_success(
            "maximum_below25_after_establish_mm", 0.50
        ),
        "maximum_below25_q90_mm_successes": q_success(
            "maximum_below25_after_establish_mm", 0.90
        ),
        "recovery_q90_median_mm_successes": q_success(
            "recovery_q90_after_establish_mm", 0.50
        ),
        "terminal_below25_median_mm_successes": q_success(
            "terminal_below25_mm", 0.50
        ),
        "bearing_neff_fraction_median": q(
            "bearing_neff_fraction_median", 0.50
        ),
        "bearing_max_load_share_median": q(
            "bearing_max_load_share_median", 0.50
        ),
        "maximum_support_gap_q90_mm": q("maximum_support_gap_mm", 0.90),
        "recontact_station_fraction_median": q(
            "recontact_station_fraction", 0.50
        ),
        "alternate_landing_station_fraction_median": q(
            "alternate_landing_station_fraction", 0.50
        ),
        "force_ratio_q99_case_median": q(
            "force_ratio_q99_all_stations", 0.50
        ),
        "force_ratio_peak_case_q90": q(
            "force_ratio_peak_all_stations", 0.90
        ),
    }


def group_aggregate(
    records: list[dict[str, Any]],
    fields: tuple[str, ...],
    scope: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in fields)].append(record)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = {"scope": scope}
        row.update({field: value for field, value in zip(fields, key)})
        row.update(aggregate_records(group))
        output.append(row)
    return output


def summarize_designs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["design_id"])].append(record)
    output: list[dict[str, Any]] = []
    metadata_fields = (
        "array_shape",
        "nx",
        "ny",
        "spine_count",
        "spacing_mm",
        "angle_pattern",
        "angle_label",
        "fixed_angle_deg",
        "tip_radius_um",
        "diameter_mm",
        "spring_stiffness_N_per_m",
        "spring_label",
        "spring_family",
    )
    for design_id, group in grouped.items():
        row = {"design_id": design_id}
        row.update({field: group[0][field] for field in metadata_fields})
        row.update(aggregate_records(group))
        strata = group_aggregate(
            group,
            ("preload_N", "terrain_stratum"),
            "design_stratum",
        )
        row["worst_stratum_establish25_success_rate"] = min(
            item["establish25_success_rate"] for item in strata
        )
        row["worst_stratum_duty25_median"] = min(
            item["duty25_median"] for item in strata
        )
        row["worst_stratum_net_force_ratio_median"] = min(
            item["net_force_ratio_mean_median"] for item in strata
        )
        output.append(row)
    output.sort(
        key=lambda row: (
            -row["establish25_success_rate"],
            row["establish25_median_mm_successes"],
            -row["duty25_median"],
            -row["positive_force_ratio_mean_clipped2_median"],
            row["design_id"],
        )
    )
    for index, row in enumerate(output, start=1):
        row["descriptive_fast_bearing_order"] = index
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def build_curve_rows(
    curve_data: dict[str, Any],
) -> list[dict[str, Any]]:
    x_mm = curve_data["path_x_mm"]
    output: list[dict[str, Any]] = []
    for (spring, preload), chunks in sorted(
        curve_data["spring_preload_curves"].items()
    ):
        array = np.stack(chunks).astype(np.float64)
        quantiles = np.quantile(array, (0.10, 0.25, 0.50, 0.75, 0.90), axis=0)
        for index, distance_mm in enumerate(x_mm):
            output.append(
                {
                    "spring_label": spring,
                    "preload_N": preload,
                    "distance_mm": float(distance_mm),
                    "force_ratio_q10": float(quantiles[0, index]),
                    "force_ratio_q25": float(quantiles[1, index]),
                    "force_ratio_median": float(quantiles[2, index]),
                    "force_ratio_q75": float(quantiles[3, index]),
                    "force_ratio_q90": float(quantiles[4, index]),
                    "force_N_q10": float(quantiles[0, index] * preload),
                    "force_N_q25": float(quantiles[1, index] * preload),
                    "force_N_median": float(quantiles[2, index] * preload),
                    "force_N_q75": float(quantiles[3, index] * preload),
                    "force_N_q90": float(quantiles[4, index] * preload),
                    "positive_station_fraction": float(
                        np.mean(array[:, index] > 0.0)
                    ),
                    "bearing25_station_fraction": float(
                        np.mean(array[:, index] >= 0.25)
                    ),
                }
            )
    return output


def plot_spring_force_curves(curve_rows: list[dict[str, Any]], path: Path) -> None:
    colors = {
        "300": "#2A9D8F",
        "800": "#457B9D",
        "2000": "#1D3557",
        "rigid": "#E76F51",
    }
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=True)
    for axis, preload in zip(axes, (0.5, 1.0, 2.0)):
        for spring in ("300", "800", "2000", "rigid"):
            subset = [
                row
                for row in curve_rows
                if row["spring_label"] == spring
                and float(row["preload_N"]) == preload
            ]
            x = np.asarray([row["distance_mm"] for row in subset])
            median = np.asarray([row["force_ratio_median"] for row in subset])
            q25 = np.asarray([row["force_ratio_q25"] for row in subset])
            q75 = np.asarray([row["force_ratio_q75"] for row in subset])
            axis.plot(x, median, label=spring, color=colors[spring], lw=2)
            axis.fill_between(
                x,
                np.clip(q25, -1.2, 1.2),
                np.clip(q75, -1.2, 1.2),
                color=colors[spring],
                alpha=0.12,
            )
        axis.axhline(0.25, color="#555555", lw=1, ls="--")
        axis.axhline(0.0, color="#888888", lw=0.8)
        axis.set_title(f"Preload = {preload:g} N")
        axis.set_xlabel("Drag distance (mm)")
        axis.set_xlim(0.0, 10.0)
        axis.set_ylim(-1.2, 1.2)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Normalized drag resistance, Fx / preload")
    axes[-1].legend(title="Spring (N/m)", frameon=False)
    figure.suptitle(
        "Force-distance trend by axial support family (median and IQR)",
        y=1.02,
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_spring_metrics(factor_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [
        row
        for row in factor_rows
        if row["scope"] == "all_by_spring"
    ]
    order = ("300", "800", "2000", "rigid")
    rows.sort(key=lambda row: order.index(str(row["spring_label"])))
    labels = [str(row["spring_label"]) for row in rows]
    x = np.arange(len(rows))
    colors = ["#2A9D8F", "#457B9D", "#1D3557", "#E76F51"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    axes[0].bar(
        x,
        [row["establish25_success_rate"] for row in rows],
        color=colors,
    )
    axes[0].set_title("Stable bearing establishment")
    axes[0].set_ylabel("Case fraction")
    axes[0].set_ylim(0, 1)
    axes[1].bar(
        x,
        [row["establish25_median_mm_successes"] for row in rows],
        color=colors,
    )
    axes[1].set_title("Distance to stable bearing")
    axes[1].set_ylabel("Median distance among successes (mm)")
    axes[2].bar(
        x,
        [row["positive_force_ratio_mean_clipped2_median"] for row in rows],
        color=colors,
    )
    axes[2].set_title("Cumulative positive resistance")
    axes[2].set_ylabel("Median mean[clip(Fx/P, 0, 2)]")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        "Rigid arrays establish load quickly despite intermittent support",
        y=1.02,
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_spring_force_curves_absolute(
    curve_rows: list[dict[str, Any]], path: Path
) -> None:
    colors = {
        "300": "#2A9D8F",
        "800": "#457B9D",
        "2000": "#1D3557",
        "rigid": "#E76F51",
    }
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=False)
    for axis, preload in zip(axes, (0.5, 1.0, 2.0)):
        for spring in ("300", "800", "2000", "rigid"):
            subset = [
                row
                for row in curve_rows
                if row["spring_label"] == spring
                and float(row["preload_N"]) == preload
            ]
            x = np.asarray([row["distance_mm"] for row in subset])
            median = np.asarray([row["force_N_median"] for row in subset])
            q25 = np.asarray([row["force_N_q25"] for row in subset])
            q75 = np.asarray([row["force_N_q75"] for row in subset])
            axis.plot(x, median, label=spring, color=colors[spring], lw=2)
            axis.fill_between(
                x,
                np.clip(q25, -2.5, 2.5),
                np.clip(q75, -2.5, 2.5),
                color=colors[spring],
                alpha=0.12,
            )
        axis.axhline(0.25, color="#555555", lw=1, ls="--")
        axis.axhline(0.0, color="#888888", lw=0.8)
        axis.set_title(f"Preload = {preload:g} N")
        axis.set_xlabel("Drag distance (mm)")
        axis.set_xlim(0.0, 10.0)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Drag resistance, Fx (N)")
    axes[-1].legend(title="Spring (N/m)", frameon=False)
    figure.suptitle(
        "Absolute force-distance trend by axial support family",
        y=1.02,
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_design_tradeoff(designs: list[dict[str, Any]], path: Path) -> None:
    colors = {
        "300": "#2A9D8F",
        "800": "#457B9D",
        "2000": "#1D3557",
        "rigid": "#E76F51",
    }
    figure, axis = plt.subplots(figsize=(8.4, 6.0))
    for spring in ("300", "800", "2000", "rigid"):
        subset = [row for row in designs if row["spring_label"] == spring]
        axis.scatter(
            [row["establish25_median_mm_successes"] for row in subset],
            [row["positive_force_ratio_mean_clipped2_median"] for row in subset],
            s=[
                25 + 120 * row["establish25_success_rate"]
                for row in subset
            ],
            alpha=0.75,
            color=colors[spring],
            edgecolor="white",
            linewidth=0.5,
            label=spring,
        )
    axis.set_xlabel("Median stable-bearing distance among successes (mm)")
    axis.set_ylabel("Median cumulative positive resistance (clipped Fx/P)")
    axis.set_title("Design trade-off: search distance versus resisting capacity")
    axis.grid(alpha=0.18)
    axis.legend(title="Spring (N/m)", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_data_dictionary(path: Path) -> None:
    path.write_text(
        """# M3细筛派生指标说明

## 力方向

`force_x_N` 是阵列/背板对外的+x方向反力，源码定义为
`-sum(wall_force_x_on_spines)`。因此：

- `force_x_N > 0`：阻碍+x拖拽，为正向抗拖承载；
- `force_x_N < 0`：地形法向分量在该站点助推+x拖拽；
- 所有归一化指标使用 `force_x_N / preload_N`。

## 承载建立

- `onsetXX_mm`：第一次达到 `Fx/preload >= XX/100` 的距离，允许单点峰值。
- `establishXX_mm`：第一个0.5 mm滑动窗口起点；窗口包含6个站点，
  至少5个站点达到阈值。未建立时为空值。
- 主分析阈值为 `Fx/preload >= 0.25`。0.10和0.50作为灵敏度对照。

## 承载维持与恢复

- `duty25`：全10 mm路径中达到 `Fx/preload >= 0.25` 的站点比例。
- `maximum_below25_after_establish_mm`：建立后最长低于阈值的连续距离。
- `recovery_q90_after_establish_mm`：能够恢复的低承载区间长度90分位。
- `terminal_below25_mm`：路径结束时仍未恢复的低承载尾段。

## 累计承载

- `positive_force_ratio_mean_raw`：`mean(max(Fx/preload, 0))`。
- `positive_force_ratio_mean_clipped2`：为避免少量准静态几何尖峰主导积分，
  使用 `mean(clip(Fx/preload, 0, 2))`。原始值同时保留。
- `net_force_ratio_mean`：`mean(Fx/preload)`，保留助推区段的负贡献。
- `positive_resisting_work_mJ_*`：10 mm路径上正向抗拖力的积分；
  同时保留原始积分和按`2×preload`截顶后的稳健积分。
- `establish_0p25N_mm`、`establish_0p50N_mm`：使用固定绝对力阈值，
  用于跨预载比较工程起载能力。

## 支撑和均载

- 未接受的空间站在承载积分中按零力处理，而不是删除。
- `support_gap_fraction` 单独记录数值上无法建立平衡的站点比例。
- `bearing_neff_fraction_median` 和 `bearing_max_load_share_median`
  仅在 `Fx/preload >= 0.25` 的承载站点计算。

## 解释边界

这些指标描述当前恒定预载、准静态、无损伤模型中的相对趋势。
刚性阵列的支撑间断当前被求解器记为数值平衡失败，不能直接等同于现实脱落概率。
红砖和混凝土地形仍是未完成实验标定的合成地形。
""",
        encoding="utf-8",
    )


def write_sha256_manifest(root: Path, output_path: Path) -> None:
    rows: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output_path:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append((path.relative_to(root).as_posix(), path.stat().st_size, digest.hexdigest()))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "size_bytes", "sha256"))
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    arguments = parser.parse_args()

    archive_root = arguments.archive_root.resolve()
    source_root = arguments.source_root.resolve()
    derived_dir = archive_root / "02_派生数据"
    figure_dir = archive_root / "04_图表"
    checksum_dir = archive_root / "99_校验"
    for directory in (derived_dir, figure_dir, checksum_dir):
        directory.mkdir(parents=True, exist_ok=True)

    design_metadata = load_design_metadata(
        source_root / "coarse" / "selected_designs.json"
    )
    records, curve_data = extract_case_metrics(
        source_root / "fine" / "paths",
        design_metadata,
    )
    design_rows = summarize_designs(records)
    flexible = [
        record for record in records if record["spring_label"] != "rigid"
    ]
    factor_rows: list[dict[str, Any]] = []
    factor_rows += group_aggregate(
        records, ("spring_label",), "all_by_spring"
    )
    for field in (
        "angle_label",
        "array_shape",
        "spacing_mm",
        "diameter_mm",
        "tip_radius_um",
        "terrain_stratum",
        "preload_N",
    ):
        factor_rows += group_aggregate(
            records,
            (field,),
            f"all_by_{field}",
        )
        factor_rows += group_aggregate(
            flexible,
            (field,),
            f"flexible_only_by_{field}",
        )
    factor_rows += group_aggregate(
        records,
        ("spring_label", "angle_label"),
        "spring_by_angle",
    )
    terrain_preload_rows = group_aggregate(
        records,
        ("terrain_stratum", "preload_N", "spring_label"),
        "terrain_preload_spring",
    )
    curve_rows = build_curve_rows(curve_data)

    write_parquet(derived_dir / "case_bearing_metrics.parquet", records)
    write_csv(derived_dir / "design_bearing_metrics.csv", design_rows)
    write_parquet(
        derived_dir / "design_bearing_metrics.parquet", design_rows
    )
    write_csv(derived_dir / "factor_bearing_metrics.csv", factor_rows)
    write_csv(
        derived_dir / "terrain_preload_bearing_metrics.csv",
        terrain_preload_rows,
    )
    write_csv(
        derived_dir / "force_distance_curves_by_spring_preload.csv",
        curve_rows,
    )
    write_data_dictionary(
        archive_root / "00_说明与索引" / "派生指标字段说明.md"
    )

    plot_spring_force_curves(
        curve_rows,
        figure_dir / "图1_弹簧类型_归一化反力距离曲线.png",
    )
    plot_spring_force_curves_absolute(
        curve_rows,
        figure_dir / "图1b_弹簧类型_绝对反力距离曲线.png",
    )
    plot_spring_metrics(
        factor_rows,
        figure_dir / "图2_弹簧类型_起载距离与累计承载.png",
    )
    plot_design_tradeoff(
        design_rows,
        figure_dir / "图3_构型_搜索距离与承载能力权衡.png",
    )

    summary = {
        "schema_version": "m3-bearing-analysis-v1",
        "source_root": str(source_root),
        "case_count": len(records),
        "design_count": len(design_rows),
        "thresholds_force_over_preload": list(THRESHOLDS),
        "primary_threshold_force_over_preload": 0.25,
        "hold_window_stations": HOLD_WINDOW_STATIONS,
        "hold_required_stations": HOLD_REQUIRED_STATIONS,
        "positive_force_clip_ratio": POSITIVE_FORCE_CLIP_RATIO,
        "global": aggregate_records(records),
    }
    (derived_dir / "analysis_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_sha256_manifest(
        archive_root,
        checksum_dir / "SHA256SUMS.csv",
    )


if __name__ == "__main__":
    main()
