"""Plot a saved dynamic-M2 pull-force history against drag distance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
    with np.load(case_dir / "path.npz", allow_pickle=False) as arrays:
        distance_mm = np.asarray(arrays["path_position_m"], dtype=float) * 1e3
        local_tangential_force_n = np.abs(
            np.asarray(arrays["tangential_force_n"], dtype=float)
        )
        pull_force_n = np.abs(
            np.asarray(
                arrays["spine_on_plate_wrench_about_holder"][:, 0],
                dtype=float,
            )
        )
        normal_force_n = np.asarray(arrays["normal_force_n"], dtype=float)
        contact_state = np.asarray(arrays["contact_state"])
        converged = np.asarray(arrays["numerical_state"]) == "converged"

    parameters = config["parameters"]
    spine = parameters["spine"]
    experiment = parameters["experiment"]
    statistic_mask = (distance_mm > 0.0) & converged & np.isfinite(pull_force_n)
    statistic_values = pull_force_n[statistic_mask]
    median = float(np.median(statistic_values))
    p10 = float(np.quantile(statistic_values, 0.10))
    p25 = float(np.quantile(statistic_values, 0.25))
    preload_n = float(experiment["constant_preload_n"])
    nominal = float(spine["kinetic_friction"]) * preload_n

    figure, axis = plt.subplots(figsize=(9.2, 4.9), constrained_layout=True)
    axis.plot(
        distance_mm,
        pull_force_n,
        color="#1677ff",
        linewidth=0.85,
        alpha=0.9,
        label=r"Global pull force $|F_x^{plate}|$",
    )
    axis.plot(
        distance_mm,
        local_tangential_force_n,
        color="#00a6a6",
        linewidth=0.65,
        linestyle=":",
        alpha=0.65,
        label=r"Local contact component $|F_t|$",
    )
    axis.axhspan(
        p10,
        p25,
        color="#16a085",
        alpha=0.14,
        label=f"P10–P25: {p10:.3f}–{p25:.3f} N",
    )
    axis.axhline(
        median,
        color="#00897b",
        linewidth=1.5,
        linestyle="--",
        label=f"Median: {median:.3f} N",
    )
    axis.axhline(
        nominal,
        color="#ef6c00",
        linewidth=1.2,
        linestyle=":",
        label=rf"Flat-surface $\mu_k W$: {nominal:.3f} N",
    )
    detached = contact_state == "detached_free"
    if np.any(detached):
        axis.scatter(
            distance_mm[detached],
            pull_force_n[detached],
            s=8,
            color="#c62828",
            alpha=0.75,
            label="Detached / free flight",
            zorder=3,
        )

    seed = parameters.get("formal_screening", {}).get("seed", "unknown")
    stiffness = spine["spring_stiffness_n_m"]
    stiffness_text = (
        "rigid" if stiffness is None else f"{float(stiffness):g} N/m"
    )
    axis.set_title(f"M2 Global Pull Force vs. Drag Distance — seed {seed}", fontsize=12)
    axis.text(
        0.012,
        0.965,
        (
            f"Continuous W = {preload_n:g} N   "
            f"R = {float(spine['tip_radius_m']) * 1e6:g} μm   "
            f"d = {float(spine['diameter_m']) * 1e3:g} mm   "
            f"k = {stiffness_text}   "
            f"α = {float(spine['installation_angle_deg']):g}°"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )
    axis.set_xlabel("Drag distance (mm)")
    axis.set_ylabel(r"Pull force along drag direction $|F_x^{plate}|$ (N)")
    axis.set_xlim(0.0, max(float(distance_mm[-1]), 1e-9))
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="#d7dde5", linewidth=0.55, alpha=0.8)
    axis.legend(loc="upper right", frameon=True, fontsize=8.5)
    for side in axis.spines.values():
        side.set_color("#6b7280")
        side.set_linewidth(0.8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(
        f"terminal={summary['termination_reason']} "
        f"normal_force_range=[{np.nanmin(normal_force_n):.6g}, "
        f"{np.nanmax(normal_force_n):.6g}] N"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
