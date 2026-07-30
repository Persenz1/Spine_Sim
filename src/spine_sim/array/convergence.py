"""Small deterministic convergence plans and proxy-trend acceptance metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from spine_sim.core.identity import identity

from .design import TOTAL_PRELOADS_N, build_full_array_design
from .models import M3_MODULE_VERSION


@dataclass(frozen=True)
class ConvergenceVariant:
    name: str
    axis: str
    time_step_s: float = 0.5e-3
    drag_speed_m_s: float = 1e-3
    projection_iterations: int = 40
    position_correction: float = 0.20
    settlement_damping_scale: float = 10.0

    @property
    def variant_id(self) -> str:
        return identity(
            "m3_convergence_variant",
            asdict(self),
            module_version=M3_MODULE_VERSION,
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "variant_id": self.variant_id}


@dataclass(frozen=True)
class TrendConvergenceThresholds:
    median_pull_relative: float = 0.03
    p10_pull_relative: float = 0.05
    neff_relative: float = 0.05
    maximum_pin_load_relative: float = 0.05
    contact_fraction_absolute: float = 0.02
    cumulative_relative_energy_error: float = 1e-3
    contact_work_identity_residual_j: float = 1e-12


def build_convergence_variants() -> tuple[ConvergenceVariant, ...]:
    """One-axis variants around a strict 0.25 ms reference."""

    variants = (
        ConvergenceVariant(
            "reference", "reference", time_step_s=0.25e-3
        ),
        ConvergenceVariant(
            "dt_0p5ms", "time_step", time_step_s=0.5e-3
        ),
        ConvergenceVariant("dt_1ms", "time_step", time_step_s=1e-3),
        ConvergenceVariant("dt_2ms", "time_step", time_step_s=2e-3),
        ConvergenceVariant("dt_5ms", "time_step", time_step_s=5e-3),
        ConvergenceVariant(
            "projection_20",
            "contact_projection",
            projection_iterations=20,
        ),
        ConvergenceVariant(
            "position_correction_0p5",
            "contact_stabilization",
            position_correction=0.50,
        ),
        ConvergenceVariant(
            "position_correction_1p0",
            "contact_stabilization",
            position_correction=1.0,
        ),
        ConvergenceVariant(
            "settlement_damping_5",
            "settlement_damping",
            settlement_damping_scale=5.0,
        ),
        ConvergenceVariant(
            "settlement_damping_20",
            "settlement_damping",
            settlement_damping_scale=20.0,
        ),
        ConvergenceVariant(
            "drag_speed_2mm_s",
            "drag_rate",
            drag_speed_m_s=2e-3,
        ),
        ConvergenceVariant(
            "drag_speed_5mm_s",
            "drag_rate",
            drag_speed_m_s=5e-3,
        ),
    )
    if len({variant.variant_id for variant in variants}) != len(variants):
        raise AssertionError("M3 convergence variant IDs must be unique")
    return variants


_SENTINEL_SPECS = (
    # Feasibility anchor: verified on the formal P40 condition with the
    # deterministic placement search. More extreme geometries remain below,
    # but an initially colliding configuration cannot establish trend
    # convergence.
    (2, 2, 4e-3, "fixed", 70.0, 100e-6, 0.8e-3, "spring", 800.0),
    (2, 2, 6e-3, "fixed", 80.0, 100e-6, 0.8e-3, "rigid", None),
    (4, 4, 5e-3, "fixed", 70.0, 100e-6, 0.8e-3, "spring", 800.0),
    (
        4,
        4,
        5e-3,
        "gradient_80_to_60",
        None,
        100e-6,
        0.8e-3,
        "spring",
        2000.0,
    ),
    (5, 2, 4e-3, "fixed", 60.0, 100e-6, 0.8e-3, "spring", 2000.0),
    (2, 5, 6e-3, "fixed", 80.0, 50e-6, 0.6e-3, "spring", 800.0),
    (6, 6, 4e-3, "fixed", 70.0, 100e-6, 0.8e-3, "spring", 300.0),
    (
        6,
        6,
        6e-3,
        "gradient_80_to_60",
        None,
        50e-6,
        0.6e-3,
        "rigid",
        None,
    ),
)


def build_convergence_sentinels() -> tuple[dict[str, Any], ...]:
    design = build_full_array_design()
    selected: list[dict[str, Any]] = []
    for spec in _SENTINEL_SPECS:
        (
            nx,
            ny,
            spacing,
            layout,
            angle,
            radius,
            diameter,
            axial_mode,
            stiffness,
        ) = spec
        matches = [
            row
            for row in design
            if row["nx"] == nx
            and row["ny"] == ny
            and math.isclose(row["spacing_m"], spacing)
            and row["angle_layout"] == layout
            and row["fixed_angle_deg"] == angle
            and math.isclose(row["tip_radius_m"], radius)
            and math.isclose(row["diameter_m"], diameter)
            and row["axial_mode"] == axial_mode
            and row["spring_stiffness_n_m"] == stiffness
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"expected one convergence sentinel for {spec}, got {len(matches)}"
            )
        selected.append(matches[0])
    return tuple(selected)


def convergence_plan_manifest() -> dict[str, Any]:
    variants = build_convergence_variants()
    sentinels = build_convergence_sentinels()
    return {
        "schema_version": "1",
        "m3_module_version": M3_MODULE_VERSION,
        "purpose": (
            "short-path numerical/rate sensitivity before any 100 mm campaign"
        ),
        "sentinel_count": len(sentinels),
        "variant_count": len(variants),
        "preloads_n": list(TOTAL_PRELOADS_N),
        "case_count_per_terrain_condition": (
            len(sentinels) * len(variants) * len(TOTAL_PRELOADS_N)
        ),
        "recommended_short_path_m": 2e-3,
        "variants": [variant.as_dict() for variant in variants],
        "sentinels": [
            {
                key: value
                for key, value in row.items()
                if key != "configuration"
            }
            for row in sentinels
        ],
        "thresholds": asdict(TrendConvergenceThresholds()),
        "formal_campaign_started": False,
    }


def _relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-12)


def compare_trend_summaries(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    thresholds: TrendConvergenceThresholds | None = None,
) -> dict[str, Any]:
    """Compare two paired summaries without treating failed cases as zero."""

    limits = thresholds or TrendConvergenceThresholds()
    required = (
        "tangential_force_median_n",
        "tangential_force_p10_n",
        "neff_resultant_median",
        "maximum_pin_normal_force_n",
        "contact_fraction",
        "steady_sample_count",
        "cumulative_relative_energy_error",
        "maximum_abs_contact_work_identity_residual_j",
    )
    missing = [
        name
        for name in required
        if reference.get(name) is None or candidate.get(name) is None
    ]
    if missing:
        return {
            "passed": False,
            "failure_reason": "missing_conditional_performance",
            "missing_fields": missing,
        }
    differences = {
        "median_pull_relative": _relative_difference(
            float(candidate["tangential_force_median_n"]),
            float(reference["tangential_force_median_n"]),
        ),
        "p10_pull_relative": _relative_difference(
            float(candidate["tangential_force_p10_n"]),
            float(reference["tangential_force_p10_n"]),
        ),
        "neff_relative": _relative_difference(
            float(candidate["neff_resultant_median"]),
            float(reference["neff_resultant_median"]),
        ),
        "maximum_pin_load_relative": _relative_difference(
            float(candidate["maximum_pin_normal_force_n"]),
            float(reference["maximum_pin_normal_force_n"]),
        ),
        "contact_fraction_absolute": abs(
            float(candidate["contact_fraction"])
            - float(reference["contact_fraction"])
        ),
    }
    numerical_checks = {
        "same_terminal_state": (
            candidate.get("run_terminal_state")
            == reference.get("run_terminal_state")
            == "path_end"
        ),
        "initialization_success": (
            candidate.get("initial_preload_success") is True
            and reference.get("initial_preload_success") is True
        ),
        "conditional_performance_available": (
            candidate.get(
                "conditional_performance_available", True
            )
            is True
            and reference.get(
                "conditional_performance_available", True
            )
            is True
        ),
        "ranking_inclusion_allowed": (
            candidate.get("ranking_inclusion_allowed", True) is True
            and reference.get("ranking_inclusion_allowed", True) is True
        ),
        "rod_clearance_ok": (
            candidate.get("rod_collision_detected") is not True
            and reference.get("rod_collision_detected") is not True
        ),
        "same_selected_origin": (
            candidate.get("selected_unit_origin_xy_m")
            == reference.get("selected_unit_origin_xy_m")
        ),
        "sufficient_steady_samples": (
            int(candidate["steady_sample_count"]) >= 20
            and int(reference["steady_sample_count"]) >= 20
        ),
        "cumulative_energy_error_ok": (
            float(candidate["cumulative_relative_energy_error"])
            <= limits.cumulative_relative_energy_error
        ),
        "contact_work_identity_ok": (
            float(
                candidate[
                    "maximum_abs_contact_work_identity_residual_j"
                ]
            )
            <= limits.contact_work_identity_residual_j
        ),
    }
    difference_checks = {
        name: value <= getattr(limits, name)
        for name, value in differences.items()
    }
    numerical_passed = all(numerical_checks.values())
    differences_passed = all(difference_checks.values())
    return {
        "passed": differences_passed and numerical_passed,
        "failure_reason": (
            None
            if numerical_passed and differences_passed
            else (
                "ineligible_or_numerically_invalid"
                if not numerical_passed
                else "trend_threshold_exceeded"
            )
        ),
        "differences": differences,
        "difference_checks": difference_checks,
        "numerical_checks": numerical_checks,
        "thresholds": asdict(limits),
    }
