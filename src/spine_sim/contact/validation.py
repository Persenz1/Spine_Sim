"""Dynamic constant-preload M2 analytic gates and M1 smoke validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.io.results import atomic_write_json, utc_now
from spine_sim.terrain import (
    RegionSpec,
    TerrainLibrary,
    compute_track_geometry,
    evaluate_analytic,
)

from .dynamics import (
    DynamicContactSettings,
    DynamicExperimentSettings,
    DynamicIntegratorSettings,
    DynamicSingleSpineExperiment,
    DynamicSingleSpineUnit,
)
from .models import AxialMode, M2_MODULE_VERSION, SpineParameters


def _fixture_track(
    kind: str,
    parameters: Mapping[str, float] | None = None,
    *,
    radius_m: float = 50e-6,
    dx_m: float = 10e-6,
):
    region = RegionSpec(
        terrain_recipe_id=f"terrain_recipe_m2_{kind}",
        origin_x_m=-0.003,
        origin_y_m=-0.0004,
        size_x_m=0.008,
        size_y_m=0.0008,
        resolution_x_m=dx_m,
        resolution_y_m=dx_m,
        purpose="module",
    )
    x = region.origin_x_m + np.arange(region.shape[1]) * dx_m
    y = region.origin_y_m + np.arange(region.shape[0]) * dx_m
    height = evaluate_analytic(kind, x, y, parameters)
    return compute_track_geometry(
        height,
        region,
        radius_m=radius_m,
        y_global_m=0.0,
    )


def _fixture_parameters(**overrides: Any) -> SpineParameters:
    values: dict[str, Any] = {
        "tip_radius_m": 50e-6,
        "diameter_m": 0.8e-3,
        "exposed_length_m": 4e-3,
        "installation_angle_deg": 70.0,
        "axial_mode": AxialMode.SPRING,
        "spring_stiffness_n_m": 2000.0,
        "spring_travel_m": 4e-3,
        "static_friction": 0.30,
        "kinetic_friction": 0.20,
        "rod_clearance_mode": "disabled_analytic_fixture",
        "density_kg_m3": 7850.0,
        "axial_modal_mass_factor": 1.0 / 3.0,
        "transverse_modal_mass_factor": 0.236,
        "axial_damping_ratio": 0.05,
        "transverse_damping_ratio": 0.05,
        "yield_strength_pa": 1.0e9,
    }
    values.update(overrides)
    return SpineParameters(**values)


def _fixture_experiment(
    *,
    initial_x_m: float,
    drag_length_m: float,
    preload_n: float = 0.5,
    output_spacing_m: float = 10e-6,
) -> DynamicExperimentSettings:
    return DynamicExperimentSettings(
        initial_center_x_m=initial_x_m,
        drag_length_m=drag_length_m,
        drag_speed_m_s=1e-3,
        constant_preload_n=preload_n,
        holder_effective_mass_kg=0.05,
        holder_vertical_damping_n_s_m=1.0,
        maximum_preload_approach_m=8e-3,
        output_spacing_m=output_spacing_m,
        effective_normal_force_min_n=0.05,
    )


def _run_fixture(
    kind: str,
    terrain_parameters: Mapping[str, float] | None,
    *,
    initial_x_m: float,
    drag_length_m: float,
    time_step_s: float = 1e-3,
    preload_n: float = 0.5,
    spine_parameters: SpineParameters | None = None,
):
    track = _fixture_track(kind, terrain_parameters)
    parameters = spine_parameters or _fixture_parameters()
    return DynamicSingleSpineExperiment(
        parameters,
        track,
        _fixture_experiment(
            initial_x_m=initial_x_m,
            drag_length_m=drag_length_m,
            preload_n=preload_n,
        ),
        DynamicContactSettings(
            normal_model="rigid_moreau",
            restitution_coefficient=0.0,
        ),
        DynamicIntegratorSettings(time_step_s=time_step_s),
    ).run()


def _gate(
    name: str,
    passed: bool,
    evidence: Mapping[str, Any],
    *,
    requirement: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "requirement": requirement,
        "evidence": dict(evidence),
    }


def run_analytic_validation(
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run dynamic constant-preload M2 acceptance gates."""

    gates: list[dict[str, Any]] = []
    plane = _run_fixture(
        "plane",
        None,
        initial_x_m=-1e-3,
        drag_length_m=0.5e-3,
    )
    plane_normal = np.asarray(
        [point.normal_force_n for point in plane.points[5:]],
        dtype=np.float64,
    )
    plane_pull = np.abs(
        np.asarray(
            [
                point.spine_on_plate_wrench_about_holder[0]
                for point in plane.points[5:]
            ],
            dtype=np.float64,
        )
    )
    normal_mean = float(np.mean(plane_normal))
    pull_median = float(np.median(plane_pull))
    expected_pull = (
        plane.parameters.kinetic_friction
        * plane.experiment.constant_preload_n
    )
    gates.append(
        _gate(
            "plane_continuous_preload",
            plane.summary.run_terminal_state.value == "path_end"
            and abs(normal_mean - plane.experiment.constant_preload_n) < 0.02
            and plane.summary.event_counts["detach_to_free"] == 0,
            {
                "normal_force_mean_n": normal_mean,
                "external_preload_n": plane.experiment.constant_preload_n,
                "events": dict(plane.summary.event_counts),
            },
            requirement=(
                "flat-plane time-average reaction balances continuous "
                "external preload without fixed Z"
            ),
        )
    )
    gates.append(
        _gate(
            "plane_coulomb_drag",
            abs(pull_median - expected_pull) < 0.01,
            {
                "pull_force_median_n": pull_median,
                "expected_muW_n": expected_pull,
            },
            requirement="steady global drag force approaches kinetic mu*W",
        )
    )

    zero_load = _run_fixture(
        "plane",
        None,
        initial_x_m=-1e-3,
        drag_length_m=0.2e-3,
        preload_n=0.0,
    )
    gates.append(
        _gate(
            "zero_preload_no_false_force",
            zero_load.summary.global_pull_force_peak_n < 1e-10
            and max(
                (point.normal_force_n for point in zero_load.points),
                default=0.0,
            )
            < 1e-10,
            {
                "normal_peak_n": max(
                    (point.normal_force_n for point in zero_load.points),
                    default=0.0,
                ),
                "pull_peak_n": zero_load.summary.global_pull_force_peak_n,
            },
            requirement="zero external preload does not fabricate contact force",
        )
    )

    slope = _run_fixture(
        "slope",
        {"slope_x": 0.10},
        initial_x_m=-1e-3,
        drag_length_m=0.3e-3,
    )
    gates.append(
        _gate(
            "slope_force_and_work_direction",
            slope.summary.run_terminal_state.value == "path_end"
            and all(
                point.drive_work_increment_j >= -1e-8
                for point in slope.points[5:]
            ),
            {
                "terminal": slope.summary.run_terminal_state.value,
                "minimum_drive_work_increment_j": min(
                    (
                        point.drive_work_increment_j
                        for point in slope.points[5:]
                    ),
                    default=0.0,
                ),
            },
            requirement="slope force and prescribed-drive work use global signs",
        )
    )

    bump_parameters = {
        "amplitude_m": 400e-6,
        "center_x_m": -1e-3,
        "sigma_x_m": 150e-6,
        "sigma_y_m": 150e-6,
    }
    bump = _run_fixture(
        "smooth_bump",
        bump_parameters,
        initial_x_m=-1e-3,
        drag_length_m=1e-3,
    )
    gates.append(
        _gate(
            "drop_recontacts_under_continuous_load",
            bump.summary.run_terminal_state.value == "path_end"
            and bump.summary.event_counts["detach_to_free"] >= 1
            and bump.summary.event_counts["recontact"] >= 1
            and bump.summary.event_counts["impact"] >= 1,
            {
                "events": dict(bump.summary.event_counts),
                "impact_velocity_peak_m_s": (
                    bump.summary.impact_velocity_peak_m_s
                ),
            },
            requirement=(
                "loss of contact continues dynamically to impact/recontact"
            ),
        )
    )

    double_parameters = {
        "amplitude_1_m": 400e-6,
        "amplitude_2_m": 400e-6,
        "center_1_x_m": -1e-3,
        "center_2_x_m": 0.5e-3,
        "sigma_x_m": 150e-6,
        "sigma_y_m": 150e-6,
    }
    double = _run_fixture(
        "double_bump",
        double_parameters,
        initial_x_m=-1e-3,
        drag_length_m=2e-3,
    )
    gates.append(
        _gate(
            "double_bump_dynamic_path",
            double.summary.run_terminal_state.value == "path_end"
            and double.summary.event_counts["detach_to_free"] >= 1
            and double.summary.event_counts["recontact"] >= 1
            and "no_admissible_contact_equilibrium"
            not in double.summary.termination_reason,
            {
                "terminal": double.summary.run_terminal_state.value,
                "events": dict(double.summary.event_counts),
                "reason": double.summary.termination_reason,
            },
            requirement=(
                "ordinary branch loss cannot terminate as model_unclosed"
            ),
        )
    )

    unit = DynamicSingleSpineUnit(
        _fixture_parameters(),
        _fixture_track("plane"),
        DynamicContactSettings(),
    )
    lower = unit.axial_response(-1e-6)
    interior = unit.axial_response(0.2e-3)
    hard = unit.axial_response(5e-3)
    gates.append(
        _gate(
            "spring_three_segments",
            lower[3].value == "lower_stop"
            and interior[3].value == "interior"
            and hard[3].value == "hard_stop"
            and lower[0] < 0.0 < interior[0] < hard[0],
            {
                "lower": lower[3].value,
                "interior": interior[3].value,
                "hard": hard[3].value,
            },
            requirement="dynamic axial restoring law preserves all spring segments",
        )
    )

    p06 = _fixture_parameters(diameter_m=0.6e-3)
    p08 = _fixture_parameters(diameter_m=0.8e-3)
    compliance_ratio = (
        p06.transverse_compliance_m_n
        / p08.transverse_compliance_m_n
    )
    frequency06 = math.sqrt(
        (1.0 / p06.transverse_compliance_m_n)
        / p06.transverse_modal_mass_kg
    )
    frequency08 = math.sqrt(
        (1.0 / p08.transverse_compliance_m_n)
        / p08.transverse_modal_mass_kg
    )
    gates.append(
        _gate(
            "diameter_static_and_dynamic_trend",
            2.8 < compliance_ratio < 3.3
            and frequency06 < frequency08,
            {
                "compliance_ratio_06_over_08": compliance_ratio,
                "omega_06_rad_s": frequency06,
                "omega_08_rad_s": frequency08,
            },
            requirement=(
                "diameter changes both static compliance and dynamic frequency"
            ),
        )
    )

    plane_half = _run_fixture(
        "plane",
        None,
        initial_x_m=-1e-3,
        drag_length_m=0.5e-3,
        time_step_s=0.5e-3,
    )
    median_delta = abs(
        plane.summary.global_pull_force_median_n
        - plane_half.summary.global_pull_force_median_n
    )
    normal_delta = abs(
        np.mean(
            [point.normal_force_n for point in plane.points[5:]]
        )
        - np.mean(
            [point.normal_force_n for point in plane_half.points[5:]]
        )
    )
    gates.append(
        _gate(
            "time_step_halving",
            median_delta < 0.01 and normal_delta < 0.02,
            {
                "pull_median_delta_n": median_delta,
                "normal_mean_delta_n": normal_delta,
            },
            requirement=(
                "halving the internal time step preserves robust force metrics"
            ),
        )
    )

    gates.append(
        _gate(
            "energy_and_dynamic_residual",
            plane.summary.maximum_abs_dynamic_residual_n < 1e-8
            and plane.summary.maximum_abs_energy_residual_j < 1e-6,
            {
                "dynamic_residual_n": (
                    plane.summary.maximum_abs_dynamic_residual_n
                ),
                "energy_residual_j": (
                    plane.summary.maximum_abs_energy_residual_j
                ),
            },
            requirement="flat dynamic balance and incremental energy audit close",
        )
    )

    replay = _run_fixture(
        "plane",
        None,
        initial_x_m=-1e-3,
        drag_length_m=0.5e-3,
    )
    deterministic = (
        len(replay.points) == len(plane.points)
        and np.array_equal(
            np.asarray(
                [
                    point.spine_on_plate_wrench_about_holder
                    for point in replay.points
                ]
            ),
            np.asarray(
                [
                    point.spine_on_plate_wrench_about_holder
                    for point in plane.points
                ]
            ),
        )
    )
    gates.append(
        _gate(
            "deterministic_replay",
            deterministic,
            {"point_count": len(plane.points)},
            requirement="identical configuration and initial state replay exactly",
        )
    )

    gates.append(
        _gate(
            "formal_ranking_blocked_until_physical_calibration",
            not plane.summary.formal_ranking_eligible,
            {
                "formal_ranking_eligible": (
                    plane.summary.formal_ranking_eligible
                ),
                "time_step_convergence_checked": (
                    plane.summary.time_step_convergence_checked
                ),
                "contact_parameter_convergence_checked": (
                    plane.summary.contact_parameter_convergence_checked
                ),
            },
            requirement=(
                "a single successful run cannot bypass convergence/calibration"
            ),
        )
    )

    report = {
        "schema_version": "2",
        "m2_module_version": M2_MODULE_VERSION,
        "created_at_utc": utc_now(),
        "gate_count": len(gates),
        "passed_count": sum(gate["passed"] for gate in gates),
        "all_passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "preload_mode": "continuous_external_force",
        "formal_random_screening_allowed": False,
        "formal_random_screening_blockers": [
            "holder/backplate effective mass is not hardware-frozen",
            "dynamic contact/restitution parameters are not material-calibrated",
            "yield, buckling and rod clearance are not fully closed",
            "production time-step convergence pairs have not been run",
        ],
    }
    if output_path is not None:
        atomic_write_json(Path(output_path), report)
    return report


def run_m1_suite_smoke(
    suite_report_path: str | Path,
    *,
    drag_length_m: float = 0.2e-3,
    path_step_m: float = 50e-6,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Exercise the dynamic M2 contract on all saved M1 conditions."""

    report_path = Path(suite_report_path)
    source = json.loads(report_path.read_text(encoding="utf-8"))
    library_root = report_path.parent / "terrain_library"
    library = TerrainLibrary(library_root)
    conditions: list[dict[str, Any]] = []
    for condition in source["conditions"]:
        recipe_id = condition["terrain_recipe_id"]
        region_id = condition["region_id"]
        metadata_candidates = list(
            (
                library_root / "tracks" / recipe_id / region_id / "50um"
            ).glob("*.json")
        )
        zero_y_candidates = []
        for candidate in metadata_candidates:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
            if math.isclose(
                float(metadata["y_global_m"]),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                zero_y_candidates.append(metadata)
        if len(zero_y_candidates) != 1:
            conditions.append(
                {
                    "name": condition["name"],
                    "passed": False,
                    "reason": "expected exactly one saved y=0, 50um track",
                }
            )
            continue
        metadata = zero_y_candidates[0]
        track = library.load_track(
            recipe_id,
            region_id,
            float(metadata["radius_m"]),
            metadata["track_id"],
        )
        valid_indices = np.flatnonzero(track.valid_mask)
        parameters = _fixture_parameters(
            rod_clearance_mode="unclosed",
            yield_strength_pa=None,
        )
        result = None
        initial_x = math.nan
        attempts = []
        for fraction in (0.50, 0.60, 0.40, 0.70, 0.30):
            center_index = valid_indices[
                int(round(fraction * (len(valid_indices) - 1)))
            ]
            candidate_x = float(track.x_global_m[center_index])
            candidate = DynamicSingleSpineExperiment(
                parameters,
                track,
                _fixture_experiment(
                    initial_x_m=candidate_x,
                    drag_length_m=drag_length_m,
                    output_spacing_m=min(path_step_m, 10e-6),
                ),
                DynamicContactSettings(),
                DynamicIntegratorSettings(time_step_s=1e-3),
            ).run()
            attempts.append(
                {
                    "valid_track_fraction": fraction,
                    "initial_center_x_m": candidate_x,
                    "initial_preload_success": (
                        candidate.summary.initial_preload_success
                    ),
                    "terminal": (
                        candidate.summary.run_terminal_state.value
                    ),
                }
            )
            result = candidate
            initial_x = candidate_x
            if (
                candidate.summary.initial_preload_success
                and candidate.summary.run_terminal_state.value == "path_end"
            ):
                break
        assert result is not None
        passed = (
            result.summary.initial_preload_success
            and result.summary.run_terminal_state.value == "path_end"
            and result.summary.numerical_state.value == "converged"
        )
        conditions.append(
            {
                "name": condition["name"],
                "description": condition["description"],
                "terrain_recipe_id": recipe_id,
                "region_id": region_id,
                "track_id": track.track_id,
                "initial_center_x_m": initial_x,
                "attempts": attempts,
                "passed": passed,
                "preload_mode": result.summary.preload_mode,
                "event_counts": dict(result.summary.event_counts),
                "contact_fraction": result.summary.contact_fraction,
                "pull_force_median_n": (
                    result.summary.global_pull_force_median_n
                ),
                "run_terminal_state": (
                    result.summary.run_terminal_state.value
                ),
                "termination_reason": result.summary.termination_reason,
                "model_state": result.summary.model_state.value,
                "formal_ranking_eligible": False,
            }
        )
    output = {
        "schema_version": "2",
        "m2_module_version": M2_MODULE_VERSION,
        "created_at_utc": utc_now(),
        "source_suite": str(report_path),
        "condition_count": len(conditions),
        "passed_count": sum(item["passed"] for item in conditions),
        "all_passed": all(item["passed"] for item in conditions),
        "purpose": "dynamic_interface_and_state_smoke_only",
        "formal_ranking_allowed": False,
        "conditions": conditions,
    }
    if output_path is not None:
        atomic_write_json(Path(output_path), output)
    return output
