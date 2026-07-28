"""M0 runner adapter and result serialization for M2 cases."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.runtime.runner import CaseOutput, RunContext
from spine_sim.terrain import TerrainLibrary

from .dynamics import (
    DynamicContactSettings,
    DynamicExperimentSettings,
    DynamicIntegratorSettings,
    DynamicSingleSpineExperiment,
    DynamicSingleSpineResult,
)
from .models import M2_MODULE_VERSION, SpineParameters


def _summary_dict(result) -> dict[str, Any]:
    if isinstance(result, DynamicSingleSpineResult):
        summary = asdict(result.summary)
        summary["numerical_state"] = result.summary.numerical_state.value
        summary["model_state"] = result.summary.model_state.value
        summary["run_terminal_state"] = (
            result.summary.run_terminal_state.value
        )
        summary["parameters"] = result.parameters.as_dict()
        summary["experiment"] = asdict(result.experiment)
        summary["dynamic_contact"] = asdict(result.contact)
        summary["dynamic_integrator"] = asdict(result.integrator)
        summary["track_id"] = result.track_id
        summary["assumptions"] = list(result.assumptions)
        summary["m2_module_version"] = M2_MODULE_VERSION
        return summary
    summary = asdict(result.summary)
    summary["numerical_state"] = result.summary.numerical_state.value
    summary["model_state"] = result.summary.model_state.value
    summary["run_terminal_state"] = result.summary.run_terminal_state.value
    summary["parameters"] = result.parameters.as_dict()
    summary["track_id"] = result.track_id
    summary["fixed_holder_z_m"] = result.fixed_holder_z_m
    summary["assumptions"] = list(result.assumptions)
    summary["m2_module_version"] = M2_MODULE_VERSION
    return summary


def _legacy_arrays(result) -> dict[str, np.ndarray]:
    points = result.points

    def numeric(getter, *, width: int | None = None):
        values = [getter(point) for point in points]
        if width is not None and not values:
            return np.empty((0, width), dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    return {
        "path_position_m": numeric(lambda point: point.path_position_m),
        "holder_xz_m": numeric(lambda point: point.response.holder_xz_m, width=2),
        "center_xz_m": numeric(lambda point: point.response.center_xz_m, width=2),
        "support_xyz_m": numeric(
            lambda point: (
                point.response.support_xyz_m
                if point.response.support_xyz_m is not None
                else (np.nan, np.nan, np.nan)
            ),
            width=3,
        ),
        "gap_m": numeric(lambda point: point.response.gap_m),
        "tangent_xz": numeric(
            lambda point: point.response.tangent_xz or (np.nan, np.nan),
            width=2,
        ),
        "normal_xz": numeric(
            lambda point: point.response.normal_xz or (np.nan, np.nan),
            width=2,
        ),
        "cap_gate_passed": np.asarray(
            [point.response.cap_gate_passed for point in points], dtype=np.bool_
        ),
        "near_tie": np.asarray(
            [point.response.near_tie for point in points], dtype=np.bool_
        ),
        "event_refined": np.asarray(
            [point.event_refined for point in points], dtype=np.bool_
        ),
        "contact_state": np.asarray(
            [point.response.contact_state.value for point in points], dtype="U32"
        ),
        "spring_state": np.asarray(
            [point.response.spring_state.value for point in points], dtype="U16"
        ),
        "event_label": np.asarray(
            [point.response.event_label.value for point in points], dtype="U24"
        ),
        "wall_on_spine_force_xz_n": numeric(
            lambda point: point.response.wall_on_spine_force_xz_n,
            width=2,
        ),
        "spine_on_plate_wrench_about_holder": numeric(
            lambda point: point.response.spine_on_plate_wrench_about_holder,
            width=6,
        ),
        "normal_force_n": numeric(lambda point: point.response.normal_force_n),
        "tangential_force_n": numeric(
            lambda point: point.response.tangential_force_n
        ),
        "axial_force_n": numeric(lambda point: point.response.axial_force_n),
        "transverse_force_n": numeric(
            lambda point: point.response.transverse_force_n
        ),
        "spring_compression_m": numeric(
            lambda point: point.response.spring_compression_m
        ),
        "beam_displacement_xz_m": numeric(
            lambda point: point.response.beam_displacement_xz_m,
            width=2,
        ),
        "static_friction_margin_n": numeric(
            lambda point: point.response.static_friction_margin_n
        ),
        "spring_travel_margin_m": numeric(
            lambda point: point.response.spring_travel_margin_m
        ),
        "elastic_energy_j": numeric(
            lambda point: point.response.elastic_energy_j
        ),
        "holder_work_increment_j": numeric(
            lambda point: point.response.holder_work_increment_j
        ),
        "contact_work_increment_j": numeric(
            lambda point: point.response.contact_work_increment_j
        ),
        "friction_dissipation_increment_j": numeric(
            lambda point: point.response.friction_dissipation_increment_j
        ),
        "energy_residual_j": numeric(
            lambda point: point.response.energy_residual_j
        ),
        "geometry_residual_m": numeric(
            lambda point: point.response.residual.geometry_m
        ),
        "structure_residual_m": numeric(
            lambda point: point.response.residual.structure_m
        ),
        "force_decomposition_residual_n": numeric(
            lambda point: point.response.residual.force_decomposition_n
        ),
        "root_iterations": np.asarray(
            [point.response.residual.root_iterations for point in points],
            dtype=np.int32,
        ),
        "numerical_state": np.asarray(
            [point.response.numerical_state.value for point in points], dtype="U24"
        ),
        "model_state": np.asarray(
            [point.response.model_state.value for point in points], dtype="U24"
        ),
    }


def _dynamic_arrays(result: DynamicSingleSpineResult) -> dict[str, np.ndarray]:
    points = result.points

    def numeric(getter, *, width: int | None = None):
        values = [getter(point) for point in points]
        if width is not None and not values:
            return np.empty((0, width), dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    return {
        "time_s": numeric(lambda point: point.time_s),
        "path_position_m": numeric(lambda point: point.path_position_m),
        "holder_xz_m": numeric(lambda point: point.holder_xz_m, width=2),
        "holder_velocity_xz_m_s": numeric(
            lambda point: point.holder_velocity_xz_m_s,
            width=2,
        ),
        "holder_acceleration_xz_m_s2": numeric(
            lambda point: point.holder_acceleration_xz_m_s2,
            width=2,
        ),
        "center_xz_m": numeric(lambda point: point.center_xz_m, width=2),
        "center_velocity_xz_m_s": numeric(
            lambda point: point.center_velocity_xz_m_s,
            width=2,
        ),
        "center_acceleration_xz_m_s2": numeric(
            lambda point: point.center_acceleration_xz_m_s2,
            width=2,
        ),
        "support_xyz_m": numeric(
            lambda point: (
                point.support_xyz_m
                if point.support_xyz_m is not None
                else (np.nan, np.nan, np.nan)
            ),
            width=3,
        ),
        "tangent_xz": numeric(
            lambda point: point.tangent_xz or (np.nan, np.nan),
            width=2,
        ),
        "normal_xz": numeric(
            lambda point: point.normal_xz or (np.nan, np.nan),
            width=2,
        ),
        "gap_m": numeric(lambda point: point.gap_m),
        "contact_state": np.asarray(
            [point.contact_state.value for point in points],
            dtype="U32",
        ),
        "spring_state": np.asarray(
            [point.spring_state.value for point in points],
            dtype="U16",
        ),
        "event_label": np.asarray(
            [point.event_label.value for point in points],
            dtype="U24",
        ),
        "external_preload_n": numeric(
            lambda point: point.external_preload_n
        ),
        "wall_on_spine_force_xz_n": numeric(
            lambda point: point.wall_on_spine_force_xz_n,
            width=2,
        ),
        "spine_on_plate_wrench_about_holder": numeric(
            lambda point: point.spine_on_plate_wrench_about_holder,
            width=6,
        ),
        "normal_force_n": numeric(lambda point: point.normal_force_n),
        "tangential_force_n": numeric(
            lambda point: point.tangential_force_n
        ),
        "normal_impulse_n_s": numeric(
            lambda point: point.normal_impulse_n_s
        ),
        "tangential_impulse_n_s": numeric(
            lambda point: point.tangential_impulse_n_s
        ),
        "impact_velocity_m_s": numeric(
            lambda point: point.impact_velocity_m_s
        ),
        "axial_displacement_m": numeric(
            lambda point: point.axial_displacement_m
        ),
        "transverse_displacement_m": numeric(
            lambda point: point.transverse_displacement_m
        ),
        "axial_velocity_m_s": numeric(
            lambda point: point.axial_velocity_m_s
        ),
        "transverse_velocity_m_s": numeric(
            lambda point: point.transverse_velocity_m_s
        ),
        "axial_force_n": numeric(lambda point: point.axial_force_n),
        "transverse_force_n": numeric(
            lambda point: point.transverse_force_n
        ),
        "spring_compression_m": numeric(
            lambda point: point.spring_compression_m
        ),
        "spring_travel_margin_m": numeric(
            lambda point: point.spring_travel_margin_m
        ),
        "kinetic_energy_j": numeric(
            lambda point: point.kinetic_energy_j
        ),
        "structural_energy_j": numeric(
            lambda point: point.structural_energy_j
        ),
        "preload_work_increment_j": numeric(
            lambda point: point.preload_work_increment_j
        ),
        "drive_work_increment_j": numeric(
            lambda point: point.drive_work_increment_j
        ),
        "friction_dissipation_increment_j": numeric(
            lambda point: point.friction_dissipation_increment_j
        ),
        "damping_dissipation_increment_j": numeric(
            lambda point: point.damping_dissipation_increment_j
        ),
        "energy_residual_j": numeric(
            lambda point: point.energy_residual_j
        ),
        "dynamic_residual_n": numeric(
            lambda point: point.dynamic_residual_n
        ),
        "actual_time_step_s": numeric(
            lambda point: point.actual_time_step_s
        ),
        "nonlinear_iterations": np.asarray(
            [point.nonlinear_iterations for point in points],
            dtype=np.int32,
        ),
        "bending_stress_pa": numeric(
            lambda point: point.bending_stress_pa
        ),
        "euler_buckling_margin_n": numeric(
            lambda point: point.euler_buckling_margin_n
        ),
        "numerical_state": np.asarray(
            [point.numerical_state.value for point in points],
            dtype="U24",
        ),
        "model_state": np.asarray(
            [point.model_state.value for point in points],
            dtype="U24",
        ),
    }


def _arrays(result) -> dict[str, np.ndarray]:
    if isinstance(result, DynamicSingleSpineResult):
        return _dynamic_arrays(result)
    return _legacy_arrays(result)


def _events(result, case_id: str) -> list[dict[str, Any]]:
    if isinstance(result, DynamicSingleSpineResult):
        events: list[dict[str, Any]] = []
        sequence = 0
        for point in result.points:
            for event_type in point.event_labels:
                events.append(
                    {
                        "sequence": sequence,
                        "case_id": case_id,
                        "event_type": event_type,
                        "path_position_m": point.path_position_m,
                        "details": {
                            "time_s": point.time_s,
                            "contact_state": point.contact_state.value,
                            "spring_state": point.spring_state.value,
                            "normal_force_n": point.normal_force_n,
                            "normal_impulse_n_s": point.normal_impulse_n_s,
                            "impact_velocity_m_s": point.impact_velocity_m_s,
                        },
                    }
                )
                sequence += 1
        return events
    events: list[dict[str, Any]] = []
    sequence = 0
    for point in result.points:
        if point.response.event_label.value == "none":
            continue
        events.append(
            {
                "sequence": sequence,
                "case_id": case_id,
                "event_type": point.response.event_label.value,
                "path_position_m": point.path_position_m,
                "details": {
                    "contact_state": point.response.contact_state.value,
                    "spring_state": point.response.spring_state.value,
                    "normal_force_n": point.response.normal_force_n,
                    "event_refined": point.event_refined,
                },
            }
        )
        sequence += 1
    return events


def run_case(parameters: Mapping[str, Any], context: RunContext) -> CaseOutput:
    """Run one saved-track M2 case through M0's campaign runner."""

    started = time.perf_counter()
    required = {
        "terrain_library_root",
        "terrain_recipe_id",
        "region_id",
        "track_id",
        "radius_m",
        "spine",
        "experiment",
    }
    missing = required - set(parameters)
    if missing:
        raise ValueError(f"M2 case is missing fields: {sorted(missing)}")
    library = TerrainLibrary(Path(str(parameters["terrain_library_root"])))
    track = library.load_track(
        str(parameters["terrain_recipe_id"]),
        str(parameters["region_id"]),
        float(parameters["radius_m"]),
        str(parameters["track_id"]),
    )
    spine = SpineParameters.from_mapping(parameters["spine"])
    if spine.rod_clearance_mode == "disabled_analytic_fixture":
        raise ValueError(
            "saved project-terrain cases cannot disable rod clearance as an analytic fixture"
        )
    experiment_data = dict(parameters["experiment"])
    required_dynamic = {
        "drag_speed_m_s",
        "constant_preload_n",
        "holder_effective_mass_kg",
        "holder_vertical_damping_n_s_m",
        "output_spacing_m",
    }
    missing_dynamic = required_dynamic - set(experiment_data)
    if missing_dynamic:
        raise ValueError(
            "M2 dynamic experiment is missing explicit fields: "
            f"{sorted(missing_dynamic)}"
        )
    if "dynamic_contact" not in parameters:
        raise ValueError("M2 case requires explicit dynamic_contact settings")
    if "dynamic_integrator" not in parameters:
        raise ValueError(
            "M2 case requires explicit dynamic_integrator settings"
        )
    experiment = DynamicExperimentSettings.from_mapping(experiment_data)
    contact = DynamicContactSettings.from_mapping(
        parameters["dynamic_contact"]
    )
    integrator = DynamicIntegratorSettings.from_mapping(
        parameters["dynamic_integrator"]
    )
    result = DynamicSingleSpineExperiment(
        spine,
        track,
        experiment,
        contact,
        integrator,
    ).run()
    summary = _summary_dict(result)
    arrays = _arrays(result)
    validation = {
        "passed": (
            result.summary.initial_preload_success
            and result.summary.numerical_state.value == "converged"
            and result.summary.run_terminal_state.value == "path_end"
        ),
        "maximum_abs_geometry_residual_m": (
            float(
                max(
                    (max(0.0, -point.gap_m) for point in result.points),
                    default=0.0,
                )
            )
        ),
        "maximum_abs_energy_residual_j": (
            result.summary.maximum_abs_energy_residual_j
        ),
        "formal_ranking_eligible": result.summary.formal_ranking_eligible,
    }
    return CaseOutput(
        summary=summary,
        arrays=arrays,
        events=_events(result, context.case_id),
        validation=validation,
        stage_times_s={"m2_total": time.perf_counter() - started},
    )
