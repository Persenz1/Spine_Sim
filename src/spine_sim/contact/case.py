"""M0 runner adapter and result serialization for M2 cases."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.runtime.runner import CaseOutput, RunContext
from spine_sim.terrain import TerrainLibrary
from spine_sim.core.states import ModelState

from .dynamics import (
    DynamicContactSettings,
    DynamicExperimentSettings,
    DynamicIntegratorSettings,
    DynamicSingleSpineExperiment,
    DynamicSingleSpineResult,
)
from .models import M2_MODULE_VERSION, SpineParameters


def _json_finite(value):
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, tuple):
        return tuple(_json_finite(item) for item in value)
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    return value


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
        return _json_finite(summary)
    summary = asdict(result.summary)
    summary["numerical_state"] = result.summary.numerical_state.value
    summary["model_state"] = result.summary.model_state.value
    summary["run_terminal_state"] = result.summary.run_terminal_state.value
    summary["parameters"] = result.parameters.as_dict()
    summary["track_id"] = result.track_id
    summary["fixed_holder_z_m"] = result.fixed_holder_z_m
    summary["assumptions"] = list(result.assumptions)
    summary["m2_module_version"] = M2_MODULE_VERSION
    return _json_finite(summary)


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


def _proxy_rod_clearance(
    *,
    library: TerrainLibrary,
    track,
    result: DynamicSingleSpineResult,
    sample_count: int = 24,
) -> np.ndarray:
    """Conservative, low-cost shank clearance audit for the M2 proxy model.

    The dynamic contact law still acts at the finite spherical tip.  This
    diagnostic samples a cylindrical shank beginning one shank radius behind
    the sphere centre and extending to the holder.  It is a post-check, not a
    distributed rod-contact model.
    """

    if sample_count < 2:
        raise ValueError("rod-clearance sample_count must be at least two")
    height = library.open_region(
        track.terrain_recipe_id,
        track.region_id,
    )
    region = library.load_region_spec(
        track.terrain_recipe_id,
        track.region_id,
    )
    points = result.points
    if not points:
        return np.empty(0, dtype=np.float64)
    rod_radius = 0.5 * result.parameters.diameter_m
    start = min(rod_radius, result.parameters.exposed_length_m)
    distances = np.linspace(
        start,
        result.parameters.exposed_length_m,
        sample_count,
        dtype=np.float64,
    )
    axis_x, axis_z = result.parameters.axis_xz
    y_float = (
        track.y_global_m - region.origin_y_m
    ) / region.resolution_y_m
    if y_float < 0.0 or y_float > height.shape[0] - 1:
        raise ValueError("rod-clearance y coordinate lies outside terrain region")
    y0 = min(int(np.floor(y_float)), height.shape[0] - 2)
    ty = y_float - y0
    minimum = np.empty(len(points), dtype=np.float64)
    chunk_size = 2048
    for first in range(0, len(points), chunk_size):
        chunk = points[first : first + chunk_size]
        center_x = np.asarray(
            [point.center_xz_m[0] for point in chunk],
            dtype=np.float64,
        )
        center_z = np.asarray(
            [point.center_xz_m[1] for point in chunk],
            dtype=np.float64,
        )
        sample_x = center_x[:, None] - axis_x * distances[None, :]
        sample_z = center_z[:, None] - axis_z * distances[None, :]
        x_float = (sample_x - region.origin_x_m) / region.resolution_x_m
        if np.any(x_float < 0.0) or np.any(x_float > height.shape[1] - 1):
            raise ValueError("rod-clearance x coordinate lies outside terrain region")
        x0 = np.minimum(
            np.floor(x_float).astype(np.int64),
            height.shape[1] - 2,
        )
        tx = x_float - x0
        h00 = np.asarray(height[y0, x0], dtype=np.float64)
        h01 = np.asarray(height[y0, x0 + 1], dtype=np.float64)
        h10 = np.asarray(height[y0 + 1, x0], dtype=np.float64)
        h11 = np.asarray(height[y0 + 1, x0 + 1], dtype=np.float64)
        terrain_z = (
            (1.0 - ty) * ((1.0 - tx) * h00 + tx * h01)
            + ty * ((1.0 - tx) * h10 + tx * h11)
        )
        clearance = sample_z - rod_radius - terrain_z
        minimum[first : first + len(chunk)] = np.min(clearance, axis=1)
    return minimum


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
    rod_clearance = None
    rod_collision = None
    minimum_rod_clearance = None
    if (
        spine.rod_clearance_mode == "proxy_cylindrical_shank_postcheck"
        and result.points
    ):
        rod_clearance = _proxy_rod_clearance(
            library=library,
            track=track,
            result=result,
        )
        arrays["rod_clearance_m"] = rod_clearance
        minimum_rod_clearance = float(np.min(rod_clearance))
        rod_collision = bool(minimum_rod_clearance < 0.0)
    summary.update(
        {
            "ranking_scope": "project_model_proxy",
            "requires_experimental_calibration": True,
            "rod_clearance_checked": rod_clearance is not None,
            "rod_collision_detected": rod_collision,
            "minimum_rod_clearance_m": minimum_rod_clearance,
            "rod_clearance_assumption": (
                "cylindrical_shank_begins_one_radius_behind_tip_center"
                if rod_clearance is not None
                else None
            ),
        }
    )
    yield_ok = (
        spine.yield_strength_pa is not None
        and result.summary.maximum_bending_stress_pa
        <= spine.yield_strength_pa
    )
    buckling_ok = result.summary.minimum_euler_buckling_margin_n >= 0.0
    clearance_ok = rod_collision is False
    path_ok = (
        result.summary.initial_preload_success
        and result.summary.numerical_state.value == "converged"
        and result.summary.run_terminal_state.value == "path_end"
    )
    project_model_baseline_eligible = (
        path_ok
        and result.summary.model_state is ModelState.COVERED
        and yield_ok
        and buckling_ok
        and clearance_ok
    )
    energy_residual = result.summary.maximum_abs_energy_residual_j
    validation = {
        "passed": path_ok,
        "maximum_abs_geometry_residual_m": (
            float(
                max(
                    (max(0.0, -point.gap_m) for point in result.points),
                    default=0.0,
                )
            )
        ),
        "maximum_abs_energy_residual_j": (
            energy_residual if np.isfinite(energy_residual) else None
        ),
        "formal_ranking_eligible": result.summary.formal_ranking_eligible,
        "project_model_baseline_eligible": project_model_baseline_eligible,
        "ranking_scope": "project_model_proxy",
        "requires_experimental_calibration": True,
        "constraint_checks": {
            "yield_ok": yield_ok,
            "buckling_ok": buckling_ok,
            "rod_clearance_ok": clearance_ok,
        },
    }
    return CaseOutput(
        summary=summary,
        arrays=arrays,
        events=_events(result, context.case_id),
        validation=validation,
        stage_times_s={"m2_total": time.perf_counter() - started},
    )
