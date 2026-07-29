"""M0 runner adapter and same-time-state serialization for dynamic M3 cases."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.contact import DynamicContactSettings, DynamicIntegratorSettings
from spine_sim.runtime.runner import CaseOutput, RunContext
from spine_sim.terrain import TerrainLibrary

from .dynamics import (
    ArrayDynamicExperimentSettings,
    DynamicCommonBackplateArray,
    DynamicCommonBackplateExperiment,
)
from .models import M3_MODEL_LEVEL, M3_MODULE_VERSION, ArrayConfiguration


_EXPLICIT_EXPERIMENT_PARAMETERS = {
    "external_total_preload_n",
    "drag_speed_m_s",
    "backplate_mass_kg",
    "backplate_vertical_damping_n_s_m",
    "backplate_rotational_dofs",
    "backplate_inertia_kg_m2",
    "maximum_preload_approach_m",
    "preload_ramp_time_s",
    "preload_ramp_profile",
    "settlement_damping_scale",
    "settling_reaction_force_tolerance_n",
    "settling_reaction_force_relative_tolerance",
    "settling_dynamic_residual_tolerance_n",
    "settling_stable_steps",
    "dynamic_residual_tolerance_n",
    "coupled_projection_relaxation",
    "output_spacing_m",
    "effective_pin_normal_force_min_n",
    "time_step_convergence_checked",
    "contact_parameter_convergence_checked",
    "settlement_damping_convergence_checked",
    "terrain_resolution_convergence_checked",
    "physical_calibration_completed",
}
_EXPLICIT_CONTACT_PARAMETERS = set(DynamicContactSettings.__dataclass_fields__)
_EXPLICIT_INTEGRATOR_PARAMETERS = set(DynamicIntegratorSettings.__dataclass_fields__)


def _summary_dict(result) -> dict[str, Any]:
    summary = asdict(result.summary)
    summary["numerical_state"] = result.summary.numerical_state.value
    summary["model_state"] = result.summary.model_state.value
    summary["run_terminal_state"] = result.summary.run_terminal_state.value
    summary["configuration"] = result.configuration.as_dict()
    summary["configuration_id"] = result.configuration.configuration_id
    summary["terrain_recipe_id"] = result.terrain_recipe_id
    summary["region_id"] = result.region_id
    summary["track_ids"] = list(result.track_ids)
    summary["experiment"] = asdict(result.experiment)
    summary["contact"] = asdict(result.contact)
    summary["integrator"] = asdict(result.integrator)
    summary["assumptions"] = list(result.assumptions)
    summary["m3_module_version"] = M3_MODULE_VERSION
    summary["model_level"] = M3_MODEL_LEVEL
    return summary


def _full_path_arrays(result) -> dict[str, np.ndarray]:
    points = result.points
    pin_count = result.configuration.pin_count

    def numeric(getter, *, trailing_shape: tuple[int, ...] = ()):
        values = [getter(point) for point in points]
        if not values:
            return np.empty((0, *trailing_shape), dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    def pin_numeric(getter, *, width: int | None = None):
        trailing = (pin_count,) if width is None else (pin_count, width)
        return numeric(
            lambda point: [getter(pin) for pin in point.pin_responses],
            trailing_shape=trailing,
        )

    def active_mask(indices):
        mask = np.zeros(pin_count, dtype=np.bool_)
        mask[list(indices)] = True
        return mask

    arrays = {
        "time_s": numeric(lambda point: point.time_s),
        "path_position_m": numeric(lambda point: point.path_position_m),
        "backplate_position_xyz_m": numeric(
            lambda point: point.backplate_position_xyz_m,
            trailing_shape=(3,),
        ),
        "backplate_velocity_xyz_m_s": numeric(
            lambda point: point.backplate_velocity_xyz_m_s,
            trailing_shape=(3,),
        ),
        "backplate_acceleration_xyz_m_s2": numeric(
            lambda point: point.backplate_acceleration_xyz_m_s2,
            trailing_shape=(3,),
        ),
        "external_total_preload_n": numeric(
            lambda point: point.external_total_preload_n
        ),
        "pin_holder_xyz_m": pin_numeric(lambda pin: pin.holder_xyz_m, width=3),
        "pin_center_xyz_m": pin_numeric(lambda pin: pin.center_xyz_m, width=3),
        "pin_center_velocity_xyz_m_s": pin_numeric(
            lambda pin: pin.center_velocity_xyz_m_s, width=3
        ),
        "pin_center_acceleration_xyz_m_s2": pin_numeric(
            lambda pin: pin.center_acceleration_xyz_m_s2, width=3
        ),
        "pin_support_xyz_m": pin_numeric(
            lambda pin: (
                pin.support_xyz_m
                if pin.support_xyz_m is not None
                else (np.nan, np.nan, np.nan)
            ),
            width=3,
        ),
        "pin_gap_m": pin_numeric(lambda pin: pin.gap_m),
        "pin_normal_force_n": pin_numeric(lambda pin: pin.normal_force_n),
        "pin_tangential_force_n": pin_numeric(
            lambda pin: pin.tangential_force_n
        ),
        "pin_normal_impulse_n_s": pin_numeric(
            lambda pin: pin.normal_impulse_n_s
        ),
        "pin_tangential_impulse_n_s": pin_numeric(
            lambda pin: pin.tangential_impulse_n_s
        ),
        "pin_impact_velocity_m_s": pin_numeric(
            lambda pin: pin.impact_velocity_m_s
        ),
        "pin_axial_displacement_m": pin_numeric(
            lambda pin: pin.axial_displacement_m
        ),
        "pin_transverse_displacement_m": pin_numeric(
            lambda pin: pin.transverse_displacement_m
        ),
        "pin_axial_velocity_m_s": pin_numeric(
            lambda pin: pin.axial_velocity_m_s
        ),
        "pin_transverse_velocity_m_s": pin_numeric(
            lambda pin: pin.transverse_velocity_m_s
        ),
        "pin_axial_force_n": pin_numeric(lambda pin: pin.axial_force_n),
        "pin_transverse_force_n": pin_numeric(
            lambda pin: pin.transverse_force_n
        ),
        "pin_spring_compression_m": pin_numeric(
            lambda pin: pin.spring_compression_m
        ),
        "pin_spring_travel_margin_m": pin_numeric(
            lambda pin: pin.spring_travel_margin_m
        ),
        "pin_bending_stress_pa": pin_numeric(
            lambda pin: pin.bending_stress_pa
        ),
        "pin_euler_buckling_margin_n": pin_numeric(
            lambda pin: pin.euler_buckling_margin_n
        ),
        "pin_wrench_about_holder": pin_numeric(
            lambda pin: pin.spine_on_plate_wrench_about_holder, width=6
        ),
        "pin_wrench_about_unit": pin_numeric(
            lambda pin: pin.spine_on_plate_wrench_about_unit, width=6
        ),
        "wall_on_unit_wrench_about_origin": numeric(
            lambda point: point.wall_on_unit_wrench_about_origin,
            trailing_shape=(6,),
        ),
        "active_thrust_wrench_about_origin": numeric(
            lambda point: point.active_thrust_wrench_about_origin,
            trailing_shape=(6,),
        ),
        "guide_reaction_wrench_about_origin": numeric(
            lambda point: point.guide_reaction_wrench_about_origin,
            trailing_shape=(6,),
        ),
        "unit_normal_force_n": numeric(
            lambda point: point.total_contact_reaction_z_n
        ),
        "tangential_force_positive_n": numeric(
            lambda point: point.tangential_force_positive_n
        ),
        "tangential_force_negative_n": numeric(
            lambda point: point.tangential_force_negative_n
        ),
        "unit_moment_nm": numeric(
            lambda point: point.wall_on_unit_wrench_about_origin[3:],
            trailing_shape=(3,),
        ),
        "active_pin_count": numeric(lambda point: point.active_pin_count),
        "effective_load_pin_count": numeric(
            lambda point: point.effective_load_pin_count
        ),
        "neff_normal": numeric(lambda point: point.sharing.neff_normal),
        "neff_target_tangential": numeric(
            lambda point: point.sharing.neff_target_tangential
        ),
        "neff_resultant": numeric(
            lambda point: point.sharing.neff_resultant
        ),
        "max_mean_normal": numeric(
            lambda point: point.sharing.max_mean_normal
        ),
        "max_mean_target_tangential": numeric(
            lambda point: point.sharing.max_mean_target_tangential
        ),
        "max_mean_resultant": numeric(
            lambda point: point.sharing.max_mean_resultant
        ),
        "gini_normal": numeric(lambda point: point.sharing.gini_normal),
        "gini_target_tangential": numeric(
            lambda point: point.sharing.gini_target_tangential
        ),
        "gini_resultant": numeric(
            lambda point: point.sharing.gini_resultant
        ),
        "total_contact_reaction_z_n": numeric(
            lambda point: point.total_contact_reaction_z_n
        ),
        "backplate_inertia_force_z_n": numeric(
            lambda point: point.backplate_inertia_force_z_n
        ),
        "backplate_damping_force_z_n": numeric(
            lambda point: point.backplate_damping_force_z_n
        ),
        "kinetic_energy_j": numeric(lambda point: point.kinetic_energy_j),
        "structural_energy_j": numeric(lambda point: point.structural_energy_j),
        "preload_work_increment_j": numeric(
            lambda point: point.preload_work_increment_j
        ),
        "drive_work_increment_j": numeric(
            lambda point: point.drive_work_increment_j
        ),
        "cumulative_preload_work_j": numeric(
            lambda point: point.cumulative_preload_work_j
        ),
        "cumulative_drive_work_j": numeric(
            lambda point: point.cumulative_drive_work_j
        ),
        "friction_dissipation_increment_j": numeric(
            lambda point: point.friction_dissipation_increment_j
        ),
        "cumulative_friction_dissipation_j": numeric(
            lambda point: point.cumulative_friction_dissipation_j
        ),
        "structural_damping_dissipation_increment_j": numeric(
            lambda point: point.structural_damping_dissipation_increment_j
        ),
        "cumulative_structural_damping_dissipation_j": numeric(
            lambda point: point.cumulative_structural_damping_dissipation_j
        ),
        "backplate_damping_dissipation_increment_j": numeric(
            lambda point: point.backplate_damping_dissipation_increment_j
        ),
        "cumulative_backplate_damping_dissipation_j": numeric(
            lambda point: point.cumulative_backplate_damping_dissipation_j
        ),
        "dynamic_residual_n": numeric(lambda point: point.dynamic_residual_n),
        "energy_residual_j": numeric(lambda point: point.energy_residual_j),
        "actual_time_step_s": numeric(
            lambda point: point.actual_time_step_s
        ),
        "nonlinear_iterations": numeric(
            lambda point: point.nonlinear_iterations
        ),
        "force_aggregation_residual_n": numeric(
            lambda point: point.force_aggregation_residual_n
        ),
        "moment_aggregation_residual_nm": numeric(
            lambda point: point.moment_aggregation_residual_nm
        ),
        "contact_state": np.asarray(
            [
                [pin.contact_state.value for pin in point.pin_responses]
                for point in points
            ],
            dtype="U32",
        ).reshape((-1, pin_count)),
        "spring_state": np.asarray(
            [
                [pin.spring_state.value for pin in point.pin_responses]
                for point in points
            ],
            dtype="U16",
        ).reshape((-1, pin_count)),
        "event_label": np.asarray(
            [
                [pin.event_label.value for pin in point.pin_responses]
                for point in points
            ],
            dtype="U24",
        ).reshape((-1, pin_count)),
        "active_nominal": np.asarray(
            [active_mask(point.activity_sets.nominal) for point in points],
            dtype=np.bool_,
        ).reshape((-1, pin_count)),
        "active_geometric": np.asarray(
            [active_mask(point.activity_sets.geometric) for point in points],
            dtype=np.bool_,
        ).reshape((-1, pin_count)),
        "active_positive_normal": np.asarray(
            [
                active_mask(point.activity_sets.positive_normal)
                for point in points
            ],
            dtype=np.bool_,
        ).reshape((-1, pin_count)),
        "active_admissible": np.asarray(
            [active_mask(point.activity_sets.admissible) for point in points],
            dtype=np.bool_,
        ).reshape((-1, pin_count)),
        "active_target_load": np.asarray(
            [active_mask(point.activity_sets.target_load) for point in points],
            dtype=np.bool_,
        ).reshape((-1, pin_count)),
        "numerical_state": np.asarray(
            [point.numerical_state.value for point in points], dtype="U24"
        ),
        "model_state": np.asarray(
            [point.model_state.value for point in points], dtype="U24"
        ),
    }
    return arrays


_AGGREGATE_TRACE_KEYS = {
    "time_s",
    "path_position_m",
    "backplate_position_xyz_m",
    "backplate_velocity_xyz_m_s",
    "backplate_acceleration_xyz_m_s2",
    "external_total_preload_n",
    "wall_on_unit_wrench_about_origin",
    "active_thrust_wrench_about_origin",
    "guide_reaction_wrench_about_origin",
    "unit_normal_force_n",
    "tangential_force_positive_n",
    "tangential_force_negative_n",
    "unit_moment_nm",
    "active_pin_count",
    "effective_load_pin_count",
    "neff_normal",
    "neff_target_tangential",
    "neff_resultant",
    "max_mean_normal",
    "max_mean_target_tangential",
    "max_mean_resultant",
    "gini_normal",
    "gini_target_tangential",
    "gini_resultant",
    "total_contact_reaction_z_n",
    "backplate_inertia_force_z_n",
    "backplate_damping_force_z_n",
    "kinetic_energy_j",
    "structural_energy_j",
    "preload_work_increment_j",
    "drive_work_increment_j",
    "cumulative_preload_work_j",
    "cumulative_drive_work_j",
    "friction_dissipation_increment_j",
    "cumulative_friction_dissipation_j",
    "structural_damping_dissipation_increment_j",
    "cumulative_structural_damping_dissipation_j",
    "backplate_damping_dissipation_increment_j",
    "cumulative_backplate_damping_dissipation_j",
    "dynamic_residual_n",
    "energy_residual_j",
    "actual_time_step_s",
    "nonlinear_iterations",
    "force_aggregation_residual_n",
    "moment_aggregation_residual_nm",
    "numerical_state",
    "model_state",
}


def _settlement_arrays(result) -> dict[str, np.ndarray]:
    trace = result.settlement_trace
    fields = {
        "time_s": "time_s",
        "ramp_fraction": "ramp_fraction",
        "applied_total_preload_n": "applied_total_preload_n",
        "damping_scale": "damping_scale",
        "backplate_position_z_m": "backplate_position_z_m",
        "actual_approach_m": "actual_approach_m",
        "maximum_mode_speed_m_s": "maximum_mode_speed_m_s",
        "total_contact_reaction_z_n": "total_contact_reaction_z_n",
        "contact_reaction_error_n": "contact_reaction_error_n",
        "dynamic_residual_n": "dynamic_residual_n",
        "active_pin_count": "active_pin_count",
        "stable_steps": "stable_steps",
    }
    return {
        f"settlement_{output_name}": np.asarray(
            [getattr(point, attribute_name) for point in trace],
            dtype=(
                np.int64
                if attribute_name in {"active_pin_count", "stable_steps"}
                else np.float64
            ),
        )
        for output_name, attribute_name in fields.items()
    }


def _arrays(
    result,
    output_level: str = "full_pin_trace",
) -> dict[str, np.ndarray]:
    if output_level not in {
        "summary",
        "aggregate_trace",
        "full_pin_trace",
    }:
        raise ValueError(
            "output level must be summary, aggregate_trace or full_pin_trace"
        )
    if output_level == "summary":
        return {}
    path_arrays = _full_path_arrays(result)
    if output_level == "aggregate_trace":
        path_arrays = {
            key: value
            for key, value in path_arrays.items()
            if key in _AGGREGATE_TRACE_KEYS
        }
    path_arrays.update(_settlement_arrays(result))
    return path_arrays


def _proxy_array_rod_clearance(
    *,
    library: TerrainLibrary,
    result,
    sample_count: int = 24,
) -> np.ndarray:
    """Post-check cylindrical shank clearance for every saved pin state."""

    if sample_count < 2:
        raise ValueError("rod-clearance sample_count must be at least two")
    point_count = len(result.points)
    pin_count = result.configuration.pin_count
    if point_count == 0:
        return np.empty((0, pin_count), dtype=np.float64)
    height = library.open_region(result.terrain_recipe_id, result.region_id)
    region = library.load_region_spec(
        result.terrain_recipe_id,
        result.region_id,
    )
    minimum = np.empty((point_count, pin_count), dtype=np.float64)
    for pin_index, parameters in enumerate(
        result.configuration.pin_parameters
    ):
        y_global_m = result.points[0].pin_responses[pin_index].holder_xyz_m[1]
        y_float = (y_global_m - region.origin_y_m) / region.resolution_y_m
        if y_float < 0.0 or y_float > height.shape[0] - 1:
            raise ValueError(
                f"pin {pin_index} rod-clearance y coordinate lies outside terrain"
            )
        y0 = min(int(np.floor(y_float)), height.shape[0] - 2)
        ty = y_float - y0
        rod_radius = 0.5 * parameters.diameter_m
        distances = np.linspace(
            min(rod_radius, parameters.exposed_length_m),
            parameters.exposed_length_m,
            sample_count,
            dtype=np.float64,
        )
        axis_x, axis_z = parameters.axis_xz
        center_x = np.asarray(
            [
                point.pin_responses[pin_index].center_xyz_m[0]
                for point in result.points
            ],
            dtype=np.float64,
        )
        center_z = np.asarray(
            [
                point.pin_responses[pin_index].center_xyz_m[2]
                for point in result.points
            ],
            dtype=np.float64,
        )
        sample_x = center_x[:, None] - axis_x * distances[None, :]
        sample_z = center_z[:, None] - axis_z * distances[None, :]
        x_float = (sample_x - region.origin_x_m) / region.resolution_x_m
        if np.any(x_float < 0.0) or np.any(x_float > height.shape[1] - 1):
            raise ValueError(
                f"pin {pin_index} rod-clearance x coordinate lies outside terrain"
            )
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
        minimum[:, pin_index] = np.min(
            sample_z - rod_radius - terrain_z,
            axis=1,
        )
    if getattr(height, "_mmap", None) is not None:
        height._mmap.close()
    return minimum


def _events(result, case_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    sequence = 0
    for point in result.points:
        for pin_index, label in point.event_labels:
            pin = point.pin_responses[pin_index]
            events.append(
                {
                    "sequence": sequence,
                    "case_id": case_id,
                    "event_type": label,
                    "time_s": point.time_s,
                    "path_position_m": point.path_position_m,
                    "details": {
                        "pin_index": pin_index,
                        "contact_state": pin.contact_state.value,
                        "spring_state": pin.spring_state.value,
                        "normal_force_n": pin.normal_force_n,
                        "normal_impulse_n_s": pin.normal_impulse_n_s,
                        "all_pin_normal_force_n": [
                            item.normal_force_n for item in point.pin_responses
                        ],
                    },
                }
            )
            sequence += 1
    return events


def run_case(parameters: Mapping[str, Any], context: RunContext) -> CaseOutput:
    started = time.perf_counter()
    required = {
        "terrain_library_root",
        "terrain_recipe_id",
        "region_id",
        "configuration",
        "unit_origin_xy_m",
        "experiment",
    }
    missing = required - set(parameters)
    if missing:
        raise ValueError(f"M3 case is missing fields: {sorted(missing)}")
    output_control = dict(parameters.get("output", {}))
    extra_output_fields = set(output_control) - {"level"}
    if extra_output_fields:
        raise ValueError(
            f"M3 output contains unknown fields: {sorted(extra_output_fields)}"
        )
    output_level = str(output_control.get("level", "full_pin_trace"))
    if output_level not in {
        "summary",
        "aggregate_trace",
        "full_pin_trace",
    }:
        raise ValueError(
            "M3 output.level must be summary, aggregate_trace or full_pin_trace"
        )
    library = TerrainLibrary(Path(str(parameters["terrain_library_root"])))
    configuration = ArrayConfiguration.from_mapping(parameters["configuration"])
    has_tracks = "tracks" in parameters
    has_requests = "track_requests" in parameters
    if has_tracks == has_requests:
        raise ValueError(
            "M3 case requires exactly one of tracks or track_requests"
        )
    if has_tracks:
        tracks = tuple(
            library.load_track(
                str(parameters["terrain_recipe_id"]),
                str(parameters["region_id"]),
                float(item["radius_m"]),
                str(item["track_id"]),
            )
            for item in parameters["tracks"]
        )
    else:
        requests = list(parameters["track_requests"])
        if len(requests) != configuration.pin_count:
            raise ValueError("M3 track_requests must contain one row per pin")
        recipe_for_tracks = library.load_recipe(
            str(parameters["terrain_recipe_id"])
        )
        region_for_tracks = library.load_region_spec(
            str(parameters["terrain_recipe_id"]),
            str(parameters["region_id"]),
        )
        tracks = tuple(
            library.cache_track(
                recipe_for_tracks,
                region_for_tracks,
                radius_m=float(request["radius_m"]),
                y_global_m=float(request["y_global_m"]),
            )
            for request in requests
        )
    if configuration.fixture_only:
        raise ValueError("fixture-only singleton arrays cannot run as saved project cases")
    if configuration.base_spine.rod_clearance_mode == "disabled_analytic_fixture":
        raise ValueError(
            "saved project-terrain M3 cases cannot disable rod clearance"
        )

    experiment_data = dict(parameters["experiment"])
    contact_data = dict(parameters.get("contact", {}))
    integrator_data = dict(parameters.get("integrator", {}))
    unclosed = set(experiment_data.get("unclosed_parameter_names", ()))
    unclosed.update(
        f"experiment.{name}"
        for name in _EXPLICIT_EXPERIMENT_PARAMETERS - set(experiment_data)
    )
    unclosed.update(
        f"contact.{name}"
        for name in _EXPLICIT_CONTACT_PARAMETERS - set(contact_data)
    )
    unclosed.update(
        f"integrator.{name}"
        for name in _EXPLICIT_INTEGRATOR_PARAMETERS - set(integrator_data)
    )
    experiment_data["unclosed_parameter_names"] = tuple(sorted(unclosed))
    contact = DynamicContactSettings.from_mapping(contact_data)
    integrator = DynamicIntegratorSettings.from_mapping(integrator_data)
    experiment = ArrayDynamicExperimentSettings.from_mapping(experiment_data)
    system = DynamicCommonBackplateArray(
        configuration,
        tracks,
        unit_origin_xy_m=tuple(parameters["unit_origin_xy_m"]),
        contact=contact,
    )
    result = DynamicCommonBackplateExperiment(
        system,
        experiment,
        integrator,
    ).run()
    recipe = library.load_recipe(str(parameters["terrain_recipe_id"]))
    validation_passed = (
        result.summary.initial_preload_success
        and result.summary.numerical_state.value == "converged"
        and result.summary.run_terminal_state.value == "path_end"
        and result.summary.maximum_abs_dynamic_residual_n is not None
        and np.isfinite(result.summary.maximum_abs_dynamic_residual_n)
        and result.summary.maximum_force_aggregation_residual_n is not None
        and result.summary.maximum_force_aggregation_residual_n <= 1e-12
        and result.summary.maximum_moment_aggregation_residual_nm is not None
        and result.summary.maximum_moment_aggregation_residual_nm <= 1e-15
    )
    summary = _summary_dict(result)
    summary["seed"] = recipe.seed
    summary["terrain_family"] = parameters.get("terrain_family")
    summary["terrain_condition_id"] = parameters.get(
        "terrain_condition_id"
    )
    summary["terrain_catalog_id"] = parameters.get("terrain_catalog_id")
    summary["loading_protocol_id"] = parameters.get(
        "loading_protocol_id"
    )
    summary["output_level"] = output_level
    summary["ranking_inclusion_allowed"] = bool(
        result.summary.initial_preload_success
        and result.summary.conditional_performance_available
        and result.summary.run_terminal_state.value == "path_end"
    )
    arrays = _arrays(result, output_level)
    rod_clearance = None
    minimum_rod_clearance = None
    rod_collision = None
    if (
        configuration.base_spine.rod_clearance_mode
        == "proxy_cylindrical_shank_postcheck"
        and result.points
    ):
        rod_clearance = _proxy_array_rod_clearance(
            library=library,
            result=result,
        )
        minimum_rod_clearance = float(np.min(rod_clearance))
        rod_collision = bool(minimum_rod_clearance < 0.0)
        if output_level == "full_pin_trace":
            arrays["pin_rod_clearance_m"] = rod_clearance
    yield_ok = (
        configuration.base_spine.yield_strength_pa is not None
        and result.summary.maximum_bending_stress_pa is not None
        and result.summary.maximum_bending_stress_pa
        <= configuration.base_spine.yield_strength_pa
    )
    buckling_ok = (
        result.summary.minimum_euler_buckling_margin_n is not None
        and result.summary.minimum_euler_buckling_margin_n >= 0.0
    )
    clearance_ok = rod_collision is False
    constraints_ok = yield_ok and buckling_ok and clearance_ok
    summary["ranking_inclusion_allowed"] = bool(
        summary["ranking_inclusion_allowed"] and constraints_ok
    )
    summary["formal_ranking_eligible"] = bool(
        result.summary.formal_ranking_eligible and constraints_ok
    )
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
    point_count = len(result.points)
    if arrays:
        arrays.update(
            {
            "seed": np.full(point_count, recipe.seed, dtype=np.int64),
            "terrain_recipe_id": np.full(
                point_count, result.terrain_recipe_id, dtype="U64"
            ),
            "configuration_id": np.full(
                point_count,
                result.configuration.configuration_id,
                dtype="U64",
            ),
            "model_level": np.full(
                point_count, M3_MODEL_LEVEL, dtype="U96"
            ),
            }
        )
    return CaseOutput(
        summary=summary,
        arrays=arrays,
        events=(
            []
            if output_level == "summary"
            else _events(result, context.case_id)
        ),
        validation={
            "passed": bool(validation_passed and constraints_ok),
            "formal_ranking_eligible": summary[
                "formal_ranking_eligible"
            ],
            "ranking_inclusion_allowed": summary[
                "ranking_inclusion_allowed"
            ],
            "initialization_coverage_member": True,
            "conditional_performance_member": bool(
                summary["ranking_inclusion_allowed"]
            ),
            "constraint_checks": {
                "yield_ok": yield_ok,
                "buckling_ok": buckling_ok,
                "rod_clearance_ok": clearance_ok,
            },
            "output_level": output_level,
            "same_time_state_sample_contract": True,
            "model_level": M3_MODEL_LEVEL,
        },
        stage_times_s={"m3_total": time.perf_counter() - started},
    )
