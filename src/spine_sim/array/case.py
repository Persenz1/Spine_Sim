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
    "output_spacing_m",
    "effective_pin_normal_force_min_n",
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


def _arrays(result) -> dict[str, np.ndarray]:
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
        "tracks",
        "configuration",
        "unit_origin_xy_m",
        "experiment",
    }
    missing = required - set(parameters)
    if missing:
        raise ValueError(f"M3 case is missing fields: {sorted(missing)}")
    library = TerrainLibrary(Path(str(parameters["terrain_library_root"])))
    tracks = tuple(
        library.load_track(
            str(parameters["terrain_recipe_id"]),
            str(parameters["region_id"]),
            float(item["radius_m"]),
            str(item["track_id"]),
        )
        for item in parameters["tracks"]
    )
    configuration = ArrayConfiguration.from_mapping(parameters["configuration"])
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
        and np.isfinite(result.summary.maximum_abs_dynamic_residual_n)
        and result.summary.maximum_force_aggregation_residual_n <= 1e-12
        and result.summary.maximum_moment_aggregation_residual_nm <= 1e-15
    )
    summary = _summary_dict(result)
    summary["seed"] = recipe.seed
    arrays = _arrays(result)
    point_count = len(result.points)
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
        events=_events(result, context.case_id),
        validation={
            "passed": bool(validation_passed),
            "formal_ranking_eligible": result.summary.formal_ranking_eligible,
            "same_time_state_sample_contract": True,
            "model_level": M3_MODEL_LEVEL,
        },
        stage_times_s={"m3_total": time.perf_counter() - started},
    )
