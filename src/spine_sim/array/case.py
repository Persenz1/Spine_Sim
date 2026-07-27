"""M0 runner adapter and same-state result serialization for M3 cases."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.contact import SolverSettings
from spine_sim.runtime.runner import CaseOutput, RunContext
from spine_sim.terrain import TerrainLibrary

from .experiment import ArrayExperimentSettings, CommonBackplateExperiment
from .models import M3_MODEL_LEVEL, M3_MODULE_VERSION, ArrayConfiguration
from .solver import CommonBackplateArray


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
    summary["fixed_common_uz_m"] = result.fixed_common_uz_m
    summary["target_preload_n"] = result.target_preload_n
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
            lambda point: [getter(response) for response in point.response.pin_responses],
            trailing_shape=trailing,
        )

    def active_mask(point, indices):
        mask = np.zeros(pin_count, dtype=np.bool_)
        mask[list(indices)] = True
        return mask

    return {
        "path_position_m": numeric(lambda point: point.path_position_m),
        "common_ux_m": numeric(lambda point: point.response.common_ux_m),
        "common_uz_m": numeric(lambda point: point.response.common_uz_m),
        "unit_origin_xyz_m": numeric(
            lambda point: point.response.unit_origin_xyz_m,
            trailing_shape=(3,),
        ),
        "pin_holder_xyz_m": numeric(
            lambda point: point.response.pin_holder_xyz_m,
            trailing_shape=(pin_count, 3),
        ),
        "pin_center_xz_m": pin_numeric(lambda response: response.center_xz_m, width=2),
        "pin_support_xyz_m": pin_numeric(
            lambda response: (
                response.support_xyz_m
                if response.support_xyz_m is not None
                else (np.nan, np.nan, np.nan)
            ),
            width=3,
        ),
        "pin_wrench_about_holder": pin_numeric(
            lambda response: response.spine_on_plate_wrench_about_holder,
            width=6,
        ),
        "pin_wrench_about_unit": numeric(
            lambda point: point.response.pin_wrench_about_unit,
            trailing_shape=(pin_count, 6),
        ),
        "wall_on_unit_wrench_about_origin": numeric(
            lambda point: point.response.wall_on_unit_wrench_about_origin,
            trailing_shape=(6,),
        ),
        "active_thrust_wrench_about_origin": numeric(
            lambda point: point.response.active_thrust_wrench_about_origin,
            trailing_shape=(6,),
        ),
        "guide_reaction_wrench_about_origin": numeric(
            lambda point: point.response.guide_reaction_wrench_about_origin,
            trailing_shape=(6,),
        ),
        "unit_normal_force_n": numeric(
            lambda point: point.response.total_normal_force_n
        ),
        "tangential_force_positive_n": numeric(
            lambda point: point.response.tangential_force_positive_n
        ),
        "tangential_force_negative_n": numeric(
            lambda point: point.response.tangential_force_negative_n
        ),
        "unit_moment_nm": numeric(
            lambda point: point.response.wall_on_unit_wrench_about_origin[3:],
            trailing_shape=(3,),
        ),
        "pin_normal_force_n": pin_numeric(lambda response: response.normal_force_n),
        "pin_tangential_force_n": pin_numeric(
            lambda response: response.tangential_force_n
        ),
        "pin_spring_compression_m": pin_numeric(
            lambda response: response.spring_compression_m
        ),
        "pin_geometry_residual_m": pin_numeric(
            lambda response: response.residual.geometry_m
        ),
        "contact_state": np.asarray(
            [
                [response.contact_state.value for response in point.response.pin_responses]
                for point in points
            ],
            dtype="U32",
        ).reshape((-1, pin_count)),
        "spring_state": np.asarray(
            [
                [response.spring_state.value for response in point.response.pin_responses]
                for point in points
            ],
            dtype="U16",
        ).reshape((-1, pin_count)),
        "event_label": np.asarray(
            [
                [response.event_label.value for response in point.response.pin_responses]
                for point in points
            ],
            dtype="U24",
        ).reshape((-1, pin_count)),
        "active_nominal": np.asarray(
            [
                active_mask(point, point.response.activity_sets.nominal)
                for point in points
            ],
            dtype=np.bool_,
        ).reshape((-1, pin_count)),
        "active_geometric": np.asarray(
            [
                active_mask(point, point.response.activity_sets.geometric)
                for point in points
            ],
            dtype=np.bool_,
        ).reshape((-1, pin_count)),
        "active_positive_normal": np.asarray(
            [
                active_mask(point, point.response.activity_sets.positive_normal)
                for point in points
            ],
            dtype=np.bool_,
        ).reshape((-1, pin_count)),
        "active_admissible": np.asarray(
            [
                active_mask(point, point.response.activity_sets.admissible)
                for point in points
            ],
            dtype=np.bool_,
        ).reshape((-1, pin_count)),
        "active_target_load": np.asarray(
            [
                active_mask(point, point.response.activity_sets.target_load)
                for point in points
            ],
            dtype=np.bool_,
        ).reshape((-1, pin_count)),
        "neff_normal": numeric(lambda point: point.response.sharing.neff_normal),
        "neff_target_tangential": numeric(
            lambda point: point.response.sharing.neff_target_tangential
        ),
        "neff_resultant": numeric(
            lambda point: point.response.sharing.neff_resultant
        ),
        "max_mean_normal": numeric(
            lambda point: point.response.sharing.max_mean_normal
        ),
        "max_mean_target_tangential": numeric(
            lambda point: point.response.sharing.max_mean_target_tangential
        ),
        "max_mean_resultant": numeric(
            lambda point: point.response.sharing.max_mean_resultant
        ),
        "gini_normal": numeric(lambda point: point.response.sharing.gini_normal),
        "gini_target_tangential": numeric(
            lambda point: point.response.sharing.gini_target_tangential
        ),
        "gini_resultant": numeric(
            lambda point: point.response.sharing.gini_resultant
        ),
        "force_aggregation_residual_n": numeric(
            lambda point: point.response.residual.force_aggregation_n
        ),
        "moment_aggregation_residual_nm": numeric(
            lambda point: point.response.residual.moment_aggregation_nm
        ),
        "event_refined": np.asarray(
            [point.event_refined for point in points],
            dtype=np.bool_,
        ),
        "numerical_state": np.asarray(
            [point.response.numerical_state.value for point in points],
            dtype="U24",
        ),
        "model_state": np.asarray(
            [point.response.model_state.value for point in points],
            dtype="U24",
        ),
    }


def _events(result, case_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    sequence = 0
    for point in result.points:
        for pin_index, label in point.response.event_labels:
            response = point.response.pin_responses[pin_index]
            events.append(
                {
                    "sequence": sequence,
                    "case_id": case_id,
                    "event_type": label,
                    "path_position_m": point.path_position_m,
                    "details": {
                        "pin_index": pin_index,
                        "contact_state": response.contact_state.value,
                        "spring_state": response.spring_state.value,
                        "normal_force_n": response.normal_force_n,
                        "all_pin_normal_force_n": [
                            item.normal_force_n
                            for item in point.response.pin_responses
                        ],
                        "event_refined": point.event_refined,
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
    system = CommonBackplateArray(
        configuration,
        tracks,
        unit_origin_xy_m=tuple(parameters["unit_origin_xy_m"]),
        solver_settings=SolverSettings(**dict(parameters.get("solver", {}))),
    )
    result = CommonBackplateExperiment(
        system,
        ArrayExperimentSettings(**dict(parameters["experiment"])),
    ).run()
    recipe = library.load_recipe(str(parameters["terrain_recipe_id"]))
    validation_passed = (
        result.summary.initial_preload_success
        and result.summary.numerical_state.value == "converged"
        and result.summary.run_terminal_state.value == "path_end"
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
                point_count,
                result.terrain_recipe_id,
                dtype="U64",
            ),
            "configuration_id": np.full(
                point_count,
                result.configuration.configuration_id,
                dtype="U64",
            ),
            "preload_n": np.full(
                point_count,
                result.target_preload_n,
                dtype=np.float64,
            ),
            "model_level": np.full(
                point_count,
                M3_MODEL_LEVEL,
                dtype="U80",
            ),
        }
    )
    return CaseOutput(
        summary=summary,
        arrays=arrays,
        events=_events(result, context.case_id),
        validation={
            "passed": validation_passed,
            "formal_ranking_eligible": (
                validation_passed
                and result.summary.model_state.value == "covered"
            ),
            "same_state_sample_contract": True,
            "model_level": M3_MODEL_LEVEL,
        },
        stage_times_s={"m3_total": time.perf_counter() - started},
    )
