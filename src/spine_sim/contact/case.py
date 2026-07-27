"""M0 runner adapter and result serialization for M2 cases."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.runtime.runner import CaseOutput, RunContext
from spine_sim.terrain import TerrainLibrary

from .experiment import ExperimentSettings, SingleSpineExperiment
from .models import M2_MODULE_VERSION, SolverSettings, SpineParameters
from .solver import PrescribedPoseConstitutiveCore


def _summary_dict(result) -> dict[str, Any]:
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


def _arrays(result) -> dict[str, np.ndarray]:
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


def _events(result, case_id: str) -> list[dict[str, Any]]:
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
    experiment = ExperimentSettings(**experiment_data)
    solver = SolverSettings(**dict(parameters.get("solver", {})))
    core = PrescribedPoseConstitutiveCore(spine, track, solver)
    result = SingleSpineExperiment(core, experiment).run()
    summary = _summary_dict(result)
    arrays = _arrays(result)
    validation = {
        "passed": (
            result.summary.initial_preload_success
            and result.summary.numerical_state.value == "converged"
            and result.summary.run_terminal_state.value == "path_end"
        ),
        "maximum_abs_geometry_residual_m": (
            result.summary.maximum_abs_geometry_residual_m
        ),
        "maximum_abs_energy_residual_j": (
            result.summary.maximum_abs_energy_residual_j
        ),
        "formal_ranking_eligible": (
            result.summary.model_state.value == "covered"
            and result.summary.run_terminal_state.value == "path_end"
        ),
    }
    return CaseOutput(
        summary=summary,
        arrays=arrays,
        events=_events(result, context.case_id),
        validation=validation,
        stage_times_s={"m2_total": time.perf_counter() - started},
    )
