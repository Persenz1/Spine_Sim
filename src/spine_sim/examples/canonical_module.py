"""Named analytic catalog used to exercise the canonical production chain.

This module is intentionally a smoke catalog, not a calibrated hardware model.
Every physical value is tagged as an analytic test fixture in the result
provenance; production studies should construct the public geometry,
single-spine, and array inputs from measured/sourced records.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from spine_sim.array import (
    ArrayAcceptedState,
    ArrayTolerances,
    ControlMode,
    MixedControl,
    SpineInstance,
    commit_array_trial,
    solve_array_equilibrium,
)
from spine_sim.core.identity import identity, lineage_hash
from spine_sim.core.versions import ARRAY_MODEL_LEVEL, GEOMETRY_SCHEMA_VERSION
from spine_sim.geometry import CandidateCursor, ContactCandidate
from spine_sim.io.schema import CanonicalResultMetadata, validate_canonical_summary
from spine_sim.runtime.runner import CaseOutput, RunContext
from spine_sim.single_spine import (
    FrictionParameters,
    SingleSpineTolerances,
    SpineGeometry,
    SpineMaterial,
    SuspensionParameters,
)
from spine_sim.terrain.envelope import RodClearanceResult


CATALOG_NAME = "analytic_flat_wall_v1"
CATALOG_VERSION = "canonical-analytic-catalog-1"


def _float_tuple(values: Any, length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _fixture_candidate(spine_id: str, x_m: float, terrain_version: str) -> ContactCandidate:
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    sphere_center = np.array([x_m, 0.0, -0.004], dtype=np.float64)
    support = np.array([[x_m, 0.0, -0.00405]], dtype=np.float64)
    payload = {
        "catalog": CATALOG_NAME,
        "spine_id": spine_id,
        "sphere_center_m": sphere_center.tolist(),
        "geometry_version": GEOMETRY_SCHEMA_VERSION,
    }
    return ContactCandidate(
        candidate_id=identity(
            "candidate", payload, module_version=GEOMETRY_SCHEMA_VERSION
        ),
        lineage=lineage_hash(CATALOG_VERSION, payload),
        terrain_version=terrain_version,
        track_id=f"analytic_track_{spine_id}",
        geometry_version=GEOMETRY_SCHEMA_VERSION,
        candidate_index=0,
        path_position_m=0.0,
        feature_id=f"analytic_plane_{spine_id}",
        sphere_center_m=sphere_center,
        support_points_m=support,
        signed_gap_m=0.0,
        curvature_radius_m=None,
        surface_normal=normal,
        envelope_normal=normal,
        contact_normal=normal,
        normal_model="contact",
        tangent_basis=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        ),
        valid=True,
        near_tie=False,
        geometry_uncertain=False,
        gap_lower_m=0.0,
        gap_upper_m=0.0,
        forward_cap_valid=True,
        rod_clearance=RodClearanceResult(False, 1e-3, 1, ()),
        search_cursor=CandidateCursor(
            next_path_index=1,
            candidate_index=1,
            last_feature_id=f"analytic_plane_{spine_id}",
            exhausted=True,
        ),
    )


def _fixture_spines(parameters: Mapping[str, Any], terrain_version: str) -> tuple[SpineInstance, ...]:
    gaps = _float_tuple(parameters["initial_gaps_m"], len(parameters["initial_gaps_m"]), "initial_gaps_m")
    spacing_m = float(parameters["spacing_m"])
    additional_compliance = float(parameters["additional_compliance_m_per_N"])
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    if not np.isfinite(additional_compliance) or additional_compliance <= 0.0:
        raise ValueError("additional_compliance_m_per_N must be finite and positive")
    count = len(gaps)
    if count < 1:
        raise ValueError("analytic catalog requires at least one spine")
    material = SpineMaterial(
        young_modulus_Pa=float(parameters["young_modulus_Pa"]),
        poisson_ratio=float(parameters["poisson_ratio"]),
        shear_correction=float(parameters["shear_correction"]),
        shaft_allowable_stress_Pa=float(parameters["shaft_allowable_stress_Pa"]),
        surface_capacity_present=False,
        fracture_topology_present=False,
        shaft_failure_is_catastrophic_disconnect=False,
        parameter_sources={
            "young_modulus_Pa": "analytic_test_fixture",
            "poisson_ratio": "analytic_test_fixture",
            "shear_correction": "analytic_test_fixture",
            "shaft_allowable_stress_Pa": "analytic_test_fixture",
        },
    )
    friction = FrictionParameters(
        static_coefficient=float(parameters["static_friction"]),
        kinetic_coefficient=float(parameters["kinetic_friction"]),
        parameter_source="analytic_test_fixture",
    )
    suspension = SuspensionParameters(
        additional_compliance_m_per_N=(
            (additional_compliance, 0.0, 0.0),
            (0.0, additional_compliance, 0.0),
            (0.0, 0.0, additional_compliance),
        ),
        axial_spring_stiffness_N_per_m=None,
        axial_spring_travel_m=None,
        rebound_recovery_distance_m=float(parameters["rebound_recovery_distance_m"]),
        parameter_sources={
            "additional_compliance_m_per_N": "analytic_test_fixture",
            "rebound_recovery_distance_m": "analytic_test_fixture",
        },
    )
    tolerances = SingleSpineTolerances(
        gap_m=float(parameters["single_tolerances"]["gap_m"]),
        force_N=float(parameters["single_tolerances"]["force_N"]),
        friction_N=float(parameters["single_tolerances"]["friction_N"]),
        spring_N=float(parameters["single_tolerances"]["spring_N"]),
        velocity_m_per_s=float(
            parameters["single_tolerances"]["velocity_m_per_s"]
        ),
        capacity_relative=float(
            parameters["single_tolerances"]["capacity_relative"]
        ),
        event_fraction=float(
            parameters["single_tolerances"]["event_fraction"]
        ),
    )
    offset = 0.5 * (count - 1)
    spines: list[SpineInstance] = []
    for index, gap in enumerate(gaps):
        spine_id = f"spine-{index:04d}"
        x_m = (index - offset) * spacing_m
        geometry = SpineGeometry(
            spine_id=spine_id,
            root_position_m=(x_m, 0.0, 0.0),
            axis_root_to_tip=(0.0, 0.0, -1.0),
            length_m=float(parameters["spine_length_m"]),
            diameter_m=float(parameters["spine_diameter_m"]),
            tip_radius_m=float(parameters["tip_radius_m"]),
            frame="wall",
            root_reference=f"root-{index:04d}",
            backplate_object="rigid_backplate",
        )
        spines.append(
            SpineInstance(
                geometry=geometry,
                material=material,
                friction=friction,
                suspension=suspension,
                tolerances=tolerances,
                initial_gap_m=gap,
                candidate=_fixture_candidate(spine_id, x_m, terrain_version),
                stable_engagement=True,
            )
        )
    return tuple(spines)


def _control(parameters: Mapping[str, Any]) -> MixedControl:
    required_force = float(parameters["required_normal_force_N"])
    loader_stiffness = float(parameters["loader_stiffness_N_per_m"])
    loader = np.zeros((6, 6), dtype=np.float64)
    loader[2, 2] = loader_stiffness
    return MixedControl(
        modes=(
            ControlMode.PRESCRIBED_POSE,
            ControlMode.PRESCRIBED_POSE,
            ControlMode.REQUIRED_WRENCH,
            ControlMode.PRESCRIBED_POSE,
            ControlMode.PRESCRIBED_POSE,
            ControlMode.PRESCRIBED_POSE,
        ),
        prescribed_q_C=(0.0, 0.0, None, 0.0, 0.0, 0.0),
        required_wrench=(None, None, required_force, None, None, None),
        loader_stiffness=tuple(tuple(float(value) for value in row) for row in loader),
        initial_q_C=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        q_rate_C=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        F_ref_N=float(parameters["reference_force_N"]),
        L_ref_m=float(parameters["reference_length_m"]),
        frame="wall",
        reference_point="array_reference",
        reference_position_m=(0.0, 0.0, 0.0),
        backplate_object="rigid_backplate",
        resistance_direction=(1.0, 0.0, 0.0),
    )


def _per_spine_rows(result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in result.per_spine:
        single = item.single_result
        rows.append(
            {
                "spine_id": item.spine_id,
                "initial_gap_m": item.initial_gap_m,
                "terrain_signed_gap_m": item.terrain_signed_gap_m,
                "closure_threshold_m": item.closure_threshold_m,
                "signed_gap_m": item.signed_gap_m,
                "loading_displacement_m": item.loading_displacement_m,
                "physical_state": single.physical_state.value,
                "contact_submode": (
                    None
                    if single.contact_submode is None
                    else single.contact_submode.value
                ),
                "spring_branch": single.spring_branch.value,
                "normal_force_N": single.normal_force_N,
                "wall_force_N": single.wall_force_N,
                "root_wrench": single.root_wrench.as_dict(),
                "generalized_wrench": item.generalized_wrench.as_dict(),
                "margins": dict(single.margins),
                "capacity_assessments": {
                    name: asdict(value)
                    for name, value in single.capacity_assessments.items()
                },
                "complementarity_residuals": dict(
                    single.complementarity_residuals
                ),
                "model_state": single.model_state.value,
                "numerical_state": single.numerical_state.value,
            }
        )
    return rows


def run_case(parameters: Mapping[str, Any], context: RunContext) -> CaseOutput:
    """Run one explicitly named analytic canonical array case."""

    if parameters.get("catalog") != CATALOG_NAME:
        raise ValueError(f"only the named smoke catalog {CATALOG_NAME!r} is supported")
    if not context.terrain_version:
        raise ValueError("canonical cases require a non-empty terrain_version")
    spines = _fixture_spines(parameters, context.terrain_version)
    accepted = ArrayAcceptedState.initial(spines)
    array_tolerances = ArrayTolerances(
        **{name: value for name, value in parameters["array_tolerances"].items()}
    )
    trial = solve_array_equilibrium(
        spines,
        accepted,
        _control(parameters),
        load_parameter=float(parameters["load_parameter"]),
        tolerances=array_tolerances,
    )
    if not trial.committable:
        raise RuntimeError(
            "analytic canonical fixture did not reach an admissible equilibrium: "
            f"{trial.result.equilibrium_status.value}/"
            f"{trial.result.quasistatic_stability.value}"
        )
    committed = commit_array_trial(accepted, trial)
    result = trial.result
    per_spine = _per_spine_rows(result)
    metadata = CanonicalResultMetadata(
        case_id=context.case_id,
        normalized_input_hash=context.normalized_input_hash,
        model_level=ARRAY_MODEL_LEVEL,
        terrain_version=context.terrain_version,
        geometry_version=context.geometry_version,
        parameter_provenance={
            "catalog": CATALOG_NAME,
            "catalog_version": CATALOG_VERSION,
            "classification": "analytic_test_fixture",
            "seed": int(parameters["seed"]),
        },
        units={
            "length": "m",
            "angle": "rad",
            "force": "N",
            "moment": "N*m",
            "stiffness": "N/m_or_generalized_SI",
        },
        frames=(
            {
                "name": "wall",
                "reference_point": "array_reference",
                "convention": "right_handed",
                "physical_backplate_pose": "-q_C",
            },
        ),
        assumptions=result.assumptions + ("named_analytic_flat_wall_fixture",),
        omissions=result.omissions,
        applicability=("canonical_chain_smoke", "rigid_backplate_quasistatic"),
        cannot_answer=(
            "calibrated_hardware_capacity",
            "impact_or_inertial_response",
            "damage_evolution",
        ),
        project_schema_version=context.project_schema_version,
        model_schema_version=context.model_schema_version,
        result_schema_version=context.result_schema_version,
        solver_semantics_version=context.solver_semantics_version,
    )
    summary: dict[str, Any] = {
        **metadata.as_dict(),
        "physical_state": [
            item.single_result.physical_state.value for item in result.per_spine
        ],
        "numerical_state": result.numerical_state.value,
        "model_state": result.model_state.value,
        "run_state": "complete",
        "residuals": {
            "scaled_equilibrium_norm": result.diagnostics.scaled_residual_norm,
            "range_norm": result.diagnostics.range_residual_norm,
        },
        "tolerances": {
            **asdict(array_tolerances),
            "single_spine": asdict(spines[0].tolerances),
        },
        "per_spine": per_spine,
        "q_C": result.q_C,
        "physical_backplate_pose": result.physical_backplate_pose,
        "total_wrench": result.total_wrench.as_dict(),
        "counts": asdict(result.counts),
        "rank_status": result.rank_status.value,
        "range_status": result.range_status.value,
        "equilibrium_status": result.equilibrium_status.value,
        "quasistatic_stability": result.quasistatic_stability.value,
        "dynamic_stability": result.dynamic_stability.value,
        "equilibrium_diagnostics": asdict(result.diagnostics),
        "accepted_revision": committed.revision,
    }
    validate_canonical_summary(summary)
    trace_row = {
        "case_id": context.case_id,
        "load_parameter": committed.load_parameter,
        "accepted": True,
        "valid": True,
        "q_C": list(result.q_C),
        "physical_backplate_pose": list(result.physical_backplate_pose),
        "total_force_N": list(result.total_wrench.force_N),
        "total_moment_Nm": list(result.total_wrench.moment_Nm),
        "counts": asdict(result.counts),
        "rank_status": result.rank_status.value,
        "range_status": result.range_status.value,
        "equilibrium_status": result.equilibrium_status.value,
        "quasistatic_stability": result.quasistatic_stability.value,
        "dynamic_stability": result.dynamic_stability.value,
        "per_spine": per_spine,
    }
    return CaseOutput(
        summary=summary,
        arrays={
            "q_C": np.asarray(result.q_C, dtype=np.float64),
            "total_wrench": result.total_wrench.vector,
        },
        trace_rows=[trace_row],
        events=[event.as_dict() for event in result.events],
        validation={
            "status": "passed",
            "checks": [
                "canonical_result_schema",
                "array_equilibrium",
                "result_trace_round_trip",
            ],
        },
    )


__all__ = ["CATALOG_NAME", "CATALOG_VERSION", "run_case"]
