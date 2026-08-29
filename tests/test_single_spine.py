from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import pytest

from spine_sim.core.errors import ConfigurationError
from spine_sim.core.states import (
    ContinuationAction,
    EventType,
    ModelState,
    NumericalEventType,
    PhysicalState,
    SpringBranch,
    validate_physical_transition,
)
from spine_sim.single_spine import (
    BaseMotion,
    FrictionParameters,
    SingleSpineTolerances,
    SpineAcceptedState,
    SpineGeometry,
    SpineMaterial,
    SuspensionParameters,
    commit_single_spine_trial,
    solve_single_spine,
    solve_unilateral_spring,
    solve_wedge_2d,
)


@dataclass(frozen=True)
class Clearance:
    collision: bool | None = False


@dataclass(frozen=True)
class Candidate:
    candidate_id: str = "candidate-1"
    selected_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    signed_gap_m: float = 0.0
    valid: bool = True
    near_tie: bool = False
    sphere_center_m: tuple[float, float, float] = (0.0, 0.0, -0.004)
    support_points_m: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, -0.00405),
    )
    tangent_basis: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    forward_cap_valid: bool | None = True
    rod_clearance: Clearance | None = Clearance()
    effective_radius_m: float | None = 40e-6
    curvature_radius_m: float | None = None
    search_cursor: int = 1


def geometry() -> SpineGeometry:
    return SpineGeometry(
        spine_id="spine-1",
        root_position_m=(0.0, 0.0, 0.0),
        axis_root_to_tip=(0.0, 0.0, -1.0),
        length_m=0.004,
        diameter_m=0.0008,
        tip_radius_m=50e-6,
        frame="surface",
        root_reference="root-1",
        backplate_object="backplate",
    )


def material(**changes) -> SpineMaterial:
    base = SpineMaterial(
        young_modulus_Pa=200e9,
        poisson_ratio=0.29,
        shear_correction=6.0 / 7.0,
        shaft_allowable_stress_Pa=1e12,
        surface_capacity_present=False,
        fracture_topology_present=False,
        shaft_failure_is_catastrophic_disconnect=False,
        parameter_sources={"shaft_allowable_stress_Pa": "test fixture"},
    )
    return replace(base, **changes)


def friction(mu_s: float = 0.45, mu_k: float = 0.35) -> FrictionParameters:
    return FrictionParameters(mu_s, mu_k, "test fixture")


def rigid_suspension() -> SuspensionParameters:
    return SuspensionParameters(
        additional_compliance_m_per_N=None,
        axial_spring_stiffness_N_per_m=None,
        axial_spring_travel_m=None,
        rebound_recovery_distance_m=0.0002,
    )


def spring_suspension() -> SuspensionParameters:
    return SuspensionParameters(
        additional_compliance_m_per_N=None,
        axial_spring_stiffness_N_per_m=300.0,
        axial_spring_travel_m=0.004,
        rebound_recovery_distance_m=0.0002,
        parameter_sources={
            "axial_spring_stiffness_N_per_m": "test fixture",
            "axial_spring_travel_m": "test fixture",
        },
    )


def tolerances() -> SingleSpineTolerances:
    return SingleSpineTolerances(
        gap_m=1e-10,
        force_N=1e-9,
        friction_N=1e-9,
        spring_N=1e-9,
        velocity_m_per_s=1e-12,
        capacity_relative=1e-9,
        event_fraction=1e-10,
    )


def motion(
    displacement=(0.0, 0.0, 1e-6),
    velocity=(0.0, 0.0, 0.0),
    load_parameter=1.0,
    search_distance=0.0,
) -> BaseMotion:
    return BaseMotion(
        relative_displacement_m=displacement,
        relative_tangential_velocity_m_per_s=velocity,
        load_parameter=load_parameter,
        search_distance_increment_m=search_distance,
    )


def solve(
    accepted: SpineAcceptedState,
    base_motion: BaseMotion,
    candidate: Candidate | None = Candidate(),
    *,
    selected_material: SpineMaterial | None = None,
    selected_friction: FrictionParameters | None = None,
    suspension: SuspensionParameters | None = None,
):
    return solve_single_spine(
        geometry(),
        selected_material or material(),
        selected_friction or friction(),
        suspension or rigid_suspension(),
        accepted,
        base_motion,
        candidate,
        tolerances=tolerances(),
    )


def test_frozen_eight_states_event_names_and_transition_validation() -> None:
    assert [state.name for state in PhysicalState] == [
        "SEARCH",
        "CONTACT",
        "STICK",
        "SLIP",
        "DETACH",
        "REBOUND",
        "HARDSTOP",
        "FAILED",
    ]
    assert {event.name for event in EventType} == {
        "CONTACT",
        "CONTACT_REJECT",
        "STICK_START",
        "SLIP_START",
        "RESTICK",
        "DETACH",
        "REBOUND_START",
        "REBOUND_COMPLETE",
        "REENGAGE",
        "HARDSTOP",
        "HARDSTOP_RELEASE",
        "MATERIAL_FAILURE",
    }
    assert "NEWTON_RETRY" in NumericalEventType.__members__
    assert "NEWTON_RETRY" not in EventType.__members__
    validate_physical_transition(
        EventType.RESTICK, PhysicalState.SLIP, PhysicalState.STICK
    )
    with pytest.raises(ConfigurationError, match="illegal physical transition"):
        validate_physical_transition(
            EventType.RESTICK, PhysicalState.SEARCH, PhysicalState.STICK
        )


def test_chapter_2_wedge_closed_form_and_bidirectional_signs() -> None:
    T, P, phi, mu = 0.7, 0.5, 0.4, 0.3
    result = solve_wedge_2d(T, P, phi, mu)
    assert result.normal_force_N == pytest.approx(
        T * math.sin(phi) + P * math.cos(phi)
    )
    assert result.signed_friction_demand_N == pytest.approx(
        T * math.cos(phi) - P * math.sin(phi)
    )
    reverse = solve_wedge_2d(0.0, 1.0, phi, mu)
    assert reverse.signed_friction_demand_N < 0.0
    assert reverse.slip_direction_sign == -1
    flat_zero_mu = solve_wedge_2d(1.0, 1.0, 0.0, 0.0)
    assert flat_zero_mu.physical_state is PhysicalState.SLIP


def test_chapter_2_self_lock_and_peel_are_not_absolute_value_shortcuts() -> None:
    mu = 0.5
    self_lock = solve_wedge_2d(
        10.0, 1.0, math.atan(1.0 / mu), mu
    )
    assert self_lock.positive_direction_self_lock
    assert math.isinf(self_lock.forward_limit_ratio)
    peel_detach = solve_wedge_2d(0.1, -1.0, 0.7, mu)
    assert peel_detach.normal_force_N < 0.0
    assert peel_detach.physical_state is PhysicalState.DETACH


def test_unilateral_spring_complementarity_and_fixed_regression() -> None:
    regression = solve_unilateral_spring(
        0.21062006,
        300.0,
        0.004,
        force_tolerance_N=1e-9,
    )
    assert regression.branch is SpringBranch.INTERIOR
    assert regression.displacement_m == pytest.approx(0.000702066867)
    assert solve_unilateral_spring(
        -0.1, 300.0, 0.004, force_tolerance_N=1e-9
    ).branch is SpringBranch.LOWER_STOP
    hard = solve_unilateral_spring(
        1.5, 300.0, 0.004, force_tolerance_N=1e-9
    )
    assert hard.branch is SpringBranch.HARDSTOP
    assert hard.displacement_m == pytest.approx(0.004)


def test_coupled_spring_branch_closes_and_hardstop_releases_through_contact() -> None:
    initial = SpineAcceptedState.initial("spine-1")
    interior = solve(
        initial,
        motion((0.0, 0.0, 0.000702067)),
        suspension=spring_suspension(),
    )
    assert interior.result.spring_branch is SpringBranch.INTERIOR
    assert interior.result.spring_displacement_m == pytest.approx(
        0.000702, rel=2e-4
    )
    hard_trial = solve(
        initial,
        motion((0.0, 0.0, 0.005)),
        suspension=spring_suspension(),
    )
    assert hard_trial.result.physical_state is PhysicalState.HARDSTOP
    assert hard_trial.result.contact_submode is PhysicalState.STICK
    hard = commit_single_spine_trial(initial, hard_trial)
    released = solve(
        hard,
        motion((0.0, 0.0, 0.003), load_parameter=2.0),
        suspension=spring_suspension(),
    )
    assert released.result.physical_state is PhysicalState.STICK
    assert released.result.spring_branch is SpringBranch.INTERIOR
    assert [event.event_type for event in released.result.events] == [
        EventType.HARDSTOP_RELEASE,
        EventType.STICK_START,
    ]


def test_trial_is_immutable_until_explicit_commit() -> None:
    accepted = SpineAcceptedState.initial("spine-1")
    trial = solve(accepted, motion())
    assert accepted.physical_state is PhysicalState.SEARCH
    assert accepted.revision == 0
    assert trial.result.physical_state is PhysicalState.STICK
    assert [event.event_type for event in trial.result.events] == [
        EventType.CONTACT,
        EventType.STICK_START,
    ]
    committed = commit_single_spine_trial(accepted, trial)
    assert committed.physical_state is PhysicalState.STICK
    assert committed.revision == 1
    with pytest.raises(ConfigurationError, match="stale"):
        commit_single_spine_trial(committed, trial)


def test_contact_reject_does_not_commit_contact_or_reengagement_history() -> None:
    accepted = SpineAcceptedState.initial("spine-1")
    rejected = solve(
        accepted,
        motion(),
        replace(Candidate(), forward_cap_valid=False),
    )
    assert rejected.result.physical_state is PhysicalState.SEARCH
    assert [event.event_type for event in rejected.result.events] == [
        EventType.CONTACT_REJECT
    ]
    committed = commit_single_spine_trial(accepted, rejected)
    assert committed.candidate_id is None
    assert committed.search_cursor == 1
    assert committed.reengagement_count == 0
    assert committed.completed_detach_cycles == 0


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (
            replace(Candidate(), valid=False, near_tie=True, selected_normal=None),
            "near_tie_requires_resolved_normal_model",
        ),
        (replace(Candidate(), selected_normal=None), "contact_normal_unclosed"),
    ],
)
def test_unresolved_contact_normal_is_parameter_unclosed(
    candidate, reason
) -> None:
    trial = solve(SpineAcceptedState.initial("spine-1"), motion(), candidate)
    assert trial.result.model_state is ModelState.PARAMETER_UNCLOSED
    assert trial.result.events[0].details["reason"] == reason


def test_3d_friction_cone_slip_force_opposes_tangential_velocity() -> None:
    accepted = SpineAcceptedState.initial("spine-1")
    trial = solve(
        accepted,
        motion(
            displacement=(100e-6, 30e-6, 1e-6),
            velocity=(1e-3, 0.3e-3, 0.0),
        ),
    )
    result = trial.result
    assert result.physical_state is PhysicalState.SLIP
    tangent = np.asarray(result.tangential_force_N)
    velocity = np.array([1e-3, 0.3e-3, 0.0])
    assert np.dot(tangent, velocity) < 0.0
    assert np.linalg.norm(tangent) == pytest.approx(
        0.35 * result.normal_force_N
    )
    assert np.linalg.norm(tangent) <= 0.45 * result.normal_force_N


def test_slip_resticks_only_at_zero_velocity_inside_static_cone() -> None:
    accepted = SpineAcceptedState.initial("spine-1")
    slipping = solve(
        accepted,
        motion((100e-6, 0.0, 1e-6), (1e-3, 0.0, 0.0)),
    )
    slip_state = commit_single_spine_trial(accepted, slipping)
    restick = solve(
        slip_state,
        motion((1e-6, 0.0, 1e-6), (0.0, 0.0, 0.0), 2.0),
    )
    assert restick.result.physical_state is PhysicalState.STICK
    assert EventType.RESTICK in {
        event.event_type for event in restick.result.events
    }


def test_root_wrench_and_fixed_branch_tangent_match_finite_difference() -> None:
    accepted = SpineAcceptedState.initial("spine-1")
    center_motion = motion((1e-6, 0.0, 2e-6), load_parameter=1.0)
    center_trial = solve(
        accepted,
        center_motion,
        selected_friction=friction(2.0, 1.5),
    )
    center = commit_single_spine_trial(accepted, center_trial)
    force = np.asarray(center_trial.result.wall_force_N)
    lever = np.array([0.0, 0.0, -0.00405])
    np.testing.assert_allclose(
        center_trial.result.root_wrench.moment_Nm,
        np.cross(lever, force),
        atol=1e-12,
    )
    step = 1e-9
    plus = solve(
        center,
        motion((1e-6 + step, 0.0, 2e-6), load_parameter=2.0),
        selected_friction=friction(2.0, 1.5),
    )
    minus = solve(
        center,
        motion((1e-6 - step, 0.0, 2e-6), load_parameter=2.0),
        selected_friction=friction(2.0, 1.5),
    )
    derivative = (
        np.asarray(plus.result.wall_force_N)
        - np.asarray(minus.result.wall_force_N)
    ) / (2.0 * step)
    tangent = np.asarray(center_trial.result.local_tangent_N_per_m)
    np.testing.assert_allclose(derivative, tangent[:, 0], rtol=1e-7, atol=1e-5)


def test_missing_capacity_parameter_is_unclosed_not_failure() -> None:
    accepted = SpineAcceptedState.initial("spine-1")
    trial = solve(
        accepted,
        motion(),
        selected_material=material(shaft_allowable_stress_Pa=None),
    )
    assert trial.result.model_state is ModelState.PARAMETER_UNCLOSED
    assert trial.result.failure is None
    assert EventType.MATERIAL_FAILURE not in {
        event.event_type for event in trial.result.events
    }
    unsourced = solve(
        accepted,
        motion(),
        selected_material=material(parameter_sources={}),
    )
    assert unsourced.result.capacity_assessments["shaft"].model_state is (
        ModelState.PARAMETER_UNCLOSED
    )


@pytest.mark.parametrize(
    ("catastrophic", "expected_action", "expected_state"),
    [
        (
            True,
            ContinuationAction.PERMANENT_REMOVE,
            PhysicalState.FAILED,
        ),
        (
            False,
            ContinuationAction.STOP_MODEL_LIMIT,
            PhysicalState.STICK,
        ),
    ],
)
def test_failure_action_distinguishes_remove_from_model_limit(
    catastrophic, expected_action, expected_state
) -> None:
    accepted = SpineAcceptedState.initial("spine-1")
    trial = solve(
        accepted,
        motion(),
        selected_material=material(
            shaft_allowable_stress_Pa=1.0,
            shaft_failure_is_catastrophic_disconnect=catastrophic,
        ),
    )
    assert trial.result.failure is not None
    assert trial.result.failure.continuation_action is expected_action
    assert trial.result.physical_state is expected_state
    assert trial.result.events[-1].event_type is EventType.MATERIAL_FAILURE
    if catastrophic:
        np.testing.assert_allclose(trial.result.wall_force_N, 0.0)
    else:
        assert trial.result.model_state is ModelState.OUT_OF_SCOPE


def test_absent_surface_and_fracture_topologies_are_out_of_scope() -> None:
    trial = solve(SpineAcceptedState.initial("spine-1"), motion())
    assert trial.result.capacity_assessments["surface"].model_state is ModelState.OUT_OF_SCOPE
    assert trial.result.capacity_assessments["fracture"].model_state is ModelState.OUT_OF_SCOPE


def test_hertz_surface_and_mode_ii_fracture_capacities_are_closed_when_sourced() -> None:
    selected = material(
        surface_capacity_present=True,
        surface_young_modulus_Pa=30e9,
        surface_poisson_ratio=0.2,
        surface_allowable_tensile_stress_Pa=1e12,
        fracture_topology_present=True,
        fracture_toughness_Pa_sqrt_m=1e6,
        crack_half_length_m=1e-5,
        fracture_area_m2=1e-6,
        fracture_geometry_factor=1.2,
        parameter_sources={
            "shaft_allowable_stress_Pa": "test fixture",
            "surface_young_modulus_Pa": "test fixture",
            "surface_poisson_ratio": "test fixture",
            "surface_allowable_tensile_stress_Pa": "test fixture",
            "fracture_toughness_Pa_sqrt_m": "test fixture",
            "crack_half_length_m": "test fixture",
            "fracture_area_m2": "test fixture",
            "fracture_geometry_factor": "test fixture",
        },
    )
    trial = solve(
        SpineAcceptedState.initial("spine-1"),
        motion(),
        selected_material=selected,
    )
    assert trial.result.model_state is ModelState.CLOSED
    assert trial.result.capacity_assessments["hertz"].model_state is ModelState.CLOSED
    assert trial.result.capacity_assessments["surface"].model_state is ModelState.CLOSED
    assert trial.result.capacity_assessments["fracture"].model_state is ModelState.CLOSED
    assert trial.result.diagnostics["hertz_contact_radius_m"] > 0.0
    expected_fracture_capacity = 1e6 / (
        1.2 * math.sqrt(math.pi * 1e-5)
    ) * 1e-6
    assert trial.result.diagnostics["fracture_force_capacity_N"] == pytest.approx(
        expected_fracture_capacity
    )


def test_surface_boundary_stops_at_model_limit_instead_of_deleting_spine() -> None:
    selected = material(
        surface_capacity_present=True,
        surface_young_modulus_Pa=30e9,
        surface_poisson_ratio=0.2,
        surface_allowable_tensile_stress_Pa=1.0,
        parameter_sources={
            "shaft_allowable_stress_Pa": "test fixture",
            "surface_young_modulus_Pa": "test fixture",
            "surface_poisson_ratio": "test fixture",
            "surface_allowable_tensile_stress_Pa": "test fixture",
        },
    )
    trial = solve(
        SpineAcceptedState.initial("spine-1"),
        motion(),
        selected_material=selected,
    )
    assert trial.result.failure is not None
    assert trial.result.failure.failure_object == "surface"
    assert (
        trial.result.failure.continuation_action
        is ContinuationAction.STOP_MODEL_LIMIT
    )
    assert trial.result.physical_state is PhysicalState.STICK
    assert trial.result.model_state is ModelState.OUT_OF_SCOPE


def test_detach_rebound_search_then_reengage_updates_only_on_commit() -> None:
    initial = SpineAcceptedState.initial("spine-1")
    first = solve(initial, motion())
    state = commit_single_spine_trial(initial, first)
    detached_trial = solve(
        state,
        motion((0.0, 0.0, -1e-6), load_parameter=2.0),
    )
    assert detached_trial.result.physical_state is PhysicalState.DETACH
    detached = commit_single_spine_trial(state, detached_trial)
    rebound = commit_single_spine_trial(
        detached, solve(detached, motion(load_parameter=3.0), None)
    )
    assert rebound.physical_state is PhysicalState.REBOUND
    searched = commit_single_spine_trial(
        rebound,
        solve(
            rebound,
            motion(load_parameter=4.0, search_distance=0.0002),
            None,
        ),
    )
    assert searched.physical_state is PhysicalState.SEARCH
    assert searched.completed_detach_cycles == 1
    reengage_trial = solve(
        searched, motion(load_parameter=5.0), Candidate(candidate_id="candidate-2")
    )
    assert EventType.REENGAGE in {
        event.event_type for event in reengage_trial.result.events
    }
    assert searched.reengagement_count == 0
    reengaged = commit_single_spine_trial(searched, reengage_trial)
    assert reengaged.reengagement_count == 1


def test_earliest_friction_event_is_independent_of_trial_step_size() -> None:
    initial = SpineAcceptedState.initial("spine-1")
    attached = commit_single_spine_trial(
        initial, solve(initial, motion((0.0, 0.0, 1e-6)))
    )
    short = solve(
        attached,
        motion((100e-6, 0.0, 1e-6), (1e-3, 0.0, 0.0), 2.0),
    )
    long = solve(
        attached,
        motion((200e-6, 0.0, 1e-6), (1e-3, 0.0, 0.0), 3.0),
    )
    assert EventType.SLIP_START in {
        event.event_type for event in short.result.events
    }
    assert EventType.SLIP_START in {
        event.event_type for event in long.result.events
    }
    assert short.result.evaluated_motion.relative_displacement_m[0] == pytest.approx(
        long.result.evaluated_motion.relative_displacement_m[0],
        rel=1e-7,
        abs=1e-11,
    )
    event_x = short.result.evaluated_motion.relative_displacement_m[0]
    before = solve(
        attached,
        motion(
            (event_x - 1e-9, 0.0, 1e-6),
            (1e-3, 0.0, 0.0),
            1.5,
        ),
    )
    assert before.result.physical_state is PhysicalState.STICK
    assert np.linalg.norm(short.result.tangential_force_N) == pytest.approx(
        0.35 * short.result.normal_force_N
    )
    assert np.linalg.norm(before.result.tangential_force_N) > np.linalg.norm(
        short.result.tangential_force_N
    )
