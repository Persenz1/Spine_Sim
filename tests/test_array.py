from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import pytest

import spine_sim.array as array_module
from spine_sim.array import (
    ArrayAcceptedState,
    ArrayTolerances,
    ControlMode,
    EquilibriumStatus,
    MixedControl,
    QuasistaticStability,
    RangeStatus,
    RankStatus,
    SpineInstance,
    commit_array_trial,
    evaluate_conservative_stability,
    solve_array_equilibrium,
)
from spine_sim.core.errors import ConfigurationError
from spine_sim.core.frames import Wrench
from spine_sim.core.states import (
    ContinuationAction,
    Event,
    EventType,
    ModelState,
    NumericalState,
    PhysicalState,
    SpringBranch,
)
from spine_sim.geometry import (
    CandidateCursor,
    ContactCandidate,
    GEOMETRY_VERSION,
    SpinePath,
    SpinePose,
    SurfaceState,
    query_next_candidate,
)
from spine_sim.single_spine import (
    BaseMotion,
    FailurePayload,
    FrictionParameters,
    SingleSpineResult,
    SingleSpineTolerances,
    SingleSpineTrial,
    SpineAcceptedState,
    SpineGeometry,
    SpineMaterial,
    SuspensionParameters,
    solve_single_spine,
)
from spine_sim.terrain.envelope import RodClearanceResult
from spine_sim.terrain.library import TerrainLibrary
from spine_sim.terrain.models import RegionSpec, TerrainRecipe


def _candidate(
    spine_id: str,
    *,
    point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0),
    terrain_gap_m: float = 0.0,
    tip_radius_m: float = 50e-6,
    candidate_index: int = 0,
) -> ContactCandidate:
    normal_vector = np.asarray(normal, dtype=np.float64)
    normal_vector /= np.linalg.norm(normal_vector)
    center = np.asarray(point, dtype=np.float64) + tip_radius_m * normal_vector
    reference = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(reference, normal_vector))) > 0.9:
        reference = np.array([0.0, 0.0, 1.0])
    tangent = reference - np.dot(reference, normal_vector) * normal_vector
    tangent /= np.linalg.norm(tangent)
    feature_id = f"feature-{spine_id}"
    return ContactCandidate(
        candidate_id=f"candidate-{spine_id}",
        lineage=f"lineage-{spine_id}",
        terrain_version="array-test-terrain",
        track_id=f"track-{spine_id}",
        geometry_version=GEOMETRY_VERSION,
        candidate_index=candidate_index,
        path_position_m=float(candidate_index),
        feature_id=feature_id,
        sphere_center_m=center,
        support_points_m=np.asarray([point], dtype=np.float64),
        signed_gap_m=terrain_gap_m,
        curvature_radius_m=200e-6,
        surface_normal=normal_vector,
        envelope_normal=normal_vector,
        contact_normal=normal_vector,
        normal_model="contact",
        tangent_basis=np.stack((tangent, np.cross(normal_vector, tangent))),
        valid=True,
        near_tie=False,
        geometry_uncertain=False,
        gap_lower_m=None,
        gap_upper_m=None,
        forward_cap_valid=True,
        rod_clearance=RodClearanceResult(False, 1e-3, 1, ()),
        search_cursor=CandidateCursor(
            next_path_index=candidate_index + 1,
            candidate_index=candidate_index + 1,
            last_feature_id=feature_id,
        ),
    )


def _material() -> SpineMaterial:
    return SpineMaterial(
        young_modulus_Pa=200e9,
        poisson_ratio=0.29,
        shear_correction=6.0 / 7.0,
        shaft_allowable_stress_Pa=1e12,
        surface_capacity_present=False,
        fracture_topology_present=False,
        shaft_failure_is_catastrophic_disconnect=False,
        parameter_sources={"shaft_allowable_stress_Pa": "array test"},
    )


def _single_tolerances() -> SingleSpineTolerances:
    return SingleSpineTolerances(
        gap_m=1e-10,
        force_N=1e-9,
        friction_N=1e-9,
        spring_N=1e-9,
        velocity_m_per_s=1e-12,
        capacity_relative=1e-9,
        event_fraction=1e-10,
    )


def _spine(
    spine_id: str,
    *,
    root: tuple[float, float, float] = (0.0, 0.0, 0.0),
    point: tuple[float, float, float] | None = None,
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0),
    initial_gap_m: float = 0.0,
    terrain_gap_m: float = 0.0,
    candidate: bool = True,
) -> SpineInstance:
    tip_radius = 50e-6
    geometry = SpineGeometry(
        spine_id=spine_id,
        root_position_m=root,
        axis_root_to_tip=normal,
        length_m=0.004,
        diameter_m=0.0008,
        tip_radius_m=tip_radius,
        frame="wall",
        root_reference=f"root-{spine_id}",
        backplate_object="rigid_backplate",
    )
    contact_point = root if point is None else point
    return SpineInstance(
        geometry=geometry,
        material=_material(),
        friction=FrictionParameters(0.45, 0.35, "array test"),
        suspension=SuspensionParameters(None, None, None, 0.0002),
        tolerances=_single_tolerances(),
        initial_gap_m=initial_gap_m,
        candidate=(
            _candidate(
                spine_id,
                point=contact_point,
                normal=normal,
                terrain_gap_m=terrain_gap_m,
                tip_radius_m=tip_radius,
            )
            if candidate
            else None
        ),
    )


def _control(
    *,
    q: tuple[float, ...] = (0.0,) * 6,
    required: Mapping[int, float] | None = None,
    loader: np.ndarray | None = None,
    initial_q: tuple[float, ...] = (0.0,) * 6,
    q_rate: tuple[float, ...] = (0.0,) * 6,
    F_ref_N: float = 1.0,
    L_ref_m: float = 1.0,
    equality_matrix: tuple[tuple[float, ...], ...] = (),
) -> MixedControl:
    required = {} if required is None else required
    return MixedControl(
        modes=tuple(
            ControlMode.REQUIRED_WRENCH
            if index in required
            else ControlMode.PRESCRIBED_POSE
            for index in range(6)
        ),
        prescribed_q_C=tuple(
            None if index in required else float(q[index])
            for index in range(6)
        ),
        required_wrench=tuple(
            float(required[index]) if index in required else None
            for index in range(6)
        ),
        loader_stiffness=tuple(
            tuple(float(value) for value in row)
            for row in (
                np.zeros((6, 6), dtype=float) if loader is None else loader
            )
        ),
        initial_q_C=initial_q,
        q_rate_C=q_rate,
        F_ref_N=F_ref_N,
        L_ref_m=L_ref_m,
        equality_matrix=equality_matrix,
    )


def _install_linear_single_solver(
    monkeypatch: pytest.MonkeyPatch,
    laws: Mapping[str, Mapping[str, Any]],
    calls: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    counter = {"count": 0}
    per_spine_count: dict[str, int] = {}

    def fake_solve(
        geometry: SpineGeometry,
        material: SpineMaterial,
        friction: FrictionParameters,
        suspension: SuspensionParameters,
        accepted: SpineAcceptedState,
        motion: BaseMotion,
        candidate: ContactCandidate | None,
        *,
        tolerances: SingleSpineTolerances,
    ) -> SingleSpineTrial:
        del material, friction, suspension
        counter["count"] += 1
        per_spine_count[geometry.spine_id] = (
            per_spine_count.get(geometry.spine_id, 0) + 1
        )
        law = laws.get(geometry.spine_id, {})
        tangent = np.asarray(
            law.get("tangent", np.diag([1.0, 0.0, 0.0])), dtype=float
        )
        resident_state = law.get("resident_state")
        if accepted.physical_state is resident_state:
            tangent = np.asarray(law["resident_tangent"], dtype=float)
        displacement = np.asarray(motion.relative_displacement_m, dtype=float)
        event_threshold_x_m = law.get("event_threshold_x_m")
        event_on_call = law.get("event_on_call")
        event_active = (
            (
                event_threshold_x_m is None
                or displacement[0] >= float(event_threshold_x_m) - 1e-15
            )
            and (
                event_on_call is None
                or per_spine_count[geometry.spine_id]
                == int(event_on_call)
            )
        )
        open_contact = bool(
            law.get("unilateral", False)
            and candidate is not None
            and candidate.signed_gap_m > tolerances.gap_m
        )
        scripted_failure = per_spine_count[geometry.spine_id] == law.get(
            "fail_on_call"
        )
        scripted_transition = per_spine_count[geometry.spine_id] == law.get(
            "transition_on_call"
        )
        scripted_reject = per_spine_count[geometry.spine_id] == law.get(
            "reject_on_call"
        )
        result_events = (
            tuple(law.get("events", ())) if event_active else ()
        )
        result_failure = law.get("failure") if event_active else None
        if accepted.physical_state is PhysicalState.FAILED:
            effective_tangent = np.zeros((3, 3), dtype=float)
            force = np.zeros(3, dtype=float)
            physical_state = PhysicalState.FAILED
            result_events = ()
            result_failure = None
        elif scripted_failure:
            force_before = tangent @ displacement
            effective_tangent = np.zeros((3, 3), dtype=float)
            force = np.zeros(3, dtype=float)
            physical_state = PhysicalState.FAILED
            result_failure = FailurePayload(
                failure_object="shaft",
                failure_mode="scripted disconnect",
                criterion="array event regression",
                demand=float(np.linalg.norm(force_before)),
                capacity=1.0,
                margin=1.0 - float(np.linalg.norm(force_before)),
                parameter_sources={"capacity": "array test"},
                continuation_action=ContinuationAction.PERMANENT_REMOVE,
            )
            result_events = (
                Event(
                    event_type=EventType.MATERIAL_FAILURE,
                    sequence=accepted.event_sequence + 1,
                    from_state=PhysicalState.STICK,
                    to_state=PhysicalState.FAILED,
                    spine_id=geometry.spine_id,
                    load_parameter=motion.load_parameter,
                    details={
                        "force_before_N": tuple(
                            float(value) for value in force_before
                        ),
                        **(
                            {
                                "event_fraction": law["failure_fraction"],
                                "located_event": "capacity:shaft",
                            }
                            if "failure_fraction" in law
                            else {}
                        ),
                    },
                ),
            )
        elif scripted_reject:
            effective_tangent = np.zeros((3, 3), dtype=float)
            force = np.zeros(3, dtype=float)
            physical_state = PhysicalState.SEARCH
            result_events = (
                Event(
                    event_type=EventType.CONTACT_REJECT,
                    sequence=accepted.event_sequence + 1,
                    from_state=PhysicalState.CONTACT,
                    to_state=PhysicalState.SEARCH,
                    spine_id=geometry.spine_id,
                    load_parameter=motion.load_parameter,
                ),
            )
        elif scripted_transition:
            effective_tangent = np.asarray(
                law["transition_tangent"], dtype=float
            )
            force = effective_tangent @ displacement
            physical_state = law["transition_state"]
            result_events = (
                Event(
                    event_type=law["transition_event"],
                    sequence=accepted.event_sequence + 1,
                    from_state=law["transition_from_state"],
                    to_state=physical_state,
                    spine_id=geometry.spine_id,
                    load_parameter=motion.load_parameter,
                    details=(
                        {
                            "located_event": "friction",
                            "event_fraction": law["transition_fraction"],
                        }
                        if "transition_fraction" in law
                        else {}
                    ),
                ),
            )
        elif open_contact:
            effective_tangent = np.zeros((3, 3), dtype=float)
            force = np.zeros(3, dtype=float)
            physical_state = PhysicalState.SEARCH
        else:
            effective_tangent = tangent
            force = np.asarray(law.get("bias", (0.0, 0.0, 0.0))) + (
                tangent @ displacement
            )
            physical_state = (
                law["event_physical_state"]
                if (
                    event_active
                    and result_events
                    and "event_physical_state" in law
                )
                else (
                    resident_state
                    if accepted.physical_state is resident_state
                    else law.get("physical_state", PhysicalState.STICK)
                )
            )
        numerical_state = law.get(
            "numerical_state", NumericalState.CONVERGED
        )
        model_state = law.get("model_state", ModelState.CLOSED)
        committable = bool(
            law.get("committable", True)
            if event_active
            else law.get("committable_before_event", True)
        )
        normal = np.asarray(
            (
                candidate.selected_normal
                if candidate is not None and candidate.selected_normal is not None
                else law.get("normal", (1.0, 0.0, 0.0))
            ),
            dtype=float,
        )
        normal /= np.linalg.norm(normal)
        normal_force = float(
            law.get("normal_force_N", float(np.dot(force, normal)))
        )
        contact_submode = law.get("contact_submode")
        spring_branch = law.get("spring_branch", SpringBranch.RIGID)
        root_moment = np.asarray(law.get("root_moment_Nm", (0.0, 0.0, 0.0)))
        root_wrench = Wrench(
            force_N=tuple(float(value) for value in force),
            moment_Nm=tuple(float(value) for value in root_moment),
            frame=geometry.frame,
            reference_point=geometry.root_reference,
            acting_on=geometry.backplate_object,
            exerted_by=geometry.spine_id,
        )
        if scripted_reject:
            proposed = replace(
                accepted,
                physical_state=PhysicalState.SEARCH,
                relative_displacement_m=motion.relative_displacement_m,
                search_cursor=candidate.search_cursor,
                last_load_parameter=motion.load_parameter,
                event_sequence=result_events[-1].sequence,
                revision=accepted.revision + 1,
            )
        elif physical_state in {
            PhysicalState.STICK,
            PhysicalState.SLIP,
            PhysicalState.HARDSTOP,
        } and candidate:
            proposed = replace(
                accepted,
                physical_state=physical_state,
                contact_submode=contact_submode,
                spring_branch=spring_branch,
                candidate_id=candidate.candidate_id,
                contact_point_m=candidate.support_points_m[0],
                contact_normal=candidate.selected_normal,
                relative_displacement_m=motion.relative_displacement_m,
                elastic_displacement_m=motion.relative_displacement_m,
                slip_displacement_m=(0.0, 0.0, 0.0),
                search_cursor=(
                    accepted.search_cursor
                    if accepted.search_cursor is not None
                    else candidate.search_cursor
                ),
                last_load_parameter=motion.load_parameter,
                event_sequence=(
                    result_events[-1].sequence
                    if result_events
                    else accepted.event_sequence
                ),
                revision=accepted.revision + 1,
            )
        else:
            proposed_physical_state = physical_state
            if physical_state in {
                PhysicalState.STICK,
                PhysicalState.SLIP,
                PhysicalState.HARDSTOP,
            }:
                proposed_physical_state = accepted.physical_state
            proposed = replace(
                accepted,
                physical_state=proposed_physical_state,
                relative_displacement_m=motion.relative_displacement_m,
                last_load_parameter=motion.load_parameter,
                event_sequence=(
                    result_events[-1].sequence
                    if result_events
                    else accepted.event_sequence
                ),
                revision=accepted.revision + 1,
            )
        tangential = force - normal_force * normal
        result = SingleSpineResult(
            wall_force_N=tuple(float(value) for value in force),
            root_wrench=root_wrench,
            local_tangent_N_per_m=tuple(
                tuple(float(value) for value in row) for row in effective_tangent
            ),
            physical_state=physical_state,
            contact_submode=contact_submode,
            spring_branch=spring_branch,
            spring_displacement_m=0.0,
            normal_force_N=normal_force,
            tangential_force_N=tuple(float(value) for value in tangential),
            elastic_displacement_m=motion.relative_displacement_m,
            slip_displacement_m=(0.0, 0.0, 0.0),
            model_state=model_state,
            numerical_state=numerical_state,
            margins={},
            capacity_assessments={},
            complementarity_residuals={},
            diagnostics={},
            events=result_events,
            failure=result_failure,
            evaluated_motion=motion,
            assumptions=("linear array test double",),
        )
        if calls is not None:
            calls.append(
                {
                    "spine_id": geometry.spine_id,
                    "accepted_state": accepted.physical_state,
                    "accepted_cursor": accepted.search_cursor,
                    "motion": motion,
                    "candidate": candidate,
                    "result": result,
                }
            )
        return SingleSpineTrial(
            spine_id=geometry.spine_id,
            base_revision=accepted.revision,
            proposed_state=proposed,
            result=result,
            committable=committable,
        )

    monkeypatch.setattr(array_module, "solve_single_spine", fake_solve)
    return counter


def test_n_equals_one_is_the_canonical_single_spine_and_pose_sign() -> None:
    spine = _spine(
        "one",
        root=(0.0, 0.0, 0.0),
        point=(0.0, 0.0, -0.00405),
        normal=(0.0, 0.0, 1.0),
    )
    accepted = ArrayAcceptedState.initial([spine])
    q = (0.0, 0.0, 1e-6, 0.0, 0.0, 0.0)
    control = _control(q=q)
    direct = solve_single_spine(
        spine.geometry,
        spine.material,
        spine.friction,
        spine.suspension,
        accepted.spine_states[0],
        BaseMotion(q[:3], (0.0, 0.0, 0.0), 1.0),
        replace(spine.candidate, signed_gap_m=0.0),
        tolerances=spine.tolerances,
    )

    array_trial = solve_array_equilibrium(
        [spine], accepted, control, load_parameter=1.0
    )

    item = array_trial.result.per_spine[0]
    assert item.single_result == direct.result
    assert item.generalized_wrench.vector == pytest.approx(
        direct.result.root_wrench.vector
    )
    assert array_trial.result.total_wrench.vector == pytest.approx(
        direct.result.root_wrench.vector
    )
    assert array_trial.result.q_C == pytest.approx(q)
    assert array_trial.result.physical_backplate_pose == pytest.approx(
        tuple(-value for value in q)
    )


def test_array_accepted_state_owns_its_local_integrity_boundary() -> None:
    initial = ArrayAcceptedState.initial([_spine("state-boundary")])
    with pytest.raises(ConfigurationError, match="q_C"):
        replace(initial, q_C=(0.0,) * 5)
    with pytest.raises(ConfigurationError, match="q_C"):
        replace(initial, q_C=(0.0, 0.0, 0.0, 0.0, 0.0, np.nan))
    with pytest.raises(ConfigurationError, match="load_parameter"):
        replace(initial, load_parameter=np.inf)
    with pytest.raises(ConfigurationError, match="revision"):
        replace(initial, revision=-1)
    with pytest.raises(ConfigurationError, match="at least one spine"):
        replace(initial, spine_states=())
    with pytest.raises(ConfigurationError, match="IDs must be unique"):
        replace(
            initial,
            spine_states=(initial.spine_states[0], initial.spine_states[0]),
        )


@pytest.mark.parametrize(
    ("control_change", "message"),
    [
        ({"frame": "another-frame"}, "frame does not match control.frame"),
        (
            {"backplate_object": "another-backplate"},
            "backplate_object does not match control.backplate_object",
        ),
    ],
)
def test_array_rejects_geometry_control_semantic_mismatch(
    control_change: Mapping[str, str], message: str
) -> None:
    spine = _spine("semantic-mismatch")
    control = replace(_control(), **control_change)

    with pytest.raises(ConfigurationError, match=message):
        solve_array_equilibrium(
            [spine],
            ArrayAcceptedState.initial([spine]),
            control,
            load_parameter=1.0,
        )


def test_candidate_continuation_requires_one_valid_version_lineage() -> None:
    spine = _spine("candidate-version")
    wrong_terrain = replace(
        _candidate("candidate-version-next", candidate_index=1),
        terrain_version="another-terrain-version",
    )
    with pytest.raises(ConfigurationError, match="share one terrain_version"):
        replace(spine, continuation_candidates=(wrong_terrain,))

    missing_terrain = replace(
        _candidate("candidate-version-empty", candidate_index=1),
        terrain_version=" ",
    )
    with pytest.raises(ConfigurationError, match="non-empty string"):
        replace(spine, continuation_candidates=(missing_terrain,))

    stale_geometry = _candidate("candidate-version-stale", candidate_index=1)
    object.__setattr__(stale_geometry, "geometry_version", "stale-geometry")
    with pytest.raises(ConfigurationError, match="current geometry_version"):
        replace(spine, continuation_candidates=(stale_geometry,))


def test_array_candidates_require_one_terrain_version_but_allow_distinct_tracks() -> None:
    first = _spine("terrain-first")
    second = _spine("terrain-second")
    accepted = ArrayAcceptedState.initial([first, second])

    # Different track/lineage identities are normal for distinct spines.
    solve_array_equilibrium(
        [first, second], accepted, _control(), load_parameter=1.0
    )

    assert second.candidate is not None
    mixed = replace(
        second,
        candidate=replace(
            second.candidate, terrain_version="another-terrain-version"
        ),
    )
    with pytest.raises(ConfigurationError, match="share one terrain_version"):
        solve_array_equilibrium(
            [first, mixed],
            ArrayAcceptedState.initial([first, mixed]),
            _control(),
            load_parameter=1.0,
        )


def test_active_candidate_missing_from_instance_is_data_integrity_error() -> None:
    spine = _spine("missing-active")
    initial = ArrayAcceptedState.initial([spine])
    active_candidate = spine.candidate
    assert active_candidate is not None
    active = replace(
        initial.spine_states[0],
        physical_state=PhysicalState.STICK,
        candidate_id="missing-candidate",
        contact_point_m=active_candidate.support_points_m[0],
        contact_normal=active_candidate.selected_normal,
        search_cursor=active_candidate.search_cursor,
    )
    with pytest.raises(ConfigurationError, match="candidate_id is missing"):
        array_module._candidate_for_state(spine, active)


def test_identical_gaps_and_stiffness_share_load_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spines = [_spine("left"), _spine("right")]
    _install_linear_single_solver(
        monkeypatch,
        {
            spine.spine_id: {"tangent": np.diag([800.0, 0.0, 0.0])}
            for spine in spines
        },
    )
    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(q=(0.002, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    forces = [item.single_result.wall_force_N[0] for item in trial.result.per_spine]
    assert forces == pytest.approx([1.6, 1.6])
    assert trial.result.total_wrench.force_N == pytest.approx((3.2, 0.0, 0.0))


def test_d_i0_and_terrain_gap_are_distinct_and_overclosure_is_elastic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spines = [
        _spine("a", initial_gap_m=0.0010, terrain_gap_m=0.0002),
        _spine("b", initial_gap_m=0.0015, terrain_gap_m=0.0003),
    ]
    calls: list[dict[str, Any]] = []
    _install_linear_single_solver(
        monkeypatch,
        {
            spine.spine_id: {"tangent": np.diag([1000.0, 0.0, 0.0])}
            for spine in spines
        },
        calls,
    )
    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(q=(0.0030, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    first, second = trial.result.per_spine
    assert first.terrain_signed_gap_m == pytest.approx(0.0002)
    assert second.terrain_signed_gap_m == pytest.approx(0.0003)
    assert first.closure_threshold_m == pytest.approx(0.0012)
    assert second.closure_threshold_m == pytest.approx(0.0018)
    assert first.loading_displacement_m == pytest.approx(0.0018)
    assert second.loading_displacement_m == pytest.approx(0.0012)
    assert first.signed_gap_m == pytest.approx(0.0)
    assert second.signed_gap_m == pytest.approx(0.0)
    assert [call["candidate"].signed_gap_m for call in calls] == pytest.approx(
        [0.0, 0.0]
    )
    motions = [call["motion"].relative_displacement_m[0] for call in calls]
    forces = [
        item.single_result.wall_force_N[0] for item in (first, second)
    ]
    assert motions == pytest.approx([0.0018, 0.0012])
    assert forces == pytest.approx([1.8, 1.2])


def test_open_contact_retains_geometric_gap_and_carries_no_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("open", initial_gap_m=0.0010, terrain_gap_m=0.0002)
    calls: list[dict[str, Any]] = []
    _install_linear_single_solver(
        monkeypatch,
        {
            "open": {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "unilateral": True,
            }
        },
        calls,
    )
    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(q=(0.0005, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    item = trial.result.per_spine[0]
    assert item.closure_threshold_m == pytest.approx(0.0012)
    assert item.loading_displacement_m == pytest.approx(-0.0007)
    assert item.signed_gap_m == pytest.approx(0.0007)
    assert calls[0]["candidate"].signed_gap_m == pytest.approx(0.0007)
    assert item.single_result.wall_force_N == pytest.approx((0.0, 0.0, 0.0))


def test_force_control_seeds_initially_open_contacts_and_matches_two_spine_closed_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spines = [
        _spine("one", initial_gap_m=0.001),
        _spine("two", initial_gap_m=0.002),
    ]
    _install_linear_single_solver(
        monkeypatch,
        {
            "one": {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "unilateral": True,
            },
            "two": {
                "tangent": np.diag([2000.0, 0.0, 0.0]),
                "unilateral": True,
            },
        },
    )
    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(required={0: 6.0}, F_ref_N=6.0, L_ref_m=0.01),
        load_parameter=1.0,
    )

    expected_q = (6.0 + 1000.0 * 0.001 + 2000.0 * 0.002) / 3000.0
    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.q_C[0] == pytest.approx(expected_q)
    assert [
        item.single_result.wall_force_N[0] for item in trial.result.per_spine
    ] == pytest.approx(
        [1000.0 * (expected_q - 0.001), 2000.0 * (expected_q - 0.002)]
    )
    assert trial.result.diagnostics.iterations >= 3


def test_real_contact_seed_clears_coarse_force_tolerance() -> None:
    spine = _spine("coarse-tolerance", initial_gap_m=1e-6)
    coarse = replace(
        _single_tolerances(),
        force_N=1e-2,
        friction_N=1e-2,
        spring_N=1e-2,
    )
    spine = replace(spine, tolerances=coarse)

    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(required={0: 0.1}, F_ref_N=1.0, L_ref_m=0.01),
        load_parameter=1.0,
    )

    assert trial.committable
    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.counts.n_active == 1
    assert trial.result.per_spine[0].single_result.physical_state is PhysicalState.STICK


def test_force_control_seeds_candidate_at_exact_zero_gap() -> None:
    spine = _spine("exact-zero-gap")
    accepted = ArrayAcceptedState.initial([spine])

    trial = solve_array_equilibrium(
        [spine],
        accepted,
        _control(required={0: 0.1}, F_ref_N=1.0, L_ref_m=0.01),
        load_parameter=1.0,
    )

    assert trial.committable
    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.counts.n_active == 1
    assert trial.result.total_wrench.force_N[0] == pytest.approx(0.1)
    assert EventType.CONTACT_REJECT not in {
        event.event_type for event in trial.result.events
    }


def test_small_terrain_library_candidate_runs_through_real_array_solver(
    tmp_path: Any,
) -> None:
    recipe = TerrainRecipe(
        seed=7,
        target_rms_height_m=0.0,
        correlation_length_x_m=20e-6,
        correlation_length_y_m=20e-6,
        kernel_truncate_sigma=2.0,
    )
    region = RegionSpec(
        terrain_recipe_id=recipe.terrain_recipe_id,
        origin_x_m=-0.3e-3,
        origin_y_m=-0.2e-3,
        size_x_m=0.6e-3,
        size_y_m=0.4e-3,
        purpose="debug",
    )
    library = TerrainLibrary(tmp_path / "terrain")
    library.generate_region(recipe, region, tile_rows=7)
    track = library.cache_track(
        recipe, region, radius_m=50e-6, y_global_m=0.0
    )
    mapped = library.open_region(
        recipe.terrain_recipe_id, region.region_id, verify_hash=True
    )
    raw_height = np.array(mapped, copy=True)
    mapped._mmap.close()
    valid_mask = np.ones(region.shape, dtype=np.bool_)
    center_index = track.x_global_m.size // 2
    path = SpinePath.from_track(
        track,
        track.envelope_height_m[[center_index]],
        track_indices=np.array([center_index]),
    )
    pose = SpinePose(
        tip_axis=np.array([0.0, 0.0, -1.0]),
        spherical_cap_axial_length_m=25e-6,
        cone_length_m=50e-6,
        rod_radius_m=30e-6,
        exposed_rod_length_m=100e-6,
    )
    candidate, _cursor = query_next_candidate(
        SurfaceState(track, region, raw_height, valid_mask),
        path,
        CandidateCursor(),
        pose,
    )
    assert candidate is not None
    assert candidate.valid
    assert candidate.rod_clearance.collision is False

    spine = replace(
        _spine(
            "terrain-pipeline",
            normal=(0.0, 0.0, -1.0),
            initial_gap_m=1e-6,
        ),
        candidate=candidate,
    )
    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(required={2: 0.1}, F_ref_N=1.0, L_ref_m=0.01),
        load_parameter=1.0,
    )

    assert trial.committable
    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.counts.n_active == 1
    assert trial.result.total_wrench.force_N[2] == pytest.approx(0.1)


def test_partly_engaged_array_can_seed_a_remaining_open_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engaged = _spine("engaged", root=(0.0, -0.01, 0.0))
    open_spine = _spine(
        "open-candidate",
        root=(0.0, 0.01, 0.0),
        initial_gap_m=0.001,
    )
    spines = [engaged, open_spine]
    _install_linear_single_solver(
        monkeypatch,
        {
            spine.spine_id: {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "unilateral": True,
            }
            for spine in spines
        },
    )
    initial = ArrayAcceptedState.initial(spines)
    engaged_candidate = engaged.candidate
    assert engaged_candidate is not None
    engaged_state = replace(
        initial.spine_states[0],
        physical_state=PhysicalState.STICK,
        candidate_id=engaged_candidate.candidate_id,
        contact_point_m=engaged_candidate.support_points_m[0],
        contact_normal=engaged_candidate.selected_normal,
        search_cursor=engaged_candidate.search_cursor,
    )
    accepted = replace(
        initial,
        spine_states=(engaged_state, initial.spine_states[1]),
        revision=1,
    )
    trial = solve_array_equilibrium(
        spines,
        accepted,
        _control(
            required={0: 3.0, 5: 0.02},
            F_ref_N=3.0,
            L_ref_m=0.01,
        ),
        load_parameter=1.0,
    )

    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.rank_status is RankStatus.FULL_RANK
    assert trial.result.range_status is RangeStatus.COMPATIBLE
    assert trial.result.q_C[0] == pytest.approx(0.002)
    assert trial.result.q_C[5] == pytest.approx(0.05)
    assert [
        item.single_result.wall_force_N[0]
        for item in trial.result.per_spine
    ] == pytest.approx([2.5, 0.5])
    assert trial.result.per_spine[1].signed_gap_m == pytest.approx(0.0)
    assert trial.committable


def test_contact_seed_signature_tracks_hardstop_mode_and_spring_branch() -> None:
    spine = _spine("signature")
    candidate = spine.candidate
    assert candidate is not None
    contact = {
        "candidate_id": candidate.candidate_id,
        "contact_point_m": candidate.support_points_m[0],
        "contact_normal": candidate.selected_normal,
    }
    hard_stick = replace(
        SpineAcceptedState.initial(spine.spine_id),
        physical_state=PhysicalState.HARDSTOP,
        contact_submode=PhysicalState.STICK,
        spring_branch=SpringBranch.HARDSTOP,
        **contact,
    )
    hard_slip = replace(
        hard_stick, contact_submode=PhysicalState.SLIP
    )
    assert array_module._active_set_signature(
        [spine], [hard_stick]
    ) != array_module._active_set_signature([spine], [hard_slip])

    stick_rigid = replace(
        hard_stick,
        physical_state=PhysicalState.STICK,
        contact_submode=None,
        spring_branch=SpringBranch.RIGID,
    )
    stick_interior = replace(
        stick_rigid, spring_branch=SpringBranch.INTERIOR
    )
    assert array_module._active_set_signature(
        [spine], [stick_rigid]
    ) != array_module._active_set_signature([spine], [stick_interior])


def test_open_contact_seed_selects_the_wall_side_for_a_required_moment() -> None:
    spines = [
        _spine(
            "upper",
            root=(0.0, 0.01, 0.0),
            normal=(1.0, 0.0, 0.0),
            initial_gap_m=0.0001,
        ),
        _spine(
            "lower",
            root=(0.0, -0.01, 0.0),
            normal=(1.0, 0.0, 0.0),
            initial_gap_m=0.0001,
        ),
    ]

    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(required={5: -0.001}, F_ref_N=0.1, L_ref_m=0.01),
        load_parameter=1.0,
    )

    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.q_C[5] < 0.0
    assert trial.result.total_wrench.moment_Nm[2] == pytest.approx(-0.001)
    assert trial.result.per_spine[0].signed_gap_m == pytest.approx(0.0)
    assert trial.result.per_spine[1].signed_gap_m > 0.0
    assert trial.committable


def test_open_contact_seed_is_built_inside_the_equality_subspace() -> None:
    spines = [
        _spine(
            "upper-equality",
            root=(0.0, 0.01, 0.0),
            normal=(1.0, 0.0, 0.0),
            initial_gap_m=0.0001,
        ),
        _spine(
            "lower-equality",
            root=(0.0, -0.01, 0.0),
            normal=(1.0, 0.0, 0.0),
            initial_gap_m=0.0001,
        ),
    ]
    equality = ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0),)

    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(
            required={0: 0.0, 5: -0.001},
            F_ref_N=0.1,
            L_ref_m=0.01,
            equality_matrix=equality,
        ),
        load_parameter=1.0,
    )

    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert np.asarray(equality) @ np.asarray(trial.result.q_C) == pytest.approx(
        (0.0,)
    )
    assert trial.result.q_C[5] < 0.0
    assert trial.result.total_wrench.moment_Nm[2] == pytest.approx(-0.001)
    assert trial.result.per_spine[0].signed_gap_m == pytest.approx(0.0)
    assert trial.result.per_spine[1].signed_gap_m > 0.0
    assert trial.committable


def test_rank_deficient_search_state_can_seed_another_open_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spines = [
        _spine("upper-reseed", root=(0.0, 0.01, 0.0), initial_gap_m=0.0001),
        _spine("lower-reseed", root=(0.0, -0.01, 0.0), initial_gap_m=0.0001),
    ]
    calls: list[dict[str, Any]] = []
    _install_linear_single_solver(
        monkeypatch,
        {
            spine.spine_id: {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "unilateral": True,
            }
            for spine in spines
        },
        calls,
    )

    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(
            required={0: 2.0, 5: 0.0},
            F_ref_N=2.0,
            L_ref_m=0.01,
        ),
        load_parameter=1.0,
    )

    gap_pairs = [
        tuple(call["candidate"].signed_gap_m for call in calls[index : index + 2])
        for index in range(0, len(calls), 2)
    ]
    gap_tolerance = spines[0].tolerances.gap_m
    one_closed_index = next(
        index
        for index, (upper_gap, lower_gap) in enumerate(gap_pairs)
        if upper_gap <= gap_tolerance and lower_gap > gap_tolerance
    )
    assert any(
        upper_gap <= gap_tolerance and lower_gap <= gap_tolerance
        for upper_gap, lower_gap in gap_pairs[one_closed_index + 1 :]
    )
    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.total_wrench.vector == pytest.approx(
        (2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert trial.committable


def test_two_spine_pose_control_returns_reaction_and_negative_physical_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spines = [
        _spine("one", initial_gap_m=0.001),
        _spine("two", initial_gap_m=0.002),
    ]
    _install_linear_single_solver(
        monkeypatch,
        {
            "one": {"tangent": np.diag([1000.0, 0.0, 0.0])},
            "two": {"tangent": np.diag([2000.0, 0.0, 0.0])},
        },
    )
    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(q=(0.003, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    assert [
        item.single_result.wall_force_N[0] for item in trial.result.per_spine
    ] == pytest.approx([2.0, 2.0])
    assert trial.result.total_wrench.force_N[0] == pytest.approx(4.0)
    assert trial.result.physical_backplate_pose[0] == pytest.approx(-0.003)
    assert trial.result.quasistatic_stability is QuasistaticStability.NO_FREE_MODE


def test_finite_loader_stiffness_closes_the_force_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spines = [_spine("one"), _spine("two")]
    _install_linear_single_solver(
        monkeypatch,
        {
            "one": {"tangent": np.diag([1000.0, 0.0, 0.0])},
            "two": {"tangent": np.diag([2000.0, 0.0, 0.0])},
        },
    )
    loader = np.zeros((6, 6))
    loader[0, 0] = 1000.0
    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(required={0: 6.0}, loader=loader),
        load_parameter=1.0,
    )

    q_x = trial.result.q_C[0]
    assert q_x == pytest.approx(6.0 / 4000.0)
    assert trial.result.total_wrench.force_N[0] == pytest.approx(4.5)
    assert trial.result.total_wrench.force_N[0] + 1000.0 * q_x == pytest.approx(
        6.0
    )


def test_spatial_wrench_and_tangent_include_contact_lever_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spines = [
        _spine("upper", root=(0.0, 0.002, 0.0)),
        _spine("lower", root=(0.0, -0.001, 0.0)),
    ]
    local = np.diag([1000.0, 0.0, 0.0])
    _install_linear_single_solver(
        monkeypatch,
        {spine.spine_id: {"tangent": local} for spine in spines},
    )
    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(q=(0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    expected_tangent = np.zeros((6, 6))
    for spine in spines:
        point = np.asarray(spine.geometry.root_position_m)
        skew = np.array(
            [
                [0.0, -point[2], point[1]],
                [point[2], 0.0, -point[0]],
                [-point[1], point[0], 0.0],
            ]
        )
        B = np.hstack((np.eye(3), -skew))
        expected_tangent += B.T @ local @ B
    assert trial.result.total_wrench.vector == pytest.approx(
        (2.0, 0.0, 0.0, 0.0, 0.0, -0.001)
    )
    assert np.asarray(trial.result.tangent) == pytest.approx(expected_tangent)


def test_scaled_rank_and_range_separate_compatible_and_incompatible_wrenches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lever = 0.2
    spine = _spine(
        "rank",
        root=(0.0, lever, 0.0),
        candidate=False,
    )
    _install_linear_single_solver(
        monkeypatch,
        {"rank": {"tangent": np.diag([500.0, 0.0, 0.0])}},
    )
    accepted = ArrayAcceptedState.initial([spine])
    compatible = solve_array_equilibrium(
        [spine],
        accepted,
        _control(
            required={0: 2.0, 5: -2.0 * lever},
            F_ref_N=2.0,
            L_ref_m=lever,
        ),
        load_parameter=1.0,
    )
    incompatible = solve_array_equilibrium(
        [spine],
        accepted,
        _control(
            required={0: 2.0, 5: 2.0 * lever},
            F_ref_N=2.0,
            L_ref_m=lever,
        ),
        load_parameter=1.0,
    )

    assert compatible.result.rank_status is RankStatus.RANK_DEFICIENT
    assert compatible.result.range_status is RangeStatus.COMPATIBLE
    assert compatible.result.equilibrium_status is EquilibriumStatus.RANK_DEFICIENT
    assert compatible.result.diagnostics.scaled_rank == 1
    assert incompatible.result.rank_status is RankStatus.RANK_DEFICIENT
    assert incompatible.result.range_status is RangeStatus.INCOMPATIBLE
    assert (
        incompatible.result.equilibrium_status
        is EquilibriumStatus.RANGE_INCOMPATIBLE
    )
    assert incompatible.result.diagnostics.range_residual_norm > 0.1


@pytest.mark.parametrize(
    ("eigenvalue", "expected"),
    [
        (2.0, QuasistaticStability.STABLE_CONSERVATIVE),
        (0.0, QuasistaticStability.MARGINAL_CONSERVATIVE),
        (-2.0, QuasistaticStability.UNSTABLE_CONSERVATIVE),
    ],
)
def test_scaled_conservative_stability_positive_zero_and_negative(
    eigenvalue: float,
    expected: QuasistaticStability,
) -> None:
    tangent = np.zeros((6, 6))
    tangent[0, 0] = eigenvalue
    status, minimum, threshold = evaluate_conservative_stability(
        tangent,
        np.array([0]),
        _control(required={0: 0.0}, F_ref_N=2.0, L_ref_m=0.5),
    )

    assert status is expected
    assert minimum == pytest.approx(eigenvalue * 0.5 / 2.0)
    assert threshold is not None and threshold > 0.0


def test_active_equality_can_remove_all_conservative_free_modes() -> None:
    equality = ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0),)
    status, minimum, threshold = evaluate_conservative_stability(
        np.eye(6),
        np.array([0]),
        _control(required={0: 0.0}, equality_matrix=equality),
    )

    assert status is QuasistaticStability.NO_FREE_MODE
    assert minimum is None
    assert threshold is None


def test_conservative_stability_is_not_evaluated_for_nonconverged_local_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("nonconverged-stick")
    _install_linear_single_solver(
        monkeypatch,
        {
            "nonconverged-stick": {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "physical_state": PhysicalState.STICK,
                "numerical_state": NumericalState.NONCONVERGED,
                "committable": False,
            }
        },
    )
    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(q=(0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.numerical_state is NumericalState.NONCONVERGED
    assert trial.result.quasistatic_stability is QuasistaticStability.NOT_EVALUATED
    assert not trial.committable


def test_equilibrium_is_solved_in_the_active_equality_subspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("equality", candidate=False)
    _install_linear_single_solver(
        monkeypatch,
        {"equality": {"tangent": np.diag([1.0, 1.0, 0.0])}},
    )
    equality = ((1.0, -1.0, 0.0, 0.0, 0.0, 0.0),)
    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(
            required={0: 1.0, 1: 2.0},
            equality_matrix=equality,
        ),
        load_parameter=1.0,
    )

    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.q_C[:2] == pytest.approx((1.5, 1.5))
    assert np.asarray(equality) @ np.asarray(trial.result.q_C) == pytest.approx(
        (0.0,)
    )


def test_slip_uses_directional_quasistatic_status_not_symmetric_stability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("slip")
    nonsymmetric = np.array(
        [[1000.0, 200.0, 0.0], [0.0, 500.0, 0.0], [0.0, 0.0, 0.0]]
    )
    _install_linear_single_solver(
        monkeypatch,
        {
            "slip": {
                "tangent": nonsymmetric,
                "physical_state": PhysicalState.SLIP,
                "normal_force_N": 1.0,
            }
        },
    )
    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(q=(0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    assert (
        trial.result.quasistatic_stability
        is QuasistaticStability.DIRECTIONALLY_ADMISSIBLE_QUASISTATIC
    )
    assert trial.result.dynamic_stability is ModelState.OUT_OF_SCOPE
    assert trial.result.diagnostics.minimum_stability_eigenvalue is None


def test_inadmissible_slip_is_not_given_directional_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("slip")
    _install_linear_single_solver(
        monkeypatch,
        {
            "slip": {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "physical_state": PhysicalState.SLIP,
                "normal_force_N": 1.0,
                "numerical_state": NumericalState.NONCONVERGED,
                "committable": False,
            }
        },
    )
    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(q=(0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    assert trial.result.quasistatic_stability is QuasistaticStability.NOT_EVALUATED
    assert trial.result.numerical_state is NumericalState.NONCONVERGED
    assert not trial.committable


@pytest.mark.parametrize(
    ("velocity_y", "expected"),
    [
        (1.0, QuasistaticStability.DIRECTIONALLY_ADMISSIBLE_QUASISTATIC),
        (-1.0, QuasistaticStability.NOT_EVALUATED),
    ],
)
def test_slip_direction_must_be_dissipative(
    monkeypatch: pytest.MonkeyPatch,
    velocity_y: float,
    expected: QuasistaticStability,
) -> None:
    spine = _spine("direction")
    _install_linear_single_solver(
        monkeypatch,
        {
            "direction": {
                "tangent": np.array(
                    [
                        [1000.0, 200.0, 0.0],
                        [0.0, 500.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                "physical_state": PhysicalState.SLIP,
            }
        },
    )
    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(
            q=(0.001, 0.001, 0.0, 0.0, 0.0, 0.0),
            q_rate=(0.0, velocity_y, 0.0, 0.0, 0.0, 0.0),
        ),
        load_parameter=1.0,
    )

    assert trial.result.quasistatic_stability is expected
    assert trial.result.dynamic_stability is ModelState.OUT_OF_SCOPE


def test_real_sliding_contact_opposes_physical_backplate_velocity() -> None:
    spine = _spine("physical-drag-sign")
    q_rate = np.array([0.0, 1e-3, 0.0, 0.0, 0.0, 0.0])
    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(
            q=(1e-6, 1e-4, 0.0, 0.0, 0.0, 0.0),
            q_rate=tuple(q_rate),
        ),
        load_parameter=1.0,
    )

    single = trial.result.per_spine[0].single_result
    assert single.physical_state is PhysicalState.SLIP
    physical_backplate_velocity = -q_rate[:3]
    assert (
        np.dot(single.tangential_force_N, physical_backplate_velocity) < 0.0
    )
    assert (
        trial.result.quasistatic_stability
        is QuasistaticStability.DIRECTIONALLY_ADMISSIBLE_QUASISTATIC
    )


def test_invalid_local_numerical_state_blocks_directional_stability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("slip-residual")
    _install_linear_single_solver(
        monkeypatch,
        {
            "slip-residual": {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "physical_state": PhysicalState.SLIP,
                "numerical_state": NumericalState.INVALID_RESIDUAL,
                "committable": False,
            }
        },
    )
    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(q=(0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.quasistatic_stability is QuasistaticStability.NOT_EVALUATED
    assert trial.result.numerical_state is NumericalState.INVALID_RESIDUAL
    assert not trial.committable


def test_hardstop_with_slip_submode_uses_directional_stability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("hardstop-slip")
    _install_linear_single_solver(
        monkeypatch,
        {
            "hardstop-slip": {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "physical_state": PhysicalState.HARDSTOP,
                "contact_submode": PhysicalState.SLIP,
                "spring_branch": SpringBranch.HARDSTOP,
            }
        },
    )
    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(required={0: 1.0}),
        load_parameter=1.0,
    )

    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert (
        trial.result.quasistatic_stability
        is QuasistaticStability.DIRECTIONALLY_ADMISSIBLE_QUASISTATIC
    )
    assert trial.result.dynamic_stability is ModelState.OUT_OF_SCOPE


def test_directional_slip_requires_an_admissible_one_sided_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("one-sided")
    _install_linear_single_solver(
        monkeypatch,
        {
            "one-sided": {
                "tangent": np.array(
                    [
                        [1000.0, 100.0, 0.0],
                        [0.0, 500.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                "physical_state": PhysicalState.SLIP,
                "normal_force_N": 1.0,
                "committable": False,
            }
        },
    )
    trial = solve_array_equilibrium(
        [spine],
        ArrayAcceptedState.initial([spine]),
        _control(q=(0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    assert trial.result.quasistatic_stability is QuasistaticStability.NOT_EVALUATED
    assert not trial.committable


def test_one_time_slip_event_advances_the_working_state_and_event_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("slip-event")
    calls: list[dict[str, Any]] = []
    stick_tangent = np.diag([1000.0, 0.0, 0.0])
    slip_tangent = np.diag([500.0, 0.0, 0.0])
    _install_linear_single_solver(
        monkeypatch,
        {
            "slip-event": {
                "tangent": stick_tangent,
                "transition_on_call": 2,
                "transition_tangent": slip_tangent,
                "transition_state": PhysicalState.SLIP,
                "transition_event": EventType.SLIP_START,
                "transition_from_state": PhysicalState.STICK,
                "resident_state": PhysicalState.SLIP,
                "resident_tangent": slip_tangent,
            }
        },
        calls,
    )
    accepted = ArrayAcceptedState.initial([spine])
    trial = solve_array_equilibrium(
        [spine],
        accepted,
        _control(required={0: 3.0}),
        load_parameter=1.0,
    )

    event_index = next(
        index
        for index, call in enumerate(calls)
        if call["result"].events
    )
    assert calls[event_index + 1 :]
    assert all(
        call["accepted_state"] is PhysicalState.SLIP
        and call["result"].physical_state is PhysicalState.SLIP
        for call in calls[event_index + 1 :]
    )
    assert trial.result.q_C[0] == pytest.approx(0.006)
    assert trial.result.total_wrench.force_N[0] == pytest.approx(3.0)
    assert trial.result.per_spine[0].single_result.physical_state is PhysicalState.SLIP
    assert [event.event_type for event in trial.result.events] == [
        EventType.SLIP_START
    ]
    assert trial.proposed_state.spine_states[0].physical_state is PhysicalState.SLIP
    assert trial.committable
    committed = commit_array_trial(accepted, trial)
    assert committed.spine_states[0].physical_state is PhysicalState.SLIP


def test_only_the_globally_earliest_spine_event_advances_before_reassembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    early = _spine("early-event")
    late = _spine("late-event")
    spines = [early, late]
    tangent = np.diag([1000.0, 0.0, 0.0])
    calls: list[dict[str, Any]] = []
    laws: dict[str, Mapping[str, Any]] = {}
    for spine, fraction in ((early, 0.2), (late, 0.8)):
        laws[spine.spine_id] = {
            "tangent": tangent,
            "transition_on_call": 2,
            "transition_tangent": tangent,
            "transition_state": PhysicalState.SLIP,
            "transition_event": EventType.SLIP_START,
            "transition_from_state": PhysicalState.STICK,
            "transition_fraction": fraction,
            "resident_state": PhysicalState.SLIP,
            "resident_tangent": tangent,
        }
    _install_linear_single_solver(monkeypatch, laws, calls)
    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(required={0: 2.0}),
        load_parameter=1.0,
    )

    final_states = [
        item.single_result.physical_state for item in trial.result.per_spine
    ]
    assert final_states == [PhysicalState.SLIP, PhysicalState.STICK]
    recorded_events = [
        (event.spine_id, event.details["event_fraction"])
        for event in trial.result.events
    ]
    assert recorded_events == [
        ("early-event", pytest.approx(0.2))
    ]
    assert trial.result.q_C[0] == pytest.approx(0.001)
    assert trial.result.total_wrench.force_N[0] == pytest.approx(2.0)
    assert trial.proposed_state.spine_states[0].physical_state is PhysicalState.SLIP
    assert trial.proposed_state.spine_states[1].physical_state is PhysicalState.STICK
    assert trial.committable


def test_later_permanent_failure_is_not_released_with_the_earliest_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    early = _spine("early-failure")
    late = _spine("late-failure")
    spines = [early, late]
    tangent = np.diag([1000.0, 0.0, 0.0])
    _install_linear_single_solver(
        monkeypatch,
        {
            "early-failure": {
                "tangent": tangent,
                "fail_on_call": 2,
                "failure_fraction": 0.2,
            },
            "late-failure": {
                "tangent": tangent,
                "fail_on_call": 2,
                "failure_fraction": 0.8,
            },
        },
    )
    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(required={0: 2.0}),
        load_parameter=1.0,
    )

    assert [
        item.single_result.physical_state for item in trial.result.per_spine
    ] == [PhysicalState.FAILED, PhysicalState.STICK]
    assert [event.spine_id for event in trial.result.events] == [
        "early-failure"
    ]
    assert [item["spine_id"] for item in trial.result.released_wrenches] == [
        "early-failure"
    ]
    assert len(trial.result.rebalance_predictions) == 1
    prediction = trial.result.rebalance_predictions[0]
    assert prediction["failed_spine_ids"] == ("early-failure",)
    assert prediction["released_force_N"] == pytest.approx((1.0, 0.0, 0.0))
    assert prediction["delta_q_C"] == pytest.approx(
        (0.001, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert trial.result.q_C[0] == pytest.approx(0.002)
    assert trial.result.total_wrench.force_N[0] == pytest.approx(2.0)
    assert trial.committable


def test_contact_reject_advances_candidate_cursor_before_re_equilibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("candidate-sequence", terrain_gap_m=0.001)
    next_candidate = _candidate(
        "candidate-sequence-next",
        terrain_gap_m=0.002,
        candidate_index=1,
    )
    spine = replace(spine, continuation_candidates=(next_candidate,))
    assert spine.candidate is not None
    next_cursor = spine.candidate.search_cursor
    calls: list[dict[str, Any]] = []
    _install_linear_single_solver(
        monkeypatch,
        {
            "candidate-sequence": {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "unilateral": True,
                "reject_on_call": 2,
            }
        },
        calls,
    )
    accepted = ArrayAcceptedState.initial([spine])
    trial = solve_array_equilibrium(
        [spine],
        accepted,
        _control(required={0: 3.0}),
        load_parameter=1.0,
    )

    reject_index = next(
        index
        for index, call in enumerate(calls)
        if any(
            event.event_type is EventType.CONTACT_REJECT
            for event in call["result"].events
        )
    )
    assert calls[reject_index + 1 :]
    assert all(
        call["accepted_cursor"] == next_cursor
        and call["candidate"].candidate_id == next_candidate.candidate_id
        for call in calls[reject_index + 1 :]
    )
    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.q_C[0] == pytest.approx(0.005)
    assert trial.result.total_wrench.force_N[0] == pytest.approx(3.0)
    assert [event.event_type for event in trial.result.events] == [
        EventType.CONTACT_REJECT,
    ]
    assert trial.proposed_state.spine_states[0].search_cursor == next_cursor
    assert trial.committable
    committed = commit_array_trial(accepted, trial)
    assert committed.spine_states[0].search_cursor == next_cursor


def test_parameter_unclosed_rejection_survives_candidate_exhaustion() -> None:
    spine = _spine("unclosed-candidate")
    assert spine.candidate is not None
    unresolved = replace(
        spine.candidate,
        geometry_uncertain=True,
        rod_clearance=RodClearanceResult(
            collision=None,
            minimum_clearance_m=None,
            sample_count=0,
            model_warning=("missing complete body geometry",),
        ),
    )
    spine = replace(spine, candidate=unresolved)
    accepted = ArrayAcceptedState.initial([spine])
    loader = np.zeros((6, 6), dtype=float)
    loader[0, 0] = 1000.0

    trial = solve_array_equilibrium(
        [spine],
        accepted,
        _control(required={0: 0.1}, loader=loader),
        load_parameter=1.0,
    )

    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.model_state is ModelState.PARAMETER_UNCLOSED
    assert trial.result.counts.n_active == 0
    assert [event.details.get("reason") for event in trial.result.events] == [
        "rod_clearance_unclosed"
    ]
    assert not trial.committable
    assert trial.proposed_state == accepted


def test_permanent_failure_cannot_revive_during_same_load_redistribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = _spine("failed-event", root=(0.0, 0.01, 0.0))
    survivor = _spine("survivor")
    spines = [failed, survivor]
    calls: list[dict[str, Any]] = []
    _install_linear_single_solver(
        monkeypatch,
        {
            "failed-event": {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "fail_on_call": 2,
            },
            "survivor": {"tangent": np.diag([1000.0, 0.0, 0.0])},
        },
        calls,
    )
    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(required={0: 3.0}),
        load_parameter=1.0,
    )

    failed_calls = [
        call for call in calls if call["spine_id"] == "failed-event"
    ]
    event_index = next(
        index
        for index, call in enumerate(failed_calls)
        if call["result"].physical_state is PhysicalState.FAILED
    )
    assert failed_calls[event_index + 1 :]
    assert all(
        call["accepted_state"] is PhysicalState.FAILED
        and call["result"].physical_state is PhysicalState.FAILED
        for call in failed_calls[event_index + 1 :]
    )
    assert (
        trial.result.per_spine[0].single_result.physical_state
        is PhysicalState.FAILED
    )
    assert trial.result.per_spine[0].single_result.wall_force_N == pytest.approx(
        (0.0, 0.0, 0.0)
    )
    assert trial.result.per_spine[1].single_result.wall_force_N[0] == pytest.approx(
        3.0
    )
    assert trial.result.q_C[0] == pytest.approx(0.003)
    assert trial.result.total_wrench.force_N[0] == pytest.approx(3.0)
    assert [event.event_type for event in trial.result.events] == [
        EventType.MATERIAL_FAILURE
    ]
    assert len(trial.result.released_wrenches) == 1
    released = trial.result.released_wrenches[0]
    assert released["force_N"] == pytest.approx((1.5, 0.0, 0.0))
    assert released["moment_Nm"] == pytest.approx((0.0, 0.0, -0.015))
    assert len(trial.result.rebalance_predictions) == 1
    prediction = trial.result.rebalance_predictions[0]
    assert prediction["equation"] == "(K_L+K_R)delta_q_C=f_R_gen_minus"
    assert prediction["failed_spine_ids"] == ("failed-event",)
    assert prediction["released_force_N"] == pytest.approx((1.5, 0.0, 0.0))
    assert prediction["released_moment_Nm"] == pytest.approx(
        (0.0, 0.0, -0.015)
    )
    assert prediction["rank_status"] == RankStatus.FULL_RANK.value
    assert prediction["range_status"] == RangeStatus.COMPATIBLE.value
    assert prediction["delta_q_C"] == pytest.approx(
        (0.0015, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert all(
        call["motion"].load_parameter == pytest.approx(1.0) for call in calls
    )
    assert trial.committable


def test_equation_9_29_redistributes_an_eccentric_failure_force_and_moment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = _spine("edge", root=(0.0, 0.02, 0.0), candidate=False)
    upper = _spine("upper", root=(0.0, 0.01, 0.0), candidate=False)
    lower = _spine("lower", root=(0.0, -0.01, 0.0), candidate=False)
    spines = [failed, upper, lower]
    local = np.diag([1000.0, 0.0, 0.0])
    _install_linear_single_solver(
        monkeypatch,
        {
            "edge": {"tangent": local, "fail_on_call": 2},
            "upper": {"tangent": local},
            "lower": {"tangent": local},
        },
    )
    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(
            required={0: 3.0, 5: -0.02},
            F_ref_N=3.0,
            L_ref_m=0.01,
        ),
        load_parameter=1.0,
    )

    assert trial.result.equilibrium_status is EquilibriumStatus.SOLVED
    assert trial.result.q_C[0] == pytest.approx(0.0015)
    assert trial.result.q_C[5] == pytest.approx(-0.1)
    assert trial.result.total_wrench.vector == pytest.approx(
        (3.0, 0.0, 0.0, 0.0, 0.0, -0.02)
    )
    assert [
        item.single_result.wall_force_N[0]
        for item in trial.result.per_spine
    ] == pytest.approx([0.0, 2.5, 0.5])
    prediction = trial.result.rebalance_predictions[0]
    assert prediction["released_force_N"] == pytest.approx((1.0, 0.0, 0.0))
    assert prediction["released_moment_Nm"] == pytest.approx(
        (0.0, 0.0, -0.02)
    )
    assert prediction["delta_q_C"] == pytest.approx(
        (0.0005, 0.0, 0.0, 0.0, 0.0, -0.1)
    )
    assert prediction["rank_status"] == RankStatus.FULL_RANK.value
    assert prediction["range_status"] == RangeStatus.COMPATIBLE.value
    assert trial.committable


def test_model_limit_trial_cannot_commit_or_mutate_accepted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("failed")
    failure = FailurePayload(
        failure_object="surface",
        failure_mode="unclosed topology",
        criterion="test boundary",
        demand=2.0,
        capacity=1.0,
        margin=-1.0,
        parameter_sources={"capacity": "array test"},
        continuation_action=ContinuationAction.STOP_MODEL_LIMIT,
    )
    model_limit_event = Event(
        event_type=EventType.MATERIAL_FAILURE,
        sequence=1,
        from_state=PhysicalState.STICK,
        to_state=None,
        spine_id=spine.spine_id,
        load_parameter=1.0,
    )
    _install_linear_single_solver(
        monkeypatch,
        {
            "failed": {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "failure": failure,
                "events": (model_limit_event,),
                "committable": False,
            }
        },
    )
    accepted = ArrayAcceptedState.initial([spine])
    trial = solve_array_equilibrium(
        [spine],
        accepted,
        _control(q=(0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    assert trial.result.equilibrium_status is EquilibriumStatus.MODEL_LIMIT
    assert trial.result.model_state is ModelState.OUT_OF_SCOPE
    assert trial.result.events == (model_limit_event,)
    assert not trial.committable
    assert accepted == ArrayAcceptedState.initial([spine])
    assert trial.proposed_state.spine_states == accepted.spine_states
    with pytest.raises(ConfigurationError, match="not a completed admissible"):
        commit_array_trial(accepted, trial)
    assert accepted == ArrayAcceptedState.initial([spine])


def test_model_limit_reassembles_every_spine_at_global_earliest_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    early = _spine("early-limit")
    late = _spine("late-slip")
    spines = [early, late]
    tangent = np.diag([1000.0, 0.0, 0.0])
    failure = FailurePayload(
        failure_object="surface",
        failure_mode="unclosed topology",
        criterion="test boundary",
        demand=2.0,
        capacity=1.0,
        margin=-1.0,
        parameter_sources={"capacity": "array test"},
        continuation_action=ContinuationAction.STOP_MODEL_LIMIT,
    )
    early_event = Event(
        event_type=EventType.MATERIAL_FAILURE,
        sequence=1,
        from_state=PhysicalState.STICK,
        to_state=None,
        spine_id=early.spine_id,
        load_parameter=0.2,
        details={"event_fraction": 0.2},
    )
    late_event = Event(
        event_type=EventType.SLIP_START,
        sequence=1,
        from_state=PhysicalState.STICK,
        to_state=PhysicalState.SLIP,
        spine_id=late.spine_id,
        load_parameter=0.8,
        details={"event_fraction": 0.8},
    )
    _install_linear_single_solver(
        monkeypatch,
        {
            early.spine_id: {
                "tangent": tangent,
                "failure": failure,
                "events": (early_event,),
                "committable": False,
                "event_threshold_x_m": 0.2e-3,
            },
            late.spine_id: {
                "tangent": tangent,
                "events": (late_event,),
                "event_physical_state": PhysicalState.SLIP,
                "event_threshold_x_m": 0.8e-3,
            },
        },
    )
    accepted = ArrayAcceptedState.initial(spines)
    trial = solve_array_equilibrium(
        spines,
        accepted,
        _control(q=(1e-3, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
    )

    assert trial.result.q_C[0] == pytest.approx(0.2e-3)
    assert [
        item.single_result.evaluated_motion.relative_displacement_m[0]
        for item in trial.result.per_spine
    ] == pytest.approx([0.2e-3, 0.2e-3])
    assert [
        item.single_result.physical_state
        for item in trial.result.per_spine
    ] == [PhysicalState.STICK, PhysicalState.STICK]
    assert trial.result.total_wrench.force_N[0] == pytest.approx(0.4)
    assert trial.result.events == (early_event,)
    assert trial.result.equilibrium_status is EquilibriumStatus.MODEL_LIMIT
    assert trial.result.model_state is ModelState.OUT_OF_SCOPE
    assert (
        trial.result.quasistatic_stability
        is QuasistaticStability.NOT_EVALUATED
    )
    assert trial.result.numerical_state is NumericalState.NONCONVERGED
    assert not trial.committable
    assert trial.proposed_state == accepted


def test_model_limit_fraction_uses_the_latest_same_load_event_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("same-load-limit")
    failure = FailurePayload(
        failure_object="surface",
        failure_mode="unclosed topology",
        criterion="same-load boundary",
        demand=2.0,
        capacity=1.0,
        margin=-1.0,
        parameter_sources={"capacity": "array test"},
        continuation_action=ContinuationAction.STOP_MODEL_LIMIT,
    )
    limit_event = Event(
        event_type=EventType.MATERIAL_FAILURE,
        sequence=2,
        from_state=PhysicalState.SLIP,
        to_state=None,
        spine_id=spine.spine_id,
        load_parameter=1.0,
        details={"event_fraction": 0.5},
    )
    calls: list[dict[str, Any]] = []
    _install_linear_single_solver(
        monkeypatch,
        {
            spine.spine_id: {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "transition_on_call": 2,
                "transition_tangent": np.diag([500.0, 0.0, 0.0]),
                "transition_state": PhysicalState.SLIP,
                "transition_event": EventType.SLIP_START,
                "transition_from_state": PhysicalState.STICK,
                "transition_fraction": 1.0,
                "resident_state": PhysicalState.SLIP,
                "resident_tangent": np.diag([500.0, 0.0, 0.0]),
                "failure": failure,
                "events": (limit_event,),
                "event_on_call": 4,
                "committable": False,
                "committable_before_event": True,
            }
        },
        calls,
    )
    accepted = ArrayAcceptedState.initial([spine])
    trial = solve_array_equilibrium(
        [spine],
        accepted,
        _control(required={0: 1.0}),
        load_parameter=1.0,
    )

    assert [
        call["motion"].relative_displacement_m[0] for call in calls
    ] == pytest.approx([0.0, 1e-3, 1e-3, 2e-3, 1.5e-3])
    assert trial.result.q_C[0] == pytest.approx(1.5e-3)
    assert (
        trial.result.per_spine[0]
        .single_result.evaluated_motion.relative_displacement_m[0]
        == pytest.approx(1.5e-3)
    )
    assert [event.event_type for event in trial.result.events] == [
        EventType.SLIP_START,
        EventType.MATERIAL_FAILURE,
    ]
    assert trial.result.events[1] == limit_event
    assert trial.result.equilibrium_status is EquilibriumStatus.MODEL_LIMIT
    assert trial.proposed_state == accepted
    assert not trial.committable


def test_same_load_event_order_uses_global_q_not_local_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_case(
        a_fraction: float, b_fraction: float
    ) -> tuple[Any, dict[str, list[float]]]:
        a = _spine("a")
        b = _spine("b")
        spines = [a, b]
        failure_a = FailurePayload(
            failure_object="surface",
            failure_mode="a boundary",
            criterion="global event order",
            demand=2.0,
            capacity=1.0,
            margin=-1.0,
            parameter_sources={"capacity": "array test"},
            continuation_action=ContinuationAction.STOP_MODEL_LIMIT,
        )
        failure_b = replace(failure_a, failure_mode="b boundary")
        event_a = Event(
            event_type=EventType.MATERIAL_FAILURE,
            sequence=2,
            from_state=PhysicalState.SLIP,
            to_state=None,
            spine_id=a.spine_id,
            load_parameter=1.0,
            details={"event_fraction": a_fraction},
        )
        event_b = Event(
            event_type=EventType.MATERIAL_FAILURE,
            sequence=1,
            from_state=PhysicalState.STICK,
            to_state=None,
            spine_id=b.spine_id,
            load_parameter=1.0,
            details={"event_fraction": b_fraction},
        )
        calls: list[dict[str, Any]] = []
        _install_linear_single_solver(
            monkeypatch,
            {
                a.spine_id: {
                    "tangent": np.diag([750.0, 0.0, 0.0]),
                    "transition_on_call": 2,
                    "transition_tangent": np.diag([250.0, 0.0, 0.0]),
                    "transition_state": PhysicalState.SLIP,
                    "transition_event": EventType.SLIP_START,
                    "transition_from_state": PhysicalState.STICK,
                    "transition_fraction": 1.0,
                    "resident_state": PhysicalState.SLIP,
                    "resident_tangent": np.diag([250.0, 0.0, 0.0]),
                    "failure": failure_a,
                    "events": (event_a,),
                    "event_on_call": 4,
                    "committable": False,
                    "committable_before_event": True,
                },
                b.spine_id: {
                    "tangent": np.diag([250.0, 0.0, 0.0]),
                    "failure": failure_b,
                    "events": (event_b,),
                    "event_on_call": 4,
                    "committable": False,
                    "committable_before_event": True,
                },
            },
            calls,
        )
        accepted = ArrayAcceptedState.initial(spines)
        trial = solve_array_equilibrium(
            spines,
            accepted,
            _control(required={0: 1.0}),
            load_parameter=1.0,
        )
        motions = {
            spine_id: [
                call["motion"].relative_displacement_m[0]
                for call in calls
                if call["spine_id"] == spine_id
            ]
            for spine_id in (a.spine_id, b.spine_id)
        }
        return trial, motions

    simultaneous, simultaneous_motions = run_case(0.5, 0.75)
    assert simultaneous.result.q_C[0] == pytest.approx(1.5e-3)
    assert simultaneous_motions == {
        "a": pytest.approx([0.0, 1e-3, 1e-3, 2e-3, 1.5e-3]),
        "b": pytest.approx([0.0, 1e-3, 1e-3, 2e-3, 1.5e-3]),
    }
    assert [
        (event.spine_id, event.event_type)
        for event in simultaneous.result.events
    ] == [
        ("a", EventType.SLIP_START),
        ("a", EventType.MATERIAL_FAILURE),
        ("b", EventType.MATERIAL_FAILURE),
    ]

    b_first, _motions = run_case(0.4, 0.6)
    assert b_first.result.q_C[0] == pytest.approx(1.2e-3)
    assert [
        (event.spine_id, event.event_type)
        for event in b_first.result.events
    ] == [
        ("a", EventType.SLIP_START),
        ("b", EventType.MATERIAL_FAILURE),
    ]


def test_nonconverged_iteration_limit_returns_one_self_consistent_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("iteration-limit")
    _install_linear_single_solver(
        monkeypatch,
        {
            "iteration-limit": {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "transition_on_call": 2,
                "transition_tangent": np.diag([500.0, 0.0, 0.0]),
                "transition_state": PhysicalState.SLIP,
                "transition_event": EventType.SLIP_START,
                "transition_from_state": PhysicalState.STICK,
                "transition_fraction": 0.5,
            }
        },
    )
    accepted = ArrayAcceptedState.initial([spine])
    trial = solve_array_equilibrium(
        [spine],
        accepted,
        _control(required={0: 1.0}),
        load_parameter=1.0,
        tolerances=ArrayTolerances(maximum_iterations=1),
    )

    q_x = trial.result.q_C[0]
    item = trial.result.per_spine[0]
    evaluated_x = item.single_result.evaluated_motion.relative_displacement_m[0]
    force_x = item.single_result.wall_force_N[0]
    assert trial.result.equilibrium_status is EquilibriumStatus.NONCONVERGED
    assert trial.result.numerical_state is NumericalState.NONCONVERGED
    assert evaluated_x == pytest.approx(q_x)
    assert force_x == pytest.approx(1000.0 * q_x)
    assert trial.result.total_wrench.force_N[0] == pytest.approx(force_x)
    assert trial.result.events == ()
    assert item.single_result.events == ()
    assert item.single_result.physical_state is PhysicalState.STICK
    assert trial.proposed_state == accepted
    assert not trial.committable


def test_final_newton_update_still_stops_at_the_first_model_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("final-limit")
    failure = FailurePayload(
        failure_object="surface",
        failure_mode="unclosed topology",
        criterion="final Newton boundary",
        demand=2.0,
        capacity=1.0,
        margin=-1.0,
        parameter_sources={"capacity": "array test"},
        continuation_action=ContinuationAction.STOP_MODEL_LIMIT,
    )
    event = Event(
        event_type=EventType.MATERIAL_FAILURE,
        sequence=1,
        from_state=PhysicalState.STICK,
        to_state=None,
        spine_id=spine.spine_id,
        load_parameter=1.0,
        details={"event_fraction": 0.5},
    )
    calls: list[dict[str, Any]] = []
    _install_linear_single_solver(
        monkeypatch,
        {
            spine.spine_id: {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "failure": failure,
                "events": (event,),
                "event_threshold_x_m": 0.5e-3,
                "committable": False,
                "committable_before_event": True,
            }
        },
        calls,
    )
    accepted = ArrayAcceptedState.initial([spine])
    trial = solve_array_equilibrium(
        [spine],
        accepted,
        _control(required={0: 1.0}),
        load_parameter=1.0,
        tolerances=ArrayTolerances(maximum_iterations=1),
    )

    assert [
        call["motion"].relative_displacement_m[0] for call in calls
    ] == pytest.approx([0.0, 1e-3, 0.5e-3])
    assert trial.result.q_C[0] == pytest.approx(0.5e-3)
    assert trial.result.equilibrium_status is EquilibriumStatus.MODEL_LIMIT
    assert trial.result.model_state is ModelState.OUT_OF_SCOPE
    assert trial.result.events == (event,)
    assert trial.proposed_state == accepted
    assert not trial.committable


def test_iteration_exhaustion_does_not_duplicate_the_last_selected_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine = _spine("last-event")
    event = Event(
        event_type=EventType.SLIP_START,
        sequence=1,
        from_state=PhysicalState.STICK,
        to_state=PhysicalState.SLIP,
        spine_id=spine.spine_id,
        load_parameter=1.0,
        details={"event_fraction": 0.5},
    )
    _install_linear_single_solver(
        monkeypatch,
        {
            spine.spine_id: {
                "tangent": np.diag([1000.0, 0.0, 0.0]),
                "events": (event,),
                "event_physical_state": PhysicalState.SLIP,
            }
        },
    )
    accepted = ArrayAcceptedState.initial([spine])
    trial = solve_array_equilibrium(
        [spine],
        accepted,
        _control(q=(1e-3, 0.0, 0.0, 0.0, 0.0, 0.0)),
        load_parameter=1.0,
        tolerances=ArrayTolerances(maximum_iterations=1),
    )

    assert trial.result.equilibrium_status is EquilibriumStatus.NONCONVERGED
    assert trial.result.events == (event,)
    assert trial.proposed_state == accepted
    assert not trial.committable


def test_3600_spine_assembly_remains_n_by_six_and_calls_each_spine_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 3600
    spines = [
        _spine(
            f"s{index}",
            root=(0.0, 1e-4 * (index % 60), 1e-4 * (index // 60)),
            candidate=False,
        )
        for index in range(count)
    ]
    counter = _install_linear_single_solver(monkeypatch, {})
    trial = solve_array_equilibrium(
        spines,
        ArrayAcceptedState.initial(spines),
        _control(),
        load_parameter=1.0,
    )

    assert counter["count"] == count
    assert trial.result.diagnostics.assembled_spine_count == count
    assert trial.result.diagnostics.largest_dense_matrix_shape == (6, 6)
    assert len(trial.result.per_spine) == count


def test_mixed_control_rejects_pose_and_wrench_on_the_same_component() -> None:
    with pytest.raises(ConfigurationError, match="requires pose only"):
        MixedControl(
            modes=(ControlMode.PRESCRIBED_POSE,) * 6,
            prescribed_q_C=(0.0,) * 6,
            required_wrench=(1.0, None, None, None, None, None),
            loader_stiffness=tuple(tuple(row) for row in np.zeros((6, 6))),
            initial_q_C=(0.0,) * 6,
            q_rate_C=(0.0,) * 6,
            F_ref_N=1.0,
            L_ref_m=1.0,
        )
