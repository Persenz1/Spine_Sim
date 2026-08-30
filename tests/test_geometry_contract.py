from __future__ import annotations

import numpy as np
import pytest

from spine_sim.geometry import (
    CandidateCursor,
    SpinePath,
    SpinePose,
    SurfaceState,
    drive_candidate_path,
    query_next_candidate,
)
from spine_sim.terrain.envelope import compute_track_geometry
from spine_sim.terrain.models import RegionSpec


def _region() -> RegionSpec:
    return RegionSpec(
        terrain_recipe_id="terrain_recipe_geometry_contract",
        origin_x_m=-0.5e-3,
        origin_y_m=-0.25e-3,
        size_x_m=1.0e-3,
        size_y_m=0.5e-3,
        resolution_x_m=10e-6,
        resolution_y_m=10e-6,
        purpose="debug",
    )


def _two_feature_track():
    region = _region()
    height = np.full(region.shape, -100e-6, dtype=np.float64)
    row = region.shape[0] // 2
    column = region.shape[1] // 2
    height[row, column - 2] = 40e-6
    height[row, column + 2] = 40e-6
    track = compute_track_geometry(
        height,
        region,
        radius_m=50e-6,
        y_global_m=0.0,
        near_tie_tolerance_m=0.0,
    )
    return region, height, track, column


def test_first_encounter_cursor_and_driver_preserve_feature_order() -> None:
    _region_spec, _height, track, column = _two_feature_track()
    indices = np.array([column - 4, column - 3, column, column + 3])
    z = track.envelope_height_m[indices] - 1e-6
    z[0] = track.envelope_height_m[indices[0]] + 1e-6
    positions = np.arange(indices.size, dtype=np.float64) * 10e-6
    path = SpinePath.from_track(
        track,
        z,
        track_indices=indices,
        path_position_m=positions,
    )
    pose = SpinePose(tip_axis=np.array([0.0, 0.0, -1.0]))
    state = SurfaceState(track)

    first, cursor = query_next_candidate(
        state, path, CandidateCursor(), pose
    )
    assert first is not None
    assert np.isclose(first.path_position_m, 5e-6)
    assert first.signed_gap_m == 0.0
    assert first.search_cursor == cursor
    second, cursor = query_next_candidate(
        state, path, first.search_cursor, pose
    )
    assert second is not None
    assert first.feature_id != second.feature_id
    independent_first, _ = query_next_candidate(
        state, path, CandidateCursor(), pose
    )
    assert independent_first is not None
    assert independent_first.candidate_id == first.candidate_id
    remaining = list(drive_candidate_path(state, path, pose, cursor))
    all_candidates = [first, second, *remaining]
    assert [item.candidate_index for item in all_candidates] == list(
        range(len(all_candidates))
    )
    assert [item.path_position_m for item in all_candidates] == sorted(
        item.path_position_m for item in all_candidates
    )


def test_near_tie_keeps_support_set_and_has_no_selected_physical_normal() -> None:
    _region_spec, _height, track, column = _two_feature_track()
    path = SpinePath.from_track(
        track,
        track.envelope_height_m[[column]],
        track_indices=np.array([column]),
    )
    candidate, cursor = query_next_candidate(
        SurfaceState(track),
        path,
        CandidateCursor(),
        SpinePose(tip_axis=np.array([1.0, 0.0, 0.0])),
    )
    assert candidate is not None
    assert cursor.candidate_index == 1
    assert candidate.near_tie
    assert candidate.support_points_m.shape == (2, 3)
    assert candidate.selected_normal is None
    assert candidate.tangent_basis is None
    assert candidate.forward_cap_valid is False
    assert candidate.valid is False
    np.testing.assert_array_equal(
        candidate.support_points_m,
        track.support_points_m[column, :2],
    )


def test_query_rejects_support_interpolation_between_track_nodes() -> None:
    _region_spec, _height, track, column = _two_feature_track()
    centers = np.array(
        [
            [
                track.x_global_m[column] + 0.5 * track.resolution_m,
                track.y_global_m,
                track.envelope_height_m[column],
            ]
        ]
    )
    path = SpinePath(
        path_position_m=np.array([0.0]),
        sphere_centers_m=centers,
        track_indices=np.array([column]),
    )
    with pytest.raises(ValueError, match="support interpolation is forbidden"):
        query_next_candidate(
            SurfaceState(track),
            path,
            CandidateCursor(),
            SpinePose(tip_axis=np.array([0.0, 0.0, -1.0])),
        )


def test_gap_crossing_does_not_interpolate_across_a_feature_switch() -> None:
    _region_spec, _height, track, column = _two_feature_track()
    indices = np.array([column, column + 1])
    assert np.all(track.feature_switch_flag[indices])
    assert not np.array_equal(
        track.support_feature_indices_yx[indices[0], 0],
        track.support_feature_indices_yx[indices[1], 0],
    )
    center_z = track.envelope_height_m[indices] + np.array([1e-6, -1e-6])
    path = SpinePath.from_track(
        track,
        center_z,
        track_indices=indices,
        path_position_m=np.array([0.0, 10e-6]),
    )

    candidate, _ = query_next_candidate(
        SurfaceState(track),
        path,
        CandidateCursor(),
        SpinePose(tip_axis=np.array([0.0, 0.0, -1.0])),
    )

    assert candidate is not None
    assert candidate.path_position_m == pytest.approx(10e-6)
    assert candidate.signed_gap_m == pytest.approx(-1e-6)
    np.testing.assert_allclose(candidate.sphere_center_m, path.sphere_centers_m[1])


def test_gap_crossing_interpolates_on_same_feature_next_to_switch() -> None:
    _region_spec, _height, track, column = _two_feature_track()
    indices = np.array([column + 1, column + 2])
    assert track.feature_switch_flag[indices[0]]
    assert not track.feature_switch_flag[indices[1]]
    np.testing.assert_array_equal(
        track.support_feature_indices_yx[indices[0], 0],
        track.support_feature_indices_yx[indices[1], 0],
    )
    assert not np.any(track.near_tie_flag[indices])
    center_z = track.envelope_height_m[indices] + np.array([1e-6, -1e-6])
    path = SpinePath.from_track(
        track,
        center_z,
        track_indices=indices,
        path_position_m=np.array([0.0, 10e-6]),
    )

    candidate, _ = query_next_candidate(
        SurfaceState(track),
        path,
        CandidateCursor(),
        SpinePose(tip_axis=np.array([0.0, 0.0, -1.0])),
    )

    assert candidate is not None
    assert candidate.path_position_m == pytest.approx(5e-6)
    assert candidate.signed_gap_m == pytest.approx(0.0)
    np.testing.assert_allclose(
        candidate.sphere_center_m,
        0.5 * (path.sphere_centers_m[0] + path.sphere_centers_m[1]),
    )


def test_surface_state_rejects_raw_clearance_inputs_from_another_source() -> None:
    region, height, track, _column = _two_feature_track()
    source_valid = np.ones(region.shape, dtype=np.bool_)

    SurfaceState(track, region, height, source_valid)
    with pytest.raises(ValueError, match="source_data_sha256"):
        SurfaceState(track, region, height + 1e-6, source_valid)

    mismatched_mask = source_valid.copy()
    mismatched_mask[0, 0] = False
    with pytest.raises(ValueError, match="source_valid_mask_sha256"):
        SurfaceState(track, region, height, mismatched_mask)


def test_candidate_reports_gap_bounds_and_unknown_incomplete_body() -> None:
    region = _region()
    height = np.zeros(region.shape, dtype=np.float64)
    lower = height - 1e-6
    upper = height + 2e-6
    track = compute_track_geometry(
        height,
        region,
        radius_m=50e-6,
        y_global_m=0.0,
        source_uncertain_mask=np.ones(region.shape, dtype=np.bool_),
        height_lower_bound_m=lower,
        height_upper_bound_m=upper,
    )
    index = region.shape[1] // 2
    path = SpinePath.from_track(
        track,
        track.envelope_height_m[[index]],
        track_indices=np.array([index]),
    )
    candidate, _ = query_next_candidate(
        SurfaceState(track),
        path,
        CandidateCursor(),
        SpinePose(tip_axis=np.array([0.0, 0.0, -1.0])),
    )
    assert candidate is not None
    assert candidate.geometry_uncertain
    assert np.isclose(candidate.gap_lower_m, -2e-6)
    assert np.isclose(candidate.gap_upper_m, 1e-6)
    assert candidate.rod_clearance.collision is None
    assert "model_unclosed_segmented_tip_rod_geometry" in (
        candidate.rod_clearance.model_warning
    )


def test_complete_pose_runs_segmented_clearance_inside_candidate_query() -> None:
    region = _region()
    height = np.zeros(region.shape, dtype=np.float64)
    source_valid = np.ones(region.shape, dtype=np.bool_)
    track = compute_track_geometry(
        height,
        region,
        radius_m=50e-6,
        y_global_m=0.0,
        source_valid_mask=source_valid,
    )
    index = region.shape[1] // 2
    path = SpinePath.from_track(
        track,
        track.envelope_height_m[[index]],
        track_indices=np.array([index]),
    )
    pose = SpinePose(
        tip_axis=np.array([1.0, 0.5, -1.0]),
        spherical_cap_axial_length_m=25e-6,
        cone_length_m=50e-6,
        rod_radius_m=30e-6,
        exposed_rod_length_m=100e-6,
    )
    candidate, _ = query_next_candidate(
        SurfaceState(track, region, height, source_valid),
        path,
        CandidateCursor(),
        pose,
    )
    assert candidate is not None
    assert candidate.forward_cap_valid
    assert candidate.rod_clearance.collision is False
    assert candidate.rod_clearance.sample_count > 0
    assert candidate.valid


def test_candidate_identity_changes_with_spine_pose() -> None:
    _region_spec, _height, track, column = _two_feature_track()
    index = column - 3
    path = SpinePath.from_track(
        track,
        track.envelope_height_m[[index]],
        track_indices=np.array([index]),
    )
    first, _ = query_next_candidate(
        SurfaceState(track),
        path,
        CandidateCursor(),
        SpinePose(tip_axis=np.array([0.0, 0.0, -1.0])),
    )
    second, _ = query_next_candidate(
        SurfaceState(track),
        path,
        CandidateCursor(),
        SpinePose(tip_axis=np.array([0.1, 0.0, -1.0])),
    )
    assert first is not None and second is not None
    assert first.candidate_id != second.candidate_id
