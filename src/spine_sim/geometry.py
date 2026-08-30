"""Canonical terrain-to-contact-candidate geometry contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spine_sim.core.identity import identity, lineage_hash, stable_hash
from spine_sim.core.versions import GEOMETRY_SCHEMA_VERSION
from spine_sim.terrain.envelope import (
    RodClearanceResult,
    array_sha256,
    check_segmented_tip_rod_clearance,
    forward_cap_gate,
)
from spine_sim.terrain.models import M1_MODULE_VERSION, RegionSpec, TrackGeometry


GEOMETRY_VERSION = GEOMETRY_SCHEMA_VERSION
NormalModel = Literal["surface", "envelope", "contact", "none"]


def _unit_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _matches_source_identity(value: ArrayLike, expected_sha256: str) -> bool:
    return array_sha256(np.asarray(value)) == expected_sha256


def _tangent_basis(
    normal: NDArray[np.float64] | None,
    preferred_direction: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    if normal is None:
        return None
    tangent = preferred_direction - np.dot(preferred_direction, normal) * normal
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, normal))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        tangent = reference - np.dot(reference, normal) * normal
        tangent_norm = float(np.linalg.norm(tangent))
    tangent /= tangent_norm
    second = np.cross(normal, tangent)
    return np.stack((tangent, second), axis=0)


@dataclass(frozen=True)
class SurfaceState:
    """One v2 track plus optional raw height data needed for body clearance."""

    track: TrackGeometry
    region: RegionSpec | None = None
    height_m: NDArray[np.floating] | None = None
    source_valid_mask: NDArray[np.bool_] | None = None
    terrain_version: str = M1_MODULE_VERSION
    surface_model: str = "single_valued_height_field_2_5d"
    general_mesh_scope: str = "OUT_OF_SCOPE"

    def __post_init__(self) -> None:
        if self.surface_model != "single_valued_height_field_2_5d":
            raise ValueError("only the explicit 2.5-D height-field model is supported")
        if self.general_mesh_scope != "OUT_OF_SCOPE":
            raise ValueError("general mesh geometry must remain explicitly OUT_OF_SCOPE")
        if (self.region is None) != (self.height_m is None):
            raise ValueError("region and height_m must be supplied together")
        if self.height_m is not None:
            assert self.region is not None
            height = np.asarray(self.height_m)
            if height.shape != self.region.shape or not np.all(np.isfinite(height)):
                raise ValueError("height_m must be finite with the RegionSpec shape")
            if (
                self.region.terrain_recipe_id != self.track.terrain_recipe_id
                or self.region.region_id != self.track.region_id
            ):
                raise ValueError("raw clearance region does not match the track identity")
            if not _matches_source_identity(
                self.height_m, self.track.source_data_sha256
            ):
                raise ValueError("height_m does not match track.source_data_sha256")
            if self.source_valid_mask is None:
                raise ValueError(
                    "raw height clearance requires its explicit source_valid_mask"
                )
            mask = np.asarray(self.source_valid_mask)
            if mask.shape != height.shape or mask.dtype != np.bool_:
                raise ValueError(
                    "source_valid_mask must be boolean with the height shape"
                )
            implicit_all_valid_sha256 = stable_hash(
                {
                    "kind": "implicit_all_valid",
                    "shape": list(self.region.shape),
                    "region_id": self.region.region_id,
                }
            )
            if not _matches_source_identity(
                self.source_valid_mask,
                self.track.source_valid_mask_sha256,
            ) and not (
                bool(np.all(mask))
                and self.track.source_valid_mask_sha256
                == implicit_all_valid_sha256
            ):
                raise ValueError(
                    "source_valid_mask does not match "
                    "track.source_valid_mask_sha256"
                )


@dataclass(frozen=True)
class SpinePath:
    """Ordered, non-interpolated track queries for one spine search path."""

    path_position_m: NDArray[np.float64]
    sphere_centers_m: NDArray[np.float64]
    track_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        positions = np.asarray(self.path_position_m, dtype=np.float64)
        centers = np.asarray(self.sphere_centers_m, dtype=np.float64)
        indices = np.asarray(self.track_indices)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError("path_position_m must be a non-empty vector")
        if centers.shape != (positions.size, 3):
            raise ValueError("sphere_centers_m must have shape (path_count, 3)")
        if indices.shape != positions.shape or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError("track_indices must be an integer path vector")
        if (
            not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(centers))
            or np.any(np.diff(positions) < 0.0)
        ):
            raise ValueError("path positions/centres must be finite and ordered")

    @classmethod
    def from_track(
        cls,
        track: TrackGeometry,
        center_z_m: ArrayLike,
        *,
        track_indices: ArrayLike | None = None,
        path_position_m: ArrayLike | None = None,
    ) -> "SpinePath":
        indices = (
            np.arange(track.x_global_m.size, dtype=np.int64)
            if track_indices is None
            else np.asarray(track_indices, dtype=np.int64)
        )
        z = np.asarray(center_z_m, dtype=np.float64)
        if z.ndim == 0:
            z = np.full(indices.size, float(z), dtype=np.float64)
        if z.shape != indices.shape:
            raise ValueError("center_z_m must be scalar or match track_indices")
        if np.any(indices < 0) or np.any(indices >= track.x_global_m.size):
            raise IndexError("track_indices contains an out-of-range index")
        positions = (
            track.x_global_m[indices] - track.x_global_m[indices[0]]
            if path_position_m is None
            else np.asarray(path_position_m, dtype=np.float64)
        )
        centers = np.column_stack(
            (
                track.x_global_m[indices],
                np.full(indices.size, track.y_global_m),
                z,
            )
        )
        return cls(positions, centers, indices)


@dataclass(frozen=True)
class SpinePose:
    """Spine direction and complete optional sphere-cap/cone/rod geometry."""

    tip_axis: NDArray[np.float64]
    normal_model: NormalModel = "contact"
    gap_tolerance_m: float = 0.0
    spherical_cap_axial_length_m: float | None = None
    cone_length_m: float | None = None
    rod_radius_m: float | None = None
    exposed_rod_length_m: float | None = None
    clearance_axial_samples: int = 32
    clearance_perimeter_samples: int = 16

    def __post_init__(self) -> None:
        _unit_vector(self.tip_axis, "tip_axis")
        if self.normal_model not in {"surface", "envelope", "contact", "none"}:
            raise ValueError("unsupported normal_model")
        if not np.isfinite(self.gap_tolerance_m) or self.gap_tolerance_m < 0.0:
            raise ValueError("gap_tolerance_m must be finite and non-negative")

    @property
    def has_complete_body_geometry(self) -> bool:
        values = (
            self.spherical_cap_axial_length_m,
            self.cone_length_m,
            self.rod_radius_m,
            self.exposed_rod_length_m,
        )
        return all(value is not None and value > 0.0 for value in values)


@dataclass(frozen=True)
class CandidateCursor:
    next_path_index: int = 0
    candidate_index: int = 0
    last_feature_id: str | None = None
    exhausted: bool = False

    def __post_init__(self) -> None:
        if self.next_path_index < 0 or self.candidate_index < 0:
            raise ValueError("cursor indices must be non-negative")


@dataclass(frozen=True, eq=False)
class ContactCandidate:
    candidate_id: str
    lineage: str
    terrain_version: str
    track_id: str
    geometry_version: str
    candidate_index: int
    path_position_m: float
    feature_id: str
    sphere_center_m: NDArray[np.float64]
    support_points_m: NDArray[np.float64]
    signed_gap_m: float
    curvature_radius_m: float | None
    surface_normal: NDArray[np.float64] | None
    envelope_normal: NDArray[np.float64] | None
    contact_normal: NDArray[np.float64] | None
    normal_model: NormalModel
    tangent_basis: NDArray[np.float64] | None
    valid: bool
    near_tie: bool
    geometry_uncertain: bool
    gap_lower_m: float | None
    gap_upper_m: float | None
    forward_cap_valid: bool | None
    rod_clearance: RodClearanceResult
    search_cursor: CandidateCursor

    @property
    def selected_normal(self) -> NDArray[np.float64] | None:
        if self.near_tie or self.normal_model == "none":
            return None
        selected = {
            "surface": self.surface_normal,
            "envelope": self.envelope_normal,
            "contact": self.contact_normal,
        }[self.normal_model]
        if selected is None:
            return None
        array = np.asarray(selected, dtype=np.float64)
        if array.shape == (3,) and np.all(np.isfinite(array)):
            return array
        if array.shape == (1, 3) and np.all(np.isfinite(array[0])):
            return array[0]
        return None


def _feature_id(track: TrackGeometry, index: int, support_count: int) -> str:
    indices = track.support_feature_indices_yx[index, :support_count]
    return "+".join(f"node:{int(y)}:{int(x)}" for y, x in indices)


def _node_feature_id(track: TrackGeometry, index: int) -> str | None:
    support_count = 2 if bool(track.near_tie_flag[index]) else 1
    indices = track.support_feature_indices_yx[index, :support_count]
    if np.any(indices < 0):
        return None
    return _feature_id(track, index, support_count)


def _candidate_normals(
    track: TrackGeometry,
    track_index: int,
    support_count: int,
    near_tie: bool,
) -> tuple[
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    surface = np.asarray(
        track.surface_normals[track_index, :support_count], dtype=np.float64
    )
    contact = np.asarray(
        track.contact_normals[track_index, :support_count], dtype=np.float64
    )
    envelope = np.asarray(track.envelope_normals[track_index], dtype=np.float64)
    surface_output = surface if np.any(np.isfinite(surface)) else None
    contact_output = contact if np.any(np.isfinite(contact)) else None
    envelope_output = (
        None if near_tie or not np.all(np.isfinite(envelope)) else envelope
    )
    return surface_output, envelope_output, contact_output


def _clearance(
    surface_state: SurfaceState,
    pose: SpinePose,
    center: NDArray[np.float64],
) -> RodClearanceResult:
    if not pose.has_complete_body_geometry:
        return RodClearanceResult(
            collision=None,
            minimum_clearance_m=None,
            sample_count=0,
            model_warning=("model_unclosed_segmented_tip_rod_geometry",),
        )
    if surface_state.height_m is None or surface_state.region is None:
        return RodClearanceResult(
            collision=None,
            minimum_clearance_m=None,
            sample_count=0,
            model_warning=("raw_height_required_for_pose_aware_rod_clearance",),
        )
    return check_segmented_tip_rod_clearance(
        surface_state.height_m,
        surface_state.region,
        sphere_center_xyz_m=center,
        tip_axis=pose.tip_axis,
        tip_radius_m=surface_state.track.radius_m,
        spherical_cap_axial_length_m=pose.spherical_cap_axial_length_m,
        cone_length_m=pose.cone_length_m,
        exposed_rod_length_m=pose.exposed_rod_length_m,
        rod_radius_m=pose.rod_radius_m,
        source_valid_mask=surface_state.source_valid_mask,
        axial_sample_count=pose.clearance_axial_samples,
        perimeter_sample_count=pose.clearance_perimeter_samples,
    )


def query_next_candidate(
    surface_state: SurfaceState,
    spine_path: SpinePath,
    cursor: CandidateCursor,
    spine_pose: SpinePose,
) -> tuple[ContactCandidate | None, CandidateCursor]:
    """Return the next distinct encountered feature without interpolating support."""

    if cursor.exhausted:
        return None, cursor
    track = surface_state.track
    axis = _unit_vector(spine_pose.tip_axis, "tip_axis")
    previous_gap: float | None = None
    previous_center: NDArray[np.float64] | None = None
    previous_position: float | None = None
    previous_track_index: int | None = None
    previous_feature_id: str | None = None
    for path_index in range(cursor.next_path_index, spine_path.path_position_m.size):
        track_index = int(spine_path.track_indices[path_index])
        if not 0 <= track_index < track.x_global_m.size:
            raise IndexError("spine_path track index is outside the track")
        center = np.asarray(spine_path.sphere_centers_m[path_index], dtype=np.float64)
        expected_x = float(track.x_global_m[track_index])
        coordinate_tolerance = max(1e-15, track.resolution_m * 1e-9)
        if (
            abs(float(center[0]) - expected_x) > coordinate_tolerance
            or abs(float(center[1]) - track.y_global_m) > coordinate_tolerance
        ):
            raise ValueError(
                "path centres must query exact track nodes; support interpolation is forbidden"
            )
        gap = float(center[2] - track.envelope_height_m[track_index])
        node_feature_id = _node_feature_id(track, track_index)
        if not np.isfinite(gap) or gap > spine_pose.gap_tolerance_m:
            previous_gap = gap
            previous_center = center
            previous_position = float(spine_path.path_position_m[path_index])
            previous_track_index = track_index
            previous_feature_id = node_feature_id
            continue
        near_tie = bool(track.near_tie_flag[track_index])
        support_count = 2 if near_tie else 1
        support_points = np.asarray(
            track.support_points_m[track_index, :support_count], dtype=np.float64
        )
        finite_support = np.all(np.isfinite(support_points), axis=1)
        support_points = support_points[finite_support]
        support_count = int(support_points.shape[0])
        if support_count == 0:
            previous_gap = gap
            previous_center = center
            previous_position = float(spine_path.path_position_m[path_index])
            previous_track_index = track_index
            previous_feature_id = None
            continue
        feature_id = _feature_id(track, track_index, support_count)
        if feature_id == cursor.last_feature_id:
            previous_gap = gap
            previous_center = center
            previous_position = float(spine_path.path_position_m[path_index])
            previous_track_index = track_index
            previous_feature_id = feature_id
            continue

        path_position = float(spine_path.path_position_m[path_index])
        signed_gap = gap
        candidate_center = center
        same_feature_segment = (
            previous_track_index is not None
            and previous_feature_id == feature_id
        )
        if same_feature_segment and previous_track_index != track_index:
            first_index = min(previous_track_index, track_index)
            last_index = max(previous_track_index, track_index)
            same_feature_segment = all(
                _node_feature_id(track, index) == feature_id
                for index in range(first_index, last_index + 1)
            )
        if (
            previous_gap is not None
            and previous_gap > spine_pose.gap_tolerance_m
            and previous_center is not None
            and previous_position is not None
            and np.isfinite(previous_gap)
            and same_feature_segment
        ):
            denominator = previous_gap - gap
            if denominator > 0.0:
                fraction = (previous_gap - spine_pose.gap_tolerance_m) / denominator
                fraction = float(np.clip(fraction, 0.0, 1.0))
                path_position = previous_position + fraction * (
                    path_position - previous_position
                )
                candidate_center = previous_center + fraction * (
                    center - previous_center
                )
                signed_gap = spine_pose.gap_tolerance_m

        forward_values = np.asarray(
            forward_cap_gate(support_points, candidate_center, axis),
            dtype=np.bool_,
        )
        forward_valid = bool(np.all(forward_values))
        rod_clearance = _clearance(surface_state, spine_pose, candidate_center)
        surface_normal, envelope_normal, contact_normal = _candidate_normals(
            track, track_index, support_count, near_tie
        )
        radial = candidate_center[None, :] - support_points
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        if np.all(radial_norm > 0.0):
            contact_normal = radial / radial_norm
        selected: NDArray[np.float64] | None = None
        if not near_tie:
            normal_source = {
                "surface": surface_normal,
                "envelope": envelope_normal,
                "contact": contact_normal,
                "none": None,
            }[spine_pose.normal_model]
            if normal_source is not None:
                normal_array = np.asarray(normal_source, dtype=np.float64)
                if normal_array.shape == (3,):
                    selected = normal_array
                elif normal_array.shape == (1, 3):
                    selected = normal_array[0]
                if selected is not None and not np.all(np.isfinite(selected)):
                    selected = None
        tangent_basis = _tangent_basis(selected, axis)
        lower = track.envelope_height_lower_m
        upper = track.envelope_height_upper_m
        gap_lower = (
            None if upper is None else float(candidate_center[2] - upper[track_index])
        )
        gap_upper = (
            None if lower is None else float(candidate_center[2] - lower[track_index])
        )
        geometry_uncertain = bool(track.geometry_uncertain_mask[track_index])
        geometry_uncertain |= near_tie or rod_clearance.collision is None
        valid = bool(track.valid_mask[track_index]) and forward_valid
        valid &= rod_clearance.collision is not True
        payload = {
            "track_id": track.track_id,
            "candidate_index": cursor.candidate_index,
            "path_position_m": round(path_position, 15),
            "feature_id": feature_id,
            "geometry_version": GEOMETRY_VERSION,
            "sphere_center_m": [
                round(float(value), 15) for value in candidate_center
            ],
            "tip_axis": [round(float(value), 15) for value in axis],
            "normal_model": spine_pose.normal_model,
            "gap_tolerance_m": round(spine_pose.gap_tolerance_m, 15),
            "segmented_body_m": {
                "spherical_cap_axial_length_m": spine_pose.spherical_cap_axial_length_m,
                "cone_length_m": spine_pose.cone_length_m,
                "rod_radius_m": spine_pose.rod_radius_m,
                "exposed_rod_length_m": spine_pose.exposed_rod_length_m,
                "clearance_axial_samples": spine_pose.clearance_axial_samples,
                "clearance_perimeter_samples": spine_pose.clearance_perimeter_samples,
            },
        }
        candidate_id = identity(
            "candidate", payload, module_version=GEOMETRY_VERSION
        )
        continuation_cursor = CandidateCursor(
            next_path_index=path_index + 1,
            candidate_index=cursor.candidate_index + 1,
            last_feature_id=feature_id,
            exhausted=False,
        )
        candidate = ContactCandidate(
            candidate_id=candidate_id,
            lineage=lineage_hash(
                track.track_id,
                track.source_data_sha256,
                track.source_valid_mask_sha256,
                track.measurement_semantics_hash,
            ),
            terrain_version=surface_state.terrain_version,
            track_id=track.track_id,
            geometry_version=GEOMETRY_VERSION,
            candidate_index=cursor.candidate_index,
            path_position_m=path_position,
            feature_id=feature_id,
            sphere_center_m=candidate_center,
            support_points_m=support_points,
            signed_gap_m=signed_gap,
            curvature_radius_m=None,
            surface_normal=surface_normal,
            envelope_normal=envelope_normal,
            contact_normal=contact_normal,
            normal_model=spine_pose.normal_model,
            tangent_basis=tangent_basis,
            valid=valid,
            near_tie=near_tie,
            geometry_uncertain=geometry_uncertain,
            gap_lower_m=gap_lower,
            gap_upper_m=gap_upper,
            forward_cap_valid=forward_valid,
            rod_clearance=rod_clearance,
            search_cursor=continuation_cursor,
        )
        return candidate, continuation_cursor
    return None, CandidateCursor(
        next_path_index=spine_path.path_position_m.size,
        candidate_index=cursor.candidate_index,
        last_feature_id=cursor.last_feature_id,
        exhausted=True,
    )


def drive_candidate_path(
    surface_state: SurfaceState,
    spine_path: SpinePath,
    spine_pose: SpinePose,
    cursor: CandidateCursor | None = None,
) -> Iterator[ContactCandidate]:
    """The sole path continuation driver used by search/reject/re-engagement."""

    current = CandidateCursor() if cursor is None else cursor
    while not current.exhausted:
        candidate, current = query_next_candidate(
            surface_state, spine_path, current, spine_pose
        )
        if candidate is None:
            break
        yield candidate


__all__ = [
    "CandidateCursor",
    "ContactCandidate",
    "GEOMETRY_VERSION",
    "SpinePath",
    "SpinePose",
    "SurfaceState",
    "drive_candidate_path",
    "query_next_candidate",
]
