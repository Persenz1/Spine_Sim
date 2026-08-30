"""从有限球尖地形轨迹到接触候选的标准几何契约。

本模块只回答“沿给定搜索路径最先遇到哪个可达表面特征”，并保留法向、间隙、
不确定性和杆体碰撞信息；是否真正接触、粘着或承载由单刺力学层决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    """校验并归一化非零三维方向向量。"""

    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _matches_source_identity(value: ArrayLike, expected_sha256: str) -> bool:
    """核对原始数组内容是否与 track 记录的来源摘要一致。"""

    return array_sha256(np.asarray(value)) == expected_sha256


def _tangent_basis(
    normal: NDArray[np.float64] | None,
    preferred_direction: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """以优选方向在接触平面上的投影构造右手正交切向基。"""

    if normal is None:
        return None
    tangent = preferred_direction - np.dot(preferred_direction, normal) * normal
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        # 刺轴近似平行法向时投影退化，改用与法向不共线的全局轴作参考。
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
    """一条 v2 track，以及杆体 clearance 所需的可选原始高度场。

    只给 track 时仍可查询球尖候选，但杆体碰撞结论为 unknown。若同时给出 region、
    height 和 mask，则三者 identity、shape 与内容哈希必须和 track 的来源完全一致。
    """

    track: TrackGeometry
    region: RegionSpec | None = None
    height_m: NDArray[np.floating] | None = None
    source_valid_mask: NDArray[np.bool_] | None = None
    terrain_version: str = M1_MODULE_VERSION
    surface_model: str = "single_valued_height_field_2_5d"
    general_mesh_scope: str = "OUT_OF_SCOPE"

    def __post_init__(self) -> None:
        """冻结对象前校验 2.5D 范围及所有来源绑定。"""

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
            # defined geometry 可把“全有效”mask 隐式记录；显式全 True 数组与该语义等价。
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
    """一根刺沿有序路径对 track 节点进行的非插值查询序列。"""

    path_position_m: NDArray[np.float64]
    sphere_centers_m: NDArray[np.float64]
    track_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        """校验站位、球心和 track 索引一一对应且路径单调。"""

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
        """由一条 track、球心高度和可选节点子集构造搜索路径。"""

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
    """刺轴方向、法向选择及可选的球冠—锥段—杆体完整几何。"""

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
        """归一化刺轴，并校验尺寸、容差和 clearance 采样密度。"""

        object.__setattr__(self, "tip_axis", _unit_vector(self.tip_axis, "tip_axis"))
        if self.normal_model not in {"surface", "envelope", "contact", "none"}:
            raise ValueError("unsupported normal_model")
        gap_tolerance_m = float(self.gap_tolerance_m)
        if not np.isfinite(gap_tolerance_m) or gap_tolerance_m < 0.0:
            raise ValueError("gap_tolerance_m must be finite and non-negative")
        object.__setattr__(self, "gap_tolerance_m", gap_tolerance_m)
        for name in (
            "spherical_cap_axial_length_m",
            "cone_length_m",
            "rod_radius_m",
            "exposed_rod_length_m",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            dimension = float(value)
            if not np.isfinite(dimension) or dimension <= 0.0:
                raise ValueError(f"{name} must be finite and positive when supplied")
            object.__setattr__(self, name, dimension)
        for name, minimum in (
            ("clearance_axial_samples", 3),
            ("clearance_perimeter_samples", 8),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))

    @property
    def has_complete_body_geometry(self) -> bool:
        """四项针体尺寸是否齐全，从而可给出确定的姿态相关 clearance。"""

        values = (
            self.spherical_cap_axial_length_m,
            self.cone_length_m,
            self.rod_radius_m,
            self.exposed_rod_length_m,
        )
        return all(value is not None for value in values)


@dataclass(frozen=True)
class CandidateCursor:
    """不可变候选 continuation：记录下一路径站、序号和上一 feature。"""

    next_path_index: int = 0
    candidate_index: int = 0
    last_feature_id: str | None = None
    exhausted: bool = False

    def __post_init__(self) -> None:
        """游标和候选序号均不得回退为负数。"""

        if self.next_path_index < 0 or self.candidate_index < 0:
            raise ValueError("cursor indices must be non-negative")


@dataclass(frozen=True, eq=False)
class ContactCandidate:
    """几何层产生的一个可追溯接触候选。

    ``valid`` 只表示尚未发现确定的几何无效，并不等价于正法向反力、摩擦可行或
    稳定承载。near-tie 会保留两组支撑/法向，禁止在本层武断合成唯一法向。
    """

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

    def __post_init__(self) -> None:
        """统一数组表示，并校验候选、法向、间隙界和 continuation 的内部一致性。"""

        for name in (
            "candidate_id",
            "lineage",
            "terrain_version",
            "track_id",
            "geometry_version",
            "feature_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")
        if self.geometry_version != GEOMETRY_VERSION:
            raise ValueError("candidate geometry_version is not current")
        if (
            isinstance(self.candidate_index, (bool, np.bool_))
            or not isinstance(self.candidate_index, (int, np.integer))
            or int(self.candidate_index) < 0
        ):
            raise ValueError("candidate_index must be a non-negative integer")
        object.__setattr__(self, "candidate_index", int(self.candidate_index))
        for name in ("valid", "near_tie", "geometry_uncertain"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
            object.__setattr__(self, name, bool(value))
        if self.forward_cap_valid is not None:
            if not isinstance(self.forward_cap_valid, (bool, np.bool_)):
                raise ValueError("forward_cap_valid must be boolean or None")
            object.__setattr__(
                self, "forward_cap_valid", bool(self.forward_cap_valid)
            )

        path_position_m = float(self.path_position_m)
        signed_gap_m = float(self.signed_gap_m)
        if not np.isfinite(path_position_m) or not np.isfinite(signed_gap_m):
            raise ValueError("candidate path position and signed gap must be finite")
        object.__setattr__(self, "path_position_m", path_position_m)
        object.__setattr__(self, "signed_gap_m", signed_gap_m)
        sphere_center = np.asarray(self.sphere_center_m, dtype=np.float64)
        support_points = np.asarray(self.support_points_m, dtype=np.float64)
        support_count = 2 if self.near_tie else 1
        # 唯一支撑固定为 (1, 3)，near-tie 固定为 (2, 3)，避免下游猜测 shape。
        if sphere_center.shape != (3,) or not np.all(np.isfinite(sphere_center)):
            raise ValueError("sphere_center_m must be a finite 3-vector")
        if support_points.shape != (support_count, 3) or not np.all(
            np.isfinite(support_points)
        ):
            raise ValueError(
                "support_points_m must contain one unique or two near-tie supports"
            )
        object.__setattr__(self, "sphere_center_m", sphere_center)
        object.__setattr__(self, "support_points_m", support_points)

        if self.normal_model not in {"surface", "envelope", "contact", "none"}:
            raise ValueError("unsupported normal_model")
        for name in ("surface_normal", "envelope_normal", "contact_normal"):
            value = getattr(self, name)
            if value is None:
                continue
            if self.near_tie and name == "envelope_normal":
                # 两条几乎并列的包络分支在交界处不可微，不能伪造单一包络法向。
                raise ValueError("near-tie candidates cannot have one envelope normal")
            normal = np.asarray(value, dtype=np.float64)
            expected_shape = (2, 3) if self.near_tie else (3,)
            if normal.shape != expected_shape or not np.all(np.isfinite(normal)):
                raise ValueError(
                    f"{name} has invalid unique/near-tie normal representation"
                )
            object.__setattr__(self, name, normal)

        tangent_basis = self.tangent_basis
        if tangent_basis is not None:
            tangent = np.asarray(tangent_basis, dtype=np.float64)
            if tangent.shape != (2, 3) or not np.all(np.isfinite(tangent)):
                raise ValueError("tangent_basis must be a finite (2, 3) array")
            if self.selected_normal is None:
                raise ValueError("tangent_basis requires one selected normal")
            object.__setattr__(self, "tangent_basis", tangent)

        if (self.gap_lower_m is None) != (self.gap_upper_m is None):
            raise ValueError("candidate gap bounds must both be present or both be None")
        if self.gap_lower_m is not None:
            lower = float(self.gap_lower_m)
            upper = float(self.gap_upper_m)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError("candidate gap bounds must be finite and ordered")
            object.__setattr__(self, "gap_lower_m", lower)
            object.__setattr__(self, "gap_upper_m", upper)
        if self.curvature_radius_m is not None:
            curvature = float(self.curvature_radius_m)
            if not np.isfinite(curvature) or curvature <= 0.0:
                raise ValueError("curvature_radius_m must be finite and positive")
            object.__setattr__(self, "curvature_radius_m", curvature)
        if not isinstance(self.rod_clearance, RodClearanceResult):
            raise ValueError("rod_clearance must be a RodClearanceResult")
        if not isinstance(self.search_cursor, CandidateCursor) or (
            self.search_cursor.candidate_index != self.candidate_index + 1
            or self.search_cursor.last_feature_id != self.feature_id
        ):
            raise ValueError("candidate continuation cursor does not match the candidate")

    @property
    def selected_normal(self) -> NDArray[np.float64] | None:
        """按配置返回唯一法向；near-tie 或 ``none`` 模型返回 ``None``。"""

        if self.near_tie:
            return None
        return {
            "surface": self.surface_normal,
            "envelope": self.envelope_normal,
            "contact": self.contact_normal,
            "none": None,
        }[self.normal_model]


def _feature_id(track: TrackGeometry, index: int, support_count: int) -> str:
    """用原始支撑网格节点构造可跨相邻 track 站保持不变的 feature ID。"""

    indices = track.support_feature_indices_yx[index, :support_count]
    return "+".join(f"node:{int(y)}:{int(x)}" for y, x in indices)


def _node_feature_id(track: TrackGeometry, index: int) -> str | None:
    """读取一个 track 节点的 feature ID；支撑缺失时返回 ``None``。"""

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
    """从 track 提取 surface/envelope/contact 三类法向并规范其 shape。"""

    surface = np.asarray(
        track.surface_normals[track_index, :support_count], dtype=np.float64
    )
    contact = np.asarray(
        track.contact_normals[track_index, :support_count], dtype=np.float64
    )
    envelope = np.asarray(track.envelope_normals[track_index], dtype=np.float64)
    surface_output = surface if np.all(np.isfinite(surface)) else None
    contact_output = contact if np.all(np.isfinite(contact)) else None
    envelope_output = (
        None if near_tie or not np.all(np.isfinite(envelope)) else envelope
    )
    if not near_tie:
        surface_output = None if surface_output is None else surface_output[0]
        contact_output = None if contact_output is None else contact_output[0]
    return surface_output, envelope_output, contact_output


def _clearance(
    surface_state: SurfaceState,
    pose: SpinePose,
    center: NDArray[np.float64],
) -> RodClearanceResult:
    """在信息足够时检查分段针体，否则显式返回模型未闭合原因。"""

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
    """按路径返回下一个不同 feature；只插值闭合位置，不插值支撑身份。"""

    if cursor.exhausted:
        return None, cursor
    # previous_* 只服务于同一 feature 内的首次 gap 闭合定位，不构成另一套搜索状态。
    track = surface_state.track
    axis = spine_pose.tip_axis
    previous_gap: float | None = None
    previous_center: NDArray[np.float64] | None = None
    previous_position: float | None = None
    previous_track_index: int | None = None
    previous_feature_id: str | None = None
    for path_index in range(cursor.next_path_index, spine_path.path_position_m.size):
        # 阶段 1：逐站核对球心确实查询精确 track 节点，并计算有符号包络间隙。
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
            # 尚未闭合时保留上一站；后续若同一 feature 过零，可线性定位首次接触。
            previous_gap = gap
            previous_center = center
            previous_position = float(spine_path.path_position_m[path_index])
            previous_track_index = track_index
            previous_feature_id = node_feature_id
            continue
        near_tie = bool(track.near_tie_flag[track_index])
        support_count = 2 if near_tie else 1
        support_points = track.support_points_m[track_index, :support_count]
        finite_support = np.all(np.isfinite(support_points), axis=1)
        support_points = support_points[finite_support]
        if support_points.shape[0] != support_count:
            # 包络高度存在但支撑坐标不完整时不能形成候选，继续沿路径搜索。
            previous_gap = gap
            previous_center = center
            previous_position = float(spine_path.path_position_m[path_index])
            previous_track_index = track_index
            previous_feature_id = None
            continue
        feature_id = _feature_id(track, track_index, support_count)
        if feature_id == cursor.last_feature_id:
            # 同一表面 feature 连续覆盖多个站，只报告首次遭遇，避免重复候选。
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
            # 路径可跳过 track 节点；只有中间所有节点仍属于同一 feature 才允许插值。
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
            # 只对球心和路径位置做 gap 过零插值；支撑点仍取当前离散 feature 的原值。
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

        # 阶段 2：执行针尖朝向和整段针体 clearance 门控，再整理三类法向。
        forward_valid = bool(
            np.all(forward_cap_gate(support_points, candidate_center, axis))
        )
        rod_clearance = _clearance(surface_state, spine_pose, candidate_center)
        surface_normal, envelope_normal, contact_normal = _candidate_normals(
            track, track_index, support_count, near_tie
        )
        radial = candidate_center[None, :] - support_points
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        if np.all(radial_norm > 0.0):
            # 插值球心后，contact normal 应由实际候选球心重新计算，而非照搬 track 节点。
            radial_normal = radial / radial_norm
            contact_normal = radial_normal if near_tie else radial_normal[0]
        selected = None if near_tie else {
            "surface": surface_normal,
            "envelope": envelope_normal,
            "contact": contact_normal,
            "none": None,
        }[spine_pose.normal_model]
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
        # near-tie 和未知 clearance 都保留为不确定，而不是误判为安全或无效。
        geometry_uncertain |= near_tie or rod_clearance.collision is None
        valid = bool(track.valid_mask[track_index]) and forward_valid
        valid &= rod_clearance.collision is not True
        # 阶段 3：identity 包含姿态、法向模型和完整针体参数，语义变化会生成新候选。
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
    # 路径扫描完成仍无新 feature；返回 exhausted cursor，后续调用可 O(1) 结束。
    return None, CandidateCursor(
        next_path_index=spine_path.path_position_m.size,
        candidate_index=cursor.candidate_index,
        last_feature_id=cursor.last_feature_id,
        exhausted=True,
    )


__all__ = [
    "CandidateCursor",
    "ContactCandidate",
    "GEOMETRY_VERSION",
    "SpinePath",
    "SpinePose",
    "SurfaceState",
    "query_next_candidate",
]
