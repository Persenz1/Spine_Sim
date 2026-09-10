"""阵列物理计数、载荷分担和路径阻力积分指标。

这些函数只从已求得的逐刺/路径数据计算描述指标，不参与接触状态或阵列平衡求解。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class SpineMetricInput:
    """一根刺用于计数和载荷分担统计的最小状态。"""

    geometric: bool
    signed_gap_m: float | None
    engagement: bool | None
    active: bool
    normal_force_N: float | None = None
    tangent_resistance_N: float | None = None
    macro_normal_force_N: float | None = None
    force_magnitude_N: float | None = None

    @classmethod
    def from_force(
        cls, *, wall_force_N: ArrayLike, contact_normal: ArrayLike | None,
        geometric: bool, signed_gap_m: float | None, engagement: bool | None,
        force_tolerance_N: float = 0.0,
    ) -> "SpineMetricInput":
        """由壁面对针的力生成主稿指标：``P=f_z``、``T=-f_x``。"""

        force = np.asarray(wall_force_N, dtype=float)
        magnitude = float(np.linalg.norm(force))
        local_normal = (
            None if contact_normal is None
            else float(np.dot(force, np.asarray(contact_normal, dtype=float)))
        )
        return cls(
            geometric, signed_gap_m, engagement, magnitude > force_tolerance_N,
            local_normal, -float(force[0]), float(force[2]), magnitude,
        )


@dataclass(frozen=True)
class ArrayCounts:
    """名义、几何、接触、挂接、承载数量及载荷分担指标。"""

    n_nominal: int
    n_geometric: int
    n_contact: int
    n_engaged: int | None
    n_engaged_lower: int
    n_engaged_upper: int
    n_active: int
    n_evaluable: int
    P_sum_N: float | None
    P_avg_N: float | None
    n_share_normal: float | None
    n_share_tangent_positive: float | None
    load_sharing_index: float | None
    N_sum_local_N: float | None = None
    n_share_local_normal: float | None = None
    normal_weight_basis: str = "legacy_local_normal"
    load_sharing_basis: str = "legacy_tangent_magnitude"


@dataclass(frozen=True)
class PathResistanceMetrics:
    """正/负/净归一化路径阻力及有效覆盖范围。"""

    J_positive: float | None
    J_negative: float | None
    J_net: float | None
    effective_length_m: float
    full_length_m: float
    coverage: float
    interval_count: int
    complete: bool
    observed_J_positive: float
    observed_J_negative: float
    observed_J_net: float
    T_peak_N: float | None
    T_min_N: float | None
    T_mean_N: float | None
    T_quantile_N: float | None
    force_quantile: float | None


def _inverse_simpson(values: Sequence[float]) -> float | None:
    """计算 ``(Σp)²/Σp²`` 的有效分担根数；总量为零时未定义。"""

    array = np.asarray(values, dtype=float)
    total = float(np.sum(array))
    if total <= 0.0:
        return None
    # 极小力平方下溢在这里的语义就是“无法分辨分担根数”，
    # 不应因调用方全局开启 ``numpy.seterr(all="raise")`` 而改变接口结果。
    with np.errstate(under="ignore"):
        denominator = float(np.dot(array, array))
    if denominator <= 0.0:
        return None
    return total * total / denominator


def compute_array_counts(
    spines: Iterable[SpineMetricInput], *, gap_tolerance_m: float
) -> ArrayCounts:
    """计算分层物理计数，且不把未知挂接状态强制当作 false。"""

    if not np.isfinite(gap_tolerance_m) or gap_tolerance_m < 0.0:
        raise ValueError("gap_tolerance_m must be finite and non-negative")
    items = tuple(spines)
    contacts = tuple(
        item
        for item in items
        if item.geometric
        and item.signed_gap_m is not None
        and np.isfinite(item.signed_gap_m)
        and item.signed_gap_m <= gap_tolerance_m
    )
    engagement_values = tuple(item.engagement for item in items)
    n_evaluable = sum(value is not None for value in engagement_values)
    n_engaged_lower = sum(value is True for value in engagement_values)
    unknown = sum(value is None for value in engagement_values)
    # 存在未知项时只报告上下界，精确 n_engaged 保持 None。
    n_engaged = n_engaged_lower if unknown == 0 else None

    normal_values: list[float] = []
    normal_known = True
    for item in contacts:
        value = item.normal_force_N
        if value is None or not np.isfinite(value):
            normal_known = False
            continue
        if value < 0.0:
            raise ValueError("contact normal force must be non-negative")
        normal_values.append(float(value))
    N_sum = float(sum(normal_values)) if normal_known else None
    local_share = _inverse_simpson(normal_values) if normal_known else None
    macro_supplied = any(item.macro_normal_force_N is not None for item in items)
    if macro_supplied:
        # 宏观 P_i 可以为负；只有分载权重裁剪，合力保持原有符号。
        macro_values = [item.macro_normal_force_N for item in contacts]
        macro_known = all(value is not None and np.isfinite(value) for value in macro_values)
        P_sum = float(sum(macro_values)) if macro_known else None
        P_avg = P_sum / len(contacts) if macro_known and contacts else None
        n_share_normal = (
            _inverse_simpson([max(float(value), 0.0) for value in macro_values])
            if macro_known else None
        )
    elif normal_known:
        # 旧求解器调用仍可读取旧值，但输出字段明确标记其权重含义。
        P_sum = N_sum
        P_avg = P_sum / len(contacts) if contacts else None
        n_share_normal = local_share
    else:
        P_sum = None
        P_avg = None
        n_share_normal = None

    # 有效分担根数只看正向抗力；load_sharing_index 则看所有 active 刺的载荷幅值。
    tangent_positive = [
        float(item.tangent_resistance_N)
        for item in items
        if item.tangent_resistance_N is not None
        and np.isfinite(item.tangent_resistance_N)
        and item.tangent_resistance_N > 0.0
    ]
    magnitude_supplied = any(item.force_magnitude_N is not None for item in items)
    active_values = [
        item.force_magnitude_N if magnitude_supplied else item.tangent_resistance_N
        for item in items if item.active
    ]
    active_known = all(value is not None and np.isfinite(value) for value in active_values)
    active_magnitudes = [abs(float(value)) for value in active_values] if active_known else []
    active_mean = float(np.mean(active_magnitudes)) if active_magnitudes else 0.0
    if active_mean > 0.0:
        sharing_index = float(np.max(active_magnitudes) / active_mean)
    else:
        sharing_index = None

    return ArrayCounts(
        n_nominal=len(items),
        n_geometric=sum(item.geometric for item in items),
        n_contact=len(contacts),
        n_engaged=n_engaged,
        n_engaged_lower=n_engaged_lower,
        n_engaged_upper=n_engaged_lower + unknown,
        n_active=sum(item.active for item in items),
        n_evaluable=n_evaluable,
        P_sum_N=P_sum,
        P_avg_N=P_avg,
        n_share_normal=n_share_normal,
        n_share_tangent_positive=_inverse_simpson(tangent_positive),
        load_sharing_index=sharing_index,
        N_sum_local_N=N_sum,
        n_share_local_normal=local_share,
        normal_weight_basis="macro_P_positive" if macro_supplied else "legacy_local_normal",
        load_sharing_basis="force_norm" if magnitude_supplied else "legacy_tangent_magnitude",
    )


def integrate_path_resistance(
    path_position_m: ArrayLike,
    resistance_force_N: ArrayLike,
    *,
    external_normal_preload_N: float,
    accepted: ArrayLike,
    valid: ArrayLike,
    window_m: tuple[float, float] | None = None,
    force_quantile: float | None = None,
) -> PathResistanceMetrics:
    """在声明窗口积分阵列总阻力，线性站间插值，事件跳变不占路径长度。

    缺口不连线、不补零、分母不缩短。窗口不完整时正式 J 和力统计为 None；
    ``observed_J_*`` 仅为已知区间对整个窗口的贡献，不能当作完整性能。
    调用方必须先求阵列总 T 再传入，不能把逐刺正负面积相加。
    """

    segments, window = _path_segments(
        path_position_m, resistance_force_N, accepted=accepted, valid=valid,
        window_m=window_m,
    )
    if not np.isfinite(external_normal_preload_N) or external_normal_preload_N <= 0.0:
        raise ValueError("external_normal_preload_N must be finite and positive")
    if force_quantile is not None and not 0.0 <= force_quantile <= 1.0:
        raise ValueError("force_quantile must lie in [0, 1]")
    full_length = window[1] - window[0]
    effective_length = sum(end - start for start, end, _, _ in segments)
    complete = bool(np.isclose(effective_length, full_length, rtol=1e-10, atol=0.0))
    positive_integral = 0.0
    negative_integral = 0.0
    for start, end, first, last in segments:
        first /= external_normal_preload_N
        last /= external_normal_preload_N
        length = end - start
        if first * last < 0.0:
            # 跨零处精确拆三角形；先裁剪站点再梯形积分会高估正负面积。
            fraction = abs(first) / (abs(first) + abs(last))
            positive_integral += 0.5 * length * (
                max(first, 0.0) * fraction + max(last, 0.0) * (1.0 - fraction)
            )
            negative_integral += 0.5 * length * (
                max(-first, 0.0) * fraction + max(-last, 0.0) * (1.0 - fraction)
            )
        else:
            positive_integral += 0.5 * length * (max(first, 0.0) + max(last, 0.0))
            negative_integral += 0.5 * length * (max(-first, 0.0) + max(-last, 0.0))
    positive = positive_integral / full_length
    negative = negative_integral / full_length
    forces = [force for _, _, first, last in segments for force in (first, last)]
    # 已接受的同 X 事件后状态可决定峰值，却不贡献额外路径面积。
    x = np.asarray(path_position_m, dtype=float)
    recorded_force = np.asarray(resistance_force_N, dtype=float)
    include = (
        np.asarray(accepted, dtype=bool) & np.asarray(valid, dtype=bool)
        & np.isfinite(recorded_force) & (x >= window[0]) & (x <= window[1])
    )
    forces.extend(recorded_force[include].tolist())
    return PathResistanceMetrics(
        J_positive=positive if complete else None,
        J_negative=negative if complete else None,
        J_net=positive - negative if complete else None,
        effective_length_m=float(effective_length), full_length_m=float(full_length),
        coverage=float(effective_length / full_length), interval_count=len(segments),
        complete=complete, observed_J_positive=positive, observed_J_negative=negative,
        observed_J_net=positive - negative,
        T_peak_N=max(forces) if complete else None,
        T_min_N=min(forces) if complete else None,
        T_mean_N=(positive - negative) * external_normal_preload_N if complete else None,
        T_quantile_N=(
            _distance_quantile(segments, force_quantile)
            if complete and force_quantile is not None else None
        ),
        force_quantile=force_quantile,
    )


def _distance_quantile(
    segments: Sequence[tuple[float, float, float, float]], quantile: float,
) -> float:
    """按路径长度加权的连续线性力分布，避免自适应站点密度改变低分位力。"""

    ends = np.asarray([(min(first, last), max(first, last)) for _, _, first, last in segments])
    lengths = np.asarray([high - low for low, high, _, _ in segments])
    low, high = float(np.min(ends)), float(np.max(ends))
    if quantile == 0.0:
        return low
    if quantile == 1.0:
        return high
    target = quantile * float(np.sum(lengths))
    changing = ends[:, 1] > ends[:, 0]
    for _ in range(60):
        midpoint = 0.5 * (low + high)
        fractions = (midpoint >= ends[:, 1]).astype(float)
        fractions[changing] = np.clip(
            (midpoint - ends[changing, 0]) / (ends[changing, 1] - ends[changing, 0]), 0.0, 1.0
        )
        if float(np.dot(lengths, fractions)) >= target:
            high = midpoint
        else:
            low = midpoint
    return high


def _path_segments(
    path_position_m: ArrayLike, resistance_force_N: ArrayLike, *,
    accepted: ArrayLike, valid: ArrayLike, window_m: tuple[float, float] | None,
) -> tuple[list[tuple[float, float, float, float]], tuple[float, float]]:
    """返回裁到窗口内的有效线性区间；同 X 的左右事件状态依输入顺序保留。"""

    x = np.asarray(path_position_m, dtype=float)
    force = np.asarray(resistance_force_N, dtype=float)
    accepted_mask = np.asarray(accepted, dtype=bool)
    valid_mask = np.asarray(valid, dtype=bool)
    if x.ndim != 1 or not (
        x.shape == force.shape == accepted_mask.shape == valid_mask.shape
    ):
        raise ValueError("path, force, accepted, and valid must be equal 1-D arrays")
    if not np.all(np.isfinite(x)):
        raise ValueError("path positions must be finite")
    dx = np.diff(x)
    if np.any(dx < 0.0):
        raise ValueError("path positions must be non-decreasing")
    if window_m is None:
        if x.size < 2:
            raise ValueError("window_m is required for fewer than two path positions")
        window_m = (float(x[0]), float(x[-1]))
    start, end = map(float, window_m)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("window_m must contain finite increasing bounds")

    # 无效点不补零，也不跨越缺口连接；只保留两个端点均有效的真实区间。
    point_valid = accepted_mask & valid_mask & np.isfinite(force)
    interval_valid = point_valid[:-1] & point_valid[1:] & (dx > 0.0)
    segments = []
    for i in np.flatnonzero(interval_valid):
        low, high = max(start, x[i]), min(end, x[i + 1])
        if high <= low:
            continue
        slope = (force[i + 1] - force[i]) / dx[i]
        first = force[i] + slope * (low - x[i])
        last = force[i] + slope * (high - x[i])
        segments.append((float(low), float(high), float(first), float(last)))
    return segments, (start, end)
