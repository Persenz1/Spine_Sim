"""论文跑批和实体试验的协议接口；不替调用方选择阈值、样本或计算预算。

所有力为 N、长度为 m、时间为 s。ResearchSample 是重复单位，路径站点不是
独立重复。后续 case adapter 可把 sample 中的种子/落点映射到自己的地形输入。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core.config import BaseCaseSpec
from .metrics import _path_segments


@dataclass(frozen=True)
class LoadProtocol:
    """跨 case 的载荷对照方式；每个 case 内仍施加恒定总 P。"""

    mode: Literal["total_force", "pressure", "nominal_per_spine"]
    value: float

    def total_preload_N(self, *, area_m2: float, n_nominal: int) -> float:
        """value 单位依次为 N、Pa、N/名义刺；不规定任何实际逐刺分力。"""

        if self.mode == "total_force":
            result = self.value
        elif self.mode == "pressure":
            result = self.value * area_m2
        elif self.mode == "nominal_per_spine":
            result = self.value * n_nominal
        else:
            raise ValueError(f"unknown load comparison mode: {self.mode}")
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError("resolved total preload must be finite and positive")
        return float(result)


@dataclass(frozen=True)
class ResearchSample:
    """配对比较中复用同一表面、随机实现和落点，并记录数据集用途。"""

    surface_id: str
    surface_seed: int
    placement_seed: int
    placement_xy_m: tuple[float, float]
    split: Literal["train", "validation", "test"]


def iter_paired_cases(
    designs: Iterable[BaseCaseSpec], samples: Sequence[ResearchSample], *,
    sample_parameter: str = "sample",
) -> Iterator[BaseCaseSpec]:
    """惰性生成 design × 已声明样本，供现有 CampaignRunner 分批运行。

    在 parameters[sample_parameter] 注入样本字典，其值参与现有 case identity。
    不改写设计内部的几何/载荷；adapter 负责消费 sample 参数。有限设计资源和
    固定面积/固定间距等比较标签应由设计 case 自身保存。
    """

    for design in designs:
        for sample in samples:
            parameters = dict(design.parameters)
            parameters[sample_parameter] = asdict(sample)
            yield replace(design, parameters=parameters, tags=(*design.tags, sample.split))


@dataclass(frozen=True)
class EstablishmentProtocol:
    """在 search_window_m 内达到目标并保持指定距离。

    S_req 是合格保持窗口的起点减搜索起点；confirmed_at_m 是确认窗口结束的
    绝对位置，必须不超过搜索上限。允许低于阈值的总比例及最长连续距离同时
    生效；默认不允许掉落。站间力使用与路径积分相同的线性插值。
    """

    target_force_N: float
    persistence_distance_m: float
    search_window_m: tuple[float, float]
    allowed_below_fraction: float = 0.0
    max_contiguous_below_m: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_force_N) or self.target_force_N <= 0.0:
            raise ValueError("target_force_N must be finite and positive")
        if not np.isfinite(self.persistence_distance_m) or self.persistence_distance_m < 0.0:
            raise ValueError("persistence_distance_m must be finite and non-negative")
        if not 0.0 <= self.allowed_below_fraction < 1.0:
            raise ValueError("allowed_below_fraction must be in [0, 1)")
        if not np.isfinite(self.max_contiguous_below_m) or self.max_contiguous_below_m < 0.0:
            raise ValueError("max_contiguous_below_m must be finite and non-negative")
        low, high = self.search_window_m
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("search_window_m must have finite increasing bounds")


@dataclass(frozen=True)
class EstablishmentMetrics:
    established: bool | None
    status: str
    S_req_m: float | None
    confirmed_at_m: float | None
    observed_S_req_m: float | None
    complete_search: bool


def _threshold_spans(
    segments: Sequence[tuple[float, float, float, float]], threshold: float,
) -> list[tuple[float, float, bool]]:
    spans: list[tuple[float, float, bool]] = []
    for low, high, first, last in segments:
        first_good, last_good = first >= threshold, last >= threshold
        if first_good == last_good:
            pieces = [(low, high, first_good)]
        else:
            crossing = low + (high - low) * (threshold - first) / (last - first)
            pieces = [(low, crossing, first_good), (crossing, high, last_good)]
        for start, end, good in pieces:
            if end <= start:
                continue
            if spans and spans[-1][1] == start and spans[-1][2] == good:
                spans[-1] = (spans[-1][0], end, good)
            else:
                spans.append((start, end, good))
    return spans


def _first_persistent_start(
    block: Sequence[tuple[float, float, float, float]], protocol: EstablishmentProtocol,
) -> float | None:
    width = protocol.persistence_distance_m
    target = protocol.target_force_N
    if width == 0.0:
        for low, high, first, last in block:
            if first >= target:
                return low
            if last >= target:
                return low + (high - low) * (target - first) / (last - first)
        return None
    spans = _threshold_spans(block, target)
    bad = [(low, high) for low, high, good in spans if not good]
    high_start = block[-1][1] - width
    breakpoints = sorted({
        point for low, high, _ in spans
        for point in (low, high, low - width, high - width)
    } | {block[0][0], high_start})

    def overlaps(start: float) -> list[float]:
        return [max(0.0, min(start + width, high) - max(start, low)) for low, high in bad]

    for good_low, good_high, good in spans:
        if not good or good_low > high_start:
            continue
        limit = min(good_high, high_start)
        starts = sorted({good_low, limit} | {p for p in breakpoints if good_low < p < limit})
        # Below-threshold lengths are affine between these breakpoints. Intersect
        # their bounds instead of searching only at recorded stations.
        intervals = list(zip(starts[:-1], starts[1:])) or [(starts[0], starts[0])]
        for left, right in intervals:
            first_overlap, last_overlap = overlaps(left), overlaps(right)
            first_values = [sum(first_overlap), *first_overlap]
            last_values = [sum(last_overlap), *last_overlap]
            budgets = [protocol.allowed_below_fraction * width] + [
                protocol.max_contiguous_below_m
            ] * len(bad)
            feasible_low, feasible_high = left, right
            for initial, final, budget in zip(first_values, last_values, budgets):
                if initial <= budget and final <= budget:
                    continue
                if initial > budget and final > budget:
                    feasible_low, feasible_high = 1.0, 0.0
                    break
                crossing = left + (right - left) * (budget - initial) / (final - initial)
                if final > initial:
                    feasible_high = min(feasible_high, crossing)
                else:
                    feasible_low = max(feasible_low, crossing)
            if feasible_low <= feasible_high:
                return feasible_low
    return None


def evaluate_establishment(
    path_position_m: ArrayLike, resistance_force_N: ArrayLike, *,
    accepted: ArrayLike, valid: ArrayLike, protocol: EstablishmentProtocol,
) -> EstablishmentMetrics:
    """判断有限搜索内的持续建立，缺失路径不伪装成建立失败或零阻力。"""

    segments, window = _path_segments(
        path_position_m, resistance_force_N, accepted=accepted, valid=valid,
        window_m=protocol.search_window_m,
    )
    covered = sum(high - low for low, high, _, _ in segments)
    complete = bool(np.isclose(covered, window[1] - window[0], rtol=1e-10, atol=0.0))
    blocks: list[list[tuple[float, float, float, float]]] = []
    for segment in segments:
        if not blocks or blocks[-1][-1][1] != segment[0]:
            blocks.append([])
        blocks[-1].append(segment)
    candidates = [
        start for block in blocks
        if (start := _first_persistent_start(block, protocol)) is not None
    ]
    if protocol.persistence_distance_m == 0.0:
        x = np.asarray(path_position_m, dtype=float)
        force = np.asarray(resistance_force_N, dtype=float)
        include = (
            np.asarray(accepted, dtype=bool) & np.asarray(valid, dtype=bool)
            & np.isfinite(force) & (force >= protocol.target_force_N)
            & (x >= window[0]) & (x <= window[1])
        )
        candidates.extend(x[include].tolist())
    if candidates:
        start = min(candidates)
        # 缺口之前/之中可能发生更早的建立；仍能确认观察到的窗口成功。
        prefix_coverage = sum(max(0.0, min(high, start) - low) for low, high, _, _ in segments if low < start)
        earliest_known = bool(np.isclose(prefix_coverage, start - window[0], rtol=1e-10, atol=0.0))
        return EstablishmentMetrics(
            True, "established" if earliest_known else "established_earliest_unknown",
            start - window[0] if earliest_known else None,
            start + protocol.persistence_distance_m, start - window[0], complete,
        )
    return EstablishmentMetrics(
        False if complete else None, "not_established" if complete else "incomplete",
        None, None, None, complete,
    )


@dataclass(frozen=True)
class EstablishmentPopulation:
    sample_count: int
    successes: int
    failures: int
    unknown: int
    probability: float | None
    probability_lower: float | None
    probability_upper: float | None


def summarize_establishment(
    results: Iterable[EstablishmentMetrics],
) -> EstablishmentPopulation:
    """每个结果须来自一个声明的表面/落点重复，不能传相邻路径站点。"""

    values = [result.established for result in results]
    count = len(values)
    successes = sum(value is True for value in values)
    failures = sum(value is False for value in values)
    unknown = count - successes - failures
    return EstablishmentPopulation(
        count, successes, failures, unknown,
        successes / count if count and not unknown else None,
        successes / count if count else None,
        (successes + unknown) / count if count else None,
    )


@dataclass(frozen=True)
class SensorCalibration:
    """六维传感器校零、符号、坐标和参考点变换。

    rotation_world_from_sensor 为传感器分量到世界分量的正交旋转；
    sensor_origin_from_backplate_m 是世界系中背板参考点指向传感器原点的向量。
    force_sign 将校零后的传感器作用力转为壁面对阵列的力，须由装置自由体图确定。
    此变换不把砝码标称载荷自动当作阵列实际 P，也不反推逐刺力。
    """

    rotation_world_from_sensor: tuple[tuple[float, float, float], ...]
    sensor_origin_from_backplate_m: tuple[float, float, float]
    tare_sensor_wrench: tuple[float, float, float, float, float, float]
    force_sign: Literal[-1, 1]
    displacement_zero_m: float = 0.0
    time_offset_s: float = 0.0
    calibration_id: str = ""

    def wrench_to_backplate(self, sensor_wrench: ArrayLike) -> NDArray[np.float64]:
        """输入 (...,6) 的 (Fx,Fy,Fz,Mx,My,Mz)，输出同形的背板力旋量。"""

        measured = np.asarray(sensor_wrench, dtype=float)
        rotation = np.asarray(self.rotation_world_from_sensor, dtype=float)
        if measured.shape[-1] != 6 or rotation.shape != (3, 3):
            raise ValueError("sensor wrench must end in six components and rotation must be 3x3")
        if self.force_sign not in (-1, 1) or not np.allclose(rotation.T @ rotation, np.eye(3)):
            raise ValueError("sensor sign must be +/-1 and rotation must be orthogonal")
        if not np.isclose(np.linalg.det(rotation), 1.0):
            raise ValueError("sensor rotation must be right-handed")
        centered = (measured - np.asarray(self.tare_sensor_wrench)) * self.force_sign
        force = centered[..., :3] @ rotation.T
        moment = centered[..., 3:] @ rotation.T
        moment += np.cross(np.asarray(self.sensor_origin_from_backplate_m), force)
        return np.concatenate((force, moment), axis=-1)

    def align_displacement(self, displacement_m: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(displacement_m, dtype=float) - self.displacement_zero_m

    def align_time(self, time_s: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(time_s, dtype=float) + self.time_offset_s
