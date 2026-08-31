"""可在任意有限全局网格上求值的解析地形夹具。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import TerrainConfigurationError


def _axis(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """校验非空有限的一维坐标轴。"""

    axis = np.asarray(value, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
        raise TerrainConfigurationError(f"{name} must be a non-empty finite 1-D array")
    return axis


def _number(parameters: Mapping[str, Any], name: str, default: float) -> float:
    """读取有限数值参数或默认值。"""

    try:
        value = float(parameters.get(name, default))
    except (TypeError, ValueError) as exc:
        raise TerrainConfigurationError(f"{name} must be numeric") from exc
    if not np.isfinite(value):
        raise TerrainConfigurationError(f"{name} must be finite")
    return value


def _positive(parameters: Mapping[str, Any], name: str, default: float) -> float:
    """读取有限正参数或默认值。"""

    value = _number(parameters, name, default)
    if value <= 0:
        raise TerrainConfigurationError(f"{name} must be positive")
    return value


def _bump(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    amplitude_m: float,
    center_x_m: float,
    center_y_m: float,
    sigma_x_m: float,
    sigma_y_m: float,
) -> NDArray[np.float64]:
    """在张量积 x/y 网格上计算一个各向异性二维高斯凸包。"""

    exponent = (
        ((x[None, :] - center_x_m) / sigma_x_m) ** 2
        + ((y[:, None] - center_y_m) / sigma_y_m) ** 2
    )
    return amplitude_m * np.exp(-0.5 * exponent)


def evaluate_analytic(
    kind: str,
    x_global_m: ArrayLike,
    y_global_m: ArrayLike,
    parameters: Mapping[str, Any] | None = None,
) -> NDArray[np.float64]:
    """在任意全局 x/y 节点数组上计算命名解析夹具。"""

    x = _axis(x_global_m, "x_global_m")
    y = _axis(y_global_m, "y_global_m")
    p = parameters or {}
    base = _number(p, "offset_m", 0.0)

    # 每种 fixture 都直接使用全局坐标，不依赖请求窗口或切片顺序。
    if kind == "plane":
        return np.full((y.size, x.size), base, dtype=np.float64)

    if kind in {"slope", "cross_slope"}:
        slope_x = _number(p, "slope_x", 0.0)
        slope_y = _number(p, "slope_y", 0.0)
        origin_x = _number(p, "origin_x_m", 0.0)
        origin_y = _number(p, "origin_y_m", 0.0)
        return (
            base
            + slope_x * (x[None, :] - origin_x)
            + slope_y * (y[:, None] - origin_y)
        )

    if kind == "smooth_bump":
        return base + _bump(
            x,
            y,
            amplitude_m=_number(p, "amplitude_m", 100e-6),
            center_x_m=_number(p, "center_x_m", 0.0),
            center_y_m=_number(p, "center_y_m", 0.0),
            sigma_x_m=_positive(p, "sigma_x_m", 100e-6),
            sigma_y_m=_positive(p, "sigma_y_m", 100e-6),
        )

    if kind == "double_bump":
        common_sigma_x = _positive(p, "sigma_x_m", 100e-6)
        common_sigma_y = _positive(p, "sigma_y_m", 100e-6)
        first = _bump(
            x,
            y,
            amplitude_m=_number(p, "amplitude_1_m", 100e-6),
            center_x_m=_number(p, "center_1_x_m", -250e-6),
            center_y_m=_number(p, "center_1_y_m", 0.0),
            sigma_x_m=_positive(p, "sigma_1_x_m", common_sigma_x),
            sigma_y_m=_positive(p, "sigma_1_y_m", common_sigma_y),
        )
        second = _bump(
            x,
            y,
            amplitude_m=_number(p, "amplitude_2_m", 100e-6),
            center_x_m=_number(p, "center_2_x_m", 250e-6),
            center_y_m=_number(p, "center_2_y_m", 0.0),
            sigma_x_m=_positive(p, "sigma_2_x_m", common_sigma_x),
            sigma_y_m=_positive(p, "sigma_2_y_m", common_sigma_y),
        )
        return base + first + second

    if kind == "sine_1d":
        amplitude = _number(p, "amplitude_m", 50e-6)
        wavelength = _positive(p, "wavelength_m", 500e-6)
        phase = _number(p, "phase_rad", 0.0)
        origin_x = _number(p, "origin_x_m", 0.0)
        line = base + amplitude * np.sin(
            2.0 * np.pi * (x - origin_x) / wavelength + phase
        )
        return np.broadcast_to(line, (y.size, x.size)).copy()

    if kind == "sine_2d":
        amplitude_x = _number(p, "amplitude_x_m", 30e-6)
        amplitude_y = _number(p, "amplitude_y_m", 30e-6)
        wavelength_x = _positive(p, "wavelength_x_m", 500e-6)
        wavelength_y = _positive(p, "wavelength_y_m", 500e-6)
        phase_x = _number(p, "phase_x_rad", 0.0)
        phase_y = _number(p, "phase_y_rad", 0.0)
        return (
            base
            + amplitude_x
            * np.sin(2.0 * np.pi * x[None, :] / wavelength_x + phase_x)
            + amplitude_y
            * np.sin(2.0 * np.pi * y[:, None] / wavelength_y + phase_y)
        )

    raise TerrainConfigurationError(
        "analytic kind must be plane, slope, smooth_bump, double_bump, "
        "sine_1d, sine_2d or cross_slope"
    )
