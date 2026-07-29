"""Geometry descriptors used to calibrate and validate height fields."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import TerrainConfigurationError


_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def _moments(values: NDArray[np.float64]) -> tuple[float, float]:
    centered = values - float(np.mean(values))
    standard_deviation = float(np.std(centered))
    if standard_deviation == 0.0:
        return 0.0, 3.0
    normalized = centered / standard_deviation
    return float(np.mean(normalized**3)), float(np.mean(normalized**4))


def _axis_acf(
    height: NDArray[np.float64],
    mask: NDArray[np.bool_],
    spacing_m: float,
    *,
    axis: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average normalized ACFs over representative valid lines."""

    lines = height if axis == 1 else height.T
    line_masks = mask if axis == 1 else mask.T
    indices = np.linspace(0, lines.shape[0] - 1, min(64, lines.shape[0]), dtype=int)
    correlations: list[NDArray[np.float64]] = []
    for index in indices:
        valid = line_masks[index]
        if np.count_nonzero(valid) < max(8, lines.shape[1] // 2):
            continue
        line = np.where(valid, lines[index], np.nan)
        mean = float(np.nanmean(line))
        line = np.where(valid, line - mean, 0.0)
        weights = valid.astype(np.float64)
        spectrum = np.fft.rfft(line, n=2 * line.size)
        weight_spectrum = np.fft.rfft(weights, n=2 * weights.size)
        numerator = np.fft.irfft(spectrum * np.conj(spectrum))[: line.size]
        denominator = np.fft.irfft(
            weight_spectrum * np.conj(weight_spectrum)
        )[: line.size]
        acf = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        if acf[0] > 0:
            correlations.append(acf / acf[0])
    if not correlations:
        return np.arange(lines.shape[1]) * spacing_m, np.full(
            lines.shape[1], np.nan
        )
    count = min(item.size for item in correlations)
    mean_acf = np.mean([item[:count] for item in correlations], axis=0)
    return np.arange(count, dtype=np.float64) * spacing_m, mean_acf


def _correlation_length(
    lags_m: NDArray[np.float64], acf: NDArray[np.float64]
) -> float:
    """Return the first 1/e ACF crossing."""

    if not np.any(np.isfinite(acf)):
        return float("nan")
    below = np.flatnonzero(acf <= math.exp(-1.0))
    return float(lags_m[below[0] if below.size else acf.size - 1])


def _axis_psd(
    height: NDArray[np.float64],
    mask: NDArray[np.bool_],
    spacing_m: float,
    *,
    axis: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lines = height if axis == 1 else height.T
    line_masks = mask if axis == 1 else mask.T
    indices = np.linspace(0, lines.shape[0] - 1, min(64, lines.shape[0]), dtype=int)
    powers: list[NDArray[np.float64]] = []
    window = np.hanning(lines.shape[1])
    window_energy = float(np.sum(window**2))
    for index in indices:
        valid = line_masks[index]
        if np.count_nonzero(valid) < max(8, lines.shape[1] // 2):
            continue
        line = np.where(valid, lines[index], np.nan)
        line = np.where(valid, line - float(np.nanmean(line)), 0.0)
        transform = np.fft.rfft(line * window)
        powers.append(np.abs(transform) ** 2 * spacing_m / window_energy)
    frequency = np.fft.rfftfreq(lines.shape[1], d=spacing_m)
    if not powers:
        return frequency, np.zeros_like(frequency)
    return frequency, np.mean(powers, axis=0)


def _radial_two_dimensional_psd(
    height: NDArray[np.float64],
    mask: NDArray[np.bool_],
    dx_m: float,
    dy_m: float,
    *,
    maximum_axis_samples: int = 512,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
    """Compute a radially averaged 2-D PSD on a bounded representative grid."""

    stride_y = max(1, math.ceil(height.shape[0] / maximum_axis_samples))
    stride_x = max(1, math.ceil(height.shape[1] / maximum_axis_samples))
    sampled = np.asarray(height[::stride_y, ::stride_x], dtype=np.float64)
    sampled_mask = mask[::stride_y, ::stride_x]
    mean = float(np.mean(sampled[sampled_mask]))
    sampled = np.where(sampled_mask, sampled - mean, 0.0)
    window_y = np.hanning(sampled.shape[0])
    window_x = np.hanning(sampled.shape[1])
    window = window_y[:, None] * window_x[None, :]
    window_energy = float(np.sum(window**2))
    transform = np.fft.rfft2(sampled * window)
    psd = (
        np.abs(transform) ** 2
        * (dx_m * stride_x)
        * (dy_m * stride_y)
        / max(window_energy, np.finfo(float).eps)
    )
    frequency_x = np.fft.rfftfreq(
        sampled.shape[1], d=dx_m * stride_x
    )
    frequency_y = np.fft.fftfreq(
        sampled.shape[0], d=dy_m * stride_y
    )
    radial_frequency = np.sqrt(
        frequency_y[:, None] ** 2 + frequency_x[None, :] ** 2
    )
    positive = radial_frequency > 0
    minimum = min(
        1.0 / (sampled.shape[1] * dx_m * stride_x),
        1.0 / (sampled.shape[0] * dy_m * stride_y),
    )
    maximum = float(np.max(radial_frequency))
    if maximum <= minimum:
        return (
            np.asarray([minimum]),
            np.asarray([float(np.mean(psd[positive]))]),
            list(sampled.shape),
        )
    edges = np.geomspace(minimum, maximum, 65)
    bins = np.digitize(radial_frequency[positive], edges) - 1
    valid_bins = (bins >= 0) & (bins < edges.size - 1)
    counts = np.bincount(
        bins[valid_bins], minlength=edges.size - 1
    ).astype(np.float64)
    sums = np.bincount(
        bins[valid_bins],
        weights=psd[positive][valid_bins],
        minlength=edges.size - 1,
    )
    populated = counts > 0
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers[populated], sums[populated] / counts[populated], list(
        sampled.shape
    )


def _local_extrema(
    height: NDArray[np.float64], mask: NDArray[np.bool_]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    center = height[1:-1, 1:-1]
    center_mask = mask[1:-1, 1:-1]
    peak = center_mask.copy()
    pit = center_mask.copy()
    strictly_peak = np.zeros_like(center_mask)
    strictly_pit = np.zeros_like(center_mask)
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == column_offset == 0:
                continue
            neighbor = height[
                1 + row_offset : height.shape[0] - 1 + row_offset,
                1 + column_offset : height.shape[1] - 1 + column_offset,
            ]
            neighbor_mask = mask[
                1 + row_offset : mask.shape[0] - 1 + row_offset,
                1 + column_offset : mask.shape[1] - 1 + column_offset,
            ]
            peak &= neighbor_mask & (center >= neighbor)
            pit &= neighbor_mask & (center <= neighbor)
            strictly_peak |= center > neighbor
            strictly_pit |= center < neighbor
    return peak & strictly_peak, pit & strictly_pit


def _typical_feature_size(
    extrema: NDArray[np.bool_],
    height: NDArray[np.float64],
    *,
    dx_m: float,
    dy_m: float,
    peak: bool,
) -> float:
    positions = np.argwhere(extrema)
    if positions.size == 0:
        return float("nan")
    sizes: list[float] = []
    maximum_radius = min(32, max(1, min(height.shape) // 8))
    for row, column in positions[:: max(1, math.ceil(len(positions) / 256))]:
        row += 1
        column += 1
        center = height[row, column]
        background = float(
            np.median(
                height[
                    max(0, row - maximum_radius) : row + maximum_radius + 1,
                    max(0, column - maximum_radius) : column + maximum_radius + 1,
                ]
            )
        )
        threshold = 0.5 * (center + background)
        radius_x = 1
        while radius_x < maximum_radius:
            left = height[row, max(0, column - radius_x)]
            right = height[row, min(height.shape[1] - 1, column + radius_x)]
            if (peak and (left < threshold or right < threshold)) or (
                not peak and (left > threshold or right > threshold)
            ):
                break
            radius_x += 1
        radius_y = 1
        while radius_y < maximum_radius:
            lower = height[max(0, row - radius_y), column]
            upper = height[min(height.shape[0] - 1, row + radius_y), column]
            if (peak and (lower < threshold or upper < threshold)) or (
                not peak and (lower > threshold or upper > threshold)
            ):
                break
            radius_y += 1
        sizes.append(2.0 * math.sqrt(radius_x * dx_m * radius_y * dy_m))
    return float(np.median(sizes))


def compute_descriptors(
    height_m: NDArray[np.floating],
    *,
    dx_m: float,
    dy_m: float,
    valid_mask: NDArray[np.bool_] | None = None,
    include_curves: bool = False,
) -> dict[str, Any]:
    """Compute distribution, spatial, slope, peak, and pit descriptors."""

    height = np.asarray(height_m, dtype=np.float64)
    if height.ndim != 2 or min(height.shape) < 3:
        raise TerrainConfigurationError("descriptor input must be at least 3x3")
    if dx_m <= 0 or dy_m <= 0:
        raise TerrainConfigurationError("descriptor spacing must be positive")
    mask = (
        np.ones(height.shape, dtype=np.bool_)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=np.bool_)
    )
    if mask.shape != height.shape:
        raise TerrainConfigurationError("descriptor mask shape mismatch")
    mask &= np.isfinite(height)
    values = height[mask]
    if values.size < 9:
        raise TerrainConfigurationError("descriptor input has too few valid points")
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values))
    skewness, kurtosis = _moments(values)
    quantiles = np.percentile(values, _PERCENTILES)

    valid_x = mask[:, 1:] & mask[:, :-1]
    valid_y = mask[1:, :] & mask[:-1, :]
    slope_x = np.diff(height, axis=1)[valid_x] / dx_m
    slope_y = np.diff(height, axis=0)[valid_y] / dy_m
    lag_x, acf_x = _axis_acf(height, mask, dx_m, axis=1)
    lag_y, acf_y = _axis_acf(height, mask, dy_m, axis=0)
    correlation_x = _correlation_length(lag_x, acf_x)
    correlation_y = _correlation_length(lag_y, acf_y)
    frequency_x, psd_x = _axis_psd(height, mask, dx_m, axis=1)
    frequency_y, psd_y = _axis_psd(height, mask, dy_m, axis=0)
    peaks, pits = _local_extrema(height, mask)
    valid_area_m2 = float(np.count_nonzero(mask)) * dx_m * dy_m
    interior_height = height[1:-1, 1:-1]
    pit_depths = -interior_height[pits]
    result: dict[str, Any] = {
        "shape_yx": list(height.shape),
        "valid_fraction": float(np.mean(mask)),
        "height": {
            "mean_m": mean,
            "rms_about_mean_m": float(
                np.sqrt(np.mean(np.square(values - mean)))
            ),
            "standard_deviation_m": standard_deviation,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "quantiles_m": {
                str(percentile): float(value)
                for percentile, value in zip(_PERCENTILES, quantiles, strict=True)
            },
        },
        "spatial": {
            "correlation_length_x_m": correlation_x,
            "correlation_length_y_m": correlation_y,
            "anisotropy_ratio_x_over_y": (
                correlation_x / correlation_y
                if correlation_y > 0 and math.isfinite(correlation_y)
                else float("nan")
            ),
        },
        "slope": {
            "x_rms": float(np.sqrt(np.mean(slope_x**2))),
            "y_rms": float(np.sqrt(np.mean(slope_y**2))),
            "x_quantiles": {
                str(item): float(value)
                for item, value in zip(
                    _PERCENTILES,
                    np.percentile(slope_x, _PERCENTILES),
                    strict=True,
                )
            },
            "y_quantiles": {
                str(item): float(value)
                for item, value in zip(
                    _PERCENTILES,
                    np.percentile(slope_y, _PERCENTILES),
                    strict=True,
                )
            },
        },
        "features": {
            "peak_density_per_m2": float(np.count_nonzero(peaks) / valid_area_m2),
            "pit_density_per_m2": float(np.count_nonzero(pits) / valid_area_m2),
            "typical_peak_size_m": _typical_feature_size(
                peaks, height, dx_m=dx_m, dy_m=dy_m, peak=True
            ),
            "typical_pit_size_m": _typical_feature_size(
                pits, height, dx_m=dx_m, dy_m=dy_m, peak=False
            ),
            "typical_pit_depth_m": (
                float(np.median(pit_depths[pit_depths > 0]))
                if np.any(pit_depths > 0)
                else 0.0
            ),
        },
        "artifact_checks": artifact_checks(height, mask),
    }
    if include_curves:
        radial_frequency, radial_psd, psd_2d_shape = _radial_two_dimensional_psd(
            height, mask, dx_m, dy_m
        )
        result["curves"] = {
            "acf_lag_x_m": lag_x.tolist(),
            "acf_x": acf_x.tolist(),
            "acf_lag_y_m": lag_y.tolist(),
            "acf_y": acf_y.tolist(),
            "frequency_x_per_m": frequency_x.tolist(),
            "psd_x_m3": psd_x.tolist(),
            "frequency_y_per_m": frequency_y.tolist(),
            "psd_y_m3": psd_y.tolist(),
            "radial_frequency_2d_per_m": radial_frequency.tolist(),
            "radial_psd_2d_m4": radial_psd.tolist(),
            "psd_2d_analysis_shape_yx": psd_2d_shape,
        }
    return result


def artifact_checks(
    height_m: NDArray[np.float64], valid_mask: NDArray[np.bool_]
) -> dict[str, Any]:
    """Return lightweight, threshold-free evidence for common synthesis artifacts."""

    height = np.asarray(height_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    values = height[valid]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    extreme_count = (
        int(np.count_nonzero(np.abs(values - median) > 12.0 * robust_sigma))
        if robust_sigma > 0
        else 0
    )
    opposite_edge_x = np.corrcoef(height[:, 0], height[:, -1])[0, 1]
    opposite_edge_y = np.corrcoef(height[0, :], height[-1, :])[0, 1]
    edge_jump_x = float(np.sqrt(np.mean((height[:, 0] - height[:, -1]) ** 2)))
    edge_jump_y = float(np.sqrt(np.mean((height[0, :] - height[-1, :]) ** 2)))
    differences_x = np.diff(height, axis=1)
    differences_y = np.diff(height, axis=0)
    interior_step_x = float(np.sqrt(np.mean(differences_x**2)))
    interior_step_y = float(np.sqrt(np.mean(differences_y**2)))
    line_rms_x = np.sqrt(np.mean(differences_x**2, axis=0))
    line_rms_y = np.sqrt(np.mean(differences_y**2, axis=1))
    mirror_x = np.corrcoef(height.ravel(), height[:, ::-1].ravel())[0, 1]
    mirror_y = np.corrcoef(height.ravel(), height[::-1, :].ravel())[0, 1]
    return {
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(height))),
        "extreme_outlier_count_over_12_robust_sigma": extreme_count,
        "opposite_edge_correlation_x": float(opposite_edge_x),
        "opposite_edge_correlation_y": float(opposite_edge_y),
        "boundary_jump_over_interior_rms_x": (
            edge_jump_x / interior_step_x if interior_step_x > 0 else 0.0
        ),
        "boundary_jump_over_interior_rms_y": (
            edge_jump_y / interior_step_y if interior_step_y > 0 else 0.0
        ),
        "maximum_column_step_rms_over_median": (
            float(np.max(line_rms_x) / np.median(line_rms_x))
            if np.median(line_rms_x) > 0
            else 0.0
        ),
        "maximum_row_step_rms_over_median": (
            float(np.max(line_rms_y) / np.median(line_rms_y))
            if np.median(line_rms_y) > 0
            else 0.0
        ),
        "mirror_correlation_x": float(mirror_x),
        "mirror_correlation_y": float(mirror_y),
    }
