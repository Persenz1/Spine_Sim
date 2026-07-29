"""Material-specific synthetic terrain algorithms.

These are intentionally separate models.  Sandpaper uses measured-patch
quilting when a calibrated Hirox source is present and a granular fallback for
provisional grits.  Brick uses base/directional/pore/fired-grain layers.
Concrete uses mortar/aggregate/air-void/finish layers.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from .errors import TerrainConfigurationError
from .measured import (
    MeasuredSurface,
    load_measured_surface,
    resample_measured_patch,
    sha256_file,
)

MaterialBackend = Literal["cpu", "cuda"]


def _normalize_rms(
    field: NDArray[np.floating],
    target_rms_m: float,
    *,
    backend: MaterialBackend = "cpu",
) -> NDArray[np.float32]:
    if backend == "cuda":
        try:
            import cupy as cp  # type: ignore
        except ImportError as exc:
            raise TerrainConfigurationError(
                "CUDA material generation requested but CuPy is not installed"
            ) from exc
        result_gpu = cp.asarray(field, dtype=cp.float32)
        mean = float(cp.mean(result_gpu, dtype=cp.float64).get())
        result_gpu -= np.float32(mean)
        rms = float(
            cp.sqrt(
                cp.mean(result_gpu * result_gpu, dtype=cp.float64)
            ).get()
        )
        if rms > 0.0:
            result_gpu *= np.float32(target_rms_m / rms)
        else:
            result_gpu.fill(0.0)
        return np.asarray(cp.asnumpy(result_gpu), dtype=np.float32)
    if backend != "cpu":
        raise TerrainConfigurationError(
            f"material backend must be 'cpu' or 'cuda', got {backend!r}"
        )
    result = np.asarray(field, dtype=np.float64)
    result -= float(np.mean(result))
    rms = float(np.sqrt(np.mean(result**2)))
    if rms > 0.0:
        result *= target_rms_m / rms
    else:
        result.fill(0.0)
    return np.asarray(result, dtype=np.float32)


def _smooth_latent(field: NDArray[np.float32], *, passes: int = 2) -> None:
    """Apply a compact non-periodic binomial smoother in place."""

    for _ in range(passes):
        padded_x = np.pad(field, ((0, 0), (2, 2)), mode="reflect")
        field[:] = (
            padded_x[:, :-4]
            + 4.0 * padded_x[:, 1:-3]
            + 6.0 * padded_x[:, 2:-2]
            + 4.0 * padded_x[:, 3:-1]
            + padded_x[:, 4:]
        ) / 16.0
        padded_y = np.pad(field, ((2, 2), (0, 0)), mode="reflect")
        field[:] = (
            padded_y[:-4, :]
            + 4.0 * padded_y[1:-3, :]
            + 6.0 * padded_y[2:-2, :]
            + 4.0 * padded_y[3:-1, :]
            + padded_y[4:, :]
        ) / 16.0


def correlated_field(
    shape: tuple[int, int],
    *,
    dx_m: float,
    dy_m: float,
    correlation_x_m: float,
    correlation_y_m: float,
    rng: np.random.Generator,
    target_rms_m: float,
    backend: MaterialBackend = "cpu",
    cuda_tile_rows: int = 512,
) -> NDArray[np.float32]:
    """Generate an overscanned, non-periodic anisotropic random field.

    A smoothed latent grid is bilinearly expanded.  Unlike a bare FFT field,
    opposite output edges do not share periodic boundary values.
    """

    ny, nx = shape
    step_x = max(1, int(round(correlation_x_m / max(3.0 * dx_m, 1e-30))))
    step_y = max(1, int(round(correlation_y_m / max(3.0 * dy_m, 1e-30))))
    latent_nx = math.ceil((nx - 1) / step_x) + 7
    latent_ny = math.ceil((ny - 1) / step_y) + 7
    latent = rng.standard_normal((latent_ny, latent_nx), dtype=np.float32)
    if backend == "cuda":
        return _correlated_field_cuda(
            latent,
            shape=shape,
            step_x=step_x,
            step_y=step_y,
            target_rms_m=target_rms_m,
            tile_rows=cuda_tile_rows,
        )
    if backend != "cpu":
        raise TerrainConfigurationError(
            f"material backend must be 'cpu' or 'cuda', got {backend!r}"
        )
    _smooth_latent(latent, passes=2)

    x_float = np.arange(nx, dtype=np.float64) / step_x + 3.0
    y_float = np.arange(ny, dtype=np.float64) / step_y + 3.0
    x0 = np.floor(x_float).astype(np.intp)
    y0 = np.floor(y_float).astype(np.intp)
    tx = np.asarray(x_float - x0, dtype=np.float32)
    ty = np.asarray(y_float - y0, dtype=np.float32)
    result = np.empty(shape, dtype=np.float32)
    for row in range(ny):
        lower = (1.0 - tx) * latent[y0[row], x0] + tx * latent[
            y0[row], x0 + 1
        ]
        upper = (1.0 - tx) * latent[y0[row] + 1, x0] + tx * latent[
            y0[row] + 1, x0 + 1
        ]
        result[row] = (1.0 - ty[row]) * lower + ty[row] * upper
    return _normalize_rms(result, target_rms_m)


def _correlated_field_cuda(
    latent: NDArray[np.float32],
    *,
    shape: tuple[int, int],
    step_x: int,
    step_y: int,
    target_rms_m: float,
    tile_rows: int,
) -> NDArray[np.float32]:
    """Expand and normalize one latent field on CUDA with bounded memory.

    Random draws remain on NumPy's PCG64 stream so the seed semantics do not
    depend on the installed CuPy version.  Interpolation is tiled in Y and
    copied directly into the final host array, which keeps full M3 campaign
    regions within an 8 GB device.
    """

    if tile_rows < 1:
        raise TerrainConfigurationError("cuda_tile_rows must be positive")
    try:
        import cupy as cp  # type: ignore
    except ImportError as exc:
        raise TerrainConfigurationError(
            "CUDA material generation requested but CuPy is not installed"
        ) from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise TerrainConfigurationError(
            "CUDA material generation requested but no CUDA device is available"
        )

    latent_gpu = cp.asarray(latent)
    for _ in range(2):
        padded_x = cp.pad(latent_gpu, ((0, 0), (2, 2)), mode="reflect")
        latent_gpu = (
            padded_x[:, :-4]
            + 4.0 * padded_x[:, 1:-3]
            + 6.0 * padded_x[:, 2:-2]
            + 4.0 * padded_x[:, 3:-1]
            + padded_x[:, 4:]
        ) / 16.0
        del padded_x
        padded_y = cp.pad(latent_gpu, ((2, 2), (0, 0)), mode="reflect")
        latent_gpu = (
            padded_y[:-4, :]
            + 4.0 * padded_y[1:-3, :]
            + 6.0 * padded_y[2:-2, :]
            + 4.0 * padded_y[3:-1, :]
            + padded_y[4:, :]
        ) / 16.0
        del padded_y

    ny, nx = shape
    x_index = cp.arange(nx, dtype=cp.int64)
    x0 = x_index // step_x + 3
    tx = (x_index % step_x).astype(cp.float32) / np.float32(step_x)
    result = np.empty(shape, dtype=np.float32)
    total = 0.0
    total_squares = 0.0
    sample_count = ny * nx
    for start in range(0, ny, tile_rows):
        stop = min(ny, start + tile_rows)
        y_index = cp.arange(start, stop, dtype=cp.int64)
        y0 = y_index // step_y + 3
        ty = (y_index % step_y).astype(cp.float32) / np.float32(step_y)
        lower = (1.0 - tx)[None, :] * latent_gpu[
            y0[:, None], x0[None, :]
        ] + tx[None, :] * latent_gpu[y0[:, None], x0[None, :] + 1]
        upper = (1.0 - tx)[None, :] * latent_gpu[
            y0[:, None] + 1, x0[None, :]
        ] + tx[None, :] * latent_gpu[y0[:, None] + 1, x0[None, :] + 1]
        tile = (1.0 - ty)[:, None] * lower + ty[:, None] * upper
        total += float(cp.sum(tile, dtype=cp.float64).get())
        total_squares += float(cp.sum(tile * tile, dtype=cp.float64).get())
        result[start:stop] = cp.asnumpy(tile)
        del y_index, y0, ty, lower, upper, tile

    mean = total / sample_count
    variance = max(0.0, total_squares / sample_count - mean * mean)
    rms = math.sqrt(variance)
    scale = target_rms_m / rms if rms > 0.0 else 0.0
    for start in range(0, ny, tile_rows):
        stop = min(ny, start + tile_rows)
        if scale:
            result[start:stop] = np.asarray(
                (result[start:stop] - mean) * scale,
                dtype=np.float32,
            )
        else:
            result[start:stop].fill(0.0)
    return result


def _non_gaussian(
    field: NDArray[np.float32],
    *,
    exponent: float,
    skew_bias: float = 0.0,
    backend: MaterialBackend = "cpu",
) -> NDArray[np.float32]:
    if backend == "cuda":
        try:
            import cupy as cp  # type: ignore
        except ImportError as exc:
            raise TerrainConfigurationError(
                "CUDA material generation requested but CuPy is not installed"
            ) from exc
        source_gpu = cp.asarray(field, dtype=cp.float32)
        rms = float(
            cp.sqrt(cp.mean(source_gpu * source_gpu, dtype=cp.float64)).get()
        )
        if rms == 0.0:
            return np.asarray(field, dtype=np.float32)
        normalized = source_gpu / np.float32(rms)
        transformed = cp.sign(normalized) * cp.abs(normalized) ** exponent
        if skew_bias:
            transformed += np.float32(skew_bias) * (
                normalized * normalized - 1.0
            )
        del source_gpu, normalized
        return _normalize_rms(
            cp.asnumpy(transformed),
            rms,
            backend="cuda",
        )
    rms = float(np.sqrt(np.mean(np.asarray(field, dtype=np.float64) ** 2)))
    if rms == 0:
        return field
    normalized = np.asarray(field, dtype=np.float64) / rms
    transformed = np.sign(normalized) * np.abs(normalized) ** exponent
    if skew_bias:
        transformed += skew_bias * (normalized**2 - 1.0)
    return _normalize_rms(transformed, rms)


def _feature_count(
    density_per_m2: float,
    shape: tuple[int, int],
    dx_m: float,
    dy_m: float,
    rng: np.random.Generator,
) -> int:
    area = max(dx_m * dy_m, (shape[1] - 1) * dx_m * (shape[0] - 1) * dy_m)
    return int(rng.poisson(max(0.0, density_per_m2 * area)))


def add_irregular_features(
    height_m: NDArray[np.float32],
    *,
    dx_m: float,
    dy_m: float,
    density_per_m2: float,
    diameter_median_m: float,
    diameter_log_sigma: float,
    aspect_ratio_median: float,
    aspect_ratio_log_sigma: float,
    amplitude_median_m: float,
    amplitude_log_sigma: float,
    edge_power: float,
    boundary_roughness: float,
    rng: np.random.Generator,
    sign_mode: str,
    positive_probability: float = 0.5,
    cluster_probability: float = 0.0,
    backend: MaterialBackend = "cpu",
) -> dict[str, Any]:
    """Stamp randomized, rough-edged, rotated features onto ``height_m``."""

    if backend == "cuda":
        return _add_irregular_features_cuda(
            height_m,
            dx_m=dx_m,
            dy_m=dy_m,
            density_per_m2=density_per_m2,
            diameter_median_m=diameter_median_m,
            diameter_log_sigma=diameter_log_sigma,
            aspect_ratio_median=aspect_ratio_median,
            aspect_ratio_log_sigma=aspect_ratio_log_sigma,
            amplitude_median_m=amplitude_median_m,
            amplitude_log_sigma=amplitude_log_sigma,
            edge_power=edge_power,
            boundary_roughness=boundary_roughness,
            rng=rng,
            sign_mode=sign_mode,
            positive_probability=positive_probability,
            cluster_probability=cluster_probability,
        )
    if backend != "cpu":
        raise TerrainConfigurationError(
            f"material backend must be 'cpu' or 'cuda', got {backend!r}"
        )
    count = _feature_count(density_per_m2, height_m.shape, dx_m, dy_m, rng)
    ny, nx = height_m.shape
    previous_center: tuple[float, float] | None = None
    positive_count = 0
    sampled_diameters: list[float] = []
    sampled_amplitudes: list[float] = []
    for _ in range(count):
        diameter = float(
            diameter_median_m * rng.lognormal(0.0, diameter_log_sigma)
        )
        diameter = min(
            diameter,
            0.45 * max((nx - 1) * dx_m, (ny - 1) * dy_m),
        )
        aspect = max(
            1.0, float(aspect_ratio_median * rng.lognormal(0.0, aspect_ratio_log_sigma))
        )
        semi_major = max(1.25 * min(dx_m, dy_m), 0.5 * diameter * math.sqrt(aspect))
        semi_minor = max(1.25 * min(dx_m, dy_m), 0.5 * diameter / math.sqrt(aspect))
        if previous_center is not None and rng.random() < cluster_probability:
            center_y = previous_center[0] + rng.normal(0.0, diameter / dy_m)
            center_x = previous_center[1] + rng.normal(0.0, diameter / dx_m)
            center_y = float(np.clip(center_y, 0, ny - 1))
            center_x = float(np.clip(center_x, 0, nx - 1))
        else:
            center_y = float(rng.uniform(0, ny - 1))
            center_x = float(rng.uniform(0, nx - 1))
        previous_center = (center_y, center_x)
        angle = float(rng.uniform(0.0, math.pi))
        cosine, sine = math.cos(angle), math.sin(angle)
        radius_x = max(
            2, int(math.ceil(1.35 * (semi_major + semi_minor) / dx_m))
        )
        radius_y = max(
            2, int(math.ceil(1.35 * (semi_major + semi_minor) / dy_m))
        )
        x_start = max(0, int(math.floor(center_x)) - radius_x)
        x_stop = min(nx, int(math.floor(center_x)) + radius_x + 1)
        y_start = max(0, int(math.floor(center_y)) - radius_y)
        y_stop = min(ny, int(math.floor(center_y)) + radius_y + 1)
        x = (np.arange(x_start, x_stop) - center_x) * dx_m
        y = (np.arange(y_start, y_stop) - center_y) * dy_m
        xx, yy = np.meshgrid(x, y)
        rotated_x = cosine * xx + sine * yy
        rotated_y = -sine * xx + cosine * yy
        normalized_x = rotated_x / semi_major
        normalized_y = rotated_y / semi_minor
        radius = np.sqrt(normalized_x**2 + normalized_y**2)
        theta = np.arctan2(normalized_y, normalized_x)
        harmonic = int(rng.integers(3, 8))
        phase_a, phase_b = rng.uniform(0.0, 2.0 * math.pi, size=2)
        boundary = 1.0 + boundary_roughness * (
            0.62 * np.sin(harmonic * theta + phase_a)
            + 0.38 * np.sin((harmonic + 2) * theta + phase_b)
        )
        normalized_radius = radius / np.maximum(boundary, 0.55)
        profile = np.clip(1.0 - normalized_radius, 0.0, 1.0) ** edge_power
        profile *= 1.0 + 0.10 * np.sin(
            (harmonic + 3) * theta + phase_b
        ) * np.clip(1.0 - normalized_radius, 0.0, 1.0)
        amplitude = float(
            amplitude_median_m * rng.lognormal(0.0, amplitude_log_sigma)
        )
        if sign_mode == "negative":
            sign = -1.0
        elif sign_mode == "positive":
            sign = 1.0
        elif sign_mode == "mixed":
            sign = 1.0 if rng.random() < positive_probability else -1.0
        else:
            raise TerrainConfigurationError(f"unsupported sign_mode {sign_mode!r}")
        positive_count += int(sign > 0)
        height_m[y_start:y_stop, x_start:x_stop] += np.asarray(
            sign * amplitude * profile, dtype=np.float32
        )
        sampled_diameters.append(diameter)
        sampled_amplitudes.append(amplitude)
    return {
        "realized_count": count,
        "realized_positive_count": positive_count,
        "sampled_diameter_median_m": (
            float(np.median(sampled_diameters)) if sampled_diameters else None
        ),
        "sampled_amplitude_median_m": (
            float(np.median(sampled_amplitudes)) if sampled_amplitudes else None
        ),
    }


@lru_cache(maxsize=1)
def _cuda_irregular_feature_kernel() -> Any:
    try:
        import cupy as cp  # type: ignore
    except ImportError as exc:
        raise TerrainConfigurationError(
            "CUDA material generation requested but CuPy is not installed"
        ) from exc
    return cp.RawKernel(
        r"""
        extern "C" __global__
        void stamp_irregular_feature(
            float* height,
            const int nx,
            const int x_start,
            const int x_stop,
            const int y_start,
            const int y_stop,
            const float center_x,
            const float center_y,
            const float dx,
            const float dy,
            const float cosine,
            const float sine,
            const float semi_major,
            const float semi_minor,
            const int harmonic,
            const float phase_a,
            const float phase_b,
            const float edge_power,
            const float boundary_roughness,
            const float signed_amplitude
        ) {
            const int width = x_stop - x_start;
            const int feature_size = width * (y_stop - y_start);
            for (
                int linear = blockDim.x * blockIdx.x + threadIdx.x;
                linear < feature_size;
                linear += blockDim.x * gridDim.x
            ) {
                const int local_y = linear / width;
                const int local_x = linear - local_y * width;
                const int ix = x_start + local_x;
                const int iy = y_start + local_y;
                const float x = ((float)ix - center_x) * dx;
                const float y = ((float)iy - center_y) * dy;
                const float rotated_x = cosine * x + sine * y;
                const float rotated_y = -sine * x + cosine * y;
                const float normalized_x = rotated_x / semi_major;
                const float normalized_y = rotated_y / semi_minor;
                const float radius = sqrtf(
                    normalized_x * normalized_x
                    + normalized_y * normalized_y
                );
                const float theta = atan2f(normalized_y, normalized_x);
                const float boundary = 1.0f + boundary_roughness * (
                    0.62f * sinf((float)harmonic * theta + phase_a)
                    + 0.38f * sinf(
                        ((float)harmonic + 2.0f) * theta + phase_b
                    )
                );
                const float normalized_radius = radius / fmaxf(boundary, 0.55f);
                const float base = fminf(
                    1.0f,
                    fmaxf(1.0f - normalized_radius, 0.0f)
                );
                float profile = powf(base, edge_power);
                profile *= 1.0f + 0.10f * sinf(
                    ((float)harmonic + 3.0f) * theta + phase_b
                ) * base;
                height[iy * nx + ix] += signed_amplitude * profile;
            }
        }
        """,
        "stamp_irregular_feature",
    )


def _add_irregular_features_cuda(
    height_m: NDArray[np.float32],
    *,
    dx_m: float,
    dy_m: float,
    density_per_m2: float,
    diameter_median_m: float,
    diameter_log_sigma: float,
    aspect_ratio_median: float,
    aspect_ratio_log_sigma: float,
    amplitude_median_m: float,
    amplitude_log_sigma: float,
    edge_power: float,
    boundary_roughness: float,
    rng: np.random.Generator,
    sign_mode: str,
    positive_probability: float,
    cluster_probability: float,
) -> dict[str, Any]:
    """Stamp features on CUDA while preserving the CPU RNG draw sequence."""

    try:
        import cupy as cp  # type: ignore
    except ImportError as exc:
        raise TerrainConfigurationError(
            "CUDA material generation requested but CuPy is not installed"
        ) from exc
    count = _feature_count(density_per_m2, height_m.shape, dx_m, dy_m, rng)
    ny, nx = height_m.shape
    previous_center: tuple[float, float] | None = None
    positive_count = 0
    sampled_diameters: list[float] = []
    sampled_amplitudes: list[float] = []
    feature_arguments: list[tuple[Any, ...]] = []
    for _ in range(count):
        diameter = float(
            diameter_median_m * rng.lognormal(0.0, diameter_log_sigma)
        )
        diameter = min(
            diameter,
            0.45 * max((nx - 1) * dx_m, (ny - 1) * dy_m),
        )
        aspect = max(
            1.0,
            float(
                aspect_ratio_median
                * rng.lognormal(0.0, aspect_ratio_log_sigma)
            ),
        )
        semi_major = max(
            1.25 * min(dx_m, dy_m),
            0.5 * diameter * math.sqrt(aspect),
        )
        semi_minor = max(
            1.25 * min(dx_m, dy_m),
            0.5 * diameter / math.sqrt(aspect),
        )
        if previous_center is not None and rng.random() < cluster_probability:
            center_y = previous_center[0] + rng.normal(0.0, diameter / dy_m)
            center_x = previous_center[1] + rng.normal(0.0, diameter / dx_m)
            center_y = float(np.clip(center_y, 0, ny - 1))
            center_x = float(np.clip(center_x, 0, nx - 1))
        else:
            center_y = float(rng.uniform(0, ny - 1))
            center_x = float(rng.uniform(0, nx - 1))
        previous_center = (center_y, center_x)
        angle = float(rng.uniform(0.0, math.pi))
        cosine, sine = math.cos(angle), math.sin(angle)
        radius_x = max(
            2,
            int(math.ceil(1.35 * (semi_major + semi_minor) / dx_m)),
        )
        radius_y = max(
            2,
            int(math.ceil(1.35 * (semi_major + semi_minor) / dy_m)),
        )
        x_start = max(0, int(math.floor(center_x)) - radius_x)
        x_stop = min(nx, int(math.floor(center_x)) + radius_x + 1)
        y_start = max(0, int(math.floor(center_y)) - radius_y)
        y_stop = min(ny, int(math.floor(center_y)) + radius_y + 1)
        harmonic = int(rng.integers(3, 8))
        phase_a, phase_b = rng.uniform(0.0, 2.0 * math.pi, size=2)
        amplitude = float(
            amplitude_median_m * rng.lognormal(0.0, amplitude_log_sigma)
        )
        if sign_mode == "negative":
            sign = -1.0
        elif sign_mode == "positive":
            sign = 1.0
        elif sign_mode == "mixed":
            sign = 1.0 if rng.random() < positive_probability else -1.0
        else:
            raise TerrainConfigurationError(
                f"unsupported sign_mode {sign_mode!r}"
            )
        positive_count += int(sign > 0)
        feature_arguments.append(
            (
                np.int32(nx),
                np.int32(x_start),
                np.int32(x_stop),
                np.int32(y_start),
                np.int32(y_stop),
                np.float32(center_x),
                np.float32(center_y),
                np.float32(dx_m),
                np.float32(dy_m),
                np.float32(cosine),
                np.float32(sine),
                np.float32(semi_major),
                np.float32(semi_minor),
                np.int32(harmonic),
                np.float32(phase_a),
                np.float32(phase_b),
                np.float32(edge_power),
                np.float32(boundary_roughness),
                np.float32(sign * amplitude),
            )
        )
        sampled_diameters.append(diameter)
        sampled_amplitudes.append(amplitude)

    height_gpu = cp.asarray(height_m)
    kernel = _cuda_irregular_feature_kernel()
    threads = 256
    for arguments in feature_arguments:
        feature_size = int(
            (arguments[2] - arguments[1]) * (arguments[4] - arguments[3])
        )
        blocks = max(1, math.ceil(feature_size / threads))
        kernel(
            (blocks,),
            (threads,),
            (height_gpu, *arguments),
        )
    height_m[:] = cp.asnumpy(height_gpu)
    return {
        "realized_count": count,
        "realized_positive_count": positive_count,
        "sampled_diameter_median_m": (
            float(np.median(sampled_diameters)) if sampled_diameters else None
        ),
        "sampled_amplitude_median_m": (
            float(np.median(sampled_amplitudes)) if sampled_amplitudes else None
        ),
    }


def _source_candidates(relative_path: str) -> list[Path]:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return [candidate]
    module_repo = Path(__file__).resolve().parents[3]
    return [
        Path.cwd() / candidate,
        module_repo / candidate,
    ]


def resolve_profile_source(
    profile: Mapping[str, Any], explicit_path: str | Path | None
) -> tuple[Path | None, Mapping[str, Any] | None]:
    sources = list(profile.get("source_data", ()))
    if explicit_path is not None:
        path = Path(explicit_path).resolve()
        # An explicit user path is new evidence, not an alias for the profile's
        # first calibrated source.  Do not apply another file's hash/label.
        return path, None
    for record in sources:
        relative_path = record.get("relative_path")
        if not relative_path:
            continue
        for candidate in _source_candidates(str(relative_path)):
            if candidate.is_file():
                return candidate.resolve(), record
    return None, None


@lru_cache(maxsize=8)
def _load_profile_surface_cached(
    path_text: str,
    file_size: int,
    modified_time_ns: int,
    source_record_json: str,
) -> MeasuredSurface:
    del file_size, modified_time_ns
    path = Path(path_text)
    source_record = (
        json.loads(source_record_json) if source_record_json else None
    )
    if source_record is not None:
        expected_hash = source_record.get("sha256")
        if expected_hash and sha256_file(path) != expected_hash:
            raise TerrainConfigurationError(
                f"measured source hash differs from profile for {path.name}"
            )
    provenance = (
        {
            key: source_record[key]
            for key in ("source_id", "doi", "license")
            if key in source_record
        }
        if source_record
        else {"source_role": "user_supplied_recalibration_input"}
    )
    return load_measured_surface(
        path,
        dataset_zero_is_invalid=bool(
            source_record
            and source_record.get(
                "dataset_zero_is_invalid",
                "Hirox" in str(source_record.get("source_id", "")),
            )
        ),
        invalid_margin_samples=(
            int(source_record.get("invalid_margin_samples", 0))
            if source_record
            else 0
        ),
        maximum_missing_fraction=0.05,
        provenance=provenance,
    )


def _load_profile_surface(
    path: Path, source_record: Mapping[str, Any] | None
) -> MeasuredSurface:
    """Load and retain verified sources for efficient multi-terrain batches."""

    stat = path.stat()
    source_record_json = (
        json.dumps(source_record, sort_keys=True, ensure_ascii=False)
        if source_record is not None
        else ""
    )
    return _load_profile_surface_cached(
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        source_record_json,
    )


def _full_resampled_source(
    surface: MeasuredSurface, resolution_m: float
) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, Any]]:
    size_x = math.floor(surface.size_x_m / resolution_m) * resolution_m
    size_y = math.floor(surface.size_y_m / resolution_m) * resolution_m
    return resample_measured_patch(
        surface.height_m,
        surface.valid_mask,
        source_dx_m=surface.dx_m,
        source_dy_m=surface.dy_m,
        target_size_x_m=size_x,
        target_size_y_m=size_y,
        target_dx_m=resolution_m,
        target_dy_m=resolution_m,
        allow_upsampling=True,
    )


def _valid_patch_origins(
    valid_mask: NDArray[np.bool_], patch_shape: tuple[int, int]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    py, px = patch_shape
    invalid = (~valid_mask).astype(np.int32)
    integral = np.pad(
        np.cumsum(np.cumsum(invalid, axis=0), axis=1),
        ((1, 0), (1, 0)),
    )
    counts = (
        integral[py:, px:]
        - integral[:-py, px:]
        - integral[py:, :-px]
        + integral[:-py, :-px]
    )
    origins = np.nonzero(counts == 0)
    if origins[0].size == 0:
        # A tiny controlled tolerance is allowed, but patch data are repaired
        # only after retaining the source mask/provenance.
        origins = np.nonzero(counts <= max(1, int(0.0025 * py * px)))
    return (
        np.asarray(origins[0], dtype=np.intp),
        np.asarray(origins[1], dtype=np.intp),
    )


def _tile_starts(length: int, patch: int, overlap: int) -> list[int]:
    if patch >= length:
        return [0]
    step = patch - overlap
    starts = list(range(0, length - patch + 1, step))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def quilt_measured_patches(
    source_height_m: NDArray[np.float32],
    source_valid_mask: NDArray[np.bool_],
    *,
    output_shape: tuple[int, int],
    patch_size_samples: int,
    overlap_fraction: float,
    candidate_count: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Texture-quilt measured patches with overlap-error selection and blending."""

    ny, nx = output_shape
    patch_y = min(patch_size_samples, source_height_m.shape[0], ny)
    patch_x = min(patch_size_samples, source_height_m.shape[1], nx)
    if min(patch_y, patch_x) < 4:
        raise TerrainConfigurationError("measured patch is too small for synthesis")
    overlap_y = min(patch_y - 1, max(2, int(round(patch_y * overlap_fraction))))
    overlap_x = min(patch_x - 1, max(2, int(round(patch_x * overlap_fraction))))
    origin_y, origin_x = _valid_patch_origins(
        source_valid_mask, (patch_y, patch_x)
    )
    if origin_y.size == 0:
        raise TerrainConfigurationError(
            "measured source contains no sufficiently valid synthesis patch"
        )

    output = np.zeros(output_shape, dtype=np.float32)
    placed = np.zeros(output_shape, dtype=np.bool_)
    starts_y = _tile_starts(ny, patch_y, overlap_y)
    starts_x = _tile_starts(nx, patch_x, overlap_x)
    selected_origins: list[tuple[int, int]] = []
    seam_errors: list[float] = []
    score_stride = max(1, min(patch_x, patch_y) // 48)
    source_y_offsets = np.arange(patch_y, dtype=np.intp)[None, :, None]
    source_x_offsets = np.arange(patch_x, dtype=np.intp)[None, None, :]
    ramp_y = np.sin(
        np.linspace(0.0, math.pi / 2.0, overlap_y, dtype=np.float32)
    ) ** 2
    ramp_x = np.sin(
        np.linspace(0.0, math.pi / 2.0, overlap_x, dtype=np.float32)
    ) ** 2
    blend_cache: dict[tuple[bool, bool], NDArray[np.float32]] = {}
    for has_top_overlap in (False, True):
        for has_left_overlap in (False, True):
            blend = np.ones((patch_y, patch_x), dtype=np.float32)
            if has_top_overlap:
                blend[:overlap_y, :] = np.minimum(
                    blend[:overlap_y, :], ramp_y[:, None]
                )
            if has_left_overlap:
                blend[:, :overlap_x] = np.minimum(
                    blend[:, :overlap_x], ramp_x[None, :]
                )
            blend_cache[(has_top_overlap, has_left_overlap)] = blend
    for destination_y in starts_y:
        for destination_x in starts_x:
            draw_count = max(1, min(candidate_count, origin_y.size))
            indices = rng.integers(0, origin_y.size, size=draw_count)
            candidate_y = origin_y[indices]
            candidate_x = origin_x[indices]
            candidates = source_height_m[
                candidate_y[:, None, None] + source_y_offsets,
                candidate_x[:, None, None] + source_x_offsets,
            ]
            existing = output[
                destination_y : destination_y + patch_y,
                destination_x : destination_x + patch_x,
            ]
            existing_mask = placed[
                destination_y : destination_y + patch_y,
                destination_x : destination_x + patch_x,
            ]
            comparison = existing_mask[::score_stride, ::score_stride]
            if np.any(comparison):
                candidate_samples = candidates[
                    :, ::score_stride, ::score_stride
                ][:, comparison]
                difference = (
                    candidate_samples
                    - existing[::score_stride, ::score_stride][comparison]
                )
                scores = np.mean(difference * difference, axis=1)
            else:
                scores = np.zeros(draw_count, dtype=np.float32)
            recent_origins = selected_origins[-16:]
            for candidate_index, (source_y, source_x) in enumerate(
                zip(candidate_y, candidate_x)
            ):
                if (int(source_y), int(source_x)) in recent_origins:
                    scores[candidate_index] += (
                        0.05 * float(np.var(candidates[candidate_index]))
                    )
            best_index = int(np.argmin(scores))
            best_score = float(scores[best_index])
            best_patch = candidates[best_index]
            best_origin = (
                int(candidate_y[best_index]),
                int(candidate_x[best_index]),
            )
            blend = blend_cache[
                (destination_y > 0, destination_x > 0)
            ]
            destination = output[
                destination_y : destination_y + patch_y,
                destination_x : destination_x + patch_x,
            ]
            destination[:] = (1.0 - blend) * destination + blend * best_patch
            placed[
                destination_y : destination_y + patch_y,
                destination_x : destination_x + patch_x,
            ] = True
            selected_origins.append(best_origin)
            seam_errors.append(best_score)
    return output, {
        "method": "overlap_error_patch_quilting_raised_cosine_seams",
        "patch_shape_yx": [patch_y, patch_x],
        "overlap_yx": [overlap_y, overlap_x],
        "candidate_count": candidate_count,
        "placed_patch_count": len(selected_origins),
        "unique_recent_origin_count": len(set(selected_origins)),
        "mean_overlap_mse_m2": float(np.mean(seam_errors)),
        "rotation_policy": "none_preserve_measured_direction",
        "periodic_tiling": False,
    }


def quantile_match(
    height_m: NDArray[np.float32],
    target_quantiles: Mapping[str, Any],
    *,
    backend: MaterialBackend = "cpu",
) -> NDArray[np.float32]:
    percentiles = np.asarray(sorted(float(key) for key in target_quantiles))
    target = np.asarray(
        [float(target_quantiles[str(int(item))]) for item in percentiles],
        dtype=np.float64,
    )
    if backend == "cuda":
        try:
            import cupy as cp  # type: ignore
        except ImportError as exc:
            raise TerrainConfigurationError(
                "CUDA material generation requested but CuPy is not installed"
            ) from exc
        height_gpu = cp.asarray(height_m, dtype=cp.float32)
        percentiles_gpu = cp.asarray(percentiles, dtype=cp.float64)
        current = cp.asnumpy(cp.percentile(height_gpu, percentiles_gpu))
    elif backend == "cpu":
        height_gpu = None
        current = np.percentile(height_m, percentiles)
    else:
        raise TerrainConfigurationError(
            f"material backend must be 'cpu' or 'cuda', got {backend!r}"
        )
    current, unique_indices = np.unique(current, return_index=True)
    target = target[unique_indices]
    if current.size < 2:
        return np.zeros_like(height_m, dtype=np.float32)
    if backend == "cuda":
        assert height_gpu is not None
        mapped_gpu = cp.interp(
            height_gpu,
            cp.asarray(current, dtype=cp.float32),
            cp.asarray(target, dtype=cp.float32),
            left=np.float32(target[0]),
            right=np.float32(target[-1]),
        )
        mapped_gpu -= cp.mean(mapped_gpu, dtype=cp.float64)
        return np.asarray(cp.asnumpy(mapped_gpu), dtype=np.float32)
    mapped = np.interp(height_m, current, target, left=target[0], right=target[-1])
    mapped -= float(np.mean(mapped))
    return np.asarray(mapped, dtype=np.float32)


def _granular_fallback(
    profile: Mapping[str, Any],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    rng: np.random.Generator,
    backend: MaterialBackend = "cpu",
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    generation = profile["generation"]
    grain = float(generation["grain_diameter_m"])
    target_rms = float(generation["rms_height_m"])
    field_a = correlated_field(
        shape,
        dx_m=resolution_m,
        dy_m=resolution_m,
        correlation_x_m=0.42 * grain,
        correlation_y_m=0.36 * grain,
        rng=rng,
        target_rms_m=target_rms,
        backend=backend,
    )
    field_b = correlated_field(
        shape,
        dx_m=resolution_m,
        dy_m=resolution_m,
        correlation_x_m=0.24 * grain,
        correlation_y_m=0.28 * grain,
        rng=rng,
        target_rms_m=target_rms,
        backend=backend,
    )
    # Max-composition creates angular protrusions and deep inter-grain valleys;
    # it is structurally different from a single Gaussian/fractal surface.
    normalized_a = field_a / max(target_rms, np.finfo(np.float32).eps)
    normalized_b = field_b / max(target_rms, np.finfo(np.float32).eps)
    granular = np.maximum(normalized_a, 0.72 * normalized_b)
    granular -= 0.28 * np.abs(normalized_a - normalized_b) ** 1.25
    height = _normalize_rms(granular, target_rms, backend=backend)
    prominent_density = 0.06 / float(generation["grain_spacing_m"]) ** 2
    features = add_irregular_features(
        height,
        dx_m=resolution_m,
        dy_m=resolution_m,
        density_per_m2=prominent_density,
        diameter_median_m=grain,
        diameter_log_sigma=float(generation["grain_height_log_sigma"]),
        aspect_ratio_median=1.35,
        aspect_ratio_log_sigma=float(generation["grain_aspect_ratio_log_sigma"]),
        amplitude_median_m=0.65 * target_rms,
        amplitude_log_sigma=float(generation["grain_height_log_sigma"]),
        edge_power=float(generation["grain_edge_power"]),
        boundary_roughness=0.19,
        rng=rng,
        sign_mode="positive",
        backend=backend,
    )
    height = quantile_match(
        height,
        generation["height_quantiles_m"],
        backend=backend,
    )
    return height, {
        "method": "provisional_multifield_grains_with_irregular_prominent_particles",
        "prominent_grains": features,
        "measured_patch_used": False,
    }


def generate_sandpaper(
    profile: Mapping[str, Any],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    rng: np.random.Generator,
    source_path: str | Path | None = None,
    backend: MaterialBackend = "cpu",
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Generate sandpaper from measured patches or an explicit provisional model."""

    resolved_source, source_record = resolve_profile_source(profile, source_path)
    generation = profile["generation"]
    if resolved_source is None:
        return _granular_fallback(
            profile,
            shape=shape,
            resolution_m=resolution_m,
            rng=rng,
            backend=backend,
        )
    surface = _load_profile_surface(resolved_source, source_record)
    source_height, source_mask, resampling = _full_resampled_source(
        surface, resolution_m
    )
    patch_samples = max(
        8, int(round(float(generation["patch_size_m"]) / resolution_m)) + 1
    )
    height, quilting = quilt_measured_patches(
        source_height,
        source_mask,
        output_shape=shape,
        patch_size_samples=patch_samples,
        overlap_fraction=float(generation["patch_overlap_fraction"]),
        candidate_count=int(generation["patch_candidates"]),
        rng=rng,
    )
    residual_fraction = float(generation["bounded_psd_residual_fraction"])
    residual = correlated_field(
        shape,
        dx_m=resolution_m,
        dy_m=resolution_m,
        correlation_x_m=max(
            resolution_m, 0.35 * float(generation["grain_diameter_m"])
        ),
        correlation_y_m=max(
            resolution_m, 0.30 * float(generation["grain_diameter_m"])
        ),
        rng=rng,
        target_rms_m=residual_fraction * float(generation["rms_height_m"]),
        backend=backend,
    )
    height = np.asarray(height + residual, dtype=np.float32)
    del residual
    height = quantile_match(
        height,
        generation["height_quantiles_m"],
        backend=backend,
    )
    return height, {
        "method": "measured_patch_quilting_plus_bounded_psd_residual",
        "measured_patch_used": True,
        "source": surface.metadata["source"],
        "source_preprocessing": surface.metadata["preprocessing"],
        "source_resampling": resampling,
        "quilting": quilting,
        "bounded_psd_residual_fraction": residual_fraction,
        "quantile_matching": "profile_1_5_25_50_75_95_99_percentiles",
    }


def generate_red_brick(
    profile: Mapping[str, Any],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    rng: np.random.Generator,
    backend: MaterialBackend = "cpu",
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Generate fired brick as base + directional + pores + fired grains."""

    generation = profile["generation"]
    base_config = generation["base_field"]
    base = correlated_field(
        shape,
        dx_m=resolution_m,
        dy_m=resolution_m,
        correlation_x_m=float(base_config["correlation_x_m"]),
        correlation_y_m=float(base_config["correlation_y_m"]),
        rng=rng,
        target_rms_m=float(base_config["rms_height_m"]),
        backend=backend,
    )
    secondary = correlated_field(
        shape,
        dx_m=resolution_m,
        dy_m=resolution_m,
        correlation_x_m=0.32 * float(base_config["correlation_x_m"]),
        correlation_y_m=0.38 * float(base_config["correlation_y_m"]),
        rng=rng,
        target_rms_m=(
            float(base_config["secondary_scale_fraction"])
            * float(base_config["rms_height_m"])
        ),
        backend=backend,
    )
    height = np.asarray(base + secondary, dtype=np.float32)
    del base, secondary

    directional_config = generation["directional_texture"]
    if directional_config["enabled"]:
        directional = correlated_field(
            shape,
            dx_m=resolution_m,
            dy_m=resolution_m,
            correlation_x_m=float(directional_config["correlation_x_m"]),
            correlation_y_m=float(directional_config["correlation_y_m"]),
            rng=rng,
            target_rms_m=float(directional_config["rms_height_m"]),
            backend=backend,
        )
        height += directional
        del directional

    pore_config = generation["pores"]
    pore_record = add_irregular_features(
        height,
        dx_m=resolution_m,
        dy_m=resolution_m,
        density_per_m2=float(pore_config["density_per_m2"]),
        diameter_median_m=float(pore_config["diameter_median_m"]),
        diameter_log_sigma=float(pore_config["diameter_log_sigma"]),
        aspect_ratio_median=float(pore_config["aspect_ratio_median"]),
        aspect_ratio_log_sigma=float(pore_config["aspect_ratio_log_sigma"]),
        amplitude_median_m=float(pore_config["depth_median_m"]),
        amplitude_log_sigma=float(pore_config["depth_log_sigma"]),
        edge_power=float(pore_config["edge_power"]),
        boundary_roughness=float(pore_config["boundary_roughness"]),
        rng=rng,
        sign_mode="negative",
        cluster_probability=float(pore_config["cluster_probability"]),
        backend=backend,
    )
    fine_config = generation["fine_roughness"]
    fine = correlated_field(
        shape,
        dx_m=resolution_m,
        dy_m=resolution_m,
        correlation_x_m=float(fine_config["correlation_x_m"]),
        correlation_y_m=float(fine_config["correlation_y_m"]),
        rng=rng,
        target_rms_m=float(fine_config["rms_height_m"]),
        backend=backend,
    )
    fine = _non_gaussian(
        fine,
        exponent=float(fine_config["non_gaussian_exponent"]),
        skew_bias=float(fine_config["skew_bias"]),
        backend=backend,
    )
    height += fine
    del fine
    height -= np.float32(np.mean(height, dtype=np.float64))
    return height, {
        "method": "brick_base_directional_irregular_pores_fired_grain",
        "layers": {
            "base": base_config,
            "directional": directional_config,
            "pores": pore_record,
            "fine_roughness": fine_config,
        },
        "periodic_boundary": False,
    }


def generate_concrete(
    profile: Mapping[str, Any],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    rng: np.random.Generator,
    backend: MaterialBackend = "cpu",
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Generate concrete as mortar + aggregate + void + finish + fine layers."""

    generation = profile["generation"]
    mortar_config = generation["mortar"]
    mortar_primary = correlated_field(
        shape,
        dx_m=resolution_m,
        dy_m=resolution_m,
        correlation_x_m=float(mortar_config["correlation_x_m"]),
        correlation_y_m=float(mortar_config["correlation_y_m"]),
        rng=rng,
        target_rms_m=float(mortar_config["rms_height_m"]),
        backend=backend,
    )
    mortar_secondary = correlated_field(
        shape,
        dx_m=resolution_m,
        dy_m=resolution_m,
        correlation_x_m=0.22 * float(mortar_config["correlation_x_m"]),
        correlation_y_m=0.24 * float(mortar_config["correlation_y_m"]),
        rng=rng,
        target_rms_m=(
            float(mortar_config["secondary_scale_fraction"])
            * float(mortar_config["rms_height_m"])
        ),
        backend=backend,
    )
    height = _non_gaussian(
        np.asarray(mortar_primary + mortar_secondary, dtype=np.float32),
        exponent=float(mortar_config["non_gaussian_exponent"]),
        skew_bias=float(mortar_config["skew_bias"]),
        backend=backend,
    )
    del mortar_primary, mortar_secondary

    aggregate_config = generation["aggregate"]
    aggregate_record = add_irregular_features(
        height,
        dx_m=resolution_m,
        dy_m=resolution_m,
        density_per_m2=float(aggregate_config["density_per_m2"]),
        diameter_median_m=float(aggregate_config["diameter_median_m"]),
        diameter_log_sigma=float(aggregate_config["diameter_log_sigma"]),
        aspect_ratio_median=float(aggregate_config["aspect_ratio_median"]),
        aspect_ratio_log_sigma=float(aggregate_config["aspect_ratio_log_sigma"]),
        amplitude_median_m=float(aggregate_config["height_median_m"]),
        amplitude_log_sigma=float(aggregate_config["height_log_sigma"]),
        edge_power=float(aggregate_config["edge_power"]),
        boundary_roughness=float(aggregate_config["boundary_roughness"]),
        rng=rng,
        sign_mode="mixed",
        positive_probability=float(aggregate_config["exposed_probability"]),
        cluster_probability=0.12,
        backend=backend,
    )
    void_config = generation["voids"]
    void_record = add_irregular_features(
        height,
        dx_m=resolution_m,
        dy_m=resolution_m,
        density_per_m2=float(void_config["density_per_m2"]),
        diameter_median_m=float(void_config["diameter_median_m"]),
        diameter_log_sigma=float(void_config["diameter_log_sigma"]),
        aspect_ratio_median=float(void_config["aspect_ratio_median"]),
        aspect_ratio_log_sigma=float(void_config["aspect_ratio_log_sigma"]),
        amplitude_median_m=float(void_config["depth_median_m"]),
        amplitude_log_sigma=float(void_config["depth_log_sigma"]),
        edge_power=float(void_config["edge_power"]),
        boundary_roughness=float(void_config["boundary_roughness"]),
        rng=rng,
        sign_mode="negative",
        cluster_probability=0.22,
        backend=backend,
    )
    finish_config = generation["finish"]
    if finish_config["enabled"] and float(finish_config["rms_height_m"]) > 0:
        finish = correlated_field(
            shape,
            dx_m=resolution_m,
            dy_m=resolution_m,
            correlation_x_m=float(finish_config["correlation_x_m"]),
            correlation_y_m=float(finish_config["correlation_y_m"]),
            rng=rng,
            target_rms_m=float(finish_config["rms_height_m"]),
            backend=backend,
        )
        height += finish
    fine_config = generation["fine_roughness"]
    fine = correlated_field(
        shape,
        dx_m=resolution_m,
        dy_m=resolution_m,
        correlation_x_m=float(fine_config["correlation_x_m"]),
        correlation_y_m=float(fine_config["correlation_y_m"]),
        rng=rng,
        target_rms_m=float(fine_config["rms_height_m"]),
        backend=backend,
    )
    height += _non_gaussian(
        fine,
        exponent=float(fine_config["non_gaussian_exponent"]),
        backend=backend,
    )
    del fine
    height -= np.float32(np.mean(height, dtype=np.float64))
    return height, {
        "method": "concrete_mortar_aggregate_air_void_finish",
        "layers": {
            "mortar": mortar_config,
            "aggregate": aggregate_record,
            "air_voids": void_record,
            "finish": finish_config,
            "fine_roughness": fine_config,
        },
        "periodic_boundary": False,
        "cement_paste_relabelled_as_concrete": False,
    }
