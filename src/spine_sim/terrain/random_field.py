"""坐标可寻址、有限核的 defined-geometry 随机场。

随机值由 seed、生成器版本和全局整数节点直接决定，因此窗口大小、tile 顺序和后端
不会改变同一坐标的 realization；production 网格只是 canonical 网格的偶数节点。
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .errors import TerrainConfigurationError
from .models import RegionSpec, TerrainRecipe


_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_X_MULTIPLIER = np.uint64(0xD2B74407B1CE6E93)
_Y_MULTIPLIER = np.uint64(0xCA5A826395121157)
_SEED_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)
_SQRT3 = math.sqrt(3.0)


def _version_counter(version: str) -> np.uint64:
    """把生成器语义版本压缩为 64 位计数器的一部分。"""

    digest = hashlib.sha256(version.encode("utf-8")).digest()
    return np.uint64(int.from_bytes(digest[:8], "little"))


def _splitmix64(value: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """对地址计数器执行向量化 SplitMix64 混合。"""

    with np.errstate(over="ignore"):
        value = (value + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
        value = ((value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK64
        value = ((value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK64
        return value ^ (value >> np.uint64(31))


def coordinate_noise(
    seed: int,
    x_indices: NDArray[np.int64],
    y_indices: NDArray[np.int64],
    *,
    generator_version: str,
) -> NDArray[np.float64]:
    """按全局网格地址返回确定性的零均值、单位方差均匀噪声。"""

    x_bits = np.asarray(x_indices, dtype=np.int64).view(np.uint64)
    y_bits = np.asarray(y_indices, dtype=np.int64).view(np.uint64)
    # uint64 溢出是哈希算法定义的一部分，不是数值错误。
    with np.errstate(over="ignore"):
        counter = (
            x_bits[None, :] * _X_MULTIPLIER
            ^ y_bits[:, None] * _Y_MULTIPLIER
            ^ np.uint64(seed) * _SEED_MULTIPLIER
            ^ _version_counter(generator_version)
        )
    hashed = _splitmix64(counter)
    unit = (hashed >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    return (2.0 * unit - 1.0) * _SQRT3


def gaussian_kernel(
    correlation_length_m: float,
    spacing_m: float,
    truncate_sigma: float,
) -> NDArray[np.float64]:
    """构造单位和、按相关长度截断的一维高斯核。"""

    radius = max(1, int(math.ceil(truncate_sigma * correlation_length_m / spacing_m)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64) * spacing_m
    weights = np.exp(-0.5 * (offsets / correlation_length_m) ** 2)
    weights /= weights.sum()
    return weights


def _filter_x_valid(
    values: NDArray[np.float64],
    kernel_x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """沿 x 做 valid 卷积；输入窗口已显式包含 halo。"""

    output_width = values.shape[1] - kernel_x.size + 1
    horizontal = np.zeros(
        (values.shape[0], output_width), dtype=np.float64
    )
    for index, weight in enumerate(kernel_x):
        horizontal += values[:, index : index + output_width] * weight
    return horizontal


def _filter_y_valid(
    horizontal: NDArray[np.float64],
    kernel_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """沿 y 做 valid 卷积，完成可分离二维滤波。"""

    output_height = horizontal.shape[0] - kernel_y.size + 1
    filtered = np.zeros(
        (output_height, horizontal.shape[1]), dtype=np.float64
    )
    for index, weight in enumerate(kernel_y):
        filtered += horizontal[index : index + output_height, :] * weight
    return filtered


def _generate_canonical_window_cpu(
    recipe: TerrainRecipe,
    *,
    start_x_index: int,
    start_y_index: int,
    count_x: int,
    count_y: int,
) -> NDArray[np.float64]:
    """在 CPU 上重建带完整核 halo 的 canonical 窗口。"""

    kernel_x = gaussian_kernel(
        recipe.correlation_length_x_m,
        recipe.canonical_dx_m,
        recipe.kernel_truncate_sigma,
    )
    kernel_y = gaussian_kernel(
        recipe.correlation_length_y_m,
        recipe.canonical_dy_m,
        recipe.kernel_truncate_sigma,
    )
    halo_x = kernel_x.size // 2
    halo_y = kernel_y.size // 2
    # 先扩展有限核 halo，再 valid 卷积，保证裁剪窗口与大窗口对应区域逐点一致。
    x_indices = np.arange(
        start_x_index - halo_x,
        start_x_index + count_x + halo_x,
        dtype=np.int64,
    )
    y_indices = np.arange(
        start_y_index - halo_y,
        start_y_index + count_y + halo_y,
        dtype=np.int64,
    )
    noise = coordinate_noise(
        recipe.seed,
        x_indices,
        y_indices,
        generator_version=recipe.generator_version,
    )
    horizontal = _filter_x_valid(noise, kernel_x)
    del noise
    filtered = _filter_y_valid(horizontal, kernel_y)
    del horizontal
    # 白噪声经可分离单位和核后的理论 RMS 为两个核 L2 能量乘积的平方根。
    theoretical_rms = math.sqrt(
        float(np.dot(kernel_x, kernel_x) * np.dot(kernel_y, kernel_y))
    )
    scale = (
        recipe.target_rms_height_m
        * recipe.amplitude_scale
        / theoretical_rms
    )
    return filtered * scale


def _generate_canonical_window_cuda(
    recipe: TerrainRecipe,
    *,
    start_x_index: int,
    start_y_index: int,
    count_x: int,
    count_y: int,
) -> NDArray[np.float64]:
    """在 CUDA 上复现与 CPU 相同的地址哈希、卷积和理论 RMS 标定。"""

    try:
        import cupy as cp  # type: ignore
    except ImportError as exc:
        raise TerrainConfigurationError(
            "CUDA generation requested but CuPy is not installed"
        ) from exc

    kernel_x_np = gaussian_kernel(
        recipe.correlation_length_x_m,
        recipe.canonical_dx_m,
        recipe.kernel_truncate_sigma,
    )
    kernel_y_np = gaussian_kernel(
        recipe.correlation_length_y_m,
        recipe.canonical_dy_m,
        recipe.kernel_truncate_sigma,
    )
    halo_x = kernel_x_np.size // 2
    halo_y = kernel_y_np.size // 2
    x = cp.arange(
        start_x_index - halo_x,
        start_x_index + count_x + halo_x,
        dtype=cp.int64,
    ).view(cp.uint64)
    y = cp.arange(
        start_y_index - halo_y,
        start_y_index + count_y + halo_y,
        dtype=cp.int64,
    ).view(cp.uint64)
    mask = cp.uint64(0xFFFFFFFFFFFFFFFF)
    seed_term = cp.uint64(
        (recipe.seed * int(_SEED_MULTIPLIER)) & 0xFFFFFFFFFFFFFFFF
    )
    counter = (
        x[None, :] * cp.uint64(int(_X_MULTIPLIER))
        ^ y[:, None] * cp.uint64(int(_Y_MULTIPLIER))
        ^ seed_term
        ^ cp.uint64(int(_version_counter(recipe.generator_version)))
    )
    hashed = (counter + cp.uint64(0x9E3779B97F4A7C15)) & mask
    hashed = (
        (hashed ^ (hashed >> cp.uint64(30))) * cp.uint64(0xBF58476D1CE4E5B9)
    ) & mask
    hashed = (
        (hashed ^ (hashed >> cp.uint64(27))) * cp.uint64(0x94D049BB133111EB)
    ) & mask
    hashed ^= hashed >> cp.uint64(31)
    noise = (
        (hashed >> cp.uint64(11)).astype(cp.float64) * (1.0 / (1 << 53))
        * 2.0
        - 1.0
    ) * _SQRT3
    kernel_x = cp.asarray(kernel_x_np)
    kernel_y = cp.asarray(kernel_y_np)
    output_width = noise.shape[1] - kernel_x.size + 1
    horizontal = cp.zeros((noise.shape[0], output_width), dtype=cp.float64)
    for index, weight in enumerate(kernel_x):
        horizontal += noise[:, index : index + output_width] * weight
    del noise
    output_height = horizontal.shape[0] - kernel_y.size + 1
    filtered = cp.zeros(
        (output_height, horizontal.shape[1]), dtype=cp.float64
    )
    for index, weight in enumerate(kernel_y):
        filtered += horizontal[index : index + output_height, :] * weight
    del horizontal
    theoretical_rms = math.sqrt(
        float(np.dot(kernel_x_np, kernel_x_np) * np.dot(kernel_y_np, kernel_y_np))
    )
    scale = (
        recipe.target_rms_height_m
        * recipe.amplitude_scale
        / theoretical_rms
    )
    return cp.asnumpy(filtered * scale)


def generate_canonical_window(
    recipe: TerrainRecipe,
    *,
    start_x_index: int,
    start_y_index: int,
    count_x: int,
    count_y: int,
    backend: Literal["cpu", "cuda"] = "cpu",
) -> NDArray[np.float64]:
    """校验窗口和后端后分派 canonical 随机场生成。"""

    if count_x < 1 or count_y < 1:
        raise TerrainConfigurationError("canonical window dimensions must be positive")
    if backend == "cpu":
        return _generate_canonical_window_cpu(
            recipe,
            start_x_index=start_x_index,
            start_y_index=start_y_index,
            count_x=count_x,
            count_y=count_y,
        )
    if backend == "cuda":
        return _generate_canonical_window_cuda(
            recipe,
            start_x_index=start_x_index,
            start_y_index=start_y_index,
            count_x=count_x,
            count_y=count_y,
        )
    raise TerrainConfigurationError("backend must be cpu or cuda")


def generate_defined_geometry(
    recipe: TerrainRecipe,
    region: RegionSpec,
    *,
    backend: Literal["cpu", "cuda"] = "cpu",
) -> NDArray[np.float32]:
    """从同一 realization 生成 canonical 或 production 区域。"""

    region.validate_against(recipe)
    start_x, start_y = recipe.canonical_indices(
        region.origin_x_m, region.origin_y_m
    )
    ny, nx = region.shape
    if math.isclose(
        region.resolution_x_m,
        recipe.canonical_dx_m,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) and math.isclose(
        region.resolution_y_m,
        recipe.canonical_dy_m,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        field = generate_canonical_window(
            recipe,
            start_x_index=start_x,
            start_y_index=start_y,
            count_x=nx,
            count_y=ny,
            backend=backend,
        )
    elif math.isclose(
        region.resolution_x_m,
        recipe.production_dx_m,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) and math.isclose(
        region.resolution_y_m,
        recipe.production_dy_m,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        # production 节点严格取 canonical realization 的 [::2, ::2]，不重新随机采样。
        canonical = generate_canonical_window(
            recipe,
            start_x_index=start_x,
            start_y_index=start_y,
            count_x=2 * (nx - 1) + 1,
            count_y=2 * (ny - 1) + 1,
            backend=backend,
        )
        field = canonical[::2, ::2]
    else:
        raise TerrainConfigurationError(
            "x and y must both use canonical spacing or both use production spacing"
        )
    return np.asarray(field, dtype=np.float32)
