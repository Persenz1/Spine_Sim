"""材料表面形貌的统计比较和可视化 QA。

本模块输出描述性证据与相对误差，不内置统一材料验收阈值；单个实测样本的拟合也不
等价于群体级材料验证。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .api import Terrain
from .descriptors import compute_descriptors
from .errors import TerrainConfigurationError


def _relative_error(value: float, reference: float) -> float:
    """计算相对参考值的有符号误差，并稳定处理零参考值。"""

    if not math.isfinite(value) or not math.isfinite(reference):
        return float("nan")
    return (value - reference) / max(abs(reference), np.finfo(float).eps)


def compare_topographies(
    reference_height_m: NDArray[np.floating],
    synthetic_height_m: NDArray[np.floating],
    *,
    reference_dx_m: float,
    reference_dy_m: float,
    synthetic_dx_m: float,
    synthetic_dy_m: float,
    reference_valid_mask: NDArray[np.bool_] | None = None,
    synthetic_valid_mask: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """比较实测与合成地形的高度、空间、坡度、峰和坑指标。"""

    reference = compute_descriptors(
        reference_height_m,
        dx_m=reference_dx_m,
        dy_m=reference_dy_m,
        valid_mask=reference_valid_mask,
        include_curves=True,
    )
    synthetic = compute_descriptors(
        synthetic_height_m,
        dx_m=synthetic_dx_m,
        dy_m=synthetic_dy_m,
        valid_mask=synthetic_valid_mask,
        include_curves=True,
    )
    # 明确列出跨样本可直接比较的标量；完整描述量和曲线仍分别保留在报告中。
    pairs = {
        "rms_height": (
            synthetic["height"]["rms_about_mean_m"],
            reference["height"]["rms_about_mean_m"],
        ),
        "skewness": (
            synthetic["height"]["skewness"],
            reference["height"]["skewness"],
        ),
        "kurtosis": (
            synthetic["height"]["kurtosis"],
            reference["height"]["kurtosis"],
        ),
        "correlation_length_x": (
            synthetic["spatial"]["correlation_length_x_m"],
            reference["spatial"]["correlation_length_x_m"],
        ),
        "correlation_length_y": (
            synthetic["spatial"]["correlation_length_y_m"],
            reference["spatial"]["correlation_length_y_m"],
        ),
        "slope_x_rms": (
            synthetic["slope"]["x_rms"],
            reference["slope"]["x_rms"],
        ),
        "slope_y_rms": (
            synthetic["slope"]["y_rms"],
            reference["slope"]["y_rms"],
        ),
        "peak_density": (
            synthetic["features"]["peak_density_per_m2"],
            reference["features"]["peak_density_per_m2"],
        ),
        "pit_density": (
            synthetic["features"]["pit_density_per_m2"],
            reference["features"]["pit_density_per_m2"],
        ),
    }
    return {
        "schema_version": "topography-comparison-v1",
        "reference": reference,
        "synthetic": synthetic,
        "relative_errors": {
            name: _relative_error(value, reference_value)
            for name, (value, reference_value) in pairs.items()
        },
        "validation_scope": (
            "single_reference_morphology_fit_not_population_validation"
        ),
    }


def _sample(values: NDArray[np.floating], maximum: int = 200_000) -> NDArray[np.float64]:
    """按固定步长下采样一维视图，限制绘图开销且保持确定性。"""

    flat = np.asarray(values, dtype=np.float64).ravel()
    return flat[:: max(1, math.ceil(flat.size / maximum))]


def render_comparison(
    synthetic: Terrain,
    output_path: str | Path,
    *,
    reference_height_m: NDArray[np.floating] | None = None,
    reference_dx_m: float | None = None,
    reference_dy_m: float | None = None,
    reference_valid_mask: NDArray[np.bool_] | None = None,
    reference_label: str = "measured reference",
    dpi: int = 170,
) -> Path:
    """绘制高度图、截面、CDF、PSD 和坡度分布对比。"""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise TerrainConfigurationError(
            "validation plots require the optional plot dependency"
        ) from exc

    # 没有实测参考时仍生成明确标注的 synthetic-only 图，避免视觉上暗示已验证。
    reference_available = reference_height_m is not None
    if reference_available and (reference_dx_m is None or reference_dy_m is None):
        raise TerrainConfigurationError(
            "reference spacing is required with reference_height_m"
        )
    synthetic_descriptors = compute_descriptors(
        synthetic.height,
        dx_m=synthetic.dx,
        dy_m=synthetic.dy,
        valid_mask=synthetic.valid_mask,
        include_curves=True,
    )
    reference_descriptors = (
        compute_descriptors(
            np.asarray(reference_height_m),
            dx_m=float(reference_dx_m),
            dy_m=float(reference_dy_m),
            valid_mask=reference_valid_mask,
            include_curves=True,
        )
        if reference_available
        else None
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    values = synthetic.height[synthetic.valid_mask]
    lower, upper = np.percentile(values, [1, 99])
    extent = (
        0,
        synthetic.size_x_m * 1e3,
        synthetic.size_y_m * 1e3,
        0,
    )
    synthetic_image = axes[0, 1].imshow(
        synthetic.height * 1e6,
        extent=extent,
        aspect="auto",
        cmap="viridis",
        vmin=lower * 1e6,
        vmax=upper * 1e6,
        interpolation="nearest",
    )
    axes[0, 1].set_title(
        f"synthetic {synthetic.material}/{synthetic.subtype}, seed={synthetic.seed}"
    )
    axes[0, 1].set_xlabel("x (mm)")
    axes[0, 1].set_ylabel("y (mm)")
    figure.colorbar(synthetic_image, ax=axes[0, 1], label="height (µm)")

    if reference_available:
        reference = np.asarray(reference_height_m, dtype=np.float64)
        reference_mask = (
            np.ones(reference.shape, dtype=np.bool_)
            if reference_valid_mask is None
            else np.asarray(reference_valid_mask, dtype=np.bool_)
        )
        reference_values = reference[reference_mask]
        ref_lower, ref_upper = np.percentile(reference_values, [1, 99])
        ref_extent = (
            0,
            (reference.shape[1] - 1) * float(reference_dx_m) * 1e3,
            (reference.shape[0] - 1) * float(reference_dy_m) * 1e3,
            0,
        )
        reference_image = axes[0, 0].imshow(
            reference * 1e6,
            extent=ref_extent,
            aspect="auto",
            cmap="viridis",
            vmin=ref_lower * 1e6,
            vmax=ref_upper * 1e6,
            interpolation="nearest",
        )
        axes[0, 0].set_title(reference_label)
        axes[0, 0].set_xlabel("x (mm)")
        axes[0, 0].set_ylabel("y (mm)")
        figure.colorbar(reference_image, ax=axes[0, 0], label="height (µm)")
    else:
        axes[0, 0].axis("off")
        axes[0, 0].text(
            0.5,
            0.55,
            "REAL HEIGHT MAP\nNOT AVAILABLE",
            ha="center",
            va="center",
            fontsize=16,
            weight="bold",
            color="#8b1a1a",
            transform=axes[0, 0].transAxes,
        )
        axes[0, 0].text(
            0.5,
            0.35,
            "Synthetic-only provisional validation;\nno measured surface is implied.",
            ha="center",
            va="center",
            fontsize=10,
            transform=axes[0, 0].transAxes,
        )

    synthetic_x = np.arange(synthetic.height.shape[1]) * synthetic.dx * 1e3
    axes[0, 2].plot(
        synthetic_x,
        synthetic.height[synthetic.height.shape[0] // 2] * 1e6,
        label="synthetic",
        linewidth=1.0,
    )
    if reference_available:
        reference = np.asarray(reference_height_m)
        reference_x = (
            np.arange(reference.shape[1]) * float(reference_dx_m) * 1e3
        )
        axes[0, 2].plot(
            reference_x,
            reference[reference.shape[0] // 2] * 1e6,
            label="measured",
            linewidth=0.8,
            alpha=0.8,
        )
    axes[0, 2].set_title("center cross-section")
    axes[0, 2].set_xlabel("x (mm)")
    axes[0, 2].set_ylabel("height (µm)")
    axes[0, 2].legend()

    synthetic_sample = np.sort(_sample(values))
    axes[1, 0].plot(
        synthetic_sample * 1e6,
        np.linspace(0, 1, synthetic_sample.size),
        label="synthetic",
    )
    if reference_available:
        reference_sample = np.sort(_sample(reference_values))
        axes[1, 0].plot(
            reference_sample * 1e6,
            np.linspace(0, 1, reference_sample.size),
            label="measured",
        )
    axes[1, 0].set_title("height CDF")
    axes[1, 0].set_xlabel("height (µm)")
    axes[1, 0].set_ylabel("cumulative probability")
    axes[1, 0].legend()

    synthetic_curves = synthetic_descriptors["curves"]
    axes[1, 1].loglog(
        synthetic_curves["frequency_x_per_m"][1:],
        synthetic_curves["psd_x_m3"][1:],
        label="synthetic x",
    )
    axes[1, 1].loglog(
        synthetic_curves["frequency_y_per_m"][1:],
        synthetic_curves["psd_y_m3"][1:],
        label="synthetic y",
        alpha=0.8,
    )
    if reference_descriptors is not None:
        reference_curves = reference_descriptors["curves"]
        axes[1, 1].loglog(
            reference_curves["frequency_x_per_m"][1:],
            reference_curves["psd_x_m3"][1:],
            label="measured x",
            linewidth=0.9,
        )
        axes[1, 1].loglog(
            reference_curves["frequency_y_per_m"][1:],
            reference_curves["psd_y_m3"][1:],
            label="measured y",
            linewidth=0.9,
            alpha=0.8,
        )
    axes[1, 1].set_title("directional PSD")
    axes[1, 1].set_xlabel("spatial frequency (1/m)")
    axes[1, 1].set_ylabel("PSD (m³)")
    axes[1, 1].legend(fontsize=8)

    synthetic_slope_x = np.diff(synthetic.height, axis=1).ravel() / synthetic.dx
    synthetic_slope_y = np.diff(synthetic.height, axis=0).ravel() / synthetic.dy
    slope_limit = max(
        np.percentile(np.abs(synthetic_slope_x), 99),
        np.percentile(np.abs(synthetic_slope_y), 99),
    )
    bins = np.linspace(-slope_limit, slope_limit, 90)
    axes[1, 2].hist(
        _sample(synthetic_slope_x),
        bins=bins,
        density=True,
        histtype="step",
        label="synthetic x",
    )
    axes[1, 2].hist(
        _sample(synthetic_slope_y),
        bins=bins,
        density=True,
        histtype="step",
        label="synthetic y",
    )
    if reference_available:
        reference_slope_x = np.diff(reference, axis=1).ravel() / float(
            reference_dx_m
        )
        reference_slope_y = np.diff(reference, axis=0).ravel() / float(
            reference_dy_m
        )
        axes[1, 2].hist(
            _sample(reference_slope_x),
            bins=bins,
            density=True,
            histtype="step",
            label="measured x",
            alpha=0.8,
        )
        axes[1, 2].hist(
            _sample(reference_slope_y),
            bins=bins,
            density=True,
            histtype="step",
            label="measured y",
            alpha=0.8,
        )
    axes[1, 2].set_title("slope distributions")
    axes[1, 2].set_xlabel("dimensionless slope")
    axes[1, 2].set_ylabel("density")
    axes[1, 2].legend(fontsize=8)

    figure.suptitle(
        f"M1 geometry validation — status: {synthetic.metadata['profile_status']}",
        fontsize=14,
    )
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi)
    plt.close(figure)
    return target


def summarize_seed_ensemble(terrains: Sequence[Terrain]) -> dict[str, Any]:
    """汇总多个 seed，并确认它们属于同一材料/subtype 参数族。"""

    if not terrains:
        raise TerrainConfigurationError("seed ensemble cannot be empty")
    identity = (terrains[0].material, terrains[0].subtype)
    if any((item.material, item.subtype) != identity for item in terrains):
        raise TerrainConfigurationError(
            "seed ensemble cannot mix material/subtype profiles"
        )
    # 保留逐 seed 描述量，同时计算跨 seed 均值和离散度，避免只展示“最好看”的样本。
    descriptors = [
        compute_descriptors(
            item.height,
            dx_m=item.dx,
            dy_m=item.dy,
            valid_mask=item.valid_mask,
            include_curves=True,
        )
        for item in terrains
    ]
    metrics = {
        "rms_height_m": [
            item["height"]["rms_about_mean_m"] for item in descriptors
        ],
        "skewness": [item["height"]["skewness"] for item in descriptors],
        "kurtosis": [item["height"]["kurtosis"] for item in descriptors],
        "correlation_length_x_m": [
            item["spatial"]["correlation_length_x_m"] for item in descriptors
        ],
        "correlation_length_y_m": [
            item["spatial"]["correlation_length_y_m"] for item in descriptors
        ],
        "slope_x_rms": [item["slope"]["x_rms"] for item in descriptors],
        "slope_y_rms": [item["slope"]["y_rms"] for item in descriptors],
        "peak_density_per_m2": [
            item["features"]["peak_density_per_m2"] for item in descriptors
        ],
        "pit_density_per_m2": [
            item["features"]["pit_density_per_m2"] for item in descriptors
        ],
    }
    return {
        "schema_version": "terrain-seed-ensemble-v1",
        "material": identity[0],
        "subtype": identity[1],
        "profile_status": terrains[0].metadata["profile_status"],
        "profile_hash": terrains[0].metadata["profile_hash"],
        "parameter_basis": terrains[0].metadata["parameter_basis"],
        "generator_version": terrains[0].metadata["generator_version"],
        "grid": terrains[0].metadata["grid"],
        "seeds": [item.seed for item in terrains],
        "metric_mean": {
            name: float(np.nanmean(values)) for name, values in metrics.items()
        },
        "metric_standard_deviation": {
            name: float(np.nanstd(values)) for name, values in metrics.items()
        },
        "per_seed_descriptors": descriptors,
    }


def write_validation_json(path: str | Path, report: Mapping[str, Any]) -> Path:
    """以 UTF-8、排序缩进 JSON 写入验证报告并返回绝对路径。"""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target
