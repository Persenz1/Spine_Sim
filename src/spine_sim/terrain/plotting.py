"""Lightweight, read-only terrain previews and finite-tip comparison plots."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spine_sim.io.results import atomic_write_json

from .library import TerrainLibrary
from .models import RegionSpec


@dataclass(frozen=True)
class TerrainPatch:
    """A bounded in-memory sample from a potentially large terrain map."""

    x_global_m: NDArray[np.float64]
    y_global_m: NDArray[np.float64]
    height_m: NDArray[np.float64]
    center_x_m: float
    center_y_m: float
    source_shape: tuple[int, int]

    @property
    def shape(self) -> tuple[int, int]:
        return self.height_m.shape

    @property
    def size_x_m(self) -> float:
        return float(self.x_global_m[-1] - self.x_global_m[0])

    @property
    def size_y_m(self) -> float:
        return float(self.y_global_m[-1] - self.y_global_m[0])


@dataclass(frozen=True)
class SpherePlacement:
    """Lowest non-penetrating placement of a sphere at a fixed x-y centre."""

    radius_m: float
    center_xyz_m: tuple[float, float, float]
    support_xyz_m: tuple[float, float, float]
    minimum_clearance_m: float
    evaluated_sample_count: int


@dataclass(frozen=True)
class GrooveSelection:
    """A locally low terrain site with higher surrounding terrain."""

    center_x_m: float
    center_y_m: float
    center_height_m: float
    inner_mean_height_m: float
    surrounding_mean_height_m: float
    depth_score_m: float
    inner_radius_m: float
    outer_radius_m: float


def _bounded_node_indices(
    center: int,
    intervals: int,
    count: int,
    *,
    maximum_points: int | None,
) -> NDArray[np.int64]:
    start = center - intervals // 2
    stop = start + intervals + 1
    if start < 0:
        stop -= start
        start = 0
    if stop > count:
        start -= stop - count
        stop = count
    node_count = stop - start
    if maximum_points is None or node_count <= maximum_points:
        return np.arange(start, stop, dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(start, stop - 1, maximum_points)).astype(np.int64)
    )


def extract_centered_patch(
    height_m: ArrayLike,
    region: RegionSpec,
    *,
    center_x_m: float | None = None,
    center_y_m: float | None = None,
    window_size_x_m: float = 10e-3,
    window_size_y_m: float | None = None,
    maximum_axis_points: int | None = None,
) -> TerrainPatch:
    """Sample a rectangular crop without materializing the full terrain."""

    window_y = window_size_x_m if window_size_y_m is None else window_size_y_m
    for name, value in (
        ("window_size_x_m", window_size_x_m),
        ("window_size_y_m", window_y),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if maximum_axis_points is not None and maximum_axis_points < 2:
        raise ValueError("maximum_axis_points must be at least 2")

    height = np.asanyarray(height_m)
    if height.ndim != 2 or height.shape != region.shape:
        raise ValueError("height_m shape must match region.shape")
    requested_x = (
        region.origin_x_m + 0.5 * region.size_x_m
        if center_x_m is None
        else float(center_x_m)
    )
    requested_y = (
        region.origin_y_m + 0.5 * region.size_y_m
        if center_y_m is None
        else float(center_y_m)
    )
    if not math.isfinite(requested_x) or not math.isfinite(requested_y):
        raise ValueError("patch centre coordinates must be finite")

    center_column = int(
        round((requested_x - region.origin_x_m) / region.resolution_x_m)
    )
    center_row = int(
        round((requested_y - region.origin_y_m) / region.resolution_y_m)
    )
    if not (0 <= center_column < region.shape[1]) or not (
        0 <= center_row < region.shape[0]
    ):
        raise ValueError("patch centre lies outside the stored terrain region")

    intervals_x = min(
        max(2, int(round(window_size_x_m / region.resolution_x_m))),
        region.shape[1] - 1,
    )
    intervals_y = min(
        max(2, int(round(window_y / region.resolution_y_m))),
        region.shape[0] - 1,
    )
    source_shape = (intervals_y + 1, intervals_x + 1)
    row_indices = _bounded_node_indices(
        center_row,
        intervals_y,
        region.shape[0],
        maximum_points=maximum_axis_points,
    )
    column_indices = _bounded_node_indices(
        center_column,
        intervals_x,
        region.shape[1],
        maximum_points=maximum_axis_points,
    )
    x = (
        region.origin_x_m
        + column_indices.astype(np.float64) * region.resolution_x_m
    )
    y = (
        region.origin_y_m
        + row_indices.astype(np.float64) * region.resolution_y_m
    )
    actual_center_x = (
        region.origin_x_m + center_column * region.resolution_x_m
    )
    actual_center_y = region.origin_y_m + center_row * region.resolution_y_m
    return TerrainPatch(
        x_global_m=x,
        y_global_m=y,
        height_m=np.array(
            height[np.ix_(row_indices, column_indices)],
            dtype=np.float64,
            copy=True,
        ),
        center_x_m=float(actual_center_x),
        center_y_m=float(actual_center_y),
        source_shape=source_shape,
    )


def place_sphere_on_patch(
    patch: TerrainPatch,
    *,
    radius_m: float,
    center_x_m: float | None = None,
    center_y_m: float | None = None,
) -> SpherePlacement:
    """Place a sphere as low as possible while clearing every sampled node."""

    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    center_x = patch.center_x_m if center_x_m is None else float(center_x_m)
    center_y = patch.center_y_m if center_y_m is None else float(center_y_m)
    x_offset = patch.x_global_m[None, :] - center_x
    y_offset = patch.y_global_m[:, None] - center_y
    radial_squared = x_offset * x_offset + y_offset * y_offset
    inside = radial_squared <= radius_m * radius_m * (1.0 + 1e-12)
    if not np.any(inside):
        raise ValueError("terrain patch contains no nodes below the sphere footprint")

    cap = np.sqrt(np.maximum(0.0, radius_m * radius_m - radial_squared))
    required_center_height = np.where(
        inside, patch.height_m + cap, -np.inf
    )
    flat_support = int(np.argmax(required_center_height))
    support_row, support_column = np.unravel_index(
        flat_support, required_center_height.shape
    )
    center_z = float(required_center_height[support_row, support_column])
    clearances = center_z - cap[inside] - patch.height_m[inside]
    support_x = float(patch.x_global_m[support_column])
    support_y = float(patch.y_global_m[support_row])
    support_z = float(patch.height_m[support_row, support_column])
    return SpherePlacement(
        radius_m=float(radius_m),
        center_xyz_m=(center_x, center_y, center_z),
        support_xyz_m=(support_x, support_y, support_z),
        minimum_clearance_m=float(np.min(clearances)),
        evaluated_sample_count=int(np.count_nonzero(inside)),
    )


def _centered_box_sum(
    values: NDArray[np.float64], half_width: int
) -> NDArray[np.float64]:
    width = 2 * half_width + 1
    integral = np.pad(
        np.cumsum(np.cumsum(values, axis=0), axis=1),
        ((1, 0), (1, 0)),
    )
    sums = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )
    output = np.full(values.shape, np.nan, dtype=np.float64)
    output[
        half_width : values.shape[0] - half_width,
        half_width : values.shape[1] - half_width,
    ] = sums
    return output


def select_groove_center(
    patch: TerrainPatch,
    *,
    sphere_radius_m: float,
) -> GrooveSelection:
    """Select a broad local depression suitable for a finite spherical tip."""

    if sphere_radius_m <= 0.0 or not math.isfinite(sphere_radius_m):
        raise ValueError("sphere_radius_m must be positive and finite")
    if patch.shape[0] < 7 or patch.shape[1] < 7:
        raise ValueError("terrain patch is too small for groove selection")
    spacing_x = float(np.median(np.diff(patch.x_global_m)))
    spacing_y = float(np.median(np.diff(patch.y_global_m)))
    inner_radius = max(0.5 * sphere_radius_m, spacing_x, spacing_y)
    outer_radius = max(2.0 * sphere_radius_m, inner_radius + spacing_x)
    inner_half = max(
        1, int(round(inner_radius / max(spacing_x, spacing_y)))
    )
    outer_half = max(
        inner_half + 1,
        int(round(outer_radius / max(spacing_x, spacing_y))),
    )
    maximum_half = (min(patch.shape) - 1) // 2
    if outer_half >= maximum_half:
        raise ValueError("terrain patch is too small around the sphere footprint")

    inner_sum = _centered_box_sum(patch.height_m, inner_half)
    outer_sum = _centered_box_sum(patch.height_m, outer_half)
    inner_count = (2 * inner_half + 1) ** 2
    outer_count = (2 * outer_half + 1) ** 2
    inner_mean = inner_sum / inner_count
    surrounding_mean = (outer_sum - inner_sum) / (
        outer_count - inner_count
    )
    depth_score = surrounding_mean - inner_mean
    valid = np.isfinite(depth_score)
    valid_values = inner_mean[valid]
    center_values = patch.height_m[valid]
    low_inner_limit = float(np.quantile(valid_values, 0.35))
    low_center_limit = float(np.quantile(center_values, 0.35))
    eligible = (
        valid
        & (inner_mean <= low_inner_limit)
        & (patch.height_m <= low_center_limit)
    )
    if not np.any(eligible):
        eligible = valid
    selected_flat = int(
        np.argmax(np.where(eligible, depth_score, -np.inf))
    )
    row, column = np.unravel_index(selected_flat, patch.shape)
    return GrooveSelection(
        center_x_m=float(patch.x_global_m[column]),
        center_y_m=float(patch.y_global_m[row]),
        center_height_m=float(patch.height_m[row, column]),
        inner_mean_height_m=float(inner_mean[row, column]),
        surrounding_mean_height_m=float(surrounding_mean[row, column]),
        depth_score_m=float(depth_score[row, column]),
        inner_radius_m=inner_half * max(spacing_x, spacing_y),
        outer_radius_m=outer_half * max(spacing_x, spacing_y),
    )


def _downsample_patch(
    patch: TerrainPatch, *, maximum_axis_points: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if maximum_axis_points < 2:
        raise ValueError("maximum_axis_points must be at least 2")

    def indices(count: int) -> NDArray[np.int64]:
        sample_count = min(count, maximum_axis_points)
        return np.unique(
            np.rint(np.linspace(0, count - 1, sample_count)).astype(np.int64)
        )

    rows = indices(patch.shape[0])
    columns = indices(patch.shape[1])
    return (
        patch.x_global_m[columns],
        patch.y_global_m[rows],
        patch.height_m[np.ix_(rows, columns)],
    )


def _nice_scale_bar(span_um: float) -> float:
    target = max(span_um * 0.2, np.finfo(float).tiny)
    power = 10.0 ** math.floor(math.log10(target))
    factor = target / power
    nice_factor = 1.0 if factor < 1.5 else 2.0 if factor < 3.5 else 5.0
    return nice_factor * power


def _save_figure(figure: Any, path: Path, *, dpi: int) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        figure.savefig(
            temporary,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            format=path.suffix.lstrip("."),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _render_oblique(
    patch: TerrainPatch,
    path: Path,
    *,
    maximum_axis_points: int,
    dpi: int,
) -> float:
    import matplotlib.pyplot as plt

    x, y, height = _downsample_patch(
        patch, maximum_axis_points=maximum_axis_points
    )
    x_mm = (x - patch.center_x_m) * 1e3
    y_mm = (y - patch.center_y_m) * 1e3
    z_um = height * 1e6
    x_mesh, y_mesh = np.meshgrid(x_mm, y_mm)
    horizontal_span_um = (
        max(float(np.ptp(x_mm)), float(np.ptp(y_mm))) * 1e3
    )
    vertical_span_um = max(float(np.ptp(z_um)), 1e-9)
    vertical_exaggeration = min(
        1.5,
        max(1.0, 0.35 * horizontal_span_um / vertical_span_um),
    )
    size_label = f"{patch.size_x_m * 1e3:g} × {patch.size_y_m * 1e3:g} mm"

    figure = plt.figure(figsize=(9.4, 7.0), layout="constrained")
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        x_mesh,
        y_mesh,
        z_um,
        cmap="terrain",
        linewidth=0.0,
        antialiased=True,
        rcount=min(z_um.shape[0], maximum_axis_points),
        ccount=min(z_um.shape[1], maximum_axis_points),
    )
    axis.set_xlabel("x from crop centre (mm)")
    axis.set_ylabel("y from crop centre (mm)")
    axis.set_zlabel("height z (µm)")
    axis.set_title(
        f"{size_label} random terrain — oblique view "
        f"(vertical ×{vertical_exaggeration:.1f})"
    )
    axis.view_init(elev=35.0, azim=-128.0)
    axis.set_box_aspect(
        (
            max(float(np.ptp(x_mm)), 1e-6) * 1e3,
            max(float(np.ptp(y_mm)), 1e-6) * 1e3,
            vertical_span_um * vertical_exaggeration,
        )
    )
    z_tick_step = 50.0
    z_tick_min = math.floor(float(np.min(z_um)) / z_tick_step) * z_tick_step
    z_tick_max = math.ceil(float(np.max(z_um)) / z_tick_step) * z_tick_step
    axis.set_zticks([z_tick_min, z_tick_max])
    colorbar = figure.colorbar(surface, ax=axis, shrink=0.68, pad=0.08)
    colorbar.set_label("height (µm)")

    scale_mm = _nice_scale_bar(float(np.ptp(x_mm)))
    scale_fraction = scale_mm / float(np.ptp(x_mm))
    bar_start = 0.08
    bar_y = 0.08
    axis.annotate(
        "",
        xy=(bar_start + scale_fraction, bar_y),
        xytext=(bar_start, bar_y),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 4.0},
    )
    axis.text2D(
        bar_start + 0.5 * scale_fraction,
        bar_y + 0.025,
        f"{scale_mm:g} mm",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
    )
    _save_figure(figure, path, dpi=dpi)
    plt.close(figure)
    return vertical_exaggeration


def _render_heightmap(
    patch: TerrainPatch,
    path: Path,
    *,
    groove: GrooveSelection,
    sphere_radius_m: float,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    x_mm = (patch.x_global_m - patch.center_x_m) * 1e3
    y_mm = (patch.y_global_m - patch.center_y_m) * 1e3
    size_label = f"{patch.size_x_m * 1e3:g} × {patch.size_y_m * 1e3:g} mm"
    figure, axis = plt.subplots(figsize=(8.2, 7.0), layout="constrained")
    image = axis.imshow(
        patch.height_m * 1e6,
        origin="lower",
        extent=(x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]),
        interpolation="nearest",
        cmap="terrain",
        aspect="equal",
    )
    axis.set_xlabel("x from crop centre (mm)")
    axis.set_ylabel("y from crop centre (mm)")
    axis.set_title(f"{size_label} random terrain — 2D height map")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.88)
    colorbar.set_label("height z (µm)")
    scale_mm = _nice_scale_bar(float(np.ptp(x_mm)))
    size_bar = AnchoredSizeBar(
        axis.transData,
        scale_mm,
        f"{scale_mm:g} mm",
        "lower left",
        pad=0.45,
        color="black",
        frameon=True,
        size_vertical=max(0.1, 0.006 * float(np.ptp(y_mm))),
    )
    axis.add_artist(size_bar)
    groove_x_mm = (groove.center_x_m - patch.center_x_m) * 1e3
    groove_y_mm = (groove.center_y_m - patch.center_y_m) * 1e3
    axis.add_patch(
        Circle(
            (groove_x_mm, groove_y_mm),
            sphere_radius_m * 1e3,
            fill=False,
            edgecolor="white",
            linewidth=1.8,
        )
    )
    axis.scatter(
        [groove_x_mm],
        [groove_y_mm],
        marker="+",
        s=60,
        linewidth=1.8,
        color="white",
    )
    axis.annotate(
        "tip detail",
        xy=(groove_x_mm, groove_y_mm),
        xytext=(8, 8),
        textcoords="offset points",
        color="white",
        fontsize=9,
        weight="bold",
    )
    _save_figure(figure, path, dpi=dpi)
    plt.close(figure)


def _render_true_scale_sphere(
    patch: TerrainPatch,
    placement: SpherePlacement,
    path: Path,
    *,
    maximum_axis_points: int,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d import proj3d

    x, y, height = _downsample_patch(
        patch, maximum_axis_points=maximum_axis_points
    )
    center = np.asarray(placement.center_xyz_m, dtype=np.float64)
    support = np.asarray(placement.support_xyz_m, dtype=np.float64)
    x_um = (x - center[0]) * 1e6
    y_um = (y - center[1]) * 1e6
    z_um = height * 1e6
    x_mesh, y_mesh = np.meshgrid(x_um, y_um)

    azimuth = np.linspace(0.0, 2.0 * np.pi, 64)
    polar = np.linspace(0.0, np.pi, 40)
    sphere_x = placement.radius_m * np.outer(
        np.cos(azimuth), np.sin(polar)
    )
    sphere_y = placement.radius_m * np.outer(
        np.sin(azimuth), np.sin(polar)
    )
    sphere_z = center[2] + placement.radius_m * np.outer(
        np.ones_like(azimuth), np.cos(polar)
    )

    figure = plt.figure(figsize=(9.4, 7.2), layout="constrained")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_proj_type("ortho")
    axis.plot_surface(
        x_mesh,
        y_mesh,
        z_um,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        alpha=0.42,
    )
    axis.plot_surface(
        sphere_x * 1e6,
        sphere_y * 1e6,
        sphere_z * 1e6,
        color="#ff8c00",
        linewidth=0.45,
        edgecolor="#7a3500",
        alpha=0.94,
        shade=True,
    )
    support_relative_um = np.array(
        [
            (support[0] - center[0]) * 1e6,
            (support[1] - center[1]) * 1e6,
            support[2] * 1e6,
        ]
    )
    center_um = np.array([0.0, 0.0, center[2] * 1e6])
    axis.plot(
        [center_um[0], support_relative_um[0]],
        [center_um[1], support_relative_um[1]],
        [center_um[2], support_relative_um[2]],
        color="#b22222",
        linewidth=3.0,
    )
    axis.scatter(
        [support_relative_um[0]],
        [support_relative_um[1]],
        [support_relative_um[2]],
        color="#b22222",
        marker="D",
        s=75,
        depthshade=False,
    )
    radius_midpoint = 0.5 * (center_um + support_relative_um)
    axis.text(
        radius_midpoint[0],
        radius_midpoint[1],
        radius_midpoint[2],
        f" R = {placement.radius_m * 1e6:g} µm",
        color="#8b1a1a",
    )

    x_limits = (float(x_um[0]), float(x_um[-1]))
    y_limits = (float(y_um[0]), float(y_um[-1]))
    z_margin = 0.06 * placement.radius_m * 1e6
    z_limits = (
        float(np.min(z_um) - z_margin),
        float((center[2] + placement.radius_m) * 1e6 + z_margin),
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_zlim(*z_limits)
    axis.set_box_aspect(
        (
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0],
        )
    )
    support_angle_degrees = math.degrees(
        math.atan2(
            support_relative_um[1],
            support_relative_um[0],
        )
    )
    axis.view_init(elev=36.0, azim=support_angle_degrees)
    projected_x, projected_y, _ = proj3d.proj_transform(
        support_relative_um[0],
        support_relative_um[1],
        support_relative_um[2],
        axis.get_proj(),
    )
    axis.annotate(
        "support",
        xy=(projected_x, projected_y),
        xycoords=axis.transData,
        xytext=(24, -28),
        textcoords="offset points",
        color="#9e1b1b",
        weight="bold",
        arrowprops={
            "arrowstyle": "->",
            "color": "#9e1b1b",
            "linewidth": 1.8,
        },
    )
    axis.set_xlabel("x from sphere centre (µm)")
    axis.set_ylabel("y from sphere centre (µm)")
    axis.set_zlabel("height z (µm)")
    radius_label = f"{placement.radius_m * 1e6:g} µm"
    axis.set_title(
        f"{radius_label} effective spherical tip in selected groove "
        "— physical 1:1 scale"
    )
    axis.text2D(
        0.02,
        0.96,
        (
            f"Groove-centred {patch.size_x_m * 1e3:g} × "
            f"{patch.size_y_m * 1e3:g} mm crop · no z exaggeration"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    axis.legend(
        handles=[
            Patch(facecolor="#2b8c6b", alpha=0.42, label="terrain"),
            Patch(
                facecolor="#ff8c00",
                alpha=0.94,
                label=f"{radius_label} sphere",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="#b22222",
                linewidth=0,
                label="discrete support",
            ),
        ],
        loc="upper right",
    )
    _save_figure(figure, path, dpi=dpi)
    plt.close(figure)


def render_terrain_views(
    library_root: str | Path,
    terrain_recipe_id: str,
    region_id: str,
    output_dir: str | Path,
    *,
    center_x_m: float | None = None,
    center_y_m: float | None = None,
    overview_size_m: float = 10e-3,
    sphere_radius_m: float = 100e-6,
    overview_maximum_axis_points: int = 1201,
    surface_maximum_axis_points: int = 181,
    dpi: int = 180,
    prefix: str = "terrain",
) -> dict[str, Any]:
    """Render two overviews and a local true-scale sphere comparison."""

    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    if not prefix or any(character in prefix for character in "\\/:*?\"<>|"):
        raise ValueError("prefix must be a non-empty filename-safe string")

    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError(
            "terrain plotting requires the optional 'plot' dependency; "
            "install the project with pip install -e \".[plot]\""
        ) from error
    matplotlib.use("Agg", force=True)

    library = TerrainLibrary(library_root)
    region = library.load_region_spec(terrain_recipe_id, region_id)
    mapped = library.open_region(terrain_recipe_id, region_id, verify_hash=False)
    try:
        overview_patch = extract_centered_patch(
            mapped,
            region,
            center_x_m=center_x_m,
            center_y_m=center_y_m,
            window_size_x_m=overview_size_m,
            window_size_y_m=overview_size_m,
            maximum_axis_points=overview_maximum_axis_points,
        )
        groove = select_groove_center(
            overview_patch,
            sphere_radius_m=sphere_radius_m,
        )
        sphere_window_m = max(5.0 * sphere_radius_m, 300e-6)
        sphere_patch = extract_centered_patch(
            mapped,
            region,
            center_x_m=groove.center_x_m,
            center_y_m=groove.center_y_m,
            window_size_x_m=sphere_window_m,
            window_size_y_m=sphere_window_m,
        )
    finally:
        mapped._mmap.close()
        del mapped

    placement = place_sphere_on_patch(
        sphere_patch,
        radius_m=sphere_radius_m,
        center_x_m=groove.center_x_m,
        center_y_m=groove.center_y_m,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "oblique_3d": output / f"{prefix}-3d-oblique.png",
        "heightmap_2d": output / f"{prefix}-2d-heightmap.png",
        "sphere_true_scale": output / f"{prefix}-sphere-true-scale.png",
    }
    vertical_exaggeration = _render_oblique(
        overview_patch,
        paths["oblique_3d"],
        maximum_axis_points=surface_maximum_axis_points,
        dpi=dpi,
    )
    _render_heightmap(
        overview_patch,
        paths["heightmap_2d"],
        groove=groove,
        sphere_radius_m=sphere_radius_m,
        dpi=dpi,
    )
    _render_true_scale_sphere(
        sphere_patch,
        placement,
        paths["sphere_true_scale"],
        maximum_axis_points=surface_maximum_axis_points,
        dpi=dpi,
    )

    metadata: dict[str, Any] = {
        "schema_version": "1",
        "source": {
            "library_root": str(Path(library_root)),
            "terrain_recipe_id": terrain_recipe_id,
            "region_id": region_id,
            "access": "read_only_memory_map_then_bounded_sampling",
        },
        "overview_patch": {
            "center_xy_m": [
                overview_patch.center_x_m,
                overview_patch.center_y_m,
            ],
            "size_xy_m": [
                overview_patch.size_x_m,
                overview_patch.size_y_m,
            ],
            "source_shape": list(overview_patch.source_shape),
            "render_shape": list(overview_patch.shape),
            "height_min_m": float(np.min(overview_patch.height_m)),
            "height_max_m": float(np.max(overview_patch.height_m)),
            "height_rms_m": float(
                np.sqrt(np.mean(np.square(overview_patch.height_m)))
            ),
        },
        "sphere": {
            "radius_m": placement.radius_m,
            "local_patch_size_xy_m": [
                sphere_patch.size_x_m,
                sphere_patch.size_y_m,
            ],
            "center_xyz_m": list(placement.center_xyz_m),
            "support_xyz_m": list(placement.support_xyz_m),
            "minimum_clearance_m": placement.minimum_clearance_m,
            "evaluated_sample_count": placement.evaluated_sample_count,
            "placement_rule": (
                "lowest sphere centre at fixed x-y with no penetration of "
                "sampled terrain nodes inside the circular footprint"
            ),
        },
        "groove_selection": {
            "center_xy_m": [groove.center_x_m, groove.center_y_m],
            "center_height_m": groove.center_height_m,
            "inner_mean_height_m": groove.inner_mean_height_m,
            "surrounding_mean_height_m": groove.surrounding_mean_height_m,
            "depth_score_m": groove.depth_score_m,
            "inner_radius_m": groove.inner_radius_m,
            "outer_radius_m": groove.outer_radius_m,
            "rule": (
                "maximum surrounding-minus-inner mean among low central "
                "terrain candidates"
            ),
        },
        "rendering": {
            "overview_maximum_axis_points": overview_maximum_axis_points,
            "surface_maximum_axis_points": surface_maximum_axis_points,
            "dpi": dpi,
            "oblique_vertical_exaggeration": vertical_exaggeration,
            "sphere_view_axis_scale": "physical_1_to_1_no_z_exaggeration",
        },
        "files": {name: path.name for name, path in paths.items()},
    }
    metadata_path = output / f"{prefix}-metadata.json"
    atomic_write_json(metadata_path, metadata)
    return {
        "metadata_path": str(metadata_path),
        "files": {name: str(path) for name, path in paths.items()},
        "sphere": metadata["sphere"],
    }
