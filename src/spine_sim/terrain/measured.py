"""Measured topography ingestion and conservative preprocessing.

The public API in :mod:`spine_sim.terrain.api` uses this module for measured
terrain.  Raw files are never modified.  Every loader returns SI units, a
separate validity mask, and a provenance/preprocessing record.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from .errors import GeometryOutOfDomainError, TerrainConfigurationError


_LENGTH_SCALES = {
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "μm": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
}


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without changing the source."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MeasuredSurface:
    """A standardized, finite 2.5-D measured surface in SI units."""

    height_m: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    dx_m: float
    dy_m: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        height = np.asarray(self.height_m)
        mask = np.asarray(self.valid_mask)
        if height.ndim != 2 or min(height.shape) < 2:
            raise TerrainConfigurationError(
                "measured height must contain at least 2x2 samples"
            )
        if height.dtype != np.float32:
            raise TerrainConfigurationError("measured height must use float32")
        if mask.shape != height.shape or mask.dtype != np.bool_:
            raise TerrainConfigurationError(
                "measured valid_mask must be bool with the height shape"
            )
        if not np.all(np.isfinite(height)):
            raise TerrainConfigurationError("measured height contains NaN or Inf")
        if not np.any(mask):
            raise TerrainConfigurationError("measured surface has no valid samples")
        if not math.isfinite(self.dx_m) or self.dx_m <= 0:
            raise TerrainConfigurationError("dx_m must be positive and finite")
        if not math.isfinite(self.dy_m) or self.dy_m <= 0:
            raise TerrainConfigurationError("dy_m must be positive and finite")

    @property
    def shape(self) -> tuple[int, int]:
        return self.height_m.shape

    @property
    def size_x_m(self) -> float:
        return (self.height_m.shape[1] - 1) * self.dx_m

    @property
    def size_y_m(self) -> float:
        return (self.height_m.shape[0] - 1) * self.dy_m


def _unit_scale(unit: str, *, field_name: str) -> float:
    try:
        return _LENGTH_SCALES[unit]
    except KeyError as exc:
        raise TerrainConfigurationError(
            f"unsupported {field_name} {unit!r}; choose {sorted(_LENGTH_SCALES)}"
        ) from exc


def _finite_placeholder(
    height: NDArray[np.float64], valid_mask: NDArray[np.bool_]
) -> NDArray[np.float32]:
    """Replace invalid storage values while retaining their mask semantics."""

    valid_values = height[valid_mask]
    if valid_values.size == 0:
        raise TerrainConfigurationError("measured source contains no valid height")
    placeholder = float(np.median(valid_values))
    finite = np.where(valid_mask, height, placeholder)
    return np.asarray(finite, dtype=np.float32)


def _exclude_invalid_margin(
    valid_mask: NDArray[np.bool_], margin_samples: int
) -> NDArray[np.bool_]:
    """Exclude a square safety margin around known invalid instrument pixels."""

    if margin_samples <= 0:
        return valid_mask
    window = 2 * margin_samples + 1
    invalid = (~valid_mask).astype(np.int32)
    padded_x = np.pad(invalid, ((0, 0), (margin_samples, margin_samples)))
    cumulative_x = np.concatenate(
        (
            np.zeros((invalid.shape[0], 1), dtype=np.int32),
            np.cumsum(padded_x, axis=1, dtype=np.int32),
        ),
        axis=1,
    )
    horizontal = (
        cumulative_x[:, window:] - cumulative_x[:, :-window]
    ) > 0
    padded_y = np.pad(
        horizontal.astype(np.int32),
        ((margin_samples, margin_samples), (0, 0)),
    )
    cumulative_y = np.concatenate(
        (
            np.zeros((1, invalid.shape[1]), dtype=np.int32),
            np.cumsum(padded_y, axis=0, dtype=np.int32),
        ),
        axis=0,
    )
    near_invalid = (
        cumulative_y[window:, :] - cumulative_y[:-window, :]
    ) > 0
    return valid_mask & ~near_invalid


def _robust_plane(
    height_m: NDArray[np.float32],
    valid_mask: NDArray[np.bool_],
    dx_m: float,
    dy_m: float,
    *,
    maximum_fit_points: int = 200_000,
) -> tuple[NDArray[np.float32], list[float]]:
    """Remove a rigid plane with three deterministic MAD-rejection iterations."""

    rows, columns = np.nonzero(valid_mask)
    if rows.size < 3:
        raise TerrainConfigurationError("at least three valid points are needed to level")
    stride = max(1, math.ceil(rows.size / maximum_fit_points))
    rows = rows[::stride]
    columns = columns[::stride]
    z = np.asarray(height_m[rows, columns], dtype=np.float64)
    x = (columns - 0.5 * (height_m.shape[1] - 1)) * dx_m
    y = (rows - 0.5 * (height_m.shape[0] - 1)) * dy_m
    design = np.column_stack((x, y, np.ones(x.size)))
    keep = np.ones(x.size, dtype=np.bool_)
    coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(3):
        coefficients, *_ = np.linalg.lstsq(design[keep], z[keep], rcond=None)
        residual = z - design @ coefficients
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        if mad <= np.finfo(np.float64).eps:
            break
        updated = np.abs(residual - median) <= 4.5 * 1.4826 * mad
        if np.count_nonzero(updated) < 3 or np.array_equal(updated, keep):
            break
        keep = updated

    x_axis = (
        np.arange(height_m.shape[1], dtype=np.float64)
        - 0.5 * (height_m.shape[1] - 1)
    ) * dx_m
    y_axis = (
        np.arange(height_m.shape[0], dtype=np.float64)
        - 0.5 * (height_m.shape[0] - 1)
    ) * dy_m
    plane = (
        coefficients[0] * x_axis[None, :]
        + coefficients[1] * y_axis[:, None]
        + coefficients[2]
    )
    leveled = np.asarray(height_m, dtype=np.float64) - plane
    leveled -= float(np.mean(leveled[valid_mask]))
    return np.asarray(leveled, dtype=np.float32), coefficients.tolist()


def _parse_hirox(path: Path) -> tuple[NDArray[np.float64], float, float, dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header_lines = [stream.readline().strip() for _ in range(5)]
    except OSError as exc:
        raise TerrainConfigurationError(f"cannot read Hirox header: {exc}") from exc
    pairs: dict[str, str] = {}
    for line in header_lines:
        if "," not in line:
            raise TerrainConfigurationError("invalid Hirox key,value header")
        key, value = line.split(",", maxsplit=1)
        pairs[key.strip()] = value.strip()
    required = {"Measured Date", "Calibration", "Height Unit", "X size", "Y size"}
    missing = required - pairs.keys()
    if missing:
        raise TerrainConfigurationError(
            f"Hirox header is missing fields: {sorted(missing)}"
        )
    calibration = pairs["Calibration"].replace("μ", "u").replace("µ", "u")
    if not calibration.lower().endswith("um/pxl"):
        raise TerrainConfigurationError(
            f"unsupported Hirox calibration {pairs['Calibration']!r}"
        )
    pitch_m = float(calibration[:-6].strip()) * 1e-6
    height_scale = _unit_scale(pairs["Height Unit"], field_name="height unit")
    try:
        height = np.loadtxt(path, delimiter=",", skiprows=5, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise TerrainConfigurationError(f"cannot parse Hirox matrix: {exc}") from exc
    expected = (int(pairs["Y size"]), int(pairs["X size"]))
    if height.shape != expected:
        raise TerrainConfigurationError(
            f"Hirox matrix shape {height.shape} does not match header {expected}"
        )
    return (
        height * height_scale,
        pitch_m,
        pitch_m,
        {
            "format": "hirox_csv",
            "measured_date": pairs["Measured Date"],
            "raw_height_unit": pairs["Height Unit"],
            "raw_lateral_calibration": pairs["Calibration"],
            "raw_shape_yx": list(expected),
        },
    )


def _read_ascii_ply_vertices(path: Path) -> NDArray[np.float64]:
    with path.open("r", encoding="ascii", errors="strict") as stream:
        if stream.readline().strip() != "ply":
            raise TerrainConfigurationError("PLY file does not start with 'ply'")
        vertex_count: int | None = None
        properties: list[str] = []
        in_vertex = False
        while True:
            line = stream.readline()
            if not line:
                raise TerrainConfigurationError("truncated PLY header")
            fields = line.strip().split()
            if fields[:2] == ["format", "binary_little_endian"] or fields[:2] == [
                "format",
                "binary_big_endian",
            ]:
                raise TerrainConfigurationError(
                    "binary PLY requires the optional SurfaceTopography backend"
                )
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertex = True
            elif fields[:1] == ["element"]:
                in_vertex = False
            elif in_vertex and fields[:1] == ["property"]:
                properties.append(fields[-1])
            elif fields[:1] == ["end_header"]:
                break
        if vertex_count is None:
            raise TerrainConfigurationError("PLY header has no vertex element")
        try:
            indices = [properties.index(axis) for axis in ("x", "y", "z")]
        except ValueError as exc:
            raise TerrainConfigurationError("PLY vertices need x, y and z") from exc
        points = np.empty((vertex_count, 3), dtype=np.float64)
        for row in range(vertex_count):
            fields = stream.readline().split()
            if len(fields) < len(properties):
                raise TerrainConfigurationError("truncated PLY vertex table")
            points[row] = [float(fields[index]) for index in indices]
    return points


def _read_stl_vertices(path: Path) -> NDArray[np.float64]:
    """Read ASCII or binary STL vertex coordinates."""

    with path.open("rb") as stream:
        prefix = stream.read(84)
        if len(prefix) < 84:
            raise TerrainConfigurationError("truncated STL file")
        triangle_count = struct.unpack("<I", prefix[80:84])[0]
        binary_size = 84 + 50 * triangle_count
        is_binary = path.stat().st_size == binary_size
        stream.seek(0)
        if is_binary:
            stream.read(84)
            points = np.empty((triangle_count * 3, 3), dtype=np.float64)
            for triangle in range(triangle_count):
                record = stream.read(50)
                if len(record) != 50:
                    raise TerrainConfigurationError("truncated binary STL")
                values = struct.unpack("<12fH", record)
                points[3 * triangle : 3 * triangle + 3] = np.asarray(
                    values[3:12], dtype=np.float64
                ).reshape(3, 3)
            return points
    vertices: list[list[float]] = []
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            fields = line.strip().split()
            if fields[:1] == ["vertex"] and len(fields) == 4:
                vertices.append([float(value) for value in fields[1:]])
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TerrainConfigurationError(f"cannot parse ASCII STL: {exc}") from exc
    if not vertices:
        raise TerrainConfigurationError("STL contains no vertices")
    return np.asarray(vertices, dtype=np.float64)


def _points_to_grid(
    points_m: NDArray[np.float64],
    *,
    spacing_x_m: float | None,
    spacing_y_m: float | None,
    maximum_missing_fraction: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], float, float, dict[str, Any]]:
    if points_m.ndim != 2 or points_m.shape[1] < 3:
        raise TerrainConfigurationError("point source must contain x, y, z columns")
    points_m = points_m[:, :3]
    points_m = points_m[np.all(np.isfinite(points_m), axis=1)]
    if points_m.shape[0] < 4:
        raise TerrainConfigurationError("point source has fewer than four finite points")

    unique_x = np.unique(points_m[:, 0])
    unique_y = np.unique(points_m[:, 1])
    regular = unique_x.size * unique_y.size == points_m.shape[0]
    if regular and unique_x.size >= 2 and unique_y.size >= 2:
        dx = np.diff(unique_x)
        dy = np.diff(unique_y)
        regular = bool(
            np.allclose(dx, dx[0], rtol=1e-6, atol=1e-15)
            and np.allclose(dy, dy[0], rtol=1e-6, atol=1e-15)
        )
    if regular:
        dx_m = float(unique_x[1] - unique_x[0])
        dy_m = float(unique_y[1] - unique_y[0])
        ix = np.searchsorted(unique_x, points_m[:, 0])
        iy = np.searchsorted(unique_y, points_m[:, 1])
        height = np.full((unique_y.size, unique_x.size), np.nan)
        height[iy, ix] = points_m[:, 2]
    else:
        if spacing_x_m is None or spacing_y_m is None:
            raise TerrainConfigurationError(
                "irregular XYZ/mesh input requires spacing_x_m and spacing_y_m"
            )
        dx_m, dy_m = float(spacing_x_m), float(spacing_y_m)
        if dx_m <= 0 or dy_m <= 0:
            raise TerrainConfigurationError("point-cloud spacing must be positive")
        ix = np.rint((points_m[:, 0] - np.min(points_m[:, 0])) / dx_m).astype(int)
        iy = np.rint((points_m[:, 1] - np.min(points_m[:, 1])) / dy_m).astype(int)
        height = np.full((int(iy.max()) + 1, int(ix.max()) + 1), np.nan)
        # A 2.5-D topography retains the uppermost surface at duplicate x/y.
        for x_index, y_index, z_value in zip(ix, iy, points_m[:, 2], strict=True):
            current = height[y_index, x_index]
            if not np.isfinite(current) or z_value > current:
                height[y_index, x_index] = z_value
    valid = np.isfinite(height)
    missing_fraction = 1.0 - float(np.mean(valid))
    if missing_fraction > maximum_missing_fraction:
        raise TerrainConfigurationError(
            f"point-cloud grid is {missing_fraction:.1%} missing; refusing large-area fill"
        )
    if not np.all(valid):
        height = _fill_sparse_invalid(height, valid)
    return height, valid, dx_m, dy_m, {
        "point_count": int(points_m.shape[0]),
        "grid_reconstruction": "regular_coordinates" if regular else "upper_envelope_bins",
        "missing_fraction": missing_fraction,
    }


def _fill_sparse_invalid(
    height: NDArray[np.float64], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Fill only sparse gaps by deterministic row/column linear interpolation."""

    result = np.array(height, dtype=np.float64, copy=True)
    x = np.arange(result.shape[1])
    for row in range(result.shape[0]):
        known = np.flatnonzero(valid[row])
        if known.size >= 2:
            missing = ~valid[row]
            result[row, missing] = np.interp(x[missing], known, result[row, known])
    y = np.arange(result.shape[0])
    remaining = ~np.isfinite(result)
    for column in range(result.shape[1]):
        known = np.flatnonzero(np.isfinite(result[:, column]))
        if known.size >= 2:
            missing = remaining[:, column]
            result[missing, column] = np.interp(
                y[missing], known, result[known, column]
            )
    if not np.all(np.isfinite(result)):
        raise TerrainConfigurationError(
            "invalid area cannot be filled without extrapolating a large gap"
        )
    return result


def load_measured_surface(
    path: str | Path,
    *,
    format: str = "auto",
    height_unit: str | None = None,
    lateral_unit: str = "m",
    spacing_x_m: float | None = None,
    spacing_y_m: float | None = None,
    invalid_values: Iterable[float] = (),
    dataset_zero_is_invalid: bool = False,
    invalid_margin_samples: int = 0,
    level: str = "robust_plane",
    maximum_missing_fraction: float = 0.05,
    provenance: Mapping[str, Any] | None = None,
) -> MeasuredSurface:
    """Load NPY/CSV/TXT/XYZ/PLY/STL or supported metrology data.

    ``dataset_zero_is_invalid`` is deliberately opt-in.  It is used for the
    public Hirox files because their zero-height wedges are treated as an
    instrument-boundary assumption; generic zero height remains valid.
    """

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    selected = format.lower()
    suffix = source.suffix.lower()
    if selected == "auto":
        if suffix == ".csv":
            first_line = source.open(
                "r", encoding="utf-8-sig", errors="replace"
            ).readline()
            selected = "hirox_csv" if first_line.startswith("Measured Date,") else "csv"
        else:
            selected = suffix.lstrip(".")

    parser_record: dict[str, Any]
    valid: NDArray[np.bool_]
    if selected == "hirox_csv":
        height, dx_m, dy_m, parser_record = _parse_hirox(source)
        valid = np.isfinite(height)
    elif selected in {"npy", "csv", "txt"}:
        try:
            if selected == "npy":
                raw = np.load(source, allow_pickle=False)
            else:
                raw = np.loadtxt(
                    source, delimiter="," if selected == "csv" else None
                )
        except (OSError, ValueError) as exc:
            raise TerrainConfigurationError(
                f"cannot parse measured height matrix: {exc}"
            ) from exc
        if raw.ndim != 2 or min(raw.shape) < 2:
            raise TerrainConfigurationError(
                "measured height matrix must contain at least 2x2 samples"
            )
        if height_unit is None or spacing_x_m is None or spacing_y_m is None:
            raise TerrainConfigurationError(
                "matrix input requires height_unit, spacing_x_m and spacing_y_m"
            )
        height = np.asarray(raw, dtype=np.float64) * _unit_scale(
            height_unit, field_name="height unit"
        )
        dx_m, dy_m = float(spacing_x_m), float(spacing_y_m)
        valid = np.isfinite(height)
        parser_record = {"format": selected, "raw_shape_yx": list(height.shape)}
    elif selected in {"xyz", "ply", "stl"}:
        if selected == "xyz":
            try:
                points = np.loadtxt(source, dtype=np.float64)
            except (OSError, ValueError) as exc:
                raise TerrainConfigurationError(f"cannot parse XYZ: {exc}") from exc
        elif selected == "ply":
            points = _read_ascii_ply_vertices(source)
        else:
            points = _read_stl_vertices(source)
        if height_unit is None:
            raise TerrainConfigurationError(
                "XYZ/PLY/STL input requires the coordinate unit in height_unit"
            )
        points *= _unit_scale(lateral_unit, field_name="lateral unit")
        # Permit a distinct z scale when an exporter used mixed units.
        points[:, 2] *= _unit_scale(height_unit, field_name="height unit") / _unit_scale(
            lateral_unit, field_name="lateral unit"
        )
        height, valid, dx_m, dy_m, reconstruction = _points_to_grid(
            points,
            spacing_x_m=spacing_x_m,
            spacing_y_m=spacing_y_m,
            maximum_missing_fraction=maximum_missing_fraction,
        )
        parser_record = {"format": selected, **reconstruction}
    elif selected in {"x3p", "sdf", "os3d"}:
        try:
            import SurfaceTopography  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TerrainConfigurationError(
                f"{selected.upper()} requires the optional SurfaceTopography package"
            ) from exc
        try:
            topography = SurfaceTopography.read_topography(str(source))
            height = np.asarray(topography.heights(), dtype=np.float64)
            size_x, size_y = topography.physical_sizes
        except Exception as exc:  # third-party format errors vary by backend
            raise TerrainConfigurationError(
                f"SurfaceTopography could not load {source}: {exc}"
            ) from exc
        dx_m = float(size_x) / (height.shape[1] - 1)
        dy_m = float(size_y) / (height.shape[0] - 1)
        valid = np.isfinite(height)
        parser_record = {"format": selected, "backend": "SurfaceTopography"}
    else:
        raise TerrainConfigurationError(f"unsupported measured format {selected!r}")

    if dx_m <= 0 or dy_m <= 0:
        raise TerrainConfigurationError("measured grid spacing must be positive")
    for invalid_value in invalid_values:
        valid &= height != float(invalid_value)
    zero_policy = "zero_is_valid"
    if dataset_zero_is_invalid:
        valid &= height != 0.0
        zero_policy = "dataset_specific_exact_zero_invalid_assumption"
    if invalid_margin_samples < 0:
        raise TerrainConfigurationError("invalid_margin_samples cannot be negative")
    original_exact_valid_fraction = float(np.mean(valid))
    valid = _exclude_invalid_margin(valid, invalid_margin_samples)
    missing_fraction = 1.0 - float(np.mean(valid))
    if missing_fraction > maximum_missing_fraction:
        raise TerrainConfigurationError(
            f"source has {missing_fraction:.1%} invalid samples, above "
            f"maximum_missing_fraction={maximum_missing_fraction:.1%}"
        )
    finite_height = _finite_placeholder(height, valid)
    preprocessing: dict[str, Any] = {
        "invalid_policy": {
            "nonfinite": "invalid",
            "explicit_invalid_values": [float(item) for item in invalid_values],
            "zero": zero_policy,
            "invalid_margin_samples": invalid_margin_samples,
        },
        "original_valid_fraction": float(np.mean(valid)),
        "exact_invalid_policy_valid_fraction_before_margin": (
            original_exact_valid_fraction
        ),
        "invalid_storage_placeholder": "valid_median_mask_preserved",
        "leveling": level,
    }
    if level == "robust_plane":
        finite_height, coefficients = _robust_plane(
            finite_height, valid, dx_m, dy_m
        )
        preprocessing["plane_coefficients_z_ax_by_c"] = coefficients
    elif level == "mean":
        finite_height = np.asarray(
            finite_height - float(np.mean(finite_height[valid])), dtype=np.float32
        )
    elif level != "none":
        raise TerrainConfigurationError(
            "level must be 'robust_plane', 'mean', or 'none'"
        )
    finite_height = np.asarray(finite_height, dtype=np.float32)
    finite_height[~valid] = np.float32(np.median(finite_height[valid]))
    metadata = {
        "schema_version": "measured-surface-v1",
        "source": {
            "original_filename": source.name,
            "path": str(source),
            "sha256": sha256_file(source),
            **dict(provenance or {}),
        },
        "parser": parser_record,
        "original_grid": {
            "spacing_x_m": dx_m,
            "spacing_y_m": dy_m,
            "shape_yx": list(height.shape),
            "physical_node_span_m": [
                (height.shape[1] - 1) * dx_m,
                (height.shape[0] - 1) * dy_m,
            ],
        },
        "preprocessing": preprocessing,
    }
    return MeasuredSurface(
        height_m=finite_height,
        valid_mask=np.asarray(valid, dtype=np.bool_),
        dx_m=dx_m,
        dy_m=dy_m,
        metadata=metadata,
    )


def _box_lowpass(
    values: NDArray[np.float64], window: int, *, axis: int
) -> NDArray[np.float64]:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padding = [(0, 0), (0, 0)]
    padding[axis] = (pad, pad)
    padded = np.pad(values, padding, mode="reflect")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate(
        (np.zeros(zero_shape, dtype=np.float64), cumulative), axis=axis
    )
    upper = [slice(None), slice(None)]
    lower = [slice(None), slice(None)]
    upper[axis] = slice(window, None)
    lower[axis] = slice(None, -window)
    return (cumulative[tuple(upper)] - cumulative[tuple(lower)]) / window


def resample_measured_patch(
    height_m: NDArray[np.float32],
    valid_mask: NDArray[np.bool_],
    *,
    source_dx_m: float,
    source_dy_m: float,
    target_size_x_m: float,
    target_size_y_m: float,
    target_dx_m: float,
    target_dy_m: float,
    allow_upsampling: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, Any]]:
    """Anti-aliased bilinear resampling for an already selected source patch."""

    if (
        target_dx_m < source_dx_m * (1.0 - 1e-12)
        or target_dy_m < source_dy_m * (1.0 - 1e-12)
    ) and not allow_upsampling:
        raise TerrainConfigurationError(
            "measured mode cannot upsample a lower-resolution source"
        )
    nx_float = target_size_x_m / target_dx_m
    ny_float = target_size_y_m / target_dy_m
    nx_intervals = int(round(nx_float))
    ny_intervals = int(round(ny_float))
    if not math.isclose(nx_float, nx_intervals, abs_tol=1e-9, rel_tol=0.0):
        raise TerrainConfigurationError("target size_x_m must align with target_dx_m")
    if not math.isclose(ny_float, ny_intervals, abs_tol=1e-9, rel_tol=0.0):
        raise TerrainConfigurationError("target size_y_m must align with target_dy_m")
    if target_size_x_m > (height_m.shape[1] - 1) * source_dx_m + 1e-15:
        raise GeometryOutOfDomainError("target x span exceeds selected measured patch")
    if target_size_y_m > (height_m.shape[0] - 1) * source_dy_m + 1e-15:
        raise GeometryOutOfDomainError("target y span exceeds selected measured patch")

    values = np.asarray(height_m, dtype=np.float64)
    window_x = max(1, int(math.floor(target_dx_m / source_dx_m)))
    window_y = max(1, int(math.floor(target_dy_m / source_dy_m)))
    if window_x > 1:
        values = _box_lowpass(values, window_x, axis=1)
    if window_y > 1:
        values = _box_lowpass(values, window_y, axis=0)

    x_float = np.arange(nx_intervals + 1) * target_dx_m / source_dx_m
    y_float = np.arange(ny_intervals + 1) * target_dy_m / source_dy_m
    x0 = np.minimum(np.floor(x_float).astype(int), height_m.shape[1] - 2)
    y0 = np.minimum(np.floor(y_float).astype(int), height_m.shape[0] - 2)
    tx = x_float - x0
    ty = y_float - y0
    result = np.empty((ny_intervals + 1, nx_intervals + 1), dtype=np.float64)
    result_mask = np.empty_like(result, dtype=np.bool_)
    for row, (source_y, weight_y) in enumerate(zip(y0, ty, strict=True)):
        top = (1.0 - tx) * values[source_y, x0] + tx * values[source_y, x0 + 1]
        bottom = (
            (1.0 - tx) * values[source_y + 1, x0]
            + tx * values[source_y + 1, x0 + 1]
        )
        result[row] = (1.0 - weight_y) * top + weight_y * bottom
        result_mask[row] = (
            valid_mask[source_y, x0]
            & valid_mask[source_y, x0 + 1]
            & valid_mask[source_y + 1, x0]
            & valid_mask[source_y + 1, x0 + 1]
        )
    result -= float(np.mean(result[result_mask])) if np.any(result_mask) else 0.0
    return (
        np.asarray(result, dtype=np.float32),
        result_mask,
        {
            "method": "box_antialias_then_bilinear",
            "antialias_window_samples_xy": [window_x, window_y],
            "upsampling": bool(
                target_dx_m < source_dx_m or target_dy_m < source_dy_m
            ),
        },
    )


def random_measured_crop(
    surface: MeasuredSurface,
    *,
    size_x_m: float,
    size_y_m: float,
    resolution_m: float,
    rng: np.random.Generator,
    maximum_invalid_fraction: float = 0.01,
    attempts: int = 128,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, Any]]:
    """Select a reproducible valid ROI, then anti-alias and resample it."""

    required_columns = math.ceil(size_x_m / surface.dx_m) + 2
    required_rows = math.ceil(size_y_m / surface.dy_m) + 2
    if required_columns > surface.shape[1] or required_rows > surface.shape[0]:
        raise GeometryOutOfDomainError(
            "requested measured terrain is larger than the measured source; "
            "use mode='synthetic' or 'auto'"
        )
    best: tuple[float, int, int] | None = None
    max_x0 = surface.shape[1] - required_columns
    max_y0 = surface.shape[0] - required_rows
    for _ in range(attempts):
        x0 = int(rng.integers(0, max_x0 + 1))
        y0 = int(rng.integers(0, max_y0 + 1))
        mask = surface.valid_mask[
            y0 : y0 + required_rows, x0 : x0 + required_columns
        ]
        invalid_fraction = 1.0 - float(np.mean(mask))
        candidate = (invalid_fraction, y0, x0)
        if best is None or candidate < best:
            best = candidate
        if invalid_fraction == 0.0:
            break
    assert best is not None
    invalid_fraction, y0, x0 = best
    if invalid_fraction > maximum_invalid_fraction:
        raise GeometryOutOfDomainError(
            f"no acceptable measured crop found; best invalid fraction "
            f"{invalid_fraction:.2%}"
        )
    height = surface.height_m[y0 : y0 + required_rows, x0 : x0 + required_columns]
    mask = surface.valid_mask[y0 : y0 + required_rows, x0 : x0 + required_columns]
    result, result_mask, resampling = resample_measured_patch(
        height,
        mask,
        source_dx_m=surface.dx_m,
        source_dy_m=surface.dy_m,
        target_size_x_m=size_x_m,
        target_size_y_m=size_y_m,
        target_dx_m=resolution_m,
        target_dy_m=resolution_m,
    )
    # Flips retain scale and do not invent unmeasured rotations or directionality.
    flip_x = bool(rng.integers(0, 2))
    flip_y = bool(rng.integers(0, 2))
    if flip_x:
        result = result[:, ::-1].copy()
        result_mask = result_mask[:, ::-1].copy()
    if flip_y:
        result = result[::-1, :].copy()
        result_mask = result_mask[::-1, :].copy()
    return result, result_mask, {
        "crop_origin_source_indices_yx": [y0, x0],
        "crop_shape_source_yx": [required_rows, required_columns],
        "crop_invalid_fraction": invalid_fraction,
        "transform": {"flip_x": flip_x, "flip_y": flip_y, "rotation_deg": 0},
        "resampling": resampling,
    }
