"""Measured/artificial file-heightmap sources with explicit provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spine_sim.core.identity import identity, stable_hash
from spine_sim.io.results import atomic_write_json

from .errors import GeometryOutOfDomainError, TerrainConfigurationError


_HEIGHT_UNITS_M = {
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_source_array(path: Path, *, mmap_mode: str | None = "r") -> NDArray[np.floating]:
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    elif path.suffix.lower() in {".csv", ".txt"}:
        array = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    else:
        raise TerrainConfigurationError(
            "file heightmap must be .npy, .csv or whitespace-delimited .txt"
        )
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise TerrainConfigurationError("file heightmap must contain a numeric 2-D array")
    if not np.all(np.isfinite(array)):
        raise TerrainConfigurationError("file heightmap contains non-finite values")
    return array


@dataclass(frozen=True)
class FileHeightMapSource:
    """Identity and grid metadata for an immutable external heightmap."""

    source_path: str
    source_sha256: str
    source_format: str
    source_height_unit: str
    origin_x_m: float
    origin_y_m: float
    spacing_x_m: float
    spacing_y_m: float
    shape_yx: tuple[int, int]

    def __post_init__(self) -> None:
        if self.source_height_unit not in _HEIGHT_UNITS_M:
            raise TerrainConfigurationError(
                f"unsupported source height unit {self.source_height_unit!r}"
            )
        if self.spacing_x_m <= 0 or self.spacing_y_m <= 0:
            raise TerrainConfigurationError("heightmap spacing must be positive")
        if len(self.shape_yx) != 2 or min(self.shape_yx) < 2:
            raise TerrainConfigurationError(
                "heightmap must contain at least 2x2 nodes"
            )
        if len(self.source_sha256) != 64:
            raise TerrainConfigurationError("invalid source SHA-256")

    @property
    def source_id(self) -> str:
        normalized = asdict(self)
        for key in ("origin_x_m", "origin_y_m", "spacing_x_m", "spacing_y_m"):
            normalized[key] = round(float(normalized[key]), 15)
        return identity("terrain_source", normalized, module_version="m1.0.0")

    @property
    def x_max_m(self) -> float:
        return self.origin_x_m + (self.shape_yx[1] - 1) * self.spacing_x_m

    @property
    def y_max_m(self) -> float:
        return self.origin_y_m + (self.shape_yx[0] - 1) * self.spacing_y_m

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        source_height_unit: str,
        origin_x_m: float,
        origin_y_m: float,
        spacing_x_m: float,
        spacing_y_m: float,
    ) -> "FileHeightMapSource":
        source_path = Path(path).resolve()
        array = _load_source_array(source_path)
        return cls(
            source_path=str(source_path),
            source_sha256=_sha256_file(source_path),
            source_format=source_path.suffix.lower().lstrip("."),
            source_height_unit=source_height_unit,
            origin_x_m=float(origin_x_m),
            origin_y_m=float(origin_y_m),
            spacing_x_m=float(spacing_x_m),
            spacing_y_m=float(spacing_y_m),
            shape_yx=(int(array.shape[0]), int(array.shape[1])),
        )


def register_heightmap_source(
    library_root: str | Path,
    source: FileHeightMapSource,
) -> Path:
    """Record source provenance without copying or relabelling the raw source."""

    target = Path(library_root).resolve() / "sources" / f"{source.source_id}.json"
    document = {
        "schema_version": "1",
        "source_id": source.source_id,
        "source": asdict(source),
        "source_role": "external_raw_input_not_preprocessed_copy",
    }
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != document:
            raise TerrainConfigurationError(
                "source ID already exists with different metadata"
            )
    else:
        atomic_write_json(target, document)
    return target


def sample_file_heightmap(
    source: FileHeightMapSource,
    x_global_m: ArrayLike,
    y_global_m: ArrayLike,
    *,
    detrend_plane: bool = False,
    preprocessing: Mapping[str, Any] | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Bilinearly sample a source grid and return an explicit preprocessing record."""

    source_path = Path(source.source_path)
    if _sha256_file(source_path) != source.source_sha256:
        raise TerrainConfigurationError("heightmap source hash changed")
    x = np.asarray(x_global_m, dtype=np.float64)
    y = np.asarray(y_global_m, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or y.size == 0:
        raise TerrainConfigurationError("query x/y must be non-empty 1-D arrays")
    if (
        x.min() < source.origin_x_m
        or x.max() > source.x_max_m
        or y.min() < source.origin_y_m
        or y.max() > source.y_max_m
    ):
        raise GeometryOutOfDomainError(
            "geometry_out_of_domain: file heightmap query exceeds source domain"
        )

    raw = np.asarray(_load_source_array(source_path, mmap_mode="r"), dtype=np.float64)
    values_m = raw * _HEIGHT_UNITS_M[source.source_height_unit]
    applied: dict[str, Any] = dict(preprocessing or {})
    if detrend_plane:
        yy, xx = np.indices(values_m.shape, dtype=np.float64)
        design = np.column_stack((xx.ravel(), yy.ravel(), np.ones(values_m.size)))
        coefficients, *_ = np.linalg.lstsq(
            design, values_m.ravel(), rcond=None
        )
        values_m = values_m - (
            coefficients[0] * xx
            + coefficients[1] * yy
            + coefficients[2]
        )
        applied["detrend_plane_coefficients_index_space"] = coefficients.tolist()
    applied["interpolation"] = "bilinear"
    applied["query_crop_bounds_m"] = {
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }

    x_float = (x - source.origin_x_m) / source.spacing_x_m
    y_float = (y - source.origin_y_m) / source.spacing_y_m
    x0 = np.minimum(np.floor(x_float).astype(int), source.shape_yx[1] - 2)
    y0 = np.minimum(np.floor(y_float).astype(int), source.shape_yx[0] - 2)
    tx = x_float - x0
    ty = y_float - y0
    result = np.empty((y.size, x.size), dtype=np.float64)
    for row, (source_y, weight_y) in enumerate(zip(y0, ty, strict=True)):
        top = (
            (1.0 - tx) * values_m[source_y, x0]
            + tx * values_m[source_y, x0 + 1]
        )
        bottom = (
            (1.0 - tx) * values_m[source_y + 1, x0]
            + tx * values_m[source_y + 1, x0 + 1]
        )
        result[row, :] = (1.0 - weight_y) * top + weight_y * bottom
    record = {
        "source_id": source.source_id,
        "source_sha256": source.source_sha256,
        "source_height_unit": source.source_height_unit,
        "source_grid": {
            "origin_x_m": source.origin_x_m,
            "origin_y_m": source.origin_y_m,
            "spacing_x_m": source.spacing_x_m,
            "spacing_y_m": source.spacing_y_m,
            "shape_yx": list(source.shape_yx),
        },
        "preprocessing": applied,
        "processed_data_hash": stable_hash(result),
        "interpretation": "file_heightmap_no_random_recipe_statistics",
    }
    return result, record
