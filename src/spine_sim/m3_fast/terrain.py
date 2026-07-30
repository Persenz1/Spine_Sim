"""Read-only terrain-track loading and vectorized interpolation for fast M3.

This module intentionally depends only on NumPy and the on-disk M1 contract.
It does not import M1 terrain classes and never queries a two-dimensional
height map during an M3 case.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


_CATALOG_SCHEMAS = frozenset({"m1-material-terrain-catalog-v1"})
_TRACK_SCHEMA = "1"
_FAMILY_ORDER = ("sandpaper", "red_brick", "concrete")
_FAMILY_ALIASES = {
    "sandpaper": "sandpaper",
    "砂纸": "sandpaper",
    "red_brick": "red_brick",
    "red-brick": "red_brick",
    "brick": "red_brick",
    "红砖": "red_brick",
    "concrete": "concrete",
    "混凝土": "concrete",
}
_REQUIRED_TRACK_ARRAYS = frozenset(
    {
        "x_global_m",
        "envelope_height_m",
        "envelope_slope_x",
        "valid_mask",
    }
)
_HEX_DIGITS = frozenset("0123456789abcdef")

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.intp]


def _normalize_family(value: object) -> str:
    raw = str(value).strip().lower()
    try:
        return _FAMILY_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(
            "terrain family must identify sandpaper, red_brick, or concrete"
        ) from exc


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX_DIGITS for character in value)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return document


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _identity(kind: str, value: Mapping[str, object], module_version: str) -> str:
    payload = {
        "input": dict(value),
        "kind": kind,
        "module_version": module_version,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{kind}_{digest[:20]}"


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _require_identifier(value: object, name: str) -> str:
    identifier = str(value).strip()
    if (
        not identifier
        or identifier in {".", ".."}
        or "/" in identifier
        or "\\" in identifier
    ):
        raise ValueError(f"{name} is not a safe non-empty cache identifier")
    return identifier


def _as_float_vector(values: object, name: str, *, allow_empty: bool) -> FloatArray:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=np.float64)
    elif isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional numeric sequence")
    else:
        try:
            array = np.asarray(list(values), dtype=np.float64)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                f"{name} must be a one-dimensional numeric sequence"
            ) from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _coordinate_tolerance(resolution_m: float) -> float:
    return max(1e-12, abs(float(resolution_m)) * 1e-6)


@dataclass(frozen=True, slots=True)
class TerrainCondition:
    """Validated, lightweight view of one formal catalog condition."""

    terrain_condition_id: str
    terrain_family: str
    seed: int
    terrain_recipe_id: str
    region_id: str
    realization_id: str
    name: str
    material: str
    subtype: str
    index: int
    resolution_m: float
    data_sha256: str
    catalog_id: str
    catalog_path: Path
    library_root: Path
    m1_module_version: str

    def __post_init__(self) -> None:
        _require_identifier(self.terrain_condition_id, "terrain_condition_id")
        _require_identifier(self.terrain_recipe_id, "terrain_recipe_id")
        _require_identifier(self.region_id, "region_id")
        if self.realization_id:
            _require_identifier(self.realization_id, "realization_id")
        if self.terrain_family not in _FAMILY_ORDER:
            raise ValueError("unsupported terrain_family")
        if isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive and finite")
        if not _is_sha256(self.data_sha256):
            raise ValueError("data_sha256 must be a lowercase SHA-256 digest")

    @property
    def terrain_id(self) -> str:
        """Stable M3 terrain identifier used in case summaries."""

        return self.terrain_condition_id

    @property
    def family(self) -> str:
        return self.terrain_family


@dataclass(frozen=True, slots=True, eq=False)
class TrackBank:
    """C-contiguous track arrays for one condition and one tip radius."""

    x_global_m: FloatArray = field(repr=False)
    y_values_m: FloatArray
    envelope_height_m: FloatArray = field(repr=False)
    envelope_slope_x: FloatArray = field(repr=False)
    arc_length_m: FloatArray = field(repr=False)
    valid_mask: BoolArray = field(repr=False)
    terrain_id: str
    seed: int
    radius_m: float
    resolution_m: float
    terrain_recipe_id: str
    region_id: str
    realization_id: str
    track_ids: tuple[str, ...]
    _y_sorted_m: FloatArray = field(init=False, repr=False)
    _y_sort_order: IndexArray = field(init=False, repr=False)
    _dx_m: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x = np.ascontiguousarray(self.x_global_m, dtype=np.float64)
        y = np.ascontiguousarray(self.y_values_m, dtype=np.float64)
        height = np.ascontiguousarray(self.envelope_height_m, dtype=np.float64)
        slope = np.ascontiguousarray(self.envelope_slope_x, dtype=np.float64)
        arc = np.ascontiguousarray(self.arc_length_m, dtype=np.float64)
        valid = np.ascontiguousarray(self.valid_mask, dtype=np.bool_)

        if x.ndim != 1 or x.size < 2:
            raise ValueError("x_global_m must be one-dimensional with at least 2 nodes")
        if y.ndim != 1 or y.size == 0:
            raise ValueError("y_values_m must be a non-empty one-dimensional array")
        expected_shape = (y.size, x.size)
        for name, array in (
            ("envelope_height_m", height),
            ("envelope_slope_x", slope),
            ("arc_length_m", arc),
            ("valid_mask", valid),
        ):
            if array.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
        if len(self.track_ids) != y.size:
            raise ValueError("track_ids must contain one identifier per track row")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("track coordinates must be finite")
        if not np.all(np.isfinite(height)):
            raise ValueError("envelope_height_m must be finite")
        if not np.all(np.isfinite(arc)):
            raise ValueError("arc_length_m must be finite")
        if np.any(valid & ~np.isfinite(slope)):
            raise ValueError("valid slope samples must be finite")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive and finite")
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive and finite")

        spacing = np.diff(x)
        if np.any(spacing <= 0.0):
            raise ValueError("x_global_m must be strictly increasing")
        dx_m = float(spacing[0])
        spacing_tolerance = max(1e-15, abs(dx_m) * 1e-9)
        if not np.allclose(
            spacing,
            dx_m,
            rtol=0.0,
            atol=spacing_tolerance,
        ):
            raise ValueError("x_global_m must be a uniform public grid")
        if not math.isclose(
            dx_m,
            self.resolution_m,
            rel_tol=0.0,
            abs_tol=spacing_tolerance,
        ):
            raise ValueError("public x spacing does not match resolution_m")

        order = np.ascontiguousarray(np.argsort(y, kind="stable"), dtype=np.intp)
        sorted_y = np.ascontiguousarray(y[order], dtype=np.float64)
        if sorted_y.size > 1 and np.any(
            np.diff(sorted_y) <= _coordinate_tolerance(self.resolution_m)
        ):
            raise ValueError("TrackBank cannot contain duplicate y coordinates")

        for array in (x, y, height, slope, arc, valid, order, sorted_y):
            array.setflags(write=False)
        object.__setattr__(self, "x_global_m", x)
        object.__setattr__(self, "y_values_m", y)
        object.__setattr__(self, "envelope_height_m", height)
        object.__setattr__(self, "envelope_slope_x", slope)
        object.__setattr__(self, "arc_length_m", arc)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "_y_sort_order", order)
        object.__setattr__(self, "_y_sorted_m", sorted_y)
        object.__setattr__(self, "_dx_m", dx_m)

    @property
    def track_count(self) -> int:
        return int(self.y_values_m.size)

    @property
    def sample_count(self) -> int:
        return int(self.x_global_m.size)

    @property
    def dx_m(self) -> float:
        return self._dx_m

    def rows_for_y(
        self,
        y_values_m: Sequence[float] | FloatArray,
        *,
        atol_m: float | None = None,
    ) -> IndexArray:
        """Map requested global y coordinates to track rows without a 2-D search."""

        query = _as_float_vector(y_values_m, "y_values_m", allow_empty=True)
        if query.size == 0:
            return np.empty(0, dtype=np.intp)
        tolerance = (
            _coordinate_tolerance(self.resolution_m)
            if atol_m is None
            else float(atol_m)
        )
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("atol_m must be finite and non-negative")

        insertion = np.searchsorted(self._y_sorted_m, query, side="left")
        right = np.minimum(insertion, self.track_count - 1)
        left = np.maximum(insertion - 1, 0)
        right_distance = np.abs(self._y_sorted_m[right] - query)
        left_distance = np.abs(self._y_sorted_m[left] - query)
        use_left = left_distance <= right_distance
        nearest = np.where(use_left, left, right)
        distance = np.where(use_left, left_distance, right_distance)
        if np.any(distance > tolerance):
            missing = query[distance > tolerance]
            preview = ", ".join(f"{value:.12g}" for value in missing[:5])
            raise KeyError(f"no cached track for requested y coordinate(s): {preview}")
        rows = self._y_sort_order[nearest]
        return np.ascontiguousarray(rows, dtype=np.intp)


def load_catalog(
    path: str | Path,
    *,
    require_formal_300: bool | None = None,
) -> list[TerrainCondition]:
    """Load and validate an M1 material catalog without importing M1 classes.

    If ``require_formal_300`` is omitted, the strict 3×100 paired-seed checks
    are enabled whenever the catalog declares ``formal_300_complete=true``.
    """

    catalog_path = Path(path).expanduser().resolve()
    catalog = _read_json_object(catalog_path)
    schema_version = str(catalog.get("schema_version", ""))
    if schema_version not in _CATALOG_SCHEMAS:
        raise ValueError(f"unsupported terrain catalog schema_version {schema_version!r}")
    if catalog.get("status") != "complete":
        raise ValueError("terrain catalog status must be complete")
    if catalog.get("all_full_hashes_verified") is not True:
        raise ValueError("terrain catalog must verify every full data hash")

    raw_conditions = catalog.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("terrain catalog contains no conditions")
    declared_count = catalog.get("condition_count")
    if declared_count is not None and int(declared_count) != len(raw_conditions):
        raise ValueError("terrain catalog condition_count does not match conditions")

    raw_library_root = catalog.get("library_root")
    if not isinstance(raw_library_root, str) or not raw_library_root.strip():
        raise ValueError("terrain catalog is missing library_root")
    library_root = Path(raw_library_root).expanduser()
    if not library_root.is_absolute():
        library_root = catalog_path.parent / library_root
    library_root = library_root.resolve()

    catalog_id = str(catalog.get("terrain_catalog_id", "")).strip()
    if not catalog_id:
        raise ValueError("terrain catalog is missing terrain_catalog_id")
    module_version = str(catalog.get("m1_module_version", "m1"))
    top_resolution_m = float(catalog.get("resolution_m", math.nan))
    formal_required = (
        catalog.get("formal_300_complete") is True
        if require_formal_300 is None
        else bool(require_formal_300)
    )
    if formal_required and catalog.get("formal_300_complete") is not True:
        raise ValueError("formal M3 requires formal_300_complete=true")

    conditions: list[TerrainCondition] = []
    seen_pairings: set[tuple[str, int]] = set()
    seen_realizations: set[str] = set()
    for ordinal, raw in enumerate(raw_conditions, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"terrain condition {ordinal} must be an object")
        family = _normalize_family(raw.get("terrain_family", raw.get("family", "")))
        try:
            seed = int(raw["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"terrain condition {ordinal} has an invalid seed") from exc
        if isinstance(raw.get("seed"), bool):
            raise TypeError(f"terrain condition {ordinal} seed must be an integer")
        pairing = (family, seed)
        if pairing in seen_pairings:
            raise ValueError(f"duplicate terrain pairing {family}/seed={seed}")
        seen_pairings.add(pairing)
        if raw.get("full_sha256_verified") is not True:
            raise ValueError(f"{family}/seed={seed} does not have a verified full hash")

        recipe_id = _require_identifier(raw.get("terrain_recipe_id"), "terrain_recipe_id")
        region_id = _require_identifier(raw.get("region_id"), "region_id")
        data_sha256 = str(raw.get("data_sha256", "")).lower()
        if not _is_sha256(data_sha256):
            raise ValueError(
                f"{family}/seed={seed} data_sha256 must be a SHA-256 digest"
            )
        realization_id = str(raw.get("realization_id", "")).strip()
        if formal_required and not realization_id:
            raise ValueError("formal terrain conditions require realization_id")
        if realization_id:
            _require_identifier(realization_id, "realization_id")
            if realization_id in seen_realizations:
                raise ValueError(f"duplicate terrain realization_id {realization_id}")
            seen_realizations.add(realization_id)

        region = raw.get("region")
        resolution_m = top_resolution_m
        if isinstance(region, dict) and "resolution_x_m" in region:
            resolution_m = float(region["resolution_x_m"])
        if not math.isfinite(resolution_m) or resolution_m <= 0.0:
            raise ValueError(f"{family}/seed={seed} has invalid x resolution")

        condition_id = _identity(
            "terrain_condition",
            {
                "terrain_family": family,
                "seed": seed,
                "terrain_recipe_id": recipe_id,
                "region_id": region_id,
                "data_sha256": data_sha256,
            },
            module_version,
        )
        material = str(raw.get("material", family)).strip()
        subtype = str(raw.get("subtype", "")).strip()
        name = str(raw.get("name", f"{family}_seed_{seed}")).strip()
        index = int(raw.get("index", ordinal))
        conditions.append(
            TerrainCondition(
                terrain_condition_id=condition_id,
                terrain_family=family,
                seed=seed,
                terrain_recipe_id=recipe_id,
                region_id=region_id,
                realization_id=realization_id,
                name=name,
                material=material,
                subtype=subtype,
                index=index,
                resolution_m=resolution_m,
                data_sha256=data_sha256,
                catalog_id=catalog_id,
                catalog_path=catalog_path,
                library_root=library_root,
                m1_module_version=module_version,
            )
        )

    family_rank = {family: rank for rank, family in enumerate(_FAMILY_ORDER)}
    conditions.sort(
        key=lambda condition: (
            family_rank[condition.terrain_family],
            condition.seed,
            condition.index,
        )
    )
    if formal_required:
        counts = Counter(condition.terrain_family for condition in conditions)
        expected_counts = {family: 100 for family in _FAMILY_ORDER}
        if len(conditions) != 300 or dict(counts) != expected_counts:
            raise ValueError(
                "formal M3 requires exactly 100 conditions for each terrain family"
            )
        seed_sets = {
            family: frozenset(
                condition.seed
                for condition in conditions
                if condition.terrain_family == family
            )
            for family in _FAMILY_ORDER
        }
        if len(set(seed_sets.values())) != 1:
            raise ValueError("formal M3 terrain families must use paired seed sets")
    return conditions


def select_conditions(
    conditions: Sequence[TerrainCondition],
    *,
    terrain_family: str | Iterable[str] | None = None,
    material: str | Iterable[str] | None = None,
    subtype: str | Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    seed_min: int | None = None,
    seed_max: int | None = None,
    terrain_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[TerrainCondition]:
    """Select catalog conditions while preserving deterministic catalog order."""

    if terrain_family is not None and material is not None:
        raise ValueError("supply only one of terrain_family and material")
    family_selector = terrain_family if terrain_family is not None else material
    if seed_min is not None and seed_max is not None and seed_max < seed_min:
        raise ValueError("seed_max cannot be less than seed_min")
    if limit is not None and (isinstance(limit, bool) or limit < 0):
        raise ValueError("limit must be a non-negative integer")

    families: set[str] | None
    if family_selector is None:
        families = None
    elif isinstance(family_selector, str):
        families = {_normalize_family(family_selector)}
    else:
        families = {_normalize_family(value) for value in family_selector}

    subtypes: set[str] | None
    if subtype is None:
        subtypes = None
    elif isinstance(subtype, str):
        subtypes = {subtype}
    else:
        subtypes = {str(value) for value in subtype}
    seed_set = None if seeds is None else {int(value) for value in seeds}
    terrain_id_set = (
        None if terrain_ids is None else {str(value) for value in terrain_ids}
    )

    selected = [
        condition
        for condition in conditions
        if (families is None or condition.terrain_family in families)
        and (subtypes is None or condition.subtype in subtypes)
        and (seed_set is None or condition.seed in seed_set)
        and (seed_min is None or condition.seed >= seed_min)
        and (seed_max is None or condition.seed <= seed_max)
        and (
            terrain_id_set is None
            or condition.terrain_id in terrain_id_set
            or condition.realization_id in terrain_id_set
        )
    ]
    return selected if limit is None else selected[:limit]


@dataclass(frozen=True, slots=True)
class _TrackRecord:
    track_id: str
    y_global_m: float
    metadata_path: Path
    metadata: dict[str, object]


def _discover_track_records(
    radius_directory: Path,
    condition: TerrainCondition,
    radius_m: float,
) -> list[_TrackRecord]:
    records: list[_TrackRecord] = []
    seen_y: list[float] = []
    for metadata_path in sorted(radius_directory.glob("*.json")):
        metadata = _read_json_object(metadata_path)
        if str(metadata.get("schema_version", "")) != _TRACK_SCHEMA:
            raise ValueError(f"unsupported track schema in {metadata_path}")
        if str(metadata.get("terrain_recipe_id", "")) != condition.terrain_recipe_id:
            raise ValueError(f"track recipe identity mismatch: {metadata_path}")
        if str(metadata.get("region_id", "")) != condition.region_id:
            raise ValueError(f"track region identity mismatch: {metadata_path}")
        track_id = _require_identifier(metadata.get("track_id"), "track_id")
        if metadata_path.stem != track_id:
            raise ValueError(f"track sidecar filename does not match track_id: {metadata_path}")
        try:
            track_radius_m = float(metadata["radius_m"])
            y_global_m = float(metadata["y_global_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid track coordinates in {metadata_path}") from exc
        if not math.isfinite(y_global_m):
            raise ValueError(f"track y coordinate must be finite: {metadata_path}")
        radius_tolerance = max(1e-15, abs(radius_m) * 1e-10)
        if not math.isclose(
            track_radius_m,
            radius_m,
            rel_tol=0.0,
            abs_tol=radius_tolerance,
        ):
            raise ValueError(f"track radius does not match radius directory: {metadata_path}")
        seen_y.append(y_global_m)
        records.append(
            _TrackRecord(
                track_id=track_id,
                y_global_m=y_global_m,
                metadata_path=metadata_path,
                metadata=metadata,
            )
        )
    if not records:
        raise FileNotFoundError(f"no track sidecars found in {radius_directory}")
    order = np.argsort(np.asarray(seen_y, dtype=np.float64), kind="stable")
    sorted_y = np.asarray(seen_y, dtype=np.float64)[order]
    if sorted_y.size > 1 and np.any(
        np.diff(sorted_y) <= _coordinate_tolerance(condition.resolution_m)
    ):
        raise ValueError(f"duplicate track y coordinates in {radius_directory}")
    return records


def _select_track_records(
    records: Sequence[_TrackRecord],
    requested_y_m: FloatArray,
    tolerance_m: float,
) -> list[_TrackRecord]:
    available_y = np.asarray(
        [record.y_global_m for record in records],
        dtype=np.float64,
    )
    order = np.argsort(available_y, kind="stable")
    sorted_y = available_y[order]
    insertion = np.searchsorted(sorted_y, requested_y_m, side="left")
    right = np.minimum(insertion, sorted_y.size - 1)
    left = np.maximum(insertion - 1, 0)
    right_distance = np.abs(sorted_y[right] - requested_y_m)
    left_distance = np.abs(sorted_y[left] - requested_y_m)
    use_left = left_distance <= right_distance
    nearest = np.where(use_left, left, right)
    distance = np.where(use_left, left_distance, right_distance)
    if np.any(distance > tolerance_m):
        missing = requested_y_m[distance > tolerance_m]
        preview = ", ".join(f"{value:.12g}" for value in missing[:5])
        raise FileNotFoundError(f"no complete cached track for y={preview}")
    source_indices = order[nearest]
    return [records[int(index)] for index in source_indices]


def _load_track_arrays(
    record: _TrackRecord,
    condition: TerrainCondition,
    radius_m: float,
    *,
    verify_hash: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray, float]:
    metadata = record.metadata
    data_path = record.metadata_path.with_suffix(".npz")
    complete_path = record.metadata_path.with_suffix(".complete")
    if not data_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError(f"track cache is incomplete: {record.track_id}")

    data_sha256 = str(metadata.get("data_sha256", "")).lower()
    if not _is_sha256(data_sha256):
        raise ValueError(f"invalid data_sha256 in {record.metadata_path}")
    try:
        marker = complete_path.read_text(encoding="ascii").strip().lower()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read track completion marker: {complete_path}") from exc
    if marker != data_sha256:
        raise ValueError(f"track completion marker does not match sidecar: {record.track_id}")
    if verify_hash and _sha256_file(data_path) != marker:
        raise ValueError(f"track NPZ hash verification failed: {record.track_id}")

    resolution_m = float(metadata.get("resolution_m", math.nan))
    resolution_tolerance = max(1e-15, condition.resolution_m * 1e-9)
    if not math.isclose(
        resolution_m,
        condition.resolution_m,
        rel_tol=0.0,
        abs_tol=resolution_tolerance,
    ):
        raise ValueError(f"track resolution does not match catalog: {record.track_id}")
    try:
        with np.load(data_path, allow_pickle=False) as archive:
            missing = _REQUIRED_TRACK_ARRAYS.difference(archive.files)
            if missing:
                raise ValueError(
                    f"track NPZ {record.track_id} is missing arrays {sorted(missing)}"
                )
            x = np.ascontiguousarray(archive["x_global_m"], dtype=np.float64)
            height = np.ascontiguousarray(
                archive["envelope_height_m"],
                dtype=np.float64,
            )
            slope = np.ascontiguousarray(
                archive["envelope_slope_x"],
                dtype=np.float64,
            )
            raw_valid = archive["valid_mask"]
            if raw_valid.dtype != np.bool_:
                raise ValueError(f"track valid_mask must be boolean: {record.track_id}")
            valid = np.ascontiguousarray(raw_valid, dtype=np.bool_)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("track "):
            raise
        raise ValueError(f"cannot load track NPZ {data_path}") from exc

    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"track x array is not a usable vector: {record.track_id}")
    if height.shape != x.shape or slope.shape != x.shape or valid.shape != x.shape:
        raise ValueError(f"track arrays do not share one public x shape: {record.track_id}")
    if int(metadata.get("sample_count", -1)) != x.size:
        raise ValueError(f"track sample_count does not match NPZ: {record.track_id}")
    if int(metadata.get("valid_count", -1)) != int(np.count_nonzero(valid)):
        raise ValueError(f"track valid_count does not match NPZ: {record.track_id}")
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError(f"track x grid must be finite and increasing: {record.track_id}")
    if not np.all(np.isfinite(height)):
        raise ValueError(f"track envelope height must be finite: {record.track_id}")
    if np.any(valid & ~np.isfinite(slope)):
        raise ValueError(f"valid track slope must be finite: {record.track_id}")
    spacing = np.diff(x)
    spacing_tolerance = max(1e-15, resolution_m * 1e-9)
    if not np.allclose(
        spacing,
        resolution_m,
        rtol=0.0,
        atol=spacing_tolerance,
    ):
        raise ValueError(f"track x grid is not uniformly spaced: {record.track_id}")
    return x, height, slope, valid, resolution_m


def _unique_requested_y(values: FloatArray, tolerance_m: float) -> FloatArray:
    unique: list[float] = []
    for value in values:
        if not any(abs(float(value) - existing) <= tolerance_m for existing in unique):
            unique.append(float(value))
    return np.ascontiguousarray(unique, dtype=np.float64)


def _derive_arc_length(x_global_m: FloatArray, height_m: FloatArray) -> FloatArray:
    arc_length_m = np.empty_like(height_m, dtype=np.float64, order="C")
    arc_length_m[:, 0] = 0.0
    np.hypot(
        np.diff(x_global_m)[None, :],
        np.diff(height_m, axis=1),
        out=arc_length_m[:, 1:],
    )
    np.cumsum(arc_length_m[:, 1:], axis=1, out=arc_length_m[:, 1:])
    return arc_length_m


def load_track_bank(
    library_root: str | Path | None,
    condition: TerrainCondition,
    radius_m: float,
    y_values_m: Sequence[float] | FloatArray,
    *,
    verify_hash: bool = True,
    coordinate_tolerance_m: float | None = None,
) -> TrackBank:
    """Load requested same-radius tracks and stack them into C-contiguous arrays."""

    if not isinstance(condition, TerrainCondition):
        raise TypeError("condition must be a TerrainCondition returned by load_catalog")
    radius_m = float(radius_m)
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    requested_y = _as_float_vector(y_values_m, "y_values_m", allow_empty=False)
    tolerance_m = (
        _coordinate_tolerance(condition.resolution_m)
        if coordinate_tolerance_m is None
        else float(coordinate_tolerance_m)
    )
    if not math.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("coordinate_tolerance_m must be finite and non-negative")
    unique_y = _unique_requested_y(requested_y, tolerance_m)

    root = condition.library_root if library_root is None else Path(library_root)
    root = root.expanduser().resolve()
    recipe_id = _require_identifier(condition.terrain_recipe_id, "terrain_recipe_id")
    region_id = _require_identifier(condition.region_id, "region_id")
    radius_um = int(round(radius_m * 1e6))
    radius_directory = (
        root / "tracks" / recipe_id / region_id / f"{radius_um}um"
    )
    if not radius_directory.is_dir():
        raise FileNotFoundError(f"track radius directory does not exist: {radius_directory}")

    records = _discover_track_records(radius_directory, condition, radius_m)
    selected = _select_track_records(records, unique_y, tolerance_m)
    height_rows: list[FloatArray] = []
    slope_rows: list[FloatArray] = []
    valid_rows: list[BoolArray] = []
    actual_y: list[float] = []
    track_ids: list[str] = []
    resolution_m = condition.resolution_m
    public_x: FloatArray | None = None
    for record in selected:
        x, height, slope, valid, resolution_m = _load_track_arrays(
            record,
            condition,
            radius_m,
            verify_hash=verify_hash,
        )
        if public_x is None:
            public_x = x
        elif not np.array_equal(x, public_x):
            raise ValueError(
                f"tracks do not share an identical public x grid: {record.track_id}"
            )
        height_rows.append(height)
        slope_rows.append(slope)
        valid_rows.append(valid)
        actual_y.append(record.y_global_m)
        track_ids.append(record.track_id)
    if public_x is None:
        raise RuntimeError("no track was selected")

    height_bank = np.ascontiguousarray(np.stack(height_rows, axis=0), dtype=np.float64)
    slope_bank = np.ascontiguousarray(np.stack(slope_rows, axis=0), dtype=np.float64)
    valid_bank = np.ascontiguousarray(np.stack(valid_rows, axis=0), dtype=np.bool_)
    arc_bank = _derive_arc_length(public_x, height_bank)
    return TrackBank(
        x_global_m=np.ascontiguousarray(public_x, dtype=np.float64),
        y_values_m=np.ascontiguousarray(actual_y, dtype=np.float64),
        envelope_height_m=height_bank,
        envelope_slope_x=slope_bank,
        arc_length_m=arc_bank,
        valid_mask=valid_bank,
        terrain_id=condition.terrain_id,
        seed=condition.seed,
        radius_m=radius_m,
        resolution_m=resolution_m,
        terrain_recipe_id=condition.terrain_recipe_id,
        region_id=condition.region_id,
        realization_id=condition.realization_id,
        track_ids=tuple(track_ids),
    )


def _prepare_output(
    output: NDArray[np.generic] | None,
    size: int,
    dtype: np.dtype[np.generic],
    name: str,
) -> NDArray[np.generic]:
    if output is None:
        return np.empty(size, dtype=dtype)
    if not isinstance(output, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if output.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}")
    if output.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not output.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not output.flags.writeable:
        raise ValueError(f"{name} must be writeable")
    return output


def interpolate_tracks(
    bank: TrackBank,
    track_rows: Sequence[int] | IndexArray,
    x_query_m: float | Sequence[float] | FloatArray,
    *,
    out_height: FloatArray | None = None,
    out_slope: FloatArray | None = None,
    out_arc_length: FloatArray | None = None,
    out_valid: BoolArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
    """Linearly interpolate all requested tracks on their one shared x grid."""

    if not isinstance(bank, TrackBank):
        raise TypeError("bank must be a TrackBank")
    raw_rows = np.asarray(track_rows)
    if raw_rows.ndim != 1:
        raise ValueError("track_rows must be one-dimensional")
    if raw_rows.dtype == np.bool_ or not np.issubdtype(raw_rows.dtype, np.integer):
        raise TypeError("track_rows must contain integers")
    rows = np.ascontiguousarray(raw_rows, dtype=np.intp)
    size = int(rows.size)
    if np.any(rows < 0) or np.any(rows >= bank.track_count):
        raise IndexError("track_rows contains an out-of-range row")

    raw_query = np.asarray(x_query_m, dtype=np.float64)
    if raw_query.ndim == 0:
        query = np.full(size, float(raw_query), dtype=np.float64)
    elif raw_query.shape == (size,):
        query = np.ascontiguousarray(raw_query, dtype=np.float64)
    else:
        raise ValueError("x_query_m must be scalar or match track_rows")

    float_dtype = np.dtype(np.float64)
    bool_dtype = np.dtype(np.bool_)
    height = _prepare_output(out_height, size, float_dtype, "out_height")
    slope = _prepare_output(out_slope, size, float_dtype, "out_slope")
    arc = _prepare_output(
        out_arc_length,
        size,
        float_dtype,
        "out_arc_length",
    )
    valid = _prepare_output(out_valid, size, bool_dtype, "out_valid")
    float_outputs = (height, slope, arc)
    if any(
        np.shares_memory(left, right)
        for index, left in enumerate(float_outputs)
        for right in float_outputs[index + 1 :]
    ):
        raise ValueError("floating interpolation outputs cannot overlap")

    height.fill(np.nan)
    slope.fill(np.nan)
    arc.fill(np.nan)
    valid.fill(False)
    if size == 0:
        return height, slope, arc, valid  # type: ignore[return-value]

    finite_query = np.isfinite(query)
    domain_tolerance_m = max(1e-15, bank.dx_m * 1e-9)
    inside = (
        finite_query
        & (query >= bank.x_global_m[0] - domain_tolerance_m)
        & (query <= bank.x_global_m[-1] + domain_tolerance_m)
    )
    safe_query = np.where(finite_query, query, bank.x_global_m[0])
    safe_query = np.clip(
        safe_query,
        bank.x_global_m[0],
        bank.x_global_m[-1],
    )
    grid_position = (safe_query - bank.x_global_m[0]) / bank.dx_m
    nearest_node = np.rint(grid_position)
    snap_to_node = np.abs(grid_position - nearest_node) <= 1e-9
    grid_position = np.where(snap_to_node, nearest_node, grid_position)
    left = np.floor(grid_position).astype(np.intp)
    right = np.ceil(grid_position).astype(np.intp)
    np.clip(left, 0, bank.sample_count - 1, out=left)
    np.clip(right, 0, bank.sample_count - 1, out=right)
    weight = grid_position - left

    left_height = bank.envelope_height_m[rows, left]
    np.subtract(
        bank.envelope_height_m[rows, right],
        left_height,
        out=height,
    )
    np.multiply(height, weight, out=height)
    np.add(height, left_height, out=height)

    left_slope = bank.envelope_slope_x[rows, left]
    np.subtract(
        bank.envelope_slope_x[rows, right],
        left_slope,
        out=slope,
    )
    np.multiply(slope, weight, out=slope)
    np.add(slope, left_slope, out=slope)

    left_arc = bank.arc_length_m[rows, left]
    np.subtract(bank.arc_length_m[rows, right], left_arc, out=arc)
    np.multiply(arc, weight, out=arc)
    np.add(arc, left_arc, out=arc)

    np.logical_and(
        bank.valid_mask[rows, left],
        bank.valid_mask[rows, right],
        out=valid,
    )
    np.logical_and(valid, inside, out=valid)
    np.logical_and(valid, np.isfinite(height), out=valid)
    np.logical_and(valid, np.isfinite(slope), out=valid)
    np.logical_and(valid, np.isfinite(arc), out=valid)
    height[~valid] = np.nan
    slope[~valid] = np.nan
    arc[~valid] = np.nan
    return height, slope, arc, valid  # type: ignore[return-value]


__all__ = [
    "TerrainCondition",
    "TrackBank",
    "interpolate_tracks",
    "load_catalog",
    "load_track_bank",
    "select_conditions",
]
