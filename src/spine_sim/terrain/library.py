"""Atomic, rebuildable local terrain library with read-only memory maps."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from spine_sim.io.results import atomic_write_bytes, atomic_write_json, atomic_write_npz, utc_now

from .envelope import compute_track_geometry
from .errors import TerrainConfigurationError
from .models import M1_MODULE_VERSION, RegionSpec, TerrainRecipe, TrackGeometry
from .random_field import generate_canonical_window, gaussian_kernel

if TYPE_CHECKING:
    from .api import Terrain


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _safe_id(value: str, name: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or any(character in value for character in "\\/:")
    ):
        raise TerrainConfigurationError(f"invalid {name}")
    return value


class TerrainLibrary:
    """Manage recipes, raw regions, tracks, manifests and validation artifacts."""

    DIRECTORIES = (
        "recipes",
        "sources",
        "regions",
        "tracks",
        "manifests",
        "validation",
    )

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        for name in self.DIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def recipe_path(self, terrain_recipe_id: str) -> Path:
        return self.root / "recipes" / f"{_safe_id(terrain_recipe_id, 'terrain_recipe_id')}.json"

    def region_dir(self, terrain_recipe_id: str, region_id: str) -> Path:
        return (
            self.root
            / "regions"
            / _safe_id(terrain_recipe_id, "terrain_recipe_id")
            / _safe_id(region_id, "region_id")
        )

    def region_data_path(self, terrain_recipe_id: str, region_id: str) -> Path:
        return self.region_dir(terrain_recipe_id, region_id) / "raw_height.npy"

    def region_manifest_path(self, terrain_recipe_id: str, region_id: str) -> Path:
        return (
            self.root
            / "manifests"
            / "regions"
            / _safe_id(terrain_recipe_id, "terrain_recipe_id")
            / f"{_safe_id(region_id, 'region_id')}.json"
        )

    def track_path(
        self,
        terrain_recipe_id: str,
        region_id: str,
        radius_m: float,
        track_id: str,
    ) -> Path:
        radius_um = int(round(radius_m * 1e6))
        return (
            self.root
            / "tracks"
            / _safe_id(terrain_recipe_id, "terrain_recipe_id")
            / _safe_id(region_id, "region_id")
            / f"{radius_um}um"
            / f"{_safe_id(track_id, 'track_id')}.npz"
        )

    def save_recipe(self, recipe: TerrainRecipe) -> Path:
        path = self.recipe_path(recipe.terrain_recipe_id)
        document = {
            "schema_version": "1",
            "m1_module_version": M1_MODULE_VERSION,
            "terrain_recipe_id": recipe.terrain_recipe_id,
            "recipe_hash": recipe.recipe_hash,
            "recipe": recipe.normalized(),
            "kernel_definition": recipe.kernel_definition,
            "production_sampling": recipe.production_sampling,
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != document:
                raise TerrainConfigurationError(
                    "recipe ID already exists with different content"
                )
            return path
        atomic_write_json(path, document)
        return path

    def load_recipe(self, terrain_recipe_id: str) -> TerrainRecipe:
        document = json.loads(
            self.recipe_path(terrain_recipe_id).read_text(encoding="utf-8")
        )
        recipe = TerrainRecipe.from_mapping(document["recipe"])
        if recipe.terrain_recipe_id != terrain_recipe_id:
            raise TerrainConfigurationError("stored recipe identity does not match content")
        if recipe.recipe_hash != document["recipe_hash"]:
            raise TerrainConfigurationError("stored recipe hash does not match content")
        return recipe

    def _write_region_tiles(
        self,
        target: Path,
        recipe: TerrainRecipe,
        region: RegionSpec,
        *,
        tile_rows: int,
        backend: Literal["cpu", "cuda"],
    ) -> dict[str, Any]:
        if tile_rows < 1:
            raise TerrainConfigurationError("tile_rows must be positive")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        started = time.perf_counter()
        peak_working_bytes = 0
        try:
            ny, nx = region.shape
            output = np.lib.format.open_memmap(
                temporary_path, mode="w+", dtype=np.float32, shape=(ny, nx)
            )
            start_x, start_y = recipe.canonical_indices(
                region.origin_x_m, region.origin_y_m
            )
            canonical_x = math.isclose(
                region.resolution_x_m,
                recipe.canonical_dx_m,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            canonical_y = math.isclose(
                region.resolution_y_m,
                recipe.canonical_dy_m,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            production_x = math.isclose(
                region.resolution_x_m,
                recipe.production_dx_m,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            production_y = math.isclose(
                region.resolution_y_m,
                recipe.production_dy_m,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            if canonical_x and canonical_y:
                stride = 1
            elif production_x and production_y:
                stride = 2
            else:
                raise TerrainConfigurationError(
                    "x and y must both use canonical spacing or both production spacing"
                )

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
            for row_start in range(0, ny, tile_rows):
                row_stop = min(ny, row_start + tile_rows)
                output_rows = row_stop - row_start
                canonical_rows = stride * (output_rows - 1) + 1
                canonical_columns = stride * (nx - 1) + 1
                tile = generate_canonical_window(
                    recipe,
                    start_x_index=start_x,
                    start_y_index=start_y + stride * row_start,
                    count_x=canonical_columns,
                    count_y=canonical_rows,
                    backend=backend,
                )
                sampled = tile[::stride, ::stride]
                output[row_start:row_stop, :] = sampled.astype(
                    np.float32, copy=False
                )
                float64_bytes = np.dtype(np.float64).itemsize
                float32_bytes = np.dtype(np.float32).itemsize
                extended_rows = canonical_rows + kernel_y.size - 1
                extended_columns = canonical_columns + kernel_x.size - 1
                noise_bytes = (
                    extended_rows
                    * extended_columns
                    * float64_bytes
                )
                horizontal_bytes = (
                    extended_rows * canonical_columns * float64_bytes
                )
                filtered_bytes = (
                    canonical_rows * canonical_columns * float64_bytes
                )
                cast_bytes = (
                    output_rows
                    * nx
                    * float32_bytes
                )
                working_bytes = max(
                    noise_bytes + 2 * horizontal_bytes,
                    horizontal_bytes + 2 * filtered_bytes,
                    filtered_bytes + cast_bytes,
                )
                peak_working_bytes = max(peak_working_bytes, working_bytes)
                del sampled, tile
            output.flush()
            del output
            data_hash = sha256_file(temporary_path)
            file_size = temporary_path.stat().st_size
            os.replace(temporary_path, target)
            return {
                "generation_time_s": time.perf_counter() - started,
                "tile_rows": tile_rows,
                "estimated_tile_peak_bytes": peak_working_bytes,
                "data_sha256": data_hash,
                "file_size_bytes": file_size,
            }
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def generate_region(
        self,
        recipe: TerrainRecipe,
        region: RegionSpec,
        *,
        tile_rows: int = 64,
        backend: Literal["cpu", "cuda"] = "cpu",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Atomically generate a raw region; the COMPLETE marker is written last."""

        region.validate_against(recipe)
        if recipe.generator_name == "material_hybrid":
            from .api import generate_terrain

            terrain = generate_terrain(
                material=str(recipe.material),
                subtype=str(recipe.subtype),
                size_x_m=region.size_x_m,
                size_y_m=region.size_y_m,
                resolution_m=region.resolution_x_m,
                seed=recipe.seed,
                mode=str(recipe.generation_mode),  # type: ignore[arg-type]
                backend=backend,
            )
            if terrain.metadata["profile_hash"] != recipe.profile_hash:
                raise TerrainConfigurationError(
                    "material profile changed; refusing non-reproducible rebuild"
                )
            return self.register_material_region(
                recipe, region, terrain, overwrite=overwrite
            )
        self.save_recipe(recipe)
        directory = self.region_dir(recipe.terrain_recipe_id, region.region_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "raw_height.npy"
        metadata_path = directory / "metadata.json"
        complete_path = directory / "COMPLETE"
        if complete_path.is_file() and not overwrite:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        if complete_path.exists():
            complete_path.unlink()

        if backend == "cuda":
            try:
                import cupy as cp  # type: ignore
            except ImportError as exc:
                raise TerrainConfigurationError(
                    "CUDA generation requested but CuPy is not installed"
                ) from exc
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        metrics = self._write_region_tiles(
            target,
            recipe,
            region,
            tile_rows=tile_rows,
            backend=backend,
        )
        if backend == "cuda":
            cp.cuda.Device().synchronize()
            device_pool = cp.get_default_memory_pool()
            pinned_pool = cp.get_default_pinned_memory_pool()
            metrics.update(
                {
                    "gpu_memory_pool_used_bytes": int(device_pool.used_bytes()),
                    "gpu_memory_pool_peak_cached_bytes": int(device_pool.total_bytes()),
                    "gpu_pinned_pool_cached_blocks": int(pinned_pool.n_free_blocks()),
                }
            )
            device_pool.free_all_blocks()
            pinned_pool.free_all_blocks()
        metadata = {
            "schema_version": "1",
            "m1_module_version": M1_MODULE_VERSION,
            "terrain_recipe_id": recipe.terrain_recipe_id,
            "recipe_hash": recipe.recipe_hash,
            "region_id": region.region_id,
            "region": region.normalized(),
            "shape": list(region.shape),
            "dtype": "float32",
            "coordinate_storage": "origin_spacing_shape_only_no_meshgrid",
            "generation_backend": backend,
            "production_sampling": "canonical_even_indices_stride2_nodal",
            "created_at_utc": utc_now(),
            **metrics,
        }
        atomic_write_json(metadata_path, metadata)
        atomic_write_json(
            self.region_manifest_path(recipe.terrain_recipe_id, region.region_id),
            metadata,
        )
        atomic_write_bytes(
            complete_path, (metadata["data_sha256"] + "\n").encode("ascii")
        )
        return metadata

    def register_material_region(
        self,
        recipe: TerrainRecipe,
        region: RegionSpec,
        terrain: "Terrain",
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Atomically store a material terrain without changing M3's mmap contract."""

        if recipe.generator_name != "material_hybrid":
            raise TerrainConfigurationError(
                "register_material_region requires a material_hybrid recipe"
            )
        region.validate_against(recipe)
        if (
            terrain.material != recipe.material
            or terrain.subtype != recipe.subtype
            or terrain.seed != recipe.seed
            or terrain.resolved_mode != recipe.generation_mode
            or terrain.metadata.get("profile_hash") != recipe.profile_hash
        ):
            raise TerrainConfigurationError(
                "material terrain identity does not match its recipe"
            )
        if terrain.height.shape != region.shape:
            raise TerrainConfigurationError(
                "material terrain shape does not match region"
            )
        if terrain.height.dtype != np.float32 or not np.all(
            np.isfinite(terrain.height)
        ):
            raise TerrainConfigurationError(
                "material terrain must be finite float32"
            )
        self.save_recipe(recipe)
        directory = self.region_dir(recipe.terrain_recipe_id, region.region_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "raw_height.npy"
        mask_target = directory / "valid_mask.npy"
        metadata_path = directory / "metadata.json"
        complete_path = directory / "COMPLETE"
        if complete_path.is_file() and not overwrite:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        if complete_path.exists():
            complete_path.unlink()

        started = time.perf_counter()
        temporary_paths: list[Path] = []
        try:
            for destination, array in (
                (target, terrain.height),
                (mask_target, terrain.valid_mask),
            ):
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=directory,
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                temporary_paths.append(temporary)
                with temporary.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                os.replace(temporary, destination)
                temporary_paths.remove(temporary)
            data_hash = sha256_file(target)
            mask_hash = sha256_file(mask_target)
            metadata = {
                "schema_version": "1",
                "m1_module_version": M1_MODULE_VERSION,
                "terrain_recipe_id": recipe.terrain_recipe_id,
                "recipe_hash": recipe.recipe_hash,
                "region_id": region.region_id,
                "region": region.normalized(),
                "shape": list(region.shape),
                "dtype": "float32",
                "coordinate_storage": "origin_spacing_shape_only_no_meshgrid",
                "generation_backend": str(
                    terrain.metadata["generation_backend"]["resolved"]
                ),
                "backend": dict(terrain.metadata["generation_backend"]),
                "production_sampling": recipe.production_sampling,
                "material": terrain.material,
                "subtype": terrain.subtype,
                "seed": terrain.seed,
                "valid_mask_file": mask_target.name,
                "valid_mask_sha256": mask_hash,
                "valid_fraction": float(np.mean(terrain.valid_mask)),
                "material_metadata": dict(terrain.metadata),
                "created_at_utc": utc_now(),
                "generation_time_s": time.perf_counter() - started,
                "data_sha256": data_hash,
                "file_size_bytes": target.stat().st_size,
            }
            atomic_write_json(metadata_path, metadata)
            atomic_write_json(
                self.region_manifest_path(recipe.terrain_recipe_id, region.region_id),
                metadata,
            )
            atomic_write_bytes(complete_path, (data_hash + "\n").encode("ascii"))
            return metadata
        finally:
            for temporary in temporary_paths:
                if temporary.exists():
                    temporary.unlink()

    def open_region(
        self,
        terrain_recipe_id: str,
        region_id: str,
        *,
        verify_hash: bool = False,
    ) -> np.memmap:
        directory = self.region_dir(terrain_recipe_id, region_id)
        complete = directory / "COMPLETE"
        metadata_path = directory / "metadata.json"
        data_path = directory / "raw_height.npy"
        if not (complete.is_file() and metadata_path.is_file() and data_path.is_file()):
            raise FileNotFoundError(f"terrain region is absent or incomplete: {region_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("terrain_recipe_id") != terrain_recipe_id
            or metadata.get("region_id") != region_id
        ):
            raise TerrainConfigurationError(
                "region path identity does not match metadata"
            )
        marker_hash = complete.read_text(encoding="ascii").strip()
        if marker_hash != metadata.get("data_sha256"):
            raise TerrainConfigurationError("region COMPLETE marker does not match metadata")
        if verify_hash and sha256_file(data_path) != marker_hash:
            raise TerrainConfigurationError("region data hash verification failed")
        height = np.load(data_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(height, np.memmap):
            raise TerrainConfigurationError("region did not open as a memory map")
        if height.dtype != np.float32 or list(height.shape) != metadata["shape"]:
            raise TerrainConfigurationError("region array shape/dtype does not match metadata")
        return height

    def load_region_spec(self, terrain_recipe_id: str, region_id: str) -> RegionSpec:
        manifest = json.loads(
            self.region_manifest_path(terrain_recipe_id, region_id).read_text(
                encoding="utf-8"
            )
        )
        region = RegionSpec.from_mapping(manifest["region"])
        if region.region_id != region_id:
            raise TerrainConfigurationError("stored region identity does not match content")
        return region

    def rebuild_region(
        self,
        terrain_recipe_id: str,
        region_id: str,
        *,
        tile_rows: int = 64,
        backend: Literal["cpu", "cuda"] = "cpu",
    ) -> dict[str, Any]:
        recipe = self.load_recipe(terrain_recipe_id)
        region = self.load_region_spec(terrain_recipe_id, region_id)
        return self.generate_region(
            recipe,
            region,
            tile_rows=tile_rows,
            backend=backend,
            overwrite=True,
        )

    def delete_region_cache(
        self,
        terrain_recipe_id: str,
        region_id: str,
        *,
        include_tracks: bool = True,
    ) -> dict[str, Any]:
        """Delete rebuildable data while retaining the recipe and region manifest."""

        region_directory = self.region_dir(terrain_recipe_id, region_id)
        tracks_directory = (
            self.root
            / "tracks"
            / _safe_id(terrain_recipe_id, "terrain_recipe_id")
            / _safe_id(region_id, "region_id")
        )
        removed: list[str] = []
        if region_directory.exists():
            try:
                shutil.rmtree(region_directory)
            except PermissionError as exc:
                raise TerrainConfigurationError(
                    "cannot delete region cache while a memory map is still open; "
                    "release downstream region handles and retry"
                ) from exc
            removed.append(str(region_directory))
        if include_tracks and tracks_directory.exists():
            try:
                shutil.rmtree(tracks_directory)
            except PermissionError as exc:
                raise TerrainConfigurationError(
                    "cannot delete track cache while a consumer still has a file open"
                ) from exc
            removed.append(str(tracks_directory))
        return {
            "removed": removed,
            "recoverable": True,
            "retained_recipe": str(self.recipe_path(terrain_recipe_id)),
            "retained_manifest": str(
                self.region_manifest_path(terrain_recipe_id, region_id)
            ),
        }

    def delete_track_caches(
        self, terrain_recipe_id: str, region_id: str
    ) -> dict[str, Any]:
        """Delete only rebuildable tracks for one region."""

        tracks_directory = (
            self.root
            / "tracks"
            / _safe_id(terrain_recipe_id, "terrain_recipe_id")
            / _safe_id(region_id, "region_id")
        )
        removed: list[str] = []
        if tracks_directory.exists():
            try:
                shutil.rmtree(tracks_directory)
            except PermissionError as exc:
                raise TerrainConfigurationError(
                    "cannot delete track cache while a consumer still has a file open"
                ) from exc
            removed.append(str(tracks_directory))
        return {"removed": removed, "recoverable": True}

    def cache_track(
        self,
        recipe: TerrainRecipe,
        region: RegionSpec,
        *,
        radius_m: float,
        y_global_m: float,
        near_tie_tolerance_m: float = 1e-10,
        overwrite: bool = False,
    ) -> TrackGeometry:
        track_id = TrackGeometry.make_id(
            terrain_recipe_id=recipe.terrain_recipe_id,
            region_id=region.region_id,
            radius_m=radius_m,
            y_global_m=y_global_m,
            envelope_algorithm_version="finite-sphere-envelope-v1",
            resolution_m=region.resolution_x_m,
        )
        path = self.track_path(
            recipe.terrain_recipe_id, region.region_id, radius_m, track_id
        )
        metadata_path = path.with_suffix(".json")
        complete_path = path.with_suffix(".complete")
        if complete_path.is_file() and not overwrite:
            return self.load_track(
                recipe.terrain_recipe_id, region.region_id, radius_m, track_id
            )
        height = self.open_region(recipe.terrain_recipe_id, region.region_id)
        started = time.perf_counter()
        try:
            track = compute_track_geometry(
                height,
                region,
                radius_m=radius_m,
                y_global_m=y_global_m,
                near_tie_tolerance_m=near_tie_tolerance_m,
            )
        finally:
            height._mmap.close()
        if track.track_id != track_id:
            raise TerrainConfigurationError("computed track ID differs from cache key")
        arrays = {
            "x_global_m": track.x_global_m,
            "envelope_height_m": track.envelope_height_m,
            "envelope_slope_x": track.envelope_slope_x,
            "support_x_m": track.support_x_m,
            "support_y_m": track.support_y_m,
            "valid_mask": track.valid_mask,
            "near_tie_flag": track.near_tie_flag,
        }
        atomic_write_npz(path, arrays)
        metadata = {
            "schema_version": "1",
            "m1_module_version": M1_MODULE_VERSION,
            "terrain_recipe_id": track.terrain_recipe_id,
            "region_id": track.region_id,
            "track_id": track.track_id,
            "radius_m": track.radius_m,
            "y_global_m": track.y_global_m,
            "resolution_m": track.resolution_m,
            "envelope_algorithm_version": track.envelope_algorithm_version,
            "model_warning": list(track.model_warning),
            "sample_count": int(track.x_global_m.size),
            "valid_count": int(np.count_nonzero(track.valid_mask)),
            "generation_time_s": time.perf_counter() - started,
            "data_sha256": sha256_file(path),
            "created_at_utc": utc_now(),
        }
        atomic_write_json(metadata_path, metadata)
        atomic_write_bytes(
            complete_path, (metadata["data_sha256"] + "\n").encode("ascii")
        )
        return track

    def load_track(
        self,
        terrain_recipe_id: str,
        region_id: str,
        radius_m: float,
        track_id: str,
    ) -> TrackGeometry:
        path = self.track_path(
            terrain_recipe_id, region_id, radius_m, track_id
        )
        metadata_path = path.with_suffix(".json")
        complete_path = path.with_suffix(".complete")
        if not (path.is_file() and metadata_path.is_file() and complete_path.is_file()):
            raise FileNotFoundError(f"track is absent or incomplete: {track_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        marker_hash = complete_path.read_text(encoding="ascii").strip()
        if marker_hash != metadata["data_sha256"]:
            raise TerrainConfigurationError("track COMPLETE marker does not match metadata")
        if sha256_file(path) != marker_hash:
            raise TerrainConfigurationError("track data hash verification failed")
        expected_id = TrackGeometry.make_id(
            terrain_recipe_id=metadata["terrain_recipe_id"],
            region_id=metadata["region_id"],
            radius_m=float(metadata["radius_m"]),
            y_global_m=float(metadata["y_global_m"]),
            envelope_algorithm_version=metadata["envelope_algorithm_version"],
            resolution_m=float(metadata["resolution_m"]),
        )
        if (
            metadata["terrain_recipe_id"] != terrain_recipe_id
            or metadata["region_id"] != region_id
            or metadata["track_id"] != track_id
            or expected_id != track_id
        ):
            raise TerrainConfigurationError(
                "track path identity does not match metadata/cache key"
            )
        with np.load(path, allow_pickle=False) as arrays:
            copied = {name: arrays[name] for name in arrays.files}
        return TrackGeometry(
            terrain_recipe_id=metadata["terrain_recipe_id"],
            region_id=metadata["region_id"],
            track_id=metadata["track_id"],
            radius_m=float(metadata["radius_m"]),
            y_global_m=float(metadata["y_global_m"]),
            resolution_m=float(metadata["resolution_m"]),
            envelope_algorithm_version=metadata["envelope_algorithm_version"],
            x_global_m=np.asarray(copied["x_global_m"], dtype=np.float64),
            envelope_height_m=np.asarray(
                copied["envelope_height_m"], dtype=np.float64
            ),
            envelope_slope_x=np.asarray(
                copied["envelope_slope_x"], dtype=np.float64
            ),
            support_x_m=np.asarray(copied["support_x_m"], dtype=np.float64),
            support_y_m=np.asarray(copied["support_y_m"], dtype=np.float64),
            valid_mask=np.asarray(copied["valid_mask"], dtype=np.bool_),
            near_tie_flag=np.asarray(copied["near_tie_flag"], dtype=np.bool_),
            model_warning=tuple(metadata.get("model_warning", ())),
        )
