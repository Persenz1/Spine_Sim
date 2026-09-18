"""原子、可重建并通过只读 memory map 访问的本地地形库。"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from spine_sim.core.identity import stable_hash
from spine_sim.io.files import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    utc_now,
)

from .envelope import array_sha256, compute_track_geometry
from .errors import TerrainConfigurationError
from .models import (
    ENVELOPE_ALGORITHM_VERSION,
    M1_MODULE_VERSION,
    TRACK_SCHEMA_VERSION,
    RegionSpec,
    TerrainRecipe,
    TrackGeometry,
)
from .random_field import generate_canonical_window, gaussian_kernel

if TYPE_CHECKING:
    from .api import Terrain


def _defined_geometry_measurement_semantics() -> dict[str, Any]:
    """返回解析/随机定义地形可安全推断的非测量语义。"""

    return {
        "status": "not_applicable",
        "probe": None,
        "measurement_tolerance_m": None,
        "determinate_mask": False,
        "bounds": False,
        "surface_model": "single_valued_height_field_2_5d",
        "general_mesh_scope": "OUT_OF_SCOPE",
    }


def _infer_legacy_region_measurement_semantics(
    recipe: TerrainRecipe,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """只为身份闭合的非测量旧 region 推断规范测量语义。"""

    if recipe.generator_name == "defined_geometry":
        return _defined_geometry_measurement_semantics()
    if recipe.generator_name != "material_hybrid":
        return None
    material_metadata = metadata.get("material_metadata")
    if not isinstance(material_metadata, dict):
        return None
    # synthetic 输出不携带探针、反卷积或测量公差语义；只有材料、seed、
    # profile 和实际生成模式都与 recipe 对齐时，才可安全补成 not_applicable。
    identity_matches = (
        recipe.generation_mode == "synthetic"
        and material_metadata.get("resolved_mode") == "synthetic"
        and metadata.get("material") == recipe.material
        and metadata.get("subtype") == recipe.subtype
        and metadata.get("seed") == recipe.seed
        and material_metadata.get("profile_hash") == recipe.profile_hash
    )
    optional_geometry_fields = {
        "geometry_uncertain_mask_file",
        "determinate_mask_file",
        "geometry_lower_bound_file",
        "geometry_upper_bound_file",
    }
    if not identity_matches or optional_geometry_fields & metadata.keys():
        return None
    return _defined_geometry_measurement_semantics()


def _close_memmap(array: np.ndarray | None) -> None:
    """显式释放 NumPy memmap 句柄，尤其用于 Windows 删除/替换前。"""

    if isinstance(array, np.memmap):
        array._mmap.close()


def _load_checked_npy(
    path: Path,
    *,
    expected_sha256: str,
    expected_shape: tuple[int, int],
    expected_dtype: np.dtype[Any],
) -> np.memmap:
    """在复核文件哈希、shape 和 dtype 后以只读 memmap 打开 NPY。"""

    if not path.is_file():
        raise TerrainConfigurationError(
            f"region geometry input is missing: {path.name}"
        )
    if sha256_file(path) != expected_sha256:
        raise TerrainConfigurationError(
            f"region geometry input hash verification failed: {path.name}"
        )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap):
        raise TerrainConfigurationError(
            f"region geometry input did not open as a memory map: {path.name}"
        )
    if array.shape != expected_shape or array.dtype != expected_dtype:
        array._mmap.close()
        raise TerrainConfigurationError(
            f"region geometry input shape/dtype mismatch: {path.name}"
        )
    return array


def _safe_id(value: str, name: str) -> str:
    """拒绝可逃逸库目录的空值、点目录和路径分隔符。"""

    if (
        not value
        or value in {".", ".."}
        or any(character in value for character in "\\/:")
    ):
        raise TerrainConfigurationError(f"invalid {name}")
    return value


class TerrainLibrary:
    """管理 recipe、原始 region、track、manifest 和验证产物。"""

    DIRECTORIES = (
        "recipes",
        "sources",
        "regions",
        "tracks",
        "manifests",
        "validation",
    )

    def __init__(self, root: str | Path):
        """解析库根目录并创建固定目录骨架。"""

        self.root = Path(root).resolve()
        for name in self.DIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def recipe_path(self, terrain_recipe_id: str) -> Path:
        """返回安全的 recipe JSON 路径。"""

        return self.root / "recipes" / f"{_safe_id(terrain_recipe_id, 'terrain_recipe_id')}.json"

    def region_dir(self, terrain_recipe_id: str, region_id: str) -> Path:
        """返回一个 recipe/region 的缓存目录。"""

        return (
            self.root
            / "regions"
            / _safe_id(terrain_recipe_id, "terrain_recipe_id")
            / _safe_id(region_id, "region_id")
        )

    def region_data_path(self, terrain_recipe_id: str, region_id: str) -> Path:
        """返回 region 主高度 NPY 路径。"""

        return self.region_dir(terrain_recipe_id, region_id) / "raw_height.npy"

    def region_manifest_path(self, terrain_recipe_id: str, region_id: str) -> Path:
        """返回独立于可删除缓存目录的 region manifest 路径。"""

        return (
            self.root
            / "manifests"
            / "regions"
            / _safe_id(terrain_recipe_id, "terrain_recipe_id")
            / f"{_safe_id(region_id, 'region_id')}.json"
        )

    def _migrate_legacy_region_metadata(
        self,
        recipe: TerrainRecipe,
        region: RegionSpec,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """为可证明非测量的旧 COMPLETE region 补齐规范语义。"""

        inferred_semantics = _infer_legacy_region_measurement_semantics(
            recipe, metadata
        )
        if inferred_semantics is None:
            return metadata
        directory = self.region_dir(recipe.terrain_recipe_id, region.region_id)
        metadata_path = directory / "metadata.json"
        manifest_path = self.region_manifest_path(
            recipe.terrain_recipe_id, region.region_id
        )
        if "measurement_semantics" in metadata:
            # 正常新缓存不需要额外 I/O；迁移过的缓存则顺手修复旧实现可能
            # 留下的 metadata/manifest 并发分叉。
            if "metadata_migrated_at_utc" not in metadata:
                return metadata
            try:
                if json.loads(manifest_path.read_text(encoding="utf-8")) == metadata:
                    return metadata
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        complete_path = directory / "COMPLETE"
        if not complete_path.is_file():
            return metadata
        lock_path = directory / ".metadata-migration.lock"
        lock_started = time.monotonic()
        while True:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                try:
                    lock_age_s = time.time() - lock_path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if lock_age_s > 5.0:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() - lock_started > 30.0:
                    raise TerrainConfigurationError(
                        "timed out waiting for legacy region metadata migration"
                    )
                time.sleep(0.01)
                continue
            else:
                with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                    handle.write(
                        f"pid={os.getpid()} created={time.time():.6f}\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                break

        try:
            # CAS：等待锁期间另一个进程可能已经迁移，必须在锁内重读。
            current = json.loads(metadata_path.read_text(encoding="utf-8"))
            if complete_path.read_text(encoding="ascii").strip() != current.get(
                "data_sha256"
            ):
                raise TerrainConfigurationError(
                    "region COMPLETE marker does not match metadata"
                )
            if "measurement_semantics" in current:
                migrated = current
                try:
                    retained = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, json.JSONDecodeError):
                    retained = None
                if retained == migrated:
                    return migrated
            else:
                migrated = dict(current)
                locked_semantics = _infer_legacy_region_measurement_semantics(
                    recipe, current
                )
                if locked_semantics is None:
                    return current
                migrated["measurement_semantics"] = locked_semantics
                migrated["metadata_migrated_at_utc"] = utc_now()
            # cache metadata 是后续是否仍需迁移的判据，因此最后发布它：若前一次
            # manifest 写入或进程在两步之间失败，下次调用仍会安全重试两步。
            atomic_write_json(manifest_path, migrated)
            atomic_write_json(metadata_path, migrated)
            return migrated
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def track_path(
        self,
        terrain_recipe_id: str,
        region_id: str,
        radius_m: float,
        track_id: str,
    ) -> Path:
        """按 recipe、region、球半径和 track ID 构造缓存路径。"""

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
        """原子保存 recipe；同一 ID 已存在不同内容时拒绝覆盖。"""

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
        """读取 recipe 并复核其内容 identity 和 recipe hash。"""

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
        """按 y tile 流式生成 defined geometry 到临时 NPY，并原子发布。"""

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
            # 每个 tile 都按全局 canonical 索引重建；相邻 tile 不依赖前一 tile 状态。
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
                # 估计各滤波阶段同时存活数组的峰值，而不是简单累加整个运行期分配。
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
        """原子生成原始 region；同一 region 只支持一个 writer。"""

        region.validate_against(recipe)
        # 材料 recipe 由统一 API 生成后注册；defined geometry 才走坐标可寻址 tile 路径。
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
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return self._migrate_legacy_region_metadata(
                recipe, region, metadata
            )
        if complete_path.exists():
            complete_path.unlink()
        # 移除旧 marker 后，任何中断都会留下“不完整”而非可被误读的旧完整缓存。

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
            "measurement_semantics": _defined_geometry_measurement_semantics(),
            "created_at_utc": utc_now(),
            **metrics,
        }
        atomic_write_json(metadata_path, metadata)
        atomic_write_json(
            self.region_manifest_path(recipe.terrain_recipe_id, region.region_id),
            metadata,
        )
        # COMPLETE 最后写，并携带主高度文件哈希以绑定本次发布。
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
        """按 M1 mmap 契约原子保存材料 Terrain 及全部可选 mask/几何界。"""

        if recipe.generator_name != "material_hybrid":
            raise TerrainConfigurationError(
                "register_material_region requires a material_hybrid recipe"
            )
        region.validate_against(recipe)
        # 内存 Terrain 必须与 recipe 的材料、seed、实际模式和 profile hash 完全一致。
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
        self.save_recipe(recipe)
        directory = self.region_dir(recipe.terrain_recipe_id, region.region_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "raw_height.npy"
        mask_target = directory / "valid_mask.npy"
        uncertain_target = directory / "geometry_uncertain_mask.npy"
        determinate_target = directory / "determinate_mask.npy"
        lower_target = directory / "geometry_lower_bound.npy"
        upper_target = directory / "geometry_upper_bound.npy"
        metadata_path = directory / "metadata.json"
        complete_path = directory / "COMPLETE"
        if complete_path.is_file() and not overwrite:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return self._migrate_legacy_region_metadata(
                recipe, region, metadata
            )
        if complete_path.exists():
            complete_path.unlink()

        started = time.perf_counter()
        temporary_paths: list[Path] = []
        try:
            geometry_arrays: list[tuple[Path, NDArray[Any]]] = [
                (target, terrain.height),
                (mask_target, terrain.valid_mask),
            ]
            if terrain.geometry_uncertain_mask is not None:
                geometry_arrays.append(
                    (uncertain_target, terrain.geometry_uncertain_mask)
                )
            if terrain.determinate_mask is not None:
                geometry_arrays.append(
                    (determinate_target, terrain.determinate_mask)
                )
            if terrain.geometry_lower_bound_m is not None:
                geometry_arrays.extend(
                    (
                        (lower_target, terrain.geometry_lower_bound_m),
                        (upper_target, terrain.geometry_upper_bound_m),
                    )
                )
            for destination, array in geometry_arrays:
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
            measurement_semantics = {
                "status": (
                    "unknown_probe"
                    if terrain.resolved_mode == "measured"
                    and terrain.measurement_probe is None
                    else (
                        "known_probe"
                        if terrain.resolved_mode == "measured"
                        else "not_applicable"
                    )
                ),
                "probe": (
                    None
                    if terrain.measurement_probe is None
                    else dict(terrain.measurement_probe)
                ),
                "measurement_tolerance_m": terrain.measurement_tolerance_m,
                "determinate_mask": terrain.determinate_mask is not None,
                "bounds": terrain.geometry_lower_bound_m is not None,
                "surface_model": "single_valued_height_field_2_5d",
                "general_mesh_scope": "OUT_OF_SCOPE",
            }
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
                "measurement_semantics": measurement_semantics,
                "material_metadata": dict(terrain.metadata),
                "created_at_utc": utc_now(),
                "generation_time_s": time.perf_counter() - started,
                "data_sha256": data_hash,
                "file_size_bytes": target.stat().st_size,
            }
            if terrain.geometry_uncertain_mask is not None:
                metadata.update(
                    {
                        "geometry_uncertain_mask_file": uncertain_target.name,
                        "geometry_uncertain_mask_sha256": sha256_file(
                            uncertain_target
                        ),
                    }
                )
            if terrain.determinate_mask is not None:
                metadata.update(
                    {
                        "determinate_mask_file": determinate_target.name,
                        "determinate_mask_sha256": sha256_file(
                            determinate_target
                        ),
                    }
                )
            if terrain.geometry_lower_bound_m is not None:
                metadata.update(
                    {
                        "geometry_lower_bound_file": lower_target.name,
                        "geometry_lower_bound_sha256": sha256_file(lower_target),
                        "geometry_upper_bound_file": upper_target.name,
                        "geometry_upper_bound_sha256": sha256_file(upper_target),
                    }
                )
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
        """完整性复核后以只读 memmap 打开 region 高度。"""

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

    def open_region_valid_mask(
        self,
        terrain_recipe_id: str,
        region_id: str,
        *,
        verify_hash: bool = True,
    ) -> np.memmap | None:
        """打开材料 valid mask；defined 全有效几何返回 ``None``。"""

        directory = self.region_dir(terrain_recipe_id, region_id)
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"terrain region is absent or incomplete: {region_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        mask_name = metadata.get("valid_mask_file")
        if mask_name is None:
            if "material_metadata" in metadata:
                raise TerrainConfigurationError(
                    "material region is missing its required valid-mask contract"
                )
            return None
        expected_hash = str(metadata.get("valid_mask_sha256", ""))
        if len(expected_hash) != 64:
            raise TerrainConfigurationError(
                "material region has an invalid valid-mask identity"
            )
        mask_path = directory / str(mask_name)
        if not verify_hash:
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            if not isinstance(mask, np.memmap):
                raise TerrainConfigurationError("valid mask did not open as a memory map")
            if mask.shape != tuple(metadata["shape"]) or mask.dtype != np.bool_:
                mask._mmap.close()
                raise TerrainConfigurationError("valid mask shape/dtype mismatch")
            return mask
        return _load_checked_npy(
            mask_path,
            expected_sha256=expected_hash,
            expected_shape=tuple(metadata["shape"]),
            expected_dtype=np.dtype(np.bool_),
        )

    def load_region_spec(self, terrain_recipe_id: str, region_id: str) -> RegionSpec:
        """从独立 manifest 读取并复核 RegionSpec identity。"""

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
        """从已保存 recipe/manifest 重建 region。"""

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
        """删除可重建 region 数据，但保留 recipe 和 region manifest。"""

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
        """只删除指定 region 的可重建 track 缓存。"""

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
        """从已验证 region 计算/复用一条 track，并以锁保护并发发布。"""

        region.validate_against(recipe)
        directory = self.region_dir(recipe.terrain_recipe_id, region.region_id)
        metadata_path_region = directory / "metadata.json"
        if not metadata_path_region.is_file():
            raise FileNotFoundError(
                f"terrain region is absent or incomplete: {region.region_id}"
            )
        region_metadata = json.loads(
            metadata_path_region.read_text(encoding="utf-8")
        )
        region_metadata = self._migrate_legacy_region_metadata(
            recipe, region, region_metadata
        )
        measurement_semantics = region_metadata.get("measurement_semantics")
        required_measurement_fields = {
            "status",
            "probe",
            "measurement_tolerance_m",
            "determinate_mask",
            "bounds",
            "surface_model",
            "general_mesh_scope",
        }
        if (
            not isinstance(measurement_semantics, dict)
            or not required_measurement_fields <= measurement_semantics.keys()
            or measurement_semantics.get("status")
            not in {"not_applicable", "known_probe", "unknown_probe"}
        ):
            raise TerrainConfigurationError(
                "region metadata is missing canonical measurement_semantics; "
                "rebuild the region"
            )
        measurement_status = measurement_semantics["status"]
        height = self.open_region(
            recipe.terrain_recipe_id,
            region.region_id,
            verify_hash=True,
        )
        mask_map = self.open_region_valid_mask(
            recipe.terrain_recipe_id,
            region.region_id,
            verify_hash=True,
        )
        source_valid: NDArray[np.bool_]
        if mask_map is None:
            source_valid = np.ones(region.shape, dtype=np.bool_)
            valid_mask_sha256 = stable_hash(
                {
                    "kind": "implicit_all_valid",
                    "shape": list(region.shape),
                    "region_id": region.region_id,
                }
            )
        else:
            source_valid = mask_map
            valid_mask_sha256 = array_sha256(source_valid)

        measurement_semantics_hash = stable_hash(
            {
                "semantics": measurement_semantics,
                "geometry_uncertain_mask_sha256": region_metadata.get(
                    "geometry_uncertain_mask_sha256"
                ),
                "determinate_mask_sha256": region_metadata.get(
                    "determinate_mask_sha256"
                ),
                "geometry_lower_bound_sha256": region_metadata.get(
                    "geometry_lower_bound_sha256"
                ),
                "geometry_upper_bound_sha256": region_metadata.get(
                    "geometry_upper_bound_sha256"
                ),
            }
        )

        optional_maps: list[np.memmap] = []
        uncertain_name = region_metadata.get("geometry_uncertain_mask_file")
        if uncertain_name is not None:
            uncertain_map = _load_checked_npy(
                directory / str(uncertain_name),
                expected_sha256=str(
                    region_metadata["geometry_uncertain_mask_sha256"]
                ),
                expected_shape=region.shape,
                expected_dtype=np.dtype(np.bool_),
            )
            optional_maps.append(uncertain_map)
            source_uncertain: NDArray[np.bool_] = uncertain_map
        elif measurement_status == "unknown_probe":
            source_uncertain = source_valid.copy()
        else:
            source_uncertain = np.zeros(region.shape, dtype=np.bool_)

        lower_name = region_metadata.get("geometry_lower_bound_file")
        upper_name = region_metadata.get("geometry_upper_bound_file")
        if (lower_name is None) != (upper_name is None):
            _close_memmap(height)
            _close_memmap(mask_map)
            for mapped in optional_maps:
                _close_memmap(mapped)
            raise TerrainConfigurationError(
                "region geometry bounds are incomplete; rebuild the region"
            )
        lower_map: np.memmap | None = None
        upper_map: np.memmap | None = None
        if lower_name is not None:
            lower_map = _load_checked_npy(
                directory / str(lower_name),
                expected_sha256=str(region_metadata["geometry_lower_bound_sha256"]),
                expected_shape=region.shape,
                expected_dtype=np.dtype(np.float32),
            )
            upper_map = _load_checked_npy(
                directory / str(upper_name),
                expected_sha256=str(region_metadata["geometry_upper_bound_sha256"]),
                expected_shape=region.shape,
                expected_dtype=np.dtype(np.float32),
            )
            optional_maps.extend((lower_map, upper_map))

        source_data_sha256 = array_sha256(height)
        track_id = TrackGeometry.make_id(
            terrain_recipe_id=recipe.terrain_recipe_id,
            region_id=region.region_id,
            radius_m=radius_m,
            y_global_m=y_global_m,
            track_schema_version=TRACK_SCHEMA_VERSION,
            envelope_algorithm_version=ENVELOPE_ALGORITHM_VERSION,
            near_tie_tolerance_m=near_tie_tolerance_m,
            resolution_m=region.resolution_x_m,
            source_data_sha256=source_data_sha256,
            source_valid_mask_sha256=valid_mask_sha256,
            measurement_semantics_hash=measurement_semantics_hash,
        )
        path = self.track_path(
            recipe.terrain_recipe_id, region.region_id, radius_m, track_id
        )
        metadata_path = path.with_suffix(".json")
        complete_path = path.with_suffix(".complete")
        if complete_path.is_file() and not overwrite:
            try:
                return self.load_track(
                    recipe.terrain_recipe_id, region.region_id, radius_m, track_id
                )
            finally:
                _close_memmap(height)
                _close_memmap(mask_map)
                for mapped in optional_maps:
                    _close_memmap(mapped)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        # O_EXCL 锁保证同一 track 只有一个 writer；读者仍只认最后发布的 .complete。
        lock_started = time.monotonic()
        owns_lock = False
        while not owns_lock:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                if complete_path.is_file() and not overwrite:
                    try:
                        return self.load_track(
                            recipe.terrain_recipe_id,
                            region.region_id,
                            radius_m,
                            track_id,
                        )
                    finally:
                        _close_memmap(height)
                        _close_memmap(mask_map)
                        for mapped in optional_maps:
                            _close_memmap(mapped)
                try:
                    lock_age_s = (
                        time.time() - lock_path.stat().st_mtime
                    )
                except FileNotFoundError:
                    continue
                if lock_age_s > 3600.0:
                    # 超过一小时的锁视为崩溃 writer 遗留；活跃锁最多等待五分钟。
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() - lock_started > 300.0:
                    _close_memmap(height)
                    _close_memmap(mask_map)
                    for mapped in optional_maps:
                        _close_memmap(mapped)
                    raise TerrainConfigurationError(
                        "timed out waiting for the track-cache writer: "
                        f"{track_id}"
                    )
                time.sleep(0.05)
                continue
            else:
                with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                    handle.write(
                        f"pid={os.getpid()} created={time.time():.6f}\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                owns_lock = True

        try:
            if complete_path.is_file() and not overwrite:
                return self.load_track(
                    recipe.terrain_recipe_id,
                    region.region_id,
                    radius_m,
                    track_id,
                )
            started = time.perf_counter()
            track = compute_track_geometry(
                height,
                region,
                radius_m=radius_m,
                y_global_m=y_global_m,
                near_tie_tolerance_m=near_tie_tolerance_m,
                source_valid_mask=source_valid,
                source_uncertain_mask=source_uncertain,
                height_lower_bound_m=lower_map,
                height_upper_bound_m=upper_map,
                source_data_sha256=source_data_sha256,
                source_valid_mask_sha256=valid_mask_sha256,
                measurement_semantics_hash=measurement_semantics_hash,
            )
            if track.track_id != track_id:
                raise TerrainConfigurationError(
                    "computed track ID differs from cache key"
                )
            arrays = {
                "x_global_m": track.x_global_m,
                "envelope_height_m": track.envelope_height_m,
                "envelope_slope_x": track.envelope_slope_x,
                "envelope_slope_y": track.envelope_slope_y,
                "support_x_m": track.support_x_m,
                "support_y_m": track.support_y_m,
                "support_points_m": track.support_points_m,
                "support_feature_indices_yx": track.support_feature_indices_yx,
                "support_value_gap_m": track.support_value_gap_m,
                "surface_normals": track.surface_normals,
                "envelope_normals": track.envelope_normals,
                "contact_normals": track.contact_normals,
                "footprint_valid_mask": track.footprint_valid_mask,
                "valid_mask": track.valid_mask,
                "near_tie_flag": track.near_tie_flag,
                "feature_switch_flag": track.feature_switch_flag,
                "geometry_uncertain_mask": track.geometry_uncertain_mask,
            }
            if track.envelope_height_lower_m is not None:
                arrays["envelope_height_lower_m"] = track.envelope_height_lower_m
                arrays["envelope_height_upper_m"] = track.envelope_height_upper_m
            atomic_write_npz(path, arrays)
            metadata = {
                "schema_version": TRACK_SCHEMA_VERSION,
                "m1_module_version": M1_MODULE_VERSION,
                "terrain_recipe_id": track.terrain_recipe_id,
                "region_id": track.region_id,
                "track_id": track.track_id,
                "radius_m": track.radius_m,
                "y_global_m": track.y_global_m,
                "resolution_m": track.resolution_m,
                "near_tie_tolerance_m": track.near_tie_tolerance_m,
                "source_data_sha256": track.source_data_sha256,
                "source_valid_mask_sha256": track.source_valid_mask_sha256,
                "measurement_semantics_hash": track.measurement_semantics_hash,
                "has_envelope_height_bounds": (
                    track.envelope_height_lower_m is not None
                ),
                "envelope_algorithm_version": (
                    track.envelope_algorithm_version
                ),
                "model_warning": list(track.model_warning),
                "sample_count": int(track.x_global_m.size),
                "valid_count": int(np.count_nonzero(track.valid_mask)),
                "footprint_valid_count": int(
                    np.count_nonzero(track.footprint_valid_mask)
                ),
                "geometry_uncertain_count": int(
                    np.count_nonzero(track.geometry_uncertain_mask)
                ),
                "generation_time_s": time.perf_counter() - started,
                "data_sha256": sha256_file(path),
                "created_at_utc": utc_now(),
            }
            atomic_write_json(metadata_path, metadata)
            # 与 region 相同，数据和 metadata 均完成后才发布携带数据哈希的 marker。
            atomic_write_bytes(
                complete_path,
                (metadata["data_sha256"] + "\n").encode("ascii"),
            )
            return track
        finally:
            _close_memmap(height)
            _close_memmap(mask_map)
            for mapped in optional_maps:
                _close_memmap(mapped)
            if owns_lock:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    def load_track(
        self,
        terrain_recipe_id: str,
        region_id: str,
        radius_m: float,
        track_id: str,
    ) -> TrackGeometry:
        """复核 marker、文件哈希、schema 和 cache-key identity 后加载 track。"""

        path = self.track_path(
            terrain_recipe_id, region_id, radius_m, track_id
        )
        metadata_path = path.with_suffix(".json")
        complete_path = path.with_suffix(".complete")
        if not (path.is_file() and metadata_path.is_file() and complete_path.is_file()):
            raise FileNotFoundError(f"track is absent or incomplete: {track_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("schema_version", "")) != TRACK_SCHEMA_VERSION:
            raise TerrainConfigurationError(
                "track cache schema is obsolete; delete/rebuild track caches"
            )
        if metadata.get("envelope_algorithm_version") != ENVELOPE_ALGORITHM_VERSION:
            raise TerrainConfigurationError(
                "track envelope algorithm is obsolete; delete/rebuild track caches"
            )
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
            track_schema_version=metadata["schema_version"],
            envelope_algorithm_version=metadata["envelope_algorithm_version"],
            near_tie_tolerance_m=float(metadata["near_tie_tolerance_m"]),
            resolution_m=float(metadata["resolution_m"]),
            source_data_sha256=metadata["source_data_sha256"],
            source_valid_mask_sha256=metadata["source_valid_mask_sha256"],
            measurement_semantics_hash=metadata["measurement_semantics_hash"],
        )
        if (
            metadata["terrain_recipe_id"] != terrain_recipe_id
            or metadata["region_id"] != region_id
            or metadata["track_id"] != track_id
            or not math.isclose(
                float(metadata["radius_m"]),
                radius_m,
                rel_tol=0.0,
                abs_tol=max(1e-15, abs(radius_m) * 1e-12),
            )
            or expected_id != track_id
        ):
            raise TerrainConfigurationError(
                "track path identity does not match metadata/cache key"
            )
        # NPZ 关闭前复制全部数组，返回的 TrackGeometry 不依赖 archive 文件句柄。
        with np.load(path, allow_pickle=False) as arrays:
            copied = {name: arrays[name] for name in arrays.files}
        required = {
            "x_global_m",
            "envelope_height_m",
            "envelope_slope_x",
            "envelope_slope_y",
            "support_x_m",
            "support_y_m",
            "support_points_m",
            "support_feature_indices_yx",
            "support_value_gap_m",
            "surface_normals",
            "envelope_normals",
            "contact_normals",
            "footprint_valid_mask",
            "valid_mask",
            "near_tie_flag",
            "feature_switch_flag",
            "geometry_uncertain_mask",
        }
        missing = required - copied.keys()
        if missing:
            raise TerrainConfigurationError(
                "track cache is incomplete for schema v2; delete/rebuild track caches"
            )
        has_bounds = bool(metadata.get("has_envelope_height_bounds"))
        if has_bounds and not {
            "envelope_height_lower_m",
            "envelope_height_upper_m",
        } <= copied.keys():
            raise TerrainConfigurationError(
                "track bound arrays are incomplete; delete/rebuild track caches"
            )
        return TrackGeometry(
            terrain_recipe_id=metadata["terrain_recipe_id"],
            region_id=metadata["region_id"],
            track_id=metadata["track_id"],
            radius_m=float(metadata["radius_m"]),
            y_global_m=float(metadata["y_global_m"]),
            resolution_m=float(metadata["resolution_m"]),
            track_schema_version=metadata["schema_version"],
            envelope_algorithm_version=metadata["envelope_algorithm_version"],
            near_tie_tolerance_m=float(metadata["near_tie_tolerance_m"]),
            source_data_sha256=metadata["source_data_sha256"],
            source_valid_mask_sha256=metadata["source_valid_mask_sha256"],
            measurement_semantics_hash=metadata["measurement_semantics_hash"],
            x_global_m=np.asarray(copied["x_global_m"], dtype=np.float64),
            envelope_height_m=np.asarray(
                copied["envelope_height_m"], dtype=np.float64
            ),
            envelope_slope_x=np.asarray(
                copied["envelope_slope_x"], dtype=np.float64
            ),
            envelope_slope_y=np.asarray(
                copied["envelope_slope_y"], dtype=np.float64
            ),
            support_x_m=np.asarray(copied["support_x_m"], dtype=np.float64),
            support_y_m=np.asarray(copied["support_y_m"], dtype=np.float64),
            support_points_m=np.asarray(
                copied["support_points_m"], dtype=np.float64
            ),
            support_feature_indices_yx=np.asarray(
                copied["support_feature_indices_yx"], dtype=np.int64
            ),
            support_value_gap_m=np.asarray(
                copied["support_value_gap_m"], dtype=np.float64
            ),
            surface_normals=np.asarray(copied["surface_normals"], dtype=np.float64),
            envelope_normals=np.asarray(
                copied["envelope_normals"], dtype=np.float64
            ),
            contact_normals=np.asarray(copied["contact_normals"], dtype=np.float64),
            footprint_valid_mask=np.asarray(
                copied["footprint_valid_mask"], dtype=np.bool_
            ),
            valid_mask=np.asarray(copied["valid_mask"], dtype=np.bool_),
            near_tie_flag=np.asarray(copied["near_tie_flag"], dtype=np.bool_),
            feature_switch_flag=np.asarray(
                copied["feature_switch_flag"], dtype=np.bool_
            ),
            geometry_uncertain_mask=np.asarray(
                copied["geometry_uncertain_mask"], dtype=np.bool_
            ),
            envelope_height_lower_m=(
                np.asarray(copied["envelope_height_lower_m"], dtype=np.float64)
                if has_bounds
                else None
            ),
            envelope_height_upper_m=(
                np.asarray(copied["envelope_height_upper_m"], dtype=np.float64)
                if has_bounds
                else None
            ),
            model_warning=tuple(metadata.get("model_warning", ())),
        )
