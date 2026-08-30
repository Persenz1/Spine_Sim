from __future__ import annotations

import json
import gc
import importlib.util
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from spine_sim.geometry import SurfaceState
from spine_sim.terrain.errors import GeometryOutOfDomainError
from spine_sim.terrain.errors import TerrainConfigurationError
from spine_sim.terrain.heightmap import (
    FileHeightMapSource,
    register_heightmap_source,
    sample_file_heightmap,
)
from spine_sim.terrain.library import TerrainLibrary
from spine_sim.terrain.models import (
    ENVELOPE_ALGORITHM_VERSION,
    TRACK_SCHEMA_VERSION,
    CampaignDesignSpace,
    RegionSpec,
    TerrainRecipe,
    TrackGeometry,
    compute_campaign_region,
)
from spine_sim.terrain.random_field import generate_defined_geometry
from spine_sim.terrain.suite import generate_terrain_suite


def _mmap_checksum(path: str) -> float:
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    return float(np.sum(mapped, dtype=np.float64))


class TerrainLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe = TerrainRecipe(
            seed=9,
            correlation_length_x_m=20e-6,
            correlation_length_y_m=20e-6,
            kernel_truncate_sigma=2.0,
        )
        self.region = RegionSpec(
            terrain_recipe_id=self.recipe.terrain_recipe_id,
            origin_x_m=-0.3e-3,
            origin_y_m=-0.2e-3,
            size_x_m=0.6e-3,
            size_y_m=0.4e-3,
            purpose="debug",
        )

    def test_atomic_region_mmap_track_and_delete_rebuild_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = TerrainLibrary(temporary)
            generated = library.generate_region(
                self.recipe, self.region, tile_rows=7
            )
            mapped = library.open_region(
                self.recipe.terrain_recipe_id,
                self.region.region_id,
                verify_hash=True,
            )
            self.assertIsInstance(mapped, np.memmap)
            self.assertEqual(mapped.dtype, np.float32)
            self.assertEqual(mapped.shape, self.region.shape)
            self.assertFalse(mapped.flags.writeable)
            track = library.cache_track(
                self.recipe,
                self.region,
                radius_m=50e-6,
                y_global_m=0.0,
            )
            loaded = library.load_track(
                self.recipe.terrain_recipe_id,
                self.region.region_id,
                50e-6,
                track.track_id,
            )
            SurfaceState(
                loaded,
                self.region,
                np.array(mapped, copy=True),
                np.ones(self.region.shape, dtype=np.bool_),
            )
            with self.assertRaisesRegex(
                ValueError, "height_m does not match track.source_data_sha256"
            ):
                SurfaceState(
                    loaded,
                    self.region,
                    mapped[::-1],
                    np.ones(self.region.shape, dtype=np.bool_),
                )
            np.testing.assert_array_equal(
                loaded.envelope_height_m, track.envelope_height_m
            )
            mapped._mmap.close()
            del mapped
            gc.collect()
            deletion = library.delete_region_cache(
                self.recipe.terrain_recipe_id, self.region.region_id
            )
            self.assertTrue(deletion["recoverable"])
            self.assertFalse(
                library.region_dir(
                    self.recipe.terrain_recipe_id, self.region.region_id
                ).exists()
            )
            rebuilt = library.rebuild_region(
                self.recipe.terrain_recipe_id,
                self.region.region_id,
                tile_rows=13,
            )
            self.assertEqual(generated["data_sha256"], rebuilt["data_sha256"])

    def test_read_only_mmap_is_consistent_across_spawned_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = TerrainLibrary(temporary)
            library.generate_region(self.recipe, self.region, tile_rows=11)
            path = library.region_data_path(
                self.recipe.terrain_recipe_id, self.region.region_id
            )
            expected = _mmap_checksum(str(path))
            with ProcessPoolExecutor(max_workers=2) as executor:
                values = list(
                    executor.map(_mmap_checksum, [str(path), str(path)])
                )
            self.assertEqual(values, [expected, expected])

    def test_interrupted_files_are_not_treated_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = TerrainLibrary(temporary)
            directory = library.region_dir(
                self.recipe.terrain_recipe_id, self.region.region_id
            )
            directory.mkdir(parents=True)
            np.save(directory / "raw_height.npy", np.zeros((2, 2), dtype=np.float32))
            with self.assertRaises(FileNotFoundError):
                library.open_region(
                    self.recipe.terrain_recipe_id, self.region.region_id
                )

    def test_track_identity_ignores_binary_unit_arithmetic_noise(self) -> None:
        common = {
            "terrain_recipe_id": self.recipe.terrain_recipe_id,
            "region_id": self.region.region_id,
            "y_global_m": 0.0,
            "track_schema_version": TRACK_SCHEMA_VERSION,
            "envelope_algorithm_version": ENVELOPE_ALGORITHM_VERSION,
            "near_tie_tolerance_m": 1e-10,
            "resolution_m": 10e-6,
            "source_data_sha256": "1" * 64,
            "source_valid_mask_sha256": "2" * 64,
            "measurement_semantics_hash": "3" * 64,
        }
        self.assertEqual(
            TrackGeometry.make_id(radius_m=100e-6, **common),
            TrackGeometry.make_id(radius_m=100.0 * 1e-6, **common),
        )

    def test_track_identity_changes_with_geometry_semantics(self) -> None:
        common = {
            "terrain_recipe_id": self.recipe.terrain_recipe_id,
            "region_id": self.region.region_id,
            "radius_m": 50e-6,
            "y_global_m": 0.0,
            "track_schema_version": TRACK_SCHEMA_VERSION,
            "envelope_algorithm_version": ENVELOPE_ALGORITHM_VERSION,
            "resolution_m": 10e-6,
            "source_data_sha256": "1" * 64,
            "source_valid_mask_sha256": "2" * 64,
            "measurement_semantics_hash": "3" * 64,
        }
        baseline = TrackGeometry.make_id(
            near_tie_tolerance_m=1e-10, **common
        )
        self.assertNotEqual(
            baseline,
            TrackGeometry.make_id(near_tie_tolerance_m=2e-10, **common),
        )
        self.assertNotEqual(
            baseline,
            TrackGeometry.make_id(
                near_tie_tolerance_m=1e-10,
                **{**common, "source_valid_mask_sha256": "4" * 64},
            ),
        )
        self.assertNotEqual(
            baseline,
            TrackGeometry.make_id(
                near_tie_tolerance_m=1e-10,
                **{**common, "source_data_sha256": "5" * 64},
            ),
        )
        self.assertNotEqual(
            baseline,
            TrackGeometry.make_id(
                near_tie_tolerance_m=1e-10,
                **{**common, "measurement_semantics_hash": "6" * 64},
            ),
        )

    def test_obsolete_track_schema_requires_explicit_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = TerrainLibrary(temporary)
            library.generate_region(self.recipe, self.region)
            track = library.cache_track(
                self.recipe,
                self.region,
                radius_m=50e-6,
                y_global_m=0.0,
            )
            sidecar = library.track_path(
                self.recipe.terrain_recipe_id,
                self.region.region_id,
                50e-6,
                track.track_id,
            ).with_suffix(".json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            metadata["schema_version"] = "1"
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(
                TerrainConfigurationError, "obsolete.*rebuild"
            ):
                library.load_track(
                    self.recipe.terrain_recipe_id,
                    self.region.region_id,
                    50e-6,
                    track.track_id,
                )
            with self.assertRaisesRegex(
                TerrainConfigurationError, "obsolete.*rebuild"
            ):
                library.cache_track(
                    self.recipe,
                    self.region,
                    radius_m=50e-6,
                    y_global_m=0.0,
                )

    def test_obsolete_envelope_algorithm_requires_explicit_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = TerrainLibrary(temporary)
            library.generate_region(self.recipe, self.region)
            track = library.cache_track(
                self.recipe,
                self.region,
                radius_m=50e-6,
                y_global_m=0.0,
            )
            sidecar = library.track_path(
                self.recipe.terrain_recipe_id,
                self.region.region_id,
                50e-6,
                track.track_id,
            ).with_suffix(".json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            metadata["envelope_algorithm_version"] = (
                "finite-sphere-envelope-v2-footprint-support2"
            )
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(
                TerrainConfigurationError, "algorithm is obsolete.*rebuild"
            ):
                library.load_track(
                    self.recipe.terrain_recipe_id,
                    self.region.region_id,
                    50e-6,
                    track.track_id,
                )

    @unittest.skipUnless(importlib.util.find_spec("cupy"), "CuPy/GPU is unavailable")
    def test_cuda_library_region_matches_cpu_and_records_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library = TerrainLibrary(temporary)
            metadata = library.generate_region(
                self.recipe,
                self.region,
                tile_rows=7,
                backend="cuda",
            )
            mapped = library.open_region(
                self.recipe.terrain_recipe_id, self.region.region_id
            )
            expected = generate_defined_geometry(
                self.recipe, self.region, backend="cpu"
            )
            np.testing.assert_allclose(mapped, expected, rtol=2e-6, atol=1e-10)
            self.assertEqual(metadata["generation_backend"], "cuda")
            self.assertGreater(
                metadata["gpu_memory_pool_peak_cached_bytes"], 0
            )
            mapped._mmap.close()


class HeightMapTests(unittest.TestCase):
    def test_source_hash_units_preprocessing_and_domain_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "measured.npy"
            raw_um = np.add.outer(np.arange(4.0), np.arange(5.0))
            np.save(source_path, raw_um)
            source = FileHeightMapSource.from_file(
                source_path,
                source_height_unit="um",
                origin_x_m=1e-3,
                origin_y_m=-1e-3,
                spacing_x_m=10e-6,
                spacing_y_m=20e-6,
            )
            metadata_path = register_heightmap_source(root / "library", source)
            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8"))["source_id"],
                source.source_id,
            )
            values, record = sample_file_heightmap(
                source,
                np.array([1.005e-3, 1.015e-3]),
                np.array([-0.99e-3, -0.97e-3]),
                preprocessing={"crop_policy": "query_bounds"},
            )
            np.testing.assert_allclose(
                values,
                np.array([[1.0, 2.0], [2.0, 3.0]]) * 1e-6,
                atol=1e-15,
            )
            self.assertEqual(record["interpretation"], "file_heightmap_no_random_recipe_statistics")
            with self.assertRaises(GeometryOutOfDomainError):
                sample_file_heightmap(
                    source,
                    np.array([0.0]),
                    np.array([-0.99e-3]),
                )


class CampaignRegionTests(unittest.TestCase):
    def test_maximum_region_is_derived_and_covers_declared_design(self) -> None:
        recipe = TerrainRecipe()
        design = CampaignDesignSpace()
        report = compute_campaign_region(recipe, design)
        region = report.region
        self.assertGreater(region.size_x_m, 0.140)
        self.assertLess(region.size_x_m, 0.151)
        self.assertGreater(region.size_y_m, 0.039)
        self.assertLess(region.size_y_m, 0.051)
        self.assertGreaterEqual(
            report.aligned_bounds_m["x_min"], report.raw_bounds_m["x_min"] - design.resolution_m
        )
        self.assertLessEqual(
            report.aligned_bounds_m["x_min"], report.raw_bounds_m["x_min"]
        )
        self.assertGreaterEqual(
            report.aligned_bounds_m["x_max"], report.raw_bounds_m["x_max"]
        )
        self.assertIn("x_spring_horizontal_projection", report.margins_m)
        self.assertIn("y_random_filter_halo", report.margins_m)
        equivalent_json_region = RegionSpec(
            terrain_recipe_id=recipe.terrain_recipe_id,
            origin_x_m=-0.02198,
            origin_y_m=-0.0201,
            size_x_m=0.14796,
            size_y_m=0.0402,
            purpose="campaign",
        )
        self.assertEqual(region.region_id, equivalent_json_region.region_id)


class TerrainSuiteTests(unittest.TestCase):
    def test_suite_rejects_non_ten_or_unsafe_condition_sets_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for suite_name, conditions in (
                ("../unsafe", [{}] * 10),
                ("valid_name", []),
            ):
                with self.subTest(suite_name=suite_name):
                    with self.assertRaises(TerrainConfigurationError):
                        generate_terrain_suite(
                            temporary,
                            {
                                "schema_version": "1",
                                "suite_name": suite_name,
                                "base_recipe": {},
                                "conditions": conditions,
                            },
                        )


if __name__ == "__main__":
    unittest.main()
