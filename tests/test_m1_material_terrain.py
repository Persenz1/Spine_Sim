from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spine_sim.terrain import (
    Terrain,
    TerrainLibrary,
    available_profiles,
    generate_terrain,
    load_material_profile,
    load_terrain,
    refine_material_terrain_same_realization,
    register_terrain,
    save_terrain,
)
from spine_sim.terrain.errors import TerrainConfigurationError
from spine_sim.terrain.measured import (
    load_measured_surface,
    resample_measured_patch,
)
from spine_sim.terrain.material_generators import add_irregular_features


class MaterialProfileTests(unittest.TestCase):
    def test_profiles_are_material_specific_and_provisional_labels_are_explicit(
        self,
    ) -> None:
        profiles = available_profiles()
        self.assertIn("P100", profiles["sandpaper"])
        self.assertIn("P200", profiles["sandpaper"])
        self.assertIn("fired_brick_standard", profiles["red_brick"])
        self.assertIn("rough_wall", profiles["concrete"])
        p200 = load_material_profile("sandpaper", "P200")
        self.assertEqual(p200["status"], "provisional")
        self.assertIn("interpolation", p200["parameter_basis"])
        self.assertEqual(p200["source_data"], [])
        with self.assertRaises(TerrainConfigurationError):
            load_material_profile("concrete", "P100")


class MaterialGenerationTests(unittest.TestCase):
    def _generate(self, material: str, subtype: str, seed: int = 912):
        return generate_terrain(
            material=material,
            subtype=subtype,
            size_x_m=0.6e-3,
            size_y_m=0.4e-3,
            resolution_m=10e-6,
            seed=seed,
            mode="synthetic",
        )

    def test_all_three_materials_are_finite_reproducible_and_distinct(self) -> None:
        specs = (
            ("sandpaper", "P200"),
            ("red_brick", "fired_brick_standard"),
            ("concrete", "rough_wall"),
        )
        terrains = []
        for material, subtype in specs:
            with self.subTest(material=material):
                first = self._generate(material, subtype)
                second = self._generate(material, subtype)
                self.assertEqual(first.height.shape, (41, 61))
                self.assertEqual(first.height.dtype, np.float32)
                self.assertEqual(first.valid_mask.dtype, np.bool_)
                self.assertTrue(np.all(first.valid_mask))
                self.assertTrue(np.all(np.isfinite(first.height)))
                np.testing.assert_array_equal(first.height, second.height)
                self.assertEqual(first.metadata, second.metadata)
                terrains.append(first.height)
        self.assertFalse(np.array_equal(terrains[0], terrains[1]))
        self.assertFalse(np.array_equal(terrains[1], terrains[2]))

    def test_seed_changes_realization_without_changing_profile(self) -> None:
        first = self._generate("red_brick", "fired_brick_standard", seed=1)
        second = self._generate("red_brick", "fired_brick_standard", seed=2)
        self.assertFalse(np.array_equal(first.height, second.height))
        self.assertEqual(
            first.metadata["profile_hash"], second.metadata["profile_hash"]
        )

    def test_same_realization_refinement_preserves_every_coarse_node(
        self,
    ) -> None:
        coarse = self._generate(
            "red_brick",
            "fired_brick_standard",
            seed=41_001,
        )
        fine_detail = generate_terrain(
            material="red_brick",
            subtype="fired_brick_standard",
            size_x_m=0.6e-3,
            size_y_m=0.4e-3,
            resolution_m=5e-6,
            seed=41_001,
            mode="synthetic",
        )
        refined = refine_material_terrain_same_realization(
            coarse,
            fine_detail,
        )
        self.assertEqual(refined.height.shape, (81, 121))
        np.testing.assert_array_equal(
            refined.height[::2, ::2],
            coarse.height,
        )
        self.assertTrue(np.all(refined.valid_mask))
        self.assertTrue(
            refined.metadata["same_realization_refinement"][
                "coarse_node_identity_exact"
            ]
        )
        other_seed = generate_terrain(
            material="red_brick",
            subtype="fired_brick_standard",
            size_x_m=0.6e-3,
            size_y_m=0.4e-3,
            resolution_m=5e-6,
            seed=41_002,
            mode="synthetic",
        )
        with self.assertRaisesRegex(
            TerrainConfigurationError,
            "material, subtype and seed",
        ):
            refine_material_terrain_same_realization(
                coarse,
                other_seed,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("cupy"),
        "CuPy/GPU is unavailable",
    )
    def test_cuda_material_generation_matches_cpu_and_records_device(self) -> None:
        cpu = generate_terrain(
            material="red_brick",
            subtype="fired_brick_standard",
            size_x_m=0.6e-3,
            size_y_m=0.4e-3,
            resolution_m=10e-6,
            seed=913,
            mode="synthetic",
            backend="cpu",
        )
        cuda = generate_terrain(
            material="red_brick",
            subtype="fired_brick_standard",
            size_x_m=0.6e-3,
            size_y_m=0.4e-3,
            resolution_m=10e-6,
            seed=913,
            mode="synthetic",
            backend="cuda",
        )
        np.testing.assert_allclose(
            cuda.height,
            cpu.height,
            rtol=2e-6,
            atol=2e-10,
        )
        backend = cuda.metadata["generation_backend"]
        self.assertEqual(backend["resolved"], "cuda")
        self.assertEqual(backend["provider"], "cupy")
        self.assertIn("NVIDIA", backend["device_name"])
        self.assertGreater(
            backend["gpu_memory_pool_peak_cached_bytes"],
            0,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("cupy"),
        "CuPy/GPU is unavailable",
    )
    def test_cuda_irregular_feature_stamping_is_reproducible(self) -> None:
        common = {
            "dx_m": 10e-6,
            "dy_m": 10e-6,
            "density_per_m2": 10_000_000.0,
            "diameter_median_m": 0.25e-3,
            "diameter_log_sigma": 0.25,
            "aspect_ratio_median": 1.4,
            "aspect_ratio_log_sigma": 0.2,
            "amplitude_median_m": 0.2e-3,
            "amplitude_log_sigma": 0.3,
            "edge_power": 1.2,
            "boundary_roughness": 0.15,
            "sign_mode": "mixed",
            "positive_probability": 0.6,
            "cluster_probability": 0.2,
        }
        cpu = np.zeros((201, 201), dtype=np.float32)
        first_cuda = np.zeros_like(cpu)
        second_cuda = np.zeros_like(cpu)
        cpu_record = add_irregular_features(
            cpu,
            rng=np.random.Generator(np.random.PCG64(412)),
            backend="cpu",
            **common,
        )
        first_record = add_irregular_features(
            first_cuda,
            rng=np.random.Generator(np.random.PCG64(412)),
            backend="cuda",
            **common,
        )
        second_record = add_irregular_features(
            second_cuda,
            rng=np.random.Generator(np.random.PCG64(412)),
            backend="cuda",
            **common,
        )
        self.assertEqual(first_record, cpu_record)
        self.assertEqual(second_record, first_record)
        np.testing.assert_array_equal(second_cuda, first_cuda)
        np.testing.assert_allclose(
            first_cuda,
            cpu,
            rtol=2e-4,
            atol=5e-8,
        )

    @unittest.skipUnless(
        Path("data/raw/mendeley_hcgcnm269w_v2/P100.csv").is_file(),
        "verified public P100 raw height is not available",
    )
    def test_local_p100_synthetic_uses_verified_measured_patches(self) -> None:
        terrain = generate_terrain(
            material="sandpaper",
            subtype="P100",
            size_x_m=0.6e-3,
            size_y_m=0.3e-3,
            resolution_m=10e-6,
            seed=88,
            mode="synthetic",
        )
        self.assertTrue(terrain.metadata["generation"]["measured_patch_used"])
        self.assertEqual(
            terrain.metadata["generation"]["source"]["sha256"],
            "a085c14fbbae9999dd29395d69f6ce77c655a5ed8d541c5e62e4a5a59bce1d18",
        )

    def test_auto_fallback_is_exactly_explicit_synthetic(self) -> None:
        automatic = generate_terrain(
            material="concrete",
            subtype="rough_wall",
            size_x_m=0.4e-3,
            size_y_m=0.3e-3,
            resolution_m=10e-6,
            seed=77,
            mode="auto",
        )
        explicit = generate_terrain(
            material="concrete",
            subtype="rough_wall",
            size_x_m=0.4e-3,
            size_y_m=0.3e-3,
            resolution_m=10e-6,
            seed=77,
            mode="synthetic",
        )
        np.testing.assert_array_equal(automatic.height, explicit.height)
        self.assertEqual(automatic.resolved_mode, "synthetic")
        self.assertIn("auto_fallback_reason", automatic.metadata)

    def test_save_load_library_and_m3_region_contract(self) -> None:
        terrain = self._generate("red_brick", "fired_brick_standard")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = save_terrain(root / "brick.npz", terrain)
            loaded = load_terrain(artifact)
            np.testing.assert_array_equal(loaded.height, terrain.height)
            recipe, region, metadata = register_terrain(
                root / "library", terrain
            )
            library = TerrainLibrary(root / "library")
            mapped = library.open_region(
                recipe.terrain_recipe_id, region.region_id, verify_hash=True
            )
            self.assertEqual(mapped.dtype, np.float32)
            np.testing.assert_array_equal(mapped, terrain.height)
            mapped._mmap.close()
            track = library.cache_track(
                recipe,
                region,
                radius_m=50e-6,
                y_global_m=0.2e-3,
            )
            self.assertEqual(track.region_id, region.region_id)
            self.assertGreater(np.count_nonzero(track.valid_mask), 0)
            manifest = json.loads(
                library.region_manifest_path(
                    recipe.terrain_recipe_id, region.region_id
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["material"], "red_brick")
            self.assertEqual(metadata["valid_fraction"], 1.0)

    def test_load_rejects_noncanonical_height_dtype(self) -> None:
        terrain = self._generate("red_brick", "fired_brick_standard")
        with tempfile.TemporaryDirectory() as temporary:
            artifact = save_terrain(Path(temporary) / "brick.npz", terrain)
            with np.load(artifact, allow_pickle=False) as archive:
                payload = {name: archive[name] for name in archive.files}
            payload["height"] = payload["height"].astype(np.float64)
            with artifact.open("wb") as stream:
                np.savez_compressed(stream, **payload)

            with self.assertRaisesRegex(TerrainConfigurationError, "float32"):
                load_terrain(artifact)

    def test_load_requires_canonical_measurement_semantics(self) -> None:
        terrain = self._generate("red_brick", "fired_brick_standard")
        with tempfile.TemporaryDirectory() as temporary:
            artifact = save_terrain(Path(temporary) / "brick.npz", terrain)
            with np.load(artifact, allow_pickle=False) as archive:
                payload = {name: archive[name] for name in archive.files}
            metadata = json.loads(str(payload["metadata_json"].item()))
            del metadata["measurement_semantics"]
            payload["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True, ensure_ascii=False)
            )
            with artifact.open("wb") as stream:
                np.savez_compressed(stream, **payload)

            with self.assertRaisesRegex(
                TerrainConfigurationError, "measurement_semantics"
            ):
                load_terrain(artifact)

    def test_cache_track_uses_material_mask_and_mask_hash_identity(self) -> None:
        terrain = self._generate("red_brick", "fired_brick_standard")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe, region, _ = register_terrain(root, terrain)
            library = TerrainLibrary(root)
            first = library.cache_track(
                recipe, region, radius_m=50e-6, y_global_m=0.2e-3
            )
            changed_mask = terrain.valid_mask.copy()
            row = changed_mask.shape[0] // 2
            column = changed_mask.shape[1] // 2
            changed_mask[row, column + 4] = False
            changed = Terrain(
                height=terrain.height,
                dx=terrain.dx,
                dy=terrain.dy,
                valid_mask=changed_mask,
                material=terrain.material,
                subtype=terrain.subtype,
                seed=terrain.seed,
                metadata=terrain.metadata,
                geometry_uncertain_mask=terrain.geometry_uncertain_mask,
            )
            register_terrain(root, changed, overwrite=True)
            second = library.cache_track(
                recipe, region, radius_m=50e-6, y_global_m=0.2e-3
            )
            self.assertNotEqual(first.track_id, second.track_id)
            self.assertFalse(second.footprint_valid_mask[column])
            self.assertFalse(second.valid_mask[column])

    def test_registered_geometry_bounds_round_trip_into_track(self) -> None:
        terrain = self._generate("sandpaper", "P200")
        bounded = Terrain(
            height=terrain.height,
            dx=terrain.dx,
            dy=terrain.dy,
            valid_mask=terrain.valid_mask,
            material=terrain.material,
            subtype=terrain.subtype,
            seed=terrain.seed,
            metadata=terrain.metadata,
            geometry_uncertain_mask=np.ones_like(
                terrain.valid_mask, dtype=np.bool_
            ),
            geometry_lower_bound_m=np.asarray(
                terrain.height - 1e-6, dtype=np.float32
            ),
            geometry_upper_bound_m=np.asarray(
                terrain.height + 2e-6, dtype=np.float32
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            recipe, region, _ = register_terrain(temporary, bounded)
            track = TerrainLibrary(temporary).cache_track(
                recipe, region, radius_m=50e-6, y_global_m=0.2e-3
            )
            index = region.shape[1] // 2
            self.assertIsNotNone(track.envelope_height_lower_m)
            self.assertIsNotNone(track.envelope_height_upper_m)
            self.assertTrue(track.geometry_uncertain_mask[index])
            self.assertGreater(
                track.envelope_height_upper_m[index]
                - track.envelope_height_lower_m[index],
                0.0,
            )

    def test_cache_track_rejects_missing_material_mask(self) -> None:
        terrain = self._generate("concrete", "rough_wall")
        with tempfile.TemporaryDirectory() as temporary:
            recipe, region, metadata = register_terrain(temporary, terrain)
            mask_path = (
                TerrainLibrary(temporary).region_dir(
                    recipe.terrain_recipe_id, region.region_id
                )
                / metadata["valid_mask_file"]
            )
            mask_path.unlink()
            with self.assertRaisesRegex(
                TerrainConfigurationError, "geometry input is missing"
            ):
                TerrainLibrary(temporary).cache_track(
                    recipe, region, radius_m=50e-6, y_global_m=0.2e-3
                )

class MeasuredPreprocessingTests(unittest.TestCase):
    def test_zero_is_valid_by_default_and_nonfinite_mask_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scan.npy"
            raw_um = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, np.nan, 3.0],
                    [2.0, 3.0, 4.0],
                ],
                dtype=np.float32,
            )
            np.save(path, raw_um)
            surface = load_measured_surface(
                path,
                height_unit="um",
                spacing_x_m=5e-6,
                spacing_y_m=5e-6,
                level="mean",
                maximum_missing_fraction=0.2,
            )
            self.assertTrue(surface.valid_mask[0, 0])
            self.assertFalse(surface.valid_mask[1, 1])
            self.assertTrue(np.all(np.isfinite(surface.height_m)))
            self.assertEqual(
                surface.metadata["preprocessing"]["invalid_policy"]["zero"],
                "zero_is_valid",
            )
            self.assertIsNone(surface.measurement_probe)
            self.assertIsNone(surface.measurement_tolerance_m)
            self.assertIsNone(surface.geometry_lower_bound_m)
            self.assertIsNone(surface.geometry_upper_bound_m)
            self.assertTrue(surface.geometry_uncertain_mask[0, 0])
            self.assertEqual(
                surface.metadata["measurement_semantics"]["status"],
                "unknown_probe",
            )
            self.assertEqual(
                surface.metadata["general_mesh_scope"], "OUT_OF_SCOPE"
            )

    def test_measured_resampling_rejects_fake_resolution(self) -> None:
        height = np.arange(25, dtype=np.float32).reshape(5, 5)
        mask = np.ones_like(height, dtype=np.bool_)
        with self.assertRaises(TerrainConfigurationError):
            resample_measured_patch(
                height,
                mask,
                source_dx_m=10e-6,
                source_dy_m=10e-6,
                target_size_x_m=40e-6,
                target_size_y_m=40e-6,
                target_dx_m=5e-6,
                target_dy_m=5e-6,
            )


if __name__ == "__main__":
    unittest.main()
