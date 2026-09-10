"""Prescribed material coarse scan; execution choices do not change physics.

The configuration is a research design recipe, compiled here to the existing
CampaignSpec. Surfaces are generated lazily and reused by all designs/loads.
No candidate selection or large-array extrapolation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .core.config import BaseCaseSpec, CampaignSpec
from .core.identity import stable_hash
from .core.versions import IJMS_VERSIONS
from .runtime.backend import BackendConfig, discover_backend
from .runtime.runner import CampaignRunner
from .terrain.api import generate_terrain
from .terrain.models import MATERIAL_TERRAIN_VERSION


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "ijms_small_array.json"
TEMP_ROOT = Path("E:/Agent_Tmp_WS/ijms_small_runtime")


def read_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # These are persistent configurations. All disposable write buffers stay
    # under the user's temporary-workspace root, even for a results folder on D:.
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ijms-small-write-", dir=TEMP_ROOT) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        shutil.move(str(temporary), str(path))


def design_table(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return unique physical assemblies, with the eight references first."""
    designs: dict[str, dict[str, Any]] = {}
    mounts = [("fixed", 0.0)] + [("spring", float(k)) for k in config["spring_stiffnesses_N_per_m"]]
    reference_mounts = [("fixed", 0.0)] + [("spring", float(k)) for k in config["layout_reference_stiffnesses_N_per_m"]]

    def add(radius: float, angle: float | None, mount: tuple[str, float], layout: Sequence[int], spacing: float, *, reference=False) -> None:
        nx, ny = layout
        spine = deepcopy(config["reference_spine"])
        spine.update(tip_radius_m=radius, mount_type=mount[0], spring_stiffness_N_per_m=mount[1],
                     max_compression_m=config["spring_travel_m"] if mount[0] == "spring" else 0.0)
        array = dict(nx=nx, ny=ny, spacing_x_m=spacing, spacing_y_m=spacing,
                     tip_positions_xy_m=[[(i-(nx-1)/2)*spacing, (j-(ny-1)/2)*spacing]
                                         for j in range(ny) for i in range(nx)],
                     loaded_area_m2=nx*ny*spacing**2, spine=spine)
        if angle is None:
            array["angle_gradient_deg"] = dict(config["angle_gradient_deg"])
            array["heel_free_length_m"] = spine.pop("free_length_m")
            angle_label = "gradient60to80"
        else:
            array["theta_deg"] = angle
            angle_label = f"a{angle:g}"
        key = stable_hash(array)
        if key in designs:
            designs[key]["full_reference"] |= reference
            return
        name = f"r{radius*1e6:g}_{angle_label}_{mount[0]}{mount[1]:g}_{nx}x{ny}_p{spacing*1e3:g}"
        designs[key] = dict(name=name, array=array, full_reference=reference)

    # A balanced, immutable first group gives overnight execution useful
    # references without inventing a reduced-fidelity or selected-data stage.
    for radius in config["tip_radii_m"]:
        for mount in reference_mounts:
            add(radius, config["layout_reference_angle_deg"], mount, config["main_layout"], config["main_spacing_m"], reference=True)
    for radius in config["tip_radii_m"]:
        for angle in [*config["angles_deg"], None]:
            for mount in mounts:
                add(radius, angle, mount, config["main_layout"], config["main_spacing_m"])
    for radius in config["tip_radii_m"]:
        for mount in reference_mounts:
            for layout in config["layouts"]:
                for spacing in config["spacings_m"]:
                    add(radius, config["layout_reference_angle_deg"], mount, layout, spacing)
    return list(designs.values())


def surface_domain(config: Mapping[str, Any]) -> dict[str, Any]:
    """One small-stage field covers every centered unloaded array and drag."""
    dx = config["surface"]["resolution_m"]
    margin = config["surface"]["edge_margin_m"]
    pitch = max(config["spacings_m"])
    extent_x = max(nx-1 for nx, _ in config["layouts"])*pitch
    extent_y = max(ny-1 for _, ny in config["layouts"])*pitch
    nx = math.ceil((extent_x+2*margin+config["path"]["search_distance_m"])/dx - 1e-10)
    ny = math.ceil((extent_y+2*margin)/dx - 1e-10)
    return dict(size_x_m=nx*dx, size_y_m=ny*dx, resolution_m=dx,
                origin_xy_m=[-extent_x/2-margin, -extent_y/2-margin],
                shape_yx=[ny+1, nx+1], height_bytes_float64=(ny+1)*(nx+1)*8)


def metric_protocols(config: Mapping[str, Any]) -> dict[str, Any]:
    metric = config["metrics"]
    protocols = []
    for key, values in (("target_force_N", metric["target_forces_N"]), ("target_fraction_P", metric["target_fractions_P"])):
        for value in values:
            for window in metric["search_windows_m"]:
                protocols.append(dict(name=f"{key}_{value:g}_S{window*1e3:g}mm", **{key: value},
                                      persistence_distance_m=metric["persistence_distance_m"],
                                      search_window_m=[0., window]))
    return dict(force_quantile=metric["force_quantile"], protocols=protocols)


@dataclass(frozen=True)
class Shard:
    sample_index: int
    material_index: int
    batch_index: int
    full: bool
    items: tuple[tuple[int, float], ...]

    def name(self, config: Mapping[str, Any]) -> str:
        material = config["materials"][self.material_index]
        return f"s{self.sample_index:03d}-{material['material']}-{material['subtype']}-{'full' if self.full else 'summary'}-{self.batch_index:03d}"


def shard_schedule(config: Mapping[str, Any], designs: Sequence[Mapping[str, Any]]) -> Iterator[Shard]:
    """Interleave material conditions before progressing to another sample."""
    size = config["execution"]["shard_size"]
    for sample_index in range(config["surface"]["realizations_per_material"]):
        full_sample = sample_index in config["execution"]["full_reference_sample_indices"]
        blocks = []
        for full in (True, False):
            items = [(i, float(preload)) for i, design in enumerate(designs)
                     if bool(full_sample and design["full_reference"]) == full
                     for preload in config["preloads_N"]]
            blocks.extend((full, i//size, tuple(items[i:i+size])) for i in range(0, len(items), size))
        for full, batch_index, items in blocks:
            for material_index in range(len(config["materials"])):
                yield Shard(sample_index, material_index, batch_index, full, items)


def describe(config: Mapping[str, Any]) -> dict[str, Any]:
    designs = design_table(config)
    schedule = list(shard_schedule(config, designs))
    return dict(name=config["name"], designs=len(designs), materials=config["materials"],
                preloads_N=config["preloads_N"], realizations_per_material=config["surface"]["realizations_per_material"],
                surfaces=len(config["materials"])*config["surface"]["realizations_per_material"],
                cases=sum(len(s.items) for s in schedule), shards=len(schedule),
                full_output_cases=sum(len(s.items) for s in schedule if s.full),
                domain=surface_domain(config), metric_protocols=len(metric_protocols(config)["protocols"]),
                semantic_versions=IJMS_VERSIONS, execution_order="sample, output/block, material; same walls and placement across designs/loads")


def freeze_config(output: Path, config: Mapping[str, Any]) -> None:
    frozen = dict(config=config, semantic_versions=IJMS_VERSIONS, terrain_version=MATERIAL_TERRAIN_VERSION)
    path = output / "scan_config.json"
    if path.exists():
        if read_config(path) != frozen:
            raise ValueError("results directory contains a different scientific configuration/version; use its matching code/config or a new output directory")
    else:
        _write_json(path, frozen)


def prepare_surface(output: Path, config: Mapping[str, Any], shard: Shard) -> dict[str, Any]:
    material = config["materials"][shard.material_index]
    seed = config["surface"]["seed_start"] + shard.sample_index
    stem = f"{material['material']}-{material['subtype']}-s{shard.sample_index:03d}"
    path = output / "surfaces" / f"{stem}.npy"
    metadata_path = path.with_suffix(".json")
    domain = surface_domain(config)
    if not (path.exists() and metadata_path.exists()):
        print(f"Generating shared surface: {stem}", flush=True)
        terrain = generate_terrain(**material, seed=seed, mode=config["surface"]["mode"], backend="cpu",
                                   **{key: domain[key] for key in ("size_x_m", "size_y_m", "resolution_m")})
        metadata = deepcopy(dict(terrain.metadata))
        metadata["measurement_probe"] = getattr(terrain, "measurement_probe", None)
        metadata["measurement_tolerance_m"] = getattr(terrain, "measurement_tolerance_m", None)
        metadata["array_storage"] = dict(dtype="float64", source_dtype="float32", origin_xy_m=domain["origin_xy_m"])
        metadata["sample_scope"] = "synthetic_realization; measured source specimens are not independent per seed"
        metadata["actual_height_statistics"] = dict(mean_m=float(np.mean(terrain.height, dtype=np.float64)), rms_about_mean_m=float(np.std(terrain.height, dtype=np.float64)),
                                                   minimum_m=float(np.min(terrain.height)), maximum_m=float(np.max(terrain.height)))
        path.parent.mkdir(parents=True, exist_ok=True)
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ijms-small-surface-", dir=TEMP_ROOT) as directory:
            temporary = Path(directory) / path.name
            # float64 avoids one converted full-size height copy in each worker.
            height = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64, shape=terrain.height.shape)
            height[:] = terrain.height
            height.flush()
            del height
            shutil.move(str(temporary), str(path))
        # Publishing metadata last means interruption during generation is
        # recoverable by recreating that same deterministic realization.
        _write_json(metadata_path, metadata)
    return dict(kind="heightfield", path=str(path.resolve()), metadata_path=str(metadata_path.resolve()),
                dx_m=domain["resolution_m"], dy_m=domain["resolution_m"], origin_xy_m=domain["origin_xy_m"])


def build_campaign(config: Mapping[str, Any], designs: Sequence[Mapping[str, Any]], shard: Shard,
                   surface: Mapping[str, Any], *, workers: int = 1) -> CampaignSpec:
    material = config["materials"][shard.material_index]
    cases = []
    for design_index, preload in shard.items:
        design = designs[design_index]
        parameters = dict(parameter_status=config["parameter_status"], design_name=design["name"],
                          array=deepcopy(design["array"]), surface=dict(surface), load=dict(mode="total_force", value=preload),
                          path=deepcopy(config["path"]), solver=deepcopy(config["solver"]), metrics=metric_protocols(config),
                          sample=dict(surface_seed=config["surface"]["seed_start"]+shard.sample_index,
                                      placement_seed=0, placement_xy_m=config["surface"]["placement_xy_m"], split="train",
                                      material=material["material"], subtype=material["subtype"], sample_index=shard.sample_index),
                          output=dict(level="full" if shard.full else "summary",
                                      progress_interval_s=config["execution"].get("progress_interval_s", 30.)))
        cases.append(BaseCaseSpec(module="ijms_guided_array", module_version=IJMS_VERSIONS["solver_semantics_version"],
                                  parameters=parameters, tags=("small_array_material_coarse", design["name"], material["subtype"]),
                                  terrain_version=MATERIAL_TERRAIN_VERSION, **IJMS_VERSIONS))
    return CampaignSpec(name=f"{config['name']}-{shard.name(config)}", module_version="ijms-small-array-campaign-1",
                        callable="spine_sim.ijms:run_case", cases=tuple(cases), mode="formal", workers=workers)


def status(output: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Read SQLite/per-file summaries; distinguish execution from physics."""
    execution, physics = Counter(), Counter()
    wall_time_s = 0.
    newest = None
    for root in sorted((output / "results").glob("*")):
        database = root / "case_summaries.sqlite3"
        if database.exists():
            with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
                rows = [json.loads(row[0]) for row in connection.execute("SELECT summary_json FROM case_summary")]
        else:
            rows = [read_config(path) for path in root.glob("paths/*/summary.json")]
        for row in rows:
            execution[row["run_state"]] += 1
            physics[row.get("status", "NO_PHYSICS_RESULT")] += 1
            wall_time_s += row.get("wall_time_s", 0.)
            timestamp = row.get("completed_at_utc")
            if timestamp and (newest is None or timestamp > newest):
                newest = timestamp
    expected = describe(config)
    return dict(planned_cases=expected["cases"], recorded_cases=sum(execution.values()),
                not_recorded_cases=expected["cases"]-sum(execution.values()), execution_states=dict(execution),
                physics_statuses=dict(physics), summed_case_wall_time_s=wall_time_s, latest_result_utc=newest,
                prepared_surfaces=len(list((output / "surfaces").glob("*.npy"))),
                prepared_shards=len(list((output / "campaigns").glob("*.json"))),
                note="not_recorded includes running and not started; incomplete physics is not a successful full path")


def execute(config: Mapping[str, Any], output: Path, *, prepare_only: bool, workers: int,
            max_shards: int | None = None, start_shard: int = 0) -> dict[str, int]:
    output = output.resolve()
    freeze_config(output, config)
    designs = design_table(config)
    backend = None if prepare_only else discover_backend(BackendConfig(preference="cpu", allow_gpu=False))
    processed = 0
    for index, shard in enumerate(shard_schedule(config, designs)):
        if index < start_shard:
            continue
        if max_shards is not None and processed >= max_shards:
            break
        surface = prepare_surface(output, config, shard)
        # Workers are an execution override; canonical campaign JSON stays the
        # same when the next resume uses another machine or worker count.
        campaign = build_campaign(config, designs, shard, surface)
        raw = asdict(campaign)
        config_path = output / "campaigns" / f"{shard.name(config)}.json"
        if config_path.exists():
            if read_config(config_path) != json.loads(json.dumps(raw)):
                raise ValueError(f"existing campaign differs: {config_path}")
        else:
            _write_json(config_path, raw)
        print(f"{'Prepared' if prepare_only else 'Run/resume'} shard {index}: {shard.name(config)} ({len(campaign.cases)} cases)", flush=True)
        if not prepare_only:
            runner = CampaignRunner(campaign, output / "results" / shard.name(config), backend)
            runner.prepare(raw)
            records = runner.run(resume=True, workers=workers)
            errors = sum(record.run_state != "complete" for record in records)
            print(f"Shard {index}: {len(records)} recorded, {errors} execution errors (see persisted summaries)", flush=True)
        processed += 1
    return dict(processed_shards=processed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2)//2)))
    parser.add_argument("--max-shards", type=int, help="execution prefix only; does not change the registered 64-realization design")
    parser.add_argument("--start-shard", type=int, default=0, help="zero-based operational partition; use the same output directory")
    parser.add_argument("--dry-run", action="store_true", help="describe the plan without generating surfaces or running cases")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.dry_run:
        print(json.dumps(describe(config), ensure_ascii=False, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is used")
    if args.workers < 1 or args.start_shard < 0 or (args.max_shards is not None and args.max_shards < 1):
        parser.error("workers/max-shards must be positive and start-shard non-negative")
    if args.action == "status":
        frozen = args.output_dir / "scan_config.json"
        if frozen.exists():
            config = read_config(frozen)["config"]
        result = status(args.output_dir.resolve(), config)
    else:
        result = execute(config, args.output_dir, prepare_only=args.action == "prepare", workers=args.workers,
                         max_shards=args.max_shards, start_shard=args.start_shard)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
