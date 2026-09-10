"""Runnable IJMS approach--preload--drag protocol and campaign adapter.

All inputs are SI except the explicitly named ``theta_deg``/``yaw_deg``.
The JSON examples are numerical demonstrations, not calibrated device designs.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from scipy.optimize import brentq

from .continuous_geometry import HeightFieldSurface, PlaneSurface
from .contact_path import check_sphere_path
from .guided_array import GuidedArray, PathSettings, PathState, THEORY_VERSION, SOLVER_VERSION
from .guided_rod import GuidedRod, GuidedRodParameters
from .metrics import SpineMetricInput, compute_array_counts, integrate_path_resistance
from .research_protocols import EstablishmentProtocol, LoadProtocol, evaluate_establishment
from .runtime.runner import CaseOutput, RunContext


@lru_cache(maxsize=1)
def _shared_heightfield(path, dx, dy, origin, mask_path, metadata_path):
    """Immutable per-worker field, reused by cases on the same saved surface."""
    height = np.load(Path(path), mmap_mode="r", allow_pickle=False)
    mask = np.load(Path(mask_path), mmap_mode="r", allow_pickle=False) if mask_path else None
    surface = HeightFieldSurface(height, dx, dy, origin, valid_mask=mask)
    surface.metadata = (json.loads(Path(metadata_path).read_text(encoding="utf-8"))
                        if metadata_path else {"source": str(path), "mode": "saved_heightfield"})
    return surface


def build_surface(config: Mapping[str, Any], sample: Mapping[str, Any] | None = None):
    """Continuous geometry from an analytic plane, sampled field or terrain API."""
    raw = dict(config)
    kind = raw.pop("kind")
    raw.pop("version", None)
    if sample and "surface_seed" in sample:
        raw["seed"] = sample["surface_seed"]
    seed = int(raw.pop("seed", 0))
    if kind == "plane":
        return PlaneSurface(**raw)
    origin = tuple(raw.pop("origin_xy_m", (0., 0.)))
    if kind == "heightfield":
        metadata_path = raw.pop("metadata_path", None)
        metadata = raw.pop("metadata", None)
        mask_path = raw.pop("valid_mask_path", None)
        if "path" in raw:
            path = str(Path(raw.pop("path")).resolve())
            dx = float(raw.pop("dx_m"))
            dy = float(raw.pop("dy_m", dx))
            if not raw and metadata is None:
                return _shared_heightfield(path, dx, dy, origin, mask_path, metadata_path)
            height = np.load(Path(path), mmap_mode="r", allow_pickle=False)
            raw.update(dx_m=dx, dy_m=dy)
        else:
            height = np.asarray(raw.pop("height_m"), float)
            raw.setdefault("dy_m", raw["dx_m"])
        if mask_path:
            raw["valid_mask"] = np.load(Path(mask_path), mmap_mode="r", allow_pickle=False)
        surface = HeightFieldSurface(height, origin_xy_m=origin, **raw)
        surface.metadata = (json.loads(Path(metadata_path).read_text(encoding="utf-8"))
                            if metadata_path else dict(metadata or {}))
        return surface
    if kind == "material":
        from .terrain.api import generate_terrain
        terrain = generate_terrain(seed=seed, **raw)
        surface = HeightFieldSurface.from_terrain(terrain, origin_xy_m=origin)
        surface.metadata = dict(terrain.metadata, material=terrain.material, subtype=terrain.subtype,
                                seed=terrain.seed, measurement_probe=terrain.measurement_probe,
                                measurement_tolerance_m=terrain.measurement_tolerance_m,
                                geometry_uncertain=bool(terrain.geometry_uncertain_mask is not None
                                                        and np.any(terrain.geometry_uncertain_mask)),
                                geometry_bounds_provided=terrain.geometry_lower_bound_m is not None)
        return surface
    dx, dy = float(raw["dx_m"]), float(raw.get("dy_m", raw["dx_m"]))
    ny, nx = map(int, raw["shape"])
    x, y = np.meshgrid(np.arange(nx)*dx, np.arange(ny)*dy)
    angle = np.deg2rad(raw.get("orientation_deg", 0.))
    u, v = np.cos(angle)*x+np.sin(angle)*y, -np.sin(angle)*x+np.cos(angle)*y
    if kind == "sinusoidal":
        phase = raw.get("phase_rad", 0.)
        height = raw["amplitude_m"] * np.sin(2*np.pi*u/raw["wavelength_m"] + phase)
        if raw.get("cross_amplitude_m", 0):
            height += raw["cross_amplitude_m"] * np.cos(2*np.pi*v/raw["cross_wavelength_m"])
    elif kind == "gaussian":
        # Rotated anisotropic Gaussian spectrum. RMS is normalised per finite
        # realization; correlation parameters describe the spectral filter.
        noise = np.random.default_rng(seed).normal(size=(ny, nx))
        kx, ky = np.meshgrid(2*np.pi*np.fft.fftfreq(nx, dx), 2*np.pi*np.fft.fftfreq(ny, dy))
        ku, kv = np.cos(angle)*kx+np.sin(angle)*ky, -np.sin(angle)*kx+np.cos(angle)*ky
        spectrum = np.exp(-0.25*((ku*raw["correlation_x_m"])**2+(kv*raw["correlation_y_m"])**2))
        height = np.fft.ifft2(np.fft.fft2(noise)*spectrum).real
        height -= height.mean()
        height *= raw["rms_height_m"] / max(float(height.std()), 1e-30)
    else:
        raise ValueError(f"unknown surface kind: {kind}")
    return HeightFieldSurface(height, dx, dy, origin)


def build_rods(config: Mapping[str, Any]) -> list[GuidedRod]:
    """Construct co-planar outermost tips and geometrically compatible guides.

    Grid spacings refer to unloaded tip locations. Different angles/lengths can
    change guide spacing. common_mouth_height_m instead determines each length.
    """
    nx, ny = int(config.get("nx", 1)), int(config.get("ny", 1))
    positions = config.get("tip_positions_xy_m")
    if positions is None:
        positions = [(i*config.get("spacing_x_m", 0.), j*config.get("spacing_y_m", 0.))
                     for j in range(ny) for i in range(nx)]
    defaults = dict(config["spine"])
    theta = config.get("theta_deg", 60.)
    gradient = config.get("angle_gradient_deg")
    if gradient is not None:
        if len(positions) != nx*ny or nx < 2:
            raise ValueError("angle gradient needs a rectangular array with at least two X columns")
        if "theta_deg" in config or "common_mouth_height_m" in config:
            raise ValueError("angle_gradient_deg determines the angles and heel-referenced mouth height")
        theta = np.linspace(float(gradient["heel"]), float(gradient["toe"]), nx)
    theta_array = np.asarray(theta, float)
    theta_values = (np.full(len(positions), float(theta_array)) if theta_array.ndim == 0
                    else np.broadcast_to(theta_array, (ny, nx)).ravel())
    if len(theta_values) != len(positions):
        raise ValueError("angle array must match the explicit tip positions")
    overrides = {int(row["index"]): row for row in config.get("per_spine", [])}
    if set(overrides)-set(range(len(positions))):
        raise ValueError("per_spine index lies outside the declared array")
    gradient_height = None
    if gradient is not None:
        heel_length = float(config.get("heel_free_length_m", defaults.get("free_length_m", 0.004)))
        heel_radius = float(overrides.get(0, {}).get("tip_radius_m", defaults["tip_radius_m"]))
        gradient_height = heel_radius+heel_length*np.sin(np.deg2rad(float(gradient["heel"])))
    rods = []
    for i, xy in enumerate(positions):
        local = dict(defaults)
        override = dict(overrides.get(i, {}))
        override.pop("index", None)
        theta_i = np.deg2rad(override.pop("theta_deg", theta_values[i]))
        if not 0 < theta_i <= np.pi/2:
            raise ValueError("guided installation theta_deg must be in (0,90]")
        yaw = np.deg2rad(override.pop("yaw_deg", config.get("yaw_deg", 0.)))
        local.update(override)
        if gradient_height is not None:
            if "free_length_m" in overrides.get(i, {}) or "theta_deg" in overrides.get(i, {}):
                raise ValueError("gradient assembly derives each row's angle and exposed length")
            local["free_length_m"] = (gradient_height-local["tip_radius_m"])/np.sin(theta_i)
        if "common_mouth_height_m" in config:
            if "free_length_m" in defaults or "free_length_m" in overrides.get(i, {}):
                raise ValueError("common_mouth_height_m determines free_length_m; omit explicit lengths")
            local["free_length_m"] = (config["common_mouth_height_m"]-local["tip_radius_m"])/np.sin(theta_i)
        p = GuidedRodParameters(**local)
        if p.mount_type == "spring" and p.max_compression_m <= 0:
            raise ValueError("IJMS guided spring requires positive max_compression_m")
        axis = np.array([np.cos(theta_i)*np.cos(yaw), np.cos(theta_i)*np.sin(yaw), -np.sin(theta_i)])
        center0 = np.array([xy[0], xy[1], p.tip_radius_m])
        rods.append(GuidedRod(p, center0 - p.free_length_m*axis, axis))
    return rods


def approach_position(model: GuidedArray, start_xy_m) -> np.ndarray:
    """First zero-force sphere contact under rigid vertical approach."""
    roots = []
    if isinstance(model.surface, HeightFieldSurface):
        surface_min = float(np.nanmin(model.surface.height_m))
        surface_max = float(np.nanmax(model.surface.height_m))
    for rod in model.rods:
        center0 = rod.evaluate(rod.zero_state(), derivatives=False).center_m
        center0[:2] += start_xy_m
        if isinstance(model.surface, PlaneSurface):
            surface = model.surface
            if surface.normal[2] <= 0:
                raise ValueError("vertical preload protocol needs an upward-facing surface")
            roots.append((rod.parameters.tip_radius_m - (center0-surface.point_m)@surface.normal)/surface.normal[2])
            continue
        surface = model.surface
        below = surface_min - center0[2] - rod.parameters.tip_radius_m
        above = surface_max - center0[2] + 2*rod.parameters.tip_radius_m
        def gap(z):
            center = center0 + np.array([0., 0., z])
            query = surface.query_sphere(center, rod.parameters.tip_radius_m)
            if query.gap_m is None:
                raise ValueError(f"initial sphere lies outside usable surface: {query.status}")
            return query.gap_m
        roots.append(brentq(gap, below, above, xtol=1e-13))
    return np.array([*start_xy_m, max(roots)])


def _row(state: PathState, phase: str, origin_x: float, detailed: bool, settings: PathSettings):
    data = state.diagnostics
    per_spine = data.get("per_spine", [])
    counts = compute_array_counts([
        SpineMetricInput.from_force(wall_force_N=item["force_N"], contact_normal=item["normal"],
                                   geometric=True, signed_gap_m=item["gap_m"], engagement=None,
                                   force_tolerance_N=max(state.preload_N/len(state.modes), 1e-4)*settings.residual_tolerance)
        for item in per_spine], gap_tolerance_m=settings.contact_tolerance_m) if per_spine else None
    positive_loads = [max(p["P_N"], 0.) for p in per_spine]
    result = dict(phase=phase, X_m=float(state.position_m[0]),
                  search_distance_m=float(state.position_m[0]-origin_x),
                  position_m=state.position_m.tolist(), preload_N=state.preload_N,
                  T_N=float(-state.forces_N[:, 0].sum()), P_N=float(state.forces_N[:, 2].sum()),
                  L_N=float(state.forces_N[:, 1].sum()), energy_J=state.energy_J,
                  accepted=True, valid=True,
                  counts=asdict(counts) if counts else None,
                  spring_energy_J=sum(p["spring_energy_J"] for p in per_spine),
                  bending_energy_J=sum(p["bending_energy_J"] for p in per_spine),
                  mean_compression_m=(float(np.mean([p["compression_m"] for p in per_spine])) if per_spine else None),
                  hard_stop_fraction=(sum("HARDSTOP" in p["mode"] for p in per_spine)/len(per_spine)
                                      if per_spine else None),
                  maximum_positive_P_share=(max(positive_loads)/sum(positive_loads)
                                            if sum(positive_loads) > 0 else None),
                  max_travel_utilization=max((p["travel_utilization"] for p in per_spine), default=0.),
                  max_stress_upper_Pa=max((p["max_stress_upper_Pa"] for p in per_spine), default=0.),
                  max_stress_utilization=(max(p["stress_utilization"] for p in per_spine)
                                          if per_spine and all(p["stress_utilization"] is not None for p in per_spine)
                                          else None),
                  **{k: v for k, v in data.items() if k not in {"per_spine"}})
    if detailed:
        result["modes"] = list(state.modes)
        result["per_spine"] = per_spine
        result["rod_coordinates"] = [x.tolist() for x in state.rod_coordinates]
    return result


def evaluate_protocols(x, force, *, preload, distance, metric_config):
    """Reduce all declared protocols before summary mode discards the path."""
    protocols, windows = {}, {}
    for index, specification in enumerate(metric_config.get("protocols", [])):
        item = dict(specification)
        name = str(item.pop("name", f"protocol_{index}"))
        fraction = item.pop("target_fraction_P", None)
        if fraction is not None:
            if "target_force_N" in item:
                raise ValueError("choose absolute or preload-relative target in each protocol")
            item["target_force_N"] = fraction*preload
        item.setdefault("search_window_m", (0., distance))
        item["search_window_m"] = tuple(item["search_window_m"])
        if not 0 <= item["search_window_m"][0] < item["search_window_m"][1] <= distance:
            raise ValueError("protocol search window must lie inside the declared drag path")
        if name in protocols:
            raise ValueError("protocol names must be unique")
        protocol = EstablishmentProtocol(**item)
        value = asdict(evaluate_establishment(x, force, accepted=[True]*len(x),
                                             valid=[True]*len(x), protocol=protocol))
        value["protocol"] = asdict(protocol)
        value["target_basis"] = "fraction_P" if fraction is not None else "absolute_force"
        protocols[name] = value
        window = protocol.search_window_m
        key = f"{window[0]:.12g}:{window[1]:.12g}"
        if key not in windows:
            windows[key] = dict(window_m=list(window), **asdict(integrate_path_resistance(
                x, force, external_normal_preload_N=preload, accepted=[True]*len(x), valid=[True]*len(x),
                window_m=window, force_quantile=metric_config.get("force_quantile", 0.1))))
    return protocols, windows


def _weighted_path_statistics(rows, windows):
    """Linear-in-distance summaries, retaining incomplete field coverage."""
    x = np.array([row["search_distance_m"] for row in rows], dtype=float)
    fields = ("n_active", "n_contact", "n_share_normal", "n_share_local_normal", "load_sharing_index",
              "spring_energy_J", "bending_energy_J", "mean_compression_m", "hard_stop_fraction",
              "maximum_positive_P_share", "max_stress_utilization")
    statistics = {}
    for key, item in windows.items():
        low, high = item["window_m"]
        dx = np.diff(x)
        left, right = np.maximum(x[:-1], low), np.minimum(x[1:], high)
        width = np.maximum(right-left, 0.)
        denominators = np.where(dx > 0, dx, 1.)
        record = {}
        for field in fields:
            raw = [((row.get("counts") or {}).get(field) if field.startswith("n_") or field == "load_sharing_index"
                    else row.get(field)) for row in rows]
            values = np.array([np.nan if v is None else v for v in raw], dtype=float)
            valid = (dx > 0) & (width > 0) & np.isfinite(values[:-1]) & np.isfinite(values[1:])
            total_length = float(width[valid].sum())
            slope = np.diff(values)/denominators
            integral = float(np.sum(width[valid]*(values[:-1][valid]
                              + .5*slope[valid]*(left[valid]+right[valid]-2*x[:-1][valid]))))
            complete = bool(np.isclose(total_length, high-low, rtol=1e-10, atol=0.))
            record[field] = dict(mean=integral/(high-low) if complete else None,
                                 observed_mean=integral/total_length if total_length > 0 else None,
                                 coverage=total_length/(high-low))
        statistics[key] = record
    return statistics


def simulate(parameters: Mapping[str, Any], *, progress=None) -> CaseOutput:
    started = perf_counter()
    rods = build_rods(parameters["array"])
    surface = build_surface(parameters["surface"], parameters.get("sample"))
    path = parameters["path"]
    start_xy = np.asarray(path.get("start_xy_m", (0., 0.)), float)
    start_xy = start_xy + np.asarray(parameters.get("sample", {}).get("placement_xy_m", (0., 0.)))
    settings_data = dict(parameters.get("solver", {}))
    settings_data.setdefault("y_locked_m", float(start_xy[1]))
    stability_setting = settings_data.pop("stability", "final")
    settings = PathSettings(**settings_data)
    model = GuidedArray(rods, surface, settings)
    load = LoadProtocol(**parameters["load"])
    area = float(parameters["array"].get("loaded_area_m2", 0.))
    if load.mode == "pressure" and area <= 0:
        raise ValueError("pressure comparison requires actual loaded_area_m2")
    preload = load.total_preload_N(area_m2=area, n_nominal=len(rods))
    distance, max_step = float(path["search_distance_m"]), float(path["max_step_m"])
    if min(distance, max_step) <= 0:
        raise ValueError("search distance and step must be positive")
    if isinstance(surface, HeightFieldSurface):
        max_step = min(max_step, 0.25*min(surface.dx_m, surface.dy_m))
    detailed = parameters.get("output", {}).get("level", "trace") == "full"
    output_level = parameters.get("output", {}).get("level", "trace")
    if output_level not in {"summary", "trace", "full"}:
        raise ValueError("output.level must be summary, trace or full")
    initial_q = approach_position(model, start_xy)
    state = model.unloaded(initial_q)
    rows = [_row(state, "first_contact", start_xy[0], detailed, settings)]
    events = [dict(kind="FIRST_CONTACT", phase="approach", X_m=float(start_xy[0]), Z_m=float(initial_q[2]))]
    status, failure = "COMPLETED", None
    stage_start = perf_counter()

    def advance(target, phase):
        nonlocal state, status, failure
        current = state.preload_N if phase == "preload" else state.position_m[0]
        step = target-current
        minimum = preload * 1e-7 if phase == "preload" else settings.minimum_step_m
        while target-current > max(1e-14, minimum*1e-4):
            if len(rows) >= settings.max_path_steps:
                status = "NUMERICAL_STEP_BUDGET"
                failure = dict(phase=phase, accepted_steps=len(rows))
                events.append(dict(kind=status, **failure))
                return False
            proposed = min(target, current+step)
            trial = model.solve(state, state.position_m[0] if phase == "preload" else proposed,
                                proposed if phase == "preload" else preload)
            if progress is not None:
                progress(dict(phase=phase, target=float(proposed), step=float(step),
                              status=trial.status, residual=trial.residual,
                              modes=trial.state.modes if trial.state is not None else None))
            next_state = trial.state
            changed = next_state is not None and next_state.modes != state.modes
            geometry_event = False
            swept_failure = None
            if next_state is not None and trial.status == "ACCEPTED":
                for i, rod in enumerate(rods):
                    swept = check_sphere_path(surface, state.centers_m[i], next_state.centers_m[i],
                                              rod.parameters.tip_radius_m,
                                              previous_normal=state.normals[i], normal=next_state.normals[i],
                                              previous_feature=state.features[i] if state.features else None,
                                              feature=next_state.features[i],
                                              penetration_tolerance_m=settings.swept_contact_tolerance_m)
                    geometry_event |= swept.geometry_event
                    if not swept.admissible:
                        swept_failure = dict(spine_index=i, geometry_status=swept.status,
                                             clearance_m=swept.clearance_m)
                        break
            # Locate the first change of physical mode in path coordinates.
            # Newton iterations remain private trial states. A final min-step
            # transition is a finite rebalance, never a fake interpolated force.
            event_step = preload*1e-5 if phase == "preload" else settings.event_tolerance_m
            event_refine = (changed or geometry_event) and step > event_step and state.preload_N > 0
            if trial.status != "ACCEPTED" or event_refine or swept_failure:
                if step > minimum * 1.01:
                    step *= 0.5
                    continue
                if event_refine and trial.status == "ACCEPTED" and not swept_failure:
                    pass
                else:
                    status = "UNRESOLVED_CONTACT_PATH" if swept_failure else trial.status
                    failure = dict(phase=phase, attempted_parameter=float(proposed),
                                   residual=trial.residual, **(swept_failure or trial.details))
                    events.append(dict(kind=status, **failure))
                    return False
            if changed or geometry_event:
                events.append(dict(kind="CONTACT_MODE_CHANGE" if changed else "GEOMETRY_FEATURE_CHANGE", phase=phase,
                                   X_m=float(next_state.position_m[0]), preload_N=next_state.preload_N,
                                   changes=[dict(spine_index=i, before=before, after=after)
                                            for i, (before, after) in enumerate(zip(state.modes, next_state.modes))
                                            if before != after],
                                   location_parameter="preload_N" if phase == "preload" else "X_m",
                                   location_uncertainty=float(step)))
            state = next_state
            rows.append(_row(state, phase, start_xy[0], detailed, settings))
            current = state.preload_N if phase == "preload" else state.position_m[0]
            step = min(target-current, step*2)
        return True

    for target in np.linspace(0, preload, int(path.get("preload_steps", 8))+1)[1:]:
        if not advance(float(target), "preload"):
            break
    preload_time = perf_counter()-stage_start
    stage_start = perf_counter()
    if status == "COMPLETED":
        # The exact preloaded internal state starts drag and is retained on every
        # subsequent call. This row makes the declared [0,Smax] window explicit.
        rows.append(_row(state, "drag", start_xy[0], detailed, settings))
        end = float(start_xy[0]+distance)
        while state.position_m[0] < end-1e-14:
            if not advance(min(end, float(state.position_m[0]+max_step)), "drag"):
                break
    drag_time = perf_counter()-stage_start
    drag = [row for row in rows if row["phase"] == "drag"]
    x, force = [r["search_distance_m"] for r in drag], [r["T_N"] for r in drag]
    # Empty/incomplete paths remain explicit; they are not zero-load designs.
    metrics = None
    establishment = None
    metric_config = parameters.get("metrics", {})
    establishments, window_metrics = evaluate_protocols(
        x, force, preload=preload, distance=distance, metric_config=metric_config)
    if drag:
        metrics = asdict(integrate_path_resistance(x, force, external_normal_preload_N=preload,
                                                  accepted=[True]*len(x), valid=[True]*len(x),
                                                  window_m=(0., distance),
                                                  force_quantile=metric_config.get("force_quantile", 0.1)))
        target = metric_config.get("target_force_N")
        if "target_fraction_P" in metric_config:
            if target is not None:
                raise ValueError("choose target_force_N or target_fraction_P")
            target = metric_config["target_fraction_P"]*preload
        if target is not None:
            protocol = EstablishmentProtocol(target, metric_config["persistence_distance_m"], (0., distance),
                                              metric_config.get("allowed_below_fraction", 0.),
                                              metric_config.get("max_contiguous_below_m", 0.))
            establishment = asdict(evaluate_establishment(x, force, accepted=[True]*len(x),
                                                           valid=[True]*len(x), protocol=protocol))
    stability = {"status": "NOT_REQUESTED"}
    if stability_setting == "final" and state.preload_N > 0:
        from .guided_stability import assess_stick_stability
        stability = assess_stick_stability(model, state)
    elif stability_setting not in {"final", "off"}:
        raise ValueError("solver.stability must be final or off")
    summary = dict(theory_version=THEORY_VERSION, solver_version=SOLVER_VERSION,
                   model_level="guided_common_backplate_finite_rod", status=status,
                   completed=status == "COMPLETED", failure=failure,
                   nominal_spines=len(rods), total_preload_N=preload,
                   load_protocol=asdict(load), loaded_area_m2=area,
                   initial_contact_position_m=initial_q.tolist(),
                   final_position_m=state.position_m.tolist(),
                   completed_search_m=float(state.position_m[0]-start_xy[0]),
                   declared_search_m=distance, effective_max_step_m=max_step,
                   metrics=metrics, establishment=establishment,
                   establishments=establishments, window_metrics=window_metrics,
                   path_statistics=_weighted_path_statistics(drag, window_metrics),
                   preload_state=(next((row for row in reversed(rows) if row["phase"] == "preload"), rows[0])),
                   surface_metadata=dict(getattr(surface, "metadata", {})),
                   needle_parameters=[asdict(r.parameters) for r in rods],
                   max_lateral_motion_m=max(abs(r["position_m"][1]-start_xy[1]) for r in rows),
                   final_counts=rows[-1].get("counts"), final_stability=stability,
                   final_total_force_N=state.forces_N.sum(axis=0).tolist(),
                   max_equilibrium_residual=max((r.get("residual", 0.) for r in rows), default=0.),
                   max_travel_utilization=max(r["max_travel_utilization"] for r in rows),
                   max_stress_upper_Pa=max(r["max_stress_upper_Pa"] for r in rows),
                   max_stress_utilization=(max(r["max_stress_utilization"] for r in rows[1:])
                                           if len(rows)>1 and all(r["max_stress_utilization"] is not None for r in rows[1:])
                                           else None),
                   accumulated_energy_residual_J=sum(r.get("incremental_energy_residual_J", 0.) for r in rows
                                                    if r["phase"] != "first_contact" and not
                                                    (r["phase"] == "drag" and r["search_distance_m"] == 0.)),
                   accepted_steps=len(rows), event_count=len(events),
                   events=events if output_level == "summary" else None,
                   sample=dict(parameters.get("sample", {})),
                   units={"length":"m", "force":"N", "energy":"J", "angle":"rad"},
                   coordinates="+x drag, +z away from wall; T=-sum(fx), P=sum(fz), L=sum(fy)",
                   numerical_method="finite rod + incremental material motion + mixed complementarity; direct small-array / sparse large-array linear solves",
                   parameter_status=parameters.get("parameter_status", "uncalibrated"))
    if output_level == "full":
        summary["final_per_spine"] = state.diagnostics.get("per_spine", [])
        summary["continuation_state"] = state.snapshot()
    return CaseOutput(summary=summary, trace_rows=rows if output_level != "summary" else [],
                      events=events if output_level != "summary" else [], validation={"status": "completed" if status == "COMPLETED" else "incomplete",
                                                 "max_equilibrium_residual": summary["max_equilibrium_residual"]},
                      stage_times_s={"preload":preload_time, "drag":drag_time,
                                     "total":perf_counter()-started})


def run_case(parameters: Mapping[str, Any], context: RunContext) -> CaseOutput:
    """Existing CampaignRunner callable; mechanics execute on CPU."""
    if context.backend.get("selected") == "cuda":
        raise ValueError("IJMS finite-rod solver uses CPU; pass --backend cpu")
    interval = float(parameters.get("output", {}).get("progress_interval_s", 0.))
    last_report, started = 0., perf_counter()
    def report(value):
        nonlocal last_report
        now = perf_counter()
        if now-last_report >= interval:
            print(json.dumps(dict(case_id=context.case_id, elapsed_s=now-started, **value)), flush=True)
            last_report = now
    output = simulate(parameters, progress=report if interval > 0 else None)
    output.summary.update(case_id=context.case_id, normalized_input_hash=context.normalized_input_hash,
                          project_schema_version=context.project_schema_version,
                          model_schema_version=context.model_schema_version,
                          result_schema_version=context.result_schema_version,
                          solver_semantics_version=context.solver_semantics_version,
                          geometry_version=context.geometry_version,
                          terrain_version=context.terrain_version,
                          parameter_registry_version=context.parameter_registry_version)
    return output
