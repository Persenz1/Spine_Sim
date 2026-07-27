"""Deterministic balanced-coverage design for the gated M3 round-one screen."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from spine_sim.contact import SpineParameters
from spine_sim.core.identity import identity

from .models import AngleLayout, M3_MODULE_VERSION


REPRESENTATIVE_SHAPES = (
    (2, 2),
    (2, 5),
    (5, 2),
    (3, 5),
    (5, 3),
    (4, 4),
    (6, 6),
)
SPACINGS_M = (4e-3, 5e-3, 6e-3)
FIXED_ANGLES_DEG = (50.0, 60.0, 70.0, 80.0)
GRADIENT_LAYOUTS = (
    AngleLayout.GRADIENT_80_TO_60,
    AngleLayout.GRADIENT_80_TO_50,
)


def build_candidate_pool(
    parameter_packs: Sequence[Mapping[str, Any]],
    *,
    shapes: Sequence[tuple[int, int]] = REPRESENTATIVE_SHAPES,
    spacings_m: Sequence[float] = SPACINGS_M,
    fixed_angles_deg: Sequence[float] = FIXED_ANGLES_DEG,
) -> list[dict[str, Any]]:
    """Build the allowed pool without taking a terrain/seed Cartesian product."""

    pool: list[dict[str, Any]] = []
    for pack in parameter_packs:
        pack_id = str(pack["parameter_pack_id"])
        spine = SpineParameters.from_mapping(pack["spine"])
        for nx, ny in shapes:
            for spacing_m in spacings_m:
                for angle_deg in fixed_angles_deg:
                    row = {
                        "parameter_pack_id": pack_id,
                        "nx": int(nx),
                        "ny": int(ny),
                        "spacing_m": float(spacing_m),
                        "angle_layout": AngleLayout.FIXED.value,
                        "fixed_angle_deg": float(angle_deg),
                        "tip_radius_m": spine.tip_radius_m,
                        "diameter_m": spine.diameter_m,
                        "axial_mode": spine.axial_mode.value,
                        "spring_stiffness_n_m": spine.spring_stiffness_n_m,
                    }
                    row["hardware_candidate_id"] = identity(
                        "hardware_candidate",
                        row,
                        module_version=M3_MODULE_VERSION,
                    )
                    pool.append(row)
                for layout in GRADIENT_LAYOUTS:
                    row = {
                        "parameter_pack_id": pack_id,
                        "nx": int(nx),
                        "ny": int(ny),
                        "spacing_m": float(spacing_m),
                        "angle_layout": layout.value,
                        "fixed_angle_deg": None,
                        "tip_radius_m": spine.tip_radius_m,
                        "diameter_m": spine.diameter_m,
                        "axial_mode": spine.axial_mode.value,
                        "spring_stiffness_n_m": spine.spring_stiffness_n_m,
                    }
                    row["hardware_candidate_id"] = identity(
                        "hardware_candidate",
                        row,
                        module_version=M3_MODULE_VERSION,
                    )
                    pool.append(row)
    unique = {row["hardware_candidate_id"]: row for row in pool}
    return [unique[key] for key in sorted(unique)]


def _tokens(row: Mapping[str, Any]) -> frozenset[str]:
    angle = (
        f"fixed_{row['fixed_angle_deg']:g}"
        if row["angle_layout"] == "fixed"
        else str(row["angle_layout"])
    )
    stiffness = (
        "rigid"
        if row["spring_stiffness_n_m"] is None
        else f"{float(row['spring_stiffness_n_m']):g}"
    )
    main = {
        f"pack={row['parameter_pack_id']}",
        f"shape={row['nx']}x{row['ny']}",
        f"nx={row['nx']}",
        f"ny={row['ny']}",
        f"spacing={float(row['spacing_m']):g}",
        f"layout={row['angle_layout']}",
        f"angle={angle}",
        f"tip={float(row['tip_radius_m']):g}",
        f"diameter={float(row['diameter_m']):g}",
        f"axial={row['axial_mode']}",
        f"stiffness={stiffness}",
    }
    interactions = {
        f"installation_mode*scale={row['axial_mode']}*{row['nx']}x{row['ny']}",
        f"stiffness*spacing={stiffness}*{float(row['spacing_m']):g}",
        f"angle*direction={angle}*{row['nx']}x{row['ny']}",
        f"gradient*nx={row['angle_layout']}*{row['nx']}",
        f"diameter*stiffness={float(row['diameter_m']):g}*{stiffness}",
        f"tip*pack={float(row['tip_radius_m']):g}*{row['parameter_pack_id']}",
    }
    return frozenset(main | interactions)


def select_balanced_candidates(
    pool: Sequence[Mapping[str, Any]],
    target_count: int,
) -> list[dict[str, Any]]:
    """Greedy balanced coverage with deterministic maximin tie breaking."""

    if target_count < 1:
        raise ValueError("target_count must be positive")
    if target_count > len(pool):
        raise ValueError("target_count cannot exceed the unique candidate pool")
    candidates = [dict(row) for row in pool]
    token_map = {
        row["hardware_candidate_id"]: _tokens(row)
        for row in candidates
    }
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    remaining = {
        row["hardware_candidate_id"]: row
        for row in candidates
    }
    while len(selected) < target_count:
        best_id: str | None = None
        best_score: tuple[float, float, float, str] | None = None
        for candidate_id, row in remaining.items():
            tokens = token_map[candidate_id]
            novelty = float(sum(counts[token] == 0 for token in tokens))
            balance = float(sum(1.0 / (1.0 + counts[token]) for token in tokens))
            if selected:
                max_overlap = max(
                    len(tokens & token_map[item["hardware_candidate_id"]])
                    / len(tokens | token_map[item["hardware_candidate_id"]])
                    for item in selected
                )
                distance = 1.0 - max_overlap
            else:
                distance = 1.0
            score = (novelty, balance, distance, candidate_id)
            if best_score is None or score > best_score:
                best_score = score
                best_id = candidate_id
        assert best_id is not None
        chosen = remaining.pop(best_id)
        selected.append(chosen)
        counts.update(token_map[best_id])
    return selected


def level_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    fields = (
        "parameter_pack_id",
        "nx",
        "ny",
        "spacing_m",
        "angle_layout",
        "fixed_angle_deg",
        "tip_radius_m",
        "diameter_m",
        "axial_mode",
        "spring_stiffness_n_m",
    )
    output: dict[str, dict[str, int]] = {}
    materialized = list(rows)
    for field in fields:
        counts = Counter(str(row[field]) for row in materialized)
        output[field] = dict(sorted(counts.items()))
    return output


def screening_gate_status(
    *,
    full_chain_manifest_present: bool,
    m2_formal_round1_completed: bool,
    m2_parameter_packs_approved: bool,
    explicit_m3_round1_approval: bool,
) -> dict[str, Any]:
    blockers = []
    if not full_chain_manifest_present:
        blockers.append("full_chain_frozen_manifest.json absent")
    if not m2_formal_round1_completed:
        blockers.append("M2 formal round one incomplete")
    if not m2_parameter_packs_approved:
        blockers.append("M2 parameter packs not user-approved")
    if not explicit_m3_round1_approval:
        blockers.append("explicit approval '开始 M3 第一轮筛选' absent")
    return {
        "formal_m3_round1_allowed": not blockers,
        "blockers": blockers,
    }
