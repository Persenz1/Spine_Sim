from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from spine_sim.core.versions import PARAMETER_REGISTRY_VERSION
from spine_sim.parameters import (
    Installation,
    equal_height_length_m,
    legacy_design_id,
    load_registry,
)


ROOT = Path(__file__).resolve().parents[1]
TERMINAL_INPUT = (
    ROOT
    / "docs"
    / "archive"
    / "legacy_simulation_evidence"
    / "manifests"
    / "terminal_input_selected_designs.json"
)


def _terminal_document() -> dict:
    return json.loads(TERMINAL_INPUT.read_text(encoding="utf-8"))


def _independent_legacy_id(value: dict) -> str:
    payload = json.dumps(
        {"package": value["package"], "geometry": value["geometry"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"m3_full_design_{hashlib.sha256(payload).hexdigest()[:20]}"


def test_registry_is_one_valid_machine_readable_source_with_evidence() -> None:
    path = ROOT / "src" / "spine_sim" / "parameters" / "registry.json"
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["schema_version"] == "spine-parameter-registry-v1"
    assert document["registry_version"] == PARAMETER_REGISTRY_VERSION
    assert load_registry().registry_version == PARAMETER_REGISTRY_VERSION

    for section_name in (
        "coordinate_contract",
        "candidate_axes",
        "legacy_model_baseline",
        "unclosed_parameters",
        "legacy_id_schemes",
        "paired_seed_sets",
        "terminal_presets",
        "selection_sets",
    ):
        for value in document[section_name].values():
            assert "value" in value
            assert value["evidence_status"]
            assert value["source"]

    for section_name in ("generators", "protocols"):
        for value in document[section_name].values():
            assert value["evidence_status"]
            assert value["source"]
            assert value["fields"]
            for field in value["fields"].values():
                assert "value" in field
                assert field["evidence_status"]
                assert field["source"]


def test_candidate_axes_do_not_implicitly_materialize_a_cartesian_scan() -> None:
    registry = load_registry()

    assert registry.candidate_axis("nx").value == [2, 3, 4, 5, 6]
    assert registry.candidate_axis("ny").value == [2, 3, 4, 5, 6]
    assert registry.candidate_axis("single_spine_normal_preload_N").value == 0.5
    assert registry.candidate_axis("array_total_normal_preloads_N").value == [
        0.5,
        1.0,
        2.0,
    ]
    assert registry.candidate_axis("drag_speed_m_per_s").value == 0.001
    assert registry.candidate_axis("maximum_drag_length_m").value == 0.1
    assert registry.candidate_axis("terrain_resolution_protocol_m").value == {
        "production": 10e-6,
        "final_convergence": 5e-6,
    }
    full_scan_shapes = {
        (design.canonical.nx, design.canonical.ny)
        for design in registry.generate_legacy_full_scan()
    }
    assert (3, 3) not in full_scan_shapes
    assert full_scan_shapes == {
        (2, 2),
        (2, 5),
        (5, 2),
        (3, 5),
        (5, 3),
        (4, 4),
        (6, 6),
    }


def test_named_source_generators_match_frozen_counts_and_inputs() -> None:
    registry = load_registry()
    m3a = registry.generate_source_defined_m3a()
    m3b = registry.generate_source_defined_m3b()

    assert len(m3a) == len({item.legacy_package_id for item in m3a}) == 80
    assert {item.tip_radius_m for item in m3a} == {50e-6, 100e-6}
    assert {item.rod_diameter_m for item in m3a} == {0.6e-3, 0.8e-3}
    assert {round(-math.degrees(item.fixed_pitch_rad)) for item in m3a} == {
        50,
        60,
        70,
        80,
    }
    assert {
        "rigid"
        if item.installation.mode == "rigid"
        else item.installation.stiffness_N_per_m
        for item in m3a
    } == {"rigid", 200.0, 500.0, 1000.0, 2000.0}

    assert len(m3b) == len({item.legacy_geometry_id for item in m3b}) == 28
    assert sum(item.legacy_angle_pattern == "fixed" for item in m3b) == 21
    assert sum(item.legacy_angle_pattern == "80_to_60" for item in m3b) == 6
    assert sum(item.legacy_angle_pattern == "80_to_50" for item in m3b) == 1


def test_legacy_full_scan_preserves_all_1344_id_payloads() -> None:
    designs = load_registry().generate_legacy_full_scan()

    assert len(designs) == len({item.legacy_design_id for item in designs}) == 1344
    for design in designs:
        assert design.legacy_design_id == _independent_legacy_id(
            dict(design.legacy_mapping)
        )
        assert design.legacy_design_id == legacy_design_id(design.legacy_mapping)
    assert all(
        round(-math.degrees(design.canonical.pitch_by_x_column_rad[0]))
        in {60, 70, 80}
        for design in designs
    )


def test_terminal_twelve_round_trip_against_archived_manifest() -> None:
    registry = load_registry()
    archive = _terminal_document()
    presets = {preset.legacy_design_id: preset for preset in registry.terminal_presets()}

    assert set(presets) == set(archive["selected_design_ids"])
    assert len(presets) == 12
    for legacy in archive["selected_designs"]:
        design_id = legacy["design_id"]
        imported = registry.import_legacy_design(
            legacy, protocol_id="legacy_terminal_archive"
        )
        generated = presets[design_id].design.canonical

        for field_name in (
            "nx",
            "ny",
            "spacing_m",
            "pitch_by_x_column_rad",
            "yaw_rad",
            "length_by_x_column_m",
            "tip_radius_m",
            "rod_diameter_m",
            "installation",
            "material_category",
            "legacy_angle_pattern",
            "legacy_design_id",
        ):
            assert getattr(imported.design, field_name) == getattr(
                generated, field_name
            )
        assert set(registry.protocol("legacy_terminal_archive").source) < set(
            imported.design.source
        )
        assert any(
            source.endswith("terminal_input_selected_designs.json")
            for source in imported.design.source
        )
        assert imported.design.legacy_design_id == design_id
        assert imported.report.legacy_design_id == design_id
        assert imported.report.protocol_id == "legacy_terminal_archive"
        assert imported.report.unexplained_differences == ()
        assert {field.canonical_field for field in imported.report.fields} == {
            "nx",
            "ny",
            "spacing_m",
            "tip_radius_m",
            "rod_diameter_m",
            "pitch_by_x_column_rad",
            "yaw_rad",
            "length_by_x_column_m",
            "installation",
            "legacy_design_id",
        }
        assert all(field.evidence_status for field in imported.report.fields)
        assert all(field.source for field in imported.report.fields)


def test_terminal_roles_and_selection_provenance_are_complete() -> None:
    registry = load_registry()
    archive = _terminal_document()
    presets = {preset.role_id: preset for preset in registry.terminal_presets()}

    for design_id, archived_role in archive["roles"].items():
        preset = next(
            value for value in presets.values() if value.legacy_design_id == design_id
        )
        assert preset.role == archived_role["role"]
        assert preset.mechanism == archived_role["mechanism"]
        assert preset.reason == archived_role["reason"]

    machine = registry.selection_set("machine_final_six")
    human = registry.selection_set("human_primary_four")
    optional = registry.selection_set("human_optional_fifth")
    assert machine.role_ids == ("C6", "A3", "A4", "C5", "C2", "C7")
    assert human.role_ids == ("A1", "A2", "A3", "A5")
    assert optional.role_ids == ("A4",)
    assert machine.legacy_design_ids[0] == presets["C6"].legacy_design_id
    assert machine.source != presets["C6"].source


def test_legacy_importer_converts_degrees_null_and_spring_immediately() -> None:
    registry = load_registry()
    archive = _terminal_document()
    rigid_input = archive["selected_designs"][0]
    spring_input = next(
        value
        for value in archive["selected_designs"]
        if value["package"]["spring_stiffness_N_per_m"] == 300.0
    )

    rigid = registry.import_legacy_design(rigid_input).design
    spring = registry.import_legacy_design(spring_input).design

    assert rigid.installation == Installation("rigid")
    assert rigid.pitch_by_x_column_rad == pytest.approx(
        (-math.radians(70.0),) * rigid.nx
    )
    assert spring.installation == Installation(
        "unilateral_spring", 300.0, 0.004
    )
    assert not hasattr(rigid, "fixed_angle_deg")
    assert rigid.yaw_rad == 0.0


def test_gradient_keeps_x_column_order_axis_sign_and_equal_height() -> None:
    registry = load_registry()
    c7 = next(
        preset for preset in registry.terminal_presets() if preset.role_id == "C7"
    ).design.canonical

    assert c7.mount_x_by_column_m == pytest.approx(
        (-0.01, -0.005, 0.0, 0.005, 0.01)
    )
    assert c7.pitch_by_x_column_rad == pytest.approx(
        tuple(-math.radians(value) for value in (60, 65, 70, 75, 80))
    )
    for pitch, axis in zip(
        c7.pitch_by_x_column_rad, c7.axis_by_x_column, strict=True
    ):
        assert axis == pytest.approx((math.cos(pitch), 0.0, math.sin(pitch)))
        assert axis[0] > 0.0
        assert axis[2] < 0.0

    reference_height = 0.004 * math.sin(math.radians(80.0))
    for pitch, length in zip(
        c7.pitch_by_x_column_rad, c7.length_by_x_column_m, strict=True
    ):
        assert length == pytest.approx(equal_height_length_m(pitch))
        assert length * abs(math.sin(pitch)) == pytest.approx(reference_height)
    assert c7.length_by_x_column_m[-1] == pytest.approx(0.004)


def test_protocols_and_paired_seeds_have_distinct_identities() -> None:
    registry = load_registry()
    required = {
        "legacy_runtime_default",
        "legacy_full_scan",
        "source_defined_m3a",
        "source_defined_m3b",
        "legacy_coarse_archive",
        "legacy_fine_archive",
        "legacy_terminal_archive",
        "current_final_source",
        "legacy_large_array_archive",
        "terrain_coverage_envelope",
        "legacy_cross_gripper_preset",
        "legacy_analysis",
    }
    protocols = {name: registry.protocol(name) for name in required}

    assert len({value.protocol_identity for value in protocols.values()}) == len(
        required
    )
    assert len(registry.paired_seeds_for_protocol("source_defined_m3a")) == 6
    assert registry.paired_seeds_for_protocol("source_defined_m3a") == (
        41005,
        41010,
        41015,
        41020,
        41025,
        41030,
    )
    assert len(registry.paired_seeds_for_protocol("legacy_coarse_archive")) == 15
    assert len(registry.paired_seeds_for_protocol("legacy_fine_archive")) == 50
    assert registry.paired_seeds_for_protocol("legacy_terminal_archive") == tuple(
        range(41001, 41101)
    )
    assert registry.paired_seeds_for_protocol("current_final_source") == tuple(
        range(41001, 41101)
    )
    assert registry.paired_seeds_for_protocol("legacy_large_array_archive") == tuple(
        range(52001, 52011)
    )

    terminal = protocols["legacy_terminal_archive"].values
    current = protocols["current_final_source"].values
    assert (terminal["path_length_m"], terminal["dx_m"]) == (0.01, 5e-05)
    assert (current["path_length_m"], current["dx_m"]) == (0.02, 0.0001)
    assert protocols["legacy_terminal_archive"].protocol_identity != protocols[
        "current_final_source"
    ].protocol_identity


def test_archive_and_coverage_protocol_values_remain_separate() -> None:
    registry = load_registry()
    coarse = registry.protocol("legacy_coarse_archive").values
    fine = registry.protocol("legacy_fine_archive").values
    terminal = registry.protocol("legacy_terminal_archive").values
    large = registry.protocol("legacy_large_array_archive").values
    coverage = registry.protocol("terrain_coverage_envelope").values

    assert (coarse["terrain_condition_count"], coarse["input_design_count"]) == (
        45,
        1344,
    )
    assert coarse["capture_full_paths"] is False
    assert (fine["terrain_condition_count"], fine["input_design_count"]) == (
        150,
        96,
    )
    assert fine["capture_full_paths"] is True
    assert (terminal["terrain_condition_count"], terminal["input_design_count"]) == (
        300,
        12,
    )
    assert large["array_shapes"] == [[20, 20], [40, 40], [60, 60]]
    assert large["normal_preloads_N"] == [2.0, 5.0, 10.0]
    assert large["terrain_resolution_m"] == 20e-6
    assert coverage["drag_length_m"] == 0.1
    assert coverage["rod_clearance_reserve_m"] == 0.00425
    assert coverage["terrain_resolution_m"] == 10e-6


def test_bad_legacy_identity_is_rejected_instead_of_silently_rebound() -> None:
    registry = load_registry()
    value = dict(_terminal_document()["selected_designs"][0])
    value["design_id"] = "m3_full_design_not_the_payload"

    with pytest.raises(ValueError, match="design_id"):
        registry.import_legacy_design(value)
