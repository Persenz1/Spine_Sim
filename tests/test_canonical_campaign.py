from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from spine_sim.cli import main as cli_main
from spine_sim.core.config import BaseCaseSpec, CampaignSpec
from spine_sim.core.versions import (
    ARRAY_MODEL_LEVEL,
    GEOMETRY_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    PARAMETER_REGISTRY_VERSION,
    PROJECT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    SOLVER_SEMANTICS_VERSION,
)
from spine_sim.examples.canonical_module import CATALOG_NAME
from spine_sim.io.results import (
    CompactResultStore,
    open_result_store,
    read_trace_table,
)
from spine_sim.io.schema import validate_canonical_summary


CALLABLE = "spine_sim.examples.canonical_module:run_case"
CASE_MODULE_VERSION = "canonical-array-case-test-1"
CAMPAIGN_MODULE_VERSION = "canonical-campaign-test-1"
TERRAIN_VERSION = "analytic-flat-wall-test-1"


def _parameters(seed: int, spacing_m: float) -> dict[str, Any]:
    return {
        "catalog": CATALOG_NAME,
        "seed": seed,
        # 从开放侧给一个极小装配间隙，让阵列 active-set seed 负责闭合接触；
        # 恰好零间隙且零初始反力只会测试 loader 自平衡，而不会测试微刺承载。
        "initial_gaps_m": [1e-6, 1e-6],
        "spacing_m": spacing_m,
        "spine_length_m": 0.004,
        "spine_diameter_m": 0.0008,
        "tip_radius_m": 50e-6,
        "young_modulus_Pa": 200e9,
        "poisson_ratio": 0.29,
        "shear_correction": 6.0 / 7.0,
        "shaft_allowable_stress_Pa": 1e12,
        "static_friction": 0.45,
        "kinetic_friction": 0.35,
        "additional_compliance_m_per_N": 0.001,
        "rebound_recovery_distance_m": 0.0002,
        "required_normal_force_N": 0.1,
        "loader_stiffness_N_per_m": 1000.0,
        "reference_force_N": 1.0,
        "reference_length_m": 0.01,
        "load_parameter": 1.0,
        "single_tolerances": {
            "gap_m": 1e-10,
            "force_N": 1e-9,
            "friction_N": 1e-9,
            "spring_N": 1e-9,
            "velocity_m_per_s": 1e-12,
            "capacity_relative": 1e-9,
            "event_fraction": 1e-10,
        },
        "array_tolerances": {
            "scaled_residual": 1e-9,
            "rank_absolute": 1e-12,
            "rank_relative": 1e-10,
            "range_residual": 1e-9,
            "stability_absolute": 1e-10,
            "stability_relative": 1e-8,
            "gap_m": 1e-10,
            "maximum_iterations": 30,
        },
    }


def _case(seed: int, spacing_m: float) -> BaseCaseSpec:
    return BaseCaseSpec(
        module="canonical_array",
        module_version=CASE_MODULE_VERSION,
        parameters=_parameters(seed, spacing_m),
        upstream_hash=f"analytic-terrain-seed-{seed}",
        tags=(
            "analytic_catalog",
            "paired_seed",
            f"spacing_m:{spacing_m}",
        ),
        terrain_version=TERRAIN_VERSION,
        geometry_version=GEOMETRY_SCHEMA_VERSION,
    )


def _campaign(*seeds: int) -> CampaignSpec:
    return CampaignSpec(
        name="canonical-analytic-smoke",
        module_version=CAMPAIGN_MODULE_VERSION,
        callable=CALLABLE,
        cases=tuple(
            _case(seed, spacing_m)
            for seed in seeds
            for spacing_m in (0.002, 0.003)
        ),
        workers=1,
        mode="small",
    )


def _write_catalog(path: Path, campaign: CampaignSpec) -> None:
    document = {
        "name": campaign.name,
        "module_version": campaign.module_version,
        "callable": campaign.callable,
        "workers": campaign.workers,
        "mode": campaign.mode,
        "cases": [
            {
                "module": case.module,
                "module_version": case.module_version,
                "parameters": case.parameters,
                "upstream_hash": case.upstream_hash,
                "tags": list(case.tags),
                "project_schema_version": case.project_schema_version,
                "model_schema_version": case.model_schema_version,
                "result_schema_version": case.result_schema_version,
                "solver_semantics_version": case.solver_semantics_version,
                "terrain_version": case.terrain_version,
                "geometry_version": case.geometry_version,
                "parameter_registry_version": case.parameter_registry_version,
            }
            for case in campaign.cases
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_cli(
    capsys: Any,
    command: str,
    catalog: Path,
    output: Path,
) -> tuple[dict[str, Any], Path]:
    assert cli_main([command, str(catalog), "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    campaign_dir = Path(report["campaign_dir"])
    assert campaign_dir.is_dir()
    return report, campaign_dir


def _assert_versioned_campaign_metadata(
    campaign_dir: Path,
    campaign: CampaignSpec,
) -> None:
    manifest = _read_json(campaign_dir / "manifest.json")
    lineage = _read_json(campaign_dir / "lineage.json")
    normalized = _read_json(campaign_dir / "config" / "normalized.json")
    normalized_campaign = CampaignSpec.from_mapping(normalized["campaign"])
    assert campaign_dir.name == campaign.campaign_id
    assert normalized["campaign_id"] == campaign.campaign_id
    assert normalized_campaign.campaign_id == campaign.campaign_id
    assert [case.case_id for case in normalized_campaign.cases] == [
        case.case_id for case in campaign.cases
    ]
    assert manifest["campaign_id"] == campaign.campaign_id
    assert manifest["schema_version"] == PROJECT_SCHEMA_VERSION
    assert manifest["model_schema_version"] == MODEL_SCHEMA_VERSION
    assert manifest["result_schema_version"] == RESULT_SCHEMA_VERSION
    assert manifest["solver_semantics_version"] == SOLVER_SEMANTICS_VERSION
    assert manifest["geometry_schema_version"] == GEOMETRY_SCHEMA_VERSION
    assert manifest["parameter_registry_version"] == PARAMETER_REGISTRY_VERSION
    assert manifest["index_format"] == "parquet"
    case_index = pq.read_table(campaign_dir / "cases.parquet").to_pylist()
    assert {row["case_id"] for row in case_index} == {
        case.case_id for case in campaign.cases
    }
    assert lineage["campaign_id"] == campaign.campaign_id
    assert set(lineage["case_lineage"]) == {
        case.case_id for case in campaign.cases
    }
    for case in campaign.cases:
        item = lineage["case_lineage"][case.case_id]
        assert item["module"] == case.module
        assert item["module_version"] == case.module_version
        assert item["config_hash"] == case.config_hash
        assert item["upstream_hash"] == case.upstream_hash
        assert item["project_schema_version"] == PROJECT_SCHEMA_VERSION
        assert item["model_schema_version"] == MODEL_SCHEMA_VERSION
        assert item["result_schema_version"] == RESULT_SCHEMA_VERSION
        assert item["solver_semantics_version"] == SOLVER_SEMANTICS_VERSION
        assert item["terrain_version"] == TERRAIN_VERSION
        assert item["geometry_version"] == GEOMETRY_SCHEMA_VERSION
        assert (
            item["parameter_registry_version"]
            == PARAMETER_REGISTRY_VERSION
        )
        assert item["normalized_input_hash"] == case.normalized_input_hash


def _assert_canonical_case_round_trip(
    campaign_dir: Path,
    case: BaseCaseSpec,
) -> dict[str, Any]:
    case_dir = campaign_dir / "paths" / case.case_id
    stored_case = BaseCaseSpec.from_mapping(_read_json(case_dir / "config.json"))
    assert stored_case.case_id == case.case_id
    summary = _read_json(case_dir / "summary.json")
    round_tripped = json.loads(json.dumps(summary))
    assert round_tripped == summary
    validate_canonical_summary(round_tripped)
    assert summary["case_id"] == case.case_id
    assert summary["normalized_input_hash"] == case.normalized_input_hash
    assert summary["model_level"] == ARRAY_MODEL_LEVEL
    assert summary["terrain_version"] == TERRAIN_VERSION
    assert summary["geometry_version"] == GEOMETRY_SCHEMA_VERSION
    assert summary["project_schema_version"] == PROJECT_SCHEMA_VERSION
    assert summary["model_schema_version"] == MODEL_SCHEMA_VERSION
    assert summary["result_schema_version"] == RESULT_SCHEMA_VERSION
    assert summary["solver_semantics_version"] == SOLVER_SEMANTICS_VERSION
    assert summary["parameter_registry_version"] == PARAMETER_REGISTRY_VERSION
    assert summary["trace_file"] == "trace.parquet"
    assert summary["trace_format"] == "parquet"
    assert summary["counts"]["n_active"] == 2
    assert summary["counts"]["n_contact"] == 2

    trace_path = case_dir / summary["trace_file"]
    assert read_trace_table(trace_path) == [
        {
            "case_id": case.case_id,
            "load_parameter": case.parameters["load_parameter"],
            "accepted": True,
            "valid": True,
            "q_C": summary["q_C"],
            "physical_backplate_pose": summary["physical_backplate_pose"],
            "total_force_N": summary["total_wrench"]["force_N"],
            "total_moment_Nm": summary["total_wrench"]["moment_Nm"],
            "counts": summary["counts"],
            "rank_status": summary["rank_status"],
            "range_status": summary["range_status"],
            "equilibrium_status": summary["equilibrium_status"],
            "quasistatic_stability": summary["quasistatic_stability"],
            "dynamic_stability": summary["dynamic_stability"],
            "per_spine": summary["per_spine"],
        }
    ]
    return summary


def test_case_identity_changes_with_input_and_every_semantic_version() -> None:
    case = _case(17, 0.002)
    assert replace(case, parameters=_parameters(19, 0.002)).case_id != case.case_id
    assert replace(case, module_version="canonical-array-case-test-2").case_id != case.case_id
    assert replace(case, terrain_version="analytic-flat-wall-test-2").case_id != case.case_id
    assert replace(case, geometry_version="contact-candidate-test-3").case_id != case.case_id
    for field_name in (
        "project_schema_version",
        "model_schema_version",
        "result_schema_version",
        "solver_semantics_version",
        "parameter_registry_version",
    ):
        changed = replace(case, **{field_name: f"{field_name}-changed"})
        assert changed.case_id != case.case_id


def test_campaign_identity_includes_callable_and_storage_mode() -> None:
    campaign = _campaign(17, 29)
    assert replace(
        campaign,
        callable="spine_sim.examples.fake_module:run_case",
    ).campaign_id != campaign.campaign_id
    assert replace(campaign, mode="formal").campaign_id != campaign.campaign_id


def test_generic_cli_runs_one_canonical_case_with_parquet_trace(
    tmp_path: Path,
    capsys: Any,
) -> None:
    source_campaign = _campaign(17, 29)
    catalog = tmp_path / "analytic_catalog.json"
    _write_catalog(catalog, source_campaign)

    report, campaign_dir = _run_cli(
        capsys, "run-case", catalog, tmp_path / "single-output"
    )
    assert report["status_counts"] == {"complete": 1}
    selected_campaign = replace(
        source_campaign,
        cases=source_campaign.cases[:1],
        workers=1,
        mode="small",
    )
    assert selected_campaign.campaign_id != source_campaign.campaign_id
    _assert_versioned_campaign_metadata(campaign_dir, selected_campaign)
    _assert_canonical_case_round_trip(
        campaign_dir, selected_campaign.cases[0]
    )


def test_shipped_canonical_example_runs_active_contact_chain(
    tmp_path: Path,
    capsys: Any,
) -> None:
    catalog = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "canonical_campaign.json"
    )
    campaign = CampaignSpec.from_mapping(_read_json(catalog))

    report, campaign_dir = _run_cli(
        capsys, "run-case", catalog, tmp_path / "example-output"
    )

    assert report["status_counts"] == {"complete": 1}
    summary = _read_json(
        campaign_dir
        / "paths"
        / campaign.cases[0].case_id
        / "summary.json"
    )
    validate_canonical_summary(summary)
    assert summary["case_id"] == campaign.cases[0].case_id
    assert summary["terrain_version"] == campaign.cases[0].terrain_version
    assert summary["counts"]["n_active"] == 2
    assert summary["counts"]["n_contact"] == 2


def test_generic_cli_runs_two_paired_seeds_and_round_trips_summaries(
    tmp_path: Path,
    capsys: Any,
) -> None:
    campaign = _campaign(17, 29)
    catalog = tmp_path / "paired_seed_catalog.json"
    _write_catalog(catalog, campaign)

    report, campaign_dir = _run_cli(
        capsys, "run-campaign", catalog, tmp_path / "campaign-output"
    )
    assert report["status_counts"] == {"complete": 4}
    _assert_versioned_campaign_metadata(campaign_dir, campaign)
    summaries = [
        _assert_canonical_case_round_trip(campaign_dir, case)
        for case in campaign.cases
    ]
    assert sorted(
        summary["parameter_provenance"]["seed"] for summary in summaries
    ) == [17, 17, 29, 29]
    assert len({summary["case_id"] for summary in summaries}) == 4
    for seed in (17, 29):
        pair = [case for case in campaign.cases if case.parameters["seed"] == seed]
        assert len(pair) == 2
        assert {case.upstream_hash for case in pair} == {
            f"analytic-terrain-seed-{seed}"
        }
        assert {case.parameters["spacing_m"] for case in pair} == {0.002, 0.003}
        assert pair[0].case_id != pair[1].case_id


def test_canonical_formal_summary_uses_compact_store_without_attachments(
    tmp_path: Path,
    capsys: Any,
) -> None:
    source = _campaign(17)
    campaign = replace(
        source,
        cases=tuple(
            replace(
                case,
                parameters={
                    **case.parameters,
                    "output": {"level": "summary"},
                },
            )
            for case in source.cases
        ),
        mode="formal",
    )
    catalog = tmp_path / "canonical_formal_summary.json"
    _write_catalog(catalog, campaign)

    report, campaign_dir = _run_cli(
        capsys, "run-campaign", catalog, tmp_path / "formal-output"
    )

    assert report["status_counts"] == {"complete": 2}
    store = open_result_store(campaign_dir)
    assert isinstance(store, CompactResultStore)
    assert not (campaign_dir / "paths").exists()
    assert [record.run_state for record in store.list_records()] == [
        "complete",
        "complete",
    ]
