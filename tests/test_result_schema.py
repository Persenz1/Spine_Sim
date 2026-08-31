from __future__ import annotations

import pytest

from spine_sim.core.versions import ARRAY_MODEL_LEVEL, PROJECT_SCHEMA_VERSION
from spine_sim.io.schema import CanonicalResultMetadata, validate_canonical_summary


def metadata() -> CanonicalResultMetadata:
    return CanonicalResultMetadata(
        case_id="case_x",
        normalized_input_hash="a" * 64,
        model_level=ARRAY_MODEL_LEVEL,
        terrain_version="analytic-1",
        geometry_version="geometry-2",
        parameter_provenance={"protocol": "fixture"},
        units={"force": "N", "length": "m", "angle": "rad"},
        frames=({"name": "wall", "convention": "right_handed"},),
        assumptions=("quasistatic",),
        omissions=("mass_and_damping",),
        applicability=("rigid_backplate",),
        cannot_answer=("impact",),
    )


def test_array_schema_requires_rank_range_and_stability_separately() -> None:
    summary = {
        **metadata().as_dict(),
        "physical_state": "STICK",
        "numerical_state": "CONVERGED",
        "model_state": "CLOSED",
        "residuals": {"scaled_norm": 0.0},
        "tolerances": {"scaled_residual": 1e-8},
        "per_spine": [],
        "rank_status": "FULL_RANK",
        "range_status": "COMPATIBLE",
        "equilibrium_status": "SOLVED",
        "quasistatic_stability": "NO_FREE_MODE",
        "dynamic_stability": "OUT_OF_SCOPE",
    }
    validate_canonical_summary(summary)
    summary["project_schema_version"] = "bogus-project-schema"
    with pytest.raises(ValueError, match="project schema version"):
        validate_canonical_summary(summary)
    summary["project_schema_version"] = PROJECT_SCHEMA_VERSION
    summary["parameter_registry_version"] = "bogus-parameter-registry"
    with pytest.raises(ValueError, match="parameter registry version"):
        validate_canonical_summary(summary)
    summary["parameter_registry_version"] = metadata().parameter_registry_version
    del summary["range_status"]
    with pytest.raises(ValueError, match="range_status"):
        validate_canonical_summary(summary)


def test_metadata_rejects_invented_model_level() -> None:
    values = metadata().as_dict()
    values["model_level"] = "M3"
    with pytest.raises(ValueError, match="unsupported model_level"):
        CanonicalResultMetadata(**values)
