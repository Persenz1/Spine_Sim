from __future__ import annotations

import csv
from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "data" / "catalog" / "public_topography_sources.csv"
LICENSES = ROOT / "data" / "catalog" / "licenses.csv"
HASHES = ROOT / "data" / "catalog" / "file_hashes.sha256"

EXPECTED_FIELDS = (
    "dataset_id",
    "title",
    "source",
    "doi",
    "publication",
    "material_family",
    "material_subclass",
    "surface_finish",
    "condition_or_wear",
    "manufacturer_or_batch",
    "specimen_count",
    "patch_count",
    "field_of_view_x",
    "field_of_view_y",
    "lateral_resolution_x",
    "lateral_resolution_y",
    "vertical_resolution",
    "instrument",
    "file_format",
    "coordinate_unit",
    "height_unit",
    "raw_or_processed",
    "contains_absolute_scale",
    "contains_height_map",
    "contains_point_cloud",
    "contains_mesh",
    "license",
    "access_method",
    "direct_download",
    "login_required",
    "author_request_required",
    "suitability_for_calibration",
    "suitability_for_pipeline_test",
    "known_limitations",
    "evidence",
    "checked_date",
)


def test_public_catalog_schema_traceability_and_unique_ids() -> None:
    with CATALOG.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == EXPECTED_FIELDS
        rows = list(reader)

    assert len(rows) >= 19
    assert len({row["dataset_id"] for row in rows}) == len(rows)
    assert {"sandpaper", "red brick", "concrete"} <= {
        row["material_family"] for row in rows
    }
    for row in rows:
        assert all(value.strip() for value in row.values())
        assert row["checked_date"] == "2026-07-29"
        assert "http" in row["evidence"]


def test_license_and_hash_manifests_are_well_formed() -> None:
    with LICENSES.open(encoding="utf-8", newline="") as stream:
        licenses = list(csv.DictReader(stream))
    assert licenses
    assert len({row["dataset_id"] for row in licenses}) == len(licenses)
    assert all(row["license"] and row["evidence"] for row in licenses)

    entries = [
        line
        for line in HASHES.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert entries
    for line in entries:
        digest, relative_path = line.split(maxsplit=1)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert relative_path.startswith("data/raw/")
