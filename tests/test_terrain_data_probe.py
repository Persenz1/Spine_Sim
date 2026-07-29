from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "terrain_data_probe.py"
SPEC = importlib.util.spec_from_file_location("terrain_data_probe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
terrain_data_probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = terrain_data_probe
SPEC.loader.exec_module(terrain_data_probe)


def _write_hirox(path: Path, matrix: np.ndarray) -> None:
    lines = [
        "Measured Date,2016/11/15 13:48",
        "Calibration,1.25μm/pxl",
        "Height Unit,μm",
        f"X size,{matrix.shape[1]}",
        f"Y size,{matrix.shape[0]}",
    ]
    rows = [",".join(str(value) for value in row) for row in matrix]
    path.write_text("\n".join(lines + rows) + "\n", encoding="utf-8-sig")


def test_hirox_csv_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    values = np.array([[0.0, 1.0, 2.0], [3.0, np.nan, 5.0]])
    _write_hirox(path, values)

    header, actual = terrain_data_probe.read_hirox_csv(path)
    summary = terrain_data_probe.summarize_hirox(path, header, actual)

    assert actual.shape == (2, 3)
    assert header.lateral_calibration_um_per_pixel == pytest.approx(1.25)
    assert summary["physical_node_span_m"] == pytest.approx(
        {"x": 2.5e-6, "y": 1.25e-6}
    )
    assert summary["missing_value_ratio"] == pytest.approx(1.0 / 6.0)
    assert summary["minimum_height_m"] == pytest.approx(0.0)
    assert summary["maximum_height_m"] == pytest.approx(5e-6)
    assert summary["mean_height_m"] == pytest.approx(2.2e-6)


def test_hirox_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "\n".join(
            [
                "Measured Date,unknown",
                "Calibration,1.0um/pxl",
                "Height Unit,um",
                "X size,4",
                "Y size,2",
                "1,2,3",
                "4,5,6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        terrain_data_probe.EvidenceError,
        match="does not match header",
    ):
        terrain_data_probe.read_hirox_csv(path)


def test_hirox_unknown_height_unit_is_rejected() -> None:
    text = "\n".join(
        [
            "Measured Date,unknown",
            "Calibration,1.0um/pxl",
            "Height Unit,nm",
            "X size,2",
            "Y size,2",
        ]
    )
    with pytest.raises(
        terrain_data_probe.EvidenceError,
        match="unsupported Hirox height unit",
    ):
        terrain_data_probe.parse_hirox_header_text(text)


def test_download_metadata_accumulates_files(tmp_path: Path) -> None:
    p240 = terrain_data_probe.SANDPAPER_FILES["P240.csv"]
    readme = terrain_data_probe.SANDPAPER_FILES["ReadMe.txt"]

    terrain_data_probe._record_download_metadata(tmp_path, "P240.csv", p240)
    terrain_data_probe._record_download_metadata(tmp_path, "ReadMe.txt", readme)

    metadata = json.loads(
        (tmp_path / "source_metadata.json").read_text(encoding="utf-8")
    )
    assert set(metadata["downloaded_files"]) == {"P240.csv", "ReadMe.txt"}
    assert metadata["license"] == "CC BY 4.0"
