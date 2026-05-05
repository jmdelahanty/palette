from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.utils import run_subject_mask_batch_pipeline as mod


def test_zarr_paths_from_report_reads_unique_result_paths(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "results": [
                    {"zarr_path": "/data/a_analysis.zarr"},
                    {"zarr_path": "/data/b_analysis.zarr"},
                    {"zarr_path": "/data/a_analysis.zarr"},
                    {"not_zarr_path": "/data/ignored.zarr"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert mod._zarr_paths_from_report(report) == [
        Path("/data/a_analysis.zarr"),
        Path("/data/b_analysis.zarr"),
    ]


def test_zarr_paths_from_report_falls_back_to_plan_paths(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"plans": [{"zarr_path": "/data/planned_analysis.zarr"}]}), encoding="utf-8")

    assert mod._zarr_paths_from_report(report) == [Path("/data/planned_analysis.zarr")]


def test_zarr_paths_from_report_requires_paths(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"results": [{"error": "missing"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="did not contain any zarr_path"):
        mod._zarr_paths_from_report(report)
