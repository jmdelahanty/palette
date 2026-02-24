from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.utils import export_detect_quality_overview as mod


PNG_BYTES = b"\x89PNG\r\n\x1a\nFAKEPNG"


def _make_zarr_with_artifact(
    root: Path,
    name: str,
    *,
    zarr_use: str = "training",
    latest_run: str = "refined_detect_1",
    include_artifact: bool = True,
) -> Path:
    zarr_path = root / name / "zarr" / f"{name}_{zarr_use}.zarr"
    group = zarr.open_group(str(zarr_path), mode="w")
    group.attrs["zarr_purpose"] = zarr_use

    refined_parent = group.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = latest_run
    refined_run = refined_parent.create_group(latest_run)
    if include_artifact:
        vis = refined_run.create_group("visualizations")
        vis.create_array(mod.ARTIFACT_NAME, data=np.frombuffer(PNG_BYTES, dtype=np.uint8), chunks=(len(PNG_BYTES),))
    return zarr_path


def test_export_detect_quality_overview_writes_png(tmp_path: Path) -> None:
    zarr_path = _make_zarr_with_artifact(tmp_path, "rec_one")
    output_dir = tmp_path / "exports"

    rc = mod.main([str(zarr_path), "--output-dir", str(output_dir)])
    assert rc == 0

    expected_path = output_dir / f"{zarr_path.stem}__refined_detect_1__{mod.ARTIFACT_NAME}.png"
    assert expected_path.exists()
    assert expected_path.read_bytes() == PNG_BYTES


def test_export_detect_quality_overview_list_mode_only_lists(tmp_path: Path) -> None:
    zarr_path = _make_zarr_with_artifact(tmp_path, "rec_list")
    output_dir = tmp_path / "exports"
    report = tmp_path / "report.json"

    rc = mod.main([str(zarr_path), "--output-dir", str(output_dir), "--list", "--json-report", str(report)])
    assert rc == 0
    assert not output_dir.exists()

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "list"
    assert payload["summary"]["listed"] == 1
    row = payload["rows"][0]
    assert row["status"] == "listed"
    assert row["reason"] == "list_mode"


def test_export_detect_quality_overview_recursive_filters_zarr_use(tmp_path: Path) -> None:
    _make_zarr_with_artifact(tmp_path, "rec_training", zarr_use="training")
    _make_zarr_with_artifact(tmp_path, "rec_analysis", zarr_use="analysis")
    output_dir = tmp_path / "exports"
    report = tmp_path / "report.json"

    rc = mod.main(
        [
            str(tmp_path),
            "--recursive",
            "--zarr-use",
            "training",
            "--output-dir",
            str(output_dir),
            "--json-report",
            str(report),
        ]
    )
    assert rc == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["scanned"] == 1
    assert payload["summary"]["exported"] == 1
    exported = sorted(output_dir.glob("*.png"))
    assert len(exported) == 1
    assert "rec_training" in exported[0].name


def test_export_detect_quality_overview_view_mode_does_not_write_files(
    monkeypatch, tmp_path: Path
) -> None:
    zarr_path = _make_zarr_with_artifact(tmp_path, "rec_view")
    output_dir = tmp_path / "exports"
    report = tmp_path / "report.json"
    viewed_calls: list[tuple[int, str]] = []

    def _fake_view(png_bytes: bytes, *, title: str) -> None:
        viewed_calls.append((len(png_bytes), title))

    monkeypatch.setattr(mod, "_view_png_bytes", _fake_view)

    rc = mod.main(
        [
            str(zarr_path),
            "--view",
            "--output-dir",
            str(output_dir),
            "--json-report",
            str(report),
        ]
    )
    assert rc == 0
    assert viewed_calls
    assert viewed_calls[0][0] == len(PNG_BYTES)
    assert "refined_detect_1" in viewed_calls[0][1]
    assert not output_dir.exists()

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "view"
    assert payload["summary"]["viewed"] == 1
    assert payload["summary"]["exported"] == 0
    row = payload["rows"][0]
    assert row["status"] == "viewed"
    assert row["reason"] == "shown"
