from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import zarr

from fisheye.utils import list_detect_group_fallbacks as mod


def _make_recording_dir(root: Path, name: str) -> Path:
    rec = root / name
    (rec / "raw").mkdir(parents=True, exist_ok=True)
    (rec / "zarr").mkdir(parents=True, exist_ok=True)
    h5_path = rec / "raw" / f"{name}.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["session_uuid"] = f"{name}_session"
        h5.attrs["camera_id"] = "2010093"
    return rec


def _make_analysis_zarr(recording_dir: Path, name: str, *, resolved_group: str) -> Path:
    zarr_path = recording_dir / "zarr" / f"{name}_analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_1"
    detect_run = detect_parent.create_group("detect_1")
    detect_run.create_array("frame_counts", data=np.array([1, 1, 0], dtype=np.int32), overwrite=True)
    quality_parent = detect_run.create_group("quality_reports")
    quality_parent.attrs["latest"] = "detect_quality_1"
    quality = quality_parent.create_group("detect_quality_1")
    quality.attrs["quality_score"] = {"grade": "A", "overall_score": 99.0}
    quality.attrs["detection_quality_summary"] = {"clean_percentage": 98.0}

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_1"
    refined_run = refined_parent.create_group("refined_1")
    if resolved_group in {"filtered", "interpolated", "manual"}:
        refined_run.create_group(resolved_group)
    return zarr_path


def test_collect_group_rows_finds_raw_fallback(tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    rec_raw = _make_recording_dir(root, "rec_raw")
    _make_analysis_zarr(rec_raw, "rec_raw", resolved_group="raw")

    rec_interp = _make_recording_dir(root, "rec_interp")
    _make_analysis_zarr(rec_interp, "rec_interp", resolved_group="interpolated")

    rows = mod._collect_group_rows(  # noqa: SLF001
        roots=[root],
        recursive=True,
        requested_use="analysis",
        target_group="raw",
        registry=None,
    )
    assert len(rows) == 1
    assert rows[0]["recording_dir"].endswith("rec_raw")
    assert rows[0]["resolved_detect_group"] == "raw"


def test_main_json_output(tmp_path: Path, capsys) -> None:
    root = tmp_path / "recordings"
    rec_raw = _make_recording_dir(root, "rec_raw")
    _make_analysis_zarr(rec_raw, "rec_raw", resolved_group="raw")

    rc = mod.main([str(root), "--recursive", "--group", "raw", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["resolved_detect_group"] == "raw"
