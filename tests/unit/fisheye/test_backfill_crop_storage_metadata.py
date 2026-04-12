from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from fisheye.utils import backfill_crop_storage_metadata as mod


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype

    def __getitem__(self, key):
        return self._data[key]


class _FakeGroup:
    def __init__(self, children: dict[str, Any] | None = None) -> None:
        self._children: dict[str, Any] = children or {}
        self.attrs: dict[str, Any] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        child = _FakeGroup()
        self._children[name] = child
        return child

    def create_array(self, name: str, data, **_kwargs) -> _FakeArray:
        array = _FakeArray(np.asarray(data))
        self._children[name] = array
        return array

    def get(self, name: str) -> Any:
        return self._children.get(name)

    def group_keys(self):
        return [key for key, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):
        return self._children.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str) -> Any:
        return self._children[key]


def _patch_scan(monkeypatch, mapping: dict[Path, _FakeGroup]) -> None:
    ordered_paths = list(mapping.keys())
    monkeypatch.setattr(mod, "_iter_zarr", lambda *_args, **_kwargs: iter(ordered_paths))

    def _open_group(path: str, mode: str = "r") -> _FakeGroup:  # noqa: ARG001
        return mapping[Path(path)]

    monkeypatch.setattr(mod.zarr, "open_group", _open_group)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _make_crop_run(
    parent: _FakeGroup,
    name: str,
    *,
    frame_indices: np.ndarray,
    bbox_norm_coords: np.ndarray,
    roi_coordinates_full: np.ndarray,
    roi_images: np.ndarray | None = None,
    roi_size: list[int] | None = None,
    created_at_utc: str | None = None,
    status: str | None = None,
) -> _FakeGroup:
    run = parent.create_group(name)
    run.create_array("frame_indices", data=frame_indices)
    run.create_array("bbox_norm_coords", data=bbox_norm_coords)
    run.create_array("roi_coordinates_full", data=roi_coordinates_full)
    if roi_images is not None:
        run.create_array("roi_images", data=roi_images)
    if roi_size is not None:
        run.attrs["roi_size"] = roi_size
    if created_at_utc is not None:
        run.attrs["created_at_utc"] = created_at_utc
    if status is not None:
        run.attrs["status"] = status
    run.attrs["detection_source_path"] = "detect_runs/detect_001"
    run.attrs["detection_source_type"] = "detect"
    run.attrs["source_detect_run"] = "detect_001"
    run.attrs["parameters"] = {"pad": 4}
    return run


def test_main_apply_backfills_materialized_crop_run_metadata(tmp_path: Path, capsys, monkeypatch) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "analysis"
    root.attrs["total_frames"] = 4

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    run = _make_crop_run(
        crop_parent,
        "crop_001",
        frame_indices=np.array([0, 2, 2], dtype=np.int32),
        bbox_norm_coords=np.zeros((3, 4), dtype=np.float32),
        roi_coordinates_full=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        roi_images=np.zeros((3, 16, 20), dtype=np.uint8),
        created_at_utc="2026-03-01T00:00:00+00:00",
    )

    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "crop_runs_scanned: 1" in out
    assert "storage_mode_updates: 1" in out
    assert "frame_counts_backfills: 1" in out
    assert "detection_indices_backfills: 1" in out
    assert "updated_runs: 1" in out
    assert "updated_parent_groups: 1" in out

    assert run.attrs["crop_storage_mode"] == "materialized"
    assert run.attrs["roi_size"] == [16, 20]
    assert run.attrs["crop_signature"]["roi_size"] == [16, 20]
    assert run.attrs["crop_signature"]["detection_source_path"] == "detect_runs/detect_001"
    np.testing.assert_array_equal(run["frame_counts"][:], np.array([1, 0, 2, 0], dtype=np.int32))
    np.testing.assert_array_equal(run["detection_indices"][:], np.array([0, 1, 2], dtype=np.int32))
    assert crop_parent.attrs["latest"] == "crop_001"
    assert crop_parent.attrs["latest_materialized"] == "crop_001"
    assert crop_parent.attrs["latest_any"] == "crop_001"


def test_main_apply_repairs_parent_latest_pointers_for_mixed_mode_runs(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "analysis"
    root.attrs["total_frames"] = 5

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_002"

    run_materialized = _make_crop_run(
        crop_parent,
        "crop_001",
        frame_indices=np.array([0, 1], dtype=np.int32),
        bbox_norm_coords=np.zeros((2, 4), dtype=np.float32),
        roi_coordinates_full=np.array([[1, 2], [3, 4]], dtype=np.int32),
        roi_images=np.zeros((2, 12, 12), dtype=np.uint8),
        created_at_utc="2026-03-01T00:00:00+00:00",
    )
    run_geometry = _make_crop_run(
        crop_parent,
        "crop_002",
        frame_indices=np.array([2, 4], dtype=np.int32),
        bbox_norm_coords=np.zeros((2, 4), dtype=np.float32),
        roi_coordinates_full=np.array([[5, 6], [7, 8]], dtype=np.int32),
        roi_size=[14, 14],
        created_at_utc="2026-03-01T00:01:00+00:00",
    )

    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "storage_mode_updates: 2" in out
    assert "parent_pointer_updates: 1" in out
    assert "issue_targets: 1" in out

    assert run_materialized.attrs["crop_storage_mode"] == "materialized"
    assert run_geometry.attrs["crop_storage_mode"] == "geometry_only"
    assert crop_parent.attrs["latest"] == "crop_001"
    assert crop_parent.attrs["latest_materialized"] == "crop_001"
    assert crop_parent.attrs["latest_any"] == "crop_002"


def test_main_apply_reports_but_does_not_rewrite_existing_mismatched_frame_counts(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "analysis"
    root.attrs["total_frames"] = 2

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    run = _make_crop_run(
        crop_parent,
        "crop_001",
        frame_indices=np.array([0, 1], dtype=np.int32),
        bbox_norm_coords=np.zeros((2, 4), dtype=np.float32),
        roi_coordinates_full=np.array([[1, 2], [3, 4]], dtype=np.int32),
        roi_images=np.zeros((2, 8, 8), dtype=np.uint8),
        created_at_utc="2026-03-01T00:00:00+00:00",
    )
    run.create_array("frame_counts", data=np.array([2, 0], dtype=np.int32))

    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "issue_targets: 1" in out
    assert "frame_counts_mismatch_existing" in out

    np.testing.assert_array_equal(run["frame_counts"][:], np.array([2, 0], dtype=np.int32))
    assert run.attrs["crop_storage_mode"] == "materialized"


def test_main_default_skips_non_completed_crop_runs(tmp_path: Path, capsys, monkeypatch) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "analysis"
    root.attrs["total_frames"] = 3

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_002"

    _make_crop_run(
        crop_parent,
        "crop_001",
        frame_indices=np.array([0, 1], dtype=np.int32),
        bbox_norm_coords=np.zeros((2, 4), dtype=np.float32),
        roi_coordinates_full=np.array([[1, 2], [3, 4]], dtype=np.int32),
        roi_images=np.zeros((2, 8, 8), dtype=np.uint8),
        created_at_utc="2026-03-01T00:00:00+00:00",
        status="failed",
    )
    completed = _make_crop_run(
        crop_parent,
        "crop_002",
        frame_indices=np.array([2], dtype=np.int32),
        bbox_norm_coords=np.zeros((1, 4), dtype=np.float32),
        roi_coordinates_full=np.array([[5, 6]], dtype=np.int32),
        roi_images=np.zeros((1, 8, 8), dtype=np.uint8),
        created_at_utc="2026-03-01T00:01:00+00:00",
        status="completed",
    )

    _patch_scan(monkeypatch, {zarr_path: root})

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "crop_runs_scanned: 1" in out
    assert "skipped_non_completed_runs: 1" in out
    assert "issue_targets: 0" in out
    assert "crop_runs/crop_001 [status=failed]" in out

    assert "crop_storage_mode" not in crop_parent["crop_001"].attrs
    assert completed.attrs["crop_storage_mode"] == "materialized"


def test_main_writes_jsonl_log_with_run_and_zarr_events(tmp_path: Path, capsys, monkeypatch) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "analysis"
    root.attrs["total_frames"] = 3

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_002"

    _make_crop_run(
        crop_parent,
        "crop_001",
        frame_indices=np.array([0, 1], dtype=np.int32),
        bbox_norm_coords=np.zeros((2, 4), dtype=np.float32),
        roi_coordinates_full=np.array([[1, 2], [3, 4]], dtype=np.int32),
        roi_images=np.zeros((2, 8, 8), dtype=np.uint8),
        created_at_utc="2026-03-01T00:00:00+00:00",
        status="failed",
    )
    _make_crop_run(
        crop_parent,
        "crop_002",
        frame_indices=np.array([2], dtype=np.int32),
        bbox_norm_coords=np.zeros((1, 4), dtype=np.float32),
        roi_coordinates_full=np.array([[5, 6]], dtype=np.int32),
        roi_images=np.zeros((1, 8, 8), dtype=np.uint8),
        created_at_utc="2026-03-01T00:01:00+00:00",
        status="completed",
    )

    _patch_scan(monkeypatch, {zarr_path: root})
    log_dir = tmp_path / "logs"

    rc = mod.main([str(zarr_path), "--log-dir", str(log_dir)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Log file:" in out

    log_files = sorted(log_dir.glob("backfill_crop_storage_metadata_*.jsonl"))
    assert len(log_files) == 1

    rows = _read_jsonl(log_files[0])
    events = [row["event"] for row in rows]
    assert events[0] == "run_start"
    assert "crop_run_skipped_non_completed" in events
    assert "crop_run_checked" in events
    assert "crop_parent_checked" in events
    assert "zarr_checked" in events
    assert events[-1] == "run_end"

    skipped_row = next(row for row in rows if row["event"] == "crop_run_skipped_non_completed")
    assert skipped_row["crop_run"] == "crop_001"
    assert skipped_row["run_status"] == "failed"

    zarr_row = next(row for row in rows if row["event"] == "zarr_checked")
    assert zarr_row["crop_runs_scanned"] == 1
    assert zarr_row["skipped_non_completed_runs"] == 1
    assert zarr_row["would_modify"] is True

    end_row = rows[-1]
    assert end_row["status"] == "ok"
    assert end_row["crop_runs_scanned"] == 1
    assert end_row["skipped_non_completed_runs"] == 1


def test_main_uses_direct_crop_parent_metadata_when_root_consolidated_is_stale(
    tmp_path: Path,
    capsys,
) -> None:
    zarr_path = tmp_path / "rec_training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["zarr_purpose"] = "training"
    root.attrs["total_frames"] = 2

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop_parent.attrs["latest_materialized"] = "crop_001"
    crop_parent.attrs["latest_any"] = "crop_001"

    run = crop_parent.create_group("crop_001")
    run.attrs["crop_storage_mode"] = "materialized"
    run.attrs["roi_size"] = [8, 8]
    run.attrs["detection_source_path"] = "detect_runs/detect_001"
    run.attrs["detection_source_type"] = "detect"
    run.attrs["source_detect_run"] = "detect_001"
    run.attrs["crop_signature"] = {
        "signature_version": 1,
        "detection_source_path": "detect_runs/detect_001",
        "detection_source_type": "detect",
        "detection_selection_policy": None,
        "source_detect_run": "detect_001",
        "source_refined_run": None,
        "roi_size": [8, 8],
        "parameter_source": None,
        "parameters_hash": None,
    }
    run.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32))
    run.create_array("bbox_norm_coords", data=np.zeros((2, 4), dtype=np.float32))
    run.create_array("roi_coordinates_full", data=np.array([[1, 2], [3, 4]], dtype=np.int32))
    run.create_array("roi_images", data=np.zeros((2, 8, 8), dtype=np.uint8))
    run.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32))
    run.create_array("detection_indices", data=np.array([0, 1], dtype=np.int32))

    root_meta_path = zarr_path / "zarr.json"
    root_meta = json.loads(root_meta_path.read_text(encoding="utf-8"))
    root_meta["consolidated_metadata"]["metadata"]["crop_runs"]["attributes"].pop("latest_materialized", None)
    root_meta["consolidated_metadata"]["metadata"]["crop_runs"]["attributes"].pop("latest_any", None)
    root_meta["consolidated_metadata"]["metadata"]["crop_runs/crop_001"]["attributes"].pop(
        "crop_storage_mode",
        None,
    )
    root_meta_path.write_text(json.dumps(root_meta, indent=2), encoding="utf-8")

    rc = mod.main([str(zarr_path)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "storage_mode_updates: 0" in out
    assert "parent_pointer_updates: 0" in out
