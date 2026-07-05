from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fisheye.utils import backfill_keypoint_heading_fields as mod
from fisheye.utils.backfill_keypoint_heading_fields import _backfill_heading_columns


class _FakeArray:
    def __init__(self, data: np.ndarray, *, chunks: tuple[int, ...] | None = None) -> None:
        self._data = np.array(data, copy=True)
        self.chunks = chunks or ((max(1, int(self._data.shape[0])),) if self._data.ndim else (1,))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self._data.shape)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        self._data[key] = value


class _FakeGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self._children: dict[str, object] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        group = _FakeGroup()
        self._children[name] = group
        return group

    def create_array(
        self,
        name: str,
        *,
        data=None,
        shape=None,
        chunks=None,
        dtype=None,
        fill_value=None,
        overwrite: bool = False,
    ) -> _FakeArray:
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        if data is None:
            dtype_obj = np.dtype(dtype) if dtype is not None else np.float64
            data = np.full(shape, fill_value, dtype=dtype_obj)
        array = _FakeArray(np.asarray(data), chunks=chunks)
        self._children[name] = array
        return array

    def get(self, name: str, default=None):
        return self._children.get(name, default)

    def keys(self):
        return self._children.keys()

    def group_keys(self):
        return [name for name, value in self._children.items() if isinstance(value, _FakeGroup)]

    def __getitem__(self, name: str):
        return self._children[name]

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def __delitem__(self, name: str) -> None:
        del self._children[name]


def _patch_scan(monkeypatch, mapping: dict[Path, _FakeGroup]) -> None:
    ordered_paths = list(mapping.keys())
    monkeypatch.setattr(mod, "_iter_zarr", lambda *_args, **_kwargs: iter(ordered_paths))

    def _open_group(path: str, mode: str = "r") -> _FakeGroup:  # noqa: ARG001
        return mapping[Path(path)]

    monkeypatch.setattr(mod.zarr, "open_group", _open_group)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _heading_spec(enabled: bool = True) -> dict[str, object]:
    return {
        "version": 1,
        "enabled": enabled,
        "origin": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
        "direction_from": {"op": "keypoint", "label": "swim_bladder"},
        "direction_to": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
        "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"],
    }


def _pose_schema(*, include_metadata: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "traditional_v2",
        "nodes": [
            {"name": "swim_bladder"},
            {"name": "eye_left"},
            {"name": "eye_right"},
        ],
    }
    if include_metadata:
        payload["metadata"] = {"heading_computation": _heading_spec()}
    return payload


def test_backfill_heading_columns_raw_keypoints_writes_and_drops_legacy() -> None:
    root = _FakeGroup()
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.attrs["pose_schema"] = _pose_schema()
    run.create_array(
        "keypoints_roi",
        data=np.array(
            [
                [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]],
                [[0.0, 0.0], [0.0, -2.0], [0.0, -2.0]],
                [[0.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]],
            ],
            dtype=np.float64,
        ),
    )
    run.create_array("heading", data=np.array([10.0, np.nan, 30.0], dtype=np.float64))
    run.create_array("detection_success", data=np.array([True, True, False], dtype=bool))
    run.create_array("detection_source", data=np.array([0, 0, 1], dtype=np.int8))
    run.create_array("heading_valid", data=np.array([True, False, False], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="detection_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    assert "heading_valid" not in run
    np.testing.assert_allclose(run["heading"][:], np.array([0.0, 90.0, -180.0], dtype=np.float64))
    np.testing.assert_array_equal(run["heading_finite"][:], np.array([True, True, True], dtype=bool))
    np.testing.assert_array_equal(run["heading_usable"][:], np.array([True, True, False], dtype=bool))


def test_backfill_heading_columns_refined_defaults_detection_source_to_real() -> None:
    root = _FakeGroup()
    run = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    run.attrs["pose_schema"] = _pose_schema()
    run.create_array(
        "keypoints_roi",
        data=np.array(
            [
                [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]],
                [[0.0, 0.0], [0.0, -2.0], [0.0, -2.0]],
                [[0.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]],
            ],
            dtype=np.float64,
        ),
    )
    run.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32))
    run.create_array("detection_indices", data=np.array([0, 0, 0], dtype=np.int32))
    run.create_array("heading", data=np.array([1.0, np.nan, 2.0], dtype=np.float64))
    run.create_array("refined_success", data=np.array([True, False, True], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="refined_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    np.testing.assert_allclose(run["heading"][:], np.array([0.0, 90.0, -180.0], dtype=np.float64))
    np.testing.assert_array_equal(run["heading_finite"][:], np.array([True, True, True], dtype=bool))
    np.testing.assert_array_equal(run["heading_usable"][:], np.array([True, False, True], dtype=bool))
    assert "heading_delta_prev_deg" in run
    assert "heading_delta_next_deg" in run
    assert "heading_temporal_outlier" in run
    summary = run.attrs["summary_statistics"]
    assert "postprocess" in summary
    assert "heading_temporal_outlier" in summary["postprocess"]
    assert summary["postprocess"]["temporal_heading_status"] == "enabled"


def test_backfill_heading_columns_skips_when_fields_present_and_temporal_summary_ready() -> None:
    root = _FakeGroup()
    run = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    run.attrs["pose_schema"] = _pose_schema()
    run.create_array(
        "keypoints_roi",
        data=np.array(
            [
                [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]],
                [[0.0, 0.0], [0.0, -2.0], [0.0, -2.0]],
            ],
            dtype=np.float64,
        ),
    )
    run.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32))
    run.create_array("detection_indices", data=np.array([0, 0], dtype=np.int32))
    run.create_array("heading", data=np.array([0.0, 90.0], dtype=np.float64))
    run.create_array("refined_success", data=np.array([True, True], dtype=bool))
    run.create_array("heading_finite", data=np.array([True, True], dtype=bool))
    run.create_array("heading_usable", data=np.array([True, True], dtype=bool))
    run.create_array("heading_delta_prev_deg", data=np.array([np.nan, np.nan], dtype=np.float32))
    run.create_array("heading_delta_next_deg", data=np.array([np.nan, np.nan], dtype=np.float32))
    run.create_array("heading_temporal_outlier", data=np.array([False, False], dtype=bool))
    run.attrs["summary_statistics"] = {
        "postprocess": {
            "heading_temporal_evaluable": 0,
            "heading_temporal_outlier": 0,
            "heading_temporal_outlier_rate_percent": 0.0,
            "temporal_heading_status": "enabled",
        }
    }

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="refined_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "skipped_existing"
    np.testing.assert_allclose(run["heading"][:], np.array([0.0, 90.0], dtype=np.float64))
    np.testing.assert_array_equal(run["heading_finite"][:], np.array([True, True], dtype=bool))
    np.testing.assert_array_equal(run["heading_usable"][:], np.array([True, True], dtype=bool))


def test_backfill_heading_columns_refined_disables_sampled_import_temporal_fields() -> None:
    root = _FakeGroup()
    raw = root.create_group("raw_video")
    raw.attrs["import_mode"] = "sampled"
    raw.attrs["frame_step"] = 100
    raw.create_array("original_frame_indices", data=np.array([0, 100, 200], dtype=np.int32))

    run = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    run.attrs["pose_schema"] = _pose_schema()
    run.create_array(
        "keypoints_roi",
        data=np.array(
            [
                [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]],
                [[0.0, 0.0], [0.0, -2.0], [0.0, -2.0]],
                [[0.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]],
            ],
            dtype=np.float64,
        ),
    )
    run.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32))
    run.create_array("detection_indices", data=np.array([0, 0, 0], dtype=np.int32))
    run.create_array("heading", data=np.array([0.0, 179.0, 2.0], dtype=np.float64))
    run.create_array("refined_success", data=np.array([True, True, True], dtype=bool))
    run.create_array("usable_keypoints", data=np.array([True, True, True], dtype=bool))
    run.create_array("confidence_valid", data=np.array([True, True, True], dtype=bool))
    run.create_array("geometry_valid", data=np.array([True, True, True], dtype=bool))
    run.create_array("flip_corrected", data=np.array([False, False, False], dtype=bool))
    run.create_array("source_success", data=np.array([True, True, True], dtype=bool))
    run.create_array("detection_source", data=np.array([0, 0, 0], dtype=np.int8))
    run.create_array("heading_delta_prev_deg", data=np.array([np.nan, 1.0, np.nan], dtype=np.float32))
    run.create_array("heading_delta_next_deg", data=np.array([1.0, np.nan, np.nan], dtype=np.float32))
    run.create_array("heading_temporal_outlier", data=np.array([False, False, False], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="refined_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    np.testing.assert_allclose(run["heading"][:], np.array([0.0, 90.0, -180.0], dtype=np.float64))
    summary = run.attrs["summary_statistics"]["postprocess"]
    assert summary["temporal_heading_status"] == "disabled_sampled_import"
    assert summary["temporal_heading_disabled_reason"] == "sampled_import"
    assert "heading_delta_prev_deg" not in run
    assert "heading_delta_next_deg" not in run
    assert "heading_temporal_outlier" not in run


def test_backfill_heading_columns_raw_skips_when_fields_already_match() -> None:
    root = _FakeGroup()
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.attrs["pose_schema"] = _pose_schema()
    run.create_array(
        "keypoints_roi",
        data=np.array(
            [
                [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]],
                [[0.0, 0.0], [0.0, -2.0], [0.0, -2.0]],
            ],
            dtype=np.float64,
        ),
    )
    run.create_array("heading", data=np.array([0.0, 90.0], dtype=np.float64))
    run.create_array("detection_success", data=np.array([True, True], dtype=bool))
    run.create_array("detection_source", data=np.array([0, 0], dtype=np.int8))
    run.create_array("heading_finite", data=np.array([True, True], dtype=bool))
    run.create_array("heading_usable", data=np.array([True, True], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="detection_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "skipped_existing"


def test_backfill_heading_columns_detects_shape_mismatch() -> None:
    root = _FakeGroup()
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.attrs["pose_schema"] = _pose_schema()
    run.create_array(
        "keypoints_roi",
        data=np.array(
            [
                [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]],
                [[0.0, 0.0], [0.0, -2.0], [0.0, -2.0]],
            ],
            dtype=np.float64,
        ),
    )
    run.create_array("heading", data=np.array([1.0, 2.0], dtype=np.float64))
    run.create_array("detection_success", data=np.array([True], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="detection_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "shape_mismatch"


def test_backfill_heading_columns_requires_resolved_heading_metadata() -> None:
    root = _FakeGroup()
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.attrs["pose_schema"] = _pose_schema(include_metadata=False)
    run.create_array(
        "keypoints_roi",
        data=np.array([[[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]]], dtype=np.float64),
    )
    run.create_array("heading", data=np.array([5.0], dtype=np.float64))
    run.create_array("detection_success", data=np.array([True], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="detection_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "no_heading_spec"
    np.testing.assert_allclose(run["heading"][:], np.array([5.0], dtype=np.float64))
    assert "heading_finite" not in run


def test_backfill_heading_columns_honors_disabled_override() -> None:
    root = _FakeGroup()
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.attrs["pose_schema"] = _pose_schema()
    run.attrs["heading_computation_override"] = _heading_spec(enabled=False)
    run.create_array(
        "keypoints_roi",
        data=np.array([[[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]]], dtype=np.float64),
    )
    run.create_array("heading", data=np.array([12.0], dtype=np.float64))
    run.create_array("detection_success", data=np.array([True], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="detection_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    assert np.isnan(run["heading"][0])
    np.testing.assert_array_equal(run["heading_finite"][:], np.array([False], dtype=bool))
    np.testing.assert_array_equal(run["heading_usable"][:], np.array([False], dtype=bool))


def test_backfill_heading_columns_canonicalizes_legacy_bladder_label_names() -> None:
    root = _FakeGroup()
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.attrs["keypoint_labels"] = ["bladder", "eye_left", "eye_right"]
    run.attrs["pose_schema"] = _pose_schema()
    run.create_array(
        "keypoints_roi",
        data=np.array([[[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]]], dtype=np.float64),
    )
    run.create_array("heading", data=np.array([10.0], dtype=np.float64))
    run.create_array("detection_success", data=np.array([True], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="detection_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    np.testing.assert_allclose(run["heading"][:], np.array([0.0], dtype=np.float64))
    np.testing.assert_array_equal(run["heading_finite"][:], np.array([True], dtype=bool))
    np.testing.assert_array_equal(run["heading_usable"][:], np.array([True], dtype=bool))


def test_iter_run_groups_includes_direct_fs_run_names(monkeypatch) -> None:
    root = _FakeGroup()
    parent = root.create_group("keypoints_runs")
    parent.attrs["latest"] = "keypoints_001"
    embedded = parent.create_group("keypoints_001")
    embedded.attrs["name"] = "embedded"
    direct_groups = {
        "keypoints_001": _FakeGroup(),
        "keypoints_002": _FakeGroup(),
    }
    direct_groups["keypoints_001"].attrs["name"] = "direct-001"
    direct_groups["keypoints_002"].attrs["name"] = "direct-002"
    zarr_path = Path("/tmp/fake_training.zarr")
    seen_modes: list[str] = []

    monkeypatch.setattr(mod, "direct_zarr_group_names", lambda path: ["keypoints_001", "keypoints_002"])
    monkeypatch.setattr(
        mod,
        "open_zarr_group_direct",
        lambda path, mode: seen_modes.append(mode) or direct_groups[Path(path).name],
    )

    groups = list(mod._iter_run_groups(root, all_runs=True, zarr_path=zarr_path, open_mode="a"))

    assert len(groups) == 2
    assert groups[0][0] == "keypoints_runs/keypoints_001"
    assert groups[0][2] is direct_groups["keypoints_001"]
    assert groups[1][0] == "keypoints_runs/keypoints_002"
    assert groups[1][2] is direct_groups["keypoints_002"]
    assert seen_modes == ["a", "a"]


def test_main_dry_run_writes_jsonl_report(tmp_path: Path, capsys, monkeypatch) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "analysis"

    parent = root.create_group("keypoints_runs")
    parent.attrs["latest"] = "keypoints_001"
    run = parent.create_group("keypoints_001")
    run.attrs["pose_schema"] = _pose_schema()
    run.create_array(
        "keypoints_roi",
        data=np.array(
            [
                [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]],
                [[0.0, 0.0], [0.0, -2.0], [0.0, -2.0]],
            ],
            dtype=np.float64,
        ),
    )
    run.create_array("heading", data=np.array([10.0, 20.0], dtype=np.float64))
    run.create_array("detection_success", data=np.array([True, True], dtype=bool))

    _patch_scan(monkeypatch, {zarr_path: root})
    log_dir = tmp_path / "logs"

    rc = mod.main([str(zarr_path), "--log-dir", str(log_dir)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Log file:" in out
    assert "Dry run: ok=1" in out

    log_files = sorted(log_dir.glob("backfill_keypoint_heading_fields_*.jsonl"))
    assert len(log_files) == 1

    rows = _read_jsonl(log_files[0])
    events = [str(row["event"]) for row in rows]
    assert events[0] == "run_start"
    assert "run_group_checked" in events
    assert events[-1] == "run_end"

    checked_row = next(row for row in rows if row["event"] == "run_group_checked")
    assert checked_row["zarr"] == str(zarr_path)
    assert checked_row["run_path"] == "keypoints_runs/keypoints_001"
    assert checked_row["status"] == "ok"
    assert checked_row["dry_run"] is True
    assert checked_row["changed"] is True

    end_row = rows[-1]
    assert end_row["status"] == "ok"
    assert end_row["mode"] == "dry-run"
    assert end_row["runs_considered"] == 1
    assert end_row["ok"] == 1
