from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fisheye.registry.db import Registry
from fisheye.utils import backfill_eye_mask_profiles as backfill_mod
from fisheye.utils.eye_mask_profile import (
    build_eye_mask_profile_summary,
    resolve_eye_mask_source,
    write_eye_mask_profile,
)


class _FakeArray:
    def __init__(self, data: Any, *, chunks: tuple[int, ...] | None = None) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.chunks = chunks

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value) -> None:
        self._data[item] = value


class _FakeGroup:
    def __init__(self, *, store: object | None = None, path: str = "") -> None:
        self._children: dict[str, _FakeGroup | _FakeArray] = {}
        self.attrs: dict[str, Any] = {}
        self.store = store
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        if name in self._children:
            raise ValueError(f"{name} already exists")
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(store=self.store, path=child_path)
        self._children[name] = child
        return child

    def require_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if isinstance(existing, _FakeGroup):
            return existing
        if existing is not None:
            raise TypeError(f"{name} already exists and is not a group")
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(store=self.store, path=child_path)
        self._children[name] = child
        return child

    def create_array(
        self,
        name: str,
        *,
        data: Any,
        chunks: tuple[int, ...] | None = None,
        overwrite: bool = False,
    ) -> _FakeArray:
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_array(tail, data=data, chunks=chunks, overwrite=overwrite)
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        arr = _FakeArray(data, chunks=chunks)
        self._children[name] = arr
        return arr

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

    def group_keys(self):
        return [key for key, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):
        return list(self._children.keys())

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except Exception:
            return False

    def __getitem__(self, key: str):
        if "/" in key:
            current: _FakeGroup | _FakeArray = self
            for token in key.split("/"):
                if not isinstance(current, _FakeGroup):
                    raise KeyError(key)
                current = current._children[token]
            return current
        return self._children[key]

    def __delitem__(self, key: str) -> None:
        if "/" in key:
            head, tail = key.split("/", 1)
            child = self._children[head]
            if not isinstance(child, _FakeGroup):
                raise KeyError(key)
            del child[tail]
            return
        del self._children[key]


def _make_mask_row(left_pixels: list[tuple[int, int]], right_pixels: list[tuple[int, int]]) -> np.ndarray:
    row = np.zeros((2, 4, 4), dtype=np.uint8)
    for y, x in left_pixels:
        row[0, y, x] = 1
    for y, x in right_pixels:
        row[1, y, x] = 1
    return row


def _make_eye_mask_root(*, zarr_use: str = "analysis") -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["recording_id"] = "rec_001"
    root.attrs["zarr_purpose"] = zarr_use
    root.attrs["rig_id"] = "omnifin0"
    root.attrs["dish_design"] = "cedar"

    analysis_meta = root.create_group("analysis_metadata")
    analysis_meta.attrs["session_context"] = json.dumps(
        {
            "camera_id": "2010094",
            "arena_id": "arena_2",
            "canvas_name": "shadow",
            "protocol_name_from_definition": "DefaultScreen",
        }
    )
    analysis_meta.attrs["subject_metadata"] = json.dumps(
        {
            "days_post_fertilization": 7,
            "dish": {"genotype": "Tg(elavl3:gcamp7f)"},
        }
    )

    keypoints_parent = root.create_group("refined_keypoints_runs")
    keypoints_parent.attrs["latest"] = "refined_keypoints_001"
    keypoints_parent.create_group("refined_keypoints_001")

    raw_parent = root.create_group("eye_masks_runs")
    raw_parent.attrs["latest"] = "eye_masks_001"
    raw_run = raw_parent.create_group("eye_masks_001")
    raw_run.attrs.update(
        {
            "method": "yolo_eye_segmentation",
            "source_crop_run": "crop_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "source_keypoints_run": "refined_keypoints_001",
            "total_rois": 5,
            "successful_roi_pairs": 3,
            "successful_roi_pair_rate": 0.6,
        }
    )

    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_parent.attrs["latest"] = "refined_eye_masks_001"
    refined_run = refined_parent.create_group("refined_eye_masks_001")

    masks = np.stack(
        [
            _make_mask_row([(0, 0), (0, 1), (1, 0), (1, 1)], [(2, 2), (2, 3), (3, 2), (3, 3)]),
            _make_mask_row([(0, 0), (1, 0), (2, 0)], [(0, 3), (1, 3), (2, 3)]),
            _make_mask_row([(1, 1), (1, 2)], [(2, 1), (2, 2)]),
            _make_mask_row([(1, 1)], [(2, 2)]),
            _make_mask_row([], []),
        ],
        axis=0,
    )

    ellipse_success = np.asarray(
        [
            [True, True],
            [True, True],
            [True, False],
            [True, True],
            [False, False],
        ],
        dtype=np.bool_,
    )
    ellipse_params = np.asarray(
        [
            [[1.0, 1.0, 18.0, 10.0, 0.0], [2.5, 2.5, 17.0, 9.0, 0.0]],
            [[1.0, 1.0, 17.0, 9.0, 0.0], [2.5, 2.5, 16.0, 8.0, 0.0]],
            [[1.2, 1.2, 16.0, 8.0, 0.0], [2.2, 2.2, 15.0, 7.0, 0.0]],
            [[1.4, 1.4, 15.0, 7.0, 0.0], [2.0, 2.0, 14.0, 6.0, 0.0]],
            [[1.5, 1.5, 14.0, 6.0, 0.0], [1.8, 1.8, 13.0, 5.0, 0.0]],
        ],
        dtype=np.float32,
    )

    refined_run.create_array("masks_roi", data=masks, chunks=(2, 2, 4, 4))
    refined_run.create_array("ellipse_success", data=ellipse_success, chunks=(2, 2))
    refined_run.create_array("ellipse_params", data=ellipse_params, chunks=(2, 2, 5))
    refined_run.create_array(
        "eye_separation",
        data=np.asarray([20.0, 21.0, 22.0, 23.0, 24.0], dtype=np.float32),
        chunks=(2,),
    )

    metrics_group = refined_run.create_group("metrics")
    metrics_group.create_array(
        "area_refined",
        data=np.asarray(
            [
                [12.0, 14.0],
                [10.0, 9.0],
                [8.0, 7.0],
                [6.0, 6.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        chunks=(2, 2),
    )
    metrics_group.create_array(
        "area_union_refined",
        data=np.asarray([22.0, 18.0, 14.0, 10.0, 0.0], dtype=np.float32),
        chunks=(2,),
    )
    metrics_group.create_array(
        "area_ratio_left_right",
        data=np.asarray([0.86, 1.11, 1.14, 1.0, np.nan], dtype=np.float32),
        chunks=(2,),
    )
    metrics_group.create_array(
        "axis_ratio",
        data=np.asarray(
            [
                [1.8, 1.7],
                [1.9, 1.8],
                [2.0, 1.9],
                [1.6, 1.5],
                [np.nan, np.nan],
            ],
            dtype=np.float32,
        ),
        chunks=(2, 2),
    )
    metrics_group.create_array(
        "circularity",
        data=np.asarray(
            [
                [0.80, 0.82],
                [0.79, 0.81],
                [0.78, 0.80],
                [0.77, 0.79],
                [np.nan, np.nan],
            ],
            dtype=np.float32,
        ),
        chunks=(2, 2),
    )
    metrics_group.create_array(
        "separation_refined",
        data=np.asarray([20.0, 21.0, 22.0, 23.0, 24.0], dtype=np.float32),
        chunks=(2,),
    )
    metrics_group.create_array(
        "separation_keypoint",
        data=np.asarray([20.0, 20.0, 21.0, 22.0, 23.0], dtype=np.float32),
        chunks=(2,),
    )

    refined_run.attrs.update(
        {
            "method": "refine_eye_masks",
            "source_eye_masks_run": "eye_masks_001",
            "source_crop_run": "crop_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "source_keypoints_run": "refined_keypoints_001",
            "total_rois": 5,
            "successful_eyes": 7,
            "successful_roi_pairs": 3,
            "successful_roi_pair_rate": 0.6,
            "reason_counts": {
                "refined": 3,
                "filtered_pair": 1,
                "keypoint_fail": 1,
            },
            "eye_mask_review_status": {
                "state": "approved",
                "method": "manual",
                "intended_use": "training",
                "reviewer": "qa_user",
                "timestamp": "2026-02-25T02:59:00+00:00",
            },
            "source_keypoint_stale": {
                "state": "fresh",
                "reason": None,
                "timestamp_utc": "2026-02-25T02:58:30+00:00",
            },
            "duration_seconds": 5.0,
        }
    )

    return root


def _is_monotonic(values: list[float | None]) -> bool:
    filtered = [float(v) for v in values if v is not None]
    return all(a <= b for a, b in zip(filtered, filtered[1:]))


def test_build_eye_mask_profile_summary_invariants() -> None:
    root = _make_eye_mask_root()
    summary = build_eye_mask_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        created_at_utc="2026-02-25T10:00:00+00:00",
    )

    source = summary["source"]
    quality = summary["quality"]
    geometry = summary["geometry"]
    composition = summary["composition"]
    freshness = summary["freshness"]

    assert source["stage_group"] == "refined_eye_masks_runs"
    assert source["eye_mask_path"] == "refined_eye_masks_runs/refined_eye_masks_001"
    assert source["eye_mask_method"] == "refine_eye_masks"
    assert source["source_keypoint_path"] == "refined_keypoints_runs/refined_keypoints_001"
    assert source["review_state"] == "approved"
    assert source["review_method"] == "manual"

    assert quality["rows_total"] == 5
    assert quality["rows_usable"] == 3
    assert quality["usable_rate"] == 0.6
    assert quality["successful_roi_pair_rate"] == 0.6
    assert quality["ellipse_success_rate"] == 0.7

    assert _is_monotonic(
        [
            geometry["area"]["stats"]["p10"],
            geometry["area"]["stats"]["p50"],
            geometry["area"]["stats"]["p90"],
        ]
    )
    assert _is_monotonic(
        [
            geometry["left_area"]["stats"]["p10"],
            geometry["left_area"]["stats"]["p50"],
            geometry["left_area"]["stats"]["p90"],
        ]
    )
    assert _is_monotonic(
        [
            geometry["eye_separation"]["stats"]["p10"],
            geometry["eye_separation"]["stats"]["p50"],
            geometry["eye_separation"]["stats"]["p90"],
        ]
    )

    assert summary["spatial"]["edge_proximity_rate"] is not None
    assert freshness["source_keypoint_stale"]["state"] == "fresh"

    assert composition["rig_id"] == "omnifin0"
    assert composition["camera_id"] == "2010094"
    assert composition["arena_id"] == "arena_2"
    assert composition["canvas_name"] == "shadow"
    assert composition["protocol_name"] == "DefaultScreen"
    assert composition["dish_design"] == "cedar"
    assert composition["genotype"] == "Tg(elavl3:gcamp7f)"
    assert composition["dpf_at_acquisition"] == 7


def test_build_eye_mask_profile_summary_labels_derived_compat_sources() -> None:
    root = _make_eye_mask_root()
    refined = root["refined_eye_masks_runs/refined_eye_masks_001"]
    refined.attrs["compatibility_role"] = "derived_from_refined_subject_masks"
    refined.attrs["source_refined_subject_masks_run"] = "refined_subject_masks_001"
    refined.attrs["source_subject_mask_run"] = "subject_masks_001"

    summary = build_eye_mask_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        created_at_utc="2026-02-25T10:02:00+00:00",
    )

    source = summary["source"]
    assert source["stage_group"] == "refined_eye_masks_runs"
    assert source["eye_stage_role"] == "derived_compat"
    assert source["eye_stage_label"] == "refined_eye_masks_runs (derived compat)"
    assert source["authority_stage_group"] == "refined_subject_masks_runs"
    assert source["compatibility_role"] == "derived_from_refined_subject_masks"
    assert source["source_refined_subject_masks_run"] == "refined_subject_masks_001"
    assert source["canonical_refined_subject_masks_run"] == "refined_subject_masks_001"
    assert source["source_subject_mask_run"] == "subject_masks_001"


def test_resolve_eye_mask_source_does_not_require_contains_lookup(monkeypatch) -> None:
    root = _make_eye_mask_root()
    monkeypatch.setattr(_FakeGroup, "__contains__", lambda _self, _key: False)
    source = resolve_eye_mask_source(root)
    assert source.eye_mask_path == "refined_eye_masks_runs/refined_eye_masks_001"
    assert source.eye_mask_method == "refine_eye_masks"


def test_build_eye_mask_profile_summary_handles_nested_path_lookup_fallback(monkeypatch) -> None:
    root = _make_eye_mask_root()
    original_getitem = _FakeGroup.__getitem__

    def _getitem_without_nested_paths(self, key: str):
        if "/" in key:
            raise KeyError(key)
        return original_getitem(self, key)

    monkeypatch.setattr(_FakeGroup, "__getitem__", _getitem_without_nested_paths)
    summary = build_eye_mask_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        created_at_utc="2026-02-25T10:05:00+00:00",
    )
    assert summary["source"]["eye_mask_path"] == "refined_eye_masks_runs/refined_eye_masks_001"


def test_write_eye_mask_profile_writes_run_attrs_and_latest_pointer() -> None:
    root = _make_eye_mask_root()
    result = write_eye_mask_profile(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        run_name="eye_mask_profile_2026-02-25_10-10-10",
        created_at_utc="2026-02-25T10:10:10+00:00",
    )

    assert result.run_name == "eye_mask_profile_2026-02-25_10-10-10"
    assert result.source_eye_mask_path == "refined_eye_masks_runs/refined_eye_masks_001"
    assert result.source_eye_mask_method == "refine_eye_masks"

    parent = root["analysis/eye_mask_profile_runs"]
    assert parent.attrs["latest"] == "eye_mask_profile_2026-02-25_10-10-10"

    run_group = parent["eye_mask_profile_2026-02-25_10-10-10"]
    assert run_group.attrs["schema_name"] == "eye_mask_dataset_profile"
    assert run_group.attrs["schema_version"] == "v1"
    assert run_group.attrs["source_stage_group"] == "refined_eye_masks_runs"
    assert run_group.attrs["source_eye_mask_path"] == "refined_eye_masks_runs/refined_eye_masks_001"
    assert run_group.attrs["source_eye_mask_method"] == "refine_eye_masks"
    assert run_group.attrs["source_keypoint_run"] == "refined_keypoints_001"
    assert run_group.attrs["source_eye_masks_run"] == "eye_masks_001"
    assert run_group.attrs["source_row_count"] == 5
    assert isinstance(run_group.attrs["profile_summary"], dict)


def test_build_eye_mask_profile_summary_reads_postprocess_reason_counts_fallback() -> None:
    root = _make_eye_mask_root()
    refined = root["refined_eye_masks_runs/refined_eye_masks_001"]
    refined.attrs.pop("reason_counts", None)
    refined.attrs["summary_statistics"] = {
        "postprocess": {
            "reason_counts": {
                "clean": 3,
                "manual_correction": 2,
                "keypoint_fail": 1,
            }
        }
    }

    summary = build_eye_mask_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        created_at_utc="2026-02-25T10:20:00+00:00",
    )
    quality = summary["quality"]
    assert quality["exclusion_reasons"] == {
        "clean": 3,
        "manual_correction": 2,
        "keypoint_fail": 1,
    }
    assert quality["excluded_reasons"] == {
        "clean": 3,
        "manual_correction": 2,
        "keypoint_fail": 1,
    }
    assert quality["exclusion_reasons_json"] == '{"clean":3,"keypoint_fail":1,"manual_correction":2}'


def test_build_eye_mask_profile_summary_prefers_mask_areas_when_metrics_stale() -> None:
    root = _make_eye_mask_root()
    metrics = root["refined_eye_masks_runs/refined_eye_masks_001/metrics"]
    metrics["area_refined"][:] = np.zeros((5, 2), dtype=np.float32)
    metrics["area_union_refined"][:] = np.zeros((5,), dtype=np.float32)

    summary = build_eye_mask_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        created_at_utc="2026-02-25T10:22:00+00:00",
    )
    geometry = summary["geometry"]
    assert geometry["left_area"]["stats"]["p50"] == pytest.approx(2.0)
    assert geometry["right_area"]["stats"]["p50"] == pytest.approx(2.0)
    assert geometry["union_area"]["stats"]["p50"] == pytest.approx(4.0)


def test_backfill_eye_mask_profiles_cli_dry_run_and_apply(capsys, monkeypatch, tmp_path: Path) -> None:
    zarr_ok = tmp_path / "ok_analysis.zarr"
    zarr_missing = tmp_path / "missing_analysis.zarr"

    root_ok = _make_eye_mask_root()
    root_missing = _FakeGroup()
    root_missing.attrs["zarr_purpose"] = "analysis"

    mapping = {
        zarr_ok: root_ok,
        zarr_missing: root_missing,
    }

    monkeypatch.setattr(backfill_mod, "_iter_zarr", lambda *_args, **_kwargs: iter(mapping.keys()))
    monkeypatch.setattr(
        backfill_mod.zarr,
        "open_group",
        lambda path, mode="r": mapping[Path(path)],  # noqa: ARG005
    )

    dry_rc = backfill_mod.main([str(tmp_path)])
    assert dry_rc == 0
    dry_out = capsys.readouterr().out
    assert "would_write" in dry_out
    assert "missing_source" in dry_out
    assert "would_write=1" in dry_out
    assert "errors=0" in dry_out
    assert root_ok.get("analysis/eye_mask_profile_runs") is None

    apply_rc = backfill_mod.main(
        [
            str(tmp_path),
            "--apply",
            "--run-name",
            "eye_mask_profile_fixed",
        ]
    )
    assert apply_rc == 0
    apply_out = capsys.readouterr().out
    assert "updated" in apply_out
    assert "missing_source=1" in apply_out
    assert "updated=1" in apply_out
    assert root_ok["analysis/eye_mask_profile_runs"].attrs["latest"] == "eye_mask_profile_fixed"
    assert "profile_summary" in root_ok["analysis/eye_mask_profile_runs/eye_mask_profile_fixed"].attrs


def test_backfill_eye_mask_profiles_populates_dataset_identity_from_registry(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "identity_training.zarr"
    root = _make_eye_mask_root(zarr_use="training")
    root.attrs.pop("recording_id", None)
    root.attrs["zarr_purpose"] = "training"
    mapping = {zarr_path: root}

    monkeypatch.setattr(backfill_mod, "_iter_zarr", lambda *_args, **_kwargs: iter(mapping.keys()))
    monkeypatch.setattr(
        backfill_mod.zarr,
        "open_group",
        lambda path, mode="r": mapping[Path(path)],  # noqa: ARG005
    )

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    registry.upsert_dataset(
        "dataset_identity",
        session_uuid="session_identity",
        zarr_path=zarr_path,
        recording_id="recording_identity",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.close()

    rc = backfill_mod.main(
        [
            str(zarr_path),
            "--apply",
            "--registry",
            str(registry_path),
            "--run-name",
            "eye_mask_profile_identity",
            "--zarr-use",
            "any",
        ]
    )
    assert rc == 0

    run_group = root["analysis/eye_mask_profile_runs/eye_mask_profile_identity"]
    profile_summary = run_group.attrs["profile_summary"]
    assert profile_summary["dataset"]["dataset_id"] == "dataset_identity"
    assert profile_summary["dataset"]["recording_id"] == "recording_identity"
    assert profile_summary["dataset"]["zarr_use"] == "training"
