from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fisheye.registry.db import Registry
from fisheye.utils import training_image_profile as profile_mod
from fisheye.utils.training_image_profile import (
    TrainingImageProfileError,
    build_training_image_profile_summary,
    sync_latest_training_image_profile_for_zarr,
    write_training_image_profile,
)


class _FakeArray:
    def __init__(self, data: Any, *, chunks: tuple[int, ...] | None = None) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.chunks = chunks

    def __getitem__(self, item):
        return self._data[item]


class _FakeGroup:
    def __init__(self, *, store: object | None = None, path: str = "") -> None:
        self._children: dict[str, _FakeGroup | _FakeArray] = {}
        self.attrs: dict[str, Any] = {}
        self.store = object() if store is None else store
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

    def create_dataset(self, name: str, *, data: Any, chunks: tuple[int, ...] | None = None) -> _FakeArray:
        return self.create_array(name, data=data, chunks=chunks)

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


def _make_training_root() -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["recording_id"] = "rec_img_001"
    root.attrs["zarr_purpose"] = "training"
    root.attrs["rig_id"] = "omnifin0"
    root.attrs["dish_design"] = "palm"

    analysis_meta = root.create_group("analysis_metadata")
    analysis_meta.attrs["session_context"] = json.dumps(
        {
            "camera_id": "2010095",
            "arena_id": "arena_1",
            "canvas_name": "sleepyfish",
            "protocol_name_from_definition": "recording_only",
        }
    )
    analysis_meta.attrs["subject_metadata"] = json.dumps(
        {
            "days_post_fertilization": 6,
            "dish": {"genotype": "wildtype"},
        }
    )

    frames = np.zeros((4, 8, 8), dtype=np.uint8)
    frames[0] = np.arange(64, dtype=np.uint8).reshape(8, 8)
    frames[1] = 40
    frames[2] = 120
    frames[3] = np.flipud(frames[0])
    raw = root.create_group("raw_video")
    raw.attrs["import_purpose"] = "training_data"
    raw.create_array("images_ds", data=frames, chunks=(1, 8, 8))

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_001"
    detect = detect_parent.create_group("detect_001")
    detect.create_array("frame_indices", data=np.asarray([0, 2], dtype=np.int32))
    detect.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    detect.create_array("frame_counts", data=np.asarray([1, 0, 1, 0], dtype=np.int32))
    return root


def _legacy_map_detection_frames_to_rows(
    root: _FakeGroup,
    frame_indices: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    mapped = np.full(frame_indices.shape, -1, dtype=np.int64)
    direct = (frame_indices >= 0) & (frame_indices < int(n_rows))
    mapped[direct] = frame_indices[direct]
    if np.all(mapped >= 0):
        return mapped

    raw_video = root.get("raw_video")
    if raw_video is None or "original_frame_indices" not in raw_video:
        return mapped
    original = np.asarray(raw_video["original_frame_indices"][:], dtype=np.int64)
    lookup = {int(frame): idx for idx, frame in enumerate(original.tolist())}
    for idx, frame in enumerate(frame_indices):
        if mapped[idx] >= 0:
            continue
        mapped[idx] = lookup.get(int(frame), -1)
    return mapped


def _make_training_root_with_source_frame_labels() -> _FakeGroup:
    root = _make_training_root()
    root["raw_video"].create_array(
        "original_frame_indices",
        data=np.asarray([10, 0, 20, 30], dtype=np.int64),
    )
    detect = root["detect_runs/detect_001"]
    detect.create_array(
        "frame_indices",
        data=np.asarray([0, 20, 99], dtype=np.int64),
        overwrite=True,
    )
    detect.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        overwrite=True,
    )
    detect.create_array("frame_counts", data=np.asarray([1, 0, 1], dtype=np.int32), overwrite=True)
    return root


def test_map_detection_frames_to_rows_preserves_direct_inverse_and_unmapped_sentinel() -> None:
    root = _make_training_root_with_source_frame_labels()

    mapped = profile_mod._map_detection_frames_to_rows(
        root,
        np.asarray([0, 20, 99], dtype=np.int64),
        n_rows=4,
    )

    np.testing.assert_array_equal(mapped, np.asarray([0, 2, -1], dtype=np.int64))


def test_map_detection_frames_to_rows_preserves_legacy_duplicate_inverse_lookup() -> None:
    root = _make_training_root()
    root["raw_video"].create_array(
        "original_frame_indices",
        data=np.asarray([10, 20, 20, 30], dtype=np.int64),
    )

    mapped = profile_mod._map_detection_frames_to_rows(
        root,
        np.asarray([20], dtype=np.int64),
        n_rows=4,
    )

    np.testing.assert_array_equal(mapped, np.asarray([2], dtype=np.int64))


def test_training_image_profile_frame_domains_matches_legacy_label_summary(monkeypatch) -> None:
    root = _make_training_root_with_source_frame_labels()

    summary = build_training_image_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_img_001_training.zarr"),
        created_at_utc="2026-05-13T12:00:00+00:00",
    )

    monkeypatch.setattr(profile_mod, "_map_detection_frames_to_rows", _legacy_map_detection_frames_to_rows)
    legacy_summary = build_training_image_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_img_001_training.zarr"),
        created_at_utc="2026-05-13T12:00:00+00:00",
    )

    assert summary == legacy_summary
    assert summary["label_conditioned"]["source"]["rows_total"] == 3
    assert summary["label_conditioned"]["source"]["rows_mapped"] == 2
    assert summary["label_conditioned"]["profiled_detection_count"] == 2


def test_build_training_image_profile_summary_computes_image_and_label_stats() -> None:
    root = _make_training_root()

    summary = build_training_image_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_img_001_training.zarr"),
        created_at_utc="2026-05-13T12:00:00+00:00",
    )

    assert summary["schema_name"] == "training_image_profile"
    assert summary["dataset"]["zarr_use"] == "training"
    assert summary["source"]["frame_array"] == "raw_video/images_ds"
    assert summary["source"]["frames_total"] == 4
    assert summary["source"]["frames_profiled"] == 4
    assert len(summary["source"]["content_hash"]) == 64
    assert summary["image_metrics"]["intensity_mean"]["count"] == 4
    assert summary["image_metrics"]["contrast_p99_p01"]["p50"] is not None
    assert sum(summary["intensity_histogram"]["counts"]) == 4 * 8 * 8

    label = summary["label_conditioned"]
    assert label["source"]["status"] == "available"
    assert label["profiled_detection_count"] == 2
    assert label["metrics"]["fish_background_abs_delta"]["count"] == 2

    json.dumps(summary, allow_nan=False)


def test_write_training_image_profile_writes_attrs_arrays_and_latest() -> None:
    root = _make_training_root()

    result = write_training_image_profile(
        root,
        zarr_path=Path("/tmp/rec_img_001_training.zarr"),
        run_name="training_image_profile_fixed",
        created_at_utc="2026-05-13T12:10:00+00:00",
    )

    assert result.run_name == "training_image_profile_fixed"
    assert result.source_frame_array == "raw_video/images_ds"
    parent = root["analysis/training_image_profile_runs"]
    assert parent.attrs["latest"] == "training_image_profile_fixed"
    run_group = parent["training_image_profile_fixed"]
    assert run_group.attrs["schema_name"] == "training_image_profile"
    assert run_group.attrs["schema_version"] == "v1"
    assert run_group.attrs["fingerprint_status"] == "complete"
    assert run_group.attrs["source_frame_count"] == 4
    assert run_group["intensity_histogram_counts"].shape == (256,)
    assert run_group["intensity_histogram_bin_edges"].shape == (257,)
    json.dumps(run_group.attrs["profile_summary"], allow_nan=False)


def test_training_image_profile_rejects_analysis_zarr_by_default() -> None:
    root = _make_training_root()
    root.attrs["zarr_purpose"] = "analysis"

    with pytest.raises(TrainingImageProfileError, match="training Zarrs"):
        build_training_image_profile_summary(root, zarr_path=Path("/tmp/rec_img_001_analysis.zarr"))


def test_training_image_profile_cli_dry_run(monkeypatch, capsys, tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_img_001_training.zarr"
    root = _make_training_root()

    monkeypatch.setattr(profile_mod, "_open_root", lambda *_args, **_kwargs: root)

    rc = profile_mod.main([str(zarr_path), "--max-frames", "2"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "would_write"
    assert out["frames_profiled"] == 2
    assert root.get("analysis/training_image_profile_runs") is None


def test_sync_latest_training_image_profile_for_zarr_updates_registry(tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_img_001_training.zarr"
    root = _make_training_root()
    write_training_image_profile(
        root,
        zarr_path=zarr_path,
        run_name="training_image_profile_sync",
        created_at_utc="2026-05-13T12:20:00+00:00",
    )

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.upsert_dataset(
            "dataset_img_sync",
            session_uuid="session_img_sync",
            zarr_path=zarr_path,
            recording_id="rec_img_001",
            artifact_kind="source_recording",
            zarr_use="training",
        )

        result = sync_latest_training_image_profile_for_zarr(
            registry,
            zarr_path,
            root=root,
            apply=True,
        )
        rows = registry.query_training_image_profile_latest(dataset_ids=["dataset_img_sync"])
    finally:
        registry.close()

    assert result["status"] == "updated"
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["profile_run"] == "training_image_profile_sync"
    assert row["source_frame_array"] == "raw_video/images_ds"
    assert row["frames_profiled"] == 4
    assert row["mean_intensity_p50"] is not None
