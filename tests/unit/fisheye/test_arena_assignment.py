from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import fisheye.tracking.arena_assignment as mod
from fisheye.tracking.arena_assignment import assign_arenas_spatial
from fisheye.tracking.single_subject_per_arena import TrackingConflictError


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)

    def __getitem__(self, key):
        return self._data[key]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape


class _FakeGroup(dict):
    def __init__(self, *args, attrs: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def create_group(self, name: str):
        group = _FakeGroup()
        self[name] = group
        return group

    def require_group(self, name: str):
        if name not in self:
            return self.create_group(name)
        return self[name]

    def create_array(self, name: str, data, **_kwargs):
        array = _FakeArray(np.asarray(data))
        self[name] = array
        return array

    def group_keys(self) -> list[str]:
        return [key for key, value in self.items() if isinstance(value, _FakeGroup)]

    def array_keys(self) -> list[str]:
        return [key for key, value in self.items() if isinstance(value, _FakeArray)]

    def get(self, key: str, default=None):
        return super().get(key, default)


@pytest.fixture(autouse=True)
def _stub_environment_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda: {
            "git": {
                "commit_hash": "a" * 40,
                "short_hash": "aaaaaaaa",
                "branch": "main",
                "is_dirty": False,
                "remote_url": "origin",
            },
            "platform": {
                "hostname": "test-host",
                "system": "Linux",
                "release": "test",
                "python_version": "3.11",
                "machine": "x86_64",
            },
            "environment": {},
        },
    )


def _build_root() -> _FakeGroup:
    root = _FakeGroup(
        attrs={
            "experiment_setup": {"fish_per_dish": 1, "total_expected_fish": 1},
            "video_height": 100,
            "video_width": 100,
        }
    )
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_001"
    detect_run = detect_parent.create_group("detect_001")
    detect_run.attrs["source_detect_run"] = "detect_source_001"
    detect_run.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32))
    detect_run.create_array(
        "bbox_norm_coords",
        data=np.array([[0.25, 0.25, 0.1, 0.1], [0.75, 0.75, 0.1, 0.1]], dtype=np.float64),
    )
    detect_run.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32))
    detect_run.create_array("instance_key", data=np.array([101, 102], dtype=np.uint64))
    detect_run.create_array("source_detect_row_index", data=np.array([0, 1], dtype=np.int32))

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_001"
    refined_run = refined_parent.create_group("refined_001")
    refined_run.attrs["source_detect_run"] = "detect_source_001"
    instances = refined_run.create_group("instances")
    instances.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32))
    instances.create_array(
        "bbox_norm_coords",
        data=np.array([[0.1, 0.1, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]], dtype=np.float64),
    )
    instances.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32))
    instances.create_array("source_kind_codes", data=np.array([0, 0], dtype=np.int8))
    instances.create_array("manual_edit_flags", data=np.array([0, 0], dtype=np.int8))
    instances.create_array("instance_key", data=np.array([201, 202], dtype=np.uint64))
    instances.create_array("refined_row_ids", data=np.array([11, 12], dtype=np.int64))
    instances.create_array("source_detect_row_index", data=np.array([0, 1], dtype=np.int32))

    crop_parent = root.create_group("crop_runs")
    crop_run = crop_parent.create_group("crop_001")
    crop_run.attrs.update(
        {
            "source_detect_run": "detect_source_001",
            "source_refined_run": "refined_001",
            "height": 1000,
            "width": 1000,
        }
    )
    crop_run.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32))
    crop_run.create_array(
        "bbox_norm_coords",
        data=np.array(
            [[0.25, 0.25, 0.1, 0.1], [0.50, 0.50, 0.1, 0.1], [0.75, 0.75, 0.1, 0.1]],
            dtype=np.float64,
        ),
    )
    crop_run.create_array("frame_counts", data=np.array([1, 1, 1], dtype=np.int32))
    crop_run.create_array("instance_key", data=np.array([301, 302, 303], dtype=np.uint64))
    crop_run.create_array("source_refined_row_ids", data=np.array([11, 12, 13], dtype=np.int64))
    crop_run.create_array("source_detect_row_index", data=np.array([0, 1, -1], dtype=np.int32))

    root.create_group("arena_assignment_runs")
    return root


def _legacy_infer_num_frames(root, detection_group, frame_indices: np.ndarray) -> int:
    candidates: list[int] = []

    for key in ("n_frames", "total_frames", "source_total_frames"):
        value = detection_group.attrs.get(key)
        if isinstance(value, (int, np.integer)) and value > 0:
            candidates.append(int(value))

    params = detection_group.attrs.get("parameters")
    if isinstance(params, dict):
        for key in ("n_frames", "total_frames"):
            value = params.get(key)
            if isinstance(value, (int, np.integer)) and value > 0:
                candidates.append(int(value))

    for key in ("palette_total_frames", "total_frames", "n_frames"):
        value = root.attrs.get(key)
        if isinstance(value, (int, np.integer)) and value > 0:
            candidates.append(int(value))

    if "raw_video" in root and "images_ds" in root["raw_video"]:
        candidates.append(int(root["raw_video/images_ds"].shape[0]))

    if candidates:
        return max(candidates)

    if frame_indices.size:
        return int(frame_indices.max()) + 1

    raise ValueError(
        "Unable to infer total frame count; detection metadata missing 'frame_counts' and video attributes."
    )


def test_infer_num_frames_matches_legacy_count_resolution(tmp_path) -> None:
    root = zarr.open_group(str(tmp_path / "arena_frames.zarr"), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", shape=(4, 2, 2), chunks=(4, 2, 2), dtype=np.uint8, overwrite=True)
    detect = root.create_group("detect_runs").create_group("detect_001")
    frame_indices = np.asarray([0, 2], dtype=np.int64)

    assert mod._infer_num_frames(root, detect, frame_indices) == _legacy_infer_num_frames(
        root,
        detect,
        frame_indices,
    )

    detect.create_array("frame_counts", data=np.ones(4, dtype=np.int32), overwrite=True)
    assert mod._count_from_domains(root, mod.FrameDomain.RUN_FRAME, run_group=detect) == int(
        detect["frame_counts"].shape[0]
    )


def _stub_assignment_runtime(
    monkeypatch: pytest.MonkeyPatch,
    root: _FakeGroup,
) -> None:
    def fake_open(_path: str, mode: str = "a"):
        assert mode == "a"
        return root

    def fake_get_run_group(_root, _stage, _console):
        assign_parent = _root["arena_assignment_runs"]
        assign_group = assign_parent.create_group("arena_assignment_001")
        return assign_group, "arena_assignment_001"

    monkeypatch.setattr(mod, "open_zarr_root", fake_open)
    monkeypatch.setattr(mod, "get_run_group", fake_get_run_group)
    monkeypatch.setattr(mod, "emit_stage_completion", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "build_stage_provenance", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(mod, "write_stage_provenance", lambda *args, **kwargs: None)


def _set_multiarena_crop_rows(
    root: _FakeGroup,
    *,
    frame_indices: np.ndarray,
    centers: np.ndarray,
) -> None:
    crop_run = root["crop_runs"]["crop_001"]
    crop_run["frame_indices"] = _FakeArray(np.asarray(frame_indices, dtype=np.int32))
    crop_run["bbox_norm_coords"] = _FakeArray(np.asarray(centers, dtype=np.float64))
    n_frames = int(frame_indices.max()) + 1 if frame_indices.size else 0
    crop_run["frame_counts"] = _FakeArray(
        np.bincount(frame_indices, minlength=n_frames).astype(np.int32)
    )
    row_count = int(frame_indices.shape[0])
    crop_run["instance_key"] = _FakeArray(
        np.arange(301, 301 + row_count, dtype=np.uint64)
    )
    crop_run["source_refined_row_ids"] = _FakeArray(
        np.arange(11, 11 + row_count, dtype=np.int64)
    )
    crop_run["source_detect_row_index"] = _FakeArray(
        np.arange(row_count, dtype=np.int32)
    )


def _four_square_rois() -> list[dict[str, object]]:
    return [
        {
            "id": 10,
            "roi_pixels": [0, 0, 50, 50],
            "source": "test",
            "image_shape": [100, 100],
        },
        {
            "id": 20,
            "roi_pixels": [50, 0, 50, 50],
            "source": "test",
            "image_shape": [100, 100],
        },
        {
            "id": 30,
            "roi_pixels": [0, 50, 50, 50],
            "source": "test",
            "image_shape": [100, 100],
        },
        {
            "id": 40,
            "roi_pixels": [50, 50, 50, 50],
            "source": "test",
            "image_shape": [100, 100],
        },
    ]


def test_assign_arenas_spatial_prefers_sparse_refined_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _build_root()
    captured: dict[str, object] = {}

    def fake_open(_path: str, mode: str = "a"):
        assert mode == "a"
        return root

    def fake_get_run_group(_root, _stage, _console):
        assign_parent = _root["arena_assignment_runs"]
        assign_group = assign_parent.create_group("arena_assignment_001")
        return assign_group, "arena_assignment_001"

    def fake_get_single_dish_roi_from_mask(_root, _console):
        return [{"id": 3, "roi_pixels": [0, 0, 100, 100], "source": "mask"}]

    def fake_write_tracking_run(**kwargs):
        captured.update(kwargs)
        track_parent = root["arena_assignment_runs"]["arena_assignment_001"].create_group("tracks")
        return "tracks_001", track_parent, {"ok": True}

    monkeypatch.setattr(mod, "open_zarr_root", fake_open)
    monkeypatch.setattr(mod, "get_run_group", fake_get_run_group)
    monkeypatch.setattr(mod, "infer_experiment_setup", lambda _attrs: SimpleNamespace(setup_type="single_dish", num_dishes=1, source="experiment_setup"))
    monkeypatch.setattr(mod, "get_single_dish_roi_from_mask", fake_get_single_dish_roi_from_mask)
    monkeypatch.setattr(mod, "write_tracking_run", fake_write_tracking_run)
    monkeypatch.setattr(mod, "emit_stage_completion", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "build_stage_provenance", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(mod, "write_stage_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "get_environment_info", lambda: {
        "git": {
            "commit_hash": "a" * 40,
            "short_hash": "aaaaaaaa",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "origin",
        },
        "platform": {
            "hostname": "test-host",
            "system": "Linux",
            "release": "test",
            "python_version": "3.11",
            "machine": "x86_64",
        },
        "environment": {},
    })

    result = assign_arenas_spatial("/tmp/fake.zarr", config={}, console=None)

    assert captured["source_rowset_path"] == "refined_detect_runs/refined_001/instances"
    assert captured["source_arena_assignment_run"] == "arena_assignment_001"
    assert captured["source_refined_run"] == "refined_001"
    assert captured["source_detect_run"] == "detect_source_001"
    assert root["arena_assignment_runs"]["arena_assignment_001"].attrs["assignment_source"] == "refined_instances"
    assert captured["instance_key"].tolist() == [201, 202]
    assert captured["source_refined_row_ids"].tolist() == [11, 12]
    assert captured["source_detect_row_index"].tolist() == [0, 1]
    assert captured["expected_source_rowset_fingerprint"].is_complete
    assert result["assigned_detections"] == 2


def test_assign_arenas_spatial_falls_back_to_raw_when_instances_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _build_root()
    refined_run = root["refined_detect_runs"]["refined_001"]
    del refined_run["instances"]
    captured: dict[str, object] = {}

    def fake_open(_path: str, mode: str = "a"):
        assert mode == "a"
        return root

    def fake_get_run_group(_root, _stage, _console):
        assign_parent = _root["arena_assignment_runs"]
        assign_group = assign_parent.create_group("arena_assignment_001")
        return assign_group, "arena_assignment_001"

    def fake_get_single_dish_roi_from_mask(_root, _console):
        return [{"id": 3, "roi_pixels": [0, 0, 100, 100], "source": "mask"}]

    def fake_write_tracking_run(**kwargs):
        captured.update(kwargs)
        track_parent = root["arena_assignment_runs"]["arena_assignment_001"].create_group("tracks")
        return "tracks_001", track_parent, {"ok": True}

    monkeypatch.setattr(mod, "open_zarr_root", fake_open)
    monkeypatch.setattr(mod, "get_run_group", fake_get_run_group)
    monkeypatch.setattr(mod, "infer_experiment_setup", lambda _attrs: SimpleNamespace(setup_type="single_dish", num_dishes=1, source="experiment_setup"))
    monkeypatch.setattr(mod, "get_single_dish_roi_from_mask", fake_get_single_dish_roi_from_mask)
    monkeypatch.setattr(mod, "write_tracking_run", fake_write_tracking_run)
    monkeypatch.setattr(mod, "emit_stage_completion", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "build_stage_provenance", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(mod, "write_stage_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "get_environment_info", lambda: {
        "git": {
            "commit_hash": "a" * 40,
            "short_hash": "aaaaaaaa",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "origin",
        },
        "platform": {
            "hostname": "test-host",
            "system": "Linux",
            "release": "test",
            "python_version": "3.11",
            "machine": "x86_64",
        },
        "environment": {},
    })

    result = assign_arenas_spatial("/tmp/fake.zarr", config={}, console=None)

    assert captured["source_rowset_path"] == "detect_runs/detect_001"
    assert captured["source_refined_run"] is None
    assert root["arena_assignment_runs"]["arena_assignment_001"].attrs["assignment_source"] == "detect_raw"
    assert result["assigned_detections"] == 2


def test_assign_arenas_spatial_can_track_explicit_crop_rowset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    captured: dict[str, object] = {}

    def fake_open(_path: str, mode: str = "a"):
        assert mode == "a"
        return root

    def fake_get_run_group(_root, _stage, _console):
        assign_parent = _root["arena_assignment_runs"]
        assign_group = assign_parent.create_group("arena_assignment_001")
        return assign_group, "arena_assignment_001"

    def fake_get_single_dish_roi_from_mask(_root, _console):
        return [
            {
                "id": 3,
                "roi_pixels": [0, 0, 100, 100],
                "source": "mask",
                "image_shape": [100, 100],
            }
        ]

    def fake_write_tracking_run(**kwargs):
        captured.update(kwargs)
        track_parent = root["arena_assignment_runs"]["arena_assignment_001"].create_group("tracks")
        return "tracks_001", track_parent, {"ok": True}

    monkeypatch.setattr(mod, "open_zarr_root", fake_open)
    monkeypatch.setattr(mod, "get_run_group", fake_get_run_group)
    monkeypatch.setattr(mod, "infer_experiment_setup", lambda _attrs: SimpleNamespace(setup_type="single_dish", num_dishes=1, source="experiment_setup"))
    monkeypatch.setattr(mod, "get_single_dish_roi_from_mask", fake_get_single_dish_roi_from_mask)
    monkeypatch.setattr(mod, "write_tracking_run", fake_write_tracking_run)
    monkeypatch.setattr(mod, "emit_stage_completion", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "build_stage_provenance", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(mod, "write_stage_provenance", lambda *args, **kwargs: None)

    result = assign_arenas_spatial(
        "/tmp/fake.zarr",
        config={},
        console=None,
        source_rowset_path="crop_runs/crop_001",
    )

    assert captured["source_rowset_path"] == "crop_runs/crop_001"
    assert captured["source_refined_run"] == "refined_001"
    assert captured["source_detect_run"] == "detect_source_001"
    assert root["arena_assignment_runs"]["arena_assignment_001"].attrs["assignment_source"] == "explicit_crop_rows"
    assert root["arena_assignment_runs"]["arena_assignment_001"].attrs["source_rowset_path"] == "crop_runs/crop_001"
    assert result["total_detections"] == 3
    assert result["assigned_detections"] == 3


def test_assign_arenas_spatial_resolves_canonical_keypoint_crop_and_exact_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    keypoints = root.create_group("keypoints_runs")
    canonical = keypoints.create_group("canonical_a")
    canonical.attrs["source_crop_run"] = "crop_001"
    captured: dict[str, object] = {}

    def fake_open(_path: str, mode: str = "a"):
        assert mode == "a"
        return root

    def fake_get_single_dish_roi_from_mask(_root, _console):
        return [
            {
                "id": 3,
                "roi_pixels": [0, 0, 100, 100],
                "source": "mask",
                "image_shape": [100, 100],
            }
        ]

    def fake_write_tracking_run(**kwargs):
        captured.update(kwargs)
        track = root["arena_assignment_runs"][
            "arena_assignment_tracks_a"
        ].create_group("tracks")
        return "tracks_a", track, {"ok": True}

    monkeypatch.setattr(mod, "open_zarr_root", fake_open)
    monkeypatch.setattr(mod, "is_run_complete_in_parent", lambda *_a, **_k: True)
    monkeypatch.setattr(mod, "is_run_selector_eligible", lambda _group: True)
    monkeypatch.setattr(
        mod,
        "require_runs_parent",
        lambda _root, name, **_kwargs: _root[name],
    )
    monkeypatch.setattr(
        mod,
        "infer_experiment_setup",
        lambda _attrs: SimpleNamespace(
            setup_type="single_dish",
            num_dishes=1,
            source="experiment_setup",
        ),
    )
    monkeypatch.setattr(
        mod,
        "get_single_dish_roi_from_mask",
        fake_get_single_dish_roi_from_mask,
    )
    monkeypatch.setattr(mod, "write_tracking_run", fake_write_tracking_run)
    monkeypatch.setattr(mod, "emit_stage_completion", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "build_stage_provenance", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(mod, "write_stage_provenance", lambda *args, **kwargs: None)

    result = assign_arenas_spatial(
        "/tmp/fake.zarr",
        config={},
        console=None,
        source_keypoint_run="canonical_a",
        arena_assignment_run_name="arena_assignment_tracks_a",
        tracking_run_name="tracks_a",
    )

    assert captured["source_rowset_path"] == "crop_runs/crop_001"
    assert captured["exact_run_name"] == "tracks_a"
    assert captured["source_detect_run"] == "detect_source_001"
    assert result["assigned_detections"] == 3


def test_crop_lineage_resolves_finalized_refined_working_source_detect() -> None:
    root = _build_root()
    crop = root["crop_runs"]["crop_001"]
    crop.attrs.clear()
    crop.attrs["provenance"] = {
        "inputs": {"source_refined_detect_run": "refined_final"}
    }
    refined = root["refined_detect_runs"]
    final = refined.create_group("refined_final")
    final.attrs["source_working_refined_run"] = "refined_working"
    working = refined.create_group("refined_working")
    working.attrs["source_detect_run"] = "detect_native_a"

    refined_name = mod._source_refined_run_from_attrs(crop.attrs)

    assert refined_name == "refined_final"
    assert mod._source_detect_run_from_refined(root, refined_name) == "detect_native_a"


def test_assign_arenas_spatial_accepts_external_crop_recorder_rowset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    crop_run = root["crop_runs"]["crop_001"]
    crop_run.attrs.pop("source_detect_run")
    crop_run.attrs.pop("source_refined_run")
    crop_run.attrs["detection_source_type"] = "external_crop_recorder_crop_meta_selected_live_detection"
    captured: dict[str, object] = {}

    def fake_get_single_dish_roi_from_mask(_root, _console):
        return [
            {
                "id": 3,
                "roi_pixels": [0, 0, 100, 100],
                "source": "mask",
                "image_shape": [100, 100],
            }
        ]

    def fake_write_tracking_run(**kwargs):
        captured.update(kwargs)
        track_parent = root["arena_assignment_runs"]["arena_assignment_001"].create_group("tracks")
        return "tracks_001", track_parent, {"ok": True}

    _stub_assignment_runtime(monkeypatch, root)
    monkeypatch.setattr(
        mod,
        "infer_experiment_setup",
        lambda _attrs: SimpleNamespace(setup_type="single_dish", num_dishes=1, source="experiment_setup"),
    )
    monkeypatch.setattr(mod, "get_single_dish_roi_from_mask", fake_get_single_dish_roi_from_mask)
    monkeypatch.setattr(mod, "write_tracking_run", fake_write_tracking_run)

    result = assign_arenas_spatial(
        "/tmp/fake.zarr",
        config={},
        console=None,
        source_rowset_path="crop_runs/crop_001",
    )

    source_label = "external_crop_recorder_crop_meta_selected_live_detection"
    assert captured["source_rowset_path"] == "crop_runs/crop_001"
    assert captured["source_refined_run"] is None
    assert captured["source_detect_run"] == source_label
    assign_group = root["arena_assignment_runs"]["arena_assignment_001"]
    assert assign_group.attrs["assignment_source"] == "explicit_crop_rows"
    assert assign_group.attrs["source_detect_run"] == source_label
    assert result["total_detections"] == 3
    assert result["assigned_detections"] == 3


def test_assign_arenas_spatial_accepts_clipped_collection_proxy_rowset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    crop_run = root["crop_runs"]["crop_001"]
    crop_run.attrs.pop("source_detect_run")
    crop_run.attrs.pop("source_refined_run")
    crop_run.attrs["detection_source_type"] = "finalized_clipped_refined_detect_collection_proxy"
    captured: dict[str, object] = {}

    def fake_get_single_dish_roi_from_mask(_root, _console):
        return [
            {
                "id": 3,
                "roi_pixels": [0, 0, 100, 100],
                "source": "mask",
                "image_shape": [100, 100],
            }
        ]

    def fake_write_tracking_run(**kwargs):
        captured.update(kwargs)
        track_parent = root["arena_assignment_runs"]["arena_assignment_001"].create_group("tracks")
        return "tracks_001", track_parent, {"ok": True}

    _stub_assignment_runtime(monkeypatch, root)
    monkeypatch.setattr(
        mod,
        "infer_experiment_setup",
        lambda _attrs: SimpleNamespace(setup_type="single_dish", num_dishes=1, source="experiment_setup"),
    )
    monkeypatch.setattr(mod, "get_single_dish_roi_from_mask", fake_get_single_dish_roi_from_mask)
    monkeypatch.setattr(mod, "write_tracking_run", fake_write_tracking_run)

    result = assign_arenas_spatial(
        "/tmp/fake.zarr",
        config={},
        console=None,
        source_rowset_path="crop_runs/crop_001",
    )

    source_label = "finalized_clipped_refined_detect_collection_proxy"
    assert captured["source_rowset_path"] == "crop_runs/crop_001"
    assert captured["source_refined_run"] is None
    assert captured["source_detect_run"] == source_label
    assign_group = root["arena_assignment_runs"]["arena_assignment_001"]
    assert assign_group.attrs["assignment_source"] == "explicit_crop_rows"
    assert assign_group.attrs["source_detect_run"] == source_label
    assert result["total_detections"] == 3
    assert result["assigned_detections"] == 3


def test_assign_arenas_spatial_tracks_four_subjects_in_four_subarenas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    _set_multiarena_crop_rows(
        root,
        frame_indices=np.array([0, 0, 0, 0], dtype=np.int32),
        centers=np.array(
            [
                [0.25, 0.25, 0.10, 0.10],
                [0.75, 0.25, 0.10, 0.10],
                [0.25, 0.75, 0.10, 0.10],
                [0.75, 0.75, 0.10, 0.10],
            ],
            dtype=np.float64,
        ),
    )
    _stub_assignment_runtime(monkeypatch, root)
    monkeypatch.setattr(
        mod,
        "infer_experiment_setup",
        lambda _attrs: SimpleNamespace(
            setup_type="multi_dish",
            num_dishes=4,
            source="experiment_setup",
        ),
    )

    result = assign_arenas_spatial(
        "/tmp/fake.zarr",
        config={"assign_ids": {"sub_dish_rois": _four_square_rois()}},
        console=None,
        source_rowset_path="crop_runs/crop_001",
    )

    assign_group = root["arena_assignment_runs"]["arena_assignment_001"]
    assert result["total_detections"] == 4
    assert result["assigned_detections"] == 4
    assert result["unassigned_detections"] == 0
    assert assign_group["arena_ids"][:].tolist() == [10, 20, 30, 40]
    assert assign_group["n_detections_per_arena"][:].tolist() == [[1, 1, 1, 1]]

    track_parent = root["tracking_runs"]
    track_group = track_parent[track_parent.attrs["latest"]]
    assert track_group.attrs["source_rowset_path"] == "crop_runs/crop_001"
    assert track_group.attrs["source_arena_assignment_run"] == "arena_assignment_001"
    assert track_group.attrs["tracking_identity_mode"] == "instance_key"
    assert track_group.attrs["source_rowset_fingerprint_status"] == "complete"
    assert assign_group.attrs["source_rowset_fingerprint"] == track_group.attrs["source_rowset_fingerprint"]
    assert track_group["track_ids"][:].tolist() == [0, 1, 2, 3]
    assert track_group["arena_ids"][:].tolist() == [10, 20, 30, 40]
    assert track_group["track_arena_ids"][:].tolist() == [10, 20, 30, 40]
    assert track_group["instance_key"][:].tolist() == [301, 302, 303, 304]
    assert track_group.attrs["num_tracks"] == 4
    assert track_group.attrs["tracking_qc_state"] == "ok"


def test_assign_arenas_spatial_fails_tracks_on_same_frame_same_arena_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    _set_multiarena_crop_rows(
        root,
        frame_indices=np.array([0, 0], dtype=np.int32),
        centers=np.array(
            [
                [0.25, 0.25, 0.10, 0.10],
                [0.35, 0.35, 0.10, 0.10],
            ],
            dtype=np.float64,
        ),
    )
    emitted: list[dict[str, object]] = []
    _stub_assignment_runtime(monkeypatch, root)
    monkeypatch.setattr(
        mod,
        "infer_experiment_setup",
        lambda _attrs: SimpleNamespace(
            setup_type="multi_dish",
            num_dishes=4,
            source="experiment_setup",
        ),
    )
    monkeypatch.setattr(
        mod,
        "emit_stage_completion",
        lambda *args, **kwargs: emitted.append(kwargs),
    )

    with pytest.raises(TrackingConflictError, match="frame 0"):
        assign_arenas_spatial(
            "/tmp/fake.zarr",
            config={"assign_ids": {"sub_dish_rois": _four_square_rois()}},
            console=None,
            source_rowset_path="crop_runs/crop_001",
        )

    assign_group = root["arena_assignment_runs"]["arena_assignment_001"]
    assert assign_group["arena_ids"][:].tolist() == [10, 10]
    assert "tracking_runs" not in root
    status_by_step = {event["step_name"]: event["status"] for event in emitted}
    reason_by_step = {
        event["step_name"]: event["details_json"]["reason"]
        for event in emitted
    }
    assert status_by_step == {"arena_assignment": "ok", "tracks": "error"}
    assert reason_by_step["tracks"] == "tracking_generation_failed"


def test_assign_arenas_spatial_keeps_unassigned_crop_rows_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    _set_multiarena_crop_rows(
        root,
        frame_indices=np.array([0, 1, 2], dtype=np.int32),
        centers=np.array(
            [
                [0.25, 0.25, 0.10, 0.10],
                [0.75, 0.25, 0.10, 0.10],
                [1.25, 0.25, 0.10, 0.10],
            ],
            dtype=np.float64,
        ),
    )
    _stub_assignment_runtime(monkeypatch, root)
    monkeypatch.setattr(
        mod,
        "infer_experiment_setup",
        lambda _attrs: SimpleNamespace(
            setup_type="multi_dish",
            num_dishes=4,
            source="experiment_setup",
        ),
    )

    result = assign_arenas_spatial(
        "/tmp/fake.zarr",
        config={"assign_ids": {"sub_dish_rois": _four_square_rois()}},
        console=None,
        source_rowset_path="crop_runs/crop_001",
    )

    assign_group = root["arena_assignment_runs"]["arena_assignment_001"]
    assert result["assigned_detections"] == 2
    assert result["unassigned_detections"] == 1
    assert assign_group["arena_ids"][:].tolist() == [10, 20, -1]
    assert assign_group["n_detections_per_arena"][:].tolist() == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]

    track_parent = root["tracking_runs"]
    track_group = track_parent[track_parent.attrs["latest"]]
    assert track_group["track_ids"][:].tolist() == [0, 1, -1]
    assert track_group["arena_ids"][:].tolist() == [10, 20, -1]
    assert track_group.attrs["n_unassigned_rows"] == 1
    assert track_group.attrs["tracking_qc_state"] == "warn"


def test_assign_arenas_spatial_missing_single_dish_mask_returns_standard_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    emitted: list[dict[str, object]] = []

    def fake_open(_path: str, mode: str = "a"):
        assert mode == "a"
        return root

    def fake_emit_stage_completion(*_args, **kwargs):
        emitted.append(kwargs)

    monkeypatch.setattr(mod, "open_zarr_root", fake_open)
    monkeypatch.setattr(
        mod,
        "infer_experiment_setup",
        lambda _attrs: SimpleNamespace(
            setup_type="single_dish",
            num_dishes=1,
            source="experiment_setup",
        ),
    )
    monkeypatch.setattr(mod, "get_single_dish_roi_from_mask", lambda _root, _console: None)
    monkeypatch.setattr(mod, "emit_stage_completion", fake_emit_stage_completion)

    result = assign_arenas_spatial("/tmp/fake.zarr", config={}, console=None)

    assert result["status"] == "missing"
    assert result["reason"] == "dish_mask_missing"
    assert result["assigned_detections"] == 0
    assert result["unassigned_detections"] == 0
    assert result["assignment_rate_percent"] == 0.0
    assert result["assigned"] == 0
    assert result["unassigned"] == 0
    assert len(emitted) == 2
    assert {event["step_name"] for event in emitted} == {"arena_assignment", "tracks"}
