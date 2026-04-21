from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fisheye.tracking.arena_assignment as mod
from fisheye.tracking.arena_assignment import assign_arenas_spatial


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

    def create_array(self, name: str, data, **_kwargs):
        array = _FakeArray(np.asarray(data))
        self[name] = array
        return array

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

    root.create_group("arena_assignment_runs")
    return root


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

    monkeypatch.setattr(mod.zarr, "open", fake_open)
    monkeypatch.setattr(mod, "get_run_group", fake_get_run_group)
    monkeypatch.setattr(mod, "infer_experiment_setup", lambda _attrs: SimpleNamespace(setup_type="single_dish", num_dishes=1, source="experiment_setup"))
    monkeypatch.setattr(mod, "get_single_dish_roi_from_mask", fake_get_single_dish_roi_from_mask)
    monkeypatch.setattr(mod, "write_single_subject_per_arena_tracking_run", fake_write_tracking_run)
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

    monkeypatch.setattr(mod.zarr, "open", fake_open)
    monkeypatch.setattr(mod, "get_run_group", fake_get_run_group)
    monkeypatch.setattr(mod, "infer_experiment_setup", lambda _attrs: SimpleNamespace(setup_type="single_dish", num_dishes=1, source="experiment_setup"))
    monkeypatch.setattr(mod, "get_single_dish_roi_from_mask", fake_get_single_dish_roi_from_mask)
    monkeypatch.setattr(mod, "write_single_subject_per_arena_tracking_run", fake_write_tracking_run)
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
