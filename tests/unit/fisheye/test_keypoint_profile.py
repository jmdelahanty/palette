from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import backfill_keypoint_profiles as backfill_mod
from fisheye.utils.keypoint_profile import build_keypoint_profile_summary, write_keypoint_profile


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


def _make_keypoint_root() -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["recording_id"] = "rec_001"
    root.attrs["zarr_purpose"] = "analysis"
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

    keypoints_parent = root.create_group("keypoints_runs")
    keypoints_parent.attrs["latest"] = "keypoints_001"
    keypoints_run = keypoints_parent.create_group("keypoints_001")
    keypoints_run.attrs["method"] = "traditional_pose"
    keypoints_run.attrs["skeleton_id"] = "fish_v1"
    keypoints_run.attrs["kpt_shape"] = [3, 2]
    keypoints_run.create_array("keypoints_roi", data=np.asarray(np.arange(30).reshape((5, 3, 2)), dtype=np.float64))
    keypoints_run.create_array("usable_keypoints", data=np.asarray([True, True, False, True, False], dtype=np.bool_))
    keypoints_run.create_array("confidence_valid", data=np.asarray([True, True, False, True, True], dtype=np.bool_))
    keypoints_run.create_array("geometry_valid", data=np.asarray([True, True, True, False, True], dtype=np.bool_))
    keypoints_run.create_array("triangle_area", data=np.asarray([0.01, 0.03, 0.02, 0.04, 0.05], dtype=np.float64))
    keypoints_run.create_array("min_angle", data=np.asarray([10.0, 15.0, 20.0, 25.0, 30.0], dtype=np.float64))
    keypoints_run.create_array("heading", data=np.asarray([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=np.float64))
    return root


def test_build_keypoint_profile_summary_invariants() -> None:
    root = _make_keypoint_root()
    summary = build_keypoint_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        created_at_utc="2026-02-24T10:00:00+00:00",
    )

    source = summary["source"]
    quality = summary["quality"]
    geometry = summary["geometry"]
    composition = summary["composition"]

    assert source["keypoint_path"] == "keypoints_runs/keypoints_001"
    assert source["keypoint_method"] == "traditional_pose"
    assert source["keypoint_run"] == "keypoints_001"
    assert source["skeleton_id"] == "fish_v1"
    assert source["kpt_shape"] == [3, 2]

    assert quality["rows_total"] == 5
    assert quality["rows_usable"] == 3
    assert quality["usable_keypoints_total"] == 3
    assert quality["usable_rate"] == 0.6
    assert quality["confidence_valid_rate"] == 0.8
    assert quality["geometry_valid_rate"] == 0.8

    assert geometry["triangle_area"]["stats"]["p10"] <= geometry["triangle_area"]["stats"]["p50"]
    assert geometry["triangle_area"]["stats"]["p50"] <= geometry["triangle_area"]["stats"]["p90"]
    assert geometry["min_angle"]["stats"]["p10"] <= geometry["min_angle"]["stats"]["p50"]
    assert geometry["min_angle"]["stats"]["p50"] <= geometry["min_angle"]["stats"]["p90"]
    assert geometry["heading"]["stats"]["p10"] <= geometry["heading"]["stats"]["p50"]
    assert geometry["heading"]["stats"]["p50"] <= geometry["heading"]["stats"]["p90"]

    assert composition["rig_id"] == "omnifin0"
    assert composition["camera_id"] == "2010094"
    assert composition["arena_id"] == "arena_2"
    assert composition["canvas_name"] == "shadow"
    assert composition["protocol_name"] == "DefaultScreen"
    assert composition["dish_design"] == "cedar"
    assert composition["genotype"] == "Tg(elavl3:gcamp7f)"
    assert composition["dpf_at_acquisition"] == 7


def test_build_keypoint_profile_summary_includes_edge_distance_metrics() -> None:
    root = _make_keypoint_root()
    keypoints_run = root["keypoints_runs/keypoints_001"]
    keypoints_run.create_array(
        "edge_pairs",
        data=np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int16),
    )
    keypoints_run.create_array(
        "edge_distances",
        data=np.asarray(
            [
                [10.0, 20.0, 30.0],
                [12.0, 22.0, np.nan],
                [14.0, 24.0, 34.0],
                [16.0, 26.0, 36.0],
                [18.0, 28.0, 38.0],
            ],
            dtype=np.float32,
        ),
    )
    keypoints_run.create_array(
        "edge_distances_norm",
        data=np.asarray(
            [
                [0.10, 0.20, 0.30],
                [0.12, 0.22, np.nan],
                [0.14, 0.24, 0.34],
                [0.16, 0.26, 0.36],
                [0.18, 0.28, 0.38],
            ],
            dtype=np.float32,
        ),
    )
    keypoints_run.create_array(
        "edge_distance_valid",
        data=np.asarray(
            [
                [True, True, True],
                [True, True, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ],
            dtype=np.bool_,
        ),
    )
    keypoints_run.attrs["edge_distance_labels"] = [
        "swim_bladder-eye_left",
        "swim_bladder-eye_right",
        "eye_left-eye_right",
    ]
    keypoints_run.attrs["edge_distance_normalization"] = {"mode": "roi_diagonal", "roi_diagonal": 90.5}

    summary = build_keypoint_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
    )
    edge_distance = summary["geometry"]["edge_distance"]

    assert edge_distance["edge_order"] == [[0, 1], [0, 2], [1, 2]]
    assert edge_distance["edge_labels"] == [
        "swim_bladder-eye_left",
        "swim_bladder-eye_right",
        "eye_left-eye_right",
    ]
    assert edge_distance["normalization"]["mode"] == "roi_diagonal"

    first_edge = edge_distance["edges"][0]
    assert first_edge["label"] == "swim_bladder-eye_left"
    assert first_edge["valid_count"] == 5
    assert first_edge["valid_rate"] == 1.0
    assert first_edge["distance"]["p50"] == pytest.approx(14.0)
    assert first_edge["distance_norm"]["p50"] == pytest.approx(0.14)

    third_edge = edge_distance["edges"][2]
    assert third_edge["valid_count"] == 4
    assert third_edge["valid_rate"] == pytest.approx(4.0 / 5.0)
    assert third_edge["distance"]["count"] == 4
    assert third_edge["distance_norm"]["count"] == 4


def test_build_keypoint_profile_summary_includes_derived_metrics() -> None:
    root = _make_keypoint_root()
    keypoints_run = root["keypoints_runs/keypoints_001"]
    keypoints_run.attrs["derived_metric_schema_id"] = "traditional_v2_derived_metrics"
    keypoints_run.attrs["derived_metric_schema_version"] = "1.0"
    keypoints_run.attrs["derived_metric_labels"] = [
        "total_length",
        "tail_length",
        "head_length",
        "eye_span",
    ]
    keypoints_run.attrs["derived_metric_normalization"] = {
        "mode": "roi_diagonal",
        "roi_diagonal": 90.5,
    }
    keypoints_run.create_array(
        "derived_metric_values",
        data=np.asarray(
            [
                [40.0, 25.0, 15.0, 8.0],
                [42.0, 26.0, 16.0, 8.5],
                [np.nan, np.nan, np.nan, np.nan],
                [44.0, 27.0, 17.0, 9.0],
                [46.0, 28.0, 18.0, 9.5],
            ],
            dtype=np.float32,
        ),
    )
    keypoints_run.create_array(
        "derived_metric_values_norm",
        data=np.asarray(
            [
                [0.40, 0.25, 0.15, 0.08],
                [0.42, 0.26, 0.16, 0.085],
                [np.nan, np.nan, np.nan, np.nan],
                [0.44, 0.27, 0.17, 0.09],
                [0.46, 0.28, 0.18, 0.095],
            ],
            dtype=np.float32,
        ),
    )
    keypoints_run.create_array(
        "derived_metric_valid",
        data=np.asarray(
            [
                [True, True, True, True],
                [True, True, True, True],
                [False, False, False, False],
                [True, True, True, True],
                [True, True, True, True],
            ],
            dtype=np.bool_,
        ),
    )

    summary = build_keypoint_profile_summary(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
    )
    derived = summary["geometry"]["derived_metrics"]

    assert derived["schema_id"] == "traditional_v2_derived_metrics"
    assert derived["labels"] == ["total_length", "tail_length", "head_length", "eye_span"]
    assert derived["normalization"]["mode"] == "roi_diagonal"

    total_length = derived["metrics"][0]
    assert total_length["name"] == "total_length"
    assert total_length["valid_count"] == 4
    assert total_length["valid_rate"] == pytest.approx(4.0 / 5.0)
    assert total_length["stats"]["p50"] == pytest.approx(43.0)
    assert total_length["stats_norm"]["p50"] == pytest.approx(0.43)


def test_write_keypoint_profile_writes_run_attrs_and_latest_pointer() -> None:
    root = _make_keypoint_root()
    result = write_keypoint_profile(
        root,
        zarr_path=Path("/tmp/rec_001_analysis.zarr"),
        run_name="keypoint_profile_2026-02-24_10-10-10",
        created_at_utc="2026-02-24T10:10:10+00:00",
        source_keypoint_path="keypoints_runs/keypoints_001",
    )

    assert result.run_name == "keypoint_profile_2026-02-24_10-10-10"
    assert result.source_keypoint_path == "keypoints_runs/keypoints_001"
    assert result.source_keypoint_method == "traditional_pose"

    parent = root["analysis/keypoint_profile_runs"]
    assert parent.attrs["latest"] == "keypoint_profile_2026-02-24_10-10-10"

    run_group = parent["keypoint_profile_2026-02-24_10-10-10"]
    assert run_group.attrs["schema_name"] == "keypoint_dataset_profile"
    assert run_group.attrs["schema_version"] == "v1"
    assert run_group.attrs["source_keypoint_path"] == "keypoints_runs/keypoints_001"
    assert run_group.attrs["source_keypoint_method"] == "traditional_pose"
    assert run_group.attrs["source_keypoint_run"] == "keypoints_001"
    assert run_group.attrs["source_skeleton_id"] == "fish_v1"
    assert run_group.attrs["source_kpt_shape"] == [3, 2]
    assert run_group.attrs["source_row_count"] == 5
    assert isinstance(run_group.attrs["profile_summary"], dict)


def test_backfill_keypoint_profiles_cli_dry_run_and_apply(capsys, monkeypatch, tmp_path: Path) -> None:
    zarr_ok = tmp_path / "ok_analysis.zarr"
    zarr_missing = tmp_path / "missing_analysis.zarr"

    root_ok = _make_keypoint_root()
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
    assert root_ok.get("analysis/keypoint_profile_runs") is None

    apply_rc = backfill_mod.main(
        [
            str(tmp_path),
            "--apply",
            "--run-name",
            "keypoint_profile_fixed",
        ]
    )
    assert apply_rc == 0
    apply_out = capsys.readouterr().out
    assert "updated" in apply_out
    assert "missing_source=1" in apply_out
    assert "updated=1" in apply_out
    assert root_ok["analysis/keypoint_profile_runs"].attrs["latest"] == "keypoint_profile_fixed"
    assert "profile_summary" in root_ok["analysis/keypoint_profile_runs/keypoint_profile_fixed"].attrs


def test_backfill_keypoint_profiles_populates_dataset_identity_from_registry(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "identity_training.zarr"
    root = _make_keypoint_root()
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
            "keypoint_profile_identity",
            "--zarr-use",
            "any",
        ]
    )
    assert rc == 0

    run_group = root["analysis/keypoint_profile_runs/keypoint_profile_identity"]
    profile_summary = run_group.attrs["profile_summary"]
    assert profile_summary["dataset"]["dataset_id"] == "dataset_identity"
    assert profile_summary["dataset"]["recording_id"] == "recording_identity"
    assert profile_summary["dataset"]["zarr_use"] == "training"
