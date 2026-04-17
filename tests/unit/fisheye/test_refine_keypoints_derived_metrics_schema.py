from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.refinement import refine_keypoints as mod


class _FakeArray:
    def __init__(
        self,
        data: Any,
        *,
        chunks: tuple[int, ...] | None = None,
        dtype: Any | None = None,
    ) -> None:
        try:
            array = np.asarray(data, dtype=dtype)
        except TypeError:
            array = np.asarray(data, dtype=object)
        self._data = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.chunks = chunks

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value) -> None:
        self._data[item] = value


class _FakeGroup:
    def __init__(self) -> None:
        self._children: dict[str, _FakeGroup | _FakeArray] = {}
        self.attrs: dict[str, Any] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        child = _FakeGroup()
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
        child = _FakeGroup()
        self._children[name] = child
        return child

    def create_array(
        self,
        name: str,
        *,
        data: Any | None = None,
        shape: tuple[int, ...] | None = None,
        chunks: tuple[int, ...] | None = None,
        dtype: Any | None = None,
        fill_value: Any | None = None,
        overwrite: bool = False,
    ) -> _FakeArray:
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_array(
                tail,
                data=data,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                fill_value=fill_value,
                overwrite=overwrite,
            )
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        if data is None:
            if shape is None:
                raise ValueError("shape is required when data is omitted")
            if fill_value is None:
                fill_value = 0
            try:
                data = np.full(shape, fill_value, dtype=dtype)
            except TypeError:
                data = np.full(shape, fill_value, dtype=object)
        arr = _FakeArray(data, chunks=chunks, dtype=dtype)
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


def _make_keypoint_root() -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["recording_id"] = "rec_derived_metrics_schema"

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["crop_signature"] = {"signature_version": 2, "crop_revision": 4}
    crop.attrs["crop_revision"] = 4
    crop.attrs["detect_review_status_ref"] = "refined_detect_runs/refined_detect_001"
    crop.attrs["video_source_path"] = "/tmp/source-video.mp4"

    keypoints_parent = root.create_group("keypoints_runs")
    keypoints_parent.attrs["latest"] = "keypoints_001"
    run = keypoints_parent.create_group("keypoints_001")
    run.attrs["method"] = "traditional_pose"
    run.attrs["source_crop_run"] = "crop_001"
    run.attrs["source_detect_run"] = "detect_001"
    run.attrs["source_refined_run"] = "refined_detect_001"
    run.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    run.attrs["pose_schema"] = {
        "name": "traditional_v1",
        "skeleton_id": "fish_v1",
        "kpt_shape": [3, 2],
        "nodes": [
            {"id": 0, "name": "swim_bladder"},
            {"id": 1, "name": "eye_left"},
            {"id": 2, "name": "eye_right"},
        ],
        "edges": [[0, 1], [0, 2], [1, 2]],
        "metadata": {
            "heading_computation": {
                "version": 1,
                "enabled": True,
                "origin": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
                "direction_from": {"op": "keypoint", "label": "swim_bladder"},
                "direction_to": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
                "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"],
            }
        },
    }

    run.create_array("frame_indices", data=np.asarray([0], dtype=np.int32), chunks=(1,))
    run.create_array("frame_counts", data=np.asarray([1], dtype=np.int32), chunks=(1,))
    run.create_array("n_rois", data=np.asarray([1], dtype=np.int32), chunks=(1,))
    run.create_array("detection_indices", data=np.asarray([0], dtype=np.int32), chunks=(1,))
    run.create_array(
        "keypoints_roi",
        data=np.asarray([[[1.0, 1.0], [5.0, 1.0], [3.0, 4.0]]], dtype=np.float64),
        chunks=(1, 3, 2),
    )
    run.create_array(
        "keypoints_img",
        data=np.asarray([[[11.0, 11.0], [15.0, 11.0], [13.0, 14.0]]], dtype=np.float64),
        chunks=(1, 3, 2),
    )
    run.create_array(
        "keypoints_norm",
        data=np.asarray([[[0.11, 0.11], [0.15, 0.11], [0.13, 0.14]]], dtype=np.float64),
        chunks=(1, 3, 2),
    )
    run.create_array("heading", data=np.asarray([0.0], dtype=np.float64), chunks=(1,))
    run.create_array("confidence", data=np.asarray([0.95], dtype=np.float64), chunks=(1,))
    run.create_array(
        "keypoint_confidences",
        data=np.asarray([[0.95, 0.95, 0.95]], dtype=np.float64),
        chunks=(1, 3),
    )
    run.create_array("effective_threshold", data=np.asarray([0.5], dtype=np.float64), chunks=(1,))
    run.create_array("effective_se2_radius", data=np.asarray([2.0], dtype=np.float64), chunks=(1,))
    run.create_array("detection_success", data=np.asarray([True], dtype=np.bool_), chunks=(1,))
    run.create_array("detection_source", data=np.asarray([0], dtype=np.int8), chunks=(1,))
    return root


def test_create_refined_keypoint_run_emits_derived_metrics_schema(monkeypatch) -> None:
    root = _make_keypoint_root()

    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(mod, "_resolve_status_context_from_root", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_status_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_emit_refined_keypoint_status", lambda **_kwargs: False)
    monkeypatch.setattr(
        mod,
        "delayed",
        lambda func: (lambda *args, **kwargs: (lambda: func(*args, **kwargs))),
    )
    monkeypatch.setattr(
        mod.dask,
        "compute",
        lambda *tasks, **_kwargs: tuple(task() for task in tasks),
    )
    monkeypatch.setattr(mod, "_write_reason_arrays", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "_process_refinement_chunk",
        lambda *_args, **_kwargs: {
            "start": 0,
            "end": 1,
            "quality": np.asarray([0], dtype=np.int8),
            "refined_success": np.asarray([True], dtype=np.bool_),
            "roi": np.asarray([[[1.0, 1.0], [5.0, 1.0], [3.0, 4.0]]], dtype=np.float64),
            "img": np.asarray([[[11.0, 11.0], [15.0, 11.0], [13.0, 14.0]]], dtype=np.float64),
            "norm": np.asarray([[[0.11, 0.11], [0.15, 0.11], [0.13, 0.14]]], dtype=np.float64),
            "heading": np.asarray([0.0], dtype=np.float64),
            "confidence": np.asarray([0.95], dtype=np.float64),
            "kp_conf": np.asarray([[0.95, 0.95, 0.95]], dtype=np.float64),
            "thresh": np.asarray([0.5], dtype=np.float64),
            "se2": np.asarray([2.0], dtype=np.float64),
            "flip_flags": np.asarray([False], dtype=np.bool_),
            "area": np.asarray([6.0], dtype=np.float64),
            "min_angle": np.asarray([45.0], dtype=np.float64),
            "triangle_angles": np.asarray([[45.0, 63.43494882, 71.56505118]], dtype=np.float64),
            "confidence_valid": np.asarray([True], dtype=np.bool_),
            "geometry_valid": np.asarray([True], dtype=np.bool_),
            "usable": np.asarray([True], dtype=np.bool_),
            "reason": np.asarray(["clean"], dtype=object),
            "stats": {
                "refined_success": 1,
                "flips_corrected": 0,
                "low_confidence": 0,
                "confidence_missing": 0,
                "geometry_issues": 0,
                "clean": 1,
                "usable": 1,
            },
        },
    )
    monkeypatch.setattr(
        mod,
        "refresh_refined_keypoint_heading_fields",
        lambda *_args, **_kwargs: {
            "heading_temporal_evaluable": 0,
            "heading_temporal_outlier_count": 0,
            "heading_temporal_outlier_rate_percent": 0.0,
            "temporal_heading_threshold_deg": 30.0,
            "temporal_heading_max_frame_gap": 1,
            "temporal_heading_status": "disabled",
            "temporal_heading_disabled_reason": "unit_test",
        },
    )
    monkeypatch.setattr(mod, "_run_post_refinement_diagnostics", lambda **_kwargs: {})
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "test",
            "is_dirty": False,
            "remote_url": "origin",
        },
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {
            "platform": {
                "hostname": "test-host",
                "python_version": "3.11",
                "system": "Linux",
                "release": "test",
                "machine": "x86_64",
            },
            "environment": {},
        },
    )

    run_name = mod.create_refined_keypoint_run(
        "/tmp/derived_metrics_schema_analysis.zarr",
        keypoint_run="keypoints_001",
        config={
            "refine_keypoints": {
                "scheduler": "single-threaded",
                "chunk_size": 1,
                "confidence_threshold": 0.1,
                "min_triangle_angle": 1.0,
                "min_triangle_area": 0.1,
            }
        },
        created_at_utc="2026-04-11T12:00:00+00:00",
    )

    refined = root["refined_keypoints_runs"][run_name]
    schema = dict(refined.attrs["derived_metrics_schema"])

    assert refined.attrs["source_refined_run"] == "refined_detect_001"
    assert refined.attrs["skeleton_id"] == "fish_v1"
    assert refined.attrs["kpt_shape"] == [3, 2]
    assert refined.attrs["pose_schema"]["skeleton_id"] == "fish_v1"
    assert refined.attrs["pose_schema"]["kpt_shape"] == [3, 2]
    assert refined.attrs["source_crop_storage_mode"] == "geometry_only"
    assert refined.attrs["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert refined.attrs["source_crop_revision"] == 4
    assert refined.attrs["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001"
    assert refined.attrs["keypoint_signature"]["source_refined_run"] == "refined_detect_001"
    assert refined.attrs["keypoint_signature"]["signature_version"] == 2
    assert refined.attrs["keypoint_signature"]["source_crop_storage_mode"] == "geometry_only"
    assert refined.attrs["keypoint_signature"]["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert refined.attrs["keypoint_signature"]["source_crop_revision"] == 4
    assert (
        refined.attrs["keypoint_signature"]["source_detect_review_status_ref"]
        == "refined_detect_runs/refined_detect_001"
    )
    assert refined.attrs["provenance"]["inputs"]["source_refined_run"] == "refined_detect_001"
    assert refined.attrs["provenance"]["inputs"]["source_crop_storage_mode"] == "geometry_only"
    assert refined.attrs["provenance"]["inputs"]["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert refined.attrs["provenance"]["inputs"]["source_crop_revision"] == 4
    assert (
        refined.attrs["provenance"]["inputs"]["source_detect_review_status_ref"]
        == "refined_detect_runs/refined_detect_001"
    )

    assert schema["schema_version"] == 1
    assert schema["entity_kind"] == "keypoint_roi"

    metric = schema["metrics"][0]
    assert metric["name"] == "eye_triangle_geometry"
    assert metric["kind"] == "triangle_3pt"
    assert metric["source"]["array"] == "keypoints_roi"
    assert metric["selectors"]["indices"] == [0, 1, 2]
    assert metric["selectors"]["labels"] == ["swim_bladder", "eye_left", "eye_right"]
    assert [item["array"] for item in metric["outputs"]] == [
        "triangle_area",
        "triangle_angles",
        "min_angle",
    ]

    gate = schema["quality_gates"][0]
    assert gate["name"] == "geometry_valid"
    assert gate["output"]["array"] == "geometry_valid"
    assert gate["conditions"][2]["threshold_attr"] == "summary_statistics.min_triangle_angle"
    assert gate["conditions"][3]["threshold_attr"] == "summary_statistics.min_triangle_area"
    assert gate["conditions"][4]["threshold_attr"] == "summary_statistics.max_triangle_area"
