from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.tune import detect_review as mod


class _FakeGroup:
    def __init__(self, *, path: str = "") -> None:
        self._children: dict[str, _FakeGroup | _FakeArray] = {}
        self.attrs: dict[str, Any] = {}
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        if name in self._children:
            raise ValueError(f"{name} already exists")
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def require_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if existing is not None:
            return existing
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

    def create_array(
        self,
        name: str,
        *,
        data: Any | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: Any | None = None,
        fill_value: Any = 0,
        overwrite: bool = False,
        **_kwargs: Any,
    ):
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        if data is None:
            if shape is None:
                raise ValueError("shape is required when data is omitted")
            try:
                resolved_dtype = np.dtype(dtype) if dtype is not None else np.float32
            except TypeError:
                resolved_dtype = object
            data = np.full(shape, fill_value, dtype=resolved_dtype)
        arr = _FakeArray(data)
        self._children[name] = arr
        return arr

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


class _FakeArray:
    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value) -> None:
        self._data[item] = value


def _seed_dense_curated_run(run: _FakeGroup) -> None:
    for name, data in (
        ("refined_row_ids", np.asarray([0, 1, 2], dtype=np.int64)),
        ("frame_indices", np.asarray([0, 1, 2], dtype=np.int32)),
        ("entity_ids", np.asarray([0, 0, 0], dtype=np.int32)),
        ("bbox_img_xyxy", np.asarray([[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1], [np.nan, np.nan, np.nan, np.nan]], dtype=np.float64)),
        ("status_codes", np.asarray([0, 0, 2], dtype=np.int8)),
        ("source_kind_codes", np.asarray([1, 3, 0], dtype=np.int8)),
        ("manual_edit_flags", np.asarray([False, True, True], dtype=bool)),
        ("source_detect_row_index", np.asarray([10, -1, -1], dtype=np.int32)),
        ("review_state_codes", np.asarray([1, 1, 1], dtype=np.int8)),
        ("keypoints_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("subject_mask_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("eye_mask_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("swim_bladder_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("confidence_scores", np.asarray([0.9, 0.8, np.nan], dtype=np.float32)),
        ("class_ids", np.asarray([0, 0, -1], dtype=np.int32)),
        ("detection_source", np.asarray([0, 0, 0], dtype=np.int8)),
        ("reason", np.asarray(["clean", "manual_correction", "manual_clear"], dtype=object)),
    ):
        run.create_array(name, data=data, overwrite=True)
    run.attrs["curated_row_storage"] = "dense_frame_entity_v3"


def test_pick_variant_prefers_dense_refined_root_when_present() -> None:
    refined = _FakeGroup(path="refined_detect_runs/refined_detect_001")
    _seed_dense_curated_run(refined)

    assert mod._pick_variant(refined, None) == "refined"


def test_load_dense_curated_edit_payload_reads_present_and_filtered_rows() -> None:
    refined = _FakeGroup(path="refined_detect_runs/refined_detect_001")
    _seed_dense_curated_run(refined)

    payload = mod._load_dense_curated_edit_payload(refined)  # type: ignore[arg-type]

    assert payload["frame_indices"].tolist() == [0, 1, 2]
    assert payload["status_labels"].tolist() == ["present", "present", "filtered_out"]
    assert payload["source_kind_labels"].tolist() == ["raw_detect", "manual", "none"]
    assert payload["manual_edit_flags"].tolist() == [False, True, True]
    assert payload["reason_labels"].tolist() == ["clean", "manual_correction", "manual_clear"]
    assert payload["storage_row_indices"].tolist() == [0, 1, 2]
    assert payload["frame_to_row"] == {0: 0, 1: 1, 2: 2}


def test_load_dense_curated_edit_payload_defaults_manual_edit_flags_to_false() -> None:
    refined = _FakeGroup(path="refined_detect_runs/refined_detect_001")
    _seed_dense_curated_run(refined)
    del refined._children["manual_edit_flags"]

    payload = mod._load_dense_curated_edit_payload(refined)  # type: ignore[arg-type]

    assert payload["manual_edit_flags"].tolist() == [False, False, False]


def test_load_dense_curated_edit_payload_rejects_multi_instance_sparse_runs() -> None:
    refined = _FakeGroup(path="refined_detect_runs/refined_detect_001")
    instances = refined.create_group("instances")
    for name, data in (
        ("refined_row_ids", np.asarray([10, 11], dtype=np.int64)),
        ("frame_indices", np.asarray([1, 1], dtype=np.int32)),
        ("frame_offsets", np.asarray([0, 0, 2, 2], dtype=np.int64)),
        ("bbox_img_xyxy", np.asarray([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]], dtype=np.float64)),
        ("source_kind_codes", np.asarray([1, 1], dtype=np.int8)),
        ("manual_edit_flags", np.asarray([False, False], dtype=bool)),
        ("source_detect_row_index", np.asarray([0, 1], dtype=np.int32)),
        ("frame_counts", np.asarray([0, 2, 0], dtype=np.int32)),
    ):
        instances.create_array(name, data=data, overwrite=True)

    try:
        mod._load_dense_curated_edit_payload(refined, total_frames=3)  # type: ignore[arg-type]
    except RuntimeError as exc:
        assert "does not yet support multi-instance refined runs" in str(exc)
    else:
        raise AssertionError("expected multi-instance sparse refined runs to be rejected by detect_review")


def test_load_refined_review_payload_uses_arena_slots_for_multi_arena_runs() -> None:
    root = _FakeGroup()
    analysis_meta = root.create_group("analysis_metadata")
    analysis_meta.attrs["subdish_mask_tuning"] = {
        "masks": [
            {"id": 0, "roi_pixels": [0, 0, 100, 100]},
            {"id": 1, "roi_pixels": [100, 0, 100, 100]},
        ]
    }

    refined = _FakeGroup(path="refined_detect_runs/refined_detect_001")
    instances = refined.create_group("instances")
    for name, data in (
        ("refined_row_ids", np.asarray([10, 11], dtype=np.int64)),
        ("frame_indices", np.asarray([0, 0], dtype=np.int32)),
        ("frame_offsets", np.asarray([0, 0, 2], dtype=np.int64)),
        ("bbox_img_xyxy", np.asarray([[40, 40, 60, 60], [140, 40, 160, 60]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.25, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2]], dtype=np.float64)),
        ("source_kind_codes", np.asarray([1, 1], dtype=np.int8)),
        ("manual_edit_flags", np.asarray([False, False], dtype=bool)),
        ("source_detect_row_index", np.asarray([0, 1], dtype=np.int32)),
        ("frame_counts", np.asarray([2, 0], dtype=np.int32)),
        ("confidence_scores", np.asarray([0.9, 0.8], dtype=np.float32)),
        ("class_ids", np.asarray([0, 0], dtype=np.int32)),
        ("reason", np.asarray(["clean", "clean"], dtype=object)),
    ):
        instances.create_array(name, data=data, overwrite=True)

    source = refined.create_group("source_detections")
    for name, data in (
        ("source_detect_row_index", np.asarray([0, 1, 2], dtype=np.int32)),
        ("frame_indices", np.asarray([0, 0, 1], dtype=np.int32)),
        ("bbox_img_xyxy", np.asarray([[40, 40, 60, 60], [140, 40, 160, 60], [140, 40, 160, 60]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.25, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2]], dtype=np.float64)),
        ("decision_codes", np.asarray([0, 0, 1], dtype=np.int8)),
        ("resolved_refined_row_id", np.asarray([10, 11, -1], dtype=np.int64)),
        ("confidence_scores", np.asarray([0.9, 0.8, 0.2], dtype=np.float32)),
        ("class_ids", np.asarray([0, 0, 0], dtype=np.int32)),
        ("review_notes", np.asarray(["", "", ""], dtype=object)),
        ("reason", np.asarray(["clean", "clean", "low_score"], dtype=object)),
    ):
        source.create_array(name, data=data, overwrite=True)

    payload = mod._load_refined_review_payload(  # type: ignore[arg-type]
        root,  # type: ignore[arg-type]
        refined,  # type: ignore[arg-type]
        total_frames=2,
        width=200,
        height=100,
    )

    assert mod._payload_review_axis(payload) == "frame_arena"
    assert payload["frame_indices"].tolist() == [0, 0, 1, 1]
    assert payload["arena_ids"].tolist() == [0, 1, 0, 1]
    assert payload["status_labels"].tolist() == ["present", "present", "missing", "filtered_out"]
    assert payload["source_detect_row_index"].tolist() == [0, 1, -1, 2]
    assert payload["refined_row_ids"].tolist() == [10, 11, -1, -1]
    assert payload["slot_to_row"][(1, 1)] == 3


def test_apply_manual_changes_to_arena_payload_updates_slot_and_source_surface() -> None:
    payload = {
        "frame_indices": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "arena_ids": np.asarray([0, 1, 0, 1], dtype=np.int32),
        "bbox_norm_coords": np.asarray(
            [
                [0.25, 0.5, 0.1, 0.2],
                [0.75, 0.5, 0.1, 0.2],
                [np.nan, np.nan, np.nan, np.nan],
                [0.75, 0.5, 0.1, 0.2],
            ],
            dtype=np.float64,
        ),
        "confidence_scores": np.asarray([0.9, 0.8, np.nan, 0.2], dtype=np.float32),
        "class_ids": np.asarray([0, 0, -1, 0], dtype=np.int32),
        "status_labels": np.asarray(["present", "present", "missing", "filtered_out"], dtype=object),
        "source_kind_labels": np.asarray(["raw_detect", "raw_detect", "none", "raw_detect"], dtype=object),
        "manual_edit_flags": np.asarray([False, False, False, False], dtype=bool),
        "reason_labels": np.asarray(["clean", "clean", "missing_detection", "low_score"], dtype=object),
        "source_detect_row_index": np.asarray([0, 1, -1, 2], dtype=np.int32),
        "detection_source": np.asarray([0, 0, 0, 0], dtype=np.int8),
        "refined_row_ids": np.asarray([10, 11, -1, -1], dtype=np.int64),
        "slot_to_row": {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3},
        "source_surface_source_detect_row_index": np.asarray([0, 1, 2], dtype=np.int32),
        "source_surface_frame_indices": np.asarray([0, 0, 1], dtype=np.int32),
        "source_surface_bbox_norm_coords": np.asarray(
            [[0.25, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2]],
            dtype=np.float64,
        ),
        "source_surface_decision_labels": np.asarray(["accepted", "accepted", "filtered"], dtype=object),
        "source_surface_reason_labels": np.asarray(["clean", "clean", "low_score"], dtype=object),
        "source_surface_confidence_scores": np.asarray([0.9, 0.8, 0.2], dtype=np.float32),
        "source_surface_class_ids": np.asarray([0, 0, 0], dtype=np.int32),
        "source_surface_review_notes": np.asarray(["", "", ""], dtype=object),
        "source_rows_by_slot": {
            (0, 0): np.asarray([0], dtype=np.int32),
            (0, 1): np.asarray([1], dtype=np.int32),
            (1, 1): np.asarray([2], dtype=np.int32),
        },
        "source_row_lookup": {0: 0, 1: 1, 2: 2},
    }

    updated, added_slots, removed_slots = mod._apply_manual_changes_to_arena_payload(
        payload,  # type: ignore[arg-type]
        manual_changes={
            (0, 0): None,
            (1, 1): np.asarray([0.74, 0.52, 0.12, 0.18], dtype=np.float64),
        },
        manual_score=0.95,
        manual_class_id=3,
    )

    assert added_slots == 1
    assert removed_slots == 1
    assert updated["status_labels"].tolist() == ["filtered_out", "present", "missing", "present"]
    assert updated["source_kind_labels"].tolist() == ["none", "raw_detect", "none", "manual"]
    assert updated["manual_edit_flags"].tolist() == [True, False, False, True]
    assert updated["reason_labels"].tolist() == ["manual_clear", "clean", "missing_detection", "manual_correction"]
    assert updated["source_surface_decision_labels"].tolist() == ["manual_clear", "accepted", "accepted"]
    assert updated["source_surface_reason_labels"].tolist() == ["manual_clear", "clean", "manual_correction"]
    assert updated["source_detect_row_index"].tolist() == [0, 1, -1, 2]


def test_update_curated_refined_after_manual_write_materializes_requested_group(monkeypatch) -> None:
    root = _FakeGroup()
    captured: dict[str, Any] = {}

    def _fake_env_info(**_kwargs):
        return {"git": {}, "platform": {}, "environment": {}}

    def _fake_materialize(root_arg, **kwargs):
        captured["root"] = root_arg
        captured["kwargs"] = kwargs
        return {
            "refined_detect_run": kwargs["refined_run_name"],
            "source_group": kwargs["source_group"],
            "rows_materialized": 2,
        }

    monkeypatch.setattr(mod, "get_environment_info", _fake_env_info)
    monkeypatch.setattr(mod, "materialize_refined_detect_curation", _fake_materialize)

    mod._update_curated_refined_after_manual_write(
        root,  # type: ignore[arg-type]
        zarr_path="/tmp/example.zarr",
        refined_run_name="refined_detect_001",
        source_group="manual_a",
    )

    assert captured["root"] is root
    assert captured["kwargs"]["refined_run_name"] == "refined_detect_001"
    assert captured["kwargs"]["source_group"] == "manual_a"


def test_update_curated_refined_after_manual_write_warns_without_raising(monkeypatch, capsys) -> None:
    root = _FakeGroup()

    def _fake_env_info(**_kwargs):
        return {"git": {}, "platform": {}, "environment": {}}

    def _fake_materialize(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "get_environment_info", _fake_env_info)
    monkeypatch.setattr(mod, "materialize_refined_detect_curation", _fake_materialize)

    mod._update_curated_refined_after_manual_write(
        root,  # type: ignore[arg-type]
        zarr_path="/tmp/example.zarr",
        refined_run_name="refined_detect_001",
        source_group="manual_a",
    )

    out = capsys.readouterr().out
    assert "Warning: refined detect curated surface update failed" in out


def test_write_dense_curated_edit_payload_maps_payload_rows_back_to_storage_rows(monkeypatch) -> None:
    root = _FakeGroup()
    root.attrs["width"] = 200
    root.attrs["height"] = 100
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["detect_review_status"] = {"state": "approved"}
    for name, data in (
        ("refined_row_ids", np.asarray([20, 10, 11], dtype=np.int64)),
        ("frame_indices", np.asarray([2, 0, 1], dtype=np.int32)),
        ("entity_ids", np.asarray([0, 0, 0], dtype=np.int32)),
        ("bbox_img_xyxy", np.asarray([[5, 5, 6, 6], [1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.3, 0.3, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]], dtype=np.float64)),
        ("status_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("source_kind_codes", np.asarray([1, 1, 1], dtype=np.int8)),
        ("manual_edit_flags", np.asarray([False, False, False], dtype=bool)),
        ("source_detect_row_index", np.asarray([2, 0, 1], dtype=np.int32)),
        ("review_state_codes", np.asarray([1, 1, 1], dtype=np.int8)),
        ("keypoints_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("subject_mask_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("eye_mask_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("swim_bladder_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("confidence_scores", np.asarray([0.7, 0.9, 0.8], dtype=np.float32)),
        ("class_ids", np.asarray([0, 0, 0], dtype=np.int32)),
        ("detection_source", np.asarray([0, 0, 0], dtype=np.int8)),
        ("reason", np.asarray(["clean", "clean", "clean"], dtype=object)),
    ):
        refined.create_array(name, data=data, overwrite=True)
    refined.attrs["curated_row_storage"] = "dense_frame_entity_v3"

    def _fake_env_info(**_kwargs):
        return {"git": {}, "platform": {}, "environment": {}}

    monkeypatch.setattr(mod, "get_environment_info", _fake_env_info)

    payload = mod._load_dense_curated_edit_payload(refined)  # type: ignore[arg-type]
    assert payload["frame_indices"].tolist() == [0, 1, 2]
    assert payload["storage_row_indices"].tolist() == [1, 2, 0]

    payload["bbox_norm_coords"][0] = np.asarray([0.55, 0.55, 0.2, 0.2], dtype=np.float64)
    payload["confidence_scores"][0] = np.float32(0.95)
    payload["source_kind_labels"][0] = "manual"
    payload["manual_edit_flags"][0] = True
    payload["reason_labels"][0] = "manual_correction"

    mod._write_dense_curated_edit_payload(
        root,  # type: ignore[arg-type]
        zarr_path="/tmp/example.zarr",
        refined_run_name="refined_detect_001",
        payload=payload,
        row_indices=np.asarray([0], dtype=np.int32),
        command_label="detect_review",
        source_context={"editor": "detect_review", "edit_mode": "manual"},
    )

    instances = refined["instances"]
    assert instances["frame_indices"][:].tolist() == [0, 1, 2]
    assert instances["bbox_norm_coords"][:].tolist() == [
        [0.55, 0.55, 0.2, 0.2],
        [0.2, 0.2, 0.1, 0.1],
        [0.3, 0.3, 0.1, 0.1],
    ]
    assert instances["manual_edit_flags"][:].tolist() == [True, False, False]
    assert read_reason_labels(instances).tolist() == ["manual_correction", "clean", "clean"]
    assert "reason" not in instances
    assert "bbox_norm_coords" not in refined


def test_write_dense_curated_edit_payload_sparse_branch_threads_instance_keys(tmp_path, monkeypatch) -> None:
    import zarr

    from fisheye.shared.instance_keys import (
        INSTANCE_KEY_CONTEXT_MANUAL_CURATION,
        INSTANCE_KEY_ORIGIN_CODE_MAP,
        mint_detection_instance_keys,
        resolve_recording_identity,
    )
    from fisheye.shared.refined_detect_curation import write_curated_refined_detect_root

    zarr_path = tmp_path / "rec.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["width"] = 200
    root.attrs["height"] = 100
    root.attrs["total_frames"] = 5

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_001"
    detect = detect_parent.create_group("detect_001")
    detect.create_array("frame_indices", data=np.asarray([1, 3], dtype=np.int32))
    detect.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.2]], dtype=np.float64),
    )
    detect.create_array("scores", data=np.asarray([0.9, 0.7], dtype=np.float32))
    detect.create_array("class_ids", data=np.asarray([0, 0], dtype=np.int32))
    detect.create_array("instance_key", data=np.asarray([111, 222], dtype=np.uint64))

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"

    write_curated_refined_detect_root(
        root,
        zarr_path=zarr_path,
        refined_run_name="refined_detect_001",
        frame_indices=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        entity_ids=np.zeros(5, dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [0.5, 0.5, 0.2, 0.4],
                [np.nan, np.nan, np.nan, np.nan],
                [0.25, 0.25, 0.1, 0.2],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        ),
        status_labels=np.asarray(["missing", "present", "missing", "present", "missing"], dtype=object),
        source_kind_labels=np.asarray(["none", "raw_detect", "none", "raw_detect", "none"], dtype=object),
        reason_labels=np.asarray(
            ["missing_detection", "clean", "missing_detection", "clean", "missing_detection"],
            dtype=object,
        ),
        source_detect_row_index=np.asarray([-1, 0, -1, 1, -1], dtype=np.int32),
        manual_edit_flags=np.zeros(5, dtype=bool),
        detection_source=np.zeros(5, dtype=np.int8),
        confidence_scores=np.asarray([np.nan, 0.9, np.nan, 0.7, np.nan], dtype=np.float32),
        class_ids=np.asarray([-1, 0, -1, 0, -1], dtype=np.int32),
    )

    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {"git": {}, "platform": {}, "environment": {}},
    )

    payload = mod._load_dense_curated_edit_payload(refined, total_frames=5)
    assert "source_surface_source_detect_row_index" in payload

    payload["bbox_norm_coords"][2] = np.asarray([0.3, 0.6, 0.1, 0.1], dtype=np.float64)
    payload["status_labels"][2] = "present"
    payload["source_kind_labels"][2] = "manual"
    payload["manual_edit_flags"][2] = True
    payload["reason_labels"][2] = "manual_correction"
    payload["confidence_scores"][2] = np.float32(1.0)
    payload["class_ids"][2] = np.int32(0)
    payload["source_detect_row_index"][2] = -1

    mod._write_dense_curated_edit_payload(
        root,
        zarr_path=str(zarr_path),
        refined_run_name="refined_detect_001",
        payload=payload,
        row_indices=np.asarray([2], dtype=np.int32),
        command_label="detect_review",
        source_context={"editor": "detect_review", "edit_mode": "manual"},
    )

    expected_minted = mint_detection_instance_keys(
        recording_identity=resolve_recording_identity(root.attrs, fallback_path=zarr_path),
        frame_indices=np.asarray([2], dtype=np.int64),
        bbox_norm_coords=np.asarray([[0.3, 0.6, 0.1, 0.1]], dtype=np.float64),
        class_ids=np.asarray([0], dtype=np.int64),
        payload_context=INSTANCE_KEY_CONTEXT_MANUAL_CURATION,
    )

    instances = refined["instances"]
    assert instances.attrs["instance_key_status"] == "present"
    assert instances["frame_indices"][:].tolist() == [1, 2, 3]
    assert instances["instance_key"][:].tolist() == [111, int(expected_minted[0]), 222]
    assert instances["instance_key_origin_codes"][:].tolist() == [
        INSTANCE_KEY_ORIGIN_CODE_MAP["copied_from_detect"],
        INSTANCE_KEY_ORIGIN_CODE_MAP["minted_at_curation"],
        INSTANCE_KEY_ORIGIN_CODE_MAP["copied_from_detect"],
    ]
    source_detections = refined["source_detections"]
    assert source_detections["instance_key"][:].tolist() == [111, 222]
