from __future__ import annotations

from typing import Any

import numpy as np
import zarr

import fisheye.shared.refined_detect_curation as refined_detect_curation_module
from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns
from fisheye.shared.refined_detect_curation import (
    REFINED_SOURCE_DETECTION_DECISION_CODE_MAP,
    REFINED_SOURCE_KIND_CODE_MAP,
    build_source_detection_decision_summary,
    extract_present_curated_rows,
    extract_source_detection_rows,
    materialize_refined_detect_curation,
    update_curated_refined_detect_rows,
    write_curated_refined_detect_root,
    write_curated_refined_detect_surfaces,
)


class _FakeArray:
    def __init__(self, data: Any, *, chunks: tuple[int, ...] | None = None) -> None:
        self._data = np.asarray(data, dtype=object) if isinstance(data, list) and data and isinstance(data[0], str) else np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.chunks = chunks

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value) -> None:
        self._data[item] = value


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
        child = type(self)(path=child_path)
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
        child = type(self)(path=child_path)
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
        fill_value: Any = 0,
        overwrite: bool = False,
        **_kwargs: object,
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
            try:
                resolved_dtype = np.dtype(dtype) if dtype is not None else np.float32
            except TypeError:
                resolved_dtype = object
            data = np.full(shape, fill_value, dtype=resolved_dtype)
        arr = _FakeArray(data, chunks=chunks)
        self._children[name] = arr
        return arr

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

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


class _OpaqueExistingGroup(_FakeGroup):
    def __init__(self, *, path: str = "") -> None:
        super().__init__(path=path)
        self._opaque_existing_groups: set[str] = set()

    def get(self, name: str):
        if "/" not in name and name in self._opaque_existing_groups and name in self._children:
            return None
        return super().get(name)

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" not in name and name in self._opaque_existing_groups and name in self._children:
            full_path = f"{self.path}/{name}" if self.path else name
            raise ValueError(
                f"A group exists in store LocalStore('file:///tmp/fake.zarr') at path '{full_path}'"
            )
        return super().create_group(name)


def _legacy_resolved_total_frames(root, refined_run) -> int:
    detect_group, _source_detect_run = refined_detect_curation_module._resolve_bound_source_detect_group(
        root,
        refined_run,
    )
    if detect_group is not None:
        total_frames = refined_detect_curation_module.as_int(detect_group.attrs.get("total_frames"))
        if total_frames is None:
            total_frames = refined_detect_curation_module.as_int(detect_group.attrs.get("n_frames"))
        if total_frames is not None and total_frames >= 0:
            return int(total_frames)
        for name in ("frame_counts", "n_detections"):
            if name in detect_group:
                return int(detect_group[name].shape[0])

    total_frames = refined_detect_curation_module.as_int(root.attrs.get("total_frames"))
    if total_frames is None:
        total_frames = refined_detect_curation_module.as_int(root.attrs.get("n_frames"))
    if total_frames is not None and total_frames >= 0:
        return int(total_frames)
    total_frames = refined_detect_curation_module.as_int(refined_run.attrs.get("coverage_frames_total"))
    if total_frames is not None and total_frames >= 0:
        return int(total_frames)
    raw = root.get("raw_video")
    if raw is not None:
        total_frames = refined_detect_curation_module.as_int(raw.attrs.get("total_frames"))
        if total_frames is None:
            total_frames = refined_detect_curation_module.as_int(raw.attrs.get("n_frames"))
        if total_frames is not None and total_frames >= 0:
            return int(total_frames)
    if refined_detect_curation_module.has_sparse_curated_refined_detect_instances_arrays(refined_run):
        instances = refined_detect_curation_module._get_child_group_if_present(refined_run, "instances")
        if instances is not None and "frame_counts" in instances:
            return int(instances["frame_counts"].shape[0])
        frame_indices_arr = (
            np.asarray(instances["frame_indices"][:], dtype=np.int32).reshape(-1)
            if instances is not None
            else np.empty((0,), dtype=np.int32)
        )
        if refined_detect_curation_module.has_curated_refined_source_detections_projection(refined_run):
            source_detections = refined_detect_curation_module._get_child_group_if_present(
                refined_run,
                "source_detections",
            )
            if source_detections is not None:
                source_frames = np.asarray(source_detections["frame_indices"][:], dtype=np.int32).reshape(-1)
                if source_frames.size:
                    frame_indices_arr = np.concatenate([frame_indices_arr, source_frames])
        if frame_indices_arr.size == 0:
            return 0
        return int(np.max(frame_indices_arr)) + 1
    frame_indices_arr = np.asarray(refined_run["frame_indices"][:], dtype=np.int32).reshape(-1)
    if frame_indices_arr.size == 0:
        return 0
    return int(np.max(frame_indices_arr)) + 1


def test_resolved_total_frames_matches_legacy_frame_count_slots(tmp_path) -> None:
    root = zarr.open_group(str(tmp_path / "curation_detect.zarr"), mode="w")
    detect = root.create_group("detect_runs").create_group("detect_001")
    detect.create_array("frame_counts", data=np.ones(4, dtype=np.int32), overwrite=True)
    refined = root.create_group("refined_detect_runs").create_group("refined_001")
    refined.attrs["source_detect_run"] = "detect_001"

    assert refined_detect_curation_module._resolved_total_frames(root, refined) == _legacy_resolved_total_frames(
        root,
        refined,
    )

    root = zarr.open_group(str(tmp_path / "curation_instances.zarr"), mode="w")
    refined = root.create_group("refined_detect_runs").create_group("refined_001")
    instances = refined.create_group("instances")
    instances.create_array("refined_row_ids", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    instances.create_array("frame_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    instances.create_array("frame_offsets", data=np.asarray([0, 1, 2, 2, 2], dtype=np.int64), overwrite=True)
    instances.create_array("bbox_img_xyxy", data=np.zeros((2, 4), dtype=np.float64), overwrite=True)
    instances.create_array("bbox_norm_coords", data=np.zeros((2, 4), dtype=np.float64), overwrite=True)
    instances.create_array("source_kind_codes", data=np.ones(2, dtype=np.int8), overwrite=True)
    instances.create_array("manual_edit_flags", data=np.zeros(2, dtype=bool), overwrite=True)
    instances.create_array("source_detect_row_index", data=np.arange(2, dtype=np.int32), overwrite=True)
    instances.create_array("frame_counts", data=np.asarray([1, 1, 0, 0], dtype=np.int32), overwrite=True)

    assert refined_detect_curation_module._resolved_total_frames(root, refined) == _legacy_resolved_total_frames(
        root,
        refined,
    )


def _write_sparse_group(
    group: _FakeGroup,
    *,
    frame_indices: np.ndarray,
    bbox_norm_coords: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    reason_labels: np.ndarray,
) -> None:
    frame_counts = np.bincount(frame_indices, minlength=5).astype(np.int32)
    group.create_array("frame_indices", data=frame_indices, overwrite=True)
    group.create_array("bbox_norm_coords", data=bbox_norm_coords, overwrite=True)
    group.create_array("scores", data=scores, overwrite=True)
    group.create_array("class_ids", data=class_ids, overwrite=True)
    group.create_array("frame_counts", data=frame_counts, overwrite=True)
    group.create_array("n_detections", data=frame_counts, overwrite=True)
    group.create_array("frame_mapping", data=frame_indices, overwrite=True)
    write_reason_columns(
        group,  # type: ignore[arg-type]
        np.asarray(reason_labels, dtype=object),
        chunk_size=max(1, int(frame_indices.shape[0])),
        overwrite=True,
    )


def _build_root(*, group_cls: type[_FakeGroup] = _FakeGroup) -> _FakeGroup:
    root = group_cls()
    root.attrs["width"] = 200
    root.attrs["height"] = 100
    root.attrs["total_frames"] = 5

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_001"
    detect_group = detect_parent.create_group("detect_001")
    _write_sparse_group(
        detect_group,
        frame_indices=np.asarray([1, 3], dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.2]],
            dtype=np.float64,
        ),
        scores=np.asarray([0.9, 0.7], dtype=np.float32),
        class_ids=np.asarray([0, 0], dtype=np.int32),
        reason_labels=np.asarray(["raw", "raw"], dtype=object),
    )

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["manual_review_latest"] = "manual_a"
    refined.attrs["detect_review_status"] = {
        "state": "approved",
        "notes": "screened",
    }

    interpolated = refined.create_group("interpolated")
    _write_sparse_group(
        interpolated,
        frame_indices=np.asarray([1, 3], dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.2]],
            dtype=np.float64,
        ),
        scores=np.asarray([0.9, 0.7], dtype=np.float32),
        class_ids=np.asarray([0, 0], dtype=np.int32),
        reason_labels=np.asarray(["clean", "interpolated"], dtype=object),
    )

    manual = refined.create_group("manual_a")
    _write_sparse_group(
        manual,
        frame_indices=np.asarray([1, 4], dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [[0.51, 0.5, 0.2, 0.4], [0.75, 0.5, 0.15, 0.2]],
            dtype=np.float64,
        ),
        scores=np.asarray([0.95, 0.8], dtype=np.float32),
        class_ids=np.asarray([0, 0], dtype=np.int32),
        reason_labels=np.asarray(["manual", "manual"], dtype=object),
    )
    return root


def _build_root_with_same_frame_raw_candidates(*, group_cls: type[_FakeGroup] = _FakeGroup) -> _FakeGroup:
    root = group_cls()
    root.attrs["width"] = 200
    root.attrs["height"] = 100
    root.attrs["total_frames"] = 3

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_dup"
    detect_group = detect_parent.create_group("detect_dup")
    _write_sparse_group(
        detect_group,
        frame_indices=np.asarray([1, 1], dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [[0.5, 0.5, 0.2, 0.4], [0.55, 0.5, 0.2, 0.4]],
            dtype=np.float64,
        ),
        scores=np.asarray([0.9, 0.6], dtype=np.float32),
        class_ids=np.asarray([0, 0], dtype=np.int32),
        reason_labels=np.asarray(["raw", "raw"], dtype=object),
    )
    frame_counts = np.asarray([0, 2, 0], dtype=np.int32)
    detect_group.create_array("frame_counts", data=frame_counts, overwrite=True)
    detect_group.create_array("n_detections", data=frame_counts, overwrite=True)

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_dup"
    refined = refined_parent.create_group("refined_detect_dup")
    refined.attrs["source_detect_run"] = "detect_dup"
    return root


def test_materialize_refined_detect_curation_writes_dense_root_from_interpolated() -> None:
    root = _build_root()

    payload = materialize_refined_detect_curation(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        source_group="interpolated",
    )

    assert payload["rows_materialized"] == 5
    refined = root["refined_detect_runs"]["refined_detect_001"]
    assert refined.attrs["curated_row_storage"] == "sparse_instances_v1"
    assert refined.attrs["curated_primary_surface"] == "instances"
    assert refined.attrs["entity_assignment_policy"] == "local_instance_index_per_frame"
    assert refined.attrs["summary_statistics"]["rows_interpolated"] == 2
    assert refined.attrs["summary_statistics"]["rows_manual_edited"] == 0
    assert "frame_indices" not in refined
    instances = refined["instances"]
    assert instances["frame_indices"][:].tolist() == [1, 3]
    assert instances["refined_row_ids"][:].tolist() == [1, 3]
    assert instances["frame_counts"][:].tolist() == [0, 1, 0, 1, 0]
    assert instances["frame_offsets"][:].tolist() == [0, 0, 1, 1, 2, 2]
    assert instances["source_kind_codes"][:].tolist() == [
        REFINED_SOURCE_KIND_CODE_MAP["interpolated"],
        REFINED_SOURCE_KIND_CODE_MAP["interpolated"],
    ]
    source_detections = refined["source_detections"]
    assert source_detections["source_detect_row_index"][:].tolist() == [0, 1]
    assert source_detections["frame_indices"][:].tolist() == [1, 3]
    assert source_detections["decision_codes"][:].tolist() == [
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["filtered"],
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["filtered"],
    ]
    assert source_detections["resolved_refined_row_id"][:].tolist() == [-1, -1]
    assert read_reason_labels(source_detections).tolist() == ["filtered", "filtered"]
    assert "reason" not in source_detections


def test_materialize_refined_detect_curation_preserves_row_ids_by_frame_identity() -> None:
    root = _build_root()

    materialize_refined_detect_curation(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        source_group="interpolated",
    )
    materialize_refined_detect_curation(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        source_group="manual_a",
    )

    refined = root["refined_detect_runs"]["refined_detect_001"]
    assert refined.attrs["summary_statistics"]["rows_manual"] == 2
    assert refined.attrs["summary_statistics"]["rows_manual_edited"] == 2
    instances = refined["instances"]
    assert instances["frame_indices"][:].tolist() == [1, 4]
    assert instances["refined_row_ids"][:].tolist() == [1, 4]
    assert instances["manual_edit_flags"][:].tolist() == [True, True]
    source_detections = refined["source_detections"]
    assert source_detections["decision_codes"][:].tolist() == [
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["filtered"],
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["filtered"],
    ]
    assert source_detections["resolved_refined_row_id"][:].tolist() == [-1, -1]
    assert read_reason_labels(source_detections).tolist() == ["filtered", "filtered"]


def test_update_curated_refined_detect_rows_rewrites_sparse_views_best_effort_for_legacy_source_free_edits() -> None:
    root = _build_root()
    materialize_refined_detect_curation(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        source_group="interpolated",
    )

    refined = root["refined_detect_runs"]["refined_detect_001"]
    refined.attrs["detect_review_status"] = {"state": "pending"}
    original_row_ids = refined["instances"]["refined_row_ids"][:].copy()
    original_frame_indices = refined["instances"]["frame_indices"][:].copy()

    payload = update_curated_refined_detect_rows(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        row_indices=np.asarray([1, 4], dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [[0.51, 0.5, 0.2, 0.4], [np.nan, np.nan, np.nan, np.nan]],
            dtype=np.float64,
        ),
        status_labels=np.asarray(["present", "filtered_out"], dtype=object),
        source_kind_labels=np.asarray(["manual", "none"], dtype=object),
        reason_labels=np.asarray(["manual_correction", "manual_clear"], dtype=object),
        source_detect_row_index=np.asarray([-1, -1], dtype=np.int32),
        manual_edit_flags=np.asarray([True, True], dtype=bool),
        detection_source=np.asarray([0, 0], dtype=np.int8),
        confidence_scores=np.asarray([0.95, np.nan], dtype=np.float32),
        class_ids=np.asarray([0, -1], dtype=np.int32),
        command="detect_review",
        env_info={"git": {}, "platform": {}, "environment": {}},
        source_context={"editor": "detect_review", "edit_mode": "manual"},
    )

    assert payload["rows_updated"] == 2
    assert refined["instances"]["refined_row_ids"][:].tolist() == original_row_ids.tolist()
    assert refined["instances"]["frame_indices"][:].tolist() == original_frame_indices.tolist()
    assert refined.attrs["summary_statistics"]["rows_manual_edited"] == 1
    instances = refined["instances"]
    assert instances["frame_indices"][:].tolist() == [1, 3]
    assert instances["refined_row_ids"][:].tolist() == [1, 3]
    assert read_reason_labels(instances).tolist() == ["manual_correction", "interpolated"]
    source_detections = refined["source_detections"]
    assert source_detections["decision_codes"][:].tolist() == [
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["filtered"],
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["filtered"],
    ]
    assert source_detections["resolved_refined_row_id"][:].tolist() == [-1, -1]
    assert read_reason_labels(source_detections).tolist() == ["filtered", "filtered"]


def test_write_curated_refined_detect_root_syncs_sparse_views_from_raw_links() -> None:
    root = _build_root()

    payload = write_curated_refined_detect_root(
        root,  # type: ignore[arg-type]
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
        reason_labels=np.asarray(["missing_detection", "clean", "missing_detection", "clean", "missing_detection"], dtype=object),
        source_detect_row_index=np.asarray([-1, 0, -1, 1, -1], dtype=np.int32),
        manual_edit_flags=np.zeros(5, dtype=bool),
        detection_source=np.zeros(5, dtype=np.int8),
        confidence_scores=np.asarray([np.nan, 0.9, np.nan, 0.7, np.nan], dtype=np.float32),
        class_ids=np.asarray([-1, 0, -1, 0, -1], dtype=np.int32),
    )

    assert payload["rows_present"] == 2
    refined = root["refined_detect_runs"]["refined_detect_001"]
    instances = refined["instances"]
    assert instances["frame_indices"][:].tolist() == [1, 3]
    assert instances["source_detect_row_index"][:].tolist() == [0, 1]
    source_detections = refined["source_detections"]
    assert source_detections["decision_codes"][:].tolist() == [
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["accepted"],
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["accepted"],
    ]
    assert source_detections["resolved_refined_row_id"][:].tolist() == [1, 3]
    assert read_reason_labels(source_detections).tolist() == ["clean", "clean"]


def test_write_curated_refined_detect_root_preserves_filtered_backlinks_in_source_detections() -> None:
    root = _build_root()

    write_curated_refined_detect_root(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        frame_indices=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        entity_ids=np.zeros(5, dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [0.25, 0.25, 0.1, 0.2],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        ),
        status_labels=np.asarray(["missing", "filtered_out", "missing", "present", "missing"], dtype=object),
        source_kind_labels=np.asarray(["none", "raw_detect", "none", "raw_detect", "none"], dtype=object),
        reason_labels=np.asarray(
            ["missing_detection", "filtered_jump", "missing_detection", "clean", "missing_detection"],
            dtype=object,
        ),
        source_detect_row_index=np.asarray([-1, 0, -1, 1, -1], dtype=np.int32),
        manual_edit_flags=np.zeros(5, dtype=bool),
        detection_source=np.zeros(5, dtype=np.int8),
        confidence_scores=np.asarray([np.nan, np.nan, np.nan, 0.7, np.nan], dtype=np.float32),
        class_ids=np.asarray([-1, -1, -1, 0, -1], dtype=np.int32),
    )

    source_detections = root["refined_detect_runs"]["refined_detect_001"]["source_detections"]
    assert source_detections["decision_codes"][:].tolist() == [
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["filtered"],
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["accepted"],
    ]
    assert source_detections["resolved_refined_row_id"][:].tolist() == [-1, 3]
    assert read_reason_labels(source_detections).tolist() == ["filtered_jump", "clean"]


def test_write_curated_refined_detect_root_does_not_infer_duplicates_from_same_frame_raw_rows() -> None:
    root = _build_root_with_same_frame_raw_candidates()

    write_curated_refined_detect_root(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_dup",
        frame_indices=np.asarray([0, 1, 2], dtype=np.int32),
        entity_ids=np.zeros(3, dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [0.5, 0.5, 0.2, 0.4],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        ),
        status_labels=np.asarray(["missing", "present", "missing"], dtype=object),
        source_kind_labels=np.asarray(["none", "raw_detect", "none"], dtype=object),
        reason_labels=np.asarray(["missing_detection", "clean", "missing_detection"], dtype=object),
        source_detect_row_index=np.asarray([-1, 0, -1], dtype=np.int32),
        manual_edit_flags=np.zeros(3, dtype=bool),
        detection_source=np.zeros(3, dtype=np.int8),
        confidence_scores=np.asarray([np.nan, 0.9, np.nan], dtype=np.float32),
        class_ids=np.asarray([-1, 0, -1], dtype=np.int32),
    )

    source_detections = root["refined_detect_runs"]["refined_detect_dup"]["source_detections"]
    assert source_detections["decision_codes"][:].tolist() == [
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["accepted"],
        REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["filtered"],
    ]
    assert read_reason_labels(source_detections).tolist() == ["clean", "filtered"]


def test_write_curated_refined_detect_surfaces_preserves_multi_instance_frames() -> None:
    root = _build_root_with_same_frame_raw_candidates()

    payload = write_curated_refined_detect_surfaces(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_dup",
        instance_frame_indices=np.asarray([1, 1], dtype=np.int32),
        instance_bbox_norm_coords=np.asarray(
            [[0.5, 0.5, 0.2, 0.4], [0.55, 0.5, 0.2, 0.4]],
            dtype=np.float64,
        ),
        instance_source_kind_labels=np.asarray(["raw_detect", "raw_detect"], dtype=object),
        instance_reason_labels=np.asarray(["clean", "clean"], dtype=object),
        instance_source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
        instance_manual_edit_flags=np.asarray([False, False], dtype=bool),
        instance_confidence_scores=np.asarray([0.9, 0.6], dtype=np.float32),
        instance_class_ids=np.asarray([0, 0], dtype=np.int32),
        source_detection_source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
        source_detection_frame_indices=np.asarray([1, 1], dtype=np.int32),
        source_detection_bbox_norm_coords=np.asarray(
            [[0.5, 0.5, 0.2, 0.4], [0.55, 0.5, 0.2, 0.4]],
            dtype=np.float64,
        ),
        source_detection_decision_labels=np.asarray(["accepted", "accepted"], dtype=object),
        source_detection_reason_labels=np.asarray(["clean", "clean"], dtype=object),
        source_detection_confidence_scores=np.asarray([0.9, 0.6], dtype=np.float32),
        source_detection_class_ids=np.asarray([0, 0], dtype=np.int32),
        command="refine_detect",
        env_info={"git": {}, "platform": {}, "environment": {}},
        source_context={"selection_policy": "quality_filtered_sparse_instances_no_interpolation"},
    )

    assert payload["rows_present"] == 2
    refined = root["refined_detect_runs"]["refined_detect_dup"]
    assert refined.attrs["entity_assignment_policy"] == "local_instance_index_per_frame"
    assert refined.attrs["row_identity_policy"] == "stable_sparse_refined_row_id"
    assert refined.attrs["summary_statistics"]["rows_present"] == 2
    assert refined.attrs["summary_statistics"]["rows_ambiguous"] == 1
    assert refined.attrs["summary_statistics"]["frames_multi_instance"] == 1
    assert refined.attrs["summary_statistics"]["max_instances_per_frame"] == 2

    instances = refined["instances"]
    assert instances["frame_indices"][:].tolist() == [1, 1]
    assert instances["source_detect_row_index"][:].tolist() == [0, 1]
    assert instances["frame_offsets"][:].tolist() == [0, 0, 2, 2]

    present_rows = extract_present_curated_rows(refined)  # type: ignore[arg-type]
    assert present_rows["frame_indices"].tolist() == [1, 1]
    assert present_rows["entity_ids"].tolist() == [0, 1]
    assert "status_codes" not in refined


def test_write_curated_refined_detect_surfaces_uses_detect_frame_source_dimensions() -> None:
    root = _FakeGroup()

    raw = root.create_group("raw_video")
    raw.attrs["video_width"] = 4512
    raw.attrs["video_height"] = 4512
    raw.attrs["total_frames"] = 3

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_seed"
    detect_group = detect_parent.create_group("detect_seed")
    detect_group.attrs["frame_source_shape"] = [3, 640, 640]
    detect_group.attrs["frame_source_path"] = "raw_video/images_ds"
    _write_sparse_group(
        detect_group,
        frame_indices=np.asarray([0], dtype=np.int32),
        bbox_norm_coords=np.asarray([[0.5, 0.5, 0.25, 0.25]], dtype=np.float64),
        scores=np.asarray([0.85], dtype=np.float32),
        class_ids=np.asarray([0], dtype=np.int32),
        reason_labels=np.asarray(["raw"], dtype=object),
    )

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_seed"
    refined = refined_parent.create_group("refined_detect_seed")
    refined.attrs["source_detect_run"] = "detect_seed"
    refined.attrs["coverage_frames_total"] = 3

    write_curated_refined_detect_surfaces(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_seed",
        instance_frame_indices=np.asarray([0], dtype=np.int32),
        instance_bbox_norm_coords=np.asarray([[0.5, 0.5, 0.25, 0.25]], dtype=np.float64),
        instance_source_kind_labels=np.asarray(["raw_detect"], dtype=object),
        instance_reason_labels=np.asarray(["clean"], dtype=object),
        instance_source_detect_row_index=np.asarray([0], dtype=np.int32),
        instance_confidence_scores=np.asarray([0.85], dtype=np.float32),
        instance_class_ids=np.asarray([0], dtype=np.int32),
        source_detection_source_detect_row_index=np.asarray([0], dtype=np.int32),
        source_detection_frame_indices=np.asarray([0], dtype=np.int32),
        source_detection_bbox_norm_coords=np.asarray([[0.5, 0.5, 0.25, 0.25]], dtype=np.float64),
        source_detection_decision_labels=np.asarray(["accepted"], dtype=object),
        source_detection_reason_labels=np.asarray(["clean"], dtype=object),
        source_detection_confidence_scores=np.asarray([0.85], dtype=np.float32),
        source_detection_class_ids=np.asarray([0], dtype=np.int32),
    )

    np.testing.assert_allclose(
        refined["instances"]["bbox_img_xyxy"][:],
        np.asarray([[240.0, 240.0, 400.0, 400.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        refined["source_detections"]["bbox_img_xyxy"][:],
        np.asarray([[240.0, 240.0, 400.0, 400.0]], dtype=np.float64),
    )
    assert refined["instances"]["bbox_img_xyxy"].chunks == (1, 4)
    assert refined["instances"]["bbox_norm_coords"].chunks == (1, 4)
    assert refined["source_detections"]["bbox_img_xyxy"].chunks == (1, 4)
    assert refined["source_detections"]["bbox_norm_coords"].chunks == (1, 4)


def test_write_curated_refined_detect_surfaces_uses_clipped_source_video_metadata() -> None:
    root = _build_root()
    del root.attrs["width"]
    del root.attrs["height"]
    root.attrs["source_video_metadata"] = {
        "schema_id": "palette.source_video_collection_metadata.v1",
        "layout": "clipped_video_collection",
        "width": 4512,
        "height": 4512,
        "total_frames": 5,
    }

    refined = root["refined_detect_runs/refined_detect_001"]
    write_curated_refined_detect_surfaces(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        instance_frame_indices=np.asarray([1], dtype=np.int32),
        instance_bbox_norm_coords=np.asarray(
            [[0.5, 0.5, 0.25, 0.25]], dtype=np.float64
        ),
        instance_source_kind_labels=np.asarray(["raw_detect"], dtype=object),
        instance_reason_labels=np.asarray(["clean"], dtype=object),
        instance_source_detect_row_index=np.asarray([0], dtype=np.int32),
        source_detection_source_detect_row_index=np.asarray([0], dtype=np.int32),
        source_detection_frame_indices=np.asarray([1], dtype=np.int32),
        source_detection_bbox_norm_coords=np.asarray(
            [[0.5, 0.5, 0.25, 0.25]], dtype=np.float64
        ),
        source_detection_decision_labels=np.asarray(["accepted"], dtype=object),
        source_detection_reason_labels=np.asarray(["clean"], dtype=object),
    )

    np.testing.assert_allclose(
        refined["instances"]["bbox_img_xyxy"][:],
        np.asarray([[1692.0, 1692.0, 2820.0, 2820.0]], dtype=np.float64),
    )


def test_write_curated_refined_detect_surfaces_uses_bound_detect_frame_counts_before_root_total() -> None:
    root = _FakeGroup()
    root.attrs["width"] = 4512
    root.attrs["height"] = 4512
    root.attrs["total_frames"] = 1_188_000

    detect_path = "clips/clip_000000/cameras/2010095/detect_runs/detect_clip"
    detect_group = root.require_group(detect_path)
    detect_group.attrs["frame_source_shape"] = [3, 640, 640]
    detect_group.create_array("frame_counts", data=np.asarray([1, 0, 1], dtype=np.int32), overwrite=True)

    refined_family_path = "clips/clip_000000/cameras/2010095/refined_detect_runs"
    refined_parent = root.require_group(refined_family_path)
    refined = refined_parent.create_group("refined_detect_clip")
    refined.attrs["source_detect_run"] = "detect_clip"
    refined.attrs["source_detect_path"] = detect_path
    refined.attrs["coverage_frames_total"] = 1_188_000

    payload = write_curated_refined_detect_surfaces(
        root,  # type: ignore[arg-type]
        refined_family_path=refined_family_path,
        refined_run_name="refined_detect_clip",
        instance_frame_indices=np.asarray([0, 2], dtype=np.int32),
        instance_bbox_norm_coords=np.asarray([[0.5, 0.5, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]], dtype=np.float64),
        instance_source_kind_labels=np.asarray(["raw_detect", "raw_detect"], dtype=object),
        instance_reason_labels=np.asarray(["clean", "clean"], dtype=object),
        instance_source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
        source_detection_source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
        source_detection_frame_indices=np.asarray([0, 2], dtype=np.int32),
        source_detection_bbox_norm_coords=np.asarray([[0.5, 0.5, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]], dtype=np.float64),
        source_detection_decision_labels=np.asarray(["accepted", "accepted"], dtype=object),
        source_detection_reason_labels=np.asarray(["clean", "clean"], dtype=object),
    )

    assert payload["rows_present"] == 2
    assert refined["instances"]["frame_counts"].shape == (3,)
    assert refined["instances"]["frame_offsets"].shape == (4,)
    assert refined["instances"]["frame_counts"][:].tolist() == [1, 0, 1]


def test_write_curated_refined_detect_surfaces_reuses_existing_sparse_groups_on_rerun() -> None:
    root = _build_root(group_cls=_OpaqueExistingGroup)

    common_kwargs = {
        "refined_run_name": "refined_detect_001",
        "instance_frame_indices": np.asarray([1, 3], dtype=np.int32),
        "instance_bbox_norm_coords": np.asarray(
            [[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.2]],
            dtype=np.float64,
        ),
        "instance_source_kind_labels": np.asarray(["raw_detect", "raw_detect"], dtype=object),
        "instance_source_detect_row_index": np.asarray([0, 1], dtype=np.int32),
        "instance_manual_edit_flags": np.asarray([False, False], dtype=bool),
        "instance_confidence_scores": np.asarray([0.9, 0.7], dtype=np.float32),
        "instance_class_ids": np.asarray([0, 0], dtype=np.int32),
        "source_detection_source_detect_row_index": np.asarray([0, 1], dtype=np.int32),
        "source_detection_frame_indices": np.asarray([1, 3], dtype=np.int32),
        "source_detection_bbox_norm_coords": np.asarray(
            [[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.2]],
            dtype=np.float64,
        ),
        "source_detection_decision_labels": np.asarray(["accepted", "accepted"], dtype=object),
        "source_detection_confidence_scores": np.asarray([0.9, 0.7], dtype=np.float32),
        "source_detection_class_ids": np.asarray([0, 0], dtype=np.int32),
    }

    write_curated_refined_detect_surfaces(
        root,  # type: ignore[arg-type]
        instance_reason_labels=np.asarray(["clean", "clean"], dtype=object),
        source_detection_reason_labels=np.asarray(["clean", "clean"], dtype=object),
        **common_kwargs,
    )

    refined = root["refined_detect_runs"]["refined_detect_001"]
    assert read_reason_labels(refined["instances"]).tolist() == ["clean", "clean"]
    assert read_reason_labels(refined["source_detections"]).tolist() == ["clean", "clean"]

    assert isinstance(refined, _OpaqueExistingGroup)
    refined._opaque_existing_groups.update({"instances", "source_detections"})

    write_curated_refined_detect_surfaces(
        root,  # type: ignore[arg-type]
        instance_reason_labels=np.asarray(["clean_rerun", "clean_rerun"], dtype=object),
        source_detection_reason_labels=np.asarray(["clean_rerun", "clean_rerun"], dtype=object),
        **common_kwargs,
    )

    assert read_reason_labels(refined["instances"]).tolist() == ["clean_rerun", "clean_rerun"]
    assert read_reason_labels(refined["source_detections"]).tolist() == ["clean_rerun", "clean_rerun"]


def test_get_or_create_child_group_uses_direct_store_path_open(monkeypatch) -> None:
    calls: list[tuple[object, str, str, bool]] = []

    class _StorePath:
        def __init__(self) -> None:
            self.store = object()

    class _GroupWithStorePath:
        def __init__(self) -> None:
            self.store_path = _StorePath()
            self.path = "refined_detect_runs/run_001"

    expected_group = object()

    def _fake_open_group(*, store, path, mode, use_consolidated=False):
        calls.append((store, path, mode, bool(use_consolidated)))
        return expected_group

    monkeypatch.setattr(refined_detect_curation_module.zarr, "open_group", _fake_open_group)

    result = refined_detect_curation_module._get_or_create_child_group(  # type: ignore[attr-defined]
        _GroupWithStorePath(),  # type: ignore[arg-type]
        "instances",
    )

    assert result is expected_group
    assert len(calls) == 1
    assert calls[0][1:] == ("refined_detect_runs/run_001/instances", "a", False)


def test_update_curated_refined_detect_rows_refreshes_review_state_without_row_changes() -> None:
    root = _build_root()
    materialize_refined_detect_curation(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        source_group="interpolated",
    )

    refined = root["refined_detect_runs"]["refined_detect_001"]
    refined.attrs["detect_review_status"] = {"state": "rejected"}

    payload = update_curated_refined_detect_rows(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        row_indices=np.asarray([], dtype=np.int32),
        command="set_detect_review_status",
        env_info={"git": {}, "platform": {}, "environment": {}},
        source_context={"editor": "detect_review", "edit_mode": "status_only"},
    )

    assert payload["rows_updated"] == 0
    assert refined.attrs["review_state_code_map"]["rejected"] == 4
    assert refined.attrs["summary_statistics"]["rows_present"] == 2


def test_extract_present_curated_rows_reads_sparse_instances_surface() -> None:
    root = _build_root()
    materialize_refined_detect_curation(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        source_group="manual_a",
    )

    refined = root["refined_detect_runs"]["refined_detect_001"]
    payload = extract_present_curated_rows(refined)  # type: ignore[arg-type]

    assert payload["frame_indices"].tolist() == [1, 4]
    assert payload["refined_row_ids"].tolist() == [1, 4]
    assert payload["entity_ids"].tolist() == [0, 0]
    assert payload["manual_edit_flags"].tolist() == [True, True]
    assert payload["detection_source"].tolist() == [0, 0]


def test_extract_source_detection_rows_and_summary_reads_projection() -> None:
    root = _build_root()

    write_curated_refined_detect_root(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        frame_indices=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        entity_ids=np.zeros(5, dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [0.25, 0.25, 0.1, 0.2],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        ),
        status_labels=np.asarray(["missing", "filtered_out", "missing", "present", "missing"], dtype=object),
        source_kind_labels=np.asarray(["none", "raw_detect", "none", "raw_detect", "none"], dtype=object),
        reason_labels=np.asarray(
            ["missing_detection", "manual_clear", "missing_detection", "clean", "missing_detection"],
            dtype=object,
        ),
        source_detect_row_index=np.asarray([-1, 0, -1, 1, -1], dtype=np.int32),
        manual_edit_flags=np.asarray([False, True, False, False, False], dtype=bool),
        detection_source=np.zeros(5, dtype=np.int8),
        confidence_scores=np.asarray([np.nan, np.nan, np.nan, 0.7, np.nan], dtype=np.float32),
        class_ids=np.asarray([-1, -1, -1, 0, -1], dtype=np.int32),
    )

    refined = root["refined_detect_runs"]["refined_detect_001"]
    payload = extract_source_detection_rows(refined)  # type: ignore[arg-type]
    summary = build_source_detection_decision_summary(refined)  # type: ignore[arg-type]

    assert payload["decision_labels"].tolist() == ["manual_clear", "accepted"]
    assert payload["resolved_refined_row_id"].tolist() == [-1, 3]
    assert summary == {
        "total_candidates": 2,
        "decision_accepted": 1,
        "decision_filtered": 0,
        "decision_duplicate": 0,
        "decision_manual_clear": 1,
    }


def _add_detect_instance_keys(root: _FakeGroup, *, keys: list[int], run: str = "detect_001") -> None:
    root["detect_runs"][run].create_array(
        "instance_key",
        data=np.asarray(keys, dtype=np.uint64),
        overwrite=True,
    )


def _dense_root_kwargs_with_manual_row() -> dict[str, Any]:
    """Five dense rows: detect-sourced present rows on frames 1 and 3, plus a
    hand-drawn manual present row on frame 2 (source_detect_row_index == -1)."""

    return dict(
        refined_run_name="refined_detect_001",
        frame_indices=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        entity_ids=np.zeros(5, dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [0.5, 0.5, 0.2, 0.4],
                [0.3, 0.6, 0.1, 0.1],
                [0.25, 0.25, 0.1, 0.2],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        ),
        status_labels=np.asarray(["missing", "present", "present", "present", "missing"], dtype=object),
        source_kind_labels=np.asarray(["none", "raw_detect", "manual", "raw_detect", "none"], dtype=object),
        reason_labels=np.asarray(
            ["missing_detection", "clean", "manual_add", "clean", "missing_detection"],
            dtype=object,
        ),
        source_detect_row_index=np.asarray([-1, 0, -1, 1, -1], dtype=np.int32),
        manual_edit_flags=np.asarray([False, False, True, False, False], dtype=bool),
        detection_source=np.zeros(5, dtype=np.int8),
        confidence_scores=np.asarray([np.nan, 0.9, np.nan, 0.7, np.nan], dtype=np.float32),
        class_ids=np.asarray([-1, 0, -1, 0, -1], dtype=np.int32),
    )


def _write_keyed_instance_rows(
    root: _FakeGroup,
    *,
    frame_indices: list[int],
    bbox_norm_coords: list[list[float]],
    source_detect_row_index: list[int],
    refined_row_ids: list[int],
    class_ids: list[int] | None = None,
    reason_labels: list[str] | None = None,
    source_kind_labels: list[str] | None = None,
    review_notes: list[str] | None = None,
) -> None:
    row_count = len(frame_indices)
    write_curated_refined_detect_surfaces(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        instance_frame_indices=np.asarray(frame_indices, dtype=np.int32),
        instance_bbox_norm_coords=np.asarray(bbox_norm_coords, dtype=np.float64).reshape(-1, 4),
        instance_source_kind_labels=np.asarray(
            source_kind_labels or ["manual"] * row_count,
            dtype=object,
        ),
        instance_reason_labels=np.asarray(reason_labels or ["manual_add"] * row_count, dtype=object),
        instance_source_detect_row_index=np.asarray(source_detect_row_index, dtype=np.int32),
        instance_manual_edit_flags=np.ones(row_count, dtype=bool),
        instance_confidence_scores=np.ones(row_count, dtype=np.float32),
        instance_class_ids=np.asarray(class_ids or [0] * row_count, dtype=np.int32),
        instance_refined_row_ids=np.asarray(refined_row_ids, dtype=np.int64),
        instance_review_notes=(
            np.asarray(review_notes, dtype=object)
            if review_notes is not None
            else None
        ),
        source_detection_source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
        source_detection_frame_indices=np.asarray([1, 3], dtype=np.int32),
        source_detection_bbox_norm_coords=np.asarray(
            [[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.2]],
            dtype=np.float64,
        ),
        source_detection_decision_labels=np.asarray(["filtered", "filtered"], dtype=object),
        source_detection_reason_labels=np.asarray(["raw", "raw"], dtype=object),
        source_detection_instance_key=np.asarray([111, 222], dtype=np.uint64),
    )


def _stored_key_by_refined_row_id(root: _FakeGroup) -> dict[int, int]:
    instances = root["refined_detect_runs/refined_detect_001/instances"]
    return dict(
        zip(
            np.asarray(instances["refined_row_ids"][:], dtype=np.int64).tolist(),
            np.asarray(instances["instance_key"][:], dtype=np.uint64).tolist(),
            strict=True,
        )
    )


def test_write_curated_refined_detect_root_mints_keys_for_manual_rows() -> None:
    from fisheye.shared.instance_keys import (
        INSTANCE_KEY_ORIGIN_CODE_MAP,
        mint_manual_curation_instance_keys,
    )

    root = _build_root()
    _add_detect_instance_keys(root, keys=[111, 222])

    write_curated_refined_detect_root(root, **_dense_root_kwargs_with_manual_row())  # type: ignore[arg-type]

    expected_minted = mint_manual_curation_instance_keys(
        recording_identity="unknown_recording",
        refined_row_ids=np.asarray([2], dtype=np.int64),
        frame_indices=np.asarray([2], dtype=np.int64),
        bbox_norm_coords=np.asarray([[0.3, 0.6, 0.1, 0.1]], dtype=np.float64),
        class_ids=np.asarray([-1], dtype=np.int64),
    )

    instances = root["refined_detect_runs"]["refined_detect_001"]["instances"]
    assert instances["frame_indices"][:].tolist() == [1, 2, 3]
    assert instances["instance_key"][:].tolist() == [111, int(expected_minted[0]), 222]
    assert instances["instance_key_origin_codes"][:].tolist() == [
        INSTANCE_KEY_ORIGIN_CODE_MAP["copied_from_detect"],
        INSTANCE_KEY_ORIGIN_CODE_MAP["minted_at_curation"],
        INSTANCE_KEY_ORIGIN_CODE_MAP["copied_from_detect"],
    ]
    assert instances.attrs["instance_key_status"] == "present"
    assert instances.attrs["instance_key_origin_code_map"] == dict(INSTANCE_KEY_ORIGIN_CODE_MAP)

    present_rows = extract_present_curated_rows(
        root["refined_detect_runs"]["refined_detect_001"]  # type: ignore[arg-type]
    )
    assert present_rows["instance_key"].tolist() == [111, int(expected_minted[0]), 222]
    assert present_rows["instance_key_origin_codes"].tolist() == [0, 1, 0]


def test_sampled_manual_instance_key_uses_acquisition_frame_identity() -> None:
    from fisheye.shared.instance_keys import mint_manual_curation_instance_keys

    root = _build_root()
    _add_detect_instance_keys(root, keys=[111, 222])
    raw_video = root.create_group("raw_video")
    raw_video.create_array(
        "original_frame_indices",
        data=np.asarray([100, 150, 200, 250, 300], dtype=np.int64),
        overwrite=True,
    )

    write_curated_refined_detect_root(root, **_dense_root_kwargs_with_manual_row())  # type: ignore[arg-type]

    expected = mint_manual_curation_instance_keys(
        recording_identity="unknown_recording",
        refined_row_ids=np.asarray([2], dtype=np.int64),
        frame_indices=np.asarray([200], dtype=np.int64),
        bbox_norm_coords=np.asarray([[0.3, 0.6, 0.1, 0.1]], dtype=np.float64),
        class_ids=np.asarray([-1], dtype=np.int64),
    )
    instances = root["refined_detect_runs/refined_detect_001/instances"]
    assert instances["frame_indices"][:].tolist() == [1, 2, 3]
    assert instances["instance_key"][:].tolist() == [111, int(expected[0]), 222]


def test_write_curated_refined_detect_root_minted_keys_are_deterministic_across_rewrites() -> None:
    root = _build_root()
    _add_detect_instance_keys(root, keys=[111, 222])

    write_curated_refined_detect_root(root, **_dense_root_kwargs_with_manual_row())  # type: ignore[arg-type]
    first = root["refined_detect_runs"]["refined_detect_001"]["instances"]["instance_key"][:].tolist()

    write_curated_refined_detect_root(root, **_dense_root_kwargs_with_manual_row())  # type: ignore[arg-type]
    second = root["refined_detect_runs"]["refined_detect_001"]["instances"]["instance_key"][:].tolist()

    assert first == second


def test_manual_instance_key_survives_bbox_metadata_edit_and_physical_reorder() -> None:
    root = _build_root()
    _add_detect_instance_keys(root, keys=[111, 222])
    _write_keyed_instance_rows(
        root,
        frame_indices=[1, 2],
        bbox_norm_coords=[[0.5, 0.5, 0.2, 0.4], [0.3, 0.6, 0.1, 0.1]],
        source_detect_row_index=[0, -1],
        refined_row_ids=[10, 11],
        source_kind_labels=["raw_detect", "manual"],
        reason_labels=["clean", "manual_add"],
        review_notes=["", "initial"],
    )
    before = _stored_key_by_refined_row_id(root)

    _write_keyed_instance_rows(
        root,
        frame_indices=[2, 1],
        bbox_norm_coords=[[0.7, 0.2, 0.2, 0.15], [0.5, 0.5, 0.2, 0.4]],
        source_detect_row_index=[-1, 0],
        refined_row_ids=[11, 10],
        class_ids=[3, 0],
        source_kind_labels=["manual", "raw_detect"],
        reason_labels=["manual_correction", "approved"],
        review_notes=["bbox, class, and note changed", "approved"],
    )

    assert _stored_key_by_refined_row_id(root) == before
    instances = root["refined_detect_runs/refined_detect_001/instances"]
    assert instances["refined_row_ids"][:].tolist() == [10, 11]
    assert instances["review_notes"][:].tolist() == ["approved", "bbox, class, and note changed"]


def test_deleted_manual_identity_is_not_reused_when_identical_box_is_readded() -> None:
    import pytest

    root = _build_root()
    _add_detect_instance_keys(root, keys=[111, 222])
    bbox = [[0.3, 0.6, 0.1, 0.1]]
    _write_keyed_instance_rows(
        root,
        frame_indices=[2],
        bbox_norm_coords=bbox,
        source_detect_row_index=[-1],
        refined_row_ids=[10],
    )
    deleted_key = _stored_key_by_refined_row_id(root)[10]

    _write_keyed_instance_rows(
        root,
        frame_indices=[],
        bbox_norm_coords=[],
        source_detect_row_index=[],
        refined_row_ids=[],
    )
    refined = root["refined_detect_runs/refined_detect_001"]
    assert refined.attrs["next_refined_row_id"] == 11

    with pytest.raises(ValueError, match="reuse retired identity"):
        _write_keyed_instance_rows(
            root,
            frame_indices=[2],
            bbox_norm_coords=bbox,
            source_detect_row_index=[-1],
            refined_row_ids=[10],
        )

    _write_keyed_instance_rows(
        root,
        frame_indices=[2],
        bbox_norm_coords=bbox,
        source_detect_row_index=[-1],
        refined_row_ids=[-1],
    )
    readded = _stored_key_by_refined_row_id(root)
    assert list(readded) == [11]
    assert readded[11] != deleted_key
    assert refined.attrs["next_refined_row_id"] == 12


def test_split_and_merge_tombstone_inputs_and_mint_fresh_identities() -> None:
    root = _build_root()
    _add_detect_instance_keys(root, keys=[111, 222])
    _write_keyed_instance_rows(
        root,
        frame_indices=[2],
        bbox_norm_coords=[[0.5, 0.5, 0.2, 0.2]],
        source_detect_row_index=[-1],
        refined_row_ids=[20],
    )
    original_key = _stored_key_by_refined_row_id(root)[20]

    _write_keyed_instance_rows(
        root,
        frame_indices=[2, 2],
        bbox_norm_coords=[[0.45, 0.5, 0.1, 0.2], [0.55, 0.5, 0.1, 0.2]],
        source_detect_row_index=[-1, -1],
        refined_row_ids=[-1, -1],
    )
    split_keys = _stored_key_by_refined_row_id(root)
    assert list(split_keys) == [21, 22]
    assert original_key not in split_keys.values()

    _write_keyed_instance_rows(
        root,
        frame_indices=[2],
        bbox_norm_coords=[[0.5, 0.5, 0.2, 0.2]],
        source_detect_row_index=[-1],
        refined_row_ids=[-1],
    )
    merged_keys = _stored_key_by_refined_row_id(root)
    assert list(merged_keys) == [23]
    assert merged_keys[23] not in split_keys.values()
    assert root["refined_detect_runs/refined_detect_001"].attrs["next_refined_row_id"] == 24


def test_surviving_detector_identity_cannot_be_retargeted_to_another_source_row() -> None:
    import pytest

    root = _build_root()
    _add_detect_instance_keys(root, keys=[111, 222])
    _write_keyed_instance_rows(
        root,
        frame_indices=[1],
        bbox_norm_coords=[[0.5, 0.5, 0.2, 0.4]],
        source_detect_row_index=[0],
        refined_row_ids=[10],
        source_kind_labels=["raw_detect"],
    )

    with pytest.raises(ValueError, match="refusing to retarget"):
        _write_keyed_instance_rows(
            root,
            frame_indices=[3],
            bbox_norm_coords=[[0.25, 0.25, 0.1, 0.2]],
            source_detect_row_index=[1],
            refined_row_ids=[10],
            source_kind_labels=["raw_detect"],
        )


def test_write_curated_refined_detect_root_rejects_duplicate_copied_instance_keys() -> None:
    import pytest

    root = _build_root()
    _add_detect_instance_keys(root, keys=[333, 333])

    with pytest.raises(ValueError, match="not unique after combining"):
        write_curated_refined_detect_root(root, **_dense_root_kwargs_with_manual_row())  # type: ignore[arg-type]


def test_write_curated_refined_detect_root_without_detect_keys_stamps_missing_and_warns(caplog) -> None:
    import logging

    root = _build_root()  # detect run has no instance_key array

    with caplog.at_level(logging.WARNING, logger="fisheye.shared.refined_detect_curation"):
        write_curated_refined_detect_root(root, **_dense_root_kwargs_with_manual_row())  # type: ignore[arg-type]

    instances = root["refined_detect_runs"]["refined_detect_001"]["instances"]
    assert "instance_key" not in instances
    assert "instance_key_origin_codes" not in instances
    assert instances.attrs["instance_key_status"] == "missing"
    assert any("no instance_key array" in record.message for record in caplog.records)
    assert any("without instance_key values" in record.message for record in caplog.records)


def test_write_curated_refined_detect_root_mints_for_out_of_range_source_rows_and_warns(caplog) -> None:
    import logging

    root = _build_root()
    _add_detect_instance_keys(root, keys=[111, 222])

    kwargs = _dense_root_kwargs_with_manual_row()
    # Point the frame-3 row at a detect row that does not exist.
    kwargs["source_detect_row_index"] = np.asarray([-1, 0, -1, 7, -1], dtype=np.int32)

    with caplog.at_level(logging.WARNING, logger="fisheye.shared.refined_detect_curation"):
        write_curated_refined_detect_root(root, **kwargs)  # type: ignore[arg-type]

    instances = root["refined_detect_runs"]["refined_detect_001"]["instances"]
    assert instances["instance_key_origin_codes"][:].tolist() == [0, 1, 1]
    keys = instances["instance_key"][:].tolist()
    assert keys[0] == 111
    assert len(set(keys)) == 3
    assert any("out-of-range" in record.message for record in caplog.records)


def test_write_curated_refined_detect_surfaces_rejects_origin_codes_without_keys() -> None:
    import pytest

    root = _build_root_with_same_frame_raw_candidates()

    with pytest.raises(ValueError, match="instance_key_origin_codes requires instance_key"):
        write_curated_refined_detect_surfaces(
            root,  # type: ignore[arg-type]
            refined_run_name="refined_detect_dup",
            instance_frame_indices=np.asarray([1], dtype=np.int32),
            instance_bbox_norm_coords=np.asarray([[0.5, 0.5, 0.2, 0.4]], dtype=np.float64),
            instance_source_kind_labels=np.asarray(["raw_detect"], dtype=object),
            instance_reason_labels=np.asarray(["clean"], dtype=object),
            instance_source_detect_row_index=np.asarray([0], dtype=np.int32),
            instance_key_origin_codes=np.asarray([0], dtype=np.int8),
        )
