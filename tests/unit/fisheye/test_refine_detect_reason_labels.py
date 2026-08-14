import numpy as np
import pytest
import zarr

from fisheye.refinement.refine_detect import (
    _apply_dish_mask_quality_gate,
    _apply_registered_detection_gate,
    _build_sparse_refined_inputs_from_filtered,
    _filtered_reason_from_quality_label,
    _get_sampled_frame_count,
    _require_quality_matches_active_canonical_manifest,
    _reject_deprecated_interpolation_overrides,
    _resolve_detection_quality_labels,
    _select_per_frame_top_k_raw_indices,
    create_refined_run,
    get_refinement_parameters,
)


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data
        self.shape = data.shape
        self.dtype = data.dtype

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value) -> None:
        self._data[item] = value


class _FakeGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self._children: dict[str, object] = {}

    def create_group(self, name: str):
        grp = _FakeGroup()
        self._children[name] = grp
        return grp

    def create_array(
        self,
        name: str,
        *,
        data=None,
        shape=None,
        fill_value=None,
        overwrite: bool = False,
        **_kwargs,
    ):
        if not overwrite and name in self._children:
            raise ValueError(f"Array '{name}' already exists")
        if data is None:
            if shape is None:
                raise ValueError("shape is required when data is None")
            arr = np.full(shape, fill_value if fill_value is not None else "", dtype=object)
        else:
            arr = np.asarray(data).copy()
        wrapped = _FakeArray(arr)
        self._children[name] = wrapped
        return wrapped

    def get(self, name: str):
        return self._children.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def __getitem__(self, name: str):
        return self._children[name]

    def __delitem__(self, name: str) -> None:
        del self._children[name]


def _fake_root() -> _FakeGroup:
    return _FakeGroup()


def _legacy_get_sampled_frame_count(root, detect_group):
    raw = root.get("raw_video")
    if raw is not None:
        if "original_frame_indices" in raw:
            return int(raw["original_frame_indices"].shape[0])
        if "images_ds" in raw:
            return int(raw["images_ds"].shape[0])
        if "images_full" in raw:
            return int(raw["images_full"].shape[0])
    if detect_group is not None and "frame_counts" in detect_group:
        return int(detect_group["frame_counts"].shape[0])
    return None


def test_get_sampled_frame_count_matches_legacy_frame_axis_resolution(tmp_path) -> None:
    zarr_path = tmp_path / "sampled.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("original_frame_indices", data=np.asarray([0, 2, 4], dtype=np.int32), overwrite=True)
    detect = root.create_group("detect_runs").create_group("detect_001")
    detect.create_array("frame_counts", data=np.ones(5, dtype=np.int32), overwrite=True)

    assert _get_sampled_frame_count(root, detect) == _legacy_get_sampled_frame_count(root, detect)

    del root["raw_video/original_frame_indices"]
    assert _get_sampled_frame_count(root, detect) == _legacy_get_sampled_frame_count(root, detect)


def test_get_sampled_frame_count_reads_strict_instance_offsets(tmp_path) -> None:
    root = zarr.open_group(str(tmp_path / "strict_sampled.zarr"), mode="w")
    detect = root.create_group("detect_runs").create_group("detect_001")
    instances = detect.create_group("instances")
    instances.create_array(
        "frame_indices",
        data=np.asarray([0, 2, 2], dtype=np.int32),
        overwrite=True,
    )
    instances.create_array(
        "bbox_norm_coords",
        data=np.zeros((3, 4), dtype=np.float32),
        overwrite=True,
    )
    instances.create_array(
        "frame_row_offsets",
        data=np.asarray([0, 1, 1, 3], dtype=np.int64),
        overwrite=True,
    )

    assert _get_sampled_frame_count(root, detect) == 3


def test_create_sampled_refined_seed_from_strict_instances_without_selectors(
    tmp_path,
) -> None:
    zarr_path = tmp_path / "strict_training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update(
        {
            "zarr_purpose": "training",
            "recording_id": "recording-a",
            "width": 8,
            "height": 8,
            "source_video_width": 8,
            "source_video_height": 8,
        }
    )
    raw = root.create_group("raw_video")
    raw.attrs.update(
        {
            "import_mode": "sampled",
            "import_purpose": "training_data",
            "source_frame_count": 100,
            "total_frames": 3,
        }
    )
    raw.create_array(
        "images_full",
        data=np.zeros((3, 8, 8), dtype=np.uint8),
        overwrite=True,
    )
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([10, 20, 30], dtype=np.int32),
        overwrite=True,
    )
    detect_parent = root.create_group("detect_runs")
    detect = detect_parent.create_group("detect_seed")
    instances = detect.create_group("instances")
    arrays = {
        "frame_indices": np.asarray([0, 2], dtype=np.int32),
        "source_acquisition_frame_index": np.asarray([10, 30], dtype=np.int64),
        "instance_key": np.asarray([101, 202], dtype=np.uint64),
        "bbox_norm_coords": np.asarray(
            [[0.5, 0.5, 0.25, 0.25], [0.25, 0.25, 0.2, 0.2]],
            dtype=np.float32,
        ),
        "bbox_img_xyxy": np.asarray(
            [[3.0, 3.0, 5.0, 5.0], [1.2, 1.2, 2.8, 2.8]],
            dtype=np.float32,
        ),
        "centers_img_xy": np.asarray([[4.0, 4.0], [2.0, 2.0]], dtype=np.float32),
        "scores": np.asarray([0.9, 0.8], dtype=np.float32),
        "class_ids": np.asarray([0, 0], dtype=np.int32),
        "frame_row_offsets": np.asarray([0, 1, 1, 2], dtype=np.int64),
    }
    for name, values in arrays.items():
        instances.create_array(name, data=values, overwrite=True)

    created = create_refined_run(
        str(zarr_path),
        detect_run="detect_seed",
        config={"refine_detect": {}},
        require_detect_quality=True,
        per_frame_top_k=1,
        refined_run_name="refined_seed",
        stage_selector_eligible=False,
        emit_completion_status=False,
    )

    assert created == "refined_seed"
    direct = zarr.open_group(
        str(zarr_path), mode="r", use_consolidated=False
    )
    refined_parent = direct["refined_detect_runs"]
    assert "latest" not in refined_parent.attrs
    assert "latest_complete" not in refined_parent.attrs
    assert "latest_pending" not in refined_parent.attrs
    refined = refined_parent[created]
    assert refined.attrs["stage_selector_eligible"] is False
    assert refined.attrs["palette_run_completion_status"] == "complete"
    assert refined["instances/frame_indices"][:].tolist() == [0, 2]
    assert refined["instances/instance_key"][:].tolist() == [101, 202]


def test_get_refinement_parameters_defaults_max_gap_to_0() -> None:
    params, source = get_refinement_parameters(config={})
    assert source == "config"
    assert params["max_gap"] == 0
    assert params["interpolation_method"] == "disabled"


def test_reject_deprecated_interpolation_overrides() -> None:
    with pytest.raises(ValueError, match="Interpolation overrides are deprecated and unsupported"):
        _reject_deprecated_interpolation_overrides(max_gap=5, interpolation_method=None)


def test_active_refinement_rejects_stale_quality_manifest_digest() -> None:
    quality = _FakeGroup()
    quality.attrs["source_detect_run_manifest_digest"] = "a" * 64
    manifest = {"payload_digest": "b" * 64}

    with pytest.raises(ValueError, match="different canonical manifest digest"):
        _require_quality_matches_active_canonical_manifest(quality, manifest)

    quality.attrs["source_detect_run_manifest_digest"] = "b" * 64
    _require_quality_matches_active_canonical_manifest(quality, manifest)


def test_registered_gate_modes_are_explicit_and_fail_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = np.asarray([10, 20, 30], dtype=np.uint64)
    labels = np.asarray([0, 2, 0], dtype=np.int8)

    off_labels, off = _apply_registered_detection_gate(
        zarr_path=tmp_path / "recording.zarr",
        source_detect_path="detect_runs/detect_native",
        raw_instance_keys=keys,
        detection_quality_labels=labels,
        requirement="off",
        gate_run=None,
    )
    np.testing.assert_array_equal(off_labels, labels)
    assert off["status"] == "off"
    assert off["applied"] is False

    available_labels, unavailable = _apply_registered_detection_gate(
        zarr_path=tmp_path / "recording.zarr",
        source_detect_path="detect_runs/detect_native",
        raw_instance_keys=keys,
        detection_quality_labels=labels,
        requirement="if_available",
        gate_run=None,
    )
    np.testing.assert_array_equal(available_labels, labels)
    assert unavailable["status"] == "unavailable"

    with pytest.raises(ValueError, match="No exact registered detection gate"):
        _apply_registered_detection_gate(
            zarr_path=tmp_path / "recording.zarr",
            source_detect_path="detect_runs/detect_native",
            raw_instance_keys=keys,
            detection_quality_labels=labels,
            requirement="required",
            gate_run=None,
        )

    def valid_gate(*_args, **_kwargs):
        return {
            "inside": np.asarray([True, False, True]),
            "gate_run": "gate_exact",
            "gate_group_path": "analysis/detection_gate_runs/gate_exact",
            "selection_record_sha256": "a" * 64,
            "selected_candidate_record_sha256": "b" * 64,
            "ordered_instance_key_coverage_exact": True,
        }

    monkeypatch.setattr(
        "fisheye.analysis_workflows.materializers.registered_detection_gate."
        "validate_registered_detection_gate_consumption",
        valid_gate,
    )
    gated_labels, applied = _apply_registered_detection_gate(
        zarr_path=tmp_path / "recording.zarr",
        source_detect_path="detect_runs/detect_native",
        raw_instance_keys=keys,
        detection_quality_labels=labels,
        requirement="required",
        gate_run="gate_exact",
    )
    assert gated_labels.tolist() == [0, 6, 0]
    assert applied["status"] == "applied"
    assert applied["rejected_count"] == 1
    assert _filtered_reason_from_quality_label(6) == (
        "outside_registered_detection_gate"
    )


def test_if_available_records_invalid_gate_but_required_raises(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_gate(*_args, **_kwargs):
        raise ValueError("ordered instance_key coverage differs")

    monkeypatch.setattr(
        "fisheye.analysis_workflows.materializers.registered_detection_gate."
        "validate_registered_detection_gate_consumption",
        invalid_gate,
    )
    keys = np.asarray([10, 20], dtype=np.uint64)
    labels = np.zeros(2, dtype=np.int8)
    result_labels, result = _apply_registered_detection_gate(
        zarr_path=tmp_path / "recording.zarr",
        source_detect_path="detect_runs/detect_native",
        raw_instance_keys=keys,
        detection_quality_labels=labels,
        requirement="if_available",
        gate_run="gate_stale",
    )
    np.testing.assert_array_equal(result_labels, labels)
    assert result["status"] == "rejected_invalid"

    with pytest.raises(ValueError, match="Required registered detection gate is invalid"):
        _apply_registered_detection_gate(
            zarr_path=tmp_path / "recording.zarr",
            source_detect_path="detect_runs/detect_native",
            raw_instance_keys=keys,
            detection_quality_labels=labels,
            requirement="required",
            gate_run="gate_stale",
        )


def test_resolve_detection_quality_labels_requires_quality_by_default() -> None:
    root = _fake_root()
    detect_group = root.create_group("detect_runs").create_group("detect_001")

    with pytest.raises(ValueError, match="Missing usable detect_quality context"):
        _resolve_detection_quality_labels(
            root,
            detect_group,
            detect_run="detect_001",
            source_detect_path="detect_runs/detect_001",
            quality_run=None,
            quality_group_path=None,
            total_detections=3,
            require_quality=True,
            allow_missing_reason="test",
            console=None,
        )


def test_resolve_detection_quality_labels_allows_explicit_opt_out() -> None:
    root = _fake_root()
    detect_group = root.create_group("detect_runs").create_group("detect_001")

    labels, resolved_run, quality_group = _resolve_detection_quality_labels(
        root,
        detect_group,
        detect_run="detect_001",
        source_detect_path="detect_runs/detect_001",
        quality_run=None,
        quality_group_path=None,
        total_detections=4,
        require_quality=False,
        allow_missing_reason="explicit opt-out",
        console=None,
    )

    assert labels.tolist() == [0, 0, 0, 0]
    assert resolved_run is None
    assert quality_group is None


def test_resolve_detection_quality_labels_uses_latest_quality_run() -> None:
    root = _fake_root()
    detect_group = root.create_group("detect_runs").create_group("detect_001")
    quality_reports = detect_group.create_group("quality_reports")
    quality_reports.attrs["latest"] = "detect_quality_001"
    quality_group = quality_reports.create_group("detect_quality_001")
    quality_group.create_array("detection_quality_labels", data=np.array([0, 2, 0], dtype=np.int8))

    labels, resolved_run, resolved_group = _resolve_detection_quality_labels(
        root,
        detect_group,
        detect_run="detect_001",
        source_detect_path="detect_runs/detect_001",
        quality_run=None,
        quality_group_path=None,
        total_detections=3,
        require_quality=True,
        allow_missing_reason="test",
        console=None,
    )

    assert labels.tolist() == [0, 2, 0]
    assert resolved_run == "detect_quality_001"
    assert resolved_group is quality_group


def test_resolve_detection_quality_labels_rejects_length_mismatch() -> None:
    root = _fake_root()
    detect_group = root.create_group("detect_runs").create_group("detect_001")
    quality_reports = detect_group.create_group("quality_reports")
    quality_reports.attrs["latest"] = "detect_quality_001"
    quality_group = quality_reports.create_group("detect_quality_001")
    quality_group.create_array("detection_quality_labels", data=np.array([0, 2], dtype=np.int8))

    with pytest.raises(ValueError, match="does not match detections"):
        _resolve_detection_quality_labels(
            root,
            detect_group,
            detect_run="detect_001",
            source_detect_path="detect_runs/detect_001",
            quality_run=None,
            quality_group_path=None,
            total_detections=3,
            require_quality=True,
            allow_missing_reason="test",
            console=None,
        )


def test_resolve_detection_quality_labels_uses_exact_collection_key_slice() -> None:
    root = _fake_root()
    detect_parent = root.create_group("detect_runs")
    detect = detect_parent.create_group("clip_detect")
    detect.create_array("instance_key", data=np.asarray([21, 22], dtype=np.uint64))
    source_parent = root.create_group("detect_collection_sources")
    source = source_parent.create_group("source_001")
    source.attrs["source_slices"] = [
        {
            "detect_group_path": "detect_runs/other",
            "start": 0,
            "stop": 2,
        },
        {
            "detect_group_path": "detect_runs/clip_detect",
            "start": 2,
            "stop": 4,
        },
    ]
    quality_parent = root.create_group("detect_quality_runs")
    quality = quality_parent.create_group("quality_001")
    quality.attrs["source_detection_group_path"] = (
        "detect_collection_sources/source_001"
    )
    quality.create_array(
        "instance_key", data=np.asarray([10, 11, 21, 22], dtype=np.uint64)
    )
    quality.create_array(
        "detection_quality_labels", data=np.asarray([0, 3, 2, 0], dtype=np.int8)
    )

    labels, resolved_run, resolved_group = _resolve_detection_quality_labels(
        root,
        detect,
        detect_run="clip_detect",
        source_detect_path="detect_runs/clip_detect",
        quality_run=None,
        quality_group_path="detect_quality_runs/quality_001",
        total_detections=2,
        require_quality=True,
        allow_missing_reason="test",
        console=None,
    )

    assert labels.tolist() == [2, 0]
    assert resolved_run == "quality_001"
    assert resolved_group is quality


def test_per_frame_top_k_keeps_highest_score_and_marks_duplicates() -> None:
    raw_frame_indices = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
    raw_scores = np.asarray([0.30, 0.90, 0.50, 0.20, 0.80], dtype=np.float32)
    raw_bboxes = np.asarray(
        [
            [0.10, 0.10, 0.10, 0.10],
            [0.20, 0.20, 0.10, 0.10],
            [0.30, 0.30, 0.10, 0.10],
            [0.40, 0.40, 0.10, 0.10],
            [0.50, 0.50, 0.10, 0.10],
        ],
        dtype=np.float64,
    )
    raw_class_ids = np.zeros(raw_frame_indices.shape[0], dtype=np.int32)
    raw_instance_keys = np.asarray([101, 102, 103, 104, 105], dtype=np.uint64)
    quality_labels = np.zeros(raw_frame_indices.shape[0], dtype=np.int8)

    selected, duplicate, stats = _select_per_frame_top_k_raw_indices(
        raw_frame_indices=raw_frame_indices,
        raw_scores=raw_scores,
        candidate_raw_indices=np.flatnonzero(quality_labels == 0).astype(np.int32),
        per_frame_top_k=1,
        score_field="scores",
    )

    assert selected.tolist() == [1, 4]
    assert duplicate.tolist() == [0, 2, 3]
    assert stats["duplicate_rows"] == 3
    assert stats["frames_with_duplicates"] == 2

    payload = _build_sparse_refined_inputs_from_filtered(
        raw_bboxes=raw_bboxes,
        raw_scores=raw_scores,
        raw_frame_indices=raw_frame_indices,
        raw_class_ids=raw_class_ids,
        raw_instance_keys=raw_instance_keys,
        detection_quality_labels=quality_labels,
        interp_bboxes=raw_bboxes[selected],
        interp_scores=raw_scores[selected],
        interp_frame_indices=raw_frame_indices[selected],
        interp_class_ids=raw_class_ids[selected],
        selected_source_detect_row_index=selected,
        duplicate_source_detect_row_index=duplicate,
    )

    assert payload["instance_frame_indices"].tolist() == [0, 1]
    assert payload["instance_source_detect_row_index"].tolist() == [1, 4]
    assert payload["instance_key"].tolist() == [102, 105]
    assert payload["source_detection_instance_key"].tolist() == [101, 102, 103, 104, 105]
    assert payload["source_detection_decision_labels"].tolist() == [
        "duplicate",
        "accepted",
        "duplicate",
        "duplicate",
        "accepted",
    ]
    assert payload["source_detection_reason_labels"].tolist() == [
        "per_frame_top_k_excluded",
        "clean",
        "per_frame_top_k_excluded",
        "per_frame_top_k_excluded",
        "clean",
    ]


def test_dish_mask_gate_marks_clean_outside_candidates_before_top_k() -> None:
    raw_frame_indices = np.asarray([0, 0, 1], dtype=np.int32)
    raw_scores = np.asarray([0.99, 0.80, 0.75], dtype=np.float32)
    raw_bboxes = np.asarray(
        [
            [0.04, 0.80, 0.05, 0.05],  # high-confidence false positive outside the dish
            [0.50, 0.50, 0.05, 0.05],
            [0.55, 0.50, 0.05, 0.05],
        ],
        dtype=np.float64,
    )
    raw_class_ids = np.zeros(raw_frame_indices.shape[0], dtype=np.int32)
    quality_labels = np.zeros(raw_frame_indices.shape[0], dtype=np.int8)

    gated_labels, gate_stats = _apply_dish_mask_quality_gate(
        bbox_coords=raw_bboxes,
        detection_quality_labels=quality_labels,
        mask_spec={
            "enabled": True,
            "shape": "circle",
            "center_norm": [0.5, 0.5],
            "radius_norm_x": 0.3,
            "radius_norm_y": 0.3,
            "source": "analysis_metadata.dish_mask",
        },
    )

    assert gate_stats["outside_clean_rows"] == 1
    assert gate_stats["base_geometry"] == {
        "center_norm": [0.5, 0.5],
        "radius_norm_x": 0.3,
        "radius_norm_y": 0.3,
    }
    assert gate_stats["effective_geometry"] == gate_stats["base_geometry"]
    assert gated_labels.tolist() == [5, 0, 0]
    assert _filtered_reason_from_quality_label(5) == "outside_dish_mask"

    selected, duplicate, stats = _select_per_frame_top_k_raw_indices(
        raw_frame_indices=raw_frame_indices,
        raw_scores=raw_scores,
        candidate_raw_indices=np.flatnonzero(gated_labels == 0).astype(np.int32),
        per_frame_top_k=1,
        score_field="scores",
    )

    assert selected.tolist() == [1, 2]
    assert duplicate.tolist() == []
    assert stats["candidate_rows"] == 2

    payload = _build_sparse_refined_inputs_from_filtered(
        raw_bboxes=raw_bboxes,
        raw_scores=raw_scores,
        raw_frame_indices=raw_frame_indices,
        raw_class_ids=raw_class_ids,
        detection_quality_labels=gated_labels,
        interp_bboxes=raw_bboxes[selected],
        interp_scores=raw_scores[selected],
        interp_frame_indices=raw_frame_indices[selected],
        interp_class_ids=raw_class_ids[selected],
        selected_source_detect_row_index=selected,
        duplicate_source_detect_row_index=duplicate,
    )

    assert payload["source_detection_decision_labels"].tolist() == [
        "filtered",
        "accepted",
        "accepted",
    ]
    assert payload["source_detection_reason_labels"].tolist() == [
        "outside_dish_mask",
        "clean",
        "clean",
    ]
