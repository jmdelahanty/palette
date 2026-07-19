from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import zarr

from fisheye.refinement.finalize_subject_masks import _component_surface_rows
from fisheye.shared.composite_subject_mask import (
    CompositeSubjectMaskArray,
    CompositeSubjectMaskError,
    assert_subject_mask_run_unreferenced,
    validate_composite_subject_mask_run,
)
from fisheye.shared.row_source_signature import (
    ROW_SOURCE_SIGNATURE_ARRAY,
    build_row_source_signatures,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
)
from fisheye.shared.zarr.stage_arrays import SUBJECT_MASKS_SPEC, validate_run
from fisheye.tune.refined_subject_mask_review import _load_source_subject_mask_run
from fisheye.utils.compact_keypoint_deltas import (
    KeypointCompactionError,
    compact_keypoint_deltas,
)
from fisheye.utils.compact_subject_mask_deltas import compact_subject_mask_deltas


def _root() -> Any:
    return zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)


def _complete(group: Any) -> None:
    group.attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_COMPLETE


def _model_provenance(role: str, sha256: str) -> dict[str, Any]:
    return {
        "schema": "palette.run_provenance.v1",
        "git_sha": "0" * 40,
        "config_hash": "1" * 64,
        "params": {},
        "input_run_ids": {},
        "input_artifacts": [{"role": role, "sha256": sha256, "path": "/fixture/model"}],
        "command": "fixture",
        "fisheye_version": None,
    }


def _crop(
    root: Any,
    name: str,
    *,
    keys: list[int],
    frames: list[int],
    content: list[int],
) -> Any:
    parent = root.require_group("crop_runs")
    group = parent.create_group(name)
    keys_array = np.asarray(keys, dtype=np.uint64)
    frames_array = np.asarray(frames, dtype=np.int32)
    signature_batch = build_row_source_signatures(
        stage="crop",
        instance_keys=keys_array,
        content_components={"crop_content": np.asarray(content, dtype=np.int32)},
        compatibility_context={"roi_size": [2, 2], "pixel_source": "fixture"},
    )
    group.create_array("instance_key", data=keys_array, chunks=(1,))
    group.create_array("frame_indices", data=frames_array, chunks=(1,))
    group.create_array("frame_counts", data=np.bincount(frames_array, minlength=4).astype(np.int32))
    group.create_array("detection_indices", data=np.arange(len(keys), dtype=np.int32))
    group.create_array("detection_source", data=np.zeros(len(keys), dtype=np.int8))
    group.create_array("source_frame_indices", data=frames_array)
    group.create_array(ROW_SOURCE_SIGNATURE_ARRAY, data=signature_batch.signatures, chunks=(1, 32))
    group.attrs.update(signature_batch.spec.to_attrs())
    group.attrs.update(
        {
            "crop_storage_mode": "materialized",
            "roi_size": [2, 2],
            "roi_pixel_contract": {"name": "fixture", "shape": [2, 2]},
        }
    )
    _complete(group)
    return group


_KP_COMPAT = {
    "keypoint_labels": ["head", "tail"],
    "keypoint_confidence_labels": ["head", "tail"],
    "skeleton_id": "fixture",
    "kpt_shape": [2, 3],
    "pose_schema": {"name": "fixture", "nodes": ["head", "tail"]},
    "model_kpt_shape": [2, 3],
    "model_path": "/models/pose.pt",
    "model_name": "pose.pt",
    "model_input_transform": {"name": "stretch"},
    "model_input_transform_name": "stretch",
    "model_input_shape_hw": [2, 2],
    "native_roi_shape_hw": [2, 2],
}


def _keypoint_run(
    parent: Any,
    name: str,
    *,
    crop_name: str,
    keys: list[int],
    crop_rows: list[int],
    values: list[float],
    delta: bool = False,
) -> Any:
    group = parent.create_group(name)
    rows = len(keys)
    scalar = np.asarray(values, dtype=np.float64)
    coords = np.repeat(scalar[:, None, None], 4, axis=1).reshape(rows, 2, 2)
    group.create_array("instance_key", data=np.asarray(keys, dtype=np.uint64), chunks=(1,))
    group.create_array("source_crop_row_ids", data=np.asarray(crop_rows, dtype=np.int64), chunks=(1,))
    group.create_array("frame_indices", data=np.arange(rows, dtype=np.int32))
    group.create_array("detection_indices", data=np.arange(rows, dtype=np.int32))
    group.create_array("detection_source", data=np.zeros(rows, dtype=np.int8))
    for array_name in ("keypoints_roi", "keypoints_img", "keypoints_norm"):
        group.create_array(array_name, data=coords, chunks=(1, 2, 2))
    group.create_array("keypoint_confidences", data=np.repeat(scalar[:, None], 2, axis=1), chunks=(1, 2))
    for array_name in ("confidence", "heading", "effective_threshold", "effective_se2_radius"):
        group.create_array(array_name, data=scalar, chunks=(1,))
    group.create_array("detection_success", data=np.ones(rows, dtype=bool), chunks=(1,))
    group.create_array("heading_finite", data=np.ones(rows, dtype=bool), chunks=(1,))
    group.create_array("heading_usable", data=np.ones(rows, dtype=bool), chunks=(1,))
    group.create_array("pose_bbox_xyxy_roi", data=np.repeat(scalar[:, None], 4, axis=1), chunks=(1, 4))
    group.create_array("frame_counts", data=np.ones(4, dtype=np.int32))
    group.create_array("n_rois", data=np.ones(4, dtype=np.int32))
    group.create_array("n_keypoints", data=np.full(4, 2, dtype=np.int32))
    group.attrs.update(
        {
            **_KP_COMPAT,
            "source_crop_run": crop_name,
            "parameters": {
                "confidence_threshold": 0.5,
                "iou_threshold": 0.7,
                "max_det": 1,
                "imgsz": 2,
                "pose_schema": "fixture",
                "n_keypoints": 2,
                "model_kpt_shape": [2, 3],
                "input_mode_requested": "tensor",
                "input_mode_effective": "tensor",
                "model_input_transform": {"name": "stretch"},
            },
            "run_provenance": _model_provenance("keypoint_model", "a" * 64),
        }
    )
    if delta:
        group.attrs.update(
            {
                "incremental_materialization_role": "delta_replacement_rows",
                "canonical_finalization_policy": "incremental_compaction_required",
            }
        )
    _complete(group)
    return group


_MASK_ATTRS = {
    "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
    "label_schema_id": "fixture_masks_v1",
    "probabilities_encoding": "linear_uint8_0_255",
    "mask_probability_threshold": 0.5,
    "probabilities_dtype": "uint8",
    "probability_semantics": "sigmoid_multilabel_logits",
    "output_semantics": "multilabel",
    "overlap_policy": "independent_sigmoid",
    "model_input_transform": {"name": "stretch"},
    "model_input_transform_name": "stretch",
    "model_input_shape_hw": [2, 2],
    "native_roi_shape_hw": [2, 2],
    "source_checkpoint": "/models/masks.pt",
}


def _mask_run(
    parent: Any,
    name: str,
    *,
    crop_name: str,
    keys: list[int],
    crop_rows: list[int],
    values: list[int],
    delta: bool = False,
) -> Any:
    group = parent.create_group(name)
    rows = len(keys)
    probabilities = np.empty((rows, 3, 2, 2), dtype=np.uint8)
    for row, value in enumerate(values):
        probabilities[row] = np.uint8(value)
    group.create_array("mask_probs_roi", data=probabilities, chunks=(1, 1, 2, 2), shards=(2, 1, 2, 2))
    group.create_array("available_channels", data=np.ones(3, dtype=bool))
    group.create_array("instance_key", data=np.asarray(keys, dtype=np.uint64))
    group.create_array("source_crop_row_ids", data=np.asarray(crop_rows, dtype=np.int64))
    group.create_array("frame_indices", data=np.arange(rows, dtype=np.int32))
    group.create_array("frame_counts", data=np.ones(4, dtype=np.int32))
    group.create_array("detection_indices", data=np.arange(rows, dtype=np.int32))
    group.create_array("detection_source", data=np.zeros(rows, dtype=np.int8))
    metrics = group.create_group("metrics")
    metric_values = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 3, axis=1)
    metrics.create_array("prob_max", data=metric_values, chunks=(1, 3))
    metrics.create_array("mask_present", data=metric_values > 0, chunks=(1, 3))
    metrics.create_array("area_px", data=metric_values, chunks=(1, 3))
    metrics.create_array("centroid_xy", data=np.repeat(metric_values[:, :, None], 2, axis=2), chunks=(1, 3, 2))
    metrics.create_array("centroid_valid", data=metric_values > 0, chunks=(1, 3))
    metrics.create_array("bbox_xyxy", data=np.repeat(metric_values[:, :, None], 4, axis=2), chunks=(1, 3, 4))
    metrics.create_array("bbox_valid", data=metric_values > 0, chunks=(1, 3))
    group.attrs.update(
        {
            **_MASK_ATTRS,
            "source_crop_run": crop_name,
            "run_provenance": _model_provenance(
                "subject_mask_unet_checkpoint", "b" * 64
            ),
        }
    )
    if delta:
        group.attrs.update(
            {
                "incremental_materialization_role": "delta_replacement_rows",
                "canonical_finalization_policy": "incremental_compaction_required",
            }
        )
    _complete(group)
    return group


def _fixture() -> Any:
    root = _root()
    _crop(root, "crop_base", keys=[10, 20, 30], frames=[0, 1, 2], content=[1, 2, 3])
    _crop(root, "crop_target", keys=[30, 10, 40], frames=[2, 0, 3], content=[9, 1, 4])
    keypoint_parent = root.require_group("keypoints_runs")
    keypoint_shards = root.require_group("keypoint_shard_runs")
    _keypoint_run(
        keypoint_parent,
        "kp_base",
        crop_name="crop_base",
        keys=[10, 20, 30],
        crop_rows=[0, 1, 2],
        values=[10.0, 20.0, 30.0],
    )
    kp_delta = _keypoint_run(
        keypoint_shards,
        "kp_delta",
        crop_name="crop_target",
        keys=[40, 30],
        crop_rows=[2, 0],
        values=[400.0, 300.0],
        delta=True,
    )
    mask_parent = root.require_group("subject_mask_runs")
    mask_shards = root.require_group("subject_mask_shard_runs")
    _mask_run(
        mask_parent,
        "mask_base",
        crop_name="crop_base",
        keys=[10, 20, 30],
        crop_rows=[0, 1, 2],
        values=[10, 20, 30],
    )
    mask_delta = _mask_run(
        mask_shards,
        "mask_delta",
        crop_name="crop_target",
        keys=[40, 30],
        crop_rows=[2, 0],
        values=[240, 230],
        delta=True,
    )
    target = root["crop_runs/crop_target"]
    delta_rows = np.asarray([2, 0], dtype=np.int64)
    delta_signatures = np.asarray(
        target[ROW_SOURCE_SIGNATURE_ARRAY][:], dtype=np.uint8
    )[delta_rows]
    signature_attrs = {
        key: value
        for key, value in target.attrs.items()
        if str(key).startswith("source_row_signature_")
    }
    for group in (kp_delta, mask_delta):
        group.create_array(
            ROW_SOURCE_SIGNATURE_ARRAY,
            data=delta_signatures,
            chunks=(1, 32),
        )
        group.attrs.update(signature_attrs)
    return root


def test_keypoint_compactor_writes_exact_target_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fixture()
    monkeypatch.setattr("fisheye.utils.compact_keypoint_deltas.open_zarr_root", lambda *_a, **_k: root)

    result = compact_keypoint_deltas(
        zarr_path="fixture.zarr",
        base_run="kp_base",
        target_crop_run="crop_target",
        replacement_runs=["kp_delta"],
        output_run="kp_snapshot",
        row_shard_rows=2,
        frame_shard_rows=2,
    )

    output = root["keypoints_runs/kp_snapshot"]
    assert result["plan"]["base_row_count"] == 1
    assert result["plan"]["replacement_row_count"] == 2
    np.testing.assert_array_equal(output["instance_key"][:], [30, 10, 40])
    np.testing.assert_array_equal(output["source_crop_row_ids"][:], [0, 1, 2])
    np.testing.assert_allclose(output["confidence"][:], [300.0, 10.0, 400.0])
    assert output.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert root["keypoints_runs"].attrs["latest"] == "kp_snapshot"
    assert output["confidence"].shards is not None


def test_keypoint_compactor_failure_does_not_promote(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fixture()
    root["keypoints_runs"].attrs["latest"] = "kp_base"
    root["keypoints_runs"].attrs["latest_complete"] = "kp_base"
    monkeypatch.setattr("fisheye.utils.compact_keypoint_deltas.open_zarr_root", lambda *_a, **_k: root)

    def mutate() -> None:
        root["crop_runs/crop_target/instance_key"][0] = np.uint64(999)

    with pytest.raises(KeypointCompactionError, match="input changed|do not match"):
        compact_keypoint_deltas(
            zarr_path="fixture.zarr",
            base_run="kp_base",
            target_crop_run="crop_target",
            replacement_runs=["kp_delta"],
            output_run="kp_failed",
            before_publish=mutate,
        )
    assert root["keypoints_runs"].attrs["latest"] == "kp_base"
    assert root["keypoints_runs"].attrs["latest_complete"] == "kp_base"
    assert root["keypoints_runs/kp_failed"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_keypoint_compactor_rejects_model_fingerprint_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture()
    root["keypoint_shard_runs/kp_delta"].attrs["run_provenance"] = _model_provenance(
        "keypoint_model", "c" * 64
    )
    monkeypatch.setattr("fisheye.utils.compact_keypoint_deltas.open_zarr_root", lambda *_a, **_k: root)

    with pytest.raises(KeypointCompactionError, match="model fingerprint differs"):
        compact_keypoint_deltas(
            zarr_path="fixture.zarr",
            base_run="kp_base",
            target_crop_run="crop_target",
            replacement_runs=["kp_delta"],
            output_run="kp_bad_model",
            dry_run=True,
        )
    assert "kp_bad_model" not in root["keypoints_runs"]


def test_keypoint_compactor_rejects_stale_inference_crop_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture()
    root["keypoint_shard_runs/kp_delta/source_row_signature"][0, 0] ^= np.uint8(1)
    monkeypatch.setattr("fisheye.utils.compact_keypoint_deltas.open_zarr_root", lambda *_a, **_k: root)

    with pytest.raises(KeypointCompactionError, match="stale or incompatible crop rows"):
        compact_keypoint_deltas(
            zarr_path="fixture.zarr",
            base_run="kp_base",
            target_crop_run="crop_target",
            replacement_runs=["kp_delta"],
            output_run="kp_stale_pixels",
            dry_run=True,
        )


def test_subject_mask_compactor_reuses_base_and_finalizer_reads_composite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture()
    monkeypatch.setattr("fisheye.utils.compact_subject_mask_deltas.open_zarr_root", lambda *_a, **_k: root)

    result = compact_subject_mask_deltas(
        zarr_path="fixture.zarr",
        base_run="mask_base",
        target_crop_run="crop_target",
        replacement_runs=["mask_delta"],
        output_run="mask_snapshot",
        tabular_shard_rows=2,
    )

    output = root["subject_mask_runs/mask_snapshot"]
    assert "mask_probs_roi" not in output
    assert output["composite_payload/mask_probs_roi_delta"].shape == (2, 3, 2, 2)
    assert result["storage"]["unchanged_probability_bytes_rewritten"] == 0
    validation = validate_composite_subject_mask_run(
        root, output, run_name="mask_snapshot", verify_identity=True
    )
    assert validation.base_rows_used == 1
    probabilities = CompositeSubjectMaskArray.open(
        root, output, run_name="mask_snapshot", verify_identity=True
    )
    np.testing.assert_array_equal(probabilities[:, 0, 0, 0], [230, 10, 240])
    source = _load_source_subject_mask_run(root, "mask_snapshot")
    surfaces, is_probability, path, encoding, *_ = _component_surface_rows(
        source,
        "subject_body",
        start_row=0,
        stop_row=3,
    )
    assert is_probability is True
    assert path == "mask_probs_roi"
    assert encoding == "linear_uint8_0_255"
    np.testing.assert_allclose(surfaces[:, 0, 0], np.asarray([230, 10, 240]) / 255.0)
    np.testing.assert_array_equal(output["metrics/prob_max"][:, 0], [230, 10, 240])
    assert output.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert root["subject_mask_runs"].attrs["latest"] == "mask_snapshot"
    stage_validation = validate_run(output, SUBJECT_MASKS_SPEC)
    assert stage_validation.valid, stage_validation.errors
    with pytest.raises(CompositeSubjectMaskError, match="composite dependents"):
        assert_subject_mask_run_unreferenced(root["subject_mask_runs"], "mask_base")


def test_subject_mask_compactor_failure_does_not_promote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture()
    root["subject_mask_runs"].attrs["latest"] = "mask_base"
    root["subject_mask_runs"].attrs["latest_complete"] = "mask_base"
    monkeypatch.setattr(
        "fisheye.utils.compact_subject_mask_deltas.open_zarr_root",
        lambda *_a, **_k: root,
    )

    def mutate() -> None:
        root["crop_runs/crop_target/instance_key"][0] = np.uint64(999)

    with pytest.raises(RuntimeError, match="input changed|do not match"):
        compact_subject_mask_deltas(
            zarr_path="fixture.zarr",
            base_run="mask_base",
            target_crop_run="crop_target",
            replacement_runs=["mask_delta"],
            output_run="mask_failed",
            before_publish=mutate,
        )
    assert root["subject_mask_runs"].attrs["latest"] == "mask_base"
    assert root["subject_mask_runs"].attrs["latest_complete"] == "mask_base"
    assert root["subject_mask_runs/mask_failed"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_subject_mask_compactor_rejects_model_fingerprint_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture()
    root["subject_mask_shard_runs/mask_delta"].attrs["run_provenance"] = _model_provenance(
        "subject_mask_unet_checkpoint", "d" * 64
    )
    monkeypatch.setattr(
        "fisheye.utils.compact_subject_mask_deltas.open_zarr_root",
        lambda *_a, **_k: root,
    )

    with pytest.raises(RuntimeError, match="model fingerprint differs"):
        compact_subject_mask_deltas(
            zarr_path="fixture.zarr",
            base_run="mask_base",
            target_crop_run="crop_target",
            replacement_runs=["mask_delta"],
            output_run="mask_bad_model",
            dry_run=True,
        )
    assert "mask_bad_model" not in root["subject_mask_runs"]
