"""Bind sampled full-frame detection review to crop-based training surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.training_crop_materialization import (
    BoundTrainingCropMaterialization,
    bind_training_crop_materialization,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import is_run_complete
from fisheye.shared.zarr.detection_frame_supervision import (
    build_detection_frame_supervision_plan,
)

TRAINING_DATASET_COMPOSITION_ATTRIBUTE = "training_dataset_composition"
TRAINING_DATASET_COMPOSITION_SCHEMA_ID = "palette.training_dataset_composition"
TRAINING_DATASET_COMPOSITION_SCHEMA_VERSION = 1
_SELECTOR_ORDER = ("authoritative_run", "latest_complete", "latest")


class TrainingDatasetCompositionError(ValueError):
    """Raised when a training Zarr is not a coherent review dataset."""


@dataclass(frozen=True)
class TrainingDetectionReviewBase:
    archive_path: Path
    frame_count: int
    original_frame_indices: np.ndarray
    detect_run_id: str
    refined_run_id: str
    refined_review_state: str
    refined_review_intended_use: str
    refined_review_authority_scope: str
    refined_review_status_digest: str
    detect_row_count: int
    refined_row_count: int
    refined_bbox_source_dtype: str
    refined_bbox_source_digest: str
    refined_instance_keys: np.ndarray
    refined_frame_indices: np.ndarray
    refined_bbox_norm_coords: np.ndarray


@dataclass(frozen=True)
class BoundTrainingDatasetComposition:
    archive_path: Path
    crop: BoundTrainingCropMaterialization
    base: TrainingDetectionReviewBase
    binding: Mapping[str, Any]


def _validate_observation_join(
    base: TrainingDetectionReviewBase,
    *,
    keys: np.ndarray,
    source_frames: np.ndarray,
    boxes: np.ndarray,
    label: str,
) -> None:
    local_by_source = {
        int(source_frame): int(local_frame)
        for local_frame, source_frame in enumerate(base.original_frame_indices.tolist())
    }
    refined_by_key = {
        int(key): (int(frame), box)
        for key, frame, box in zip(
            base.refined_instance_keys.tolist(),
            base.refined_frame_indices.tolist(),
            base.refined_bbox_norm_coords,
            strict=True,
        )
    }
    for row, (key, source_frame, box) in enumerate(
        zip(keys.tolist(), source_frames.tolist(), boxes, strict=True)
    ):
        if int(source_frame) not in local_by_source:
            raise TrainingDatasetCompositionError(
                f"{label} row {row} source frame {source_frame} is absent from sampled full frames."
            )
        refined = refined_by_key.get(int(key))
        if refined is None:
            raise TrainingDatasetCompositionError(
                f"{label} row {row} instance_key={key} is absent from refined detection review."
            )
        local_frame, refined_box = refined
        if local_frame != local_by_source[int(source_frame)]:
            raise TrainingDatasetCompositionError(
                f"{label} row {row} frame lineage differs from refined detection review."
            )
        if not np.array_equal(np.asarray(box, dtype=np.float32), refined_box):
            raise TrainingDatasetCompositionError(
                f"{label} row {row} bbox_norm_coords differs from refined detection review."
            )


def _sha256_array(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(b"palette.training_dataset_identity.v1\x00")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\x00")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _selected_run(parent: Any, requested: str | None, *, family: str) -> str:
    if requested is not None:
        candidate = str(requested).strip()
    else:
        candidate = ""
        for attr in _SELECTOR_ORDER:
            value = parent.attrs.get(attr)
            if value is not None and str(value).strip():
                candidate = str(value).strip()
                break
    if not candidate or candidate not in parent:
        raise TrainingDatasetCompositionError(
            f"Training dataset has no selected {family} review run."
        )
    return candidate


def _instances(run: Any, *, family: str) -> Any:
    table = run.get("instances")
    if table is None:
        table = run
    missing = [
        name
        for name in ("frame_indices", "bbox_norm_coords", "instance_key")
        if name not in table
    ]
    if missing:
        raise TrainingDatasetCompositionError(
            f"{family} review run lacks stable multi-instance arrays: {missing}."
        )
    return table


def _read_detection_table(
    run: Any,
    *,
    family: str,
    frame_count: int,
    allow_float64_boxes: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    table = _instances(run, family=family)
    frame_indices = np.asarray(table["frame_indices"][:])
    boxes = np.asarray(table["bbox_norm_coords"][:])
    keys = np.asarray(table["instance_key"][:])
    count = int(frame_indices.shape[0])
    if frame_indices.ndim != 1 or not np.issubdtype(frame_indices.dtype, np.integer):
        raise TrainingDatasetCompositionError(
            f"{family} frame_indices must be one integer vector."
        )
    allowed_box_dtypes = {np.dtype(np.float32)}
    if allow_float64_boxes:
        allowed_box_dtypes.add(np.dtype(np.float64))
    if boxes.shape != (count, 4) or boxes.dtype not in allowed_box_dtypes:
        raise TrainingDatasetCompositionError(
            f"{family} bbox_norm_coords must have shape ({count}, 4) and dtype "
            f"in {sorted(str(dtype) for dtype in allowed_box_dtypes)}."
        )
    boxes_source_dtype = str(boxes.dtype)
    boxes_source_digest = _sha256_array(boxes)
    boxes = np.asarray(boxes, dtype=np.float32)
    if count:
        half = np.float32(0.5)
        if (
            not np.isfinite(boxes).all()
            or np.any(boxes[:, 2:] <= 0)
            or np.any(boxes[:, :2] - boxes[:, 2:] * half < 0)
            or np.any(boxes[:, :2] + boxes[:, 2:] * half > 1)
        ):
            raise TrainingDatasetCompositionError(
                f"{family} bbox_norm_coords must remain finite, positive, and contained after float32 canonicalization."
            )
    if keys.shape != (count,) or keys.dtype != np.dtype(np.uint64):
        raise TrainingDatasetCompositionError(
            f"{family} instance_key must have exact uint64 shape ({count},)."
        )
    if np.unique(keys).shape[0] != count:
        raise TrainingDatasetCompositionError(
            f"{family} instance_key values must be unique."
        )
    if count and (
        int(frame_indices.min()) < 0 or int(frame_indices.max()) >= int(frame_count)
    ):
        raise TrainingDatasetCompositionError(
            f"{family} frame_indices are outside the sampled training frame axis."
        )
    order = np.lexsort((keys, frame_indices)) if count else np.empty(0, dtype=np.int64)
    if count and not np.array_equal(order, np.arange(count, dtype=np.int64)):
        raise TrainingDatasetCompositionError(
            f"{family} rows must be sorted by frame_indices then instance_key."
        )
    return (
        np.asarray(frame_indices, dtype=np.int64),
        boxes,
        np.asarray(keys, dtype=np.uint64),
        boxes_source_dtype,
        boxes_source_digest,
    )


def validate_training_detection_review_base(
    archive: str | Path,
    *,
    detect_run_id: str | None = None,
    refined_run_id: str | None = None,
) -> TrainingDetectionReviewBase:
    """Require sampled pixels plus canonical and refined detection-review rows."""

    path = Path(archive).expanduser().resolve()
    root = open_zarr_group_direct(path, mode="r")
    if str(root.attrs.get("zarr_purpose") or "").strip().lower() != "training":
        raise TrainingDatasetCompositionError(
            "Detection review composition requires a training-purpose Zarr."
        )
    raw = root.get("raw_video")
    if raw is None:
        raise TrainingDatasetCompositionError("Training Zarr lacks raw_video.")
    missing_raw = [
        name
        for name in ("images_full", "images_ds", "original_frame_indices")
        if name not in raw
    ]
    if missing_raw:
        raise TrainingDatasetCompositionError(
            f"Training Zarr lacks first-class full-frame review arrays: {missing_raw}."
        )
    images_full = raw["images_full"]
    images_ds = raw["images_ds"]
    original = np.asarray(raw["original_frame_indices"][:])
    if len(images_full.shape) not in {3, 4} or np.dtype(images_full.dtype) != np.uint8:
        raise TrainingDatasetCompositionError(
            "raw_video/images_full must be rank-3/4 uint8."
        )
    if len(images_ds.shape) not in {3, 4} or np.dtype(images_ds.dtype) != np.uint8:
        raise TrainingDatasetCompositionError(
            "raw_video/images_ds must be rank-3/4 uint8."
        )
    frame_count = int(images_full.shape[0])
    if int(images_ds.shape[0]) != frame_count:
        raise TrainingDatasetCompositionError(
            "images_full and images_ds must share one sampled frame axis."
        )
    if original.shape != (frame_count,) or not np.issubdtype(
        original.dtype, np.integer
    ):
        raise TrainingDatasetCompositionError(
            "raw_video/original_frame_indices must be one integer per sampled frame."
        )
    original = np.asarray(original, dtype=np.int64)
    if frame_count and (
        np.unique(original).shape[0] != frame_count
        or np.any(original[1:] <= original[:-1])
    ):
        raise TrainingDatasetCompositionError(
            "original_frame_indices must be unique and strictly increasing."
        )

    detect_parent = root.get("detect_runs")
    refined_parent = root.get("refined_detect_runs")
    if detect_parent is None or refined_parent is None:
        raise TrainingDatasetCompositionError(
            "Training Zarr must contain detect_runs and refined_detect_runs before crop enrichment."
        )
    detect_id = _selected_run(detect_parent, detect_run_id, family="detect")
    refined_id = _selected_run(
        refined_parent,
        refined_run_id,
        family="refined-detect",
    )
    for family, run in (
        ("detect", detect_parent[detect_id]),
        ("refined-detect", refined_parent[refined_id]),
    ):
        legacy_status_complete = str(run.attrs.get("status") or "").strip().lower() in {
            "complete",
            "completed",
        }
        if not legacy_status_complete and not is_run_complete(
            run, legacy_default=False
        ):
            raise TrainingDatasetCompositionError(
                f"Selected {family} review run is not complete."
            )
    review_status = refined_parent[refined_id].attrs.get("detect_review_status")
    if not isinstance(review_status, Mapping):
        raise TrainingDatasetCompositionError(
            "Refined detection review must be approved before crop enrichment."
        )
    review_state = str(review_status.get("state") or "").strip().lower()
    intended_use = str(review_status.get("intended_use") or "").strip().lower()
    if review_state != "approved" or intended_use not in {
        "training",
        "analysis_and_training",
    }:
        raise TrainingDatasetCompositionError(
            "Refined detection review must be approved for training before crop enrichment."
        )
    authority_scope = str(review_status.get("authority_scope") or "").strip()
    review_status_digest = canonical_json_sha256(dict(review_status))
    detect_frames, _detect_boxes, _detect_keys, _detect_dtype, _detect_digest = (
        _read_detection_table(
            detect_parent[detect_id], family="detect", frame_count=frame_count
        )
    )
    (
        refined_frames,
        refined_boxes,
        refined_keys,
        refined_bbox_source_dtype,
        refined_bbox_source_digest,
    ) = _read_detection_table(
        refined_parent[refined_id],
        family="refined-detect",
        frame_count=frame_count,
        allow_float64_boxes=True,
    )
    if authority_scope == "selector_ineligible_training_candidate":
        receipt = review_status.get("selector_ineligible_candidate_receipt")
        approval = review_status.get("authoritative_approval")
        if not isinstance(receipt, Mapping) or not isinstance(approval, Mapping):
            raise TrainingDatasetCompositionError(
                "Selector-ineligible detection approval lacks its candidate receipt."
            )
        decision_path = f"detect_frame_decision_runs/{refined_id}"
        table = _instances(refined_parent[refined_id], family="refined-detect")
        table_prefix = f"refined_detect_runs/{refined_id}"
        if table is not refined_parent[refined_id]:
            table_prefix += "/instances"
        supervision = build_detection_frame_supervision_plan(
            root,
            bbox_path=f"{table_prefix}/bbox_norm_coords",
            frame_indices_path=f"{table_prefix}/frame_indices",
            n_frames=frame_count,
        )
        would_materialize = receipt.get("frame_decision_would_materialize")
        if type(would_materialize) is not bool:
            raise TrainingDatasetCompositionError(
                "Candidate receipt frame_decision_would_materialize must be boolean."
            )
        expected_receipt = {
            "schema_id": "palette.selector_ineligible_detection_review_receipt",
            "schema_version": 1,
            "status": "complete",
            "authority_scope": authority_scope,
            "source_refined_detect_run": refined_id,
            "frame_decision_path": decision_path,
            "frame_decision_digest": supervision.source_decision_digest,
            "frame_decision_would_materialize": would_materialize,
            "frame_count": frame_count,
            "instance_count": int(refined_frames.shape[0]),
            "positive_frame_count": supervision.positive_frame_count,
            "negative_frame_count": supervision.negative_frame_count,
            "parent_selectors_updated": False,
            "stage_selector_eligible": False,
            "metadata_mode": "direct_mutable",
        }
        if dict(receipt) != expected_receipt:
            raise TrainingDatasetCompositionError(
                "Selector-ineligible detection approval receipt is stale or malformed."
            )
        if dict(approval) != {
            "status": "deferred_selector_ineligible",
            "reason_code": "CANDIDATE_ONLY",
            "run": refined_id,
        }:
            raise TrainingDatasetCompositionError(
                "Selector-ineligible detection approval must defer authoritative selection."
            )
        if (
            root.attrs.get("stage_selector_eligible") is not False
            or refined_parent[refined_id].attrs.get("stage_selector_eligible")
            is not False
            or root.attrs.get("training_artifact_status")
            not in {"review_complete", "complete"}
        ):
            raise TrainingDatasetCompositionError(
                "Selector-ineligible reviewed training artifacts must remain ineligible "
                "and be review-complete or complete."
            )
        for parent in (detect_parent, refined_parent):
            for selector in _SELECTOR_ORDER:
                if parent.attrs.get(selector) in {detect_id, refined_id}:
                    raise TrainingDatasetCompositionError(
                        "Selector-ineligible review receipt conflicts with a parent selector."
                    )
    return TrainingDetectionReviewBase(
        archive_path=path,
        frame_count=frame_count,
        original_frame_indices=original,
        detect_run_id=detect_id,
        refined_run_id=refined_id,
        refined_review_state=review_state,
        refined_review_intended_use=intended_use,
        refined_review_authority_scope=authority_scope,
        refined_review_status_digest=review_status_digest,
        detect_row_count=int(detect_frames.shape[0]),
        refined_row_count=int(refined_frames.shape[0]),
        refined_bbox_source_dtype=refined_bbox_source_dtype,
        refined_bbox_source_digest=refined_bbox_source_digest,
        refined_instance_keys=refined_keys,
        refined_frame_indices=refined_frames,
        refined_bbox_norm_coords=refined_boxes,
    )


def validate_training_crop_source_join(
    base: TrainingDetectionReviewBase,
    *,
    source_zarr: str | Path,
    source_crop_run: str,
    source_instance_keys: list[int] | tuple[int, ...] | None = None,
) -> None:
    """Fail before materialization when the source crop cannot join review rows."""

    source = open_zarr_group_direct(Path(source_zarr).expanduser().resolve(), mode="r")
    parent = source.get("crop_runs")
    if parent is None or str(source_crop_run) not in parent:
        raise TrainingDatasetCompositionError(
            f"Source crop run not found: crop_runs/{source_crop_run}."
        )
    run = parent[str(source_crop_run)]
    required = ("instance_key", "bbox_norm_coords")
    missing = [name for name in required if name not in run]
    if missing:
        raise TrainingDatasetCompositionError(
            f"Source crop lacks detection-review join arrays: {missing}."
        )
    frame_name = (
        "source_acquisition_frame_index"
        if "source_acquisition_frame_index" in run
        else "frame_indices"
    )
    if frame_name not in run:
        raise TrainingDatasetCompositionError(
            "Source crop lacks source acquisition frame identity."
        )
    keys = np.asarray(run["instance_key"][:], dtype=np.uint64)
    source_frames = np.asarray(run[frame_name][:], dtype=np.int64)
    boxes = np.asarray(run["bbox_norm_coords"][:], dtype=np.float32)
    if source_instance_keys is not None:
        requested = {int(value) for value in source_instance_keys}
        source_indices = np.asarray(
            [index for index, key in enumerate(keys.tolist()) if int(key) in requested],
            dtype=np.int64,
        )
        observed = {int(keys[index]) for index in source_indices.tolist()}
        missing_keys = sorted(requested - observed)
        if missing_keys:
            raise TrainingDatasetCompositionError(
                f"Requested source instance keys are absent: {missing_keys[:10]}."
            )
        keys = keys[source_indices]
        source_frames = source_frames[source_indices]
        boxes = boxes[source_indices]
    _validate_observation_join(
        base,
        keys=keys,
        source_frames=source_frames,
        boxes=boxes,
        label="Source crop",
    )


def build_training_dataset_composition(
    archive: str | Path,
    *,
    crop_run_id: str,
    detect_run_id: str | None = None,
    refined_run_id: str | None = None,
    require_consolidated_crop: bool = True,
) -> tuple[
    TrainingDetectionReviewBase, BoundTrainingCropMaterialization, dict[str, Any]
]:
    """Build an exact cross-surface receipt and validate all observation joins."""

    base = validate_training_detection_review_base(
        archive,
        detect_run_id=detect_run_id,
        refined_run_id=refined_run_id,
    )
    crop = bind_training_crop_materialization(
        archive,
        run_id=str(crop_run_id),
        require_consolidated=require_consolidated_crop,
    )
    root = open_zarr_group_direct(base.archive_path, mode="r")
    crop_run = root[f"crop_runs/{crop_run_id}"]
    crop_keys = np.asarray(crop_run["instance_key"][:], dtype=np.uint64)
    crop_source_frames = np.asarray(crop_run["source_frame_indices"][:], dtype=np.int64)
    crop_boxes = np.asarray(crop_run["bbox_norm_coords"][:], dtype=np.float32)
    if crop_keys.shape != (crop.row_count,):
        raise TrainingDatasetCompositionError(
            "Crop instance_key length differs from materialized pixels."
        )
    _validate_observation_join(
        base,
        keys=crop_keys,
        source_frames=crop_source_frames,
        boxes=crop_boxes,
        label="Crop",
    )

    payload = {
        "schema_id": TRAINING_DATASET_COMPOSITION_SCHEMA_ID,
        "schema_version": TRAINING_DATASET_COMPOSITION_SCHEMA_VERSION,
        "status": "complete",
        "review_surfaces": {
            "detection": {
                "full_frame_path": "raw_video/images_full",
                "display_frame_path": "raw_video/images_ds",
                "original_frame_indices_path": "raw_video/original_frame_indices",
                "detect_run": base.detect_run_id,
                "refined_detect_run": base.refined_run_id,
                "review_state": base.refined_review_state,
                "review_intended_use": base.refined_review_intended_use,
                "review_authority_scope": base.refined_review_authority_scope,
                "review_status_digest": base.refined_review_status_digest,
                "detect_row_count": base.detect_row_count,
                "refined_row_count": base.refined_row_count,
                "refined_bbox_source_dtype": base.refined_bbox_source_dtype,
                "refined_bbox_source_digest": base.refined_bbox_source_digest,
                "canonical_crop_bbox_dtype": "float32",
                "row_identity": "instance_key",
            },
            "crop": {
                "run": str(crop_run_id),
                "row_count": crop.row_count,
                "roi_shape": list(crop.roi_shape),
                "binding_digest": crop.binding["payload_digest"],
                "source_join": "instance_key_and_original_frame_index_and_bbox_v1",
            },
            "keypoints": {"status": "candidate_finalization_pending"},
            "subject_masks": {"status": "candidate_finalization_pending"},
        },
        "identity_digests": {
            "original_frame_indices": _sha256_array(base.original_frame_indices),
            "refined_instance_keys": _sha256_array(base.refined_instance_keys),
            "crop_instance_keys": _sha256_array(crop_keys),
            "crop_source_frame_indices": _sha256_array(crop_source_frames),
        },
        "stage_selector_eligible": False,
        "registry_activation": "deferred",
    }
    return (
        base,
        crop,
        {
            "payload": payload,
            "payload_digest": canonical_json_sha256(payload),
        },
    )


def bind_training_dataset_composition(
    archive: str | Path,
    *,
    crop_run_id: str,
    detect_run_id: str | None = None,
    refined_run_id: str | None = None,
) -> BoundTrainingDatasetComposition:
    """Validate the persisted direct and consolidated composition receipt."""

    path = Path(archive).expanduser().resolve()
    base, crop, expected = build_training_dataset_composition(
        path,
        crop_run_id=crop_run_id,
        detect_run_id=detect_run_id,
        refined_run_id=refined_run_id,
        require_consolidated_crop=True,
    )
    direct = open_zarr_group_direct(path, mode="r")
    persisted = direct.attrs.get(TRAINING_DATASET_COMPOSITION_ATTRIBUTE)
    if not isinstance(persisted, Mapping) or dict(persisted) != expected:
        raise TrainingDatasetCompositionError(
            "Direct training dataset composition receipt is absent or stale."
        )
    consolidated = zarr.open_group(str(path), mode="r", use_consolidated=True)
    consolidated_value = consolidated.attrs.get(TRAINING_DATASET_COMPOSITION_ATTRIBUTE)
    if (
        not isinstance(consolidated_value, Mapping)
        or dict(consolidated_value) != expected
    ):
        raise TrainingDatasetCompositionError(
            "Consolidated training dataset composition receipt is absent or stale."
        )
    return BoundTrainingDatasetComposition(
        archive_path=path,
        crop=crop,
        base=base,
        binding=expected,
    )


__all__ = [
    "BoundTrainingDatasetComposition",
    "TRAINING_DATASET_COMPOSITION_ATTRIBUTE",
    "TRAINING_DATASET_COMPOSITION_SCHEMA_ID",
    "TRAINING_DATASET_COMPOSITION_SCHEMA_VERSION",
    "TrainingDatasetCompositionError",
    "TrainingDetectionReviewBase",
    "bind_training_dataset_composition",
    "build_training_dataset_composition",
    "validate_training_detection_review_base",
    "validate_training_crop_source_join",
]
