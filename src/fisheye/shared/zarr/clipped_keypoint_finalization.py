"""Finalize bounded clip keypoint results into immutable recording snapshots.

Clip-local caches and inference are compute partitions, never public storage
partitions.  This module verifies that every clip result is terminal and bound
to the same crop, coordinate, model, preprocessing, and input-package
contracts, then reorders the complete payload to the recording-level crop row
order.  Publication always delegates physical layout to the shared byte-based
planners; legacy clip chunk/shard metadata is neither inspected nor copied.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.pose_inference_failure import (
    validate_pose_inference_failure_codes,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.body_frame_producer import (
    BodyFrameSourceReference,
    build_keypoint_body_frame_recipe,
    prepare_keypoint_body_frame,
)
from fisheye.shared.zarr.body_frame_publication import (
    BodyFrameShadowPublication,
    publish_selector_ineligible_body_frame_snapshot,
)
from fisheye.shared.zarr.coordinate_manifest import (
    build_coordinate_catalog_envelope,
)
from fisheye.shared.zarr.crop_shadow import (
    CropGeometryShadowPublication,
    validate_crop_geometry_shadow_publication,
)
from fisheye.shared.zarr.keypoint_manifest import (
    KeypointPreprocessingReference,
    keypoint_skeleton_digest,
)
from fisheye.shared.zarr.keypoint_publication_mode import (
    KeypointChainPublicationDispositions,
)
from fisheye.shared.zarr.keypoint_publication import (
    KeypointShadowPublication,
    PreparedRawKeypointSnapshot,
    prepare_raw_keypoint_v2_snapshot,
    publish_selector_ineligible_keypoint_snapshot,
)
from fisheye.shared.zarr.keypoint_quality_producer import (
    ObservationLocalKeypointQualityPolicy,
    prepare_observation_local_keypoint_quality,
)
from fisheye.shared.zarr.keypoint_quality_publication import (
    KeypointQualityShadowPublication,
    publish_selector_ineligible_keypoint_quality_snapshot,
)
from fisheye.shared.zarr.keypoint_quality_schema import (
    KeypointQualitySourceReference,
)
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.keypoint_successor import (
    TerminalKeypointInferenceBatch,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_keypoint_manifest import (
    RefinedKeypointSnapshotIdentity,
    build_refined_keypoint_source_bindings,
)
from fisheye.shared.zarr.refined_keypoint_producer import (
    prepare_refined_keypoint_snapshot,
)
from fisheye.shared.zarr.refined_keypoint_publication import (
    RefinedKeypointShadowPublication,
    publish_selector_ineligible_refined_keypoint_snapshot,
)
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile


CLIPPED_KEYPOINT_RESULT_SCHEMA_ID = "palette.keypoint.clip_terminal_result"
CLIPPED_KEYPOINT_RESULT_SCHEMA_VERSION = 1
CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_ID = (
    "palette.keypoint.clip_terminal_result_receipt"
)
CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_VERSION = 1
CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_ID = (
    "palette.keypoint.clipped_recording_finalization"
)
CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_VERSION = 1
CLIPPED_KEYPOINT_FINALIZATION_RECEIPT_NAME = "finalization_receipt.json"
_SHA256_HEX = frozenset("0123456789abcdef")


class ClippedKeypointFinalizationError(ValueError):
    """Raised when clip results cannot form one complete immutable snapshot."""


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value).strip()
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ClippedKeypointFinalizationError(
            f"{name} must be a lowercase hexadecimal SHA-256 digest."
        )
    return text


def _require_component(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text or "/" in text:
        raise ClippedKeypointFinalizationError(
            f"{name} must be one nonempty path-safe component."
        )
    return text


def _values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def _manifest_payload_digest(manifest: Mapping[str, Any]) -> str:
    return _require_sha256(
        manifest.get("payload_digest"), name="run manifest payload_digest"
    )


@dataclass(frozen=True)
class ClipTerminalKeypointResult:
    """One complete clip result plus all immutable inference input bindings."""

    clip_id: str
    clip_index: int
    terminal_status: str
    inference: TerminalKeypointInferenceBatch
    source_crop_row_signature: np.ndarray
    crop_run_id: str
    crop_manifest_digest: str
    crop_coordinate_catalog_digest: str
    keypoint_coordinate_catalog_digest: str
    pose_model_binding_digest: str
    preprocessing_digest: str
    input_package_manifest_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "clip_id", _require_component(self.clip_id, name="clip_id")
        )
        if type(self.clip_index) is not int or self.clip_index < 0:
            raise ClippedKeypointFinalizationError(
                "clip_index must be a nonnegative exact integer."
            )
        status = str(self.terminal_status).strip()
        if status != "complete":
            raise ClippedKeypointFinalizationError(
                "Only a complete clip artifact may enter recording finalization."
            )
        object.__setattr__(self, "terminal_status", status)
        if not isinstance(self.inference, TerminalKeypointInferenceBatch):
            raise TypeError("inference must be TerminalKeypointInferenceBatch.")
        signatures = np.asarray(self.source_crop_row_signature)
        if signatures.dtype != np.dtype(np.uint8) or signatures.shape != (
            self.inference.n_rows,
            32,
        ):
            raise ClippedKeypointFinalizationError(
                "source_crop_row_signature must have exact uint8 shape [M,32]."
            )
        object.__setattr__(
            self,
            "source_crop_row_signature",
            np.ascontiguousarray(signatures),
        )
        object.__setattr__(
            self,
            "crop_run_id",
            _require_component(self.crop_run_id, name="crop_run_id"),
        )
        for name in (
            "crop_manifest_digest",
            "crop_coordinate_catalog_digest",
            "keypoint_coordinate_catalog_digest",
            "pose_model_binding_digest",
            "preprocessing_digest",
            "input_package_manifest_digest",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )

    def as_manifest(self) -> dict[str, object]:
        keys = np.asarray(self.inference.instance_key, dtype=np.uint64)
        success = np.asarray(self.inference.pose_success, dtype=bool)
        return {
            "schema_id": CLIPPED_KEYPOINT_RESULT_SCHEMA_ID,
            "schema_version": CLIPPED_KEYPOINT_RESULT_SCHEMA_VERSION,
            "clip_id": self.clip_id,
            "clip_index": self.clip_index,
            "terminal_status": self.terminal_status,
            "row_count": self.inference.n_rows,
            "terminal_success_count": int(np.count_nonzero(success)),
            "terminal_failure_count": int(np.count_nonzero(~success)),
            "instance_keys_digest": sha256_array(keys),
            "source_crop_row_signatures_digest": sha256_array(
                self.source_crop_row_signature
            ),
            "source_crop": {
                "run_id": self.crop_run_id,
                "manifest_digest": self.crop_manifest_digest,
                "coordinate_catalog_digest": (
                    self.crop_coordinate_catalog_digest
                ),
            },
            "keypoint_coordinate_catalog_digest": (
                self.keypoint_coordinate_catalog_digest
            ),
            "pose_model_binding_digest": self.pose_model_binding_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "input_package_manifest_digest": self.input_package_manifest_digest,
            "row_terminal_semantics": (
                "every_expected_row_present_pose_success_true_or_false_v1"
            ),
        }


@dataclass(frozen=True)
class PreparedClippedKeypointFinalization:
    prepared: PreparedRawKeypointSnapshot
    clip_receipts: tuple[Mapping[str, object], ...]
    preparation_receipt: Mapping[str, object]


@dataclass(frozen=True)
class ClippedKeypointPublicationChain:
    crop: CropGeometryShadowPublication
    raw: KeypointShadowPublication
    quality: KeypointQualityShadowPublication
    refined: RefinedKeypointShadowPublication
    body_frame: BodyFrameShadowPublication
    receipt_path: Path
    receipt: Mapping[str, object]


def validate_clip_terminal_result_receipt(
    receipt: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate one persisted clip-result sidecar before finalization."""

    errors: list[str] = []
    if set(receipt) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("clip terminal receipt has an unexpected field set")
    if (
        receipt.get("schema_id") != CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_ID
        or receipt.get("schema_version")
        != CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_VERSION
        or receipt.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("clip terminal receipt schema header mismatch")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "clip terminal receipt payload must be an object")
    expected_payload = {
        "status",
        "analysis_zarr",
        "source_group_path",
        "source_crop_group_path",
        "source_coordinate_contract_mode",
        "source_array_hashes",
        "source_crop_array_hashes",
        "source_crop_roi_shape_hw",
        "input_package_manifest_path",
        "result",
        "production_state_changes",
    }
    if set(payload) != expected_payload:
        errors.append("clip terminal receipt payload has an unexpected field set")
    try:
        digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        errors.append(f"clip terminal receipt payload is not canonical JSON: {exc}")
    else:
        if receipt.get("payload_digest") != digest:
            errors.append("clip terminal receipt payload digest mismatch")
    if payload.get("status") != "complete":
        errors.append("clip terminal receipt status is not complete")
    if payload.get("production_state_changes") != []:
        errors.append("clip terminal receipt reports production-state changes")
    package_path = payload.get("input_package_manifest_path")
    if not isinstance(package_path, str) or not package_path.strip():
        errors.append("clip terminal input-package manifest path is invalid")
    source_group = payload.get("source_group_path")
    if not isinstance(source_group, str) or not source_group.strip("/"):
        errors.append("clip terminal source_group_path is invalid")
    source_crop_group = payload.get("source_crop_group_path")
    if not isinstance(source_crop_group, str) or not source_crop_group.strip("/"):
        errors.append("clip terminal source_crop_group_path is invalid")
    if payload.get("source_coordinate_contract_mode") != "legacy_noncanonical":
        errors.append("clip terminal source coordinate mode is invalid")
    hashes = payload.get("source_array_hashes")
    required_hash_paths = {
        "instance_key",
        "source_crop_row_ids",
        "frame_indices",
        "keypoints_roi",
        "keypoint_confidences",
        "confidence",
        "pose_bbox_xyxy_roi",
        "detection_success",
    }
    if not isinstance(hashes, Mapping) or set(hashes) != required_hash_paths:
        errors.append("clip terminal source-array hashes are incomplete")
    elif any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
        for value in hashes.values()
    ):
        errors.append("clip terminal source-array hash is invalid")
    crop_hashes = payload.get("source_crop_array_hashes")
    required_crop_hash_paths = {
        "instance_key",
        "source_crop_row_ids",
        "frame_indices",
        "roi_coordinates_full",
    }
    if not isinstance(crop_hashes, Mapping) or set(crop_hashes) != (
        required_crop_hash_paths
    ):
        errors.append("clip terminal source-crop hashes are incomplete")
    elif any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
        for value in crop_hashes.values()
    ):
        errors.append("clip terminal source-crop hash is invalid")
    roi_shape = payload.get("source_crop_roi_shape_hw")
    if (
        not isinstance(roi_shape, list)
        or len(roi_shape) != 2
        or any(type(value) is not int or value <= 0 for value in roi_shape)
    ):
        errors.append("clip terminal source-crop ROI shape is invalid")
    result = payload.get("result")
    expected_result_fields = {
        "schema_id",
        "schema_version",
        "clip_id",
        "clip_index",
        "terminal_status",
        "row_count",
        "terminal_success_count",
        "terminal_failure_count",
        "instance_keys_digest",
        "source_crop_row_signatures_digest",
        "source_crop",
        "keypoint_coordinate_catalog_digest",
        "pose_model_binding_digest",
        "preprocessing_digest",
        "input_package_manifest_digest",
        "row_terminal_semantics",
    }
    if not isinstance(result, Mapping) or set(result) != expected_result_fields:
        errors.append("clip terminal result declaration is invalid")
    elif (
        result.get("schema_id") != CLIPPED_KEYPOINT_RESULT_SCHEMA_ID
        or result.get("schema_version") != CLIPPED_KEYPOINT_RESULT_SCHEMA_VERSION
        or result.get("terminal_status") != "complete"
        or result.get("row_terminal_semantics")
        != "every_expected_row_present_pose_success_true_or_false_v1"
    ):
        errors.append("clip terminal result contract mismatch")
    return tuple(errors)


def clipped_keypoint_binding_digests(
    *,
    crop: CropGeometryShadowPublication,
    pose_model_schema_binding: Mapping[str, Any],
    preprocessing: KeypointPreprocessingReference,
) -> dict[str, str]:
    payload = crop.manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise ClippedKeypointFinalizationError("Crop run manifest payload is missing.")
    coordinate = payload.get("coordinate_contract")
    if not isinstance(coordinate, Mapping):
        raise ClippedKeypointFinalizationError(
            "Clipped keypoint v2 requires a coordinate-catalog crop-v2 manifest."
        )
    keypoint_coordinate = build_coordinate_catalog_envelope(
        KEYPOINT_SCHEMA_V2.coordinate_contract_manifest()
    )
    return {
        "crop_manifest_digest": canonical_json_sha256(crop.manifest),
        "crop_coordinate_catalog_digest": _require_sha256(
            coordinate.get("digest"), name="crop coordinate catalog digest"
        ),
        "keypoint_coordinate_catalog_digest": _require_sha256(
            keypoint_coordinate.get("digest"),
            name="keypoint coordinate catalog digest",
        ),
        "pose_model_binding_digest": canonical_json_sha256(
            pose_model_schema_binding
        ),
        "preprocessing_digest": canonical_json_sha256(
            preprocessing.as_manifest()
        ),
    }


def clip_terminal_result_from_yolo_arrays(
    crop: CropGeometryShadowPublication,
    yolo_arrays: Mapping[str, Any],
    *,
    source_crop_arrays: Mapping[str, Any],
    clip_id: str,
    clip_index: int,
    pose_model_schema_binding: Mapping[str, Any],
    preprocessing: KeypointPreprocessingReference,
    input_package_manifest_digest: str,
) -> ClipTerminalKeypointResult:
    """Bind one legacy YOLO clip payload to the strict recording crop.

    Float64 legacy pose coordinates are narrowed only at this explicit adapter.
    The resulting terminal batch already obeys raw-keypoint-v2 dtypes and NaN
    failure semantics.  The source arrays remain immutable evidence and their
    exact hashes belong in the persisted clip receipt written by the CLI.
    """

    required = {
        "instance_key",
        "source_crop_row_ids",
        "frame_indices",
        "keypoints_roi",
        "keypoint_confidences",
        "confidence",
        "pose_bbox_xyxy_roi",
        "detection_success",
    }
    missing = sorted(required - set(yolo_arrays))
    if missing:
        raise ClippedKeypointFinalizationError(
            f"YOLO clip payload is missing required arrays: {missing!r}."
        )
    keys = _values(yolo_arrays["instance_key"])
    if keys.dtype != np.dtype(np.uint64) or keys.ndim != 1:
        raise ClippedKeypointFinalizationError(
            "YOLO clip instance_key must have exact uint64 shape [M]."
        )
    crop_keys = _values(crop.arrays["instance_key"])
    crop_signatures = _values(crop.arrays["source_row_signature"])
    crop_rows_by_key = {int(key): row for row, key in enumerate(crop_keys)}
    if any(int(key) not in crop_rows_by_key for key in keys):
        raise ClippedKeypointFinalizationError(
            "YOLO clip contains an instance_key absent from the crop snapshot."
        )
    rows = np.asarray([crop_rows_by_key[int(key)] for key in keys], dtype=np.int64)
    source_rows = _values(yolo_arrays["source_crop_row_ids"])
    source_frames = _values(yolo_arrays["frame_indices"])
    if source_rows.dtype != np.dtype(np.int64) or source_rows.shape != keys.shape:
        raise ClippedKeypointFinalizationError(
            "YOLO clip source_crop_row_ids must have exact int64 shape [M]."
        )
    if (
        source_frames.dtype != np.dtype(np.int64)
        or source_frames.shape != keys.shape
    ):
        raise ClippedKeypointFinalizationError(
            "YOLO clip frame_indices must have exact int64 shape [M]."
        )
    required_source_crop = {
        "instance_key",
        "source_crop_row_ids",
        "frame_indices",
        "roi_coordinates_full",
        "roi_sizes_full",
    }
    missing_crop = sorted(required_source_crop - set(source_crop_arrays))
    if missing_crop:
        raise ClippedKeypointFinalizationError(
            f"YOLO source crop is missing required arrays: {missing_crop!r}."
        )
    source_crop_keys = _values(source_crop_arrays["instance_key"])
    source_crop_row_ids = _values(source_crop_arrays["source_crop_row_ids"])
    source_crop_frames = _values(source_crop_arrays["frame_indices"])
    source_crop_origins = _values(source_crop_arrays["roi_coordinates_full"])
    source_crop_sizes = _values(source_crop_arrays["roi_sizes_full"])
    if (
        source_crop_keys.dtype != np.dtype(np.uint64)
        or source_crop_keys.ndim != 1
        or source_crop_row_ids.dtype != np.dtype(np.int64)
        or source_crop_row_ids.shape != source_crop_keys.shape
        or source_crop_frames.dtype != np.dtype(np.int64)
        or source_crop_frames.shape != source_crop_keys.shape
        or source_crop_origins.dtype != np.dtype(np.int32)
        or source_crop_origins.shape != (source_crop_keys.shape[0], 2)
        or source_crop_sizes.dtype != np.dtype(np.int32)
        or source_crop_sizes.shape != (source_crop_keys.shape[0], 2)
        or np.any(source_rows < 0)
        or np.any(source_rows >= source_crop_keys.shape[0])
    ):
        raise ClippedKeypointFinalizationError(
            "YOLO source crop arrays do not have the exact direct-row mapping "
            "dtypes and shapes."
        )
    if not np.array_equal(source_crop_row_ids, np.arange(source_crop_keys.size)):
        raise ClippedKeypointFinalizationError(
            "YOLO source crop row ids must be one direct contiguous mapping."
        )
    comparisons = (
        ("instance_key", keys, crop_keys[rows]),
        ("source crop instance_key", source_crop_keys[source_rows], keys),
        (
            "frame_indices",
            source_frames,
            _values(crop.arrays["frame_indices"])[rows],
        ),
        (
            "source crop frame_indices",
            source_crop_frames[source_rows],
            source_frames,
        ),
        (
            "source crop origins",
            source_crop_origins[source_rows],
            _values(crop.arrays["roi_coordinates_full"])[rows],
        ),
        (
            "source crop sizes",
            source_crop_sizes[source_rows],
            _values(crop.arrays["roi_sizes_full"])[rows],
        ),
    )
    for name, observed, expected in comparisons:
        if not np.array_equal(observed, expected):
            raise ClippedKeypointFinalizationError(
                f"YOLO clip {name} differs from the bound crop-v2 rows."
            )
    source_success = _values(yolo_arrays["detection_success"])
    if source_success.dtype != np.dtype(bool) or source_success.shape != keys.shape:
        raise ClippedKeypointFinalizationError(
            "YOLO clip detection_success must have exact bool shape [M]."
        )
    if "pose_failure_codes" in yolo_arrays:
        try:
            validate_pose_inference_failure_codes(
                _values(yolo_arrays["pose_failure_codes"]),
                pose_success=source_success,
            )
        except ValueError as exc:
            raise ClippedKeypointFinalizationError(str(exc)) from exc
    inference = TerminalKeypointInferenceBatch(
        instance_key=np.array(keys, copy=True),
        keypoints_roi=_values(yolo_arrays["keypoints_roi"]).astype(np.float32),
        keypoint_confidences=_values(
            yolo_arrays["keypoint_confidences"]
        ).astype(np.float32),
        pose_confidence=_values(yolo_arrays["confidence"]).astype(np.float32),
        pose_bbox_xyxy_roi=_values(yolo_arrays["pose_bbox_xyxy_roi"]).astype(
            np.float32
        ),
        pose_success=source_success,
    )
    digests = clipped_keypoint_binding_digests(
        crop=crop,
        pose_model_schema_binding=pose_model_schema_binding,
        preprocessing=preprocessing,
    )
    return ClipTerminalKeypointResult(
        clip_id=clip_id,
        clip_index=clip_index,
        terminal_status="complete",
        inference=inference,
        source_crop_row_signature=np.array(
            crop_signatures[rows], dtype=np.uint8, copy=True
        ),
        crop_run_id=crop.run_id,
        crop_manifest_digest=digests["crop_manifest_digest"],
        crop_coordinate_catalog_digest=digests[
            "crop_coordinate_catalog_digest"
        ],
        keypoint_coordinate_catalog_digest=digests[
            "keypoint_coordinate_catalog_digest"
        ],
        pose_model_binding_digest=digests["pose_model_binding_digest"],
        preprocessing_digest=digests["preprocessing_digest"],
        input_package_manifest_digest=input_package_manifest_digest,
    )


def prepare_clipped_keypoint_finalization(
    crop: CropGeometryShadowPublication,
    clips: Sequence[ClipTerminalKeypointResult],
    *,
    pose_model_schema_binding: Mapping[str, Any],
    preprocessing: KeypointPreprocessingReference,
) -> PreparedClippedKeypointFinalization:
    """Validate and reorder terminal clip results to recording crop-row order."""

    crop_errors = validate_crop_geometry_shadow_publication(crop)
    if crop_errors:
        raise ClippedKeypointFinalizationError(
            "Source crop publication is invalid: " + "; ".join(crop_errors)
        )
    resolved_clips = tuple(clips)
    if not resolved_clips:
        raise ClippedKeypointFinalizationError(
            "At least one terminal clip result is required."
        )
    clip_ids = [clip.clip_id for clip in resolved_clips]
    clip_indices = [clip.clip_index for clip in resolved_clips]
    if len(set(clip_ids)) != len(clip_ids):
        raise ClippedKeypointFinalizationError("Clip ids must be unique.")
    if sorted(clip_indices) != list(range(len(resolved_clips))):
        raise ClippedKeypointFinalizationError(
            "Clip indices must form the contiguous domain [0, clip_count)."
        )
    ordered_clips = tuple(sorted(resolved_clips, key=lambda item: item.clip_index))
    expected = clipped_keypoint_binding_digests(
        crop=crop,
        pose_model_schema_binding=pose_model_schema_binding,
        preprocessing=preprocessing,
    )
    for clip in ordered_clips:
        if clip.crop_run_id != crop.run_id:
            raise ClippedKeypointFinalizationError(
                f"Clip {clip.clip_id!r} binds a different crop run."
            )
        for name, value in expected.items():
            if getattr(clip, name) != value:
                raise ClippedKeypointFinalizationError(
                    f"Clip {clip.clip_id!r} {name} differs from finalization."
                )

    target_keys = _values(crop.arrays["instance_key"])
    target_signatures = _values(crop.arrays["source_row_signature"])
    target_frames = _values(crop.arrays["frame_indices"])
    target_acquisition = _values(crop.arrays["source_acquisition_frame_index"])
    if target_keys.dtype != np.dtype(np.uint64):
        raise ClippedKeypointFinalizationError(
            "Crop instance_key must have exact uint64 dtype."
        )
    if np.unique(target_keys).shape[0] != target_keys.shape[0]:
        raise ClippedKeypointFinalizationError(
            "Crop instance_key values are not unique."
        )
    all_keys = np.concatenate(
        [np.asarray(clip.inference.instance_key) for clip in ordered_clips]
    )
    if np.unique(all_keys).shape[0] != all_keys.shape[0]:
        raise ClippedKeypointFinalizationError(
            "Clip results repeat one or more instance_key values."
        )
    if not np.array_equal(np.sort(all_keys), np.sort(target_keys)):
        raise ClippedKeypointFinalizationError(
            "Terminal clip rows must exactly partition the crop instance keys."
        )
    keypoints = {clip.inference.n_keypoints for clip in ordered_clips}
    if len(keypoints) != 1:
        raise ClippedKeypointFinalizationError(
            "Clip results disagree on the pose skeleton cardinality."
        )
    n_keypoints = next(iter(keypoints))
    row_count = int(target_keys.shape[0])
    points = np.full((row_count, n_keypoints, 2), np.nan, dtype=np.float32)
    confidences = np.full((row_count, n_keypoints), np.nan, dtype=np.float32)
    pose_confidence = np.full(row_count, np.nan, dtype=np.float32)
    bbox = np.full((row_count, 4), np.nan, dtype=np.float32)
    pose_success = np.zeros(row_count, dtype=bool)
    assigned = np.zeros(row_count, dtype=bool)
    target_order = np.argsort(target_keys, kind="stable")
    sorted_target_keys = target_keys[target_order]
    for clip in ordered_clips:
        clip_keys = np.asarray(clip.inference.instance_key, dtype=np.uint64)
        positions = np.searchsorted(sorted_target_keys, clip_keys)
        if np.any(positions >= sorted_target_keys.shape[0]):
            raise ClippedKeypointFinalizationError(
                f"Clip {clip.clip_id!r} contains an unknown instance_key."
            )
        target_rows = target_order[positions]
        if not np.array_equal(target_keys[target_rows], clip_keys):
            raise ClippedKeypointFinalizationError(
                f"Clip {clip.clip_id!r} contains an unknown instance_key."
            )
        if not np.array_equal(
            clip.source_crop_row_signature,
            target_signatures[target_rows],
        ):
            raise ClippedKeypointFinalizationError(
                f"Clip {clip.clip_id!r} crop signatures differ from crop-v2."
            )
        points[target_rows] = clip.inference.keypoints_roi
        confidences[target_rows] = clip.inference.keypoint_confidences
        pose_confidence[target_rows] = clip.inference.pose_confidence
        bbox[target_rows] = clip.inference.pose_bbox_xyxy_roi
        pose_success[target_rows] = clip.inference.pose_success
        assigned[target_rows] = True
    if not np.all(assigned):
        raise ClippedKeypointFinalizationError(
            "One or more crop rows lacks terminal inference evidence."
        )

    valid = (
        pose_success[:, None]
        & np.all(np.isfinite(points), axis=2)
        & np.isfinite(confidences)
    )
    origins = _values(crop.arrays["roi_coordinates_full"]).astype(
        np.float32, copy=False
    )
    points_img = points + origins[:, None, :]
    bbox_img = bbox + np.column_stack((origins, origins)).astype(
        np.float32, copy=False
    )
    skeleton_digest = keypoint_skeleton_digest(pose_model_schema_binding)
    arrays: dict[str, np.ndarray] = {
        "instance_key": np.array(target_keys, copy=True),
        "source_crop_row_ids": np.arange(row_count, dtype=np.int64),
        "source_acquisition_frame_index": np.array(
            target_acquisition, dtype=np.int64, copy=True
        ),
        "frame_indices": np.array(target_frames, dtype=np.int64, copy=True),
        "frame_row_offsets": np.array(
            _values(crop.arrays["frame_row_offsets"]), dtype=np.int64, copy=True
        ),
        "source_crop_row_signature": np.array(
            target_signatures, dtype=np.uint8, copy=True
        ),
        "keypoints_roi": points,
        "keypoints_img": points_img,
        "keypoint_confidences": confidences,
        "keypoint_valid": valid,
        "pose_confidence": pose_confidence,
        "pose_bbox_xyxy_roi": bbox,
        "pose_bbox_xyxy_img": bbox_img,
        "pose_success": pose_success,
    }
    arrays["keypoint_row_signature"] = derive_keypoint_row_signatures(
        instance_key=arrays["instance_key"],
        source_crop_row_signature=arrays["source_crop_row_signature"],
        keypoints_roi=arrays["keypoints_roi"],
        keypoint_valid=arrays["keypoint_valid"],
        skeleton_digest=skeleton_digest,
    )
    dimensions = KeypointDimensions(
        n_frames=crop.dimensions.n_frames,
        n_instances=row_count,
        n_keypoints=n_keypoints,
        source_width=crop.dimensions.source_width,
        source_height=crop.dimensions.source_height,
    )
    prepared = prepare_raw_keypoint_v2_snapshot(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=crop.arrays,
        source_crop_manifest=crop.manifest,
        pose_model_schema_binding=pose_model_schema_binding,
        preprocessing=preprocessing,
    )
    clip_receipts = tuple(clip.as_manifest() for clip in ordered_clips)
    preparation_receipt = {
        "schema_id": CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_ID,
        "schema_version": CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_VERSION,
        "status": "prepared",
        "crop_run_id": crop.run_id,
        "crop_manifest_digest": expected["crop_manifest_digest"],
        "clip_count": len(ordered_clips),
        "row_count": row_count,
        "terminal_success_count": int(np.count_nonzero(pose_success)),
        "terminal_failure_count": int(np.count_nonzero(~pose_success)),
        "pending_row_count": 0,
        "instance_keys_digest": sha256_array(target_keys),
        "clip_receipts_digest": canonical_json_sha256(list(clip_receipts)),
        "pose_model_binding_digest": expected["pose_model_binding_digest"],
        "preprocessing_digest": expected["preprocessing_digest"],
        "physical_layout_source": "shared_byte_planner_not_clip_metadata",
        "selector_eligible": False,
        "production_state_changes": [],
    }
    return PreparedClippedKeypointFinalization(
        prepared=prepared,
        clip_receipts=clip_receipts,
        preparation_receipt=preparation_receipt,
    )


def _output_binding(publication: Any, *, stage: str) -> dict[str, object]:
    manifest = publication.manifest
    payload = manifest.get("payload")
    logical = payload.get("logical_content") if isinstance(payload, Mapping) else None
    return {
        "stage": stage,
        "path": str(publication.output_path),
        "run_id": publication.run_id,
        "manifest_payload_digest": _manifest_payload_digest(manifest),
        "manifest_digest": canonical_json_sha256(manifest),
        "logical_content_digest": (
            logical.get("digest") if isinstance(logical, Mapping) else None
        ),
        "storage_profile_id": publication.plans.profile.profile_id,
        "selector_eligible": False,
    }


def validate_clipped_keypoint_finalization_receipt(
    receipt: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the exact digest-bound finalization envelope."""

    errors: list[str] = []
    if set(receipt) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("clipped finalization receipt has an unexpected field set")
    if (
        receipt.get("schema_id") != CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_ID
        or receipt.get("schema_version")
        != CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_VERSION
        or receipt.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("clipped finalization receipt schema header mismatch")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "clipped finalization payload must be an object")
    expected_payload = {
        "status",
        "selector_eligible",
        "registry_registered",
        "crop",
        "clip_inputs",
        "preparation",
        "outputs",
        "selector_activation",
        "production_state_changes",
    }
    if set(payload) != expected_payload:
        errors.append("clipped finalization payload has an unexpected field set")
    try:
        digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        errors.append(f"clipped finalization payload is not canonical JSON: {exc}")
    else:
        if receipt.get("payload_digest") != digest:
            errors.append("clipped finalization payload digest mismatch")
    if payload.get("status") != "complete":
        errors.append("clipped finalization status is not complete")
    if payload.get("selector_eligible") is not False:
        errors.append("clipped finalization must remain selector-ineligible")
    if payload.get("registry_registered") is not False:
        errors.append("clipped finalization must remain unregistered")
    if payload.get("selector_activation") != "none_direct_path_only":
        errors.append("clipped finalization selector activation is invalid")
    if payload.get("production_state_changes") != []:
        errors.append("clipped finalization reports production-state changes")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "raw_keypoints",
        "keypoint_quality",
        "refined_keypoints",
        "body_frame",
    }:
        errors.append("clipped finalization output bindings are incomplete")
    elif any(
        not isinstance(value, Mapping) or value.get("selector_eligible") is not False
        for value in outputs.values()
    ):
        errors.append("clipped finalization output became selector-eligible")
    return tuple(errors)


def publish_selector_ineligible_clipped_keypoint_chain(
    crop: CropGeometryShadowPublication,
    clips: Sequence[ClipTerminalKeypointResult],
    *,
    pose_model_schema_binding: Mapping[str, Any],
    preprocessing: KeypointPreprocessingReference,
    bundle_root: Path,
    raw_run_id: str,
    quality_run_id: str,
    refined_run_id: str,
    body_frame_run_id: str,
    refined_identity: RefinedKeypointSnapshotIdentity,
    quality_policy: ObservationLocalKeypointQualityPolicy = (
        ObservationLocalKeypointQualityPolicy()
    ),
    review_state_map: Mapping[int, str] | None = None,
    reason_code_map: Mapping[int, str] | None = None,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    created_by: str = "clipped_keypoint_v2_finalizer",
    dispositions: KeypointChainPublicationDispositions = (
        KeypointChainPublicationDispositions()
    ),
) -> ClippedKeypointPublicationChain:
    """Publish raw, quality, refined, and body-frame runs plus one receipt."""

    root = bundle_root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Clipped finalization bundle already exists: {root}")
    review_codes = {0: "unreviewed"} if review_state_map is None else review_state_map
    reason_codes = {0: "none"} if reason_code_map is None else reason_code_map
    prepared = prepare_clipped_keypoint_finalization(
        crop,
        clips,
        pose_model_schema_binding=pose_model_schema_binding,
        preprocessing=preprocessing,
    )
    skeleton_digest = keypoint_skeleton_digest(pose_model_schema_binding)
    pose_schema = pose_model_schema_binding.get("pose_schema")
    if not isinstance(pose_schema, Mapping):
        raise ClippedKeypointFinalizationError(
            "Pose-model binding lacks the exact pose_schema document."
        )
    skeleton_id = str(pose_schema.get("skeleton_id") or "").strip()
    if not skeleton_id:
        raise ClippedKeypointFinalizationError("Pose schema lacks skeleton_id.")

    raw = publish_selector_ineligible_keypoint_snapshot(
        prepared.prepared,
        destination=root / "raw_keypoints.zarr",
        run_id=raw_run_id,
        shadow_root=root,
        storage_profile=storage_profile,
        created_by=created_by,
        disposition=dispositions.raw,
    )
    raw_manifest_digest = canonical_json_sha256(raw.manifest)
    row_signatures_digest = sha256_array(
        np.asarray(raw.prepared.arrays["keypoint_row_signature"])
    )
    quality_source = KeypointQualitySourceReference(
        run_name=raw.run_id,
        manifest_digest=raw_manifest_digest,
        skeleton_id=skeleton_id,
        skeleton_digest=skeleton_digest,
        keypoint_row_signatures_digest=row_signatures_digest,
    )
    quality_prepared = prepare_observation_local_keypoint_quality(
        raw.prepared.arrays,
        source_dimensions=raw.prepared.dimensions,
        source_crop_arrays=crop.arrays,
        source=quality_source,
        skeleton_digest=skeleton_digest,
        policy=quality_policy,
    )
    quality = publish_selector_ineligible_keypoint_quality_snapshot(
        quality_prepared,
        source_manifest=raw.manifest,
        destination=root / "keypoint_quality.zarr",
        run_id=quality_run_id,
        shadow_root=root,
        storage_profile=storage_profile,
        created_by=created_by,
        disposition=dispositions.quality,
    )
    refined_prepared = prepare_refined_keypoint_snapshot(
        raw.prepared.arrays,
        dimensions=raw.prepared.dimensions,
        source_crop_arrays=crop.arrays,
        skeleton_digest=skeleton_digest,
        keypoint_quality_arrays=quality.prepared.arrays,
        quality_dimensions=quality.prepared.dimensions,
        quality_profile=quality.prepared.profile,
        decisions=(),
        review_state_map=review_codes,
        reason_code_map=reason_codes,
    )
    refined_source = build_refined_keypoint_source_bindings(
        raw_manifest=raw.manifest,
        quality_manifest=quality.manifest,
        crop_manifest=crop.manifest,
    )
    if refined_identity.recording_identity != refined_source.recording_identity:
        raise ClippedKeypointFinalizationError(
            "Refined snapshot identity binds a different recording."
        )
    refined = publish_selector_ineligible_refined_keypoint_snapshot(
        refined_prepared,
        source=refined_source,
        raw_manifest=raw.manifest,
        quality_manifest=quality.manifest,
        crop_manifest=crop.manifest,
        raw_arrays=raw.prepared.arrays,
        quality_arrays=quality.prepared.arrays,
        source_crop_arrays=crop.arrays,
        identity=refined_identity,
        review_state_map=review_codes,
        reason_code_map=reason_codes,
        destination=root / "refined_keypoints.zarr",
        run_id=refined_run_id,
        shadow_root=root,
        storage_profile=storage_profile,
        created_by=created_by,
        disposition=dispositions.refined,
    )
    recipe = build_keypoint_body_frame_recipe(
        pose_schema=pose_schema,
        skeleton_digest=skeleton_digest,
        keypoint_count=raw.prepared.dimensions.n_keypoints,
    )
    body_source = BodyFrameSourceReference(
        stage="refined_keypoints",
        run_name=refined.run_id,
        manifest_digest=canonical_json_sha256(refined.manifest),
        skeleton_id=skeleton_id,
        skeleton_digest=skeleton_digest,
        keypoint_row_signatures_digest=sha256_array(
            np.asarray(refined.prepared.arrays["keypoint_row_signature"])
        ),
    )
    body_prepared = prepare_keypoint_body_frame(
        refined.prepared.arrays,
        source_dimensions=refined.prepared.dimensions,
        source_crop_arrays=crop.arrays,
        source=body_source,
        source_manifest=refined.manifest,
        recipe=recipe,
        review_state_map=review_codes,
        reason_code_map=reason_codes,
    )
    body_frame = publish_selector_ineligible_body_frame_snapshot(
        body_prepared,
        source_manifest=refined.manifest,
        destination=root / "body_frame.zarr",
        run_id=body_frame_run_id,
        shadow_root=root,
        storage_profile=storage_profile,
        created_by=created_by,
        disposition=dispositions.body_frame,
    )
    crop_binding = {
        "stage": "crop",
        "path": str(crop.output_path),
        "run_id": crop.run_id,
        "manifest_payload_digest": _manifest_payload_digest(crop.manifest),
        "manifest_digest": canonical_json_sha256(crop.manifest),
        "logical_content_digest": crop.receipt["logical_content_digest"],
        "storage_profile_id": crop.plans.profile.profile_id,
        "selector_eligible": False,
    }
    payload = {
        "status": "complete",
        "selector_eligible": False,
        "registry_registered": False,
        "crop": crop_binding,
        "clip_inputs": list(prepared.clip_receipts),
        "preparation": dict(prepared.preparation_receipt),
        "outputs": {
            "raw_keypoints": _output_binding(raw, stage="keypoints"),
            "keypoint_quality": _output_binding(
                quality, stage="keypoint_quality"
            ),
            "refined_keypoints": _output_binding(
                refined, stage="refined_keypoints"
            ),
            "body_frame": _output_binding(body_frame, stage="body_frame"),
        },
        "selector_activation": "none_direct_path_only",
        "production_state_changes": [],
    }
    receipt = {
        "schema_id": CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_ID,
        "schema_version": CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    errors = validate_clipped_keypoint_finalization_receipt(receipt)
    if errors:
        raise ClippedKeypointFinalizationError(
            "Finalization receipt is invalid: " + "; ".join(errors)
        )
    receipt_path = root / CLIPPED_KEYPOINT_FINALIZATION_RECEIPT_NAME
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(
            receipt,
            handle,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        handle.write("\n")
    return ClippedKeypointPublicationChain(
        crop=crop,
        raw=raw,
        quality=quality,
        refined=refined,
        body_frame=body_frame,
        receipt_path=receipt_path,
        receipt=receipt,
    )


__all__ = [
    "CLIPPED_KEYPOINT_FINALIZATION_RECEIPT_NAME",
    "CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_ID",
    "CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_VERSION",
    "CLIPPED_KEYPOINT_RESULT_SCHEMA_ID",
    "CLIPPED_KEYPOINT_RESULT_SCHEMA_VERSION",
    "CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_ID",
    "CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_VERSION",
    "ClipTerminalKeypointResult",
    "ClippedKeypointFinalizationError",
    "ClippedKeypointPublicationChain",
    "PreparedClippedKeypointFinalization",
    "clip_terminal_result_from_yolo_arrays",
    "clipped_keypoint_binding_digests",
    "prepare_clipped_keypoint_finalization",
    "publish_selector_ineligible_clipped_keypoint_chain",
    "validate_clip_terminal_result_receipt",
    "validate_clipped_keypoint_finalization_receipt",
]
