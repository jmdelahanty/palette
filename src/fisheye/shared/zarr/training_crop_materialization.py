"""Strict source binding for self-contained training crop pixels.

The geometry-only crop contract remains the authority for observation identity
and coordinate semantics.  This module binds a copied ``roi_images`` surface
to that authority without pretending the training archive owns a second
crop-v2 authority.  Pixel payloads are deliberately not re-hashed when a
consumer opens a multi-gigabyte training store; publication is responsible for
physical-copy checksums, while this binding revalidates the small identity
columns that join inference output back to the source observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    derive_crop_placement_geometry,
)
from fisheye.shared.zarr.detection_schema import (
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct

TRAINING_CROP_MATERIALIZATION_SCHEMA_ID = "palette.training_crop_materialization.v1"
TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE = (
    "training_crop_materialization_binding"
)
TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_ID = (
    "palette.training_crop_materialization_binding"
)
TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_VERSION = 1
SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER = "sampled_training_images_full"
SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER = (
    "sampled_acquisition_crop_video_hybrid"
)
SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_ID = "palette.sampled_training_crop_geometry"
SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_VERSION = 1
ACQUISITION_HYBRID_PIXEL_SOURCE_CODE_MAP = {
    "0": "acquisition_crop_video_lossless",
    "1": "sampled_images_full_fallback",
}
ACQUISITION_HYBRID_FALLBACK_REASON_CODE_MAP = {
    "0": "none",
    "1": "missing_crop_metadata",
    "2": "blank_crop_frame",
    "3": "crop_has_no_detection",
    "4": "invalid_crop_geometry",
    "5": "reviewed_bbox_outside_recorded_crop",
}
TRAINING_CROP_MATERIALIZATION_PROVIDERS = (
    "source_video_pynvvc_luma",
    "verified_flat_roi_cache",
    SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER,
    SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER,
)

_BASE_IDENTITY_ARRAYS = (
    *CROP_GEOMETRY_SCHEMA_V1.binding_paths,
    "source_frame_indices",
)
_SOURCE_CROP_IDENTITY_ARRAYS = ("source_crop_row_ids",)
_OPTIONAL_CLIPPED_IDENTITY_ARRAYS = (
    "source_clip_local_frame_indices",
    "source_clip_indices",
)
_ACQUISITION_HYBRID_IDENTITY_ARRAYS = (
    "source_training_row_indices",
    "source_crop_meta_row_indices",
    "source_crop_video_frame_indices",
    "source_crop_local_frame_ids",
    "pixel_source_codes",
    "fallback_reason_codes",
)


class TrainingCropMaterializationError(RuntimeError):
    """Raised when training pixels are not exactly bound to crop geometry."""


@dataclass(frozen=True)
class BoundTrainingCropMaterialization:
    archive_path: Path
    run_id: str
    run_path: str
    run_group: zarr.Group
    roi_images: zarr.Array
    instance_key: zarr.Array
    source_frame_indices: zarr.Array
    binding: Mapping[str, Any]

    @property
    def row_count(self) -> int:
        return int(self.roi_images.shape[0])

    @property
    def roi_shape(self) -> tuple[int, int]:
        return (int(self.roi_images.shape[1]), int(self.roi_images.shape[2]))


def _array_declaration(array: Any) -> dict[str, Any]:
    return {
        "dtype": np.dtype(array.dtype).str,
        "shape": [int(value) for value in array.shape],
    }


def _identity_digest(array: Any) -> str:
    return sha256_array(np.asarray(array[:]))


def _exact_array(
    run: zarr.Group,
    name: str,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    node = run[name]
    if np.dtype(node.dtype) != dtype or tuple(int(v) for v in node.shape) != shape:
        raise TrainingCropMaterializationError(
            f"{name} must have exact {dtype} shape {shape}; "
            f"got dtype={node.dtype}, shape={node.shape}."
        )
    return np.asarray(node[:], dtype=dtype)


def validate_sampled_training_images_full_materialization(
    run: zarr.Group,
) -> None:
    """Require the explicit local-frame/acquisition-frame sampled geometry.

    Recording crop-v1 uses the complete acquisition frame axis. A sampled
    training artifact instead owns a compact local image axis and maps it to
    acquisition frames. This validator freezes that distinction rather than
    weakening or falsely relabelling the recording-level crop contract.
    """

    raw_shape = run.attrs.get("source_images_shape")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 3
        or any(type(value) is not int or value <= 0 for value in raw_shape)
    ):
        raise TrainingCropMaterializationError(
            "sampled images_full provider requires source_images_shape=[F,H,W]."
        )
    frame_count, source_height, source_width = (int(v) for v in raw_shape)
    if run.attrs.get("source_images_path") != "raw_video/images_full":
        raise TrainingCropMaterializationError(
            "sampled provider source_images_path must be raw_video/images_full."
        )
    if str(run.attrs.get("source_images_dtype") or "") not in {"uint8", "|u1"}:
        raise TrainingCropMaterializationError(
            "sampled images_full provider requires exact uint8 source pixels."
        )
    if (
        int(run.attrs.get("height") or 0) != source_height
        or int(run.attrs.get("width") or 0) != source_width
    ):
        raise TrainingCropMaterializationError(
            "sampled provider source dimensions differ from source_images_shape."
        )
    source_refined_run = str(run.attrs.get("source_refined_detect_run") or "").strip()
    if not source_refined_run or "/" in source_refined_run:
        raise TrainingCropMaterializationError(
            "sampled provider requires one safe source refined-detect run name."
        )
    if run.attrs.get("source_frame_decision_path") != (
        f"detect_frame_decision_runs/{source_refined_run}"
    ):
        raise TrainingCropMaterializationError(
            "source frame-decision path must bind the selected refined run."
        )
    row_count = int(run["roi_images"].shape[0])
    keys = _exact_array(
        run,
        "instance_key",
        dtype=np.dtype(np.uint64),
        shape=(row_count,),
    )
    if np.unique(keys).shape[0] != row_count:
        raise TrainingCropMaterializationError("instance_key values must be unique.")
    local_frames = _exact_array(
        run,
        "frame_indices",
        dtype=np.dtype(np.int64),
        shape=(row_count,),
    )
    acquisition_frames = _exact_array(
        run,
        "source_acquisition_frame_index",
        dtype=np.dtype(np.int64),
        shape=(row_count,),
    )
    source_frames = _exact_array(
        run,
        "source_frame_indices",
        dtype=np.dtype(np.int64),
        shape=(row_count,),
    )
    if not np.array_equal(source_frames, acquisition_frames):
        raise TrainingCropMaterializationError(
            "source_frame_indices must equal source acquisition-frame identity."
        )
    if row_count and (
        int(local_frames.min()) < 0
        or int(local_frames.max()) >= frame_count
        or np.any(local_frames[1:] < local_frames[:-1])
    ):
        raise TrainingCropMaterializationError(
            "frame_indices must be sorted within the sampled images_full axis."
        )
    if np.any(acquisition_frames < 0):
        raise TrainingCropMaterializationError(
            "source acquisition-frame identities must be nonnegative."
        )
    expected_offsets = np.zeros(frame_count + 1, dtype=np.int64)
    expected_offsets[1:] = np.cumsum(
        np.bincount(local_frames, minlength=frame_count), dtype=np.int64
    )
    offsets = _exact_array(
        run,
        "frame_row_offsets",
        dtype=np.dtype(np.int64),
        shape=(frame_count + 1,),
    )
    if not np.array_equal(offsets, expected_offsets):
        raise TrainingCropMaterializationError(
            "frame_row_offsets does not exactly index sampled local frames."
        )

    bbox_norm = _exact_array(
        run,
        "bbox_norm_coords",
        dtype=np.dtype(np.float32),
        shape=(row_count, 4),
    )
    if row_count:
        half = np.float32(0.5)
        if (
            not np.isfinite(bbox_norm).all()
            or np.any(bbox_norm[:, 2:] <= 0)
            or np.any(bbox_norm[:, :2] - bbox_norm[:, 2:] * half < 0)
            or np.any(bbox_norm[:, :2] + bbox_norm[:, 2:] * half > 1)
        ):
            raise TrainingCropMaterializationError(
                "bbox_norm_coords must be finite, positive-area, and contained."
            )
    expected_bbox_img, expected_centers = derive_canonical_detection_geometry(
        bbox_norm,
        source_width=source_width,
        source_height=source_height,
    )
    bbox_img = _exact_array(
        run,
        "bbox_img_xyxy",
        dtype=np.dtype(np.float32),
        shape=(row_count, 4),
    )
    centers = _exact_array(
        run,
        "centers_img_xy",
        dtype=np.dtype(np.float32),
        shape=(row_count, 2),
    )
    if not np.array_equal(bbox_img, expected_bbox_img) or not np.array_equal(
        centers, expected_centers
    ):
        raise TrainingCropMaterializationError(
            "Sampled crop pixel geometry is not the exact float32 bbox projection."
        )
    sizes = _exact_array(
        run,
        "roi_sizes_full",
        dtype=np.dtype(np.int32),
        shape=(row_count, 2),
    )
    if np.any(sizes <= 0):
        raise TrainingCropMaterializationError("roi_sizes_full must be positive.")
    expected_coordinates, expected_source_crop, expected_bbox_roi = (
        derive_crop_placement_geometry(centers, bbox_img, sizes)
    )
    for name, expected, dtype in (
        ("roi_coordinates_full", expected_coordinates, np.dtype(np.int32)),
        ("source_crop_xywh", expected_source_crop, np.dtype(np.float32)),
        ("bbox_roi_xyxy", expected_bbox_roi, np.dtype(np.float32)),
    ):
        observed = _exact_array(
            run,
            name,
            dtype=dtype,
            shape=tuple(int(v) for v in expected.shape),
        )
        if not np.array_equal(observed, expected):
            raise TrainingCropMaterializationError(
                f"{name} differs from the sampled crop placement rule."
            )
    roi_images = run["roi_images"]
    if np.dtype(roi_images.dtype) != np.dtype(np.uint8) or len(roi_images.shape) != 3:
        raise TrainingCropMaterializationError(
            "roi_images must be one rank-3 uint8 crop payload."
        )
    roi_height, roi_width = (int(v) for v in roi_images.shape[1:])
    if row_count and not np.array_equal(
        sizes,
        np.repeat(
            np.asarray([[roi_width, roi_height]], dtype=np.int32),
            row_count,
            axis=0,
        ),
    ):
        raise TrainingCropMaterializationError(
            "roi_images extent must equal every persisted roi_sizes_full row."
        )
    _exact_array(
        run,
        "source_refined_row_ids",
        dtype=np.dtype(np.int64),
        shape=(row_count,),
    )
    _exact_array(
        run,
        "source_row_signature",
        dtype=np.dtype(np.uint8),
        shape=(row_count, 32),
    )
    digest = str(run.attrs.get("source_frame_decision_digest") or "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise TrainingCropMaterializationError(
            "source_frame_decision_digest must be one lowercase SHA-256."
        )
    if run.attrs.get("padding_mode") != "zero_outside_source_frame":
        raise TrainingCropMaterializationError(
            "sampled images_full crops require explicit zero padding semantics."
        )
    if run.attrs.get("pixel_verification") != (
        "all_rows_byte_equal_to_source_window_v1"
    ):
        raise TrainingCropMaterializationError(
            "sampled images_full publication requires all-row pixel verification."
        )


def validate_sampled_acquisition_crop_hybrid_materialization(
    run: zarr.Group,
) -> None:
    """Validate explicit acquisition-video versus full-frame fallback rows."""

    raw_shape = run.attrs.get("source_images_shape")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 3
        or any(type(value) is not int or value <= 0 for value in raw_shape)
    ):
        raise TrainingCropMaterializationError(
            "acquisition hybrid provider requires source_images_shape=[F,H,W]."
        )
    frame_count, source_height, source_width = (int(v) for v in raw_shape)
    if run.attrs.get("source_images_path") != "raw_video/images_full":
        raise TrainingCropMaterializationError(
            "acquisition hybrid fallback must bind raw_video/images_full."
        )
    row_count = int(run["roi_images"].shape[0])
    roi_images = run["roi_images"]
    if np.dtype(roi_images.dtype) != np.dtype(np.uint8) or len(roi_images.shape) != 3:
        raise TrainingCropMaterializationError(
            "roi_images must be one rank-3 uint8 crop payload."
        )
    roi_height, roi_width = (int(value) for value in roi_images.shape[1:])
    declared_shape = run.attrs.get("acquisition_crop_shape")
    if declared_shape != [roi_height, roi_width]:
        raise TrainingCropMaterializationError(
            "acquisition_crop_shape must equal the persisted roi_images extent."
        )
    if (
        int(run.attrs.get("height") or 0) != source_height
        or int(run.attrs.get("width") or 0) != source_width
    ):
        raise TrainingCropMaterializationError(
            "acquisition hybrid source dimensions differ from source_images_shape."
        )

    local_frames = _exact_array(
        run, "frame_indices", dtype=np.dtype(np.int64), shape=(row_count,)
    )
    acquisition_frames = _exact_array(
        run,
        "source_acquisition_frame_index",
        dtype=np.dtype(np.int64),
        shape=(row_count,),
    )
    source_frames = _exact_array(
        run, "source_frame_indices", dtype=np.dtype(np.int64), shape=(row_count,)
    )
    if not np.array_equal(source_frames, acquisition_frames):
        raise TrainingCropMaterializationError(
            "source_frame_indices must equal source acquisition-frame identity."
        )
    if row_count and (
        int(local_frames.min()) < 0
        or int(local_frames.max()) >= frame_count
        or np.any(local_frames[1:] < local_frames[:-1])
    ):
        raise TrainingCropMaterializationError(
            "frame_indices must be sorted within the sampled full-frame axis."
        )
    expected_offsets = np.zeros(frame_count + 1, dtype=np.int64)
    expected_offsets[1:] = np.cumsum(
        np.bincount(local_frames, minlength=frame_count), dtype=np.int64
    )
    offsets = _exact_array(
        run,
        "frame_row_offsets",
        dtype=np.dtype(np.int64),
        shape=(frame_count + 1,),
    )
    if not np.array_equal(offsets, expected_offsets):
        raise TrainingCropMaterializationError(
            "frame_row_offsets does not exactly index sampled local frames."
        )

    keys = _exact_array(
        run, "instance_key", dtype=np.dtype(np.uint64), shape=(row_count,)
    )
    if np.unique(keys).shape[0] != row_count:
        raise TrainingCropMaterializationError("instance_key values must be unique.")
    _exact_array(
        run,
        "source_refined_row_ids",
        dtype=np.dtype(np.int64),
        shape=(row_count,),
    )
    source_training_rows = _exact_array(
        run,
        "source_training_row_indices",
        dtype=np.dtype(np.int64),
        shape=(row_count,),
    )
    if not np.array_equal(source_training_rows, local_frames):
        raise TrainingCropMaterializationError(
            "source_training_row_indices must equal the compact sampled frame index."
        )
    meta_rows = _exact_array(
        run,
        "source_crop_meta_row_indices",
        dtype=np.dtype(np.int64),
        shape=(row_count,),
    )
    video_rows = _exact_array(
        run,
        "source_crop_video_frame_indices",
        dtype=np.dtype(np.int64),
        shape=(row_count,),
    )
    local_crop_rows = _exact_array(
        run,
        "source_crop_local_frame_ids",
        dtype=np.dtype(np.int64),
        shape=(row_count,),
    )
    source_codes = _exact_array(
        run, "pixel_source_codes", dtype=np.dtype(np.uint8), shape=(row_count,)
    )
    reason_codes = _exact_array(
        run, "fallback_reason_codes", dtype=np.dtype(np.uint8), shape=(row_count,)
    )
    if np.any(source_codes > 1) or np.any(reason_codes > 5):
        raise TrainingCropMaterializationError(
            "Acquisition hybrid source/reason codes are outside the v1 registries."
        )
    acquisition = source_codes == 0
    fallback = source_codes == 1
    if np.any(acquisition & (reason_codes != 0)) or np.any(
        fallback & (reason_codes == 0)
    ):
        raise TrainingCropMaterializationError(
            "Acquisition rows require reason 0 and fallback rows require a reason."
        )
    if np.any(
        acquisition & ((meta_rows < 0) | (video_rows < 0) | (local_crop_rows < 0))
    ):
        raise TrainingCropMaterializationError(
            "Acquisition rows require exact nonnegative crop-video lineage."
        )
    if np.any(
        fallback & ((meta_rows != -1) | (video_rows != -1) | (local_crop_rows != -1))
    ):
        raise TrainingCropMaterializationError(
            "Fallback rows must use -1 for unavailable crop-video lineage."
        )

    bbox_norm = _exact_array(
        run, "bbox_norm_coords", dtype=np.dtype(np.float32), shape=(row_count, 4)
    )
    expected_bbox_img, expected_centers = derive_canonical_detection_geometry(
        bbox_norm, source_width=source_width, source_height=source_height
    )
    bbox_img = _exact_array(
        run, "bbox_img_xyxy", dtype=np.dtype(np.float32), shape=(row_count, 4)
    )
    centers = _exact_array(
        run, "centers_img_xy", dtype=np.dtype(np.float32), shape=(row_count, 2)
    )
    if not np.array_equal(bbox_img, expected_bbox_img) or not np.array_equal(
        centers, expected_centers
    ):
        raise TrainingCropMaterializationError(
            "Acquisition hybrid detection projections differ from their authority."
        )
    sizes = _exact_array(
        run, "roi_sizes_full", dtype=np.dtype(np.int32), shape=(row_count, 2)
    )
    expected_sizes = np.repeat(
        np.asarray([[roi_width, roi_height]], dtype=np.int32), row_count, axis=0
    )
    if not np.array_equal(sizes, expected_sizes):
        raise TrainingCropMaterializationError(
            "Every acquisition hybrid row must use the native crop-video extent."
        )
    coordinates = _exact_array(
        run,
        "roi_coordinates_full",
        dtype=np.dtype(np.int32),
        shape=(row_count, 2),
    )
    source_crop = _exact_array(
        run, "source_crop_xywh", dtype=np.dtype(np.float32), shape=(row_count, 4)
    )
    expected_source_crop = np.concatenate(
        (coordinates.astype(np.float32), sizes.astype(np.float32)), axis=1
    )
    if not np.array_equal(source_crop, expected_source_crop):
        raise TrainingCropMaterializationError(
            "source_crop_xywh must exactly describe persisted placement and extent."
        )
    bbox_roi = _exact_array(
        run, "bbox_roi_xyxy", dtype=np.dtype(np.float32), shape=(row_count, 4)
    )
    expected_bbox_roi = bbox_img - np.concatenate(
        (coordinates.astype(np.float32), coordinates.astype(np.float32)), axis=1
    )
    if not np.array_equal(bbox_roi, expected_bbox_roi):
        raise TrainingCropMaterializationError(
            "bbox_roi_xyxy must translate authoritative source-camera boxes."
        )
    if row_count and (
        np.any(bbox_roi[:, :2] < 0)
        or np.any(bbox_roi[:, 2] > roi_width)
        or np.any(bbox_roi[:, 3] > roi_height)
    ):
        raise TrainingCropMaterializationError(
            "Every reviewed box must be fully represented by its persisted pixels."
        )
    _exact_array(
        run,
        "source_row_signature",
        dtype=np.dtype(np.uint8),
        shape=(row_count, 32),
    )
    if run.attrs.get("pixel_verification") != (
        "all_rows_byte_equal_to_declared_provider_v1"
    ):
        raise TrainingCropMaterializationError(
            "Acquisition hybrid publication requires all-row pixel verification."
        )
    if run.attrs.get("fallback_policy") != "sampled_images_full_zero_padded_v1":
        raise TrainingCropMaterializationError(
            "Acquisition hybrid fallback policy is missing or ambiguous."
        )
    if run.attrs.get("pixel_source_code_map") != (
        ACQUISITION_HYBRID_PIXEL_SOURCE_CODE_MAP
    ) or run.attrs.get("fallback_reason_code_map") != (
        ACQUISITION_HYBRID_FALLBACK_REASON_CODE_MAP
    ):
        raise TrainingCropMaterializationError(
            "Acquisition hybrid code registries differ from the frozen v1 maps."
        )


def build_training_crop_materialization_binding(
    run: zarr.Group,
) -> dict[str, Any]:
    """Build the exact persisted binding after all pixels have been written."""

    provider = str(run.attrs.get("training_materialization_provider") or "")
    if provider not in TRAINING_CROP_MATERIALIZATION_PROVIDERS:
        raise TrainingCropMaterializationError(
            f"Unsupported training materialization provider: {provider!r}."
        )
    if run.attrs.get("training_materialization_schema") != (
        TRAINING_CROP_MATERIALIZATION_SCHEMA_ID
    ):
        raise TrainingCropMaterializationError(
            "Training crop materialization schema identity is missing or wrong."
        )
    if provider == SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER:
        required_identity_arrays = _BASE_IDENTITY_ARRAYS
    elif provider == SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER:
        required_identity_arrays = (
            *_BASE_IDENTITY_ARRAYS,
            *_ACQUISITION_HYBRID_IDENTITY_ARRAYS,
        )
    else:
        required_identity_arrays = (
            *_BASE_IDENTITY_ARRAYS,
            *_SOURCE_CROP_IDENTITY_ARRAYS,
        )
    missing = [
        name for name in (*required_identity_arrays, "roi_images") if name not in run
    ]
    if missing:
        raise TrainingCropMaterializationError(
            f"Training crop materialization is missing required arrays: {missing}."
        )

    roi_images = run["roi_images"]
    if np.dtype(roi_images.dtype) != np.dtype(np.uint8) or len(roi_images.shape) != 3:
        raise TrainingCropMaterializationError(
            "roi_images must be a rank-3 uint8 array."
        )
    if provider == SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER:
        validate_sampled_training_images_full_materialization(run)
    elif provider == SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER:
        validate_sampled_acquisition_crop_hybrid_materialization(run)
    row_count = int(roi_images.shape[0])
    declarations: dict[str, dict[str, Any]] = {}
    identity_sha256: dict[str, str] = {}
    for name in (
        *_BASE_IDENTITY_ARRAYS,
        *_SOURCE_CROP_IDENTITY_ARRAYS,
        *_OPTIONAL_CLIPPED_IDENTITY_ARRAYS,
        *_ACQUISITION_HYBRID_IDENTITY_ARRAYS,
    ):
        if name not in run:
            continue
        array = run[name]
        if int(array.shape[0]) != row_count and name != "frame_row_offsets":
            raise TrainingCropMaterializationError(
                f"{name} first-axis length differs from roi_images."
            )
        declarations[name] = _array_declaration(array)
        identity_sha256[name] = _identity_digest(array)
    declarations["roi_images"] = _array_declaration(roi_images)

    source_binding = run.attrs.get("source_crop_manifest_binding")
    if not isinstance(source_binding, Mapping):
        raise TrainingCropMaterializationError(
            "Training crop materialization lacks source_crop_manifest_binding."
        )
    provider_evidence: dict[str, Any]
    if provider == "verified_flat_roi_cache":
        provider_evidence = {
            "manifest_path": run.attrs.get("source_roi_cache_manifest"),
            "manifest_sha256": run.attrs.get("source_roi_cache_manifest_sha256"),
            "payload_sha256": run.attrs.get("source_roi_cache_payload_sha256"),
            "verified": run.attrs.get("source_roi_cache_verified"),
            "runtime_dependency": run.attrs.get("source_roi_cache_independence"),
        }
    elif provider == SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER:
        provider_evidence = {
            "source_images_path": run.attrs.get("source_images_path"),
            "source_images_dtype": run.attrs.get("source_images_dtype"),
            "source_images_shape": run.attrs.get("source_images_shape"),
            "source_refined_detect_run": run.attrs.get("source_refined_detect_run"),
            "source_frame_decision_path": run.attrs.get("source_frame_decision_path"),
            "source_frame_decision_digest": run.attrs.get(
                "source_frame_decision_digest"
            ),
            "padding_mode": run.attrs.get("padding_mode"),
            "pixel_verification": run.attrs.get("pixel_verification"),
        }
    elif provider == SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER:
        provider_evidence = {
            "source_images_path": run.attrs.get("source_images_path"),
            "source_images_shape": run.attrs.get("source_images_shape"),
            "source_refined_detect_run": run.attrs.get("source_refined_detect_run"),
            "source_frame_decision_path": run.attrs.get("source_frame_decision_path"),
            "source_frame_decision_digest": run.attrs.get(
                "source_frame_decision_digest"
            ),
            "acquisition_crop_video_path": run.attrs.get("acquisition_crop_video_path"),
            "acquisition_crop_video_stat": run.attrs.get("acquisition_crop_video_stat"),
            "acquisition_crop_meta_path": run.attrs.get("acquisition_crop_meta_path"),
            "acquisition_crop_meta_sha256": run.attrs.get(
                "acquisition_crop_meta_sha256"
            ),
            "acquisition_crop_summary_path": run.attrs.get(
                "acquisition_crop_summary_path"
            ),
            "acquisition_crop_summary_sha256": run.attrs.get(
                "acquisition_crop_summary_sha256"
            ),
            "acquisition_encoder_contract": run.attrs.get(
                "acquisition_encoder_contract"
            ),
            "pixel_source_code_map": run.attrs.get("pixel_source_code_map"),
            "fallback_reason_code_map": run.attrs.get("fallback_reason_code_map"),
            "fallback_policy": run.attrs.get("fallback_policy"),
            "decode_backend": run.attrs.get("decode_backend"),
            "pixel_verification": run.attrs.get("pixel_verification"),
        }
    else:
        provider_evidence = {
            "source_video_path": run.attrs.get("source_video_path"),
            "decode_backend": run.attrs.get("decode_backend"),
            "pixel_contract_name": run.attrs.get("roi_pixel_contract_name"),
        }

    payload = {
        "schema_id": TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_ID,
        "schema_version": TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_VERSION,
        "provider": provider,
        "stage_selector_eligible": False,
        "source": {
            "archive_path": str(run.attrs.get("source_crop_archive_path") or ""),
            "crop_run": str(run.attrs.get("source_crop_run") or ""),
            "crop_path": str(run.attrs.get("source_crop_path") or ""),
            "crop_manifest": dict(source_binding),
        },
        "dimensions": {
            "row_count": row_count,
            "roi_shape": [int(roi_images.shape[1]), int(roi_images.shape[2])],
            "source_height": int(run.attrs.get("height") or 0),
            "source_width": int(run.attrs.get("width") or 0),
        },
        "array_declarations": declarations,
        "identity_array_sha256": identity_sha256,
        "provider_evidence": provider_evidence,
        "pixel_payload_validation": (
            "physical_publication_checksum_plus_provider_evidence_v1"
        ),
    }
    return {
        "payload": payload,
        "payload_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
    }


def _parse_binding(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "payload",
        "payload_digest_algorithm",
        "payload_digest",
    }:
        raise TrainingCropMaterializationError(
            "Training crop binding envelope has an unexpected field set."
        )
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise TrainingCropMaterializationError(
            "Training crop binding payload must be an object."
        )
    if value.get("payload_digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        raise TrainingCropMaterializationError(
            "Training crop binding digest algorithm mismatch."
        )
    if value.get("payload_digest") != canonical_json_sha256(payload):
        raise TrainingCropMaterializationError(
            "Training crop binding payload digest mismatch."
        )
    return value


def bind_training_crop_materialization(
    archive_path: str | Path,
    *,
    run_id: str,
    require_consolidated: bool = True,
) -> BoundTrainingCropMaterialization:
    """Open one candidate input for keypoint or subject-mask inference."""

    path = Path(archive_path).expanduser().resolve()
    root = open_zarr_group_direct(path, mode="r")
    if str(root.attrs.get("zarr_purpose") or "").strip().lower() != "training":
        raise TrainingCropMaterializationError(
            "Training crop materialization requires a training-purpose Zarr."
        )
    parent = root.get("crop_runs")
    if parent is None or str(run_id) not in parent:
        raise TrainingCropMaterializationError(
            f"Training crop run not found: {run_id!r}."
        )
    run = parent[str(run_id)]
    if run.attrs.get("status") != "completed":
        raise TrainingCropMaterializationError(
            "Training crop materialization is not completed."
        )
    if run.attrs.get("stage_selector_eligible") is not False:
        raise TrainingCropMaterializationError(
            "Training crop materialization must remain selector-ineligible."
        )
    persisted = _parse_binding(
        run.attrs.get(TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE)
    )
    recomputed = build_training_crop_materialization_binding(run)
    if dict(persisted) != recomputed:
        raise TrainingCropMaterializationError(
            "Training crop materialization binding differs from decoded arrays or attrs."
        )
    if require_consolidated:
        try:
            consolidated_root = zarr.open_group(
                str(path), mode="r", zarr_format=3, use_consolidated=True
            )
            consolidated_run = consolidated_root[f"crop_runs/{run_id}"]
        except Exception as exc:
            raise TrainingCropMaterializationError(
                f"Published training crop lacks readable consolidated metadata: {exc}"
            ) from exc
        if (
            consolidated_run.attrs.get(TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE)
            != persisted
        ):
            raise TrainingCropMaterializationError(
                "Direct and consolidated training crop bindings differ."
            )
        for name, expected in persisted["payload"]["array_declarations"].items():
            observed = _array_declaration(consolidated_run[name])
            if observed != expected:
                raise TrainingCropMaterializationError(
                    f"Direct and consolidated declaration differ for {name}."
                )
    payload = persisted["payload"]
    if payload.get("stage_selector_eligible") is not False:
        raise TrainingCropMaterializationError(
            "Training crop binding is not selector-ineligible."
        )
    keys = np.asarray(run["instance_key"][:], dtype=np.uint64).reshape(-1)
    if int(np.unique(keys).shape[0]) != int(keys.shape[0]):
        raise TrainingCropMaterializationError(
            "Training crop instance_key values are not unique."
        )
    return BoundTrainingCropMaterialization(
        archive_path=path,
        run_id=str(run_id),
        run_path=f"crop_runs/{run_id}",
        run_group=run,
        roi_images=run["roi_images"],
        instance_key=run["instance_key"],
        source_frame_indices=run["source_frame_indices"],
        binding=persisted,
    )


__all__ = [
    "ACQUISITION_HYBRID_FALLBACK_REASON_CODE_MAP",
    "ACQUISITION_HYBRID_PIXEL_SOURCE_CODE_MAP",
    "TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE",
    "TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_ID",
    "TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_VERSION",
    "TRAINING_CROP_MATERIALIZATION_PROVIDERS",
    "TRAINING_CROP_MATERIALIZATION_SCHEMA_ID",
    "SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER",
    "SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER",
    "SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_ID",
    "SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_VERSION",
    "BoundTrainingCropMaterialization",
    "TrainingCropMaterializationError",
    "bind_training_crop_materialization",
    "build_training_crop_materialization_binding",
    "validate_sampled_training_images_full_materialization",
    "validate_sampled_acquisition_crop_hybrid_materialization",
]
