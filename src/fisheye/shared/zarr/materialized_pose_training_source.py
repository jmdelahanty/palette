"""Validate a self-contained materialized pose crop as a training source.

This is an input contract, not a dataset-specific exporter. A source run owns
pixels, ROI keypoints, per-point visibility, crop geometry, and source-row
lineage in the same immutable run. Consumers select it by an exact run ID.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr_run_completion import is_run_complete


MATERIALIZED_POSE_CROP_SCHEMA_IDS = frozenset(
    {"palette.training.pose_head_materialized_crop"}
)
_REQUIRED_ROW_ARRAYS = {
    "roi_coordinates_full": (2,),
    "source_keypoint_row_ids": (),
    "source_crop_row_ids": (),
    "source_training_row_indices": (),
    "source_frame_indices": (),
    "bbox_img_xyxy": (4,),
}


@dataclass(frozen=True)
class MaterializedPoseCropSource:
    schema_id: str
    schema_version: int
    run_id: str
    row_count: int
    roi_shape: tuple[int, int]
    keypoint_count: int
    skeleton_id: str
    keypoint_labels: tuple[str, ...]
    source_bindings: dict[str, Any]
    pixel_sha256: str
    keypoints_roi_sha256: str
    keypoint_visibility_sha256: str
    source_row_lineage_sha256: str


def _sha256_array(array: zarr.Array) -> str:
    digest = hashlib.sha256()
    digest.update(str(np.dtype(array.dtype)).encode("ascii"))
    digest.update(json.dumps(tuple(array.shape)).encode("ascii"))
    row_step = max(1, int(array.chunks[0]) if array.chunks else 32)
    for start in range(0, int(array.shape[0]), row_step):
        values = np.ascontiguousarray(array[start : start + row_step])
        digest.update(values.tobytes())
    return digest.hexdigest()


def inspect_materialized_pose_crop_source(
    run: zarr.Group,
    *,
    run_id: str,
    verify_digests: bool = True,
) -> MaterializedPoseCropSource:
    """Refuse an incomplete, inconsistent, or unbound materialized source."""

    schema_id = str(run.attrs.get("schema_id") or "")
    if schema_id not in MATERIALIZED_POSE_CROP_SCHEMA_IDS:
        raise ValueError(f"Unsupported materialized pose crop schema: {schema_id!r}")
    version = int(run.attrs.get("schema_version") or 0)
    if version not in {1, 2, 3}:
        raise ValueError(f"Unsupported materialized pose crop version: {version}")
    if run.attrs.get("crop_storage_mode") != "materialized":
        raise ValueError("Materialized pose crop must own its ROI pixels")
    if run.attrs.get("stage_selector_eligible") is not False:
        raise ValueError(
            "Materialized pose training source must be selector-ineligible"
        )
    if not is_run_complete(run, legacy_default=False):
        raise ValueError("Materialized pose crop source is incomplete")

    for name in (
        "roi_images",
        "keypoints_roi",
        "keypoint_visibility",
        *_REQUIRED_ROW_ARRAYS,
    ):
        if name not in run:
            raise ValueError(f"Materialized pose crop is missing {name}")
    roi = run["roi_images"]
    keypoints = run["keypoints_roi"]
    visibility = run["keypoint_visibility"]
    if roi.ndim != 3 or np.dtype(roi.dtype) != np.dtype(np.uint8):
        raise ValueError("Materialized pose roi_images must be (N,H,W) uint8")
    n, height, width = (int(v) for v in roi.shape)
    if n <= 0 or min(height, width) <= 0:
        raise ValueError("Materialized pose crop has an empty or invalid ROI shape")
    if (
        tuple(keypoints.shape)[:1] != (n,)
        or keypoints.ndim != 3
        or keypoints.shape[2] != 2
    ):
        raise ValueError("Materialized pose keypoints_roi must be (N,K,2)")
    if np.dtype(keypoints.dtype) not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("Materialized pose keypoints must be float32 or float64")
    k = int(keypoints.shape[1])
    if tuple(visibility.shape) != (n, k) or np.dtype(visibility.dtype) != np.dtype(
        np.uint8
    ):
        raise ValueError("Materialized pose visibility must be (N,K) uint8")
    for name, suffix in _REQUIRED_ROW_ARRAYS.items():
        if tuple(run[name].shape) != (n, *suffix):
            raise ValueError(f"Materialized pose {name} has the wrong row shape")

    schema = run.attrs.get("pose_schema")
    if not isinstance(schema, Mapping):
        raise ValueError("Materialized pose crop has no pose_schema")
    skeleton_id = str(schema.get("skeleton_id") or "").strip()
    labels_raw = schema.get("keypoint_labels")
    if not skeleton_id or not isinstance(labels_raw, (list, tuple)):
        raise ValueError(
            "Materialized pose schema needs skeleton ID and ordered labels"
        )
    labels = tuple(str(label).strip() for label in labels_raw)
    if len(labels) != k or not all(labels) or len(set(labels)) != k:
        raise ValueError(
            "Materialized pose keypoint labels disagree with keypoint shape"
        )
    bindings = run.attrs.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("Materialized pose crop has no source bindings")
    source_keypoint_run = bindings.get("source_keypoint_run") or (
        bindings.get("refined_keypoint_run") if version == 1 else None
    )
    if not bindings.get("source_crop_run") or not source_keypoint_run:
        raise ValueError("Materialized pose crop lacks bound source run IDs")
    if version >= 2 and not bindings.get("source_keypoint_group"):
        raise ValueError("Materialized pose v2+ source lacks keypoint group binding")
    if version == 3 and (
        not bindings.get("source_box_mode") or not bindings.get("source_detection_run")
    ):
        raise ValueError("Materialized pose v3 source lacks detection box binding")
    if tuple(int(v) for v in run.attrs.get("roi_size", ())) != (height, width):
        raise ValueError("Materialized pose ROI size declaration disagrees with pixels")
    image_shape = tuple(int(v) for v in bindings.get("source_images_shape", ()))
    if len(image_shape) != 3 or image_shape[1] < height or image_shape[2] < width:
        raise ValueError("Materialized pose crop has invalid source image shape")
    origins = np.asarray(run["roi_coordinates_full"][:])
    if (
        not np.issubdtype(origins.dtype, np.integer)
        or np.any(origins < 0)
        or np.any(origins[:, 0] + width > image_shape[2])
        or np.any(origins[:, 1] + height > image_shape[1])
    ):
        raise ValueError("Materialized pose crop origins escape the source image")

    points = np.asarray(keypoints[:])
    visible = np.asarray(visibility[:])
    if not np.isin(visible, (0, 2)).all():
        raise ValueError("Materialized pose visibility must contain only 0 or 2")
    valid_xy = np.isfinite(points).all(axis=2)
    inside = (
        (points[..., 0] >= 0)
        & (points[..., 0] < width)
        & (points[..., 1] >= 0)
        & (points[..., 1] < height)
    )
    if not np.all(valid_xy[visible == 2] & inside[visible == 2]):
        raise ValueError("Visible pose keypoints must be finite and inside the ROI")

    digest_names = {
        "roi_images": "pixel_sha256",
        "keypoints_roi": "keypoints_roi_sha256",
        "keypoint_visibility": "keypoint_visibility_sha256",
    }
    digests = {attr: str(run.attrs.get(attr) or "") for attr in digest_names.values()}
    if any(len(value) != 64 for value in digests.values()):
        raise ValueError("Materialized pose crop has incomplete content digests")
    if verify_digests:
        for array_name, attr_name in digest_names.items():
            if _sha256_array(run[array_name]) != digests[attr_name]:
                raise ValueError(f"Materialized pose crop {array_name} digest mismatch")
    lineage_arrays = {name: _sha256_array(run[name]) for name in _REQUIRED_ROW_ARRAYS}
    lineage_digest = hashlib.sha256(
        json.dumps(lineage_arrays, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return MaterializedPoseCropSource(
        schema_id=schema_id,
        schema_version=version,
        run_id=run_id,
        row_count=n,
        roi_shape=(height, width),
        keypoint_count=k,
        skeleton_id=skeleton_id,
        keypoint_labels=labels,
        source_bindings=dict(bindings),
        pixel_sha256=digests["pixel_sha256"],
        keypoints_roi_sha256=digests["keypoints_roi_sha256"],
        keypoint_visibility_sha256=digests["keypoint_visibility_sha256"],
        source_row_lineage_sha256=lineage_digest,
    )
