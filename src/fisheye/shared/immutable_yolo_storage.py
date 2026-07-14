"""Fail-closed completion contract for immutable serial YOLO outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import math
from typing import Any, Literal

import numpy as np


IMMUTABLE_YOLO_STORAGE_SCHEMA = "palette.immutable_yolo_storage_completion.v1"
IMMUTABLE_YOLO_STORAGE_ATTR = "immutable_yolo_storage_validation"

_NUMERIC_KINDS = frozenset("biufc")
_FRAME_ARRAYS = {
    "detect": frozenset({"frame_counts", "n_detections"}),
    "keypoints": frozenset({"frame_counts", "n_keypoints", "n_rois"}),
}
_REQUIRED_ARRAYS = {
    "detect": frozenset(
        {
            "frame_indices",
            "bbox_norm_coords",
            "scores",
            "class_ids",
            "instance_key",
            "frame_counts",
            "n_detections",
        }
    ),
    "keypoints": frozenset(
        {
            "keypoints_roi",
            "keypoints_img",
            "keypoints_norm",
            "confidence",
            "keypoint_confidences",
            "detection_success",
            "pose_bbox_xyxy_roi",
            "heading",
            "heading_finite",
            "heading_usable",
            "effective_threshold",
            "effective_se2_radius",
            "detection_source",
            "frame_indices",
            "source_crop_row_ids",
            "instance_key",
            "frame_counts",
            "n_keypoints",
            "n_rois",
        }
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _effective_shard_rows(requested: int, inner_rows: int) -> int:
    if requested <= 0 or inner_rows <= 0:
        raise ValueError("Requested shard rows and inner chunk rows must be positive.")
    return int(math.ceil(int(requested) / int(inner_rows)) * int(inner_rows))


def _dtype(array: Any) -> np.dtype[Any] | None:
    try:
        return np.dtype(array.dtype)
    except (TypeError, ValueError):
        return None


def _shards(array: Any) -> tuple[int, ...] | None:
    value = getattr(array, "shards", None)
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _chunks(array: Any) -> tuple[int, ...] | None:
    value = getattr(array, "chunks", None)
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _direct_arrays(run_group: Any) -> dict[str, Any]:
    arrays = getattr(run_group, "arrays", None)
    if not callable(arrays):
        raise TypeError("Run group does not expose direct Zarr arrays().")
    return {str(name): array for name, array in arrays()}


def _shape(array: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in array.shape)


def _summary_errors(
    summary: Any,
    *,
    stage: str,
    row_count: int,
) -> list[str]:
    if not isinstance(summary, Mapping):
        return [f"{stage} shard-write summary is missing or not a mapping"]
    errors: list[str] = []
    if str(summary.get("status") or "") != "complete":
        errors.append(f"{stage} shard-write summary status is not complete")
    if summary.get("exact_match") is not True:
        errors.append(f"{stage} shard-write summary exact_match is not true")
    source_hashes = summary.get("source_sha256_by_array")
    destination_hashes = summary.get("destination_sha256_by_array")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        errors.append(f"{stage} shard-write summary has no source hashes")
    elif dict(source_hashes) != dict(destination_hashes or {}):
        errors.append(f"{stage} shard-write source and destination hashes differ")
    count_name = "detection_row_count" if stage == "detect" else "row_count"
    try:
        summary_rows = int(summary.get(count_name))
    except (TypeError, ValueError):
        summary_rows = None
    if summary_rows != int(row_count):
        errors.append(
            f"{stage} shard-write summary {count_name}={summary_rows!r}, expected {row_count}"
        )
    return errors


def _record_failure(
    run_group: Any,
    *,
    stage: str,
    layout: str,
    policy: str,
    errors: Sequence[str],
) -> None:
    run_group.attrs[IMMUTABLE_YOLO_STORAGE_ATTR] = {
        "schema_id": IMMUTABLE_YOLO_STORAGE_SCHEMA,
        "status": "error",
        "stage": stage,
        "storage_layout": layout,
        "storage_policy": policy,
        "validated_at_utc": _utc_now(),
        "errors": list(errors),
    }


def validate_immutable_yolo_storage(
    run_group: Any,
    *,
    stage: Literal["detect", "keypoints"],
    row_shard_rows: int | None,
    frame_shard_rows: int | None,
) -> dict[str, Any]:
    """Validate the physical raw-YOLO storage contract before completion.

    Passing ``row_shard_rows=None`` is the explicit ordinary-chunk compatibility
    contract. Any positive row target requires every fixed-width numeric/bool
    direct array to use indexed shards aligned to its retained inner chunk grid.
    """

    if stage not in _REQUIRED_ARRAYS:
        raise ValueError(f"Unsupported immutable YOLO stage: {stage!r}.")
    prefix = "detect" if stage == "detect" else "keypoint"
    expected_layout = (
        "indexed_sharding_v1" if row_shard_rows is not None else "regular_chunks_v1"
    )
    expected_policy = (
        "default_indexed_sharding_v1"
        if row_shard_rows is not None
        else "explicit_regular_chunks_override"
    )
    layout = str(run_group.attrs.get(f"{prefix}_storage_layout") or "")
    policy = str(run_group.attrs.get(f"{prefix}_storage_policy") or "")
    errors: list[str] = []

    if layout != expected_layout:
        errors.append(
            f"declared {prefix}_storage_layout={layout!r}, expected {expected_layout!r}"
        )
    if policy != expected_policy:
        errors.append(
            f"declared {prefix}_storage_policy={policy!r}, expected {expected_policy!r}"
        )

    row_domain = "row" if stage == "detect" else "roi"
    declared_row_rows = run_group.attrs.get(f"{prefix}_{row_domain}_shard_rows")
    declared_frame_rows = run_group.attrs.get(f"{prefix}_frame_shard_rows")
    expected_declared_row = int(row_shard_rows) if row_shard_rows is not None else None
    expected_declared_frame = (
        int(frame_shard_rows)
        if row_shard_rows is not None and frame_shard_rows is not None
        else None
    )
    if declared_row_rows != expected_declared_row:
        errors.append(
            f"declared row shard rows={declared_row_rows!r}, expected {expected_declared_row!r}"
        )
    if declared_frame_rows != expected_declared_frame:
        errors.append(
            f"declared frame shard rows={declared_frame_rows!r}, expected {expected_declared_frame!r}"
        )
    if row_shard_rows is not None and (
        int(row_shard_rows) <= 0
        or frame_shard_rows is None
        or int(frame_shard_rows) <= 0
    ):
        errors.append("indexed sharding requires positive row and frame shard targets")

    try:
        arrays = _direct_arrays(run_group)
    except Exception as exc:
        errors.append(f"could not enumerate direct arrays: {exc}")
        arrays = {}
    missing = sorted(_REQUIRED_ARRAYS[stage] - set(arrays))
    if missing:
        errors.append(f"missing required arrays: {missing}")

    row_anchor = "frame_indices" if stage == "detect" else "keypoints_roi"
    row_count = int(arrays[row_anchor].shape[0]) if row_anchor in arrays else 0
    frame_anchor = "n_detections" if stage == "detect" else "frame_counts"
    frame_count = int(arrays[frame_anchor].shape[0]) if frame_anchor in arrays else 0

    checked: list[dict[str, Any]] = []
    for name, array in sorted(arrays.items()):
        shape = _shape(array)
        dtype = _dtype(array)
        if not shape or dtype is None or dtype.kind not in _NUMERIC_KINDS:
            continue
        domain = "frame" if name in _FRAME_ARRAYS[stage] else "row"
        expected_count = frame_count if domain == "frame" else row_count
        if shape[0] != expected_count:
            errors.append(
                f"{name} has {shape[0]} {domain} rows, expected {expected_count}"
            )
        chunks = _chunks(array)
        actual_shards = _shards(array)
        expected_shards: tuple[int, ...] | None = None
        if row_shard_rows is not None:
            if chunks is None:
                errors.append(f"{name} has no inner chunk contract")
            else:
                requested = (
                    int(frame_shard_rows)
                    if domain == "frame"
                    else int(row_shard_rows)
                )
                expected_shards = (
                    _effective_shard_rows(requested, chunks[0]),
                    *chunks[1:],
                )
                if actual_shards != expected_shards:
                    errors.append(
                        f"{name} shards={actual_shards!r}, expected {expected_shards!r}"
                    )
        elif actual_shards is not None:
            errors.append(f"{name} is sharded under the explicit regular-chunk contract")
        checked.append(
            {
                "name": name,
                "domain": domain,
                "shape": list(shape),
                "dtype": str(dtype),
                "chunks": list(chunks) if chunks is not None else None,
                "shards": list(actual_shards) if actual_shards is not None else None,
                "expected_shards": list(expected_shards) if expected_shards is not None else None,
            }
        )

    identity = arrays.get("instance_key")
    identity_unique = False
    if identity is not None:
        identity_shape = _shape(identity)
        identity_dtype = _dtype(identity)
        if identity_shape != (row_count,):
            errors.append(f"instance_key shape={identity_shape}, expected {(row_count,)}")
        if identity_dtype != np.dtype(np.uint64):
            errors.append(f"instance_key dtype={identity_dtype}, expected uint64")
        try:
            values = np.asarray(identity[:], dtype=np.uint64).reshape(-1)
            identity_unique = int(np.unique(values).shape[0]) == int(values.shape[0])
            if not identity_unique:
                errors.append("instance_key contains duplicate values")
        except Exception as exc:
            errors.append(f"could not reread instance_key: {exc}")

    if "frame_counts" in arrays:
        try:
            counts = np.asarray(arrays["frame_counts"][:], dtype=np.int64).reshape(-1)
            if int(counts.sum(dtype=np.int64)) != row_count:
                errors.append(
                    f"frame_counts sums to {int(counts.sum(dtype=np.int64))}, expected {row_count}"
                )
            if stage == "detect" and "n_detections" in arrays:
                if not np.array_equal(
                    counts,
                    np.asarray(arrays["n_detections"][:], dtype=np.int64),
                ):
                    errors.append("frame_counts and n_detections differ")
            if stage == "keypoints" and "n_rois" in arrays:
                if not np.array_equal(
                    counts,
                    np.asarray(arrays["n_rois"][:], dtype=np.int64),
                ):
                    errors.append("frame_counts and n_rois differ")
        except Exception as exc:
            errors.append(f"could not validate frame counts: {exc}")

    summary_name = f"{prefix}_shard_write"
    if row_shard_rows is not None:
        errors.extend(
            _summary_errors(
                run_group.attrs.get(summary_name),
                stage=stage,
                row_count=row_count,
            )
        )
    elif run_group.attrs.get(summary_name) not in (None, {}):
        errors.append(f"{summary_name} must be empty under the explicit regular-chunk contract")

    if errors:
        _record_failure(
            run_group,
            stage=stage,
            layout=layout,
            policy=policy,
            errors=errors,
        )
        raise RuntimeError(
            f"Refusing to complete immutable YOLO {stage} run: " + "; ".join(errors)
        )

    report = {
        "schema_id": IMMUTABLE_YOLO_STORAGE_SCHEMA,
        "status": "ok",
        "stage": stage,
        "storage_layout": layout,
        "storage_policy": policy,
        "row_count": row_count,
        "frame_count": frame_count,
        "row_shard_rows_requested": expected_declared_row,
        "frame_shard_rows_requested": expected_declared_frame,
        "eligible_arrays_checked": len(checked),
        "arrays": checked,
        "instance_key_present": identity is not None,
        "instance_key_unique": identity_unique,
        "validated_at_utc": _utc_now(),
        "errors": [],
    }
    run_group.attrs[IMMUTABLE_YOLO_STORAGE_ATTR] = report
    return report
