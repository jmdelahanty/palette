from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from .refined_detect_curation import (
    CURATED_REFINED_INSTANCES_REQUIRED_ARRAYS,
    CURATED_REFINED_SOURCE_DETECTIONS_REQUIRED_ARRAYS,
    REFINED_SOURCE_DETECTION_DECISION_CODE_MAP,
    REFINED_SOURCE_KIND_CODE_MAP,
)
from .type_conversions import normalize_attr


@dataclass(frozen=True)
class RefinedDetectIdentityIssue:
    severity: str
    code: str
    message: str
    path: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.path:
            payload["path"] = self.path
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def _node_path(parent: Any, child: str) -> str:
    base = normalize_attr(getattr(parent, "path", None))
    return f"{base}/{child}" if base else child


def _child(group: Any, name: str) -> Any:
    try:
        child = group.get(name)
    except Exception:
        child = None
    if child is not None:
        return child
    try:
        return group[name]
    except Exception:
        return None


def _array(group: Any, name: str, *, dtype: Any) -> Optional[np.ndarray]:
    node = _child(group, name)
    if node is None:
        return None
    try:
        return np.asarray(node[:], dtype=dtype)
    except Exception:
        return None


def _issue(
    issues: list[RefinedDetectIdentityIssue],
    severity: str,
    code: str,
    message: str,
    *,
    path: Optional[str] = None,
    **details: Any,
) -> None:
    issues.append(
        RefinedDetectIdentityIssue(
            severity=severity,
            code=code,
            message=message,
            path=path,
            details={key: value for key, value in details.items() if value is not None},
        )
    )


def _duplicate_values(values: np.ndarray) -> list[int]:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return []
    unique, counts = np.unique(arr, return_counts=True)
    return [int(value) for value, count in zip(unique.tolist(), counts.tolist()) if int(count) > 1]


def _validate_instances(
    refined_run: Any,
    issues: list[RefinedDetectIdentityIssue],
) -> tuple[Optional[Any], dict[str, np.ndarray]]:
    instances = _child(refined_run, "instances")
    if instances is None:
        _issue(
            issues,
            "error",
            "missing_instances",
            "Refined detect run is missing canonical instances/ surface.",
            path=_node_path(refined_run, "instances"),
        )
        return None, {}

    missing = [
        name
        for name in CURATED_REFINED_INSTANCES_REQUIRED_ARRAYS
        if _child(instances, name) is None
    ]
    for name in missing:
        _issue(
            issues,
            "error",
            "missing_instances_array",
            f"instances/{name} is required for sparse refined-detect identity.",
            path=_node_path(instances, name),
        )
    if missing:
        return instances, {}

    arrays = {
        "refined_row_ids": _array(instances, "refined_row_ids", dtype=np.int64),
        "frame_indices": _array(instances, "frame_indices", dtype=np.int32),
        "frame_offsets": _array(instances, "frame_offsets", dtype=np.int64),
        "bbox_img_xyxy": _array(instances, "bbox_img_xyxy", dtype=np.float64),
        "bbox_norm_coords": _array(instances, "bbox_norm_coords", dtype=np.float64),
        "source_kind_codes": _array(instances, "source_kind_codes", dtype=np.int8),
        "manual_edit_flags": _array(instances, "manual_edit_flags", dtype=bool),
        "source_detect_row_index": _array(instances, "source_detect_row_index", dtype=np.int32),
        "frame_counts": _array(instances, "frame_counts", dtype=np.int32),
    }
    unreadable = [name for name, arr in arrays.items() if arr is None]
    for name in unreadable:
        _issue(
            issues,
            "error",
            "unreadable_instances_array",
            f"instances/{name} could not be read.",
            path=_node_path(instances, name),
        )
    if unreadable:
        return instances, {}

    out = {name: np.asarray(arr) for name, arr in arrays.items() if arr is not None}
    row_count = int(out["frame_indices"].reshape(-1).shape[0])
    for name in (
        "refined_row_ids",
        "source_kind_codes",
        "manual_edit_flags",
        "source_detect_row_index",
    ):
        if int(out[name].reshape(-1).shape[0]) != row_count:
            _issue(
                issues,
                "error",
                "instances_length_mismatch",
                f"instances/{name} length does not match frame_indices.",
                path=_node_path(instances, name),
                expected=row_count,
                actual=int(out[name].reshape(-1).shape[0]),
            )
    for name in ("bbox_img_xyxy", "bbox_norm_coords"):
        if int(out[name].reshape(-1, 4).shape[0]) != row_count:
            _issue(
                issues,
                "error",
                "instances_length_mismatch",
                f"instances/{name} row count does not match frame_indices.",
                path=_node_path(instances, name),
                expected=row_count,
                actual=int(out[name].reshape(-1, 4).shape[0]),
            )

    refined_row_ids = out["refined_row_ids"].reshape(-1)
    frame_indices = out["frame_indices"].reshape(-1)
    source_kind_codes = out["source_kind_codes"].reshape(-1)
    source_detect_row_index = out["source_detect_row_index"].reshape(-1)
    frame_counts = out["frame_counts"].reshape(-1)
    frame_offsets = out["frame_offsets"].reshape(-1)

    if np.any(refined_row_ids < 0):
        _issue(
            issues,
            "error",
            "negative_refined_row_id",
            "Written instances/refined_row_ids must not contain temporary or negative IDs.",
            path=_node_path(instances, "refined_row_ids"),
            count=int(np.count_nonzero(refined_row_ids < 0)),
        )
    duplicate_row_ids = _duplicate_values(refined_row_ids)
    if duplicate_row_ids:
        _issue(
            issues,
            "error",
            "duplicate_refined_row_id",
            "instances/refined_row_ids must be unique within a refined run.",
            path=_node_path(instances, "refined_row_ids"),
            duplicate_values=duplicate_row_ids[:20],
        )
    if np.any(frame_indices < 0):
        _issue(
            issues,
            "error",
            "negative_frame_index",
            "instances/frame_indices must not contain negative frame IDs.",
            path=_node_path(instances, "frame_indices"),
            count=int(np.count_nonzero(frame_indices < 0)),
        )
    if (
        frame_counts.size
        and frame_indices.size
        and int(np.max(frame_indices)) >= int(frame_counts.shape[0])
    ):
        _issue(
            issues,
            "error",
            "frame_index_outside_frame_counts",
            "instances/frame_indices contains values beyond frame_counts length.",
            path=_node_path(instances, "frame_indices"),
            max_frame_index=int(np.max(frame_indices)),
            frame_counts_length=int(frame_counts.shape[0]),
        )

    if frame_offsets.shape[0] != frame_counts.shape[0] + 1:
        _issue(
            issues,
            "error",
            "frame_offsets_length_mismatch",
            "instances/frame_offsets length must equal len(frame_counts) + 1.",
            path=_node_path(instances, "frame_offsets"),
            expected=int(frame_counts.shape[0] + 1),
            actual=int(frame_offsets.shape[0]),
        )
    elif frame_offsets.size:
        expected_offsets = np.zeros(frame_counts.shape[0] + 1, dtype=np.int64)
        expected_offsets[1:] = np.cumsum(frame_counts, dtype=np.int64)
        if not np.array_equal(frame_offsets, expected_offsets):
            _issue(
                issues,
                "error",
                "frame_offsets_do_not_match_counts",
                "instances/frame_offsets must be the cumulative sum of frame_counts.",
                path=_node_path(instances, "frame_offsets"),
            )
        if int(frame_offsets[-1]) != row_count:
            _issue(
                issues,
                "error",
                "frame_offsets_row_count_mismatch",
                "instances/frame_offsets[-1] must equal the number of instance rows.",
                path=_node_path(instances, "frame_offsets"),
                expected=row_count,
                actual=int(frame_offsets[-1]),
            )

    if frame_counts.size:
        valid_frames = frame_indices[(frame_indices >= 0) & (frame_indices < frame_counts.shape[0])]
        expected_counts = np.bincount(
            valid_frames,
            minlength=int(frame_counts.shape[0]),
        ).astype(np.int32)
        if not np.array_equal(frame_counts, expected_counts):
            _issue(
                issues,
                "error",
                "frame_counts_do_not_match_rows",
                "instances/frame_counts must match frame_indices.",
                path=_node_path(instances, "frame_counts"),
            )

    if row_count:
        expected_order = np.lexsort((refined_row_ids, frame_indices))
        if not np.array_equal(expected_order, np.arange(row_count)):
            _issue(
                issues,
                "error",
                "instances_not_frame_sorted",
                "instances rows must be sorted by frame_indices then refined_row_ids "
                "for frame_offsets reads.",
                path=_node_path(instances, "frame_indices"),
            )

    raw_code = int(REFINED_SOURCE_KIND_CODE_MAP["raw_detect"])
    raw_missing_source = np.where(
        (source_kind_codes == raw_code) & (source_detect_row_index < 0)
    )[0]
    if raw_missing_source.size:
        _issue(
            issues,
            "error",
            "raw_instance_missing_source_row",
            "raw_detect instance rows must retain source_detect_row_index lineage.",
            path=_node_path(instances, "source_detect_row_index"),
            count=int(raw_missing_source.size),
        )

    duplicate_source_rows = _duplicate_values(source_detect_row_index[source_detect_row_index >= 0])
    if duplicate_source_rows:
        _issue(
            issues,
            "error",
            "duplicate_instance_source_detect_row_index",
            "A raw source detection row must not resolve to multiple instance rows.",
            path=_node_path(instances, "source_detect_row_index"),
            duplicate_values=duplicate_source_rows[:20],
        )

    manual_latest = normalize_attr(refined_run.attrs.get("manual_review_latest"))
    if manual_latest:
        _issue(
            issues,
            "warning",
            "legacy_manual_review_pointer_present",
            "manual_review_latest is a legacy compatibility pointer; current "
            "readers should prefer instances/.",
            path=getattr(refined_run, "path", None),
            manual_review_latest=manual_latest,
        )
    primary_surface = normalize_attr(refined_run.attrs.get("curated_primary_surface"))
    if primary_surface and primary_surface != "instances":
        _issue(
            issues,
            "warning",
            "unexpected_curated_primary_surface",
            "Sparse refined-detect runs should advertise curated_primary_surface='instances'.",
            path=getattr(refined_run, "path", None),
            curated_primary_surface=primary_surface,
        )
    row_identity_policy = normalize_attr(refined_run.attrs.get("row_identity_policy"))
    if row_identity_policy and row_identity_policy != "stable_sparse_refined_row_id":
        _issue(
            issues,
            "warning",
            "unexpected_row_identity_policy",
            "Sparse refined-detect runs should advertise stable_sparse_refined_row_id.",
            path=getattr(refined_run, "path", None),
            row_identity_policy=row_identity_policy,
        )

    return instances, out


def _validate_source_detections(
    refined_run: Any,
    instance_arrays: Mapping[str, np.ndarray],
    issues: list[RefinedDetectIdentityIssue],
) -> None:
    source = _child(refined_run, "source_detections")
    if source is None:
        _issue(
            issues,
            "error",
            "missing_source_detections",
            "Current sparse refined-detect runs need source_detections/ to audit raw-row lineage.",
            path=_node_path(refined_run, "source_detections"),
        )
        return

    missing = [
        name
        for name in CURATED_REFINED_SOURCE_DETECTIONS_REQUIRED_ARRAYS
        if _child(source, name) is None
    ]
    for name in missing:
        _issue(
            issues,
            "error",
            "missing_source_detections_array",
            f"source_detections/{name} is required for raw-row lineage validation.",
            path=_node_path(source, name),
        )
    if missing:
        return

    arrays = {
        "source_detect_row_index": _array(source, "source_detect_row_index", dtype=np.int32),
        "frame_indices": _array(source, "frame_indices", dtype=np.int32),
        "bbox_img_xyxy": _array(source, "bbox_img_xyxy", dtype=np.float64),
        "bbox_norm_coords": _array(source, "bbox_norm_coords", dtype=np.float64),
        "decision_codes": _array(source, "decision_codes", dtype=np.int8),
        "resolved_refined_row_id": _array(source, "resolved_refined_row_id", dtype=np.int64),
    }
    unreadable = [name for name, arr in arrays.items() if arr is None]
    for name in unreadable:
        _issue(
            issues,
            "error",
            "unreadable_source_detections_array",
            f"source_detections/{name} could not be read.",
            path=_node_path(source, name),
        )
    if unreadable:
        return

    source_arrays = {name: np.asarray(arr) for name, arr in arrays.items() if arr is not None}
    source_rows = source_arrays["source_detect_row_index"].reshape(-1)
    row_count = int(source_rows.shape[0])
    for name in ("frame_indices", "decision_codes", "resolved_refined_row_id"):
        if int(source_arrays[name].reshape(-1).shape[0]) != row_count:
            _issue(
                issues,
                "error",
                "source_detections_length_mismatch",
                f"source_detections/{name} length does not match source_detect_row_index.",
                path=_node_path(source, name),
                expected=row_count,
                actual=int(source_arrays[name].reshape(-1).shape[0]),
            )
    for name in ("bbox_img_xyxy", "bbox_norm_coords"):
        if int(source_arrays[name].reshape(-1, 4).shape[0]) != row_count:
            _issue(
                issues,
                "error",
                "source_detections_length_mismatch",
                f"source_detections/{name} row count does not match source_detect_row_index.",
                path=_node_path(source, name),
                expected=row_count,
                actual=int(source_arrays[name].reshape(-1, 4).shape[0]),
            )

    if np.any(source_rows < 0):
        _issue(
            issues,
            "error",
            "negative_source_detect_row_index",
            "source_detections/source_detect_row_index must not contain negative raw-row IDs.",
            path=_node_path(source, "source_detect_row_index"),
            count=int(np.count_nonzero(source_rows < 0)),
        )
    duplicate_source_rows = _duplicate_values(source_rows)
    if duplicate_source_rows:
        _issue(
            issues,
            "error",
            "duplicate_source_detect_row_index",
            "source_detections/source_detect_row_index must be unique.",
            path=_node_path(source, "source_detect_row_index"),
            duplicate_values=duplicate_source_rows[:20],
        )

    instance_row_ids = np.asarray(instance_arrays["refined_row_ids"], dtype=np.int64).reshape(-1)
    instance_source_rows = np.asarray(
        instance_arrays["source_detect_row_index"],
        dtype=np.int32,
    ).reshape(-1)
    instance_ids = {int(value) for value in instance_row_ids.tolist()}
    accepted_code = int(REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["accepted"])
    decision_codes = source_arrays["decision_codes"].reshape(-1)
    resolved_ids = source_arrays["resolved_refined_row_id"].reshape(-1)
    source_lookup = {int(row): idx for idx, row in enumerate(source_rows.tolist())}

    accepted_mask = decision_codes == accepted_code
    accepted_bad_ids = [
        int(value)
        for value in resolved_ids[accepted_mask].tolist()
        if int(value) < 0 or int(value) not in instance_ids
    ]
    if accepted_bad_ids:
        _issue(
            issues,
            "error",
            "accepted_source_resolves_missing_instance",
            "Accepted source rows must resolve to an existing instances/refined_row_ids value.",
            path=_node_path(source, "resolved_refined_row_id"),
            resolved_refined_row_ids=accepted_bad_ids[:20],
        )

    nonaccepted_resolved = np.where((decision_codes != accepted_code) & (resolved_ids >= 0))[0]
    if nonaccepted_resolved.size:
        _issue(
            issues,
            "warning",
            "nonaccepted_source_has_resolved_row_id",
            "Non-accepted source rows usually should not point at a resolved refined row.",
            path=_node_path(source, "resolved_refined_row_id"),
            count=int(nonaccepted_resolved.size),
        )

    for refined_row_id, source_row in zip(instance_row_ids.tolist(), instance_source_rows.tolist()):
        source_row_int = int(source_row)
        if source_row_int < 0:
            continue
        source_idx = source_lookup.get(source_row_int)
        if source_idx is None:
            _issue(
                issues,
                "error",
                "instance_source_row_missing_from_source_detections",
                "Instance row has source_detect_row_index that is absent from source_detections/.",
                path=_node_path(source, "source_detect_row_index"),
                refined_row_id=int(refined_row_id),
                source_detect_row_index=source_row_int,
            )
            continue
        if int(decision_codes[source_idx]) != accepted_code:
            _issue(
                issues,
                "error",
                "instance_source_row_not_accepted",
                "Instance row with raw-source lineage must point at an accepted "
                "source_detections row.",
                path=_node_path(source, "decision_codes"),
                refined_row_id=int(refined_row_id),
                source_detect_row_index=source_row_int,
                decision_code=int(decision_codes[source_idx]),
            )
        if int(resolved_ids[source_idx]) != int(refined_row_id):
            _issue(
                issues,
                "error",
                "source_link_mismatch",
                "source_detections/resolved_refined_row_id does not match the instance row ID.",
                path=_node_path(source, "resolved_refined_row_id"),
                refined_row_id=int(refined_row_id),
                source_detect_row_index=source_row_int,
                resolved_refined_row_id=int(resolved_ids[source_idx]),
            )


def validate_refined_detect_identity(refined_run: Any) -> list[RefinedDetectIdentityIssue]:
    """Validate sparse refined-detect row identity and raw-source linkage.

    This validator is intentionally stricter than generic read compatibility.
    It answers whether a current `refined_detect_runs/<run>/instances` surface is
    safe to use for row-local downstream stale handling.
    """

    issues: list[RefinedDetectIdentityIssue] = []
    _instances, instance_arrays = _validate_instances(refined_run, issues)
    if instance_arrays:
        _validate_source_detections(refined_run, instance_arrays, issues)
    return issues


def collect_refined_detect_identity_validation(refined_run: Any) -> dict[str, Any]:
    issues = validate_refined_detect_identity(refined_run)
    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    return {
        "ok": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": [issue.as_dict() for issue in issues],
    }
