"""Read-only transition from current sparse refined detections to v1 arrays.

The adapter never mutates its source.  It makes legacy/current implicit fields
explicit, records every representation conversion, and validates the complete
28-array full-acquisition contract before a shadow publisher may write it.
Clipped recording snapshots remain a separate transition because they require
an external finalized collection/media binding.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.shared.instance_keys import mint_manual_curation_instance_keys
from fisheye.shared.zarr.detection_schema import derive_canonical_detection_geometry
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_KIND_CODE_MAP,
    RefinedDetectionDimensions,
)


TRANSITION_REPORT_SCHEMA_ID = "palette.refined_detection.transition_report"
TRANSITION_REPORT_SCHEMA_VERSION = 1

_NO_REASON_LABELS = frozenset({"", "none", "clean", "present", "accepted"})
_REASON_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class RefinedDetectionTransitionResult:
    """Validated target arrays plus a JSON-safe conversion report."""

    dimensions: RefinedDetectionDimensions
    arrays: Mapping[str, np.ndarray]
    instance_reason_codes: Mapping[int, str]
    source_reason_codes: Mapping[int, str]
    report: Mapping[str, object]


class RefinedDetectionTransitionError(ValueError):
    """Raised when current data cannot produce a contract-valid v1 snapshot."""

    def __init__(self, report: Mapping[str, object]) -> None:
        self.report = dict(report)
        blockers = self.report.get("blockers", [])
        super().__init__(
            "Refined-detection transition is blocked: "
            + "; ".join(str(item) for item in blockers)
        )


def _child(group: Any, name: str) -> Any:
    try:
        value = group.get(name)
    except AttributeError:
        value = group[name] if name in group else None
    if value is None:
        raise ValueError(f"Current refined run is missing subgroup {name!r}.")
    return value


def _array(group: Any, *names: str) -> tuple[np.ndarray | None, str | None]:
    for name in names:
        try:
            value = group.get(name)
        except AttributeError:
            value = group[name] if name in group else None
        if value is None:
            continue
        try:
            return np.asarray(value[...]), name
        except (IndexError, TypeError):
            return np.asarray(value), name
    return None, None


def _required_array(group: Any, *names: str) -> tuple[np.ndarray, str]:
    values, selected = _array(group, *names)
    if values is None or selected is None:
        raise ValueError(f"Current refined table requires one of {names!r}.")
    return values, selected


def _reason_labels(group: Any, row_count: int) -> np.ndarray:
    labels = read_reason_labels(group)
    if labels is None:
        return np.full(row_count, "", dtype=object)
    values = np.asarray(labels, dtype=object).reshape(-1)
    if values.shape != (row_count,):
        raise ValueError("Current reason labels do not match their table row count.")
    return values


def _normalize_reason_label(value: object) -> str | None:
    label = str(value).strip().lower()
    if label in _NO_REASON_LABELS:
        return None
    if not _REASON_LABEL_RE.fullmatch(label):
        raise ValueError(
            f"Reason label {value!r} is not a lowercase snake-case identifier."
        )
    return label


def _encode_reason_labels(
    values: np.ndarray,
) -> tuple[np.ndarray, dict[int, str]]:
    normalized = [_normalize_reason_label(value) for value in values.tolist()]
    labels = sorted({label for label in normalized if label is not None})
    if len(labels) > int(np.iinfo(np.uint16).max):
        raise ValueError("Reason registry exceeds the canonical uint16 domain.")
    code_by_label = {label: index + 1 for index, label in enumerate(labels)}
    codes = np.asarray(
        [0 if label is None else code_by_label[label] for label in normalized],
        dtype=np.uint16,
    )
    registry = {0: "none", **{code: label for label, code in code_by_label.items()}}
    return codes, registry


def _offsets(frame_indices: np.ndarray, *, n_frames: int) -> np.ndarray:
    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if frames.size and (
        int(np.min(frames)) < 0 or int(np.max(frames)) >= int(n_frames)
    ):
        raise ValueError("Frame indices fall outside the declared frame domain.")
    if frames.size > 1 and np.any(np.diff(frames) < 0):
        raise ValueError("Rows must be frame sorted before building CSR offsets.")
    counts = np.bincount(frames, minlength=int(n_frames))
    offsets = np.zeros(int(n_frames) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets


def _source_fallback_table(source_detect_group: Any | None) -> Any | None:
    if source_detect_group is None:
        return None
    try:
        instances = source_detect_group.get("instances")
    except AttributeError:
        instances = None
    return instances if instances is not None else source_detect_group


def _source_or_fallback(
    source: Any,
    fallback: Any | None,
    *names: str,
) -> tuple[np.ndarray | None, str | None]:
    values, selected = _array(source, *names)
    if values is not None:
        return values, f"source_detections/{selected}"
    if fallback is not None:
        values, selected = _array(fallback, *names)
        if values is not None:
            return values, f"source_detect/{selected}"
    return None, None


def _row_vector(
    values: np.ndarray,
    *,
    row_count: int,
    dtype: Any,
    label: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).reshape(-1)
    if result.shape != (row_count,):
        raise ValueError(f"{label} does not match its table row count.")
    return result


def _bbox_matrix(
    values: np.ndarray,
    *,
    row_count: int,
    label: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (row_count, 4):
        raise ValueError(f"{label} must have shape ({row_count}, 4).")
    return np.array(result, copy=True, order="C")


def _mapping(
    target: str,
    source: str,
    operation: str,
) -> dict[str, str]:
    return {"target": target, "source": source, "operation": operation}


def build_refined_detection_transition(
    refined_run: Any,
    *,
    n_frames: int,
    source_width: int,
    source_height: int,
    recording_identity: str,
    source_detect_group: Any | None = None,
    allow_manual_score_reset: bool = False,
) -> RefinedDetectionTransitionResult:
    """Convert a current full-acquisition sparse run to exact v1 arrays.

    ``allow_manual_score_reset`` is the only lossy opt-in. Current manual rows
    with nonzero legacy confidence have no valid v1 model-score meaning; when
    explicitly allowed their score becomes exact zero with ``score_valid=false``.
    """

    dimensions_input = {
        "n_frames": int(n_frames),
        "source_width": int(source_width),
        "source_height": int(source_height),
    }
    mappings: list[dict[str, str]] = []
    blockers: list[str] = []
    lossy: list[dict[str, object]] = []
    excluded: list[str] = []
    try:
        if not str(recording_identity).strip():
            raise ValueError("recording_identity cannot be empty.")
        instances = _child(refined_run, "instances")
        source = _child(refined_run, "source_detections")
        fallback = _source_fallback_table(source_detect_group)

        inst_frames_raw, inst_frames_name = _required_array(instances, "frame_indices")
        inst_frames = np.asarray(inst_frames_raw, dtype=np.int32).reshape(-1)
        n_instances = int(inst_frames.shape[0])
        refined_ids_raw, refined_ids_name = _required_array(
            instances,
            "refined_row_ids",
        )
        refined_ids = _row_vector(
            refined_ids_raw,
            row_count=n_instances,
            dtype=np.int64,
            label="instances/refined_row_ids",
        )
        order = (
            np.lexsort((refined_ids, inst_frames))
            if n_instances
            else np.empty((0,), dtype=np.int64)
        )
        inst_frames = inst_frames[order]
        refined_ids = refined_ids[order]

        inst_bbox_raw, inst_bbox_name = _required_array(
            instances,
            "bbox_norm_coords",
        )
        inst_bbox = _bbox_matrix(
            inst_bbox_raw,
            row_count=n_instances,
            label="instances/bbox_norm_coords",
        )[order]
        kind_raw, kind_name = _required_array(instances, "source_kind_codes")
        kind = _row_vector(
            kind_raw,
            row_count=n_instances,
            dtype=np.uint8,
            label="instances/source_kind_codes",
        )[order]
        source_rows_raw, source_rows_name = _required_array(
            instances,
            "source_detect_row_index",
        )
        source_rows = _row_vector(
            source_rows_raw,
            row_count=n_instances,
            dtype=np.int64,
            label="instances/source_detect_row_index",
        )[order]
        manual_flags_raw, manual_flags_name = _required_array(
            instances,
            "manual_edit_flags",
        )
        manual_flags = _row_vector(
            manual_flags_raw,
            row_count=n_instances,
            dtype=bool,
            label="instances/manual_edit_flags",
        )[order]
        inst_class_raw, inst_class_name = _required_array(instances, "class_ids")
        inst_class = _row_vector(
            inst_class_raw,
            row_count=n_instances,
            dtype=np.int32,
            label="instances/class_ids",
        )[order]

        allowed_kinds = set(SOURCE_KIND_CODE_MAP.values())
        unknown_kinds = sorted(
            set(int(value) for value in kind.tolist()) - allowed_kinds
        )
        if unknown_kinds:
            raise ValueError(
                "Current refined rows contain unsupported source kind codes "
                f"{unknown_kinds}; interpolation is outside refined v1."
            )
        raw_mask = kind == SOURCE_KIND_CODE_MAP["raw_detect"]
        manual_mask = kind == SOURCE_KIND_CODE_MAP["manual"]
        if np.any(~manual_flags[manual_mask]):
            raise ValueError(
                "Current manual rows must already be marked manually edited."
            )

        source_rows_identity_raw, source_rows_identity_name = _required_array(
            source,
            "source_detect_row_index",
        )
        source_row_ids = np.asarray(source_rows_identity_raw, dtype=np.int64).reshape(
            -1
        )
        n_source = int(source_row_ids.shape[0])
        if not np.array_equal(source_row_ids, np.arange(n_source, dtype=np.int64)):
            raise ValueError(
                "Source audit row identities are not exact contiguous rows."
            )
        source_frames_raw, source_frames_name = _required_array(source, "frame_indices")
        source_frames = _row_vector(
            source_frames_raw,
            row_count=n_source,
            dtype=np.int32,
            label="source_detections/frame_indices",
        )
        if source_frames.size > 1 and np.any(np.diff(source_frames) < 0):
            raise ValueError(
                "Source audit rows are not frame sorted within contiguous source identity."
            )
        source_bbox_raw, source_bbox_name = _required_array(
            source,
            "bbox_norm_coords",
        )
        source_bbox = _bbox_matrix(
            source_bbox_raw,
            row_count=n_source,
            label="source_detections/bbox_norm_coords",
        )
        decisions_raw, decisions_name = _required_array(source, "decision_codes")
        decisions = _row_vector(
            decisions_raw,
            row_count=n_source,
            dtype=np.uint8,
            label="source_detections/decision_codes",
        )
        resolved_raw, resolved_name = _required_array(
            source,
            "resolved_refined_row_id",
        )
        resolved = _row_vector(
            resolved_raw,
            row_count=n_source,
            dtype=np.int64,
            label="source_detections/resolved_refined_row_id",
        )

        source_scores_raw, source_scores_name = _source_or_fallback(
            source,
            fallback,
            "scores",
            "confidence_scores",
        )
        source_classes_raw, source_classes_name = _source_or_fallback(
            source,
            fallback,
            "class_ids",
        )
        source_keys_raw, source_keys_name = _source_or_fallback(
            source,
            fallback,
            "instance_key",
        )
        if source_scores_raw is None:
            raise ValueError(
                "Source audit table lacks model scores and no fallback supplies them."
            )
        if source_classes_raw is None:
            raise ValueError(
                "Source audit table lacks class IDs and no fallback supplies them."
            )
        if source_keys_raw is None:
            raise ValueError(
                "Source audit table lacks durable keys and no fallback supplies them."
            )
        source_scores = _row_vector(
            source_scores_raw,
            row_count=n_source,
            dtype=np.float32,
            label="source_detections/scores",
        )
        source_classes = _row_vector(
            source_classes_raw,
            row_count=n_source,
            dtype=np.int32,
            label="source_detections/class_ids",
        )
        source_keys = _row_vector(
            source_keys_raw,
            row_count=n_source,
            dtype=np.uint64,
            label="source_detections/instance_key",
        )

        inst_scores_raw, inst_scores_name = _array(
            instances, "scores", "confidence_scores"
        )
        if inst_scores_raw is None:
            inst_scores = np.zeros(n_instances, dtype=np.float32)
            if np.any(raw_mask):
                inst_scores[raw_mask] = source_scores[source_rows[raw_mask]]
            inst_scores_name = "derived_from_source_or_manual_zero"
        else:
            inst_scores = _row_vector(
                inst_scores_raw,
                row_count=n_instances,
                dtype=np.float32,
                label="instances/scores",
            )[order]
            if np.any(raw_mask) and not np.array_equal(
                inst_scores[raw_mask],
                source_scores[source_rows[raw_mask]],
            ):
                raise ValueError(
                    "Raw-backed current scores differ from source-audit scores."
                )
        nonzero_manual = np.flatnonzero(manual_mask & (inst_scores != np.float32(0.0)))
        if nonzero_manual.size:
            detail = {
                "field": "instances/scores",
                "operation": "reset_manual_score_to_zero",
                "row_count": int(nonzero_manual.size),
            }
            if not allow_manual_score_reset:
                raise ValueError(
                    "Manual rows carry nonzero legacy confidence; explicit "
                    "allow_manual_score_reset is required."
                )
            lossy.append(detail)
        inst_scores[manual_mask] = np.float32(0.0)
        score_valid = raw_mask.astype(bool, copy=True)

        inst_keys_raw, inst_keys_name = _array(instances, "instance_key")
        if inst_keys_raw is None:
            inst_keys = np.zeros(n_instances, dtype=np.uint64)
            if np.any(raw_mask):
                inst_keys[raw_mask] = source_keys[source_rows[raw_mask]]
            if np.any(manual_mask):
                inst_keys[manual_mask] = mint_manual_curation_instance_keys(
                    recording_identity=str(recording_identity),
                    refined_row_ids=refined_ids[manual_mask],
                    frame_indices=inst_frames[manual_mask],
                    bbox_norm_coords=inst_bbox[manual_mask],
                    class_ids=inst_class[manual_mask],
                )
            inst_keys_name = "derived_from_source_or_manual_allocator"
        else:
            inst_keys = _row_vector(
                inst_keys_raw,
                row_count=n_instances,
                dtype=np.uint64,
                label="instances/instance_key",
            )[order]
            if np.any(manual_mask):
                expected_manual_keys = mint_manual_curation_instance_keys(
                    recording_identity=str(recording_identity),
                    refined_row_ids=refined_ids[manual_mask],
                    frame_indices=inst_frames[manual_mask],
                    bbox_norm_coords=inst_bbox[manual_mask],
                    class_ids=inst_class[manual_mask],
                )
                if not np.array_equal(
                    inst_keys[manual_mask],
                    expected_manual_keys,
                ):
                    raise ValueError(
                        "Current manual instance_key values do not match the "
                        "frozen root-snapshot allocator; transition requires "
                        "explicit parent identity evidence instead of reminting."
                    )

        inst_reason_labels = _reason_labels(instances, n_instances)[order]
        source_reason_labels = _reason_labels(source, n_source)
        inst_reason_codes, inst_reason_registry = _encode_reason_labels(
            inst_reason_labels
        )
        source_reason_codes, source_reason_registry = _encode_reason_labels(
            source_reason_labels
        )

        inst_bbox_img, inst_centers = derive_canonical_detection_geometry(
            inst_bbox,
            source_width=int(source_width),
            source_height=int(source_height),
        )
        source_bbox_img, source_centers = derive_canonical_detection_geometry(
            source_bbox,
            source_width=int(source_width),
            source_height=int(source_height),
        )
        dimensions = RefinedDetectionDimensions(
            n_frames=int(n_frames),
            n_instances=n_instances,
            n_source_detections=n_source,
            source_width=int(source_width),
            source_height=int(source_height),
        )
        arrays = {
            "instances/frame_indices": inst_frames,
            "instances/source_acquisition_frame_index": inst_frames.astype(np.int64),
            "instances/instance_key": inst_keys,
            "instances/refined_row_ids": refined_ids,
            "instances/bbox_norm_coords": inst_bbox,
            "instances/bbox_img_xyxy": inst_bbox_img,
            "instances/centers_img_xy": inst_centers,
            "instances/scores": inst_scores,
            "instances/score_valid": score_valid,
            "instances/class_ids": inst_class,
            "instances/source_kind_codes": kind,
            "instances/manual_edit_flags": manual_flags,
            "instances/source_detect_row_index": source_rows,
            "instances/reason_codes": inst_reason_codes,
            "instances/frame_row_offsets": _offsets(
                inst_frames,
                n_frames=int(n_frames),
            ),
            "source_detections/source_detect_row_index": source_row_ids,
            "source_detections/frame_indices": source_frames,
            "source_detections/source_acquisition_frame_index": source_frames.astype(
                np.int64
            ),
            "source_detections/instance_key": source_keys,
            "source_detections/bbox_norm_coords": source_bbox,
            "source_detections/bbox_img_xyxy": source_bbox_img,
            "source_detections/centers_img_xy": source_centers,
            "source_detections/scores": source_scores,
            "source_detections/class_ids": source_classes,
            "source_detections/decision_codes": decisions,
            "source_detections/resolved_refined_row_id": resolved,
            "source_detections/reason_codes": source_reason_codes,
            "source_detections/frame_row_offsets": _offsets(
                source_frames,
                n_frames=int(n_frames),
            ),
        }

        mappings.extend(
            (
                _mapping(
                    "instances/frame_indices",
                    f"instances/{inst_frames_name}",
                    "cast_int32_and_sort",
                ),
                _mapping(
                    "instances/refined_row_ids",
                    f"instances/{refined_ids_name}",
                    "cast_int64_and_sort",
                ),
                _mapping(
                    "instances/bbox_norm_coords",
                    f"instances/{inst_bbox_name}",
                    "cast_float32_authority",
                ),
                _mapping(
                    "instances/source_kind_codes",
                    f"instances/{kind_name}",
                    "cast_uint8",
                ),
                _mapping(
                    "instances/manual_edit_flags",
                    f"instances/{manual_flags_name}",
                    "exact_bool",
                ),
                _mapping(
                    "instances/source_detect_row_index",
                    f"instances/{source_rows_name}",
                    "cast_int64",
                ),
                _mapping(
                    "instances/class_ids", f"instances/{inst_class_name}", "cast_int32"
                ),
                _mapping(
                    "instances/scores",
                    f"instances/{inst_scores_name}",
                    "raw_join_plus_manual_missing_encoding",
                ),
                _mapping(
                    "instances/score_valid",
                    "instances/source_kind_codes",
                    "derive_raw_true_manual_false",
                ),
                _mapping(
                    "instances/instance_key",
                    f"instances/{inst_keys_name}",
                    "preserve_or_frozen_allocator",
                ),
                _mapping(
                    "instances/source_acquisition_frame_index",
                    "instances/frame_indices",
                    "full_acquisition_identity_int64",
                ),
                _mapping(
                    "instances/frame_row_offsets",
                    "instances/frame_indices",
                    "csr_bincount",
                ),
                _mapping(
                    "source_detections/source_detect_row_index",
                    f"source_detections/{source_rows_identity_name}",
                    "cast_int64",
                ),
                _mapping(
                    "source_detections/frame_indices",
                    f"source_detections/{source_frames_name}",
                    "cast_int32",
                ),
                _mapping(
                    "source_detections/bbox_norm_coords",
                    f"source_detections/{source_bbox_name}",
                    "cast_float32_authority",
                ),
                _mapping(
                    "source_detections/decision_codes",
                    f"source_detections/{decisions_name}",
                    "cast_uint8",
                ),
                _mapping(
                    "source_detections/resolved_refined_row_id",
                    f"source_detections/{resolved_name}",
                    "cast_int64",
                ),
                _mapping(
                    "source_detections/scores", str(source_scores_name), "cast_float32"
                ),
                _mapping(
                    "source_detections/class_ids",
                    str(source_classes_name),
                    "cast_int32",
                ),
                _mapping(
                    "source_detections/instance_key",
                    str(source_keys_name),
                    "preserve_uint64",
                ),
                _mapping(
                    "source_detections/source_acquisition_frame_index",
                    "source_detections/frame_indices",
                    "full_acquisition_identity_int64",
                ),
                _mapping(
                    "source_detections/frame_row_offsets",
                    "source_detections/frame_indices",
                    "csr_bincount",
                ),
            )
        )
        for group_name, group, names in (
            (
                "instances",
                instances,
                (
                    "frame_counts",
                    "frame_offsets",
                    "reason_bytes",
                    "reason",
                    "review_notes",
                    "bbox_img_xyxy",
                ),
            ),
            (
                "source_detections",
                source,
                ("reason_bytes", "reason", "review_notes", "bbox_img_xyxy"),
            ),
        ):
            for name in names:
                try:
                    present = group.get(name) is not None
                except AttributeError:
                    present = name in group
                if present:
                    excluded.append(f"{group_name}/{name}")

        schema_issues = REFINED_DETECTION_SCHEMA_V1.validate(
            arrays,
            dimensions=dimensions,
        )
        if schema_issues:
            blockers.extend(
                f"{issue.code} at {issue.path}: {issue.message}"
                for issue in schema_issues
            )
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        blockers.append(str(exc))
        dimensions = None
        arrays = {}
        inst_reason_registry = {0: "none"}
        source_reason_registry = {0: "none"}

    report: dict[str, object] = {
        "schema_id": TRANSITION_REPORT_SCHEMA_ID,
        "schema_version": TRANSITION_REPORT_SCHEMA_VERSION,
        "source_profile": "current_sparse_refined_detect_compatibility",
        "target_schema": {
            "id": REFINED_DETECTION_SCHEMA_V1.schema_id,
            "version": REFINED_DETECTION_SCHEMA_V1.schema_version,
            "lineage_profile": "full_acquisition",
        },
        "status": "blocked" if blockers else "contract_ready",
        "selector_eligible": False,
        "dimensions_input": dimensions_input,
        "row_counts": (
            None
            if dimensions is None
            else {
                "instances": dimensions.n_instances,
                "source_detections": dimensions.n_source_detections,
            }
        ),
        "mappings": mappings,
        "lossy_conversions": lossy,
        "excluded_compatibility_arrays": sorted(set(excluded)),
        "blockers": blockers,
    }
    if blockers or dimensions is None:
        raise RefinedDetectionTransitionError(report)
    return RefinedDetectionTransitionResult(
        dimensions=dimensions,
        arrays=arrays,
        instance_reason_codes=inst_reason_registry,
        source_reason_codes=source_reason_registry,
        report=report,
    )


__all__ = [
    "TRANSITION_REPORT_SCHEMA_ID",
    "TRANSITION_REPORT_SCHEMA_VERSION",
    "RefinedDetectionTransitionError",
    "RefinedDetectionTransitionResult",
    "build_refined_detection_transition",
]
