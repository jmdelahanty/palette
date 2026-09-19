"""Durable browser checkpoints for mutable keypoint review rows.

The labeling SQLite sidecar owns checkpoint lifecycle.  Canonical keypoint
arrays remain unchanged until :func:`apply_keypoint_checkpoints` runs under the
archive mutation lock.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fisheye.shared.frame_flags import row_identity_payload
from fisheye.shared.subject_mask_stale import mark_downstream_subject_mask_runs_stale
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


KEYPOINT_CHECKPOINT_COMPONENT = "keypoints"
KEYPOINT_CHECKPOINT_SAVE_MODE = "checkpoint_v1"
KEYPOINT_IMMUTABLE_DIRECT_SAVE_MODE = "immutable_delta_direct_v1"
KEYPOINT_CHECKPOINT_PAYLOAD_SCHEMA = (
    "palette.web_labeling_keypoint_checkpoint_payload.v1"
)
KEYPOINT_CHECKPOINT_METADATA_SCHEMA = (
    "palette.web_labeling_keypoint_checkpoint_metadata.v1"
)
KEYPOINT_APPLY_RECEIPTS_ATTR = "keypoint_checkpoint_apply_receipts_v1"
KEYPOINT_APPLY_INFLIGHT_ATTR = "keypoint_checkpoint_apply_inflight_v1"
_RECEIPT_HISTORY_LIMIT = 32
_APPLY_SNAPSHOT_LIMIT = 1_000
_SUPPORTED_OPERATIONS = frozenset(
    {
        "replace_points",
        "mark_no_keypoints",
        "mark_detection_issue",
        "clear_failure_label",
    }
)
_ROW_ARRAY_FIELDS = {
    "keypoints_roi": "kp_roi_arr",
    "keypoints_img": "kp_img_arr",
    "keypoints_norm": "kp_norm_arr",
    "heading": "heading_arr",
    "confidence": "confidence_arr",
    "keypoint_confidences": "conf_arr",
    "triangle_area": "triangle_area_arr",
    "min_angle": "min_angle_arr",
    "triangle_angles": "triangle_angles_arr",
    "refined_success": "refined_success_arr",
    "flip_corrected": "flip_corrected_arr",
    "quality_labels": "quality_labels_arr",
    "confidence_valid": "confidence_valid_arr",
    "geometry_valid": "geometry_valid_arr",
    "usable_keypoints": "usable_arr",
    "edit_applied": "edit_applied_arr",
    "heading_finite": "heading_finite_arr",
    "heading_usable": "heading_usable_arr",
    "detection_source": "detection_source_arr",
}


class KeypointCheckpointConflict(RuntimeError):
    """A checkpoint no longer describes the server-bound canonical row."""


def keypoint_browser_save_mode(session: object) -> str:
    return (
        KEYPOINT_IMMUTABLE_DIRECT_SAVE_MODE
        if bool(getattr(session, "immutable_base", False))
        else KEYPOINT_CHECKPOINT_SAVE_MODE
    )


def _json_value(value: object) -> object:
    """Return strict-JSON row data while preserving non-finite distinctions."""

    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if isinstance(value, float):
        if math.isnan(value):
            return {"nonfinite": "nan"}
        if value == math.inf:
            return {"nonfinite": "+inf"}
        if value == -math.inf:
            return {"nonfinite": "-inf"}
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _decoded_json_value(value: object) -> object:
    if isinstance(value, Mapping) and set(value) == {"nonfinite"}:
        marker = value.get("nonfinite")
        if marker == "nan":
            return float("nan")
        if marker == "+inf":
            return float("inf")
        if marker == "-inf":
            return float("-inf")
    if isinstance(value, Mapping):
        return {str(key): _decoded_json_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_decoded_json_value(child) for child in value]
    return value


def _point_payload(points: object) -> list[list[float | None]]:
    values = np.asarray(points, dtype=np.float64)
    return [
        [float(x) if np.isfinite(x) else None, float(y) if np.isfinite(y) else None]
        for x, y in values
    ]


def _row_array_value(array: object | None, roi_idx: int) -> object:
    if array is None:
        return None
    return _json_value(np.asarray(array[int(roi_idx)]).copy())  # type: ignore[index]


def _reason_value(session: object, roi_idx: int) -> str:
    reason = getattr(session, "reason_arr", None)
    if reason is None:
        return ""
    value = reason[int(roi_idx)]
    return "" if value is None else str(value)


def _row_state_document(session: object, roi_idx: int) -> dict[str, object]:
    """Capture every row surface coupled by established keypoint save actions."""

    values = {
        name: _row_array_value(getattr(session, attribute, None), roi_idx)
        for name, attribute in _ROW_ARRAY_FIELDS.items()
    }
    values["reason"] = _reason_value(session, roi_idx)
    refined = getattr(session, "refined")
    if bool(getattr(session, "recovered_roi_only", False)):
        getter = getattr(refined, "get", None)
        for name in ("keypoint_origin", "keypoint_manual_edit", "training_eligible"):
            array = getter(name) if callable(getter) else None
            values[name] = _row_array_value(array, roi_idx)
    derived = getattr(session, "derived_metric_storage", None)
    if derived is not None:
        for name in ("values", "values_norm", "valid"):
            values[f"derived_{name}"] = _row_array_value(
                getattr(derived, name, None), roi_idx
            )
    return {
        "schema": "palette.keypoint_review_coupled_row_state.v1",
        "roi_idx": int(roi_idx),
        "fields": values,
    }


def _edit_revision(session: object) -> int:
    attrs = getattr(getattr(session, "refined"), "attrs")
    try:
        return int(attrs.get("edit_revision", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _row_identity(session: object, roi_idx: int) -> dict[str, object]:
    identity: dict[str, object] = {
        "roi_idx": int(roi_idx),
        "frame_idx": int(getattr(session, "frame_indices")[int(roi_idx)]),
    }
    identity.update(
        row_identity_payload(
            int(roi_idx),
            source_refined_row_ids=getattr(session, "source_refined_row_ids", None),
            source_detect_row_index=getattr(session, "source_detect_row_index", None),
        )
    )
    instance_keys = getattr(session, "instance_keys", None)
    if instance_keys is not None:
        identity["instance_key"] = int(instance_keys[int(roi_idx)])
    return identity


def _skeleton_binding(session: object) -> dict[str, object]:
    labels = [str(value) for value in getattr(session, "keypoint_labels")]
    policy = getattr(session, "manual_qc_policy", None)
    attrs = getattr(getattr(session, "refined"), "attrs")
    pose_schema = attrs.get("pose_schema")
    pose_mapping = pose_schema if isinstance(pose_schema, Mapping) else {}
    skeleton_id = str(
        getattr(policy, "skeleton_id", "")
        or attrs.get("skeleton_id")
        or pose_mapping.get("skeleton_id")
        or pose_mapping.get("name")
        or "undeclared"
    )
    declared_digest = str(
        getattr(policy, "skeleton_digest", "")
        or attrs.get("skeleton_digest")
        or pose_mapping.get("skeleton_digest")
        or ""
    )
    document = {
        "skeleton_id": skeleton_id,
        "keypoint_labels": labels,
        "keypoint_count": int(getattr(session, "keypoint_count")),
        "coordinate_space": "crop_roi_pixels_xy",
    }
    return {
        **document,
        "declared_skeleton_digest": declared_digest or None,
        "checkpoint_skeleton_digest": canonical_json_sha256(document),
    }


_ROOT_BINDING_ATTRS = (
    "schema_id",
    "schema_version",
    "zarr_purpose",
    "recording_id",
    "recording_identity",
    "initial_contract_digest",
)
_RUN_BINDING_ATTRS = (
    "schema_id",
    "schema_version",
    "initial_contract_digest",
    "coordinate_system",
    "frame_index_domain",
    "source_bindings",
    "source_crop_run",
    "source_refined_run",
    "source_keypoints_run",
    "source_detect_run",
    "run_manifest",
    "refined_keypoint_run_manifest",
    "pose_schema",
    "skeleton_id",
    "skeleton_digest",
    "kpt_shape",
    "keypoint_labels",
    "heading_computation",
    "heading_computation_override",
)


def _selected_attrs(attrs: Mapping[str, object], names: Sequence[str]) -> dict[str, object]:
    return {name: _json_value(attrs[name]) for name in names if name in attrs}


def _scientific_contract_binding(session: object) -> dict[str, object]:
    """Bind effective row-write semantics without hashing archive payload arrays."""

    refined = getattr(session, "refined")
    crop = getattr(session, "crop")
    root = getattr(session, "root")
    policy = getattr(session, "manual_qc_policy", None)
    derived = getattr(session, "derived_metric_storage", None)
    archive_path = Path(str(getattr(session, "zarr_path"))).expanduser().resolve()
    archive_stat: dict[str, int] | None = None
    try:
        stat = archive_path.stat()
        archive_stat = {"device": int(stat.st_dev), "inode": int(stat.st_ino)}
    except OSError:
        pass
    document = {
        "archive_identity": str(archive_path),
        "archive_filesystem_identity": archive_stat,
        "root_contract_attrs": _selected_attrs(
            getattr(root, "attrs", {}), _ROOT_BINDING_ATTRS
        ),
        "refined_run": str(getattr(session, "refined_run")),
        "refined_contract_attrs": _selected_attrs(
            getattr(refined, "attrs", {}), _RUN_BINDING_ATTRS
        ),
        "crop_run": str(getattr(session, "crop_run")),
        "crop_contract_attrs": _selected_attrs(
            getattr(crop, "attrs", {}), _RUN_BINDING_ATTRS
        ),
        "effective_review": {
            "min_triangle_angle": float(getattr(session, "min_triangle_angle")),
            "min_triangle_area": float(getattr(session, "min_triangle_area")),
            "max_triangle_area": (
                None
                if getattr(session, "max_triangle_area", None) is None
                else float(getattr(session, "max_triangle_area"))
            ),
            "confidence_threshold": float(
                getattr(session, "confidence_threshold")
            ),
            "roi_diagonal": (
                None
                if getattr(session, "roi_diagonal", None) is None
                else float(getattr(session, "roi_diagonal"))
            ),
            "norm_factor": _json_value(
                np.asarray(getattr(session, "norm_factor"), dtype=np.float64)
            ),
            "head_triangle_indices": _json_value(
                getattr(session, "head_triangle_indices")
            ),
            "manual_qc_policy": _json_value(policy),
            "derived_metric_schema": _json_value(
                getattr(derived, "schema", None) if derived is not None else None
            ),
        },
    }
    return {
        "schema": "palette.keypoint_checkpoint_scientific_binding.v1",
        "document": document,
        "document_sha256": canonical_json_sha256(document),
    }


def _binding(session: object, roi_idx: int) -> dict[str, object]:
    row_state = _row_state_document(session, roi_idx)
    roi_coordinates = getattr(session, "roi_coordinates_full", None)
    return {
        "target_run_path": f"refined_keypoints_runs/{getattr(session, 'refined_run')}",
        "source_rowset_path": f"crop_runs/{getattr(session, 'crop_run')}",
        "row_identity": _row_identity(session, roi_idx),
        "skeleton": _skeleton_binding(session),
        "scientific_contract": _scientific_contract_binding(session),
        "expected_edit_revision": _edit_revision(session),
        "expected_row_state": row_state,
        "expected_row_state_sha256": canonical_json_sha256(row_state),
        "write_inputs": {
            "roi_coordinates_full": _row_array_value(roi_coordinates, roi_idx),
        },
        "delta_run": getattr(session, "delta_run", None),
        "delta_generation": getattr(session, "delta_generation", None),
    }


def _current_roi_idx(runtime: object) -> int:
    session = getattr(runtime, "review_session")
    failures = getattr(session, "failures")
    if int(failures.size) <= 0:
        raise IndexError("No ROIs are currently loaded for review.")
    position = int(getattr(runtime, "position"))
    if position < 0 or position >= int(failures.size):
        raise IndexError("ROI position is out of range.")
    return int(failures[position])


def _persisted_task_scope(runtime: object, roi_idx: int) -> dict[str, object]:
    rows = getattr(runtime, "task_roi_indices", None)
    if rows is not None and int(roi_idx) not in {
        int(value) for value in np.asarray(rows).tolist()
    }:
        raise KeypointCheckpointConflict(
            f"Keypoint checkpoint row {roi_idx} is outside the current task scope."
        )
    task_scope_sha256 = str(getattr(runtime, "task_scope_sha256", None) or "")
    if not task_scope_sha256:
        task_scope_sha256 = canonical_json_sha256(
            {
                "task_id": str(getattr(runtime, "task_id")),
                "recording_id": str(getattr(runtime, "recording_id")),
            }
        )
    document = {
        "schema": "palette.keypoint_checkpoint_task_row_scope.v1",
        "admitted_roi_idx": int(roi_idx),
        "admission": "server_resolved_current_task_row",
        "task_scope_sha256": task_scope_sha256,
    }
    return {**document, "document_sha256": canonical_json_sha256(document)}


def _validate_points(session: object, points: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    expected = (int(getattr(session, "keypoint_count")), 2)
    if values.shape != expected:
        raise ValueError(f"Expected points shape {expected}, got {values.shape}.")
    if not np.isfinite(values).all():
        labels = getattr(session, "keypoint_labels")
        missing = [
            str(label)
            for label, point in zip(labels, values)
            if not np.isfinite(point).all()
        ]
        raise ValueError(
            "Cannot save incomplete keypoints. Missing: " + ", ".join(missing)
        )
    if bool(getattr(session, "recovered_roi_only", False)):
        height, width = getattr(session, "roi_images").shape[1:3]
        if not (
            np.all(values >= 0)
            and np.all(values[:, 0] < int(width))
            and np.all(values[:, 1] < int(height))
        ):
            raise ValueError("Every training keypoint must be visible inside the crop")
    return values


def stage_keypoint_checkpoint(
    store: object,
    runtime: object,
    *,
    user: str,
    operation: str,
    points: Sequence[Sequence[float]] | None = None,
) -> dict[str, object]:
    """Persist one final-row browser edit without touching canonical Zarr."""

    session = getattr(runtime, "review_session")
    if keypoint_browser_save_mode(session) != KEYPOINT_CHECKPOINT_SAVE_MODE:
        raise ValueError("Immutable keypoint delta review uses its direct compatibility writer.")
    operation_value = str(operation)
    if operation_value not in _SUPPORTED_OPERATIONS:
        raise ValueError(f"Unsupported keypoint checkpoint operation: {operation_value}")
    points_array: np.ndarray | None = None
    if operation_value == "replace_points":
        if points is None:
            raise ValueError("Missing points.")
        points_array = _validate_points(session, points)
    elif points is not None:
        raise ValueError(f"Operation {operation_value} does not accept points.")

    roi_idx = _current_roi_idx(runtime)
    payload: dict[str, object] = {
        "schema": KEYPOINT_CHECKPOINT_PAYLOAD_SCHEMA,
        "operation": operation_value,
    }
    if points_array is not None:
        payload["points"] = _point_payload(points_array)
    binding = _binding(session, roi_idx)
    intended_row_state = _intended_row_state(
        session,
        roi_idx=roi_idx,
        payload=payload,
        base_row_state=binding["expected_row_state"],  # type: ignore[arg-type]
        base_write_inputs=binding["write_inputs"],  # type: ignore[arg-type]
    )
    metadata = {
        "schema": KEYPOINT_CHECKPOINT_METADATA_SCHEMA,
        "binding": binding,
        "intended_row_state": intended_row_state,
        "intended_row_state_sha256": canonical_json_sha256(intended_row_state),
        "save_mode": KEYPOINT_CHECKPOINT_SAVE_MODE,
        # This is the immutable row authorization observed when the checkpoint
        # was admitted.  A corrected row may disappear from a freshly filtered
        # presentation queue before crash recovery, but it remains in this
        # task-bound scope.
        "task_row_scope": _persisted_task_scope(runtime, roi_idx),
        "session_provenance": {
            "session_id": str(getattr(runtime, "session_id")),
            "task_id": str(getattr(runtime, "task_id")),
            "recording_id": str(getattr(runtime, "recording_id")),
            "user": str(user),
            "workflow_kind": "keypoints",
            "reopen_policy": "same_task_recording_user_may_resume",
        },
    }
    checkpoint = store.upsert_session_checkpoint(
        session_id=str(getattr(runtime, "session_id")),
        task_id=str(getattr(runtime, "task_id")),
        recording_id=str(getattr(runtime, "recording_id")),
        user=str(user),
        workflow_kind="keypoints",
        target_run_path=str(binding["target_run_path"]),
        target_edit_revision=int(binding["expected_edit_revision"]),
        source_rowset_path=str(binding["source_rowset_path"]),
        roi_idx=roi_idx,
        component_name=KEYPOINT_CHECKPOINT_COMPONENT,
        payload=payload,
        metadata=metadata,
    )
    return {
        "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
        "roi_idx": roi_idx,
        "frame_idx": int(getattr(session, "frame_indices")[roi_idx]),
        "operation": operation_value,
        "target_edit_revision": int(binding["expected_edit_revision"]),
        "canonical_zarr_mutated": False,
        "saved": True,
        "applied": False,
        "save_mode": KEYPOINT_CHECKPOINT_SAVE_MODE,
    }


def _checkpoint_payload(checkpoint: Mapping[str, object]) -> Mapping[str, object]:
    payload = checkpoint.get("payload")
    if not isinstance(payload, Mapping):
        raise KeypointCheckpointConflict("Keypoint checkpoint payload is missing.")
    if payload.get("schema") != KEYPOINT_CHECKPOINT_PAYLOAD_SCHEMA:
        raise KeypointCheckpointConflict("Keypoint checkpoint payload schema changed.")
    if str(payload.get("operation") or "") not in _SUPPORTED_OPERATIONS:
        raise KeypointCheckpointConflict("Keypoint checkpoint operation is unsupported.")
    return payload


def _checkpoint_metadata(checkpoint: Mapping[str, object]) -> Mapping[str, object]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise KeypointCheckpointConflict("Keypoint checkpoint metadata is missing.")
    if metadata.get("schema") != KEYPOINT_CHECKPOINT_METADATA_SCHEMA:
        raise KeypointCheckpointConflict("Keypoint checkpoint metadata schema changed.")
    binding = metadata.get("binding")
    if not isinstance(binding, Mapping):
        raise KeypointCheckpointConflict("Keypoint checkpoint binding is missing.")
    return metadata


def validate_keypoint_checkpoint(
    checkpoint: Mapping[str, object],
    runtime: object,
    *,
    allowed_row_state_sha256: Sequence[str] = (),
    allowed_intermediate_row_state: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Revalidate ownership, lineage, schema, row identity, and exact base row."""

    session = getattr(runtime, "review_session")
    expected_owner = {
        "task_id": str(getattr(runtime, "task_id")),
        "recording_id": str(getattr(runtime, "recording_id")),
        "user": str(getattr(runtime, "user")),
        "workflow_kind": "keypoints",
        "component_name": KEYPOINT_CHECKPOINT_COMPONENT,
    }
    for key, expected in expected_owner.items():
        if str(checkpoint.get(key) or "") != expected:
            raise KeypointCheckpointConflict(
                f"Keypoint checkpoint ownership mismatch for {key}."
            )
    payload = _checkpoint_payload(checkpoint)
    metadata = _checkpoint_metadata(checkpoint)
    if metadata.get("save_mode") != KEYPOINT_CHECKPOINT_SAVE_MODE:
        raise KeypointCheckpointConflict("Keypoint checkpoint save mode changed.")
    task_scope = metadata.get("task_row_scope")
    if not isinstance(task_scope, Mapping):
        raise KeypointCheckpointConflict("Keypoint checkpoint task row scope is missing.")
    scope_document = {
        "schema": task_scope.get("schema"),
        "admitted_roi_idx": task_scope.get("admitted_roi_idx"),
        "admission": task_scope.get("admission"),
        "task_scope_sha256": task_scope.get("task_scope_sha256"),
    }
    if (
        scope_document["schema"] != "palette.keypoint_checkpoint_task_row_scope.v1"
        or str(task_scope.get("document_sha256") or "")
        != canonical_json_sha256(scope_document)
    ):
        raise KeypointCheckpointConflict("Keypoint checkpoint task row scope changed.")
    intended_row_state = metadata.get("intended_row_state")
    if (
        not isinstance(intended_row_state, Mapping)
        or str(metadata.get("intended_row_state_sha256") or "")
        != canonical_json_sha256(intended_row_state)
    ):
        raise KeypointCheckpointConflict(
            "Keypoint checkpoint intended row state changed."
        )
    session_provenance = metadata.get("session_provenance")
    if not isinstance(session_provenance, Mapping):
        raise KeypointCheckpointConflict("Keypoint checkpoint session provenance is missing.")
    if str(session_provenance.get("session_id") or "") != str(
        checkpoint.get("session_id") or ""
    ):
        raise KeypointCheckpointConflict("Keypoint checkpoint session identity changed.")
    for key in ("task_id", "recording_id", "user", "workflow_kind"):
        if str(session_provenance.get(key) or "") != expected_owner[key]:
            raise KeypointCheckpointConflict(
                f"Keypoint checkpoint session provenance mismatch for {key}."
            )
    if session_provenance.get("reopen_policy") != "same_task_recording_user_may_resume":
        raise KeypointCheckpointConflict("Keypoint checkpoint reopen policy changed.")
    stored = metadata["binding"]
    assert isinstance(stored, Mapping)
    roi_idx = int(checkpoint.get("roi_idx") or 0)
    if (
        intended_row_state.get("schema")
        != "palette.keypoint_review_coupled_row_state.v1"
        or int(intended_row_state.get("roi_idx", -1)) != roi_idx
    ):
        raise KeypointCheckpointConflict(
            "Keypoint checkpoint intended row identity changed."
        )
    if (
        int(task_scope.get("admitted_roi_idx", -1)) != roi_idx
        or task_scope.get("admission") != "server_resolved_current_task_row"
    ):
        raise KeypointCheckpointConflict(
            f"Keypoint checkpoint row {roi_idx} is outside its admitted task scope."
        )
    current_task_scope_sha256 = str(
        getattr(runtime, "task_scope_sha256", None) or ""
    )
    if current_task_scope_sha256 and str(
        task_scope.get("task_scope_sha256") or ""
    ) != current_task_scope_sha256:
        raise KeypointCheckpointConflict(
            "Keypoint checkpoint static task scope changed."
        )
    current = _binding(session, roi_idx)
    for key in (
        "target_run_path",
        "source_rowset_path",
        "row_identity",
        "skeleton",
        "scientific_contract",
        "write_inputs",
        "delta_run",
        "delta_generation",
    ):
        if _json_value(stored.get(key)) != _json_value(current.get(key)):
            raise KeypointCheckpointConflict(
                f"Keypoint checkpoint binding mismatch for {key}."
            )
    if str(checkpoint.get("target_run_path") or "") != str(
        current["target_run_path"]
    ):
        raise KeypointCheckpointConflict("Keypoint checkpoint target run changed.")
    if str(checkpoint.get("source_rowset_path") or "") != str(
        current["source_rowset_path"]
    ):
        raise KeypointCheckpointConflict("Keypoint checkpoint source rowset changed.")
    # ``expected_edit_revision`` records the target generation observed at save
    # time, but it is not a row revision.  Another row may be applied while
    # this checkpoint waits.  Exact coupled-row content decides freshness.
    allowed_digests = {
        str(stored.get("expected_row_state_sha256") or ""),
        *(str(value) for value in allowed_row_state_sha256 if str(value)),
    }
    if str(current["expected_row_state_sha256"]) not in allowed_digests:
        base_row_state = stored.get("expected_row_state")
        if not (
            isinstance(base_row_state, Mapping)
            and isinstance(allowed_intermediate_row_state, Mapping)
            and _recognized_intermediate_row_state(
                base=base_row_state,
                intended=allowed_intermediate_row_state,
                current=current["expected_row_state"],  # type: ignore[arg-type]
                recovered=bool(getattr(session, "recovered_roi_only", False)),
            )
        ):
            raise KeypointCheckpointConflict("Keypoint checkpoint base row changed.")
    operation = str(payload.get("operation") or "")
    if operation == "replace_points":
        points = payload.get("points")
        if not isinstance(points, Sequence):
            raise KeypointCheckpointConflict("Keypoint checkpoint points are missing.")
        _validate_points(session, points)  # type: ignore[arg-type]
    return payload


def _recognized_intermediate_row_state(
    *,
    base: Mapping[str, object],
    intended: Mapping[str, object],
    current: Mapping[str, object],
    recovered: bool,
) -> bool:
    """Accept only values emitted along the deterministic one-row writer path."""

    if any(
        _json_value(document.get(key)) != _json_value(base.get(key))
        for document in (intended, current)
        for key in ("schema", "roi_idx")
    ):
        return False
    base_fields = base.get("fields")
    intended_fields = intended.get("fields")
    current_fields = current.get("fields")
    if not all(
        isinstance(value, Mapping)
        for value in (base_fields, intended_fields, current_fields)
    ):
        return False
    assert isinstance(base_fields, Mapping)
    assert isinstance(intended_fields, Mapping)
    assert isinstance(current_fields, Mapping)
    if set(current_fields) != set(base_fields) or set(current_fields) != set(
        intended_fields
    ):
        return False
    for field_name, current_value in current_fields.items():
        allowed = [base_fields[field_name], intended_fields[field_name]]
        # Recovered saves deliberately force eligibility false before any
        # other row field and restore it from final QC only at the end.
        if recovered and field_name in {"training_eligible", "usable_keypoints"}:
            allowed.append(False)
        if not any(_json_value(current_value) == _json_value(value) for value in allowed):
            return False
    return True


def overlay_keypoint_checkpoint(
    payload: Mapping[str, object], checkpoint: Mapping[str, object]
) -> dict[str, object]:
    """Overlay one already-validated checkpoint onto a canonical ROI payload."""

    out = dict(payload)
    saved = _checkpoint_payload(checkpoint)
    operation = str(saved.get("operation") or "")
    reason = str(out.get("reason") or "")
    tags = [token.strip() for token in reason.split("|") if token.strip()]
    if operation == "replace_points":
        out["points"] = saved["points"]
    elif operation in {"mark_no_keypoints", "mark_detection_issue"}:
        out["points"] = [[None, None] for _ in out.get("labels", [])]
        tag = (
            "fish_present_no_keypoints"
            if operation == "mark_no_keypoints"
            else "detection_issue"
        )
        tags = [value for value in tags if value != "manual_correction"]
        if tag not in tags:
            tags.append(tag)
        out["reason"] = "|".join(tags)
    elif operation == "clear_failure_label":
        out["reason"] = "|".join(
            value
            for value in tags
            if value not in {"fish_present_no_keypoints", "detection_issue"}
        )
    out["session_checkpoint"] = {
        "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
        "state": str(checkpoint.get("state") or ""),
        "operation": operation,
        "saved": True,
        "applied": False,
        "updated_at_utc": str(checkpoint.get("updated_at_utc") or ""),
    }
    return out


def current_keypoint_payload(
    store: object,
    runtime: object,
    backend_module: object,
    *,
    state_payload: Mapping[str, object],
) -> dict[str, object]:
    """Load one ROI image once, then overlay its exact durable checkpoint."""

    session = getattr(runtime, "review_session")
    canonical = dict(
        backend_module.load_roi_payload(  # type: ignore[attr-defined]
            session, position=int(getattr(runtime, "position"))
        )
    )
    roi_idx = int(canonical["roi_idx"])
    checkpoint = store.get_session_checkpoint(
        task_id=str(getattr(runtime, "task_id")),
        roi_idx=roi_idx,
        component_name=KEYPOINT_CHECKPOINT_COMPONENT,
        state=None,
    )
    if checkpoint is not None and str(checkpoint.get("state") or "") in {
        "active",
        "applying",
    }:
        allowed: list[str] = []
        intermediate: Mapping[str, object] | None = None
        current_write_uncertain = False
        if str(checkpoint.get("state") or "") == "applying":
            attrs = getattr(getattr(session, "refined"), "attrs")
            inflight = attrs.get(KEYPOINT_APPLY_INFLIGHT_ATTR)
            # A claim is committed before the archive lock is acquired.  In
            # that pre-write window the exact staged base remains valid and is
            # safe to overlay even though no Zarr inflight record exists yet.
            if inflight is not None and not isinstance(inflight, Mapping):
                raise KeypointCheckpointConflict(
                    "Applying keypoint checkpoint has an invalid target recovery receipt."
                )
            if isinstance(inflight, Mapping):
                if str(inflight.get("apply_id") or "") != str(
                    checkpoint.get("apply_id") or ""
                ):
                    raise KeypointCheckpointConflict(
                        "Applying keypoint checkpoint belongs to another target recovery receipt."
                    )
                checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
                completed = inflight.get("completed_row_state_sha256")
                if isinstance(completed, Mapping):
                    digest = completed.get(checkpoint_id)
                    if digest:
                        allowed.append(str(digest))
                if str(inflight.get("current_checkpoint_id") or "") == checkpoint_id:
                    candidate = inflight.get("current_expected_row_state")
                    candidate_sha256 = str(
                        inflight.get("current_expected_row_state_sha256") or ""
                    )
                    if not isinstance(candidate, Mapping) or candidate_sha256 != canonical_json_sha256(candidate):
                        raise KeypointCheckpointConflict(
                            "Applying keypoint checkpoint recovery intent changed."
                        )
                    intermediate = candidate
                    current_write_uncertain = True
        validate_keypoint_checkpoint(
            checkpoint,
            runtime,
            allowed_row_state_sha256=allowed,
            allowed_intermediate_row_state=intermediate,
        )
        if current_write_uncertain:
            raise KeypointCheckpointConflict(
                "This keypoint row is currently being applied or recovered; reload after apply completes."
            )
        canonical = overlay_keypoint_checkpoint(canonical, checkpoint)
    canonical["state"] = dict(state_payload)
    canonical["ok"] = True
    return canonical


def _checkpoint_snapshot_row(checkpoint: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "palette.labeling_session_checkpoint_snapshot_row.v1",
        "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
        "task_id": str(checkpoint.get("task_id") or ""),
        "recording_id": str(checkpoint.get("recording_id") or ""),
        "user": str(checkpoint.get("user") or ""),
        "workflow_kind": str(checkpoint.get("workflow_kind") or ""),
        "target_run_path": str(checkpoint.get("target_run_path") or ""),
        "target_edit_revision": int(checkpoint.get("target_edit_revision") or 0),
        "source_rowset_path": str(checkpoint.get("source_rowset_path") or ""),
        "roi_idx": int(checkpoint.get("roi_idx") or 0),
        "component_name": str(checkpoint.get("component_name") or ""),
        "payload": _json_value(checkpoint.get("payload")),
        "metadata": _json_value(checkpoint.get("metadata")),
    }


def _checkpoint_snapshot_row_sha256(checkpoint: Mapping[str, object]) -> str:
    declared = str(checkpoint.get("snapshot_row_sha256") or "")
    if "payload" not in checkpoint or "metadata" not in checkpoint:
        if not declared:
            raise KeypointCheckpointConflict(
                "Keypoint checkpoint snapshot descriptor has no row digest."
            )
        return declared
    computed = canonical_json_sha256(_checkpoint_snapshot_row(checkpoint))
    if declared and declared != computed:
        raise KeypointCheckpointConflict(
            "Keypoint checkpoint persisted row digest changed."
        )
    return computed


def checkpoint_snapshot_digest(checkpoints: Sequence[Mapping[str, object]]) -> str:
    rows = [
        {
            "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
            "snapshot_row_sha256": _checkpoint_snapshot_row_sha256(checkpoint),
        }
        for checkpoint in sorted(
            checkpoints,
            key=lambda value: (
                int(value.get("roi_idx") or 0),
                str(value.get("checkpoint_id") or ""),
            ),
        )
    ]
    return canonical_json_sha256(
        {"schema": "palette.keypoint_checkpoint_apply_snapshot.v1", "rows": rows}
    )


def _pending_keypoint_apply_effects(
    store: object, *, task_id: str
) -> list[Mapping[str, object]]:
    reader = getattr(store, "list_pending_session_checkpoint_apply_effects", None)
    if not callable(reader):
        return []
    return [
        row
        for row in reader(
            task_id=task_id,
            component_name=KEYPOINT_CHECKPOINT_COMPONENT,
            limit=_APPLY_SNAPSHOT_LIMIT,
        )
        if isinstance(row, Mapping)
    ]


def count_unfinished_keypoint_checkpoint_edits(store: object, *, task_id: str) -> int:
    """Count rows that still need canonical apply or apply side effects."""

    unapplied = int(
        store.count_unapplied_session_checkpoints(
            task_id=task_id, component_name=KEYPOINT_CHECKPOINT_COMPONENT
        )
    )
    pending_effects = _pending_keypoint_apply_effects(store, task_id=task_id)
    pending_rows = sum(
        max(1, int(row.get("checkpoint_count") or 0)) for row in pending_effects
    )
    exact_counter = getattr(
        store, "count_pending_session_checkpoint_apply_effects", None
    )
    if callable(exact_counter):
        pending_receipt_count = int(
            exact_counter(
                task_id=task_id, component_name=KEYPOINT_CHECKPOINT_COMPONENT
            )
        )
        pending_rows = max(pending_rows, pending_receipt_count)
    return unapplied + pending_rows


def keypoint_checkpoint_state(store: object, runtime: object) -> dict[str, object]:
    task_id = str(getattr(runtime, "task_id"))
    descriptor_reader = getattr(
        store, "list_session_checkpoint_snapshot_descriptors", None
    )
    list_method = (
        descriptor_reader
        if callable(descriptor_reader)
        else store.list_session_checkpoints
    )
    active = list_method(
        task_id=task_id,
        state="active",
        component_name=KEYPOINT_CHECKPOINT_COMPONENT,
        limit=_APPLY_SNAPSHOT_LIMIT,
    )
    applying = list_method(
        task_id=task_id,
        state="applying",
        component_name=KEYPOINT_CHECKPOINT_COMPONENT,
        limit=_APPLY_SNAPSHOT_LIMIT,
    )
    unapplied_checkpoint_count = int(
        store.count_unapplied_session_checkpoints(
            task_id=task_id, component_name=KEYPOINT_CHECKPOINT_COMPONENT
        )
    )
    pending_effects = _pending_keypoint_apply_effects(store, task_id=task_id)
    pending_effects.sort(
        key=lambda value: (
            str(value.get("applied_at_utc") or ""),
            str(value.get("apply_id") or ""),
        )
    )
    pending_effect = pending_effects[0] if pending_effects else None
    pending_effect_rows = sum(
        max(1, int(row.get("checkpoint_count") or 0)) for row in pending_effects
    )
    pending_effect_counter = getattr(
        store, "count_pending_session_checkpoint_apply_effects", None
    )
    pending_effect_count = (
        int(
            pending_effect_counter(
                task_id=task_id, component_name=KEYPOINT_CHECKPOINT_COMPONENT
            )
        )
        if callable(pending_effect_counter)
        else len(pending_effects)
    )
    pending_effect_rows = max(pending_effect_rows, pending_effect_count)
    state_counter = getattr(store, "count_session_checkpoints", None)
    active_count = (
        int(
            state_counter(
                task_id=task_id,
                state="active",
                component_name=KEYPOINT_CHECKPOINT_COMPONENT,
            )
        )
        if callable(state_counter)
        else len(active)
    )
    applying_count = (
        int(
            state_counter(
                task_id=task_id,
                state="applying",
                component_name=KEYPOINT_CHECKPOINT_COMPONENT,
            )
        )
        if callable(state_counter)
        else len(applying)
    )
    # Recovery always owns its original applying snapshot.  Otherwise apply the
    # same deterministic bounded active window that the store claim API uses.
    selected = applying if applying else active
    applying_ids = sorted(
        {
            str(row.get("apply_id") or "")
            for row in applying
            if str(row.get("apply_id") or "")
        }
    )
    immutable = bool(
        getattr(getattr(runtime, "review_session"), "immutable_base", False)
    )
    resumable_apply_id = (
        str(pending_effect.get("apply_id") or "")
        if pending_effect is not None
        else (applying_ids[0] if len(applying_ids) == 1 else "")
    )
    resumable_digest = (
        str(pending_effect.get("checkpoint_snapshot_sha256") or "")
        if pending_effect is not None
        else (
            checkpoint_snapshot_digest(applying)
            if applying and len(applying_ids) == 1
            else ""
        )
    )
    selected_digest = (
        resumable_digest
        if pending_effect is not None
        else (checkpoint_snapshot_digest(selected) if selected else "")
    )
    selected_count = (
        int(pending_effect.get("checkpoint_count") or 0)
        if pending_effect is not None
        else len(selected)
    )
    return {
        "save_mode": keypoint_browser_save_mode(getattr(runtime, "review_session")),
        "checkpoint_save_supported": not immutable,
        "checkpoint_save_semantics": (
            "local_checkpoint_no_canonical_zarr_write"
            if not immutable
            else "not_applicable_immutable_delta_direct_write"
        ),
        "unapplied_session_edit_count": (
            unapplied_checkpoint_count + pending_effect_rows
        ),
        "active_session_edit_count": active_count,
        "applying_session_edit_count": applying_count,
        "pending_apply_effect_count": pending_effect_count,
        "checkpoint_snapshot_sha256": selected_digest or None,
        "selected_session_edit_count": selected_count,
        "checkpoint_apply_batch_limit": _APPLY_SNAPSHOT_LIMIT,
        "resumable_apply_id": resumable_apply_id or None,
        "resumable_checkpoint_snapshot_sha256": resumable_digest or None,
        "apply_available": bool(pending_effect is not None or selected)
        and not immutable,
    }


class _DryRunRowArray:
    """One-row mutable array view used to derive an exact apply intent."""

    def __init__(
        self, source: object, roi_idx: int, *, initial_value: object | None = None
    ) -> None:
        self._source = source
        self._roi_idx = int(roi_idx)
        self._row = np.asarray(
            source[self._roi_idx]  # type: ignore[index]
            if initial_value is None
            else _decoded_json_value(initial_value),
            dtype=getattr(source, "dtype", None),
        ).copy()
        self.shape = getattr(source, "shape")
        self.dtype = getattr(source, "dtype", self._row.dtype)
        self.chunks = getattr(source, "chunks", None)

    def _row_selector(self, item: object) -> tuple[bool, object | None]:
        first = item[0] if isinstance(item, tuple) else item
        suffix: object | None = item[1:] if isinstance(item, tuple) else None
        if isinstance(first, (int, np.integer)):
            return int(first) == self._roi_idx, suffix
        if isinstance(first, slice):
            start, stop, step = first.indices(int(self.shape[0]))
            return (start, stop, step) == (self._roi_idx, self._roi_idx + 1, 1), suffix
        return False, suffix

    def __getitem__(self, item: object) -> object:
        is_row, suffix = self._row_selector(item)
        if not is_row:
            return self._source[item]  # type: ignore[index]
        first = item[0] if isinstance(item, tuple) else item
        value: object = (
            self._row[np.newaxis, ...] if isinstance(first, slice) else self._row.copy()
        )
        if suffix:
            value = np.asarray(value)[suffix]  # type: ignore[index]
        return value

    def __setitem__(self, item: object, value: object) -> None:
        is_row, suffix = self._row_selector(item)
        if not is_row:
            raise RuntimeError("Dry-run keypoint apply attempted to write another row.")
        first = item[0] if isinstance(item, tuple) else item
        values = np.asarray(value, dtype=self.dtype)
        if isinstance(first, slice):
            values = values[0]
        if suffix:
            target = self._row.copy()
            target[suffix] = values  # type: ignore[index]
            values = target
        self._row = np.asarray(values, dtype=self.dtype).copy()


class _DryRunGroup:
    def __init__(self, source: object, overrides: Mapping[str, object]) -> None:
        self._source = source
        self._overrides = dict(overrides)
        self.attrs = getattr(source, "attrs", {})

    def __contains__(self, key: object) -> bool:
        return str(key) in self._overrides or key in self._source  # type: ignore[operator]

    def __getitem__(self, key: str) -> object:
        if key in self._overrides:
            return self._overrides[key]
        return self._source[key]  # type: ignore[index]

    def get(self, key: str, default: object = None) -> object:
        if key in self._overrides:
            return self._overrides[key]
        getter = getattr(self._source, "get", None)
        return getter(key, default) if callable(getter) else default


class _DryRunRoot:
    def __init__(self, source: object) -> None:
        self._source = source
        self.attrs = getattr(source, "attrs", {})

    def get(self, key: str, default: object = None) -> object:
        # The row intent covers the canonical keypoint surfaces.  Downstream
        # stale publication is idempotently performed only by the real writer.
        if key == "refined_subject_masks_runs":
            return default
        getter = getattr(self._source, "get", None)
        return getter(key, default) if callable(getter) else default


def _intended_row_state(
    session: object,
    *,
    roi_idx: int,
    payload: Mapping[str, object],
    base_row_state: Mapping[str, object] | None = None,
    base_write_inputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run established science logic against bounded one-row in-memory views."""

    from fisheye.tune import keypoint_review_backend as established_backend

    dry = copy.copy(session)
    overrides: dict[str, object] = {}
    base_fields = (
        base_row_state.get("fields")
        if isinstance(base_row_state, Mapping)
        else None
    )
    base_fields = base_fields if isinstance(base_fields, Mapping) else {}
    for dataset_name, attribute_name in _ROW_ARRAY_FIELDS.items():
        source = getattr(session, attribute_name, None)
        if source is None:
            continue
        wrapped = _DryRunRowArray(
            source, roi_idx, initial_value=base_fields.get(dataset_name)
        )
        setattr(dry, attribute_name, wrapped)
        overrides[dataset_name] = wrapped
    reason = getattr(session, "reason_arr", None)
    if reason is not None:
        wrapped_reason = _DryRunRowArray(
            reason, roi_idx, initial_value=base_fields.get("reason")
        )
        dry.reason_arr = wrapped_reason
        overrides["reason"] = wrapped_reason
    refined = getattr(session, "refined")
    if bool(getattr(session, "recovered_roi_only", False)):
        getter = getattr(refined, "get", None)
        for name in ("keypoint_origin", "keypoint_manual_edit", "training_eligible"):
            source = getter(name) if callable(getter) else None
            if source is not None:
                overrides[name] = _DryRunRowArray(
                    source, roi_idx, initial_value=base_fields.get(name)
                )
    dry.refined = _DryRunGroup(refined, overrides)
    dry.root = _DryRunRoot(getattr(session, "root"))
    roi_coordinates = getattr(session, "roi_coordinates_full", None)
    if roi_coordinates is not None:
        dry.roi_coordinates_full = _DryRunRowArray(
            roi_coordinates,
            roi_idx,
            initial_value=(
                base_write_inputs.get("roi_coordinates_full")
                if isinstance(base_write_inputs, Mapping)
                else None
            ),
        )
    derived = getattr(session, "derived_metric_storage", None)
    if derived is not None:
        dry.derived_metric_storage = type(derived)(
            schema=derived.schema,
            values=_DryRunRowArray(
                derived.values,
                roi_idx,
                initial_value=base_fields.get("derived_values"),
            ),
            values_norm=_DryRunRowArray(
                derived.values_norm,
                roi_idx,
                initial_value=base_fields.get("derived_values_norm"),
            ),
            valid=_DryRunRowArray(
                derived.valid,
                roi_idx,
                initial_value=base_fields.get("derived_valid"),
            ),
        )
    _apply_operation(established_backend, dry, roi_idx=roi_idx, payload=payload)
    return _row_state_document(dry, roi_idx)


def _apply_operation(
    backend_module: object,
    session: object,
    *,
    roi_idx: int,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    # Presentation filters may remove a corrected row before recovery.  Use a
    # private session view addressed by the already-validated stable ROI row.
    apply_session = copy.copy(session)
    apply_session.failures = np.asarray([int(roi_idx)], dtype=np.int64)
    operation = str(payload.get("operation") or "")
    if operation == "replace_points":
        return backend_module.save_roi_correction(  # type: ignore[attr-defined]
            apply_session, position=0, points=payload["points"]
        )
    function_name = {
        "mark_no_keypoints": "mark_no_keypoints",
        "mark_detection_issue": "mark_detection_issue",
        "clear_failure_label": "clear_failure_label",
    }[operation]
    return getattr(backend_module, function_name)(apply_session, position=0)


def _ensure_downstream_stale_after_row_apply(
    session: object,
    *,
    roi_idx: int,
    payload: Mapping[str, object],
    base_row_state: Mapping[str, object],
    intended_row_state: Mapping[str, object],
) -> int:
    """Close the crash window between row writes and stale publication."""

    if canonical_json_sha256(base_row_state) == canonical_json_sha256(
        intended_row_state
    ):
        return 0
    stale_reason = {
        "replace_points": "keypoint_manual_correction",
        "mark_no_keypoints": "keypoint_mark_no_keypoints",
        "mark_detection_issue": "keypoint_mark_detection_issue",
        "clear_failure_label": "keypoint_clear_failure_label",
    }[str(payload.get("operation") or "")]
    return int(
        mark_downstream_subject_mask_runs_stale(
            getattr(session, "root"),
            source_keypoint_group="refined_keypoints_runs",
            source_keypoints_run=str(getattr(session, "refined_run")),
            roi_indices=[int(roi_idx)],
            frame_indices=[int(getattr(session, "frame_indices")[int(roi_idx)])],
            reason=stale_reason,
        )
    )


def _converge_row_to_intended_state(
    session: object, *, roi_idx: int, intended: Mapping[str, object]
) -> None:
    """Write the precomputed established-backend result for every coupled field."""

    fields = intended.get("fields")
    if not isinstance(fields, Mapping):
        raise KeypointCheckpointConflict("Keypoint apply intended row fields are missing.")
    for field_name, attribute_name in _ROW_ARRAY_FIELDS.items():
        array = getattr(session, attribute_name, None)
        value = fields.get(field_name)
        if array is not None and value is not None:
            array[int(roi_idx)] = _decoded_json_value(value)  # type: ignore[index]
    reason = getattr(session, "reason_arr", None)
    if reason is not None and fields.get("reason") is not None:
        reason[int(roi_idx) : int(roi_idx) + 1] = np.asarray(
            [_decoded_json_value(fields["reason"])], dtype=object
        )
    if bool(getattr(session, "recovered_roi_only", False)):
        refined = getattr(session, "refined")
        getter = getattr(refined, "get", None)
        for name in ("keypoint_origin", "keypoint_manual_edit", "training_eligible"):
            array = getter(name) if callable(getter) else None
            value = fields.get(name)
            if array is not None and value is not None:
                array[int(roi_idx)] = _decoded_json_value(value)
    derived = getattr(session, "derived_metric_storage", None)
    if derived is not None:
        for name in ("values", "values_norm", "valid"):
            value = fields.get(f"derived_{name}")
            if value is not None:
                getattr(derived, name)[int(roi_idx)] = _decoded_json_value(value)


from .web_keypoint_checkpoint_apply import apply_keypoint_checkpoints  # noqa: E402


__all__ = [
    "KEYPOINT_APPLY_INFLIGHT_ATTR",
    "KEYPOINT_APPLY_RECEIPTS_ATTR",
    "KEYPOINT_CHECKPOINT_COMPONENT",
    "KEYPOINT_CHECKPOINT_SAVE_MODE",
    "KEYPOINT_IMMUTABLE_DIRECT_SAVE_MODE",
    "KeypointCheckpointConflict",
    "apply_keypoint_checkpoints",
    "checkpoint_snapshot_digest",
    "count_unfinished_keypoint_checkpoint_edits",
    "current_keypoint_payload",
    "keypoint_browser_save_mode",
    "keypoint_checkpoint_state",
    "overlay_keypoint_checkpoint",
    "stage_keypoint_checkpoint",
    "validate_keypoint_checkpoint",
]
