"""Frozen operator decisions for registered-dish geometry publication.

The browser builds and persists one content-addressed request.  Canonical Zarr
writes are performed later by a commit-pinned LSF job after revalidating every
binding in this record.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    ACQUISITION_CANDIDATE_KIND,
    ArenaGeometryCandidatePlan,
    CANDIDATE_RUNS_PARENT,
    PALETTE_CANDIDATE_KIND,
    plan_reviewed_palette_geometry_candidate,
    validate_arena_geometry_candidate_record,
)
from fisheye.analysis_workflows.materializers.arena_geometry_comparison import (
    MANUAL_REVIEW_POLICY_ID,
    SEMANTIC_COMPATIBILITY_STATES,
)
from fisheye.analysis_workflows.materializers.arena_geometry_fit_review import (
    FIT_REVIEW_RECORD_SCHEMA_ID,
    FIT_REVIEW_RUN_SCHEMA_ID,
    FIT_REVIEW_RUNS_PARENT,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.crop_defaults import DEFAULT_ZEBRAFISH_CROP_SIZE_PX
from fisheye.shared.detection_tables import (
    resolve_detection_instance_table,
    resolve_detection_source_pixel_authority,
)
from fisheye.shared.json_safety import strict_json_dumps, write_json_atomic
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    canonical_detection_dimensions_from_manifest,
    require_active_coordinate_canonical_detection,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr_io import open_zarr_root

from .geometry_review import GeometryReviewRegistryError, load_geometry_review_queue

APPROVAL_REQUEST_SCHEMA_ID = "palette.geometry_review_approval_request"
APPROVAL_REQUEST_SCHEMA_VERSION = 1
APPROVAL_CHOICES = frozenset({"palette", "acquisition"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class GeometryReviewApprovalError(ValueError):
    """An approval request is incomplete, stale, ambiguous, or unsafe."""


@dataclass(frozen=True)
class GeometryReviewApprovalRequest:
    request_id: str
    request_sha256: str
    payload: Mapping[str, Any]

    @property
    def analysis_zarr(self) -> Path:
        return Path(str(self.payload["identity"]["dataset"]["analysis_zarr"]))

    @property
    def gate_run(self) -> str:
        return str(self.payload["pipeline"]["gate_run"])


def _canonical(value: Any) -> Any:
    return json.loads(strict_json_dumps(value))


def _sha256(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def _required_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise GeometryReviewApprovalError(f"{label} cannot be empty.")
    return text


def _safe_name(value: object, *, label: str) -> str:
    text = _required_text(value, label=label)
    if Path(text).name != text or text in {".", ".."}:
        raise GeometryReviewApprovalError(f"{label} must be one safe group name.")
    return text


def _required_sha256(value: object, *, label: str) -> str:
    digest = _required_text(value, label=label).lower().removeprefix("sha256:")
    if _SHA256_RE.fullmatch(digest) is None:
        raise GeometryReviewApprovalError(f"{label} must be one SHA-256 digest.")
    return digest


def _required_git_commit(value: object, *, label: str) -> str:
    commit = _required_text(value, label=label).lower()
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise GeometryReviewApprovalError(f"{label} must be one full Git commit.")
    return commit


def _required_utc_timestamp(value: object, *, label: str) -> str:
    text = _required_text(value, label=label)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise GeometryReviewApprovalError(
            f"{label} must be one ISO-8601 UTC timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise GeometryReviewApprovalError(f"{label} must be expressed in UTC.")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _group(root: Any, path: str, *, label: str) -> Any:
    try:
        return root[path]
    except KeyError as exc:
        raise GeometryReviewApprovalError(f"{label} is missing: {path}") from exc


def _candidate_binding(root: Any, run_name: str) -> dict[str, Any]:
    run = _safe_name(run_name, label="candidate_run")
    group = _group(
        root,
        f"analysis/{CANDIDATE_RUNS_PARENT}/{run}",
        label="Arena-geometry candidate",
    )
    attrs = dict(group.attrs)
    if (
        attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("stage_selector_eligible") is not True
        or attrs.get("candidate_id") != run
    ):
        raise GeometryReviewApprovalError(
            f"Candidate {run!r} is not complete immutable candidate evidence."
        )
    record = attrs.get("candidate_record")
    if not isinstance(record, Mapping):
        raise GeometryReviewApprovalError(f"Candidate {run!r} lacks candidate_record.")
    validate_arena_geometry_candidate_record(record)
    digest = _sha256(record)
    if (
        _required_sha256(
            attrs.get("candidate_record_sha256"),
            label=f"candidate {run} digest",
        )
        != digest
    ):
        raise GeometryReviewApprovalError(f"Candidate {run!r} digest is invalid.")
    return {
        "run_name": run,
        "candidate_kind": record["candidate_kind"],
        "candidate_record_sha256": digest,
        "arena_binding": _canonical(record["arena_binding"]),
        "coordinate_binding": _canonical(record["coordinate_binding"]),
    }


def _fit_review_binding(root: Any, run_name: str) -> dict[str, Any]:
    run = _safe_name(run_name, label="fit_review_run")
    group = _group(
        root,
        f"analysis/{FIT_REVIEW_RUNS_PARENT}/{run}",
        label="Fit-review evidence",
    )
    attrs = dict(group.attrs)
    if (
        attrs.get("schema_id") != FIT_REVIEW_RUN_SCHEMA_ID
        or attrs.get("schema_version") != 1
        or attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("review_status") != "awaiting_explicit_human_review"
    ):
        raise GeometryReviewApprovalError(
            f"Fit-review run {run!r} is not complete pending immutable evidence."
        )
    record = attrs.get("review_record")
    if not isinstance(record, Mapping) or (
        record.get("schema_id") != FIT_REVIEW_RECORD_SCHEMA_ID
        or record.get("schema_version") != 1
    ):
        raise GeometryReviewApprovalError(f"Fit-review run {run!r} is invalid.")
    digest = _sha256(record)
    if (
        _required_sha256(
            attrs.get("review_record_sha256"),
            label=f"fit-review {run} digest",
        )
        != digest
    ):
        raise GeometryReviewApprovalError(f"Fit-review run {run!r} digest is invalid.")
    source = record.get("source")
    if not isinstance(source, Mapping):
        raise GeometryReviewApprovalError(
            f"Fit-review run {run!r} lacks source binding."
        )
    return {
        "run_name": run,
        "review_record_sha256": digest,
        "camera_serial": _required_text(
            source.get("camera_serial"), label="fit-review camera_serial"
        ),
        "frame_count": int(source.get("frame_count")),
        "video_path": str(
            Path(
                _required_text(source.get("video_path"), label="fit-review video_path")
            )
            .expanduser()
            .resolve()
        ),
    }


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or int(value) <= 0:
        raise GeometryReviewApprovalError(f"{label} must be one positive integer.")
    return int(value)


def _detection_frame_count_binding(
    group: Any,
    table: Any,
    *,
    group_path: str,
    attrs: Mapping[str, Any],
    manifest_frame_count: int,
    row_count: int,
    instance_key_sha256: str,
) -> tuple[int, dict[str, Any]]:
    """Cross-check exact canonical-v3 detection frame cardinality."""

    declarations: dict[str, int] = {
        "canonical_run_manifest": _positive_int(
            manifest_frame_count,
            label="Canonical detection manifest frame count",
        )
    }
    for name in ("frame_count", "recording_frame_count"):
        if attrs.get(name) is not None:
            declarations[f"attribute:{name}"] = _positive_int(
                attrs.get(name), label=f"Detection {name}"
            )

    temporal = attrs.get("source_row_temporal_authority")
    temporal_digest: str | None = None
    if temporal is not None:
        if not isinstance(temporal, Mapping):
            raise GeometryReviewApprovalError(
                "Detection source_row_temporal_authority must be an object."
            )
        if (
            temporal.get("schema_id") != "palette.source_row_temporal_authority"
            or temporal.get("schema_version") != 1
            or temporal.get("source_rowset_ref") != f"/{group_path}"
            or temporal.get("source_identity_domain") != "observation_instance"
            or temporal.get("source_identity_mode") != "instance_key"
            or temporal.get("source_leading_dimension") != row_count
        ):
            raise GeometryReviewApprovalError(
                "Detection source_row_temporal_authority has incompatible identity."
            )
        temporal_digest = _required_sha256(
            attrs.get("source_row_temporal_authority_sha256"),
            label="Detection source-row temporal-authority digest",
        )
        if temporal_digest != _sha256(temporal):
            raise GeometryReviewApprovalError(
                "Detection source-row temporal-authority digest is stale."
            )
        observation_key = temporal.get("observation_instance_key")
        if (
            not isinstance(observation_key, Mapping)
            or observation_key.get("ref")
            != f"/{group_path}/instances/instance_key"
            or observation_key.get("shape") != [row_count]
            or observation_key.get("content_sha256") != instance_key_sha256
        ):
            raise GeometryReviewApprovalError(
                "Detection temporal authority does not bind the exact instance keys."
            )
        source_frames = temporal.get("source_acquisition_frame_index")
        if (
            not isinstance(source_frames, Mapping)
            or source_frames.get("ref")
            != f"/{group_path}/instances/source_acquisition_frame_index"
            or source_frames.get("shape") != [row_count]
            or "source_acquisition_frame_index" not in table
        ):
            raise GeometryReviewApprovalError(
                "Detection temporal authority lacks its exact source-frame array."
            )
        source_frame_values = np.asarray(table["source_acquisition_frame_index"][:])
        if (
            tuple(source_frame_values.shape) != (row_count,)
            or source_frames.get("content_sha256")
            != array_values_sha256(source_frame_values)
        ):
            raise GeometryReviewApprovalError(
                "Detection source-frame array differs from its temporal authority."
            )
        declarations["source_row_temporal_authority"] = _positive_int(
            temporal.get("source_total_frames"),
            label="Detection source_total_frames",
        )

    storage = attrs.get("immutable_yolo_storage_validation")
    if storage is not None:
        if (
            not isinstance(storage, Mapping)
            or storage.get("schema_id")
            != "palette.immutable_yolo_storage_completion.v1"
            or storage.get("status") != "ok"
            or storage.get("stage") != "detect"
            or storage.get("row_count") != row_count
            or storage.get("errors") not in (None, [])
        ):
            raise GeometryReviewApprovalError(
                "Detection immutable-YOLO storage validation is incompatible."
            )
        declarations["immutable_yolo_storage_validation"] = _positive_int(
            storage.get("frame_count"),
            label="Detection storage-validation frame_count",
        )

    if attrs.get("validated_backend_result_count") is not None:
        declarations["validated_backend_result_count"] = _positive_int(
            attrs.get("validated_backend_result_count"),
            label="Detection validated_backend_result_count",
        )

    for name in ("frame_counts", "n_detections"):
        node = table.get(name)
        if node is None and table is not group:
            node = group.get(name)
        if node is not None:
            shape = tuple(int(value) for value in node.shape)
            if len(shape) != 1 or shape[0] <= 0:
                raise GeometryReviewApprovalError(
                    f"Detection {name} has invalid frame-domain cardinality."
                )
            declarations[f"array:{name}"] = int(shape[0])

    frame_row_offsets = table.get("frame_row_offsets")
    if frame_row_offsets is not None:
        shape = tuple(int(value) for value in frame_row_offsets.shape)
        if len(shape) != 1 or shape[0] <= 1:
            raise GeometryReviewApprovalError(
                "Detection frame_row_offsets has invalid frame-domain cardinality."
            )
        declarations["array:frame_row_offsets"] = int(shape[0]) - 1

    if not declarations:
        raise GeometryReviewApprovalError(
            "Detection source lacks an exact frame-count authority."
        )
    values = set(declarations.values())
    if len(values) != 1:
        raise GeometryReviewApprovalError(
            "Detection frame-count authorities disagree: "
            + ", ".join(f"{name}={value}" for name, value in declarations.items())
        )
    frame_count = next(iter(values))
    return frame_count, {
        "schema_id": "palette.geometry_review_detection_frame_count_binding",
        "schema_version": 1,
        "frame_count": frame_count,
        "declarations": declarations,
        "source_row_temporal_authority_sha256": temporal_digest,
    }


def detection_source_binding(root: Any, group_path: str) -> dict[str, Any]:
    """Freeze metadata identity for one exact complete canonical detection run."""

    path = _required_text(group_path, label="source_detection_group_path").strip("/")
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != "detect_runs":
        raise GeometryReviewApprovalError(
            "Approval postprocessing currently requires one exact detect_runs/<run>."
        )
    run_name = _safe_name(parts[1], label="source detection run")
    try:
        manifest = require_active_coordinate_canonical_detection(
            root,
            group_path=path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GeometryReviewApprovalError(
            f"Detection source {path!r} is not the active canonical-v3 authority: {exc}"
        ) from exc
    group = _group(root, path, label="Detection source")
    attrs = dict(group.attrs)
    if attrs.get("palette_run_completion_status") != "complete":
        raise GeometryReviewApprovalError(
            f"Detection source {path!r} is not a complete immutable run."
        )
    if manifest.get("schema_version") != (
        CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    ):
        raise GeometryReviewApprovalError(
            f"Detection source {path!r} is not a coordinate-aware canonical-v3 run."
        )
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping) or payload.get("run_id") != run_name:
        raise GeometryReviewApprovalError(
            f"Detection source {path!r} run_manifest does not bind its exact run name."
        )
    try:
        dimensions = canonical_detection_dimensions_from_manifest(manifest)
    except (TypeError, ValueError) as exc:
        raise GeometryReviewApprovalError(
            f"Detection source {path!r} has an invalid canonical run_manifest: {exc}"
        ) from exc
    table = resolve_detection_instance_table(group)
    canonical_arrays: dict[str, Any] = {}
    for canonical_path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
        name = canonical_path.removeprefix("instances/")
        if name not in table:
            raise GeometryReviewApprovalError(
                f"Detection source {path!r} lacks canonical array {canonical_path!r}."
            )
        canonical_arrays[canonical_path] = table[name]
    try:
        CANONICAL_DETECTION_SCHEMA_V1.require(
            canonical_arrays,
            dimensions=dimensions,
        )
    except (TypeError, ValueError) as exc:
        raise GeometryReviewApprovalError(
            f"Detection source {path!r} violates the canonical detection schema: {exc}"
        ) from exc
    row_count = int(table["instance_key"].shape[0])
    if row_count != int(dimensions.n_instances):
        raise GeometryReviewApprovalError(
            f"Detection source {path!r} row count disagrees with its run_manifest."
        )
    if (
        tuple(table["bbox_norm_coords"].shape) != (row_count, 4)
        or tuple(table["frame_indices"].shape) != (row_count,)
        or np.dtype(table["instance_key"].dtype) != np.dtype(np.uint64)
    ):
        raise GeometryReviewApprovalError(
            f"Detection source {path!r} has incompatible row cardinality."
        )
    instance_keys = np.asarray(table["instance_key"][:], dtype=np.uint64).reshape(-1)
    if int(np.unique(instance_keys).shape[0]) != row_count:
        raise GeometryReviewApprovalError(
            f"Detection source {path!r} contains duplicate instance_key values."
        )
    frame_indices = np.asarray(table["frame_indices"][:])
    bbox_norm_coords = np.asarray(table["bbox_norm_coords"][:])
    instance_key_sha256 = array_values_sha256(instance_keys)
    frame_count, frame_count_authority = _detection_frame_count_binding(
        group,
        table,
        group_path=path,
        attrs=attrs,
        manifest_frame_count=int(dimensions.n_frames),
        row_count=row_count,
        instance_key_sha256=instance_key_sha256,
    )
    snapshot = {
        "group_path": path,
        "run_name": run_name,
        "schema_id": manifest.get("schema_id"),
        "canonical_run_manifest_schema_version": manifest.get("schema_version"),
        "row_count": row_count,
        "frame_count": frame_count,
        "frame_count_authority": frame_count_authority,
        "source_video_width": int(dimensions.source_width),
        "source_video_height": int(dimensions.source_height),
        "instance_key_sha256": instance_key_sha256,
        "frame_indices_sha256": array_values_sha256(frame_indices),
        "bbox_norm_coords_sha256": array_values_sha256(bbox_norm_coords),
        "canonical_run_manifest_payload_digest": manifest.get("payload_digest"),
        "decoded_array_sha256": attrs.get("decoded_array_sha256"),
        "instance_key_contract": attrs.get("instance_key_contract"),
        "instance_key_frame_domain": attrs.get("instance_key_frame_domain"),
        "source_pixel_authority": resolve_detection_source_pixel_authority(attrs),
    }
    snapshot["binding_sha256"] = _sha256(snapshot)
    return _canonical(snapshot)


def _expected_subject_count(root: Any) -> int:
    attrs = dict(root.attrs)
    setup = attrs.get("experiment_setup")
    raw = setup.get("expected_subject_count") if isinstance(setup, Mapping) else None
    if raw is None:
        raw = attrs.get("subject_count")
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise GeometryReviewApprovalError(
            "Canonical Zarr lacks expected subject count."
        ) from exc
    if count <= 0:
        raise GeometryReviewApprovalError("Expected subject count must be positive.")
    return count


def build_geometry_review_approval_request(
    *,
    registry_path: str | Path,
    dataset_id: str,
    recording_id: str,
    analysis_zarr: str | Path,
    fit_review_run: str,
    acquisition_candidate_run: str,
    source_detection_group_path: str,
    selected_candidate_kind: str,
    semantic_compatibility: str,
    reviewer: str,
    reviewed_at_utc: str,
    decision_reason: str,
    palette_commit: str,
    crop_roi_width: int = DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
    crop_roi_height: int = DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
) -> GeometryReviewApprovalRequest:
    """Build one exact request without writing the registry or analysis Zarr."""

    choice = _required_text(
        selected_candidate_kind, label="selected_candidate_kind"
    ).lower()
    if choice not in APPROVAL_CHOICES:
        raise GeometryReviewApprovalError(
            "selected_candidate_kind must be palette or acquisition."
        )
    semantic = _required_text(semantic_compatibility, label="semantic_compatibility")
    if semantic not in SEMANTIC_COMPATIBILITY_STATES:
        raise GeometryReviewApprovalError(
            f"Unsupported semantic compatibility: {semantic!r}."
        )
    commit = _required_git_commit(palette_commit, label="palette_commit")
    zarr_path = Path(analysis_zarr).expanduser().resolve()
    root = open_zarr_root(zarr_path, mode="r", use_consolidated=True)
    attrs = dict(root.attrs)
    canonical_recording_id = _required_text(
        attrs.get("recording_id"), label="canonical recording_id"
    )
    expected_recording_id = _required_text(recording_id, label="recording_id")
    if canonical_recording_id != expected_recording_id:
        raise GeometryReviewApprovalError(
            "Request recording_id does not match the canonical Zarr."
        )
    fit = _fit_review_binding(root, fit_review_run)
    acquisition = _candidate_binding(root, acquisition_candidate_run)
    if acquisition["candidate_kind"] != ACQUISITION_CANDIDATE_KIND:
        raise GeometryReviewApprovalError(
            "The acquisition choice does not bind an acquisition candidate."
        )
    if acquisition["arena_binding"].get("camera_serial") != fit["camera_serial"]:
        raise GeometryReviewApprovalError(
            "Acquisition candidate and fit-review camera bindings disagree."
        )
    detection = detection_source_binding(root, source_detection_group_path)
    coordinate = acquisition["coordinate_binding"]
    if (
        int(detection.get("frame_count") or -1) != int(fit["frame_count"])
        or int(detection.get("source_video_width") or -1)
        != int(coordinate.get("native_width_px") or -2)
        or int(detection.get("source_video_height") or -1)
        != int(coordinate.get("native_height_px") or -2)
    ):
        raise GeometryReviewApprovalError(
            "Detection source dimensions or frame count disagree with the frozen "
            "recording geometry evidence."
        )
    reviewer_text = _required_text(reviewer, label="reviewer")
    reviewed_at = _required_utc_timestamp(reviewed_at_utc, label="reviewed_at_utc")
    reason = _required_text(decision_reason, label="decision_reason")
    palette_plan = plan_reviewed_palette_geometry_candidate(
        source_zarr=zarr_path,
        fit_review_run=fit["run_name"],
        reviewer=reviewer_text,
        reviewed_at_utc=reviewed_at,
    )
    planned_palette = {
        "run_name": palette_plan.candidate_id,
        "candidate_kind": PALETTE_CANDIDATE_KIND,
        "candidate_record_sha256": palette_plan.candidate_record_sha256,
    }
    registry = Path(registry_path).expanduser().resolve()
    identity = {
        "dataset": {
            "registry_path": str(registry),
            "dataset_id": _required_text(dataset_id, label="dataset_id"),
            "recording_id": expected_recording_id,
            "analysis_zarr": str(zarr_path),
            "recording_dir": str(
                Path(
                    _required_text(attrs.get("recording_path"), label="recording_path")
                )
                .expanduser()
                .resolve()
            ),
            "camera_serial": fit["camera_serial"],
            "arena_id": _required_text(attrs.get("arena_id"), label="arena_id"),
        },
        "evidence": {
            "fit_review": fit,
            "acquisition_candidate": acquisition,
            "planned_palette_candidate": planned_palette,
        },
        "detection_source": detection,
        "decision": {
            "selected_candidate_kind": choice,
            "semantic_compatibility": semantic,
            "reviewer": reviewer_text,
            "reviewed_at_utc": reviewed_at,
            "decision_reason": reason,
            "decision_source": "manual_review",
            "comparison_policy_id": MANUAL_REVIEW_POLICY_ID,
        },
        "recording_processing": {
            "video_path": fit["video_path"],
            "frame_count": fit["frame_count"],
            "expected_subject_count": _expected_subject_count(root),
            "crop_roi_width": int(crop_roi_width),
            "crop_roi_height": int(crop_roi_height),
            "crop_purpose": "zebrafish_keypoints_and_subject_masks",
        },
        "execution": {
            "palette_commit": commit,
            "required_ci_state": "success_before_submission",
        },
    }
    if int(crop_roi_width) <= 0 or int(crop_roi_height) <= 0:
        raise GeometryReviewApprovalError("Crop ROI dimensions must be positive.")
    identity_digest = _sha256(identity)
    request_id = f"geometry_review_approval_{identity_digest[:24]}"
    suffix = identity_digest[:20]
    pipeline = {
        "gate_run": f"registered_detection_gate_review_{suffix}",
        "quality_run": f"detect_quality_geometry_review_{suffix}",
        "refined_run": f"refined_detect_geometry_review_{suffix}",
        "crop_run": f"crop_geometry_review_{suffix}",
        "registered_gate_requirement": "required",
        "selection_policy_id": MANUAL_REVIEW_POLICY_ID,
    }
    payload = _canonical(
        {
            "schema_id": APPROVAL_REQUEST_SCHEMA_ID,
            "schema_version": APPROVAL_REQUEST_SCHEMA_VERSION,
            "request_id": request_id,
            "identity": identity,
            "pipeline": pipeline,
        }
    )
    return GeometryReviewApprovalRequest(
        request_id=request_id,
        request_sha256=_sha256(payload),
        payload=payload,
    )


def validate_geometry_review_approval_request(
    payload: Mapping[str, Any],
) -> GeometryReviewApprovalRequest:
    normalized = _canonical(payload)
    if normalized != dict(payload):
        raise GeometryReviewApprovalError(
            "Approval request is not canonical JSON data."
        )
    if (
        payload.get("schema_id") != APPROVAL_REQUEST_SCHEMA_ID
        or payload.get("schema_version") != APPROVAL_REQUEST_SCHEMA_VERSION
    ):
        raise GeometryReviewApprovalError("Unsupported approval request schema.")
    identity = payload.get("identity")
    pipeline = payload.get("pipeline")
    if not isinstance(identity, Mapping) or not isinstance(pipeline, Mapping):
        raise GeometryReviewApprovalError(
            "Approval request identity or pipeline is missing."
        )
    identity_digest = _sha256(identity)
    expected_id = f"geometry_review_approval_{identity_digest[:24]}"
    if payload.get("request_id") != expected_id:
        raise GeometryReviewApprovalError("Approval request identity digest disagrees.")
    decision = identity.get("decision")
    if not isinstance(decision, Mapping):
        raise GeometryReviewApprovalError("Approval request lacks decision.")
    if decision.get("selected_candidate_kind") not in APPROVAL_CHOICES:
        raise GeometryReviewApprovalError("Approval candidate choice is unsupported.")
    if decision.get("semantic_compatibility") not in SEMANTIC_COMPATIBILITY_STATES:
        raise GeometryReviewApprovalError("Approval semantic state is unsupported.")
    for name in ("reviewer", "decision_reason"):
        _required_text(decision.get(name), label=f"decision.{name}")
    if _required_utc_timestamp(
        decision.get("reviewed_at_utc"), label="decision.reviewed_at_utc"
    ) != decision.get("reviewed_at_utc"):
        raise GeometryReviewApprovalError(
            "decision.reviewed_at_utc is not canonical UTC."
        )
    suffix = identity_digest[:20]
    expected_pipeline = {
        "gate_run": f"registered_detection_gate_review_{suffix}",
        "quality_run": f"detect_quality_geometry_review_{suffix}",
        "refined_run": f"refined_detect_geometry_review_{suffix}",
        "crop_run": f"crop_geometry_review_{suffix}",
        "registered_gate_requirement": "required",
        "selection_policy_id": MANUAL_REVIEW_POLICY_ID,
    }
    if dict(pipeline) != expected_pipeline:
        raise GeometryReviewApprovalError("Approval pipeline identity disagrees.")
    return GeometryReviewApprovalRequest(
        request_id=expected_id,
        request_sha256=_sha256(payload),
        payload=normalized,
    )


def load_geometry_review_approval_request(
    path: str | Path,
) -> GeometryReviewApprovalRequest:
    request_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeometryReviewApprovalError(
            f"Approval request is unreadable: {request_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise GeometryReviewApprovalError(
            "Approval request must contain one JSON object."
        )
    return validate_geometry_review_approval_request(payload)


def persist_geometry_review_approval_request(
    request: GeometryReviewApprovalRequest,
    *,
    request_root: str | Path,
) -> Path:
    root = Path(request_root).expanduser().resolve()
    path = root / f"{request.request_id}.json"
    if path.exists():
        existing = load_geometry_review_approval_request(path)
        if existing.request_sha256 != request.request_sha256:
            raise FileExistsError(
                f"Existing approval request differs from {request.request_id}."
            )
        return path
    write_json_atomic(path, request.payload)
    persisted = load_geometry_review_approval_request(path)
    if persisted.request_sha256 != request.request_sha256:
        raise RuntimeError("Persisted approval request digest changed.")
    return path


def verify_geometry_review_registry_precondition(
    request: GeometryReviewApprovalRequest,
) -> None:
    dataset = request.payload["identity"]["dataset"]
    evidence = request.payload["identity"]["evidence"]["fit_review"]
    detection = request.payload["identity"]["detection_source"]
    try:
        queue = load_geometry_review_queue(
            dataset["registry_path"], include_inactive=True
        )
    except GeometryReviewRegistryError as exc:
        raise GeometryReviewApprovalError(str(exc)) from exc
    matches = [item for item in queue if item.dataset_id == dataset["dataset_id"]]
    if len(matches) != 1:
        raise GeometryReviewApprovalError(
            "Approval registry precondition requires one exact dataset row."
        )
    item = matches[0]
    if (
        item.recording_id != dataset["recording_id"]
        or item.zarr_path.expanduser().resolve() != Path(dataset["analysis_zarr"])
        or not item.actionable
        or item.detection_run != detection["run_name"]
        or item.detection_manifest_digest
        != detection["canonical_run_manifest_payload_digest"]
    ):
        raise GeometryReviewApprovalError(
            "Approval registry dataset or canonical detection authority is stale, "
            "non-actionable, or rebound."
        )
    fit_stage = item.stage("arena_geometry_offline_fit")
    review = fit_stage.review_status if fit_stage is not None else None
    pending_runs = review.get("runs") if isinstance(review, Mapping) else None
    if (
        fit_stage is None
        or fit_stage.status != "ok"
        or fit_stage.review_state != "evidence_complete_review_pending"
        or evidence["run_name"] not in (pending_runs or ())
    ):
        raise GeometryReviewApprovalError(
            "Approval fit-review run is not the registry's exact pending evidence."
        )


def revalidate_geometry_review_approval_sources(
    request: GeometryReviewApprovalRequest,
) -> ArenaGeometryCandidatePlan:
    """Recheck all immutable scientific inputs immediately before publication."""

    identity = request.payload["identity"]
    dataset = identity["dataset"]
    evidence = identity["evidence"]
    zarr_path = Path(dataset["analysis_zarr"]).expanduser().resolve()
    root = open_zarr_root(zarr_path, mode="r", use_consolidated=False)
    attrs = dict(root.attrs)
    if (
        attrs.get("recording_id") != dataset["recording_id"]
        or attrs.get("camera_id") != dataset["camera_serial"]
        or attrs.get("arena_id") != dataset["arena_id"]
    ):
        raise GeometryReviewApprovalError(
            "Canonical dataset identity changed after approval was frozen."
        )
    fit = _fit_review_binding(root, evidence["fit_review"]["run_name"])
    if fit != evidence["fit_review"]:
        raise GeometryReviewApprovalError(
            "Fit-review evidence changed after approval was frozen."
        )
    acquisition = _candidate_binding(
        root, evidence["acquisition_candidate"]["run_name"]
    )
    if acquisition != evidence["acquisition_candidate"]:
        raise GeometryReviewApprovalError(
            "Acquisition candidate changed after approval was frozen."
        )
    detection = detection_source_binding(
        root, identity["detection_source"]["group_path"]
    )
    if detection != identity["detection_source"]:
        raise GeometryReviewApprovalError(
            "Detection source changed after approval was frozen."
        )
    decision = identity["decision"]
    palette_plan = plan_reviewed_palette_geometry_candidate(
        source_zarr=zarr_path,
        fit_review_run=fit["run_name"],
        reviewer=decision["reviewer"],
        reviewed_at_utc=decision["reviewed_at_utc"],
    )
    planned = {
        "run_name": palette_plan.candidate_id,
        "candidate_kind": PALETTE_CANDIDATE_KIND,
        "candidate_record_sha256": palette_plan.candidate_record_sha256,
    }
    if planned != evidence["planned_palette_candidate"]:
        raise GeometryReviewApprovalError(
            "Planned Palette candidate changed after approval was frozen."
        )
    return palette_plan


__all__ = [
    "APPROVAL_CHOICES",
    "APPROVAL_REQUEST_SCHEMA_ID",
    "GeometryReviewApprovalError",
    "GeometryReviewApprovalRequest",
    "build_geometry_review_approval_request",
    "detection_source_binding",
    "load_geometry_review_approval_request",
    "persist_geometry_review_approval_request",
    "revalidate_geometry_review_approval_sources",
    "validate_geometry_review_approval_request",
    "verify_geometry_review_registry_precondition",
]
