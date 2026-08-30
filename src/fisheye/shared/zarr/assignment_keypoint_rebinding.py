"""Immutable rebinding from mask-assignment evidence to active keypoint v2.

The recording subject-mask bundle seals which historical keypoint rowset was
used to split the eye mask.  A later strict-v2 keypoint publication may be
used by downstream eye analysis only after this publisher proves that the two
rowsets and the assignment-relevant scientific values are identical under the
declared float64-to-float32 normalization.  The resulting artifact is small:
it stores evidence only and never copies mask or keypoint payloads.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import zarr

from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.subject_position_keypoint_source import (
    load_keypoint_coordinate_successor_admission,
    load_keypoint_coordinate_successor_source,
)
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_bundle_coordinate_authority import (
    BoundRecordingSubjectMaskCoordinateAuthority,
    load_recording_subject_mask_coordinate_authority,
    require_bound_recording_subject_mask_coordinate_authority,
)
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
)

ASSIGNMENT_KEYPOINT_REBINDING_FAMILY = "subject_mask_assignment_keypoint_rebinding_runs"
ASSIGNMENT_KEYPOINT_REBINDING_MANIFEST_ATTR = "run_manifest"
ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_ID = (
    "palette.subject_mask.assignment_keypoint_rebinding_manifest"
)
ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_VERSION = 1
ASSIGNMENT_KEYPOINT_REBINDING_POLICY = (
    "exact_historical_assignment_to_active_keypoint_bundle_member_v1"
)
ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE = "keypoint_coordinate_successor_v1"

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_NAMES = (
    "source_crop_row_ids",
    "instance_key",
    "source_acquisition_frame_index",
)
_EQUIVALENCE_PAIRS = (
    ("source_crop_row_ids", "source_crop_row_ids", None),
    ("instance_key", "instance_key", None),
    (
        "source_acquisition_frame_index",
        "source_acquisition_frame_index",
        None,
    ),
    ("keypoints_roi", "keypoints_roi", np.dtype("float32")),
    ("detection_success", "pose_success", None),
)


class AssignmentKeypointRebindingError(ValueError):
    """Raised when an assignment dependency cannot be rebound exactly."""


def _fail(message: str) -> None:
    raise AssignmentKeypointRebindingError(message)


def _run_id(value: object, *, label: str) -> str:
    result = str(value or "").strip()
    if _RUN_ID.fullmatch(result) is None:
        _fail(f"{label} must be one safe nonempty run ID.")
    return result


def _manifest_arrays(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        arrays = manifest["payload"]["logical_content"]["document"]["arrays"]
    except (KeyError, TypeError) as exc:
        raise AssignmentKeypointRebindingError(
            "Keypoint manifest lacks its logical array inventory."
        ) from exc
    if not isinstance(arrays, Mapping):
        _fail("Keypoint manifest logical array inventory is malformed.")
    return arrays


def _assignment_collection_source_run(collection: Mapping[str, Any]) -> str:
    if (
        collection.get("schema_id")
        != "palette.subject_mask.assignment_keypoint_collection"
        or collection.get("schema_version") != 1
        or collection.get("mode") != "exact_worker_partition"
        or collection.get("row_policy") != "ordered_contiguous_recording_crop_rows_v1"
    ):
        _fail("Subject-mask assignment collection profile is unsupported.")
    n_rois = collection.get("n_rois")
    workers = collection.get("workers")
    if type(n_rois) is not int or n_rois < 0 or not isinstance(workers, list):
        _fail("Subject-mask assignment collection dimensions are invalid.")
    cursor = 0
    run_ids: set[str] = set()
    for worker in workers:
        interval = (
            worker.get("global_row_interval") if isinstance(worker, Mapping) else None
        )
        assignment = worker.get("assignment") if isinstance(worker, Mapping) else None
        if (
            not isinstance(interval, Mapping)
            or set(interval) != {"start_row", "stop_row"}
            or interval.get("start_row") != cursor
            or type(interval.get("stop_row")) is not int
            or interval["stop_row"] <= cursor
            or interval["stop_row"] > n_rois
            or not isinstance(assignment, Mapping)
            or assignment.get("assignment_keypoint_group") != "keypoints_runs"
            or assignment.get("assignment_keypoint_success_dataset")
            != "detection_success"
        ):
            _fail("Subject-mask assignment worker partition is not exact.")
        run_ids.add(
            _run_id(
                assignment.get("assignment_keypoints_run"),
                label="assignment keypoint source",
            )
        )
        cursor = int(interval["stop_row"])
    if cursor != n_rois or len(run_ids) != 1:
        _fail(
            "Subject-mask assignment collection must cover every row with one "
            "recording-wide keypoint run."
        )
    return next(iter(run_ids))


def _chunked_equivalence(
    historical: Any,
    canonical: Any,
    *,
    normalized_dtype: np.dtype[Any] | None,
    block_rows: int,
) -> dict[str, Any]:
    if len(historical.shape) == 0 or len(canonical.shape) == 0:
        _fail("Assignment equivalence arrays must have a row axis.")
    if tuple(historical.shape) != tuple(canonical.shape):
        _fail("Assignment equivalence array shapes differ.")
    target_dtype = np.dtype(normalized_dtype or historical.dtype)
    if target_dtype != np.dtype(canonical.dtype):
        _fail("Assignment equivalence dtypes differ after declared normalization.")
    historical_digest = hashlib.sha256()
    canonical_digest = hashlib.sha256()
    rows = int(historical.shape[0])
    for start in range(0, rows, block_rows):
        stop = min(rows, start + block_rows)
        left = np.ascontiguousarray(historical[start:stop], dtype=target_dtype)
        right = np.ascontiguousarray(canonical[start:stop])
        if left.shape != right.shape or not np.array_equal(
            left,
            right,
            equal_nan=True,
        ):
            _fail(f"Assignment values differ in rows [{start}, {stop}).")
        historical_digest.update(left.tobytes(order="C"))
        canonical_digest.update(right.tobytes(order="C"))
    left_sha = historical_digest.hexdigest()
    right_sha = canonical_digest.hexdigest()
    if left_sha != right_sha:
        _fail("Assignment values differ at normalized byte level.")
    return {
        "shape": [int(value) for value in canonical.shape],
        "historical_dtype": str(np.dtype(historical.dtype)),
        "canonical_dtype": str(np.dtype(canonical.dtype)),
        "normalization": (
            "identity"
            if normalized_dtype is None
            else f"numpy_astype_{target_dtype.name}_c_order_v1"
        ),
        "digest_algorithm": "sha256_c_contiguous_bytes_v1",
        "normalized_sha256": right_sha,
    }


def inspect_assignment_keypoint_rebinding(
    *,
    analysis_zarr: Path,
    subject_mask_bundle_id: str,
    keypoint_run_id: str,
    rebinding_run_id: str,
    block_rows: int = 131_072,
) -> dict[str, Any]:
    """Return one exhaustive read-only rebinding plan."""

    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    bundle_id = _run_id(subject_mask_bundle_id, label="subject_mask_bundle_id")
    requested_keypoints = _run_id(keypoint_run_id, label="keypoint_run_id")
    rebinding_id = _run_id(rebinding_run_id, label="rebinding_run_id")
    if type(block_rows) is not int or block_rows <= 0:
        _fail("block_rows must be one positive integer.")
    target_path = archive / ASSIGNMENT_KEYPOINT_REBINDING_FAMILY / rebinding_id
    if target_path.exists():
        raise FileExistsError(f"Immutable rebinding target exists: {target_path}")

    bundle = load_recording_subject_mask_coordinate_authority(
        archive,
        bundle_id=bundle_id,
        allow_inactive=True,
    )
    collection = bundle.assignment_keypoint_collection
    historical_id = _assignment_collection_source_run(collection)
    canonical_source = load_keypoint_coordinate_successor_source(
        archive,
        run_path=f"keypoints_runs/{requested_keypoints}",
    )
    root = open_zarr_root(archive, mode="r")
    canonical_run = canonical_source.run_group
    canonical_manifest = canonical_source.manifest
    active_authority = canonical_source.active_keypoint_bundle_authority
    canonical_path = f"keypoints_runs/{requested_keypoints}"
    if canonical_run.path != canonical_path:
        _fail("Keypoint coordinate-successor path differs from its authority.")
    source_crop = canonical_manifest["payload"].get("source_crop_snapshot")
    if (
        not isinstance(source_crop, Mapping)
        or source_crop.get("run_path") != bundle.crop_run_path
    ):
        _fail("Canonical keypoints and subject masks bind different crop authority.")

    historical_path = f"keypoints_runs/{historical_id}"
    try:
        historical_run = root[historical_path]
    except Exception as exc:
        raise AssignmentKeypointRebindingError(
            "Historical assignment keypoint rowset is absent."
        ) from exc
    if (
        historical_run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or historical_run.attrs.get("stage_selector_eligible") is not True
    ):
        _fail("Historical assignment keypoint rowset is not complete and selected.")

    canonical_arrays = _manifest_arrays(canonical_manifest)
    bundle_identity = bundle.bundle_manifest["payload"]["cross_binding"][
        "raw_refined_identity_array_values_sha256"
    ]
    for name in _IDENTITY_NAMES:
        declaration = canonical_arrays.get(name)
        if (
            not isinstance(declaration, Mapping)
            or declaration.get("digest_algorithm") != "sha256_c_contiguous_bytes_v1"
            or declaration.get("sha256") != bundle_identity.get(name)
        ):
            _fail(f"Canonical keypoint {name} digest differs from the mask bundle.")

    pose_schema = (
        canonical_manifest["payload"]
        .get("pose_model_schema_binding", {})
        .get("pose_schema")
    )
    labels_value = (
        pose_schema.get("keypoint_labels") if isinstance(pose_schema, Mapping) else None
    )
    historical_labels_value = historical_run.attrs.get("keypoint_labels")
    labels = list(labels_value) if isinstance(labels_value, (list, tuple)) else None
    historical_labels = (
        list(historical_labels_value)
        if isinstance(historical_labels_value, (list, tuple))
        else None
    )
    if (
        not isinstance(labels, list)
        or labels != historical_labels
        or any(not isinstance(label, str) or not label for label in labels)
        or len(labels) != len(set(labels))
        or "eye_left" not in labels
        or "eye_right" not in labels
    ):
        _fail("Historical and canonical keypoint label authorities differ.")

    equivalence: dict[str, Any] = {}
    for historical_name, canonical_name, normalized_dtype in _EQUIVALENCE_PAIRS:
        if historical_name not in historical_run or canonical_name not in canonical_run:
            _fail("Assignment equivalence input array is absent.")
        evidence = _chunked_equivalence(
            historical_run[historical_name],
            canonical_run[canonical_name],
            normalized_dtype=normalized_dtype,
            block_rows=block_rows,
        )
        if evidence["shape"][0] != bundle.n_rois:
            _fail(f"Assignment equivalence row count differs for {canonical_name}.")
        declaration = canonical_arrays.get(canonical_name)
        if (
            not isinstance(declaration, Mapping)
            or declaration.get("sha256") != evidence["normalized_sha256"]
        ):
            _fail(f"Canonical {canonical_name} values differ from its manifest.")
        equivalence[f"{historical_name}_to_{canonical_name}"] = evidence

    payload = json_attr_safe(
        {
            "rebinding_run_id": rebinding_id,
            "policy": ASSIGNMENT_KEYPOINT_REBINDING_POLICY,
            "recording_identity": bundle.recording_identity,
            "camera_identity": bundle.camera_identity,
            "row_count": bundle.n_rois,
            "assignment_state": "used",
            "subject_mask_source": {
                "bundle_id": bundle.bundle_id,
                "bundle_manifest_payload_digest": bundle.bundle_manifest[
                    "payload_digest"
                ],
                "bundle_coordinate_authority_digest": bundle.authority_digest,
                "refined_run_path": bundle.refined_run_path,
                "assignment_collection_digest": canonical_json_sha256(
                    json_attr_safe(collection)
                ),
                "historical_keypoint_run_path": historical_path,
            },
            "canonical_keypoint_source": {
                "authority_profile": ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE,
                "run_path": canonical_path,
                "run_manifest_payload_digest": canonical_manifest["payload_digest"],
                "run_manifest_document_digest": canonical_source.manifest_digest,
                "keypoint_bundle_authority_generation": active_authority["generation"],
                "keypoint_bundle_authority_digest": (
                    canonical_source.active_keypoint_bundle_authority_digest
                ),
                "coordinate_successor_authority_digest": (
                    canonical_source.successor_authority_digest
                ),
                "keypoint_labels": labels,
                "eye_keypoint_indices": {
                    "eye_left": labels.index("eye_left"),
                    "eye_right": labels.index("eye_right"),
                },
                "keypoints_dataset": "keypoints_roi",
                "success_dataset": "pose_success",
            },
            "equivalence": equivalence,
            "selection_policy": "explicit_bundle_and_keypoint_run_no_fallback_v1",
            "stage_selector_eligible": False,
            "production_state_changes": [],
        }
    )
    return {
        "schema_id": ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_ID,
        "schema_version": ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_VERSION,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def validate_assignment_keypoint_rebinding_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the closed rebinding envelope without trusting live sources."""

    errors: list[str] = []
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        return ("assignment rebinding envelope is not exact",)
    payload = manifest.get("payload")
    if (
        manifest.get("schema_id") != ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_ID
        or manifest.get("schema_version")
        != ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_VERSION
        or manifest.get("digest_algorithm") != "sha256_canonical_json_v1"
        or not isinstance(payload, Mapping)
    ):
        errors.append("assignment rebinding schema header is invalid")
        return tuple(errors)
    try:
        if manifest.get("payload_digest") != canonical_json_sha256(payload):
            errors.append("assignment rebinding payload digest differs")
    except (TypeError, ValueError):
        errors.append("assignment rebinding payload is not strict JSON")
        return tuple(errors)
    expected = {
        "rebinding_run_id",
        "policy",
        "recording_identity",
        "camera_identity",
        "row_count",
        "assignment_state",
        "subject_mask_source",
        "canonical_keypoint_source",
        "equivalence",
        "selection_policy",
        "stage_selector_eligible",
        "production_state_changes",
    }
    if set(payload) != expected:
        errors.append("assignment rebinding payload fields are not exact")
    if (
        payload.get("policy") != ASSIGNMENT_KEYPOINT_REBINDING_POLICY
        or payload.get("assignment_state") != "used"
        or payload.get("selection_policy")
        != "explicit_bundle_and_keypoint_run_no_fallback_v1"
        or payload.get("stage_selector_eligible") is not False
        or payload.get("production_state_changes") != []
        or type(payload.get("row_count")) is not int
        or payload.get("row_count") < 0
    ):
        errors.append("assignment rebinding lifecycle or dimensions are invalid")
    try:
        _run_id(payload.get("rebinding_run_id"), label="rebinding_run_id")
    except AssignmentKeypointRebindingError:
        errors.append("assignment rebinding run ID is invalid")
    if (
        not isinstance(payload.get("recording_identity"), str)
        or not payload["recording_identity"]
        or not isinstance(payload.get("camera_identity"), str)
        or not payload["camera_identity"]
    ):
        errors.append("assignment rebinding recording identity is invalid")

    subject = payload.get("subject_mask_source")
    subject_fields = {
        "bundle_id",
        "bundle_manifest_payload_digest",
        "bundle_coordinate_authority_digest",
        "refined_run_path",
        "assignment_collection_digest",
        "historical_keypoint_run_path",
    }
    if not isinstance(subject, Mapping) or set(subject) != subject_fields:
        errors.append("subject-mask rebinding source is not exact")
    else:
        try:
            _run_id(subject.get("bundle_id"), label="subject-mask bundle ID")
        except AssignmentKeypointRebindingError:
            errors.append("subject-mask bundle ID is invalid")
        for name in (
            "bundle_manifest_payload_digest",
            "bundle_coordinate_authority_digest",
            "assignment_collection_digest",
        ):
            if _SHA256.fullmatch(str(subject.get(name) or "")) is None:
                errors.append(f"subject-mask {name} is invalid")
        for name, prefix in (
            ("refined_run_path", "refined_subject_masks_runs/"),
            ("historical_keypoint_run_path", "keypoints_runs/"),
        ):
            path = subject.get(name)
            if (
                not isinstance(path, str)
                or not path.startswith(prefix)
                or path.count("/") != 1
            ):
                errors.append(f"subject-mask {name} is invalid")

    keypoints = payload.get("canonical_keypoint_source")
    keypoint_fields = {
        "authority_profile",
        "run_path",
        "run_manifest_payload_digest",
        "run_manifest_document_digest",
        "keypoint_bundle_authority_generation",
        "keypoint_bundle_authority_digest",
        "coordinate_successor_authority_digest",
        "keypoint_labels",
        "eye_keypoint_indices",
        "keypoints_dataset",
        "success_dataset",
    }
    if not isinstance(keypoints, Mapping) or set(keypoints) != keypoint_fields:
        errors.append("canonical keypoint source is not exact")
    else:
        run_path = keypoints.get("run_path")
        labels = keypoints.get("keypoint_labels")
        indices = keypoints.get("eye_keypoint_indices")
        if (
            keypoints.get("authority_profile") != ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE
            or not isinstance(run_path, str)
            or not run_path.startswith("keypoints_runs/")
            or run_path.count("/") != 1
            or type(keypoints.get("keypoint_bundle_authority_generation")) is not int
            or keypoints["keypoint_bundle_authority_generation"] <= 0
            or keypoints.get("keypoints_dataset") != "keypoints_roi"
            or keypoints.get("success_dataset") != "pose_success"
        ):
            errors.append("canonical keypoint source profile is invalid")
        for name in (
            "run_manifest_payload_digest",
            "run_manifest_document_digest",
            "keypoint_bundle_authority_digest",
            "coordinate_successor_authority_digest",
        ):
            if _SHA256.fullmatch(str(keypoints.get(name) or "")) is None:
                errors.append(f"canonical keypoint {name} is invalid")
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label for label in labels)
            or len(labels) != len(set(labels))
            or not isinstance(indices, Mapping)
            or set(indices) != {"eye_left", "eye_right"}
            or any(
                type(indices.get(name)) is not int
                or indices[name] < 0
                or indices[name] >= len(labels)
                or labels[indices[name]] != name
                for name in ("eye_left", "eye_right")
            )
        ):
            errors.append("canonical keypoint label authority is invalid")

    equivalence = payload.get("equivalence")
    expected_equivalence = {
        f"{historical}_to_{canonical}"
        for historical, canonical, _dtype in _EQUIVALENCE_PAIRS
    }
    if not isinstance(equivalence, Mapping) or set(equivalence) != expected_equivalence:
        errors.append("assignment equivalence inventory is not exact")
    else:
        evidence_fields = {
            "shape",
            "historical_dtype",
            "canonical_dtype",
            "normalization",
            "digest_algorithm",
            "normalized_sha256",
        }
        for name, evidence in equivalence.items():
            if not isinstance(evidence, Mapping) or set(evidence) != evidence_fields:
                errors.append(f"assignment equivalence {name} is not exact")
                continue
            shape = evidence.get("shape")
            try:
                historical_dtype = np.dtype(evidence.get("historical_dtype"))
                canonical_dtype = np.dtype(evidence.get("canonical_dtype"))
            except (TypeError, ValueError):
                errors.append(f"assignment equivalence {name} dtype is invalid")
                continue
            if (
                not isinstance(shape, list)
                or not shape
                or any(type(value) is not int or value < 0 for value in shape)
                or evidence.get("digest_algorithm") != "sha256_c_contiguous_bytes_v1"
                or _SHA256.fullmatch(str(evidence.get("normalized_sha256") or ""))
                is None
                or str(historical_dtype) != evidence.get("historical_dtype")
                or str(canonical_dtype) != evidence.get("canonical_dtype")
                or not isinstance(evidence.get("normalization"), str)
            ):
                errors.append(f"assignment equivalence {name} is invalid")
    return tuple(errors)


def load_assignment_keypoint_rebinding_manifest(
    analysis_zarr: Path,
    *,
    rebinding_run_id: str,
    subject_mask_authority: BoundRecordingSubjectMaskCoordinateAuthority | None = None,
) -> dict[str, Any]:
    """Load one complete rebinding and revalidate both live authorities."""

    archive = analysis_zarr.expanduser().resolve()
    run_id = _run_id(rebinding_run_id, label="rebinding_run_id")
    run_path = f"{ASSIGNMENT_KEYPOINT_REBINDING_FAMILY}/{run_id}"
    direct = open_zarr_root(archive, mode="r")
    consolidated = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )
    manifests: list[Mapping[str, Any]] = []
    for view in (direct, consolidated):
        try:
            run = view[run_path]
        except Exception as exc:
            raise AssignmentKeypointRebindingError(
                f"Assignment rebinding is absent: {run_path}."
            ) from exc
        manifest = run.attrs.get(ASSIGNMENT_KEYPOINT_REBINDING_MANIFEST_ATTR)
        errors = (
            validate_assignment_keypoint_rebinding_manifest(manifest)
            if isinstance(manifest, Mapping)
            else ("assignment rebinding manifest is absent",)
        )
        if (
            errors
            or run.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
            or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            or run.attrs.get("stage_selector_eligible") is not False
            or run.attrs.get("production_candidate") is not True
        ):
            _fail(
                "Assignment rebinding is not one complete ineligible publication: "
                + "; ".join(errors)
            )
        manifests.append(manifest)
    if manifests[0] != manifests[1]:
        _fail("Direct and consolidated assignment rebinding manifests disagree.")
    manifest = copy.deepcopy(dict(manifests[0]))
    payload = manifest["payload"]

    subject = payload["subject_mask_source"]
    if subject_mask_authority is None:
        bundle = load_recording_subject_mask_coordinate_authority(
            archive,
            bundle_id=str(subject["bundle_id"]),
            allow_inactive=True,
        )
    else:
        bundle = require_bound_recording_subject_mask_coordinate_authority(
            subject_mask_authority
        )
        if bundle.archive_path != archive:
            _fail("Provided subject-mask authority belongs to another archive.")
    if (
        bundle.recording_identity != payload["recording_identity"]
        or bundle.camera_identity != payload["camera_identity"]
        or bundle.n_rois != payload["row_count"]
        or bundle.bundle_manifest.get("payload_digest")
        != subject.get("bundle_manifest_payload_digest")
        or bundle.authority_digest != subject.get("bundle_coordinate_authority_digest")
        or bundle.refined_run_path != subject.get("refined_run_path")
        or canonical_json_sha256(json_attr_safe(bundle.assignment_keypoint_collection))
        != subject.get("assignment_collection_digest")
    ):
        _fail("Subject-mask authority changed after assignment rebinding.")

    keypoints = payload["canonical_keypoint_source"]
    source = load_keypoint_coordinate_successor_admission(
        archive,
        run_path=str(keypoints["run_path"]),
    )
    authority = source.active_keypoint_bundle_authority
    raw_manifest = source.manifest
    if (
        source.active_keypoint_bundle_authority_digest
        != keypoints.get("keypoint_bundle_authority_digest")
        or authority.get("generation")
        != keypoints.get("keypoint_bundle_authority_generation")
        or source.run_path != keypoints.get("run_path")
        or source.successor_authority_digest
        != keypoints.get("coordinate_successor_authority_digest")
        or not isinstance(raw_manifest, Mapping)
        or raw_manifest.get("payload_digest")
        != keypoints.get("run_manifest_payload_digest")
        or source.manifest_digest != keypoints.get("run_manifest_document_digest")
    ):
        _fail("Canonical keypoint authority changed after assignment rebinding.")
    return manifest


def publish_assignment_keypoint_rebinding(
    *,
    analysis_zarr: Path,
    subject_mask_bundle_id: str,
    keypoint_run_id: str,
    rebinding_run_id: str,
    block_rows: int = 131_072,
) -> dict[str, Any]:
    """Publish one immutable evidence-only rebinding after exhaustive proof."""

    archive = analysis_zarr.expanduser().resolve()
    initial = inspect_assignment_keypoint_rebinding(
        analysis_zarr=archive,
        subject_mask_bundle_id=subject_mask_bundle_id,
        keypoint_run_id=keypoint_run_id,
        rebinding_run_id=rebinding_run_id,
        block_rows=block_rows,
    )
    run_id = str(initial["payload"]["rebinding_run_id"])
    with archive_metadata_publication_lock(archive):
        checked = inspect_assignment_keypoint_rebinding(
            analysis_zarr=archive,
            subject_mask_bundle_id=subject_mask_bundle_id,
            keypoint_run_id=keypoint_run_id,
            rebinding_run_id=run_id,
            block_rows=block_rows,
        )
        if checked != initial:
            _fail("Assignment evidence changed between inspection and publication.")
        root = open_zarr_root(archive, mode="a")
        parent = root.require_group(ASSIGNMENT_KEYPOINT_REBINDING_FAMILY)
        if run_id in parent:
            raise FileExistsError(f"Immutable assignment rebinding exists: {run_id}")
        run = parent.create_group(run_id)
        owner = uuid4().hex
        try:
            mark_run_started(
                run, run_name=run_id, stage="assignment_keypoint_rebinding"
            )
            run.attrs.update(
                {
                    "status": RUN_STATUS_COMPLETE,
                    "stage_selector_eligible": False,
                    "production_candidate": True,
                    "publication_owner_uuid": owner,
                    ASSIGNMENT_KEYPOINT_REBINDING_MANIFEST_ATTR: initial,
                }
            )
            mark_run_complete(run, run_name=run_id)
            consolidate_metadata_capture_expected_warnings(archive)
            direct = open_zarr_root(archive, mode="r")
            consolidated = zarr.open_group(
                str(archive), mode="r", zarr_format=3, use_consolidated=True
            )
            for view in (direct, consolidated):
                persisted = view[f"{ASSIGNMENT_KEYPOINT_REBINDING_FAMILY}/{run_id}"]
                if (
                    persisted.attrs.get(RUN_COMPLETION_CONTRACT_ATTR)
                    != RUN_COMPLETION_CONTRACT
                    or persisted.attrs.get(RUN_COMPLETION_STATUS_ATTR)
                    != RUN_STATUS_COMPLETE
                    or persisted.attrs.get("stage_selector_eligible") is not False
                    or persisted.attrs.get(ASSIGNMENT_KEYPOINT_REBINDING_MANIFEST_ATTR)
                    != initial
                ):
                    _fail("Published assignment rebinding did not persist exactly.")
        except BaseException as exc:
            try:
                run.attrs["stage_selector_eligible"] = False
                mark_run_failed(run, run_name=run_id, error=str(exc))
                consolidate_metadata_capture_expected_warnings(archive)
            except BaseException:
                pass
            raise
    return {
        "status": "complete",
        "published_at_utc": utc_now(),
        "analysis_zarr": str(archive),
        "run_path": f"{ASSIGNMENT_KEYPOINT_REBINDING_FAMILY}/{run_id}",
        "manifest": copy.deepcopy(initial),
        "selector_eligible": False,
        "production_state_changes": [],
    }


__all__ = [
    "ASSIGNMENT_KEYPOINT_REBINDING_FAMILY",
    "ASSIGNMENT_KEYPOINT_REBINDING_MANIFEST_ATTR",
    "ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_ID",
    "ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_VERSION",
    "ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE",
    "AssignmentKeypointRebindingError",
    "inspect_assignment_keypoint_rebinding",
    "load_assignment_keypoint_rebinding_manifest",
    "publish_assignment_keypoint_rebinding",
    "validate_assignment_keypoint_rebinding_manifest",
]
