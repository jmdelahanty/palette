"""Seal one authorized source evaluation for immutable position publication.

This is the integration boundary between strict persisted-source adapters, the
pure Phase 1 evaluator, and the generic Phase 2 materializer.  It selects no
source and resolves no ``latest`` pointer: callers provide one already-bound
source and one exact estimator ID.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from fisheye.shared.subject_position_prepared_input import (
    SubjectPositionPreparedInput,
)
from fisheye.shared.coordinate_descriptor import (
    CanonicalFrameRecord,
    DigestBoundCoordinateRecordRef,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    build_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.coordinate_surface_contract import SOURCE_CAMERA_POINT_XY
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.subject_position_contract import (
    require_estimator_anatomy_expression,
)
from fisheye.shared.subject_position_detection_source import (
    BoundDetectionPositionSource,
    require_bound_detection_position_source,
)
from fisheye.shared.subject_position_expression import (
    DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
    KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
    estimator_profile_digest,
    evaluate_estimator_profile,
    get_estimator_profile,
)
from fisheye.shared.subject_position_keypoint_source import (
    BoundKeypointPositionSource,
    revalidate_bound_keypoint_position_source,
)
from fisheye.shared.subject_position_mask_source import (
    BoundSubjectMaskPositionSource,
)
from fisheye.shared.subject_position_policy import (
    SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID,
    get_subject_position_selection_policy,
    subject_position_selection_policy_digest,
)
from fisheye.shared.subject_position_storage import (
    canonical_source_camera_coordinate_metadata,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SUBJECT_POSITION_SOURCE_AUTHORITY_SCHEMA_ID = (
    "palette.subject_position_source_authority"
)
SUBJECT_POSITION_SOURCE_AUTHORITY_SCHEMA_VERSION = 1
NON_ANATOMICAL_POSITION_BINDING_SCHEMA_ID = (
    "palette.subject_position_non_anatomical_binding"
)

_RECIPE_BY_ESTIMATOR = {
    KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID: "head_triad_equal_mean",
    MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID: "head_triad_equal_mean",
    SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID: "subject_body_centroid",
}


class SubjectPositionPreparationError(ValueError):
    """Raised when a sealed source cannot authorize one estimator result."""


def _canonical_record(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise SubjectPositionPreparationError(f"{name} must be a nonempty mapping.")
    try:
        payload = json.dumps(
            json_attr_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SubjectPositionPreparationError(
            f"{name} is not canonical JSON: {exc}."
        ) from exc
    result = json.loads(payload)
    if not isinstance(result, dict):  # pragma: no cover - defensive
        raise SubjectPositionPreparationError(f"{name} must encode one object.")
    return result


def _require_source(
    value: object,
) -> (
    BoundDetectionPositionSource
    | BoundKeypointPositionSource
    | BoundSubjectMaskPositionSource
):
    if type(value) is BoundDetectionPositionSource:
        return require_bound_detection_position_source(value)
    if type(value) is BoundKeypointPositionSource:
        return revalidate_bound_keypoint_position_source(value)
    if type(value) is BoundSubjectMaskPositionSource:
        return value.revalidate()
    raise SubjectPositionPreparationError(
        "Position preparation requires one sealed detection, keypoint, or "
        "subject-mask source adapter result."
    )


def _coordinate_record(source: Any) -> dict[str, Any]:
    frame = source.source_camera_frame
    identity = source.row_identity
    try:
        endpoint = frame.endpoint
        descriptor = build_canonical_coordinate_descriptor(
            **SOURCE_CAMERA_POINT_XY.descriptor_kwargs(),
            reference_width=int(endpoint.width),
            reference_height=int(endpoint.height),
            reference_authority=DigestBoundCoordinateRecordRef(
                record_ref=frame.record_ref,
                record_sha256=frame.record_sha256,
            ),
            # ``BoundPixelFrameAuthority.endpoint.selector`` names the
            # persisted attribute that stores the pixel-frame record.  A
            # canonical descriptor that also supplies ``frame_record`` must
            # select that complete typed record, not repeat its attribute
            # name as an extent selector.
            reference_selector="record",
            row_identity_contract=identity.contract,
            row_identity_record_ref=identity.record_ref,
            overlay_transform_refs=(),
            frame_record=CanonicalFrameRecord(
                kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
                record_ref=frame.record_ref,
                record_sha256=frame.record_sha256,
            ),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise SubjectPositionPreparationError(
            f"Source-camera coordinate authority is incomplete: {exc}."
        ) from exc
    return canonical_source_camera_coordinate_metadata(descriptor)


def _source_evidence(source: Any) -> dict[str, Any]:
    if type(source) is BoundDetectionPositionSource:
        return {
            "adapter_record": source.source_binding_record,
            "adapter_record_sha256": source.source_binding_digest,
        }
    if type(source) is BoundKeypointPositionSource:
        return {
            "authority_mode": getattr(source, "authority_mode", "canonical_selector"),
            "keypoint_bundle_authority": getattr(
                source, "keypoint_bundle_authority", None
            ),
            "keypoint_bundle_authority_sha256": getattr(
                source, "keypoint_bundle_authority_digest", None
            ),
            "anatomy_source_binding": source.source_binding_record,
            "anatomy_source_binding_sha256": source.source_binding_digest,
            "run_manifest_sha256": source.run_manifest_digest,
            "logical_content_sha256": source.logical_content_digest,
            "metadata_declarations_sha256": source.metadata_declarations_digest,
            "skeleton_id": source.skeleton_id,
            "skeleton_sha256": source.skeleton_digest,
            "pose_schema_binding_sha256": source.pose_schema_binding_digest,
        }
    return {
        "authority_mode": getattr(source, "authority_mode", "family_selector"),
        "bundle_run_path": getattr(source, "bundle_run_path", None),
        "anatomy_source_binding": source.source_binding_record,
        "anatomy_source_binding_sha256": source.source_binding_digest,
        "source_payload_sha256": source.source_payload_digest,
        "anatomy_profile_sha256": source.anatomy_profile_digest,
        "direct_consolidated_evidence": source.direct_consolidated_evidence,
    }


def _source_record(source: Any) -> dict[str, Any]:
    record = {
        "schema_id": SUBJECT_POSITION_SOURCE_AUTHORITY_SCHEMA_ID,
        "schema_version": SUBJECT_POSITION_SOURCE_AUTHORITY_SCHEMA_VERSION,
        "source_modality": source.source_modality,
        "source_kind": source.source_kind,
        "run_path": source.run_path,
        "row_axis": "observation_instance",
        "row_identity": {
            "record_ref": source.row_identity.record_ref,
            "record_sha256": source.row_identity.record_sha256,
            "key_content_sha256": array_values_sha256(source.instance_key),
        },
        "source_arrays": {
            "instance_key_sha256": array_values_sha256(source.instance_key),
            "source_acquisition_frame_index_sha256": array_values_sha256(
                source.source_acquisition_frame_index
            ),
            "source_row_index_sha256": array_values_sha256(
                source.source_row_index
            ),
        },
        "source_camera_frame": {
            "record_ref": source.source_camera_frame.record_ref,
            "record_sha256": source.source_camera_frame.record_sha256,
        },
        "adapter_evidence": _source_evidence(source),
    }
    return _canonical_record(record, name="source authority")


def _anatomy_record(source: Any, estimator: Mapping[str, Any]) -> dict[str, Any]:
    estimator_id = str(estimator["estimator_id"])
    if estimator_id == DETECTION_BBOX_CENTROID_ESTIMATOR_ID:
        return {
            "schema_id": NON_ANATOMICAL_POSITION_BINDING_SCHEMA_ID,
            "schema_version": 1,
            "anatomy_profile_id": None,
            "source_modality": "detection",
            "estimator_id": estimator_id,
            "expression": estimator["expression"],
        }
    try:
        recipe_id = _RECIPE_BY_ESTIMATOR[estimator_id]
        resolved = require_estimator_anatomy_expression(
            estimator,
            source.anatomy_profile,
            binding_id=source.binding_id,
            recipe_id=recipe_id,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise SubjectPositionPreparationError(
            f"Estimator anatomy binding is incompatible: {exc}."
        ) from exc
    return resolved.record


def _result_arrays(source: Any, result: Any) -> dict[str, np.ndarray]:
    arrays = {
        "position_xy": result.position_xy,
        "valid": result.valid,
        "failure_reason_codes": result.failure_reason_codes,
        "instance_key": source.instance_key,
        "source_acquisition_frame_index": source.source_acquisition_frame_index,
        "source_row_index": source.source_row_index,
    }
    optional = {
        "support/source_points_xy": result.source_points_xy,
        "support/source_points_valid": result.source_points_valid,
        "support/source_point_reason_codes": result.source_point_reason_codes,
        "support/source_point_confidence": result.source_point_confidence,
    }
    arrays.update({path: value for path, value in optional.items() if value is not None})
    return arrays


def prepare_subject_position_input(
    source: object,
    *,
    estimator_id: str,
    software_record: Mapping[str, Any],
    policy_id: str = SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID,
) -> SubjectPositionPreparedInput:
    """Evaluate one exact provider and return a publication-ready immutable input."""

    bound = _require_source(source)
    estimator = get_estimator_profile(estimator_id)
    if estimator["source_modality"] != bound.source_modality:
        raise SubjectPositionPreparationError(
            "Estimator modality differs from the sealed source modality."
        )
    policy = get_subject_position_selection_policy(policy_id)
    if estimator_id not in policy["allowed_estimator_ids"]:
        raise SubjectPositionPreparationError(
            "Estimator is not allowed by the explicit no-default canary policy."
        )
    result = evaluate_estimator_profile(
        estimator,
        bound.expression_bindings,
        row_count=int(bound.instance_key.shape[0]),
    )
    anatomy = _canonical_record(
        _anatomy_record(bound, estimator), name="anatomy binding"
    )
    source_record = _source_record(bound)
    software = _canonical_record(software_record, name="software record")
    coordinate = _coordinate_record(bound)
    return SubjectPositionPreparedInput(
        arrays=_result_arrays(bound, result),
        estimator_record=estimator,
        estimator_sha256=estimator_profile_digest(estimator),
        anatomy_record=anatomy,
        anatomy_sha256=canonical_json_sha256(anatomy),
        source_record=source_record,
        source_sha256=canonical_json_sha256(source_record),
        policy_record=policy,
        policy_sha256=subject_position_selection_policy_digest(policy),
        software_record=software,
        software_sha256=canonical_json_sha256(software),
        coordinate_record=coordinate,
        coordinate_sha256=canonical_json_sha256(coordinate),
    )


__all__ = [
    "NON_ANATOMICAL_POSITION_BINDING_SCHEMA_ID",
    "SUBJECT_POSITION_SOURCE_AUTHORITY_SCHEMA_ID",
    "SUBJECT_POSITION_SOURCE_AUTHORITY_SCHEMA_VERSION",
    "SubjectPositionPreparationError",
    "prepare_subject_position_input",
]
