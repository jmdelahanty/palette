"""Exact keypoint source envelope for self-contained training crop materializations.

Analysis keypoints bind a coordinate-catalog crop-v2 manifest.  Sampled training
artifacts intentionally use a compact local frame axis and therefore must not be
misrepresented as recording-level crop-v2.  This module gives that existing
training crop contract an equally strict, digest-bound source envelope while
leaving the keypoint logical arrays and dtypes unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.training_crop_materialization import (
    SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER,
    SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER,
    TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_ID,
    TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_VERSION,
    TRAINING_CROP_MATERIALIZATION_PROVIDERS,
)


TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_ID = (
    "palette.keypoint.training_crop_source_manifest"
)
TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_VERSION = 1
TRAINING_KEYPOINT_CROP_COORDINATE_SCHEMA_ID = "palette.coordinate.sampled_training_crop"
TRAINING_KEYPOINT_CROP_COORDINATE_SCHEMA_VERSION = 1
TRAINING_KEYPOINT_CROP_LOGICAL_CONTENT_SCHEMA_ID = (
    "palette.keypoint.training_crop_source_logical_content"
)
TRAINING_KEYPOINT_CROP_LOGICAL_CONTENT_SCHEMA_VERSION = 1


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _require_digest(value: object, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    return text


def _require_run_id(value: object) -> str:
    run_id = str(value).strip()
    if not run_id or "/" in run_id or run_id.startswith("."):
        raise ValueError("run_id must be one safe non-hidden group name.")
    return run_id


def _parse_training_binding(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(value) != {
        "payload",
        "payload_digest_algorithm",
        "payload_digest",
    }:
        raise ValueError("Training crop binding envelope has an unexpected field set.")
    payload = _require_mapping(
        value.get("payload"), name="training crop binding payload"
    )
    if value.get(
        "payload_digest_algorithm"
    ) != CANONICAL_JSON_DIGEST_ALGORITHM or value.get(
        "payload_digest"
    ) != canonical_json_sha256(payload):
        raise ValueError("Training crop binding digest is invalid.")
    if (
        payload.get("schema_id") != TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_ID
        or payload.get("schema_version")
        != TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_VERSION
        or payload.get("stage_selector_eligible") is not False
    ):
        raise ValueError("Training crop binding identity or lifecycle is invalid.")
    if set(payload) != {
        "schema_id",
        "schema_version",
        "provider",
        "stage_selector_eligible",
        "source",
        "dimensions",
        "array_declarations",
        "identity_array_sha256",
        "provider_evidence",
        "pixel_payload_validation",
    }:
        raise ValueError("Training crop binding payload has an unexpected field set.")
    if payload.get("provider") not in TRAINING_CROP_MATERIALIZATION_PROVIDERS:
        raise ValueError("Training crop binding provider is unsupported.")
    source = _require_mapping(payload.get("source"), name="training crop source")
    if set(source) != {"archive_path", "crop_run", "crop_path", "crop_manifest"}:
        raise ValueError("Training crop binding source has an unexpected field set.")
    dimensions = _require_mapping(
        payload.get("dimensions"), name="training crop dimensions"
    )
    if set(dimensions) != {
        "row_count",
        "roi_shape",
        "source_height",
        "source_width",
    }:
        raise ValueError(
            "Training crop binding dimensions have an unexpected field set."
        )
    evidence = _require_mapping(
        payload.get("provider_evidence"), name="training crop provider evidence"
    )
    expected_evidence_fields = {
        "source_video_pynvvc_luma": {
            "source_video_path",
            "decode_backend",
            "pixel_contract_name",
        },
        "verified_flat_roi_cache": {
            "manifest_path",
            "manifest_sha256",
            "payload_sha256",
            "verified",
            "runtime_dependency",
        },
        SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER: {
            "source_images_path",
            "source_images_dtype",
            "source_images_shape",
            "source_refined_detect_run",
            "source_frame_decision_path",
            "source_frame_decision_digest",
            "padding_mode",
            "pixel_verification",
        },
        SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER: {
            "source_images_path",
            "source_images_shape",
            "source_refined_detect_run",
            "source_frame_decision_path",
            "source_frame_decision_digest",
            "acquisition_crop_video_path",
            "acquisition_crop_video_stat",
            "acquisition_crop_meta_path",
            "acquisition_crop_meta_sha256",
            "acquisition_crop_summary_path",
            "acquisition_crop_summary_sha256",
            "acquisition_encoder_contract",
            "pixel_source_code_map",
            "fallback_reason_code_map",
            "fallback_policy",
            "decode_backend",
            "pixel_verification",
        },
    }
    if set(evidence) != expected_evidence_fields[str(payload["provider"])]:
        raise ValueError("Training crop provider evidence has an unexpected field set.")
    return payload


def build_training_keypoint_crop_source_manifest(
    *,
    run_id: str,
    recording_identity: str,
    training_crop_binding: Mapping[str, Any],
) -> dict[str, object]:
    """Build the exact source manifest consumed by raw keypoint-v2 preparation."""

    resolved_run = _require_run_id(run_id)
    resolved_recording = str(recording_identity).strip()
    if not resolved_recording:
        raise ValueError("recording_identity cannot be empty.")
    binding = dict(training_crop_binding)
    binding_payload = _parse_training_binding(binding)
    dimensions = _require_mapping(
        binding_payload.get("dimensions"), name="training crop dimensions"
    )
    declarations = _require_mapping(
        binding_payload.get("array_declarations"),
        name="training crop array declarations",
    )
    identities = _require_mapping(
        binding_payload.get("identity_array_sha256"),
        name="training crop identity digests",
    )
    if "roi_images" not in declarations or set(identities) != (
        set(declarations) - {"roi_images"}
    ):
        raise ValueError(
            "Training crop declarations and identity digests have different surfaces."
        )
    for path, declaration_value in declarations.items():
        declaration = _require_mapping(declaration_value, name=f"{path} declaration")
        shape = declaration.get("shape")
        if (
            set(declaration) != {"dtype", "shape"}
            or not isinstance(declaration.get("dtype"), str)
            or not isinstance(shape, list)
            or any(type(size) is not int or size < 0 for size in shape)
        ):
            raise ValueError(f"Training crop declaration for {path!r} is invalid.")
    for path, digest in identities.items():
        _require_digest(digest, name=f"{path} identity digest")
    required_identity = {
        "instance_key",
        "source_refined_row_ids",
        "frame_indices",
        "source_acquisition_frame_index",
        "source_frame_indices",
        "frame_row_offsets",
        "bbox_norm_coords",
        "bbox_img_xyxy",
        "centers_img_xy",
        "roi_coordinates_full",
        "roi_sizes_full",
        "source_crop_xywh",
        "bbox_roi_xyxy",
        "source_row_signature",
    }
    missing = sorted(required_identity - set(identities))
    if missing:
        raise ValueError(
            f"Training crop binding lacks keypoint source identity digests: {missing!r}."
        )
    for path in required_identity:
        if path not in declarations:
            raise ValueError(f"Training crop binding lacks declaration for {path!r}.")

    row_count = dimensions.get("row_count")
    roi_shape = dimensions.get("roi_shape")
    source_height = dimensions.get("source_height")
    source_width = dimensions.get("source_width")
    offsets_shape = _require_mapping(
        declarations["frame_row_offsets"], name="frame_row_offsets declaration"
    ).get("shape")
    if (
        type(row_count) is not int
        or row_count < 0
        or not isinstance(roi_shape, list)
        or len(roi_shape) != 2
        or any(type(value) is not int or value <= 0 for value in roi_shape)
        or type(source_height) is not int
        or source_height <= 0
        or type(source_width) is not int
        or source_width <= 0
        or not isinstance(offsets_shape, list)
        or len(offsets_shape) != 1
        or type(offsets_shape[0]) is not int
        or offsets_shape[0] <= 1
    ):
        raise ValueError("Training crop dimensions are invalid for keypoint binding.")
    n_frames = int(offsets_shape[0]) - 1

    provider_evidence = _require_mapping(
        binding_payload.get("provider_evidence"),
        name="training crop provider evidence",
    )
    if (
        binding_payload["provider"]
        == SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER
    ):
        padding_mode: object = (
            "acquisition_crop_pixels_plus_zero_padded_full_frame_fallback_v1"
        )
    else:
        padding_mode = provider_evidence.get("padding_mode")
    coordinate_document: dict[str, object] = {
        "schema_id": TRAINING_KEYPOINT_CROP_COORDINATE_SCHEMA_ID,
        "schema_version": TRAINING_KEYPOINT_CROP_COORDINATE_SCHEMA_VERSION,
        "source_kind": "self_contained_training_crop_materialization",
        "frame_index_domain": "sampled_training_local_frame",
        "source_frame_index_domain": "acquisition_camera_frame",
        "roi_coordinate_space": "sampled_crop_pixels",
        "source_coordinate_space": "source_camera_pixels",
        "roi_to_source_mapping": "integer_origin_translation_xy_v1",
        "padding_mode": padding_mode,
        "dimensions": {
            "n_frames": n_frames,
            "n_instances": int(row_count),
            "source_width": int(source_width),
            "source_height": int(source_height),
            "roi_shape": list(roi_shape),
        },
    }
    coordinate_digest = canonical_json_sha256(coordinate_document)
    logical_document: dict[str, object] = {
        "schema_id": TRAINING_KEYPOINT_CROP_LOGICAL_CONTENT_SCHEMA_ID,
        "schema_version": TRAINING_KEYPOINT_CROP_LOGICAL_CONTENT_SCHEMA_VERSION,
        "run_id": resolved_run,
        "recording_identity": resolved_recording,
        "training_crop_binding_digest": binding["payload_digest"],
        "dimensions": coordinate_document["dimensions"],
        "identity_array_sha256": {
            path: identities[path] for path in sorted(required_identity)
        },
    }
    logical_digest = canonical_json_sha256(logical_document)
    payload: dict[str, object] = {
        "run_id": resolved_run,
        "run_path": f"crop_runs/{resolved_run}",
        "recording_identity": resolved_recording,
        "stage": "crop",
        "source_kind": "self_contained_training_crop_materialization",
        "training_crop_binding": binding,
        "dimensions": {
            "n_frames": n_frames,
            "n_instances": int(row_count),
            "source_width": int(source_width),
            "source_height": int(source_height),
        },
        "coordinate_contract": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": coordinate_digest,
            "document": coordinate_document,
        },
        "logical_content": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": logical_digest,
            "document": logical_document,
        },
        "row_signatures_digest": identities["source_row_signature"],
        "publication": {
            "stage_selector_eligible": False,
            "metadata_mode": "direct_while_training_review_mutable",
        },
    }
    manifest: dict[str, object] = {
        "schema_id": TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_ID,
        "schema_version": TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    canonical_json_bytes(manifest)
    return manifest


def validate_training_keypoint_crop_source_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Deeply reconstruct one persisted training crop source manifest."""

    errors: list[str] = []
    if set(manifest) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("training keypoint crop manifest has an unexpected field set")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "training keypoint crop payload must be an object")
    if (
        manifest.get("schema_id") != TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_ID
        or manifest.get("schema_version")
        != TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_VERSION
        or manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("training keypoint crop manifest identity mismatch")
    if manifest.get("payload_digest") != canonical_json_sha256(payload):
        errors.append("training keypoint crop payload digest mismatch")
    try:
        rebuilt = build_training_keypoint_crop_source_manifest(
            run_id=payload.get("run_id"),
            recording_identity=payload.get("recording_identity"),
            training_crop_binding=_require_mapping(
                payload.get("training_crop_binding"),
                name="training crop binding",
            ),
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    else:
        if dict(manifest) != rebuilt:
            errors.append("training keypoint crop manifest differs from its builder")
    return tuple(errors)


__all__ = [
    "TRAINING_KEYPOINT_CROP_COORDINATE_SCHEMA_ID",
    "TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_ID",
    "TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_VERSION",
    "build_training_keypoint_crop_source_manifest",
    "validate_training_keypoint_crop_source_manifest",
]
