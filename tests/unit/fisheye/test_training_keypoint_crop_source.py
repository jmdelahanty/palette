from __future__ import annotations

import copy

import pytest

from fisheye.shared.zarr.keypoint_manifest import (
    keypoint_crop_source_from_manifest,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.training_crop_materialization import (
    ACQUISITION_HYBRID_FALLBACK_REASON_CODE_MAP,
    ACQUISITION_HYBRID_PIXEL_SOURCE_CODE_MAP,
    SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER,
    TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_ID,
    TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_VERSION,
)
from fisheye.shared.zarr.training_keypoint_crop_source import (
    TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_ID,
    build_training_keypoint_crop_source_manifest,
    validate_training_keypoint_crop_source_manifest,
)


def _binding() -> dict[str, object]:
    declarations = {
        "instance_key": {"dtype": "<u8", "shape": [3]},
        "source_refined_row_ids": {"dtype": "<i8", "shape": [3]},
        "frame_indices": {"dtype": "<i8", "shape": [3]},
        "source_acquisition_frame_index": {"dtype": "<i8", "shape": [3]},
        "frame_row_offsets": {"dtype": "<i8", "shape": [5]},
        "bbox_norm_coords": {"dtype": "<f4", "shape": [3, 4]},
        "bbox_img_xyxy": {"dtype": "<f4", "shape": [3, 4]},
        "centers_img_xy": {"dtype": "<f4", "shape": [3, 2]},
        "roi_coordinates_full": {"dtype": "<i4", "shape": [3, 2]},
        "roi_sizes_full": {"dtype": "<i4", "shape": [3, 2]},
        "source_crop_xywh": {"dtype": "<f4", "shape": [3, 4]},
        "bbox_roi_xyxy": {"dtype": "<f4", "shape": [3, 4]},
        "source_row_signature": {"dtype": "|u1", "shape": [3, 32]},
        "source_frame_indices": {"dtype": "<i8", "shape": [3]},
        "roi_images": {"dtype": "|u1", "shape": [3, 348, 348]},
    }
    identities = {
        name: canonical_json_sha256({"path": name})
        for name in declarations
        if name != "roi_images"
    }
    payload = {
        "schema_id": TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_ID,
        "schema_version": TRAINING_CROP_MATERIALIZATION_BINDING_SCHEMA_VERSION,
        "provider": "sampled_training_images_full",
        "stage_selector_eligible": False,
        "source": {
            "archive_path": "/tmp/source.zarr",
            "crop_run": "refined_review",
            "crop_path": "refined_detect_runs/refined_review/instances",
            "crop_manifest": {"authority_kind": "sampled_detection_review"},
        },
        "dimensions": {
            "row_count": 3,
            "roi_shape": [348, 348],
            "source_height": 4512,
            "source_width": 4512,
        },
        "array_declarations": declarations,
        "identity_array_sha256": identities,
        "provider_evidence": {
            "source_images_path": "raw_video/images_full",
            "source_images_dtype": "uint8",
            "source_images_shape": [4, 4512, 4512],
            "source_refined_detect_run": "refined_review",
            "source_frame_decision_path": ("detect_frame_decision_runs/refined_review"),
            "source_frame_decision_digest": "d" * 64,
            "padding_mode": "zero_outside_source_frame",
            "pixel_verification": "all_rows_byte_equal_to_source_window_v1",
        },
        "pixel_payload_validation": (
            "physical_publication_checksum_plus_provider_evidence_v1"
        ),
    }
    return {
        "payload": payload,
        "payload_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
    }


def test_training_crop_source_builds_keypoint_reference_without_relabelling_axis() -> (
    None
):
    manifest = build_training_keypoint_crop_source_manifest(
        run_id="crop_reviewed_348",
        recording_identity="batman_canary",
        training_crop_binding=_binding(),
    )
    assert manifest["schema_id"] == TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_ID
    assert validate_training_keypoint_crop_source_manifest(manifest) == ()

    source = keypoint_crop_source_from_manifest(manifest)
    assert source.run_id == "crop_reviewed_348"
    assert source.n_frames == 4
    assert source.n_instances == 3
    coordinate = manifest["payload"]["coordinate_contract"]["document"]
    assert coordinate["frame_index_domain"] == "sampled_training_local_frame"
    assert coordinate["source_frame_index_domain"] == "acquisition_camera_frame"


def test_training_crop_source_rejects_recomputed_digest_nested_tampering() -> None:
    manifest = build_training_keypoint_crop_source_manifest(
        run_id="crop_reviewed_348",
        recording_identity="batman_canary",
        training_crop_binding=_binding(),
    )
    tampered = copy.deepcopy(manifest)
    tampered["payload"]["coordinate_contract"]["document"]["padding_mode"] = "reflect"
    tampered["payload"]["coordinate_contract"]["digest"] = canonical_json_sha256(
        tampered["payload"]["coordinate_contract"]["document"]
    )
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    errors = validate_training_keypoint_crop_source_manifest(tampered)
    assert "differs from its builder" in "; ".join(errors)


def test_training_crop_source_rejects_extra_binding_fields() -> None:
    binding = _binding()
    binding["payload"]["unexpected"] = "accepted_if_only_digest_checked"
    binding["payload_digest"] = canonical_json_sha256(binding["payload"])

    with pytest.raises(ValueError, match="unexpected field set"):
        build_training_keypoint_crop_source_manifest(
            run_id="crop_reviewed_348",
            recording_identity="batman_canary",
            training_crop_binding=binding,
        )


def test_training_crop_source_accepts_exact_acquisition_crop_hybrid() -> None:
    binding = _binding()
    payload = binding["payload"]
    payload["provider"] = SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER
    payload["provider_evidence"] = {
        "source_images_path": "raw_video/images_full",
        "source_images_shape": [4, 4512, 4512],
        "source_refined_detect_run": "reviewed",
        "source_frame_decision_path": "detect_frame_decision_runs/reviewed",
        "source_frame_decision_digest": "d" * 64,
        "acquisition_crop_video_path": "/recording/crop.mp4",
        "acquisition_crop_video_stat": {
            "strategy": "stat_v1",
            "size_bytes": 123,
            "mtime_ns": 456,
        },
        "acquisition_crop_meta_path": "/recording/crop_meta.csv",
        "acquisition_crop_meta_sha256": "e" * 64,
        "acquisition_crop_summary_path": "/recording/crop_summary.json",
        "acquisition_crop_summary_sha256": "f" * 64,
        "acquisition_encoder_contract": {
            "schema_id": "palette.orange_acquisition_crop_encoder_binding",
            "schema_version": 1,
        },
        "pixel_source_code_map": ACQUISITION_HYBRID_PIXEL_SOURCE_CODE_MAP,
        "fallback_reason_code_map": ACQUISITION_HYBRID_FALLBACK_REASON_CODE_MAP,
        "fallback_policy": "sampled_images_full_zero_padded_v1",
        "decode_backend": "pynvvc_luma_exact_gop1_seek_torch",
        "pixel_verification": "all_rows_byte_equal_to_declared_provider_v1",
    }
    for name in (
        "source_training_row_indices",
        "source_crop_meta_row_indices",
        "source_crop_video_frame_indices",
        "source_crop_local_frame_ids",
        "pixel_source_codes",
        "fallback_reason_codes",
    ):
        payload["array_declarations"][name] = {"dtype": "<i8", "shape": [3]}
        payload["identity_array_sha256"][name] = canonical_json_sha256(
            {"path": name}
        )
    binding["payload_digest"] = canonical_json_sha256(payload)

    manifest = build_training_keypoint_crop_source_manifest(
        run_id="crop_reviewed_384",
        recording_identity="batman_canary",
        training_crop_binding=binding,
    )

    coordinate = manifest["payload"]["coordinate_contract"]["document"]
    assert coordinate["padding_mode"] == (
        "acquisition_crop_pixels_plus_zero_padded_full_frame_fallback_v1"
    )
    assert validate_training_keypoint_crop_source_manifest(manifest) == ()
