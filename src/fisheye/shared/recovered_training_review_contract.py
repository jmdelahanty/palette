"""Versioned ROI-only annotation identity for recovered training archives."""

from fisheye.shared.run_provenance import sha256_payload

REVIEW_SCHEMA = "palette.training.recovered_mask_tail_review.v1"
COORDINATE_SYSTEM = "recovered_training_roi_xy"
# Explicit native ROI snapshot contract; historical recovery identities stay fixed.
NATIVE_REVIEW_SCHEMA = "palette.training.native_mask_tail_review.v1"
NATIVE_COORDINATE_SYSTEM = "native_training_roi_xy"


def initial_contract_digest(group):
    """Bind scientific/identity attrs, excluding operational and edit status.

    The initial payload hashes stay fixed when editable annotations change.
    This digest validates their declaration; full initial-content validation
    additionally reads those arrays and is inappropriate after manual edits.
    """
    keys = (
        "schema_id",
        "schema_version",
        "stage_selector_eligible",
        "coordinate_system",
        "frame_index_domain",
        "sensor_pixel_origin_available",
        "row_count",
        "source_bindings",
        "source_crop_run",
        "mask_labels",
        "label_schema_id",
        "method",
        "pose_schema",
        "skeleton_id",
        "keypoint_labels",
        "kpt_shape",
        "keypoint_origin_codes",
        "derivation_recipe",
        "source_subject_mask_run",
        "source_mask_sha256",
        "source_seed_run",
        "training_row_policy",
        "mask_edit_policy",
        "recovery_payload_mutability",
        "recovered_review_qc",
        "initial_array_sha256",
    )
    return sha256_payload({key: group.attrs.get(key) for key in keys})
