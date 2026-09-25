"""Versioned ROI-only annotation identity for recovered training archives."""

from collections.abc import Mapping

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


# Tail-successor format v2: successors reference the existing, immutable crop
# run through ``source_crop_run`` instead of publishing an identity copy. The
# refresh proof binds that crop's contract digest.
REFERENCED_CROP_SUCCESSOR_POLICY = (
    "new_mask_seed_and_review_version_reference_crop_sharded_v2"
)
_REFRESH_PROOF_KEY = "mask_apply_refresh"


def _supplier_binding(binding):
    return {key: value for key, value in binding.items() if key != _REFRESH_PROOF_KEY}


def review_run_crop_binding_matches(run_attrs, crop):
    """Whether ``crop`` supplies the pixels of a review run.

    Copied-crop runs (initial payloads and v1 successors) share the crop's
    exact ``source_bindings``. A v2 successor instead carries its own refresh
    proof, which must name this crop run and bind its contract digest, while
    the underlying supplier declaration stays identical.
    """
    run_binding = run_attrs.get("source_bindings")
    crop_binding = crop.attrs.get("source_bindings")
    if run_binding == crop_binding:
        return True
    if not isinstance(run_binding, Mapping) or not isinstance(crop_binding, Mapping):
        return False
    proof = run_binding.get(_REFRESH_PROOF_KEY)
    crop_name = str(crop.path).rstrip("/").split("/")[-1]
    return (
        isinstance(proof, Mapping)
        and proof.get("policy") == REFERENCED_CROP_SUCCESSOR_POLICY
        and str(proof.get("source_crop_run") or "").split("/")[-1] == crop_name
        and proof.get("source_crop_contract_sha256") == initial_contract_digest(crop)
        and _supplier_binding(run_binding) == _supplier_binding(crop_binding)
    )
