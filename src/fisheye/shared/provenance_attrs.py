"""Helpers for source lineage attribute compatibility."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .roi_pixel_contract import ROI_IMAGE_REPRESENTATION, normalize_pixel_contract
from .type_conversions import as_int, normalize_attr

CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR = "source_keypoints_run"
LEGACY_SOURCE_KEYPOINT_RUN_ATTR = "source_keypoint_run"
ASSIGNMENT_KEYPOINTS_RUN_ATTR = "assignment_keypoints_run"
ASSIGNMENT_KEYPOINT_GROUP_ATTR = "assignment_keypoint_group"
ASSIGNMENT_KEYPOINT_CONTRACT_ATTR = "assignment_keypoint_contract"
ASSIGNMENT_KEYPOINT_ROLE_ATTR = "assignment_keypoint_role"
ASSIGNMENT_KEYPOINT_SELECTION_ATTR = "assignment_keypoint_selection"
ASSIGNMENT_KEYPOINT_CONTRACT_VALUE = "subject_eyes_union_assignment_keypoints_v1"
EYES_UNION_LR_ASSIGNMENT_KEYPOINT_ROLE = "eyes_union_lr_assignment"
CANONICAL_SOURCE_DETECT_REVIEW_STATUS_REF_ATTR = "source_detect_review_status_ref"
SOURCE_CROP_STORAGE_MODE_ATTR = "source_crop_storage_mode"
SOURCE_CROP_SIGNATURE_ATTR = "source_crop_signature"
SOURCE_CROP_REVISION_ATTR = "source_crop_revision"
SOURCE_ROI_IMAGE_REPRESENTATION_ATTR = "source_roi_image_representation"
SOURCE_ROI_PIXEL_CONTRACT_ATTR = "source_roi_pixel_contract"
SOURCE_ROI_PIXEL_CONTRACT_NAME_ATTR = "source_roi_pixel_contract_name"
SOURCE_ROI_READ_MODE_ATTR = "source_roi_read_mode"
ROI_CACHE_POLICY_ATTR = "roi_cache_policy"
SOURCE_ROI_CACHE_USED_ATTR = "source_roi_cache_used"
SOURCE_ROI_CACHE_BACKEND_ATTR = "source_roi_cache_backend"
SOURCE_ROI_CACHE_KEY_ATTR = "source_roi_cache_key"
SOURCE_ROI_CACHE_PATH_ATTR = "source_roi_cache_path"
SOURCE_ROI_CACHE_CANONICAL_PATH_ATTR = "source_roi_cache_canonical_path"
SOURCE_ROI_CACHE_EXPECTED_ARCHIVE_PATH_ATTR = "source_roi_cache_expected_archive_path"
SOURCE_ROI_LIVE_ACCELERATION_REQUESTED_ATTR = "source_roi_live_acceleration_requested"
SOURCE_ROI_LIVE_ACCELERATION_EFFECTIVE_ATTR = "source_roi_live_acceleration_effective"
SOURCE_ROI_LIVE_ACCELERATION_FALLBACK_REASON_ATTR = "source_roi_live_acceleration_fallback_reason"
SOURCE_ROI_LIVE_GPU_CHUNK_FRAMES_ATTR = "source_roi_live_gpu_chunk_frames"
CANONICAL_SOURCE_CROP_SNAPSHOT_ATTRS = (
    SOURCE_CROP_STORAGE_MODE_ATTR,
    SOURCE_CROP_SIGNATURE_ATTR,
    SOURCE_CROP_REVISION_ATTR,
    CANONICAL_SOURCE_DETECT_REVIEW_STATUS_REF_ATTR,
    SOURCE_ROI_IMAGE_REPRESENTATION_ATTR,
    SOURCE_ROI_PIXEL_CONTRACT_NAME_ATTR,
)


def resolve_source_keypoints_run(attrs: Optional[Mapping[str, Any]]) -> Any:
    """Return canonical keypoint-run lineage value with legacy fallback."""
    if attrs is None:
        return None
    value = attrs.get(CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR)
    if value is None:
        value = attrs.get(LEGACY_SOURCE_KEYPOINT_RUN_ATTR)
    return value


def resolve_assignment_keypoints_run(attrs: Optional[Mapping[str, Any]]) -> Any:
    """Return the keypoint run selected for post-segmentation mask assignment."""
    if attrs is None:
        return None
    return attrs.get(ASSIGNMENT_KEYPOINTS_RUN_ATTR)


def resolve_assignment_keypoint_group(attrs: Optional[Mapping[str, Any]]) -> Any:
    """Return the keypoint group selected for post-segmentation mask assignment."""
    if attrs is None:
        return None
    return attrs.get(ASSIGNMENT_KEYPOINT_GROUP_ATTR)


def build_source_keypoints_attrs(
    source_keypoints_run: Any,
    *,
    include_legacy_alias: bool = True,
) -> Dict[str, Any]:
    """Build attrs payload with canonical key and optional legacy alias."""
    payload: Dict[str, Any] = {
        CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR: source_keypoints_run,
    }
    if include_legacy_alias:
        payload[LEGACY_SOURCE_KEYPOINT_RUN_ATTR] = source_keypoints_run
    return payload


def build_assignment_keypoint_attrs(
    assignment_keypoints_run: Any,
    *,
    assignment_keypoint_group: Any,
    selection: str,
) -> Dict[str, Any]:
    """Build attrs for keypoints used by deterministic post-segmentation assignment."""
    return {
        ASSIGNMENT_KEYPOINTS_RUN_ATTR: assignment_keypoints_run,
        ASSIGNMENT_KEYPOINT_GROUP_ATTR: assignment_keypoint_group,
        ASSIGNMENT_KEYPOINT_CONTRACT_ATTR: ASSIGNMENT_KEYPOINT_CONTRACT_VALUE,
        ASSIGNMENT_KEYPOINT_ROLE_ATTR: EYES_UNION_LR_ASSIGNMENT_KEYPOINT_ROLE,
        ASSIGNMENT_KEYPOINT_SELECTION_ATTR: str(selection),
    }


def build_source_crop_snapshot_attrs(
    crop_attrs: Optional[Mapping[str, Any]],
    *,
    source_crop_storage_mode: Any,
    include_review_status_ref: bool = True,
) -> Dict[str, Any]:
    """Build a normalized crop snapshot payload for downstream provenance."""
    attrs = crop_attrs or {}
    payload: Dict[str, Any] = {}

    storage_mode = normalize_attr(source_crop_storage_mode)
    if storage_mode is not None:
        payload["source_crop_storage_mode"] = storage_mode

    crop_signature = normalize_attr(attrs.get("crop_signature"))
    if crop_signature is not None:
        payload["source_crop_signature"] = crop_signature

    crop_revision = as_int(attrs.get("crop_revision"))
    if crop_revision is not None and crop_revision >= 0:
        payload["source_crop_revision"] = int(crop_revision)

    if include_review_status_ref:
        review_status_ref = normalize_attr(attrs.get("detect_review_status_ref"))
        if review_status_ref is not None:
            payload[CANONICAL_SOURCE_DETECT_REVIEW_STATUS_REF_ATTR] = review_status_ref

    representation = normalize_attr(attrs.get("roi_image_representation"))
    if representation is not None:
        payload[SOURCE_ROI_IMAGE_REPRESENTATION_ATTR] = representation

    contract = normalize_pixel_contract(attrs.get("roi_pixel_contract") or attrs.get("crop_pixel_contract"))
    if contract is not None:
        payload[SOURCE_ROI_PIXEL_CONTRACT_ATTR] = contract
        contract_name = normalize_attr(contract.get("name"))
        if contract_name is not None:
            payload[SOURCE_ROI_PIXEL_CONTRACT_NAME_ATTR] = contract_name

    return payload


def build_source_roi_pixel_attrs(roi_source: Any = None) -> Dict[str, Any]:
    """Build attrs describing the actual ROI pixels consumed by a reader."""

    if roi_source is None:
        return {}

    contract = normalize_pixel_contract(getattr(roi_source, "roi_pixel_contract", None))
    representation = normalize_attr(getattr(roi_source, "roi_image_representation", None))
    if representation is None and contract is not None:
        representation = normalize_attr(contract.get("image_representation"))
    if representation is None:
        representation = ROI_IMAGE_REPRESENTATION

    payload: Dict[str, Any] = {
        SOURCE_ROI_IMAGE_REPRESENTATION_ATTR: representation,
    }
    if contract is not None:
        payload[SOURCE_ROI_PIXEL_CONTRACT_ATTR] = contract
        name = normalize_attr(contract.get("name"))
        if name is not None:
            payload[SOURCE_ROI_PIXEL_CONTRACT_NAME_ATTR] = name
    return payload


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def extract_source_crop_snapshot_attrs(attrs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return normalized canonical crop snapshot attrs from an attr mapping."""
    attr_map = attrs or {}
    payload: Dict[str, Any] = {}

    storage_mode = normalize_attr(attr_map.get(SOURCE_CROP_STORAGE_MODE_ATTR))
    if storage_mode is not None:
        payload[SOURCE_CROP_STORAGE_MODE_ATTR] = storage_mode

    crop_signature = normalize_attr(attr_map.get(SOURCE_CROP_SIGNATURE_ATTR))
    if crop_signature is not None:
        payload[SOURCE_CROP_SIGNATURE_ATTR] = crop_signature

    crop_revision = as_int(attr_map.get(SOURCE_CROP_REVISION_ATTR))
    if crop_revision is not None and crop_revision >= 0:
        payload[SOURCE_CROP_REVISION_ATTR] = int(crop_revision)

    review_status_ref = normalize_attr(attr_map.get(CANONICAL_SOURCE_DETECT_REVIEW_STATUS_REF_ATTR))
    if review_status_ref is not None:
        payload[CANONICAL_SOURCE_DETECT_REVIEW_STATUS_REF_ATTR] = review_status_ref

    representation = normalize_attr(attr_map.get(SOURCE_ROI_IMAGE_REPRESENTATION_ATTR))
    if representation is not None:
        payload[SOURCE_ROI_IMAGE_REPRESENTATION_ATTR] = representation

    contract = normalize_pixel_contract(attr_map.get(SOURCE_ROI_PIXEL_CONTRACT_ATTR))
    if contract is not None:
        payload[SOURCE_ROI_PIXEL_CONTRACT_ATTR] = contract

    contract_name = normalize_attr(attr_map.get(SOURCE_ROI_PIXEL_CONTRACT_NAME_ATTR))
    if contract_name is not None:
        payload[SOURCE_ROI_PIXEL_CONTRACT_NAME_ATTR] = contract_name

    cache_used = _as_bool(attr_map.get(SOURCE_ROI_CACHE_USED_ATTR))
    if cache_used is not None:
        payload[SOURCE_ROI_CACHE_USED_ATTR] = cache_used

    for key in (
        SOURCE_ROI_READ_MODE_ATTR,
        ROI_CACHE_POLICY_ATTR,
        SOURCE_ROI_CACHE_BACKEND_ATTR,
        SOURCE_ROI_CACHE_KEY_ATTR,
        SOURCE_ROI_CACHE_PATH_ATTR,
        SOURCE_ROI_CACHE_CANONICAL_PATH_ATTR,
        SOURCE_ROI_CACHE_EXPECTED_ARCHIVE_PATH_ATTR,
        SOURCE_ROI_LIVE_ACCELERATION_REQUESTED_ATTR,
        SOURCE_ROI_LIVE_ACCELERATION_EFFECTIVE_ATTR,
        SOURCE_ROI_LIVE_ACCELERATION_FALLBACK_REASON_ATTR,
    ):
        value = normalize_attr(attr_map.get(key))
        if value is not None:
            payload[key] = value

    gpu_chunk_frames = as_int(attr_map.get(SOURCE_ROI_LIVE_GPU_CHUNK_FRAMES_ATTR))
    if gpu_chunk_frames is not None and gpu_chunk_frames >= 0:
        payload[SOURCE_ROI_LIVE_GPU_CHUNK_FRAMES_ATTR] = int(gpu_chunk_frames)

    return payload
