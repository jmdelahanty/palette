"""Helpers for source lineage attribute compatibility."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .type_conversions import as_int, normalize_attr

CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR = "source_keypoints_run"
LEGACY_SOURCE_KEYPOINT_RUN_ATTR = "source_keypoint_run"
CANONICAL_SOURCE_DETECT_REVIEW_STATUS_REF_ATTR = "source_detect_review_status_ref"


def resolve_source_keypoints_run(attrs: Optional[Mapping[str, Any]]) -> Any:
    """Return canonical keypoint-run lineage value with legacy fallback."""
    if attrs is None:
        return None
    value = attrs.get(CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR)
    if value is None:
        value = attrs.get(LEGACY_SOURCE_KEYPOINT_RUN_ATTR)
    return value


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

    return payload
