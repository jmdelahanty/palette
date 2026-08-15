from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional

from .type_conversions import as_int


def _hash_parameters(params: object) -> Optional[str]:
    if params is None:
        return None
    try:
        payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError):
        payload = str(params).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_crop_revision(attrs: Mapping[str, object]) -> int:
    revision = as_int(attrs.get("crop_revision"))
    return int(revision) if revision is not None and revision >= 0 else 0


def build_crop_signature(attrs: Mapping[str, object]) -> Dict[str, object]:
    return {
        "signature_version": 2,
        "detection_source_path": attrs.get("detection_source_path"),
        "source_coords_path": attrs.get("source_coords_path"),
        "detection_source_type": attrs.get("detection_source_type"),
        "source_detection_manifest_digest": attrs.get(
            "source_detection_manifest_digest"
        ),
        "detection_selection_policy": attrs.get(
            "detection_selection_policy",
            attrs.get("detection_preferred_policy"),
        ),
        "crop_storage_mode": attrs.get("crop_storage_mode"),
        "source_detect_run": attrs.get("source_detect_run"),
        "source_refined_run": attrs.get("source_refined_run"),
        "source_background_run": attrs.get("source_background_run"),
        "roi_size": attrs.get("roi_size"),
        "crop_geometry_policy_digest": attrs.get("crop_geometry_policy_digest"),
        "crop_padding_mode": attrs.get("crop_padding_mode"),
        "crop_revision": current_crop_revision(attrs),
        "parameter_source": attrs.get("parameter_source"),
        "parameters_hash": _hash_parameters(attrs.get("parameters")),
    }


def bump_crop_revision(
    attrs: Dict[str, object],
    *,
    reason: str,
    timestamp_utc: Optional[str] = None,
) -> Dict[str, object]:
    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc).isoformat()
    attrs["crop_revision"] = current_crop_revision(attrs) + 1
    attrs["crop_revision_reason"] = str(reason)
    attrs["crop_revision_updated_at_utc"] = str(timestamp_utc)
    attrs["crop_signature"] = build_crop_signature(attrs)
    return attrs
