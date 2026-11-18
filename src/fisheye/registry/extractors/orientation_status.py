"""Registry row for a current-source recording's orientation status (migration 078).

The status is derived on every receipt-bound refresh by the read-only resolver
``fisheye.shared.unified_h5.orientation_status`` from the recording's Citrus
unified H5, the one ``recording_manifest.json`` binds as ``h5_relative_path``.
It is total: a recording whose H5 cannot be found or read gets an
``unverified`` row with a reason, never an exception, and a ``verified`` result
from an H5 whose camera differs from the registry's verified camera is
downgraded to ``unverified`` (``registry_camera_mismatch``). The registry's
``session_uuid`` is Orange's session and the H5's is Citrus's
(``citsess_…``), so they are kept in separate columns and never compared. A malformed packaged evidence file still raises.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fisheye.shared.acquisition_video_streams import resolve_acquisition_manifest_file
from fisheye.shared.unified_h5.orientation_status import (
    IDENTITY_FIELDS,
    VERIFIED,
    resolve_orientation_status,
    unverified_status,
)

MANIFEST_NAME = "recording_manifest.json"


def _status(recording_dir: Path, camera_id: Optional[str]) -> tuple[Optional[str], Dict[str, Any]]:
    try:
        manifest = json.loads((recording_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, unverified_status("recording_manifest_unreadable")
    relative = manifest.get("h5_relative_path") if isinstance(manifest, dict) else None
    if relative is None:
        return None, unverified_status("no_manifest_citrus_h5")
    relative = relative if isinstance(relative, str) else None
    try:
        h5_path = resolve_acquisition_manifest_file(
            recording_dir, relative, label="h5_relative_path"
        )
    except ValueError:
        return relative, unverified_status("citrus_h5_unresolved")
    try:
        status = resolve_orientation_status(h5_path)
    except OSError:
        return relative, unverified_status("citrus_h5_unreadable")
    if (
        status["orientation_status"] == VERIFIED
        and status["camera_serial"] != camera_id
    ):
        return relative, unverified_status(
            "registry_camera_mismatch",
            **{name: status[name] for name in IDENTITY_FIELDS},
        )
    return relative, status


def extract_recording_orientation_status(
    recording_dir: Path, *, session_uuid: Optional[str], camera_id: Optional[str]
) -> Dict[str, Any]:
    """The ``recording_orientation_status`` row (without ``recording_id``).

    ``session_uuid`` and ``camera_id`` are the registry's verified values.
    """

    relative, status = _status(Path(recording_dir), camera_id)
    return {
        "session_uuid": session_uuid,
        "citrus_session_uuid": status["session_uuid"],
        "h5_relative_path": relative,
        "rig_id": status["rig_id"],
        "arena_id": status["arena_id"],
        "camera_serial": status["camera_serial"],
        "orientation_status": status["orientation_status"],
        "verified_link": status["verified_link"],
        "basis": status["basis"],
        "evidence_sha256": status["evidence_sha256"],
        "reason": status["reason"],
        "camera_to_dish_parity": status["camera_to_dish_parity"],
        "status_schema_id": status["schema_id"],
        "status_schema_version": status["schema_version"],
        "status_json": json.dumps(status, sort_keys=True),
    }


__all__ = ["extract_recording_orientation_status"]
