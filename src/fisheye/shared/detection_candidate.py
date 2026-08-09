"""Shared construction boundary for disposable YOLO detection candidates.

This module deliberately does not publish canonical data.  Full-recording
workflows hand the completed candidate to the atomic publisher; clipped
workflows package it as an importable artifact.  Keeping the low-level detector
call here makes those two transport policies share one construction boundary.
"""

from __future__ import annotations

DETECTION_CANDIDATE_BUILD_AUTHORITY_ATTR = (
    "palette_detection_candidate_build_authority"
)
DETECTION_CANDIDATE_BUILD_AUTHORITY_SCHEMA = (
    "palette.detection_candidate_build_authority.v1"
)
NODE_LOCAL_DETECTION_CANDIDATE_KIND = "node_local_atomic_publish_candidate"
DEFAULT_DETECT_ROW_SHARD_ROWS = 131_072
DEFAULT_DETECT_FRAME_SHARD_ROWS = 131_072


def node_local_detection_candidate_authority() -> dict[str, str]:
    """Return the marker authorizing writes to one disposable local overlay."""

    return {
        "schema_id": DETECTION_CANDIDATE_BUILD_AUTHORITY_SCHEMA,
        "kind": NODE_LOCAL_DETECTION_CANDIDATE_KIND,
    }


__all__ = [
    "DETECTION_CANDIDATE_BUILD_AUTHORITY_ATTR",
    "DETECTION_CANDIDATE_BUILD_AUTHORITY_SCHEMA",
    "DEFAULT_DETECT_FRAME_SHARD_ROWS",
    "DEFAULT_DETECT_ROW_SHARD_ROWS",
    "NODE_LOCAL_DETECTION_CANDIDATE_KIND",
    "node_local_detection_candidate_authority",
]
