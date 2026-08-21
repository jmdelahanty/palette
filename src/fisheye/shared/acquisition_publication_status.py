"""Typed publication state for archive-level acquisition-frame authority.

The publication record is deliberately duplicated at the archive root and on
``raw_video``.  It is the small commit marker that tells metadata-only readers
whether the larger ownership/frame/manifest graph is usable.  Materialized and
external-video authorities have different completion evidence, so their mode
is part of the record rather than something a consumer may infer from nearby
arrays.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import re
from typing import Any, Mapping


ACQUISITION_AUTHORITY_STATUS_SCHEMA_ID = (
    "palette.acquisition_authority_publication_status"
)
ACQUISITION_AUTHORITY_STATUS_SCHEMA_VERSION = 2
ACQUISITION_AUTHORITY_STATUS_ATTR = "acquisition_authority_publication_status"

ACQUISITION_AUTHORITY_PUBLISHED = "published_canonical_v1"
ACQUISITION_AUTHORITY_PENDING = "publication_pending_resumable_v1"
ACQUISITION_AUTHORITY_NOT_PUBLISHED = "not_published_noncanonical_v1"

MATERIALIZED_ACQUISITION_AUTHORITY_MODE = "materialized_source_frames_v1"
EXTERNAL_ACQUISITION_AUTHORITY_MODE = "external_video_v1"
CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE = "external_clipped_videos_v1"

MATERIALIZED_ACQUISITION_PENDING_REASON = "verified_bytes_completion_in_progress"
MATERIALIZED_ACQUISITION_PUBLISHED_REASON = "completed_full_import_verified"
EXTERNAL_ACQUISITION_PENDING_REASON = "external_authority_publication_in_progress"
EXTERNAL_ACQUISITION_PUBLISHED_REASON = "completed_external_video_authority_verified"
CLIPPED_EXTERNAL_ACQUISITION_PENDING_REASON = (
    "external_clipped_authority_publication_in_progress"
)
CLIPPED_EXTERNAL_ACQUISITION_PUBLISHED_REASON = (
    "completed_external_clipped_authority_verified"
)

ACQUISITION_NONCANONICAL_REASON_CODES = frozenset(
    {
        "organized_recording_identity_absent",
        "images_full_not_materialized",
        "sampled_or_training_import",
        "incomplete_source_frame_import",
    }
)

MATERIALIZED_ACQUISITION_COMPLETION_SEMANTICS = (
    "published_only_after_manifest_ownership_frame_loadback_v1"
)
EXTERNAL_ACQUISITION_COMPLETION_SEMANTICS = (
    "published_only_after_external_ownership_frame_loadback_v1"
)
CLIPPED_EXTERNAL_ACQUISITION_COMPLETION_SEMANTICS = (
    "published_only_after_clipped_collection_ownership_frame_loadback_v1"
)
NONCANONICAL_ACQUISITION_COMPLETION_SEMANTICS = (
    "no_canonical_acquisition_authority_published_v1"
)
ACQUISITION_AUTHORITY_RESUMPTION_POLICY = "retry_exact_archive_completion_idempotent_v1"

_CANONICAL_ID_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")
_STATUS_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "status",
        "reason_code",
        "authority_mode",
        "authority_path",
        "completion_semantics",
        "resumption_policy",
    }
)


class AcquisitionPublicationStatusError(ValueError):
    """Raised for malformed, contradictory, or partially written status."""


def _canonical_segment(value: Any, *, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or _CANONICAL_ID_SEGMENT_RE.fullmatch(value) is None
    ):
        raise AcquisitionPublicationStatusError(
            f"{field_name} must be one exact canonical path segment."
        )
    return value


def _canonical_authority_path(value: Any) -> str:
    if type(value) is not str:
        raise AcquisitionPublicationStatusError(
            "Published or pending acquisition status requires an authority_path."
        )
    parts = value.split("/")
    if len(parts) != 3 or parts[:2] != ["analysis", "acquisition_camera_frames"]:
        raise AcquisitionPublicationStatusError(
            "authority_path must select one canonical acquisition-camera node."
        )
    _canonical_segment(parts[2], field_name="authority_path camera_id")
    return value


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


@dataclass(frozen=True)
class AcquisitionAuthorityPublicationStatus:
    status: str
    reason_code: str
    authority_mode: str | None
    authority_path: str | None
    completion_semantics: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": ACQUISITION_AUTHORITY_STATUS_SCHEMA_ID,
            "schema_version": ACQUISITION_AUTHORITY_STATUS_SCHEMA_VERSION,
            "status": self.status,
            "reason_code": self.reason_code,
            "authority_mode": self.authority_mode,
            "authority_path": self.authority_path,
            "completion_semantics": self.completion_semantics,
            "resumption_policy": ACQUISITION_AUTHORITY_RESUMPTION_POLICY,
        }


def build_acquisition_authority_publication_status(
    *,
    status: str,
    reason_code: str,
    authority_mode: str | None = None,
    authority_path: str | None = None,
) -> AcquisitionAuthorityPublicationStatus:
    """Build one valid status/mode/reason combination or fail closed."""

    if status == ACQUISITION_AUTHORITY_NOT_PUBLISHED:
        if reason_code not in ACQUISITION_NONCANONICAL_REASON_CODES:
            raise AcquisitionPublicationStatusError(
                "Noncanonical acquisition status requires a controlled reason_code."
            )
        if authority_mode is not None or authority_path is not None:
            raise AcquisitionPublicationStatusError(
                "Noncanonical acquisition status cannot claim an authority mode/path."
            )
        completion_semantics = NONCANONICAL_ACQUISITION_COMPLETION_SEMANTICS
    else:
        authority_path = _canonical_authority_path(authority_path)
        expected: dict[tuple[str, str], tuple[str, str]] = {
            (
                MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
                ACQUISITION_AUTHORITY_PENDING,
            ): (
                MATERIALIZED_ACQUISITION_PENDING_REASON,
                MATERIALIZED_ACQUISITION_COMPLETION_SEMANTICS,
            ),
            (
                MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
                ACQUISITION_AUTHORITY_PUBLISHED,
            ): (
                MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
                MATERIALIZED_ACQUISITION_COMPLETION_SEMANTICS,
            ),
            (
                EXTERNAL_ACQUISITION_AUTHORITY_MODE,
                ACQUISITION_AUTHORITY_PENDING,
            ): (
                EXTERNAL_ACQUISITION_PENDING_REASON,
                EXTERNAL_ACQUISITION_COMPLETION_SEMANTICS,
            ),
            (
                EXTERNAL_ACQUISITION_AUTHORITY_MODE,
                ACQUISITION_AUTHORITY_PUBLISHED,
            ): (
                EXTERNAL_ACQUISITION_PUBLISHED_REASON,
                EXTERNAL_ACQUISITION_COMPLETION_SEMANTICS,
            ),
            (
                CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE,
                ACQUISITION_AUTHORITY_PENDING,
            ): (
                CLIPPED_EXTERNAL_ACQUISITION_PENDING_REASON,
                CLIPPED_EXTERNAL_ACQUISITION_COMPLETION_SEMANTICS,
            ),
            (
                CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE,
                ACQUISITION_AUTHORITY_PUBLISHED,
            ): (
                CLIPPED_EXTERNAL_ACQUISITION_PUBLISHED_REASON,
                CLIPPED_EXTERNAL_ACQUISITION_COMPLETION_SEMANTICS,
            ),
        }
        expected_pair = expected.get((authority_mode, status))
        if expected_pair is None:
            raise AcquisitionPublicationStatusError(
                "Acquisition publication status/mode combination is unsupported."
            )
        expected_reason, completion_semantics = expected_pair
        if reason_code != expected_reason:
            raise AcquisitionPublicationStatusError(
                "Acquisition publication reason conflicts with its status/mode."
            )
    return AcquisitionAuthorityPublicationStatus(
        status=status,
        reason_code=reason_code,
        authority_mode=authority_mode,
        authority_path=authority_path,
        completion_semantics=completion_semantics,
    )


def parse_acquisition_authority_publication_status(
    value: Any,
) -> AcquisitionAuthorityPublicationStatus:
    """Parse the exact persisted status schema without coercion or adapters."""

    if type(value) is not dict or set(value) != _STATUS_FIELDS:
        raise AcquisitionPublicationStatusError(
            "Acquisition publication status must contain the exact schema fields."
        )
    if value.get("schema_id") != ACQUISITION_AUTHORITY_STATUS_SCHEMA_ID:
        raise AcquisitionPublicationStatusError(
            "Unsupported acquisition publication status schema_id."
        )
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != ACQUISITION_AUTHORITY_STATUS_SCHEMA_VERSION
    ):
        raise AcquisitionPublicationStatusError(
            "Unsupported acquisition publication status schema_version."
        )
    if value.get("resumption_policy") != ACQUISITION_AUTHORITY_RESUMPTION_POLICY:
        raise AcquisitionPublicationStatusError(
            "Unsupported acquisition publication resumption_policy."
        )
    if (
        type(value.get("status")) is not str
        or type(value.get("reason_code")) is not str
    ):
        raise AcquisitionPublicationStatusError(
            "Acquisition publication status and reason_code must be exact strings."
        )
    record = build_acquisition_authority_publication_status(
        status=value["status"],
        reason_code=value["reason_code"],
        authority_mode=value.get("authority_mode"),
        authority_path=value.get("authority_path"),
    )
    if not _exact_json_equal(value, record.to_dict()):
        raise AcquisitionPublicationStatusError(
            "Acquisition publication status is not its canonical parsed form."
        )
    return record


def load_acquisition_authority_publication_status(
    root: Any,
) -> AcquisitionAuthorityPublicationStatus:
    """Load and exact-compare the mirrored root/``raw_video`` status record."""

    root_attrs = getattr(root, "attrs", None)
    if not isinstance(root_attrs, Mapping):
        raise AcquisitionPublicationStatusError(
            "Archive root must expose persisted attrs."
        )
    try:
        raw_video = root.get("raw_video")
    except Exception as exc:
        raise AcquisitionPublicationStatusError(
            "Archive root cannot resolve raw_video publication status."
        ) from exc
    raw_attrs = getattr(raw_video, "attrs", None)
    if not isinstance(raw_attrs, Mapping):
        raise AcquisitionPublicationStatusError(
            "Acquisition publication status requires a raw_video group."
        )
    root_value = root_attrs.get(ACQUISITION_AUTHORITY_STATUS_ATTR)
    raw_value = raw_attrs.get(ACQUISITION_AUTHORITY_STATUS_ATTR)
    if root_value is None or raw_value is None:
        raise AcquisitionPublicationStatusError(
            "Acquisition publication status is missing at root or raw_video."
        )
    if not _exact_json_equal(root_value, raw_value):
        raise AcquisitionPublicationStatusError(
            "Root and raw_video acquisition publication statuses conflict."
        )
    return parse_acquisition_authority_publication_status(root_value)


def stamp_acquisition_authority_publication_status(
    root: Any,
    raw_video: Any,
    *,
    status: str,
    reason_code: str,
    authority_mode: str | None = None,
    authority_path: str | None = None,
) -> AcquisitionAuthorityPublicationStatus:
    """Transactionally mirror one exact status on root and ``raw_video``."""

    record = build_acquisition_authority_publication_status(
        status=status,
        reason_code=reason_code,
        authority_mode=authority_mode,
        authority_path=authority_path,
    )
    intended = record.to_dict()
    attrs_pairs = (root.attrs, raw_video.attrs)
    snapshots = tuple(copy.deepcopy(dict(attrs)) for attrs in attrs_pairs)
    try:
        for attrs in attrs_pairs:
            attrs[ACQUISITION_AUTHORITY_STATUS_ATTR] = copy.deepcopy(intended)
        if any(
            not _exact_json_equal(
                attrs.get(ACQUISITION_AUTHORITY_STATUS_ATTR), intended
            )
            for attrs in attrs_pairs
        ):
            raise AcquisitionPublicationStatusError(
                "Acquisition publication status did not round-trip exactly."
            )
    except Exception as exc:
        try:
            for attrs, snapshot in zip(attrs_pairs, snapshots, strict=True):
                put = getattr(attrs, "put", None)
                if callable(put):
                    put(copy.deepcopy(snapshot))
                else:
                    attrs.clear()
                    attrs.update(copy.deepcopy(snapshot))
        except Exception as rollback_exc:
            raise AcquisitionPublicationStatusError(
                "Acquisition status write failed and rollback was incomplete: "
                f"{rollback_exc}"
            ) from exc
        if isinstance(exc, AcquisitionPublicationStatusError):
            raise
        raise AcquisitionPublicationStatusError(
            f"Acquisition publication status write failed: {exc}"
        ) from exc
    return record


__all__ = [
    "ACQUISITION_AUTHORITY_NOT_PUBLISHED",
    "ACQUISITION_AUTHORITY_PENDING",
    "ACQUISITION_AUTHORITY_PUBLISHED",
    "ACQUISITION_AUTHORITY_RESUMPTION_POLICY",
    "ACQUISITION_AUTHORITY_STATUS_ATTR",
    "ACQUISITION_AUTHORITY_STATUS_SCHEMA_ID",
    "ACQUISITION_AUTHORITY_STATUS_SCHEMA_VERSION",
    "ACQUISITION_NONCANONICAL_REASON_CODES",
    "AcquisitionAuthorityPublicationStatus",
    "AcquisitionPublicationStatusError",
    "CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE",
    "CLIPPED_EXTERNAL_ACQUISITION_COMPLETION_SEMANTICS",
    "CLIPPED_EXTERNAL_ACQUISITION_PENDING_REASON",
    "CLIPPED_EXTERNAL_ACQUISITION_PUBLISHED_REASON",
    "EXTERNAL_ACQUISITION_AUTHORITY_MODE",
    "EXTERNAL_ACQUISITION_COMPLETION_SEMANTICS",
    "EXTERNAL_ACQUISITION_PENDING_REASON",
    "EXTERNAL_ACQUISITION_PUBLISHED_REASON",
    "MATERIALIZED_ACQUISITION_AUTHORITY_MODE",
    "MATERIALIZED_ACQUISITION_COMPLETION_SEMANTICS",
    "MATERIALIZED_ACQUISITION_PENDING_REASON",
    "MATERIALIZED_ACQUISITION_PUBLISHED_REASON",
    "NONCANONICAL_ACQUISITION_COMPLETION_SEMANTICS",
    "build_acquisition_authority_publication_status",
    "load_acquisition_authority_publication_status",
    "parse_acquisition_authority_publication_status",
    "stamp_acquisition_authority_publication_status",
]
