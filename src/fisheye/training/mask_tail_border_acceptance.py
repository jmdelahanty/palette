"""Per-row, source-bound acceptance of a visible tail endpoint at a crop edge."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import re

import numpy as np

from fisheye.shared.keypoint_motion_authority import (
    keypoint_source_crop_run_from_attributes,
)
from fisheye.training.recover_merged_training_recording import _sha256_array

ATTR = "tail_crop_border_acceptances_v1"
SCHEMA = "palette.training.tail_crop_border_acceptance.v1"
ACTION = "tail_crop_border_action"
ROW_IDENTITY_ARRAYS = (
    "source_crop_row_ids",
    "source_refined_row_ids",
    "source_detect_row_index",
    "instance_key",
    "frame_indices",
)


def body_digest(mask: np.ndarray) -> str:
    return _sha256_array(np.asarray(mask, dtype=np.uint8))


def acceptance_action(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Tail border action must be an object")
    action = value.get("action")
    if action not in {"accept", "revoke"}:
        raise ValueError("Tail border action must be accept or revoke")
    raw_reason = value.get("reason")
    if action == "accept" and not isinstance(raw_reason, str):
        raise ValueError("Tail border acceptance reason must be text")
    reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
    if action == "accept" and (len(reason) < 3 or len(reason) > 240):
        raise ValueError("Tail border acceptance requires a 3–240 character reason")
    return {"action": str(action), "reason": reason if action == "accept" else ""}


def active_acceptances(group) -> dict[str, dict[str, object]]:
    raw = group.attrs.get(ATTR, {})
    if not isinstance(raw, Mapping):
        raise ValueError("Malformed tail border acceptance records")
    records = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise ValueError("Malformed tail border acceptance record")
        records[key] = dict(value)
    return records


def expected_row_identity(arrays, row: int) -> dict[str, int]:
    identity = {"roi_idx": int(row)}
    for name in ROW_IDENTITY_ARRAYS:
        if name in arrays:
            identity[name] = int(np.asarray(arrays[name][row]).item())
    if "frame_indices" in identity:
        identity["source_frame_idx"] = identity["frame_indices"]
    return identity


def validate_acceptance_record(
    key: str,
    record: Mapping[str, object],
    *,
    body: np.ndarray,
    source_crop_run: str,
    row_identity: Mapping[str, object],
) -> bool:
    """Return false only for a stale source; malformed evidence is an error."""
    try:
        row = int(key)
    except ValueError as exc:
        raise ValueError("Invalid accepted tail ROI key") from exc
    if (
        str(row) != key
        or row < 0
        or record.get("schema") != SCHEMA
        or type(record.get("roi_idx")) is not int
        or record.get("roi_idx") != row
    ):
        raise ValueError("Malformed accepted tail ROI identity")
    actor = record.get("accepted_by")
    timestamp = record.get("accepted_at_utc")
    reason = record.get("reason")
    if not isinstance(timestamp, str):
        raise ValueError("Accepted tail timestamp is missing")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Accepted tail timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("Accepted tail timestamp must be UTC")
    if (
        record.get("tip_semantics")
        != "visible_centerline_endpoint_at_crop_border_no_extrapolation"
        or not isinstance(actor, str)
        or not actor.strip()
        or not isinstance(reason, str)
        or not (3 <= len(reason.strip()) <= 240)
        or type(record.get("accepted_at_mask_revision")) is not int
        or int(record["accepted_at_mask_revision"]) < 0
        or not isinstance(record.get("source_crop_run"), str)
        or not str(record["source_crop_run"]).strip()
        or not isinstance(record.get("body_mask_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", record["body_mask_sha256"]) is None
    ):
        raise ValueError("Incomplete accepted tail ROI evidence")
    recorded_identity = record.get("row_identity")
    if not isinstance(recorded_identity, Mapping):
        raise ValueError("Accepted tail ROI row identity is missing")
    if "source_crop_row_ids" not in recorded_identity or not (
        "frame_indices" in recorded_identity or "source_frame_idx" in recorded_identity
    ):
        raise ValueError("Accepted tail ROI needs crop-row and frame identity")
    if any(type(value) is not int for value in recorded_identity.values()):
        raise ValueError("Accepted tail ROI identity values must be integers")
    if dict(recorded_identity) != dict(row_identity):
        return False
    if record.get("source_crop_run") != source_crop_run:
        return False
    return record.get("body_mask_sha256") == body_digest(body)


def bound_acceptances(
    group, *, mask_labels: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    records = active_acceptances(group)
    if not records:
        return {}
    body_idx = mask_labels.index("subject_body")
    n = int(group["masks_roi"].shape[0])
    source_crop_run = keypoint_source_crop_run_from_attributes(group.attrs)
    valid = {}
    for key, record in records.items():
        row = int(key)
        if not (0 <= row < n):
            raise ValueError("Accepted tail ROI is outside the source run")
        body = np.asarray(group["masks_roi"][row, body_idx], dtype=np.uint8)
        identity = expected_row_identity(group, row)
        if validate_acceptance_record(
            key,
            record,
            body=body,
            source_crop_run=source_crop_run,
            row_identity=identity,
        ):
            valid[key] = record
    return valid


def apply_acceptance_actions(
    group,
    *,
    actions: list[dict[str, object]],
    mask_labels: tuple[str, ...],
    revision: int,
) -> dict[str, dict[str, object]]:
    """Update sparse active records after the corresponding mask rows are written."""
    records = active_acceptances(group)
    body_idx = mask_labels.index("subject_body")
    for item in actions:
        row = int(item["roi_idx"])
        key = str(row)
        action = acceptance_action(item.get("action"))
        before_digest = str(item["before_body_sha256"])
        body = np.asarray(group["masks_roi"][row, body_idx], dtype=np.uint8)
        after_digest = body_digest(body)
        if action is None:
            if after_digest != before_digest:
                records.pop(key, None)
            continue
        if action["action"] == "revoke":
            records.pop(key, None)
            continue
        if not (np.any(body[(0, -1), :]) or np.any(body[:, (0, -1)])):
            raise ValueError(
                "Only a body mask touching the crop border can be accepted"
            )
        records[key] = {
            "schema": SCHEMA,
            "roi_idx": row,
            "row_identity": dict(item["row_identity"]),
            "source_crop_run": keypoint_source_crop_run_from_attributes(group.attrs),
            "body_mask_sha256": after_digest,
            "accepted_at_mask_revision": int(revision),
            "accepted_by": str(item["user"]),
            "accepted_at_utc": str(item["timestamp"]),
            "reason": action["reason"],
            "tip_semantics": "visible_centerline_endpoint_at_crop_border_no_extrapolation",
        }
    if records or ATTR in group.attrs:
        group.attrs[ATTR] = records
    return records
