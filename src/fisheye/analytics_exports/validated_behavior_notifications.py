"""Explicit availability announcements for exact validated-behavior exports."""

from __future__ import annotations

from dataclasses import dataclass
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.labeling.notifications import (
    LabelingNotification,
    LabelingNotificationConfig,
    send_labeling_notification,
)

from .validated_behavior_cohort import validated_behavior_manifest_path
from .validated_behavior_dataset import ValidatedBehaviorExportDataset


@dataclass(frozen=True)
class ExportAvailabilityAnnouncement:
    """A validated, prepared message. Construction has no delivery side effects."""

    notification: LabelingNotification
    context: Mapping[str, object]


def _single_line(value: str | None, *, field: str, required: bool = False) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{field} is required")
    if any(character in result for character in "\r\n\x00"):
        raise ValueError(f"{field} must be one line")
    return result


def _recipients(values: Sequence[str]) -> tuple[str, ...]:
    recipients: list[str] = []
    for value in values:
        address = _single_line(value, field="recipient", required=True)
        if (
            parseaddr(address) != ("", address)
            or address.count("@") != 1
            or any(character.isspace() for character in address)
            or "," in address
        ):
            raise ValueError(f"recipient must be one plain email address: {address!r}")
        if address not in recipients:
            recipients.append(address)
    if not recipients:
        raise ValueError("at least one --to recipient is required")
    return tuple(recipients)


def prepare_validated_behavior_export_announcement(
    *,
    publication_root: str | Path,
    export_run_id: str,
    to: Sequence[str],
    audience: str | None = None,
    location: str | None = None,
    handoff: str | None = None,
    access_note: str | None = None,
    note: str | None = None,
) -> ExportAvailabilityAnnouncement:
    """Validate the selected publication before describing it as available."""

    recipients = _recipients(to)
    audience = _single_line(audience, field="audience")
    location = _single_line(location, field="location")
    handoff = _single_line(handoff, field="handoff")
    access_note = _single_line(access_note, field="access note")
    note = _single_line(note, field="note")

    dataset = ValidatedBehaviorExportDataset.open(
        publication_root, export_run_id, validate=True, full_part_hashes=False
    )
    manifest = dataset.manifest
    root = dataset.root
    manifest_path = validated_behavior_manifest_path(root, dataset.export_run_id)
    profile = manifest["export_profile"]["profile_id"]
    digest = manifest["record_sha256"]
    location = location or str(root)

    lines = [
        "A validated Palette behavior dataset is available for reading.",
        "",
        f"Export run: {dataset.export_run_id}",
        f"Profile: {profile}",
        f"Publication status: {manifest['status']}",
        f"Manifest record SHA-256: {digest}",
        f"Manifest: {manifest_path}",
        f"Validation mode: {dataset.validation_mode}",
        f"Validated publication root: {root}",
        f"Access location (provided by sender): {location}",
        f"Tables: {', '.join(dataset.table_names)}",
    ]
    if audience:
        lines.append(f"Intended audience: {audience}")
    if handoff:
        lines.append(f"Reading guide: {handoff}")
    if access_note:
        lines.append(f"Access instructions: {access_note}")
    if note:
        lines.append(f"Note: {note}")
    lines.extend(
        [
            "",
            "Read the exact manifest-selected tables with Palette's "
            "ValidatedBehaviorExportDataset.open(publication_root, export_run_id) "
            "reader. Use table(name).collect_bounded(max_rows=...) for a small sample "
            "or table(name).scan(...) for a lazy query.",
            "",
            "This message does not grant filesystem access or activate a production selector. "
            "If the access location differs from the validated publication root, "
            "confirm that it refers to this manifest before reading it.",
        ]
    )
    notification = LabelingNotification(
        kind="validated_behavior_export_available",
        to_email=", ".join(recipients),
        to_user=dataset.export_run_id,
        subject=f"Palette dataset available: {dataset.export_run_id}",
        text_body="\n".join(lines),
    )
    return ExportAvailabilityAnnouncement(
        notification=notification,
        context={
            "export_run_id": dataset.export_run_id,
            "manifest_path": str(manifest_path),
            "manifest_record_sha256": digest,
            "publication_root": str(root),
            "access_location": location,
            "profile_id": profile,
        },
    )


def deliver_validated_behavior_export_announcement(
    announcement: ExportAvailabilityAnnouncement,
    *,
    config: LabelingNotificationConfig,
) -> dict[str, Any]:
    """Queue to the existing outbox or send using its configured SMTP relay."""

    return send_labeling_notification(
        announcement.notification,
        config=config,
        context=announcement.context,
    )
