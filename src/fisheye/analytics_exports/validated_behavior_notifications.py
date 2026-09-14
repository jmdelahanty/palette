"""Explicit availability announcements for exact validated-behavior exports."""

from __future__ import annotations

from dataclasses import dataclass
from email.utils import parseaddr
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

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


def _html_reference(value: str) -> str:
    safe = escape(value)
    try:
        parsed = urlparse(value)
    except ValueError:
        parsed = None
    if parsed is not None and parsed.scheme in {"http", "https"} and parsed.netloc:
        return f'<a href="{safe}">{safe}</a>'
    return f'<span style="overflow-wrap:anywhere">{safe}</span>'


def _html_fields(fields: Sequence[tuple[str, str]]) -> str:
    return "".join(
        f'<p style="margin:0 0 8px"><strong>{escape(label)}:</strong> '
        f"{_html_reference(value)}</p>"
        for label, value in fields
    )


def _html_section(title: str, content: str) -> str:
    return (
        '<section style="margin:24px 0">'
        f'<h2 style="font-size:18px;color:#17384a;margin:0 0 10px;'
        f'padding-bottom:5px;border-bottom:1px solid #d6e0e5">{escape(title)}</h2>'
        f"{content}</section>"
    )


def _render_html(
    *,
    dataset_fields: Sequence[tuple[str, str]],
    access_fields: Sequence[tuple[str, str]],
    table_names: Sequence[str],
    provenance_fields: Sequence[tuple[str, str]],
    note: str,
) -> str:
    table_items = "".join(
        f"<li><code>{escape(name)}</code></li>" for name in table_names
    )
    reading = (
        "<p>Open the exact manifest-selected publication with "
        "<code>ValidatedBehaviorExportDataset.open(publication_root, export_run_id)</code>. "
        "Use <code>table(name).collect_bounded(max_rows=...)</code> for a small sample "
        "or <code>table(name).scan(...)</code> for a lazy query.</p>"
    )
    sections = [
        _html_section("Dataset", _html_fields(dataset_fields)),
        _html_section("Access", _html_fields(access_fields)),
        _html_section(
            "Tables", f'<ul style="margin:0;padding-left:22px">{table_items}</ul>'
        ),
        _html_section("How to read it", reading),
        _html_section("Provenance", _html_fields(provenance_fields)),
    ]
    if note:
        sections.append(_html_section("Sender note", f"<p>{escape(note)}</p>"))
    sections.append(
        '<p style="color:#52626b;font-size:13px;border-top:1px solid #d6e0e5;'
        'padding-top:14px">This email does not grant filesystem access or activate a '
        "production selector. If the access location differs from the validated "
        "publication root, confirm that it refers to this manifest before reading it.</p>"
    )
    return (
        '<!doctype html><html><body><main style="font-family:Arial,Helvetica,sans-serif;'
        'color:#263b47;line-height:1.5;max-width:720px">'
        '<h1 style="font-size:25px;color:#17384a;margin:0">Palette dataset available</h1>'
        '<p style="margin:8px 0 20px">A validated behavior dataset is ready for reading.</p>'
        + "".join(sections)
        + "</main></body></html>"
    )


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

    dataset_fields = [
        ("Export run", dataset.export_run_id),
        ("Profile", str(profile)),
        ("Publication status", str(manifest["status"])),
    ]
    access_fields = [
        ("Location (provided by sender)", location),
        ("Validated publication root", str(root)),
    ]
    if audience:
        access_fields.append(("Intended audience", audience))
    if handoff:
        access_fields.append(("Reading guide", handoff))
    if access_note:
        access_fields.append(("Access instructions", access_note))
    provenance_fields = [
        ("Manifest", str(manifest_path)),
        ("Manifest record SHA-256", str(digest)),
        ("Validation mode", dataset.validation_mode),
    ]

    def text_fields(fields: Sequence[tuple[str, str]]) -> list[str]:
        return [f"{label}: {value}" for label, value in fields]

    lines = [
        "PALETTE DATASET AVAILABLE",
        "A validated behavior dataset is ready for reading.",
        "",
        "DATASET",
        *text_fields(dataset_fields),
        "",
        "ACCESS",
        *text_fields(access_fields),
        "",
        f"TABLES ({len(dataset.table_names)})",
        *(f"- {name}" for name in dataset.table_names),
        "",
        "HOW TO READ IT",
        "Open the exact manifest-selected publication with "
        "ValidatedBehaviorExportDataset.open(publication_root, export_run_id).",
        "Use table(name).collect_bounded(max_rows=...) for a small sample "
        "or table(name).scan(...) for a lazy query.",
        "",
        "PROVENANCE",
        *text_fields(provenance_fields),
    ]
    if note:
        lines.extend(["", "SENDER NOTE", note])
    lines.extend(
        [
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
        html_body=_render_html(
            dataset_fields=dataset_fields,
            access_fields=access_fields,
            table_names=dataset.table_names,
            provenance_fields=provenance_fields,
            note=note,
        ),
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
