from __future__ import annotations

import h5py

from .models import CoreInfo, Finding
from .reader import dataset_fields, dataset_row_count

REQUIRED_EVENT_FIELDS = (
    "timestamp_ns_epoch",
    "timestamp_ns_session",
    "event_type_id",
    "current_step_index",
    "stimulus_frame_num",
    "camera_frame_id",
    "name_or_context",
    "stimulus_mode_id",
    "details_json",
)
REQUIRED_FRAME_METADATA_FIELDS = (
    "stimulus_frame_num",
    "triggering_camera_frame_id",
    "timestamp_ns_epoch",
)
FRAME_METADATA_FIELD_ALIASES = {
    "timestamp_ns_epoch": ("timestamp_ns", "timestamp_ns_session", "relative_timestamp_ns"),
}


def missing_required_frame_metadata_fields(fields: list[str]) -> list[str]:
    missing: list[str] = []
    for field in REQUIRED_FRAME_METADATA_FIELDS:
        aliases = FRAME_METADATA_FIELD_ALIASES.get(field, ())
        if field not in fields and not any(alias in fields for alias in aliases):
            missing.append(field)
    return missing


def inspect_core(handle: h5py.File, *, profile: str = "palette-import") -> tuple[CoreInfo, list[Finding]]:
    info = CoreInfo(status="pass", profile=profile, h5_opened=True, root_keys=sorted(handle.keys()))
    findings: list[Finding] = []

    events = handle.get("/events")
    info.events_present = events is not None
    if events is None:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="h5.events_missing",
                summary="Required /events dataset is missing.",
                component="core",
                kind="core",
            )
        )
    else:
        info.events_rows = dataset_row_count(events)
        missing_fields = [field for field in REQUIRED_EVENT_FIELDS if field not in dataset_fields(events)]
        info.missing_event_fields = missing_fields
        if missing_fields:
            info.status = "fail"
            findings.append(
                Finding(
                    severity="fail",
                    code="h5.events_fields_missing",
                    summary="Required /events fields are missing.",
                    details=", ".join(missing_fields),
                    component="core",
                    kind="core",
                )
            )

    frame_metadata = handle.get("/video_metadata/frame_metadata")
    info.frame_metadata_present = frame_metadata is not None
    frame_required = profile == "palette-import"
    if frame_metadata is None:
        severity = "fail" if frame_required else "warn"
        info.status = "fail" if severity == "fail" else ("warn" if info.status == "pass" else info.status)
        findings.append(
            Finding(
                severity=severity,
                code="h5.frame_metadata_missing",
                summary="/video_metadata/frame_metadata dataset is missing.",
                details=(
                    "Palette import requires this dataset."
                    if frame_required
                    else "Citrus may omit this when video logging is disabled."
                ),
                component="core",
                kind="core",
            )
        )
    else:
        info.frame_metadata_rows = dataset_row_count(frame_metadata)
        missing_fields = missing_required_frame_metadata_fields(dataset_fields(frame_metadata))
        info.missing_frame_metadata_fields = missing_fields
        if missing_fields:
            info.status = "fail"
            findings.append(
                Finding(
                    severity="fail",
                    code="h5.frame_metadata_fields_missing",
                    summary="Required frame_metadata fields are missing.",
                    details=", ".join(missing_fields),
                    component="core",
                    kind="core",
                )
            )

    return info, findings
