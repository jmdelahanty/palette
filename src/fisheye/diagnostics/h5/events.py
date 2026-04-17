from __future__ import annotations

import json

import h5py
import numpy as np

from .core import REQUIRED_EVENT_FIELDS
from .models import EventsInfo, Finding
from .reader import dataset_fields, dataset_row_count, decode_scalar


def inspect_events(handle: h5py.File) -> tuple[EventsInfo, list[Finding]]:
    dataset = handle.get("/events")
    if dataset is None:
        return (
            EventsInfo(status="fail", error="/events dataset missing"),
            [
                Finding(
                    severity="fail",
                    code="h5.events_missing",
                    summary="Required /events dataset is missing.",
                    component="events",
                    kind="core",
                )
            ],
        )

    fields = dataset_fields(dataset)
    rows = dataset_row_count(dataset)
    info = EventsInfo(status="pass", rows=rows, fields=fields, has_camera_frame_id=("camera_frame_id" in fields))
    findings: list[Finding] = []

    missing_fields = [field for field in REQUIRED_EVENT_FIELDS if field not in fields]
    if missing_fields:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="h5.events_fields_missing",
                summary="Required /events fields are missing.",
                details=", ".join(missing_fields),
                component="events",
                kind="core",
            )
        )
        return info, findings

    if rows == 0:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="h5.events_empty",
                summary="/events dataset is empty.",
                component="events",
                kind="core",
            )
        )
        return info, findings

    data = dataset[:]
    timestamps = data["timestamp_ns_session"].astype(np.int64, copy=False)
    info.timestamp_monotonic = bool(np.all(np.diff(timestamps) >= 0)) if timestamps.size > 1 else True
    if info.timestamp_monotonic is False:
        info.status = "fail"
        findings.append(
            Finding(
                severity="fail",
                code="h5.events_timestamp_nonmonotonic",
                summary="/events timestamps are not monotonic.",
                component="events",
                kind="core",
            )
        )

    event_ids = data["event_type_id"].astype(np.int64, copy=False)
    unique_ids, counts = np.unique(event_ids, return_counts=True)
    info.event_type_counts = {str(int(event_id)): int(count) for event_id, count in zip(unique_ids, counts)}

    parse_failures = 0
    nonempty = 0
    for raw in data["details_json"]:
        text = str(decode_scalar(raw)).strip()
        if not text:
            continue
        nonempty += 1
        try:
            json.loads(text)
        except json.JSONDecodeError:
            parse_failures += 1
    info.details_json_nonempty = nonempty
    info.details_json_parse_failures = parse_failures
    return info, findings
