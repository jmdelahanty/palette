from __future__ import annotations

import h5py

from .models import Finding, SnapshotsInfo
from .reader import load_json_dataset


def inspect_snapshots(handle: h5py.File) -> tuple[SnapshotsInfo, list[Finding]]:
    info = SnapshotsInfo(status="skip")
    findings: list[Finding] = []
    statuses: list[str] = []

    protocol_group = handle.get("/protocol_snapshot")
    info.protocol_snapshot_present = protocol_group is not None and "protocol_definition_json" in protocol_group
    if info.protocol_snapshot_present:
        _, error = load_json_dataset(protocol_group["protocol_definition_json"])
        info.protocol_json_parseable = error is None
        statuses.append("pass" if error is None else "warn")
        if error is not None:
            findings.append(
                Finding(
                    severity="warn",
                    code="h5.protocol_snapshot_invalid_json",
                    summary="protocol_snapshot/protocol_definition_json is not valid JSON.",
                    details=error,
                    component="snapshots",
                    kind="optional",
                )
            )

    calibration_group = handle.get("/calibration_snapshot")
    info.calibration_snapshot_present = calibration_group is not None and "arena_config_json" in calibration_group
    if info.calibration_snapshot_present:
        _, error = load_json_dataset(calibration_group["arena_config_json"])
        info.calibration_json_parseable = error is None
        statuses.append("pass" if error is None else "warn")
        if error is not None:
            findings.append(
                Finding(
                    severity="warn",
                    code="h5.calibration_snapshot_invalid_json",
                    summary="calibration_snapshot/arena_config_json is not valid JSON.",
                    details=error,
                    component="snapshots",
                    kind="optional",
                )
            )

    recording_group = handle.get("/recording_snapshot")
    info.recording_snapshot_present = recording_group is not None and any(
        name in recording_group for name in ("recording_snapshot_json", "recording_pointer_json")
    )
    if info.recording_snapshot_present:
        errors = []
        for name in ("recording_snapshot_json", "recording_pointer_json"):
            if name not in recording_group:
                continue
            _, error = load_json_dataset(recording_group[name])
            if error is not None:
                errors.append(f"{name}: {error}")
        info.recording_json_parseable = not errors
        statuses.append("pass" if not errors else "warn")
        if errors:
            findings.append(
                Finding(
                    severity="warn",
                    code="h5.recording_snapshot_invalid_json",
                    summary="recording_snapshot JSON payloads are not fully parseable.",
                    details="; ".join(errors),
                    component="snapshots",
                    kind="optional",
                )
            )

    subject_metadata = handle.get("/subject_metadata")
    info.subject_metadata_present = subject_metadata is not None
    if subject_metadata is not None:
        info.subject_metadata_keys = sorted(str(key) for key in subject_metadata.attrs.keys())
        statuses.append("pass")

    session_metadata = handle.get("/session_metadata")
    info.session_metadata_present = session_metadata is not None
    if session_metadata is not None:
        info.session_metadata_keys = sorted(str(key) for key in session_metadata.attrs.keys())
        statuses.append("pass")

    stimulus_coordinates = handle.get("/stimulus_coordinates")
    info.stimulus_coordinates_present = stimulus_coordinates is not None
    if stimulus_coordinates is not None:
        info.stimulus_coordinate_arenas = sorted(stimulus_coordinates.keys())
        statuses.append("pass")

    if not statuses:
        info.status = "skip"
    elif "warn" in set(statuses):
        info.status = "warn"
    else:
        info.status = "pass"
    return info, findings
