"""Inventory one Citrus staging batch after organization/import.

The manifest is an audit and cleanup-planning surface.  It never deletes data.
Every source observed in organizer events or still present in the batch receives
one conservative disposition.  Copy residues are only called verified after
content hashing; producer diagnostics are only called disposable after their
merged recording artifacts and aggregate shard summary validate.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from fisheye.shared.json_safety import write_json_atomic


STAGING_BATCH_DISPOSITION_SCHEMA_ID = "palette.staging_batch_disposition.v1"
STAGING_BATCH_DISPOSITION_FILENAME = "_palette_batch_disposition.json"

MOVED = "moved"
VERIFIED_FANOUT_COPY = "verified_fanout_copy"
RETAINED_AUTHORITY = "retained_authority"
DISPOSABLE_DIAGNOSTIC = "disposable_diagnostic"
UNKNOWN = "unknown"

_DISPOSITIONS = (
    MOVED,
    VERIFIED_FANOUT_COPY,
    RETAINED_AUTHORITY,
    DISPOSABLE_DIAGNOSTIC,
    UNKNOWN,
)


class StagingBatchDispositionError(ValueError):
    """Raised when a staging disposition cannot be constructed safely."""


def _safe_resolve(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _relative_source(batch_root: Path, source: Path) -> str:
    try:
        return _safe_resolve(source).relative_to(_safe_resolve(batch_root)).as_posix()
    except ValueError:
        return str(_safe_resolve(source))


def _read_jsonl(path: Path | None) -> Iterable[dict[str, Any]]:
    if path is None or not path.is_file():
        return ()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _sha256_file(path: Path, cache: dict[Path, str]) -> str:
    resolved = _safe_resolve(path)
    cached = cache.get(resolved)
    if cached is not None:
        return cached
    digest = sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    cache[resolved] = value
    return value


def _file_evidence(
    path: Path,
    hash_cache: dict[Path, str],
    *,
    include_sha256: bool = False,
) -> dict[str, Any]:
    resolved = _safe_resolve(path)
    if not resolved.is_file():
        return {"path": str(resolved), "present": False}
    stat = resolved.stat()
    evidence: dict[str, Any] = {
        "path": str(resolved),
        "present": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        evidence["sha256"] = _sha256_file(resolved, hash_cache)
    return evidence


def _organizer_events(
    organize_log: Path | None,
    *,
    batch_root: Path,
) -> dict[Path, list[dict[str, Any]]]:
    events: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for payload in _read_jsonl(organize_log):
        event = str(payload.get("event") or "")
        if event not in {"file_moved", "file_copied"}:
            continue
        source_text = str(payload.get("source") or "").strip()
        dest_text = str(payload.get("dest") or "").strip()
        if not source_text or not dest_text:
            continue
        source = _safe_resolve(Path(source_text))
        try:
            source.relative_to(_safe_resolve(batch_root))
        except ValueError:
            continue
        events[source].append(
            {
                "action": "move" if event == "file_moved" else "copy",
                "destination": str(_safe_resolve(Path(dest_text))),
                "recording_name": str(payload.get("recording_name") or ""),
            }
        )
    return events


def _is_retained_authority(relative_path: str) -> bool:
    path = Path(relative_path)
    if relative_path == STAGING_BATCH_DISPOSITION_FILENAME:
        return False
    if path.parts and path.parts[0] in {
        "citrus",
        "external_recorder",
        "external_crop_recorder",
        "recording_geometry_assets",
    }:
        return True
    if path.suffix.lower() in {".h5", ".hdf5", ".mp4"}:
        return True
    return path.name in {
        "_citrus_transfer_complete.json",
        "recording_session.json",
        "recording_snapshot.json",
        "recording_geometry_contract.json",
        "ptp_sync_summary.json",
        "orange_local_control.events.jsonl",
        "external_recorder_contract.json",
        "external_crop_recorder_contract.json",
        "external_recorder_supervisor_plan.json",
        "external_crop_recorder_supervisor_plan.json",
    }


def _is_recording_geometry_authority(relative_path: str) -> bool:
    path = Path(relative_path)
    return (
        path.name == "recording_geometry_contract.json"
        or bool(path.parts and path.parts[0] == "recording_geometry_assets")
    )


def _event_destination(
    events: Mapping[Path, list[dict[str, Any]]], source: Path
) -> Path | None:
    candidates = [
        Path(str(event["destination"]))
        for event in events.get(_safe_resolve(source), ())
        if event.get("action") == "move"
    ]
    return candidates[-1] if candidates else None


def _completed_shard_diagnostics(
    batch_root: Path,
    *,
    events: Mapping[Path, list[dict[str, Any]]],
    workflow_complete: bool,
) -> set[Path]:
    """Return shard diagnostics superseded by validated merged artifacts."""

    disposable: set[Path] = set()
    if not workflow_complete:
        return disposable
    recorder = batch_root / "external_recorder"
    summary_sources = sorted(
        source
        for source in events
        if source.parent == _safe_resolve(recorder)
        and source.name.startswith("Cam")
        and source.name.endswith("_external_summary.json")
    )
    for summary_source in summary_sources:
        camera_prefix = summary_source.name.removesuffix("_external_summary.json")
        summary_path = _event_destination(events, summary_source)
        merged_video = _event_destination(
            events, recorder / f"{camera_prefix}_external.mp4"
        )
        merged_keyframes = _event_destination(
            events, recorder / f"{camera_prefix}_external_keyframes.json"
        )
        if (
            summary_path is None
            or merged_video is None
            or merged_keyframes is None
            or not summary_path.is_file()
            or not merged_video.is_file()
            or not merged_keyframes.is_file()
        ):
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, Mapping):
            continue
        try:
            frames_encoded = int(summary.get("frames_encoded") or 0)
            frames_received = int(summary.get("frames_received") or 0)
            encode_dropped = int(summary.get("encode_dropped") or 0)
        except (TypeError, ValueError):
            continue
        shards = summary.get("external_encode_shards")
        if (
            summary.get("worker_failed") is not False
            or frames_encoded <= 0
            or frames_encoded != frames_received
            or encode_dropped != 0
            or not isinstance(shards, list)
            or not shards
        ):
            continue
        candidates: set[Path] = set()
        all_shards_complete = True
        for raw_shard in shards:
            if not isinstance(raw_shard, Mapping):
                all_shards_complete = False
                break
            retention = raw_shard.get("mp4_retention")
            if (
                raw_shard.get("worker_failed") is not False
                or not isinstance(retention, Mapping)
                or retention.get("removed_after_merge") is not True
                or retention.get("retained") is not False
            ):
                all_shards_complete = False
                break
            for field in ("encode_csv", "mp4_keyframe"):
                basename = Path(str(raw_shard.get(field) or "")).name
                if not basename:
                    all_shards_complete = False
                    break
                candidate = recorder / basename
                if not candidate.is_file():
                    all_shards_complete = False
                    break
                candidates.add(_safe_resolve(candidate))
        if all_shards_complete:
            disposable.update(candidates)
    return disposable


def build_staging_batch_disposition(
    batch_root: str | Path,
    *,
    organize_log: str | Path | None,
    workflow_status: str,
    apply: bool,
    organized_recording_dirs: Iterable[str | Path],
    zarr_paths: Iterable[str | Path],
) -> dict[str, Any]:
    """Build a complete, non-mutating disposition for one staging batch."""

    root = _safe_resolve(Path(batch_root))
    if not root.is_dir():
        raise StagingBatchDispositionError(f"Staging batch is not a directory: {root}")
    log_path = _safe_resolve(Path(organize_log)) if organize_log is not None else None
    recordings = sorted({_safe_resolve(Path(path)) for path in organized_recording_dirs})
    zarrs = sorted({_safe_resolve(Path(path)) for path in zarr_paths})
    events = _organizer_events(log_path, batch_root=root)
    current_files = {
        _safe_resolve(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != STAGING_BATCH_DISPOSITION_FILENAME
    }
    sources = sorted(current_files | set(events), key=lambda path: str(path))
    workflow_complete = bool(
        apply
        and workflow_status == "ok"
        and recordings
        and len(zarrs) == len(recordings)
        and all(path.exists() for path in zarrs)
    )
    disposable = _completed_shard_diagnostics(
        root,
        events=events,
        workflow_complete=workflow_complete,
    )
    hash_cache: dict[Path, str] = {}
    artifacts: list[dict[str, Any]] = []
    cleanup_blockers: list[str] = []
    geometry_authority_present = False

    for source in sources:
        relative = _relative_source(root, source)
        source_present = source.is_file()
        geometry_authority_present = geometry_authority_present or bool(
            source_present and _is_recording_geometry_authority(relative)
        )
        source_evidence = _file_evidence(source, hash_cache)
        source_events = events.get(source, [])
        destination_evidence = [
            {
                **event,
                **_file_evidence(Path(str(event["destination"])), hash_cache),
            }
            for event in source_events
        ]
        move_events = [event for event in source_events if event["action"] == "move"]
        copy_events = [event for event in source_events if event["action"] == "copy"]

        if move_events and not source_present:
            move_destinations_present = all(
                evidence.get("present") is True
                for evidence in destination_evidence
                if evidence.get("action") == "move"
            )
            if move_destinations_present:
                disposition = MOVED
                reason = "organizer_move_completed_source_absent_destination_present"
            else:
                disposition = UNKNOWN
                reason = "moved_source_and_destination_are_missing"
        elif move_events:
            disposition = UNKNOWN
            reason = "organizer_logged_move_but_source_is_still_present"
        elif copy_events and source_present:
            source_evidence = _file_evidence(
                source, hash_cache, include_sha256=True
            )
            destination_evidence = [
                (
                    {
                        **event,
                        **_file_evidence(
                            Path(str(event["destination"])),
                            hash_cache,
                            include_sha256=event.get("action") == "copy",
                        ),
                    }
                )
                for event in source_events
            ]
            source_sha = str(source_evidence.get("sha256") or "")
            verified = bool(
                source_sha
                and all(
                    evidence.get("present") is True
                    and evidence.get("sha256") == source_sha
                    for evidence in destination_evidence
                    if evidence.get("action") == "copy"
                )
            )
            if verified:
                disposition = VERIFIED_FANOUT_COPY
                reason = "all_logged_copy_destinations_match_source_sha256"
            else:
                disposition = UNKNOWN
                reason = "copy_destination_missing_or_content_mismatch"
        elif source in disposable:
            disposition = DISPOSABLE_DIAGNOSTIC
            reason = "merged_output_and_zero_drop_shard_summary_verified"
        elif source_present and _is_retained_authority(relative):
            disposition = RETAINED_AUTHORITY
            reason = "source_authority_not_yet_archived_or_fanout_verified"
        else:
            disposition = UNKNOWN
            reason = "no_explicit_disposition_rule"

        if source_present and disposition in {RETAINED_AUTHORITY, UNKNOWN}:
            cleanup_blockers.append(f"{disposition}:{relative}")
        artifacts.append(
            {
                "source_path": str(source),
                "relative_path": relative,
                "disposition": disposition,
                "reason": reason,
                "source": source_evidence,
                "destinations": destination_evidence,
            }
        )

    counts = {name: 0 for name in _DISPOSITIONS}
    present_bytes = {name: 0 for name in _DISPOSITIONS}
    for artifact in artifacts:
        disposition = str(artifact["disposition"])
        counts[disposition] += 1
        source = artifact["source"]
        if isinstance(source, Mapping) and source.get("present") is True:
            present_bytes[disposition] += int(source.get("size_bytes") or 0)
    if not workflow_complete:
        cleanup_blockers.insert(0, "workflow_not_complete_or_zarr_coverage_incomplete")
    if geometry_authority_present:
        cleanup_blockers.append(
            "recording_geometry_candidate_publication_not_implemented"
        )

    return {
        "schema_id": STAGING_BATCH_DISPOSITION_SCHEMA_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_path": str(root),
        "organize_log": str(log_path) if log_path is not None else None,
        "workflow": {
            "status": workflow_status,
            "apply": bool(apply),
            "organized_recording_dirs": [str(path) for path in recordings],
            "zarr_paths": [str(path) for path in zarrs],
            "complete_for_cleanup_assessment": workflow_complete,
        },
        "summary": {
            "artifact_count": len(artifacts),
            "counts_by_disposition": counts,
            "present_bytes_by_disposition": present_bytes,
        },
        "cleanup_assessment": {
            "safe_to_delete_batch": bool(workflow_complete and not cleanup_blockers),
            "blockers": cleanup_blockers,
            "automatic_deletion_performed": False,
        },
        "artifacts": artifacts,
    }


def write_staging_batch_disposition(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(Path(path), payload, overwrite=True)


__all__ = [
    "DISPOSABLE_DIAGNOSTIC",
    "MOVED",
    "RETAINED_AUTHORITY",
    "STAGING_BATCH_DISPOSITION_FILENAME",
    "STAGING_BATCH_DISPOSITION_SCHEMA_ID",
    "StagingBatchDispositionError",
    "UNKNOWN",
    "VERIFIED_FANOUT_COPY",
    "build_staging_batch_disposition",
    "write_staging_batch_disposition",
]
