#!/usr/bin/env python3
"""Audit and remove superseded external-IPC split recording directories.

This command is dry-run by default.  It derives deletion targets only from the
published per-camera consolidation receipts, verifies every source file has an
inode-identical archived counterpart, and validates the recording frame index,
single master analysis Zarr, and imported camera calibration.  ``--apply``
first renames all targets into a same-filesystem quarantine, revalidates the
four surviving recordings, and only then removes the quarantine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

MODULE_NAME = "fisheye.utils.cleanup_external_ipc_rolling_recordings"
RECEIPT_NAME = "rolling_clip_source_cleanup_receipt.json"
RECEIPT_SCHEMA = "palette.rolling_clip_source_cleanup.v1"
CONSOLIDATION_RECEIPT = "rolling_clip_consolidation_receipt.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _inventory_digest(rows: Sequence[tuple[str, int, int, int]]) -> str:
    canonical = json.dumps(list(rows), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _zarr_attrs(node: Path) -> dict[str, Any]:
    metadata = _load_json(node / "zarr.json")
    attrs = metadata.get("attributes")
    if attrs is None:
        return {}
    if not isinstance(attrs, Mapping):
        raise ValueError(f"Zarr attributes are not an object: {node / 'zarr.json'}")
    return dict(attrs)


def _validate_survivor(
    recording: Path, *, session_id: str, camera: str
) -> dict[str, Any]:
    expected_name = f"{session_id}_cam{camera}"
    if recording.name != expected_name or recording.parent == recording:
        raise ValueError(f"unexpected survivor path: {recording}")
    if not recording.is_dir():
        raise FileNotFoundError(f"surviving camera recording is missing: {recording}")

    consolidation = _load_json(recording / CONSOLIDATION_RECEIPT)
    validation = consolidation.get("validation")
    if (
        consolidation.get("status") != "published"
        or consolidation.get("recording_id") != expected_name
        or str(consolidation.get("camera_serial") or "") != camera
        or not isinstance(validation, Mapping)
        or validation.get("status") != "pass"
        or int(validation.get("failure_count") or 0) != 0
    ):
        raise ValueError(
            f"consolidation receipt is not a passing publication: {recording}"
        )

    frame_manifest = _load_json(recording / "recording_frame_index_manifest.json")
    if frame_manifest.get("status") != "ok":
        raise ValueError(f"recording frame index is not ok: {recording}")
    row_count = int(frame_manifest.get("row_count") or 0)
    if row_count <= 0 or [
        str(value) for value in frame_manifest.get("camera_serials", [])
    ] != [camera]:
        raise ValueError(f"recording frame index identity is invalid: {recording}")

    zarr_path = recording / "zarr" / f"{expected_name}_analysis.zarr"
    root_attrs = _zarr_attrs(zarr_path)
    if (
        root_attrs.get("recording_id") != expected_name
        or [str(value) for value in root_attrs.get("camera_serials", [])] != [camera]
        or int(root_attrs.get("clip_count") or 0) != 55
        or int(root_attrs.get("clip_camera_row_count") or 0) != 55
        or int(root_attrs.get("recording_frame_index_row_count") or 0) != row_count
        or root_attrs.get("source_layout") != "rolling_clips"
    ):
        raise ValueError(f"master analysis Zarr identity is invalid: {zarr_path}")
    if any(path.is_dir() for path in (recording / "clips").rglob("*.zarr")):
        raise ValueError(f"clip-level Zarr found below {recording / 'clips'}")

    calibration = _zarr_attrs(zarr_path / "analysis" / "calibration")
    calibration_receipt = _load_json(
        zarr_path.parent / f"{zarr_path.name}_calibration_import_receipt.json"
    )
    if (
        calibration_receipt.get("status") != "pass"
        or str(calibration.get("active_camera_id") or "") != camera
        or str(calibration.get("primary_camera_id") or "") != camera
        or calibration.get("operator_configuration_verified") is not True
        or not str(calibration.get("immediate_donor_zarr") or "").startswith(
            str(recording.parent / f"sleepyfish_2026_05_05_17_45_30_cam{camera}")
        )
    ):
        raise ValueError(f"camera calibration import is invalid: {zarr_path}")

    return {
        "recording": str(recording),
        "recording_id": expected_name,
        "camera_serial": camera,
        "frame_index_rows": row_count,
        "clip_count": 55,
        "master_analysis_zarr": str(zarr_path),
        "pixels_per_mm_camera": calibration.get("pixels_per_mm_camera"),
        "calibration_donor": calibration.get("immediate_donor_zarr"),
        "consolidation_validation": "pass",
        "calibration_validation": "pass",
    }


def _audit_sources(
    recording: Path,
    *,
    recordings_root: Path,
    session_id: str,
    camera: str,
) -> tuple[list[Path], dict[str, Any]]:
    receipt = _load_json(recording / CONSOLIDATION_RECEIPT)
    raw_sources = receipt.get("source_recordings")
    if not isinstance(raw_sources, list) or len(raw_sources) != 55:
        raise ValueError(
            f"expected 55 source recordings in {recording / CONSOLIDATION_RECEIPT}"
        )
    expression = re.compile(
        rf"^{re.escape(session_id)}_clip_[0-9]{{6}}_Cam{re.escape(camera)}$"
    )
    sources: list[Path] = []
    inventory: list[tuple[str, int, int, int]] = []
    for raw_source in raw_sources:
        source = Path(str(raw_source)).resolve()
        if (
            source.parent != recordings_root
            or expression.fullmatch(source.name) is None
        ):
            raise ValueError(f"unsafe source recording path in receipt: {source}")
        if not source.is_dir():
            raise FileNotFoundError(
                f"source recording is missing before cleanup: {source}"
            )
        archive = recording / "raw" / "legacy_split_recordings" / source.name
        if not archive.is_dir():
            raise FileNotFoundError(f"legacy archive is missing: {archive}")
        source_files = _files(source)
        archive_files = _files(archive)
        if len(source_files) != len(archive_files):
            raise ValueError(f"source/archive file count differs: {source}")
        for source_file in source_files:
            relative = source_file.relative_to(source)
            archived_file = archive / relative
            if not archived_file.is_file():
                raise FileNotFoundError(
                    f"archived source file is missing: {archived_file}"
                )
            source_stat = source_file.stat()
            archive_stat = archived_file.stat()
            if (
                source_stat.st_dev != archive_stat.st_dev
                or source_stat.st_ino != archive_stat.st_ino
                or source_stat.st_size != archive_stat.st_size
            ):
                raise ValueError(
                    f"source file is not inode-identical to archive: {source_file}"
                )
            inventory.append(
                (
                    f"{source.name}/{relative.as_posix()}",
                    source_stat.st_dev,
                    source_stat.st_ino,
                    source_stat.st_size,
                )
            )
        sources.append(source)
    if len(set(sources)) != 55:
        raise ValueError(f"duplicate source recordings in receipt: {recording}")
    return sorted(sources), {
        "camera_serial": camera,
        "source_recording_count": len(sources),
        "source_file_count": len(inventory),
        "source_logical_bytes": sum(row[3] for row in inventory),
        "inode_archive_match_count": len(inventory),
        "inventory_sha256": _inventory_digest(sorted(inventory)),
    }


def _audit_provisional(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"provisional combined recording is missing: {path}")
    generated = re.compile(
        r"^(?:clips/clip_[0-9]{6}/clip_manifest\.json|recording_clip_index\.(?:csv|json)|"
        r"recording_manifest\.json|rolling_clip_consolidation_receipt\.json)$"
    )
    files = _files(path)
    unique = [item for item in files if item.stat().st_nlink == 1]
    unexpected_unique = [
        item.relative_to(path).as_posix()
        for item in unique
        if generated.fullmatch(item.relative_to(path).as_posix()) is None
    ]
    if unexpected_unique:
        raise ValueError(
            f"provisional recording contains unexpected unique files: {unexpected_unique[:8]}"
        )
    unique_large = [item for item in unique if item.stat().st_size >= 1024 * 1024]
    if unique_large:
        raise ValueError(
            f"provisional recording contains unique large files: {unique_large[:8]}"
        )
    return {
        "path": str(path),
        "file_count": len(files),
        "logical_bytes": sum(item.stat().st_size for item in files),
        "unique_generated_file_count": len(unique),
        "unique_generated_bytes": sum(item.stat().st_size for item in unique),
        "unique_large_file_count": 0,
    }


def cleanup_rolling_recordings(
    recordings_root: str | Path,
    session_id: str,
    *,
    cameras: Sequence[str],
    apply: bool = False,
) -> dict[str, Any]:
    root = Path(recordings_root).expanduser().resolve()
    if not root.is_dir() or not session_id.strip():
        raise ValueError(
            "recordings_root and session_id must identify an existing session"
        )
    camera_ids = tuple(str(camera).strip() for camera in cameras)
    if (
        len(camera_ids) != 4
        or len(set(camera_ids)) != 4
        or not all(value.isdigit() for value in camera_ids)
    ):
        raise ValueError("exactly four unique numeric camera serials are required")

    survivors: list[Path] = []
    survivor_evidence: list[dict[str, Any]] = []
    sources: list[Path] = []
    source_evidence: list[dict[str, Any]] = []
    for camera in camera_ids:
        recording = root / f"{session_id}_cam{camera}"
        survivors.append(recording)
        survivor_evidence.append(
            _validate_survivor(recording, session_id=session_id, camera=camera)
        )
        camera_sources, evidence = _audit_sources(
            recording,
            recordings_root=root,
            session_id=session_id,
            camera=camera,
        )
        sources.extend(camera_sources)
        source_evidence.append(evidence)
    if len(sources) != 220 or len(set(sources)) != 220:
        raise ValueError(
            "cleanup plan must contain exactly 220 unique split recordings"
        )
    if set(sources) & set(survivors):
        raise ValueError("cleanup plan overlaps surviving camera recordings")

    provisional = root / session_id
    provisional_evidence = _audit_provisional(provisional)
    targets = [*sorted(sources), provisional]
    quarantine = root / f".{session_id}.legacy-cleanup-quarantine.incomplete"
    if quarantine.exists():
        raise FileExistsError(f"cleanup quarantine already exists: {quarantine}")
    receipt: dict[str, Any] = {
        "status": "planned" if not apply else "in_progress",
        "schema_id": RECEIPT_SCHEMA,
        "generated_by": MODULE_NAME,
        "generated_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "recordings_root": str(root),
        "session_id": session_id,
        "camera_serials": list(camera_ids),
        "survivors": survivor_evidence,
        "source_audit": source_evidence,
        "provisional_audit": provisional_evidence,
        "source_recording_count": len(sources),
        "source_file_count": sum(item["source_file_count"] for item in source_evidence),
        "inode_archive_match_count": sum(
            item["inode_archive_match_count"] for item in source_evidence
        ),
        "deletion_targets": [str(path) for path in targets],
        "deletion_target_count": len(targets),
        "quarantine": str(quarantine),
    }
    if not apply:
        return receipt

    for survivor in survivors:
        _atomic_write_json(survivor / RECEIPT_NAME, receipt)
    quarantine.mkdir()
    moved: list[tuple[Path, Path]] = []
    try:
        for source in targets:
            quarantined = quarantine / source.name
            if quarantined.exists():
                raise FileExistsError(f"duplicate quarantine target: {quarantined}")
            os.replace(source, quarantined)
            moved.append((source, quarantined))
        post_move = [
            _validate_survivor(
                root / f"{session_id}_cam{camera}",
                session_id=session_id,
                camera=camera,
            )
            for camera in camera_ids
        ]
    except Exception:
        for source, quarantined in reversed(moved):
            if quarantined.exists() and not source.exists():
                os.replace(quarantined, source)
        if quarantine.exists() and not any(quarantine.iterdir()):
            quarantine.rmdir()
        raise

    shutil.rmtree(quarantine)
    remaining = [str(path) for path in targets if path.exists()]
    if remaining:
        raise RuntimeError(
            f"cleanup targets remain after quarantine removal: {remaining[:8]}"
        )
    receipt.update(
        {
            "status": "pass",
            "completed_at_utc": _utc_now(),
            "deleted_source_recording_count": len(sources),
            "deleted_provisional_recording": True,
            "post_cleanup_survivors": post_move,
            "quarantine_removed": not quarantine.exists(),
        }
    )
    for survivor in survivors:
        _atomic_write_json(survivor / RECEIPT_NAME, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings_root", type=Path)
    parser.add_argument("session_id")
    parser.add_argument("--camera", action="append", required=True, dest="cameras")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = cleanup_rolling_recordings(
            args.recordings_root,
            args.session_id,
            cameras=args.cameras,
            apply=bool(args.apply),
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        else:
            print(f"error: {exc}")
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"{result['status']}: sources={result['source_recording_count']} "
            f"files={result['source_file_count']} targets={result['deletion_target_count']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
