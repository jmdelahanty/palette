#!/usr/bin/env python3
"""Consolidate accidental external-IPC clip recordings by camera stream.

The historical recording-only organizer expanded every ``(clip, camera)`` row
of an Orange rolling session into a top-level Palette recording.  This utility
repairs that shape without copying the large video payload or deleting the
accidental directories:

* required artifacts are hard-linked into a hidden work directory;
* one camera-specific parent recording owns all clips for that camera;
* clip paths and manifests are rewritten relative to that camera recording;
* the complete clip/camera grid and frame clocks are validated; and
* the work directory is atomically renamed to the requested destination.

Dry-run is the default.  ``--apply`` is required to write or publish anything.
The source directories are never removed by this utility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MODULE_NAME = "fisheye.utils.consolidate_external_ipc_rolling_recordings"
INDEX_NAME = "recording_clip_index.json"
INDEX_CSV_NAME = "recording_clip_index.csv"
RECEIPT_NAME = "rolling_clip_consolidation_receipt.json"
ARTIFACT_SCHEMA_ID = "orange_rolling_clips_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected safe relative path, got {value!r}")
    return path


def _resolve_under(root: Path, relative: str | Path) -> Path:
    rel = _safe_relative(relative)
    resolved_root = root.resolve()
    resolved = (resolved_root / rel).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Path escapes root {root}: {relative}")
    return resolved


def _resolve_manifest_path(recording_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return _resolve_under(recording_dir, path)


def _remap_staging_path(staging_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value)
    candidates: list[Path] = []
    if not raw.is_absolute():
        candidates.append(staging_dir / raw)
    parts = raw.parts
    if staging_dir.name in parts:
        index = parts.index(staging_dir.name)
        candidates.append(staging_dir.joinpath(*parts[index + 1 :]))
    for marker in ("external_recorder", "external_crop_recorder"):
        if marker in parts:
            index = parts.index(marker)
            candidates.append(staging_dir.joinpath(*parts[index:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else None


def _row_list(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("rows", "clips", "camera_artifacts"):
        raw = payload.get(key)
        if isinstance(raw, list):
            rows = [dict(item) for item in raw if isinstance(item, Mapping)]
            if len(rows) != len(raw):
                raise ValueError(f"{key} contains a non-object row")
            return rows
    raise ValueError("recording clip index has no rows/clips/camera_artifacts list")


@dataclass(frozen=True)
class SplitRecording:
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    clip_id: str
    clip_index: int
    camera_serial: str


@dataclass(frozen=True)
class Artifact:
    source: Path
    relative_path: Path
    role: str
    required: bool = True


@dataclass(frozen=True)
class ConsolidationPlan:
    session_id: str
    recording_id: str
    camera_serial: str
    recordings_root: Path
    staging_dir: Path
    destination: Path
    work_dir: Path
    split_recordings: Mapping[tuple[str, str], SplitRecording]
    source_index: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    normalized_rows: tuple[Mapping[str, Any], ...]
    artifacts: tuple[Artifact, ...]
    clip_manifests: Mapping[str, Mapping[str, Any]]
    camera_serials: tuple[str, ...]
    clip_ids: tuple[str, ...]


def discover_split_recordings(
    recordings_root: Path, session_id: str
) -> dict[tuple[str, str], SplitRecording]:
    expression = re.compile(
        rf"^{re.escape(session_id)}_(clip_(?P<clip>[0-9]{{6}}))_Cam(?P<camera>[0-9]+)$"
    )
    discovered: dict[tuple[str, str], SplitRecording] = {}
    for root in sorted(recordings_root.glob(f"{session_id}_clip_*_Cam*")):
        if not root.is_dir():
            continue
        match = expression.fullmatch(root.name)
        if match is None:
            continue
        clip_id = f"clip_{match.group('clip')}"
        camera = match.group("camera")
        manifest_path = root / "recording_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing split recording manifest: {manifest_path}"
            )
        manifest = _load_json(manifest_path)
        if str(manifest.get("orange_session_id") or "") != session_id:
            raise ValueError(f"Unexpected orange_session_id in {manifest_path}")
        if str(manifest.get("camera_id") or "") != camera:
            raise ValueError(f"Camera identity mismatch in {manifest_path}")
        key = (clip_id, camera)
        if key in discovered:
            raise ValueError(f"Duplicate split recording for {key}")
        discovered[key] = SplitRecording(
            root=root.resolve(),
            manifest_path=manifest_path.resolve(),
            manifest=manifest,
            clip_id=clip_id,
            clip_index=int(match.group("clip")),
            camera_serial=camera,
        )
    if not discovered:
        raise FileNotFoundError(
            f"No accidental split recordings found for {session_id} below {recordings_root}"
        )
    return discovered


def _stream(split: SplitRecording, name: str) -> Mapping[str, Any]:
    streams = split.manifest.get("video_streams")
    if not isinstance(streams, Mapping):
        raise ValueError(f"Missing video_streams in {split.manifest_path}")
    payloads = streams.get("streams")
    if not isinstance(payloads, Mapping) or not isinstance(payloads.get(name), Mapping):
        raise ValueError(f"Missing {name!r} stream in {split.manifest_path}")
    return payloads[name]


def _required_stream_path(split: SplitRecording, stream_name: str, key: str) -> Path:
    value = _stream(split, stream_name).get(key)
    path = _resolve_manifest_path(split.root, value)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"Missing {stream_name}.{key} for {split.root.name}: {value!r}"
        )
    return path.resolve()


def _looks_like_frame_clock_csv(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            header = fh.readline(16 * 1024).decode("utf-8", errors="strict")
    except (OSError, UnicodeError):
        return False
    fields = {field.strip() for field in header.rstrip("\r\n").split(",")}
    return "recording_frame_id" in fields or "frame_id" in fields


def _frame_clock_kind(path: Path) -> str:
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            header = next(csv.reader([fh.readline()]))
    except (OSError, UnicodeError, StopIteration, csv.Error):
        return "unknown"
    fields = {field.strip() for field in header}
    if {"gop_index", "frame_index_within_gop", "bytes"}.issubset(fields):
        return "full"
    if {"crop_video_frame_index", "crop_state"}.issubset(fields):
        return "crop"
    return "unknown"


def _full_metadata_source(
    *,
    split: SplitRecording,
    source_row: Mapping[str, Any],
    staging_dir: Path,
) -> Path:
    staged = _remap_staging_path(staging_dir, source_row.get("metadata"))
    if staged is not None and staged.is_file() and _frame_clock_kind(staged) == "full":
        return staged

    # The affected organizer version treated Orange's full metadata CSV as a
    # legacy summary and renamed it with a .json suffix. Detect by content, not
    # by the misleading extension.
    full = _stream(split, "full")
    for key in ("summary", "frame_clock_metadata"):
        candidate = _resolve_manifest_path(split.root, full.get(key))
        if (
            candidate is not None
            and candidate.is_file()
            and _frame_clock_kind(candidate) == "full"
        ):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not recover full-frame metadata for {split.root.name}"
    )


def _add_artifact(
    artifacts: dict[Path, Artifact],
    *,
    source: Path,
    relative_path: str | Path,
    role: str,
    required: bool = True,
) -> None:
    rel = _safe_relative(relative_path)
    source = source.resolve()
    existing = artifacts.get(rel)
    if existing is not None:
        if existing.source == source:
            return
        if existing.source.stat().st_size != source.stat().st_size or _sha256_file(
            existing.source
        ) != _sha256_file(source):
            raise ValueError(
                f"Conflicting sources for {rel}: {existing.source} and {source}"
            )
        return
    artifacts[rel] = Artifact(
        source=source, relative_path=rel, role=role, required=required
    )


def _copy_row_with_paths(
    row: Mapping[str, Any],
    *,
    recording_id: str,
    session_id: str,
    clip_id: str,
    camera: str,
    video_path: Path,
    metadata_path: Path,
    keyframe_path: Path,
    clip_manifest_path: Path,
) -> dict[str, Any]:
    normalized = dict(row)
    normalized.update(
        {
            "recording_id": recording_id,
            "session_id": session_id,
            "clip_id": clip_id,
            "clip_index": int(str(clip_id).split("_")[-1]),
            "camera_serial": camera,
            "clip_directory": f"clips/{clip_id}",
            "clip_recording_folder": f"clips/{clip_id}",
            "clip_manifest_path": clip_manifest_path.as_posix(),
            "video": video_path.as_posix(),
            "metadata": metadata_path.as_posix(),
            "keyframes": keyframe_path.as_posix(),
            "video_path": video_path.as_posix(),
            "metadata_path": metadata_path.as_posix(),
            "keyframe_path": keyframe_path.as_posix(),
            "recording_backend_mode": "external_ipc",
            "source_layout": "rolling_clips",
        }
    )
    return normalized


def _original_clip_manifest(staging_dir: Path, clip_id: str) -> Path | None:
    candidate = (
        staging_dir / "external_recorder" / "clips" / clip_id / "clip_manifest.json"
    )
    return candidate.resolve() if candidate.is_file() else None


def _clip_manifest_payload(
    *,
    recording_id: str,
    session_id: str,
    clip_id: str,
    rows: Sequence[Mapping[str, Any]],
    crop_paths: Mapping[tuple[str, str], Mapping[str, str]],
    source_manifest: Path | None,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["camera_serial"]))
    first = ordered[0]
    camera_artifacts: list[dict[str, Any]] = []
    recording_outputs: dict[str, Any] = {}
    for row in ordered:
        camera = str(row["camera_serial"])
        full = {
            "role": "ingest_authoritative",
            "output_kind": "full",
            "video": row["video_path"],
            "metadata": row["metadata_path"],
            "keyframes": row["keyframe_path"],
            "frame_count": row.get("frame_count"),
            "first_recording_frame_id": row.get("first_recording_frame_id"),
            "last_recording_frame_id": row.get("last_recording_frame_id"),
            "recording_frame_id_gaps": row.get("recording_frame_id_gaps"),
            "frame_rate": row.get("frame_rate", 30),
            "codec": "hevc",
            "container": "mp4",
        }
        crop = dict(crop_paths[(clip_id, camera)])
        crop.update(
            {
                "role": "sidecar",
                "output_kind": "crop",
                "frame_count": row.get("frame_count"),
                "first_recording_frame_id": row.get("first_recording_frame_id"),
                "last_recording_frame_id": row.get("last_recording_frame_id"),
                "recording_frame_id_gaps": row.get("recording_frame_id_gaps"),
                "frame_rate": row.get("frame_rate", 30),
                "codec": "hevc",
                "container": "mp4",
            }
        )
        camera_artifacts.append(
            {
                "camera_serial": camera,
                "video_path": row["video_path"],
                "metadata_path": row["metadata_path"],
                "keyframe_path": row["keyframe_path"],
                "frame_count": row.get("frame_count"),
                "first_recording_frame_id": row.get("first_recording_frame_id"),
                "last_recording_frame_id": row.get("last_recording_frame_id"),
                "recording_frame_id_gaps": row.get("recording_frame_id_gaps"),
            }
        )
        recording_outputs[camera] = {"full": full, "crop": crop}

    source: dict[str, Any] | None = None
    if source_manifest is not None:
        source = {
            "original_manifest_path": (f"raw/orange_clip_manifests/{clip_id}.json"),
            "original_manifest_sha256": _sha256_file(source_manifest),
        }
    return {
        "schema_id": "palette.orange_external_ipc_rolling_clip.v1",
        "schema_version": 1,
        "generated_by": MODULE_NAME,
        "generated_at_utc": _utc_now(),
        "recording_id": recording_id,
        "session_id": session_id,
        "clip_id": clip_id,
        "clip_index": int(first["clip_index"]),
        "status": first.get("status"),
        "start_reason": first.get("start_reason"),
        "stop_reason": first.get("stop_reason"),
        "final_clip": first.get("final_clip"),
        "actual_duration_s": first.get("actual_duration_s"),
        "clip_directory": f"clips/{clip_id}",
        "camera_artifacts": camera_artifacts,
        "recording_outputs": recording_outputs,
        "source": source,
    }


def build_plan(
    *,
    recordings_root: Path,
    staging_dir: Path,
    session_id: str,
    camera_serial: str,
    destination: Path | None = None,
) -> ConsolidationPlan:
    recordings_root = recordings_root.expanduser().resolve()
    staging_dir = staging_dir.expanduser().resolve()
    camera_serial = str(camera_serial).strip()
    if not camera_serial.isdigit():
        raise ValueError(f"Invalid camera serial: {camera_serial!r}")
    source_index_path = staging_dir / INDEX_NAME
    if not source_index_path.is_file():
        raise FileNotFoundError(
            f"Missing original Orange clip index: {source_index_path}"
        )
    source_index = _load_json(source_index_path)
    if str(source_index.get("session_id") or "") != session_id:
        raise ValueError(f"Session mismatch in {source_index_path}")
    all_rows = _row_list(source_index)
    all_split_recordings = discover_split_recordings(recordings_root, session_id)
    available_cameras = sorted(
        {str(row.get("camera_serial") or "") for row in all_rows}
    )
    if camera_serial not in available_cameras:
        raise ValueError(
            f"Camera {camera_serial} is not present in the source index; "
            f"available cameras: {available_cameras}"
        )
    rows = [
        row for row in all_rows if str(row.get("camera_serial") or "") == camera_serial
    ]
    split_recordings = {
        key: split
        for key, split in all_split_recordings.items()
        if key[1] == camera_serial
    }
    recording_id = f"{session_id}_cam{camera_serial}"
    destination = (
        destination.expanduser().resolve()
        if destination is not None
        else recordings_root / recording_id
    )
    work_dir = destination.with_name(f".{destination.name}.rolling-import.incomplete")
    if destination.parent != recordings_root:
        raise ValueError("Destination must be a direct child of recordings_root")

    expected_keys: set[tuple[str, str]] = set()
    artifacts: dict[Path, Artifact] = {}
    normalized_rows: list[dict[str, Any]] = []
    crop_paths: dict[tuple[str, str], dict[str, str]] = {}
    rows_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_manifests: dict[str, Path | None] = {}

    for row in rows:
        clip_id = str(row.get("clip_id") or "")
        camera = str(row.get("camera_serial") or "")
        if not re.fullmatch(r"clip_[0-9]{6}", clip_id) or not camera.isdigit():
            raise ValueError(f"Invalid clip/camera identity in source index row: {row}")
        key = (clip_id, camera)
        if key in expected_keys:
            raise ValueError(f"Duplicate source index row for {key}")
        expected_keys.add(key)
        split = split_recordings.get(key)
        if split is None:
            raise FileNotFoundError(f"Missing accidental split recording for {key}")

        full_video_source = _required_stream_path(split, "full", "video")
        full_metadata_source = _full_metadata_source(
            split=split,
            source_row=row,
            staging_dir=staging_dir,
        )
        full_keyframe_source = _required_stream_path(split, "full", "keyframes")
        crop_video_source = _required_stream_path(split, "crop", "video")
        crop_metadata_source = _required_stream_path(split, "crop", "metadata")
        crop_keyframe_source = _required_stream_path(split, "crop", "keyframes")

        clip_dir = Path("clips") / clip_id
        camera_base = f"Cam{camera}_{recording_id}"
        full_video_rel = clip_dir / f"{camera_base}.mp4"
        full_metadata_rel = clip_dir / f"{camera_base}_meta.csv"
        full_keyframe_rel = clip_dir / f"{camera_base}_keyframe.json"
        clip_manifest_rel = clip_dir / "clip_manifest.json"
        crop_dir = clip_dir / "crop"
        crop_video_rel = crop_dir / f"Cam{camera}_crop_external.mp4"
        crop_metadata_rel = crop_dir / f"Cam{camera}_crop_meta.csv"
        crop_keyframe_rel = crop_dir / f"Cam{camera}_crop_external_keyframe.json"

        for source, rel, role in (
            (full_video_source, full_video_rel, "authoritative_full_video"),
            (full_metadata_source, full_metadata_rel, "authoritative_full_metadata"),
            (full_keyframe_source, full_keyframe_rel, "authoritative_full_keyframes"),
            (crop_video_source, crop_video_rel, "runtime_crop_video"),
            (crop_metadata_source, crop_metadata_rel, "runtime_crop_metadata"),
            (crop_keyframe_source, crop_keyframe_rel, "runtime_crop_keyframes"),
        ):
            _add_artifact(
                artifacts,
                source=source,
                relative_path=rel,
                role=role,
            )

        normalized = _copy_row_with_paths(
            row,
            recording_id=recording_id,
            session_id=session_id,
            clip_id=clip_id,
            camera=camera,
            video_path=full_video_rel,
            metadata_path=full_metadata_rel,
            keyframe_path=full_keyframe_rel,
            clip_manifest_path=clip_manifest_rel,
        )
        normalized_rows.append(normalized)
        rows_by_clip[clip_id].append(normalized)
        crop_paths[key] = {
            "video": crop_video_rel.as_posix(),
            "metadata": crop_metadata_rel.as_posix(),
            "keyframes": crop_keyframe_rel.as_posix(),
        }

        # Preserve remaining Orange clip-local evidence when it survived the
        # historical move. These are optional and not authoritative paths.
        for source_dir, target_dir in (
            (
                staging_dir / "external_recorder" / "clips" / clip_id,
                clip_dir / "orange_sidecars",
            ),
            (
                staging_dir / "external_crop_recorder" / "clips" / clip_id,
                crop_dir / "orange_sidecars",
            ),
        ):
            if source_dir.is_dir():
                for source in sorted(source_dir.iterdir()):
                    if not source.is_file() or source.name == "clip_manifest.json":
                        continue
                    named_camera = re.search(r"Cam([0-9]+)", source.name)
                    if (
                        named_camera is not None
                        and named_camera.group(1) != camera_serial
                    ):
                        continue
                    _add_artifact(
                        artifacts,
                        source=source,
                        relative_path=target_dir / source.name,
                        role="orange_clip_sidecar",
                        required=False,
                    )

        if clip_id not in original_manifests:
            original = _original_clip_manifest(staging_dir, clip_id)
            original_manifests[clip_id] = original
            if original is not None:
                _add_artifact(
                    artifacts,
                    source=original,
                    relative_path=Path("raw")
                    / "orange_clip_manifests"
                    / f"{clip_id}.json",
                    role="original_orange_clip_manifest",
                )

    extra_split_keys = set(split_recordings) - expected_keys
    if extra_split_keys:
        raise ValueError(
            f"Unexpected split recordings not present in clip index: {sorted(extra_split_keys)[:8]}"
        )

    cameras = tuple(sorted({camera for _clip, camera in expected_keys}))
    clip_ids = tuple(sorted({clip for clip, _camera in expected_keys}))
    expected_grid = {(clip, camera) for clip in clip_ids for camera in cameras}
    if expected_keys != expected_grid:
        missing = sorted(expected_grid - expected_keys)
        raise ValueError(f"Incomplete clip/camera grid; missing {missing[:8]}")

    # One split directory contains a complete copy of raw session context.
    first_split = split_recordings[sorted(split_recordings)[0]]
    raw_dir = first_split.root / "raw"
    if raw_dir.is_dir():
        for source in sorted(path for path in raw_dir.rglob("*") if path.is_file()):
            _add_artifact(
                artifacts,
                source=source,
                relative_path=Path("raw") / source.relative_to(raw_dir),
                role="session_context",
            )

    # The first clip's split directories received session-scoped diagnostics.
    first_clip = clip_ids[0]
    core_sources = {artifact.source for artifact in artifacts.values()}
    for camera in cameras:
        derived_dir = split_recordings[(first_clip, camera)].root / "derived"
        if not derived_dir.is_dir():
            continue
        for source in sorted(path for path in derived_dir.rglob("*") if path.is_file()):
            if source.resolve() in core_sources:
                continue
            _add_artifact(
                artifacts,
                source=source,
                relative_path=Path("derived") / source.relative_to(derived_dir),
                role="session_diagnostic",
                required=False,
            )

    # Preserve remaining top-level staging evidence once. Clip subdirectories
    # are already captured above and would otherwise duplicate many sidecars.
    for source in sorted(path for path in staging_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(staging_dir)
        if "clips" in relative.parts:
            continue
        named_camera = re.search(r"Cam([0-9]+)", source.name)
        if named_camera is not None and named_camera.group(1) != camera_serial:
            continue
        if source == source_index_path or source == staging_dir / INDEX_CSV_NAME:
            target_name = f"{source.stem}.original{source.suffix}"
            target = Path("raw") / target_name
        else:
            target = Path("derived") / "original_staging_context" / relative
        _add_artifact(
            artifacts,
            source=source,
            relative_path=target,
            role="original_staging_context",
            required=False,
        )

    # Preserve an exact hard-linked archive of every file in the accidental
    # split recordings for this camera. This makes later deletion of those
    # top-level directories content-safe and keeps their original manifests.
    for split in split_recordings.values():
        for source in sorted(path for path in split.root.rglob("*") if path.is_file()):
            _add_artifact(
                artifacts,
                source=source,
                relative_path=(
                    Path("raw")
                    / "legacy_split_recordings"
                    / split.root.name
                    / source.relative_to(split.root)
                ),
                role="legacy_split_recording_evidence",
            )

    clip_manifests = {
        clip_id: _clip_manifest_payload(
            recording_id=recording_id,
            session_id=session_id,
            clip_id=clip_id,
            rows=rows_by_clip[clip_id],
            crop_paths=crop_paths,
            source_manifest=original_manifests[clip_id],
        )
        for clip_id in clip_ids
    }
    return ConsolidationPlan(
        session_id=session_id,
        recording_id=recording_id,
        camera_serial=camera_serial,
        recordings_root=recordings_root,
        staging_dir=staging_dir,
        destination=destination,
        work_dir=work_dir,
        split_recordings=split_recordings,
        source_index=source_index,
        rows=tuple(rows),
        normalized_rows=tuple(
            sorted(
                normalized_rows,
                key=lambda item: (int(item["clip_index"]), str(item["camera_serial"])),
            )
        ),
        artifacts=tuple(
            sorted(artifacts.values(), key=lambda item: item.relative_path.as_posix())
        ),
        clip_manifests=clip_manifests,
        camera_serials=cameras,
        clip_ids=clip_ids,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _materialize_artifact(artifact: Artifact, *, root: Path, link_mode: str) -> str:
    if not artifact.source.is_file():
        if artifact.required:
            raise FileNotFoundError(artifact.source)
        return "missing_optional"
    target = _resolve_under(root, artifact.relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        source_stat = artifact.source.stat()
        target_stat = target.stat()
        if source_stat.st_size != target_stat.st_size:
            raise FileExistsError(f"Existing target differs in size: {target}")
        if link_mode == "hardlink" and (
            source_stat.st_dev != target_stat.st_dev
            or source_stat.st_ino != target_stat.st_ino
        ):
            raise FileExistsError(
                f"Existing target is not the planned hard link: {target}"
            )
        return "existing"

    temporary = target.with_name(f".{target.name}.linktmp")
    if temporary.exists():
        temporary.unlink()
    if link_mode == "hardlink":
        os.link(artifact.source, temporary)
    elif link_mode == "copy":
        shutil.copy2(artifact.source, temporary)
    else:
        raise ValueError(f"Unsupported link_mode: {link_mode}")
    os.replace(temporary, target)
    return "created"


def _normalized_index(plan: ConsolidationPlan) -> dict[str, Any]:
    payload = dict(plan.source_index)
    payload.update(
        {
            "schema_id": "palette.orange_external_ipc_recording_clip_index.v1",
            "schema_version": 1,
            "generated_by": MODULE_NAME,
            "generated_at_utc": _utc_now(),
            "recording_id": plan.recording_id,
            "session_id": plan.session_id,
            "recording_folder": ".",
            "source_layout": "rolling_clips",
            "recording_backend_mode": "external_ipc",
            "row_granularity": "clip_camera",
            "row_count": len(plan.normalized_rows),
            "clip_count": len(plan.clip_ids),
            "cameras": list(plan.camera_serials),
            "rows": list(plan.normalized_rows),
        }
    )
    payload.pop("clips", None)
    payload.pop("camera_artifacts", None)
    columns = list(payload.get("columns") or [])
    for name in (
        "recording_id",
        "video_path",
        "metadata_path",
        "keyframe_path",
        "recording_backend_mode",
        "source_layout",
    ):
        if name not in columns:
            columns.append(name)
    payload["columns"] = columns
    return payload


def _write_index_csv(path: Path, index: Mapping[str, Any]) -> None:
    rows = _row_list(index)
    columns = list(index.get("columns") or [])
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _frame_id_from_line(line: bytes, index: int) -> int:
    fields = next(csv.reader([line.decode("utf-8")]))
    return int(fields[index])


def _fast_csv_stats(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        header_line = fh.readline()
        if not header_line:
            raise ValueError(f"Empty CSV: {path}")
        header = next(csv.reader([header_line.decode("utf-8")]))
        header_fields = {field.strip() for field in header}
        key = "recording_frame_id" if "recording_frame_id" in header else "frame_id"
        if key not in header:
            raise ValueError(f"CSV has no frame clock column: {path}")
        key_index = header.index(key)
        first_line = fh.readline()
        if not first_line:
            raise ValueError(f"CSV has no data rows: {path}")
        rows = 1
        last_line = first_line
        for line in fh:
            if line.strip():
                rows += 1
                last_line = line
    return {
        "rows": rows,
        "first_recording_frame_id": _frame_id_from_line(first_line, key_index),
        "last_recording_frame_id": _frame_id_from_line(last_line, key_index),
        "stream_kind": (
            "full"
            if {"gop_index", "frame_index_within_gop", "bytes"}.issubset(header_fields)
            else (
                "crop"
                if {"crop_video_frame_index", "crop_state"}.issubset(header_fields)
                else "unknown"
            )
        ),
    }


def validate_publication(
    plan: ConsolidationPlan,
    *,
    root: Path,
    verify_hardlinks: bool,
    scan_metadata: bool,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checks = 0
    linked_bytes = 0
    for artifact in plan.artifacts:
        checks += 1
        target = _resolve_under(root, artifact.relative_path)
        if (
            not target.is_file()
            or target.stat().st_size != artifact.source.stat().st_size
        ):
            failures.append(
                {"code": "artifact_missing_or_size_mismatch", "path": str(target)}
            )
            continue
        linked_bytes += target.stat().st_size
        if verify_hardlinks:
            source_stat = artifact.source.stat()
            target_stat = target.stat()
            if (
                source_stat.st_dev != target_stat.st_dev
                or source_stat.st_ino != target_stat.st_ino
            ):
                failures.append(
                    {"code": "artifact_not_hardlinked", "path": str(target)}
                )

    index_path = root / INDEX_NAME
    if not index_path.is_file():
        failures.append(
            {"code": "recording_clip_index_missing", "path": str(index_path)}
        )
        rows: list[dict[str, Any]] = []
    else:
        index = _load_json(index_path)
        rows = _row_list(index)
        if int(index.get("row_count") or -1) != len(plan.normalized_rows):
            failures.append({"code": "index_row_count_mismatch"})
        if int(index.get("clip_count") or -1) != len(plan.clip_ids):
            failures.append({"code": "index_clip_count_mismatch"})

    previous_by_camera: dict[str, int] = {}
    metadata_scans = 0
    crop_metadata_scans = 0
    seen_clip_manifests: set[str] = set()
    for row in rows:
        camera = str(row.get("camera_serial") or "")
        clip_id = str(row.get("clip_id") or "")
        if camera != plan.camera_serial:
            failures.append(
                {
                    "code": "unexpected_camera_in_index",
                    "expected": plan.camera_serial,
                    "observed": camera,
                }
            )
        expected_first = int(row["first_recording_frame_id"])
        expected_last = int(row["last_recording_frame_id"])
        expected_count = int(row["frame_count"])
        if expected_count != expected_last - expected_first + 1:
            failures.append(
                {
                    "code": "declared_frame_range_not_dense",
                    "camera": camera,
                    "clip_id": clip_id,
                }
            )
        if row.get("recording_frame_id_gaps") not in (None, 0, "0", [], {}):
            failures.append(
                {
                    "code": "reported_recording_frame_id_gaps",
                    "camera": camera,
                    "clip_id": clip_id,
                    "value": row.get("recording_frame_id_gaps"),
                }
            )
        previous = previous_by_camera.get(camera)
        if previous is not None and expected_first != previous + 1:
            failures.append(
                {
                    "code": "inter_clip_frame_gap",
                    "camera_serial": camera,
                    "clip_id": clip_id,
                    "previous": previous,
                    "current": expected_first,
                }
            )
        previous_by_camera[camera] = expected_last
        for key in (
            "video_path",
            "metadata_path",
            "keyframe_path",
            "clip_manifest_path",
        ):
            path = _resolve_under(root, str(row[key]))
            if not path.is_file() or path.stat().st_size <= 0:
                failures.append({"code": f"{key}_missing", "path": str(path)})
        keyframes = _load_json(_resolve_under(root, str(row["keyframe_path"])))
        if int(keyframes.get("total_frames") or -1) != expected_count:
            failures.append(
                {
                    "code": "keyframe_frame_count_mismatch",
                    "clip_id": clip_id,
                    "camera": camera,
                }
            )
        if scan_metadata:
            stats = _fast_csv_stats(_resolve_under(root, str(row["metadata_path"])))
            metadata_scans += 1
            observed_clock = {
                key: value for key, value in stats.items() if key != "stream_kind"
            }
            if observed_clock != {
                "rows": expected_count,
                "first_recording_frame_id": expected_first,
                "last_recording_frame_id": expected_last,
            }:
                failures.append(
                    {
                        "code": "full_metadata_clock_mismatch",
                        "clip_id": clip_id,
                        "camera": camera,
                        "expected": [expected_count, expected_first, expected_last],
                        "observed": stats,
                    }
                )
            if stats["stream_kind"] != "full":
                failures.append(
                    {
                        "code": "full_metadata_semantic_columns_mismatch",
                        "clip_id": clip_id,
                        "camera": camera,
                        "observed_stream_kind": stats["stream_kind"],
                    }
                )
        if clip_id not in seen_clip_manifests:
            manifest = _load_json(_resolve_under(root, str(row["clip_manifest_path"])))
            seen_clip_manifests.add(clip_id)
            if str(manifest.get("clip_id") or "") != clip_id:
                failures.append(
                    {"code": "clip_manifest_identity_mismatch", "clip_id": clip_id}
                )
        manifest = _load_json(_resolve_under(root, str(row["clip_manifest_path"])))
        output = (
            (manifest.get("recording_outputs") or {}).get(camera, {}).get("crop", {})
        )
        for key in ("video", "metadata", "keyframes"):
            path = _resolve_under(root, str(output.get(key) or ""))
            if not path.is_file() or path.stat().st_size <= 0:
                failures.append({"code": f"crop_{key}_missing", "path": str(path)})
        if scan_metadata:
            stats = _fast_csv_stats(_resolve_under(root, str(output["metadata"])))
            crop_metadata_scans += 1
            observed_clock = {
                key: value for key, value in stats.items() if key != "stream_kind"
            }
            if observed_clock != {
                "rows": expected_count,
                "first_recording_frame_id": expected_first,
                "last_recording_frame_id": expected_last,
            }:
                failures.append(
                    {
                        "code": "crop_metadata_clock_mismatch",
                        "clip_id": clip_id,
                        "camera": camera,
                        "expected": [expected_count, expected_first, expected_last],
                        "observed": stats,
                    }
                )
            if stats["stream_kind"] != "crop":
                failures.append(
                    {
                        "code": "crop_metadata_semantic_columns_mismatch",
                        "clip_id": clip_id,
                        "camera": camera,
                        "observed_stream_kind": stats["stream_kind"],
                    }
                )

    source_dirs_missing = [
        str(split.root)
        for split in plan.split_recordings.values()
        if not split.root.is_dir()
    ]
    if source_dirs_missing:
        failures.append(
            {"code": "source_directories_missing", "paths": source_dirs_missing}
        )
    return {
        "status": "pass" if not failures else "fail",
        "checked_at_utc": _utc_now(),
        "artifact_checks": checks,
        "linked_bytes": linked_bytes,
        "row_count": len(rows),
        "clip_count": len(seen_clip_manifests),
        "camera_count": len(previous_by_camera),
        "metadata_scans": metadata_scans,
        "crop_metadata_scans": crop_metadata_scans,
        "source_directories_preserved": len(plan.split_recordings)
        - len(source_dirs_missing),
        "legacy_source_file_checks": sum(
            artifact.role == "legacy_split_recording_evidence"
            for artifact in plan.artifacts
        ),
        "failure_count": len(failures),
        "failures": failures,
    }


def _root_manifest(
    plan: ConsolidationPlan, validation: Mapping[str, Any]
) -> dict[str, Any]:
    session_path = (
        plan.split_recordings[sorted(plan.split_recordings)[0]].root
        / "raw"
        / "recording_session.json"
    )
    session = _load_json(session_path)
    recording = (
        session.get("recording")
        if isinstance(session.get("recording"), Mapping)
        else {}
    )
    raw_files = [
        artifact.relative_path.as_posix()
        for artifact in plan.artifacts
        if artifact.relative_path.parts[0] == "raw"
    ]
    derived_files = [
        artifact.relative_path.as_posix()
        for artifact in plan.artifacts
        if artifact.relative_path.parts[0] == "derived"
    ]
    clip_files = [
        artifact.relative_path.as_posix()
        for artifact in plan.artifacts
        if artifact.relative_path.parts[0] == "clips"
    ]
    clip_files.extend(
        f"clips/{clip_id}/clip_manifest.json" for clip_id in plan.clip_ids
    )
    return {
        "recording_name": plan.recording_id,
        "recording_id": plan.recording_id,
        "session_uuid": plan.recording_id,
        "session_start_iso8601_utc": (
            recording.get("started_at_utc") or session.get("created_at_utc")
        ),
        "recording_type": "behavior",
        "recording_subtype": "free",
        "behavior_mode": "free",
        "artifact_schema_id": ARTIFACT_SCHEMA_ID,
        "camera_id": plan.camera_serial,
        "camera_ids": list(plan.camera_serials),
        "recording_backend": "external_ipc",
        "orange_session_id": plan.session_id,
        "orange_producer": session.get("producer"),
        "orange_recording_mode": "rolling_clips",
        "source_layout": "rolling_clips",
        "recording_clip_index": INDEX_NAME,
        "organized_utc": _utc_now(),
        "organized_by": MODULE_NAME,
        "files": {
            "raw": sorted(set(raw_files)),
            "cams": [],
            "derived": sorted(set(derived_files)),
            "clips": sorted(set(clip_files)),
        },
        "rolling_clip_streams": {
            "schema_id": "palette.orange_rolling_clip_streams.v1",
            "recording_clip_index": INDEX_NAME,
            "row_granularity": "clip_camera",
            "clip_count": len(plan.clip_ids),
            "camera_ids": list(plan.camera_serials),
            "frame_clock": "recording_frame_id",
        },
        "migration_validation": dict(validation),
        "preflight": {
            "status": "not_run",
            "checked_at_utc": None,
            "video": None,
            "h5": None,
        },
    }


def apply_plan(
    plan: ConsolidationPlan,
    *,
    link_mode: str = "hardlink",
    scan_metadata: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    if plan.destination.exists():
        raise FileExistsError(f"Destination already exists: {plan.destination}")
    if link_mode == "hardlink":
        destination_device = plan.destination.parent.stat().st_dev
        mismatched = [
            artifact.source
            for artifact in plan.artifacts
            if artifact.source.stat().st_dev != destination_device
        ]
        if mismatched:
            raise OSError(
                f"Hard-link publication crosses filesystems; first mismatch: {mismatched[0]}"
            )
    plan.work_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    existing = 0
    optional_missing = 0
    for artifact in plan.artifacts:
        outcome = _materialize_artifact(
            artifact, root=plan.work_dir, link_mode=link_mode
        )
        if outcome == "created":
            created += 1
        elif outcome == "existing":
            existing += 1
        else:
            optional_missing += 1

    index = _normalized_index(plan)
    _atomic_write_json(plan.work_dir / INDEX_NAME, index)
    _write_index_csv(plan.work_dir / INDEX_CSV_NAME, index)
    for clip_id, payload in plan.clip_manifests.items():
        _atomic_write_json(
            plan.work_dir / "clips" / clip_id / "clip_manifest.json", payload
        )

    validation = validate_publication(
        plan,
        root=plan.work_dir,
        verify_hardlinks=link_mode == "hardlink",
        scan_metadata=scan_metadata,
    )
    if validation["status"] != "pass":
        raise RuntimeError(
            f"Consolidation validation failed; work directory preserved at {plan.work_dir}: "
            f"{validation['failures'][:5]}"
        )
    _atomic_write_json(
        plan.work_dir / "recording_manifest.json", _root_manifest(plan, validation)
    )
    receipt = {
        "schema_id": "palette.external_ipc_rolling_consolidation_receipt.v1",
        "status": "validated",
        "generated_by": MODULE_NAME,
        "generated_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "session_id": plan.session_id,
        "recording_id": plan.recording_id,
        "camera_serial": plan.camera_serial,
        "destination": str(plan.destination),
        "work_dir": str(plan.work_dir),
        "link_mode": link_mode,
        "source_recording_count": len(plan.split_recordings),
        "source_recordings": [
            str(split.root)
            for split in sorted(
                plan.split_recordings.values(), key=lambda item: item.root.name
            )
        ],
        "source_directories_deleted": 0,
        "artifact_count": len(plan.artifacts),
        "artifacts_created": created,
        "artifacts_resumed": existing,
        "optional_artifacts_missing": optional_missing,
        "validation": validation,
    }
    _atomic_write_json(plan.work_dir / RECEIPT_NAME, receipt)
    os.replace(plan.work_dir, plan.destination)
    receipt["status"] = "published"
    receipt["published_at_utc"] = _utc_now()
    receipt["duration_seconds"] = time.perf_counter() - started
    _atomic_write_json(plan.destination / RECEIPT_NAME, receipt)
    return receipt


def plan_summary(plan: ConsolidationPlan) -> dict[str, Any]:
    required = [artifact for artifact in plan.artifacts if artifact.required]
    missing = [
        str(artifact.source) for artifact in required if not artifact.source.is_file()
    ]
    return {
        "status": "ready" if not missing else "blocked",
        "generated_by": MODULE_NAME,
        "generated_at_utc": _utc_now(),
        "session_id": plan.session_id,
        "recording_id": plan.recording_id,
        "camera_serial": plan.camera_serial,
        "recordings_root": str(plan.recordings_root),
        "staging_dir": str(plan.staging_dir),
        "destination": str(plan.destination),
        "work_dir": str(plan.work_dir),
        "source_recording_count": len(plan.split_recordings),
        "clip_count": len(plan.clip_ids),
        "camera_count": len(plan.camera_serials),
        "camera_serials": list(plan.camera_serials),
        "index_row_count": len(plan.normalized_rows),
        "artifact_count": len(plan.artifacts),
        "artifact_logical_bytes": sum(
            artifact.source.stat().st_size
            for artifact in plan.artifacts
            if artifact.source.is_file()
        ),
        "required_source_missing_count": len(missing),
        "required_sources_missing": missing,
        "source_directories_will_be_deleted": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-root", type=Path, required=True)
    parser.add_argument("--staging-session-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--camera-serial",
        required=True,
        help="Create one recording containing all clips for this camera stream.",
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of hard-linking payload files.",
    )
    parser.add_argument(
        "--skip-metadata-scan",
        action="store_true",
        help="Skip full/crop CSV row and boundary validation (not recommended for publication).",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        plan = build_plan(
            recordings_root=args.recordings_root,
            staging_dir=args.staging_session_dir,
            session_id=str(args.session_id),
            camera_serial=str(args.camera_serial),
            destination=args.destination,
        )
        result = (
            apply_plan(
                plan,
                link_mode="copy" if args.copy else "hardlink",
                scan_metadata=not bool(args.skip_metadata_scan),
            )
            if args.apply
            else plan_summary(plan)
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "generated_by": MODULE_NAME,
            "generated_at_utc": _utc_now(),
            "error": str(exc),
        }
        if args.output_json is not None:
            _atomic_write_json(args.output_json, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    if args.output_json is not None:
        _atomic_write_json(args.output_json, result)
    if args.json or args.output_json is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"status={result.get('status')}")
        print(f"output_json={args.output_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
