"""Canonical source metadata for one camera stream stored as video clips.

The acquisition timeline is recording-wide even though the encoded pixels are
split across several MP4 files.  This module binds that timeline to the exact
recording frame index, the clip index, and cheap live-file fingerprints without
copying or decoding video payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs


SOURCE_VIDEO_COLLECTION_METADATA_SCHEMA_ID = (
    "palette.source_video_collection_metadata.v1"
)
SOURCE_VIDEO_COLLECTION_LAYOUT = "clipped_video_collection"
SOURCE_VIDEO_COLLECTION_SCHEMA_ID = "palette.clipped_video_collection.v1"
SOURCE_VIDEO_COLLECTION_LOCATOR_KIND = "recording_relative_frame_index"
SOURCE_VIDEO_COLLECTION_FINGERPRINT_STRATEGY = "member_stat_and_index_sha256_v1"


class ClippedVideoCollectionEvidenceError(ValueError):
    """Raised when persisted clipped-source evidence differs from live files."""


@dataclass(frozen=True)
class VerifiedClippedVideoCollectionFiles:
    """Exact live files selected by one persisted clipped collection contract."""

    recording_dir: Path
    index_paths: tuple[Path, Path, Path]
    member_paths: tuple[Path, ...]
    collection_sha256: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_under(recording_dir: Path, value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is missing.")
    raw = Path(text).expanduser()
    path = raw.resolve() if raw.is_absolute() else (recording_dir / raw).resolve()
    try:
        path.relative_to(recording_dir)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the recording directory: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _relative_path(recording_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(recording_dir).as_posix()


def _file_evidence(recording_dir: Path, path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "relative_path": _relative_path(recording_dir, path),
        "sha256": _file_sha256(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _verify_file_evidence(
    recording_dir: Path,
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, Mapping):
        raise ClippedVideoCollectionEvidenceError(f"{label} evidence is missing.")
    expected_fields = {"relative_path", "sha256", "size_bytes", "mtime_ns"}
    if set(value) != expected_fields:
        raise ClippedVideoCollectionEvidenceError(
            f"{label} evidence has an unexpected field set."
        )
    try:
        path = _resolve_under(
            recording_dir,
            value["relative_path"],
            label=label,
        )
        stat = path.stat()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ClippedVideoCollectionEvidenceError(
            f"Cannot resolve live {label}: {exc}"
        ) from exc
    if (
        type(value["size_bytes"]) is not int
        or type(value["mtime_ns"]) is not int
        or int(stat.st_size) != value["size_bytes"]
        or int(stat.st_mtime_ns) != value["mtime_ns"]
    ):
        raise ClippedVideoCollectionEvidenceError(
            f"Live {label} stat differs from persisted evidence."
        )
    expected_sha256 = value["sha256"]
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or _file_sha256(path) != expected_sha256
    ):
        raise ClippedVideoCollectionEvidenceError(
            f"Live {label} content differs from persisted SHA-256 evidence."
        )
    return path


def verify_clipped_video_collection_live_files(
    recording_dir: str | Path,
    metadata: Mapping[str, Any],
) -> VerifiedClippedVideoCollectionFiles:
    """Resolve and verify every file named by canonical clipped-source metadata.

    The importer already validates video geometry and constructs the exact
    recording-wide frame map.  This read-side verifier checks that the three
    indexed mapping artifacts still have their persisted content hashes and
    that every encoded member still has the exact cheap ``stat_v1`` identity
    captured at import.  It intentionally does not decode or content-hash the
    large videos.
    """

    recording = Path(recording_dir).expanduser().resolve()
    if not recording.is_dir():
        raise ClippedVideoCollectionEvidenceError(
            f"Clipped recording directory not found: {recording}"
        )
    if not isinstance(metadata, Mapping) or (
        metadata.get("schema_id") != SOURCE_VIDEO_COLLECTION_METADATA_SCHEMA_ID
        or metadata.get("layout") != SOURCE_VIDEO_COLLECTION_LAYOUT
    ):
        raise ClippedVideoCollectionEvidenceError(
            "Canonical clipped source-video metadata is required."
        )
    collection = metadata.get("collection")
    if not isinstance(collection, Mapping) or (
        collection.get("schema_id") != SOURCE_VIDEO_COLLECTION_SCHEMA_ID
        or collection.get("schema_version") != 1
        or collection.get("fingerprint_strategy")
        != SOURCE_VIDEO_COLLECTION_FINGERPRINT_STRATEGY
    ):
        raise ClippedVideoCollectionEvidenceError(
            "Canonical clipped collection evidence is required."
        )

    index_paths = tuple(
        _verify_file_evidence(
            recording,
            collection.get(name),
            label=name,
        )
        for name in (
            "recording_clip_index",
            "recording_frame_index",
            "recording_frame_index_manifest",
        )
    )
    raw_members = collection.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ClippedVideoCollectionEvidenceError(
            "Clipped collection contains no source-video members."
        )
    camera_id = metadata.get("camera_id")
    member_paths: list[Path] = []
    seen_paths: set[Path] = set()
    expected_start = 0
    previous_clip_index = -1
    for member_index, member in enumerate(raw_members):
        label = f"collection member {member_index}"
        if not isinstance(member, Mapping):
            raise ClippedVideoCollectionEvidenceError(f"{label} is not an object.")
        try:
            path = _resolve_under(
                recording,
                member.get("relative_path"),
                label=label,
            )
            clip_index = member["clip_index"]
            frame_count = member["frame_count"]
            first_frame = member["first_frame_index"]
            last_frame = member["last_frame_index_inclusive"]
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            raise ClippedVideoCollectionEvidenceError(
                f"Cannot resolve live {label}: {exc}"
            ) from exc
        if path in seen_paths:
            raise ClippedVideoCollectionEvidenceError(
                "Clipped collection resolves multiple members to the same video file."
            )
        seen_paths.add(path)
        if (
            type(clip_index) is not int
            or clip_index <= previous_clip_index
            or type(frame_count) is not int
            or frame_count <= 0
            or type(first_frame) is not int
            or first_frame != expected_start
            or type(last_frame) is not int
            or last_frame != expected_start + frame_count - 1
        ):
            raise ClippedVideoCollectionEvidenceError(
                "Clipped collection members do not exactly tile the acquisition "
                "timeline in clip-index order."
            )
        previous_clip_index = clip_index
        expected_start += frame_count
        try:
            fingerprint = source_stat_fingerprint_attrs(
                path,
                attr_prefix="source_video",
                extra={
                    "relative_path": member["relative_path"],
                    "clip_id": member["clip_id"],
                    "clip_index": clip_index,
                    "camera_id": camera_id,
                    "width": member["width"],
                    "height": member["height"],
                    "fps": member["fps"],
                    "frame_count": frame_count,
                    "codec": member["codec"],
                    "pix_fmt": member["pix_fmt"],
                },
            )
        except (KeyError, OSError) as exc:
            raise ClippedVideoCollectionEvidenceError(
                f"Cannot fingerprint live {label}: {exc}"
            ) from exc
        live_identity = {
            "strategy": fingerprint["source_video_fingerprint_strategy"],
            "value": fingerprint["source_video_fingerprint"],
            "size_bytes": fingerprint["source_video_size_bytes"],
            "mtime_ns": fingerprint["source_video_mtime_ns"],
            "relocation_stable": False,
        }
        if member.get("file_fingerprint") != live_identity:
            raise ClippedVideoCollectionEvidenceError(
                f"Live {label} differs from its persisted stat_v1 fingerprint."
            )
        member_paths.append(path)
    if expected_start != metadata.get("total_frames"):
        raise ClippedVideoCollectionEvidenceError(
            "Clipped collection coverage differs from source total_frames."
        )
    collection_sha256 = collection.get("collection_sha256")
    if type(collection_sha256) is not str or len(collection_sha256) != 64:
        raise ClippedVideoCollectionEvidenceError(
            "Clipped collection digest is missing or malformed."
        )
    return VerifiedClippedVideoCollectionFiles(
        recording_dir=recording,
        index_paths=index_paths,
        member_paths=tuple(member_paths),
        collection_sha256=collection_sha256,
    )


def _clip_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = payload.get("clips")
    if raw_rows is None:
        raw_rows = payload.get("rows")
    if raw_rows is None:
        raw_rows = payload.get("camera_artifacts")
    if not isinstance(raw_rows, list):
        raise ValueError(
            "recording_clip_index must include clips, rows, or camera_artifacts."
        )
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("recording_clip_index contains a non-object row.")
        artifacts = raw.get("camera_artifacts")
        if isinstance(artifacts, list):
            base = {
                key: value for key, value in raw.items() if key != "camera_artifacts"
            }
            for artifact in artifacts:
                if not isinstance(artifact, Mapping):
                    raise ValueError("camera_artifacts contains a non-object row.")
                rows.append({**base, **dict(artifact)})
        else:
            rows.append(dict(raw))
    if not rows:
        raise ValueError("recording_clip_index contains no clip-camera rows.")
    return rows


def _frame_index_units(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    required = {
        "camera_serial",
        "clip_id",
        "clip_index",
        "clip_local_frame_index",
        "parent_frame_index",
        "video_path",
    }
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"recording_frame_index is missing columns: {missing}")
    table = pq.read_table(path, columns=sorted(required)).combine_chunks()
    grouped = table.group_by(
        ["camera_serial", "clip_id", "clip_index", "video_path"]
    ).aggregate(
        [
            ("parent_frame_index", "min"),
            ("parent_frame_index", "max"),
            ("parent_frame_index", "count"),
            ("clip_local_frame_index", "min"),
            ("clip_local_frame_index", "max"),
        ]
    )
    units: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in grouped.to_pylist():
        key = (
            str(row["camera_serial"]),
            str(row["clip_id"]),
            int(row["clip_index"]),
        )
        if key in units:
            raise ValueError(
                f"recording_frame_index has multiple video paths for {key}."
            )
        start = int(row["parent_frame_index_min"])
        stop = int(row["parent_frame_index_max"]) + 1
        count = int(row["parent_frame_index_count"])
        if stop - start != count:
            raise ValueError(
                f"recording_frame_index parent frames are not contiguous for {key}."
            )
        if (
            int(row["clip_local_frame_index_min"]) != 0
            or int(row["clip_local_frame_index_max"]) != count - 1
        ):
            raise ValueError(
                f"recording_frame_index clip-local frames are not dense for {key}."
            )
        units[key] = {
            "frame_start": start,
            "frame_stop": stop,
            "frame_count": count,
            "video_path": str(row["video_path"]),
        }
    ordered = sorted(units.values(), key=lambda value: int(value["frame_start"]))
    expected_start = 0
    for unit in ordered:
        if int(unit["frame_start"]) != expected_start:
            raise ValueError(
                "recording_frame_index work units do not cover one contiguous parent timeline."
            )
        expected_start = int(unit["frame_stop"])
    if expected_start != int(table.num_rows):
        raise ValueError(
            "recording_frame_index row count differs from its complete work-unit coverage."
        )
    return units


def _keyframe_metadata(
    recording_dir: Path, row: Mapping[str, Any]
) -> Mapping[str, Any]:
    value = row.get("keyframe_path") or row.get("keyframes")
    if value in (None, ""):
        return {}
    path = _resolve_under(recording_dir, value, label="clip keyframe metadata")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid keyframe metadata JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Keyframe metadata must be an object: {path}")
    return payload


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be an exact positive integer; got {value!r}.")
    return value


def _positive_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive; got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive; got {value!r}.")
    return result


def _consistent_number(
    probed: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    name: str,
    *,
    integer: bool,
) -> int | float:
    parser = _positive_int if integer else _positive_float
    observed = probed.get(name)
    declared = sidecar.get(name)
    if observed is None and declared is None:
        raise ValueError(f"Could not resolve clip video {name}.")
    resolved = parser(observed if observed is not None else declared, label=name)
    if observed is not None and declared is not None:
        other = parser(declared, label=f"keyframe {name}")
        equal = (
            resolved == other
            if integer
            else math.isclose(float(resolved), float(other), rel_tol=0.0, abs_tol=1e-6)
        )
        if not equal:
            raise ValueError(
                f"Clip video and keyframe metadata disagree on {name}: "
                f"{resolved!r} != {other!r}."
            )
    return resolved


def build_clipped_video_collection_metadata(
    recording_dir: str | Path,
    *,
    clip_index_path: str | Path | None = None,
    frame_index_path: str | Path | None = None,
    frame_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one camera's clipped source and return canonical metadata."""

    recording = Path(recording_dir).expanduser().resolve()
    # Local import keeps the metadata schema usable by pixel_frame_authority
    # without creating an import cycle through import_video_metadata.
    from fisheye.shared.import_video_metadata import probe_ffprobe_video_metadata

    if not recording.is_dir():
        raise NotADirectoryError(recording)
    clip_index = _resolve_under(
        recording,
        clip_index_path or "recording_clip_index.json",
        label="recording clip index",
    )
    frame_index = _resolve_under(
        recording,
        frame_index_path or "recording_frame_index.parquet",
        label="recording frame index",
    )
    frame_manifest = _resolve_under(
        recording,
        frame_manifest_path or "recording_frame_index_manifest.json",
        label="recording frame-index manifest",
    )
    try:
        clip_payload = json.loads(clip_index.read_text(encoding="utf-8"))
        manifest_payload = json.loads(frame_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid clipped recording JSON: {exc}") from exc
    if not isinstance(clip_payload, Mapping) or not isinstance(
        manifest_payload, Mapping
    ):
        raise ValueError("Clipped recording indexes must be JSON objects.")
    if manifest_payload.get("status") != "ok":
        raise ValueError("recording frame-index manifest is not complete.")
    rows = _clip_rows(clip_payload)
    units = _frame_index_units(frame_index)
    camera_serials = sorted({str(row.get("camera_serial") or "") for row in rows})
    if len(camera_serials) != 1 or not camera_serials[0]:
        raise ValueError(
            "Canonical clipped acquisition authority requires one camera stream per recording."
        )
    camera_id = camera_serials[0]
    if list(manifest_payload.get("camera_serials") or []) != [camera_id]:
        raise ValueError("Frame-index manifest camera identity differs from clip rows.")
    if (
        int(manifest_payload.get("row_count") or 0)
        != pq.ParquetFile(frame_index).metadata.num_rows
    ):
        raise ValueError(
            "Frame-index manifest row count differs from Parquet metadata."
        )

    members: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for row in sorted(
        rows,
        key=lambda value: (
            int(value.get("clip_index") or 0),
            str(value.get("clip_id") or ""),
        ),
    ):
        clip_index_value = int(row.get("clip_index") or 0)
        clip_id = str(row.get("clip_id") or f"clip_{clip_index_value:06d}")
        key = (camera_id, clip_id, clip_index_value)
        if key in seen:
            raise ValueError(f"Duplicate clipped acquisition member: {key}")
        seen.add(key)
        try:
            unit = units[key]
        except KeyError as exc:
            raise ValueError(
                f"Frame index has no work unit for clipped member {key}."
            ) from exc
        video = _resolve_under(recording, row.get("video_path"), label="clip video")
        indexed_video = _resolve_under(
            recording, unit["video_path"], label="frame-index clip video"
        )
        if indexed_video != video:
            raise ValueError(
                f"Clip index and frame index select different videos for {key}."
            )
        declared_count = _positive_int(
            int(row.get("frame_count") or 0), label="clip frame_count"
        )
        if declared_count != int(unit["frame_count"]):
            raise ValueError(
                f"Clip and frame indexes disagree on frame count for {key}."
            )
        sidecar = _keyframe_metadata(recording, row)
        probed = probe_ffprobe_video_metadata(video)
        width = _positive_int(probed.get("width"), label="video width")
        height = _positive_int(probed.get("height"), label="video height")
        total_frames = int(
            _consistent_number(
                probed,
                sidecar,
                "total_frames",
                integer=True,
            )
        )
        fps = float(_consistent_number(probed, sidecar, "fps", integer=False))
        if total_frames != declared_count:
            raise ValueError(
                f"Decoded clip extent differs from indexed frame count for {key}: "
                f"{total_frames} != {declared_count}."
            )
        codec = probed.get("codec") or sidecar.get("codec")
        pix_fmt = probed.get("pix_fmt")
        fingerprint = source_stat_fingerprint_attrs(
            video,
            attr_prefix="source_video",
            extra={
                "relative_path": _relative_path(recording, video),
                "clip_id": clip_id,
                "clip_index": clip_index_value,
                "camera_id": camera_id,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": total_frames,
                "codec": codec,
                "pix_fmt": pix_fmt,
            },
        )
        members.append(
            {
                "clip_id": clip_id,
                "clip_index": clip_index_value,
                "relative_path": _relative_path(recording, video),
                "frame_count": total_frames,
                "first_frame_index": int(unit["frame_start"]),
                "last_frame_index_inclusive": int(unit["frame_stop"]) - 1,
                "width": width,
                "height": height,
                "fps": fps,
                "codec": str(codec) if codec not in (None, "") else None,
                "pix_fmt": str(pix_fmt) if pix_fmt not in (None, "") else None,
                "file_fingerprint": {
                    "strategy": fingerprint["source_video_fingerprint_strategy"],
                    "value": fingerprint["source_video_fingerprint"],
                    "size_bytes": fingerprint["source_video_size_bytes"],
                    "mtime_ns": fingerprint["source_video_mtime_ns"],
                    "relocation_stable": False,
                },
            }
        )
    if set(units) != seen:
        raise ValueError("Frame index contains work units absent from the clip index.")
    widths = {int(member["width"]) for member in members}
    heights = {int(member["height"]) for member in members}
    fps_values = {float(member["fps"]) for member in members}
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError("Clipped camera geometry changes across source members.")
    first_fps = float(members[0]["fps"])
    if any(
        not math.isclose(value, first_fps, rel_tol=0.0, abs_tol=1e-6)
        for value in fps_values
    ):
        raise ValueError("Clipped camera frame rate changes across source members.")
    total_frames = sum(int(member["frame_count"]) for member in members)
    if total_frames != pq.ParquetFile(frame_index).metadata.num_rows:
        raise ValueError(
            "Clipped member frame counts do not equal the parent timeline."
        )

    collection_basis = {
        "schema_id": SOURCE_VIDEO_COLLECTION_SCHEMA_ID,
        "schema_version": 1,
        "fingerprint_strategy": SOURCE_VIDEO_COLLECTION_FINGERPRINT_STRATEGY,
        "recording_clip_index": _file_evidence(recording, clip_index),
        "recording_frame_index": _file_evidence(recording, frame_index),
        "recording_frame_index_manifest": _file_evidence(recording, frame_manifest),
        "members": members,
    }
    collection = {
        **collection_basis,
        "collection_sha256": _canonical_sha256(collection_basis),
    }
    codecs = {member["codec"] for member in members}
    pixel_formats = {member["pix_fmt"] for member in members}
    return {
        "schema_id": SOURCE_VIDEO_COLLECTION_METADATA_SCHEMA_ID,
        "layout": SOURCE_VIDEO_COLLECTION_LAYOUT,
        "camera_id": camera_id,
        "width": widths.pop(),
        "height": heights.pop(),
        "total_frames": total_frames,
        "fps": first_fps,
        "codec": codecs.pop() if len(codecs) == 1 else None,
        "pix_fmt": pixel_formats.pop() if len(pixel_formats) == 1 else None,
        "locator": {
            "kind": SOURCE_VIDEO_COLLECTION_LOCATOR_KIND,
            "relative_path": _relative_path(recording, frame_index),
        },
        "collection": collection,
    }


def clipped_video_collection_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    collection = metadata.get("collection")
    members = collection.get("members") if isinstance(collection, Mapping) else []
    return {
        "schema_id": metadata.get("schema_id"),
        "layout": metadata.get("layout"),
        "camera_id": metadata.get("camera_id"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "total_frames": metadata.get("total_frames"),
        "fps": metadata.get("fps"),
        "member_count": len(members) if isinstance(members, list) else None,
        "collection_sha256": (
            collection.get("collection_sha256")
            if isinstance(collection, Mapping)
            else None
        ),
    }


__all__ = [
    "ClippedVideoCollectionEvidenceError",
    "SOURCE_VIDEO_COLLECTION_FINGERPRINT_STRATEGY",
    "SOURCE_VIDEO_COLLECTION_LAYOUT",
    "SOURCE_VIDEO_COLLECTION_LOCATOR_KIND",
    "SOURCE_VIDEO_COLLECTION_METADATA_SCHEMA_ID",
    "SOURCE_VIDEO_COLLECTION_SCHEMA_ID",
    "VerifiedClippedVideoCollectionFiles",
    "build_clipped_video_collection_metadata",
    "clipped_video_collection_summary",
    "verify_clipped_video_collection_live_files",
]
