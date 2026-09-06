"""Palette's read-only Citrus transfer-v2 boundary and parent intake plans.

Protocol-specific reconstruction is adapted from Citrus's reference validator at
e881f5258be83231b62a00a6f9c4e5fcd69cd548 (scripts/recording_transfer_snapshot.py).
The exact external canonicalization, frame-map and finalization grammar is kept
here; Palette identity and JSON reading use their existing owners. This module
never copies payloads, submits jobs, publishes authority, or mints import receipts.
Transport verification and a parent plan are NOT scientific/media admission.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from fisheye.shared.source_recording_identity import (
    load_strict_json_object,
    recording_id_from_session_camera,
)

SNAPSHOT_SCHEMA = "citrus.recording_transfer_snapshot"
MARKER_SCHEMA = "citrus.transfer_completion_marker.v2"
CONTROL_DIR = "_citrus_transfer"
SNAPSHOT_PATH = f"{CONTROL_DIR}/snapshot.json"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_CSV_LINE = 65536
MARKER_NAME = "_citrus_transfer_complete.json"


class TransferSnapshotError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransferSnapshotError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def strict_json(path: Path) -> dict:
    info = path.lstat()
    require(
        stat.S_ISREG(info.st_mode) and info.st_size <= MAX_JSON_BYTES,
        f"nonregular or oversized JSON: {path}",
    )
    return load_strict_json_object(path, max_bytes=MAX_JSON_BYTES)


def uint(value: Any, label: str, *, positive: bool = False) -> int:
    require(
        type(value) is int and (1 if positive else 0) <= value <= 2**64 - 1,
        f"invalid uint64 {label}",
    )
    return value


def identifier(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and bool(value)
        and len(value) <= 1024
        and all(ord(c) >= 32 for c in value),
        f"invalid {label}",
    )
    return value


def normalized_path(value: Any) -> str:
    require(
        isinstance(value, str)
        and value
        and value != "."
        and "\\" not in value
        and all(ord(c) >= 32 for c in value),
        "invalid artifact path",
    )
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and str(path) == value
        and all(p not in (".", "..") for p in path.parts),
        "non-normalized artifact path",
    )
    return value


def require_no_errors(
    record: dict, *, codes: tuple[str, ...] = (), messages: tuple[str, ...] = ()
) -> None:
    """Absent/null errors and integer zero/empty text are not failure evidence.

    Orange emits null for successful operations. Do not let success booleans
    override a present error, or Python's bool/int equality admit malformed codes.
    """
    for field in codes:
        value = record.get(field)
        require(
            value is None or (type(value) is int and value == 0),
            f"{field} contradicts successful finalization",
        )
    for field in messages:
        value = record.get(field)
        require(
            value is None or (type(value) is str and value == ""),
            f"{field} contradicts successful finalization",
        )


def require_clip_closure(
    clip: dict, session: str, *, final: bool, rolling: bool
) -> None:
    require(clip.get("drain_completed") is True, "clip not drained")
    # Native single_clip entries omit these redundant fields. Rolling entries
    # require them; when supplied in either layout they must agree with the parent.
    for field, expected in (
        ("session_id", session),
        ("status", "completed"),
        ("final_clip", final),
    ):
        if rolling or field in clip:
            require(
                type(clip.get(field)) is type(expected) and clip[field] == expected,
                f"clip {field} contradicts parent/final closure",
            )
    if rolling or "rollover" in clip:
        rollover = clip.get("rollover")
        require(
            isinstance(rollover, dict) and rollover.get("pending_next_clip") is False,
            "clip rollover is pending or invalid",
        )


def source_relative(root: Path, value: Any, declared_root: str) -> str:
    identifier(value, "source path")
    path = Path(value)
    if path.is_absolute():
        # Preserve historical manifest bytes while materializing portable refs.
        # Only the explicitly declared original recording root may be rebased.
        require(
            bool(declared_root) and Path(declared_root).is_absolute(),
            "absolute source reference lacks a declared recording root",
        )
        try:
            path = path.relative_to(declared_root)
        except ValueError as error:
            raise TransferSnapshotError(
                "source reference escapes declared recording root"
            ) from error
    rel = normalized_path(path.as_posix())
    require((root / rel).is_file(), f"missing artifact: {rel}")
    require(
        not any(
            (root / Path(*Path(rel).parts[:i])).is_symlink()
            for i in range(1, len(Path(rel).parts) + 1)
        ),
        f"symlink artifact: {rel}",
    )
    return rel


def file_ref(root: Path, relative: str) -> dict:
    normalized_path(relative)
    path = root / relative
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f"not a regular artifact: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        require(
            os.fstat(stream.fileno()).st_ino == before.st_ino,
            "artifact replaced during open",
        )
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
        opened = os.fstat(stream.fileno())
    after = path.lstat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(
        signature(before) == signature(opened) == signature(after),
        f"artifact mutated while hashing: {relative}",
    )
    return {"path": relative, "size_bytes": after.st_size, "sha256": digest.hexdigest()}


def inventory(root: Path, marker_name: str, *, destination: bool = False) -> list[dict]:
    result = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            require(not path.is_symlink(), f"symlink directory: {path}")
            if path == root / CONTROL_DIR:
                require(
                    destination, "source contains reserved transfer control directory"
                )
                dirs.remove(name)
        for name in files:
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if path == root / marker_name and destination:
                continue
            require(
                name != marker_name and name != "_citrus_transfer_complete.json",
                f"nested or source completion marker: {rel}",
            )
            result.append(file_ref(root, rel))
    return sorted(result, key=lambda entry: entry["path"])


def frame_map(
    path: Path,
    kind: str,
    previous: int,
    crop_offset: int,
    *,
    on_row: Callable[[int, int, int, int], None] | None = None,
) -> dict:
    """Validate source CSV and optionally stream exact validated rows.

    The callback receives local index, recording ID, timestamp and timestamp_sys.
    It does not change the transport serialization/digest or clock claim. A
    callback writer must remain unpublished until the complete map and immutable
    source generation have been revalidated.
    """
    count = first = last = gaps = 0
    mapping_hash = hashlib.sha256()
    with path.open("r", encoding="utf-8", newline="") as stream:

        def lines():
            while line := stream.readline(MAX_CSV_LINE + 1):
                require(
                    len(line) <= MAX_CSV_LINE and line.endswith("\n"),
                    "oversized or partial metadata row",
                )
                yield line

        rows = csv.reader(lines(), strict=True)
        header = next(rows, [])
        require(
            len(header) == len(set(header)) and "" not in header,
            "duplicate/empty metadata column",
        )
        required = {"recording_frame_id", "timestamp", "timestamp_sys"}
        require(
            required <= set(header), "metadata lacks exact source identity/timestamps"
        )
        alias = "frame_id" in header
        local_crop = "crop_video_frame_index" in header
        session_crop = "session_crop_video_frame_index" in header
        if kind == "crop":
            require(
                local_crop and session_crop,
                "crop metadata lacks local/session output indices",
            )
        for row in rows:
            require(len(row) == len(header), "malformed metadata row")
            fields = dict(zip(header, row))

            def integer(key: str) -> int:
                require(
                    re.fullmatch(r"[0-9]+", fields[key]) is not None,
                    f"noninteger metadata {key}",
                )
                return uint(int(fields[key]), key)

            frame = integer("recording_frame_id")
            require(
                frame > (last or previous),
                "duplicate, reset, or misordered recording frame",
            )
            if alias:
                require(integer("frame_id") == frame, "recording alias mismatch")
            timestamp = integer("timestamp")
            timestamp_sys = integer("timestamp_sys")
            if kind == "crop":
                require(
                    integer("crop_video_frame_index") == count
                    and integer("session_crop_video_frame_index")
                    == crop_offset + count,
                    "mispaired crop-local/session index",
                )
            if not count:
                first = frame
            else:
                gaps += frame - last - 1
            mapping_hash.update(f"{count},{frame}\n".encode("ascii"))
            if on_row is not None:
                on_row(count, frame, timestamp, timestamp_sys)
            last = frame
            count += 1
    require(count > 0, "empty output frame domain")
    return {
        "recording_frame_id_field": "recording_frame_id",
        "recording_frame_id_base": 1,
        "frame_id_alias_present": alias,
        "video_frame_index_base": 0,
        "video_frame_index_rule": "metadata_row_order",
        "continuity_policy": "strictly_increasing_recording_id_subset",
        "timestamp_fields": ["timestamp", "timestamp_sys"],
        "timestamp_units": "nanoseconds",
        "clock_validity": "not_evaluated_by_transfer",
        "frame_count": count,
        "first_recording_frame_id": first,
        "last_recording_frame_id": last,
        "recording_frame_id_gaps": gaps,
        "gap_from_previous_output": first - previous - 1 if previous else None,
        "row_correspondence_sha256": mapping_hash.hexdigest(),
        "row_correspondence_digest_domain": "ascii_video_index_comma_recording_id_lf_v1",
        "correspondence_authority": "producer_metadata_row_order_with_packet_count_parity",
    }


def build_snapshot(root: Path, marker_name: str, *, destination: bool = False) -> dict:
    root = root.resolve(strict=True)
    require(root != Path("/") and root.is_dir(), "invalid recording root")
    normalized_path(marker_name)
    require(
        Path(marker_name).name == marker_name
        and marker_name not in (".", "..", CONTROL_DIR),
        "marker must be a filename",
    )
    # Hash before parsing, then verify these same bytes again before publication.
    items = inventory(root, marker_name, destination=destination)
    refs = {item["path"]: item for item in items}
    require(
        "recording_session.json" in refs, "v2 requires Orange recording_session.json"
    )
    manifest = strict_json(root / "recording_session.json")
    require(
        manifest.get("schema_id") == "orange.recording_session"
        and type(manifest.get("schema_version")) is int
        and manifest["schema_version"] == 1,
        "unsupported Orange session manifest",
    )
    require(
        manifest.get("status") == "completed"
        and manifest.get("recording", {}).get("drain_completed") is True,
        "recording is not completed and drained",
    )
    mode = manifest.get("mode")
    require(mode in ("single_clip", "rolling_clips"), "unknown recording layout")
    rolling = mode == "rolling_clips"
    layout = "rolling_clips" if rolling else "single_video"
    session = identifier(manifest.get("session_id"), "session identity")
    cameras = manifest.get("cameras")
    require(isinstance(cameras, list) and cameras, "missing cameras")
    cameras = [identifier(c, "camera serial") for c in cameras]
    require(len(cameras) == len(set(cameras)), "duplicate camera serial")
    declared_root = manifest.get("recording_folder", "")
    clips = manifest.get("clips")
    require(
        isinstance(clips, list) and clips and (rolling or len(clips) == 1),
        "invalid clip membership for declared mode",
    )
    ids = set()
    parents = {
        c: {"parent_key": {"recording_id": session, "camera_serial": c}, "clips": []}
        for c in sorted(cameras)
    }
    previous: dict[tuple, int] = {}
    offsets: dict[tuple, int] = {}
    used_media: set[str] = set()
    used_metadata: set[str] = set()
    used_dirs: set[str] = set()
    output_kinds: dict[str, set[str]] = {}

    def ref(value: str) -> dict:
        rel = source_relative(root, value, declared_root)
        require(rel in refs, f"artifact absent from frozen inventory: {rel}")
        return refs[rel]

    indexes = manifest.get("indexes", {})
    for field in ("clip_index_json", "clip_index_csv"):
        if field in indexes:
            ref(indexes[field])

    for index, clip in enumerate(clips):
        require(
            type(clip.get("clip_index")) is int and clip["clip_index"] == index,
            "missing/duplicate/misordered clip index",
        )
        clip_id = identifier(clip.get("clip_id"), "clip identity")
        require(clip_id not in ids, "duplicate clip identity")
        ids.add(clip_id)
        require_clip_closure(
            clip, session, final=index == len(clips) - 1, rolling=rolling
        )
        directory = clip.get("directory")
        if rolling:
            directory = normalized_path(directory)
            require(directory not in used_dirs, "duplicate clip directory")
            used_dirs.add(directory)
        else:
            require(directory == ".", "single-video clip must use parent directory")
        outputs = clip.get("recording_outputs")
        require(
            isinstance(outputs, dict) and set(outputs) == set(cameras),
            "clip camera membership mismatch",
        )
        if not rolling:
            require(
                manifest.get("recording_outputs") == outputs,
                "single-video parent/child output declarations disagree",
            )
        clip_manifest_path = (
            f"{directory}/clip_manifest.json" if rolling else "clip_manifest.json"
        )
        if clip_manifest_path in refs:
            child = strict_json(root / clip_manifest_path)
            require(
                child.get("schema_id") == "orange.recording_clip"
                and type(child.get("schema_version")) is int
                and child["schema_version"] == 1,
                "unsupported child manifest",
            )
            require_clip_closure(
                child, session, final=index == len(clips) - 1, rolling=rolling
            )
            for field in (
                "session_id",
                "clip_id",
                "clip_index",
                "status",
                "drain_completed",
                "final_clip",
                "recording_outputs",
            ):
                require(
                    child.get(field) == clip.get(field),
                    f"child manifest mismatch: {field}",
                )
        for camera in sorted(cameras):
            kinds = outputs[camera]
            require(
                isinstance(kinds, dict) and kinds and set(kinds) <= {"full", "crop"},
                "unsupported or missing output kind",
            )
            if camera in output_kinds:
                require(
                    set(kinds) == output_kinds[camera], "mixed per-clip output layout"
                )
            output_kinds[camera] = set(kinds)
            media = []
            for kind, output in sorted(kinds.items()):
                require(
                    output.get("camera_serial") == camera
                    and output.get("output_kind") == kind
                    and output.get("status") in ("completed", "finalized"),
                    "output camera/kind/finalization mismatch",
                )
                video, metadata = ref(output.get("video")), ref(output.get("metadata"))
                artifacts = (clip if rolling else manifest).get("camera_artifacts", {})
                if kind == "full" and artifacts:
                    require(
                        isinstance(artifacts, dict) and set(artifacts) == set(cameras),
                        "camera artifact membership mismatch",
                    )
                    for field, expected in (("video", video), ("metadata", metadata)):
                        require(
                            ref(artifacts[camera].get(field)) == expected,
                            "camera artifact/output correspondence mismatch",
                        )
                require(
                    video["path"] not in used_media
                    and metadata["path"] not in used_metadata,
                    "duplicate/mispaired output artifact",
                )
                used_media.add(video["path"])
                used_metadata.add(metadata["path"])
                if rolling:
                    require(
                        all(
                            PurePosixPath(p["path"]).is_relative_to(directory)
                            for p in (video, metadata)
                        ),
                        "output outside its clip directory",
                    )
                sidecars = [
                    {"role": field, "artifact": ref(output[field])}
                    for field in ("keyframes", "perf", "sidecar_perf", "summary")
                    if output.get(field)
                ]
                key = camera, kind
                mapping = frame_map(
                    root / metadata["path"],
                    kind,
                    previous.get(key, 0),
                    offsets.get(key, 0),
                )
                for field in (
                    "frame_count",
                    "first_recording_frame_id",
                    "last_recording_frame_id",
                    "recording_frame_id_gaps",
                ):
                    if field in output:
                        require(
                            uint(output[field], field) == mapping[field],
                            f"declared {field} mismatch",
                        )
                final_ref = ref(video["path"] + ".finalization.json")
                final = strict_json(root / final_ref["path"])
                require(
                    final.get("schema_id") == "orange.video_container_finalization"
                    and type(final.get("schema_version")) is int
                    and final["schema_version"] == 2
                    and final.get("terminal") is True
                    and final.get("status") == "complete"
                    and final.get("container", {}).get("finalized") is True,
                    "missing/failed container finalization",
                )
                require(
                    uint(
                        final.get("container", {}).get("file_size_bytes"),
                        "container size",
                    )
                    == video["size_bytes"],
                    "finalization media size mismatch",
                )
                require(
                    all(
                        final["container"].get(field) is True
                        for field in (
                            "header_written",
                            "trailer_written",
                            "output_closed",
                        )
                    ),
                    "contradictory container finalization evidence",
                )
                require_no_errors(
                    final["container"],
                    codes=("trailer_error_code", "output_close_error_code"),
                    messages=("trailer_error", "output_close_error", "file_size_error"),
                )
                require(
                    source_relative(root, final.get("video_path"), declared_root)
                    == video["path"],
                    "finalization video binding mismatch",
                )
                packet = final.get("packet_writes", {})
                require(
                    packet.get("complete") is True
                    and packet.get("writer_error_latched") is False
                    and packet.get("muxer_flush_attempted") is True
                    and packet.get("muxer_flush_succeeded") is True,
                    "incomplete mux evidence",
                )
                require_no_errors(
                    packet,
                    codes=("first_write_error_code", "muxer_flush_error_code"),
                    messages=("muxer_flush_error",),
                )
                for field in (
                    "submissions_accepted",
                    "write_attempts",
                    "packets_written",
                ):
                    require(
                        uint(packet.get(field), field) == mapping["frame_count"],
                        "packet/metadata mismatch",
                    )
                require(
                    uint(packet.get("submissions_rejected"), "rejections") == 0
                    and uint(packet.get("write_failures"), "write failures") == 0,
                    "packet failure in finalized output",
                )
                previous[key] = mapping["last_recording_frame_id"]
                offsets[key] = offsets.get(key, 0) + mapping["frame_count"]
                media.append(
                    {
                        "output_kind": kind,
                        "role": identifier(output.get("role"), "output role"),
                        "video": video,
                        "metadata": metadata,
                        "container_finalization": final_ref,
                        "frame_map": mapping,
                        "sidecars": sidecars,
                    }
                )
            parents[camera]["clips"].append(
                {
                    "clip_index": index,
                    "clip_id": clip_id,
                    "directory": directory,
                    "outputs": media,
                }
            )
    # Current supported envelope has only declared full/crop media. A retained
    # shard/preview needs an explicit future auxiliary-media profile, not guessing.
    videos = {
        p
        for p in refs
        if Path(p).suffix.lower() in (".mp4", ".mkv", ".avi", ".h265", ".hevc")
    }
    require(videos == used_media, "extra/undeclared or unsupported video artifact")
    child_manifests = {p for p in refs if PurePosixPath(p).name == "clip_manifest.json"}
    expected_children = (
        {f"{d}/clip_manifest.json" for d in used_dirs}
        if rolling
        else {"clip_manifest.json"}
    )
    require(child_manifests <= expected_children, "undeclared child manifest")
    payload = (
        "citrus_h5"
        if any(Path(p).suffix.lower() in (".h5", ".hdf5") for p in refs)
        else "video_only"
    )
    return {
        "schema_id": SNAPSHOT_SCHEMA,
        "schema_version": 1,
        "canonicalization": "json_sort_keys_ascii_compact_lf_v1",
        "recording_layout": layout,
        "recording_payload_kind": payload,
        "acquisition_session_id": session,
        "parent_identity_policy": "orange_session_id_and_camera_serial_v1",
        "source_manifest": {
            **refs["recording_session.json"],
            "schema_id": "orange.recording_session",
            "schema_version": 1,
        },
        "finalization": {
            "manifest_status": "completed",
            "drain_completed": True,
            "semantic_receipt_acceptance": "not_evaluated_by_transfer",
        },
        "parents": list(parents.values()),
        "inventory": items,
    }


def snapshot_bytes_and_id(snapshot: dict) -> tuple[bytes, str]:
    data = canonical_bytes(snapshot)
    return data, "sha256:" + hashlib.sha256(data).hexdigest()


def verify_inventory(
    root: Path, snapshot: dict, marker_name: str, *, destination: bool
) -> None:
    require(
        inventory(root, marker_name, destination=destination) == snapshot["inventory"],
        "snapshot inventory mismatch (missing, extra, changed, or replaced artifact)",
    )


@dataclass(frozen=True)
class VerifiedTransferSnapshot:
    """Ephemeral observation of an immutable delivery, never a reuse receipt."""

    root: Path
    snapshot_id: str
    attempt_id: str
    recording_layout: str
    recording_payload_kind: str
    snapshot: dict


@dataclass(frozen=True)
class IntakeOutput:
    output_kind: str
    role: str
    video: str
    metadata: str
    frame_count: int
    first_recording_frame_id: int
    last_recording_frame_id: int


@dataclass(frozen=True)
class IntakeClip:
    clip_index: int
    clip_id: str
    directory: str
    outputs: tuple[IntakeOutput, ...]


@dataclass(frozen=True)
class ParentRecordingIntakePlan:
    """One source recording, with original clip/output names and row identities."""

    recording_id: str
    session_uuid: str
    camera_id: str
    recording_layout: str
    recording_payload_kind: str
    snapshot_id: str
    total_frames: int
    clips: tuple[IntakeClip, ...]


def _verify_transfer_snapshot(root: Path) -> VerifiedTransferSnapshot:
    require(not root.is_symlink(), "delivery root cannot be a symlink")
    root = root.resolve(strict=True)
    marker_path = root / MARKER_NAME
    snapshot_path = root / SNAPSHOT_PATH
    require(
        not (root / CONTROL_DIR).is_symlink(), "snapshot control directory is a symlink"
    )
    marker = strict_json(marker_path)
    snapshot = strict_json(snapshot_path)
    marker_bytes = marker_path.read_bytes()
    snapshot_bytes = snapshot_path.read_bytes()
    rebuilt = build_snapshot(root, MARKER_NAME, destination=True)
    require(
        canonical_bytes(snapshot) == canonical_bytes(rebuilt),
        "snapshot schema, membership or source semantics mismatch",
    )
    data, identity = snapshot_bytes_and_id(snapshot)
    require(snapshot_bytes == data, "noncanonical snapshot bytes")
    expected = {
        "schema_id": MARKER_SCHEMA,
        "schema_version": 2,
        "status": "transfer_complete",
        "required_consumer_profile": "parent_recording_intake_v1",
        "snapshot_id": identity,
        "snapshot": {
            "path": SNAPSHOT_PATH,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "schema_id": SNAPSHOT_SCHEMA,
            "schema_version": 1,
        },
        "recording_layout": snapshot["recording_layout"],
        "recording_payload_kind": snapshot["recording_payload_kind"],
        "acquisition_session_id": snapshot["acquisition_session_id"],
        "parent_recording_count": len(snapshot["parents"]),
        "semantic_receipt_acceptance": "not_evaluated_by_transfer",
    }
    require(set(marker) == set(expected) | {"delivery"}, "unsupported marker fields")
    for key, value in expected.items():
        require(
            canonical_bytes(marker[key]) == canonical_bytes(value),
            f"marker binding mismatch: {key}",
        )
    delivery = marker["delivery"]
    require(
        type(delivery) is dict
        and set(delivery)
        == {
            "attempt_id",
            "created_utc",
            "source_dir",
            "destination_dir",
            "verification",
            "source_retention",
        },
        "unsupported delivery fields",
    )
    require(
        type(delivery["attempt_id"]) is str
        and re.fullmatch(r"[0-9a-f]{32}", delivery["attempt_id"]) is not None,
        "invalid delivery attempt identity",
    )
    for key in ("source_dir", "destination_dir", "created_utc"):
        identifier(delivery[key], f"delivery {key}")
    require(
        re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}"
            r"(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2})",
            delivery["created_utc"],
        )
        is not None,
        "delivery timestamp must use RFC3339 syntax",
    )
    require(
        dt.datetime.fromisoformat(delivery["created_utc"].upper()).utcoffset()
        == dt.timedelta(0),
        "delivery timestamp must be UTC",
    )
    require(
        delivery["verification"] == "sha256_all_inventory_bytes"
        and delivery["source_retention"] == "retained_pending_consumer_receipt",
        "unsupported verification or retention policy",
    )
    # Close the parse/hash observation window. The storage contract still
    # requires immutable deliveries throughout subsequent execution.
    verify_inventory(root, snapshot, MARKER_NAME, destination=True)
    require(
        marker_path.read_bytes() == marker_bytes and snapshot_path.read_bytes() == data,
        "transfer control bytes changed during validation",
    )
    require(
        canonical_bytes(strict_json(marker_path)) == canonical_bytes(marker),
        "marker changed during validation",
    )
    return VerifiedTransferSnapshot(
        root,
        identity,
        delivery["attempt_id"],
        snapshot["recording_layout"],
        snapshot["recording_payload_kind"],
        snapshot,
    )


def verify_transfer_snapshot(root: Path) -> VerifiedTransferSnapshot:
    """Verify exact bytes and independently reconstruct the closed v2 envelope.

    A v1, unknown, incomplete or malformed delivery never falls back to the
    legacy organizer. This does not evaluate codecs, clocks or scientific
    receipts and does not authorize consumption of source pixels.
    """

    try:
        return _verify_transfer_snapshot(Path(root))
    except TransferSnapshotError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        csv.Error,
        OverflowError,
        RecursionError,
    ) as error:
        raise TransferSnapshotError(f"invalid transfer-v2 delivery: {error}") from error


def _require_optional_proofs(root: Path, output: dict) -> None:
    for sidecar in output["sidecars"]:
        if sidecar["role"] != "summary":
            continue
        summary = strict_json(root / sidecar["artifact"]["path"])
        if "frame_identity_proof" not in summary:
            continue
        proof = summary["frame_identity_proof"]
        status = proof.get("status") if type(proof) is dict else None
        require(
            status not in ("failed", "fail", "error", "rejected"),
            f"frame_identity_proof is failed: {sidecar['artifact']['path']}",
        )
        # The transfer contract defines a negative-proof fixture, not a
        # positive semantic proof schema. Do not invent one from a status word.
        raise TransferSnapshotError(
            "present frame_identity_proof needs a supported semantic proof validator: "
            + sidecar["artifact"]["path"]
        )


def plan_parent_recordings(
    transfer: VerifiedTransferSnapshot,
) -> tuple[ParentRecordingIntakePlan, ...]:
    """Plan the supported dense-full acquisition profile, preserving crop subsets.

    Reopen transport evidence rather than trusting a mutable Python dictionary.
    Consumers must still perform codec, clock, context and import-receipt gates.
    No source data, IDs, timestamps, registry rows or completion state are written.
    """

    current = verify_transfer_snapshot(transfer.root)
    require(
        current.snapshot_id == transfer.snapshot_id,
        "snapshot generation changed before parent planning",
    )
    snapshot = current.snapshot
    plans = []
    for parent in snapshot["parents"]:
        session = parent["parent_key"]["recording_id"]
        camera = parent["parent_key"]["camera_serial"]
        total_frames = 0
        clips = []
        for clip in parent["clips"]:
            outputs = []
            full = None
            for output in clip["outputs"]:
                _require_optional_proofs(current.root, output)
                mapping = output["frame_map"]
                if output["output_kind"] == "full":
                    full = mapping
                outputs.append(
                    IntakeOutput(
                        output["output_kind"],
                        output["role"],
                        output["video"]["path"],
                        output["metadata"]["path"],
                        mapping["frame_count"],
                        mapping["first_recording_frame_id"],
                        mapping["last_recording_frame_id"],
                    )
                )
            require(
                full is not None, "parent intake requires a declared full-frame stream"
            )
            require(
                full["recording_frame_id_gaps"] == 0
                and full["first_recording_frame_id"] == total_frames + 1
                and full["last_recording_frame_id"]
                == total_frames + full["frame_count"],
                "dense full-frame acquisition profile rejects frame-id gaps; IDs are not renumbered",
            )
            total_frames += full["frame_count"]
            require(
                total_frames <= 2**63 - 1,
                "parent frame domain exceeds the supported signed-int64 collection index",
            )
            clips.append(
                IntakeClip(
                    clip["clip_index"],
                    clip["clip_id"],
                    clip["directory"],
                    tuple(outputs),
                )
            )
        plans.append(
            ParentRecordingIntakePlan(
                recording_id_from_session_camera(
                    session_uuid=session, camera_id=camera
                ),
                session,
                camera,
                current.recording_layout,
                current.recording_payload_kind,
                current.snapshot_id,
                total_frames,
                tuple(clips),
            )
        )
    return tuple(plans)
