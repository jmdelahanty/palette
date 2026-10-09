"""Palette's read-only Citrus transfer-v2 boundary and parent intake plans.

Protocol-specific reconstruction is adapted from Citrus's reference validator at
859a7972104a829e1cc603a614cd4cf79e5cda06 (scripts/recording_transfer_snapshot.py):
snapshot v2 with a producer-declared parent recording context (v1 or v2).
The exact external canonicalization, frame-map and finalization grammar is kept
here; Palette identity and JSON reading use their existing owners. This module
never copies payloads, submits jobs, publishes authority, or mints import receipts.
Transport verification and a parent plan are NOT scientific/media admission.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import datetime as dt
from functools import lru_cache
import hashlib
from importlib.resources import files as resource_files
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
SNAPSHOT_VERSION = 2
CONSUMER_PROFILE = "parent_recording_intake_v2"
CONTEXT_SCHEMA = "citrus.parent_recording_context"
CONTEXT_FIELDS = (
    "schema_id", "schema_version", "recording_type", "recording_subtype",
    "behavior_mode", "recording_intent", "data_origin",
)
OBSERVATION_BINDING_DIR = "recording_observation_bindings"
OBSERVATION_FINALIZATION_PATH = f"{OBSERVATION_BINDING_DIR}/finalized_collection.json"
MARKER_SCHEMA = "citrus.transfer_completion_marker.v2"
# Sealer >= 2.0.0 writes v3: v2 plus a provenance-only "sealer" record.
MARKER_SCHEMA_V3 = "citrus.transfer_completion_marker.v3"
# schema_id -> (schema_version, envelope schema definition)
MARKER_SCHEMAS = {MARKER_SCHEMA: (2, "marker"), MARKER_SCHEMA_V3: (3, "marker_v3")}
CONTROL_DIR = "_citrus_transfer"
SNAPSHOT_PATH = f"{CONTROL_DIR}/snapshot.json"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_CSV_LINE = 65536
MARKER_NAME = "_citrus_transfer_complete.json"
# Producer layouts that share parents[].clips[]; both are stored as clip collections.
TRANSFER_PARENT_LAYOUTS = ("rolling_clips", "single_video")


class TransferSnapshotError(ValueError):
    pass


# Citrus's transfer-v2 envelope grammar, byte-identical to
# citrus-recording-transfer 2.0.0 (citrus cfd4774; adds marker v3, v2 unchanged)
# python/citrus_recording_transfer/src/citrus_recording_transfer/schemas/;
# Palette's reliance is in agent-contracts citrus-recording-transfer-consumers.
ENVELOPE_SCHEMA_FILE = "recording_transfer_v2.schema.json"
ENVELOPE_SCHEMA_SHA256 = "c12832b35657f21514392215d838f401be670e76ac22ae6dab28bf304391503c"


@lru_cache(maxsize=None)
def envelope_validator(definition: str | None = None):
    """The whole envelope schema, or (``"marker"``/``"snapshot"``) one of its definitions."""

    from jsonschema import Draft202012Validator, FormatChecker

    data = resource_files("fisheye.shared").joinpath("contracts").joinpath(ENVELOPE_SCHEMA_FILE).read_bytes()
    if hashlib.sha256(data).hexdigest() != ENVELOPE_SCHEMA_SHA256:
        raise TransferSnapshotError("packaged_contract_drift:" + ENVELOPE_SCHEMA_FILE)
    schema = json.loads(data)
    if definition is not None:
        schema = {key: value for key, value in schema.items() if key != "oneOf"}
        schema["$ref"] = f"#/$defs/{definition}"
    return Draft202012Validator(schema, format_checker=FormatChecker())


def require_envelope_schema(document: Any, definition: str) -> None:
    """The producer's own grammar (closed objects, constants, formats)."""

    from jsonschema.exceptions import best_match

    error = best_match(envelope_validator(definition).iter_errors(document))
    if error is not None:
        where = "/".join(str(part) for part in error.absolute_path) or "(root)"
        raise TransferSnapshotError(
            f"{definition} violates the transfer-v2 envelope schema at {where}: {error.message}"
        )


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


# Video and H5 files at least this large take their sha256 from the sealed
# snapshot when the Citrus sealer attests the destination (see
# sealer_attested_inventory). Every other file, including each JSON, CSV and
# JSONL document Palette parses, is always hashed here.
SEALER_ATTESTED_MIN_BYTES = 16 * 1024 * 1024
SEALER_ATTESTED_SUFFIXES = frozenset(
    {".mp4", ".mkv", ".avi", ".h265", ".hevc", ".h5", ".hdf5"}
)
SEALER_VERIFICATION = "sha256_all_inventory_bytes"


def _attested_file_ref(
    root: Path, relative: str, sealed: dict, marker_mtime_ns: int
) -> dict:
    """Stat-only reference that reuses the sealer's digest for a large file.

    The sealer hashed this exact destination file before writing the marker.
    Here it must still be a regular file of the sealed size that was last
    modified no later than the marker was written.
    """

    normalized_path(relative)
    info = (root / relative).lstat()
    require(stat.S_ISREG(info.st_mode), f"not a regular artifact: {relative}")
    require(
        info.st_size == sealed["size_bytes"],
        f"artifact size differs from the sealed snapshot: {relative}",
    )
    require(
        info.st_mtime_ns <= marker_mtime_ns,
        f"artifact modified after the delivery was sealed: {relative}",
    )
    return {"path": relative, "size_bytes": info.st_size, "sha256": sealed["sha256"]}


def _sha256_prefixed(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_contract_sha256(value: Any) -> str:
    return _sha256_prefixed(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def parent_contexts(manifest: dict, cameras: list[str]) -> dict:
    """Producer-declared context per camera parent, copied exactly.

    Context v2 makes ``recording_subtype`` optional: omission means "not
    specified" and stays absent. Media or H5 presence never implies intent.
    """
    contexts = manifest.get("recording_contexts")
    require(
        type(contexts) is dict and set(contexts) == set(cameras),
        "recording_contexts must declare exactly every camera parent",
    )
    fields = set(CONTEXT_FIELDS)
    for context in contexts.values():
        require(
            type(context) is dict
            and set(context) <= fields
            and fields - {"recording_subtype"} <= set(context),
            "unsupported parent recording context fields",
        )
        require(
            context["schema_id"] == CONTEXT_SCHEMA
            and type(context["schema_version"]) is int
            and context["schema_version"] in (1, 2),
            "unsupported parent recording context version",
        )
        require(
            context["schema_version"] != 1 or "recording_subtype" in context,
            "parent context v1 requires recording_subtype",
        )
        for key in ("recording_type", "recording_subtype"):
            if key not in context:
                continue
            label = identifier(context[key], key)
            try:
                encoded = label.encode("utf-8", errors="strict")
            except UnicodeError as error:
                raise TransferSnapshotError(f"invalid {key} UTF-8") from error
            require(len(encoded) <= 1024, f"invalid {key} UTF-8 byte budget")
            require(
                not any(ord(c) < 0x20 or 0x7F <= ord(c) <= 0x9F for c in label),
                f"invalid {key} control character",
            )
            require(
                label.strip() == label
                and not label.startswith("\ufeff")
                and not label.endswith("\ufeff"),
                f"invalid {key} whitespace",
            )
        for key, values in (
            ("behavior_mode", ("free", "embedded", "none")),
            ("recording_intent", ("stimulus_experiment", "recording_only")),
            ("data_origin", ("acquired", "synthetic")),
        ):
            require(context[key] in values, f"unsupported parent context {key}")
    # A bound Citrus observation cannot belong to a recording-only parent.
    observations = manifest.get("observation_contexts", [])
    require(type(observations) is list, "invalid observation context list")
    for observation in observations:
        require(type(observation) is dict, "invalid observation context")
        envelope = observation.get("observation_identity")
        require(
            type(envelope) is dict and type(envelope.get("identity")) is dict,
            "invalid observation identity envelope",
        )
        identity = envelope["identity"]
        require(type(identity.get("camera")) is dict, "missing observation camera identity")
        camera = identity["camera"].get("source_camera_stream_id")
        require(isinstance(camera, str), "invalid observation source camera stream")
        require(camera in contexts, "observation context camera missing from recording_contexts")
        require(
            contexts[camera]["recording_intent"] == "stimulus_experiment",
            "recording_only parent has a Citrus observation binding",
        )
    return contexts


def require_observation_binding_transfer_admission(
    root: Path, manifest: dict, refs: dict[str, dict]
) -> None:
    """Rebuild the producer's transfer gate for bound sessions.

    Transfer safety only, not Palette admission: a bound session must carry
    Orange's finalized collection, each H5 and its post-close receipt with
    matching bytes. Organization re-reads these for Palette's own checks.
    """
    binding_paths = {
        path
        for path in refs
        if PurePosixPath(path).parts[:1] == (OBSERVATION_BINDING_DIR,)
    }
    projected = manifest.get("recording_observation_bindings")
    if projected is None:
        require(
            not binding_paths,
            "observation binding exists without finalized manifest projection",
        )
        return
    require(isinstance(projected, dict), "invalid recording-observation finalization projection")
    require(
        OBSERVATION_FINALIZATION_PATH in refs,
        "bound recording lacks finalized observation collection",
    )
    collection = strict_json(root / OBSERVATION_FINALIZATION_PATH)
    require(
        collection == projected,
        "recording manifest observation projection differs from finalized collection",
    )
    require(
        collection.get("schema_id") == "orange.recording.observation_binding_finalization"
        and type(collection.get("schema_version")) is int
        and collection["schema_version"] == 1
        and collection.get("status") == "finalized"
        and collection.get("binding_status") == "bound",
        "recording observation binding is not finalized and bound",
    )
    require(
        collection.get("recording_id") == manifest.get("session_id"),
        "observation finalization recording identity mismatch",
    )
    contexts = collection.get("observation_contexts")
    require(
        isinstance(contexts, list)
        and bool(contexts)
        and type(collection.get("context_count")) is int
        and collection["context_count"] == len(contexts),
        "invalid finalized observation context set",
    )
    require(
        manifest.get("observation_contexts") == contexts,
        "recording manifest observation contexts differ from finalization",
    )
    seen_contexts: set[str] = set()
    seen_h5: set[str] = set()
    experiment_id = identifier(
        collection.get("citrus_experiment_id"), "Citrus experiment identity"
    )
    for context in contexts:
        require(
            isinstance(context, dict) and context.get("status") == "bound",
            "observation context is not bound",
        )
        context_id = identifier(
            context.get("observation_context_id"), "observation context identity"
        )
        require(context_id not in seen_contexts, "duplicate observation context identity")
        seen_contexts.add(context_id)
        h5 = context.get("citrus_h5")
        require(isinstance(h5, dict), "missing finalized Citrus H5 reference")
        h5_relative = normalized_path(h5.get("relative_path"))
        require(
            h5_relative in refs and Path(h5_relative).suffix.lower() in (".h5", ".hdf5"),
            "finalized Citrus H5 is absent from transfer inventory",
        )
        require(h5_relative not in seen_h5, "duplicate finalized Citrus H5 reference")
        seen_h5.add(h5_relative)
        h5_ref = refs[h5_relative]
        require(
            type(h5.get("size_bytes")) is int
            and h5["size_bytes"] == h5_ref["size_bytes"]
            and h5.get("sha256") == "sha256:" + h5_ref["sha256"],
            "finalized Citrus H5 size or SHA-256 mismatch",
        )
        receipt_ref = context.get("finalized_receipt")
        require(isinstance(receipt_ref, dict), "missing finalized Citrus H5 receipt reference")
        receipt_relative = normalized_path(receipt_ref.get("relative_path"))
        require(
            receipt_relative in refs
            and PurePosixPath(receipt_relative).parts[:2]
            == (OBSERVATION_BINDING_DIR, "receipts"),
            "finalized receipt is absent from transfer inventory",
        )
        receipt_inventory = refs[receipt_relative]
        declared_size = receipt_ref.get("size_bytes", receipt_inventory["size_bytes"])
        require(
            type(declared_size) is int
            and declared_size == receipt_inventory["size_bytes"]
            and receipt_ref.get("sha256") == "sha256:" + receipt_inventory["sha256"],
            "finalized receipt byte binding mismatch",
        )
        receipt = strict_json(root / receipt_relative)
        contract = receipt.get("contract")
        require(
            receipt.get("schema_id") == "citrus.recording_observation_finalized_receipt"
            and type(receipt.get("schema_version")) is int
            and receipt["schema_version"] == 1
            and receipt.get("canonicalization") == "canonical_json_utf8_sort_keys_compact_v1"
            and isinstance(contract, dict),
            "unsupported finalized Citrus H5 receipt",
        )
        contract_sha = _canonical_contract_sha256(contract)
        require(
            receipt.get("contract_sha256") == contract_sha
            and receipt.get("receipt_id") == "obsbindfin_" + contract_sha.removeprefix("sha256:")
            and receipt_ref.get("contract_sha256") == contract_sha
            and receipt_ref.get("receipt_id") == receipt.get("receipt_id"),
            "finalized Citrus H5 receipt envelope mismatch",
        )
        require(
            contract.get("schema_id") == "citrus.recording_observation_finalized_receipt"
            and type(contract.get("schema_version")) is int
            and contract["schema_version"] == 1
            and contract.get("session_status") == "COMPLETE"
            and contract.get("observation_context_id") == context_id
            and contract.get("citrus_experiment_id") == experiment_id
            and contract.get("h5_artifact") == h5,
            "finalized Citrus H5 receipt contract mismatch",
        )
        identity_camera = context["observation_identity"]["identity"]["camera"]
        target = contract.get("target")
        require(
            type(target) is dict
            and target.get("source_camera_stream_id")
            == identity_camera.get("source_camera_stream_id")
            and target.get("camera_id") == identity_camera.get("camera_id"),
            "finalized receipt camera differs from observation context",
        )


def inventory(
    root: Path,
    marker_name: str,
    *,
    destination: bool = False,
    attested: "SealerAttestation | None" = None,
) -> list[dict]:
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
            sealed = attested.items.get(rel) if attested is not None else None
            if (
                sealed is not None
                and sealed["size_bytes"] >= SEALER_ATTESTED_MIN_BYTES
                and Path(name).suffix.lower() in SEALER_ATTESTED_SUFFIXES
            ):
                result.append(
                    _attested_file_ref(root, rel, sealed, attested.marker_mtime_ns)
                )
            else:
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


def build_snapshot(
    root: Path,
    marker_name: str,
    *,
    destination: bool = False,
    attested: "SealerAttestation | None" = None,
) -> dict:
    root = root.resolve(strict=True)
    require(root != Path("/") and root.is_dir(), "invalid recording root")
    normalized_path(marker_name)
    require(
        Path(marker_name).name == marker_name
        and marker_name not in (".", "..", CONTROL_DIR),
        "marker must be a filename",
    )
    # Hash before parsing, then verify these same bytes again before publication.
    items = inventory(root, marker_name, destination=destination, attested=attested)
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
    contexts = parent_contexts(manifest, cameras)
    declared_root = manifest.get("recording_folder", "")
    require_observation_binding_transfer_admission(root, manifest, refs)
    clips = manifest.get("clips")
    require(
        isinstance(clips, list) and clips and (rolling or len(clips) == 1),
        "invalid clip membership for declared mode",
    )
    ids = set()
    parents = {
        c: {
            "parent_key": {"recording_id": session, "camera_serial": c},
            "recording_context": contexts[c],
            "clips": [],
        }
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
        "schema_version": SNAPSHOT_VERSION,
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
    root: Path,
    snapshot: dict,
    marker_name: str,
    *,
    destination: bool,
    attested: "SealerAttestation | None" = None,
) -> None:
    require(
        inventory(root, marker_name, destination=destination, attested=attested)
        == snapshot["inventory"],
        "snapshot inventory mismatch (missing, extra, changed, or replaced artifact)",
    )


@dataclass(frozen=True)
class SealerAttestation:
    """The sealed inventory a v3 marker attests, and when the marker was written."""

    items: dict
    marker_mtime_ns: int


SEALER_PACKAGE = "citrus-recording-transfer"
# 2.0.1 is the first sealer whose destination hash reads storage: each file is
# fsynced, then read with O_DIRECT (on /groups NFS this bypasses the copying
# host's page cache). 2.0.0 read the destination back through that cache.
SEALER_STORAGE_READ_MIN_VERSION = (2, 0, 1)
_SEALER_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def sealer_reads_storage(sealer: Any) -> bool:
    """Whether a marker v3 sealer record names a sealer that hashed storage."""

    if not isinstance(sealer, dict) or sealer.get("package") != SEALER_PACKAGE:
        return False
    match = _SEALER_VERSION.fullmatch(str(sealer.get("version", "")))
    if match is None:
        return False
    return tuple(int(part) for part in match.groups()) >= SEALER_STORAGE_READ_MIN_VERSION


def sealer_attested_inventory(
    marker: dict, snapshot: dict, marker_mtime_ns: int
) -> SealerAttestation | None:
    """Trust the sealer's destination hashes only when it read them from storage.

    Citrus's sealer (marker v3) rebuilds the snapshot from the destination copy,
    hashing every file, and writes the marker only when it matches. From
    citrus-recording-transfer 2.0.1 that hash reads storage rather than the
    copying host's page cache (sealer_reads_storage). A v2 marker, or a sealer
    before 2.0.1, gets no shortcut: Palette hashes every byte itself.
    """

    if marker.get("schema_id") != MARKER_SCHEMA_V3:
        return None
    if not sealer_reads_storage(marker.get("sealer")):
        return None
    require(
        marker["delivery"]["verification"] == SEALER_VERIFICATION,
        "sealer verification policy is not sha256_all_inventory_bytes",
    )
    return SealerAttestation(
        items={item["path"]: item for item in snapshot["inventory"]},
        marker_mtime_ns=marker_mtime_ns,
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
    # Marker v3 provenance ({"package", "version"}); None for a v2 marker.
    sealer: dict | None = None
    # "palette_sha256_all_bytes", or "sealer_attested_large_files" when large
    # video/H5 files may reuse the sealer's digests (marker v3).
    content_verification: str = "palette_sha256_all_bytes"


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
    # Producer-declared citrus.parent_recording_context, exactly as delivered.
    recording_context: dict


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
    marker_schema = marker.get("schema_id") if isinstance(marker, dict) else None
    require(marker_schema in MARKER_SCHEMAS, f"unsupported completion marker {marker_schema!r}")
    marker_version, marker_definition = MARKER_SCHEMAS[marker_schema]
    require_envelope_schema(marker, marker_definition)
    require_envelope_schema(snapshot, "snapshot")
    marker_bytes = marker_path.read_bytes()
    snapshot_bytes = snapshot_path.read_bytes()
    attested = sealer_attested_inventory(
        marker, snapshot, marker_path.lstat().st_mtime_ns
    )
    rebuilt = build_snapshot(root, MARKER_NAME, destination=True, attested=attested)
    require(
        canonical_bytes(snapshot) == canonical_bytes(rebuilt),
        "snapshot schema, membership or source semantics mismatch",
    )
    data, identity = snapshot_bytes_and_id(snapshot)
    require(snapshot_bytes == data, "noncanonical snapshot bytes")
    expected = {
        "schema_id": marker_schema,
        "schema_version": marker_version,
        "status": "transfer_complete",
        "required_consumer_profile": CONSUMER_PROFILE,
        "snapshot_id": identity,
        "snapshot": {
            "path": SNAPSHOT_PATH,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "schema_id": SNAPSHOT_SCHEMA,
            "schema_version": SNAPSHOT_VERSION,
        },
        "recording_layout": snapshot["recording_layout"],
        "recording_payload_kind": snapshot["recording_payload_kind"],
        "acquisition_session_id": snapshot["acquisition_session_id"],
        "parent_recording_count": len(snapshot["parents"]),
        "semantic_receipt_acceptance": "not_evaluated_by_transfer",
    }
    # Field sets, constants and delivery formats are the envelope schema's;
    # what remains is binding the marker to this snapshot and Palette's UTC rule.
    for key, value in expected.items():
        require(
            canonical_bytes(marker[key]) == canonical_bytes(value),
            f"marker binding mismatch: {key}",
        )
    delivery = marker["delivery"]
    for key in ("source_dir", "destination_dir"):
        identifier(delivery[key], f"delivery {key}")  # <= 1024, no C0 controls
        # The sealer's own rule (citrus-recording-transfer >= 1.0.1): no C0 or
        # C1 controls, DEL included (U+0000-U+001F, U+007F-U+009F).
        require(
            not any(0x7F <= ord(c) <= 0x9F for c in delivery[key]),
            f"invalid delivery {key}: control character",
        )
    require(
        dt.datetime.fromisoformat(delivery["created_utc"].upper()).utcoffset()
        == dt.timedelta(0),
        "delivery timestamp must be UTC",
    )
    # Close the parse/hash observation window. The storage contract still
    # requires immutable deliveries throughout subsequent execution.
    verify_inventory(root, snapshot, MARKER_NAME, destination=True, attested=attested)
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
        marker.get("sealer"),
        (
            "palette_sha256_all_bytes"
            if attested is None
            else "sealer_attested_large_files"
        ),
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


# Orange's frame-identity proof grammar, byte-identical to Orange 8b359de
# docs/schemas/orange_external_recorder_frame_identity_proof_v2.schema.json
# (producer: tools/external_recorder_ipc_probe.cpp frame_identity_proof_json).
FRAME_IDENTITY_PROOF_SCHEMA_FILE = "orange_external_recorder_frame_identity_proof_v2.schema.json"
FRAME_IDENTITY_PROOF_SCHEMA_SHA256 = "c9c2544526044bc1a934ab90a0dbe3fca24f34230a27ec280e58b6afc2ffed8d"
FRAME_IDENTITY_PROOF_SCHEMA_ID = "orange.external_recorder.frame_identity_proof"
FRAME_IDENTITY_PROOF_VERSION = 2
# Counters that must all equal the stream's encoded frame count (Orange's
# "verified" condition plus the packet submission counters).
_PROOF_EQUAL_COUNTERS = (
    "submitted_frame_identities",
    "returned_identity_matches",
    "encoded_video_frames",
    "packets_written",
    "metadata_rows",
    "packet_submissions_accepted",
    "packet_write_attempts",
)


@lru_cache(maxsize=None)
def _frame_identity_proof_validator():
    from jsonschema import Draft202012Validator

    data = (
        resource_files("fisheye.shared")
        .joinpath("contracts")
        .joinpath(FRAME_IDENTITY_PROOF_SCHEMA_FILE)
        .read_bytes()
    )
    if hashlib.sha256(data).hexdigest() != FRAME_IDENTITY_PROOF_SCHEMA_SHA256:
        raise TransferSnapshotError("packaged_contract_drift:" + FRAME_IDENTITY_PROOF_SCHEMA_FILE)
    return Draft202012Validator(json.loads(data))


def _stream_proofs(root: Path, output: dict) -> list[tuple[str, dict]]:
    """The output's summary sidecars that carry a frame_identity_proof.

    A failed proof refuses here, before its grammar is checked, so a failed
    proof of any version is reported as failed.
    """

    proofs = []
    for sidecar in output["sidecars"]:
        if sidecar["role"] != "summary":
            continue
        path = sidecar["artifact"]["path"]
        summary = strict_json(root / path)
        if "frame_identity_proof" not in summary:
            continue
        proof = summary["frame_identity_proof"]
        status = proof.get("status") if type(proof) is dict else None
        require(
            status not in ("failed", "fail", "error", "rejected"),
            f"frame_identity_proof is failed: {path}",
        )
        proofs.append((path, proof))
    return proofs


def _require_frame_identity_proof(
    path: str, proof: Any, *, output_kind: str, frame_count: int
) -> None:
    """Accept only Orange's v2 proof, passed, binding exactly this stream's frames.

    The proof covers one recording session and camera stream (all of its
    clips), so ``frame_count`` is the stream's total over the parent's clips.
    Any other schema or version has no validator and is refused.
    """

    from jsonschema.exceptions import best_match

    require(
        type(proof) is dict
        and proof.get("schema_id") == FRAME_IDENTITY_PROOF_SCHEMA_ID
        and proof.get("schema_version") == FRAME_IDENTITY_PROOF_VERSION,
        "present frame_identity_proof needs a supported semantic proof validator: " + path,
    )
    error = best_match(_frame_identity_proof_validator().iter_errors(proof))
    require(error is None, f"frame_identity_proof grammar: {path}: {error and error.message}")
    binding = proof["video_binding"]
    require(
        proof["status"] == "passed" and binding["verified"] is True,
        f"frame_identity_proof is not passed: {path}",
    )
    counts = {name: binding[name] for name in _PROOF_EQUAL_COUNTERS}
    require(
        set(counts.values()) == {frame_count},
        f"frame_identity_proof does not bind the stream's {frame_count} frames: {path}: {counts}",
    )
    require(
        proof["source_frames_dropped"] == 0,
        f"frame_identity_proof reports dropped source frames: {path}",
    )
    require(
        output_kind != "full" or proof["source_frames_skipped_by_policy"] == 0,
        f"full-frame stream skipped source frames by policy: {path}",
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
        # summary path -> (proof, output kinds, frames over the parent's clips)
        stream_proofs: dict[str, tuple[Any, set[str], int]] = {}
        for clip in parent["clips"]:
            outputs = []
            full = None
            for output in clip["outputs"]:
                mapping = output["frame_map"]
                for path, proof in _stream_proofs(current.root, output):
                    _, kinds, frames = stream_proofs.get(path, (proof, set(), 0))
                    stream_proofs[path] = (
                        proof,
                        kinds | {output["output_kind"]},
                        frames + mapping["frame_count"],
                    )
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
        for path, (proof, kinds, frames) in sorted(stream_proofs.items()):
            require(
                len(kinds) == 1,
                f"one frame_identity_proof shared by different output kinds: {path}",
            )
            _require_frame_identity_proof(
                path, proof, output_kind=next(iter(kinds)), frame_count=frames
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
                dict(parent["recording_context"]),
            )
        )
    return tuple(plans)
