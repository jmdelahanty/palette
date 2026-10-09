"""Orange's realtime products: their declaration, model blocks and event logs.

Orange declares, per camera, the realtime detection and pose event logs it
wrote and the models that produced them (``recording_session.json``
``realtime_products``, ``orange.recording_realtime_products`` v1). This module
validates those declarations against Orange's pinned schemas and checks each
event log against what its declaration claims. It reads files and raises; it
writes nothing. Design: docs/design/2026-10-09-realtime-products-intake.

Orange's v2 event logs are the per-frame source of record for what the models
produced. Version-1 logs (before Orange d99e759) have no line schema, so only
their line count is checked.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from importlib.resources import files as resource_files
import json
from pathlib import Path
from typing import Any, Mapping

REALTIME_PRODUCTS_SCHEMA_ID = "orange.recording_realtime_products"
PRODUCTS = ("detections", "pose")
PRODUCT_MODEL_KEY = {"detections": "detect", "pose": "pose"}
LINE_SCHEMA_ID = {"detections": "orange.yolo_event", "pose": "orange.pose_event"}
FRAME_EVENT_KIND = {"detections": "yolo_result", "pose": "pose_result"}
STATUS_BLOCK = {"detections": "yolo", "pose": "pose"}
# Header-type lines a v2 log may carry besides its first session_header line.
EXTRA_HEADER_KINDS = {"detections": {"spatial_mask_policy"}, "pose": set()}
# Orange's mapping of a frame line's status to rows_by_kind
# (orange_recording_realtime_products_v1 description).
RESULT_STATUSES = {"detections": {"detections", "zero_detections"}, "pose": {"poses"}}
NO_RESULT_STATUSES = {"detections": set(), "pose": {"no_result"}}
FAILED_STATUSES = {"failed", "timeout", "error"}

# Byte-identical to Orange 0927ecaf docs/schemas/ (see contracts/README.md).
_SCHEMAS = {
    "realtime_products_v1": (
        "orange_recording_realtime_products_v1.schema.json",
        "cfe02e6e0897b5a799f0b15347ca0a2fba2f76b4cd2877ca2b312e1a871fc227",
    ),
    "detect_model_v1": (
        "orange_recording_detect_model_v1.schema.json",
        "02be96562ed67329c5e1088049fec99bb21a4b10326e62cdeb7cfb2a9871626f",
    ),
    "pose_model_v2": (
        "orange_recording_pose_model_v2.schema.json",
        "f1916316c7e01a4e45157620e827fab27e2310c58a548a9ff31cf8a52047198b",
    ),
    "pose_model_v1": (
        "orange_recording_pose_model_v1.schema.json",
        "778f172e59750cdce987082f221f83e0b348a3edc96bb72f57a5feda5c15a0ee",
    ),
    "yolo_event_v2": (
        "orange_yolo_event_v2.schema.json",
        "393a77ba7056a879fcdd1506a0714a473d6113c3f63dc1a8f19c7d8b1ab650f9",
    ),
    "pose_event_v2": (
        "orange_pose_event_v2.schema.json",
        "835d413c52c4c01893da9b0f6d662e69fc695f8aefbabd0f1393d0d186774fb7",
    ),
}
LINE_SCHEMA = {"detections": "yolo_event_v2", "pose": "pose_event_v2"}
# Model-block schemas, newest first; a block is recorded with the first that
# validates it, or with none (a block that predates Orange's schemas).
MODEL_SCHEMAS = {"detect": ("detect_model_v1",), "pose": ("pose_model_v2", "pose_model_v1")}
MODEL_DIGEST_FIELDS = ("engine_sha256", "weights_sha256", "onnx_sha256")


class RealtimeProductsError(ValueError):
    """A realtime-products declaration, model block or event log contradicts itself."""


@lru_cache(maxsize=None)
def _validator(name: str):
    from jsonschema import Draft202012Validator

    file_name, digest = _SCHEMAS[name]
    data = resource_files("fisheye.shared").joinpath("contracts").joinpath(file_name).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RealtimeProductsError("packaged_contract_drift:" + file_name)
    return Draft202012Validator(json.loads(data))


@lru_cache(maxsize=None)
def _line_validator(name: str):
    """A compiled validator for the per-line hot loop, and its name.

    jsonschema-rs (Rust) gives the same verdict as jsonschema about 150x faster
    on Orange's v2 event lines (measured 2026-10-09); jsonschema remains the
    reference and supplies the error message. Without jsonschema-rs the scan
    falls back to jsonschema: same verdict, slower.
    """

    try:
        import jsonschema_rs
    except ImportError:
        reference = _validator(name)
        return reference.is_valid, "jsonschema"
    _validator(name)  # verifies the pinned bytes
    schema = json.loads(
        resource_files("fisheye.shared").joinpath("contracts").joinpath(_SCHEMAS[name][0]).read_bytes()
    )
    return jsonschema_rs.validator_for(schema).is_valid, "jsonschema-rs"


def schema_error(name: str, document: Any) -> str | None:
    """The best validation error of ``document`` against a pinned schema, or None."""

    from jsonschema.exceptions import best_match

    error = best_match(_validator(name).iter_errors(document))
    return None if error is None else error.message


def realtime_products_declaration(
    session: Mapping[str, Any], *, cameras: list[str]
) -> dict | None:
    """The validated ``realtime_products`` block, or None when not declared.

    It must validate against Orange's schema and declare exactly ``cameras``.
    """

    block = session.get("realtime_products")
    if block is None:
        return None
    error = schema_error("realtime_products_v1", block)
    if error is not None:
        raise RealtimeProductsError(f"realtime_products violates its schema: {error}")
    declared = sorted(block["cameras"])
    if declared != sorted(cameras):
        raise RealtimeProductsError(
            f"realtime_products declares cameras {declared}, the transfer delivers {sorted(cameras)}"
        )
    return block


def model_block_schema(kind: str, block: Mapping[str, Any] | None) -> str | None:
    """The newest pinned schema the start snapshot's model block validates against."""

    if block is None:
        return None
    for name in MODEL_SCHEMAS[kind]:
        if schema_error(name, block) is None:
            return name
    return None


def model_digests(block: Mapping[str, Any] | None) -> dict[str, str]:
    """The non-empty engine/weights/onnx digests of a model block's runtime."""

    runtime = block.get("runtime") if isinstance(block, Mapping) else None
    if not isinstance(runtime, Mapping):
        return {}
    return {
        name: runtime[name]
        for name in MODEL_DIGEST_FIELDS
        if isinstance(runtime.get(name), str) and runtime[name]
    }


def _require_same_digests(label: str, declared: Mapping[str, Any], expected: Mapping[str, str]) -> None:
    for name in MODEL_DIGEST_FIELDS:
        value = declared.get(name)
        if isinstance(value, str) and value and name in expected and value != expected[name]:
            raise RealtimeProductsError(
                f"{label} {name} {value[:12]}… differs from the start snapshot's {expected[name][:12]}…"
            )


@dataclass(frozen=True)
class EventLogSummary:
    """What a scan of one event log found."""

    line_schema_version: int
    frame_rows: int
    header_rows: int
    validation: str
    line_validator: str | None = None
    first_recording_frame_id: int | None = None
    last_recording_frame_id: int | None = None
    rows_by_status: dict[str, int] = field(default_factory=dict)
    rows_by_kind: dict[str, int] = field(default_factory=dict)
    header: dict | None = None


def _kind(product: str, status: str) -> str:
    if status in RESULT_STATUSES[product]:
        return "result"
    if status in NO_RESULT_STATUSES[product]:
        return "no_result"
    if status in FAILED_STATUSES:
        return "failed"
    return "other"


def scan_event_log(path: Path, *, product: str, line_schema_version: int) -> EventLogSummary:
    """Scan one event log, validating every line of a version-2 log.

    A v2 log starts with one ``session_header`` line; detection logs may also
    carry ``spatial_mask_policy`` lines. Frame lines must validate against
    Orange's line schema, number ``event_sequence`` 1..N without a gap, and
    have strictly increasing ``frame.recording_frame_id``. A v1 log is only
    counted: every line is a frame line.
    """

    if line_schema_version == 1:
        with open(path, "rb") as stream:
            rows = sum(1 for line in stream if line.strip())
        return EventLogSummary(1, rows, 0, "file_level_v1")
    if line_schema_version != 2:
        raise RealtimeProductsError(f"{path.name}: unsupported line schema version {line_schema_version}")

    is_valid, line_validator = _line_validator(LINE_SCHEMA[product])
    frame_kind = FRAME_EVENT_KIND[product]
    status_block = STATUS_BLOCK[product]
    header: dict | None = None
    header_rows = frame_rows = 0
    first = last = None
    statuses: Counter[str] = Counter()
    with open(path, "r", encoding="utf-8") as stream:
        for number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise RealtimeProductsError(f"{path.name}:{number}: blank line")
            try:
                line = json.loads(raw)
            except ValueError as exc:
                raise RealtimeProductsError(f"{path.name}:{number}: not JSON: {exc}") from exc
            if not is_valid(line):
                message = schema_error(LINE_SCHEMA[product], line) or "rejected by the compiled validator"
                raise RealtimeProductsError(f"{path.name}:{number}: violates {LINE_SCHEMA[product]}: {message}")
            kind = line.get("event_kind")
            if number == 1:
                if kind != "session_header":
                    raise RealtimeProductsError(f"{path.name}: first line is not a session_header")
                header = line
                header_rows += 1
                continue
            if kind in EXTRA_HEADER_KINDS[product]:
                header_rows += 1
                continue
            if kind != frame_kind:
                raise RealtimeProductsError(f"{path.name}:{number}: unexpected event_kind {kind!r}")
            frame_rows += 1
            if line.get("event_sequence") != frame_rows:
                raise RealtimeProductsError(
                    f"{path.name}:{number}: event_sequence {line.get('event_sequence')} is not {frame_rows}"
                )
            frame_id = line["frame"]["recording_frame_id"]
            if last is not None and frame_id <= last:
                raise RealtimeProductsError(
                    f"{path.name}:{number}: recording_frame_id {frame_id} does not increase"
                )
            first = frame_id if first is None else first
            last = frame_id
            statuses[str(line[status_block]["status"])] += 1
    if header is None:
        raise RealtimeProductsError(f"{path.name}: empty event log")
    kinds = Counter({"result": 0, "no_result": 0, "failed": 0, "other": 0})
    for status, count in statuses.items():
        kinds[_kind(product, status)] += count
    return EventLogSummary(
        2, frame_rows, header_rows, "every_line_v2", line_validator, first, last,
        dict(sorted(statuses.items())), dict(kinds), header,
    )


def check_product(
    product: str,
    declared: Mapping[str, Any],
    summary: EventLogSummary,
    *,
    camera_serial: str,
    recording_id: str,
    snapshot_model_digests: Mapping[str, str],
) -> None:
    """Refuse when an event log disagrees with its declaration or the start snapshot."""

    label = f"Cam{camera_serial} {product}"
    line_schema = declared["line_schema"]
    if line_schema["schema_id"] != LINE_SCHEMA_ID[product]:
        raise RealtimeProductsError(f"{label}: line schema {line_schema['schema_id']!r} is not {LINE_SCHEMA_ID[product]!r}")
    _require_same_digests(f"{label} model_ref", declared["model_ref"], snapshot_model_digests)
    expected = {"row_count": summary.frame_rows, "header_rows": summary.header_rows}
    if summary.line_schema_version == 2:
        expected.update(
            first_recording_frame_id=summary.first_recording_frame_id,
            last_recording_frame_id=summary.last_recording_frame_id,
            rows_by_status=summary.rows_by_status,
            rows_by_kind=summary.rows_by_kind,
        )
    for name, observed in expected.items():
        if declared.get(name) != observed:
            raise RealtimeProductsError(
                f"{label}: declared {name} {declared.get(name)!r} but the log has {observed!r}"
            )
    if summary.header is not None:
        header = summary.header
        if header.get("camera_serial") != camera_serial or header.get("recording_id") != recording_id:
            raise RealtimeProductsError(f"{label}: session_header names another camera or recording")
        _require_same_digests(
            f"{label} session_header", header.get(STATUS_BLOCK[product]) or {}, snapshot_model_digests
        )


__all__ = [
    "EventLogSummary",
    "LINE_SCHEMA_ID",
    "MODEL_DIGEST_FIELDS",
    "PRODUCTS",
    "PRODUCT_MODEL_KEY",
    "REALTIME_PRODUCTS_SCHEMA_ID",
    "RealtimeProductsError",
    "check_product",
    "model_block_schema",
    "model_digests",
    "realtime_products_declaration",
    "scan_event_log",
    "schema_error",
]
