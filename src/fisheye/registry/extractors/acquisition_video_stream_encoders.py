"""Encoder settings of each acquisition video stream, from its recorder summary.

Orange's external recorders write one ``*_external_summary.json`` per stream
(full frame and crop). Its top-level ``tuning`` and ``rate_control`` and its
``video_metadata.encoder`` block are what the encoder was actually run with.
This module projects those settings, and for crop streams the admitted crop
compression (``fisheye.shared.orange_crop_compression.classify_crop_summary``:
lossless, or ``palette.orange_crop_cqp_gop1.v1`` with its QP), into registry
rows keyed like ``acquisition_video_streams``.

The rows are a projection, not an authority: intake already refused crops that
match no admitted rule, and the summary bytes are bound by ``summary_sha256``.

Summary location, relative to the recording directory:

- streams that record ``summary_path`` (legacy per-stream inventory) use it;
- rolling clip collection streams (``source_profile`` ``rolling_clip_collection_v1``)
  use the fixed Orange layout ``cams/acquisition/external_recorder/Cam<id>_external_summary.json``
  or ``cams/acquisition/external_crop_recorder/Cam<id>_crop_external_summary.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.orange_crop_compression import (
    CropCompressionRefusal,
    classify_crop_summary,
)

ROLLING_SOURCE_PROFILE = "rolling_clip_collection_v1"

SUMMARY_READ = "read"
SUMMARY_MISSING = "missing"
SUMMARY_UNREADABLE = "unreadable"
SUMMARY_UNLOCATED = "unlocated"

CROP_ADMITTED = "admitted"
CROP_REFUSED = "refused"

ENCODER_COLUMNS = (
    "dataset_id", "stream_key", "recording_id", "output_kind", "camera_id",
    "summary_relative_path", "summary_status", "summary_sha256", "encoder_block_present",
    "encoder_name", "encoder_codec", "encoder_profile_name", "tuning", "encoder_tuning",
    "preset", "rate_control_strategy", "const_qp", "quality_value", "target_bps",
    "max_bps", "gop_length", "source_format", "output_width", "output_height",
    "crop_compression_status", "crop_compression_rule", "crop_compression_qp",
    "crop_compression_lossless", "crop_compression_refusal", "updated_utc",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> Optional[int]:
    return value if type(value) is int else None


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def summary_relative_path(stream: Mapping[str, Any]) -> Optional[str]:
    """Where a stream's recorder summary lives, relative to its recording directory."""

    recorded = _text(stream.get("summary_path"))
    if recorded:
        return recorded
    contract = _mapping(stream.get("contract"))
    if not contract and isinstance(stream.get("contract_json"), str):
        try:
            contract = _mapping(json.loads(stream["contract_json"]))
        except ValueError:
            contract = {}
    camera_id = _text(stream.get("camera_id")) or _text(contract.get("camera_id"))
    output_kind = _text(stream.get("output_kind"))
    if contract.get("source_profile") != ROLLING_SOURCE_PROFILE or not camera_id:
        return None
    if output_kind == "full":
        return f"cams/acquisition/external_recorder/Cam{camera_id}_external_summary.json"
    if output_kind == "crop":
        return f"cams/acquisition/external_crop_recorder/Cam{camera_id}_crop_external_summary.json"
    return None


def encoder_facts(summary: Mapping[str, Any], *, output_kind: Optional[str]) -> Dict[str, Any]:
    """Registry encoder columns of one recorder summary."""

    encoder = _mapping(_mapping(summary.get("video_metadata")).get("encoder"))
    rate = _mapping(summary.get("rate_control"))
    const_qp = _int(rate.get("const_qp")) if rate.get("const_qp_active") is True else None
    facts: Dict[str, Any] = {
        "encoder_block_present": 1 if encoder else 0,
        "encoder_name": _text(encoder.get("name")),
        "encoder_codec": _text(encoder.get("codec")),
        "encoder_profile_name": _text(encoder.get("profile_name")),
        "tuning": _text(summary.get("tuning")),
        "encoder_tuning": _text(encoder.get("tuning")),
        "preset": _text(encoder.get("preset")),
        "rate_control_strategy": _text(rate.get("resolved_strategy"))
        or _text(encoder.get("rate_control_strategy")),
        "const_qp": const_qp,
        "quality_value": _int(rate.get("quality_value", encoder.get("quality_value"))),
        "target_bps": _int(encoder.get("target_bps")),
        "max_bps": _int(encoder.get("max_bps")),
        "gop_length": _int(encoder.get("resolved_gop_length")),
        "source_format": _text(encoder.get("source_format")),
        "output_width": _int(encoder.get("output_width")),
        "output_height": _int(encoder.get("output_height")),
        "crop_compression_status": None,
        "crop_compression_rule": None,
        "crop_compression_qp": None,
        "crop_compression_lossless": None,
        "crop_compression_refusal": None,
    }
    if output_kind == "crop":
        try:
            compression = classify_crop_summary(summary)
        except CropCompressionRefusal as exc:
            facts["crop_compression_status"] = CROP_REFUSED
            facts["crop_compression_refusal"] = str(exc)
        else:
            facts["crop_compression_status"] = CROP_ADMITTED
            facts["crop_compression_rule"] = compression["rule"]
            facts["crop_compression_qp"] = 0 if compression["lossless"] else compression["qp"]
            facts["crop_compression_lossless"] = 1 if compression["lossless"] else 0
    return facts


def extract_acquisition_video_stream_encoder_rows(
    recording_dir: Path,
    streams: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
) -> List[Dict[str, Any]]:
    """One encoder row per acquisition video stream row of a dataset."""

    updated_utc = utc_now()
    rows: List[Dict[str, Any]] = []
    for stream in streams:
        output_kind = _text(stream.get("output_kind"))
        relative = summary_relative_path(stream)
        row: Dict[str, Any] = {name: None for name in ENCODER_COLUMNS}
        row.update(
            dataset_id=str(dataset_id),
            stream_key=stream.get("stream_key"),
            recording_id=stream.get("recording_id"),
            output_kind=output_kind,
            camera_id=stream.get("camera_id"),
            summary_relative_path=relative,
            updated_utc=updated_utc,
        )
        if relative is None:
            row["summary_status"] = SUMMARY_UNLOCATED
        else:
            path = Path(recording_dir) / relative
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                row["summary_status"] = SUMMARY_MISSING
            except OSError:
                row["summary_status"] = SUMMARY_UNREADABLE
            else:
                row["summary_sha256"] = hashlib.sha256(data).hexdigest()
                try:
                    summary = json.loads(data)
                except ValueError:
                    summary = None
                if isinstance(summary, Mapping):
                    row["summary_status"] = SUMMARY_READ
                    row.update(encoder_facts(summary, output_kind=output_kind))
                else:
                    row["summary_status"] = SUMMARY_UNREADABLE
        rows.append(row)
    return rows


__all__ = [
    "ENCODER_COLUMNS",
    "encoder_facts",
    "extract_acquisition_video_stream_encoder_rows",
    "summary_relative_path",
]
