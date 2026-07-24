"""Write metadata-only raw_video attributes for production (no frame import)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping
import json
import math
import subprocess

import cv2
import zarr

from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PENDING,
    ACQUISITION_AUTHORITY_PUBLISHED,
    ACQUISITION_AUTHORITY_STATUS_ATTR,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    EXTERNAL_ACQUISITION_PENDING_REASON,
    EXTERNAL_ACQUISITION_PUBLISHED_REASON,
    AcquisitionPublicationStatusError,
    build_acquisition_authority_publication_status,
    parse_acquisition_authority_publication_status,
    stamp_acquisition_authority_publication_status,
)
from fisheye.shared.encoder_tags import parse_encoder_comment
from fisheye.shared.import_profile_contract import (
    IMPORT_PROFILE_SCHEMA_ID,
    PROFILE_METADATA_ONLY_ANALYSIS,
)
from fisheye.shared.import_source_fingerprint import optional_source_stat_fingerprint_attrs
from fisheye.shared.pixel_frame_authority import (
    PixelFrameAuthorityError,
    parse_source_video_metadata,
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
)
from fisheye.shared.source_video_metadata import build_source_video_metadata_v2


SOURCE_VIDEO_COLORIMETRY_FIELDS = (
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
)
SOURCE_VIDEO_COLORIMETRY_ATTRS = tuple(f"video_{field}" for field in SOURCE_VIDEO_COLORIMETRY_FIELDS)
VIDEO_METADATA_AUTHORITY_SCHEMA_ID = "palette.video_metadata_authority.v1"
VIDEO_METADATA_AUTHORITY_SCHEMA_VERSION = 1


def _stream_colorimetry_attrs(stream: Dict[str, Any]) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for field in SOURCE_VIDEO_COLORIMETRY_FIELDS:
        value = stream.get(field)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text.lower() in {"", "unknown", "unspecified", "none", "null"}:
            continue
        attrs[f"video_{field}"] = text
    if attrs:
        attrs["source_video_colorimetry_source"] = "ffprobe_stream"
    return attrs


def _colorimetry_payload(meta: Dict[str, Any]) -> Dict[str, Any]:
    payload = {attr: meta.get(attr) for attr in SOURCE_VIDEO_COLORIMETRY_ATTRS if meta.get(attr) not in (None, "")}
    if payload and meta.get("source_video_colorimetry_source"):
        payload["source_video_colorimetry_source"] = meta.get("source_video_colorimetry_source")
    return payload


def probe_video_colorimetry_attrs(video_path: Path, *, ffprobe_bin: str = "ffprobe") -> Dict[str, str]:
    """Return ffprobe stream colorimetry attrs for a source video.

    The returned fields describe the encoded source stream and are deliberately
    named ``video_color_*`` so they do not collide with Palette's downstream
    pixel-contract color semantics.
    """

    try:
        result = subprocess.run(
            [
                str(ffprobe_bin),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=color_range,color_space,color_transfer,color_primaries",
                "-of",
                "json",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout:
            return {}
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        if not isinstance(streams, list) or not streams:
            return {}
        stream = streams[0]
        if not isinstance(stream, dict):
            return {}
        return _stream_colorimetry_attrs(stream)
    except Exception:
        return {}


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _parse_frame_rate(value: Any) -> float | None:
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        num = _positive_float(numerator)
        den = _positive_float(denominator)
        if num is None or den is None:
            return None
        return num / den
    return _positive_float(value)


def _probe_ffprobe(video_path: Path) -> Dict[str, Any]:
    """Read container/stream metadata without decoding the video surface."""

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                (
                    "format=duration:format_tags=title,comment,encoder:"
                    "stream=codec_name,codec_tag_string,pix_fmt,width,height,"
                    "avg_frame_rate,r_frame_rate,nb_frames,duration,"
                    "color_range,color_space,color_transfer,color_primaries"
                ),
                "-of",
                "json",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, PermissionError, OSError):
        return {}
    if result.returncode != 0 or not result.stdout:
        return {}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    streams = payload.get("streams", [])
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        return {}
    stream = streams[0]
    format_payload = payload.get("format", {})
    if not isinstance(format_payload, dict):
        format_payload = {}
    tags = format_payload.get("tags", {})
    if not isinstance(tags, dict):
        tags = {}
    fps = _parse_frame_rate(stream.get("avg_frame_rate"))
    if fps is None:
        fps = _parse_frame_rate(stream.get("r_frame_rate"))
    duration = _positive_float(stream.get("duration"))
    if duration is None:
        duration = _positive_float(format_payload.get("duration"))
    metadata: Dict[str, Any] = {
        "width": _positive_int(stream.get("width")),
        "height": _positive_int(stream.get("height")),
        "total_frames": _positive_int(stream.get("nb_frames")),
        "fps": fps,
        "duration_seconds": duration,
        "codec": stream.get("codec_name") or stream.get("codec_tag_string"),
        "pix_fmt": stream.get("pix_fmt"),
        "format_tags": tags or None,
    }
    metadata.update(_stream_colorimetry_attrs(stream))
    return {key: value for key, value in metadata.items() if value is not None}


def _probe_opencv(video_path: Path) -> Dict[str, Any]:
    """Read the weak fallback/cross-check metadata exposed by OpenCV."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {video_path}")
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    return {
        "width": width,
        "height": height,
        "total_frames": n_frames,
        "fps": fps,
        "fourcc": (
            "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
            if fourcc > 0
            else None
        ),
    }


def _same_numeric(left: int | float, right: int | float) -> bool:
    if type(left) is int and type(right) is int:
        return left == right
    return math.isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-6)


def _resolve_critical_field(
    name: str,
    *,
    producer: Mapping[str, Any],
    ffprobe: Mapping[str, Any],
    opencv: Mapping[str, Any],
    parse: Any,
) -> tuple[int | float, str, list[str]]:
    candidates: list[tuple[str, int | float]] = []
    for source_name, source in (
        ("producer", producer),
        ("ffprobe", ffprobe),
        ("opencv", opencv),
    ):
        value = parse(source.get(name))
        if value is not None:
            candidates.append((source_name, value))
    if not candidates:
        raise ValueError(f"Could not resolve positive video metadata field {name!r}.")
    authoritative_source, authoritative_value = candidates[0]
    disagreements = [
        source_name
        for source_name, value in candidates[1:]
        if not _same_numeric(authoritative_value, value)
    ]
    if authoritative_source == "producer" and "ffprobe" in disagreements:
        raise ValueError(
            f"Producer and ffprobe disagree on {name}: "
            f"producer={authoritative_value!r}, ffprobe={parse(ffprobe.get(name))!r}."
        )
    return authoritative_value, authoritative_source, disagreements


def _probe_video(
    video_path: Path,
    *,
    producer_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Resolve producer, container, then OpenCV metadata with explicit evidence."""

    producer = dict(producer_metadata or {})
    ffprobe = _probe_ffprobe(video_path)
    try:
        opencv = _probe_opencv(video_path)
    except (OSError, RuntimeError, ValueError):
        opencv = {}
    resolved: Dict[str, Any] = {
        "source_video": str(video_path.name),
        "source_path": str(video_path.absolute()),
    }
    field_sources: dict[str, str] = {}
    crosscheck_disagreements: dict[str, list[str]] = {}
    for name, parse in (
        ("width", _positive_int),
        ("height", _positive_int),
        ("total_frames", _positive_int),
        ("fps", _positive_float),
    ):
        value, source_name, disagreements = _resolve_critical_field(
            name,
            producer=producer,
            ffprobe=ffprobe,
            opencv=opencv,
            parse=parse,
        )
        resolved[name] = value
        field_sources[name] = source_name
        if disagreements:
            crosscheck_disagreements[name] = disagreements
    resolved["duration_seconds"] = float(resolved["total_frames"]) / float(resolved["fps"])
    resolved["codec"] = str(
        producer.get("codec") or ffprobe.get("codec") or opencv.get("fourcc") or "unknown"
    )
    resolved["pix_fmt"] = str(ffprobe.get("pix_fmt") or "unknown")
    format_tags = ffprobe.get("format_tags")
    if isinstance(format_tags, Mapping) and format_tags:
        resolved["format_tags"] = dict(format_tags)
        encoder_fields = parse_encoder_comment(format_tags.get("comment"))
        if encoder_fields:
            resolved["encoder_fields"] = encoder_fields
    for attr in SOURCE_VIDEO_COLORIMETRY_ATTRS:
        if ffprobe.get(attr) is not None:
            resolved[attr] = ffprobe[attr]
    if ffprobe.get("source_video_colorimetry_source") is not None:
        resolved["source_video_colorimetry_source"] = ffprobe[
            "source_video_colorimetry_source"
        ]
    resolved["metadata_authority"] = {
        "schema_id": VIDEO_METADATA_AUTHORITY_SCHEMA_ID,
        "schema_version": VIDEO_METADATA_AUTHORITY_SCHEMA_VERSION,
        "resolution_precedence": ["producer", "ffprobe", "opencv_fallback"],
        "downstream_crosscheck": "decoded_observation",
        "producer_source": producer.get("_source"),
        "field_sources": field_sources,
        "opencv_crosscheck_disagreements": crosscheck_disagreements,
    }
    return resolved


def _set_attr(attrs: Any, key: str, value: Any, *, overwrite: bool) -> bool:
    if value is None:
        return False
    if overwrite or key not in attrs or attrs.get(key) in (None, ""):
        attrs[key] = value
        return True
    return False


def _preview_updates(attrs: Any, payload: Dict[str, Any], *, overwrite: bool) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if overwrite or key not in attrs or attrs.get(key) in (None, ""):
            updates[key] = value
    return updates


def _write_metadata(
    root: zarr.Group,
    meta: Dict[str, Any],
    *,
    overwrite: bool,
    import_purpose: str,
    recording_path: str | Path | None = None,
) -> Dict[str, Any]:
    raw = root.require_group("raw_video")
    has_arrays = any(name in raw for name in ("images_full", "images_ds", "images_ds_rgb"))
    imported_frames = None
    if has_arrays:
        if "images_ds" in raw:
            imported_frames = int(raw["images_ds"].shape[0])
        elif "images_full" in raw:
            imported_frames = int(raw["images_full"].shape[0])
        elif "images_ds_rgb" in raw:
            imported_frames = int(raw["images_ds_rgb"].shape[0])

    now = datetime.now(timezone.utc).isoformat()
    raw_updates: Dict[str, Any] = {}
    root_updates: Dict[str, Any] = {}
    source_video_fingerprint_attrs = optional_source_stat_fingerprint_attrs(
        meta.get("source_path"),
        attr_prefix="source_video",
        extra={
            "codec": meta.get("codec"),
            "pix_fmt": meta.get("pix_fmt"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "fps": meta.get("fps"),
            "frame_count": meta.get("total_frames"),
        },
    )
    resolved_recording_path = recording_path or root.attrs.get("recording_path")
    if resolved_recording_path is None and meta.get("source_path"):
        source_path = Path(str(meta["source_path"])).expanduser().resolve()
        if source_path.parent.name == "cams":
            resolved_recording_path = source_path.parent.parent
    source_metadata_input = dict(meta)
    camera_id = root.attrs.get("camera_id")
    if isinstance(camera_id, str) and camera_id.strip():
        source_metadata_input["camera_id"] = camera_id.strip()
    versioned_source_video_metadata = build_source_video_metadata_v2(
        source_metadata_input,
        recording_path=resolved_recording_path,
        fingerprint_attrs=source_video_fingerprint_attrs,
    )

    raw_payload = {
        "import_profile_schema_id": IMPORT_PROFILE_SCHEMA_ID,
        "import_profile": PROFILE_METADATA_ONLY_ANALYSIS,
        "import_method": "metadata_only",
        "import_mode": "metadata_only",
        "import_stage": "metadata_only",
        "import_timestamp": now,
        "import_purpose": import_purpose,
        "fps": meta.get("fps"),
        "total_frames": imported_frames if imported_frames is not None else meta.get("total_frames"),
        "source_video": meta.get("source_video"),
        "source_path": meta.get("source_path"),
        "original_resolution": (meta.get("height"), meta.get("width")),
        "video_codec": meta.get("codec"),
        "video_pix_fmt": meta.get("pix_fmt"),
        "format_title": (meta.get("format_tags") or {}).get("title") if meta.get("format_tags") else None,
        "format_comment": (meta.get("format_tags") or {}).get("comment") if meta.get("format_tags") else None,
        "format_encoder": (meta.get("format_tags") or {}).get("encoder") if meta.get("format_tags") else None,
        "format_tags": meta.get("format_tags"),
        "has_full_resolution": has_arrays and "images_full" in raw,
        "has_downsampled": has_arrays and ("images_ds" in raw or "images_ds_rgb" in raw),
    }
    raw_payload.update(_colorimetry_payload(meta))
    raw_payload.update(source_video_fingerprint_attrs)
    encoder_fields = meta.get("encoder_fields") or {}
    if isinstance(encoder_fields, dict) and encoder_fields:
        raw_payload.update(encoder_fields)
    if meta.get("total_frames") is not None:
        raw_payload.setdefault("original_video_length", meta.get("total_frames"))
        raw_payload.setdefault("source_video_total_frames", meta.get("total_frames"))
    for key, value in raw_payload.items():
        if _set_attr(raw.attrs, key, value, overwrite=overwrite):
            raw_updates[key] = value

    if import_purpose == "training_data":
        zarr_purpose = "training"
    elif import_purpose == "production":
        zarr_purpose = "production"
    else:
        zarr_purpose = "analysis"

    root_payload = {
        "import_profile_schema_id": IMPORT_PROFILE_SCHEMA_ID,
        "import_profile": PROFILE_METADATA_ONLY_ANALYSIS,
        "has_raw_video": has_arrays,
        "zarr_purpose": zarr_purpose,
        "source_video": meta.get("source_video"),
        "source_video_path": meta.get("source_path"),
        "source_path": meta.get("source_path"),
        "recording_path": (
            str(Path(resolved_recording_path).expanduser().resolve())
            if resolved_recording_path is not None
            else None
        ),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "fps": meta.get("fps"),
        "total_frames": imported_frames if imported_frames is not None else meta.get("total_frames"),
        "n_frames": imported_frames if imported_frames is not None else meta.get("total_frames"),
        "duration_seconds": meta.get("duration_seconds"),
        "video_codec": meta.get("codec"),
        "video_pix_fmt": meta.get("pix_fmt"),
        "source_video_metadata": versioned_source_video_metadata,
        "source_video_format_tags": meta.get("format_tags"),
        "source_video_total_frames": meta.get("total_frames"),
    }
    root_payload.update(_colorimetry_payload(meta))
    root_payload.update(source_video_fingerprint_attrs)
    if isinstance(encoder_fields, dict) and encoder_fields:
        root_payload.update(encoder_fields)
    for key, value in root_payload.items():
        if _set_attr(root.attrs, key, value, overwrite=overwrite):
            root_updates[key] = value

    return {"raw_video": raw_updates, "root": root_updates}


def publish_external_video_acquisition_authority(root: zarr.Group) -> dict[str, str]:
    """Publish the canonical acquisition frame for a metadata-only archive.

    This boundary is intentionally external-video-only. A materialized
    ``raw_video/images_full`` archive must be finalized by its pixel writer with
    exact encoded physical-chunk evidence; metadata import cannot upgrade it.
    """

    raw = root.get("raw_video")
    if raw is None:
        raw = root.require_group("raw_video")
    if not isinstance(raw, zarr.Group):
        raise PixelFrameAuthorityError("raw_video must be an exact Zarr group.")
    if "images_full" in raw:
        raise PixelFrameAuthorityError(
            "Metadata-only acquisition publication cannot authorize materialized "
            "raw_video/images_full; the pixel importer must publish exact chunk "
            "evidence."
        )
    recording_id = root.attrs.get("recording_id")
    camera_id = root.attrs.get("camera_id")
    if (
        not isinstance(recording_id, str)
        or not recording_id.strip()
        or recording_id != recording_id.strip()
        or not isinstance(camera_id, str)
        or not camera_id.strip()
        or camera_id != camera_id.strip()
    ):
        raise PixelFrameAuthorityError(
            "External-video acquisition authority requires exact recording_id "
            "and camera_id root attrs from recording context."
        )
    try:
        metadata = parse_source_video_metadata(
            root.attrs.get("source_video_metadata")
        )
    except PixelFrameAuthorityError as exc:
        raise PixelFrameAuthorityError(
            "source_video_metadata must satisfy the exact acquisition source contract."
        ) from exc
    if metadata.get("camera_id") != camera_id:
        raise PixelFrameAuthorityError(
            "source_video_metadata must carry the exact recording camera_id before "
            "acquisition authority is published."
        )
    if "manifests" in raw:
        manifests = raw["manifests"]
        if isinstance(manifests, zarr.Group) and "images_full_materialization" in manifests:
            raise PixelFrameAuthorityError(
                "External-video authority cannot coexist with a materialization manifest."
            )
    authority_path = f"analysis/acquisition_camera_frames/{camera_id}"
    try:
        pending_status = build_acquisition_authority_publication_status(
            status=ACQUISITION_AUTHORITY_PENDING,
            reason_code=EXTERNAL_ACQUISITION_PENDING_REASON,
            authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
            authority_path=authority_path,
        )
        published_status = build_acquisition_authority_publication_status(
            status=ACQUISITION_AUTHORITY_PUBLISHED,
            reason_code=EXTERNAL_ACQUISITION_PUBLISHED_REASON,
            authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
            authority_path=authority_path,
        )
        root_status_value = root.attrs.get(ACQUISITION_AUTHORITY_STATUS_ATTR)
        raw_status_value = raw.attrs.get(ACQUISITION_AUTHORITY_STATUS_ATTR)
        if (root_status_value is None) != (raw_status_value is None):
            raise AcquisitionPublicationStatusError(
                "Root/raw acquisition publication status is incomplete."
            )
        existing_status = None
        if root_status_value is not None:
            root_status = parse_acquisition_authority_publication_status(
                root_status_value
            )
            raw_status = parse_acquisition_authority_publication_status(
                raw_status_value
            )
            if root_status != raw_status or root_status not in {
                pending_status,
                published_status,
            }:
                raise AcquisitionPublicationStatusError(
                    "Existing acquisition publication status conflicts with external mode."
                )
            existing_status = root_status
    except AcquisitionPublicationStatusError as exc:
        raise PixelFrameAuthorityError(
            "External-video acquisition publication status is malformed or conflicting."
        ) from exc

    analysis = root.require_group("analysis")
    authorities = analysis.require_group("acquisition_camera_frames")
    if any(name != camera_id for name in authorities.keys()):
        raise PixelFrameAuthorityError(
            "External-video publication found another acquisition-camera authority."
        )
    if existing_status is None and camera_id in authorities:
        raise PixelFrameAuthorityError(
            "Preexisting external acquisition authority without publication status "
            "is ambiguous and requires explicit repair."
        )
    authority_node = authorities.require_group(camera_id)
    if existing_status != published_status:
        try:
            stamp_acquisition_authority_publication_status(
                root,
                raw,
                status=pending_status.status,
                reason_code=pending_status.reason_code,
                authority_mode=pending_status.authority_mode,
                authority_path=pending_status.authority_path,
            )
        except AcquisitionPublicationStatusError as exc:
            raise PixelFrameAuthorityError(
                "External-video acquisition pending status could not be persisted."
            ) from exc
    ownership = stamp_acquisition_import_ownership(root, authority_node)
    frame = stamp_acquisition_camera_frame(
        root,
        authority_node,
        import_ownership=ownership,
    )
    try:
        stamp_acquisition_authority_publication_status(
            root,
            raw,
            status=published_status.status,
            reason_code=published_status.reason_code,
            authority_mode=published_status.authority_mode,
            authority_path=published_status.authority_path,
        )
    except AcquisitionPublicationStatusError as exc:
        raise PixelFrameAuthorityError(
            "External-video acquisition published status could not be persisted."
        ) from exc
    return {
        "authority_path": authority_path,
        "ownership_record_ref": ownership.record_ref,
        "ownership_record_sha256": ownership.record_sha256,
        "frame_record_ref": frame.record_ref,
        "frame_record_sha256": frame.record_sha256,
    }


def probe_video_metadata(
    video_path: Path,
    *,
    producer_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Resolve source metadata using producer, ffprobe, then OpenCV evidence."""

    return _probe_video(video_path, producer_metadata=producer_metadata)


def probe_ffprobe_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Return finite container/stream metadata from the shared ffprobe path."""

    return _probe_ffprobe(video_path)


def write_video_metadata(
    root: zarr.Group,
    meta: Dict[str, Any],
    *,
    overwrite: bool,
    import_purpose: str,
    recording_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Public wrapper for writing normalized video metadata onto a Zarr root."""
    return _write_metadata(
        root,
        meta,
        overwrite=overwrite,
        import_purpose=import_purpose,
        recording_path=recording_path,
    )
