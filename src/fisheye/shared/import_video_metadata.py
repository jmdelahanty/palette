"""Write metadata-only raw_video attributes for production (no frame import)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import json
import subprocess

import cv2
import imageio.v3 as iio
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


def _probe_video(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()

    try:
        iio_meta = iio.immeta(str(video_path))
    except Exception:
        iio_meta = {}

    format_tags: Dict[str, Any] = {}
    stream_codec: Optional[str] = None
    stream_pix_fmt: Optional[str] = None
    stream_colorimetry: Dict[str, str] = {}
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format_tags=title,comment,encoder:stream=codec_name,codec_tag_string,pix_fmt,color_range,color_space,color_transfer,color_primaries",
                "-of",
                "json",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout:
            payload = json.loads(result.stdout)
            tags = payload.get("format", {}).get("tags", {})
            if isinstance(tags, dict):
                format_tags = tags
            streams = payload.get("streams", [])
            if isinstance(streams, list) and streams:
                stream = streams[0]
                if isinstance(stream, dict):
                    stream_codec = stream.get("codec_name") or stream.get("codec_tag_string")
                    stream_pix_fmt = stream.get("pix_fmt")
                    stream_colorimetry = _stream_colorimetry_attrs(stream)
    except Exception:
        format_tags = {}
        stream_colorimetry = {}

    encoder_fields = parse_encoder_comment(format_tags.get("comment") if format_tags else None)

    codec = None
    pix_fmt = None
    if iio_meta:
        codec = iio_meta.get("codec")
        pix_fmt = iio_meta.get("pix_fmt")
    if not codec and stream_codec:
        codec = stream_codec
    if not codec:
        if fourcc > 0:
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        else:
            codec = "unknown"
    if not pix_fmt and stream_pix_fmt:
        pix_fmt = stream_pix_fmt
    if not pix_fmt:
        pix_fmt = "unknown"

    duration_seconds = n_frames / fps if fps and fps > 0 else 0.0

    return {
        "source_video": str(video_path.name),
        "source_path": str(video_path.absolute()),
        "width": width,
        "height": height,
        "total_frames": n_frames,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "codec": codec,
        "pix_fmt": pix_fmt,
        "imageio_metadata": iio_meta if iio_meta else None,
        "format_tags": format_tags if format_tags else None,
        "encoder_fields": encoder_fields if encoder_fields else None,
        **stream_colorimetry,
    }


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

    if meta.get("imageio_metadata") is not None:
        _set_attr(root.attrs, "imageio_metadata", meta["imageio_metadata"], overwrite=overwrite)
        root_updates.setdefault("imageio_metadata", meta["imageio_metadata"])

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


def probe_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Public wrapper for probing source-video metadata."""
    return _probe_video(video_path)


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
