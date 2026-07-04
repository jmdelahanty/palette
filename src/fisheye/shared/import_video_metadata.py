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

from fisheye.shared.encoder_tags import parse_encoder_comment

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
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format_tags=title,comment,encoder:stream=codec_name,codec_tag_string,pix_fmt",
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
    except Exception:
        format_tags = {}

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


def _write_metadata(root: zarr.Group, meta: Dict[str, Any], *, overwrite: bool, import_purpose: str) -> Dict[str, Any]:
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

    raw_payload = {
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
        "has_raw_video": has_arrays,
        "zarr_purpose": zarr_purpose,
        "source_video": meta.get("source_video"),
        "source_video_path": meta.get("source_path"),
        "source_path": meta.get("source_path"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "fps": meta.get("fps"),
        "total_frames": imported_frames if imported_frames is not None else meta.get("total_frames"),
        "n_frames": imported_frames if imported_frames is not None else meta.get("total_frames"),
        "duration_seconds": meta.get("duration_seconds"),
        "video_codec": meta.get("codec"),
        "video_pix_fmt": meta.get("pix_fmt"),
        "source_video_metadata": meta,
        "source_video_format_tags": meta.get("format_tags"),
        "source_video_total_frames": meta.get("total_frames"),
    }
    if isinstance(encoder_fields, dict) and encoder_fields:
        root_payload.update(encoder_fields)
    for key, value in root_payload.items():
        if _set_attr(root.attrs, key, value, overwrite=overwrite):
            root_updates[key] = value

    if meta.get("imageio_metadata") is not None:
        _set_attr(root.attrs, "imageio_metadata", meta["imageio_metadata"], overwrite=overwrite)
        root_updates.setdefault("imageio_metadata", meta["imageio_metadata"])

    return {"raw_video": raw_updates, "root": root_updates}


def probe_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Public wrapper for probing source-video metadata."""
    return _probe_video(video_path)


def write_video_metadata(
    root: zarr.Group,
    meta: Dict[str, Any],
    *,
    overwrite: bool,
    import_purpose: str,
) -> Dict[str, Any]:
    """Public wrapper for writing normalized video metadata onto a Zarr root."""
    return _write_metadata(root, meta, overwrite=overwrite, import_purpose=import_purpose)
