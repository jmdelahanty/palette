from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fisheye.shared.source_video_metadata import (
    SourceVideoMetadataMissingError,
    resolve_source_video_from_attrs,
)


def _read_attrs(group_dir: Path) -> dict[str, object]:
    zarr_json = group_dir / "zarr.json"
    try:
        payload = json.loads(zarr_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    attrs = payload.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _norm_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _path_recording_dir(path_text: Optional[str]) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    parent_name = path.parent.name.lower()
    if parent_name in {"cams", "raw", "zarr"}:
        return path.parent.parent.resolve()
    return None


@dataclass(frozen=True)
class ZarrRecordingContext:
    recording_dir: Path
    recording_id: Optional[str]
    source_video_path: Optional[Path]


def infer_recording_context(zarr_path: Path) -> ZarrRecordingContext:
    resolved_zarr = zarr_path.expanduser().resolve()
    fallback_recording_dir = (
        resolved_zarr.parent.parent if resolved_zarr.parent.name == "zarr" else resolved_zarr.parent
    )

    root_attrs = _read_attrs(resolved_zarr)
    raw_video_attrs = _read_attrs(resolved_zarr / "raw_video")
    analysis_attrs = _read_attrs(resolved_zarr / "analysis_metadata")

    recording_id = (
        _norm_text(root_attrs.get("recording_id"))
        or _norm_text(analysis_attrs.get("recording_id"))
        or _norm_text(analysis_attrs.get("source_recording_id"))
        or _norm_text(analysis_attrs.get("session_uuid"))
    )

    try:
        resolved_source = resolve_source_video_from_attrs(
            root_attrs,
            raw_video_attrs=raw_video_attrs,
            zarr_path=resolved_zarr,
        )
    except SourceVideoMetadataMissingError:
        source_video_path = None
    else:
        source_video_path = resolved_source.path

    source_video_text = str(source_video_path) if source_video_path is not None else None
    declared_recording_path = _norm_text(root_attrs.get("recording_path"))

    recording_dir = (
        Path(declared_recording_path).expanduser().resolve()
        if declared_recording_path
        else _path_recording_dir(source_video_text)
        or _path_recording_dir(_norm_text(root_attrs.get("source_h5_path")))
        or fallback_recording_dir
    )

    return ZarrRecordingContext(
        recording_dir=recording_dir,
        recording_id=recording_id,
        source_video_path=source_video_path,
    )
