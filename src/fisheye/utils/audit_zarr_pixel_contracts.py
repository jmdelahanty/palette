"""Audit persisted pixel/decode contract metadata in Palette Zarr archives.

The audit is metadata-only: it reads ``zarr.json`` / ``.zattrs`` files directly
instead of opening Zarr stores. This keeps large recording scans fast and avoids
sync-zarr hangs in sandboxed environments.
"""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now as _utc_now
from fisheye.shared.json_safety import write_json_atomic as _write_json
from fisheye.shared.json_safety import write_jsonl_atomic
import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fisheye.shared.roi_pixel_contract import ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME
from fisheye.shared.roi_pixel_contract import crop_run_pixel_contract
from fisheye.shared.roi_pixel_contract import normalize_pixel_contract
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs


RAW_VIDEO_IMAGE_SURFACES = ("images_full", "images_ds", "images_ds_rgb", "images_ds_color", "images")
PIXEL_CONTRACT_ATTRS = (
    "pixel_contract",
    "pixel_contract_name",
    "roi_pixel_contract",
    "roi_pixel_contract_name",
    "crop_pixel_contract",
    "source_roi_pixel_contract",
    "source_roi_pixel_contract_name",
)
DECODE_BACKEND_ATTRS = ("decode_backend", "decode_backend_effective", "source_decode_backend")
SOURCE_VIDEO_PATH_ATTRS = ("source_video_path", "source_path", "video_path", "source_video")
SOURCE_VIDEO_CODEC_ATTRS = ("video_codec", "codec", "codec_name", "encoder_codec")
SOURCE_VIDEO_PIX_FMT_ATTRS = ("video_pix_fmt", "pix_fmt", "pixel_format")
SOURCE_VIDEO_WIDTH_ATTRS = ("video_width", "width", "source_video_width", "encoder_res_width")
SOURCE_VIDEO_HEIGHT_ATTRS = ("video_height", "height", "source_video_height", "encoder_res_height")
SOURCE_VIDEO_FPS_ATTRS = ("fps", "video_fps", "source_video_fps", "encoder_fps", "frame_rate", "avg_frame_rate", "r_frame_rate")
SOURCE_VIDEO_FRAME_COUNT_ATTRS = ("source_video_total_frames", "total_frames", "frame_count", "nb_frames", "n_frames")
SOURCE_VIDEO_FINGERPRINT_ATTRS = (
    "source_video_fingerprint",
    "source_video_sha256",
    "video_sha256",
    "video_fingerprint",
    "content_hash",
)
SOURCE_VIDEO_COLOR_ATTRS = ("color_range", "color_space", "color_matrix", "color_transfer", "color_primaries")
SOURCE_VIDEO_STAT_FINGERPRINT_STRATEGY = "stat_v1"


@dataclass(frozen=True)
class ZarrCandidate:
    path: Path
    source: str
    registry: dict[str, Any] | None = None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if isinstance(parsed, Mapping):
            return {str(key): item for key, item in parsed.items()}
    return {}


def _read_node(path: Path) -> dict[str, Any]:
    zarr_json = path / "zarr.json"
    if _path_exists(zarr_json):
        payload = _read_json(zarr_json) or {}
        attrs = payload.get("attributes")
        if not isinstance(attrs, dict):
            attrs = {}
        return {
            "exists": True,
            "metadata_format": "zarr.json",
            "node_type": payload.get("node_type"),
            "shape": payload.get("shape"),
            "data_type": payload.get("data_type"),
            "attributes": attrs,
        }

    zattrs = path / ".zattrs"
    zarray = path / ".zarray"
    zgroup = path / ".zgroup"
    if _path_exists(zattrs) or _path_exists(zarray) or _path_exists(zgroup):
        attrs = _read_json(zattrs) if _path_exists(zattrs) else {}
        array_payload = _read_json(zarray) if _path_exists(zarray) else None
        return {
            "exists": True,
            "metadata_format": "zarr_v2",
            "node_type": "array" if _path_exists(zarray) else "group",
            "shape": array_payload.get("shape") if isinstance(array_payload, dict) else None,
            "data_type": array_payload.get("dtype") if isinstance(array_payload, dict) else None,
            "attributes": attrs if isinstance(attrs, dict) else {},
        }

    return {
        "exists": False,
        "metadata_format": None,
        "node_type": None,
        "shape": None,
        "data_type": None,
        "attributes": {},
    }


def _has_zarr_metadata(path: Path) -> bool:
    return _path_exists(path / "zarr.json") or _path_exists(path / ".zgroup") or _path_exists(path / ".zarray")


def _iter_child_names(path: Path) -> Iterable[str]:
    if not _path_exists(path) or not _path_is_dir(path):
        return ()
    names: list[str] = []
    try:
        children = list(path.iterdir())
    except OSError:
        return ()
    for child in children:
        if child.name.startswith("."):
            continue
        if not _path_is_dir(child):
            continue
        if _path_exists(child / "zarr.json") or _path_exists(child / ".zgroup") or _path_exists(child / ".zarray"):
            names.append(child.name)
    return sorted(names)


def _contract_from_attrs(attrs: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None, str | None]:
    for attr_name in ("roi_pixel_contract", "pixel_contract", "crop_pixel_contract", "source_roi_pixel_contract"):
        contract = normalize_pixel_contract(attrs.get(attr_name))
        if contract:
            name = contract.get("name")
            return contract, str(name) if name else None, attr_name

    for attr_name in ("roi_pixel_contract_name", "pixel_contract_name", "source_roi_pixel_contract_name"):
        value = attrs.get(attr_name)
        if value:
            return None, str(value), attr_name

    return None, None, None


def _crop_run_name_from_surface_path(surface_path: Any) -> str | None:
    if not isinstance(surface_path, str):
        return None
    parts = surface_path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "crop_runs":
        return parts[1]
    return None


def _first_text(attrs: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = attrs.get(name)
        if value is None or value == "":
            continue
        return str(value)
    return None


def _first_value(mappings: Sequence[Mapping[str, Any]], names: Sequence[str]) -> Any:
    for mapping in mappings:
        for name in names:
            value = mapping.get(name)
            if value is None or value == "":
                continue
            return value
    return None


def _first_text_from(mappings: Sequence[Mapping[str, Any]], names: Sequence[str]) -> str | None:
    value = _first_value(mappings, names)
    if value is None:
        return None
    return str(value)


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, str) and "/" in value:
        left, right = value.split("/", 1)
        try:
            denominator = float(right)
            if denominator == 0:
                return None
            return float(left) / denominator
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _recording_dir_for_zarr(zarr_path: Path) -> Path:
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent
    return zarr_path.parent


def _resolve_recording_relative_path(candidate: ZarrCandidate, raw_value: Any) -> Path | None:
    if raw_value is None or raw_value == "":
        return None
    raw_text = str(raw_value)
    path = Path(raw_text).expanduser()
    if path.is_absolute():
        return path
    recording_dir = _recording_dir_for_zarr(candidate.path)
    candidates = [
        recording_dir / path,
        recording_dir / "cams" / path.name,
        recording_dir / "raw" / path.name,
        recording_dir / "clips" / path,
        candidate.path.parent / path,
    ]
    for option in candidates:
        if _path_exists(option):
            return option
    return recording_dir / path


def _relevant_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        *PIXEL_CONTRACT_ATTRS,
        *DECODE_BACKEND_ATTRS,
        "format",
        "resolution",
        "import_method",
        "import_mode",
        "source_layout",
        "downsample_formats",
        "downsample_method",
        "crop_storage_mode",
        "video_source_type",
        "roi_live_acceleration_effective",
        "roi_image_representation",
        "generated_by",
        "source_crop_run",
        "crop_pixel_migration_version",
        "training_export",
    }
    relevant: dict[str, Any] = {}
    for key in sorted(keys):
        if key not in attrs:
            continue
        value = attrs[key]
        if key == "training_export" and isinstance(value, Mapping):
            relevant[key] = {
                name: _json_safe(value.get(name))
                for name in (
                    "schema_version",
                    "task",
                    "set_id",
                    "set_name",
                    "set_version",
                    "input_format",
                    "include_rgb",
                    "created_at_utc",
                )
                if name in value
            }
            source_paths = value.get("source_zarr_paths")
            if isinstance(source_paths, Sequence) and not isinstance(source_paths, (str, bytes)):
                relevant[key]["source_zarr_count"] = len(source_paths)
            continue
        relevant[key] = _json_safe(value)
    return relevant


def _zarr_kind(path: Path, root_attrs: Mapping[str, Any], raw_attrs: Mapping[str, Any]) -> str:
    name = path.name
    if root_attrs.get("training_export"):
        return "merged_training"
    if name.endswith("_clipped_training.zarr") or raw_attrs.get("source_layout") in {"rolling_clips", "clipped"}:
        return "clipped_training"
    if name.endswith("_training.zarr"):
        return "training"
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_merged.zarr"):
        return "merged_training"
    return "unknown"


def _missing_fields(*, surface_type: str, attrs: Mapping[str, Any], inherited_contract_name: str | None = None) -> list[str]:
    missing: list[str] = []
    has_decode_backend = any(attrs.get(name) for name in DECODE_BACKEND_ATTRS)
    _, contract_name, _ = _contract_from_attrs(attrs)
    if inherited_contract_name and not contract_name:
        contract_name = inherited_contract_name

    if surface_type in {"raw_video", "raw_video_array", "merged_training_root"}:
        if not has_decode_backend:
            missing.append("decode_backend")
        if not contract_name:
            missing.append("pixel_contract")
    elif surface_type in {"crop_run", "crop_roi_images"}:
        if not contract_name:
            missing.append("roi_pixel_contract")
        if contract_name and not attrs.get("roi_pixel_contract_name") and surface_type == "crop_run":
            missing.append("roi_pixel_contract_name")
    return missing


def _safe_inferred_crop_contract(attrs: Mapping[str, Any]) -> dict[str, Any] | None:
    storage_mode = _first_text(attrs, ("crop_storage_mode", "storage_mode"))
    source_type = _first_text(attrs, ("video_source_type", "frame_source_type"))
    acceleration = _first_text(attrs, ("roi_live_acceleration_effective", "roi_live_acceleration", "acceleration"))
    if storage_mode or source_type or acceleration:
        return crop_run_pixel_contract(
            crop_storage_mode=storage_mode or "",
            video_source_type=source_type,
            acceleration=acceleration,
        )
    return None


def _backfill_guidance(
    *,
    zarr_kind: str,
    surface_type: str,
    surface_path: str,
    attrs: Mapping[str, Any],
    root_attrs: Mapping[str, Any],
    raw_attrs: Mapping[str, Any],
    inherited_contract_name: str | None = None,
) -> dict[str, Any]:
    contract, contract_name, contract_attr = _contract_from_attrs(attrs)
    if inherited_contract_name and not contract_name:
        contract_name = inherited_contract_name
    has_decode_backend = any(attrs.get(name) for name in DECODE_BACKEND_ATTRS)
    missing = _missing_fields(surface_type=surface_type, attrs=attrs, inherited_contract_name=inherited_contract_name)

    if not missing:
        return {
            "status": "present",
            "confidence": "high",
            "action": "none",
        }

    if contract_name and "roi_pixel_contract_name" in missing and surface_type == "crop_run":
        return {
            "status": "safe_scalar_name_backfill",
            "confidence": "high",
            "action": f"copy {contract_attr}.name to roi_pixel_contract_name",
            "suggested_roi_pixel_contract_name": contract_name,
        }

    if surface_type == "crop_roi_images" and inherited_contract_name:
        return {
            "status": "inherits_parent_crop_contract",
            "confidence": "high",
            "action": "none; parent crop run defines ROI pixel contract",
            "inherited_roi_pixel_contract_name": inherited_contract_name,
        }
    if surface_type == "crop_roi_images":
        return {
            "status": "parent_crop_contract_missing",
            "confidence": "low",
            "action": "fix or regenerate the parent crop run contract; roi_images should inherit the parent ROI pixel contract",
        }

    if surface_type == "crop_run":
        if str(attrs.get("decode_backend") or "") == "pynvvc_luma" or str(surface_path).endswith("_pynvvc_luma_v1"):
            return {
                "status": "safe_pynvvc_luma_crop_backfill",
                "confidence": "medium",
                "action": "stamp roi_pixel_contract and roi_pixel_contract_name from orange_mono_pynvvc_luma_uint8_v1 if row parity has passed",
                "suggested_roi_pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
            }
        inferred = _safe_inferred_crop_contract(attrs)
        if inferred:
            return {
                "status": "infer_from_crop_run_attrs",
                "confidence": "medium",
                "action": "stamp a historical crop pixel contract inferred from crop_storage_mode/video_source_type/acceleration",
                "suggested_roi_pixel_contract_name": inferred.get("name"),
                "suggested_roi_pixel_contract": inferred,
            }
        return {
            "status": "unknown_crop_contract",
            "confidence": "low",
            "action": "do not backfill as canonical; regenerate/materialize with an explicit crop pixel contract",
        }

    if surface_type in {"raw_video", "raw_video_array"}:
        import_method = str(raw_attrs.get("import_method") or root_attrs.get("import_method") or "")
        import_mode = str(raw_attrs.get("import_mode") or root_attrs.get("import_mode") or "")
        if zarr_kind == "clipped_training" and "create_clipped_training_zarr" in import_method:
            return {
                "status": "infer_legacy_opencv_gray_from_writer",
                "confidence": "medium",
                "action": "historical backfill can label as OpenCV VideoCapture BGR2GRAY; preferred strict fix is regenerate/copy from explicit pynvvc_luma source pixels",
                "suggested_decode_backend": "opencv",
                "suggested_pixel_contract_name": "opencv_bgr2gray_uint8",
            }
        if "metadata_only" in import_method or "metadata_only" in import_mode or "fisheye.capture.import_video" in import_method:
            return {
                "status": "legacy_import_gray_under_labeled",
                "confidence": "low",
                "action": "do not infer pynvvc; either stamp a legacy import/decord gray contract after writer audit or regenerate if strict model-input parity is required",
                "suggested_pixel_contract_name": "legacy_import_gray_uint8",
            }
        if zarr_kind == "merged_training":
            return {
                "status": "requires_source_contract_audit",
                "confidence": "low",
                "action": "merged raw frames are copied from source zarrs; re-export after source raw_video pixel contracts are present or mark as mixed/legacy",
            }
        return {
            "status": "unknown_raw_video_contract",
            "confidence": "low",
            "action": "do not backfill as canonical; inspect writer provenance or regenerate",
        }

    if surface_type == "merged_training_root":
        return {
            "status": "missing_export_contract",
            "confidence": "medium",
            "action": "future exporter should aggregate source pixel contracts and refuse incompatible mixes; existing artifact needs source audit before backfill",
        }

    return {
        "status": "not_applicable",
        "confidence": "medium",
        "action": "none",
    }


def _surface_row(
    *,
    candidate: ZarrCandidate,
    zarr_kind: str,
    surface_type: str,
    surface_path: str,
    node: Mapping[str, Any],
    root_attrs: Mapping[str, Any],
    raw_attrs: Mapping[str, Any],
    inherited_contract_name: str | None = None,
) -> dict[str, Any]:
    attrs = node.get("attributes")
    if not isinstance(attrs, Mapping):
        attrs = {}
    contract, contract_name, contract_attr = _contract_from_attrs(attrs)
    if inherited_contract_name and not contract_name:
        contract_name = inherited_contract_name
    decode_backend = _first_text(attrs, DECODE_BACKEND_ATTRS)
    missing = _missing_fields(surface_type=surface_type, attrs=attrs, inherited_contract_name=inherited_contract_name)
    row = {
        "record_type": "pixel_contract_surface",
        "audit_utc": _utc_now(),
        "zarr_path": str(candidate.path),
        "zarr_name": candidate.path.name,
        "zarr_kind": zarr_kind,
        "discovery_source": candidate.source,
        "registry": candidate.registry,
        "surface_type": surface_type,
        "surface_path": surface_path,
        "exists": bool(node.get("exists")),
        "metadata_format": node.get("metadata_format"),
        "node_type": node.get("node_type"),
        "shape": _json_safe(node.get("shape")),
        "data_type": _json_safe(node.get("data_type")),
        "decode_backend": decode_backend,
        "has_decode_backend": bool(decode_backend),
        "pixel_contract_attr": contract_attr,
        "pixel_contract_name": contract_name,
        "pixel_contract": _json_safe(contract),
        "has_pixel_contract": bool(contract_name),
        "inherited_pixel_contract_name": inherited_contract_name,
        "missing_fields": missing,
        "relevant_attrs": _relevant_attrs(attrs),
    }
    row["backfill"] = _backfill_guidance(
        zarr_kind=zarr_kind,
        surface_type=surface_type,
        surface_path=surface_path,
        attrs=attrs,
        root_attrs=root_attrs,
        raw_attrs=raw_attrs,
        inherited_contract_name=inherited_contract_name,
    )
    return row


def _source_backfill_guidance(*, missing: Sequence[str], source_scope: str, metadata_sources: Sequence[str]) -> dict[str, Any]:
    missing_set = set(missing)
    if not missing_set:
        return {
            "status": "present",
            "confidence": "high",
            "action": "none",
        }
    if "source_video_path" in missing_set:
        return {
            "status": "missing_source_video_path",
            "confidence": "low",
            "action": "cannot backfill encoded source-video metadata without a source path or sidecar row",
        }
    if source_scope == "clipped_sidecar":
        return {
            "status": "clipped_sidecar_partial",
            "confidence": "medium",
            "action": "promote sidecar metadata into a structured zarr/registry source-video manifest and fill missing fields with ffprobe when needed",
        }
    if missing_set <= {"colorimetry", "fingerprint"}:
        return {
            "status": "missing_colorimetry_or_fingerprint",
            "confidence": "medium",
            "action": "backfill colorimetry with ffprobe and add a stable source-video fingerprint before enforcing strict regeneration parity",
        }
    if any(source in {"raw_video_attrs", "root_attrs", "registry"} for source in metadata_sources):
        return {
            "status": "registry_or_zarr_partial",
            "confidence": "medium",
            "action": "existing zarr/registry metadata is partial; fill codec/pix_fmt/dimensions/fps/frame_count/colorimetry/fingerprint with ffprobe or sidecars",
        }
    return {
        "status": "needs_ffprobe_backfill",
        "confidence": "low",
        "action": "probe the source video and stamp encoded source-video metadata before strict export/regeneration",
    }


def _source_video_row(
    *,
    candidate: ZarrCandidate,
    zarr_kind: str,
    source_scope: str,
    source_id: str,
    metadata: Sequence[Mapping[str, Any]],
    metadata_sources: Sequence[str],
    source_path_value: Any | None = None,
) -> dict[str, Any]:
    source_video_path = source_path_value
    if source_video_path is None:
        source_video_path = _first_value(metadata, SOURCE_VIDEO_PATH_ATTRS)
    resolved_path = _resolve_recording_relative_path(candidate, source_video_path)
    encoder_params = _as_mapping(_first_value(metadata, ("encoder_params", "encoder_params_json")))
    width = _coerce_int(_first_value(metadata, SOURCE_VIDEO_WIDTH_ATTRS))
    height = _coerce_int(_first_value(metadata, SOURCE_VIDEO_HEIGHT_ATTRS))
    fps = _coerce_float(_first_value(metadata, SOURCE_VIDEO_FPS_ATTRS))
    frame_count = _coerce_int(_first_value(metadata, SOURCE_VIDEO_FRAME_COUNT_ATTRS))
    colorimetry = {
        name: _json_safe(_first_value(metadata, (name,)))
        for name in SOURCE_VIDEO_COLOR_ATTRS
        if _first_value(metadata, (name,)) is not None
    }
    if "encoder_color" not in colorimetry:
        encoder_color = _first_value(metadata, ("encoder_color",))
        if encoder_color is not None:
            colorimetry["encoder_color"] = _json_safe(encoder_color)
    fingerprint = _first_text_from(metadata, SOURCE_VIDEO_FINGERPRINT_ATTRS)

    missing: list[str] = []
    if source_video_path is None:
        missing.append("source_video_path")
    if not _first_text_from(metadata, SOURCE_VIDEO_CODEC_ATTRS):
        missing.append("codec")
    if not _first_text_from(metadata, SOURCE_VIDEO_PIX_FMT_ATTRS):
        missing.append("pix_fmt")
    if width is None:
        missing.append("width")
    if height is None:
        missing.append("height")
    if fps is None:
        missing.append("fps")
    if frame_count is None:
        missing.append("frame_count")
    if not colorimetry:
        missing.append("colorimetry")
    if not fingerprint:
        missing.append("fingerprint")

    row = {
        "record_type": "source_video_metadata",
        "audit_utc": _utc_now(),
        "zarr_path": str(candidate.path),
        "zarr_name": candidate.path.name,
        "zarr_kind": zarr_kind,
        "discovery_source": candidate.source,
        "registry": candidate.registry,
        "source_scope": source_scope,
        "source_id": source_id,
        "source_video_path": str(source_video_path) if source_video_path is not None else None,
        "source_video_resolved_path": str(resolved_path) if resolved_path is not None else None,
        "source_video_exists": bool(_path_exists(resolved_path)) if resolved_path is not None else False,
        "metadata_sources": list(metadata_sources),
        "codec": _first_text_from(metadata, SOURCE_VIDEO_CODEC_ATTRS),
        "pix_fmt": _first_text_from(metadata, SOURCE_VIDEO_PIX_FMT_ATTRS),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "colorimetry": colorimetry,
        "fingerprint": fingerprint,
        "encoder_settings": _json_safe(
            _first_value(metadata, ("format_comment", "format_encoder", "encoder_name", "encoder_params_json"))
        ),
        "encoder_params": _json_safe(encoder_params) if encoder_params else None,
        "missing_fields": missing,
    }
    row["backfill"] = _source_backfill_guidance(
        missing=missing,
        source_scope=source_scope,
        metadata_sources=metadata_sources,
    )
    return row


def _iter_clip_index_entries(recording_dir: Path) -> Iterable[tuple[str, Mapping[str, Any], list[str]]]:
    candidate_paths = [
        recording_dir / "recording_clip_index.json",
        recording_dir / "clips" / "recording_clip_index.json",
        recording_dir / "clip_index.json",
    ]
    for clip_index_path in candidate_paths:
        payload = _read_json(clip_index_path)
        if not payload:
            continue
        raw_entries: Any = None
        for key in ("clips", "clip_index", "entries", "clip_records"):
            value = payload.get(key)
            if isinstance(value, (list, tuple, dict)):
                raw_entries = value
                break
        if raw_entries is None and isinstance(payload.get("clip_manifest"), (list, tuple, dict)):
            raw_entries = payload.get("clip_manifest")
        if isinstance(raw_entries, Mapping):
            iterable = raw_entries.values()
        elif isinstance(raw_entries, (list, tuple)):
            iterable = raw_entries
        else:
            continue
        for index, entry in enumerate(iterable):
            if not isinstance(entry, Mapping):
                continue
            normalized = {str(key): value for key, value in entry.items()}
            clip_id = str(normalized.get("clip_id") or normalized.get("id") or f"clip_{index:06d}")
            sources = [str(clip_index_path)]
            keyframe_path = _resolve_recording_relative_path(
                ZarrCandidate(path=recording_dir / "zarr" / "placeholder.zarr", source="sidecar"),
                normalized.get("source_keyframe_path") or normalized.get("keyframe_path"),
            )
            keyframe_payload = _read_json(keyframe_path) if keyframe_path is not None and _path_exists(keyframe_path) else None
            if keyframe_payload:
                merged = {**keyframe_payload, **normalized}
                sources.append(str(keyframe_path))
                yield clip_id, merged, sources
            else:
                yield clip_id, normalized, sources
        return


def _source_video_rows(
    *,
    candidate: ZarrCandidate,
    zarr_kind: str,
    root_attrs: Mapping[str, Any],
    raw_attrs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    registry = candidate.registry or {}
    registry_metadata = registry.get("provenance") if isinstance(registry.get("provenance"), Mapping) else registry
    metadata_sources: list[str] = []
    metadata: list[Mapping[str, Any]] = []
    if raw_attrs:
        metadata.append(raw_attrs)
        metadata_sources.append("raw_video_attrs")
    if root_attrs:
        metadata.append(root_attrs)
        metadata_sources.append("root_attrs")
    if isinstance(registry_metadata, Mapping) and registry_metadata:
        metadata.append(registry_metadata)
        metadata_sources.append("registry")

    recording_dir = _recording_dir_for_zarr(candidate.path)
    clip_entries = list(_iter_clip_index_entries(recording_dir))
    if clip_entries:
        for clip_id, clip_metadata, clip_sources in clip_entries:
            rows.append(
                _source_video_row(
                    candidate=candidate,
                    zarr_kind=zarr_kind,
                    source_scope="clipped_sidecar",
                    source_id=clip_id,
                    metadata=[clip_metadata, *metadata],
                    metadata_sources=[*clip_sources, *metadata_sources],
                    source_path_value=clip_metadata.get("video_path") or clip_metadata.get("source_video_path"),
                )
            )
        return rows

    if metadata:
        rows.append(
            _source_video_row(
                candidate=candidate,
                zarr_kind=zarr_kind,
                source_scope="single_video",
                source_id="source_video",
                metadata=metadata,
                metadata_sources=metadata_sources,
            )
        )
    return rows


def audit_zarr_path(candidate: ZarrCandidate, *, include_source_video_metadata: bool = False) -> list[dict[str, Any]]:
    root_node = _read_node(candidate.path)
    if not root_node["exists"]:
        return [
            {
                "record_type": "pixel_contract_zarr_error",
                "audit_utc": _utc_now(),
                "zarr_path": str(candidate.path),
                "discovery_source": candidate.source,
                "registry": candidate.registry,
                "error": "missing_zarr_metadata",
            }
        ]

    root_attrs = root_node["attributes"]
    raw_node = _read_node(candidate.path / "raw_video")
    raw_attrs = raw_node["attributes"]
    zarr_kind = _zarr_kind(candidate.path, root_attrs, raw_attrs)

    rows: list[dict[str, Any]] = []
    if include_source_video_metadata:
        rows.extend(
            _source_video_rows(
                candidate=candidate,
                zarr_kind=zarr_kind,
                root_attrs=root_attrs,
                raw_attrs=raw_attrs,
            )
        )

    if root_attrs.get("training_export"):
        rows.append(
            _surface_row(
                candidate=candidate,
                zarr_kind=zarr_kind,
                surface_type="merged_training_root",
                surface_path=".",
                node=root_node,
                root_attrs=root_attrs,
                raw_attrs=raw_attrs,
            )
        )

    if raw_node["exists"]:
        rows.append(
            _surface_row(
                candidate=candidate,
                zarr_kind=zarr_kind,
                surface_type="raw_video",
                surface_path="raw_video",
                node=raw_node,
                root_attrs=root_attrs,
                raw_attrs=raw_attrs,
            )
        )
        for array_name in RAW_VIDEO_IMAGE_SURFACES:
            array_node = _read_node(candidate.path / "raw_video" / array_name)
            if array_node["exists"]:
                rows.append(
                    _surface_row(
                        candidate=candidate,
                        zarr_kind=zarr_kind,
                        surface_type="raw_video_array",
                        surface_path=f"raw_video/{array_name}",
                        node=array_node,
                        root_attrs=root_attrs,
                        raw_attrs=raw_attrs,
                    )
                )

    crop_parent = candidate.path / "crop_runs"
    if _path_exists(crop_parent):
        crop_parent_node = _read_node(crop_parent)
        crop_parent_attrs = crop_parent_node["attributes"]
        latest_crop_run = None
        current_crop_selector = None
        for selector_attr in ("latest", "latest_complete", "latest_any"):
            latest_crop_run = _first_text(crop_parent_attrs, (selector_attr,))
            if latest_crop_run:
                current_crop_selector = f"crop_runs.attrs.{selector_attr}"
                break
        for run_name in _iter_child_names(crop_parent):
            run_path = crop_parent / run_name
            run_node = _read_node(run_path)
            run_attrs = run_node["attributes"]
            _, parent_contract_name, _ = _contract_from_attrs(run_attrs)
            crop_row = _surface_row(
                candidate=candidate,
                zarr_kind=zarr_kind,
                surface_type="crop_run",
                surface_path=f"crop_runs/{run_name}",
                node=run_node,
                root_attrs=root_attrs,
                raw_attrs=raw_attrs,
            )
            crop_row["crop_run_name"] = run_name
            crop_row["latest_crop_run_name"] = latest_crop_run
            crop_row["is_current_crop_run"] = bool(latest_crop_run and latest_crop_run == run_name)
            crop_row["current_crop_selector"] = current_crop_selector
            rows.append(crop_row)
            roi_node = _read_node(run_path / "roi_images")
            if roi_node["exists"]:
                roi_row = _surface_row(
                    candidate=candidate,
                    zarr_kind=zarr_kind,
                    surface_type="crop_roi_images",
                    surface_path=f"crop_runs/{run_name}/roi_images",
                    node=roi_node,
                    root_attrs=root_attrs,
                    raw_attrs=raw_attrs,
                    inherited_contract_name=parent_contract_name,
                )
                roi_row["crop_run_name"] = run_name
                roi_row["latest_crop_run_name"] = latest_crop_run
                roi_row["is_current_crop_run"] = bool(latest_crop_run and latest_crop_run == run_name)
                roi_row["current_crop_selector"] = current_crop_selector
                rows.append(roi_row)

    return rows


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[1]) for row in rows}


def _discover_registry_candidates(
    registry_path: Path,
    *,
    zarr_use: str,
    path_contains: str | None,
    limit: int | None,
) -> list[ZarrCandidate]:
    conn = sqlite3.connect(str(registry_path))
    conn.row_factory = sqlite3.Row
    try:
        columns = _table_columns(conn, "datasets")
        if not columns:
            return []
        provenance_columns = _table_columns(conn, "provenance")
        wanted = [
            "dataset_id",
            "zarr_path",
            "zarr_use",
            "zarr_origin",
            "artifact_kind",
            "status",
            "recording_id",
            "source_layout",
        ]
        selected = [f"d.{name} AS {name}" for name in wanted if name in columns]
        if "zarr_path" not in columns:
            return []
        provenance_wanted = [
            "fps",
            "video_codec",
            "video_pix_fmt",
            "source_video",
            "format_title",
            "format_comment",
            "format_encoder",
            "encoder_name",
            "encoder_codec",
            "encoder_preset",
            "encoder_tuning",
            "encoder_rc",
            "encoder_bpp",
            "encoder_target_bps",
            "encoder_res",
            "encoder_res_width",
            "encoder_res_height",
            "encoder_fps",
            "encoder_color",
            "encoder_params_json",
        ]
        selected.extend(f"p.{name} AS provenance_{name}" for name in provenance_wanted if name in provenance_columns)
        sql = [f"SELECT {', '.join(selected)} FROM datasets d"]
        if provenance_columns and "dataset_id" in provenance_columns:
            sql.append("LEFT JOIN provenance p ON p.dataset_id = d.dataset_id")
        clauses = []
        params: list[Any] = []
        if "status" in columns:
            clauses.append("(d.status IS NULL OR d.status != 'missing')")
        if zarr_use != "all" and "zarr_use" in columns:
            clauses.append("d.zarr_use = ?")
            params.append(zarr_use)
        if path_contains:
            clauses.append("d.zarr_path LIKE ?")
            params.append(f"%{path_contains}%")
        if clauses:
            sql.append("WHERE " + " AND ".join(clauses))
        sql.append("ORDER BY d.zarr_path")
        if limit is not None:
            sql.append("LIMIT ?")
            params.append(int(limit))
        rows = conn.execute(" ".join(sql), params).fetchall()
    finally:
        conn.close()

    candidates: list[ZarrCandidate] = []
    for row in rows:
        raw_path = row["zarr_path"]
        if not raw_path:
            continue
        provenance = {
            key.removeprefix("provenance_"): _json_safe(row[key])
            for key in row.keys()
            if key.startswith("provenance_") and row[key] is not None
        }
        registry = {
            key: _json_safe(row[key])
            for key in row.keys()
            if key != "zarr_path" and not key.startswith("provenance_")
        }
        if provenance:
            registry["provenance"] = provenance
        candidates.append(
            ZarrCandidate(
                path=Path(str(raw_path)).expanduser().resolve(),
                source="registry",
                registry=registry,
            )
        )
    return candidates


def _discover_candidates(args: argparse.Namespace) -> list[ZarrCandidate]:
    candidates: list[ZarrCandidate] = []
    if args.registry is not None:
        candidates.extend(
            _discover_registry_candidates(
                args.registry.expanduser().resolve(),
                zarr_use=str(args.zarr_use),
                path_contains=args.path_contains,
                limit=args.limit,
            )
        )

    paths = [path.expanduser() for path in args.paths]
    if paths:
        for zarr_path in iter_filesystem_zarrs(paths, recursive=bool(args.recursive)):
            if args.path_contains and args.path_contains not in str(zarr_path):
                continue
            candidates.append(
                ZarrCandidate(
                    path=zarr_path.expanduser().resolve(),
                    source="filesystem",
                    registry=None,
                )
            )

    deduped: dict[str, ZarrCandidate] = {}
    for candidate in candidates:
        if args.skip_missing_zarrs and not _has_zarr_metadata(candidate.path):
            continue
        key = str(candidate.path)
        existing = deduped.get(key)
        if existing is None or (existing.registry is None and candidate.registry is not None):
            deduped[key] = candidate
    ordered = sorted(deduped.values(), key=lambda item: str(item.path))
    if args.limit is not None and args.registry is None:
        ordered = ordered[: int(args.limit)]
    return ordered


def _summary(rows: Sequence[Mapping[str, Any]], *, candidates: Sequence[ZarrCandidate]) -> dict[str, Any]:
    by_surface = Counter(str(row.get("surface_type")) for row in rows if row.get("record_type") == "pixel_contract_surface")
    by_missing = Counter()
    by_backfill = Counter()
    by_kind = Counter()
    by_source_video_scope = Counter()
    by_source_video_missing = Counter()
    by_source_video_backfill = Counter()
    by_safe_scalar_action = Counter()
    by_inferred_legacy_crop_action = Counter()
    by_source_video_fingerprint_action = Counter()
    by_source_video_colorimetry_action = Counter()
    for row in rows:
        if row.get("record_type") == "inferred_legacy_crop_contract_action":
            by_inferred_legacy_crop_action[str(row.get("status"))] += 1
            continue
        if row.get("record_type") == "source_video_ffprobe_colorimetry_action":
            by_source_video_colorimetry_action[str(row.get("status"))] += 1
            continue
        if row.get("record_type") == "source_video_stat_fingerprint_action":
            by_source_video_fingerprint_action[str(row.get("status"))] += 1
            continue
        if row.get("record_type") == "safe_scalar_name_backfill_action":
            by_safe_scalar_action[str(row.get("status"))] += 1
            continue
        if row.get("record_type") == "source_video_metadata":
            by_source_video_scope[str(row.get("source_scope"))] += 1
            for field in row.get("missing_fields") or ():
                by_source_video_missing[str(field)] += 1
            backfill = row.get("backfill") if isinstance(row.get("backfill"), Mapping) else {}
            by_source_video_backfill[str(backfill.get("status"))] += 1
            continue
        if row.get("record_type") != "pixel_contract_surface":
            continue
        by_kind[str(row.get("zarr_kind"))] += 1
        for field in row.get("missing_fields") or ():
            by_missing[str(field)] += 1
        backfill = row.get("backfill") if isinstance(row.get("backfill"), Mapping) else {}
        by_backfill[str(backfill.get("status"))] += 1
    return {
        "record_type": "pixel_contract_audit_summary",
        "audit_utc": _utc_now(),
        "zarr_count": len(candidates),
        "row_count": len(rows),
        "surface_counts": dict(sorted(by_surface.items())),
        "zarr_kind_surface_counts": dict(sorted(by_kind.items())),
        "missing_field_counts": dict(sorted(by_missing.items())),
        "backfill_status_counts": dict(sorted(by_backfill.items())),
        "source_video_scope_counts": dict(sorted(by_source_video_scope.items())),
        "source_video_missing_field_counts": dict(sorted(by_source_video_missing.items())),
        "source_video_backfill_status_counts": dict(sorted(by_source_video_backfill.items())),
        "safe_scalar_name_backfill_action_counts": dict(sorted(by_safe_scalar_action.items())),
        "inferred_legacy_crop_contract_action_counts": dict(sorted(by_inferred_legacy_crop_action.items())),
        "source_video_stat_fingerprint_action_counts": dict(sorted(by_source_video_fingerprint_action.items())),
        "source_video_ffprobe_colorimetry_action_counts": dict(sorted(by_source_video_colorimetry_action.items())),
    }


def _crop_contract_report(rows: Sequence[Mapping[str, Any]], *, candidates: Sequence[ZarrCandidate]) -> dict[str, Any]:
    crop_rows = [
        row
        for row in rows
        if row.get("record_type") == "pixel_contract_surface" and row.get("surface_type") == "crop_run"
    ]
    current_rows = [row for row in crop_rows if row.get("is_current_crop_run")]
    zarrs_with_crop_runs = {str(row.get("zarr_path")) for row in crop_rows if row.get("zarr_path")}
    zarrs_with_current_crop = {str(row.get("zarr_path")) for row in current_rows if row.get("zarr_path")}
    zarrs_missing_current_selector = sorted(zarrs_with_crop_runs - zarrs_with_current_crop)

    contract_counts = Counter(str(row.get("pixel_contract_name") or "missing") for row in current_rows)
    backfill_counts = Counter(
        str((row.get("backfill") or {}).get("status"))
        for row in current_rows
        if isinstance(row.get("backfill"), Mapping)
    )
    storage_counts = Counter(
        str((row.get("relevant_attrs") or {}).get("crop_storage_mode") or "missing")
        for row in current_rows
        if isinstance(row.get("relevant_attrs"), Mapping)
    )
    representation_counts = Counter(
        str((row.get("relevant_attrs") or {}).get("roi_image_representation") or "missing")
        for row in current_rows
        if isinstance(row.get("relevant_attrs"), Mapping)
    )
    missing_contract_rows = [
        {
            "zarr_path": row.get("zarr_path"),
            "zarr_kind": row.get("zarr_kind"),
            "crop_run": row.get("crop_run_name") or _crop_run_name_from_surface_path(row.get("surface_path")),
            "surface_path": row.get("surface_path"),
            "missing_fields": list(row.get("missing_fields") or []),
            "backfill_status": (row.get("backfill") or {}).get("status") if isinstance(row.get("backfill"), Mapping) else None,
            "recommended_action": (row.get("backfill") or {}).get("action") if isinstance(row.get("backfill"), Mapping) else None,
        }
        for row in current_rows
        if row.get("missing_fields")
    ]

    return {
        "record_type": "current_crop_contract_report",
        "audit_utc": _utc_now(),
        "zarr_count": len(candidates),
        "zarrs_with_crop_runs": len(zarrs_with_crop_runs),
        "zarrs_with_current_crop_run": len(zarrs_with_current_crop),
        "zarrs_missing_current_crop_selector": len(zarrs_missing_current_selector),
        "crop_run_rows_scanned": len(crop_rows),
        "current_crop_run_rows": len(current_rows),
        "current_crop_runs_with_contract": sum(1 for row in current_rows if not row.get("missing_fields")),
        "current_crop_runs_missing_contract": sum(1 for row in current_rows if row.get("missing_fields")),
        "contract_counts": dict(sorted(contract_counts.items())),
        "backfill_status_counts": dict(sorted(backfill_counts.items())),
        "crop_storage_mode_counts": dict(sorted(storage_counts.items())),
        "roi_image_representation_counts": dict(sorted(representation_counts.items())),
        "missing_contract_rows": missing_contract_rows,
        "zarrs_missing_current_selector_examples": zarrs_missing_current_selector[:25],
    }


def _write_audit_jsonl(path: Path | None, rows: Sequence[Mapping[str, Any]]) -> None:
    if path is None:
        for row in rows:
            print(json.dumps(_json_safe(row), sort_keys=True))
        return
    write_jsonl_atomic(path, [_json_safe(row) for row in rows])


def _set_node_attr(path: Path, *, metadata_format: str | None, name: str, value: Any) -> tuple[bool, str]:
    return _set_node_attrs(path, metadata_format=metadata_format, values={name: value})


def _set_node_attrs(path: Path, *, metadata_format: str | None, values: Mapping[str, Any]) -> tuple[bool, str]:
    if metadata_format == "zarr.json":
        metadata_path = path / "zarr.json"
        payload = _read_json(metadata_path)
        if not payload:
            return False, "missing_or_invalid_zarr_json"
        attrs = payload.get("attributes")
        if not isinstance(attrs, dict):
            attrs = {}
            payload["attributes"] = attrs
        present = [name for name in values if attrs.get(name) not in (None, "")]
        if present:
            return False, f"already_present:{','.join(sorted(present))}"
        attrs.update(values)
        _write_json(metadata_path, payload)
        return True, "updated"

    if metadata_format == "zarr_v2":
        metadata_path = path / ".zattrs"
        attrs = _read_json(metadata_path) if _path_exists(metadata_path) else {}
        if not isinstance(attrs, dict):
            return False, "invalid_zattrs"
        present = [name for name in values if attrs.get(name) not in (None, "")]
        if present:
            return False, f"already_present:{','.join(sorted(present))}"
        attrs.update(values)
        _write_json(metadata_path, attrs)
        return True, "updated"

    return False, f"unsupported_metadata_format:{metadata_format}"


def _source_video_stat_fingerprint_payload(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    path_text = row.get("source_video_resolved_path") or row.get("source_video_path")
    if not path_text:
        return None, "missing_source_video_path"
    source_path = Path(str(path_text))
    try:
        stat = source_path.stat()
    except OSError as exc:
        return None, f"source_video_stat_failed:{exc}"

    components = {
        "strategy": SOURCE_VIDEO_STAT_FINGERPRINT_STRATEGY,
        "source_video_path": str(source_path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "codec": row.get("codec"),
        "pix_fmt": row.get("pix_fmt"),
        "width": row.get("width"),
        "height": row.get("height"),
        "fps": row.get("fps"),
        "frame_count": row.get("frame_count"),
    }
    digest_payload = json.dumps(_json_safe(components), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    return {
        "source_video_fingerprint": digest,
        "source_video_fingerprint_strategy": SOURCE_VIDEO_STAT_FINGERPRINT_STRATEGY,
        "source_video_fingerprint_payload": components,
        "source_video_size_bytes": int(stat.st_size),
        "source_video_mtime_ns": int(stat.st_mtime_ns),
    }, None


def apply_source_video_stat_fingerprints(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record_type") != "source_video_metadata":
            continue
        missing = set(str(item) for item in (row.get("missing_fields") or ()))
        if "fingerprint" not in missing:
            continue
        if not row.get("source_video_exists"):
            actions.append(
                {
                    "record_type": "source_video_stat_fingerprint_action",
                    "audit_utc": _utc_now(),
                    "zarr_path": row.get("zarr_path"),
                    "source_id": row.get("source_id"),
                    "status": "skipped",
                    "reason": "source_video_missing",
                }
            )
            continue
        payload, error = _source_video_stat_fingerprint_payload(row)
        if payload is None:
            actions.append(
                {
                    "record_type": "source_video_stat_fingerprint_action",
                    "audit_utc": _utc_now(),
                    "zarr_path": row.get("zarr_path"),
                    "source_id": row.get("source_id"),
                    "status": "skipped",
                    "reason": error or "fingerprint_payload_failed",
                }
            )
            continue
        node_path = Path(str(row["zarr_path"])) / "raw_video"
        try:
            updated, reason = _set_node_attrs(node_path, metadata_format="zarr.json", values=payload)
            if not updated and reason.startswith("unsupported_metadata_format"):
                updated, reason = _set_node_attrs(node_path, metadata_format="zarr_v2", values=payload)
        except OSError as exc:
            updated, reason = False, f"os_error:{exc}"
        actions.append(
            {
                "record_type": "source_video_stat_fingerprint_action",
                "audit_utc": _utc_now(),
                "zarr_path": row.get("zarr_path"),
                "zarr_kind": row.get("zarr_kind"),
                "source_scope": row.get("source_scope"),
                "source_id": row.get("source_id"),
                "source_video_path": row.get("source_video_path"),
                "source_video_resolved_path": row.get("source_video_resolved_path"),
                "source_video_fingerprint": payload.get("source_video_fingerprint"),
                "source_video_fingerprint_strategy": SOURCE_VIDEO_STAT_FINGERPRINT_STRATEGY,
                "status": "updated" if updated else "skipped",
                "reason": reason,
            }
        )
    return actions


def _probe_source_video_colorimetry(source_path: Path, *, ffprobe_bin: str) -> tuple[dict[str, Any] | None, str | None]:
    command = [
        str(ffprobe_bin),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=color_range,color_space,color_transfer,color_primaries",
        "-of",
        "json",
        str(source_path),
    ]
    try:
        output = subprocess.check_output(command, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, f"ffprobe_failed:{exc}"
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        return None, f"ffprobe_invalid_json:{exc}"
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return None, "ffprobe_no_video_stream"
    stream = streams[0]
    if not isinstance(stream, Mapping):
        return None, "ffprobe_invalid_stream"
    colorimetry = {
        name: str(stream[name])
        for name in SOURCE_VIDEO_COLOR_ATTRS
        if stream.get(name) not in (None, "")
    }
    if not colorimetry:
        return None, "ffprobe_no_colorimetry"
    return colorimetry, None


def apply_source_video_ffprobe_colorimetry(rows: Sequence[Mapping[str, Any]], *, ffprobe_bin: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record_type") != "source_video_metadata":
            continue
        missing = set(str(item) for item in (row.get("missing_fields") or ()))
        if "colorimetry" not in missing:
            continue
        if not row.get("source_video_exists"):
            actions.append(
                {
                    "record_type": "source_video_ffprobe_colorimetry_action",
                    "audit_utc": _utc_now(),
                    "zarr_path": row.get("zarr_path"),
                    "source_id": row.get("source_id"),
                    "status": "skipped",
                    "reason": "source_video_missing",
                }
            )
            continue
        path_text = row.get("source_video_resolved_path") or row.get("source_video_path")
        if not path_text:
            actions.append(
                {
                    "record_type": "source_video_ffprobe_colorimetry_action",
                    "audit_utc": _utc_now(),
                    "zarr_path": row.get("zarr_path"),
                    "source_id": row.get("source_id"),
                    "status": "skipped",
                    "reason": "missing_source_video_path",
                }
            )
            continue
        colorimetry, error = _probe_source_video_colorimetry(Path(str(path_text)), ffprobe_bin=ffprobe_bin)
        if colorimetry is None:
            actions.append(
                {
                    "record_type": "source_video_ffprobe_colorimetry_action",
                    "audit_utc": _utc_now(),
                    "zarr_path": row.get("zarr_path"),
                    "source_id": row.get("source_id"),
                    "source_video_resolved_path": str(path_text),
                    "status": "skipped",
                    "reason": error or "ffprobe_colorimetry_failed",
                }
            )
            continue
        values = {
            **colorimetry,
            "source_video_colorimetry_source": "ffprobe_stream",
        }
        node_path = Path(str(row["zarr_path"])) / "raw_video"
        try:
            updated, reason = _set_node_attrs(node_path, metadata_format="zarr.json", values=values)
            if not updated and reason.startswith("unsupported_metadata_format"):
                updated, reason = _set_node_attrs(node_path, metadata_format="zarr_v2", values=values)
        except OSError as exc:
            updated, reason = False, f"os_error:{exc}"
        actions.append(
            {
                "record_type": "source_video_ffprobe_colorimetry_action",
                "audit_utc": _utc_now(),
                "zarr_path": row.get("zarr_path"),
                "zarr_kind": row.get("zarr_kind"),
                "source_scope": row.get("source_scope"),
                "source_id": row.get("source_id"),
                "source_video_path": row.get("source_video_path"),
                "source_video_resolved_path": str(path_text),
                "colorimetry": colorimetry,
                "status": "updated" if updated else "skipped",
                "reason": reason,
            }
        )
    return actions


def apply_safe_scalar_name_backfill(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_crop_runs_only: bool = False,
) -> list[dict[str, Any]]:
    """Apply only the high-confidence crop scalar-name metadata backfill."""

    actions: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record_type") != "pixel_contract_surface":
            continue
        if row.get("surface_type") != "crop_run":
            continue
        if current_crop_runs_only and not row.get("is_current_crop_run"):
            continue
        backfill = row.get("backfill") if isinstance(row.get("backfill"), Mapping) else {}
        if backfill.get("status") != "safe_scalar_name_backfill":
            continue
        suggested_name = backfill.get("suggested_roi_pixel_contract_name") or row.get("pixel_contract_name")
        if not suggested_name:
            actions.append(
                {
                    "record_type": "safe_scalar_name_backfill_action",
                    "audit_utc": _utc_now(),
                    "zarr_path": row.get("zarr_path"),
                    "surface_path": row.get("surface_path"),
                    "status": "skipped",
                    "reason": "missing_suggested_name",
                }
            )
            continue
        node_path = Path(str(row["zarr_path"])) / str(row["surface_path"])
        try:
            updated, reason = _set_node_attr(
                node_path,
                metadata_format=str(row.get("metadata_format")) if row.get("metadata_format") else None,
                name="roi_pixel_contract_name",
                value=str(suggested_name),
            )
        except OSError as exc:
            updated, reason = False, f"os_error:{exc}"
        actions.append(
            {
                "record_type": "safe_scalar_name_backfill_action",
                "audit_utc": _utc_now(),
                "zarr_path": row.get("zarr_path"),
                "zarr_kind": row.get("zarr_kind"),
                "surface_path": row.get("surface_path"),
                "metadata_format": row.get("metadata_format"),
                "roi_pixel_contract_name": str(suggested_name),
                "status": "updated" if updated else "skipped",
                "reason": reason,
            }
        )
    return actions


def apply_inferred_legacy_crop_contracts(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_crop_runs_only: bool = False,
) -> list[dict[str, Any]]:
    """Apply medium-confidence legacy crop contracts inferred from crop-run attrs.

    This does not infer or promote current canonical PyNvVC-luma contracts. It
    only labels historical materialized crop runs with the legacy representation
    implied by their existing storage/source/acceleration metadata.
    """

    actions: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record_type") != "pixel_contract_surface":
            continue
        if row.get("surface_type") != "crop_run":
            continue
        if current_crop_runs_only and not row.get("is_current_crop_run"):
            continue
        backfill = row.get("backfill") if isinstance(row.get("backfill"), Mapping) else {}
        if backfill.get("status") != "infer_from_crop_run_attrs":
            continue

        suggested_contract = backfill.get("suggested_roi_pixel_contract")
        if not isinstance(suggested_contract, Mapping):
            actions.append(
                {
                    "record_type": "inferred_legacy_crop_contract_action",
                    "audit_utc": _utc_now(),
                    "zarr_path": row.get("zarr_path"),
                    "surface_path": row.get("surface_path"),
                    "status": "skipped",
                    "reason": "missing_suggested_contract",
                }
            )
            continue

        suggested_name = str(
            backfill.get("suggested_roi_pixel_contract_name") or suggested_contract.get("name") or ""
        ).strip()
        if not suggested_name:
            actions.append(
                {
                    "record_type": "inferred_legacy_crop_contract_action",
                    "audit_utc": _utc_now(),
                    "zarr_path": row.get("zarr_path"),
                    "surface_path": row.get("surface_path"),
                    "status": "skipped",
                    "reason": "missing_suggested_name",
                }
            )
            continue
        if suggested_name == ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME:
            actions.append(
                {
                    "record_type": "inferred_legacy_crop_contract_action",
                    "audit_utc": _utc_now(),
                    "zarr_path": row.get("zarr_path"),
                    "surface_path": row.get("surface_path"),
                    "roi_pixel_contract_name": suggested_name,
                    "status": "skipped",
                    "reason": "refusing_to_infer_current_canonical_contract",
                }
            )
            continue

        values = {
            "roi_pixel_contract": _json_safe(dict(suggested_contract)),
            "roi_pixel_contract_name": suggested_name,
        }
        node_path = Path(str(row["zarr_path"])) / str(row["surface_path"])
        try:
            updated, reason = _set_node_attrs(
                node_path,
                metadata_format=str(row.get("metadata_format")) if row.get("metadata_format") else None,
                values=values,
            )
        except OSError as exc:
            updated, reason = False, f"os_error:{exc}"

        actions.append(
            {
                "record_type": "inferred_legacy_crop_contract_action",
                "audit_utc": _utc_now(),
                "zarr_path": row.get("zarr_path"),
                "zarr_kind": row.get("zarr_kind"),
                "surface_path": row.get("surface_path"),
                "metadata_format": row.get("metadata_format"),
                "roi_pixel_contract_name": suggested_name,
                "roi_pixel_contract": _json_safe(dict(suggested_contract)),
                "status": "updated" if updated else "skipped",
                "reason": reason,
            }
        )
    return actions


def source_video_backfill_plan_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record_type") != "source_video_metadata":
            continue
        missing = list(row.get("missing_fields") or [])
        if not missing:
            continue
        missing_set = set(str(item) for item in missing)
        probe_fields = sorted(missing_set - {"fingerprint", "source_video_path"})
        source_path = row.get("source_video_resolved_path") or row.get("source_video_path")
        can_probe = bool(source_path) and "source_video_path" not in missing_set
        plan.append(
            {
                "record_type": "source_video_backfill_plan",
                "audit_utc": _utc_now(),
                "zarr_path": row.get("zarr_path"),
                "zarr_kind": row.get("zarr_kind"),
                "source_scope": row.get("source_scope"),
                "source_id": row.get("source_id"),
                "source_video_path": row.get("source_video_path"),
                "source_video_resolved_path": row.get("source_video_resolved_path"),
                "source_video_exists": row.get("source_video_exists"),
                "missing_fields": missing,
                "ffprobe_needed": bool(probe_fields),
                "ffprobe_fields": probe_fields,
                "fingerprint_needed": "fingerprint" in missing_set,
                "fingerprint_strategy": "record source path, size, mtime_ns, and optional full sha256/stream hash before strict enforcement",
                "can_probe_without_path_repair": can_probe,
                "recommended_action": (row.get("backfill") or {}).get("action") if isinstance(row.get("backfill"), Mapping) else None,
                "source_backfill_status": (row.get("backfill") or {}).get("status") if isinstance(row.get("backfill"), Mapping) else None,
            }
        )
    return plan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Filesystem roots or Zarr paths to scan. Use --recursive for deep recording/dataset trees.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively discover *.zarr under positional paths.")
    parser.add_argument("--registry", type=Path, help="Optional Palette registry SQLite path to include registered zarrs.")
    parser.add_argument(
        "--zarr-use",
        choices=("all", "analysis", "training", "training_data", "analysis_data"),
        default="all",
        help="Registry zarr_use filter when --registry is provided.",
    )
    parser.add_argument("--path-contains", help="Only include zarr paths containing this substring.")
    parser.add_argument("--limit", type=int, help="Limit discovered zarrs.")
    parser.add_argument("--skip-missing-zarrs", action="store_true", help="Drop discovered paths without Zarr metadata.")
    parser.add_argument(
        "--include-source-video-metadata",
        action="store_true",
        help="Also emit read-only encoded source-video metadata coverage rows.",
    )
    parser.add_argument("--output-jsonl", type=Path, help="Write audit rows to this JSONL path. Defaults to stdout.")
    parser.add_argument("--summary-json", type=Path, help="Write aggregate summary JSON.")
    parser.add_argument(
        "--crop-contract-report-json",
        type=Path,
        help="Write a focused report for the current crop run in each Zarr, using crop_runs latest/latest_complete/latest_any attrs.",
    )
    parser.add_argument(
        "--source-video-backfill-plan-jsonl",
        type=Path,
        help="Write a dry-run source-video metadata backfill plan derived from source_video_metadata rows.",
    )
    parser.add_argument(
        "--apply-safe-scalar-name-backfill",
        action="store_true",
        help="Apply only the safe crop-run roi_pixel_contract.name -> roi_pixel_contract_name metadata backfill.",
    )
    parser.add_argument(
        "--apply-inferred-legacy-crop-contracts",
        action="store_true",
        help=(
            "Apply medium-confidence legacy crop-run contracts inferred from existing "
            "crop_storage_mode/video_source_type/acceleration attrs. This refuses to "
            "infer the current canonical PyNvVC-luma contract."
        ),
    )
    parser.add_argument(
        "--apply-current-crop-runs-only",
        action="store_true",
        help=(
            "Limit crop-contract apply modes to the current crop run selected by "
            "crop_runs latest/latest_complete/latest_any attrs."
        ),
    )
    parser.add_argument(
        "--apply-source-video-stat-fingerprint",
        action="store_true",
        help="Apply a fast stat_v1 source-video fingerprint to raw_video attrs for rows missing only/including fingerprint and with an existing source video.",
    )
    parser.add_argument(
        "--apply-source-video-ffprobe-colorimetry",
        action="store_true",
        help="Apply ffprobe-reported stream colorimetry attrs to raw_video for rows missing colorimetry.",
    )
    parser.add_argument("--ffprobe-bin", default="ffprobe", help="ffprobe binary for --apply-source-video-ffprobe-colorimetry.")
    parser.add_argument("--summary-to-stderr", action="store_true", help="Print summary to stderr.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.registry is None and not args.paths:
        parser.error("provide at least one path or --registry")

    candidates = _discover_candidates(args)
    rows: list[dict[str, Any]] = []
    include_source_video_metadata = bool(
        args.include_source_video_metadata
        or args.source_video_backfill_plan_jsonl
        or args.apply_source_video_stat_fingerprint
        or args.apply_source_video_ffprobe_colorimetry
    )
    for candidate in candidates:
        rows.extend(audit_zarr_path(candidate, include_source_video_metadata=include_source_video_metadata))

    if args.apply_safe_scalar_name_backfill:
        rows.extend(
            apply_safe_scalar_name_backfill(
                rows,
                current_crop_runs_only=bool(args.apply_current_crop_runs_only),
            )
        )
    if args.apply_inferred_legacy_crop_contracts:
        rows.extend(
            apply_inferred_legacy_crop_contracts(
                rows,
                current_crop_runs_only=bool(args.apply_current_crop_runs_only),
            )
        )
    if args.apply_source_video_stat_fingerprint:
        rows.extend(apply_source_video_stat_fingerprints(rows))
    if args.apply_source_video_ffprobe_colorimetry:
        rows.extend(apply_source_video_ffprobe_colorimetry(rows, ffprobe_bin=str(args.ffprobe_bin)))
    if args.source_video_backfill_plan_jsonl is not None:
        _write_audit_jsonl(args.source_video_backfill_plan_jsonl, source_video_backfill_plan_rows(rows))
    summary = _summary(rows, candidates=candidates)
    if args.crop_contract_report_json is not None:
        args.crop_contract_report_json.parent.mkdir(parents=True, exist_ok=True)
        args.crop_contract_report_json.write_text(
            json.dumps(_json_safe(_crop_contract_report(rows, candidates=candidates)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _write_audit_jsonl(args.output_jsonl, rows)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_to_stderr or args.output_jsonl is not None:
        print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
