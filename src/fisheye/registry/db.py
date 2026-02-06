"""SQLite-backed registry for datasets, provenance, and training runs."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class RegistryPaths:
    path: Path

    @staticmethod
    def from_env(default_root: Path) -> "RegistryPaths":
        env_path = os.environ.get("PALETTE_REGISTRY_PATH")
        if env_path:
            return RegistryPaths(path=Path(env_path))
        config_path = _load_registry_path(default_root)
        if config_path:
            return RegistryPaths(path=config_path)
        return RegistryPaths(path=default_root / "runs" / "registry" / "palette_registry.sqlite")


def _load_registry_path(default_root: Path) -> Optional[Path]:
    config_path = default_root / "configs" / "fisheye" / "registry.yaml"
    if not config_path.exists():
        return None
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    path_value = None
    if "registry_path" in data:
        path_value = data.get("registry_path")
    elif isinstance(data.get("registry"), dict):
        path_value = data["registry"].get("path")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _import_zarr():
    """Lazy import to keep SQL-only registry commands independent of zarr."""
    try:
        import zarr  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise ModuleNotFoundError(
            "zarr is required for scan/register operations. Install zarr to read Zarr archives."
        ) from exc
    return zarr


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _json_loads(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


def _first_value(payload: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            if value is not None:
                return value
    return None


def _normalize_parents(value: Any) -> List[Dict[str, Optional[str]]]:
    if value is None:
        return []
    if isinstance(value, list):
        parents: List[Dict[str, Optional[str]]] = []
        for item in value:
            if isinstance(item, dict):
                parents.append(
                    {
                        "identifier": item.get("identifier"),
                        "sex": item.get("sex"),
                    }
                )
            elif isinstance(item, str):
                parents.append({"identifier": item, "sex": None})
        return parents
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        parsed = _json_loads(value)
        if isinstance(parsed, list):
            return _normalize_parents(parsed)
        parents = []
        for part in value.split(";"):
            ident = part.strip()
            if ident:
                parents.append({"identifier": ident, "sex": None})
        return parents
    return []


def _compute_path_hash(path: Path) -> str:
    return sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def _extract_session_uuid(root: zarr.Group) -> Optional[str]:
    for key in ("session_uuid", "session_id"):
        value = root.attrs.get(key)
        if value:
            return str(value)
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        value = analysis.attrs.get("session_uuid")
        if value:
            return str(value)
    return None


def resolve_dataset_id(root: zarr.Group, zarr_path: Path) -> Tuple[str, Optional[str]]:
    session_uuid = _extract_session_uuid(root)
    dataset_id = session_uuid or f"path-{_compute_path_hash(zarr_path)[:12]}"
    return dataset_id, session_uuid


def _extract_protocol(root: zarr.Group) -> Tuple[Optional[str], Optional[str]]:
    stim_parent = None
    if "analysis" in root and "stimulus_runs" in root["analysis"]:
        stim_parent = root["analysis"]["stimulus_runs"]
    if stim_parent is None:
        return None, None
    latest = stim_parent.attrs.get("latest")
    if not latest or latest not in stim_parent:
        return None, None
    stim_group = stim_parent[latest]
    raw = stim_group.attrs.get("protocol_json")
    payload = _json_loads(raw)
    if not payload:
        return None, None
    name = payload.get("protocol_name")
    proto_hash = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return str(name) if name else None, proto_hash


def _extract_snapshot(root: zarr.Group) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        for key in ("zebrobot_snapshot", "subject_metadata"):
            raw = analysis.attrs.get(key)
            payload = _json_loads(raw)
            if payload:
                return payload, key
    return None, None


def _extract_session_context(root: zarr.Group) -> Dict[str, Any]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return {}
    raw = analysis.attrs.get("session_context")
    payload = _json_loads(raw)
    return payload if isinstance(payload, dict) else {}


def _extract_arena_config(root: zarr.Group) -> Dict[str, Any]:
    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        return {}
    stim_parent = analysis["stimulus_runs"]
    latest = stim_parent.attrs.get("latest")
    if not latest or latest not in stim_parent:
        return {}
    run_group = stim_parent[latest]
    raw = run_group.attrs.get("arena_config_json")
    payload = _json_loads(raw)
    return payload if isinstance(payload, dict) else {}


def _extract_dish_design(root: zarr.Group) -> Optional[str]:
    value = root.attrs.get("dish_design")
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, str) and value.strip():
        return value.strip()
    arena_config = _extract_arena_config(root)
    dish_name = arena_config.get("selected_dish_type_name")
    if isinstance(dish_name, (bytes, bytearray)):
        dish_name = dish_name.decode("utf-8", "ignore")
    if isinstance(dish_name, str) and dish_name.strip():
        return dish_name.strip()
    return None


def _extract_camera_metadata(root: zarr.Group) -> Optional[Dict[str, Any]]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return None
    raw = analysis.attrs.get("camera_metadata")
    payload = _json_loads(raw)
    return payload if isinstance(payload, dict) else None


def _normalize_downsample_format(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"gray", "grey", "grayscale"}:
        return "gray"
    if text in {"rgb", "color", "colour"}:
        return "rgb"
    return None


def _extract_acquisition(root: zarr.Group) -> Dict[str, Any]:
    raw_video = root.get("raw_video")
    video_codec = None
    video_pix_fmt = None
    fps = None
    source_video = None
    format_title = None
    format_comment = None
    format_encoder = None
    encoder_name = None
    encoder_codec = None
    encoder_preset = None
    encoder_tuning = None
    encoder_rc = None
    encoder_bpp = None
    encoder_target_bps = None
    encoder_res = None
    encoder_res_width = None
    encoder_res_height = None
    encoder_fps = None
    encoder_color = None
    encoder_params = None
    compression_name = None
    compression_level = None
    has_images_ds = None
    has_images_ds_rgb = None
    downsample_formats: List[str] = []
    if raw_video is not None:
        fps = _as_float(raw_video.attrs.get("fps") or raw_video.attrs.get("frames_per_second"))
        video_codec = raw_video.attrs.get("video_codec") or raw_video.attrs.get("codec")
        video_pix_fmt = raw_video.attrs.get("video_pix_fmt") or raw_video.attrs.get("pix_fmt")
        source_video = raw_video.attrs.get("source_video")
        format_title = raw_video.attrs.get("format_title")
        format_comment = raw_video.attrs.get("format_comment")
        format_encoder = raw_video.attrs.get("format_encoder")
        encoder_name = raw_video.attrs.get("encoder_name")
        encoder_codec = raw_video.attrs.get("encoder_codec")
        encoder_preset = raw_video.attrs.get("encoder_preset")
        encoder_tuning = raw_video.attrs.get("encoder_tuning")
        encoder_rc = raw_video.attrs.get("encoder_rc")
        encoder_bpp = raw_video.attrs.get("encoder_bpp")
        encoder_target_bps = raw_video.attrs.get("encoder_target_bps")
        encoder_res = raw_video.attrs.get("encoder_res")
        encoder_res_width = raw_video.attrs.get("encoder_res_width")
        encoder_res_height = raw_video.attrs.get("encoder_res_height")
        encoder_fps = raw_video.attrs.get("encoder_fps")
        encoder_color = raw_video.attrs.get("encoder_color")
        encoder_params = raw_video.attrs.get("encoder_params")
        has_images_ds = "images_ds" in raw_video
        has_images_ds_rgb = "images_ds_rgb" in raw_video
        raw_formats = raw_video.attrs.get("downsample_formats")
        if isinstance(raw_formats, (list, tuple)):
            for item in raw_formats:
                normalized = _normalize_downsample_format(item)
                if normalized and normalized not in downsample_formats:
                    downsample_formats.append(normalized)
        compressor = raw_video.attrs.get("compressor")
        if isinstance(compressor, dict):
            compression_name = compressor.get("name")
            compression_level = _as_int(compressor.get("clevel"))
    if has_images_ds and "gray" not in downsample_formats:
        downsample_formats.append("gray")
    if has_images_ds_rgb and "rgb" not in downsample_formats:
        downsample_formats.append("rgb")
    if format_title is None:
        format_title = root.attrs.get("format_title")
    if format_comment is None:
        format_comment = root.attrs.get("format_comment")
    if format_encoder is None:
        format_encoder = root.attrs.get("format_encoder")
    if encoder_name is None:
        encoder_name = root.attrs.get("encoder_name")
    if encoder_codec is None:
        encoder_codec = root.attrs.get("encoder_codec")
    if encoder_preset is None:
        encoder_preset = root.attrs.get("encoder_preset")
    if encoder_tuning is None:
        encoder_tuning = root.attrs.get("encoder_tuning")
    if encoder_rc is None:
        encoder_rc = root.attrs.get("encoder_rc")
    if encoder_bpp is None:
        encoder_bpp = root.attrs.get("encoder_bpp")
    if encoder_target_bps is None:
        encoder_target_bps = root.attrs.get("encoder_target_bps")
    if encoder_res is None:
        encoder_res = root.attrs.get("encoder_res")
    if encoder_res_width is None:
        encoder_res_width = root.attrs.get("encoder_res_width")
    if encoder_res_height is None:
        encoder_res_height = root.attrs.get("encoder_res_height")
    if encoder_fps is None:
        encoder_fps = root.attrs.get("encoder_fps")
    if encoder_color is None:
        encoder_color = root.attrs.get("encoder_color")
    if encoder_params is None:
        encoder_params = root.attrs.get("encoder_params")

    camera_meta = _extract_camera_metadata(root) or {}
    exposure = _as_float(_first_value(camera_meta, ("exposure", "exposure_ms", "exposure_us")))
    gain = _as_float(_first_value(camera_meta, ("gain", "camera_gain")))
    frame_rate = _as_float(_first_value(camera_meta, ("frame_rate", "fps", "framerate")))
    pixel_format = _first_value(camera_meta, ("pixel_format", "pixelFormat"))
    binning = _first_value(camera_meta, ("bin", "binning"))
    adc = _first_value(camera_meta, ("adc", "bit_depth"))
    camera_model = _first_value(camera_meta, ("device_model_name", "camera_model"))
    camera_serial = _first_value(camera_meta, ("device_serial_number", "serial_number", "camera_id"))

    return {
        "dish_design": _extract_dish_design(root),
        "fps": fps,
        "video_codec": str(video_codec) if video_codec is not None else None,
        "video_pix_fmt": str(video_pix_fmt) if video_pix_fmt is not None else None,
        "source_video": str(source_video) if source_video is not None else None,
        "format_title": _as_text(format_title),
        "format_comment": _as_text(format_comment),
        "format_encoder": _as_text(format_encoder),
        "encoder_name": _as_text(encoder_name),
        "encoder_codec": _as_text(encoder_codec),
        "encoder_preset": _as_text(encoder_preset),
        "encoder_tuning": _as_text(encoder_tuning),
        "encoder_rc": _as_text(encoder_rc),
        "encoder_bpp": _as_float(encoder_bpp),
        "encoder_target_bps": _as_int(encoder_target_bps),
        "encoder_res": _as_text(encoder_res),
        "encoder_res_width": _as_int(encoder_res_width),
        "encoder_res_height": _as_int(encoder_res_height),
        "encoder_fps": _as_float(encoder_fps),
        "encoder_color": _as_int(encoder_color),
        "encoder_params_json": _json_dumps(encoder_params) if encoder_params else None,
        "compression_name": str(compression_name) if compression_name is not None else None,
        "compression_level": compression_level,
        "exposure": exposure,
        "exposure_unit": "us" if exposure is not None else None,
        "gain": gain,
        "frame_rate": frame_rate,
        "pixel_format": str(pixel_format) if pixel_format is not None else None,
        "binning": str(binning) if binning is not None else None,
        "adc": str(adc) if adc is not None else None,
        "camera_model": str(camera_model) if camera_model is not None else None,
        "camera_serial": str(camera_serial) if camera_serial is not None else None,
        "camera_metadata_json": _json_dumps(camera_meta) if camera_meta else None,
        "has_images_ds": bool(has_images_ds) if has_images_ds is not None else None,
        "has_images_ds_rgb": bool(has_images_ds_rgb) if has_images_ds_rgb is not None else None,
        "downsample_formats_json": _json_dumps(downsample_formats) if downsample_formats else None,
    }


def _extract_provenance(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not snapshot:
        return {}
    dish = snapshot.get("dish") or snapshot
    cross = snapshot.get("cross") or {}
    return {
        "fish_id": snapshot.get("fish_id"),
        "subject_count": snapshot.get("subject_count"),
        "dish_id": snapshot.get("dish_id") or dish.get("dish_id"),
        "cross_id": dish.get("cross_id") or cross.get("cross_id"),
        "line_strain": cross.get("line_strain") or dish.get("line_strain"),
        "genotype": dish.get("genotype"),
        "parents": _normalize_parents(cross.get("parents") or dish.get("parents")),
        "species": dish.get("species"),
        "sex": dish.get("sex"),
        "dpf_at_acquisition": snapshot.get("dpf_at_acquisition"),
        "snapshot_status": snapshot.get("status"),
        "snapshot_missing": snapshot.get("missing"),
    }


def _extract_zarr_purpose(root: zarr.Group) -> Optional[str]:
    value = root.attrs.get("zarr_purpose")
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    text = str(value).strip()
    return text if text else None


def _normalize_path_text(value: Any) -> Optional[str]:
    text = _as_text(value)
    if text is None:
        return None
    normalized = text.strip("/")
    return normalized or None


def _canonical_run_path(path: Optional[str]) -> Optional[str]:
    normalized = _normalize_path_text(path)
    if normalized is None:
        return None
    parts = normalized.split("/")
    if len(parts) >= 2 and parts[0] in {"refined_detect_runs", "refined_runs", "detect_runs"}:
        return "/".join(parts[:2])
    return normalized


def _infer_detection_source_type(path: Optional[str], fallback: Optional[Any]) -> str:
    fallback_text = _as_text(fallback)
    fallback_norm = fallback_text.lower() if fallback_text else None
    normalized_path = _normalize_path_text(path)
    if normalized_path:
        parts = normalized_path.split("/")
        tail = parts[-1].lower()
        if tail in {"detect", "filtered", "interpolated", "manual", "retune"}:
            return tail
        if parts[0] == "detect_runs":
            return "detect"
    if fallback_norm:
        return fallback_norm
    return "detect"


def _resolve_latest_group_name(parent: Optional[zarr.Group]) -> Optional[str]:
    if parent is None:
        return None
    latest = _as_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    if hasattr(parent, "group_keys"):
        names = sorted(
            name
            for name in parent.group_keys()
            if isinstance(name, str)
        )
    else:
        names = sorted(
            name
            for name in parent.keys()
            if isinstance(name, str)
        )
    if not names:
        return None
    return names[-1]


def _build_detection_source_records(root: zarr.Group) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    crop_parent = root.get("crop_runs")
    crop_run_name = _resolve_latest_group_name(crop_parent)
    if crop_parent is not None and crop_run_name and crop_run_name in crop_parent:
        crop_group = crop_parent[crop_run_name]
        source_path = _normalize_path_text(crop_group.attrs.get("detection_source_path"))
        source_type = _infer_detection_source_type(source_path, crop_group.attrs.get("detection_source_type"))

        refined_ref = _canonical_run_path(crop_group.attrs.get("detect_review_status_ref"))
        if refined_ref is None:
            refined_ref = _canonical_run_path(source_path)
        if refined_ref is None:
            source_refined = _as_text(crop_group.attrs.get("source_refined_run"))
            if source_refined:
                preferred = f"refined_detect_runs/{source_refined}"
                legacy = f"refined_runs/{source_refined}"
                if preferred in root:
                    refined_ref = preferred
                elif legacy in root:
                    refined_ref = legacy
                else:
                    refined_ref = preferred
        if refined_ref is None:
            refined_ref = "unknown"

        total_detections = int(crop_group["bbox_norm_coords"].shape[0]) if "bbox_norm_coords" in crop_group else 0
        source_code_counts: Dict[str, int] = {}
        if "detection_source" in crop_group:
            raw_source = np.asarray(crop_group["detection_source"][:], dtype=np.int64)
            if raw_source.size > 0:
                unique, counts = np.unique(raw_source, return_counts=True)
                source_code_counts = {
                    str(int(code)): int(count)
                    for code, count in zip(unique.tolist(), counts.tolist())
                }

        n_real_attr = _as_int(crop_group.attrs.get("n_real_detections"))
        n_interp_attr = _as_int(crop_group.attrs.get("n_interpolated_detections"))
        n_real = source_code_counts.get("0", n_real_attr if n_real_attr is not None else total_detections)
        n_interpolated = source_code_counts.get("1", n_interp_attr if n_interp_attr is not None else 0)
        includes_interpolated = bool(
            crop_group.attrs.get("includes_interpolated", n_interpolated > 0)
        )

        counts_payload = {
            "crop_run": crop_run_name,
            "detection_source_path": source_path,
            "total_detections": int(total_detections),
            "n_real_detections": int(max(n_real, 0)),
            "n_interpolated_detections": int(max(n_interpolated, 0)),
            "includes_interpolated": includes_interpolated,
        }
        if source_code_counts:
            counts_payload["detection_source_codes"] = source_code_counts

        records.append(
            {
                "refined_run": refined_ref,
                "source_type": source_type,
                "counts": counts_payload,
            }
        )
        return records

    detect_parent = root.get("detect_runs")
    detect_run_name = _resolve_latest_group_name(detect_parent)
    if detect_parent is None or detect_run_name is None or detect_run_name not in detect_parent:
        return records

    detect_group = detect_parent[detect_run_name]
    total_detections = int(detect_group["bbox_norm_coords"].shape[0]) if "bbox_norm_coords" in detect_group else 0
    detect_path = f"detect_runs/{detect_run_name}"
    records.append(
        {
            "refined_run": detect_path,
            "source_type": "detect",
            "counts": {
                "detect_run": detect_run_name,
                "detection_source_path": detect_path,
                "total_detections": total_detections,
                "n_real_detections": total_detections,
                "n_interpolated_detections": 0,
                "includes_interpolated": False,
            },
        }
    )
    return records


class Registry:
    def __init__(self, path: Path):
        self.path = path
        _ensure_parent(self.path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute("PRAGMA user_version = 1;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                session_uuid TEXT,
                zarr_path TEXT NOT NULL,
                path_hash TEXT,
                created_utc TEXT,
                last_seen_utc TEXT,
                status TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS provenance (
                dataset_id TEXT PRIMARY KEY,
                fish_id TEXT,
                subject_count INTEGER,
                dish_id TEXT,
                dish_design TEXT,
                cross_id TEXT,
                line_strain TEXT,
                genotype TEXT,
                parents_json TEXT,
                species TEXT,
                sex TEXT,
                dpf_at_acquisition INTEGER,
                rig_id TEXT,
                arena_id TEXT,
                camera_id TEXT,
                canvas_name TEXT,
                fps REAL,
                video_codec TEXT,
                video_pix_fmt TEXT,
                format_title TEXT,
                format_comment TEXT,
                format_encoder TEXT,
                encoder_name TEXT,
                encoder_codec TEXT,
                encoder_preset TEXT,
                encoder_tuning TEXT,
                encoder_rc TEXT,
                encoder_bpp REAL,
                encoder_target_bps INTEGER,
                encoder_res TEXT,
                encoder_res_width INTEGER,
                encoder_res_height INTEGER,
                encoder_fps REAL,
                encoder_color INTEGER,
                encoder_params_json TEXT,
                source_video TEXT,
                compression_name TEXT,
                compression_level INTEGER,
                exposure REAL,
                exposure_unit TEXT,
                gain REAL,
                frame_rate REAL,
                pixel_format TEXT,
                binning TEXT,
                adc TEXT,
                camera_model TEXT,
                camera_serial TEXT,
                camera_metadata_json TEXT,
                has_images_ds INTEGER,
                has_images_ds_rgb INTEGER,
                downsample_formats_json TEXT,
                zarr_purpose TEXT,
                protocol_name TEXT,
                protocol_hash TEXT,
                snapshot_status TEXT,
                snapshot_missing_json TEXT,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_sources (
                dataset_id TEXT NOT NULL,
                refined_run TEXT,
                source_type TEXT,
                counts_json TEXT,
                created_utc TEXT,
                PRIMARY KEY (dataset_id, refined_run, source_type),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_sets (
                set_id TEXT PRIMARY KEY,
                name TEXT,
                query_filter TEXT,
                dataset_ids_json TEXT,
                invocation_json TEXT,
                created_utc TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                set_id TEXT,
                config_path TEXT,
                manifest_path TEXT,
                model_path TEXT,
                metrics_path TEXT,
                config_sha256 TEXT,
                manifest_sha256 TEXT,
                model_sha256 TEXT,
                metrics_sha256 TEXT,
                status TEXT,
                final_metrics_json TEXT,
                invocation_json TEXT,
                created_utc TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_exports (
                run_id TEXT NOT NULL,
                export_type TEXT NOT NULL,
                path TEXT,
                manifest_path TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                PRIMARY KEY (run_id, export_type),
                FOREIGN KEY(run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()
        self._ensure_columns(
            "provenance",
            {
                "fish_id": "TEXT",
                "subject_count": "INTEGER",
                "zarr_purpose": "TEXT",
                "rig_id": "TEXT",
                "arena_id": "TEXT",
                "camera_id": "TEXT",
                "canvas_name": "TEXT",
                "dish_design": "TEXT",
                "fps": "REAL",
                "video_codec": "TEXT",
                "video_pix_fmt": "TEXT",
                "format_title": "TEXT",
                "format_comment": "TEXT",
                "format_encoder": "TEXT",
                "encoder_name": "TEXT",
                "encoder_codec": "TEXT",
                "encoder_preset": "TEXT",
                "encoder_tuning": "TEXT",
                "encoder_rc": "TEXT",
                "encoder_bpp": "REAL",
                "encoder_target_bps": "INTEGER",
                "encoder_res": "TEXT",
                "encoder_res_width": "INTEGER",
                "encoder_res_height": "INTEGER",
                "encoder_fps": "REAL",
                "encoder_color": "INTEGER",
                "encoder_params_json": "TEXT",
                "source_video": "TEXT",
                "compression_name": "TEXT",
                "compression_level": "INTEGER",
                "exposure": "REAL",
                "exposure_unit": "TEXT",
                "gain": "REAL",
                "frame_rate": "REAL",
                "pixel_format": "TEXT",
                "binning": "TEXT",
                "adc": "TEXT",
                "camera_model": "TEXT",
                "camera_serial": "TEXT",
                "camera_metadata_json": "TEXT",
                "has_images_ds": "INTEGER",
                "has_images_ds_rgb": "INTEGER",
                "downsample_formats_json": "TEXT",
            },
        )
        self._ensure_columns("training_sets", {"invocation_json": "TEXT"})
        self._ensure_columns(
            "training_runs",
            {
                "invocation_json": "TEXT",
                "config_sha256": "TEXT",
                "manifest_sha256": "TEXT",
                "model_sha256": "TEXT",
                "metrics_sha256": "TEXT",
                "status": "TEXT",
                "final_metrics_json": "TEXT",
            },
        )

    def _ensure_columns(self, table: str, columns: Dict[str, str]) -> None:
        existing = {
            row["name"]
            for row in self.conn.execute(f"PRAGMA table_info({table});").fetchall()
        }
        for name, ddl in columns.items():
            if name in existing:
                continue
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl};")
        self.conn.commit()

    def upsert_dataset(self, dataset_id: str, *, session_uuid: Optional[str], zarr_path: Path) -> None:
        now = _utc_now()
        payload = {
            "dataset_id": dataset_id,
            "session_uuid": session_uuid,
            "zarr_path": str(zarr_path),
            "path_hash": _compute_path_hash(zarr_path),
            "created_utc": now,
            "last_seen_utc": now,
            "status": "active",
        }
        self.conn.execute(
            """
            INSERT INTO datasets (dataset_id, session_uuid, zarr_path, path_hash, created_utc, last_seen_utc, status)
            VALUES (:dataset_id, :session_uuid, :zarr_path, :path_hash, :created_utc, :last_seen_utc, :status)
            ON CONFLICT(dataset_id) DO UPDATE SET
                session_uuid=excluded.session_uuid,
                zarr_path=excluded.zarr_path,
                path_hash=excluded.path_hash,
                last_seen_utc=excluded.last_seen_utc,
                status=excluded.status;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_provenance(
        self,
        dataset_id: str,
        *,
        provenance: Dict[str, Any],
        context: Dict[str, Any],
        protocol_name: Optional[str],
        protocol_hash: Optional[str],
        acquisition: Optional[Dict[str, Any]] = None,
        zarr_purpose: Optional[str] = None,
    ) -> None:
        acquisition = acquisition or {}
        payload = {
            "dataset_id": dataset_id,
            "fish_id": provenance.get("fish_id"),
            "subject_count": provenance.get("subject_count"),
            "dish_id": provenance.get("dish_id"),
            "dish_design": acquisition.get("dish_design"),
            "cross_id": provenance.get("cross_id"),
            "line_strain": provenance.get("line_strain"),
            "genotype": provenance.get("genotype"),
            "parents_json": _json_dumps(provenance.get("parents")),
            "species": provenance.get("species"),
            "sex": provenance.get("sex"),
            "dpf_at_acquisition": provenance.get("dpf_at_acquisition"),
            "rig_id": context.get("rig_id"),
            "arena_id": context.get("arena_id"),
            "camera_id": context.get("camera_id"),
            "canvas_name": context.get("canvas_name"),
            "fps": acquisition.get("fps"),
            "video_codec": acquisition.get("video_codec"),
            "video_pix_fmt": acquisition.get("video_pix_fmt"),
            "format_title": acquisition.get("format_title"),
            "format_comment": acquisition.get("format_comment"),
            "format_encoder": acquisition.get("format_encoder"),
            "encoder_name": acquisition.get("encoder_name"),
            "encoder_codec": acquisition.get("encoder_codec"),
            "encoder_preset": acquisition.get("encoder_preset"),
            "encoder_tuning": acquisition.get("encoder_tuning"),
            "encoder_rc": acquisition.get("encoder_rc"),
            "encoder_bpp": acquisition.get("encoder_bpp"),
            "encoder_target_bps": acquisition.get("encoder_target_bps"),
            "encoder_res": acquisition.get("encoder_res"),
            "encoder_res_width": acquisition.get("encoder_res_width"),
            "encoder_res_height": acquisition.get("encoder_res_height"),
            "encoder_fps": acquisition.get("encoder_fps"),
            "encoder_color": acquisition.get("encoder_color"),
            "encoder_params_json": acquisition.get("encoder_params_json"),
            "source_video": acquisition.get("source_video"),
            "compression_name": acquisition.get("compression_name"),
            "compression_level": acquisition.get("compression_level"),
            "exposure": acquisition.get("exposure"),
            "exposure_unit": acquisition.get("exposure_unit"),
            "gain": acquisition.get("gain"),
            "frame_rate": acquisition.get("frame_rate"),
            "pixel_format": acquisition.get("pixel_format"),
            "binning": acquisition.get("binning"),
            "adc": acquisition.get("adc"),
            "camera_model": acquisition.get("camera_model"),
            "camera_serial": acquisition.get("camera_serial"),
            "camera_metadata_json": acquisition.get("camera_metadata_json"),
            "has_images_ds": acquisition.get("has_images_ds"),
            "has_images_ds_rgb": acquisition.get("has_images_ds_rgb"),
            "downsample_formats_json": acquisition.get("downsample_formats_json"),
            "zarr_purpose": zarr_purpose,
            "protocol_name": protocol_name,
            "protocol_hash": protocol_hash,
            "snapshot_status": provenance.get("snapshot_status"),
            "snapshot_missing_json": _json_dumps(provenance.get("snapshot_missing")),
        }
        self.conn.execute(
            """
            INSERT INTO provenance (
                dataset_id, fish_id, subject_count, dish_id, dish_design, cross_id, line_strain, genotype, parents_json,
                species, sex, dpf_at_acquisition, rig_id, arena_id, camera_id, canvas_name,
                fps, video_codec, video_pix_fmt, format_title, format_comment, format_encoder,
                encoder_name, encoder_codec, encoder_preset, encoder_tuning, encoder_rc, encoder_bpp,
                encoder_target_bps, encoder_res, encoder_res_width, encoder_res_height, encoder_fps, encoder_color,
                encoder_params_json,
                source_video, compression_name, compression_level,
                exposure, exposure_unit, gain, frame_rate, pixel_format, binning, adc, camera_model, camera_serial,
                camera_metadata_json, has_images_ds, has_images_ds_rgb, downsample_formats_json,
                zarr_purpose, protocol_name, protocol_hash, snapshot_status, snapshot_missing_json
            )
            VALUES (
                :dataset_id, :fish_id, :subject_count, :dish_id, :dish_design, :cross_id, :line_strain, :genotype, :parents_json,
                :species, :sex, :dpf_at_acquisition, :rig_id, :arena_id, :camera_id, :canvas_name,
                :fps, :video_codec, :video_pix_fmt, :format_title, :format_comment, :format_encoder,
                :encoder_name, :encoder_codec, :encoder_preset, :encoder_tuning, :encoder_rc, :encoder_bpp,
                :encoder_target_bps, :encoder_res, :encoder_res_width, :encoder_res_height, :encoder_fps, :encoder_color,
                :encoder_params_json,
                :source_video, :compression_name, :compression_level,
                :exposure, :exposure_unit, :gain, :frame_rate, :pixel_format, :binning, :adc, :camera_model, :camera_serial,
                :camera_metadata_json, :has_images_ds, :has_images_ds_rgb, :downsample_formats_json,
                :zarr_purpose, :protocol_name, :protocol_hash, :snapshot_status, :snapshot_missing_json
            )
            ON CONFLICT(dataset_id) DO UPDATE SET
                fish_id=excluded.fish_id,
                subject_count=excluded.subject_count,
                dish_id=excluded.dish_id,
                dish_design=excluded.dish_design,
                cross_id=excluded.cross_id,
                line_strain=excluded.line_strain,
                genotype=excluded.genotype,
                parents_json=excluded.parents_json,
                species=excluded.species,
                sex=excluded.sex,
                dpf_at_acquisition=excluded.dpf_at_acquisition,
                rig_id=excluded.rig_id,
                arena_id=excluded.arena_id,
                camera_id=excluded.camera_id,
                canvas_name=excluded.canvas_name,
                fps=excluded.fps,
                video_codec=excluded.video_codec,
                video_pix_fmt=excluded.video_pix_fmt,
                format_title=excluded.format_title,
                format_comment=excluded.format_comment,
                format_encoder=excluded.format_encoder,
                encoder_name=excluded.encoder_name,
                encoder_codec=excluded.encoder_codec,
                encoder_preset=excluded.encoder_preset,
                encoder_tuning=excluded.encoder_tuning,
                encoder_rc=excluded.encoder_rc,
                encoder_bpp=excluded.encoder_bpp,
                encoder_target_bps=excluded.encoder_target_bps,
                encoder_res=excluded.encoder_res,
                encoder_res_width=excluded.encoder_res_width,
                encoder_res_height=excluded.encoder_res_height,
                encoder_fps=excluded.encoder_fps,
                encoder_color=excluded.encoder_color,
                encoder_params_json=excluded.encoder_params_json,
                source_video=excluded.source_video,
                compression_name=excluded.compression_name,
                compression_level=excluded.compression_level,
                exposure=excluded.exposure,
                exposure_unit=excluded.exposure_unit,
                gain=excluded.gain,
                frame_rate=excluded.frame_rate,
                pixel_format=excluded.pixel_format,
                binning=excluded.binning,
                adc=excluded.adc,
                camera_model=excluded.camera_model,
                camera_serial=excluded.camera_serial,
                camera_metadata_json=excluded.camera_metadata_json,
                has_images_ds=excluded.has_images_ds,
                has_images_ds_rgb=excluded.has_images_ds_rgb,
                downsample_formats_json=excluded.downsample_formats_json,
                zarr_purpose=excluded.zarr_purpose,
                protocol_name=excluded.protocol_name,
                protocol_hash=excluded.protocol_hash,
                snapshot_status=excluded.snapshot_status,
                snapshot_missing_json=excluded.snapshot_missing_json;
            """,
            payload,
        )
        self.conn.commit()

    def replace_detection_sources(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        """Replace detection source lineage rows for a dataset."""
        now = _utc_now()
        with self.conn:
            self.conn.execute(
                "DELETE FROM detection_sources WHERE dataset_id = ?;",
                (dataset_id,),
            )
            for record in records:
                refined_run = _normalize_path_text(record.get("refined_run"))
                source_type = _as_text(record.get("source_type"))
                if refined_run is None or source_type is None:
                    continue
                payload = {
                    "dataset_id": dataset_id,
                    "refined_run": refined_run,
                    "source_type": source_type.lower(),
                    "counts_json": _json_dumps(record.get("counts")),
                    "created_utc": _as_text(record.get("created_utc")) or now,
                }
                self.conn.execute(
                    """
                    INSERT INTO detection_sources (dataset_id, refined_run, source_type, counts_json, created_utc)
                    VALUES (:dataset_id, :refined_run, :source_type, :counts_json, :created_utc)
                    ON CONFLICT(dataset_id, refined_run, source_type) DO UPDATE SET
                        counts_json=excluded.counts_json,
                        created_utc=excluded.created_utc;
                    """,
                    payload,
                )

    def register_from_root(self, root: zarr.Group, zarr_path: Path) -> str:
        dataset_id, session_uuid = resolve_dataset_id(root, zarr_path)
        self.upsert_dataset(dataset_id, session_uuid=session_uuid, zarr_path=zarr_path)

        protocol_name, protocol_hash = _extract_protocol(root)
        snapshot, _ = _extract_snapshot(root)
        provenance = _extract_provenance(snapshot)
        context = _extract_session_context(root)
        acquisition = _extract_acquisition(root)
        zarr_purpose = _extract_zarr_purpose(root)
        self.upsert_provenance(
            dataset_id,
            provenance=provenance,
            context=context,
            protocol_name=protocol_name,
            protocol_hash=protocol_hash,
            acquisition=acquisition,
            zarr_purpose=zarr_purpose,
        )
        detection_records = _build_detection_source_records(root)
        self.replace_detection_sources(dataset_id, detection_records)
        return dataset_id

    def record_training_run(
        self,
        *,
        run_id: str,
        set_id: Optional[str],
        config_path: Optional[Path],
        manifest_path: Optional[Path],
        model_path: Optional[Path],
        metrics_path: Optional[Path],
        config_sha256: Optional[str] = None,
        manifest_sha256: Optional[str] = None,
        model_sha256: Optional[str] = None,
        metrics_sha256: Optional[str] = None,
        status: Optional[str] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
        invocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "run_id": run_id,
            "set_id": set_id,
            "config_path": str(config_path) if config_path else None,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "model_path": str(model_path) if model_path else None,
            "metrics_path": str(metrics_path) if metrics_path else None,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "model_sha256": model_sha256,
            "metrics_sha256": metrics_sha256,
            "status": status,
            "final_metrics_json": _json_dumps(final_metrics),
            "invocation_json": _json_dumps(invocation),
            "created_utc": _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO training_runs (
                run_id, set_id, config_path, manifest_path, model_path, metrics_path,
                config_sha256, manifest_sha256, model_sha256, metrics_sha256,
                status, final_metrics_json,
                invocation_json, created_utc
            )
            VALUES (
                :run_id, :set_id, :config_path, :manifest_path, :model_path, :metrics_path,
                :config_sha256, :manifest_sha256, :model_sha256, :metrics_sha256,
                :status, :final_metrics_json,
                :invocation_json, :created_utc
            )
            ON CONFLICT(run_id) DO UPDATE SET
                set_id=excluded.set_id,
                config_path=excluded.config_path,
                manifest_path=excluded.manifest_path,
                model_path=excluded.model_path,
                metrics_path=excluded.metrics_path,
                config_sha256=excluded.config_sha256,
                manifest_sha256=excluded.manifest_sha256,
                model_sha256=excluded.model_sha256,
                metrics_sha256=excluded.metrics_sha256,
                status=excluded.status,
                final_metrics_json=excluded.final_metrics_json,
                invocation_json=excluded.invocation_json,
                created_utc=excluded.created_utc;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_training_set(
        self,
        *,
        set_id: str,
        name: Optional[str],
        query_filter: Optional[Dict[str, Any]],
        dataset_ids: Iterable[str],
        invocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        dataset_ids_norm = sorted({str(dataset_id) for dataset_id in dataset_ids if dataset_id})
        payload = {
            "set_id": str(set_id),
            "name": name,
            "query_filter": _json_dumps(query_filter),
            "dataset_ids_json": _json_dumps(dataset_ids_norm),
            "invocation_json": _json_dumps(invocation),
            "created_utc": _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO training_sets (
                set_id, name, query_filter, dataset_ids_json, invocation_json, created_utc
            )
            VALUES (
                :set_id, :name, :query_filter, :dataset_ids_json, :invocation_json, :created_utc
            )
            ON CONFLICT(set_id) DO UPDATE SET
                name=excluded.name,
                query_filter=excluded.query_filter,
                dataset_ids_json=excluded.dataset_ids_json,
                invocation_json=excluded.invocation_json,
                created_utc=excluded.created_utc;
            """,
            payload,
        )
        self.conn.commit()

    def record_model_export(
        self,
        *,
        run_id: str,
        export_type: str,
        path: Optional[Path],
        manifest_path: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "run_id": run_id,
            "export_type": export_type,
            "path": str(path) if path else None,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "metadata_json": _json_dumps(metadata),
            "created_utc": _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO model_exports (run_id, export_type, path, manifest_path, metadata_json, created_utc)
            VALUES (:run_id, :export_type, :path, :manifest_path, :metadata_json, :created_utc)
            ON CONFLICT(run_id, export_type) DO UPDATE SET
                path=excluded.path,
                manifest_path=excluded.manifest_path,
                metadata_json=excluded.metadata_json,
                created_utc=excluded.created_utc;
            """,
            payload,
        )
        self.conn.commit()

    def scan_zarr(self, zarr_path: Path) -> Optional[str]:
        if not zarr_path.exists():
            return None
        zarr = _import_zarr()
        root = zarr.open(str(zarr_path), mode="r")
        return self.register_from_root(root, zarr_path)

    def reconcile_missing_datasets(self, *, scope_paths: Optional[Iterable[Path]] = None) -> Dict[str, int]:
        """
        Mark datasets as missing when their registered zarr_path no longer exists.

        When scope_paths are provided, reconciliation is limited to dataset paths
        inside those roots (or exact path matches).
        """
        scope_roots = [_normalize_fs_path(path) for path in (scope_paths or [])]
        rows = self.conn.execute(
            "SELECT dataset_id, zarr_path FROM datasets WHERE status IS NULL OR status != 'missing';"
        ).fetchall()

        checked = 0
        marked_missing = 0
        with self.conn:
            for row in rows:
                dataset_path = _normalize_fs_path(row["zarr_path"])
                if scope_roots and not _path_matches_scope(dataset_path, scope_roots):
                    continue
                checked += 1
                if _is_zarr_root(dataset_path):
                    continue
                self.conn.execute(
                    "UPDATE datasets SET status = 'missing' WHERE dataset_id = ?;",
                    (row["dataset_id"],),
                )
                marked_missing += 1

        return {"checked": checked, "marked_missing": marked_missing}

    def query_datasets(
        self,
        *,
        dish_design: Optional[str] = None,
        dish_design_like: Optional[str] = None,
        fish_id: Optional[str] = None,
        subject_count_min: Optional[int] = None,
        subject_count_max: Optional[int] = None,
        zarr_purpose: Optional[str] = None,
        fps_min: Optional[float] = None,
        fps_max: Optional[float] = None,
        exposure_min: Optional[float] = None,
        exposure_max: Optional[float] = None,
        frame_rate_min: Optional[float] = None,
        frame_rate_max: Optional[float] = None,
        gain_min: Optional[float] = None,
        gain_max: Optional[float] = None,
        video_codec: Optional[str] = None,
        video_pix_fmt: Optional[str] = None,
        format_encoder: Optional[str] = None,
        format_title: Optional[str] = None,
        format_comment: Optional[str] = None,
        encoder_name: Optional[str] = None,
        encoder_codec: Optional[str] = None,
        encoder_preset: Optional[str] = None,
        encoder_tuning: Optional[str] = None,
        encoder_rc: Optional[str] = None,
        compression_name: Optional[str] = None,
        camera_model: Optional[str] = None,
        camera_serial: Optional[str] = None,
        camera_id: Optional[str] = None,
        rig_id: Optional[str] = None,
        arena_id: Optional[str] = None,
        model_input: Optional[str] = None,
        path_contains: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[sqlite3.Row]:
        sql = [
            "SELECT d.dataset_id, d.session_uuid, d.zarr_path,",
            "p.dish_design, p.fish_id, p.subject_count, p.zarr_purpose, p.fps, p.exposure, p.exposure_unit, p.frame_rate, p.gain,",
            "p.video_codec, p.video_pix_fmt, p.format_title, p.format_comment, p.format_encoder,",
            "p.encoder_name, p.encoder_codec, p.encoder_preset, p.encoder_tuning, p.encoder_rc,",
            "p.encoder_bpp, p.encoder_target_bps, p.encoder_res, p.encoder_res_width, p.encoder_res_height,",
            "p.encoder_fps, p.encoder_color, p.encoder_params_json,",
            "p.compression_name, p.compression_level,",
            "p.camera_model, p.camera_serial, p.camera_id, p.rig_id, p.arena_id, p.canvas_name,",
            "p.has_images_ds, p.has_images_ds_rgb, p.downsample_formats_json",
            "FROM datasets d",
            "LEFT JOIN provenance p ON d.dataset_id = p.dataset_id",
            "WHERE 1=1",
        ]
        params: List[Any] = []

        def add_clause(clause: str, value: Any) -> None:
            if value is None:
                return
            sql.append(clause)
            params.append(value)

        add_clause("AND p.dish_design = ?", dish_design)
        if dish_design_like:
            sql.append("AND p.dish_design LIKE ?")
            params.append(f"%{dish_design_like}%")
        add_clause("AND p.fish_id = ?", fish_id)
        add_clause("AND p.subject_count >= ?", subject_count_min)
        add_clause("AND p.subject_count <= ?", subject_count_max)
        add_clause("AND p.zarr_purpose = ?", zarr_purpose)
        add_clause("AND p.fps >= ?", fps_min)
        add_clause("AND p.fps <= ?", fps_max)
        add_clause("AND p.exposure >= ?", exposure_min)
        add_clause("AND p.exposure <= ?", exposure_max)
        add_clause("AND p.frame_rate >= ?", frame_rate_min)
        add_clause("AND p.frame_rate <= ?", frame_rate_max)
        add_clause("AND p.gain >= ?", gain_min)
        add_clause("AND p.gain <= ?", gain_max)
        add_clause("AND p.video_codec = ?", video_codec)
        add_clause("AND p.video_pix_fmt = ?", video_pix_fmt)
        add_clause("AND p.format_encoder = ?", format_encoder)
        add_clause("AND p.format_title = ?", format_title)
        add_clause("AND p.format_comment = ?", format_comment)
        add_clause("AND p.encoder_name = ?", encoder_name)
        add_clause("AND p.encoder_codec = ?", encoder_codec)
        add_clause("AND p.encoder_preset = ?", encoder_preset)
        add_clause("AND p.encoder_tuning = ?", encoder_tuning)
        add_clause("AND p.encoder_rc = ?", encoder_rc)
        add_clause("AND p.compression_name = ?", compression_name)
        add_clause("AND p.camera_model = ?", camera_model)
        add_clause("AND p.camera_serial = ?", camera_serial)
        add_clause("AND p.camera_id = ?", camera_id)
        add_clause("AND p.rig_id = ?", rig_id)
        add_clause("AND p.arena_id = ?", arena_id)
        if model_input is not None:
            mode = str(model_input).strip().lower()
            if mode == "gray":
                sql.append(
                    "AND (COALESCE(p.has_images_ds, 0) = 1 "
                    "OR p.downsample_formats_json LIKE '%\"gray\"%')"
                )
            elif mode == "rgb":
                sql.append(
                    "AND (COALESCE(p.has_images_ds_rgb, 0) = 1 "
                    "OR p.downsample_formats_json LIKE '%\"rgb\"%')"
                )
            else:
                raise ValueError(f"Unsupported model_input '{model_input}'. Expected 'gray' or 'rgb'.")
        if path_contains:
            sql.append("AND d.zarr_path LIKE ?")
            params.append(f"%{path_contains}%")

        sql.append("ORDER BY p.dish_design, p.fps, d.dataset_id")
        if limit is not None:
            sql.append("LIMIT ?")
            params.append(int(limit))

        query = " ".join(sql)
        return list(self.conn.execute(query, params).fetchall())


def scan_paths(
    registry: Registry,
    paths: Iterable[Path],
    *,
    recursive: bool = False,
) -> List[str]:
    normalized_paths = [Path(path).expanduser() for path in paths]
    dataset_ids: List[str] = []
    for path in normalized_paths:
        if path.is_dir() and _is_zarr_root(path):
            dataset_id = registry.scan_zarr(path)
            if dataset_id:
                dataset_ids.append(dataset_id)
            continue
        if path.is_dir() and recursive:
            for candidate in _find_zarr_roots(path):
                dataset_id = registry.scan_zarr(candidate)
                if dataset_id:
                    dataset_ids.append(dataset_id)
    registry.reconcile_missing_datasets(scope_paths=normalized_paths)
    return dataset_ids


def _is_zarr_root(path: Path) -> bool:
    return (path / "zarr.json").exists() or (path / ".zgroup").exists()


def _normalize_fs_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve()
    except Exception:
        return candidate.absolute()


def _path_matches_scope(candidate: Path, scope_roots: List[Path]) -> bool:
    for root in scope_roots:
        if candidate == root:
            return True
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _find_zarr_roots(root: Path) -> List[Path]:
    roots: List[Path] = []
    seen: set[Path] = set()
    for candidate in root.rglob("*.zarr"):
        if not candidate.is_dir():
            continue
        if not _is_zarr_root(candidate):
            continue
        resolved = _normalize_fs_path(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(candidate)
    roots.sort(key=lambda path: str(path))
    return roots
