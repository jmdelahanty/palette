#!/usr/bin/env python3
"""
Plan a recording re-organization by mapping H5 files to camera artifacts.

This script inspects H5 root attributes to derive camera IDs and pairs each
recording with its Cam<id>.mp4 and Cam<id>_meta.csv. Use --dry-run to print
an ASCII tree of the proposed folder layout.
"""

import argparse
import csv
import json
import os
import re
import sys
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import h5py

from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import normalize_attr as _normalize_attr
from fisheye.utils.recording_preflight import (
    PRECHECK_FAIL,
    PRECHECK_NOT_RUN,
    PRECHECK_PASS,
    PRECHECK_WARN,
    build_h5_preflight_payload,
    build_manifest_preflight_payload,
    build_video_preflight_payload,
    default_preflight_payload,
)

try:
    from fisheye.diagnostics.video.container import check_hevc_keyframe_flags
except ModuleNotFoundError:
    _THIS_DIR = Path(__file__).resolve().parent
    _SRC_DIR = _THIS_DIR.parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from fisheye.diagnostics.video.container import check_hevc_keyframe_flags


_utc_now = utc_now
JsonLogger = SharedJsonLogger
_PLACEHOLDER_METADATA_VALUES = {"unknown", "none", "null", "n/a", "na"}


@dataclass(frozen=True)
class PlannedFile:
    source: Path
    dest_name: str
    action: str = "move"


@dataclass
class RecordingPlan:
    name: str
    source_dir: Path
    dest_dir: Path
    raw_files: List[PlannedFile]
    cam_files: List[PlannedFile]
    derived_files: List[PlannedFile]
    camera_id: Optional[str]
    meta: Dict[str, Any] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    keyframe_checks: Dict[str, Dict[str, object]] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoDiagnosticsHookResult:
    manifest_payload: Dict[str, object]
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class H5DiagnosticsHookResult:
    manifest_payload: Dict[str, object]
    warnings: List[str] = field(default_factory=list)


def _derive_camera_id(ipc_source_name: object) -> Optional[str]:
    if ipc_source_name is None:
        return None
    text = _normalize_attr(ipc_source_name)
    if text is None:
        return None
    match = re.search(r"cam_(\d+)", text)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", text)
    return digits[-1] if digits else None


def _derive_camera_id_from_path(path: Path) -> Optional[str]:
    match = re.search(r"Cam(\d+)", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", path.stem)
    return digits[-1] if digits else None


def _sanitize_for_filename(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned)


def _set_meta_if_present(meta: Dict[str, Any], key: str, value: object) -> None:
    if meta.get(key):
        return
    normalized = _normalize_attr(value)
    if normalized and normalized.lower() in _PLACEHOLDER_METADATA_VALUES:
        return
    if normalized:
        meta[key] = normalized


def _read_h5_json_object(h5: h5py.File, path: str) -> Optional[Dict[str, Any]]:
    node = h5.get(path)
    if not isinstance(node, h5py.Dataset):
        return None
    try:
        raw_value = node[()]
    except Exception:
        return None
    if hasattr(raw_value, "item"):
        try:
            raw_value = raw_value.item()
        except Exception:
            pass
    text = _normalize_attr(raw_value)
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _dish_design_from_arena_config(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("selected_dish_type_name", "dish_name"):
        value = _normalize_attr(payload.get(key))
        if value:
            return value
    dish_config = payload.get("dish_config")
    if isinstance(dish_config, dict):
        for key in ("dish_name", "name"):
            value = _normalize_attr(dish_config.get(key))
            if value:
                return value
    return None


def _augment_h5_manifest_context(h5: h5py.File, meta: Dict[str, Any]) -> None:
    protocol_snapshot = h5.get("protocol_snapshot")
    if isinstance(protocol_snapshot, h5py.Group):
        _set_meta_if_present(
            meta,
            "protocol_name",
            protocol_snapshot.attrs.get("protocol_name"),
        )
        protocol_definition = _read_h5_json_object(h5, "protocol_snapshot/protocol_definition_json")
        if protocol_definition:
            _set_meta_if_present(
                meta,
                "protocol_name",
                protocol_definition.get("protocol_name"),
            )
            _set_meta_if_present(
                meta,
                "protocol_name_from_definition",
                protocol_definition.get("protocol_name"),
            )

    subject_metadata = h5.get("subject_metadata")
    if isinstance(subject_metadata, h5py.Group):
        _set_meta_if_present(meta, "genotype", subject_metadata.attrs.get("genotype"))
        _set_meta_if_present(
            meta,
            "dpf_at_acquisition",
            subject_metadata.attrs.get("days_post_fertilization"),
        )
        _set_meta_if_present(
            meta,
            "dpf_at_acquisition",
            subject_metadata.attrs.get("dpf_at_acquisition"),
        )

    arena_config = _read_h5_json_object(h5, "calibration_snapshot/arena_config_json")
    if arena_config:
        _set_meta_if_present(meta, "dish_design", _dish_design_from_arena_config(arena_config))


def _read_camera_context(h5_path: Path) -> Tuple[Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    try:
        with h5py.File(h5_path, "r") as h5:
            root = h5.attrs
            keys = (
                "session_uuid",
                "session_start_iso8601_utc",
                "rig_id",
                "arena_id",
                "camera_id",
                "canvas_name",
                "protocol_name_from_definition",
                "loaded_protocol_filepath",
                "stimulus_output_width",
                "stimulus_output_height",
                "ipc_source_name",
                "active_ipc_source",
                "hostname",
                "software_version",
                "protocol_name",
                "dish_design",
                "genotype",
                "dpf_at_acquisition",
                "num_dishes",
                "fish_per_dish",
            )
            for key in keys:
                if key in root:
                    _set_meta_if_present(meta, key, root.get(key))
            _augment_h5_manifest_context(h5, meta)
            camera_id = meta.get("camera_id")
            if not camera_id:
                derived = _derive_camera_id(meta.get("ipc_source_name"))
                if derived:
                    meta["camera_id"] = derived
                    meta["camera_id_source"] = "ipc_source_name"
                camera_id = derived
            return camera_id, meta
    except Exception as exc:
        meta["error"] = f"failed to read H5: {exc}"
        return None, meta


def _find_h5_files(source: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(source.rglob("*.h5"))
    return sorted(source.glob("*.h5"))


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _unique_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _cam_file_candidates(camera_id: str, root: Path) -> Tuple[Path, Path]:
    return root / f"Cam{camera_id}.mp4", root / f"Cam{camera_id}_meta.csv"


def _extend_cam_roots(
    roots: List[Path],
    h5_path: Path,
    camera_id: str,
) -> List[Path]:
    cam_roots = _unique_paths(roots)
    has_cam = False
    for root in cam_roots:
        cam_mp4, cam_meta = _cam_file_candidates(camera_id, root)
        if cam_mp4.exists() or cam_meta.exists():
            has_cam = True
            break

    if has_cam:
        return cam_roots

    for ancestor in h5_path.parents:
        if ancestor in cam_roots:
            continue
        cam_mp4, cam_meta = _cam_file_candidates(camera_id, ancestor)
        if cam_mp4.exists() or cam_meta.exists():
            cam_roots.append(ancestor)
            break

    return _unique_paths(cam_roots)


def _unique_planned(files: List[PlannedFile]) -> List[PlannedFile]:
    seen: Set[Tuple[Path, str, str]] = set()
    unique: List[PlannedFile] = []
    for planned in files:
        key = (planned.source, planned.dest_name, planned.action)
        if key in seen:
            continue
        seen.add(key)
        unique.append(planned)
    return unique


def _resolve_video_only_source_path(
    raw_value: str,
    *,
    source_root: Path,
    metadata_csv_path: Path,
) -> Path:
    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    source_candidate = (source_root / candidate).resolve()
    if source_candidate.exists():
        return source_candidate

    csv_candidate = (metadata_csv_path.parent / candidate).resolve()
    if csv_candidate.exists():
        return csv_candidate

    return source_candidate


def _load_video_only_rows(
    metadata_csv_path: Path,
    *,
    source_root: Path,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with metadata_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            row = {
                str(key).strip(): str(value).strip()
                for key, value in raw_row.items()
                if key is not None and value is not None and str(value).strip()
            }
            if not row:
                continue

            source_video_raw = (
                row.get("source_video")
                or row.get("video_path")
                or row.get("camera_video")
            )
            if not source_video_raw:
                raise ValueError(
                    f"Metadata CSV row is missing source_video/video_path/camera_video: {raw_row}"
                )
            source_video = _resolve_video_only_source_path(
                source_video_raw,
                source_root=source_root,
                metadata_csv_path=metadata_csv_path,
            )
            row["source_video"] = str(source_video)

            optional_camera_csv = (
                row.get("source_camera_metadata_csv")
                or row.get("camera_metadata_csv")
            )
            if optional_camera_csv:
                resolved_csv = _resolve_video_only_source_path(
                    optional_camera_csv,
                    source_root=source_root,
                    metadata_csv_path=metadata_csv_path,
                )
                row["source_camera_metadata_csv"] = str(resolved_csv)

            rows.append(row)
    return rows


def _build_video_only_plan(
    row: Dict[str, str],
    *,
    dest_root: Path,
    rename_cams: bool,
) -> RecordingPlan:
    video_path = Path(row["source_video"]).expanduser().resolve()
    camera_id = row.get("camera_id") or _derive_camera_id_from_path(video_path)
    session_uuid = row.get("session_uuid") or row.get("recording_id") or video_path.stem
    recording_name = row.get("recording_name") or session_uuid or video_path.stem
    folder_name = _sanitize_for_filename(recording_name)
    dest_dir = dest_root / folder_name

    session_tag = _sanitize_for_filename(session_uuid)
    if rename_cams and camera_id:
        cam_base = f"Cam{camera_id}_{session_tag}"
        video_dest_name = f"{cam_base}{video_path.suffix.lower() or '.mp4'}"
    else:
        video_dest_name = video_path.name
    cam_files = [PlannedFile(video_path, video_dest_name)]
    optional_camera_csv = row.get("source_camera_metadata_csv")
    if optional_camera_csv:
        camera_csv_path = Path(optional_camera_csv).expanduser().resolve()
        if rename_cams and camera_id:
            cam_files.append(PlannedFile(camera_csv_path, f"Cam{camera_id}_{session_tag}_meta.csv"))
        else:
            cam_files.append(PlannedFile(camera_csv_path, camera_csv_path.name))

    raw_files: List[PlannedFile] = []
    derived_files: List[PlannedFile] = []
    if camera_id:
        keyframe_sidecar = video_path.with_name(f"Cam{camera_id}_keyframe.json")
        if keyframe_sidecar.exists():
            if rename_cams:
                dest_name = f"Cam{camera_id}_{session_tag}_keyframe.json"
            else:
                dest_name = keyframe_sidecar.name
            cam_files.append(PlannedFile(keyframe_sidecar, dest_name))

        for suffix in ("pipeline_perf.csv", "acquisition_cadence_probe.csv"):
            sidecar = video_path.with_name(f"Cam{camera_id}_{suffix}")
            if sidecar.exists():
                if rename_cams:
                    dest_name = f"Cam{camera_id}_{session_tag}_{suffix}"
                else:
                    dest_name = sidecar.name
                derived_files.append(PlannedFile(sidecar, dest_name))

    for shared_name, dest_name in (
        ("ptp_sync_summary.json", "ptp_sync_summary.json"),
        ("recording_snapshot.json", "recording_snapshot_runtime.json"),
        ("recording_snapshot", "recording_snapshot_runtime.json"),
    ):
        shared_path = video_path.parent / shared_name
        if shared_path.exists():
            raw_files.append(PlannedFile(shared_path, dest_name, action="copy"))
            if shared_name.startswith("recording_snapshot"):
                break

    meta: Dict[str, str] = {
        "session_uuid": session_uuid,
        "recording_id": row.get("recording_id") or session_uuid,
        "recording_name": recording_name,
        "recording_type": row.get("recording_type") or "behavior",
        "recording_subtype": row.get("recording_subtype") or "free",
        "behavior_mode": row.get("behavior_mode") or "free",
        "artifact_schema_id": row.get("artifact_schema_id") or "video_only_v1",
    }
    for key in (
        "session_start_iso8601_utc",
        "dish_design",
        "rig_id",
        "arena_id",
        "camera_id",
        "canvas_name",
        "protocol_name",
        "protocol_name_from_definition",
        "genotype",
        "dpf_at_acquisition",
        "num_dishes",
        "fish_per_dish",
    ):
        value = row.get(key)
        if value:
            meta[key] = value
    if "protocol_name_from_definition" not in meta and row.get("protocol_name"):
        meta["protocol_name_from_definition"] = row["protocol_name"]
    if camera_id and "camera_id" not in meta:
        meta["camera_id"] = str(camera_id)

    missing: List[str] = []
    if not video_path.exists():
        missing.append(video_path.name)
    if not row.get("dish_design"):
        missing.append("dish_design (missing in metadata CSV)")

    return RecordingPlan(
        name=folder_name,
        source_dir=video_path.parent,
        dest_dir=dest_dir,
        raw_files=_unique_planned(raw_files),
        cam_files=_unique_planned(cam_files),
        derived_files=_unique_planned(derived_files),
        camera_id=str(camera_id) if camera_id else None,
        meta=meta,
        missing=missing,
    )


def _load_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _runtime_snapshot_software_version(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        payload = _load_json_object(path)
    except Exception:
        return None
    source_version = payload.get("source_version")
    if isinstance(source_version, dict):
        for key in ("describe", "commit_short", "commit"):
            value = _normalize_attr(source_version.get(key))
            if value and value.lower() not in _PLACEHOLDER_METADATA_VALUES:
                return value
    value = _normalize_attr(payload.get("producer_version"))
    if value and value.lower() not in _PLACEHOLDER_METADATA_VALUES:
        return value
    return None


def _looks_like_external_ipc_batch(source: Path) -> bool:
    session_path = source / "recording_session.json"
    if not session_path.exists():
        return False
    try:
        payload = _load_json_object(session_path)
    except Exception:
        return False
    producer = str(payload.get("producer") or "")
    backend = str(payload.get("recording_backend") or "")
    outputs = payload.get("recording_outputs")
    return (
        isinstance(outputs, dict)
        and bool(outputs)
        and ("external_ipc" in producer or "external_ipc" in backend)
    )


def _resolve_external_ipc_path(
    raw_value: object,
    *,
    batch_root: Path,
    preferred_dir: Optional[Path] = None,
) -> Optional[Path]:
    text = _normalize_attr(raw_value)
    if not text:
        return None
    raw_path = Path(text).expanduser()
    candidates: List[Path] = []

    def add(candidate: Path) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    if raw_path.is_absolute():
        add(raw_path)
    else:
        if preferred_dir is not None:
            add(preferred_dir / raw_path)
        add(batch_root / raw_path)

    # Orange manifests are often authored on the acquisition host with absolute
    # paths. After transfer, the batch root name is stable, so remap anything
    # below that path segment back under the local staging batch root.
    parts = raw_path.parts
    if batch_root.name in parts:
        idx = parts.index(batch_root.name)
        rel_parts = parts[idx + 1 :]
        if rel_parts:
            add(batch_root.joinpath(*rel_parts))
    for marker in ("external_recorder", "external_crop_recorder", "citrus"):
        if marker in parts:
            idx = parts.index(marker)
            add(batch_root.joinpath(*parts[idx:]))

    if raw_path.name:
        if preferred_dir is not None:
            add(preferred_dir / raw_path.name)
        add(batch_root / raw_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if candidates:
        return candidates[0].resolve()
    return None


def _external_ipc_output_for_camera(
    session: Dict[str, Any],
    camera_id: Optional[str],
) -> Dict[str, Any]:
    if not camera_id:
        return {}
    outputs = session.get("recording_outputs")
    if not isinstance(outputs, dict):
        return {}
    payload = outputs.get(str(camera_id))
    return payload if isinstance(payload, dict) else {}


def _dict_or_empty(value: object) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _pick_stream_value(output: Dict[str, Any], key: str) -> object:
    details = _dict_or_empty(output.get("details"))
    if key in output:
        return output.get(key)
    return details.get(key)


def _drop_none_values(payload: Dict[str, object]) -> Dict[str, object]:
    cleaned: Dict[str, object] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, dict):
            nested = _drop_none_values(value)
            if nested:
                cleaned[key] = nested
            continue
        cleaned[key] = value
    return cleaned


def _external_ipc_video_streams_payload(
    *,
    camera_id: str,
    cam_base: str,
    full_output: Dict[str, Any],
    crop_output: Dict[str, Any],
) -> Dict[str, object]:
    full_stream = _drop_none_values(
        {
            "role": "ingest_authoritative_full_frame",
            "output_kind": "full",
            "source": "orange_external_ipc",
            "camera_id": camera_id,
            "stream_id": _pick_stream_value(full_output, "stream_id"),
            "orange_declared_role": _pick_stream_value(full_output, "role"),
            "video": f"cams/{cam_base}.mp4",
            "frame_clock_metadata": f"cams/{cam_base}_meta.csv",
            "keyframes": f"cams/{cam_base}_keyframe.json",
            "summary": f"cams/{cam_base}_external_summary.json",
            "frame_clock": "recording_frame_id",
            "coordinate_space": _pick_stream_value(full_output, "coordinate_space"),
            "width": _pick_stream_value(full_output, "width"),
            "height": _pick_stream_value(full_output, "height"),
            "frame_count": _pick_stream_value(full_output, "frame_count"),
            "frame_rate": _pick_stream_value(full_output, "frame_rate"),
            "codec": _pick_stream_value(full_output, "codec"),
            "container": _pick_stream_value(full_output, "container"),
            "encoded_format": _pick_stream_value(full_output, "encoded_format"),
            "pixel_source_format": _pick_stream_value(full_output, "pixel_source_format"),
        }
    )
    streams: Dict[str, object] = {"full": full_stream}

    if crop_output:
        crop_stream = _drop_none_values(
            {
                "role": "runtime_derived_acquisition_input",
                "output_kind": "crop",
                "source": "orange_external_ipc",
                "camera_id": camera_id,
                "stream_id": _pick_stream_value(crop_output, "stream_id"),
                "orange_declared_role": _pick_stream_value(crop_output, "role"),
                "video": f"derived/external_crop_recorder/{cam_base}_crop_external.mp4",
                "metadata": f"derived/external_crop_recorder/{cam_base}_crop_meta.csv",
                "keyframes": (
                    f"derived/external_crop_recorder/"
                    f"{cam_base}_crop_external_keyframe.json"
                ),
                "summary": (
                    f"derived/external_crop_recorder/"
                    f"{cam_base}_crop_external_summary.json"
                ),
                "frame_clock": "recording_frame_id",
                "video_pixel_coordinate_space": "crop_frame_pixels",
                "source_geometry_coordinate_space": (
                    _pick_stream_value(crop_output, "coordinate_space")
                    or "full_frame_pixels"
                ),
                "geometry_columns": [
                    "crop_x",
                    "crop_y",
                    "crop_w",
                    "crop_h",
                    "detection_x",
                    "detection_y",
                    "detection_w",
                    "detection_h",
                ],
                "blank_frame_policy": _pick_stream_value(crop_output, "blank_frame_policy"),
                "selection_policy": _pick_stream_value(crop_output, "selection_policy"),
                "width": _pick_stream_value(crop_output, "width"),
                "height": _pick_stream_value(crop_output, "height"),
                "frame_count": _pick_stream_value(crop_output, "frame_count"),
                "frame_rate": _pick_stream_value(crop_output, "frame_rate"),
                "codec": _pick_stream_value(crop_output, "codec"),
                "container": _pick_stream_value(crop_output, "container"),
                "encoded_format": _pick_stream_value(crop_output, "encoded_format"),
                "pixel_source_format": _pick_stream_value(crop_output, "pixel_source_format"),
            }
        )
        streams["crop"] = crop_stream

    return {
        "schema_id": "orange_runtime_video_streams_v1",
        "frame_clock": "recording_frame_id",
        "streams": streams,
    }


def _append_planned_if_present(
    files: List[PlannedFile],
    source: Optional[Path],
    dest_name: str,
    *,
    missing: List[str],
    required: bool = False,
    action: str = "move",
    missing_label: Optional[str] = None,
) -> None:
    if source is not None and source.exists():
        files.append(PlannedFile(source, dest_name, action=action))
        return
    if required:
        missing.append(missing_label or dest_name)


def _build_external_ipc_plan(
    h5_path: Path,
    *,
    batch_root: Path,
    session: Dict[str, Any],
    dest_root: Path,
    rename_cams: bool,
) -> RecordingPlan:
    camera_id, meta = _read_camera_context(h5_path)
    name = h5_path.stem
    dest_dir = dest_root / name
    raw_files: List[PlannedFile] = [PlannedFile(h5_path, h5_path.name)]
    cam_files: List[PlannedFile] = []
    derived_files: List[PlannedFile] = []
    missing: List[str] = []

    rendered = h5_path.with_suffix(".mp4")
    if rendered.exists():
        raw_files.append(PlannedFile(rendered, rendered.name))
    else:
        missing.append(rendered.name)

    update_timing = h5_path.with_name(f"{h5_path.stem}_update_timing.csv")
    if update_timing.exists():
        raw_files.append(PlannedFile(update_timing, update_timing.name))

    for shared_name, dest_name in (
        ("recording_session.json", "recording_session.json"),
        ("recording_snapshot.json", "recording_snapshot_runtime.json"),
        ("ptp_sync_summary.json", "ptp_sync_summary.json"),
        ("_citrus_transfer_complete.json", "transfer_complete.json"),
        ("orange_local_control.events.jsonl", "orange_local_control.events.jsonl"),
        ("external_recorder_contract.json", "external_recorder_contract.json"),
        ("external_crop_recorder_contract.json", "external_crop_recorder_contract.json"),
        ("external_recorder_supervisor_plan.json", "external_recorder_supervisor_plan.json"),
        ("external_crop_recorder_supervisor_plan.json", "external_crop_recorder_supervisor_plan.json"),
    ):
        shared_path = batch_root / shared_name
        if shared_path.exists():
            raw_files.append(PlannedFile(shared_path, dest_name, action="copy"))

    threading_candidates = list(batch_root.glob("*threading_startup*.json"))
    citrus_dir = batch_root / "citrus"
    if citrus_dir.exists():
        threading_candidates.extend(citrus_dir.glob("*threading_startup*.json"))
    for threading_path in sorted(threading_candidates):
        derived_files.append(
            PlannedFile(
                threading_path,
                f"citrus/{threading_path.name}",
                action="copy",
            )
        )

    session_id = str(session.get("session_id") or batch_root.name)
    meta.setdefault("session_uuid", _choose_session_tag(meta, h5_path))
    meta.setdefault("recording_id", meta.get("session_uuid") or name)
    meta.setdefault("recording_name", name)
    meta.setdefault("recording_type", "behavior")
    meta.setdefault("recording_subtype", "free")
    meta.setdefault("behavior_mode", "free")
    meta["artifact_schema_id"] = "orange_external_ipc_single_clip_v1"
    meta["recording_backend"] = "external_ipc"
    meta["orange_session_id"] = session_id
    meta["orange_producer"] = str(session.get("producer") or "")
    meta["orange_recording_mode"] = str(session.get("mode") or "")
    _set_meta_if_present(
        meta,
        "software_version",
        _runtime_snapshot_software_version(batch_root / "recording_snapshot.json"),
    )

    if not camera_id:
        missing.append("camera_id (missing in H5 attrs)")
        return RecordingPlan(
            name=name,
            source_dir=batch_root,
            dest_dir=dest_dir,
            raw_files=_unique_planned(raw_files),
            cam_files=[],
            derived_files=_unique_planned(derived_files),
            camera_id=None,
            meta=meta,
            missing=missing,
        )

    outputs = _external_ipc_output_for_camera(session, camera_id)
    full_output = outputs.get("full") if isinstance(outputs.get("full"), dict) else {}
    crop_output = outputs.get("crop") if isinstance(outputs.get("crop"), dict) else {}
    full_dir = batch_root / "external_recorder"
    crop_dir = batch_root / "external_crop_recorder"
    session_tag = _sanitize_for_filename(_choose_session_tag(meta, h5_path))
    cam_base = f"Cam{camera_id}_{session_tag}" if rename_cams else f"Cam{camera_id}"
    meta["video_streams"] = _external_ipc_video_streams_payload(
        camera_id=camera_id,
        cam_base=cam_base,
        full_output=full_output,
        crop_output=crop_output,
    )

    if not full_output:
        missing.append(f"recording_outputs/{camera_id}/full")
    full_video = _resolve_external_ipc_path(
        full_output.get("video") if full_output else None,
        batch_root=batch_root,
        preferred_dir=full_dir,
    )
    full_summary = _resolve_external_ipc_path(
        full_output.get("metadata") if full_output else None,
        batch_root=batch_root,
        preferred_dir=full_dir,
    )
    full_keyframes = _resolve_external_ipc_path(
        full_output.get("keyframes") if full_output else None,
        batch_root=batch_root,
        preferred_dir=full_dir,
    )
    crop_meta = _resolve_external_ipc_path(
        crop_output.get("metadata") if crop_output else None,
        batch_root=batch_root,
        preferred_dir=crop_dir,
    )

    _append_planned_if_present(
        cam_files,
        full_video,
        f"{cam_base}.mp4",
        missing=missing,
        required=True,
        missing_label=f"external_recorder/Cam{camera_id}_external.mp4",
    )
    _append_planned_if_present(
        cam_files,
        crop_meta,
        f"{cam_base}_meta.csv",
        missing=missing,
        required=True,
        action="copy",
        missing_label=f"Cam{camera_id}_crop_meta.csv (compatibility camera metadata)",
    )
    _append_planned_if_present(
        cam_files,
        full_keyframes,
        f"{cam_base}_keyframe.json",
        missing=missing,
        required=True,
        missing_label=f"external_recorder/Cam{camera_id}_external_keyframes.json",
    )
    _append_planned_if_present(
        cam_files,
        full_summary,
        f"{cam_base}_external_summary.json",
        missing=missing,
        required=True,
        missing_label=f"external_recorder/Cam{camera_id}_external_summary.json",
    )

    for suffix in ("detach.csv", "gop_routing.csv", "status.json", "recorder.log"):
        source = full_dir / f"Cam{camera_id}_external_{suffix}"
        _append_planned_if_present(
            derived_files,
            source,
            f"external_recorder/{cam_base}_external_{suffix}",
            missing=missing,
            required=False,
        )
    for shared_name in (
        "external_recorder_finalization.json",
        "external_recorder_session.json",
        "external_recorder_supervisor_plan.json",
        "external_recorder_supervisor_runtime.json",
        "external_recorder_verifier_handoff.json",
    ):
        source = full_dir / shared_name
        _append_planned_if_present(
            derived_files,
            source,
            f"external_recorder/{shared_name}",
            missing=missing,
            required=False,
            action="copy",
        )

    if crop_output:
        crop_video = _resolve_external_ipc_path(
            crop_output.get("video"),
            batch_root=batch_root,
            preferred_dir=crop_dir,
        )
        crop_keyframe = _resolve_external_ipc_path(
            crop_output.get("keyframes"),
            batch_root=batch_root,
            preferred_dir=crop_dir,
        )
        crop_summary = _resolve_external_ipc_path(
            crop_output.get("summary"),
            batch_root=batch_root,
            preferred_dir=crop_dir,
        )
        crop_perf = _resolve_external_ipc_path(
            crop_output.get("perf"),
            batch_root=batch_root,
            preferred_dir=batch_root,
        )
        crop_sidecar_perf = _resolve_external_ipc_path(
            crop_output.get("sidecar_perf"),
            batch_root=batch_root,
            preferred_dir=batch_root,
        )
        _append_planned_if_present(
            derived_files,
            crop_video,
            f"external_crop_recorder/{cam_base}_crop_external.mp4",
            missing=missing,
            required=True,
            missing_label=f"external_crop_recorder/Cam{camera_id}_crop_external.mp4",
        )
        _append_planned_if_present(
            derived_files,
            crop_meta,
            f"external_crop_recorder/{cam_base}_crop_meta.csv",
            missing=missing,
            required=True,
            missing_label=f"Cam{camera_id}_crop_meta.csv",
        )
        _append_planned_if_present(
            derived_files,
            crop_keyframe,
            f"external_crop_recorder/{cam_base}_crop_external_keyframe.json",
            missing=missing,
            required=True,
            missing_label=f"external_crop_recorder/Cam{camera_id}_crop_external_keyframe.json",
        )
        _append_planned_if_present(
            derived_files,
            crop_summary,
            f"external_crop_recorder/{cam_base}_crop_external_summary.json",
            missing=missing,
            required=True,
            missing_label=f"external_crop_recorder/Cam{camera_id}_crop_external_summary.json",
        )
        _append_planned_if_present(
            derived_files,
            crop_perf,
            f"external_crop_recorder/{cam_base}_crop_perf.csv",
            missing=missing,
            required=False,
        )
        _append_planned_if_present(
            derived_files,
            crop_sidecar_perf,
            f"external_crop_recorder/{cam_base}_crop_sidecar_perf.csv",
            missing=missing,
            required=False,
        )
        for suffix in (
            "detach.csv",
            "encode.csv",
            "gop_routing.csv",
            "status.json",
            "recorder.log",
        ):
            source = crop_dir / f"Cam{camera_id}_crop_external_{suffix}"
            _append_planned_if_present(
                derived_files,
                source,
                f"external_crop_recorder/{cam_base}_crop_external_{suffix}",
                missing=missing,
                required=False,
            )
        for suffix in ("yolo_perf.csv", "yolo_events.jsonl"):
            source = batch_root / f"Cam{camera_id}_{suffix}"
            _append_planned_if_present(
                derived_files,
                source,
                f"external_crop_recorder/{cam_base}_{suffix}",
                missing=missing,
                required=False,
            )
        for shared_name in (
            "external_recorder_finalization.json",
            "external_recorder_session.json",
            "external_recorder_supervisor_plan.json",
            "external_recorder_supervisor_runtime.json",
            "external_recorder_verifier_handoff.json",
        ):
            source = crop_dir / shared_name
            _append_planned_if_present(
                derived_files,
                source,
                f"external_crop_recorder/{shared_name}",
                missing=missing,
                required=False,
                action="copy",
            )

    for suffix in ("pipeline_perf.csv", "acquisition_cadence_probe.csv"):
        source = batch_root / f"Cam{camera_id}_{suffix}"
        _append_planned_if_present(
            derived_files,
            source,
            f"external_ipc/{cam_base}_{suffix}",
            missing=missing,
            required=False,
        )

    return RecordingPlan(
        name=name,
        source_dir=batch_root,
        dest_dir=dest_dir,
        raw_files=_unique_planned(raw_files),
        cam_files=_unique_planned(cam_files),
        derived_files=_unique_planned(derived_files),
        camera_id=camera_id,
        meta=meta,
        missing=missing,
    )


def _build_external_ipc_plans(
    source_root: Path,
    *,
    dest_root: Path,
    rename_cams: bool,
) -> List[RecordingPlan]:
    session_path = source_root / "recording_session.json"
    session = _load_json_object(session_path)
    h5_files = _find_h5_files(source_root, recursive=True)
    return [
        _build_external_ipc_plan(
            h5_path,
            batch_root=source_root,
            session=session,
            dest_root=dest_root,
            rename_cams=rename_cams,
        )
        for h5_path in h5_files
    ]


def _choose_session_tag(meta: Dict[str, Any], h5_path: Path) -> str:
    return meta.get("session_uuid") or meta.get("session_start_iso8601_utc") or h5_path.stem


def _build_plan(
    h5_path: Path,
    dest_root: Path,
    cam_root: Optional[Path],
    rename_cams: bool,
) -> RecordingPlan:
    camera_id, meta = _read_camera_context(h5_path)
    name = h5_path.stem
    dest_dir = dest_root / name
    raw_files: List[PlannedFile] = [PlannedFile(h5_path, h5_path.name)]
    cam_files: List[PlannedFile] = []
    derived_files: List[PlannedFile] = []
    missing: List[str] = []

    rendered = h5_path.with_suffix(".mp4")
    if rendered.exists():
        raw_files.append(PlannedFile(rendered, rendered.name))
    else:
        missing.append(rendered.name)

    update_timing = h5_path.with_name(f"{h5_path.stem}_update_timing.csv")
    if update_timing.exists():
        raw_files.append(PlannedFile(update_timing, update_timing.name))

    # Move the full recording_snapshot.json into raw/ as
    # recording_snapshot_runtime.json — the unfiltered original from Citrus,
    # preserved alongside the H5 as a recovery backup. The per-camera
    # filtered version is written separately by --snapshot / _write_snapshot
    # into derived/recording_snapshot.json.
    for snapshot_name in ("recording_snapshot.json", "recording_snapshot"):
        snapshot_candidate = h5_path.parent / snapshot_name
        if snapshot_candidate.exists():
            raw_files.append(PlannedFile(snapshot_candidate, "recording_snapshot_runtime.json"))
            break

    if camera_id:
        search_roots = [h5_path.parent]
        if cam_root is not None:
            search_roots.append(cam_root)
        search_roots = _extend_cam_roots(search_roots, h5_path, str(camera_id))

        cam_mp4 = _first_existing(
            [root / f"Cam{camera_id}.mp4" for root in search_roots]
        )
        cam_meta = _first_existing(
            [root / f"Cam{camera_id}_meta.csv" for root in search_roots]
        )
        session_tag = _sanitize_for_filename(_choose_session_tag(meta, h5_path))
        cam_base = f"Cam{camera_id}_{session_tag}" if rename_cams else f"Cam{camera_id}"

        if cam_mp4:
            dest_name = f"{cam_base}.mp4"
            cam_files.append(PlannedFile(cam_mp4, dest_name))
        else:
            missing.append(f"Cam{camera_id}.mp4")
        if cam_meta:
            dest_name = f"{cam_base}_meta.csv"
            cam_files.append(PlannedFile(cam_meta, dest_name))
        else:
            missing.append(f"Cam{camera_id}_meta.csv")

        derived_patterns = [
            h5_path.parent / f"extracted_{camera_id}_*_image.png",
            h5_path.parent / f"extracted_{camera_id}_*.png",
        ]
        for pattern in derived_patterns:
            derived_files.extend(
                PlannedFile(path, path.name) for path in sorted(pattern.parent.glob(pattern.name))
            )
    else:
        missing.append("camera_id (missing in H5 attrs)")

    return RecordingPlan(
        name=name,
        source_dir=h5_path.parent,
        dest_dir=dest_dir,
        raw_files=raw_files,
        cam_files=cam_files,
        derived_files=_unique_planned(derived_files),
        camera_id=camera_id,
        meta=meta,
        missing=missing,
    )


def _format_recording_summary(plan: RecordingPlan) -> List[str]:
    lines = [f"Recording: {plan.name}"]
    camera_label = plan.camera_id or "unknown"
    lines.append(f"  camera_id: {camera_label}")
    if plan.cam_files:
        cam_names = ", ".join(
            f"{file.source.name} -> {file.dest_name}" if file.source.name != file.dest_name else file.source.name
            for file in plan.cam_files
        )
        lines.append(f"  cam files: {cam_names}")
    if plan.raw_files:
        raw_names = ", ".join(file.dest_name for file in plan.raw_files)
        lines.append(f"  raw files: {raw_names}")
    if plan.derived_files:
        derived_names = ", ".join(file.dest_name for file in plan.derived_files)
        lines.append(f"  derived files: {derived_names}")
    if plan.missing:
        missing = ", ".join(plan.missing)
        lines.append(f"  missing: {missing}")
    if "error" in plan.meta:
        lines.append(f"  error: {plan.meta['error']}")
    return lines


def _append_folder(lines: List[str], prefix: str, name: str, is_last: bool) -> str:
    connector = "`-- " if is_last else "|-- "
    lines.append(f"{prefix}{connector}{name}/")
    return prefix + ("    " if is_last else "|   ")


def _append_files(lines: List[str], prefix: str, files: List[PlannedFile]) -> None:
    for idx, planned in enumerate(files):
        is_last = idx == len(files) - 1
        connector = "`-- " if is_last else "|-- "
        lines.append(f"{prefix}{connector}{planned.dest_name}")


def _render_tree(root_label: str, plans: List[RecordingPlan]) -> List[str]:
    lines = [root_label]
    for idx, plan in enumerate(plans):
        is_last_plan = idx == len(plans) - 1
        child_prefix = _append_folder(lines, "", plan.name, is_last_plan)

        subfolders = [
            ("raw", plan.raw_files),
            ("cams", plan.cam_files),
            ("zarr", []),
            ("derived", plan.derived_files),
        ]
        for sub_idx, (folder_name, files) in enumerate(subfolders):
            is_last_sub = sub_idx == len(subfolders) - 1
            sub_prefix = _append_folder(lines, child_prefix, folder_name, is_last_sub)
            if files:
                _append_files(lines, sub_prefix, files)
    return lines


def _write_manifest(
    plan: RecordingPlan,
    organized_utc: str,
    run_id: str,
    log_path: Optional[Path],
) -> Optional[str]:
    manifest_path = plan.dest_dir / "recording_manifest.json"
    if manifest_path.exists():
        return f"Manifest exists, skipping: {manifest_path}"

    files = {
        "raw": [f"raw/{file.dest_name}" for file in plan.raw_files],
        "cams": [f"cams/{file.dest_name}" for file in plan.cam_files],
        "derived": [f"derived/{file.dest_name}" for file in plan.derived_files],
    }
    snapshot_path = plan.dest_dir / "derived" / "recording_snapshot.json"
    payload = {
        "recording_name": plan.meta.get("recording_name") or plan.name,
        "organized_utc": organized_utc,
        "organize_run_id": run_id,
        "organize_log": str(log_path) if log_path else None,
        "session_uuid": plan.meta.get("session_uuid"),
        "recording_id": plan.meta.get("recording_id"),
        "session_start_iso8601_utc": plan.meta.get("session_start_iso8601_utc"),
        "recording_type": plan.meta.get("recording_type"),
        "recording_subtype": plan.meta.get("recording_subtype"),
        "behavior_mode": plan.meta.get("behavior_mode"),
        "artifact_schema_id": plan.meta.get("artifact_schema_id"),
        "rig_id": plan.meta.get("rig_id"),
        "arena_id": plan.meta.get("arena_id"),
        "camera_id": plan.camera_id,
        "canvas_name": plan.meta.get("canvas_name"),
        "protocol_name_from_definition": (
            plan.meta.get("protocol_name_from_definition")
            or plan.meta.get("protocol_name")
        ),
        "protocol_name": plan.meta.get("protocol_name"),
        "dish_design": plan.meta.get("dish_design"),
        "genotype": plan.meta.get("genotype"),
        "dpf_at_acquisition": plan.meta.get("dpf_at_acquisition"),
        "num_dishes": plan.meta.get("num_dishes"),
        "fish_per_dish": plan.meta.get("fish_per_dish"),
        "loaded_protocol_filepath": plan.meta.get("loaded_protocol_filepath"),
        "ipc_source_name": plan.meta.get("ipc_source_name"),
        "active_ipc_source": plan.meta.get("active_ipc_source"),
        "hostname": plan.meta.get("hostname"),
        "software_version": plan.meta.get("software_version"),
        "stimulus_output_width": plan.meta.get("stimulus_output_width"),
        "stimulus_output_height": plan.meta.get("stimulus_output_height"),
        "camera_id_source": plan.meta.get("camera_id_source"),
        "recording_backend": plan.meta.get("recording_backend"),
        "orange_session_id": plan.meta.get("orange_session_id"),
        "orange_producer": plan.meta.get("orange_producer"),
        "orange_recording_mode": plan.meta.get("orange_recording_mode"),
        "video_streams": plan.meta.get("video_streams"),
        "source_dir": str(plan.source_dir),
        "files": files,
        "hevc_keyframe_flags": plan.keyframe_checks if plan.keyframe_checks else None,
        "recording_snapshot": f"derived/{snapshot_path.name}" if snapshot_path.exists() else None,
        "preflight": default_preflight_payload(),
    }
    try:
        plan.dest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        return f"Failed to write manifest for {plan.name}: {exc}"
    return None


def _build_snapshot_payload(
    snapshot: Dict[str, object],
    camera_id: Optional[str],
    mode: str,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    if mode == "copy":
        return snapshot, None
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, dict):
        return None, "Snapshot missing cameras map; use --snapshot-mode copy."
    if not camera_id:
        return None, "Missing camera_id; cannot split snapshot."
    camera_payload = cameras.get(str(camera_id))
    if camera_payload is None:
        return None, f"Snapshot has no entry for camera_id {camera_id}."
    filtered = {k: v for k, v in snapshot.items() if k != "cameras"}
    filtered["cameras"] = {str(camera_id): camera_payload}
    return filtered, None


def _write_snapshot(
    plan: RecordingPlan,
    snapshot: Dict[str, object],
    mode: str,
) -> Optional[str]:
    payload, warning = _build_snapshot_payload(snapshot, plan.camera_id, mode)
    if warning:
        return warning
    if payload is None:
        return "Snapshot payload is empty."
    dest_dir = plan.dest_dir / "derived"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "recording_snapshot.json"
    if dest_path.exists():
        return f"Snapshot exists, skipping: {dest_path}"
    try:
        dest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        return f"Failed to write snapshot for {plan.name}: {exc}"
    return None


def _apply_plan(
    plans: List[RecordingPlan],
    create_empty: bool,
    write_manifest: bool,
    snapshot: Optional[Dict[str, object]],
    snapshot_mode: str,
    logger: Optional[JsonLogger],
    run_id: str,
    log_path: Optional[Path],
) -> List[str]:
    warnings: List[str] = []
    moved: Set[Path] = set()
    planned_destinations: Set[Path] = set()

    def record_video_keyframe_status(
        *,
        plan: RecordingPlan,
        folder_name: str,
        planned: PlannedFile,
        dest: Path,
        session_uuid: Optional[str],
        logger: Optional[JsonLogger],
    ) -> None:
        rel_path = f"{folder_name}/{planned.dest_name}"
        result: Dict[str, object] = dict(check_hevc_keyframe_flags(dest))
        result["checked_at_utc"] = _utc_now()
        try:
            stat = dest.stat()
            result["file_size_bytes"] = int(stat.st_size)
            result["file_mtime_ns"] = int(stat.st_mtime_ns)
        except OSError as exc:
            result["fingerprint_error"] = str(exc)
        plan.keyframe_checks[rel_path] = result

        if logger:
            logger.log(
                "hevc_keyframe_check",
                recording_name=plan.name,
                session_uuid=session_uuid,
                path=str(dest),
                path_rel=rel_path,
                **result,
            )

    for plan in plans:
        moved_count = 0
        session_uuid = plan.meta.get("session_uuid")
        plan.keyframe_checks = {}

        def record_warning(message: str) -> None:
            warnings.append(message)
            if logger:
                logger.log(
                    "warning",
                    recording_name=plan.name,
                    session_uuid=session_uuid,
                    message=message,
                )

        targets = [
            ("raw", plan.raw_files),
            ("cams", plan.cam_files),
            ("derived", plan.derived_files),
        ]
        for folder_name, files in targets:
            if not files and not create_empty:
                continue
            dest_dir = plan.dest_dir / folder_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for planned in files:
                src = planned.source
                should_copy = planned.action == "copy"
                if planned.action not in {"move", "copy"}:
                    record_warning(f"Unknown planned file action '{planned.action}' for {src}")
                    continue
                if not should_copy and src in moved:
                    record_warning(f"Skipping duplicate source: {src}")
                    continue
                if not src.exists():
                    record_warning(f"Missing source: {src}")
                    continue
                dest = dest_dir / planned.dest_name
                if dest in planned_destinations:
                    record_warning(f"Duplicate destination planned, skipping: {dest}")
                    continue
                if dest.exists():
                    record_warning(f"Destination exists, skipping: {dest}")
                    continue
                verb = "Copy" if should_copy else "Move"
                print(f"{verb}: {src} -> {dest}")
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if should_copy:
                        shutil.copy2(str(src), str(dest))
                    else:
                        shutil.move(str(src), str(dest))
                except Exception as exc:
                    record_warning(f"Failed to {planned.action} {src} -> {dest}: {exc}")
                    continue
                if not should_copy:
                    moved.add(src)
                planned_destinations.add(dest)
                moved_count += 1
                if logger:
                    logger.log(
                        "file_copied" if should_copy else "file_moved",
                        recording_name=plan.name,
                        session_uuid=session_uuid,
                        source=str(src),
                        dest=str(dest),
                        action=planned.action,
                    )
                if dest.suffix.lower() == ".mp4":
                    record_video_keyframe_status(
                        plan=plan,
                        folder_name=folder_name,
                        planned=planned,
                        dest=dest,
                        session_uuid=session_uuid,
                        logger=logger,
                    )
                    check = plan.keyframe_checks.get(f"{folder_name}/{planned.dest_name}", {})
                    if bool(check.get("needs_fix", False)):
                        check_message = str(check.get("message", "")).strip()
                        record_warning(
                            f"HEVC keyframe flags issue for {dest}: "
                            f"{check_message or 'missing stss sync sample table'}"
                        )

        if snapshot is not None:
            warning = _write_snapshot(plan, snapshot, snapshot_mode)
            if warning:
                record_warning(warning)
            elif logger:
                logger.log(
                    "snapshot_written",
                    recording_name=plan.name,
                    session_uuid=session_uuid,
                    dest=str(plan.dest_dir / "derived" / "recording_snapshot.json"),
                )

        if write_manifest:
            organized_utc = _utc_now()
            warning = _write_manifest(plan, organized_utc, run_id, log_path)
            if warning:
                record_warning(warning)
            elif logger:
                logger.log(
                    "manifest_written",
                    recording_name=plan.name,
                    session_uuid=session_uuid,
                    organized_utc=organized_utc,
                    dest=str(plan.dest_dir / "recording_manifest.json"),
                )

        if logger:
            logger.log(
                "recording_applied",
                recording_name=plan.name,
                session_uuid=session_uuid,
                camera_id=plan.camera_id,
                dest_dir=str(plan.dest_dir),
                moved_files=moved_count,
            )

    return warnings


def _cleanup_empty_dirs(root: Path) -> List[str]:
    warnings: List[str] = []
    if not root.exists() or not root.is_dir():
        return warnings

    candidates = [path for path in root.rglob("*") if path.is_dir()]
    candidates.append(root)
    candidates.sort(key=lambda p: len(p.parts), reverse=True)

    for path in candidates:
        try:
            if any(path.iterdir()):
                continue
            path.rmdir()
            print(f"Removed empty directory: {path}")
        except Exception as exc:
            warnings.append(f"Failed to remove {path}: {exc}")
    return warnings


def _cleanup_staging_dirs(
    root: Path,
    ignore_names: Set[str],
    logger: Optional[JsonLogger],
    batch_source: Optional[str],
) -> List[str]:
    warnings: List[str] = []
    if not root.exists() or not root.is_dir():
        return warnings

    candidates = [path for path in root.rglob("*") if path.is_dir()]
    candidates.append(root)
    candidates.sort(key=lambda p: len(p.parts), reverse=True)

    for path in candidates:
        try:
            entries = list(path.iterdir())
        except Exception as exc:
            warnings.append(f"Failed to inspect {path}: {exc}")
            continue

        removable = True
        for entry in entries:
            if entry.is_dir():
                if entry.exists():
                    removable = False
                continue
            if entry.name in ignore_names:
                try:
                    entry.unlink()
                    if logger:
                        logger.log(
                            "cleanup_removed",
                            batch_source=batch_source,
                            path=str(entry),
                        )
                except Exception as exc:
                    warnings.append(f"Failed to remove {entry}: {exc}")
                    removable = False
            else:
                removable = False

        if removable:
            try:
                path.rmdir()
                print(f"Removed empty directory: {path}")
                if logger:
                    logger.log(
                        "cleanup_removed",
                        batch_source=batch_source,
                        path=str(path),
                    )
            except Exception as exc:
                warnings.append(f"Failed to remove {path}: {exc}")

    return warnings


_VIDEO_DIAGNOSTICS_SAMPLE_FRAMES = 120
_VIDEO_DIAGNOSTICS_DECODE_FRAMES = 30
_VIDEO_DIAGNOSTICS_SEEK_SAMPLES = 10


def _diagnostic_finding_codes(findings: List[object], limit: int = 3) -> List[str]:
    codes: List[str] = []
    for finding in findings:
        code = getattr(finding, "code", None)
        if code is None:
            continue
        code_text = str(code)
        if not code_text or code_text in codes:
            continue
        codes.append(code_text)
        if len(codes) >= limit:
            break
    return codes


def _coerce_video_diagnostics_hook_result(result: object) -> VideoDiagnosticsHookResult:
    if isinstance(result, VideoDiagnosticsHookResult):
        return result
    if isinstance(result, list):
        return VideoDiagnosticsHookResult(
            manifest_payload=build_video_preflight_payload(
                status=PRECHECK_NOT_RUN,
                media_status=PRECHECK_NOT_RUN,
                tooling_status="skip",
                videos_scanned=0,
                finding_codes=[],
            ),
            warnings=[str(item) for item in result],
        )
    raise TypeError(f"unexpected video diagnostics hook result: {type(result)!r}")


def _coerce_h5_diagnostics_hook_result(result: object) -> H5DiagnosticsHookResult:
    if isinstance(result, H5DiagnosticsHookResult):
        return result
    if isinstance(result, list):
        return H5DiagnosticsHookResult(
            manifest_payload=build_h5_preflight_payload(
                status=PRECHECK_NOT_RUN,
                core_status=PRECHECK_NOT_RUN,
                optional_status=PRECHECK_NOT_RUN,
                tooling_status="skip",
                finding_codes=[],
            ),
            warnings=[str(item) for item in result],
        )
    raise TypeError(f"unexpected h5 diagnostics hook result: {type(result)!r}")


def _persist_preflight_to_manifest(
    plan: RecordingPlan,
    *,
    video_result: Optional[VideoDiagnosticsHookResult],
    h5_result: Optional[H5DiagnosticsHookResult],
) -> Optional[str]:
    manifest_path = plan.dest_dir / "recording_manifest.json"
    if not manifest_path.exists():
        return f"Missing manifest, cannot persist preflight for {plan.name}: {manifest_path}"

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Failed to read manifest for {plan.name}: {exc}"
    if not isinstance(payload, dict):
        return f"Manifest root is not a JSON object: {manifest_path}"

    existing_preflight = payload.get("preflight")
    if not isinstance(existing_preflight, dict):
        existing_preflight = {}

    existing_video = existing_preflight.get("video") if isinstance(existing_preflight.get("video"), dict) else None
    existing_h5 = existing_preflight.get("h5") if isinstance(existing_preflight.get("h5"), dict) else None
    payload["preflight"] = build_manifest_preflight_payload(
        checked_at_utc=_utc_now(),
        video=video_result.manifest_payload if video_result is not None else existing_video,
        h5=h5_result.manifest_payload if h5_result is not None else existing_h5,
    )

    try:
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        return f"Failed to update preflight manifest for {plan.name}: {exc}"
    return None


def _run_video_diagnostics_for_plan(
    plan: RecordingPlan,
    logger: Optional[JsonLogger],
) -> VideoDiagnosticsHookResult:
    session_uuid = plan.meta.get("session_uuid")
    try:
        from fisheye.diagnostics.video.batch import build_batch_report
    except Exception as exc:
        message = f"Video diagnostics unavailable for {plan.name}: {exc}"
        if logger:
            logger.log(
                "video_diagnostics_error",
                recording_name=plan.name,
                session_uuid=session_uuid,
                recording_dir=str(plan.dest_dir),
                message=message,
            )
        return VideoDiagnosticsHookResult(
            manifest_payload=build_video_preflight_payload(
                status=PRECHECK_WARN,
                media_status=PRECHECK_NOT_RUN,
                tooling_status="error",
                videos_scanned=0,
                finding_codes=[],
                error=str(exc),
            ),
            warnings=[message],
        )

    try:
        report = build_batch_report(
            [plan.dest_dir],
            recursive=True,
            source="all",
            full_scan=False,
            sample_frames=_VIDEO_DIAGNOSTICS_SAMPLE_FRAMES,
            decode_backend="opencv",
            decode_frames=_VIDEO_DIAGNOSTICS_DECODE_FRAMES,
            seek_samples=_VIDEO_DIAGNOSTICS_SEEK_SAMPLES,
            include_probe=True,
            include_timing=True,
            include_gop=True,
            include_decode=True,
        )
    except Exception as exc:
        message = f"Video diagnostics failed for {plan.name}: {exc}"
        if logger:
            logger.log(
                "video_diagnostics_error",
                recording_name=plan.name,
                session_uuid=session_uuid,
                recording_dir=str(plan.dest_dir),
                message=message,
            )
        return VideoDiagnosticsHookResult(
            manifest_payload=build_video_preflight_payload(
                status=PRECHECK_WARN,
                media_status=PRECHECK_NOT_RUN,
                tooling_status="error",
                videos_scanned=0,
                finding_codes=[],
                error=str(exc),
            ),
            warnings=[message],
        )

    recording = next(
        (item for item in report.recordings if item.recording_root == str(plan.dest_dir)),
        None,
    )
    media_status = str(recording.media_status if recording is not None else report.overall_status)
    tooling_status = str(recording.tooling_status if recording is not None else "skip")
    scanned = int(recording.item_count if recording is not None else report.summary.scanned)
    finding_codes = _diagnostic_finding_codes(
        [finding for item in report.items for finding in item.findings]
    )
    finding_suffix = f" ({', '.join(finding_codes)})" if finding_codes else ""
    print(
        f"Video diagnostics [{plan.name}]: media={media_status} tooling={tooling_status} videos={scanned}{finding_suffix}"
    )
    if logger:
        logger.log(
            "video_diagnostics",
            recording_name=plan.name,
            session_uuid=session_uuid,
            recording_dir=str(plan.dest_dir),
            media_status=media_status,
            tooling_status=tooling_status,
            videos_scanned=scanned,
            finding_codes=finding_codes,
        )

    status = PRECHECK_PASS
    warnings: List[str] = []
    if scanned == 0:
        status = PRECHECK_WARN
        warnings.append(f"Video diagnostics for {plan.name}: no videos found under {plan.dest_dir}")
        media_payload_status = PRECHECK_NOT_RUN
    else:
        media_payload_status = media_status
        if media_status == PRECHECK_FAIL:
            status = PRECHECK_FAIL
        elif media_status in {PRECHECK_WARN, "error"} or tooling_status in {PRECHECK_WARN, PRECHECK_FAIL, "error"}:
            status = PRECHECK_WARN
        if status in {PRECHECK_WARN, PRECHECK_FAIL}:
            warnings.append(
                f"Video diagnostics for {plan.name}: media={media_status} tooling={tooling_status}{finding_suffix}"
            )

    return VideoDiagnosticsHookResult(
        manifest_payload=build_video_preflight_payload(
            status=status,
            media_status=media_payload_status,
            tooling_status=tooling_status,
            videos_scanned=scanned,
            finding_codes=finding_codes,
        ),
        warnings=warnings,
    )


def _run_h5_diagnostics_for_plan(
    plan: RecordingPlan,
    logger: Optional[JsonLogger],
) -> H5DiagnosticsHookResult:
    session_uuid = plan.meta.get("session_uuid")
    try:
        from fisheye.diagnostics.h5 import build_h5_report
    except Exception as exc:
        message = f"H5 diagnostics unavailable for {plan.name}: {exc}"
        if logger:
            logger.log(
                "h5_diagnostics_error",
                recording_name=plan.name,
                session_uuid=session_uuid,
                recording_dir=str(plan.dest_dir),
                message=message,
            )
        return H5DiagnosticsHookResult(
            manifest_payload=build_h5_preflight_payload(
                status=PRECHECK_WARN,
                core_status=PRECHECK_NOT_RUN,
                optional_status=PRECHECK_NOT_RUN,
                tooling_status="error",
                finding_codes=[],
                error=str(exc),
            ),
            warnings=[message],
        )

    try:
        report = build_h5_report(plan.dest_dir, profile="palette-import")
    except Exception as exc:
        message = f"H5 diagnostics failed for {plan.name}: {exc}"
        if logger:
            logger.log(
                "h5_diagnostics_error",
                recording_name=plan.name,
                session_uuid=session_uuid,
                recording_dir=str(plan.dest_dir),
                message=message,
            )
        return H5DiagnosticsHookResult(
            manifest_payload=build_h5_preflight_payload(
                status=PRECHECK_WARN,
                core_status=PRECHECK_NOT_RUN,
                optional_status=PRECHECK_NOT_RUN,
                tooling_status="error",
                finding_codes=[],
                error=str(exc),
            ),
            warnings=[message],
        )

    finding_codes = _diagnostic_finding_codes(report.findings)
    finding_suffix = f" ({', '.join(finding_codes)})" if finding_codes else ""
    print(
        f"H5 diagnostics [{plan.name}]: core={report.core_status} optional={report.optional_status} tooling={report.tooling_status}{finding_suffix}"
    )
    if logger:
        logger.log(
            "h5_diagnostics",
            recording_name=plan.name,
            session_uuid=session_uuid,
            recording_dir=str(plan.dest_dir),
            h5_path=report.file_info.path,
            core_status=report.core_status,
            optional_status=report.optional_status,
            tooling_status=report.tooling_status,
            finding_codes=finding_codes,
        )

    status = PRECHECK_PASS
    warnings: List[str] = []
    if report.core_status == PRECHECK_FAIL:
        status = PRECHECK_FAIL
    elif report.core_status in {PRECHECK_WARN, "error"} or report.optional_status in {PRECHECK_WARN, PRECHECK_FAIL, "error"} or report.tooling_status in {PRECHECK_WARN, PRECHECK_FAIL, "error"}:
        status = PRECHECK_WARN

    if status in {PRECHECK_WARN, PRECHECK_FAIL}:
        warnings.append(
            f"H5 diagnostics for {plan.name}: core={report.core_status} optional={report.optional_status} tooling={report.tooling_status}{finding_suffix}"
        )

    return H5DiagnosticsHookResult(
        manifest_payload=build_h5_preflight_payload(
            status=status,
            core_status=str(report.core_status),
            optional_status=str(report.optional_status),
            tooling_status=str(report.tooling_status),
            finding_codes=finding_codes,
        ),
        warnings=warnings,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan recording re-organization by mapping H5 files to camera artifacts.",
        epilog=(
            "Environment variables:\n"
            "  PALETTE_STAGING_ROOT   default source root when no positional source is given\n"
            "  PALETTE_RECORDINGS_ROOT used by scripts/organize_staging.sh for --dest-root\n"
            "  PALETTE_LOG_ROOT       default log directory for JSONL logs\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Directory containing H5 recordings "
            "(defaults to PALETTE_STAGING_ROOT or /nvme1/staging)."
        ),
    )
    default_dest_root = Path(os.environ.get("PALETTE_RECORDINGS_ROOT", "/nvme1/recordings"))
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=default_dest_root,
        help="Planned root directory for reorganized recordings (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to write JSONL logs (defaults to PALETTE_LOG_ROOT or <dest-root>/logs/organize_recordings).",
    )
    parser.add_argument(
        "--cam-root",
        type=Path,
        default=None,
        help="Optional root to search for Cam<id>.* files (defaults to source when --recursive).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for H5 files recursively under the source directory.",
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process every immediate subdirectory of the source (staging root).",
    )
    parser.add_argument(
        "--require-done",
        action="store_true",
        help="Only process batches that include the done marker file.",
    )
    parser.add_argument(
        "--done-name",
        default="TRANSFER_DONE",
        help="Marker filename that indicates transfer completion (default: TRANSFER_DONE).",
    )
    parser.add_argument(
        "--rename-cams",
        dest="rename_cams",
        action="store_true",
        default=True,
        help="Rename Cam files to include the session identifier (Cam<id>_<session>.*).",
    )
    parser.add_argument(
        "--no-rename-cams",
        dest="rename_cams",
        action="store_false",
        help="Keep original Cam file names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print an ASCII tree of the planned layout (no changes are made).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move files into the planned layout.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Optional recording_snapshot.json to attach to each recording.",
    )
    parser.add_argument(
        "--snapshot-mode",
        choices=("split", "copy"),
        default="split",
        help="How to attach the snapshot: split per camera or copy whole file.",
    )
    parser.add_argument(
        "--cleanup-empty",
        action="store_true",
        help="Remove empty directories under the source path after --apply.",
    )
    parser.add_argument(
        "--cleanup-staging",
        action="store_true",
        help="Remove staging batches that only contain marker/snapshot files after --apply.",
    )
    parser.add_argument(
        "--cleanup-ignore",
        action="append",
        default=[],
        help="Additional filenames to ignore during --cleanup-staging (can be repeated).",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write recording_manifest.json into each recording folder during --apply.",
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Organize MP4-only recordings using metadata from --metadata-csv instead of H5 discovery.",
    )
    parser.add_argument(
        "--external-ipc",
        action="store_true",
        help=(
            "Organize an Orange external_ipc batch from recording_session.json. "
            "This is auto-detected when recording_outputs are present."
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="CSV describing video-only recordings. Required with --video-only.",
    )
    parser.add_argument(
        "--run-video-diagnostics",
        action="store_true",
        help="Run unified video diagnostics against each organized recording after --apply.",
    )
    parser.add_argument(
        "--run-h5-diagnostics",
        action="store_true",
        help="Run unified H5 diagnostics against each organized recording after --apply.",
    )

    args = parser.parse_args()

    source = args.source
    if source is None:
        source = Path(os.environ.get("PALETTE_STAGING_ROOT", "/nvme1/staging"))

    if not source.exists():
        print(f"Source path does not exist: {source}", file=sys.stderr)
        return 1

    if args.video_only:
        if args.metadata_csv is None:
            print("--metadata-csv is required with --video-only.", file=sys.stderr)
            return 1
        if args.process_all:
            print("--process-all is not supported with --video-only.", file=sys.stderr)
            return 1
        if args.require_done:
            print("--require-done is not supported with --video-only.", file=sys.stderr)
            return 1
        if args.run_h5_diagnostics:
            print("--run-h5-diagnostics is not supported with --video-only.", file=sys.stderr)
            return 1
        if args.external_ipc:
            print("--external-ipc is not supported with --video-only.", file=sys.stderr)
            return 1

    if (args.run_video_diagnostics or args.run_h5_diagnostics) and not args.apply:
        print("--run-video-diagnostics and --run-h5-diagnostics require --apply.", file=sys.stderr)
        return 1

    effective_write_manifest = bool(
        args.write_manifest or args.run_video_diagnostics or args.run_h5_diagnostics
    )

    if args.process_all:
        sources = sorted([path for path in source.iterdir() if path.is_dir()])
        if not sources:
            print(f"No subdirectories found under staging root: {source}")
            return 0
    else:
        sources = [source]

    run_id = make_run_id()
    log_dir: Optional[Path]
    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        env_log = os.environ.get("PALETTE_LOG_ROOT")
        if env_log:
            log_dir = Path(env_log)
        else:
            log_dir = args.dest_root / "logs" / "organize_recordings"

    logger: Optional[JsonLogger] = None
    log_path: Optional[Path] = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"organize_recordings_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            source_root=str(source),
            process_all=args.process_all,
            recursive=args.recursive,
            dest_root=str(args.dest_root),
            rename_cams=args.rename_cams,
            write_manifest=effective_write_manifest,
            requested_write_manifest=bool(args.write_manifest),
            snapshot_mode=args.snapshot_mode,
            video_only=args.video_only,
            external_ipc=args.external_ipc,
            metadata_csv=str(args.metadata_csv) if args.metadata_csv else None,
            run_video_diagnostics=args.run_video_diagnostics,
            run_h5_diagnostics=args.run_h5_diagnostics,
        )
    except Exception as exc:
        print(f"Warning: could not create log file: {exc}", file=sys.stderr)
        logger = None
        log_path = None

    for source_path in sources:
        if args.process_all:
            print(f"\n=== Processing {source_path} ===")
        if logger:
            logger.log("batch_start", batch_source=str(source_path))

        if args.require_done:
            marker = source_path / args.done_name
            if not marker.exists():
                print(f"Skipping {source_path}: missing marker {args.done_name}")
                if logger:
                    logger.log(
                        "batch_skipped",
                        batch_source=str(source_path),
                        reason=f"missing {args.done_name}",
                    )
                continue

        snapshot_payload: Optional[Dict[str, object]] = None
        plans: List[RecordingPlan]
        if args.video_only:
            try:
                rows = _load_video_only_rows(args.metadata_csv.expanduser().resolve(), source_root=source_path)
            except Exception as exc:
                print(f"Failed to read metadata CSV: {exc}", file=sys.stderr)
                return 1
            plans = [
                _build_video_only_plan(row, dest_root=args.dest_root, rename_cams=args.rename_cams)
                for row in rows
            ]
            print(f"Found {len(plans)} video-only recording(s) from metadata CSV.")
        elif args.external_ipc or _looks_like_external_ipc_batch(source_path):
            try:
                plans = _build_external_ipc_plans(
                    source_path,
                    dest_root=args.dest_root,
                    rename_cams=args.rename_cams,
                )
            except Exception as exc:
                print(f"Failed to build external_ipc plan: {exc}", file=sys.stderr)
                return 1
            print(f"Found {len(plans)} external_ipc H5 recording(s).")

            snapshot_path: Optional[Path] = args.snapshot
            if snapshot_path is None:
                default_json = source_path / "recording_snapshot.json"
                default_plain = source_path / "recording_snapshot"
                if default_json.exists():
                    snapshot_path = default_json
                elif default_plain.exists():
                    snapshot_path = default_plain

            if snapshot_path is not None:
                if not snapshot_path.exists():
                    print(f"Snapshot path does not exist: {snapshot_path}", file=sys.stderr)
                else:
                    try:
                        snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        print(f"Failed to read snapshot JSON: {exc}", file=sys.stderr)
                        snapshot_payload = None
                    else:
                        if not isinstance(snapshot_payload, dict):
                            print("Snapshot JSON must be an object at the top level.", file=sys.stderr)
                            snapshot_payload = None
        else:
            h5_files = _find_h5_files(source_path, args.recursive)
            if not h5_files:
                print(f"No .h5 files found in {source_path}.")
                if not args.recursive:
                    print("Hint: use --recursive if H5 files live in subfolders.")
                continue

            cam_root = args.cam_root
            if cam_root is None and args.recursive:
                cam_root = source_path

            snapshot_path: Optional[Path] = args.snapshot
            if snapshot_path is None:
                default_json = source_path / "recording_snapshot.json"
                default_plain = source_path / "recording_snapshot"
                if default_json.exists():
                    snapshot_path = default_json
                elif default_plain.exists():
                    snapshot_path = default_plain

            if snapshot_path is not None:
                if not snapshot_path.exists():
                    print(f"Snapshot path does not exist: {snapshot_path}", file=sys.stderr)
                else:
                    try:
                        snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        print(f"Failed to read snapshot JSON: {exc}", file=sys.stderr)
                        snapshot_payload = None
                    else:
                        if not isinstance(snapshot_payload, dict):
                            print("Snapshot JSON must be an object at the top level.", file=sys.stderr)
                            snapshot_payload = None

            plans = [
                _build_plan(h5_path, args.dest_root, cam_root, args.rename_cams)
                for h5_path in h5_files
            ]

            print(f"Found {len(plans)} H5 recording(s).")
        for plan in plans:
            for line in _format_recording_summary(plan):
                print(line)
            if logger:
                logger.log(
                    "recording_plan",
                    recording_name=plan.name,
                    session_uuid=plan.meta.get("session_uuid"),
                    camera_id=plan.camera_id,
                    missing=plan.missing,
                    raw_files=[file.dest_name for file in plan.raw_files],
                    cam_files=[file.dest_name for file in plan.cam_files],
                    derived_files=[file.dest_name for file in plan.derived_files],
                )

        if args.dry_run:
            print("\nPlanned layout (dry-run):")
            tree_lines = _render_tree(str(args.dest_root), plans)
            for line in tree_lines:
                print(line)

        if args.apply:
            print("\nApplying moves:")
            warnings = _apply_plan(
                plans,
                create_empty=False,
                write_manifest=effective_write_manifest,
                snapshot=snapshot_payload,
                snapshot_mode=args.snapshot_mode,
                logger=logger,
                run_id=run_id,
                log_path=log_path,
            )
            if args.cleanup_empty:
                cleanup_warnings = _cleanup_empty_dirs(source_path)
                warnings.extend(cleanup_warnings)
                if logger:
                    for warning in cleanup_warnings:
                        logger.log("warning", batch_source=str(source_path), message=warning)
            if args.cleanup_staging:
                # Snapshot names kept as safety net — normally moved to raw/
                # by _build_plan, but may remain if no H5 was found.
                ignore_names = {"TRANSFER_DONE", "recording_snapshot.json", "recording_snapshot"}
                ignore_names.update(args.cleanup_ignore)
                cleanup_warnings = _cleanup_staging_dirs(
                    source_path,
                    ignore_names=ignore_names,
                    logger=logger,
                    batch_source=str(source_path),
                )
                warnings.extend(cleanup_warnings)
                if logger:
                    for warning in cleanup_warnings:
                        logger.log("warning", batch_source=str(source_path), message=warning)
            if args.run_video_diagnostics or args.run_h5_diagnostics:
                print("\nRunning post-organize diagnostics:")
                for plan in plans:
                    video_result: Optional[VideoDiagnosticsHookResult] = None
                    h5_result: Optional[H5DiagnosticsHookResult] = None
                    if args.run_video_diagnostics:
                        video_result = _coerce_video_diagnostics_hook_result(
                            _run_video_diagnostics_for_plan(plan, logger)
                        )
                        warnings.extend(video_result.warnings)
                    if args.run_h5_diagnostics:
                        h5_result = _coerce_h5_diagnostics_hook_result(
                            _run_h5_diagnostics_for_plan(plan, logger)
                        )
                        warnings.extend(h5_result.warnings)
                    manifest_warning = _persist_preflight_to_manifest(
                        plan,
                        video_result=video_result,
                        h5_result=h5_result,
                    )
                    if manifest_warning:
                        warnings.append(manifest_warning)
                        if logger:
                            logger.log(
                                "warning",
                                recording_name=plan.name,
                                session_uuid=plan.meta.get("session_uuid"),
                                message=manifest_warning,
                            )
                    elif logger:
                        logger.log(
                            "preflight_written",
                            recording_name=plan.name,
                            session_uuid=plan.meta.get("session_uuid"),
                            manifest_path=str(plan.dest_dir / "recording_manifest.json"),
                            video_status=(video_result.manifest_payload.get("status") if video_result else None),
                            h5_status=(h5_result.manifest_payload.get("status") if h5_result else None),
                        )
            if warnings:
                print("\nWarnings:")
                for warning in warnings:
                    print(f"  - {warning}")

    if logger:
        logger.log("run_end")
        logger.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
