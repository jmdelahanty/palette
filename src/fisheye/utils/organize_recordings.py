#!/usr/bin/env python3
"""Shared helpers from the retired per-H5 organizer.

New recordings enter only through transfer-v2 parent intake
(``organize_transfer_recordings``). What remains here is used by that intake
(H5 camera context, geometry bundle discovery), by the refresh tools that
maintain existing organized recordings (preflight/diagnostics hooks, external-IPC
stream payloads), and by the video-only sidecar backfill. The legacy organizer
CLI and its per-H5, external-IPC and recording-only plan builders were removed.
"""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import h5py

from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import normalize_attr as _normalize_attr
from fisheye.shared.recording_preflight import (
    PRECHECK_FAIL,
    PRECHECK_NOT_RUN,
    PRECHECK_PASS,
    PRECHECK_WARN,
    build_h5_preflight_payload,
    build_manifest_preflight_payload,
    build_video_preflight_payload,
)
from fisheye.shared.recording_geometry import (
    RECORDING_GEOMETRY_ASSETS_NAME,
    RECORDING_GEOMETRY_CONTRACT_NAME,
    RECORDING_SNAPSHOT_NAME,
    RecordingGeometryBundleVerification,
)
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
    SourceRecordingIdentity,
    SourceRecordingIdentityError,
    require_source_identity_text,
)

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
    geometry_bundle_source: Optional[Path] = None
    geometry_bundle_verification: Optional[RecordingGeometryBundleVerification] = None


class RecordingGeometryApplyError(RuntimeError):
    """Raised before ordinary recording moves when geometry preservation fails."""




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
    match = re.search(r"(?:^|[/_-])cam[_-]?(\d+)(?:$|[^0-9])", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _derive_camera_id_from_path(path: Path) -> Optional[str]:
    match = re.search(
        r"(?:^|[^A-Za-z0-9])Cam(\d+)(?=$|[^0-9])",
        path.name,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


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
        protocol_definition = _read_h5_json_object(
            h5, "protocol_snapshot/protocol_definition_json"
        )
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
        _set_meta_if_present(
            meta, "dish_design", _dish_design_from_arena_config(arena_config)
        )


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
                    if key in {"session_uuid", "camera_id"}:
                        meta[key] = require_source_identity_text(
                            root.get(key), field=f"H5 {key}"
                        )
                    else:
                        _set_meta_if_present(meta, key, root.get(key))
            _augment_h5_manifest_context(h5, meta)
            camera_id = meta.get("camera_id")
            derived = _derive_camera_id(meta.get("ipc_source_name"))
            if camera_id and derived and camera_id != derived:
                raise SourceRecordingIdentityError(
                    "H5 camera_id conflicts with ipc_source_name camera: "
                    f"{camera_id!r} != {derived!r}"
                )
            if not camera_id and derived:
                meta["camera_id"] = derived
                meta["camera_id_source"] = "ipc_source_name"
                camera_id = derived
            return camera_id, meta
    except SourceRecordingIdentityError:
        raise
    except Exception as exc:
        meta["error"] = f"failed to read H5: {exc}"
        return None, meta




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

            optional_camera_csv = row.get("source_camera_metadata_csv") or row.get(
                "camera_metadata_csv"
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
    identity = SourceRecordingIdentity.from_mapping(row)
    video_path = Path(row["source_video"]).expanduser().resolve()
    derived_camera_id = _derive_camera_id_from_path(video_path)
    if derived_camera_id is not None and derived_camera_id != identity.camera_id:
        raise SourceRecordingIdentityError(
            "camera_id conflicts with the unambiguous camera ID in source_video: "
            f"{identity.camera_id!r} != {derived_camera_id!r}"
        )
    camera_id = identity.camera_id
    session_uuid = identity.session_uuid
    recording_name = row.get("recording_name") or identity.recording_id
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
            cam_files.append(
                PlannedFile(camera_csv_path, f"Cam{camera_id}_{session_tag}_meta.csv")
            )
        else:
            cam_files.append(PlannedFile(camera_csv_path, camera_csv_path.name))

    raw_files: List[PlannedFile] = []
    derived_files: List[PlannedFile] = []
    if camera_id:
        keyframe_sidecar = _first_existing(
            [
                video_path.with_name(f"Cam{camera_id}_keyframe.json"),
                video_path.with_name(f"{video_path.stem}_keyframe.json"),
            ]
        )
        if keyframe_sidecar is not None:
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

    # Video-only imports may be re-intaken from an already organized legacy
    # recording. Search the video directory, its ancestors, and a sibling raw/
    # directory so shared runtime context is retained without requiring a
    # second physical copy of the camera video.
    shared_roots = _unique_paths(
        [video_path.parent]
        + list(video_path.parents)
        + [ancestor / "raw" for ancestor in video_path.parents]
    )
    for shared_names, dest_name in (
        (("ptp_sync_summary.json",), "ptp_sync_summary.json"),
        (
            (
                "recording_snapshot.json",
                "recording_snapshot",
                "recording_snapshot_runtime.json",
            ),
            "recording_snapshot_runtime.json",
        ),
    ):
        shared_path = _first_existing(
            [root / name for root in shared_roots for name in shared_names]
        )
        if shared_path is not None:
            raw_files.append(PlannedFile(shared_path, dest_name, action="copy"))

    meta: Dict[str, str] = {
        "session_uuid": session_uuid,
        "recording_id": identity.recording_id,
        SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: SOURCE_RECORDING_IDENTITY_PROFILE,
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
        "organizer_recording_id",
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
    full_frame_clock_metadata: Optional[str],
    has_full_summary: bool,
    has_full_status: bool,
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
            "frame_clock_metadata": full_frame_clock_metadata,
            "keyframes": f"cams/{cam_base}_keyframe.json",
            "summary": (
                f"cams/{cam_base}_external_summary.json" if has_full_summary else None
            ),
            "status": (
                f"derived/external_recorder/{cam_base}_external_status.json"
                if has_full_status
                else None
            ),
            "frame_clock": "recording_frame_id",
            "coordinate_space": _pick_stream_value(full_output, "coordinate_space"),
            "width": _pick_stream_value(full_output, "width"),
            "height": _pick_stream_value(full_output, "height"),
            "frame_count": _pick_stream_value(full_output, "frame_count"),
            "frame_rate": _pick_stream_value(full_output, "frame_rate"),
            "codec": _pick_stream_value(full_output, "codec"),
            "container": _pick_stream_value(full_output, "container"),
            "encoded_format": _pick_stream_value(full_output, "encoded_format"),
            "pixel_source_format": _pick_stream_value(
                full_output, "pixel_source_format"
            ),
        }
    )
    streams: Dict[str, object] = {"full": full_stream}

    if crop_output:
        crop_stream = _drop_none_values(
            {
                "role": "runtime_derived_acquisition_input",
                "output_kind": "crop",
                "stream_kind": _pick_stream_value(crop_output, "stream_kind"),
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
                "blank_frame_policy": _pick_stream_value(
                    crop_output, "blank_frame_policy"
                ),
                "selection_policy": _pick_stream_value(crop_output, "selection_policy"),
                "width": _pick_stream_value(crop_output, "width"),
                "height": _pick_stream_value(crop_output, "height"),
                "frame_count": _pick_stream_value(crop_output, "frame_count"),
                "packet_count": _pick_stream_value(crop_output, "packet_count"),
                "frame_rate": _pick_stream_value(crop_output, "frame_rate"),
                "codec": _pick_stream_value(crop_output, "codec"),
                "container": _pick_stream_value(crop_output, "container"),
                "tuning": _pick_stream_value(crop_output, "tuning"),
                "encoded_format": _pick_stream_value(crop_output, "encoded_format"),
                "pixel_source_format": _pick_stream_value(
                    crop_output, "pixel_source_format"
                ),
            }
        )
        streams["crop"] = crop_stream

    return {
        "schema_id": "orange_runtime_video_streams_v1",
        "frame_clock": "recording_frame_id",
        "streams": streams,
    }






def _recording_geometry_bundle_source(root: Path) -> Optional[Path]:
    """Return a fixed-layout geometry root when any v1 child is present.

    Returning partial roots is intentional: apply-time verification then fails
    closed instead of silently organizing the recording without its geometry.
    """

    children = (
        root / RECORDING_SNAPSHOT_NAME,
        root / RECORDING_GEOMETRY_CONTRACT_NAME,
        root / RECORDING_GEOMETRY_ASSETS_NAME,
    )
    return root if any(path.exists() for path in children[1:]) else None










































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

    existing_video = (
        existing_preflight.get("video")
        if isinstance(existing_preflight.get("video"), dict)
        else None
    )
    existing_h5 = (
        existing_preflight.get("h5")
        if isinstance(existing_preflight.get("h5"), dict)
        else None
    )
    payload["preflight"] = build_manifest_preflight_payload(
        checked_at_utc=_utc_now(),
        video=(
            video_result.manifest_payload
            if video_result is not None
            else existing_video
        ),
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
        (
            item
            for item in report.recordings
            if item.recording_root == str(plan.dest_dir)
        ),
        None,
    )
    media_status = str(
        recording.media_status if recording is not None else report.overall_status
    )
    tooling_status = str(recording.tooling_status if recording is not None else "skip")
    scanned = int(
        recording.item_count if recording is not None else report.summary.scanned
    )
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
        warnings.append(
            f"Video diagnostics for {plan.name}: no videos found under {plan.dest_dir}"
        )
        media_payload_status = PRECHECK_NOT_RUN
    else:
        media_payload_status = media_status
        if media_status == PRECHECK_FAIL:
            status = PRECHECK_FAIL
        elif media_status in {PRECHECK_WARN, "error"} or tooling_status in {
            PRECHECK_WARN,
            PRECHECK_FAIL,
            "error",
        }:
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
    elif (
        report.core_status in {PRECHECK_WARN, "error"}
        or report.optional_status in {PRECHECK_WARN, PRECHECK_FAIL, "error"}
        or report.tooling_status in {PRECHECK_WARN, PRECHECK_FAIL, "error"}
    ):
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




