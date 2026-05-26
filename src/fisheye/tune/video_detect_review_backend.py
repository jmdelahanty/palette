"""Video-backed detection review primitives.

This module is deliberately separate from ``detect_review_backend``: that UI
serves persisted image arrays, while this one serves source videos plus refined
detection rows. Persistence still goes through the same curated refined-detect
writer used by the existing review tools.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import zarr

from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import resolve_latest_complete_run_name
from fisheye.tune import detect_review as detect_review_mod


@dataclass(frozen=True)
class VideoSource:
    video_id: str
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int | None = None
    clip_id: str | None = None
    camera_serial: str | None = None
    media_kind: str = "source_video"
    source_path: Path | None = None
    media_width: int | None = None
    media_height: int | None = None


@dataclass(frozen=True)
class FrameRecord:
    parent_frame_index: int
    source_frame_index: int
    video_id: str
    refined_family_path: str | None
    refined_run_name: str
    refined_group_path: str
    clip_id: str | None = None
    camera_serial: str | None = None
    recording_frame_id: int | None = None


@dataclass
class RefinedPayloadCacheEntry:
    group: zarr.Group
    payload: dict[str, Any]
    total_frames: int
    family_path: str | None
    run_name: str


@dataclass
class VideoDetectReviewSession:
    zarr_path: str
    root: zarr.Group
    mode: str
    editable: bool
    videos: dict[str, VideoSource]
    frame_records: Any
    refined_cache: dict[str, RefinedPayloadCacheEntry] = field(default_factory=dict)
    collection_id: str | None = None
    review_proxy_manifest: str | None = None
    manual_score: float = 1.0
    manual_class_id: int = 0
    current_frame: int = 0


class ClippedFrameRecords:
    def __init__(
        self,
        *,
        table: Any,
        selected_by_pair: Mapping[tuple[str, str], Mapping[str, Any]],
        video_id_by_path: Mapping[str, str],
        video_id_by_pair: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        self._table = table
        self._columns = {name: table[name] for name in table.column_names}
        self._selected_by_pair = dict(selected_by_pair)
        self._video_id_by_path = dict(video_id_by_path)
        self._video_id_by_pair = dict(video_id_by_pair or {})
        self._video_frame_ranges: dict[str, dict[str, int]] | None = None

    def __len__(self) -> int:
        return int(self._table.num_rows)

    def _value(self, name: str, index: int, default: Any = None) -> Any:
        column = self._columns.get(name)
        if column is None:
            return default
        value = column[index].as_py()
        return default if value is None else value

    def __getitem__(self, parent_frame_index: int) -> FrameRecord:
        if parent_frame_index < 0 or parent_frame_index >= len(self):
            raise IndexError(parent_frame_index)
        stored_parent = self._value("parent_frame_index", parent_frame_index, parent_frame_index)
        if int(stored_parent) != int(parent_frame_index):
            raise RuntimeError(
                "Clipped video review currently requires parent_frame_index to be contiguous and row-aligned."
            )
        camera_serial = str(self._value("camera_serial", parent_frame_index, ""))
        clip_id = str(self._value("clip_id", parent_frame_index, ""))
        selected = self._selected_by_pair.get((camera_serial, clip_id))
        if selected is None:
            raise RuntimeError(f"No selected finalized run for camera={camera_serial!r} clip={clip_id!r}")
        refined_group_path = str(selected.get("refined_group_path") or "").strip("/")
        if not refined_group_path:
            raise RuntimeError(f"Selected run missing refined_group_path for camera={camera_serial!r} clip={clip_id!r}")
        video_path = str(self._value("video_path", parent_frame_index, ""))
        video_id = self._video_id_by_path.get(video_path)
        if not video_id:
            raise RuntimeError(f"No video source registered for frame-index path: {video_path}")
        return FrameRecord(
            parent_frame_index=int(parent_frame_index),
            source_frame_index=int(self._value("clip_local_frame_index", parent_frame_index, 0)),
            video_id=video_id,
            refined_family_path=refined_group_path.rsplit("/", 1)[0],
            refined_run_name=str(selected.get("refined_detect_run") or Path(refined_group_path).name),
            refined_group_path=refined_group_path,
            clip_id=clip_id,
            camera_serial=camera_serial,
            recording_frame_id=int(self._value("recording_frame_id", parent_frame_index, 0)),
        )

    def video_frame_ranges(self) -> dict[str, dict[str, int]]:
        if self._video_frame_ranges is not None:
            return self._video_frame_ranges

        ranges: dict[str, dict[str, int]] = {}
        for row_index in range(len(self)):
            camera_serial = str(self._value("camera_serial", row_index, ""))
            clip_id = str(self._value("clip_id", row_index, ""))
            video_id = self._video_id_by_pair.get((camera_serial, clip_id))
            if not video_id:
                video_path = str(self._value("video_path", row_index, ""))
                video_id = self._video_id_by_path.get(video_path)
            if not video_id:
                continue

            parent_frame = int(self._value("parent_frame_index", row_index, row_index))
            source_frame = int(self._value("clip_local_frame_index", row_index, row_index))
            current = ranges.setdefault(
                video_id,
                {
                    "parent_frame_start": parent_frame,
                    "parent_frame_end": parent_frame,
                    "source_frame_start": source_frame,
                    "source_frame_end": source_frame,
                },
            )
            current["parent_frame_start"] = min(current["parent_frame_start"], parent_frame)
            current["parent_frame_end"] = max(current["parent_frame_end"], parent_frame)
            current["source_frame_start"] = min(current["source_frame_start"], source_frame)
            current["source_frame_end"] = max(current["source_frame_end"], source_frame)

        self._video_frame_ranges = ranges
        return ranges


def _json_scalar(value: object) -> object:
    try:
        scalar = np.asarray(value).item()
    except Exception:
        return str(value)
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    if isinstance(scalar, (bytes, bytearray, memoryview)):
        try:
            return bytes(scalar).decode("utf-8")
        except Exception:
            return str(scalar)
    if isinstance(scalar, float) and not np.isfinite(scalar):
        return None
    if isinstance(scalar, (str, int, float, bool)) or scalar is None:
        return scalar
    return str(scalar)


def _json_float(value: object) -> float | None:
    scalar = _json_scalar(value)
    try:
        if scalar is None:
            return None
        out = float(scalar)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _as_positive_int(value: object) -> int | None:
    try:
        out = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _as_positive_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0 else None


def _resolve_dimensions(root: zarr.Group, refined_group: zarr.Group | None = None) -> tuple[int, int]:
    attrs = dict(root.attrs)
    if refined_group is not None:
        attrs = {**attrs, **dict(refined_group.attrs)}
    width = (
        _as_positive_int(attrs.get("width"))
        or _as_positive_int(attrs.get("source_video_width"))
        or _as_positive_int(attrs.get("video_width"))
        or _as_positive_int(attrs.get("palette_video_width"))
    )
    height = (
        _as_positive_int(attrs.get("height"))
        or _as_positive_int(attrs.get("source_video_height"))
        or _as_positive_int(attrs.get("video_height"))
        or _as_positive_int(attrs.get("palette_video_height"))
    )
    resolution = attrs.get("source_video_resolution")
    if (width is None or height is None) and isinstance(resolution, Sequence) and len(resolution) >= 2:
        # Palette stores source_video_resolution as [height, width].
        height = height or _as_positive_int(resolution[0])
        width = width or _as_positive_int(resolution[1])
    if width is None or height is None:
        raise RuntimeError("Could not resolve source video width/height from Zarr attrs.")
    return int(width), int(height)


def _resolve_fps(root: zarr.Group, default: float = 30.0) -> float:
    attrs = dict(root.attrs)
    for key in ("source_video_fps", "fps", "frame_rate", "video_fps", "palette_video_fps"):
        value = _as_positive_float(attrs.get(key))
        if value is not None:
            return float(value)
    return float(default)


def _video_id(path: Path, *, prefix: str = "video") -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _resolve_traditional_video_path(zarr_path: Path, root: zarr.Group) -> Path:
    attrs = dict(root.attrs)
    raw_path = attrs.get("source_video_path") or attrs.get("video_path")
    if raw_path:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = zarr_path.parent / path
        return path.resolve()

    source_video = attrs.get("source_video")
    if source_video:
        recording_dir = zarr_path.parent.parent if zarr_path.parent.name == "zarr" else zarr_path.parent
        candidates = [
            recording_dir / "cams" / str(source_video),
            recording_dir / str(source_video),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()
    raise RuntimeError("Could not resolve source video path from analysis Zarr attrs.")


def _latest_refined_run(root: zarr.Group, refined_run: str | None = None) -> tuple[str, zarr.Group]:
    parent = root.get("refined_detect_runs")
    if parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")
    run_name = str(
        refined_run
        or resolve_latest_complete_run_name(parent, legacy_default=True)
        or ""
    ).strip()
    if not run_name or run_name not in parent:
        raise RuntimeError("Refined detect run not found.")
    return run_name, parent[run_name]


def _normalize_bbox_or_none(bbox_norm: Optional[Sequence[object]]) -> np.ndarray | None:
    if bbox_norm is None:
        return None
    bbox = np.asarray(bbox_norm, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(bbox)):
        raise ValueError("bbox_norm must contain four finite values or be null.")
    bbox[:2] = np.clip(bbox[:2], 0.0, 1.0)
    bbox[2:] = np.clip(bbox[2:], 0.0, 1.0)
    if float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
        raise ValueError("bbox_norm width and height must be positive.")
    return bbox


def _finite_bbox_or_none(value: object) -> list[float] | None:
    bbox = np.asarray(value, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(bbox)):
        return None
    return [float(v) for v in bbox.tolist()]


def _norm_to_xyxy(bbox: Sequence[float], *, width: int, height: int) -> list[float]:
    cx, cy, bw, bh = [float(v) for v in bbox]
    return [
        (cx - bw * 0.5) * width,
        (cy - bh * 0.5) * height,
        (cx + bw * 0.5) * width,
        (cy + bh * 0.5) * height,
    ]


def _copy_payload(payload: Mapping[str, object]) -> dict[str, object]:
    copied: dict[str, object] = {}
    for key, value in payload.items():
        copied[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    return copied


def _group_at_path(root: zarr.Group, path: str) -> zarr.Group:
    try:
        group = root[str(path).strip("/")]
    except Exception as exc:
        raise RuntimeError(f"Zarr group not found: {path}") from exc
    if not hasattr(group, "attrs"):
        raise RuntimeError(f"Zarr path is not a group: {path}")
    return group


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path_from_base(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _load_review_proxy_manifest(path: str | Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    manifest_path = Path(path).expanduser().resolve()
    payload = _read_json(manifest_path)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Review proxy manifest is not a JSON object: {manifest_path}")
    schema = str(payload.get("schema_version") or "")
    if schema != "palette.review_proxy.video.v1":
        raise RuntimeError(f"Unsupported review proxy manifest schema {schema!r}: {manifest_path}")
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise RuntimeError(f"Review proxy manifest has no clips: {manifest_path}")
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in clips:
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"Review proxy manifest clip entry is not an object: {manifest_path}")
        camera_serial = str(raw.get("camera_serial") or "")
        clip_id = str(raw.get("clip_id") or "")
        if not camera_serial or not clip_id:
            raise RuntimeError(f"Review proxy manifest clip missing camera_serial/clip_id: {raw}")
        proxy_video_path = raw.get("proxy_video_path")
        if not proxy_video_path:
            raise RuntimeError(f"Review proxy manifest clip missing proxy_video_path: {raw}")
        proxy_path = _resolve_path_from_base(manifest_path.parent, proxy_video_path)
        if not proxy_path.is_file():
            raise RuntimeError(f"Review proxy video does not exist: {proxy_path}")
        by_pair[(camera_serial, clip_id)] = {**dict(raw), "proxy_video_path": str(proxy_path)}
    return by_pair


def _resolve_clipped_collection(root: zarr.Group, collection_id: str | None) -> tuple[str, zarr.Group]:
    resolved = str(collection_id or "").strip()
    refined_parent = root.get("refined_detect_runs")
    if not resolved and refined_parent is not None:
        resolved = str(refined_parent.attrs.get("latest_collection") or "").strip()
    if not resolved:
        raise RuntimeError("No clipped refined-detect collection id was provided or recorded.")
    path = f"experiment_index/finalized_runs/{resolved}"
    return resolved, _group_at_path(root, path)


def _resolve_recording_frame_index_path(
    zarr_path: Path,
    root: zarr.Group,
    collection: zarr.Group,
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    root_value = root.attrs.get("recording_frame_index_path")
    if root_value:
        return Path(str(root_value)).expanduser().resolve()
    plan_path_value = collection.attrs.get("plan_path")
    if plan_path_value:
        plan = _read_json(Path(str(plan_path_value)).expanduser().resolve())
        if isinstance(plan, Mapping) and plan.get("recording_dir"):
            recording_dir = Path(str(plan["recording_dir"])).expanduser().resolve()
            manifest_path = recording_dir / "recording_frame_index_manifest.json"
            if manifest_path.exists():
                manifest = _read_json(manifest_path)
                raw_path = manifest.get("recording_frame_index_path") if isinstance(manifest, Mapping) else None
                frame_index_path = Path(str(raw_path or "recording_frame_index.parquet")).expanduser()
                if not frame_index_path.is_absolute():
                    frame_index_path = recording_dir / frame_index_path
                return frame_index_path.resolve()
    recording_dir = zarr_path.parent.parent if zarr_path.parent.name == "zarr" else zarr_path.parent
    return (recording_dir / "recording_frame_index.parquet").resolve()


def _frame_count_from_payload(payload: Mapping[str, object]) -> int:
    frames = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
    if frames.size == 0:
        return 0
    return int(np.max(frames)) + 1


def _load_refined_payload(
    session: VideoDetectReviewSession,
    record: FrameRecord,
) -> RefinedPayloadCacheEntry:
    cached = session.refined_cache.get(record.refined_group_path)
    if cached is not None:
        return cached
    group = _group_at_path(session.root, record.refined_group_path)
    total_frames: int | None = None
    if "instances" in group and "frame_counts" in group["instances"]:
        total_frames = int(group["instances"]["frame_counts"].shape[0])
    payload = detect_review_mod._load_dense_curated_edit_payload(  # type: ignore[attr-defined]
        group,
        total_frames=total_frames,
    )
    entry = RefinedPayloadCacheEntry(
        group=group,
        payload=dict(payload),
        total_frames=total_frames or _frame_count_from_payload(payload),
        family_path=record.refined_family_path,
        run_name=record.refined_run_name,
    )
    session.refined_cache[record.refined_group_path] = entry
    return entry


def _reload_refined_payload(session: VideoDetectReviewSession, record: FrameRecord) -> None:
    session.refined_cache.pop(record.refined_group_path, None)
    _load_refined_payload(session, record)


def _write_payload(
    session: VideoDetectReviewSession,
    record: FrameRecord,
    payload: Mapping[str, object],
    *,
    row_indices: np.ndarray,
    added: int,
    removed: int,
) -> None:
    source_context = {
        "editor": "video_detect_review_web",
        "edit_mode": "manual",
        "review_surface": session.mode,
        "manual_review_frames": int(row_indices.shape[0]),
        "manual_review_added": int(added),
        "manual_review_removed": int(removed),
    }
    if record.refined_family_path is None:
        detect_review_mod._write_dense_curated_edit_payload(  # type: ignore[attr-defined]
            session.root,
            zarr_path=session.zarr_path,
            refined_run_name=record.refined_run_name,
            payload=dict(payload),  # type: ignore[arg-type]
            row_indices=row_indices,
            command_label="video_detect_review_web",
            source_context=source_context,
        )
        return

    if "source_surface_source_detect_row_index" not in payload:
        raise RuntimeError("Nested clipped refined-detect edits require canonical sparse instances.")

    status_labels = np.asarray(payload["status_labels"], dtype=object).reshape(-1)
    bbox_norm = np.asarray(payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)
    present_mask = (status_labels == "present") & np.all(np.isfinite(bbox_norm), axis=1)
    detect_review_mod.write_curated_refined_detect_surfaces(
        session.root,
        zarr_path=Path(session.zarr_path),
        refined_family_path=record.refined_family_path,
        refined_run_name=record.refined_run_name,
        instance_frame_indices=np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)[present_mask],
        instance_bbox_norm_coords=bbox_norm[present_mask],
        instance_source_kind_labels=np.asarray(payload["source_kind_labels"], dtype=object).reshape(-1)[present_mask],
        instance_reason_labels=np.asarray(payload["reason_labels"], dtype=object).reshape(-1)[present_mask],
        instance_source_detect_row_index=np.asarray(payload["source_detect_row_index"], dtype=np.int32).reshape(-1)[present_mask],
        instance_manual_edit_flags=np.asarray(payload["manual_edit_flags"], dtype=bool).reshape(-1)[present_mask],
        instance_confidence_scores=np.asarray(payload["confidence_scores"], dtype=np.float32).reshape(-1)[present_mask],
        instance_class_ids=np.asarray(payload["class_ids"], dtype=np.int32).reshape(-1)[present_mask],
        instance_refined_row_ids=np.asarray(payload["refined_row_ids"], dtype=np.int64).reshape(-1)[present_mask],
        source_detection_source_detect_row_index=np.asarray(
            payload["source_surface_source_detect_row_index"],
            dtype=np.int32,
        ).reshape(-1),
        source_detection_frame_indices=np.asarray(
            payload["source_surface_frame_indices"],
            dtype=np.int32,
        ).reshape(-1),
        source_detection_bbox_norm_coords=np.asarray(
            payload["source_surface_bbox_norm_coords"],
            dtype=np.float64,
        ).reshape(-1, 4),
        source_detection_decision_labels=np.asarray(
            payload["source_surface_decision_labels"],
            dtype=object,
        ).reshape(-1),
        source_detection_reason_labels=np.asarray(
            payload["source_surface_reason_labels"],
            dtype=object,
        ).reshape(-1),
        source_detection_confidence_scores=np.asarray(
            payload["source_surface_confidence_scores"],
            dtype=np.float32,
        ).reshape(-1),
        source_detection_class_ids=np.asarray(
            payload["source_surface_class_ids"],
            dtype=np.int32,
        ).reshape(-1),
        source_detection_review_notes=np.asarray(
            payload["source_surface_review_notes"],
            dtype=object,
        ).reshape(-1),
        command="video_detect_review_web",
        source_context=source_context,
    )


def _resolve_source_surface_row_for_frame(payload: Mapping[str, object], *, frame: int, row_idx: int) -> int | None:
    current_source_row_index = int(np.asarray(payload["source_detect_row_index"], dtype=np.int32).reshape(-1)[row_idx])
    return detect_review_mod._resolve_source_surface_row_for_frame(  # type: ignore[attr-defined]
        dict(payload),
        frame=int(frame),
        preferred_source_detect_row_index=current_source_row_index,
    )


def _traditional_session(
    zarr_path: Path,
    root: zarr.Group,
    *,
    refined_run: str | None,
    editable: bool,
    manual_score: float,
    manual_class_id: int,
) -> VideoDetectReviewSession:
    run_name, refined_group = _latest_refined_run(root, refined_run)
    width, height = _resolve_dimensions(root, refined_group)
    video_path = _resolve_traditional_video_path(zarr_path, root)
    payload = detect_review_mod._load_dense_curated_edit_payload(refined_group)  # type: ignore[attr-defined]
    total_frames = _frame_count_from_payload(payload)
    video = VideoSource(
        video_id=_video_id(video_path),
        path=video_path,
        fps=_resolve_fps(root, default=60.0),
        width=width,
        height=height,
        frame_count=total_frames,
    )
    records = [
        FrameRecord(
            parent_frame_index=frame,
            source_frame_index=frame,
            video_id=video.video_id,
            refined_family_path=None,
            refined_run_name=run_name,
            refined_group_path=f"refined_detect_runs/{run_name}",
            recording_frame_id=frame + 1,
        )
        for frame in range(total_frames)
    ]
    session = VideoDetectReviewSession(
        zarr_path=str(zarr_path),
        root=root,
        mode="traditional",
        editable=editable,
        videos={video.video_id: video},
        frame_records=records,
        manual_score=float(manual_score),
        manual_class_id=int(manual_class_id),
    )
    session.refined_cache[f"refined_detect_runs/{run_name}"] = RefinedPayloadCacheEntry(
        group=refined_group,
        payload=dict(payload),
        total_frames=total_frames,
        family_path=None,
        run_name=run_name,
    )
    return session


def _clipped_session(
    zarr_path: Path,
    root: zarr.Group,
    *,
    collection_id: str | None,
    recording_frame_index: str | Path | None,
    review_proxy_manifest: str | Path | None,
    editable: bool,
    manual_score: float,
    manual_class_id: int,
) -> VideoDetectReviewSession:
    resolved_collection_id, collection = _resolve_clipped_collection(root, collection_id)
    selected_runs = [dict(row) for row in collection.attrs.get("selected_runs", [])]
    if not selected_runs:
        raise RuntimeError(f"Clipped collection has no selected_runs: {resolved_collection_id}")
    selected_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected_runs:
        key = (str(row.get("camera_serial") or ""), str(row.get("clip_id") or ""))
        if not all(key):
            raise RuntimeError(f"Selected run missing camera_serial/clip_id: {row}")
        selected_by_pair[key] = row

    frame_index_path = _resolve_recording_frame_index_path(
        zarr_path,
        root,
        collection,
        recording_frame_index,
    )
    parquet_file = pq.ParquetFile(frame_index_path)
    names = set(parquet_file.schema_arrow.names)
    required = {"camera_serial", "clip_id", "clip_local_frame_index", "recording_frame_id", "video_path"}
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"recording_frame_index.parquet missing required columns for video review: {missing}")
    columns = ["camera_serial", "clip_id", "clip_local_frame_index", "recording_frame_id", "video_path"]
    if "parent_frame_index" in names:
        columns.append("parent_frame_index")
    table = pq.read_table(frame_index_path, columns=columns).combine_chunks()
    if int(table.num_rows) <= 0:
        raise RuntimeError("recording_frame_index.parquet has zero rows.")

    first_selected = selected_runs[0]
    first_refined_group_path = str(first_selected.get("refined_group_path") or "").strip("/")
    first_refined_group = _group_at_path(root, first_refined_group_path) if first_refined_group_path else None
    try:
        width, height = _resolve_dimensions(root, first_refined_group)
    except RuntimeError:
        first_detect_group_path = str(first_selected.get("detect_group_path") or "").strip("/")
        first_detect_group = _group_at_path(root, first_detect_group_path) if first_detect_group_path else None
        width, height = _resolve_dimensions(root, first_detect_group)
    fps = _resolve_fps(root, default=30.0)
    videos: dict[str, VideoSource] = {}
    video_id_by_path: dict[str, str] = {}
    video_id_by_pair: dict[tuple[str, str], str] = {}
    proxy_by_pair = _load_review_proxy_manifest(review_proxy_manifest)
    for row in selected_runs:
        source = row.get("source") if isinstance(row.get("source"), Mapping) else {}
        video_path_value = source.get("video_path") if isinstance(source, Mapping) else None
        if not video_path_value:
            continue
        video_path = Path(str(video_path_value)).expanduser().resolve()
        clip_id = str(row.get("clip_id") or "")
        camera_serial = str(row.get("camera_serial") or "")
        proxy = proxy_by_pair.get((camera_serial, clip_id))
        if review_proxy_manifest is not None and proxy is None:
            raise RuntimeError(f"Review proxy manifest has no entry for camera={camera_serial!r} clip={clip_id!r}")
        media_path = Path(str(proxy["proxy_video_path"])).expanduser().resolve() if proxy else video_path
        media_kind = "review_proxy_video" if proxy else "source_video"
        video_id = _video_id(media_path, prefix=clip_id or "clip")
        video_id_by_path[str(video_path)] = video_id
        video_id_by_pair[(camera_serial, clip_id)] = video_id
        videos[video_id] = VideoSource(
            video_id=video_id,
            path=media_path,
            fps=fps,
            width=width,
            height=height,
            frame_count=_as_positive_int(proxy.get("frame_count")) if proxy else None,
            clip_id=clip_id,
            camera_serial=camera_serial,
            media_kind=media_kind,
            source_path=video_path,
            media_width=_as_positive_int(proxy.get("proxy_width")) if proxy else width,
            media_height=_as_positive_int(proxy.get("proxy_height")) if proxy else height,
        )
    if not videos:
        if proxy_by_pair:
            raise RuntimeError("Review proxy manifest was provided, but selected runs had no source video paths to map.")
        # Fall back to unique paths in the frame-index table if selected_runs came from
        # an older finalizer without nested source metadata.
        for value in table["video_path"].unique().to_pylist():
            video_path = Path(str(value)).expanduser().resolve()
            video_id = _video_id(video_path, prefix="clip")
            video_id_by_path[str(video_path)] = video_id
            videos[video_id] = VideoSource(video_id=video_id, path=video_path, fps=fps, width=width, height=height)

    records = ClippedFrameRecords(
        table=table,
        selected_by_pair=selected_by_pair,
        video_id_by_path=video_id_by_path,
        video_id_by_pair=video_id_by_pair,
    )
    return VideoDetectReviewSession(
        zarr_path=str(zarr_path),
        root=root,
        mode="clipped",
        editable=editable,
        videos=videos,
        frame_records=records,
        collection_id=resolved_collection_id,
        review_proxy_manifest=str(Path(review_proxy_manifest).expanduser().resolve()) if review_proxy_manifest else None,
        manual_score=float(manual_score),
        manual_class_id=int(manual_class_id),
    )


def resolve_video_detect_review_session(
    zarr_path: str | Path,
    *,
    collection_id: str | None = None,
    refined_run: str | None = None,
    recording_frame_index: str | Path | None = None,
    review_proxy_manifest: str | Path | None = None,
    editable: bool = False,
    manual_score: float = 1.0,
    manual_class_id: int = 0,
) -> VideoDetectReviewSession:
    archive_path = Path(zarr_path).expanduser().resolve()
    root = open_zarr_group_direct(archive_path, mode="a" if editable else "r")
    refined_parent = root.get("refined_detect_runs")
    latest_collection = (
        str(refined_parent.attrs.get("latest_collection") or "").strip()
        if refined_parent is not None
        else ""
    )
    if collection_id or latest_collection:
        return _clipped_session(
            archive_path,
            root,
            collection_id=collection_id,
            recording_frame_index=recording_frame_index,
            review_proxy_manifest=review_proxy_manifest,
            editable=editable,
            manual_score=manual_score,
            manual_class_id=manual_class_id,
        )
    return _traditional_session(
        archive_path,
        root,
        refined_run=refined_run,
        editable=editable,
        manual_score=manual_score,
        manual_class_id=manual_class_id,
    )


def review_session_summary(session: VideoDetectReviewSession) -> dict[str, object]:
    present = 0
    manual = 0
    missing_or_filtered = 0
    for cached in session.refined_cache.values():
        payload = cached.payload
        status_labels = np.asarray(payload["status_labels"], dtype=object).reshape(-1)
        manual_flags = np.asarray(payload["manual_edit_flags"], dtype=bool).reshape(-1)
        present += int(np.sum(status_labels == "present"))
        missing_or_filtered += int(np.sum(status_labels != "present"))
        manual += int(np.sum(manual_flags))
    has_loaded_counts = bool(session.refined_cache)
    return {
        "zarr_path": session.zarr_path,
        "mode": session.mode,
        "collection_id": session.collection_id,
        "review_proxy_manifest": session.review_proxy_manifest,
        "editable": bool(session.editable),
        "total_frames": int(len(session.frame_records)),
        "video_count": int(len(session.videos)),
        "loaded_refined_runs": int(len(session.refined_cache)),
        "summary_counts_scope": "loaded_refined_runs",
        "width": int(next(iter(session.videos.values())).width) if session.videos else 0,
        "height": int(next(iter(session.videos.values())).height) if session.videos else 0,
        "present_frames": present if has_loaded_counts else None,
        "missing_or_filtered_frames": missing_or_filtered if has_loaded_counts else None,
        "manual_edits": manual if has_loaded_counts else None,
    }


def video_sources_payload(session: VideoDetectReviewSession) -> list[dict[str, object]]:
    ranges_by_video = _video_frame_ranges_by_source(session)
    return [
        {
            "video_id": source.video_id,
            "path": str(source.path),
            "media_url": f"/media/{source.video_id}",
            "media_kind": source.media_kind,
            "fps": float(source.fps),
            "width": int(source.width),
            "height": int(source.height),
            "media_width": int(source.media_width) if source.media_width is not None else int(source.width),
            "media_height": int(source.media_height) if source.media_height is not None else int(source.height),
            "frame_count": int(source.frame_count) if source.frame_count is not None else None,
            "source_path": str(source.source_path) if source.source_path is not None else str(source.path),
            "clip_id": source.clip_id,
            "camera_serial": source.camera_serial,
            **ranges_by_video.get(source.video_id, {}),
        }
        for source in session.videos.values()
    ]


def _video_frame_ranges_by_source(session: VideoDetectReviewSession) -> dict[str, dict[str, int]]:
    if hasattr(session.frame_records, "video_frame_ranges"):
        return session.frame_records.video_frame_ranges()

    total_frames = int(len(session.frame_records))
    if len(session.videos) == 1:
        video = next(iter(session.videos.values()))
        source_end = max(0, int(video.frame_count) - 1) if video.frame_count is not None else max(0, total_frames - 1)
        return {
            video.video_id: {
                "parent_frame_start": 0,
                "parent_frame_end": max(0, total_frames - 1),
                "source_frame_start": 0,
                "source_frame_end": source_end,
            }
        }

    ranges: dict[str, dict[str, int]] = {}
    for record in session.frame_records:
        current = ranges.setdefault(
            record.video_id,
            {
                "parent_frame_start": int(record.parent_frame_index),
                "parent_frame_end": int(record.parent_frame_index),
                "source_frame_start": int(record.source_frame_index),
                "source_frame_end": int(record.source_frame_index),
            },
        )
        current["parent_frame_start"] = min(current["parent_frame_start"], int(record.parent_frame_index))
        current["parent_frame_end"] = max(current["parent_frame_end"], int(record.parent_frame_index))
        current["source_frame_start"] = min(current["source_frame_start"], int(record.source_frame_index))
        current["source_frame_end"] = max(current["source_frame_end"], int(record.source_frame_index))
    return ranges


def _status_payload(payload: Mapping[str, object], row_idx: int) -> dict[str, object]:
    return {
        "status_label": _json_scalar(np.asarray(payload["status_labels"], dtype=object).reshape(-1)[row_idx]),
        "source_kind_label": _json_scalar(np.asarray(payload["source_kind_labels"], dtype=object).reshape(-1)[row_idx]),
        "reason_label": _json_scalar(np.asarray(payload["reason_labels"], dtype=object).reshape(-1)[row_idx]),
        "manual_edit": bool(np.asarray(payload["manual_edit_flags"], dtype=bool).reshape(-1)[row_idx]),
        "source_detect_row_index": int(np.asarray(payload["source_detect_row_index"], dtype=np.int32).reshape(-1)[row_idx]),
        "confidence_score": _json_float(np.asarray(payload["confidence_scores"], dtype=np.float32).reshape(-1)[row_idx]),
        "class_id": int(np.asarray(payload["class_ids"], dtype=np.int32).reshape(-1)[row_idx]),
    }


def load_frame_payload(session: VideoDetectReviewSession, parent_frame_index: int) -> dict[str, object]:
    if parent_frame_index < 0 or parent_frame_index >= len(session.frame_records):
        raise IndexError("parent_frame_index is out of range.")
    record = session.frame_records[int(parent_frame_index)]
    source = session.videos[record.video_id]
    cache = _load_refined_payload(session, record)
    frame_to_row = cache.payload.get("frame_to_row")
    if not isinstance(frame_to_row, dict):
        raise RuntimeError("Refined payload is missing frame_to_row.")
    row_idx = frame_to_row.get(int(record.source_frame_index))
    if row_idx is None:
        raise RuntimeError(f"Refined run is missing source frame {record.source_frame_index}.")
    row_idx = int(row_idx)
    bbox_norm = _finite_bbox_or_none(np.asarray(cache.payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)[row_idx])
    media_width = int(source.media_width) if source.media_width is not None else int(source.width)
    media_height = int(source.media_height) if source.media_height is not None else int(source.height)
    return {
        "parent_frame_index": int(record.parent_frame_index),
        "source_frame_index": int(record.source_frame_index),
        "clip_local_frame_index": int(record.source_frame_index),
        "recording_frame_id": int(record.recording_frame_id) if record.recording_frame_id is not None else None,
        "video_id": record.video_id,
        "media_url": f"/media/{record.video_id}",
        "media_kind": source.media_kind,
        "video_time_s": float(record.source_frame_index) / float(source.fps),
        "fps": float(source.fps),
        "width": int(source.width),
        "height": int(source.height),
        "source_width": int(source.width),
        "source_height": int(source.height),
        "media_width": media_width,
        "media_height": media_height,
        "bbox_norm": bbox_norm,
        "bbox_img_xyxy": _norm_to_xyxy(bbox_norm, width=source.width, height=source.height) if bbox_norm else None,
        "bbox_media_xyxy": _norm_to_xyxy(bbox_norm, width=media_width, height=media_height) if bbox_norm else None,
        "row_idx": row_idx,
        "status": _status_payload(cache.payload, row_idx),
        "refined_group_path": record.refined_group_path,
        "refined_run_name": record.refined_run_name,
        "clip_id": record.clip_id,
        "camera_serial": record.camera_serial,
    }


def _apply_manual_edit_to_payload(
    session: VideoDetectReviewSession,
    record: FrameRecord,
    payload: Mapping[str, object],
    *,
    bbox_norm: Optional[Sequence[object]],
) -> dict[str, object]:
    frame_to_row = payload.get("frame_to_row")
    if not isinstance(frame_to_row, dict):
        raise RuntimeError("Refined payload is missing frame_to_row.")
    row_idx = frame_to_row.get(int(record.source_frame_index))
    if row_idx is None:
        raise RuntimeError(f"Refined run is missing source frame {record.source_frame_index}.")
    row_idx = int(row_idx)
    normalized_bbox = _normalize_bbox_or_none(bbox_norm)
    source_surface_row_idx = _resolve_source_surface_row_for_frame(
        payload,
        frame=record.source_frame_index,
        row_idx=row_idx,
    )

    source_detect_row_index = np.asarray(payload["source_detect_row_index"], dtype=np.int32).reshape(-1)
    detection_source = np.asarray(payload["detection_source"], dtype=np.int8).reshape(-1)
    bbox_arr = np.asarray(payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(payload["confidence_scores"], dtype=np.float32).reshape(-1)
    class_ids = np.asarray(payload["class_ids"], dtype=np.int32).reshape(-1)
    status_labels = np.asarray(payload["status_labels"], dtype=object).reshape(-1)
    source_kind_labels = np.asarray(payload["source_kind_labels"], dtype=object).reshape(-1)
    manual_edit_flags = np.asarray(payload["manual_edit_flags"], dtype=bool).reshape(-1)
    reason_labels = np.asarray(payload["reason_labels"], dtype=object).reshape(-1)

    chosen_source_row_index = (
        int(np.asarray(payload["source_surface_source_detect_row_index"], dtype=np.int32).reshape(-1)[source_surface_row_idx])
        if source_surface_row_idx is not None and "source_surface_source_detect_row_index" in payload
        else -1
    )
    source_detect_row_index[row_idx] = chosen_source_row_index
    detection_source[row_idx] = 0

    if normalized_bbox is None:
        bbox_arr[row_idx] = np.full((4,), np.nan, dtype=np.float64)
        scores[row_idx] = np.float32(np.nan)
        class_ids[row_idx] = np.int32(-1)
        status_labels[row_idx] = "filtered_out"
        source_kind_labels[row_idx] = "none"
        manual_edit_flags[row_idx] = True
        reason_labels[row_idx] = "manual_clear"
        action = "manual_clear"
        added = 0
        removed = 1
        if source_surface_row_idx is not None:
            np.asarray(payload["source_surface_decision_labels"], dtype=object).reshape(-1)[source_surface_row_idx] = "manual_clear"
            np.asarray(payload["source_surface_reason_labels"], dtype=object).reshape(-1)[source_surface_row_idx] = "manual_clear"
    else:
        bbox_arr[row_idx] = normalized_bbox
        scores[row_idx] = np.float32(session.manual_score)
        class_ids[row_idx] = np.int32(session.manual_class_id)
        status_labels[row_idx] = "present"
        source_kind_labels[row_idx] = "manual"
        manual_edit_flags[row_idx] = True
        reason_labels[row_idx] = "manual_correction"
        action = "manual_correction"
        added = 1
        removed = 0
        if source_surface_row_idx is not None:
            np.asarray(payload["source_surface_decision_labels"], dtype=object).reshape(-1)[source_surface_row_idx] = "accepted"
            np.asarray(payload["source_surface_reason_labels"], dtype=object).reshape(-1)[source_surface_row_idx] = "manual_correction"

    return {
        "action": action,
        "row_idx": row_idx,
        "added": added,
        "removed": removed,
    }


def _manual_edit_result(session: VideoDetectReviewSession, *, parent_frame_index: int, action: str) -> dict[str, object]:
    payload = load_frame_payload(session, parent_frame_index)
    return {
        "action": action,
        "parent_frame_index": int(parent_frame_index),
        "source_frame_index": int(payload.get("source_frame_index", parent_frame_index)),
        "bbox_norm": payload["bbox_norm"],
        "status": payload["status"],
    }


def apply_manual_edit(
    session: VideoDetectReviewSession,
    *,
    parent_frame_index: int,
    bbox_norm: Optional[Sequence[object]],
) -> dict[str, object]:
    if not session.editable:
        raise RuntimeError("This review server was started read-only. Restart with --edit to persist changes.")
    if parent_frame_index < 0 or parent_frame_index >= len(session.frame_records):
        raise IndexError("parent_frame_index is out of range.")
    record = session.frame_records[int(parent_frame_index)]
    cache = _load_refined_payload(session, record)
    updated = _copy_payload(cache.payload)
    mutation = _apply_manual_edit_to_payload(
        session,
        record,
        updated,
        bbox_norm=bbox_norm,
    )
    _write_payload(
        session,
        record,
        updated,
        row_indices=np.asarray([int(mutation["row_idx"])], dtype=np.int32),
        added=int(mutation["added"]),
        removed=int(mutation["removed"]),
    )
    _reload_refined_payload(session, record)
    return _manual_edit_result(session, parent_frame_index=int(parent_frame_index), action=str(mutation["action"]))


def apply_manual_edits_batch(
    session: VideoDetectReviewSession,
    *,
    edits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply pending manual bbox edits with one analysis write per refined group."""
    if not session.editable:
        raise RuntimeError("This review server was started read-only. Restart with --edit to persist changes.")
    if not isinstance(edits, Sequence):
        raise ValueError("edits must be a sequence of {frame, bbox_norm} objects.")

    normalized: list[dict[str, object]] = []
    seen: set[int] = set()
    for index, raw in enumerate(edits):
        if not isinstance(raw, Mapping):
            raise ValueError(f"edit at index {index} is not an object.")
        try:
            frame = int(raw.get("frame"))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"edit at index {index} has invalid frame.") from exc
        if frame in seen:
            raise ValueError(f"duplicate frame in batch edit request: {frame}")
        if frame < 0 or frame >= len(session.frame_records):
            raise IndexError(f"parent_frame_index is out of range: {frame}")
        seen.add(frame)
        normalized.append(
            {
                "index": int(index),
                "frame": int(frame),
                "bbox_norm": raw.get("bbox_norm"),
                "record": session.frame_records[frame],
            }
        )

    groups: dict[str, list[dict[str, object]]] = {}
    for item in normalized:
        record = item["record"]
        assert isinstance(record, FrameRecord)
        groups.setdefault(record.refined_group_path, []).append(item)

    results: list[dict[str, object] | None] = [None] * len(normalized)
    group_timings: list[dict[str, object]] = []
    for group_key, group_items in groups.items():
        group_started = time.perf_counter()
        first_record = group_items[0]["record"]
        assert isinstance(first_record, FrameRecord)
        cache = _load_refined_payload(session, first_record)
        updated = _copy_payload(cache.payload)
        successes: list[dict[str, object]] = []
        added = 0
        removed = 0
        for item in group_items:
            record = item["record"]
            assert isinstance(record, FrameRecord)
            frame = int(item["frame"])
            try:
                mutation = _apply_manual_edit_to_payload(
                    session,
                    record,
                    updated,
                    bbox_norm=item["bbox_norm"],  # type: ignore[arg-type]
                )
            except Exception as exc:
                results[int(item["index"])] = {
                    "ok": False,
                    "frame": frame,
                    "error": "save_failed",
                    "details": str(exc),
                }
                continue
            successes.append({"item": item, "mutation": mutation})
            added += int(mutation["added"])
            removed += int(mutation["removed"])

        write_error: Exception | None = None
        if successes:
            try:
                _write_payload(
                    session,
                    first_record,
                    updated,
                    row_indices=np.asarray(
                        [int(entry["mutation"]["row_idx"]) for entry in successes],
                        dtype=np.int32,
                    ),
                    added=added,
                    removed=removed,
                )
                _reload_refined_payload(session, first_record)
            except Exception as exc:
                write_error = exc

        elapsed = time.perf_counter() - group_started
        group_timings.append(
            {
                "refined_group_path": group_key,
                "frames": [int(entry["item"]["frame"]) for entry in successes],
                "saved": 0 if write_error is not None else len(successes),
                "failed": len(group_items) - len(successes) + (len(successes) if write_error is not None else 0),
                "analysis_write_s": elapsed,
            }
        )
        for entry in successes:
            item = entry["item"]
            frame = int(item["frame"])
            if write_error is not None:
                results[int(item["index"])] = {
                    "ok": False,
                    "frame": frame,
                    "error": "save_failed",
                    "details": str(write_error),
                }
                continue
            mutation = entry["mutation"]
            results[int(item["index"])] = {
                "ok": True,
                "frame": frame,
                "result": _manual_edit_result(
                    session,
                    parent_frame_index=frame,
                    action=str(mutation["action"]),
                ),
            }

    return {
        "items": [item for item in results if item is not None],
        "groups": group_timings,
    }
