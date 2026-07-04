"""Runtime session containers and payload helpers for the labeling web UI."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .web_responses import _decode_uint8_payload, _raw_array_payload


@dataclass
class KeypointRuntimeSession:
    session_id: str
    task_id: str
    recording_id: str
    user: str
    review_session: Any
    position: int = 0
    filter_mode: str = "failed"
    search: str = ""
    review_method: str = "manual"
    review_intended_use: str | None = None
    review_notes: str | None = None
    auto_advance_on_save: bool = False
    target_token: str | None = None
    target_token_position: int | None = None

@dataclass
class DetectRuntimeSession:
    session_id: str
    task_id: str
    recording_id: str
    user: str
    review_session: Any
    position: int = 0
    auto_advance_on_save: bool = False
    target_token: str | None = None
    target_token_position: int | None = None

@dataclass
class DetectAnalysisPromotionConfig:
    training_zarr: str
    target_crop_run: str | None = None
    target_refined_run: str | None = None
    label_origin: str = "palette_labeling_work"
    include_negative: bool = True
    allow_unreviewed_negative: bool = False
    target_size: tuple[int, int] | None = None

@dataclass
class VideoDetectRuntimeSession:
    session_id: str
    task_id: str
    recording_id: str
    user: str
    review_session: Any
    frame_indices: np.ndarray
    position: int = 0
    editable: bool = False
    auto_advance_on_save: bool = False
    promotion: DetectAnalysisPromotionConfig | None = None
    target_token: str | None = None
    target_token_position: int | None = None

@dataclass
class SubjectMaskRuntimeSession:
    session_id: str
    task_id: str
    recording_id: str
    user: str
    zarr_path: str
    root: Any
    source: Any
    refined: Any
    roi_images: Any
    component_name: str
    comp_idx: int
    roi_indices: np.ndarray
    position: int = 0
    review_method: str = "manual"
    review_intended_use: str = "training"
    review_notes: str | None = None
    auto_advance_on_save: bool = False
    target_token: str | None = None
    target_token_position: int | None = None
    component_source_stage: str = ""
    component_source_run: str = ""
    component_source_component: str = ""
    component_source_resolution: str = ""
    component_source_seed_masks_present: bool = False

def _session_scope(session: Mapping[str, object]) -> Mapping[str, object]:
    scope = session.get("scope")
    return scope if isinstance(scope, Mapping) else {}

LABELER_RUNTIME_REDACTED_KEYS = frozenset(
    {
        "zarr_path",
        "training_zarr",
        "training_zarr_path",
        "promote_training_zarr",
        "promote_training_zarr_path",
        "analysis_zarr",
        "analysis_zarr_path",
        "path",
        "source_path",
    }
)

LABELER_RUNTIME_REDACTED_KEY_SUFFIXES = ("_path", "_zarr")

LABELER_RUNTIME_SAFE_URL_PREFIXES = ("/api/sessions/",)

LABELER_REDACT_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])(?:[A-Za-z]:\\|/)[^\s\"'<>),;]+")

LABELER_REDACT_ZARR_TOKEN_RE = re.compile(r"[^\s\"'<>),;]*\.zarr[^\s\"'<>),;]*")

def _is_labeler_runtime_redacted_key(key: object) -> bool:
    text = str(key)
    return text in LABELER_RUNTIME_REDACTED_KEYS or text.endswith(LABELER_RUNTIME_REDACTED_KEY_SUFFIXES)

def _redact_labeler_path_string(value: str) -> str:
    if value.startswith(LABELER_RUNTIME_SAFE_URL_PREFIXES) and ".zarr" not in value.lower():
        return value
    redacted = LABELER_REDACT_ABSOLUTE_PATH_RE.sub("[redacted_path]", value)
    return LABELER_REDACT_ZARR_TOKEN_RE.sub("[redacted_zarr_path]", redacted)

def _redact_labeler_runtime_payload(value: object) -> object:
    if isinstance(value, Mapping):
        is_base64_raw_payload = str(value.get("encoding") or "") == "base64_raw"
        redacted: dict[str, object] = {}
        for key, child in value.items():
            key_text = str(key)
            if _is_labeler_runtime_redacted_key(key):
                continue
            if is_base64_raw_payload and key_text == "pixels":
                redacted[key_text] = child
            else:
                redacted[key_text] = _redact_labeler_runtime_payload(child)
        return redacted
    if isinstance(value, list):
        return [_redact_labeler_runtime_payload(child) for child in value]
    if isinstance(value, tuple):
        return [_redact_labeler_runtime_payload(child) for child in value]
    if isinstance(value, str):
        return _redact_labeler_path_string(value)
    return value

LABELER_ERROR_ABSOLUTE_PATH_RE = LABELER_REDACT_ABSOLUTE_PATH_RE

LABELER_ERROR_ZARR_TOKEN_RE = LABELER_REDACT_ZARR_TOKEN_RE

def _labeler_safe_error_details(details: object) -> str | None:
    text = str(details or "").strip()
    if not text:
        return None
    redacted = LABELER_ERROR_ABSOLUTE_PATH_RE.sub("[redacted_path]", text)
    redacted = LABELER_ERROR_ZARR_TOKEN_RE.sub("[redacted_zarr_path]", redacted)
    if redacted != text and "browser_path_redacted" not in redacted:
        redacted = f"{redacted} [browser_path_redacted]"
    return redacted

def _browser_runtime_target_token(runtime: object) -> str:
    position = int(getattr(runtime, "position", 0))
    token = getattr(runtime, "target_token", None)
    token_position = getattr(runtime, "target_token_position", None)
    if not isinstance(token, str) or not token or token_position is None or int(token_position) != position:
        token = uuid.uuid4().hex
        setattr(runtime, "target_token", token)
        setattr(runtime, "target_token_position", position)
    return token

def _keypoint_runtime_state(runtime: KeypointRuntimeSession, backend_module: Any) -> dict[str, object]:
    session = runtime.review_session
    total = int(session.failures.size)
    current: dict[str, object] = {}
    if total > 0:
        runtime.position = max(0, min(int(runtime.position), total - 1))
        roi_idx = int(session.failures[runtime.position])
        current = {
            "position": int(runtime.position),
            "roi_idx": roi_idx,
            "frame_idx": int(session.frame_indices[roi_idx]),
        }
    summary: dict[str, object]
    try:
        summary = dict(backend_module.review_session_summary(session))
    except Exception as exc:
        summary = {"error": str(exc)}
    review_status = session.refined.attrs.get("keypoint_review_status")
    return dict(_redact_labeler_runtime_payload({
        "session_id": runtime.session_id,
        "task_id": runtime.task_id,
        "recording_id": runtime.recording_id,
        "user": runtime.user,
        "zarr_path": str(session.zarr_path),
        "refined_run": str(session.refined_run),
        "crop_run": str(session.crop_run),
        "position": int(runtime.position),
        "target_token": _browser_runtime_target_token(runtime),
        "total": total,
        "filter_mode": runtime.filter_mode,
        "search": runtime.search,
        "labels": list(session.keypoint_labels),
        "current": current,
        "summary": summary,
        "review_status": dict(review_status) if isinstance(review_status, Mapping) else None,
        "auto_advance_on_save": bool(runtime.auto_advance_on_save),
    }))

def _refresh_keypoint_queue(runtime: KeypointRuntimeSession, backend_module: Any) -> None:
    runtime.review_session.failures = backend_module.filter_review_rois(
        runtime.review_session,
        filter_mode=runtime.filter_mode,
        search=runtime.search,
    )
    total = int(runtime.review_session.failures.size)
    runtime.position = 0 if total <= 0 else max(0, min(int(runtime.position), total - 1))

def _advance_keypoint(runtime: KeypointRuntimeSession, *, advance: bool) -> None:
    if not advance:
        return
    total = int(runtime.review_session.failures.size)
    if total <= 0:
        runtime.position = 0
    elif runtime.position < total - 1:
        runtime.position += 1

def _detect_runtime_state(runtime: DetectRuntimeSession, backend_module: Any) -> dict[str, object]:
    session = runtime.review_session
    total = int(session.review_rows.shape[0])
    runtime.position = 0 if total <= 0 else max(0, min(int(runtime.position), total - 1))
    current: dict[str, object] = {}
    if total > 0:
        try:
            payload = backend_module.load_frame_payload(session, position=runtime.position)
            current = {
                "position": int(runtime.position),
                "row_idx": payload.get("row_idx"),
                "frame_idx": payload.get("frame_idx"),
            }
        except Exception:
            current = {"position": int(runtime.position)}
    try:
        summary = dict(backend_module.review_session_summary(session))
    except Exception as exc:
        summary = {"error": str(exc)}
    return dict(_redact_labeler_runtime_payload({
        "session_id": runtime.session_id,
        "task_id": runtime.task_id,
        "recording_id": runtime.recording_id,
        "user": runtime.user,
        "zarr_path": str(session.zarr_path),
        "refined_run": str(session.refined_run_name),
        "position": int(runtime.position),
        "target_token": _browser_runtime_target_token(runtime),
        "total": total,
        "frame_width": int(session.width),
        "frame_height": int(session.height),
        "current": current,
        "summary": summary,
        "auto_advance_on_save": bool(runtime.auto_advance_on_save),
    }))

def _positive_bbox_size_hint(width_norm: object, height_norm: object) -> tuple[float, float] | None:
    try:
        width = float(width_norm)  # type: ignore[arg-type]
        height = float(height_norm)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0.0 or height <= 0.0:
        return None
    return min(1.0, float(width)), min(1.0, float(height))

def _profile_json_bbox_mean(profile_json: object) -> tuple[float, float] | None:
    if not profile_json:
        return None
    try:
        payload = json.loads(str(profile_json))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    geometry = payload.get("geometry_norm")
    if not isinstance(geometry, Mapping):
        return None
    width_stats = geometry.get("w")
    height_stats = geometry.get("h")
    if not isinstance(width_stats, Mapping) or not isinstance(height_stats, Mapping):
        return None
    return _positive_bbox_size_hint(width_stats.get("mean"), height_stats.get("mean"))

def _current_detect_payload_bbox_median(runtime: DetectRuntimeSession) -> tuple[float, float] | None:
    try:
        bbox = np.asarray(runtime.review_session.payload.get("bbox_norm_coords"), dtype=np.float64).reshape(-1, 4)
    except Exception:
        return None
    if bbox.size <= 0:
        return None
    sizes = bbox[:, 2:4]
    finite = np.isfinite(sizes).all(axis=1) & (sizes[:, 0] > 0.0) & (sizes[:, 1] > 0.0)
    if not bool(np.any(finite)):
        return None
    return _positive_bbox_size_hint(float(np.median(sizes[finite, 0])), float(np.median(sizes[finite, 1])))

def _detect_bbox_size_hint_payload(
    *,
    session: Mapping[str, object],
    runtime: DetectRuntimeSession,
) -> dict[str, object] | None:
    """Return a normalized fixed-size bbox hint for the browser detect editor."""

    scope = _session_scope(session)
    registry_path = str(scope.get("registry_path") or "").strip()
    dataset_id = str(session.get("dataset_id") or scope.get("dataset_id") or "").strip()

    if registry_path and dataset_id:
        try:
            with sqlite3.connect(str(Path(registry_path).expanduser())) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT w_p50, h_p50, profile_json
                    FROM detection_data_profile_latest
                    WHERE dataset_id = ?
                    LIMIT 1;
                    """,
                    (dataset_id,),
                ).fetchone()
                if row is not None:
                    size = _positive_bbox_size_hint(row["w_p50"], row["h_p50"])
                    if size is not None:
                        return {
                            "schema": "palette.web_labeling_detect_bbox_size_hint.v1",
                            "source": "dataset_detection_data_profile_latest_p50",
                            "dataset_id": dataset_id,
                            "bbox_norm_coordinate_space": "source_image_xywhn",
                            "width_norm": size[0],
                            "height_norm": size[1],
                        }
                    size = _profile_json_bbox_mean(row["profile_json"])
                    if size is not None:
                        return {
                            "schema": "palette.web_labeling_detect_bbox_size_hint.v1",
                            "source": "dataset_detection_data_profile_profile_json_mean",
                            "dataset_id": dataset_id,
                            "bbox_norm_coordinate_space": "source_image_xywhn",
                            "width_norm": size[0],
                            "height_norm": size[1],
                        }
        except Exception:
            pass

    size = _current_detect_payload_bbox_median(runtime)
    if size is not None:
        return {
            "schema": "palette.web_labeling_detect_bbox_size_hint.v1",
            "source": "current_refined_run_median",
            "bbox_norm_coordinate_space": "source_image_xywhn",
            "width_norm": size[0],
            "height_norm": size[1],
        }

    if registry_path:
        try:
            with sqlite3.connect(str(Path(registry_path).expanduser())) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT
                        AVG(w_p50) AS w_p50,
                        AVG(h_p50) AS h_p50,
                        AVG(json_extract(profile_json, '$.geometry_norm.w.mean')) AS w_mean,
                        AVG(json_extract(profile_json, '$.geometry_norm.h.mean')) AS h_mean
                    FROM detection_data_profile_latest
                    WHERE zarr_use = 'training';
                    """
                ).fetchone()
                if row is not None:
                    size = _positive_bbox_size_hint(row["w_p50"], row["h_p50"])
                    source = "global_training_detection_data_profile_latest_p50"
                    if size is None:
                        size = _positive_bbox_size_hint(row["w_mean"], row["h_mean"])
                        source = "global_training_detection_data_profile_profile_json_mean"
                    if size is not None:
                        return {
                            "schema": "palette.web_labeling_detect_bbox_size_hint.v1",
                            "source": source,
                            "bbox_norm_coordinate_space": "source_image_xywhn",
                            "width_norm": size[0],
                            "height_norm": size[1],
                        }
        except Exception:
            pass

    return None

def _get_video_detect_parent_frame(runtime: VideoDetectRuntimeSession) -> int:
    total = int(runtime.frame_indices.shape[0])
    if total <= 0:
        raise ValueError("Video detection task has no frames.")
    runtime.position = max(0, min(int(runtime.position), total - 1))
    return int(runtime.frame_indices[runtime.position])

def _video_detect_runtime_state(runtime: VideoDetectRuntimeSession, backend_module: Any) -> dict[str, object]:
    total = int(runtime.frame_indices.shape[0])
    runtime.position = 0 if total <= 0 else max(0, min(int(runtime.position), total - 1))
    parent_frame = int(runtime.frame_indices[runtime.position]) if total > 0 else None
    summary: dict[str, object]
    try:
        summary = dict(backend_module.review_session_summary(runtime.review_session))
    except Exception as exc:
        summary = {"error": str(exc)}
    videos = []
    try:
        for raw_video in backend_module.video_sources_payload(runtime.review_session):
            video = dict(raw_video)
            video_id = str(video.get("video_id") or "")
            video["media_url"] = f"/api/sessions/{runtime.session_id}/detect-analysis/media/{video_id}"
            video.pop("path", None)
            video.pop("source_path", None)
            videos.append(video)
    except Exception as exc:
        videos = [{"error": str(exc)}]
    return dict(_redact_labeler_runtime_payload({
        "session_id": runtime.session_id,
        "task_id": runtime.task_id,
        "recording_id": runtime.recording_id,
        "user": runtime.user,
        "zarr_path": str(runtime.review_session.zarr_path),
        "mode": str(getattr(runtime.review_session, "mode", "")),
        "collection_id": getattr(runtime.review_session, "collection_id", None),
        "position": int(runtime.position),
        "target_token": _browser_runtime_target_token(runtime),
        "total": total,
        "parent_frame_index": parent_frame,
        "editable": bool(runtime.editable),
        "promotion": {
            "enabled": runtime.promotion is not None,
            "training_zarr": runtime.promotion.training_zarr if runtime.promotion is not None else None,
            "target_crop_run": runtime.promotion.target_crop_run if runtime.promotion is not None else None,
            "target_refined_run": runtime.promotion.target_refined_run if runtime.promotion is not None else None,
            "label_origin": runtime.promotion.label_origin if runtime.promotion is not None else None,
        },
        "summary": summary,
        "videos": videos,
        "auto_advance_on_save": bool(runtime.auto_advance_on_save),
    }))

def _video_detect_frame_payload(runtime: VideoDetectRuntimeSession, backend_module: Any) -> dict[str, object]:
    parent_frame = _get_video_detect_parent_frame(runtime)
    payload = dict(backend_module.load_frame_payload(runtime.review_session, parent_frame))
    video_id = str(payload.get("video_id") or "")
    payload["media_url"] = f"/api/sessions/{runtime.session_id}/detect-analysis/media/{video_id}"
    payload["ok"] = True
    payload["state"] = _video_detect_runtime_state(runtime, backend_module)
    return dict(_redact_labeler_runtime_payload(payload))

def _subject_mask_target_run_path(runtime: SubjectMaskRuntimeSession) -> str:
    return f"refined_subject_masks_runs/{runtime.refined.run_name}"

def _subject_mask_source_rowset_path(runtime: SubjectMaskRuntimeSession) -> str:
    crop_run = getattr(runtime.source, "crop_run", None)
    if crop_run:
        return f"crop_runs/{crop_run}"
    source_run = getattr(runtime.source, "run_name", None)
    if source_run:
        return f"subject_mask_runs/{source_run}"
    return ""

def _subject_mask_edit_revision(runtime: SubjectMaskRuntimeSession) -> int:
    try:
        return int(runtime.refined.group.attrs.get("edit_revision") or 0)
    except (TypeError, ValueError):
        return 0

def _subject_mask_row_identity(runtime: SubjectMaskRuntimeSession, roi_idx: int) -> dict[str, object]:
    identity: dict[str, object] = {"roi_idx": int(roi_idx)}
    for name in (
        "source_crop_row_ids",
        "source_refined_row_ids",
        "source_detect_row_index",
        "instance_key",
        "frame_indices",
    ):
        if name not in runtime.refined.group:
            continue
        try:
            identity[name] = int(np.asarray(runtime.refined.group[name][int(roi_idx)]).item())
        except Exception:
            continue
    if runtime.source.frame_indices is not None:
        try:
            identity["source_frame_idx"] = int(np.asarray(runtime.source.frame_indices[int(roi_idx)]).item())
        except Exception:
            pass
    return identity

def _subject_mask_checkpoint_mask(checkpoint: Mapping[str, object]) -> np.ndarray:
    payload = checkpoint.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Subject-mask checkpoint is missing payload.")
    mask_payload = payload.get("mask")
    if not isinstance(mask_payload, Mapping):
        raise ValueError("Subject-mask checkpoint is missing mask payload.")
    mask = (_decode_uint8_payload(mask_payload) > 0).astype(np.uint8)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]
    return mask

def _subject_mask_unapplied_checkpoint_count(
    store: LabelingStore | None,
    runtime: SubjectMaskRuntimeSession,
) -> int:
    if store is None:
        return 0
    try:
        return int(
            store.count_unapplied_session_checkpoints(
                task_id=runtime.task_id,
                component_name=runtime.component_name,
            )
        )
    except Exception:
        return 0

SUBJECT_MASK_COMPLETABLE_REVIEW_STATES = frozenset({"approved", "needs_review", "rejected"})

def _subject_mask_component_review_state(runtime: SubjectMaskRuntimeSession) -> str:
    component_reviews = runtime.refined.group.attrs.get("component_review_statuses")
    if isinstance(component_reviews, Mapping):
        raw_review = component_reviews.get(runtime.component_name)
        if isinstance(raw_review, Mapping):
            state = str(raw_review.get("state") or "").strip()
            if state:
                return state
    return "pending"

def _subject_mask_component_completion_guard(runtime: SubjectMaskRuntimeSession) -> dict[str, object]:
    review_state = _subject_mask_component_review_state(runtime)
    ready = review_state in SUBJECT_MASK_COMPLETABLE_REVIEW_STATES
    return {
        "ready": ready,
        "component_name": runtime.component_name,
        "component_review_state": review_state,
        "completable_review_states": sorted(SUBJECT_MASK_COMPLETABLE_REVIEW_STATES),
        "not_ready_reason": "" if ready else "component_review_pending",
        "required_action": "" if ready else "set_component_review_status_before_completing_task",
    }

def _subject_mask_runtime_state(
    runtime: SubjectMaskRuntimeSession,
    *,
    store: LabelingStore | None = None,
) -> dict[str, object]:
    total = int(runtime.roi_indices.shape[0])
    runtime.position = 0 if total <= 0 else max(0, min(int(runtime.position), total - 1))
    current: dict[str, object] = {}
    if total > 0:
        roi_idx = int(runtime.roi_indices[runtime.position])
        current = {"position": int(runtime.position), "roi_idx": roi_idx}
        frame_indices = runtime.source.frame_indices
        if frame_indices is not None:
            try:
                current["frame_idx"] = int(np.asarray(frame_indices[roi_idx]).item())
            except Exception:
                pass
    component_reviews = runtime.refined.group.attrs.get("component_review_statuses")
    component_review = None
    if isinstance(component_reviews, Mapping):
        raw_review = component_reviews.get(runtime.component_name)
        component_review = dict(raw_review) if isinstance(raw_review, Mapping) else None
    run_review = runtime.refined.group.attrs.get("refined_subject_mask_review_status")
    unapplied_checkpoint_count = _subject_mask_unapplied_checkpoint_count(store, runtime)
    return dict(_redact_labeler_runtime_payload({
        "session_id": runtime.session_id,
        "task_id": runtime.task_id,
        "recording_id": runtime.recording_id,
        "user": runtime.user,
        "zarr_path": runtime.zarr_path,
        "source_run": str(runtime.source.run_name),
        "refined_run": str(runtime.refined.run_name),
        "component_name": runtime.component_name,
        "components": list(runtime.refined.component_names),
        "position": int(runtime.position),
        "target_token": _browser_runtime_target_token(runtime),
        "total": total,
        "current": current,
        "component_review_status": component_review,
        "run_review_status": dict(run_review) if isinstance(run_review, Mapping) else None,
        "auto_advance_on_save": bool(runtime.auto_advance_on_save),
        "edit_revision": _subject_mask_edit_revision(runtime),
        "target_run_path": _subject_mask_target_run_path(runtime),
        "source_rowset_path": _subject_mask_source_rowset_path(runtime),
        "unapplied_session_edit_count": unapplied_checkpoint_count,
        "has_unapplied_session_edits": bool(unapplied_checkpoint_count > 0),
        "component_review_completion_guard": _subject_mask_component_completion_guard(runtime),
        "component_review_completion_ready": bool(_subject_mask_component_completion_guard(runtime).get("ready")),
    }))

def _subject_mask_current_payload(
    runtime: SubjectMaskRuntimeSession,
    *,
    store: LabelingStore | None = None,
) -> dict[str, object]:
    total = int(runtime.roi_indices.shape[0])
    if total <= 0:
        raise ValueError("Subject-mask task has no ROI rows.")
    runtime.position = max(0, min(int(runtime.position), total - 1))
    roi_idx = int(runtime.roi_indices[runtime.position])
    roi_image = np.asarray(runtime.roi_images[roi_idx], dtype=np.uint8)
    mask = (np.asarray(runtime.refined.group["masks_roi"][roi_idx, runtime.comp_idx], dtype=np.uint8) > 0).astype(np.uint8)
    session_checkpoint: dict[str, object] | None = None
    if store is not None:
        checkpoint = store.get_session_checkpoint(
            task_id=runtime.task_id,
            roi_idx=roi_idx,
            component_name=runtime.component_name,
            state="active",
        )
        if checkpoint is None:
            checkpoint = store.get_session_checkpoint(
                task_id=runtime.task_id,
                roi_idx=roi_idx,
                component_name=runtime.component_name,
                state="applying",
            )
        if checkpoint is not None:
            checkpoint_mask = _subject_mask_checkpoint_mask(checkpoint)
            if tuple(checkpoint_mask.shape) != tuple(mask.shape):
                raise ValueError(
                    f"checkpoint mask shape mismatch: expected {tuple(mask.shape)}, got {tuple(checkpoint_mask.shape)}"
                )
            mask = checkpoint_mask
            session_checkpoint = {
                "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
                "state": str(checkpoint.get("state") or ""),
                "updated_at_utc": str(checkpoint.get("updated_at_utc") or ""),
                "target_edit_revision": int(checkpoint.get("target_edit_revision") or 0),
                "is_overlay": True,
            }
    frame_idx: int | None = None
    frame_indices = runtime.source.frame_indices
    if frame_indices is not None:
        try:
            frame_idx = int(np.asarray(frame_indices[roi_idx]).item())
        except Exception:
            frame_idx = None
    return {
        "ok": True,
        "roi_idx": roi_idx,
        "frame_idx": frame_idx,
        "position": int(runtime.position),
        "component_name": runtime.component_name,
        "source_run": str(runtime.source.run_name),
        "refined_run": str(runtime.refined.run_name),
        "component_source": {
            "source_stage": runtime.component_source_stage,
            "source_run": runtime.component_source_run,
            "source_component": runtime.component_source_component,
            "resolution": runtime.component_source_resolution,
            "source_seed_masks_roi_present": bool(runtime.component_source_seed_masks_present),
            "runtime_source_run": str(runtime.source.run_name),
            "target_refined_run": str(runtime.refined.run_name),
        },
        "roi_image": _raw_array_payload(roi_image),
        "mask": _raw_array_payload(mask),
        "mask_area_px": int(mask.sum()),
        "session_checkpoint": session_checkpoint,
        "state": _subject_mask_runtime_state(runtime, store=store),
    }
