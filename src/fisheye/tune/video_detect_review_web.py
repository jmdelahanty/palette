"""Browser video-backed refined detection review server."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from . import video_detect_review_backend


@dataclass(frozen=True)
class _PromotionHookConfig:
    training_zarr: str
    target_crop_run: Optional[str]
    label_origin: str
    include_negative: bool
    allow_unreviewed_negative: bool
    target_size: tuple[int, int] | None


@dataclass(frozen=True)
class _ServerConfig:
    zarr_path: str
    host: str
    port: int
    collection_id: Optional[str]
    refined_run: Optional[str]
    recording_frame_index: Optional[str]
    review_proxy_manifest: Optional[str]
    editable: bool
    manual_score: float
    manual_class_id: int
    promotion_hook: Optional[_PromotionHookConfig] = None


@dataclass
class _ServerState:
    session: "video_detect_review_backend.VideoDetectReviewSession"  # type: ignore[name-defined]
    current_frame: int = 0
    promotion_hook: Optional[_PromotionHookConfig] = None


_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}
_MEDIA_COPY_CHUNK_BYTES = 1024 * 1024
_SEARCH_TARGETS = {"missing_or_filtered", "low_confidence", "manual_edit"}
_DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.5


def _format_error(error: str, *, details: Optional[str] = None, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> dict[str, object]:
    payload: dict[str, object] = {
        "ok": False,
        "error": error,
        "status": int(status),
    }
    if details:
        payload["details"] = details
    return payload


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    raw_len = handler.headers.get("Content-Length")
    try:
        length = int(raw_len or "0")
    except ValueError:
        length = 0
    if length <= 0:
        return {}
    payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON request body must be an object.")
    return payload


def _state_payload(state: _ServerState, backend_module) -> dict[str, object]:
    return {
        "current_frame": int(state.current_frame),
        "summary": backend_module.review_session_summary(state.session),
        "videos": backend_module.video_sources_payload(state.session),
        "promotion_hook": {
            "enabled": state.promotion_hook is not None,
            "training_zarr": state.promotion_hook.training_zarr if state.promotion_hook else None,
            "target_crop_run": state.promotion_hook.target_crop_run if state.promotion_hook else None,
            "label_origin": state.promotion_hook.label_origin if state.promotion_hook else None,
        },
    }


def _parse_frame_from_path(path: str, prefix: str) -> int | None:
    if not path.startswith(prefix):
        return None
    raw = path[len(prefix) :].strip("/")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_clip_index(clip_id: str | None) -> int:
    raw = str(clip_id or "").strip()
    if raw.startswith("clip_"):
        raw = raw[len("clip_") :]
    try:
        return int(raw)
    except ValueError:
        return -1


def _query_flag(query: dict[str, Sequence[str]], key: str, *, default: bool) -> bool:
    values = query.get(key)
    if not values:
        return default
    raw = str(values[0]).strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return default


def _query_int(query: dict[str, Sequence[str]], key: str, *, default: int) -> int:
    values = query.get(key)
    if not values:
        return int(default)
    return int(str(values[0]).strip())


def _query_float(query: dict[str, Sequence[str]], key: str, *, default: float) -> float:
    values = query.get(key)
    if not values:
        return float(default)
    return float(str(values[0]).strip())


def _payload_status(payload: Mapping[str, object]) -> Mapping[str, object]:
    raw = payload.get("status")
    return raw if isinstance(raw, Mapping) else {}


def _payload_matches_search(
    payload: Mapping[str, object],
    *,
    target: str,
    low_confidence_threshold: float = _DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> bool:
    status = _payload_status(payload)
    if target == "missing_or_filtered":
        status_label = str(status.get("status_label") or "").strip()
        return payload.get("bbox_norm") is None or (bool(status_label) and status_label != "present")
    if target == "low_confidence":
        value = status.get("confidence_score")
        if value is None:
            return False
        try:
            score = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(score) and score < float(low_confidence_threshold)
    if target == "manual_edit":
        return bool(status.get("manual_edit"))
    raise ValueError(f"Unknown search target: {target}")


def _parse_save_frame_from_path(path: str) -> int | None:
    prefix = "/api/frame/"
    suffix = "/save"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    raw = path[len(prefix) : -len(suffix)].strip("/")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _promote_detection_frames(
    *,
    analysis_zarr: str,
    training_zarr: str,
    frame: int,
    refined_run: str,
    hook: _PromotionHookConfig,
) -> dict[str, object]:
    from fisheye.tune.detect_training_promotion_backend import PromotionOptions, promote_detection_frames

    return promote_detection_frames(
        analysis_zarr,
        training_zarr,
        [int(frame)],
        options=PromotionOptions(
            refined_run=str(refined_run),
            target_crop_run=hook.target_crop_run,
            label_origin=hook.label_origin,
            include_negative=hook.include_negative,
            allow_unreviewed_negative=hook.allow_unreviewed_negative,
            target_size=hook.target_size,
        ),
        apply=True,
    )


def _promote_detection_frames_batch(
    *,
    analysis_zarr: str,
    training_zarr: str,
    frames: Sequence[int],
    refined_run: str,
    hook: _PromotionHookConfig,
) -> dict[str, object]:
    from fisheye.tune.detect_training_promotion_backend import PromotionOptions, promote_detection_frames

    return promote_detection_frames(
        analysis_zarr,
        training_zarr,
        [int(frame) for frame in frames],
        options=PromotionOptions(
            refined_run=str(refined_run),
            target_crop_run=hook.target_crop_run,
            label_origin=hook.label_origin,
            include_negative=hook.include_negative,
            allow_unreviewed_negative=hook.allow_unreviewed_negative,
            target_size=hook.target_size,
        ),
        apply=True,
    )


def _promote_clipped_detection_frames(
    *,
    analysis_zarr: str,
    training_zarr: str,
    frame_context: Mapping[str, object],
    hook: _PromotionHookConfig,
) -> dict[str, object]:
    from fisheye.tune.detect_training_promotion_backend import (
        ClippedPromotionFrame,
        PromotionOptions,
        promote_clipped_detection_frames,
    )

    frame = ClippedPromotionFrame(
        parent_frame_index=int(frame_context["parent_frame_index"]),
        clip_local_frame_index=int(frame_context["clip_local_frame_index"]),
        refined_group_path=str(frame_context["refined_group_path"]),
        refined_run=str(frame_context["refined_run"]),
        collection_id=str(frame_context["collection_id"]),
        clip_id=str(frame_context["clip_id"]),
        clip_index=int(frame_context["clip_index"]),
        camera_serial=str(frame_context["camera_serial"]),
        recording_frame_id=(
            None
            if frame_context.get("recording_frame_id") is None
            else int(frame_context["recording_frame_id"])  # type: ignore[arg-type]
        ),
        source_video_path=str(frame_context["source_video_path"]),
    )
    return promote_clipped_detection_frames(
        analysis_zarr,
        training_zarr,
        [frame],
        options=PromotionOptions(
            target_crop_run=hook.target_crop_run,
            label_origin=hook.label_origin,
            include_negative=hook.include_negative,
            allow_unreviewed_negative=hook.allow_unreviewed_negative,
            target_size=hook.target_size,
        ),
        apply=True,
    )


def _promote_clipped_detection_frames_batch(
    *,
    analysis_zarr: str,
    training_zarr: str,
    frame_contexts: Sequence[Mapping[str, object]],
    hook: _PromotionHookConfig,
) -> dict[str, object]:
    from fisheye.tune.detect_training_promotion_backend import (
        ClippedPromotionFrame,
        PromotionOptions,
        promote_clipped_detection_frames,
    )

    frames = [
        ClippedPromotionFrame(
            parent_frame_index=int(frame_context["parent_frame_index"]),
            clip_local_frame_index=int(frame_context["clip_local_frame_index"]),
            refined_group_path=str(frame_context["refined_group_path"]),
            refined_run=str(frame_context["refined_run"]),
            collection_id=str(frame_context["collection_id"]),
            clip_id=str(frame_context["clip_id"]),
            clip_index=int(frame_context["clip_index"]),
            camera_serial=str(frame_context["camera_serial"]),
            recording_frame_id=(
                None
                if frame_context.get("recording_frame_id") is None
                else int(frame_context["recording_frame_id"])  # type: ignore[arg-type]
            ),
            source_video_path=str(frame_context["source_video_path"]),
        )
        for frame_context in frame_contexts
    ]
    return promote_clipped_detection_frames(
        analysis_zarr,
        training_zarr,
        frames,
        options=PromotionOptions(
            target_crop_run=hook.target_crop_run,
            label_origin=hook.label_origin,
            include_negative=hook.include_negative,
            allow_unreviewed_negative=hook.allow_unreviewed_negative,
            target_size=hook.target_size,
        ),
        apply=True,
    )


def _clipped_frame_context(state: _ServerState, *, parent_frame_index: int) -> dict[str, object]:
    record = state.session.frame_records[int(parent_frame_index)]
    if record.refined_family_path is None:
        raise RuntimeError("Clipped training promotion requires a resolved per-clip refined detect group path.")
    source = state.session.videos.get(record.video_id)
    if source is None:
        raise RuntimeError(f"No video source registered for promotion video_id={record.video_id!r}")
    source_video_path = source.source_path or source.path
    return {
        "parent_frame_index": int(parent_frame_index),
        "clip_local_frame_index": int(record.source_frame_index),
        "refined_group_path": record.refined_group_path,
        "refined_run": record.refined_run_name,
        "collection_id": state.session.collection_id or "",
        "clip_id": record.clip_id or "",
        "clip_index": _parse_clip_index(record.clip_id),
        "camera_serial": record.camera_serial or "",
        "recording_frame_id": record.recording_frame_id,
        "source_video_path": str(source_video_path),
    }


def _promotion_summary_payload(
    *,
    ok: bool,
    analysis_frame: int,
    source_frame: int,
    training_zarr: str,
    result: Mapping[str, object],
    item: Mapping[str, object] | None,
    clip_id: str | None = None,
    camera_serial: str | None = None,
) -> dict[str, object]:
    compact_result: dict[str, object] = {
        "status": result.get("status"),
        "target_crop_run": result.get("target_crop_run"),
        "action_counts": result.get("action_counts", {}),
        "item": dict(item) if item is not None else None,
    }
    if result.get("decode_groups") is not None:
        compact_result["decode_groups"] = result.get("decode_groups")
    return {
        "ok": bool(ok),
        "analysis_frame": int(analysis_frame),
        "source_frame": int(source_frame),
        "clip_id": clip_id,
        "camera_serial": camera_serial,
        "training_zarr": training_zarr,
        "target_crop_run": result.get("target_crop_run"),
        "result": compact_result,
    }


def _run_promotion_hook(state: _ServerState, *, parent_frame_index: int) -> dict[str, object] | None:
    hook = state.promotion_hook
    if hook is None:
        return None
    record = state.session.frame_records[int(parent_frame_index)]
    if state.session.mode == "traditional":
        if record.refined_family_path is not None:
            raise RuntimeError("Post-save training promotion for traditional sessions requires a top-level refined detect run.")
        result = _promote_detection_frames(
            analysis_zarr=state.session.zarr_path,
            training_zarr=hook.training_zarr,
            frame=int(record.source_frame_index),
            refined_run=record.refined_run_name,
            hook=hook,
        )
        return {
            "ok": result.get("status") == "ok",
            "analysis_frame": int(parent_frame_index),
            "source_frame": int(record.source_frame_index),
            "training_zarr": hook.training_zarr,
            "target_crop_run": result.get("target_crop_run"),
            "result": result,
        }
    if state.session.mode != "clipped":
        raise RuntimeError(f"Unsupported review session mode for training promotion: {state.session.mode!r}")
    frame_context = _clipped_frame_context(state, parent_frame_index=int(parent_frame_index))
    result = _promote_clipped_detection_frames(
        analysis_zarr=state.session.zarr_path,
        training_zarr=hook.training_zarr,
        frame_context=frame_context,
        hook=hook,
    )
    return {
        "ok": result.get("status") == "ok",
        "analysis_frame": int(parent_frame_index),
        "source_frame": int(record.source_frame_index),
        "clip_id": record.clip_id,
        "camera_serial": record.camera_serial,
        "training_zarr": hook.training_zarr,
        "target_crop_run": result.get("target_crop_run"),
        "result": result,
    }


def _run_promotion_hooks_batch(state: _ServerState, *, parent_frame_indices: Sequence[int]) -> tuple[dict[int, dict[str, object]], dict[int, dict[str, object]], dict[str, object]]:
    hook = state.promotion_hook
    if hook is None or not parent_frame_indices:
        return {}, {}, {}
    parent_frames = [int(frame) for frame in parent_frame_indices]
    started = time.perf_counter()
    promotions: dict[int, dict[str, object]] = {}
    errors: dict[int, dict[str, object]] = {}
    telemetry: dict[str, object] = {"mode": state.session.mode, "requested": len(parent_frames)}
    try:
        if state.session.mode == "traditional":
            records = [state.session.frame_records[frame] for frame in parent_frames]
            refined_runs = {str(record.refined_run_name) for record in records}
            refined_family_paths = {record.refined_family_path for record in records}
            if len(refined_runs) != 1 or refined_family_paths != {None}:
                raise RuntimeError("Batch training promotion for traditional sessions requires one top-level refined detect run.")
            source_frames = [int(record.source_frame_index) for record in records]
            result = _promote_detection_frames_batch(
                analysis_zarr=state.session.zarr_path,
                training_zarr=hook.training_zarr,
                frames=source_frames,
                refined_run=next(iter(refined_runs)),
                hook=hook,
            )
            result_items = result.get("items", [])
            if not isinstance(result_items, list):
                result_items = []
            item_by_source = {
                int(item.get("frame")): item
                for item in result_items
                if isinstance(item, Mapping) and item.get("frame") is not None
            }
            for parent_frame, record in zip(parent_frames, records):
                item = item_by_source.get(int(record.source_frame_index))
                promotions[parent_frame] = _promotion_summary_payload(
                    ok=result.get("status") == "ok",
                    analysis_frame=parent_frame,
                    source_frame=int(record.source_frame_index),
                    training_zarr=hook.training_zarr,
                    result=result,
                    item=item,
                )
        elif state.session.mode == "clipped":
            contexts = [_clipped_frame_context(state, parent_frame_index=frame) for frame in parent_frames]
            result = _promote_clipped_detection_frames_batch(
                analysis_zarr=state.session.zarr_path,
                training_zarr=hook.training_zarr,
                frame_contexts=contexts,
                hook=hook,
            )
            result_items = result.get("items", [])
            if not isinstance(result_items, list):
                result_items = []
            item_by_parent = {
                int(item.get("parent_frame")): item
                for item in result_items
                if isinstance(item, Mapping) and item.get("parent_frame") is not None
            }
            for parent_frame, context in zip(parent_frames, contexts):
                item = item_by_parent.get(int(parent_frame))
                promotions[parent_frame] = _promotion_summary_payload(
                    ok=result.get("status") == "ok",
                    analysis_frame=parent_frame,
                    source_frame=int(context["clip_local_frame_index"]),
                    clip_id=str(context.get("clip_id") or ""),
                    camera_serial=str(context.get("camera_serial") or ""),
                    training_zarr=hook.training_zarr,
                    result=result,
                    item=item,
                )
        else:
            raise RuntimeError(f"Unsupported review session mode for training promotion: {state.session.mode!r}")
        telemetry["status"] = "ok"
        telemetry["target_crop_run"] = result.get("target_crop_run")
        telemetry["action_counts"] = result.get("action_counts", {})
        timing = result.get("timing")
        if isinstance(timing, Mapping):
            telemetry["backend_total_s"] = timing.get("total_seconds")
            telemetry["backend_timing"] = dict(timing)
            for key in (
                "dataset_resize_seconds",
                "image_write_seconds",
                "payload_append_seconds",
                "existing_row_update_seconds",
                "zarr_metadata_write_seconds",
            ):
                if key in timing:
                    telemetry[key] = timing[key]
        decode_groups = result.get("decode_groups", [])
        if isinstance(decode_groups, list):
            telemetry["decode_groups"] = decode_groups
            telemetry["decode_group_count"] = len(decode_groups)
            decode_total = 0.0
            for group in decode_groups:
                if not isinstance(group, Mapping):
                    continue
                try:
                    seconds = float(group.get("seconds"))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if math.isfinite(seconds):
                    decode_total += seconds
            telemetry["decode_total_s"] = decode_total
    except Exception as exc:
        error = _format_error("promotion_failed", details=str(exc), status=HTTPStatus.BAD_REQUEST)
        errors = {frame: error for frame in parent_frames}
        telemetry["status"] = "failed"
        telemetry["error"] = str(exc)
    finally:
        telemetry["total_s"] = _elapsed(started)
    return promotions, errors, telemetry


def _save_frame_edit(
    state: _ServerState,
    backend_module,
    *,
    parent_frame_index: int,
    bbox_norm: object,
) -> dict[str, object]:
    save_started = time.perf_counter()
    analysis_started = time.perf_counter()
    result = backend_module.apply_manual_edit(
        state.session,
        parent_frame_index=int(parent_frame_index),
        bbox_norm=bbox_norm,
    )
    analysis_write_s = _elapsed(analysis_started)
    promotion: dict[str, object] | None = None
    promotion_error: dict[str, object] | None = None
    promotion_s: float | None = None
    if state.promotion_hook is not None:
        promotion_started = time.perf_counter()
        try:
            promotion = _run_promotion_hook(state, parent_frame_index=int(parent_frame_index))
        except Exception as exc:
            promotion_error = _format_error("promotion_failed", details=str(exc), status=HTTPStatus.BAD_REQUEST)
        finally:
            promotion_s = _elapsed(promotion_started)
    return {
        "result": result,
        "promotion": promotion,
        "promotion_error": promotion_error,
        "timing": {
            "analysis_write_s": analysis_write_s,
            "promotion_s": promotion_s,
            "total_save_s": _elapsed(save_started),
        },
    }


def _frame_timing_context(state: _ServerState, frame: int) -> dict[str, object]:
    try:
        record = state.session.frame_records[int(frame)]
    except Exception:
        return {}
    return {
        "source_frame_index": int(record.source_frame_index),
        "clip_id": record.clip_id,
        "camera_serial": record.camera_serial,
        "refined_group_path": record.refined_group_path,
    }


def _finite_timing_values(items: Sequence[Mapping[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        timing = item.get("timing")
        if not isinstance(timing, Mapping):
            continue
        try:
            value = float(timing.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _timing_stats(values: Sequence[float], *, prefix: str) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_total_s": None,
            f"{prefix}_mean_s": None,
            f"{prefix}_max_s": None,
        }
    return {
        f"{prefix}_total_s": float(sum(values)),
        f"{prefix}_mean_s": float(sum(values) / len(values)),
        f"{prefix}_max_s": float(max(values)),
    }


def _batch_timing_payload(items: Sequence[Mapping[str, object]], *, total_batch_s: float) -> dict[str, object]:
    timing: dict[str, object] = {"total_batch_s": float(total_batch_s)}
    timing.update(_timing_stats(_finite_timing_values(items, "analysis_write_s"), prefix="analysis_write"))
    timing.update(_timing_stats(_finite_timing_values(items, "promotion_s"), prefix="promotion"))
    timing.update(_timing_stats(_finite_timing_values(items, "total_save_s"), prefix="total_save"))

    slowest: list[dict[str, object]] = []
    for item in items:
        item_timing = item.get("timing")
        if not isinstance(item_timing, Mapping):
            continue
        try:
            total_save_s = float(item_timing.get("total_save_s"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(total_save_s):
            continue
        slowest.append(
            {
                "frame": item.get("frame"),
                "source_frame_index": item.get("source_frame_index"),
                "clip_id": item.get("clip_id"),
                "analysis_write_s": item_timing.get("analysis_write_s"),
                "promotion_s": item_timing.get("promotion_s"),
                "total_save_s": total_save_s,
            }
        )
    slowest.sort(key=lambda row: float(row["total_save_s"]), reverse=True)
    timing["slowest_frames"] = slowest[:5]

    groups: dict[tuple[str, str], int] = {}
    for item in items:
        key = (str(item.get("clip_id") or ""), str(item.get("refined_group_path") or ""))
        if not any(key):
            continue
        groups[key] = groups.get(key, 0) + 1
    timing["groups_touched"] = [
        {"clip_id": clip_id or None, "refined_group_path": refined_group_path or None, "frames": count}
        for (clip_id, refined_group_path), count in sorted(groups.items())
    ]
    return timing


def _attach_promotion_batch_timing(timing: dict[str, object], promotion_batch: Mapping[str, object] | None) -> None:
    if not promotion_batch:
        return
    timing["promotion_batch"] = dict(promotion_batch)
    for source_key, target_key in (
        ("total_s", "promotion_batch_total_s"),
        ("backend_total_s", "promotion_backend_total_s"),
        ("decode_total_s", "promotion_decode_total_s"),
        ("decode_group_count", "promotion_decode_group_count"),
        ("dataset_resize_seconds", "promotion_dataset_resize_s"),
        ("image_write_seconds", "promotion_image_write_s"),
        ("payload_append_seconds", "promotion_payload_append_s"),
        ("existing_row_update_seconds", "promotion_existing_row_update_s"),
        ("zarr_metadata_write_seconds", "promotion_zarr_metadata_write_s"),
    ):
        if source_key in promotion_batch:
            timing[target_key] = promotion_batch[source_key]
    if "decode_groups" in promotion_batch:
        timing["promotion_decode_groups"] = promotion_batch["decode_groups"]


def _log_batch_save_timing(*, summary: Mapping[str, object], timing: Mapping[str, object]) -> None:
    def fmt(value: object) -> str:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "n/a"
        return f"{numeric:.3f}s" if math.isfinite(numeric) else "n/a"

    groups = timing.get("groups_touched")
    group_count = len(groups) if isinstance(groups, list) else 0
    print(
        "[video_detect_review_web] batch_save "
        f"requested={summary.get('requested')} saved={summary.get('saved')} "
        f"failed={summary.get('failed')} promotion_failed={summary.get('promotion_failed')} "
        f"total={fmt(timing.get('total_batch_s'))} "
        f"analysis_total={fmt(timing.get('analysis_write_total_s'))} "
        f"analysis_mean={fmt(timing.get('analysis_write_mean_s'))} "
        f"analysis_max={fmt(timing.get('analysis_write_max_s'))} "
        f"promotion_total={fmt(timing.get('promotion_total_s') or timing.get('promotion_batch_total_s'))} "
        f"decode_total={fmt(timing.get('promotion_decode_total_s'))} "
        f"image_write={fmt(timing.get('promotion_image_write_s'))} "
        f"metadata_write={fmt(timing.get('promotion_zarr_metadata_write_s'))} "
        f"groups={group_count}",
        flush=True,
    )


def _save_frame_edits(state: _ServerState, backend_module, edits: object) -> dict[str, object]:
    if not isinstance(edits, list):
        raise ValueError("edits must be a list of {frame, bbox_norm} objects.")
    started = time.perf_counter()
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
        seen.add(frame)
        normalized.append({"frame": frame, "bbox_norm": raw.get("bbox_norm")})

    batch_apply = getattr(backend_module, "apply_manual_edits_batch", None)
    items: list[dict[str, object]] = []
    promotion_batch_telemetry: dict[str, object] = {}
    if callable(batch_apply):
        backend_started = time.perf_counter()
        batch_result = batch_apply(state.session, edits=normalized)
        backend_elapsed = _elapsed(backend_started)
        raw_items = batch_result.get("items", []) if isinstance(batch_result, Mapping) else []
        raw_groups = batch_result.get("groups", []) if isinstance(batch_result, Mapping) else []
        analysis_share_by_frame: dict[int, float] = {}
        if isinstance(raw_groups, list):
            for group in raw_groups:
                if not isinstance(group, Mapping):
                    continue
                frames = [int(value) for value in group.get("frames", [])] if isinstance(group.get("frames"), list) else []
                try:
                    group_seconds = float(group.get("analysis_write_s"))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if not frames or not math.isfinite(group_seconds):
                    continue
                share = group_seconds / len(frames)
                for frame in frames:
                    analysis_share_by_frame[frame] = share
        if not analysis_share_by_frame and isinstance(raw_items, list):
            ok_count = sum(1 for item in raw_items if isinstance(item, Mapping) and item.get("ok"))
            if ok_count:
                share = backend_elapsed / ok_count
                for item in raw_items:
                    if isinstance(item, Mapping) and item.get("ok"):
                        analysis_share_by_frame[int(item.get("frame", -1))] = share

        if not isinstance(raw_items, list):
            raise RuntimeError("Batch edit backend returned invalid items payload.")
        saved_frames = [
            int(item.get("frame"))
            for item in raw_items
            if isinstance(item, Mapping) and item.get("ok") and item.get("frame") is not None
        ]
        promotion_by_frame: dict[int, dict[str, object]] = {}
        promotion_error_by_frame: dict[int, dict[str, object]] = {}
        promotion_share_s: float | None = None
        if state.promotion_hook is not None and saved_frames:
            promotion_by_frame, promotion_error_by_frame, promotion_batch_telemetry = _run_promotion_hooks_batch(
                state,
                parent_frame_indices=saved_frames,
            )
            try:
                promotion_total_s = float(promotion_batch_telemetry.get("total_s"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                promotion_total_s = math.nan
            if math.isfinite(promotion_total_s):
                promotion_share_s = promotion_total_s / len(saved_frames)

        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            frame = int(raw_item.get("frame", -1))
            context = _frame_timing_context(state, frame)
            analysis_write_s = analysis_share_by_frame.get(frame)
            if not raw_item.get("ok"):
                items.append(
                    {
                        "ok": False,
                        "frame": frame,
                        **context,
                        "error": raw_item.get("error", "save_failed"),
                        "details": raw_item.get("details", "save failed"),
                        "timing": {"total_save_s": None, "analysis_write_s": None, "promotion_s": None},
                    }
                )
                continue
            promotion = promotion_by_frame.get(frame)
            promotion_error = promotion_error_by_frame.get(frame)
            promotion_s = promotion_share_s if (promotion is not None or promotion_error is not None) else None
            total_save_s = None
            if analysis_write_s is not None:
                total_save_s = float(analysis_write_s) + (float(promotion_s) if promotion_s is not None else 0.0)
            items.append(
                {
                    "ok": True,
                    "frame": frame,
                    **context,
                    "result": raw_item.get("result"),
                    "promotion": promotion,
                    "promotion_error": promotion_error,
                    "timing": {
                        "analysis_write_s": analysis_write_s,
                        "promotion_s": promotion_s,
                        "total_save_s": total_save_s,
                    },
                }
            )
    else:
        for raw in normalized:
            frame = int(raw["frame"])
            context = _frame_timing_context(state, frame)
            try:
                item = _save_frame_edit(
                    state,
                    backend_module,
                    parent_frame_index=frame,
                    bbox_norm=raw.get("bbox_norm"),
                )
                items.append({"ok": True, "frame": frame, **context, **item})
            except Exception as exc:
                items.append(
                    {
                        "ok": False,
                        "frame": frame,
                        **context,
                        "error": "save_failed",
                        "details": str(exc),
                        "timing": {"total_save_s": None, "analysis_write_s": None, "promotion_s": None},
                    }
                )

    failed = [item for item in items if not item.get("ok")]
    promotion_failed = [
        item
        for item in items
        if item.get("promotion_error")
        or (isinstance(item.get("promotion"), Mapping) and item["promotion"].get("ok") is False)  # type: ignore[index]
    ]
    summary = {
        "requested": len(normalized),
        "saved": len(items) - len(failed),
        "failed": len(failed),
        "promotion_failed": len(promotion_failed),
    }
    timing = _batch_timing_payload(items, total_batch_s=_elapsed(started))
    _attach_promotion_batch_timing(timing, promotion_batch_telemetry)
    _log_batch_save_timing(summary=summary, timing=timing)
    return {
        "items": items,
        "summary": summary,
        "timing": timing,
    }


def _parse_range_header(value: str | None, *, file_size: int) -> tuple[int, int] | None:
    if not value:
        return None
    if not value.startswith("bytes="):
        raise ValueError("Only byte ranges are supported.")
    spec = value[len("bytes=") :].split(",", 1)[0].strip()
    if "-" not in spec:
        raise ValueError("Invalid Range header.")
    start_raw, end_raw = spec.split("-", 1)
    if start_raw == "":
        suffix = int(end_raw)
        if suffix <= 0:
            raise ValueError("Invalid suffix byte range.")
        start = max(0, file_size - suffix)
        end = file_size - 1
    else:
        start = int(start_raw)
        end = int(end_raw) if end_raw else file_size - 1
    if start < 0 or end < start or start >= file_size:
        raise ValueError("Unsatisfiable byte range.")
    return start, min(end, file_size - 1)


def _make_handler(state: _ServerState, static_root: Path, backend_module):
    class VideoDetectReviewRequestHandler(BaseHTTPRequestHandler):
        server_version = "PaletteVideoDetectReviewWeb/0.1"
        sys_version = ""

        def _write_bytes(
            self,
            payload: bytes,
            *,
            status: HTTPStatus = HTTPStatus.OK,
            content_type: str = "application/octet-stream",
            extra_headers: Optional[dict[str, str]] = None,
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def _send_media_headers(
            self,
            *,
            status: HTTPStatus,
            content_type: str,
            content_length: int,
            extra_headers: Optional[dict[str, str]] = None,
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Accept-Ranges", "bytes")
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()

        def _stream_file_range(self, path: Path, *, start: int, length: int) -> None:
            remaining = int(length)
            with path.open("rb") as handle:
                handle.seek(start)
                while remaining > 0:
                    chunk = handle.read(min(_MEDIA_COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        def _write_json(self, payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            self._write_bytes(data, status=status, content_type="application/json; charset=utf-8")

        def _write_not_found(self, message: str = "Not found") -> None:
            self._write_json(_format_error(message, status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)

        def _write_bad_request(self, message: str, details: Optional[str] = None) -> None:
            self._write_json(_format_error(message, details=details, status=HTTPStatus.BAD_REQUEST), status=HTTPStatus.BAD_REQUEST)

        def _serve_static(self, relative_path: str) -> None:
            candidate = (static_root / relative_path).resolve()
            if not candidate.is_relative_to(static_root) or not candidate.is_file():
                self._write_not_found("Static asset not found.")
                return
            content_type = _CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
            self._write_bytes(
                candidate.read_bytes(),
                content_type=content_type,
                extra_headers={"Cache-Control": "no-store"},
            )

        def _serve_media(self, video_id: str) -> None:
            source = state.session.videos.get(video_id)
            if source is None:
                self._write_not_found("Video source not found.")
                return
            if not source.path.is_file():
                self._write_not_found(f"Video file not found: {source.path}")
                return
            file_size = source.path.stat().st_size
            content_type = mimetypes.guess_type(source.path.name)[0] or "video/mp4"
            try:
                byte_range = _parse_range_header(self.headers.get("Range"), file_size=file_size)
            except Exception as exc:
                self.send_response(int(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes */{file_size}")
                payload = json.dumps(
                    _format_error(
                        "Invalid Range header.",
                        details=str(exc),
                        status=HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    ),
                    allow_nan=False,
                ).encode("utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if self.command != "HEAD":
                    try:
                        self.wfile.write(payload)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                return
            start, end = byte_range if byte_range is not None else (0, file_size - 1)
            length = max(0, end - start + 1)
            headers = {}
            if byte_range is not None:
                headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            self._send_media_headers(
                status=HTTPStatus.PARTIAL_CONTENT if byte_range is not None else HTTPStatus.OK,
                content_type=content_type,
                content_length=length,
                extra_headers=headers,
            )
            if self.command != "HEAD":
                self._stream_file_range(source.path, start=start, length=length)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"", "/"}:
                self._serve_static("index.html")
                return
            if path.startswith("/static/"):
                self._serve_static(path[len("/static/") :])
                return
            if path.startswith("/media/"):
                self._serve_media(path[len("/media/") :].strip("/"))
                return
            if path == "/api/state":
                self._write_json({"ok": True, "state": _state_payload(state, backend_module)})
                return
            if path == "/api/frame/current":
                try:
                    payload = backend_module.load_frame_payload(state.session, state.current_frame)
                    payload["ok"] = True
                    payload["state"] = _state_payload(state, backend_module)
                except Exception as exc:
                    self._write_json(
                        _format_error("frame_load_error", details=str(exc), status=HTTPStatus.NOT_FOUND),
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._write_json(payload)
                return
            frame_index = _parse_frame_from_path(path, "/api/frame/")
            if frame_index is not None:
                try:
                    query = parse_qs(parsed.query)
                    if _query_flag(query, "update_current", default=True):
                        state.current_frame = frame_index
                    payload = backend_module.load_frame_payload(state.session, frame_index)
                    payload["ok"] = True
                    payload["state"] = _state_payload(state, backend_module)
                except Exception as exc:
                    self._write_json(
                        _format_error("frame_load_error", details=str(exc), status=HTTPStatus.NOT_FOUND),
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._write_json(payload)
                return
            if path == "/api/search":
                query = parse_qs(parsed.query)
                direction = str((query.get("direction") or ["next"])[0]).strip().lower()
                target = str((query.get("target") or ["missing_or_filtered"])[0]).strip().lower()
                if direction not in {"next", "prev"}:
                    self._write_bad_request("Search direction must be 'next' or 'prev'.")
                    return
                if target not in _SEARCH_TARGETS:
                    self._write_bad_request(
                        "Unknown search target.",
                        details=f"target={target!r}; valid={sorted(_SEARCH_TARGETS)!r}",
                    )
                    return
                try:
                    start = _query_int(query, "start", default=state.current_frame)
                    low_confidence_threshold = _query_float(
                        query,
                        "low_confidence_threshold",
                        default=_DEFAULT_LOW_CONFIDENCE_THRESHOLD,
                    )
                except (TypeError, ValueError) as exc:
                    self._write_bad_request("Invalid search parameter.", details=str(exc))
                    return
                if not math.isfinite(low_confidence_threshold):
                    self._write_bad_request("low_confidence_threshold must be finite.")
                    return

                total = len(state.session.frame_records)
                if total <= 0:
                    self._write_json(_format_error("No frames are available.", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
                    return
                step = -1 if direction == "prev" else 1
                candidate = min(max(0, start + step), total - 1)
                while 0 <= candidate < total:
                    try:
                        payload = backend_module.load_frame_payload(state.session, candidate)
                    except Exception:
                        candidate += step
                        continue
                    if _payload_matches_search(
                        payload,
                        target=target,
                        low_confidence_threshold=low_confidence_threshold,
                    ):
                        state.current_frame = candidate
                        payload["ok"] = True
                        payload["state"] = _state_payload(state, backend_module)
                        payload["search"] = {
                            "target": target,
                            "direction": direction,
                            "start": int(start),
                            "matched_frame": int(candidate),
                            "low_confidence_threshold": float(low_confidence_threshold),
                        }
                        self._write_json(payload)
                        return
                    candidate += step
                self._write_json(
                    _format_error(
                        "No matching frame found.",
                        details=f"target={target!r}, direction={direction!r}, start={start}",
                        status=HTTPStatus.NOT_FOUND,
                    ),
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self._write_not_found()

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = _read_json_body(self)
            except Exception as exc:
                self._write_bad_request("Invalid JSON body.", details=str(exc))
                return

            if path == "/api/nav":
                try:
                    delta = int(body.get("delta") or 0)
                    target = body.get("frame")
                    if target is not None:
                        next_frame = int(target)
                    else:
                        next_frame = state.current_frame + delta
                except (TypeError, ValueError):
                    self._write_bad_request("frame/delta must be integers.")
                    return
                total = len(state.session.frame_records)
                state.current_frame = min(max(0, next_frame), max(0, total - 1))
                self._write_json({"ok": True, "state": _state_payload(state, backend_module)})
                return

            if path == "/api/frames/save_batch":
                try:
                    batch = _save_frame_edits(state, backend_module, body.get("edits"))
                except Exception as exc:
                    self._write_bad_request("batch_save_failed", details=str(exc))
                    return
                self._write_json({"ok": True, **batch, "state": _state_payload(state, backend_module)})
                return

            save_current = path == "/api/frame/current/save"
            frame_index = _parse_save_frame_from_path(path)
            save_by_frame = frame_index is not None
            if save_current or save_by_frame:
                target_frame = state.current_frame if save_current else int(frame_index)
                try:
                    saved = _save_frame_edit(
                        state,
                        backend_module,
                        parent_frame_index=target_frame,
                        bbox_norm=body.get("bbox_norm"),
                    )
                    result = saved["result"]
                    promotion = saved["promotion"]
                    promotion_error = saved["promotion_error"]
                    timing = saved["timing"]
                    state.current_frame = target_frame
                    if bool(body.get("advance")):
                        total = len(state.session.frame_records)
                        state.current_frame = min(state.current_frame + 1, max(0, total - 1))
                except Exception as exc:
                    self._write_json(
                        _format_error("save_failed", details=str(exc), status=HTTPStatus.BAD_REQUEST),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._write_json(
                    {
                        "ok": True,
                        "result": result,
                        "promotion": promotion,
                        "promotion_error": promotion_error,
                        "timing": timing,
                        "state": _state_payload(state, backend_module),
                    }
                )
                return

            self._write_not_found()

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return VideoDetectReviewRequestHandler


def run_server(config: _ServerConfig) -> int:
    from . import video_detect_review_backend as backend_module

    session = backend_module.resolve_video_detect_review_session(
        config.zarr_path,
        collection_id=config.collection_id,
        refined_run=config.refined_run,
        recording_frame_index=config.recording_frame_index,
        review_proxy_manifest=config.review_proxy_manifest,
        editable=config.editable,
        manual_score=config.manual_score,
        manual_class_id=config.manual_class_id,
    )
    if config.promotion_hook is not None:
        if not config.editable:
            raise RuntimeError("--promote-training-zarr requires --edit because promotion runs after a saved edit.")
        if session.mode not in {"traditional", "clipped"}:
            raise RuntimeError(f"--promote-training-zarr does not support review mode {session.mode!r}.")
    state = _ServerState(session=session, current_frame=0, promotion_hook=config.promotion_hook)
    static_root = Path(__file__).resolve().parent / "video_detect_review_web" / "static"
    handler = _make_handler(state, static_root, backend_module)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    summary = backend_module.review_session_summary(session)
    print(
        f"Serving Palette video detect review at http://{config.host}:{config.port} "
        f"mode={summary['mode']} frames={summary['total_frames']} videos={summary['video_count']} "
        f"editable={summary['editable']} promotion_hook={config.promotion_hook is not None}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping video detect review server.")
    finally:
        server.server_close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a video-backed Palette detection review UI.")
    parser.add_argument("zarr_path", type=Path, help="Analysis Zarr path")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Use 0.0.0.0 only on trusted networks.")
    parser.add_argument("--port", type=int, default=8790, help="Bind port")
    parser.add_argument("--collection-id", default=None, help="Finalized clipped refined-detect collection id")
    parser.add_argument("--refined-run", default=None, help="Traditional refined detect run name")
    parser.add_argument("--recording-frame-index", type=Path, default=None, help="Override recording_frame_index.parquet")
    parser.add_argument("--review-proxy-manifest", type=Path, default=None, help="Use derived review-proxy videos for clipped media.")
    parser.add_argument("--edit", action="store_true", help="Allow saving bbox edits back into the analysis Zarr")
    parser.add_argument("--manual-score", type=float, default=1.0, help="Confidence score for manually added boxes")
    parser.add_argument("--manual-class-id", type=int, default=0, help="Class id for manually added boxes")
    parser.add_argument(
        "--promote-training-zarr",
        type=Path,
        default=None,
        help="After each successful save, promote that source frame into this per-recording training Zarr.",
    )
    parser.add_argument(
        "--promote-target-crop-run",
        default=None,
        help="Target crop_runs/<run> for post-save promotion. Defaults to the training Zarr latest crop run.",
    )
    parser.add_argument(
        "--promote-label-origin",
        default="video_detect_review_web",
        help="label_origin value written by post-save promotion.",
    )
    parser.add_argument("--promote-no-negative", action="store_true", help="Do not promote clear/no-box saves as negative rows.")
    parser.add_argument(
        "--promote-allow-unreviewed-negative",
        action="store_true",
        help="Allow non-present rows without manual_edit=True to become negative examples during promotion.",
    )
    parser.add_argument(
        "--promote-target-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Target promoted image size when the training Zarr does not already define raw_video/images_ds.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_server(
        _ServerConfig(
            zarr_path=str(args.zarr_path),
            host=str(args.host),
            port=int(args.port),
            collection_id=args.collection_id,
            refined_run=args.refined_run,
            recording_frame_index=str(args.recording_frame_index) if args.recording_frame_index else None,
            review_proxy_manifest=str(args.review_proxy_manifest) if args.review_proxy_manifest else None,
            editable=bool(args.edit),
            manual_score=float(args.manual_score),
            manual_class_id=int(args.manual_class_id),
            promotion_hook=(
                _PromotionHookConfig(
                    training_zarr=str(args.promote_training_zarr),
                    target_crop_run=args.promote_target_crop_run,
                    label_origin=str(args.promote_label_origin),
                    include_negative=not bool(args.promote_no_negative),
                    allow_unreviewed_negative=bool(args.promote_allow_unreviewed_negative),
                    target_size=tuple(args.promote_target_size) if args.promote_target_size else None,
                )
                if args.promote_training_zarr
                else None
            ),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
