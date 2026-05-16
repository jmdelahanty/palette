from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import zarr
from rich.console import Console
from ultralytics import YOLO, __version__ as ultralytics_version

from ..registry.db import RegistryPaths
from ..pose.heading import compute_heading_from_attrs
from ..pose.schema import resolve_keypoint_labels_from_attrs
from ..shared.crop_image_source import CropImageSource
from ..shared.detect_reason_codec import read_reason_labels
from ..shared.provenance_attrs import build_source_crop_snapshot_attrs, build_source_roi_pixel_attrs
from ..shared.registry_stage_complete import emit_stage_completion
from ..shared.type_conversions import normalize_attr as _as_text
from ..shared.zarr.schema import get_run_group
from ..detection.detect_keypoints_yolo import (
    _extract_keypoint_confidences,
    _prepare_refined_roi_overrides,
    _repeat_to_rgb,
    _select_detection,
)

_KEYPOINT_STEP_NAME = "keypoints"
_KEYPOINT_STATUS_SOURCE = "runtime_keypoints_retry"


def _excluded_reason_tags(*, include_fish_present_no_keypoints: bool) -> tuple[str, ...]:
    tags = {"detection_issue"}
    if not include_fish_present_no_keypoints:
        tags.add("fish_present_no_keypoints")
    return tuple(sorted(tags))


def _resolve_registry_path(registry: Optional[Path]) -> Optional[Path]:
    if registry is not None:
        return registry.expanduser().resolve()
    inferred = RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    if not inferred.exists():
        return None
    return inferred


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    refined = root.get("refined_keypoints_runs")
    if refined is not None:
        return refined
    return root.get("keypoints_refined_runs")


def _load_raw_failure_indices(refined: zarr.Group) -> np.ndarray:
    if "refined_success" in refined:
        success = np.asarray(refined["refined_success"][:], dtype=bool)
        return np.where(~success)[0].astype("i4", copy=False)
    if "source_success" in refined:
        success = np.asarray(refined["source_success"][:], dtype=bool)
        return np.where(~success)[0].astype("i4", copy=False)
    if "failure_indices" in refined:
        return np.asarray(refined["failure_indices"][:], dtype="i4")
    return np.zeros(0, dtype="i4")


def _resolve_source_keypoints_run(
    keypoints_parent: zarr.Group,
    source_keypoints_run: Optional[str],
) -> str:
    if source_keypoints_run:
        if source_keypoints_run not in keypoints_parent:
            raise ValueError(f"Source keypoint run not found: {source_keypoints_run}")
        return str(source_keypoints_run)
    latest = _as_text(keypoints_parent.attrs.get("latest"))
    if latest and latest in keypoints_parent:
        return latest
    raise ValueError("No source keypoint run available under keypoints_runs.")


def _resolve_refined_run_for_source(
    refined_parent: zarr.Group,
    source_keypoints_run: str,
    refined_run: Optional[str],
) -> str:
    if refined_run:
        if refined_run not in refined_parent:
            raise ValueError(f"Refined keypoint run not found: {refined_run}")
        return str(refined_run)

    latest = _as_text(refined_parent.attrs.get("latest"))
    if latest and latest in refined_parent:
        latest_group = refined_parent[latest]
        if _as_text(latest_group.attrs.get("source_keypoints_run")) == source_keypoints_run:
            return latest

    matches: list[str] = []
    for run_name in refined_parent.group_keys():
        run_group = refined_parent[run_name]
        if _as_text(run_group.attrs.get("source_keypoints_run")) == source_keypoints_run:
            matches.append(str(run_name))
    if not matches:
        raise ValueError(
            "No refined keypoint run found for source keypoint run "
            f"'{source_keypoints_run}'."
        )
    return sorted(matches)[-1]


def _filter_retry_indices(
    refined: zarr.Group,
    raw_failures: np.ndarray,
    *,
    excluded_reason_tags: Sequence[str],
) -> np.ndarray:
    if raw_failures.size == 0:
        return raw_failures

    reason_vals = read_reason_labels(refined)
    if reason_vals is None:
        return raw_failures
    reason_vals = np.asarray(reason_vals, dtype=object)
    if reason_vals.size == 0:
        return raw_failures

    excluded = {str(tag) for tag in excluded_reason_tags if str(tag)}

    keep: list[bool] = []
    for idx in raw_failures:
        try:
            text = str(reason_vals[int(idx)]) if reason_vals[int(idx)] is not None else ""
        except Exception:
            text = ""
        tags = {token.strip() for token in text.split("|") if token.strip()}
        keep.append(tags.isdisjoint(excluded))
    return raw_failures[np.asarray(keep, dtype=bool)]


def _hash_indices(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype=np.int64)
    if values.size == 0:
        return "none"
    return hashlib.sha256(values.tobytes()).hexdigest()


def _build_retry_fingerprint(
    *,
    source_keypoints_run: str,
    source_refined_run: str,
    model_path: str,
    target_indices: np.ndarray,
    parameters: Dict[str, Any],
) -> str:
    payload = {
        "source_keypoints_run": source_keypoints_run,
        "source_refined_run": source_refined_run,
        "model_path": model_path,
        "target_indices_sha256": _hash_indices(target_indices),
        "parameters": parameters,
        "policy": "replace_on_success_only",
        "selector": "failed_only",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _find_existing_retry_run(
    keypoints_parent: zarr.Group,
    *,
    retry_fingerprint: str,
) -> Optional[str]:
    for run_name in sorted(keypoints_parent.group_keys(), reverse=True):
        attrs = keypoints_parent[run_name].attrs
        if _as_text(attrs.get("retry_fingerprint")) == retry_fingerprint:
            return str(run_name)
    return None


def _coverage_pct_from_attrs(attrs: Dict[str, Any]) -> Optional[float]:
    summary = attrs.get("summary_statistics")
    if isinstance(summary, dict) and summary.get("success_rate_percent") is not None:
        try:
            return float(summary["success_rate_percent"])
        except (TypeError, ValueError):
            pass
    success_rate = attrs.get("success_rate")
    if success_rate is None:
        return None
    try:
        value = float(success_rate)
    except (TypeError, ValueError):
        return None
    return value * 100.0 if value <= 1.0 else value


def _copy_run_arrays(source: zarr.Group, dest: zarr.Group) -> None:
    for name in source.array_keys():
        src_arr = source[name]
        chunks = getattr(src_arr, "chunks", None)
        kwargs: Dict[str, Any] = {"overwrite": True}
        if chunks is not None:
            kwargs["chunks"] = chunks
        dest.create_array(name, data=src_arr[:], **kwargs)


def _resolve_retry_keypoint_contract(
    source_run: zarr.Group,
    retry_group: zarr.Group,
) -> tuple[tuple[str, ...], int]:
    keypoints_roi = retry_group.get("keypoints_roi")
    if keypoints_roi is None or len(keypoints_roi.shape) < 3:
        raise ValueError("Retry keypoint run is missing a valid 'keypoints_roi' array.")
    expected_keypoint_count = int(keypoints_roi.shape[1])
    labels = resolve_keypoint_labels_from_attrs(
        source_run.attrs,
        keypoint_count=expected_keypoint_count,
    )
    if not labels:
        if expected_keypoint_count == 3:
            labels = ("swim_bladder", "eye_left", "eye_right")
        else:
            raise ValueError(
                "Source keypoint run is missing keypoint label metadata needed for retry "
                "(expected keypoint_labels or pose_schema nodes)."
            )

    for name in ("keypoints_img", "keypoints_norm"):
        arr = retry_group.get(name)
        if arr is not None and len(arr.shape) >= 3 and int(arr.shape[1]) != expected_keypoint_count:
            raise ValueError(
                f"Retry keypoint run array '{name}' has keypoint count {int(arr.shape[1])}, "
                f"expected {expected_keypoint_count}."
            )
    keypoint_confidences = retry_group.get("keypoint_confidences")
    if (
        keypoint_confidences is not None
        and len(keypoint_confidences.shape) >= 2
        and int(keypoint_confidences.shape[1]) != expected_keypoint_count
    ):
        raise ValueError(
            "Retry keypoint run 'keypoint_confidences' shape does not match 'keypoints_roi'."
        )

    return tuple(labels), expected_keypoint_count


def _ensure_retry_mask(dest: zarr.Group, name: str, total_rois: int, chunk_len: int) -> zarr.Array:
    if name in dest:
        arr = dest[name]
        if arr.shape == (total_rois,) and str(arr.dtype) == "bool":
            arr[:] = False
            return arr
    return dest.create_array(
        name,
        shape=(total_rois,),
        chunks=(chunk_len,),
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )


def _emit_retry_step_status(
    *,
    root: zarr.Group,
    zarr_path: Path,
    run_name: str,
    method: Optional[str],
    coverage_pct: Optional[float],
    details: Dict[str, object],
    console: Optional[Console],
    registry: Optional[Path],
) -> None:
    registry_path = _resolve_registry_path(registry)
    if registry_path is None:
        return
    emit_stage_completion(
        root,
        zarr_path,
        step_name=_KEYPOINT_STEP_NAME,
        status="ok",
        source=_KEYPOINT_STATUS_SOURCE,
        run_name=run_name,
        method=method,
        coverage_pct=coverage_pct,
        details_json=details,
        console=console,
        registry=registry_path,
        auto_registry_from_env=False,
        trigger_run_name=run_name,
    )


def retry_failed_keypoints_yolo(
    *,
    zarr_path: str,
    model_path: str,
    source_keypoints_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    batch_size: int = 256,
    device: Optional[str] = None,
    imgsz: Optional[int] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 1,
    verbose: bool = False,
    mask_threshold: float = 0.5,
    include_fish_present_no_keypoints: bool = False,
    force_new: bool = False,
    roi_cache_policy: str = "auto",
    roi_cache_dir: Optional[Path] = None,
    console: Optional[Console] = None,
    registry: Optional[Path] = None,
) -> Dict[str, Any]:
    console = console or Console()
    resolved_zarr = Path(zarr_path).expanduser().resolve()
    root = zarr.open_group(str(resolved_zarr), mode="a")

    keypoints_parent = root.get("keypoints_runs")
    if keypoints_parent is None:
        raise ValueError("Zarr archive is missing keypoints_runs; run keypoints first.")
    refined_parent = _pick_refined_parent(root)
    if refined_parent is None:
        raise ValueError("Zarr archive is missing refined_keypoints_runs; run refinement first.")

    source_run_name = _resolve_source_keypoints_run(keypoints_parent, source_keypoints_run)
    source_run = keypoints_parent[source_run_name]
    refined_run_name = _resolve_refined_run_for_source(refined_parent, source_run_name, refined_run)
    refined_group = refined_parent[refined_run_name]
    excluded_reason_tags = _excluded_reason_tags(
        include_fish_present_no_keypoints=include_fish_present_no_keypoints
    )

    raw_failures = _load_raw_failure_indices(refined_group)
    target_indices = _filter_retry_indices(
        refined_group,
        raw_failures,
        excluded_reason_tags=excluded_reason_tags,
    )

    if target_indices.size == 0:
        return {
            "run_name": source_run_name,
            "source_keypoints_run": source_run_name,
            "source_refined_run": refined_run_name,
            "retry_target_count": 0,
            "retry_replaced_count": 0,
            "retry_policy": "replace_on_success_only",
            "retry_selector": "failed_only",
            "updated": False,
            "created_new_run": False,
            "reused_existing": False,
            "skipped_reason": "no_retry_targets",
        }

    model_path_obj = Path(model_path).expanduser().resolve()
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model path not found: {model_path_obj}")

    params_payload = {
        "batch_size": int(batch_size),
        "imgsz": imgsz,
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "mask_threshold": float(mask_threshold),
        "device": device,
        "include_fish_present_no_keypoints": bool(include_fish_present_no_keypoints),
    }
    retry_fingerprint = _build_retry_fingerprint(
        source_keypoints_run=source_run_name,
        source_refined_run=refined_run_name,
        model_path=str(model_path_obj),
        target_indices=target_indices,
        parameters=params_payload,
    )

    existing_run = None if force_new else _find_existing_retry_run(
        keypoints_parent,
        retry_fingerprint=retry_fingerprint,
    )
    if existing_run is not None:
        existing_group = keypoints_parent[existing_run]
        existing_attrs = dict(existing_group.attrs)
        if "re_predicted_used" in existing_group:
            replaced_count = int(np.count_nonzero(np.asarray(existing_group["re_predicted_used"][:], dtype=bool)))
        else:
            try:
                replaced_count = int(existing_attrs.get("retry_replaced_count", 0))
            except (TypeError, ValueError):
                replaced_count = 0
        try:
            target_count = int(existing_attrs.get("retry_target_count", int(target_indices.size)))
        except (TypeError, ValueError):
            target_count = int(target_indices.size)
        status_details: Dict[str, object] = {
            "reason": "present",
            "run_group": "keypoints_runs",
            "source_keypoints_run": source_run_name,
            "source_refined_run": refined_run_name,
            "retry_selector": "failed_only",
            "retry_policy": "replace_on_success_only",
            "retry_target_count": int(target_count),
            "retry_replaced_count": int(replaced_count),
            "reused_existing": True,
        }
        keypoints_parent.attrs["latest"] = existing_run
        _emit_retry_step_status(
            root=root,
            zarr_path=resolved_zarr,
            run_name=existing_run,
            method=_as_text(existing_attrs.get("method")) or "yolo_pose",
            coverage_pct=_coverage_pct_from_attrs(existing_attrs),
            details=status_details,
            console=console,
            registry=registry,
        )
        return {
            "run_name": existing_run,
            "source_keypoints_run": source_run_name,
            "source_refined_run": refined_run_name,
            "retry_target_count": int(target_count),
            "retry_replaced_count": int(replaced_count),
            "retry_fingerprint": retry_fingerprint,
            "retry_policy": "replace_on_success_only",
            "retry_selector": "failed_only",
            "updated": False,
            "created_new_run": False,
            "reused_existing": True,
        }

    requested_crop_run = _as_text(source_run.attrs.get("source_crop_run"))
    crop_source = CropImageSource.open(
        root,
        crop_run=requested_crop_run,
        zarr_path=resolved_zarr,
        roi_cache_policy=roi_cache_policy,
        roi_cache_dir=roi_cache_dir,
        console=console,
    )
    try:
        crop_run_name = crop_source.crop_run_name
        crop_group = crop_source.crop_group
        roi_coords = np.asarray(crop_source.roi_coordinates_full, dtype=np.float64)
        total_rois = int(crop_source.total_rois)
        if np.max(target_indices, initial=-1) >= total_rois:
            raise ValueError("Retry indices exceed crop ROI count; source/refined runs are misaligned.")
        roi_h, roi_w = map(int, crop_source.roi_shape)

        override_data = _prepare_refined_roi_overrides(
            root,
            crop_group,
            total_rois,
            (roi_h, roi_w),
            console,
        )
        frame_indices_override: Optional[np.ndarray] = None
        if override_data is not None:
            idx = np.asarray(override_data["indices"], dtype=np.int64)
            roi_coords[idx] = np.asarray(override_data["coords"], dtype=np.float64)
            frame_override = override_data.get("frame_indices")
            if frame_override is not None and "frame_indices" in source_run:
                frame_indices_src = np.asarray(source_run["frame_indices"][:], dtype=np.int64)
                frame_indices_src[idx] = np.asarray(frame_override, dtype=np.int64)
                frame_indices_override = frame_indices_src.astype("i4", copy=False)

        model = YOLO(str(model_path_obj))
        if device:
            model.to(device)
        try:
            model_device = str(next(model.model.parameters()).device)
        except (AttributeError, StopIteration):
            model_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        retry_group, retry_run_name = get_run_group(root, "keypoints", console=console, create_new=True)
        _copy_run_arrays(source_run, retry_group)
        retry_group.attrs.put(dict(source_run.attrs))
        if frame_indices_override is not None:
            frame_chunks = source_run["frame_indices"].chunks if "frame_indices" in source_run else None
            kwargs: Dict[str, Any] = {"overwrite": True}
            if frame_chunks is not None:
                kwargs["chunks"] = frame_chunks
            retry_group.create_array("frame_indices", data=frame_indices_override, **kwargs)

        chunk_len = min(max(int(batch_size) * 4, 1), total_rois) if total_rois > 0 else 1
        attempted_arr = _ensure_retry_mask(retry_group, "re_predicted_attempted", total_rois, chunk_len)
        success_arr = _ensure_retry_mask(retry_group, "re_predicted_success", total_rois, chunk_len)
        used_arr = _ensure_retry_mask(retry_group, "re_predicted_used", total_rois, chunk_len)

        kp_roi_arr = retry_group["keypoints_roi"]
        kp_img_arr = retry_group["keypoints_img"]
        kp_norm_arr = retry_group["keypoints_norm"]
        heading_arr = retry_group["heading"]
        confidence_arr = retry_group["confidence"]
        keypoint_conf_arr = retry_group["keypoint_confidences"]
        keypoint_labels, expected_keypoint_count = _resolve_retry_keypoint_contract(
            source_run,
            retry_group,
        )
        detection_success_arr = retry_group["detection_success"]
        heading_finite_arr = retry_group["heading_finite"]
        heading_usable_arr = retry_group["heading_usable"]
        detection_source_arr = retry_group.get("detection_source")
        effective_threshold_arr = retry_group.get("effective_threshold")
        effective_se2_radius_arr = retry_group.get("effective_se2_radius")

        try:
            full_h, full_w = root["raw_video/images_full"].shape[1:3]
        except Exception:
            full_w = int(root.attrs.get("video_width") or root.attrs.get("width") or roi_w)
            full_h = int(root.attrs.get("video_height") or root.attrs.get("height") or roi_h)
        norm_factor = np.array([full_w, full_h], dtype=np.float64)

        attempted_flags = np.zeros(total_rois, dtype=bool)
        success_flags = np.zeros(total_rois, dtype=bool)
        used_flags = np.zeros(total_rois, dtype=bool)
        replaced_count = 0

        start_time = time.time()
        for start in range(0, int(target_indices.size), int(batch_size)):
            end = min(start + int(batch_size), int(target_indices.size))
            batch_indices = np.asarray(target_indices[start:end], dtype=np.int64)
            batch_coords = roi_coords[batch_indices]
            batch_roi_np = crop_source.read_indices(batch_indices)

            if override_data is not None:
                override_map = np.asarray(override_data["map"], dtype=np.int64)
                override_rois = np.asarray(override_data["rois"])
                local_map = override_map[batch_indices]
                valid = local_map >= 0
                if np.any(valid):
                    batch_roi_np[valid] = override_rois[local_map[valid]]

            results = model.predict(
                _repeat_to_rgb(batch_roi_np),
                imgsz=imgsz or max(roi_h, roi_w),
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=verbose,
                stream=False,
            )

            attempted_flags[batch_indices] = True
            for local_idx, (res, top_left) in enumerate(zip(results, batch_coords)):
                roi_idx = int(batch_indices[local_idx])
                det_idx = _select_detection(res)
                if det_idx is None:
                    continue
                keypoints = getattr(res, "keypoints", None)
                if keypoints is None or keypoints.xy is None:
                    continue
                kp_xy = keypoints.xy
                if kp_xy is None or kp_xy.ndim != 3 or kp_xy.shape[0] == 0:
                    continue

                kp = kp_xy[det_idx].detach().cpu().numpy()
                if int(kp.shape[0]) != expected_keypoint_count:
                    raise ValueError(
                        "YOLO retry prediction keypoint count does not match source run contract: "
                        f"predicted {int(kp.shape[0])}, expected {expected_keypoint_count} "
                        f"for keypoints_runs/{source_run_name}."
                    )

                kp[:, 0] = np.clip(kp[:, 0], 0.0, roi_w - 1)
                kp[:, 1] = np.clip(kp[:, 1], 0.0, roi_h - 1)
                top_left_arr = np.asarray(top_left, dtype=np.float64)
                kp_img = kp + np.array([top_left_arr[0], top_left_arr[1]], dtype=np.float64)
                kp_norm = kp_img / norm_factor
                heading = compute_heading_from_attrs(
                    source_run.attrs,
                    labels=keypoint_labels,
                    points=np.asarray(kp, dtype=np.float64),
                )
                kp_conf = _extract_keypoint_confidences(
                    keypoints,
                    det_idx,
                    n_keypoints=expected_keypoint_count,
                )

                boxes = getattr(res, "boxes", None)
                if boxes is not None and boxes.conf is not None and boxes.conf.numel() > 0:
                    det_conf = float(boxes.conf[det_idx].detach().cpu())
                else:
                    conf_arr = getattr(keypoints, "conf", None)
                    det_conf = float(conf_arr[det_idx].detach().cpu().mean()) if conf_arr is not None else 0.0

                kp_roi_arr[roi_idx] = kp
                kp_img_arr[roi_idx] = kp_img
                kp_norm_arr[roi_idx] = kp_norm
                heading_arr[roi_idx] = float(heading)
                confidence_arr[roi_idx] = det_conf
                keypoint_conf_arr[roi_idx] = kp_conf
                detection_success_arr[roi_idx] = True

                heading_finite = bool(np.isfinite(heading))
                heading_finite_arr[roi_idx] = heading_finite
                if detection_source_arr is not None:
                    det_src = int(detection_source_arr[roi_idx])
                else:
                    det_src = 0
                heading_usable_arr[roi_idx] = bool(heading_finite and det_src == 0)
                if effective_threshold_arr is not None:
                    effective_threshold_arr[roi_idx] = np.nan
                if effective_se2_radius_arr is not None:
                    effective_se2_radius_arr[roi_idx] = np.nan

                success_flags[roi_idx] = True
                used_flags[roi_idx] = True
                replaced_count += 1

        attempted_arr[:] = attempted_flags
        success_arr[:] = success_flags
        used_arr[:] = used_flags

        detection_success = np.asarray(detection_success_arr[:], dtype=bool)
        success_total = int(np.count_nonzero(detection_success))
        failure_total = int(total_rois - success_total)
        success_rate = (float(success_total) / float(total_rois) * 100.0) if total_rois > 0 else 0.0

        confidence_values = np.asarray(confidence_arr[:], dtype=np.float64)
        finite_conf = confidence_values[np.isfinite(confidence_values)]
        mean_conf = float(np.mean(finite_conf)) if finite_conf.size > 0 else 0.0

        if "frame_indices" in retry_group:
            frame_indices = np.asarray(retry_group["frame_indices"][:], dtype=np.int64)
            if "n_rois" in retry_group:
                total_frames = int(retry_group["n_rois"].shape[0])
            elif frame_indices.size > 0:
                total_frames = int(frame_indices.max() + 1)
            else:
                total_frames = 0
            success_counts = (
                np.bincount(frame_indices[detection_success], minlength=total_frames).astype("i4", copy=False)
                if total_frames > 0
                else np.zeros(0, dtype="i4")
            )
            chunks = (min(max(1, int(batch_size) * 4), success_counts.shape[0]),) if success_counts.size > 0 else None
            retry_group.create_array("n_keypoints", data=success_counts, chunks=chunks, overwrite=True)

        run_attrs = dict(retry_group.attrs)
        now_utc = datetime.now(timezone.utc).isoformat()
        run_attrs["keypoints_timestamp_utc"] = now_utc
        run_attrs["method"] = _as_text(run_attrs.get("method")) or "yolo_pose"
        run_attrs["model_path"] = str(model_path_obj)
        run_attrs["model_name"] = model_path_obj.name
        run_attrs["ultralytics_version"] = ultralytics_version
        run_attrs["device"] = model_device
        run_attrs["source_crop_run"] = crop_run_name
        crop_snapshot_attrs = build_source_crop_snapshot_attrs(
            crop_source.crop_group.attrs,
            source_crop_storage_mode=crop_source.storage_mode,
        )
        run_attrs.update(crop_snapshot_attrs)
        run_attrs.update(build_source_roi_pixel_attrs(crop_source))
        run_attrs["source_roi_read_mode"] = crop_source.roi_read_mode
        run_attrs["roi_cache_policy"] = crop_source.roi_cache_policy
        run_attrs["source_roi_cache_used"] = bool(crop_source.roi_cache_used)
        if crop_source.roi_cache_key:
            run_attrs["source_roi_cache_key"] = crop_source.roi_cache_key
        if crop_source.roi_cache_path:
            run_attrs["source_roi_cache_path"] = crop_source.roi_cache_path
        run_attrs["source_keypoints_run"] = source_run_name
        run_attrs["source_refined_keypoints_run"] = refined_run_name
        run_attrs["retry_fingerprint"] = retry_fingerprint
        run_attrs["retry_selector"] = "failed_only"
        run_attrs["retry_policy"] = "replace_on_success_only"
        run_attrs["retry_target_count"] = int(target_indices.size)
        run_attrs["retry_replaced_count"] = int(replaced_count)
        run_attrs["retry_excluded_reason_tags"] = list(excluded_reason_tags)
        run_attrs["keypoints_processed"] = int(total_rois)
        run_attrs["success_rate"] = round(float(success_rate), 2)
        run_attrs["inference_duration_seconds"] = float(time.time() - start_time)
        run_attrs["summary_statistics"] = {
            "total_rois": int(total_rois),
            "successful_detections": int(success_total),
            "failed_detections": int(failure_total),
            "success_rate_percent": round(float(success_rate), 2),
            "mean_confidence": mean_conf,
            "retry": {
                "selector": "failed_only",
                "policy": "replace_on_success_only",
                "source_keypoints_run": source_run_name,
                "source_refined_run": refined_run_name,
                "targeted_failures": int(target_indices.size),
                "replaced_successes": int(replaced_count),
            },
        }

        params = dict(run_attrs.get("parameters") or {})
        params.update(
            {
                "batch_size": int(batch_size),
                "imgsz": imgsz or max(roi_h, roi_w),
                "conf": float(conf),
                "iou": float(iou),
                "max_det": int(max_det),
                "mask_threshold": float(mask_threshold),
                "device": model_device,
                "retry_failed_only": True,
                "retry_selector": "failed_only",
                "retry_policy": "replace_on_success_only",
                "retry_source_keypoints_run": source_run_name,
                "retry_source_refined_run": refined_run_name,
                "roi_cache_policy": crop_source.roi_cache_policy,
            }
        )
        run_attrs["parameters"] = params
        retry_group.attrs.put(run_attrs)

        status_details: Dict[str, object] = {
            "reason": "present",
            "run_group": "keypoints_runs",
            "source_crop_run": crop_run_name,
            **crop_snapshot_attrs,
            "source_roi_read_mode": crop_source.roi_read_mode,
            "roi_cache_policy": crop_source.roi_cache_policy,
            "roi_cache_used": bool(crop_source.roi_cache_used),
            "source_keypoints_run": source_run_name,
            "source_refined_run": refined_run_name,
            "retry_selector": "failed_only",
            "retry_policy": "replace_on_success_only",
            "retry_target_count": int(target_indices.size),
            "retry_replaced_count": int(replaced_count),
            "total_rois": int(total_rois),
            "successful_detections": int(success_total),
            "failed_detections": int(failure_total),
            "success_rate_percent": round(float(success_rate), 2),
        }
        _emit_retry_step_status(
            root=root,
            zarr_path=resolved_zarr,
            run_name=retry_run_name,
            method=_as_text(run_attrs.get("method")) or "yolo_pose",
            coverage_pct=float(success_rate),
            details=status_details,
            console=console,
            registry=registry,
        )

        return {
            "run_name": retry_run_name,
            "source_keypoints_run": source_run_name,
            "source_refined_run": refined_run_name,
            "retry_target_count": int(target_indices.size),
            "retry_replaced_count": int(replaced_count),
            "retry_fingerprint": retry_fingerprint,
            "retry_policy": "replace_on_success_only",
            "retry_selector": "failed_only",
            "updated": True,
            "created_new_run": True,
            "reused_existing": False,
        }
    finally:
        crop_source.close()


__all__ = ["retry_failed_keypoints_yolo"]
