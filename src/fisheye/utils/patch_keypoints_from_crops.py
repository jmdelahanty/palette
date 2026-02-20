"""Patch keypoints in-place for flagged frames using updated crops."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr

from ..detection.detect_keypoints_traditional import detect_keypoints_traditional
from ..refinement.refine_keypoints import _process_refinement_chunk
from ..shared.keypoint_stale import mark_downstream_eye_mask_runs_stale
from ..tune.keypoint_review import _update_postprocess_summary


@dataclass(frozen=True)
class PatchPlan:
    zarr_path: Path
    frames: List[int]


def _load_frame_flags(path: Path) -> Dict[str, List[int]]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Frame flag file must be a JSON object: {path}")
    out: Dict[str, List[int]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            frames: List[int] = []
            for item in value:
                if isinstance(item, dict):
                    frame_val = item.get("frame_idx")
                    if frame_val is None:
                        frame_val = item.get("frame")
                    if frame_val is None:
                        frame_val = item.get("frame_index")
                    if frame_val is None:
                        continue
                    try:
                        frames.append(int(frame_val))
                    except (TypeError, ValueError):
                        continue
                else:
                    try:
                        frames.append(int(item))
                    except (TypeError, ValueError):
                        continue
            out[str(key)] = sorted(set(frames))
    return out


def _read_file_list(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    items: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(Path(line))
    return items


def _parse_frames(text: Optional[str]) -> List[int]:
    if not text:
        return []
    frames: List[int] = []
    tokens = [tok for tok in text.replace(",", " ").split() if tok]
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) == 2:
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                except ValueError:
                    continue
                if end < start:
                    start, end = end, start
                frames.extend(list(range(start, end + 1)))
                continue
        try:
            frames.append(int(token))
        except ValueError:
            continue
    return sorted(set(frames))


def _collect_plans(
    paths: Sequence[Path],
    file_list: Optional[Path],
    frame_flags: Dict[str, List[int]],
    explicit_frames: List[int],
) -> List[PatchPlan]:
    targets: List[Path] = []
    if file_list:
        targets.extend(_read_file_list(file_list))
    if paths:
        targets.extend(paths)
    if not targets:
        targets = [Path(p) for p in frame_flags.keys()]

    plans: List[PatchPlan] = []
    for path in targets:
        path = Path(path)
        frames = explicit_frames or frame_flags.get(str(path), [])
        frames = sorted({int(f) for f in frames})
        if not frames:
            continue
        plans.append(PatchPlan(zarr_path=path, frames=frames))
    return plans


def _resolve_keypoints_run(root: zarr.Group, run_name: Optional[str]) -> Tuple[zarr.Group, str]:
    parent = root.get("keypoints_runs")
    if parent is None:
        raise RuntimeError("No keypoints_runs found in archive.")
    if run_name is None:
        run_name = parent.attrs.get("latest")
    if not run_name or run_name not in parent:
        raise RuntimeError("Keypoint run not found.")
    return parent[run_name], run_name


def _resolve_crop_run(
    root: zarr.Group,
    run_name: Optional[str],
    keypoints_group: zarr.Group,
) -> Tuple[zarr.Group, str]:
    parent = root.get("crop_runs")
    if parent is None:
        raise RuntimeError("No crop_runs found in archive.")
    if run_name is None:
        run_name = keypoints_group.attrs.get("source_crop_run") or parent.attrs.get("latest")
    if not run_name or run_name not in parent:
        raise RuntimeError("Crop run not found.")
    return parent[run_name], run_name


def _resolve_background_run(
    root: zarr.Group,
    run_name: Optional[str],
    keypoints_group: zarr.Group,
) -> Tuple[zarr.Group, str]:
    parent = root.get("background_runs")
    if parent is None:
        raise RuntimeError("No background_runs found in archive.")
    if run_name is None:
        run_name = keypoints_group.attrs.get("source_background_run") or parent.attrs.get("latest")
    if not run_name or run_name not in parent:
        raise RuntimeError("Background run not found.")
    return parent[run_name], run_name


def _load_keypoint_parameters(group: zarr.Group) -> Dict[str, object]:
    params = group.attrs.get("parameters")
    return params if isinstance(params, dict) else {}


def _update_index_array(group: zarr.Group, name: str, values: np.ndarray) -> np.ndarray:
    values = np.unique(values.astype(np.int64, copy=False))
    if name in group:
        existing = group[name][:]
        values = np.unique(np.concatenate([existing.astype(np.int64, copy=False), values]))
    group.create_array(name, data=values, chunks=(min(1024, values.size),), overwrite=True)
    return values


def _compute_keypoints_for_indices(
    roi_images: zarr.Array,
    roi_coords_full: zarr.Array,
    background_full: np.ndarray,
    indices: np.ndarray,
    params: Dict[str, object],
) -> Dict[str, np.ndarray]:
    n = int(indices.size)
    roi_shape = roi_images.shape[1:]
    full_shape = background_full.shape

    roi_out = np.full((n, 3, 2), np.nan, dtype=np.float64)
    img_out = np.full((n, 3, 2), np.nan, dtype=np.float64)
    norm_out = np.full((n, 3, 2), np.nan, dtype=np.float64)
    heading_out = np.full(n, np.nan, dtype=np.float64)
    conf_out = np.full(n, np.nan, dtype=np.float64)
    thresh_out = np.full(n, np.nan, dtype=np.float64)
    se2_out = np.full(n, np.nan, dtype=np.float64)
    success_out = np.zeros(n, dtype=bool)
    kp_conf_out = np.full((n, 3), np.nan, dtype=np.float64)
    tri_angles_out = np.full((n, 3), np.nan, dtype=np.float64)
    tri_angles_raw_out = np.full((n, 3), np.nan, dtype=np.float64)
    tri_area_out = np.full(n, np.nan, dtype=np.float64)

    for i, roi_idx in enumerate(indices):
        roi_img = roi_images[roi_idx]
        roi_coord = roi_coords_full[roi_idx]
        x1, y1 = int(roi_coord[0]), int(roi_coord[1])
        x2, y2 = x1 + roi_shape[1], y1 + roi_shape[0]
        if x1 < 0 or y1 < 0 or x2 > full_shape[1] or y2 > full_shape[0]:
            continue
        background_roi = background_full[y1:y2, x1:x2]

        keypoints = detect_keypoints_traditional(
            roi_img,
            background_roi,
            roi_thresh=int(params.get("roi_thresh", 50)),
            se1_radius=int(params.get("se1_radius", 1)),
            se2_radius=int(params.get("se2_radius", 2)),
            min_area=int(params.get("min_area", 5)),
            min_valid_angle=float(params.get("min_valid_angle", 10.0)),
            max_valid_angle=float(params.get("max_valid_angle", 90.0)),
            min_triangle_area=float(params.get("min_triangle_area", 100.0)),
            max_triangle_area=params.get("max_triangle_area"),
        )
        if keypoints is None:
            continue

        success_out[i] = True
        roi_pts = np.array(
            [keypoints["bladder"], keypoints["eye_left"], keypoints["eye_right"]],
            dtype=np.float64,
        )
        roi_out[i] = roi_pts
        img_pts = roi_pts + np.array([x1, y1], dtype=np.float64)
        img_out[i] = img_pts
        norm_factor = np.array(full_shape[::-1], dtype=np.float64)
        norm_out[i] = img_pts / norm_factor
        heading_out[i] = float(keypoints.get("heading", np.nan))
        conf_out[i] = float(keypoints.get("confidence", np.nan))
        thresh_out[i] = float(keypoints.get("effective_threshold", np.nan))
        se2_out[i] = float(keypoints.get("effective_se2_radius", np.nan))
        kp_conf_out[i] = np.asarray(keypoints.get("keypoint_confidences", [np.nan, np.nan, np.nan]), dtype=np.float64)
        tri_angles_out[i] = np.asarray(keypoints.get("triangle_angles", [np.nan, np.nan, np.nan]), dtype=np.float64)
        tri_angles_raw_out[i] = np.asarray(
            keypoints.get("triangle_angles_raw", [np.nan, np.nan, np.nan]), dtype=np.float64
        )
        tri_area_out[i] = float(keypoints.get("triangle_area", np.nan))

    return {
        "keypoints_roi": roi_out,
        "keypoints_img": img_out,
        "keypoints_norm": norm_out,
        "heading": heading_out,
        "confidence": conf_out,
        "effective_threshold": thresh_out,
        "effective_se2_radius": se2_out,
        "detection_success": success_out,
        "keypoint_confidences": kp_conf_out,
        "triangle_angles": tri_angles_out,
        "triangle_angles_raw": tri_angles_raw_out,
        "triangle_area": tri_area_out,
    }


def _update_keypoints_summary(
    root: zarr.Group,
    keypoints_group: zarr.Group,
) -> None:
    detection_success = np.asarray(keypoints_group["detection_success"][:], dtype=bool)
    frame_indices = np.asarray(keypoints_group["frame_indices"][:], dtype=np.int64)

    if "frame_counts" in keypoints_group:
        frame_count_len = int(keypoints_group["frame_counts"].shape[0])
    elif "n_rois" in keypoints_group:
        frame_count_len = int(keypoints_group["n_rois"].shape[0])
    elif "raw_video" in root and "images_full" in root["raw_video"]:
        frame_count_len = int(root["raw_video"]["images_full"].shape[0])
    else:
        frame_count_len = int(frame_indices.max() + 1) if frame_indices.size else 0

    if detection_success.size and np.any(detection_success):
        success_counts = np.bincount(frame_indices[detection_success], minlength=frame_count_len).astype("i4", copy=False)
    else:
        success_counts = np.zeros(frame_count_len, dtype="i4")

    if "n_keypoints" in keypoints_group:
        keypoints_group["n_keypoints"][:] = success_counts
    else:
        chunk_len = max(1, min(10000, success_counts.size))
        keypoints_group.create_array("n_keypoints", data=success_counts, chunks=(chunk_len,), overwrite=True)

    detection_source = keypoints_group.get("detection_source")
    source_is_real = np.ones_like(detection_success, dtype=bool)
    if detection_source is not None:
        source_is_real = np.asarray(detection_source[:], dtype=np.int8) == 0
    if "heading" in keypoints_group:
        heading_finite = np.isfinite(np.asarray(keypoints_group["heading"][:], dtype=np.float64))
    else:
        heading_finite = np.zeros_like(detection_success, dtype=bool)
    heading_usable = detection_success & source_is_real & heading_finite

    bool_chunks = keypoints_group["detection_success"].chunks or (max(1, min(1024, detection_success.size)),)
    if "heading_finite" in keypoints_group:
        keypoints_group["heading_finite"][:] = heading_finite
    else:
        keypoints_group.create_array(
            "heading_finite",
            data=heading_finite,
            chunks=bool_chunks,
            overwrite=True,
        )
    if "heading_usable" in keypoints_group:
        keypoints_group["heading_usable"][:] = heading_usable
    else:
        keypoints_group.create_array(
            "heading_usable",
            data=heading_usable,
            chunks=bool_chunks,
            overwrite=True,
        )
    if "heading_valid" in keypoints_group:
        del keypoints_group["heading_valid"]

    total_rois = int(keypoints_group["keypoints_roi"].shape[0])
    total_success = int(np.sum(detection_success))
    total_failed = int(total_rois - total_success)
    success_rate = (total_success / total_rois * 100.0) if total_rois else 0.0
    frames_with_keypoints = int(np.sum(success_counts > 0))

    summary_raw = keypoints_group.attrs.get("summary_statistics")
    summary: Dict[str, object] = summary_raw if isinstance(summary_raw, dict) else {}
    summary.update(
        {
            "total_rois": total_rois,
            "successful_detections": total_success,
            "failed_detections": total_failed,
            "success_rate_percent": round(success_rate, 2),
            "successful_keypoint_detections": total_success,
            "failed_keypoint_detections": total_failed,
            "frames_with_keypoints": frames_with_keypoints,
        }
    )
    keypoints_group.attrs["summary_statistics"] = summary
    keypoints_group.attrs["success_rate"] = round(success_rate, 2)


def _patch_keypoints_run(
    root: zarr.Group,
    keypoints_group: zarr.Group,
    crop_group: zarr.Group,
    background_full: np.ndarray,
    frames: List[int],
    *,
    apply: bool,
    patch_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    frame_indices = keypoints_group["frame_indices"][:].astype(np.int64, copy=False)
    frames_arr = np.array(frames, dtype=np.int64)
    target_mask = np.isin(frame_indices, frames_arr)
    target_indices = np.where(target_mask)[0]
    if target_indices.size == 0:
        return {"patched": 0, "frames": 0, "success": 0, "failed": 0}

    roi_images = crop_group["roi_images"]
    roi_coords_full = crop_group["roi_coordinates_full"]
    params = _load_keypoint_parameters(keypoints_group)

    outputs = _compute_keypoints_for_indices(
        roi_images,
        roi_coords_full,
        background_full,
        target_indices,
        params,
    )

    success_mask = outputs["detection_success"]
    success = int(np.sum(success_mask))
    failed = int(target_indices.size - success)

    if not apply:
        return {
            "patched": int(target_indices.size),
            "frames": int(np.unique(frame_indices[target_indices]).size),
            "success": success,
            "failed": failed,
        }

    nan_roi = np.full((3, 2), np.nan, dtype=np.float64)
    nan_tri = np.full(3, np.nan, dtype=np.float64)
    for i, roi_idx in enumerate(target_indices):
        if success_mask[i]:
            keypoints_group["keypoints_roi"][roi_idx] = outputs["keypoints_roi"][i]
            if "keypoints_img" in keypoints_group:
                keypoints_group["keypoints_img"][roi_idx] = outputs["keypoints_img"][i]
            if "keypoints_norm" in keypoints_group:
                keypoints_group["keypoints_norm"][roi_idx] = outputs["keypoints_norm"][i]
            if "heading" in keypoints_group:
                keypoints_group["heading"][roi_idx] = outputs["heading"][i]
            if "confidence" in keypoints_group:
                keypoints_group["confidence"][roi_idx] = outputs["confidence"][i]
            if "effective_threshold" in keypoints_group:
                keypoints_group["effective_threshold"][roi_idx] = outputs["effective_threshold"][i]
            if "effective_se2_radius" in keypoints_group:
                keypoints_group["effective_se2_radius"][roi_idx] = outputs["effective_se2_radius"][i]
            if "keypoint_confidences" in keypoints_group:
                keypoints_group["keypoint_confidences"][roi_idx] = outputs["keypoint_confidences"][i]
            if "triangle_angles" in keypoints_group:
                keypoints_group["triangle_angles"][roi_idx] = outputs["triangle_angles"][i]
            if "triangle_angles_raw" in keypoints_group:
                keypoints_group["triangle_angles_raw"][roi_idx] = outputs["triangle_angles_raw"][i]
            if "triangle_area" in keypoints_group:
                keypoints_group["triangle_area"][roi_idx] = outputs["triangle_area"][i]
        else:
            keypoints_group["keypoints_roi"][roi_idx] = nan_roi
            if "keypoints_img" in keypoints_group:
                keypoints_group["keypoints_img"][roi_idx] = nan_roi
            if "keypoints_norm" in keypoints_group:
                keypoints_group["keypoints_norm"][roi_idx] = nan_roi
            if "heading" in keypoints_group:
                keypoints_group["heading"][roi_idx] = np.nan
            if "confidence" in keypoints_group:
                keypoints_group["confidence"][roi_idx] = np.nan
            if "effective_threshold" in keypoints_group:
                keypoints_group["effective_threshold"][roi_idx] = np.nan
            if "effective_se2_radius" in keypoints_group:
                keypoints_group["effective_se2_radius"][roi_idx] = np.nan
            if "keypoint_confidences" in keypoints_group:
                keypoints_group["keypoint_confidences"][roi_idx] = np.full(3, np.nan, dtype=np.float64)
            if "triangle_angles" in keypoints_group:
                keypoints_group["triangle_angles"][roi_idx] = nan_tri
            if "triangle_angles_raw" in keypoints_group:
                keypoints_group["triangle_angles_raw"][roi_idx] = nan_tri
            if "triangle_area" in keypoints_group:
                keypoints_group["triangle_area"][roi_idx] = np.nan

        keypoints_group["detection_success"][roi_idx] = bool(success_mask[i])

    _update_keypoints_summary(root, keypoints_group)

    patched_idx = _update_index_array(keypoints_group, "patched_keypoint_indices", target_indices)
    patched_frames = _update_index_array(
        keypoints_group,
        "patched_keypoint_frames",
        np.unique(frame_indices[target_indices]),
    )

    attrs = dict(keypoints_group.attrs)
    history = attrs.get("keypoint_patch_history")
    if not isinstance(history, list):
        history = []
    patch_entry: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "patched_keypoints": int(target_indices.size),
        "patched_frames": int(np.unique(frame_indices[target_indices]).size),
        "success": success,
        "failed": failed,
    }
    if patch_context:
        patch_entry.update(patch_context)
    history.append(patch_entry)
    attrs["keypoint_patch_history"] = history
    attrs["keypoint_patch_count"] = len(history)
    attrs["patched_keypoint_total"] = int(patched_idx.size)
    attrs["patched_keypoint_frame_total"] = int(patched_frames.size)
    attrs["keypoint_patch_last_utc"] = patch_entry["timestamp_utc"]
    keypoints_group.attrs.put(attrs)

    return {
        "patched": int(target_indices.size),
        "frames": int(np.unique(frame_indices[target_indices]).size),
        "success": success,
        "failed": failed,
    }


def _patch_refined_keypoints(
    zarr_path: Path,
    root: zarr.Group,
    refined_run: Optional[str],
    keypoints_run: str,
    target_indices: np.ndarray,
    *,
    apply: bool,
    force: bool,
    patch_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    refined_group_name = "refined_keypoints_runs" if root.get("refined_keypoints_runs") is not None else "keypoints_refined_runs"
    refined_parent = root.get(refined_group_name)
    if refined_parent is None:
        return {"patched": 0, "frames": 0, "refined_group_name": refined_group_name, "refined_run": refined_run}
    if refined_run is None:
        refined_run = refined_parent.attrs.get("latest")
    if not refined_run or refined_run not in refined_parent:
        raise RuntimeError("Refined keypoint run not found.")

    refined = refined_parent[refined_run]
    source_run = refined.attrs.get("source_keypoints_run")
    if source_run and source_run != keypoints_run:
        msg = (
            f"refined_keypoints_run source_keypoints_run ({source_run}) does not match "
            f"keypoints_run ({keypoints_run})."
        )
        if not force:
            raise RuntimeError(msg + " Use --force to override.")
        print(f"  [warn] {msg} Proceeding due to --force.")

    if target_indices.size == 0:
        return {"patched": 0, "frames": 0, "refined_group_name": refined_group_name, "refined_run": refined_run}

    if not apply:
        return {
            "patched": int(target_indices.size),
            "frames": 0,
            "refined_group_name": refined_group_name,
            "refined_run": refined_run,
        }

    params = refined.attrs.get("parameters")
    if not isinstance(params, dict):
        params = {}
    params_dict = {
        "confidence_threshold": params.get("confidence_threshold", 0.3),
        "min_triangle_angle": params.get("min_triangle_angle", 10.0),
        "min_triangle_area": params.get("min_triangle_area", 100.0),
        "max_triangle_area": params.get("max_triangle_area"),
    }

    for roi_idx in target_indices:
        result = _process_refinement_chunk(
            str(zarr_path),
            keypoints_run,
            int(roi_idx),
            int(roi_idx + 1),
            params_dict,
        )
        idx = slice(result["start"], result["end"])

        refined["keypoints_roi"][idx] = result["roi"]
        if "keypoints_img" in refined:
            refined["keypoints_img"][idx] = result["img"]
        if "keypoints_norm" in refined:
            refined["keypoints_norm"][idx] = result["norm"]
        refined["heading"][idx] = result["heading"]
        refined["confidence"][idx] = result["confidence"]
        if "keypoint_confidences" in refined and result.get("kp_conf") is not None:
            refined["keypoint_confidences"][idx] = result["kp_conf"]
        if "effective_threshold" in refined and result.get("thresh") is not None:
            refined["effective_threshold"][idx] = result["thresh"]
        if "effective_se2_radius" in refined and result.get("se2") is not None:
            refined["effective_se2_radius"][idx] = result["se2"]
        refined["triangle_area"][idx] = result["area"]
        refined["min_angle"][idx] = result["min_angle"]
        refined["triangle_angles"][idx] = result["triangle_angles"]
        refined["quality_labels"][idx] = result["quality"]
        refined["refined_success"][idx] = result["refined_success"]
        refined["confidence_valid"][idx] = result["confidence_valid"]
        refined["geometry_valid"][idx] = result["geometry_valid"]
        refined["usable_keypoints"][idx] = result["usable"]
        refined["reason"][idx] = result["reason"]
        refined["flip_corrected"][idx] = result["flip_flags"]

    keypoint_group = root[f"keypoints_runs/{keypoints_run}"]
    refined["source_success"][target_indices] = keypoint_group["detection_success"][target_indices]

    source_success = np.asarray(refined["source_success"][:], dtype=bool)
    failure_indices = np.where(~source_success)[0].astype("i4", copy=False)
    refined.create_array(
        "failure_indices",
        data=failure_indices,
        chunks=(max(1, min(10000, failure_indices.size)),),
        overwrite=True,
    )

    detection_source = refined.get("detection_source")
    refined_success = np.asarray(refined["refined_success"][:], dtype=bool)
    source_is_real = np.ones_like(refined_success, dtype=bool)
    if detection_source is not None:
        source_is_real = np.asarray(detection_source[:], dtype=np.int8) == 0
    heading_finite = np.isfinite(np.asarray(refined["heading"][:], dtype=np.float64))
    heading_usable = refined_success & source_is_real & heading_finite

    bool_chunks = refined["refined_success"].chunks or (max(1, min(1024, refined_success.size)),)
    if "heading_finite" in refined:
        refined["heading_finite"][:] = heading_finite
    else:
        refined.create_array(
            "heading_finite",
            data=heading_finite,
            chunks=bool_chunks,
            overwrite=True,
        )
    if "heading_usable" in refined:
        refined["heading_usable"][:] = heading_usable
    else:
        refined.create_array(
            "heading_usable",
            data=heading_usable,
            chunks=bool_chunks,
            overwrite=True,
        )
    if "heading_valid" in refined:
        del refined["heading_valid"]

    _update_postprocess_summary(refined, print_summary=False)

    frame_indices = keypoint_group["frame_indices"][:].astype(np.int64, copy=False)
    patched_idx = _update_index_array(refined, "patched_keypoint_indices", target_indices)
    patched_frames = _update_index_array(
        refined,
        "patched_keypoint_frames",
        np.unique(frame_indices[target_indices]),
    )
    attrs = dict(refined.attrs)
    history = attrs.get("refined_keypoint_patch_history")
    if not isinstance(history, list):
        history = []
    patch_entry: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "patched_keypoints": int(target_indices.size),
        "patched_frames": int(np.unique(frame_indices[target_indices]).size),
    }
    if patch_context:
        patch_entry.update(patch_context)
    history.append(patch_entry)
    attrs["refined_keypoint_patch_history"] = history
    attrs["refined_keypoint_patch_count"] = len(history)
    attrs["refined_patched_keypoint_total"] = int(patched_idx.size)
    attrs["refined_patched_keypoint_frame_total"] = int(patched_frames.size)
    attrs["refined_keypoint_patch_last_utc"] = patch_entry["timestamp_utc"]
    refined.attrs.put(attrs)

    return {
        "patched": int(target_indices.size),
        "frames": int(np.unique(frame_indices[target_indices]).size),
        "refined_group_name": refined_group_name,
        "refined_run": refined_run,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Patch keypoints in-place for flagged frames using updated crops.",
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Zarr path(s) to patch.")
    parser.add_argument(
        "--frame-flag-file",
        type=Path,
        default=Path("retune_frame_flags.json"),
        help="JSON file mapping zarr path to flagged frame indices.",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        help="Optional file with zarr paths to patch (one per line).",
    )
    parser.add_argument(
        "--frames",
        type=str,
        help="Comma-separated list of frame indices to patch (overrides frame-flag-file).",
    )
    parser.add_argument(
        "--keypoints-run",
        type=str,
        help="Keypoints run to patch (defaults to latest).",
    )
    parser.add_argument(
        "--crop-run",
        type=str,
        help="Crop run to use (defaults to keypoints source or latest).",
    )
    parser.add_argument(
        "--background-run",
        type=str,
        help="Background run to use (defaults to keypoints source or latest).",
    )
    parser.add_argument(
        "--refined-run",
        type=str,
        help="Refined keypoints run to patch (defaults to latest).",
    )
    parser.add_argument(
        "--refined-only",
        action="store_true",
        help="Only patch refined keypoints (skip recomputing raw keypoints).",
    )
    parser.add_argument(
        "--skip-refined",
        action="store_true",
        help="Skip patching refined keypoints runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow patching even if source runs or method do not match.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to the keypoint runs.",
    )

    args = parser.parse_args(argv)
    if args.refined_only and args.skip_refined:
        raise SystemExit("--refined-only cannot be combined with --skip-refined.")
    frame_flags = _load_frame_flags(args.frame_flag_file)
    explicit_frames = _parse_frames(args.frames)
    plans = _collect_plans(args.paths, args.file_list, frame_flags, explicit_frames)
    if not plans:
        print("No recordings to patch.")
        return 0

    for plan in plans:
        print(f"\nPatching {plan.zarr_path}")
        root = zarr.open_group(str(plan.zarr_path), mode="a")

        keypoints_group, keypoints_run = _resolve_keypoints_run(root, args.keypoints_run)
        frame_indices = keypoints_group["frame_indices"][:].astype(np.int64, copy=False)
        target_indices = np.where(np.isin(frame_indices, np.array(plan.frames, dtype=np.int64)))[0]
        target_frames = np.unique(frame_indices[target_indices]) if target_indices.size else np.array([], dtype=np.int64)
        method = keypoints_group.attrs.get("method")
        if method and str(method).lower() not in {"traditional_pose", "traditional"}:
            msg = f"keypoints_run method is '{method}', patching expects traditional_pose."
            if not args.force:
                raise RuntimeError(msg + " Use --force to override.")
            print(f"  [warn] {msg} Proceeding due to --force.")

        patch_context = {
            "keypoints_run": keypoints_run,
            "reason": "detection_patch",
        }
        if args.frame_flag_file:
            patch_context["frame_flag_file"] = str(args.frame_flag_file)

        if not args.refined_only:
            crop_group, crop_run = _resolve_crop_run(root, args.crop_run, keypoints_group)
            background_group, background_run = _resolve_background_run(root, args.background_run, keypoints_group)
            crop_source = keypoints_group.attrs.get("source_crop_run")
            if crop_source and crop_source != crop_run:
                msg = f"keypoints_run source_crop_run ({crop_source}) does not match crop_run ({crop_run})."
                if not args.force:
                    raise RuntimeError(msg + " Use --force to override.")
                print(f"  [warn] {msg} Proceeding due to --force.")

            if "background_full" not in background_group:
                raise RuntimeError("background_full is required for patching keypoints.")
            background_full = background_group["background_full"][:]

            patch_context.update(
                {
                    "crop_run": crop_run,
                    "background_run": background_run,
                }
            )
            keypoint_result = _patch_keypoints_run(
                root,
                keypoints_group,
                crop_group,
                background_full,
                plan.frames,
                apply=args.apply,
                patch_context=patch_context,
            )
            print(
                f"  keypoints_run={keypoints_run} crop_run={crop_run} "
                f"background_run={background_run}"
            )
            print(
                f"  frames={keypoint_result['frames']} keypoints={keypoint_result['patched']} "
                f"success={keypoint_result['success']} failed={keypoint_result['failed']}"
            )
            if args.apply and int(keypoint_result.get("patched", 0)) > 0:
                stale_marked = mark_downstream_eye_mask_runs_stale(
                    root,
                    source_keypoint_group="keypoints_runs",
                    source_keypoints_run=str(keypoints_run),
                    roi_indices=target_indices.tolist(),
                    frame_indices=target_frames.tolist(),
                    reason="keypoint_patch_from_crops_raw",
                )
                if stale_marked:
                    print(f"  marked_downstream_eye_masks_stale={stale_marked}")
        else:
            print(f"  keypoints_run={keypoints_run} (raw patch skipped)")

        if not args.skip_refined:
            refined_context = dict(patch_context)
            refined_context["refined_run"] = args.refined_run or "latest"
            refined_result = _patch_refined_keypoints(
                plan.zarr_path,
                root,
                args.refined_run,
                keypoints_run,
                target_indices,
                apply=args.apply,
                force=args.force,
                patch_context=refined_context,
            )
            if refined_result.get("patched"):
                print(
                    f"  refined_keypoints patched={refined_result['patched']} "
                    f"frames={refined_result['frames']}"
                )
            if args.apply and int(refined_result.get("patched", 0)) > 0:
                stale_marked = mark_downstream_eye_mask_runs_stale(
                    root,
                    source_keypoint_group=str(refined_result.get("refined_group_name") or "refined_keypoints_runs"),
                    source_keypoints_run=str(refined_result.get("refined_run") or ""),
                    roi_indices=target_indices.tolist(),
                    frame_indices=target_frames.tolist(),
                    reason="keypoint_patch_from_crops_refined",
                )
                if stale_marked:
                    print(f"  marked_downstream_eye_masks_stale={stale_marked}")

        if not args.apply:
            print("  (dry-run) use --apply to write changes")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
