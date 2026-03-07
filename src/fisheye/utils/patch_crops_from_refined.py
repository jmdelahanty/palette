"""Patch crop ROIs in-place using refined detections for flagged frames.

This is intended for small, targeted fixes where detection counts per frame do
NOT change. It rewrites the affected crop ROIs (and coordinates) in the chosen
crop run while preserving the rest of the dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr

from ..shared.crop_signature import bump_crop_revision
from ..shared.refined_detect_review import (
    DEFAULT_DETECT_GROUP_PREFERENCE,
    resolve_refined_detect_group,
)


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
            frames = [int(v) for v in value]
            out[str(key)] = frames
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
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        frames.append(int(token))
    return frames


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


def _resolve_refined_run(root: zarr.Group, run_name: Optional[str]) -> Tuple[zarr.Group, str, str]:
    parent = root.get("refined_detect_runs") or root.get("refined_runs")
    if parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")
    if run_name is None:
        run_name = parent.attrs.get("latest")
    if not run_name or run_name not in parent:
        raise RuntimeError("Refined detect run not found.")
    return parent, run_name, parent.name


def _resolve_crop_run(root: zarr.Group, run_name: Optional[str]) -> Tuple[zarr.Group, str]:
    parent = root.get("crop_runs")
    if parent is None:
        raise RuntimeError("No crop_runs found in archive.")
    if run_name is None:
        run_name = parent.attrs.get("latest")
    if not run_name or run_name not in parent:
        raise RuntimeError("Crop run not found.")
    return parent[run_name], run_name


def _resolve_detect_group(
    root: zarr.Group,
    refined_parent: zarr.Group,
    refined_run: str,
    refined_group_override: Optional[str],
) -> Tuple[zarr.Group, str, str]:
    refined_group = refined_parent[refined_run]
    resolution = resolve_refined_detect_group(
        refined_group,
        preference=DEFAULT_DETECT_GROUP_PREFERENCE,
        override_group=refined_group_override,
    )
    if resolution.group is None:
        detect_run = resolution.source_detect_run or refined_group.attrs.get("source_detect_run")
        detect_parent = root.get("detect_runs")
        if not detect_run or detect_parent is None or detect_run not in detect_parent:
            raise RuntimeError("Source detect run not found for refined detect patch.")
        return detect_parent[detect_run], f"detect_runs/{detect_run}", "raw"
    if resolution.group not in refined_group:
        raise RuntimeError(f"Refined detect group '{resolution.group}' not found.")
    label = resolution.label or resolution.group
    return refined_group[resolution.group], f"{refined_group.path}/{resolution.group}", label


def _extract_scale_factor(root: zarr.Group, crop_group: zarr.Group) -> Optional[float]:
    raw = root.get("raw_video")
    if raw is not None and "images_full" in raw and "images_ds" in raw:
        full = raw["images_full"]
        ds = raw["images_ds"]
        if full.shape[1] > 0:
            return ds.shape[1] / full.shape[1]
    summary = crop_group.attrs.get("summary_statistics")
    if isinstance(summary, dict):
        scale = summary.get("scale_factor")
        try:
            return float(scale)
        except (TypeError, ValueError):
            return None
    return None


def _read_frame(images_full: np.ndarray, frame_idx: int) -> np.ndarray:
    frame = images_full[frame_idx]
    if frame.ndim == 3:
        frame = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]).astype(np.uint8)
    return frame


def _crop_single(
    frame: np.ndarray,
    bbox: np.ndarray,
    roi_sz: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    roi_h, roi_w = roi_sz
    H, W = frame.shape[:2]
    center_norm = bbox[:2]
    full_centroid_px = np.round(center_norm * np.array([W, H])).astype(int)
    x1 = int(full_centroid_px[0] - roi_w // 2)
    y1 = int(full_centroid_px[1] - roi_h // 2)
    y2 = y1 + roi_h
    x2 = x1 + roi_w

    vy1 = max(0, y1)
    vy2 = min(H, y2)
    vx1 = max(0, x1)
    vx2 = min(W, x2)

    if (vy2 - vy1) == roi_h and (vx2 - vx1) == roi_w and 0 <= y1 < H and 0 <= x1 < W:
        roi = frame[vy1:vy2, vx1:vx2]
    else:
        roi = np.zeros((roi_h, roi_w), dtype=frame.dtype)
        if vy2 > vy1 and vx2 > vx1:
            py1 = max(0, -y1)
            px1 = max(0, -x1)
            py2 = py1 + (vy2 - vy1)
            px2 = px1 + (vx2 - vx1)
            roi[py1:py2, px1:px2] = frame[vy1:vy2, vx1:vx2]

    return roi.astype(np.uint8), (x1, y1)


def _update_index_array(group: zarr.Group, name: str, values: np.ndarray) -> np.ndarray:
    values = np.unique(values.astype(np.int64, copy=False))
    if name in group:
        existing = group[name][:]
        values = np.unique(np.concatenate([existing.astype(np.int64, copy=False), values]))
    group.create_array(name, data=values, chunks=(min(1024, values.size),), overwrite=True)
    return values


def _patch_crop_run(
    root: zarr.Group,
    crop_group: zarr.Group,
    detect_group: zarr.Group,
    frames: List[int],
    *,
    apply: bool,
    patch_context: Optional[Dict[str, object]] = None,
    detection_source_path: Optional[str] = None,
    detection_source_type: Optional[str] = None,
    source_refined_run: Optional[str] = None,
) -> Dict[str, object]:
    frame_indices = detect_group["frame_indices"][:].astype(np.int64, copy=False)
    bbox_norm = detect_group["bbox_norm_coords"][:]

    roi_images = crop_group["roi_images"]
    total_detections = int(roi_images.shape[0])
    if frame_indices.shape[0] != total_detections:
        raise RuntimeError(
            f"Detection count {frame_indices.shape[0]} does not match crop count {total_detections}."
        )

    frames_arr = np.array(frames, dtype=np.int64)
    target_mask = np.isin(frame_indices, frames_arr)
    target_indices = np.where(target_mask)[0]
    if target_indices.size == 0:
        return {"patched": 0, "frames": 0}

    roi_size = crop_group.attrs.get("roi_size")
    if isinstance(roi_size, (list, tuple)) and len(roi_size) == 2:
        roi_sz = (int(roi_size[0]), int(roi_size[1]))
    else:
        roi_sz = (int(roi_images.shape[1]), int(roi_images.shape[2]))

    raw = root.get("raw_video")
    if raw is None or "images_full" not in raw:
        raise RuntimeError("raw_video/images_full is required for patching crops.")
    images_full = raw["images_full"]
    scale_factor = _extract_scale_factor(root, crop_group)

    coords_full = np.zeros((target_indices.size, 2), dtype=np.int32)
    coords_ds = np.zeros((target_indices.size, 2), dtype=np.int32) if "roi_coordinates_ds" in crop_group and scale_factor is not None else None
    rois = np.zeros((target_indices.size, roi_sz[0], roi_sz[1]), dtype=np.uint8)

    frame_cache: Dict[int, np.ndarray] = {}
    for i, det_idx in enumerate(target_indices):
        frame_idx = int(frame_indices[det_idx])
        frame = frame_cache.get(frame_idx)
        if frame is None:
            if frame_idx < 0 or frame_idx >= images_full.shape[0]:
                raise RuntimeError(f"Frame index {frame_idx} out of bounds for images_full.")
            frame = _read_frame(images_full, frame_idx)
            frame_cache[frame_idx] = frame
        roi, (x1, y1) = _crop_single(frame, bbox_norm[det_idx], roi_sz)
        rois[i] = roi
        coords_full[i] = (x1, y1)
        if coords_ds is not None and scale_factor is not None:
            coords_ds[i] = (int(x1 * scale_factor), int(y1 * scale_factor))

    if not apply:
        return {
            "patched": int(target_indices.size),
            "frames": int(np.unique(frame_indices[target_indices]).size),
        }

    for i, det_idx in enumerate(target_indices):
        roi_images[det_idx] = rois[i]
        crop_group["roi_coordinates_full"][det_idx] = coords_full[i]
        if coords_ds is not None:
            crop_group["roi_coordinates_ds"][det_idx] = coords_ds[i]
        crop_group["bbox_norm_coords"][det_idx] = bbox_norm[det_idx]

    patched_det = _update_index_array(crop_group, "patched_detection_indices", target_indices)
    patched_frames = _update_index_array(
        crop_group,
        "patched_frame_indices",
        np.unique(frame_indices[target_indices]),
    )

    attrs = dict(crop_group.attrs)
    history = attrs.get("crop_patch_history")
    if not isinstance(history, list):
        history = []
    patch_entry: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "patched_detections": int(target_indices.size),
        "patched_frames": int(np.unique(frame_indices[target_indices]).size),
    }
    if patch_context:
        patch_entry.update(patch_context)
    history.append(patch_entry)
    attrs["crop_patch_history"] = history
    attrs["crop_patch_count"] = len(history)
    attrs["patched_detection_total"] = int(patched_det.size)
    attrs["patched_frame_total"] = int(patched_frames.size)
    attrs["crop_patch_last_utc"] = patch_entry["timestamp_utc"]
    if detection_source_path:
        attrs["detection_source_path"] = str(detection_source_path)
    if detection_source_type:
        attrs["detection_source_type"] = str(detection_source_type)
    if source_refined_run:
        attrs["source_refined_run"] = str(source_refined_run)
    bump_crop_revision(
        attrs,
        reason="manual_bbox_patch",
        timestamp_utc=str(patch_entry["timestamp_utc"]),
    )
    crop_group.attrs.put(attrs)

    return {
        "patched": int(target_indices.size),
        "frames": int(np.unique(frame_indices[target_indices]).size),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Patch crop ROIs in-place using refined detections for flagged frames.",
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
        "--refined-run",
        type=str,
        help="Refined detect run to use (defaults to latest).",
    )
    parser.add_argument(
        "--refined-group",
        type=str,
        help="Refined detect subgroup to use (manual/interpolated/filtered/raw).",
    )
    parser.add_argument(
        "--crop-run",
        type=str,
        help="Crop run to patch (defaults to latest).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow patching even if crop detection_source_path does not match refined group.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to the crop run.",
    )

    args = parser.parse_args(argv)
    frame_flags = _load_frame_flags(args.frame_flag_file)
    explicit_frames = _parse_frames(args.frames)

    plans = _collect_plans(args.paths, args.file_list, frame_flags, explicit_frames)
    if not plans:
        print("No recordings to patch.")
        return 0

    for plan in plans:
        print(f"\nPatching {plan.zarr_path}")
        root = zarr.open_group(str(plan.zarr_path), mode="a")
        refined_parent, refined_run, _ = _resolve_refined_run(root, args.refined_run)
        crop_group, crop_run = _resolve_crop_run(root, args.crop_run)
        detect_group, detect_path, detect_label = _resolve_detect_group(
            root, refined_parent, refined_run, args.refined_group
        )
        crop_source_path = crop_group.attrs.get("detection_source_path")
        if crop_source_path and str(crop_source_path) != str(detect_path):
            msg = (
                f"crop_run detection_source_path ({crop_source_path}) does not match "
                f"requested detect group ({detect_path})."
            )
            if not args.force:
                raise RuntimeError(msg + " Use --force to override.")
            print(f"  [warn] {msg} Proceeding due to --force.")

        patch_context = {
            "refined_run": refined_run,
            "refined_group": detect_label,
            "detect_group_path": detect_path,
            "crop_run": crop_run,
            "reason": "keypoint_detection_issue",
        }
        if args.frame_flag_file:
            patch_context["frame_flag_file"] = str(args.frame_flag_file)
        result = _patch_crop_run(
            root,
            crop_group,
            detect_group,
            plan.frames,
            apply=args.apply,
            patch_context=patch_context,
            detection_source_path=detect_path,
            detection_source_type=detect_label,
            source_refined_run=refined_run,
        )
        print(
            f"  crop_run={crop_run} refined_run={refined_run} "
            f"detect_group={detect_label} ({detect_path})"
        )
        print(f"  frames={result['frames']} detections={result['patched']}")
        if not args.apply:
            print("  (dry-run) use --apply to write changes")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
