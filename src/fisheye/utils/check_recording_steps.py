import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np
import zarr

from fisheye.registry.db import Registry
from fisheye.shared.experiment_setup import subdish_required
from fisheye.shared.refined_detect_review import (
    DEFAULT_DETECT_GROUP_PREFERENCE,
    resolve_refined_detect_group,
)
try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - rich is optional
    Console = None  # type: ignore
    Table = None  # type: ignore


DEFAULT_TUNING_KEYS = [
    "dish_mask",
    "detection_tuning",
    "keypoint_tuning",
    "eye_mask_tuning",
    "subdish_mask_tuning",
]


@dataclass
class RecordingStatus:
    recording_dir: Path
    h5_path: Path
    camera_id: Optional[str]
    recording_id: Optional[str]
    zarr_path: Path
    zarr_use: Optional[str]
    zarr_exists: bool
    pipeline_type: Optional[str]
    zarr_purpose: Optional[str]
    has_raw_video_attr: Optional[bool]
    raw_present: bool
    full_present: bool
    ds_present: bool
    sampled_present: bool
    background_full_present: bool
    background_ds_present: bool
    detect_present: bool
    detect_method: Optional[str]
    detect_coverage: Optional[float]
    detect_coverage_basis: Optional[str]
    detect_quality_present: bool
    detect_quality_run: Optional[str]
    detect_quality_grade: Optional[str]
    detect_quality_score: Optional[float]
    detect_quality_clean_percent: Optional[float]
    detect_quality_artifacts: Optional[int]
    refined_detect_present: bool
    refined_detect_coverage: Optional[float]
    refined_detect_method: Optional[str]
    refined_detect_resolved_group: Optional[str]
    detect_review_status: Optional[Dict[str, object]]
    crop_present: bool
    crop_status: Optional[str]
    crop_review_status: Optional[Dict[str, object]]
    keypoints_present: bool
    refined_keypoints_present: bool
    refined_keypoints_coverage: Optional[float]
    refined_keypoints_success: Optional[float]
    keypoint_review_status: Optional[Dict[str, object]]
    eye_masks_present: bool
    assign_ids_present: bool
    track_present: bool
    stimulus_runs: int
    calibration_present: bool
    tuning_present: int
    tuning_total: int
    tuning_missing: List[str]
    tuning_status: Dict[str, str]


def _normalize_attr(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return int(value)
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_mapping(value: object) -> Optional[Dict[str, object]]:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.flat[0]
        else:
            try:
                return dict(value.tolist())  # type: ignore[arg-type]
            except Exception:
                return None
    if isinstance(value, dict):
        return value
    try:
        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if hasattr(value, "items"):
        try:
            return dict(value)  # type: ignore[arg-type]
        except Exception:
            pass
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(value, (list, tuple)):
        try:
            return dict(value)
        except Exception:
            return None
    return None


def _extract_coverage_from_group(group: zarr.Group) -> Optional[float]:
    frame_counts = group.get("frame_counts") or group.get("n_detections")
    if frame_counts is not None:
        try:
            counts = frame_counts[:]
        except Exception:
            counts = None
        if counts is not None:
            total = counts.shape[0]
            if total > 0:
                present = (counts > 0).sum()
                return float(present) / float(total) * 100.0
    return None


def _dataset_len(group: Optional[zarr.Group], key: str) -> Optional[int]:
    if group is None:
        return None
    arr = group.get(key)
    if arr is None:
        return None
    try:
        return int(arr.shape[0])
    except Exception:
        return None


def _extract_detect_coverage(
    detect_group: zarr.Group,
    root: Optional[zarr.Group] = None,
) -> tuple[Optional[float], Optional[str]]:
    frame_counts = detect_group.get("frame_counts") or detect_group.get("n_detections")
    if frame_counts is not None:
        try:
            counts = np.asarray(frame_counts[:])
        except Exception:
            counts = None
        if counts is not None and counts.size > 0:
            total = int(counts.shape[0])
            present = int(np.sum(counts > 0))
            basis = "full"
            raw = root.get("raw_video") if root is not None else None
            sampled_total = _dataset_len(raw, "original_frame_indices")
            if sampled_total is not None and sampled_total == total:
                basis = "sampled"
            return (float(present) / float(total) * 100.0), basis

    frame_indices = detect_group.get("frame_indices")
    if frame_indices is not None:
        try:
            values = np.asarray(frame_indices[:])
        except Exception:
            values = None
        if values is not None and values.size > 0:
            try:
                flat = values.reshape(-1).astype(np.int64, copy=False)
                total = int(flat.max()) + 1
                if total > 0:
                    present = int(np.unique(flat).size)
                    return (float(present) / float(total) * 100.0), "inferred"
            except Exception:
                return None, None

    return None, None


def _extract_detect_method(detect_group: zarr.Group) -> Optional[str]:
    direct = _normalize_attr(detect_group.attrs.get("detection_method"))
    if direct:
        return direct
    provenance = _coerce_mapping(detect_group.attrs.get("provenance"))
    if provenance is not None:
        prov_method = _normalize_attr(provenance.get("method"))
        if prov_method:
            return prov_method
    return None


def _extract_detect_quality(detect_group: zarr.Group) -> Dict[str, object]:
    quality_parent = detect_group.get("quality_reports")
    if quality_parent is None:
        return {
            "present": False,
            "run": None,
            "grade": None,
            "score": None,
            "clean_percent": None,
            "artifacts": None,
        }

    latest_quality = _normalize_attr(quality_parent.attrs.get("latest"))
    quality_candidate = None
    if latest_quality and latest_quality in quality_parent:
        quality_candidate = latest_quality
    else:
        if hasattr(quality_parent, "group_keys"):
            names = list(quality_parent.group_keys())
        else:
            names = list(quality_parent.keys())
        if names:
            quality_candidate = sorted(names)[-1]

    if not quality_candidate:
        return {
            "present": False,
            "run": None,
            "grade": None,
            "score": None,
            "clean_percent": None,
            "artifacts": None,
        }

    quality_group = quality_parent[quality_candidate]
    quality_score = _coerce_mapping(quality_group.attrs.get("quality_score")) or {}
    quality_summary = _coerce_mapping(quality_group.attrs.get("detection_quality_summary")) or {}
    blip = _coerce_int(quality_summary.get("blip_detections"))
    jump = _coerce_int(quality_summary.get("jump_detections"))
    multi = _coerce_int(quality_summary.get("multi_detections"))
    artifact_total = None
    if blip is not None or jump is not None or multi is not None:
        artifact_total = int((blip or 0) + (jump or 0) + (multi or 0))

    return {
        "present": True,
        "run": str(quality_candidate),
        "grade": _normalize_attr(quality_score.get("grade")),
        "score": _coerce_float(quality_score.get("overall_score")),
        "clean_percent": _coerce_float(quality_summary.get("clean_percentage")),
        "artifacts": artifact_total,
    }


def _sampled_total_frames(root: Optional[zarr.Group]) -> Optional[int]:
    if root is None:
        return None
    raw = root.get("raw_video")
    if raw is not None:
        if "original_frame_indices" in raw:
            return int(raw["original_frame_indices"].shape[0])
        if "images_ds" in raw:
            return int(raw["images_ds"].shape[0])
        if "images_full" in raw:
            return int(raw["images_full"].shape[0])
    return None


def _extract_refined_coverage(
    refined_group: zarr.Group,
    root: Optional[zarr.Group] = None,
) -> tuple[Optional[float], Optional[str]]:
    manual_group_name = _normalize_attr(refined_group.attrs.get("manual_review_latest"))
    if manual_group_name and manual_group_name in refined_group:
        manual_group = refined_group[manual_group_name]
        coverage = _extract_coverage_from_group(manual_group)
        if coverage is not None:
            return coverage, "manual"

    parameters = _coerce_mapping(refined_group.attrs.get("parameters"))
    refine_mode = _normalize_attr(parameters.get("refine_mode")) if parameters is not None else None
    sampled_import = bool(parameters.get("sampled_import")) if parameters is not None else False
    operations = refined_group.attrs.get("operations")
    if refine_mode == "passthrough" or operations == ["passthrough"]:
        if sampled_import:
            total_frames = refined_group.attrs.get("coverage_frames_total")
            if total_frames is None:
                total_frames = _sampled_total_frames(root)
            try:
                total_frames = int(total_frames) if total_frames is not None else None
            except Exception:
                total_frames = None
            base_group = refined_group.get("interpolated") or refined_group.get("filtered")
            if base_group is not None:
                frame_counts = base_group.get("frame_counts") or base_group.get("n_detections")
                if frame_counts is not None:
                    counts = frame_counts[:]
                    if total_frames is not None:
                        if counts.shape[0] < total_frames:
                            counts = np.pad(counts, (0, total_frames - counts.shape[0]), mode="constant")
                        elif counts.shape[0] > total_frames:
                            counts = counts[:total_frames]
                        coverage = (float(np.sum(counts > 0)) / float(total_frames)) * 100.0
                        return coverage, "passthrough"
                    if counts.shape[0] > 0:
                        coverage = (float(np.sum(counts > 0)) / float(counts.shape[0])) * 100.0
                        return coverage, "passthrough"
        coverage = None
        comparison = _coerce_mapping(refined_group.attrs.get("coverage_comparison"))
        if comparison is not None:
            original = comparison.get("original")
            if isinstance(original, dict):
                coverage = _coerce_float(original.get("coverage_percent"))
        if coverage is None:
            interp_group = refined_group.get("interpolated")
            if interp_group is not None:
                coverage = _coerce_float(interp_group.attrs.get("coverage_percent"))
        return coverage, "passthrough"

    comparison = _coerce_mapping(refined_group.attrs.get("coverage_comparison"))
    if comparison is not None:
        filtered = comparison.get("filtered")
        interpolated = comparison.get("interpolated")
        if isinstance(filtered, dict) and isinstance(interpolated, dict):
            removed = _coerce_float(filtered.get("detections_removed"))
            added = _coerce_float(interpolated.get("detections_added"))
            if removed == 0 and added == 0:
                original = comparison.get("original")
                if isinstance(original, dict):
                    coverage = _coerce_float(original.get("coverage_percent"))
                    if coverage is not None:
                        return coverage, "unchanged"

        interpolated = comparison.get("interpolated")
        if isinstance(interpolated, dict):
            coverage = _coerce_float(interpolated.get("coverage_percent"))
            if coverage is not None:
                return coverage, "interpolated"
        filtered = comparison.get("filtered")
        if isinstance(filtered, dict):
            coverage = _coerce_float(filtered.get("coverage_percent"))
            if coverage is not None:
                return coverage, "filtered"

    stats = _coerce_mapping(refined_group.attrs.get("coverage_stats"))
    if stats is not None:
        final = stats.get("final")
        if isinstance(final, dict):
            coverage = _coerce_float(final.get("coverage_percent"))
            if coverage is not None:
                return coverage, "interpolated"
        clean = stats.get("clean")
        if isinstance(clean, dict):
            coverage = _coerce_float(clean.get("coverage_percent"))
            if coverage is not None:
                return coverage, "filtered"

    interp_group = refined_group.get("interpolated")
    if interp_group is not None:
        coverage = _coerce_float(interp_group.attrs.get("coverage_percent"))
        if coverage is not None:
            return coverage, "interpolated"

    return None, None


def _pick_keypoint_summary_block(summary: Dict[str, object]) -> Dict[str, object]:
    block = summary.get("postprocess")
    if isinstance(block, dict):
        return block
    block = summary.get("refine")
    if isinstance(block, dict):
        return block
    return summary


def _extract_refined_keypoints_stats(
    refined_group: zarr.Group,
) -> tuple[Optional[float], Optional[float]]:
    summary = _coerce_mapping(refined_group.attrs.get("summary_statistics"))
    total: Optional[float] = None
    refined_success: Optional[float] = None
    usable: Optional[float] = None
    success_percent: Optional[float] = None

    if summary is not None:
        summary_block = _pick_keypoint_summary_block(summary)
        total = _coerce_float(summary_block.get("total_rois")) or _coerce_float(summary_block.get("total"))
        refined_success = _coerce_float(summary_block.get("refined_success"))
        usable = _coerce_float(summary_block.get("usable_keypoints")) or _coerce_float(summary_block.get("usable"))
        success_percent = _coerce_float(summary_block.get("success_rate_percent")) or _coerce_float(
            summary_block.get("pass_rate_percent")
        )

    if total is None:
        for key in ("refined_success", "usable_keypoints", "keypoints_roi"):
            arr = refined_group.get(key)
            if arr is not None:
                try:
                    total = float(arr.shape[0])
                    break
                except Exception:
                    continue

    if refined_success is None:
        refined_arr = refined_group.get("refined_success")
        if refined_arr is not None:
            try:
                refined_vals = np.asarray(refined_arr[:], dtype=bool)
                refined_success = float(np.sum(refined_vals))
            except Exception:
                refined_success = None

    if usable is None:
        usable_arr = refined_group.get("usable_keypoints")
        if usable_arr is not None:
            try:
                usable_vals = np.asarray(usable_arr[:], dtype=bool)
                usable = float(np.sum(usable_vals))
            except Exception:
                usable = None

    usable_percent: Optional[float] = None
    if total and total > 0:
        if success_percent is None and refined_success is not None:
            success_percent = float(refined_success) / float(total) * 100.0
        if usable is not None:
            usable_percent = float(usable) / float(total) * 100.0

    return success_percent, usable_percent


def _derive_camera_id(ipc_source_name: object) -> Optional[str]:
    if ipc_source_name is None:
        return None
    text = _normalize_attr(ipc_source_name)
    if text is None:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits if digits else None


def _read_camera_id(h5_path: Path) -> Optional[str]:
    with h5py.File(h5_path, "r") as h5:
        root = h5.attrs
        if "camera_id" in root:
            cam = _normalize_attr(root.get("camera_id"))
            if cam:
                return cam
        ipc = _normalize_attr(root.get("ipc_source_name"))
        return _derive_camera_id(ipc)


def _read_recording_id(h5_path: Path) -> Optional[str]:
    with h5py.File(h5_path, "r") as h5:
        root = h5.attrs
        for key in ("session_uuid", "session_id"):
            value = _normalize_attr(root.get(key))
            if value:
                return value
    return None


def _resolve_root(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _iter_h5(paths: List[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            if path.suffix.lower() in {".h5", ".hdf5"}:
                yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("raw/*.h5")
            yield from path.rglob("raw/*.hdf5")
        else:
            yield from path.glob("*/raw/*.h5")
            yield from path.glob("*/raw/*.hdf5")


def _infer_use_from_path(zarr_path: Path) -> Optional[str]:
    name = zarr_path.name
    if name.endswith("_training.zarr"):
        return "training"
    if name.endswith("_analysis.zarr"):
        return "analysis"
    return None


def _resolve_zarr_candidates(
    *,
    recording_dir: Path,
    recording_id: Optional[str],
    requested_use: str,
    registry: Optional[Registry],
) -> List[tuple[Path, Optional[str]]]:
    candidates: Dict[str, tuple[Path, Optional[str]]] = {}

    def _accept_use(use_value: Optional[str]) -> bool:
        if requested_use == "all":
            return True
        return (use_value or "").strip().lower() == requested_use

    if registry is not None and recording_id:
        rows = registry.conn.execute(
            """
            SELECT zarr_path, zarr_use
            FROM datasets
            WHERE recording_id = ?
              AND zarr_path IS NOT NULL
              AND TRIM(zarr_path) <> '';
            """,
            (recording_id,),
        ).fetchall()
        for row in rows:
            path_text = str(row["zarr_path"]).strip()
            if not path_text:
                continue
            path = Path(path_text).expanduser()
            zarr_use = _normalize_attr(row["zarr_use"])
            if not _accept_use(zarr_use):
                continue
            candidates[str(path)] = (path, zarr_use)

    zarr_dir = recording_dir / "zarr"
    if zarr_dir.exists():
        for path in sorted(zarr_dir.glob("*.zarr")):
            inferred_use = _infer_use_from_path(path)
            if not _accept_use(inferred_use):
                continue
            key = str(path)
            if key in candidates:
                existing = candidates[key]
                candidates[key] = (existing[0], existing[1] or inferred_use)
            else:
                candidates[key] = (path, inferred_use)

    if candidates:
        return sorted(candidates.values(), key=lambda item: str(item[0]))

    recording_name = recording_dir.name
    if requested_use == "analysis":
        return [(recording_dir / "zarr" / f"{recording_name}_analysis.zarr", "analysis")]
    if requested_use == "training":
        return [(recording_dir / "zarr" / f"{recording_name}_training.zarr", "training")]
    return [
        (recording_dir / "zarr" / f"{recording_name}_analysis.zarr", "analysis"),
        (recording_dir / "zarr" / f"{recording_name}_training.zarr", "training"),
    ]


def _registry_crop_review_status_for_zarr(
    *,
    registry: Registry,
    zarr_path: Path,
) -> Optional[Dict[str, object]]:
    rows = registry.conn.execute(
        """
        SELECT
            cqc.crop_run,
            cqc.review_state,
            cqc.review_method,
            cqc.review_intended_use,
            cqc.review_reviewer,
            cqc.review_timestamp_utc,
            cqc.review_notes
        FROM crop_quality_current cqc
        JOIN datasets d ON d.dataset_id = cqc.dataset_id
        WHERE d.zarr_path = ?
        ORDER BY
            COALESCE(cqc.review_timestamp_utc, cqc.crop_created_utc, cqc.updated_utc) DESC,
            COALESCE(cqc.crop_created_utc, '') DESC,
            cqc.crop_run DESC
        LIMIT 1;
        """,
        (str(zarr_path),),
    ).fetchall()
    if not rows:
        return None
    row = rows[0]

    status: Dict[str, object] = {}
    state = _normalize_attr(row["review_state"])
    method = _normalize_attr(row["review_method"])
    intended_use = _normalize_attr(row["review_intended_use"])
    reviewer = _normalize_attr(row["review_reviewer"])
    timestamp = _normalize_attr(row["review_timestamp_utc"])
    notes = _normalize_attr(row["review_notes"])
    if state:
        status["state"] = state
    if method:
        status["method"] = method
    if intended_use:
        status["intended_use"] = intended_use
    if reviewer:
        status["reviewer"] = reviewer
    if timestamp:
        status["timestamp_utc"] = timestamp
    if notes:
        status["notes"] = notes

    return {
        "crop_run": _normalize_attr(row["crop_run"]),
        "crop_review_status": status or None,
    }


def _load_group_attrs(zarr_path: Path, group_path: str) -> Dict[str, object]:
    group_dir = zarr_path / group_path
    zarr_json = group_dir / "zarr.json"
    attrs: Dict[str, object] = {}
    if zarr_json.exists():
        try:
            data = json.loads(zarr_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        attrs_raw = data.get("attributes") if isinstance(data, dict) else None
        if isinstance(attrs_raw, dict):
            attrs = dict(attrs_raw)

    parent_zarr = group_dir.parent / "zarr.json"
    if parent_zarr.exists():
        try:
            parent_data = json.loads(parent_zarr.read_text(encoding="utf-8"))
        except Exception:
            parent_data = {}
        meta = None
        if isinstance(parent_data, dict):
            consolidated = parent_data.get("consolidated_metadata")
            if isinstance(consolidated, dict):
                meta = consolidated.get("metadata")
        if isinstance(meta, dict):
            entry = meta.get(group_dir.name)
            if isinstance(entry, dict):
                child_attrs = entry.get("attributes")
                if isinstance(child_attrs, dict):
                    for key, value in child_attrs.items():
                        attrs.setdefault(key, value)

    if attrs:
        return attrs
    zattrs = group_dir / ".zattrs"
    if zattrs.exists():
        try:
            data = json.loads(zattrs.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        return data if isinstance(data, dict) else {}
    return {}


def _check_zarr(zarr_path: Path, tuning_keys: List[str]) -> Dict[str, object]:
    if not zarr_path.exists():
        return {
            "zarr_exists": False,
            "pipeline_type": None,
            "zarr_purpose": None,
            "has_raw_video_attr": None,
            "raw_present": False,
            "full_present": False,
            "ds_present": False,
            "sampled_present": False,
            "background_full_present": False,
            "background_ds_present": False,
            "detect_present": False,
            "detect_method": None,
            "detect_coverage": None,
            "detect_coverage_basis": None,
            "detect_quality_present": False,
            "detect_quality_run": None,
            "detect_quality_grade": None,
            "detect_quality_score": None,
            "detect_quality_clean_percent": None,
            "detect_quality_artifacts": None,
            "refined_detect_present": False,
            "refined_detect_coverage": None,
            "refined_detect_method": None,
            "refined_detect_resolved_group": None,
            "detect_review_status": None,
            "crop_present": False,
            "crop_status": None,
            "crop_review_status": None,
            "keypoints_present": False,
            "refined_keypoints_present": False,
            "refined_keypoints_coverage": None,
            "refined_keypoints_success": None,
            "keypoint_review_status": None,
            "eye_masks_present": False,
            "assign_ids_present": False,
            "track_present": False,
            "stimulus_runs": 0,
            "calibration_present": False,
            "tuning_present": 0,
            "tuning_total": len(tuning_keys),
            "tuning_missing": tuning_keys,
            "tuning_status": {key: "miss" for key in tuning_keys},
        }

    try:
        root = zarr.open_group(str(zarr_path), mode="r", consolidated=False)
    except TypeError:
        root = zarr.open_group(str(zarr_path), mode="r")
    pipeline_type = _normalize_attr(root.attrs.get("pipeline_type"))
    zarr_purpose = _normalize_attr(root.attrs.get("zarr_purpose"))
    has_raw_video_attr = root.attrs.get("has_raw_video")
    if isinstance(has_raw_video_attr, (bytes, bytearray)):
        has_raw_video_attr = has_raw_video_attr.decode("utf-8", "ignore")
    if isinstance(has_raw_video_attr, str):
        if has_raw_video_attr.lower() in {"true", "1", "yes"}:
            has_raw_video_attr = True
        elif has_raw_video_attr.lower() in {"false", "0", "no"}:
            has_raw_video_attr = False
        else:
            has_raw_video_attr = None

    raw = root.get("raw_video")
    raw_present = raw is not None
    full_present = raw_present and "images_full" in raw
    ds_present = raw_present and "images_ds" in raw
    sampled_present = raw_present and "original_frame_indices" in raw

    background_full_present = False
    background_ds_present = False
    bg_runs = root.get("background_runs")
    if bg_runs is not None:
        latest_bg = bg_runs.attrs.get("latest")
        if latest_bg and latest_bg in bg_runs:
            latest_group = bg_runs[latest_bg]
            background_full_present = "background_full" in latest_group
            background_ds_present = "background_ds" in latest_group
    if not (background_full_present and background_ds_present):
        legacy_bg = root.get("background")
        if legacy_bg is not None:
            background_full_present = "background_full" in legacy_bg
            background_ds_present = "background_ds" in legacy_bg

    detect_present = False
    detect_method: Optional[str] = None
    detect_coverage: Optional[float] = None
    detect_coverage_basis: Optional[str] = None
    detect_quality_present = False
    detect_quality_run: Optional[str] = None
    detect_quality_grade: Optional[str] = None
    detect_quality_score: Optional[float] = None
    detect_quality_clean_percent: Optional[float] = None
    detect_quality_artifacts: Optional[int] = None
    detect_parent = root.get("detect_runs")
    if detect_parent is not None:
        latest_detect = detect_parent.attrs.get("latest")
        detect_candidate = None
        if latest_detect and latest_detect in detect_parent:
            detect_candidate = latest_detect
        else:
            if hasattr(detect_parent, "group_keys"):
                names = list(detect_parent.group_keys())
            else:
                names = list(detect_parent.keys())
            if names:
                detect_candidate = sorted(names)[-1]
        if detect_candidate:
            detect_present = True
            detect_group = detect_parent[detect_candidate]
            detect_method = _extract_detect_method(detect_group)
            detect_coverage, detect_coverage_basis = _extract_detect_coverage(detect_group, root)
            quality_info = _extract_detect_quality(detect_group)
            detect_quality_present = bool(quality_info["present"])
            detect_quality_run = _normalize_attr(quality_info["run"])
            detect_quality_grade = _normalize_attr(quality_info["grade"])
            detect_quality_score = _coerce_float(quality_info["score"])
            detect_quality_clean_percent = _coerce_float(quality_info["clean_percent"])
            detect_quality_artifacts = _coerce_int(quality_info["artifacts"])

    refined_detect_present = False
    refined_detect_coverage: Optional[float] = None
    refined_detect_method: Optional[str] = None
    refined_detect_resolved_group: Optional[str] = None
    detect_review_status: Optional[Dict[str, object]] = None
    refined_parent = root.get("refined_detect_runs") or root.get("refined_runs")
    if refined_parent is not None:
        latest_refined = refined_parent.attrs.get("latest")
        candidate_run = None
        if latest_refined and latest_refined in refined_parent:
            candidate_run = latest_refined
        else:
            if hasattr(refined_parent, "group_keys"):
                names = list(refined_parent.group_keys())
            else:
                names = list(refined_parent.keys())
            if names:
                candidate_run = sorted(names)[-1]
        if candidate_run:
            refined_detect_present = True
            refined_group = refined_parent[candidate_run]
            refined_detect_coverage, refined_detect_method = _extract_refined_coverage(refined_group, root)
            resolution = resolve_refined_detect_group(
                refined_group, preference=DEFAULT_DETECT_GROUP_PREFERENCE
            )
            refined_detect_resolved_group = resolution.group or resolution.label
            detect_review_status = _coerce_mapping(refined_group.attrs.get("detect_review_status"))

    crop_present = False
    crop_status: Optional[str] = None
    crop_review_status: Optional[Dict[str, object]] = None
    crop_parent = root.get("crop_runs")
    if crop_parent is not None:
        latest_crop = crop_parent.attrs.get("latest")
        if latest_crop and latest_crop in crop_parent:
            crop_present = True
            crop_status = _normalize_attr(crop_parent[latest_crop].attrs.get("status"))
            crop_review_status = _coerce_mapping(
                crop_parent[latest_crop].attrs.get("crop_review_status")
            )
        else:
            if hasattr(crop_parent, "group_keys"):
                names = list(crop_parent.group_keys())
            else:
                names = list(crop_parent.keys())
            crop_present = len(names) > 0
            if names:
                fallback_crop = sorted(names)[-1]
                try:
                    crop_status = _normalize_attr(crop_parent[fallback_crop].attrs.get("status"))
                except Exception:
                    crop_status = None

    keypoints_present = False
    keypoints_parent = root.get("keypoints_runs")
    if keypoints_parent is not None:
        latest_keypoints = _normalize_attr(keypoints_parent.attrs.get("latest"))
        if latest_keypoints and latest_keypoints in keypoints_parent:
            keypoints_present = True
        else:
            if hasattr(keypoints_parent, "group_keys"):
                keypoints_present = len(list(keypoints_parent.group_keys())) > 0
            else:
                keypoints_present = len(list(keypoints_parent.keys())) > 0

    refined_keypoints_present = False
    refined_keypoints_coverage: Optional[float] = None
    refined_keypoints_success: Optional[float] = None
    keypoint_review_status: Optional[Dict[str, object]] = None
    refined_keypoints_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
    refined_keypoints_group_name = (
        "refined_keypoints_runs" if "refined_keypoints_runs" in root else "keypoints_refined_runs"
    )
    if refined_keypoints_parent is not None:
        latest_refined_keypoints = _normalize_attr(refined_keypoints_parent.attrs.get("latest"))
        candidate_run = None
        if latest_refined_keypoints and latest_refined_keypoints in refined_keypoints_parent:
            candidate_run = latest_refined_keypoints
        else:
            if hasattr(refined_keypoints_parent, "group_keys"):
                names = list(refined_keypoints_parent.group_keys())
            else:
                names = list(refined_keypoints_parent.keys())
            if names:
                candidate_run = sorted(names)[-1]
        if candidate_run:
            refined_keypoints_present = True
            refined_kp_group = refined_keypoints_parent[candidate_run]
            refined_keypoints_success, refined_keypoints_coverage = _extract_refined_keypoints_stats(refined_kp_group)
            keypoint_review_status = _coerce_mapping(
                refined_kp_group.attrs.get("keypoint_review_status")
            )
            if not keypoint_review_status:
                attrs = _load_group_attrs(
                    zarr_path, f"{refined_keypoints_group_name}/{candidate_run}"
                )
                keypoint_review_status = _coerce_mapping(attrs.get("keypoint_review_status"))
            if not keypoint_review_status:
                review_latest = _normalize_attr(
                    refined_keypoints_parent.attrs.get("keypoint_review_status_latest")
                )
                if review_latest:
                    attrs = _load_group_attrs(
                        zarr_path, f"{refined_keypoints_group_name}/{review_latest}"
                    )
                    keypoint_review_status = _coerce_mapping(attrs.get("keypoint_review_status"))
        else:
            if hasattr(refined_keypoints_parent, "group_keys"):
                refined_keypoints_present = len(list(refined_keypoints_parent.group_keys())) > 0
            else:
                refined_keypoints_present = len(list(refined_keypoints_parent.keys())) > 0

    eye_masks_present = False
    eye_masks_parent = root.get("eye_masks_runs")
    if eye_masks_parent is not None:
        latest_eye_masks = eye_masks_parent.attrs.get("latest")
        if latest_eye_masks and latest_eye_masks in eye_masks_parent:
            eye_masks_present = True
        else:
            if hasattr(eye_masks_parent, "group_keys"):
                eye_masks_present = len(list(eye_masks_parent.group_keys())) > 0
            else:
                eye_masks_present = len(list(eye_masks_parent.keys())) > 0

    assign_ids_present = False
    assign_ids_parent = root.get("id_assignment_runs")
    if assign_ids_parent is not None:
        latest_assign = assign_ids_parent.attrs.get("latest")
        if latest_assign and latest_assign in assign_ids_parent:
            assign_ids_present = True
        else:
            if hasattr(assign_ids_parent, "group_keys"):
                assign_ids_present = len(list(assign_ids_parent.group_keys())) > 0
            else:
                assign_ids_present = len(list(assign_ids_parent.keys())) > 0

    track_present = False
    track_parent = root.get("tracking_runs")
    if track_parent is not None:
        latest_track = track_parent.attrs.get("latest")
        if latest_track and latest_track in track_parent:
            track_present = True
        else:
            if hasattr(track_parent, "group_keys"):
                track_present = len(list(track_parent.group_keys())) > 0
            else:
                track_present = len(list(track_parent.keys())) > 0

    stim_runs = 0
    analysis = root.get("analysis")
    if analysis is not None and "stimulus_runs" in analysis:
        stim_group = analysis["stimulus_runs"]
        if hasattr(stim_group, "group_keys"):
            stim_runs = len(list(stim_group.group_keys()))
        else:
            stim_runs = len(list(stim_group.keys()))

    calibration_present = "calibration" in root

    subdish_needed = subdish_required(root.attrs)

    tuning_missing: List[str] = []
    tuning_present = 0
    tuning_total = 0
    tuning_status: Dict[str, str] = {}
    analysis_meta = root.get("analysis_metadata")
    attrs = analysis_meta.attrs if analysis_meta is not None else {}
    for key in tuning_keys:
        if key in attrs:
            tuning_present += 1
            tuning_total += 1
            tuning_status[key] = "ok"
            continue
        if key == "subdish_mask_tuning" and not subdish_needed:
            tuning_status[key] = "na"
            continue
        tuning_total += 1
        tuning_status[key] = "miss"
        if tuning_status[key] == "miss":
            tuning_missing.append(key)

    return {
        "zarr_exists": True,
        "pipeline_type": pipeline_type,
        "zarr_purpose": zarr_purpose,
        "has_raw_video_attr": has_raw_video_attr,
        "raw_present": raw_present,
        "full_present": full_present,
        "ds_present": ds_present,
        "sampled_present": sampled_present,
        "background_full_present": background_full_present,
        "background_ds_present": background_ds_present,
        "detect_present": detect_present,
        "detect_method": detect_method,
        "detect_coverage": detect_coverage,
        "detect_coverage_basis": detect_coverage_basis,
        "detect_quality_present": detect_quality_present,
        "detect_quality_run": detect_quality_run,
        "detect_quality_grade": detect_quality_grade,
        "detect_quality_score": detect_quality_score,
        "detect_quality_clean_percent": detect_quality_clean_percent,
        "detect_quality_artifacts": detect_quality_artifacts,
        "refined_detect_present": refined_detect_present,
        "refined_detect_coverage": refined_detect_coverage,
        "refined_detect_method": refined_detect_method,
        "refined_detect_resolved_group": refined_detect_resolved_group,
        "detect_review_status": detect_review_status,
        "crop_present": crop_present,
        "crop_status": crop_status,
        "crop_review_status": crop_review_status,
        "keypoints_present": keypoints_present,
        "refined_keypoints_present": refined_keypoints_present,
        "refined_keypoints_coverage": refined_keypoints_coverage,
        "refined_keypoints_success": refined_keypoints_success,
        "keypoint_review_status": keypoint_review_status,
        "eye_masks_present": eye_masks_present,
        "assign_ids_present": assign_ids_present,
        "track_present": track_present,
        "stimulus_runs": stim_runs,
        "calibration_present": calibration_present,
        "tuning_present": tuning_present,
        "tuning_total": tuning_total,
        "tuning_missing": tuning_missing,
        "tuning_status": tuning_status,
    }


def _status_text(value: Optional[bool]) -> str:
    if value is None:
        return "N/A"
    return "OK" if value else "MISS"


def _tuning_status_text(value: str) -> str:
    if value == "ok":
        return "OK"
    if value == "na":
        return "N/A"
    return "MISS"


def _status_rich(value: Optional[bool]) -> str:
    if value is None:
        return "N/A"
    return "[chartreuse1]OK[/chartreuse1]" if value else "[red]MISS[/red]"


def _crop_status_text(present: bool, status: Optional[str]) -> str:
    if not present:
        return "MISS"
    normalized = str(status or "").strip().lower()
    return normalized or "OK"


def _crop_status_rich(present: bool, status: Optional[str]) -> str:
    if not present:
        return "[red]MISS[/red]"
    normalized = str(status or "").strip().lower()
    if normalized == "completed":
        return "[chartreuse1]completed[/chartreuse1]"
    if normalized == "running":
        return "[yellow]running[/yellow]"
    if normalized == "failed":
        return "[red]failed[/red]"
    if normalized:
        return f"[dim]{normalized}[/dim]"
    return "[chartreuse1]OK[/chartreuse1]"


def _tuning_status_rich(value: str) -> str:
    if value == "ok":
        return "[chartreuse1]OK[/chartreuse1]"
    if value == "na":
        return "N/A"
    return "[red]MISS[/red]"


def _percent_text(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if value >= 99.999:
        return "100%"
    return f"{value:.1f}%"


def _percent_rich(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if value >= 99.999:
        return "[chartreuse1]100%[/chartreuse1]"
    return f"[yellow]{value:.1f}%[/yellow]"


def _refined_status_text(coverage: Optional[float], method: Optional[str]) -> str:
    if coverage is None:
        return "MISS"
    percent = _percent_text(coverage) or "—"
    if method:
        return f"{percent} ({method})"
    return percent


def _detect_status_text(
    present: bool,
    method: Optional[str],
    coverage: Optional[float],
    basis: Optional[str],
) -> str:
    if not present:
        return "MISS"
    percent = _percent_text(coverage)
    if percent is None and method:
        return f"OK ({method})"
    if percent is None:
        return "OK"
    if basis and method:
        return f"OK ({percent}, {basis}, {method})"
    if method:
        return f"OK ({percent}, {method})"
    if basis:
        return f"OK ({percent}, {basis})"
    return f"OK ({percent})"


def _detect_status_rich(
    present: bool,
    method: Optional[str],
    coverage: Optional[float],
    basis: Optional[str],
) -> str:
    if not present:
        return "[red]MISS[/red]"
    percent = _percent_rich(coverage)
    method_text = f"[dim]{method}[/dim]" if method else None
    if percent is None and method_text is not None:
        return f"[chartreuse1]OK[/chartreuse1] ({method_text})"
    if percent is None:
        return "[chartreuse1]OK[/chartreuse1]"
    if basis and method_text is not None:
        return f"[chartreuse1]OK[/chartreuse1] ({percent}, [dim]{basis}[/dim], {method_text})"
    if method_text is not None:
        return f"[chartreuse1]OK[/chartreuse1] ({percent}, {method_text})"
    if basis:
        return f"[chartreuse1]OK[/chartreuse1] ({percent}, [dim]{basis}[/dim])"
    return f"[chartreuse1]OK[/chartreuse1] ({percent})"


def _refined_status_rich(coverage: Optional[float], method: Optional[str]) -> str:
    if coverage is None:
        return "[red]MISS[/red]"
    percent = _percent_rich(coverage) or "[dim]—[/dim]"
    if method:
        return f"{percent} [dim]({method})[/dim]"
    return percent


def _detect_quality_status_text(
    present: bool,
    grade: Optional[str],
    score: Optional[float],
    clean_percent: Optional[float],
    artifacts: Optional[int],
) -> str:
    if not present:
        return "MISS"
    parts: List[str] = []
    if grade and score is not None:
        parts.append(f"{grade} {score:.1f}")
    elif grade:
        parts.append(grade)
    elif score is not None:
        parts.append(f"{score:.1f}")
    if clean_percent is not None:
        parts.append(f"clean {clean_percent:.1f}%")
    if artifacts is not None:
        parts.append(f"art {artifacts}")
    if parts:
        return f"OK ({', '.join(parts)})"
    return "OK"


def _detect_quality_status_rich(
    present: bool,
    grade: Optional[str],
    score: Optional[float],
    clean_percent: Optional[float],
    artifacts: Optional[int],
) -> str:
    text = _detect_quality_status_text(
        present=present,
        grade=grade,
        score=score,
        clean_percent=clean_percent,
        artifacts=artifacts,
    )
    if not present:
        return "[red]MISS[/red]"
    if grade == "A":
        return f"[chartreuse1]{text}[/chartreuse1]"
    if grade in {"B", "C"}:
        return f"[yellow]{text}[/yellow]"
    if grade in {"D", "F"}:
        return f"[red]{text}[/red]"
    return f"[chartreuse1]{text}[/chartreuse1]"


def _keypoint_status_text(success: Optional[float], usable: Optional[float]) -> str:
    if success is None and usable is None:
        return "MISS"
    success_text = _percent_text(success) or "—"
    if usable is None:
        return success_text
    usable_text = _percent_text(usable) or "—"
    return f"{success_text} (train {usable_text})"


def _keypoint_status_rich(success: Optional[float], usable: Optional[float]) -> str:
    if success is None and usable is None:
        return "[red]MISS[/red]"
    success_text = _percent_rich(success) or "[dim]—[/dim]"
    if usable is None:
        return success_text
    usable_text = _percent_rich(usable) or "[dim]—[/dim]"
    return f"{success_text} (train {usable_text})"


def _review_status_text(status: Optional[Dict[str, object]]) -> str:
    if not status:
        return "—"
    state = str(status.get("state", "")).strip()
    method = str(status.get("method", "")).strip()
    intended_use = str(status.get("intended_use", "")).strip()
    resolved_group = str(status.get("resolved_group", "")).strip()
    parts: List[str] = []
    if method:
        parts.append(method)
    if intended_use:
        parts.append(intended_use)
    if resolved_group:
        parts.append(f"group={resolved_group}")
    label = state or "review"
    if parts:
        return f"{label} ({', '.join(parts)})"
    if state:
        return state
    return "—"


def _review_status_rich(status: Optional[Dict[str, object]]) -> str:
    if not status:
        return "[dim]—[/dim]"
    state = str(status.get("state", "")).strip().lower()
    label = _review_status_text(status)
    if state == "approved":
        return f"[chartreuse1]{label}[/chartreuse1]"
    if state in ("rejected", "fail", "failed"):
        return f"[red]{label}[/red]"
    if state in ("pending", "needs_review", "review"):
        return f"[yellow]{label}[/yellow]"
    return f"[dim]{label}[/dim]"


def _resolved_group_text(group: Optional[str]) -> str:
    return group or "—"


def _resolved_group_rich(group: Optional[str]) -> str:
    if not group:
        return "[dim]—[/dim]"
    return group


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check which processing steps have been completed for recordings.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording root(s) to scan (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for recordings under each root.",
    )
    parser.add_argument(
        "--tuning-keys",
        type=str,
        help="Comma-separated tuning keys to check (default: dish_mask,detection_tuning,keypoint_tuning,eye_mask_tuning,subdish_mask_tuning).",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich table output.",
    )
    parser.add_argument(
        "--zarr-use",
        choices=("all", "training", "analysis"),
        default="all",
        help="Filter status rows by zarr use type (default: all).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path to resolve recording zarr artifacts.",
    )
    parser.add_argument(
        "--registry-prefer-crop-review",
        action="store_true",
        help="Prefer crop review status from crop_quality_current when --registry is provided.",
    )

    args = parser.parse_args(argv)
    roots = _resolve_root(args.paths)
    tuning_keys = (
        [key.strip() for key in args.tuning_keys.split(",") if key.strip()]
        if args.tuning_keys
        else DEFAULT_TUNING_KEYS
    )
    registry: Optional[Registry] = None
    if args.registry is not None:
        registry_path = args.registry.expanduser().resolve()
        if not registry_path.exists():
            print(f"Registry not found: {registry_path}")
            return 1
        registry = Registry(registry_path)
    if args.registry_prefer_crop_review and registry is None:
        print("--registry-prefer-crop-review requires --registry.")
        return 1

    plans: List[RecordingStatus] = []
    try:
        for h5_path in _iter_h5(roots, args.recursive):
            recording_dir = h5_path.parent.parent
            camera_id = _read_camera_id(h5_path)
            recording_id = _read_recording_id(h5_path)
            zarr_candidates = _resolve_zarr_candidates(
                recording_dir=recording_dir,
                recording_id=recording_id,
                requested_use=str(args.zarr_use),
                registry=registry,
            )
            for zarr_path, zarr_use in zarr_candidates:
                zarr_info = _check_zarr(zarr_path, tuning_keys)
                if args.registry_prefer_crop_review and registry is not None:
                    crop_registry = _registry_crop_review_status_for_zarr(
                        registry=registry,
                        zarr_path=zarr_path,
                    )
                    if crop_registry is not None:
                        zarr_info["crop_present"] = True
                        if crop_registry.get("crop_review_status") is not None:
                            zarr_info["crop_review_status"] = crop_registry["crop_review_status"]
                plans.append(
                    RecordingStatus(
                        recording_dir=recording_dir,
                        h5_path=h5_path,
                        camera_id=camera_id,
                        recording_id=recording_id,
                        zarr_path=zarr_path,
                        zarr_use=zarr_use,
                        zarr_exists=zarr_info["zarr_exists"],
                        pipeline_type=zarr_info["pipeline_type"],
                        zarr_purpose=zarr_info["zarr_purpose"],
                        has_raw_video_attr=zarr_info["has_raw_video_attr"],
                        raw_present=zarr_info["raw_present"],
                        full_present=zarr_info["full_present"],
                        ds_present=zarr_info["ds_present"],
                        sampled_present=zarr_info["sampled_present"],
                        background_full_present=zarr_info["background_full_present"],
                        background_ds_present=zarr_info["background_ds_present"],
                        detect_present=zarr_info["detect_present"],
                        detect_method=zarr_info["detect_method"],
                        detect_coverage=zarr_info["detect_coverage"],
                        detect_coverage_basis=zarr_info["detect_coverage_basis"],
                        detect_quality_present=zarr_info["detect_quality_present"],
                        detect_quality_run=zarr_info["detect_quality_run"],
                        detect_quality_grade=zarr_info["detect_quality_grade"],
                        detect_quality_score=zarr_info["detect_quality_score"],
                        detect_quality_clean_percent=zarr_info["detect_quality_clean_percent"],
                        detect_quality_artifacts=zarr_info["detect_quality_artifacts"],
                        refined_detect_present=zarr_info["refined_detect_present"],
                        refined_detect_coverage=zarr_info["refined_detect_coverage"],
                        refined_detect_method=zarr_info["refined_detect_method"],
                        refined_detect_resolved_group=zarr_info["refined_detect_resolved_group"],
                        detect_review_status=zarr_info["detect_review_status"],
                        crop_present=zarr_info["crop_present"],
                        crop_status=zarr_info["crop_status"],
                        crop_review_status=zarr_info["crop_review_status"],
                        keypoints_present=zarr_info["keypoints_present"],
                        refined_keypoints_present=zarr_info["refined_keypoints_present"],
                        refined_keypoints_coverage=zarr_info["refined_keypoints_coverage"],
                        refined_keypoints_success=zarr_info["refined_keypoints_success"],
                        keypoint_review_status=zarr_info["keypoint_review_status"],
                        eye_masks_present=zarr_info["eye_masks_present"],
                        assign_ids_present=zarr_info["assign_ids_present"],
                        track_present=zarr_info["track_present"],
                        stimulus_runs=zarr_info["stimulus_runs"],
                        calibration_present=zarr_info["calibration_present"],
                        tuning_present=zarr_info["tuning_present"],
                        tuning_total=zarr_info["tuning_total"],
                        tuning_missing=zarr_info["tuning_missing"],
                        tuning_status=zarr_info["tuning_status"],
                    )
                )
    finally:
        if registry is not None:
            registry.close()

    if not plans:
        print("No recordings found.")
        return 1

    use_rich = not args.no_rich and Console is not None and Table is not None
    if use_rich:
        console = Console()
        table = Table(title="Recording Step Status", show_lines=False)
        table.add_column("Recording", style="cyan")
        table.add_column("Camera", style="magenta")
        table.add_column("Zarr")
        table.add_column("Use")
        table.add_column("Purpose")
        table.add_column("Import")
        table.add_column("BG Full")
        table.add_column("BG DS")
        table.add_column("Detect")
        table.add_column("Detect Quality")
        table.add_column("Refine Detect")
        table.add_column("Detect Group")
        table.add_column("Detect Review")
        table.add_column("Crop")
        table.add_column("Crop Review")
        table.add_column("Keypoints")
        table.add_column("Refined Keypoints (analysis/train)")
        table.add_column("Keypoint Review")
        table.add_column("Eye Masks")
        table.add_column("Assign IDs")
        table.add_column("Track")
        table.add_column("Stimulus")
        table.add_column("Calib")
        table.add_column("Tuning")
        for key in tuning_keys:
            table.add_column(key, style="dim")
        for plan in plans:
            is_production = (
                (plan.zarr_purpose == "production")
                or (plan.pipeline_type == "yolo_inference")
                or (plan.has_raw_video_attr is False and not (plan.full_present or plan.ds_present))
            )
            import_ok = None if is_production else (plan.raw_present and (plan.full_present or plan.ds_present))
            stimulus_ok = plan.stimulus_runs > 0
            background_full_ok = None if is_production else plan.background_full_present
            background_ds_ok = None if is_production else plan.background_ds_present
            tuning_text = f"{plan.tuning_present}/{plan.tuning_total}"
            stimulus_text = f"{plan.stimulus_runs} ({_status_rich(stimulus_ok)})"
            row = [
                plan.recording_dir.name,
                plan.camera_id or "unknown",
                _status_rich(plan.zarr_exists),
                plan.zarr_use or "—",
                plan.zarr_purpose or "—",
                _status_rich(import_ok),
                _status_rich(background_full_ok),
                _status_rich(background_ds_ok),
                _detect_status_rich(
                    plan.detect_present,
                    plan.detect_method,
                    plan.detect_coverage,
                    plan.detect_coverage_basis,
                ),
                _detect_quality_status_rich(
                    plan.detect_quality_present,
                    plan.detect_quality_grade,
                    plan.detect_quality_score,
                    plan.detect_quality_clean_percent,
                    plan.detect_quality_artifacts,
                ),
                _refined_status_rich(plan.refined_detect_coverage, plan.refined_detect_method),
                _resolved_group_rich(plan.refined_detect_resolved_group),
                _review_status_rich(plan.detect_review_status),
                _crop_status_rich(plan.crop_present, plan.crop_status),
                _review_status_rich(plan.crop_review_status),
                _status_rich(plan.keypoints_present),
                _keypoint_status_rich(plan.refined_keypoints_success, plan.refined_keypoints_coverage),
                _review_status_rich(plan.keypoint_review_status),
                _status_rich(plan.eye_masks_present),
                _status_rich(plan.assign_ids_present),
                _status_rich(plan.track_present),
                stimulus_text,
                _status_rich(plan.calibration_present),
                "N/A" if is_production else tuning_text,
            ]
            for key in tuning_keys:
                status = "na" if is_production else plan.tuning_status.get(key, "miss")
                row.append(_tuning_status_rich(status))
            table.add_row(*row)
        console.print(table)
    else:
        for plan in plans:
            is_production = (
                (plan.zarr_purpose == "production")
                or (plan.pipeline_type == "yolo_inference")
                or (plan.has_raw_video_attr is False and not (plan.full_present or plan.ds_present))
            )
            import_ok = None if is_production else (plan.raw_present and (plan.full_present or plan.ds_present))
            stimulus_ok = plan.stimulus_runs > 0
            background_full_ok = None if is_production else plan.background_full_present
            background_ds_ok = None if is_production else plan.background_ds_present
            print(plan.recording_dir.name)
            print(f"  camera_id: {plan.camera_id or 'unknown'}")
            print(f"  recording_id: {plan.recording_id or 'unknown'}")
            print(f"  zarr: {_status_text(plan.zarr_exists)}")
            print(f"  use: {plan.zarr_use or '—'}")
            if plan.zarr_purpose:
                print(f"  purpose: {plan.zarr_purpose}")
            print(f"  import: {_status_text(import_ok)}")
            print(f"  background_full: {_status_text(background_full_ok)}")
            print(f"  background_ds: {_status_text(background_ds_ok)}")
            print(
                "  detect: "
                f"{_detect_status_text(plan.detect_present, plan.detect_method, plan.detect_coverage, plan.detect_coverage_basis)}"
            )
            print(
                "  detect_quality: "
                f"{_detect_quality_status_text(plan.detect_quality_present, plan.detect_quality_grade, plan.detect_quality_score, plan.detect_quality_clean_percent, plan.detect_quality_artifacts)}"
            )
            if plan.detect_quality_run:
                print(f"  detect_quality_run: {plan.detect_quality_run}")
            print(
                f"  refined_detect: {_refined_status_text(plan.refined_detect_coverage, plan.refined_detect_method)}"
            )
            print(f"  detect_group: {_resolved_group_text(plan.refined_detect_resolved_group)}")
            print(f"  detect_review_status: {_review_status_text(plan.detect_review_status)}")
            print(f"  crop: {_crop_status_text(plan.crop_present, plan.crop_status)}")
            print(f"  crop_review_status: {_review_status_text(plan.crop_review_status)}")
            print(f"  keypoints: {_status_text(plan.keypoints_present)}")
            print(
                f"  refined_keypoints: "
                f"{_keypoint_status_text(plan.refined_keypoints_success, plan.refined_keypoints_coverage)}"
            )
            print(f"  keypoint_review_status: {_review_status_text(plan.keypoint_review_status)}")
            print(f"  eye_masks: {_status_text(plan.eye_masks_present)}")
            print(f"  assign_ids: {_status_text(plan.assign_ids_present)}")
            print(f"  track: {_status_text(plan.track_present)}")
            print(f"  stimulus_runs: {plan.stimulus_runs} ({_status_text(stimulus_ok)})")
            print(f"  calibration: {_status_text(plan.calibration_present)}")
            if is_production:
                print("  tuning: N/A (production)")
            else:
                print(
                    f"  tuning: {plan.tuning_present}/{plan.tuning_total} "
                    f"(missing: {', '.join(plan.tuning_missing) if plan.tuning_missing else 'none'})"
                )
            for key in tuning_keys:
                status = "na" if is_production else plan.tuning_status.get(key, "miss")
                print(f"    {key}: {_tuning_status_text(status)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
