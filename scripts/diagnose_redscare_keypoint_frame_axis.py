"""Inspect RedScare training zarr keypoint frame-axis consistency read-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr


DEFAULT_ROOTS = (
    Path("/groups/johnson/johnsonlab/jeremy/recordings"),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _shape(group: Any, name: str) -> list[int] | None:
    if group is None or name not in group:
        return None
    return [int(dim) for dim in group[name].shape]


def _attrs_subset(group: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if group is None:
        return {}
    attrs = getattr(group, "attrs", {})
    return {key: _jsonable(attrs.get(key)) for key in keys if key in attrs}


def _read_int_array(group: Any, name: str) -> np.ndarray | None:
    if group is None or name not in group:
        return None
    return np.asarray(group[name][:], dtype=np.int64)


def _leading_dim(group: Any, name: str) -> int | None:
    shape = _shape(group, name)
    if not shape:
        return None
    return int(shape[0])


def _safe_minmax(values: np.ndarray | None) -> dict[str, int | None]:
    if values is None or values.size == 0:
        return {"min": None, "max": None}
    return {"min": int(np.min(values)), "max": int(np.max(values))}


def _count_missing_in_range(values: np.ndarray | None, length: int | None) -> int | None:
    if values is None or length is None:
        return None
    if length <= 0:
        return 0
    valid = values[(values >= 0) & (values < int(length))]
    return int(length - np.unique(valid).size)


def _array_length_gaps(group: Any, frame_count_len: int | None, row_count: int | None) -> dict[str, Any]:
    frame_axis_arrays = ("frame_counts", "n_rois", "n_keypoints")
    row_axis_arrays = (
        "frame_indices",
        "detection_indices",
        "source_refined_row_ids",
        "source_detect_row_index",
        "keypoints_roi",
        "keypoints_img",
        "keypoints_norm",
        "source_keypoints_img",
        "confidence",
        "keypoint_confidences",
        "detection_success",
        "heading",
        "heading_finite",
        "heading_usable",
    )
    return {
        "frame_axis": {
            name: {
                "shape": _shape(group, name),
                "delta_vs_frame_counts": (
                    None if frame_count_len is None or _leading_dim(group, name) is None else int(_leading_dim(group, name) - frame_count_len)
                ),
            }
            for name in frame_axis_arrays
            if name in group
        },
        "row_axis": {
            name: {
                "shape": _shape(group, name),
                "delta_vs_keypoint_rows": (
                    None if row_count is None or _leading_dim(group, name) is None else int(_leading_dim(group, name) - row_count)
                ),
            }
            for name in row_axis_arrays
            if name in group
        },
    }


def _summarize_keypoint_run(group: Any, run_name: str) -> dict[str, Any]:
    frame_indices = _read_int_array(group, "frame_indices")
    frame_counts = _read_int_array(group, "frame_counts")
    n_keypoints = _read_int_array(group, "n_keypoints")
    detection_success = None
    if "detection_success" in group:
        detection_success = np.asarray(group["detection_success"][:], dtype=bool)
    keypoints_shape = _shape(group, "keypoints_roi")
    row_count = int(keypoints_shape[0]) if keypoints_shape else None
    frame_count_len = int(frame_counts.shape[0]) if frame_counts is not None else None
    n_keypoints_len = int(n_keypoints.shape[0]) if n_keypoints is not None else None
    unique_frames = int(np.unique(frame_indices).size) if frame_indices is not None else None
    bincount_matches = None
    if frame_indices is not None and frame_counts is not None:
        expected = np.bincount(
            frame_indices.astype(np.int64, copy=False),
            minlength=int(frame_counts.shape[0]),
        )[: int(frame_counts.shape[0])]
        bincount_matches = bool(np.array_equal(expected.astype(np.int64), frame_counts.astype(np.int64)))

    return {
        "run_name": run_name,
        "attrs": _attrs_subset(
            group,
            (
                "source_crop_run",
                "source_keypoints_run",
                "source_keypoint_run",
                "source_analysis_zarr",
                "run_id",
                "method",
                "created_at_utc",
                "source_frame_domain",
                "frame_index_domain",
            ),
        ),
        "shapes": {
            "frame_indices": _shape(group, "frame_indices"),
            "frame_counts": _shape(group, "frame_counts"),
            "n_rois": _shape(group, "n_rois"),
            "n_keypoints": _shape(group, "n_keypoints"),
            "keypoints_roi": _shape(group, "keypoints_roi"),
            "keypoints_img": _shape(group, "keypoints_img"),
            "keypoints_norm": _shape(group, "keypoints_norm"),
        },
        "frame_axis": {
            "frame_counts_len": frame_count_len,
            "n_keypoints_len": n_keypoints_len,
            "gap_frame_counts_minus_n_keypoints": (
                None if frame_count_len is None or n_keypoints_len is None else int(frame_count_len - n_keypoints_len)
            ),
            "frame_indices_minmax": _safe_minmax(frame_indices),
            "frame_indices_unique_count": unique_frames,
            "frame_indices_missing_in_0_to_frame_counts_len_minus_1": _count_missing_in_range(frame_indices, frame_count_len),
            "frame_counts_sum": None if frame_counts is None else int(np.sum(frame_counts)),
            "frame_counts_nonzero": None if frame_counts is None else int(np.count_nonzero(frame_counts)),
            "n_keypoints_nonzero": None if n_keypoints is None else int(np.count_nonzero(n_keypoints)),
            "n_keypoints_unique_values": None if n_keypoints is None else [int(value) for value in np.unique(n_keypoints)[:20]],
            "frame_counts_matches_bincount_frame_indices": bincount_matches,
            "rows_beyond_n_keypoints_axis": (
                None
                if frame_indices is None or n_keypoints_len is None
                else int(np.count_nonzero(frame_indices >= int(n_keypoints_len)))
            ),
            "successful_rows_beyond_n_keypoints_axis": (
                None
                if frame_indices is None or n_keypoints_len is None or detection_success is None
                else int(np.count_nonzero((frame_indices >= int(n_keypoints_len)) & detection_success))
            ),
        },
        "row_axis": {
            "keypoint_row_count": row_count,
            "keypoints_landmarks": None if not keypoints_shape or len(keypoints_shape) < 2 else int(keypoints_shape[1]),
        },
        "array_consistency": _array_length_gaps(group, frame_count_len, row_count),
    }


def _pick_crop_runs(root: Any) -> list[tuple[str, Any]]:
    parent = root.get("crop_runs") if hasattr(root, "get") else None
    if parent is None:
        return []
    names = sorted(str(name) for name in parent.group_keys())
    preferred = [name for name in names if "red_scare" in name.lower() or "acquisition_crop" in name.lower()]
    names = preferred or names
    return [(name, parent[name]) for name in names]


def _summarize_crop_run(group: Any, run_name: str) -> dict[str, Any]:
    frame_indices = _read_int_array(group, "frame_indices")
    frame_counts = _read_int_array(group, "frame_counts")
    source_training_rows = _read_int_array(group, "source_training_row_indices")
    crop_video_frames = _read_int_array(group, "source_crop_video_frame_indices")
    crop_meta_rows = _read_int_array(group, "source_crop_meta_row_indices")
    roi_shape = _shape(group, "roi_images")
    row_count = int(roi_shape[0]) if roi_shape else _leading_dim(group, "frame_indices")
    frame_count_len = int(frame_counts.shape[0]) if frame_counts is not None else None
    return {
        "run_name": run_name,
        "attrs": _attrs_subset(
            group,
            (
                "source_sample_count",
                "selected_sample_count",
                "rejected_missing_crop_meta_frame",
                "rejected_blank_crop_frame",
                "rejected_crop_has_no_detection",
                "rejected_nonfinite_crop_geometry",
                "blank_crop_frames_excluded",
                "crop_detection_required",
                "source_crop_video_frame_indices_semantics",
                "source_training_row_indices_semantics",
                "total_frames",
            ),
        ),
        "shapes": {
            "roi_images": roi_shape,
            "frame_indices": _shape(group, "frame_indices"),
            "source_training_row_indices": _shape(group, "source_training_row_indices"),
            "source_crop_video_frame_indices": _shape(group, "source_crop_video_frame_indices"),
            "source_crop_meta_row_indices": _shape(group, "source_crop_meta_row_indices"),
            "source_crop_local_frame_ids": _shape(group, "source_crop_local_frame_ids"),
            "source_recording_frame_ids": _shape(group, "source_recording_frame_ids"),
            "frame_counts": _shape(group, "frame_counts"),
        },
        "frame_axis": {
            "frame_counts_len": frame_count_len,
            "frame_indices_minmax": _safe_minmax(frame_indices),
            "frame_indices_unique_count": None if frame_indices is None else int(np.unique(frame_indices).size),
            "frame_indices_missing_in_0_to_frame_counts_len_minus_1": _count_missing_in_range(frame_indices, frame_count_len),
            "frame_counts_sum": None if frame_counts is None else int(np.sum(frame_counts)),
        },
        "row_axis": {
            "crop_row_count": row_count,
            "source_training_row_minmax": _safe_minmax(source_training_rows),
            "source_training_rows_unique_count": None if source_training_rows is None else int(np.unique(source_training_rows).size),
            "source_crop_video_frame_minmax": _safe_minmax(crop_video_frames),
            "source_crop_video_frames_unique_count": None if crop_video_frames is None else int(np.unique(crop_video_frames).size),
            "source_crop_meta_row_minmax": _safe_minmax(crop_meta_rows),
        },
    }


def _summarize_raw_video(root: Any) -> dict[str, Any]:
    raw = root.get("raw_video") if hasattr(root, "get") else None
    if raw is None:
        return {}
    original = _read_int_array(raw, "original_frame_indices")
    images_full = _shape(raw, "images_full")
    return {
        "attrs": _attrs_subset(raw, ("frame_step", "import_mode", "source_total_frames", "decode_backend")),
        "shapes": {
            "images_full": images_full,
            "images_ds": _shape(raw, "images_ds"),
            "original_frame_indices": _shape(raw, "original_frame_indices"),
        },
        "original_frame_indices_minmax": _safe_minmax(original),
        "original_frame_indices_unique_count": None if original is None else int(np.unique(original).size),
    }


def summarize_zarr(path: Path) -> dict[str, Any]:
    root = zarr.open_group(str(path), mode="r", use_consolidated=False)
    keypoint_parent = root.get("keypoints_runs")
    keypoint_runs: list[dict[str, Any]] = []
    if keypoint_parent is not None:
        for run_name in sorted(str(name) for name in keypoint_parent.group_keys()):
            if "red_scare" in run_name.lower() or "training_review" in run_name.lower():
                keypoint_runs.append(_summarize_keypoint_run(keypoint_parent[run_name], run_name))
    return {
        "path": str(path),
        "root_attrs": _attrs_subset(
            root,
            (
                "zarr_purpose",
                "zarr_use",
                "schema_id",
                "training_source_type",
                "source_analysis_zarr",
                "source_crop_meta_path",
                "source_crop_video_path",
                "created_at_utc",
                "total_frames",
                "n_frames",
            ),
        ),
        "raw_video": _summarize_raw_video(root),
        "crop_runs": [_summarize_crop_run(group, run_name) for run_name, group in _pick_crop_runs(root)],
        "keypoints_runs": keypoint_runs,
    }


def discover_zarrs(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file() or root.suffix == ".zarr":
            paths.append(root)
            continue
        paths.extend(sorted(root.glob("*RedScare*_training.zarr")))
        paths.extend(sorted(root.glob("*red*scare*_training.zarr")))
        paths.extend(sorted(root.glob("*RedScare/zarr/*RedScare*_training.zarr")))
        paths.extend(sorted(root.glob("*red*scare/zarr/*red*scare*_training.zarr")))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path
    return [unique[key] for key in sorted(unique)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Training zarr path(s) or directories to scan.")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = list(args.paths) if args.paths else list(DEFAULT_ROOTS)
    report = {"zarrs": []}
    for zarr_path in discover_zarrs(roots):
        try:
            report["zarrs"].append(summarize_zarr(zarr_path))
        except Exception as exc:  # pragma: no cover - diagnostic script
            report["zarrs"].append({"path": str(zarr_path), "error": repr(exc)})
    text = json.dumps(_jsonable(report), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
