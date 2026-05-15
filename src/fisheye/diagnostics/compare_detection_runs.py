"""Compare two Palette detect_runs for backend/model parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import zarr


def _read_array(group: zarr.Group, name: str) -> np.ndarray:
    if name not in group:
        raise KeyError(f"detect run {group.path!r} is missing required array {name!r}")
    return np.asarray(group[name][:])


def _resolve_run(parent: zarr.Group, run_name: str | None) -> tuple[str, zarr.Group]:
    if run_name in {None, "", "latest"}:
        latest = parent.attrs.get("latest")
        if not latest:
            raise KeyError("detect_runs/latest is not set; pass --run explicitly")
        run_name = str(latest)
    if run_name not in parent:
        raise KeyError(f"detect run {run_name!r} not found")
    return str(run_name), parent[run_name]


def _frame_selection(
    *,
    total_frames: int,
    frames: Sequence[int] | None,
    frame_start: int,
    frame_stop: int | None,
    frame_step: int,
) -> np.ndarray:
    if frames:
        selected = sorted({int(frame) for frame in frames})
    else:
        stop = total_frames if frame_stop is None else min(int(frame_stop), total_frames)
        selected = list(range(int(frame_start), stop, int(frame_step)))
    return np.asarray(
        [frame for frame in selected if 0 <= frame < total_frames],
        dtype=np.int64,
    )


def _rows_for_frames(
    frame_indices: np.ndarray,
    frames: Iterable[int],
) -> dict[int, np.ndarray]:
    rows: dict[int, list[int]] = {int(frame): [] for frame in frames}
    if not rows:
        return {}
    for row_index, frame in enumerate(frame_indices):
        frame_int = int(frame)
        if frame_int in rows:
            rows[frame_int].append(int(row_index))
    return {
        frame: np.asarray(row_indices, dtype=np.int64)
        for frame, row_indices in rows.items()
    }


def compare_detection_runs(
    *,
    zarr_path: Path,
    run_a: str | None,
    run_b: str | None,
    frames: Sequence[int] | None = None,
    frame_start: int = 0,
    frame_stop: int | None = None,
    frame_step: int = 1,
) -> dict[str, object]:
    try:
        root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    except TypeError:  # pragma: no cover - older zarr compatibility
        root = zarr.open_group(str(zarr_path), mode="r")
    if "detect_runs" not in root:
        raise KeyError(f"{zarr_path} does not contain detect_runs")

    parent = root["detect_runs"]
    name_a, group_a = _resolve_run(parent, run_a)
    name_b, group_b = _resolve_run(parent, run_b)

    counts_a = _read_array(group_a, "n_detections")
    counts_b = _read_array(group_b, "n_detections")
    total_frames = int(min(counts_a.shape[0], counts_b.shape[0]))
    selected_frames = _frame_selection(
        total_frames=total_frames,
        frames=frames,
        frame_start=frame_start,
        frame_stop=frame_stop,
        frame_step=frame_step,
    )
    if selected_frames.size == 0:
        raise ValueError("No frames selected for comparison")

    selected_counts_a = counts_a[selected_frames]
    selected_counts_b = counts_b[selected_frames]
    count_mismatch_mask = selected_counts_a != selected_counts_b
    count_mismatch_frames = selected_frames[count_mismatch_mask]

    frame_indices_a = _read_array(group_a, "frame_indices")
    frame_indices_b = _read_array(group_b, "frame_indices")
    bbox_a = _read_array(group_a, "bbox_norm_coords")
    bbox_b = _read_array(group_b, "bbox_norm_coords")
    scores_a = _read_array(group_a, "scores")
    scores_b = _read_array(group_b, "scores")
    class_a = np.asarray(group_a["class_ids"][:]) if "class_ids" in group_a else None
    class_b = np.asarray(group_b["class_ids"][:]) if "class_ids" in group_b else None

    rows_a = _rows_for_frames(frame_indices_a, selected_frames)
    rows_b = _rows_for_frames(frame_indices_b, selected_frames)
    bbox_abs_diffs: list[np.ndarray] = []
    score_abs_diffs: list[np.ndarray] = []
    class_mismatches = 0
    equal_count_frames = 0

    for frame in selected_frames:
        frame_int = int(frame)
        idx_a = rows_a[frame_int]
        idx_b = rows_b[frame_int]
        if idx_a.size != idx_b.size:
            continue
        equal_count_frames += 1
        if idx_a.size == 0:
            continue
        bbox_abs_diffs.append(np.abs(bbox_a[idx_a] - bbox_b[idx_b]))
        score_abs_diffs.append(np.abs(scores_a[idx_a] - scores_b[idx_b]))
        if class_a is not None and class_b is not None:
            class_mismatches += int(np.sum(class_a[idx_a] != class_b[idx_b]))

    bbox_diff = np.concatenate([arr.reshape(-1) for arr in bbox_abs_diffs]) if bbox_abs_diffs else np.array([])
    score_diff = np.concatenate(score_abs_diffs) if score_abs_diffs else np.array([])

    return {
        "zarr_path": str(zarr_path),
        "run_a": name_a,
        "run_b": name_b,
        "frames_compared": int(selected_frames.size),
        "frame_start": int(selected_frames[0]),
        "frame_stop": int(selected_frames[-1]) + 1,
        "detections_a": int(np.sum(selected_counts_a)),
        "detections_b": int(np.sum(selected_counts_b)),
        "count_mismatch_frames": int(count_mismatch_frames.size),
        "count_exact_match_fraction": float(1.0 - (count_mismatch_frames.size / selected_frames.size)),
        "first_count_mismatch_frames": [int(frame) for frame in count_mismatch_frames[:20]],
        "equal_count_frames": int(equal_count_frames),
        "bbox_abs_diff_max": float(np.max(bbox_diff)) if bbox_diff.size else 0.0,
        "bbox_abs_diff_mean": float(np.mean(bbox_diff)) if bbox_diff.size else 0.0,
        "score_abs_diff_max": float(np.max(score_diff)) if score_diff.size else 0.0,
        "score_abs_diff_mean": float(np.mean(score_diff)) if score_diff.size else 0.0,
        "class_mismatches": int(class_mismatches),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--run-a", required=True, help="First detect run name, or latest.")
    parser.add_argument("--run-b", required=True, help="Second detect run name, or latest.")
    parser.add_argument("--frames", nargs="+", type=int, default=None, help="Explicit frame IDs to compare.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--fail-on-count-mismatch", action="store_true")
    parser.add_argument("--max-bbox-diff", type=float, default=None)
    parser.add_argument("--max-score-diff", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args(argv)

    result = compare_detection_runs(
        zarr_path=args.zarr_path.expanduser(),
        run_a=args.run_a,
        run_b=args.run_b,
        frames=args.frames,
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
        frame_step=args.frame_step,
    )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"zarr: {result['zarr_path']}")
        print(f"run_a: {result['run_a']}")
        print(f"run_b: {result['run_b']}")
        print(
            "frames={frames_compared} detections_a={detections_a} detections_b={detections_b} "
            "count_match={count_exact_match_fraction:.4f}".format(**result)
        )
        print(
            "bbox_abs_diff max={bbox_abs_diff_max:.6g} mean={bbox_abs_diff_mean:.6g}; "
            "score_abs_diff max={score_abs_diff_max:.6g} mean={score_abs_diff_mean:.6g}; "
            "class_mismatches={class_mismatches}".format(**result)
        )
        if result["first_count_mismatch_frames"]:
            print(f"first count mismatch frames: {result['first_count_mismatch_frames']}")

    failed = False
    if args.fail_on_count_mismatch and int(result["count_mismatch_frames"]) > 0:
        failed = True
    if args.max_bbox_diff is not None and float(result["bbox_abs_diff_max"]) > args.max_bbox_diff:
        failed = True
    if args.max_score_diff is not None and float(result["score_abs_diff_max"]) > args.max_score_diff:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
