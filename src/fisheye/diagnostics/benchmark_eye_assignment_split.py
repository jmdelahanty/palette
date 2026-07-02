"""Benchmark eyes-union keypoint split implementations.

This diagnostic is read-only. Synthetic mode is sandbox-safe; real-zarr mode is
intended for local/outside-sandbox validation against subject-mask runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..refinement.assemble_refined_subject_masks import (
    _resolve_eye_keypoint_indices,
    _resolve_keypoint_success_array,
)
from ..refinement.subject_eye_assignment import (
    _split_union_by_keypoints_distance_batch_into,
    _split_union_by_keypoints_halfplane_batch_into,
    _split_union_by_keypoints_sparse_batch_into,
    assign_eyes_union_to_lr,
)
from ..shared.json_safety import json_attr_safe
from ..shared.mask_probability_encoding import decode_probability_values_from_attrs
from ..utils.zarr_io import open_zarr_root


SplitFn = Callable[..., None]


def _as_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _synthetic_inputs(
    *,
    row_count: int,
    height: int,
    width: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    total = max(1, int(row_count))
    h = max(8, int(height))
    w = max(8, int(width))
    union = np.zeros((total, h, w), dtype=np.uint8)
    for row_idx in range(total):
        if row_idx % 11 == 0:
            continue
        y0 = int(rng.integers(0, max(1, h - 8)))
        x0 = int(rng.integers(0, max(1, w - 8)))
        y1 = min(h, y0 + int(rng.integers(4, max(5, min(24, h - y0 + 1)))))
        x1 = min(w, x0 + int(rng.integers(4, max(5, min(24, w - x0 + 1)))))
        union[row_idx, y0:y1, x0:x1] = 1
        if row_idx % 3 == 0:
            y2 = int(rng.integers(0, max(1, h - 8)))
            x2 = int(rng.integers(0, max(1, w - 8)))
            union[row_idx, y2 : min(h, y2 + 8), x2 : min(w, x2 + 8)] = 1

    left = np.column_stack(
        [
            rng.uniform(0.0, max(1.0, float(w) * 0.45), size=total),
            rng.uniform(0.0, max(1.0, float(h)), size=total),
        ]
    ).astype(np.float32)
    right = np.column_stack(
        [
            rng.uniform(float(w) * 0.55, max(1.0, float(w) - 1.0), size=total),
            rng.uniform(0.0, max(1.0, float(h)), size=total),
        ]
    ).astype(np.float32)
    rows = np.arange(total, dtype=np.int64)
    success = np.ones((total,), dtype=bool)
    return union, left, right, success, rows


def _decode_probability_slice(values: np.ndarray, *, threshold: float) -> np.ndarray:
    return (values.astype(np.float32, copy=False) >= float(threshold)).astype(np.uint8, copy=False)


def _resolve_keypoint_group(root: object, subject_group: object, keypoint_group: Optional[str], keypoint_run: Optional[str]):
    run_name = keypoint_run or _as_text(subject_group.attrs.get("source_keypoints_run"))
    if run_name is None:
        run_name = _as_text(subject_group.attrs.get("source_keypoint_run"))
    parent_name = keypoint_group or _as_text(subject_group.attrs.get("source_keypoint_group"))
    if parent_name and run_name:
        parent = root.get(parent_name)
        if parent is None or run_name not in parent:
            raise KeyError(f"Keypoint run {parent_name}/{run_name} not found.")
        return parent[run_name], run_name, parent_name
    if not run_name:
        raise ValueError("Keypoint run not supplied and subject run has no source_keypoints_run attr.")
    matches: list[tuple[object, str]] = []
    for candidate_parent_name in ("refined_keypoints_runs", "keypoints_runs"):
        parent = root.get(candidate_parent_name)
        if parent is not None and run_name in parent:
            matches.append((parent[run_name], candidate_parent_name))
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely resolve keypoint run {run_name!r}; "
            f"matches={[name for _group, name in matches]!r}."
        )
    group, resolved_parent = matches[0]
    return group, run_name, resolved_parent


def _real_inputs(
    *,
    zarr_path: Path,
    subject_run: str,
    keypoint_group: Optional[str],
    keypoint_run: Optional[str],
    start_row: int,
    row_count: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    root = open_zarr_root(zarr_path, mode="r")
    subject_group = root["subject_mask_runs"][str(subject_run)]
    labels = tuple(str(label) for label in subject_group.attrs.get("mask_labels", ()))
    if "eyes_union" not in labels:
        raise KeyError(f"subject_mask_runs/{subject_run} does not expose eyes_union; labels={labels!r}.")
    comp_idx = labels.index("eyes_union")
    if "mask_probs_roi" not in subject_group:
        raise KeyError(f"subject_mask_runs/{subject_run} missing mask_probs_roi.")
    probs_arr = subject_group["mask_probs_roi"]
    total_rows = int(probs_arr.shape[0])
    start = max(0, int(start_row))
    stop = min(total_rows, start + max(1, int(row_count)))
    if stop <= start:
        raise ValueError(f"Empty requested row range start={start} row_count={row_count} total_rows={total_rows}.")

    kp_group, kp_run_name, kp_parent_name = _resolve_keypoint_group(root, subject_group, keypoint_group, keypoint_run)
    if "keypoints_roi" not in kp_group:
        raise KeyError(f"{kp_parent_name}/{kp_run_name} missing keypoints_roi.")
    keypoints_arr = kp_group["keypoints_roi"]
    if int(keypoints_arr.shape[0]) < stop:
        raise ValueError(
            f"{kp_parent_name}/{kp_run_name}/keypoints_roi has {int(keypoints_arr.shape[0])} rows; "
            f"cannot slice rows {start}:{stop}."
        )
    keypoint_success, success_dataset = _resolve_keypoint_success_array(kp_group, kp_run_name)
    eye_left_idx, eye_right_idx = _resolve_eye_keypoint_indices(kp_group, kp_run_name)

    source_path = f"subject_mask_runs/{subject_run}/mask_probs_roi"
    union_probs = decode_probability_values_from_attrs(
        np.asarray(probs_arr[start:stop, comp_idx]),
        attrs=subject_group.attrs,
        source_path=source_path,
    )
    union = _decode_probability_slice(union_probs, threshold=float(threshold))
    keypoints = np.asarray(keypoints_arr[start:stop], dtype=np.float32)
    success = np.asarray(keypoint_success[start:stop], dtype=bool)
    left = np.asarray(keypoints[:, eye_left_idx, :2], dtype=np.float32)
    right = np.asarray(keypoints[:, eye_right_idx, :2], dtype=np.float32)
    finite = np.all(np.isfinite(left), axis=1) & np.all(np.isfinite(right), axis=1)
    nonempty = np.any(np.asarray(union, dtype=bool), axis=(1, 2))
    noncoincident = ~np.all(np.isclose(left, right, atol=1e-3), axis=1)
    rows = np.flatnonzero(success & finite & nonempty & noncoincident).astype(np.int64)
    metadata = {
        "zarr_path": str(zarr_path),
        "subject_run": str(subject_run),
        "keypoint_group": kp_parent_name,
        "keypoint_run": kp_run_name,
        "keypoint_success_dataset": success_dataset,
        "eye_keypoint_indices": [int(eye_left_idx), int(eye_right_idx)],
        "source_rows": [int(start), int(stop)],
        "candidate_rows": int(rows.size),
    }
    return union, left, right, success, rows, metadata


def _run_once(
    fn: SplitFn,
    union: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    rows: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    left_out = np.zeros_like(union, dtype=np.uint8)
    right_out = np.zeros_like(union, dtype=np.uint8)
    started = time.perf_counter()
    fn(
        union,
        left,
        right,
        row_indices=rows,
        left_out=left_out,
        right_out=right_out,
        batch_size=int(batch_size),
    )
    seconds = float(time.perf_counter() - started)
    return left_out, right_out, seconds


def _benchmark(
    name: str,
    fn: SplitFn,
    union: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    rows: np.ndarray,
    *,
    batch_size: int,
    repeat: int,
) -> dict[str, object]:
    timings: list[float] = []
    for _ in range(max(1, int(repeat))):
        _left_out, _right_out, seconds = _run_once(fn, union, left, right, rows, batch_size=batch_size)
        timings.append(float(seconds))
    best = min(timings) if timings else 0.0
    rows_per_second = float(rows.size / best) if best > 0.0 else None
    return {
        "name": name,
        "repeat": int(max(1, int(repeat))),
        "seconds": timings,
        "best_seconds": best,
        "candidate_rows_per_second": rows_per_second,
    }


def _assignment_inputs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    total = int(np.asarray(left).shape[0])
    keypoints = np.zeros((total, 2, 2), dtype=np.float32)
    keypoints[:, 0, :2] = np.asarray(left, dtype=np.float32)
    keypoints[:, 1, :2] = np.asarray(right, dtype=np.float32)
    return keypoints


def _run_assignment_once(
    union: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    success: np.ndarray,
    *,
    use_component_fast_path: bool,
    measure_ellipses: bool,
    batch_size: int,
):
    keypoints = _assignment_inputs(left, right)
    started = time.perf_counter()
    result = assign_eyes_union_to_lr(
        union,
        keypoints_roi=keypoints,
        keypoint_success=np.asarray(success, dtype=bool),
        eye_keypoint_indices=(0, 1),
        split_batch_size=int(batch_size),
        use_component_fast_path=bool(use_component_fast_path),
        measure_ellipses=bool(measure_ellipses),
    )
    return result, float(time.perf_counter() - started)


def _benchmark_assignment(
    name: str,
    union: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    success: np.ndarray,
    *,
    use_component_fast_path: bool,
    measure_ellipses: bool,
    batch_size: int,
    repeat: int,
) -> dict[str, object]:
    timings: list[float] = []
    last_result = None
    for _ in range(max(1, int(repeat))):
        last_result, seconds = _run_assignment_once(
            union,
            left,
            right,
            success,
            use_component_fast_path=bool(use_component_fast_path),
            measure_ellipses=bool(measure_ellipses),
            batch_size=int(batch_size),
        )
        timings.append(float(seconds))
    best = min(timings) if timings else 0.0
    rows_per_second = float(union.shape[0] / best) if best > 0.0 else None
    summary = dict(last_result.summary) if last_result is not None else {}
    return {
        "name": name,
        "repeat": int(max(1, int(repeat))),
        "seconds": timings,
        "best_seconds": best,
        "rows_per_second": rows_per_second,
        "summary": summary,
    }


def _assignment_delta(reference: object, candidate: object) -> dict[str, object]:
    mismatches: dict[str, int] = {}
    for component in ("eye_left", "eye_right"):
        mismatches[f"mask_pixels_{component}"] = int(
            np.count_nonzero(reference.masks[component] != candidate.masks[component])
        )
        mismatches[f"reason_labels_{component}"] = int(
            np.count_nonzero(reference.reason_labels[component] != candidate.reason_labels[component])
        )
    mismatches["assignment_status"] = int(np.count_nonzero(reference.assignment_status != candidate.assignment_status))
    mismatches["ellipse_success"] = int(
        np.count_nonzero(
            np.asarray(reference.eye_geometry["ellipse_success"], dtype=bool)
            != np.asarray(candidate.eye_geometry["ellipse_success"], dtype=bool)
        )
    )
    return {
        "status": "match" if int(sum(mismatches.values())) == 0 else "semantic_delta",
        "mismatches": mismatches,
        "reference_status_counts": dict(reference.summary.get("status_counts") or {}),
        "candidate_status_counts": dict(candidate.summary.get("status_counts") or {}),
    }


def _assignment_parity(reference: object, candidate: object) -> dict[str, object]:
    mismatches: dict[str, int] = {}
    for component in ("eye_left", "eye_right"):
        mismatches[f"mask_pixels_{component}"] = int(
            np.count_nonzero(reference.masks[component] != candidate.masks[component])
        )
        mismatches[f"reason_labels_{component}"] = int(
            np.count_nonzero(reference.reason_labels[component] != candidate.reason_labels[component])
        )
    mismatches["assignment_status"] = int(np.count_nonzero(reference.assignment_status != candidate.assignment_status))
    mismatches["ellipse_success"] = int(
        np.count_nonzero(
            np.asarray(reference.eye_geometry["ellipse_success"], dtype=bool)
            != np.asarray(candidate.eye_geometry["ellipse_success"], dtype=bool)
        )
    )
    ref_ellipse = np.asarray(reference.eye_geometry["ellipse_params"], dtype=np.float32)
    cand_ellipse = np.asarray(candidate.eye_geometry["ellipse_params"], dtype=np.float32)
    ellipse_equal = bool(np.allclose(ref_ellipse, cand_ellipse, equal_nan=True))
    if not ellipse_equal:
        mismatches["ellipse_params_allclose"] = 1
    total = int(sum(mismatches.values()))
    return {
        "status": "match" if total == 0 else "mismatch",
        "mismatches": mismatches,
    }


def _parity_against_reference(
    *,
    reference_left: np.ndarray,
    reference_right: np.ndarray,
    candidate_left: np.ndarray,
    candidate_right: np.ndarray,
) -> dict[str, object]:
    mismatch_pixels = int(
        np.count_nonzero(reference_left != candidate_left)
        + np.count_nonzero(reference_right != candidate_right)
    )
    return {
        "status": "match" if mismatch_pixels == 0 else "mismatch",
        "mismatch_pixels": mismatch_pixels,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, default=None, help="Optional analysis zarr for real-slice mode.")
    parser.add_argument("--subject-run", default=None, help="subject_mask_runs/<run> to load in real-slice mode.")
    parser.add_argument("--keypoint-group", default=None, help="Optional keypoint parent group override.")
    parser.add_argument("--keypoint-run", default=None, help="Optional keypoint run override.")
    parser.add_argument("--start-row", type=int, default=0, help="Start row for real-slice mode.")
    parser.add_argument("--row-count", type=int, default=512, help="Rows to generate or load.")
    parser.add_argument("--height", type=int, default=128, help="Synthetic mask height.")
    parser.add_argument("--width", type=int, default=128, help="Synthetic mask width.")
    parser.add_argument("--seed", type=int, default=123, help="Synthetic RNG seed.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for real eyes_union masks.")
    parser.add_argument("--batch-size", type=int, default=32, help="Rows per split batch.")
    parser.add_argument("--repeat", type=int, default=5, help="Benchmark repeats.")
    parser.add_argument(
        "--assignment-repeat",
        type=int,
        default=0,
        help="Also benchmark full assignment variants, including no-ellipse-QC diagnostic mode.",
    )
    parser.add_argument(
        "--skip-component-fast-path-assignment",
        action="store_true",
        help=(
            "When --assignment-repeat is set, skip the disabled component-fast-path assignment candidate. "
            "Useful when benchmarking only the no-ellipse-QC diagnostic path."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.zarr is not None:
        if not args.subject_run:
            raise SystemExit("--subject-run is required with --zarr.")
        union, left, right, success, rows, metadata = _real_inputs(
            zarr_path=args.zarr,
            subject_run=str(args.subject_run),
            keypoint_group=args.keypoint_group,
            keypoint_run=args.keypoint_run,
            start_row=int(args.start_row),
            row_count=int(args.row_count),
            threshold=float(args.threshold),
        )
        mode = "real_zarr"
    else:
        union, left, right, success, rows = _synthetic_inputs(
            row_count=int(args.row_count),
            height=int(args.height),
            width=int(args.width),
            seed=int(args.seed),
        )
        metadata = {}
        mode = "synthetic"

    distance_left, distance_right, _distance_seconds = _run_once(
        _split_union_by_keypoints_distance_batch_into,
        union,
        left,
        right,
        rows,
        batch_size=int(args.batch_size),
    )
    candidate_fns: list[tuple[str, SplitFn]] = [
        ("halfplane_batch", _split_union_by_keypoints_halfplane_batch_into),
        ("sparse_batch", _split_union_by_keypoints_sparse_batch_into),
    ]
    parity: dict[str, object] = {}
    for candidate_name, candidate_fn in candidate_fns:
        candidate_left, candidate_right, _candidate_seconds = _run_once(
            candidate_fn,
            union,
            left,
            right,
            rows,
            batch_size=int(args.batch_size),
        )
        parity[candidate_name] = _parity_against_reference(
            reference_left=distance_left,
            reference_right=distance_right,
            candidate_left=candidate_left,
            candidate_right=candidate_right,
        )

    nonzero_pixels = int(np.count_nonzero(union[rows])) if int(rows.size) > 0 else 0
    candidate_pixel_capacity = int(rows.size * int(union.shape[1]) * int(union.shape[2]))
    payload = {
        "mode": mode,
        "metadata": metadata,
        "shape": list(union.shape),
        "row_count": int(union.shape[0]),
        "candidate_rows": int(rows.size),
        "nonzero_union_pixels": nonzero_pixels,
        "foreground_density": (
            float(nonzero_pixels / candidate_pixel_capacity)
            if candidate_pixel_capacity > 0
            else None
        ),
        "batch_size": int(args.batch_size),
        "repeat": int(args.repeat),
        "parity_vs_distance_batch": parity,
        "benchmarks": [
            _benchmark(
                "distance_batch",
                _split_union_by_keypoints_distance_batch_into,
                union,
                left,
                right,
                rows,
                batch_size=int(args.batch_size),
                repeat=int(args.repeat),
            ),
            _benchmark(
                "halfplane_batch",
                _split_union_by_keypoints_halfplane_batch_into,
                union,
                left,
                right,
                rows,
                batch_size=int(args.batch_size),
                repeat=int(args.repeat),
            ),
            _benchmark(
                "sparse_batch",
                _split_union_by_keypoints_sparse_batch_into,
                union,
                left,
                right,
                rows,
                batch_size=int(args.batch_size),
                repeat=int(args.repeat),
            ),
        ],
    }
    if int(args.assignment_repeat) > 0:
        standard_assignment, _standard_seconds = _run_assignment_once(
            union,
            left,
            right,
            success,
            use_component_fast_path=False,
            measure_ellipses=True,
            batch_size=int(args.batch_size),
        )
        no_ellipse_assignment, _no_ellipse_seconds = _run_assignment_once(
            union,
            left,
            right,
            success,
            use_component_fast_path=False,
            measure_ellipses=False,
            batch_size=int(args.batch_size),
        )
        assignment_parity: dict[str, object] = {}
        if not bool(args.skip_component_fast_path_assignment):
            fast_assignment, _fast_seconds = _run_assignment_once(
                union,
                left,
                right,
                success,
                use_component_fast_path=True,
                measure_ellipses=True,
                batch_size=int(args.batch_size),
            )
            assignment_parity["component_fast_path"] = _assignment_parity(standard_assignment, fast_assignment)
        payload["assignment_parity"] = assignment_parity
        payload["assignment_deltas"] = {
            "no_ellipse_qc": _assignment_delta(standard_assignment, no_ellipse_assignment)
        }
        assignment_benchmarks = [
            _benchmark_assignment(
                "standard_assignment",
                union,
                left,
                right,
                success,
                use_component_fast_path=False,
                measure_ellipses=True,
                batch_size=int(args.batch_size),
                repeat=int(args.assignment_repeat),
            ),
            _benchmark_assignment(
                "standard_assignment_no_ellipse_qc",
                union,
                left,
                right,
                success,
                use_component_fast_path=False,
                measure_ellipses=False,
                batch_size=int(args.batch_size),
                repeat=int(args.assignment_repeat),
            ),
        ]
        if not bool(args.skip_component_fast_path_assignment):
            assignment_benchmarks.insert(
                1,
                _benchmark_assignment(
                    "component_fast_path_assignment",
                    union,
                    left,
                    right,
                    success,
                    use_component_fast_path=True,
                    measure_ellipses=True,
                    batch_size=int(args.batch_size),
                    repeat=int(args.assignment_repeat),
                ),
            )
        payload["assignment_benchmarks"] = assignment_benchmarks
    print(json.dumps(json_attr_safe(payload), sort_keys=True, indent=2))
    split_ok = all(str(item.get("status")) == "match" for item in parity.values())
    assignment_ok = True
    if "assignment_parity" in payload:
        assignment_ok = all(
            str(item.get("status")) == "match"
            for item in dict(payload["assignment_parity"]).values()
        )
    return 0 if split_ok and assignment_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
