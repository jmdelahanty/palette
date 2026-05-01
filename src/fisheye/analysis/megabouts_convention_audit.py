"""Read-only audit for Palette tail geometry versus Megabouts tail angles."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import zarr

from ..utils.zarr_io import open_zarr_root

DEFAULT_MEGABOUTS_KEYPOINT_COUNT = 11

ComputeAnglesFn = Callable[..., tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class MegaboutsAngleAuditArrays:
    """Arrays used by the read-only Megabouts convention audit."""

    source_tail_sample_s: np.ndarray
    tail_sample_xy: np.ndarray
    head_xy: np.ndarray
    palette_tail_angle_rad: np.ndarray
    frame_index: np.ndarray
    valid: np.ndarray


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _require_group(parent: zarr.Group, name: str) -> zarr.Group:
    group = parent.get(name)
    if not isinstance(group, zarr.Group):
        raise ValueError(f"Missing required group: {parent.name}/{name}")
    return group


def _require_array(group: zarr.Group, name: str) -> object:
    arr = group.get(name)
    if arr is None:
        raise ValueError(f"Missing required array: {group.name}/{name}")
    return arr


def _resolve_run(parent: zarr.Group, run_name: Optional[str], *, kind: str) -> tuple[str, zarr.Group]:
    name = str(run_name or parent.attrs.get("latest") or "")
    if not name:
        raise ValueError(f"No {kind} run specified and {parent.name}.attrs['latest'] is missing.")
    if name not in parent:
        raise ValueError(f"{parent.name}/{name} not found.")
    run = parent[name]
    if not isinstance(run, zarr.Group):
        raise ValueError(f"{parent.name}/{name} is not a group.")
    return name, run


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    valid: Optional[np.ndarray] = None
    for arr in arrays:
        data = np.asarray(arr)
        row_valid = np.all(np.isfinite(data.reshape((data.shape[0], -1))), axis=1)
        valid = row_valid if valid is None else (valid & row_valid)
    if valid is None:
        raise ValueError("At least one array is required.")
    return valid


def resample_tail_keypoints(
    *,
    source_tail_sample_s: np.ndarray,
    tail_sample_xy: np.ndarray,
    target_count: int = DEFAULT_MEGABOUTS_KEYPOINT_COUNT,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Resample ordered Palette tail geometry to fixed-count keypoints.

    Points remain ordered from tail base to tail tip.
    """

    source_s = np.asarray(source_tail_sample_s, dtype=np.float64).reshape(-1)
    if int(target_count) < 2:
        raise ValueError("target_count must be >= 2.")
    if source_s.ndim != 1 or int(source_s.shape[0]) < 2:
        raise ValueError("source_tail_sample_s must contain at least two positions.")
    if np.any(~np.isfinite(source_s)) or np.any(np.diff(source_s) <= 0.0):
        raise ValueError("source_tail_sample_s must be finite and strictly increasing.")

    xy = np.asarray(tail_sample_xy, dtype=np.float64)
    if xy.ndim != 3 or int(xy.shape[2]) != 2:
        raise ValueError("tail_sample_xy must have shape (N, S, 2).")
    if int(xy.shape[1]) != int(source_s.shape[0]):
        raise ValueError("source_tail_sample_s length must match tail_sample_xy.shape[1].")

    row_count = int(xy.shape[0])
    row_valid = np.ones((row_count,), dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    if int(row_valid.shape[0]) != row_count:
        raise ValueError("valid must have one entry per row.")

    target_s = np.linspace(float(source_s[0]), float(source_s[-1]), int(target_count), dtype=np.float64)
    out = np.full((row_count, int(target_count), 2), np.nan, dtype=np.float32)
    for row_idx in range(row_count):
        if not bool(row_valid[row_idx]):
            continue
        row = xy[row_idx]
        if not np.all(np.isfinite(row)):
            continue
        out[row_idx, :, 0] = np.interp(target_s, source_s, row[:, 0]).astype(np.float32)
        out[row_idx, :, 1] = np.interp(target_s, source_s, row[:, 1]).astype(np.float32)
    return out


def _load_megabouts_compute_angles(megabouts_path: Optional[str | Path] = None) -> ComputeAnglesFn:
    """Import Megabouts' keypoint-to-angle function from package or checkout."""

    if megabouts_path is not None:
        path = str(Path(megabouts_path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        module = importlib.import_module("megabouts.tracking_data.convert_tracking")
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Unable to import megabouts.tracking_data.convert_tracking. "
            "Install Megabouts or pass --megabouts-path /path/to/megabouts."
        ) from exc
    fn = getattr(module, "compute_angles_from_keypoints", None)
    if not callable(fn):
        raise RuntimeError("Megabouts compute_angles_from_keypoints is not callable.")
    return fn


def compute_megabouts_angles_from_tail_keypoints(
    *,
    head_xy: np.ndarray,
    tail_keypoints_xy: np.ndarray,
    compute_angles_fn: ComputeAnglesFn,
) -> tuple[np.ndarray, np.ndarray]:
    """Call Megabouts' conversion function on Palette-derived keypoints."""

    head = np.asarray(head_xy, dtype=np.float64)
    tail = np.asarray(tail_keypoints_xy, dtype=np.float64)
    if head.ndim != 2 or int(head.shape[1]) != 2:
        raise ValueError("head_xy must have shape (N, 2).")
    if tail.ndim != 3 or int(tail.shape[2]) != 2:
        raise ValueError("tail_keypoints_xy must have shape (N, K, 2).")
    if int(tail.shape[0]) != int(head.shape[0]):
        raise ValueError("head_xy and tail_keypoints_xy must have the same row count.")
    tail_angle, head_yaw = compute_angles_fn(
        head_x=head[:, 0],
        head_y=head[:, 1],
        tail_x=tail[:, :, 0],
        tail_y=tail[:, :, 1],
    )
    if tail_angle is None:
        raise ValueError("Megabouts returned no tail_angle; at least two keypoints are required.")
    return np.asarray(tail_angle, dtype=np.float64), np.asarray(head_yaw, dtype=np.float64)


def _angle_residual(megabouts_angle: np.ndarray, palette_angle: np.ndarray, *, sign: float) -> np.ndarray:
    residual = np.asarray(megabouts_angle, dtype=np.float64) - float(sign) * np.asarray(palette_angle, dtype=np.float64)
    return np.arctan2(np.sin(residual), np.cos(residual))


def _residual_summary(residual: np.ndarray) -> dict[str, object]:
    finite = np.asarray(residual, dtype=np.float64)[np.isfinite(residual)]
    if finite.size == 0:
        return {
            "count": 0,
            "rmse_rad": None,
            "median_abs_rad": None,
            "p95_abs_rad": None,
            "max_abs_rad": None,
            "rmse_deg": None,
            "median_abs_deg": None,
            "p95_abs_deg": None,
            "max_abs_deg": None,
        }
    abs_values = np.abs(finite)
    return {
        "count": int(finite.size),
        "rmse_rad": float(np.sqrt(np.mean(np.square(finite)))),
        "median_abs_rad": float(np.median(abs_values)),
        "p95_abs_rad": float(np.percentile(abs_values, 95.0)),
        "max_abs_rad": float(np.max(abs_values)),
        "rmse_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(finite))))),
        "median_abs_deg": float(np.rad2deg(np.median(abs_values))),
        "p95_abs_deg": float(np.rad2deg(np.percentile(abs_values, 95.0))),
        "max_abs_deg": float(np.rad2deg(np.max(abs_values))),
    }


def compare_megabouts_to_palette_angles(
    *,
    megabouts_tail_angle_rad: np.ndarray,
    palette_tail_angle_rad: np.ndarray,
    valid: np.ndarray,
    frame_index: Optional[np.ndarray] = None,
    max_worst_rows: int = 10,
) -> dict[str, object]:
    """Summarize whether Megabouts angles match Palette angles directly or sign-flipped."""

    meg = np.asarray(megabouts_tail_angle_rad, dtype=np.float64)
    pal = np.asarray(palette_tail_angle_rad, dtype=np.float64)
    if meg.ndim != 2 or pal.ndim != 2:
        raise ValueError("Angle arrays must have shape (N, K).")
    if int(meg.shape[0]) != int(pal.shape[0]):
        raise ValueError("Angle arrays must have the same row count.")

    row_count = int(meg.shape[0])
    channel_count = min(int(meg.shape[1]), int(pal.shape[1]))
    if channel_count < 1:
        raise ValueError("Angle arrays must have at least one channel.")

    row_valid = np.asarray(valid, dtype=bool).reshape(-1)
    if int(row_valid.shape[0]) != row_count:
        raise ValueError("valid must have one entry per row.")
    frame = (
        np.arange(row_count, dtype=np.int64)
        if frame_index is None
        else np.asarray(frame_index).reshape(-1).astype(np.int64, copy=False)
    )
    if int(frame.shape[0]) != row_count:
        raise ValueError("frame_index must have one entry per row.")

    meg_common = meg[:, :channel_count]
    pal_common = pal[:, :channel_count]
    finite = row_valid & _finite_rows(meg_common, pal_common)
    direct = _angle_residual(meg_common[finite], pal_common[finite], sign=1.0)
    flipped = _angle_residual(meg_common[finite], pal_common[finite], sign=-1.0)
    direct_summary = _residual_summary(direct)
    flipped_summary = _residual_summary(flipped)
    direct_rmse = direct_summary["rmse_rad"]
    flipped_rmse = flipped_summary["rmse_rad"]
    if direct_rmse is None and flipped_rmse is None:
        best_sign = None
        best_label = "insufficient_data"
        best_residual = np.empty((0, channel_count), dtype=np.float64)
    elif flipped_rmse is None or (direct_rmse is not None and float(direct_rmse) <= float(flipped_rmse)):
        best_sign = 1
        best_label = "direct"
        best_residual = direct
    else:
        best_sign = -1
        best_label = "sign_flipped"
        best_residual = flipped

    per_channel: list[dict[str, object]] = []
    for channel_idx in range(channel_count):
        channel_direct = _angle_residual(meg_common[finite, channel_idx], pal_common[finite, channel_idx], sign=1.0)
        channel_flipped = _angle_residual(meg_common[finite, channel_idx], pal_common[finite, channel_idx], sign=-1.0)
        per_channel.append(
            {
                "channel": int(channel_idx),
                "direct": _residual_summary(channel_direct),
                "sign_flipped": _residual_summary(channel_flipped),
            }
        )

    worst_rows: list[dict[str, object]] = []
    finite_rows = np.flatnonzero(finite)
    if best_residual.size:
        row_abs = np.max(np.abs(best_residual), axis=1)
        order = np.argsort(row_abs)[::-1][: max(0, int(max_worst_rows))]
        for idx in order:
            row_idx = int(finite_rows[int(idx)])
            worst_rows.append(
                {
                    "row": row_idx,
                    "frame_index": int(frame[row_idx]),
                    "max_abs_residual_rad": float(row_abs[int(idx)]),
                    "max_abs_residual_deg": float(np.rad2deg(row_abs[int(idx)])),
                }
            )

    return {
        "row_count": row_count,
        "valid_row_count": int(np.count_nonzero(finite)),
        "megabouts_channel_count": int(meg.shape[1]),
        "palette_channel_count": int(pal.shape[1]),
        "comparison_channel_count": int(channel_count),
        "best_mapping": best_label,
        "best_palette_to_megabouts_sign": best_sign,
        "direct": direct_summary,
        "sign_flipped": flipped_summary,
        "per_channel": per_channel,
        "worst_rows": worst_rows,
    }


def _read_audit_arrays(
    root: zarr.Group,
    *,
    subject_shape_run: Optional[str] = None,
    tail_kinematics_run: Optional[str] = None,
    head_source: str = "head_endpoint_xy",
) -> tuple[str, str, MegaboutsAngleAuditArrays]:
    analysis = _require_group(root, "analysis")
    shape_parent = _require_group(analysis, "subject_shape_runs")
    shape_name, shape = _resolve_run(shape_parent, subject_shape_run, kind="subject-shape")
    tail_parent = _require_group(analysis, "tail_kinematics_runs")
    tail_name, tail_run = _resolve_run(tail_parent, tail_kinematics_run, kind="tail-kinematics")

    body = _require_group(_require_group(shape, "components"), "subject_body")
    tail_sample_xy = np.asarray(_require_array(body, "tail_sample_xy")[:], dtype=np.float32)
    row_count = int(tail_sample_xy.shape[0])
    source_tail_sample_s = np.asarray(_require_array(body, "tail_sample_s")[:], dtype=np.float32)
    head_xy = np.asarray(_require_array(body, head_source)[:], dtype=np.float32)
    palette_tail_angle = np.asarray(_require_array(tail_run, "tail_angle_rad")[:], dtype=np.float32)

    valid = _finite_rows(tail_sample_xy, head_xy, palette_tail_angle)
    for group, name in ((body, "tail_sample_valid"), (body, "bspline_valid"), (tail_run, "valid")):
        arr = group.get(name)
        if arr is not None:
            data = np.asarray(arr[:], dtype=bool).reshape(-1)
            if int(data.shape[0]) != row_count:
                raise ValueError(f"{group.name}/{name} row count does not match subject-shape rows.")
            valid &= data

    frame_arr = tail_run.get("frame_index")
    if frame_arr is None:
        row_index = tail_run.get("row_index")
        if isinstance(row_index, zarr.Group) and row_index.get("frame_indices") is not None:
            frame_arr = row_index["frame_indices"]
    frame_index = (
        np.arange(row_count, dtype=np.int64)
        if frame_arr is None
        else np.asarray(frame_arr[:], dtype=np.int64).reshape(-1)
    )

    if head_xy.shape != (row_count, 2):
        raise ValueError(f"{body.name}/{head_source} must have shape (N, 2).")
    if int(palette_tail_angle.shape[0]) != row_count:
        raise ValueError("tail_kinematics tail_angle_rad row count does not match subject-shape rows.")
    if int(frame_index.shape[0]) != row_count:
        raise ValueError("frame_index row count does not match subject-shape rows.")

    return (
        shape_name,
        tail_name,
        MegaboutsAngleAuditArrays(
            source_tail_sample_s=source_tail_sample_s,
            tail_sample_xy=tail_sample_xy,
            head_xy=head_xy,
            palette_tail_angle_rad=palette_tail_angle,
            frame_index=frame_index,
            valid=valid,
        ),
    )


def audit_megabouts_tail_convention_group(
    root: zarr.Group,
    *,
    subject_shape_run: Optional[str] = None,
    tail_kinematics_run: Optional[str] = None,
    megabouts_path: Optional[str | Path] = None,
    compute_angles_fn: Optional[ComputeAnglesFn] = None,
    head_source: str = "head_endpoint_xy",
    keypoint_count: int = DEFAULT_MEGABOUTS_KEYPOINT_COUNT,
    max_worst_rows: int = 10,
) -> dict[str, object]:
    """Run a read-only convention audit from already-materialized Palette runs."""

    shape_name, tail_name, arrays = _read_audit_arrays(
        root,
        subject_shape_run=subject_shape_run,
        tail_kinematics_run=tail_kinematics_run,
        head_source=head_source,
    )
    tail_keypoints = resample_tail_keypoints(
        source_tail_sample_s=arrays.source_tail_sample_s,
        tail_sample_xy=arrays.tail_sample_xy,
        target_count=int(keypoint_count),
        valid=arrays.valid,
    )
    compute_fn = compute_angles_fn or _load_megabouts_compute_angles(megabouts_path)
    megabouts_angle, head_yaw = compute_megabouts_angles_from_tail_keypoints(
        head_xy=arrays.head_xy,
        tail_keypoints_xy=tail_keypoints,
        compute_angles_fn=compute_fn,
    )
    comparison = compare_megabouts_to_palette_angles(
        megabouts_tail_angle_rad=megabouts_angle,
        palette_tail_angle_rad=arrays.palette_tail_angle_rad,
        valid=arrays.valid,
        frame_index=arrays.frame_index,
        max_worst_rows=int(max_worst_rows),
    )
    direct_p95_deg = comparison.get("direct", {}).get("p95_abs_deg")  # type: ignore[union-attr]
    direct_mapping_small_residual = (
        comparison.get("best_mapping") == "direct"
        and direct_p95_deg is not None
        and float(direct_p95_deg) <= 5.0
    )
    recommendation = (
        "direct_mapping_may_be_safe_after_visual_review"
        if direct_mapping_small_residual
        else "derive_megabouts_tail_angle_from_k11_keypoints_for_classifier_adapter"
    )
    summary: dict[str, object] = {
        "status": "ok",
        "mutates_archive": False,
        "source_subject_shape_run": shape_name,
        "source_tail_kinematics_run": tail_name,
        "head_source": head_source,
        "megabouts_keypoint_count": int(keypoint_count),
        "megabouts_segment_count": int(megabouts_angle.shape[1]),
        "palette_tail_angle_sample_count": int(arrays.palette_tail_angle_rad.shape[1]),
        "input_valid_row_count": int(np.count_nonzero(arrays.valid)),
        "head_yaw_finite_count": int(np.count_nonzero(np.isfinite(head_yaw))),
        "comparison": comparison,
        "direct_mapping_small_residual": bool(direct_mapping_small_residual),
        "direct_mapping_small_residual_threshold_p95_deg": 5.0,
        "recommendation": recommendation,
    }
    return dict(_json_safe(summary))


def audit_megabouts_tail_convention(
    zarr_path: str | Path,
    *,
    subject_shape_run: Optional[str] = None,
    tail_kinematics_run: Optional[str] = None,
    megabouts_path: Optional[str | Path] = None,
    head_source: str = "head_endpoint_xy",
    keypoint_count: int = DEFAULT_MEGABOUTS_KEYPOINT_COUNT,
    max_worst_rows: int = 10,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r")
    return audit_megabouts_tail_convention_group(
        root,
        subject_shape_run=subject_shape_run,
        tail_kinematics_run=tail_kinematics_run,
        megabouts_path=megabouts_path,
        head_source=head_source,
        keypoint_count=keypoint_count,
        max_worst_rows=max_worst_rows,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit of Palette tail angles against Megabouts K=11 keypoint conversion."
    )
    parser.add_argument("zarr_path", type=Path, help="Palette analysis zarr archive.")
    parser.add_argument("--subject-shape-run", help="analysis/subject_shape_runs/<run>; defaults to latest.")
    parser.add_argument("--tail-kinematics-run", help="analysis/tail_kinematics_runs/<run>; defaults to latest.")
    parser.add_argument(
        "--megabouts-path",
        type=Path,
        help="Optional local Megabouts checkout path to prepend to sys.path.",
    )
    parser.add_argument(
        "--head-source",
        default="head_endpoint_xy",
        choices=("head_endpoint_xy", "snout_tip_xy"),
        help="Subject-shape head point used by Megabouts keypoint conversion.",
    )
    parser.add_argument(
        "--keypoint-count",
        type=int,
        default=DEFAULT_MEGABOUTS_KEYPOINT_COUNT,
        help="Megabouts tail keypoint count; 11 gives 10 cumulative angle channels.",
    )
    parser.add_argument("--max-worst-rows", type=int, default=10, help="Number of largest-residual rows to report.")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = audit_megabouts_tail_convention(
        args.zarr_path,
        subject_shape_run=args.subject_shape_run,
        tail_kinematics_run=args.tail_kinematics_run,
        megabouts_path=args.megabouts_path,
        head_source=str(args.head_source),
        keypoint_count=int(args.keypoint_count),
        max_worst_rows=int(args.max_worst_rows),
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
