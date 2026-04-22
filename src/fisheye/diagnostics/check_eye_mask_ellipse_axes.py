"""Validate major/minor ellipse axis ordering for legacy eye-mask runs.

This diagnostic intentionally checks historical/compat ``eye_masks_runs`` and
``refined_eye_masks_runs`` layouts. Current reviewed eye geometry is canonical
under ``refined_subject_masks_runs``.

This checks whether successful eye ellipse fits satisfy `major >= minor`
consistently for `eye_masks_runs` and/or `refined_eye_masks_runs`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import zarr


StageName = str


@dataclass
class RunAxisReport:
    stage: StageName
    run_name: str
    total_rois: int = 0
    channels: int = 0
    success_eyes: int = 0
    validated_eyes: int = 0
    positive_valid_eyes: int = 0
    noncanonical_eyes: int = 0
    nonpositive_eyes: int = 0
    nonfinite_eyes: int = 0
    equal_axis_eyes: int = 0
    missing: List[str] = field(default_factory=list)
    shape_error: Optional[str] = None
    violation_samples: List[dict[str, int | float]] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.missing
            or self.shape_error
            or self.noncanonical_eyes
            or self.nonpositive_eyes
            or self.nonfinite_eyes
        )


def _group_names(parent: zarr.Group) -> List[str]:
    if hasattr(parent, "group_keys"):
        return sorted(str(name) for name in parent.group_keys())
    names: List[str] = []
    for name in parent.keys():
        try:
            obj = parent[name]
        except Exception:
            continue
        if isinstance(obj, zarr.Group):
            names.append(str(name))
    return sorted(names)


def _resolve_run_names(parent: zarr.Group, explicit_run: Optional[str], all_runs: bool) -> List[str]:
    if explicit_run:
        if explicit_run not in parent:
            raise ValueError(f"Run '{explicit_run}' not found.")
        return [explicit_run]
    if all_runs:
        names = _group_names(parent)
        if names:
            return names
        return []
    latest = parent.attrs.get("latest")
    if isinstance(latest, str) and latest in parent:
        return [latest]
    names = _group_names(parent)
    return names[-1:] if names else []


def _analyze_run_group(
    stage: StageName,
    run_name: str,
    run_group: zarr.Group,
    *,
    tolerance: float,
    sample_limit: int,
    chunk_size: int,
) -> RunAxisReport:
    report = RunAxisReport(stage=stage, run_name=run_name)
    required = ("ellipse_params", "ellipse_success")
    for name in required:
        if name not in run_group:
            report.missing.append(name)
    if report.missing:
        return report

    ellipse_params_arr = run_group["ellipse_params"]
    ellipse_success_arr = run_group["ellipse_success"]

    try:
        params_shape = tuple(int(v) for v in ellipse_params_arr.shape)
        success_shape = tuple(int(v) for v in ellipse_success_arr.shape)
    except Exception as exc:
        report.shape_error = f"unable to read shapes: {exc}"
        return report

    if len(params_shape) != 3 or params_shape[2] < 4:
        report.shape_error = f"ellipse_params shape {params_shape} is invalid (expected (N, C, >=4))"
        return report
    if len(success_shape) != 2 or success_shape != params_shape[:2]:
        report.shape_error = (
            f"ellipse_success shape {success_shape} does not match ellipse_params[:2] {params_shape[:2]}"
        )
        return report

    report.total_rois = params_shape[0]
    report.channels = params_shape[1]

    frame_indices_arr = run_group.get("frame_indices")
    frame_indices_ok = False
    if frame_indices_arr is not None:
        try:
            frame_indices_ok = int(frame_indices_arr.shape[0]) == report.total_rois
        except Exception:
            frame_indices_ok = False

    step = max(1, int(chunk_size))
    for start in range(0, report.total_rois, step):
        stop = min(start + step, report.total_rois)
        params_chunk = np.asarray(ellipse_params_arr[start:stop], dtype=np.float32)
        success_chunk = np.asarray(ellipse_success_arr[start:stop], dtype=bool)

        major = params_chunk[:, :, 2]
        minor = params_chunk[:, :, 3]

        finite = np.isfinite(major) & np.isfinite(minor)
        success_and_finite = success_chunk & finite
        positive = success_and_finite & (major > 0.0) & (minor > 0.0)
        nonfinite = success_chunk & ~finite
        nonpositive = success_and_finite & ((major <= 0.0) | (minor <= 0.0))
        noncanonical = positive & ((major + tolerance) < minor)
        equal_axes = positive & np.isclose(major, minor, rtol=0.0, atol=max(tolerance, 0.0))

        report.success_eyes += int(success_chunk.sum())
        report.validated_eyes += int(success_and_finite.sum())
        report.positive_valid_eyes += int(positive.sum())
        report.nonfinite_eyes += int(nonfinite.sum())
        report.nonpositive_eyes += int(nonpositive.sum())
        report.noncanonical_eyes += int(noncanonical.sum())
        report.equal_axis_eyes += int(equal_axes.sum())

        if len(report.violation_samples) < sample_limit and np.any(noncanonical):
            frame_chunk = None
            if frame_indices_ok:
                frame_chunk = np.asarray(frame_indices_arr[start:stop])
            for local_roi, eye_idx in np.argwhere(noncanonical):
                if len(report.violation_samples) >= sample_limit:
                    break
                local_roi_i = int(local_roi)
                eye_idx_i = int(eye_idx)
                sample = {
                    "roi_idx": int(start + local_roi_i),
                    "eye_idx": eye_idx_i,
                    "major": float(major[local_roi_i, eye_idx_i]),
                    "minor": float(minor[local_roi_i, eye_idx_i]),
                }
                if frame_chunk is not None:
                    sample["frame_idx"] = int(frame_chunk[local_roi_i])
                report.violation_samples.append(sample)

    return report


def _print_report(report: RunAxisReport) -> None:
    print(f"\n{report.stage}/{report.run_name}")
    if report.missing:
        print(f"  status: missing datasets ({', '.join(report.missing)})")
        return
    if report.shape_error:
        print(f"  status: shape error ({report.shape_error})")
        return

    print(f"  total_rois: {report.total_rois}")
    print(f"  channels: {report.channels}")
    print(f"  success_eyes: {report.success_eyes}")
    print(f"  validated_eyes (finite major/minor): {report.validated_eyes}")
    print(f"  positive_valid_eyes: {report.positive_valid_eyes}")
    print(f"  noncanonical_eyes (major < minor): {report.noncanonical_eyes}")
    print(f"  nonpositive_eyes (major<=0 or minor<=0): {report.nonpositive_eyes}")
    print(f"  nonfinite_eyes: {report.nonfinite_eyes}")
    print(f"  equal_axis_eyes (major≈minor): {report.equal_axis_eyes}")
    if report.violation_samples:
        print("  sample_noncanonical:")
        for sample in report.violation_samples:
            frame_text = (
                f" frame={sample['frame_idx']}" if "frame_idx" in sample else ""
            )
            print(
                "    roi={roi} eye={eye}{frame} major={major:.4f} minor={minor:.4f}".format(
                    roi=int(sample["roi_idx"]),
                    eye=int(sample["eye_idx"]),
                    frame=frame_text,
                    major=float(sample["major"]),
                    minor=float(sample["minor"]),
                )
            )
    status = "PASS" if not report.has_issues else "FAIL"
    print(f"  status: {status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--stage",
        choices=("eye_masks_runs", "refined_eye_masks_runs", "both"),
        default="both",
        help="Which legacy eye-mask stage(s) to check (default: both).",
    )
    parser.add_argument(
        "--eye-run",
        help="Specific run under eye_masks_runs to check.",
    )
    parser.add_argument(
        "--refined-run",
        help="Specific run under refined_eye_masks_runs to check.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Check all runs in each selected stage (default: latest only).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Treat major+tol < minor as noncanonical (default: 0.0).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Maximum noncanonical samples to print per run (default: 10).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of ROIs to process per chunk (default: 4096).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when any selected run has axis issues.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    root = zarr.open_group(str(args.zarr_path), mode="r", use_consolidated=False)

    selected_stages: List[StageName]
    if args.stage == "both":
        selected_stages = ["eye_masks_runs", "refined_eye_masks_runs"]
    else:
        selected_stages = [str(args.stage)]

    overall_has_issues = False
    checked_runs = 0

    for stage in selected_stages:
        parent = root.get(stage)
        if parent is None:
            print(f"\n{stage}: not found")
            overall_has_issues = True
            continue

        explicit_run = args.eye_run if stage == "eye_masks_runs" else args.refined_run
        try:
            run_names = _resolve_run_names(parent, explicit_run, bool(args.all_runs))
        except ValueError as exc:
            print(f"\n{stage}: {exc}")
            overall_has_issues = True
            continue

        if not run_names:
            print(f"\n{stage}: no runs to check")
            overall_has_issues = True
            continue

        for run_name in run_names:
            report = _analyze_run_group(
                stage,
                run_name,
                parent[run_name],
                tolerance=float(max(0.0, args.tolerance)),
                sample_limit=max(0, int(args.sample_limit)),
                chunk_size=max(1, int(args.chunk_size)),
            )
            _print_report(report)
            checked_runs += 1
            overall_has_issues = overall_has_issues or report.has_issues

    print(
        "\nSummary: checked_runs={checked} issues={issues}".format(
            checked=checked_runs,
            issues="yes" if overall_has_issues else "no",
        )
    )

    if checked_runs == 0:
        return 1
    if args.strict and overall_has_issues:
        return 1
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
