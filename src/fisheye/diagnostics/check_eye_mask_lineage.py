"""Validate eye-mask lineage arrays against source crop runs.

This checks that eye-mask run lineage arrays:

- `frame_indices`
- `detection_indices`
- `frame_counts`
- `source_refined_row_ids` when present on the crop source
- `source_detect_row_index` when present on the crop source

match exactly against `crop_runs/<source_crop_run>`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import zarr

from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.row_lineage import PER_ROI_ROW_LINEAGE_ARRAYS, ROW_LINEAGE_ARRAYS
from fisheye.shared.type_conversions import normalize_attr


LINEAGE_ARRAYS = ROW_LINEAGE_ARRAYS
REQUIRED_LINEAGE_ARRAYS = ("frame_indices", "detection_indices", "frame_counts")


_utc_now = utc_now
_run_id = make_run_id
JsonLogger = SharedJsonLogger


@dataclass
class RunLineageReport:
    stage: str
    run_name: str
    source_crop_run: Optional[str]
    total_rois: Optional[int]
    missing_source_crop_run: bool = False
    missing_crop_group: bool = False
    missing_run_arrays: List[str] = field(default_factory=list)
    missing_crop_arrays: List[str] = field(default_factory=list)
    row_length_issues: List[str] = field(default_factory=list)
    frame_count_sum_issue: Optional[str] = None
    mismatched_arrays: List[str] = field(default_factory=list)
    mismatch_samples: List[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.missing_source_crop_run
            or self.missing_crop_group
            or self.missing_run_arrays
            or self.missing_crop_arrays
            or self.row_length_issues
            or self.frame_count_sum_issue
            or self.mismatched_arrays
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
        return _group_names(parent)
    latest = parent.attrs.get("latest")
    if isinstance(latest, str) and latest in parent:
        return [latest]
    names = _group_names(parent)
    return names[-1:] if names else []


def _first_mismatch_sample(name: str, lhs: np.ndarray, rhs: np.ndarray) -> str:
    if lhs.shape != rhs.shape:
        return f"{name}: shape {lhs.shape} != {rhs.shape}"
    unequal = np.flatnonzero(lhs != rhs)
    if unequal.size == 0:
        return f"{name}: values differ (no index found)"
    idx = int(unequal[0])
    return f"{name}: idx {idx} run={lhs[idx]!r} crop={rhs[idx]!r}"


def _analyze_run_group(
    *,
    root: zarr.Group,
    stage: str,
    run_name: str,
    run_group: zarr.Group,
) -> RunLineageReport:
    total_rois: Optional[int] = None
    masks_roi = run_group.get("masks_roi")
    if masks_roi is not None:
        total_rois = int(masks_roi.shape[0])
    elif run_group.get("frame_indices") is not None:
        total_rois = int(run_group["frame_indices"].shape[0])

    source_crop_run = run_group.attrs.get("source_crop_run")
    if source_crop_run is not None:
        source_crop_run = str(source_crop_run)

    report = RunLineageReport(
        stage=stage,
        run_name=run_name,
        source_crop_run=source_crop_run,
        total_rois=total_rois,
    )

    if not source_crop_run:
        report.missing_source_crop_run = True
        return report

    crop_parent = root.get("crop_runs")
    if crop_parent is None or source_crop_run not in crop_parent:
        report.missing_crop_group = True
        return report
    crop_group = crop_parent[source_crop_run]

    for array_name in LINEAGE_ARRAYS:
        run_arr = run_group.get(array_name)
        crop_arr = crop_group.get(array_name)
        should_require = array_name in REQUIRED_LINEAGE_ARRAYS or crop_arr is not None or run_arr is not None
        if run_arr is None and should_require:
            report.missing_run_arrays.append(array_name)
        if crop_arr is None and should_require:
            report.missing_crop_arrays.append(array_name)

        run_data = np.asarray(run_arr[:]) if run_arr is not None else None
        crop_data = np.asarray(crop_arr[:]) if crop_arr is not None else None

        if run_data is not None and total_rois is not None and array_name in PER_ROI_ROW_LINEAGE_ARRAYS:
            if int(run_data.shape[0]) != int(total_rois):
                report.row_length_issues.append(
                    f"{array_name}: len={run_data.shape[0]} expected={total_rois}"
                )

        if run_data is not None and total_rois is not None and array_name == "frame_counts":
            sum_counts = int(np.sum(run_data, dtype=np.int64))
            if sum_counts != int(total_rois):
                report.frame_count_sum_issue = f"frame_counts sum={sum_counts} expected={total_rois}"

        if run_data is not None and crop_data is not None:
            if run_data.shape != crop_data.shape or not np.array_equal(run_data, crop_data):
                report.mismatched_arrays.append(array_name)
                report.mismatch_samples.append(_first_mismatch_sample(array_name, run_data, crop_data))

    return report


def _print_report(report: RunLineageReport) -> None:
    print(f"\n{report.stage}/{report.run_name}")
    if report.source_crop_run:
        print(f"  source_crop_run: {report.source_crop_run}")
    else:
        print("  source_crop_run: <missing>")
    if report.total_rois is not None:
        print(f"  total_rois: {report.total_rois}")

    if report.missing_source_crop_run:
        print("  status: FAIL (missing source_crop_run)")
        return
    if report.missing_crop_group:
        print("  status: FAIL (source crop run not found)")
        return

    if report.missing_run_arrays:
        print(f"  missing_run_arrays: {', '.join(report.missing_run_arrays)}")
    if report.missing_crop_arrays:
        print(f"  missing_crop_arrays: {', '.join(report.missing_crop_arrays)}")
    if report.row_length_issues:
        print(f"  row_length_issues: {'; '.join(report.row_length_issues)}")
    if report.frame_count_sum_issue:
        print(f"  frame_count_sum_issue: {report.frame_count_sum_issue}")
    if report.mismatched_arrays:
        print(f"  mismatched_arrays: {', '.join(report.mismatched_arrays)}")
    for sample in report.mismatch_samples:
        print(f"  mismatch_sample: {sample}")

    print(f"  status: {'PASS' if not report.has_issues else 'FAIL'}")


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    for key in ("zarr_use", "zarr_purpose"):
        value = normalize_attr(root.attrs.get(key))
        if value is not None:
            value = value.lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _iter_zarr(paths: List[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.suffix == ".zarr":
            yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("*.zarr")
        else:
            yield from path.glob("*/zarr/*.zarr")
            yield from path.glob("*.zarr")


def _collect_zarr_paths(paths: List[Path], recursive: bool) -> List[Path]:
    seen: set[Path] = set()
    out: List[Path] = []
    for path in _iter_zarr(paths, recursive):
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: List[Path]) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "check_eye_mask_lineage"
    if roots:
        return roots[0] / "logs" / "check_eye_mask_lineage"
    return Path.cwd() / "logs" / "check_eye_mask_lineage"


def _stage_names(stage: str) -> tuple[str, ...]:
    return ("eye_masks_runs", "refined_eye_masks_runs") if stage == "both" else (str(stage),)


def _check_root(
    *,
    root: zarr.Group,
    stage: str,
    explicit_run: Optional[str],
    all_runs: bool,
    strict: bool,
) -> List[RunLineageReport]:
    reports: List[RunLineageReport] = []
    for stage_name in _stage_names(stage):
        parent = root.get(stage_name)
        if parent is None:
            print(f"\n{stage_name}\n  status: missing stage")
            if strict:
                reports.append(
                    RunLineageReport(
                        stage=stage_name,
                        run_name="<missing-stage>",
                        source_crop_run=None,
                        total_rois=None,
                        missing_crop_group=True,
                    )
                )
            continue
        run_names = _resolve_run_names(
            parent,
            explicit_run=explicit_run if stage_name == stage else None,
            all_runs=bool(all_runs),
        )
        if not run_names:
            print(f"\n{stage_name}\n  status: no runs")
            continue
        for run_name in run_names:
            run_group = parent[run_name]
            report = _analyze_run_group(
                root=root,
                stage=stage_name,
                run_name=run_name,
                run_group=run_group,
            )
            reports.append(report)
            _print_report(report)
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Zarr path(s) and/or recording roots (default: /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan roots for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=("analysis", "training", "any"),
        default="any",
        help="Filter zarr archives by use (default: any).",
    )
    parser.add_argument(
        "--stage",
        choices=("eye_masks_runs", "refined_eye_masks_runs", "both"),
        default="both",
        help="Stage(s) to inspect (default: both).",
    )
    parser.add_argument(
        "--run",
        help="Specific run name to inspect (only valid when --stage is not 'both').",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Inspect all runs in each selected stage (default: latest only).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when any issue is found.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for JSONL logs (default: $PALETTE_LOG_ROOT/check_eye_mask_lineage or <root>/logs/check_eye_mask_lineage).",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable JSONL logging.")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.run and args.stage == "both":
        raise ValueError("--run requires a concrete --stage (not 'both').")

    input_paths = list(args.paths) if args.paths else [Path("/nvme1/recordings")]
    zarr_paths = _collect_zarr_paths(input_paths, recursive=bool(args.recursive))
    if not zarr_paths:
        print("No zarr files found.")
        return 1

    non_zarr_roots = [path.expanduser() for path in input_paths if path.suffix != ".zarr"]
    logger: Optional[JsonLogger] = None
    run_id = _run_id()
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, non_zarr_roots)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"check_eye_mask_lineage_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            paths=[str(path) for path in input_paths],
            recursive=bool(args.recursive),
            zarr_use=args.zarr_use,
            stage=args.stage,
            run=args.run,
            all_runs=bool(args.all_runs),
            strict=bool(args.strict),
        )

    counts = {
        "zarr_scanned": 0,
        "zarr_checked": 0,
        "zarr_pass": 0,
        "zarr_fail": 0,
        "filtered_zarr_use": 0,
        "checked_runs": 0,
        "errors": 0,
    }
    any_issues = False

    for zarr_path in zarr_paths:
        counts["zarr_scanned"] += 1
        try:
            root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        except Exception as exc:
            counts["errors"] += 1
            print(f"\n{zarr_path}\n  status: error ({exc})")
            if logger is not None:
                logger.log("zarr_error", zarr=str(zarr_path), reason=str(exc))
            continue

        observed_use = _infer_zarr_use(root, zarr_path)
        if args.zarr_use != "any" and observed_use != args.zarr_use:
            counts["filtered_zarr_use"] += 1
            if logger is not None:
                logger.log(
                    "zarr_filtered",
                    zarr=str(zarr_path),
                    observed_zarr_use=observed_use,
                    requested_zarr_use=args.zarr_use,
                )
            continue

        print(f"\n=== {zarr_path} ===")
        try:
            reports = _check_root(
                root=root,
                stage=str(args.stage),
                explicit_run=args.run,
                all_runs=bool(args.all_runs),
                strict=bool(args.strict),
            )
        except Exception as exc:
            counts["errors"] += 1
            print(f"  status: error ({exc})")
            if logger is not None:
                logger.log("zarr_error", zarr=str(zarr_path), reason=str(exc))
            continue

        zarr_has_issues = any(report.has_issues for report in reports)
        any_issues = any_issues or zarr_has_issues
        counts["checked_runs"] += len(reports)
        counts["zarr_checked"] += 1
        if zarr_has_issues:
            counts["zarr_fail"] += 1
        else:
            counts["zarr_pass"] += 1

        print(f"\nZarr summary: checked_runs={len(reports)} issues={'yes' if zarr_has_issues else 'no'}")
        if logger is not None:
            logger.log(
                "zarr_checked",
                zarr=str(zarr_path),
                observed_zarr_use=observed_use,
                checked_runs=len(reports),
                issues=bool(zarr_has_issues),
            )

    print(
        "\nSummary: "
        f"zarr_scanned={counts['zarr_scanned']} "
        f"zarr_checked={counts['zarr_checked']} "
        f"zarr_pass={counts['zarr_pass']} "
        f"zarr_fail={counts['zarr_fail']} "
        f"filtered_zarr_use={counts['filtered_zarr_use']} "
        f"checked_runs={counts['checked_runs']} "
        f"errors={counts['errors']} "
        f"issues={'yes' if any_issues else 'no'}"
    )

    if logger is not None:
        logger.log(
            "run_end",
            **counts,
            issues=bool(any_issues),
            strict=bool(args.strict),
        )
        logger.close()

    if counts["errors"] > 0:
        return 1
    if args.strict and any_issues:
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
