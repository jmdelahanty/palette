"""Check legacy eye-mask label coverage against keypoint-valid rows.

This diagnostic intentionally reads ``eye_masks_runs`` and
``refined_eye_masks_runs``. For the current reviewed subject-mask surface, use
``fisheye.diagnostics.check_subject_mask_keypoint_coverage``.

Rule enforced per selected eye-mask run:
every ROI with valid keypoints must have two valid eye labels
(`ellipse_success[:, 0]` and `ellipse_success[:, 1]`).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr

from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.provenance_attrs import resolve_source_keypoints_run
from fisheye.shared.row_alignment import assert_row_alignment
from fisheye.shared.type_conversions import normalize_attr as _normalize_attr


SUCCESS_DATASET_CANDIDATES = (
    "usable_keypoints",
    "detection_success",
    "refined_success",
    "source_success",
)
EYE_STAGE_CHOICES = ("auto", "eye_masks_runs", "refined_eye_masks_runs")
KEYPOINT_GROUP_CHOICES = ("keypoints_runs", "refined_keypoints_runs")


def _is_group(value: object) -> bool:
    return hasattr(value, "attrs") and callable(getattr(value, "get", None))


_utc_now = utc_now
_run_id = make_run_id
JsonLogger = SharedJsonLogger


@dataclass
class CoverageReport:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    eye_stage: Optional[str] = None
    eye_run: Optional[str] = None
    keypoint_group: Optional[str] = None
    keypoint_run: Optional[str] = None
    success_dataset: Optional[str] = None
    total_rois: int = 0
    keypoint_valid_rows: int = 0
    rows_with_two_eye_labels: int = 0
    rows_missing_two_eye_labels: int = 0
    sample_missing: List[dict[str, int]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return self.status in {"fail", "missing", "error"}


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    for key in ("zarr_use", "zarr_purpose"):
        value = _normalize_attr(root.attrs.get(key))
        if value:
            lowered = value.lower()
            if lowered in {"analysis", "training"}:
                return lowered
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _iter_zarr(paths: Sequence[Path], recursive: bool) -> Iterable[Path]:
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


def _collect_zarr_paths(paths: Sequence[Path], recursive: bool) -> List[Path]:
    seen: set[Path] = set()
    out: List[Path] = []
    for path in _iter_zarr(paths, recursive):
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _load_paths_file(path: Path) -> List[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[Path] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        out.append(Path(value))
    return out


def _group_names(parent: zarr.Group) -> List[str]:
    if hasattr(parent, "group_keys"):
        return sorted(str(name) for name in parent.group_keys())
    names: List[str] = []
    for name in parent.keys():
        try:
            obj = parent[name]
        except Exception:
            continue
        if _is_group(obj):
            names.append(str(name))
    return sorted(names)


def _latest_run_name(parent: zarr.Group) -> Optional[str]:
    latest = _normalize_attr(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    names = _group_names(parent)
    return names[-1] if names else None


def _as_array(array_like: object, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if hasattr(array_like, "shape") and hasattr(array_like, "__getitem__"):
        data = array_like[:]  # type: ignore[index]
    else:
        data = array_like
    if dtype is None:
        return np.asarray(data)
    return np.asarray(data, dtype=dtype)


def _resolve_eye_run(
    root: zarr.Group,
    *,
    stage: str,
    explicit_run: Optional[str],
) -> Tuple[str, str, zarr.Group]:
    if stage not in EYE_STAGE_CHOICES:
        raise ValueError(f"Unsupported stage '{stage}'.")

    if stage == "auto":
        stage_order = ("refined_eye_masks_runs", "eye_masks_runs")
    else:
        stage_order = (stage,)

    if explicit_run:
        for stage_name in stage_order:
            parent = root.get(stage_name)
            if _is_group(parent) and explicit_run in parent:
                return stage_name, str(explicit_run), parent[str(explicit_run)]
        raise ValueError(f"Eye-mask run '{explicit_run}' not found in selected stage(s): {stage_order}.")

    for stage_name in stage_order:
        parent = root.get(stage_name)
        if not _is_group(parent):
            continue
        run_name = _latest_run_name(parent)
        if run_name is None:
            continue
        return stage_name, run_name, parent[run_name]
    raise ValueError(f"No eye-mask runs found in selected stage(s): {stage_order}.")


def _resolve_keypoint_run(
    root: zarr.Group,
    eye_group: zarr.Group,
    *,
    explicit_group: Optional[str],
    explicit_run: Optional[str],
) -> Tuple[str, str, zarr.Group, List[str]]:
    notes: List[str] = []

    def _resolve_within_group(group_name: str, run_name: Optional[str]) -> Tuple[str, str, zarr.Group]:
        group = root.get(group_name)
        if not _is_group(group):
            raise ValueError(f"Keypoint group '{group_name}' not found.")
        if run_name:
            if run_name not in group:
                raise ValueError(f"Keypoint run '{run_name}' not found in {group_name}.")
            return group_name, str(run_name), group[str(run_name)]
        latest = _latest_run_name(group)
        if latest is None:
            raise ValueError(f"No runs found in keypoint group '{group_name}'.")
        notes.append(f"using_latest:{group_name}/{latest}")
        return group_name, latest, group[latest]

    if explicit_group:
        return (*_resolve_within_group(str(explicit_group), explicit_run), notes)

    refined = root.get("refined_keypoints_runs")
    raw = root.get("keypoints_runs")

    if explicit_run:
        in_refined = _is_group(refined) and explicit_run in refined
        in_raw = _is_group(raw) and explicit_run in raw
        if in_refined and in_raw:
            notes.append(
                f"keypoint_run '{explicit_run}' found in both groups; preferring refined_keypoints_runs"
            )
            return "refined_keypoints_runs", str(explicit_run), refined[str(explicit_run)], notes
        if in_refined:
            return "refined_keypoints_runs", str(explicit_run), refined[str(explicit_run)], notes
        if in_raw:
            return "keypoints_runs", str(explicit_run), raw[str(explicit_run)], notes
        raise ValueError(f"Keypoint run '{explicit_run}' not found in refined/raw keypoint groups.")

    source_group = _normalize_attr(eye_group.attrs.get("source_keypoint_group"))
    source_run = _normalize_attr(resolve_source_keypoints_run(eye_group.attrs))
    if source_group and source_run:
        source_parent = root.get(source_group)
        if _is_group(source_parent) and source_run in source_parent:
            notes.append(f"using_source_lineage:{source_group}/{source_run}")
            return source_group, source_run, source_parent[source_run], notes
        notes.append(f"source lineage keypoint run missing: {source_group}/{source_run}")
    elif source_run:
        matches: List[Tuple[str, zarr.Group]] = []
        for group_name in KEYPOINT_GROUP_CHOICES:
            group = root.get(group_name)
            if _is_group(group) and source_run in group:
                matches.append((group_name, group[source_run]))
        if len(matches) == 1:
            group_name, group = matches[0]
            notes.append(f"using_source_run:{group_name}/{source_run}")
            return group_name, source_run, group, notes
        if len(matches) > 1:
            notes.append(
                f"source keypoint run '{source_run}' found in both groups; preferring refined_keypoints_runs"
            )
            refined_group = root["refined_keypoints_runs"]
            return "refined_keypoints_runs", source_run, refined_group[source_run], notes
        notes.append(f"source keypoint run '{source_run}' not found in keypoint groups")

    for group_name in ("refined_keypoints_runs", "keypoints_runs"):
        group = root.get(group_name)
        if not _is_group(group):
            continue
        latest = _latest_run_name(group)
        if latest:
            notes.append(f"fallback_latest:{group_name}/{latest}")
            return group_name, latest, group[latest], notes
    raise ValueError("No keypoint runs found in refined/raw keypoint groups.")


def _resolve_success_dataset_name(kp_group: zarr.Group) -> Optional[str]:
    for name in SUCCESS_DATASET_CANDIDATES:
        if name in kp_group:
            return name
    return None


def _compute_keypoint_valid_mask(
    keypoints_roi: np.ndarray,
    success_flags: np.ndarray,
) -> np.ndarray:
    if keypoints_roi.ndim != 3 or keypoints_roi.shape[1] < 3 or keypoints_roi.shape[2] < 2:
        raise ValueError(
            "keypoints_roi must have shape (N, >=3, >=2) with swim-bladder, left-eye, right-eye coordinates."
        )
    if success_flags.ndim != 1:
        raise ValueError("Keypoint success array must have shape (N,).")
    if int(keypoints_roi.shape[0]) != int(success_flags.shape[0]):
        raise ValueError(
            f"Row mismatch between keypoints_roi ({keypoints_roi.shape[0]}) and success flags ({success_flags.shape[0]})."
        )

    left = np.asarray(keypoints_roi[:, 1, :2], dtype=np.float32)
    right = np.asarray(keypoints_roi[:, 2, :2], dtype=np.float32)
    finite = np.all(np.isfinite(left), axis=1) & np.all(np.isfinite(right), axis=1)
    distinct = np.any(np.abs(left - right) > 1e-6, axis=1)
    return np.asarray(success_flags, dtype=bool) & finite & distinct


def _resolve_frame_indices(root: zarr.Group, eye_group: zarr.Group, total_rois: int) -> Optional[np.ndarray]:
    direct = eye_group.get("frame_indices")
    if direct is not None:
        candidate = _as_array(direct, dtype=np.int64)
        if candidate.ndim == 1 and int(candidate.shape[0]) == int(total_rois):
            return candidate

    source_crop_run = _normalize_attr(eye_group.attrs.get("source_crop_run"))
    if not source_crop_run:
        return None
    crop_parent = root.get("crop_runs")
    if not _is_group(crop_parent) or source_crop_run not in crop_parent:
        return None
    crop_group = crop_parent[source_crop_run]
    crop_frame_indices = crop_group.get("frame_indices")
    if crop_frame_indices is None:
        return None
    candidate = _as_array(crop_frame_indices, dtype=np.int64)
    if candidate.ndim != 1 or int(candidate.shape[0]) != int(total_rois):
        return None
    return candidate


def _collect_failure_samples(
    fail_indices: np.ndarray,
    frame_indices: Optional[np.ndarray],
    limit: int,
) -> List[dict[str, int]]:
    samples: List[dict[str, int]] = []
    for roi_idx in fail_indices[: max(0, int(limit))]:
        payload = {"roi_idx": int(roi_idx)}
        if frame_indices is not None:
            payload["frame_idx"] = int(frame_indices[int(roi_idx)])
        samples.append(payload)
    return samples


def _analyze_root(
    *,
    root: zarr.Group,
    zarr_path: Path,
    stage: str,
    eye_run: Optional[str],
    keypoint_group: Optional[str],
    keypoint_run: Optional[str],
    sample_limit: int,
) -> CoverageReport:
    try:
        resolved_stage, resolved_eye_run, eye_group = _resolve_eye_run(
            root,
            stage=stage,
            explicit_run=eye_run,
        )
    except ValueError as exc:
        return CoverageReport(
            zarr_path=zarr_path,
            status="missing",
            reason=str(exc),
        )

    report = CoverageReport(
        zarr_path=zarr_path,
        status="pass",
        eye_stage=resolved_stage,
        eye_run=resolved_eye_run,
    )

    ellipse_success_arr = eye_group.get("ellipse_success")
    if ellipse_success_arr is None:
        report.status = "missing"
        report.reason = f"{resolved_stage}/{resolved_eye_run} missing ellipse_success."
        return report
    ellipse_success = _as_array(ellipse_success_arr, dtype=bool)
    if ellipse_success.ndim != 2 or ellipse_success.shape[1] < 2:
        report.status = "missing"
        report.reason = (
            f"{resolved_stage}/{resolved_eye_run}/ellipse_success must have shape (N, >=2), "
            f"got {tuple(ellipse_success.shape)}."
        )
        return report
    report.total_rois = int(ellipse_success.shape[0])

    try:
        kp_group_name, kp_run_name, kp_group, kp_notes = _resolve_keypoint_run(
            root,
            eye_group,
            explicit_group=keypoint_group,
            explicit_run=keypoint_run,
        )
    except ValueError as exc:
        report.status = "missing"
        report.reason = str(exc)
        return report
    report.keypoint_group = kp_group_name
    report.keypoint_run = kp_run_name
    report.notes.extend(kp_notes)

    keypoints_arr = kp_group.get("keypoints_roi")
    if keypoints_arr is None:
        report.status = "missing"
        report.reason = f"{kp_group_name}/{kp_run_name} missing keypoints_roi."
        return report

    success_name = _resolve_success_dataset_name(kp_group)
    if success_name is None:
        report.status = "missing"
        report.reason = (
            f"{kp_group_name}/{kp_run_name} missing success dataset "
            f"({', '.join(SUCCESS_DATASET_CANDIDATES)})."
        )
        return report
    success_arr = kp_group.get(success_name)
    if success_arr is None:
        report.status = "missing"
        report.reason = f"{kp_group_name}/{kp_run_name} missing {success_name}."
        return report
    report.success_dataset = success_name

    keypoints_roi = _as_array(keypoints_arr, dtype=np.float32)
    success_flags = _as_array(success_arr, dtype=bool)
    try:
        assert_row_alignment(
            report.total_rois,
            (
                (f"{kp_group_name}/{kp_run_name}/keypoints_roi", keypoints_roi),
                (f"{kp_group_name}/{kp_run_name}/{success_name}", success_flags),
            ),
            stage=f"{resolved_stage}/{resolved_eye_run} coverage inputs",
        )
        keypoint_valid = _compute_keypoint_valid_mask(keypoints_roi, success_flags)
    except ValueError as exc:
        report.status = "missing"
        report.reason = str(exc)
        return report

    pair_success = np.all(ellipse_success[:, :2], axis=1)
    fail_rows = np.flatnonzero(keypoint_valid & ~pair_success)
    frame_indices = _resolve_frame_indices(root, eye_group, report.total_rois)

    report.keypoint_valid_rows = int(keypoint_valid.sum())
    report.rows_with_two_eye_labels = int((keypoint_valid & pair_success).sum())
    report.rows_missing_two_eye_labels = int(fail_rows.size)
    report.sample_missing = _collect_failure_samples(fail_rows, frame_indices, sample_limit)

    if fail_rows.size > 0:
        report.status = "fail"
        report.reason = "keypoint-valid rows missing one/both eye labels."
    return report


def _report_to_log_payload(report: CoverageReport) -> dict[str, object]:
    return {
        "zarr": str(report.zarr_path),
        "status": report.status,
        "reason": report.reason,
        "eye_stage": report.eye_stage,
        "eye_run": report.eye_run,
        "keypoint_group": report.keypoint_group,
        "keypoint_run": report.keypoint_run,
        "success_dataset": report.success_dataset,
        "total_rois": report.total_rois,
        "keypoint_valid_rows": report.keypoint_valid_rows,
        "rows_with_two_eye_labels": report.rows_with_two_eye_labels,
        "rows_missing_two_eye_labels": report.rows_missing_two_eye_labels,
        "sample_missing": report.sample_missing,
        "notes": report.notes,
    }


def _print_report(report: CoverageReport, *, show_pass: bool) -> None:
    if report.status == "pass" and not show_pass:
        return

    print(f"\n=== {report.zarr_path} ===")
    print(f"status: {report.status.upper()}")
    if report.reason:
        print(f"reason: {report.reason}")
    if report.eye_stage and report.eye_run:
        print(f"eye_run: {report.eye_stage}/{report.eye_run}")
    if report.keypoint_group and report.keypoint_run:
        print(f"keypoint_run: {report.keypoint_group}/{report.keypoint_run}")
    if report.success_dataset:
        print(f"success_dataset: {report.success_dataset}")
    if report.total_rois:
        print(f"total_rois: {report.total_rois}")
        print(f"keypoint_valid_rows: {report.keypoint_valid_rows}")
        print(f"rows_with_two_eye_labels: {report.rows_with_two_eye_labels}")
        print(f"rows_missing_two_eye_labels: {report.rows_missing_two_eye_labels}")
    for note in report.notes:
        print(f"note: {note}")
    for sample in report.sample_missing:
        if "frame_idx" in sample:
            print(f"sample_missing: roi={sample['roi_idx']} frame={sample['frame_idx']}")
        else:
            print(f"sample_missing: roi={sample['roi_idx']}")


def _append_review_list(path: Path, zarr_paths: Sequence[Path]) -> int:
    raw_lines: List[str] = []
    existing: set[str] = set()
    if path.exists():
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        for line in raw_lines:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            existing.add(value)

    added = 0
    for zarr_path in zarr_paths:
        value = str(zarr_path)
        if value in existing:
            continue
        existing.add(value)
        raw_lines.append(value)
        added += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    output = "\n".join(raw_lines)
    if output and not output.endswith("\n"):
        output += "\n"
    path.write_text(output, encoding="utf-8")
    return added


def _resolve_default_roots(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: Sequence[Path]) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "check_eye_mask_keypoint_coverage"
    if roots:
        return roots[0] / "logs" / "check_eye_mask_keypoint_coverage"
    return Path.cwd() / "logs" / "check_eye_mask_keypoint_coverage"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or zarr paths to scan (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["all", "analysis", "training"],
        default="all",
        help="Restrict zarr scope by use (default: all).",
    )
    parser.add_argument(
        "--stage",
        choices=EYE_STAGE_CHOICES,
        default="auto",
        help=(
            "Legacy eye-mask stage to check (default: auto prefers "
            "refined_eye_masks_runs). Use check_subject_mask_keypoint_coverage "
            "for current refined subject masks."
        ),
    )
    parser.add_argument("--eye-run", help="Specific eye-mask run name to check.")
    parser.add_argument(
        "--keypoint-group",
        choices=KEYPOINT_GROUP_CHOICES,
        help="Explicit keypoint group override.",
    )
    parser.add_argument("--keypoint-run", help="Explicit keypoint run override.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Maximum sample failing rows to print per zarr (default: 10).",
    )
    parser.add_argument(
        "--append-review-list",
        type=Path,
        help="Append failing zarr paths to this file for review_eye_masks_batch --file-list.",
    )
    parser.add_argument("--show-pass", action="store_true", help="Print PASS records too.")
    parser.add_argument("--strict", action="store_true", help="Exit with status 1 on any fail/missing/error.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        help=(
            "Directory for JSONL logs (default: $PALETTE_LOG_ROOT/check_eye_mask_keypoint_coverage "
            "or <recordings_root>/logs/check_eye_mask_keypoint_coverage)."
        ),
    )
    parser.add_argument("--no-log", action="store_true", help="Disable JSONL logging.")
    return parser


def run(args: argparse.Namespace) -> int:
    file_list_paths: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(_load_paths_file(path))

    explicit_paths: List[Path] = []
    if args.paths:
        explicit_paths.extend(args.paths)
    if file_list_paths:
        explicit_paths.extend(file_list_paths)

    roots = _resolve_default_roots(explicit_paths if explicit_paths else None)
    zarr_paths = _collect_zarr_paths(roots, recursive=bool(args.recursive))
    if not zarr_paths:
        print("No zarr files found.")
        return 1

    logger: Optional[JsonLogger] = None
    run_id = _run_id()
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, roots)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"check_eye_mask_keypoint_coverage_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            roots=[str(path) for path in roots],
            file_list=[str(path) for path in (args.file_list or [])],
            recursive=bool(args.recursive),
            zarr_use=str(args.zarr_use),
            stage=str(args.stage),
            eye_run=args.eye_run,
            keypoint_group=args.keypoint_group,
            keypoint_run=args.keypoint_run,
            sample_limit=int(max(0, args.sample_limit)),
            append_review_list=str(args.append_review_list) if args.append_review_list else None,
        )

    zarr_scanned = 0
    zarr_checked = 0
    zarr_pass = 0
    zarr_fail = 0
    zarr_missing = 0
    zarr_error = 0
    filtered_zarr_use = 0
    failing_paths: List[Path] = []

    for zarr_path in zarr_paths:
        zarr_scanned += 1
        try:
            root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        except Exception as exc:
            zarr_error += 1
            report = CoverageReport(zarr_path=zarr_path, status="error", reason=str(exc))
            _print_report(report, show_pass=bool(args.show_pass))
            if logger is not None:
                logger.log("zarr_error", **_report_to_log_payload(report))
            continue

        observed_use = _infer_zarr_use(root, zarr_path)
        if args.zarr_use != "all" and observed_use != args.zarr_use:
            filtered_zarr_use += 1
            if logger is not None:
                logger.log(
                    "zarr_filtered",
                    zarr=str(zarr_path),
                    reason=f"zarr_use={observed_use or 'unknown'}",
                )
            continue

        report = _analyze_root(
            root=root,
            zarr_path=zarr_path,
            stage=str(args.stage),
            eye_run=args.eye_run,
            keypoint_group=args.keypoint_group,
            keypoint_run=args.keypoint_run,
            sample_limit=max(0, int(args.sample_limit)),
        )
        _print_report(report, show_pass=bool(args.show_pass))

        zarr_checked += 1
        if report.status == "pass":
            zarr_pass += 1
        elif report.status == "fail":
            zarr_fail += 1
            failing_paths.append(zarr_path)
        elif report.status == "missing":
            zarr_missing += 1
        else:
            zarr_error += 1
        if logger is not None:
            logger.log("zarr_checked", **_report_to_log_payload(report))

    review_list_added = 0
    if args.append_review_list:
        review_list_path = args.append_review_list.expanduser().resolve()
        review_list_added = _append_review_list(review_list_path, failing_paths)
        print(
            f"\nReview list updated: {review_list_path} "
            f"(added={review_list_added}, failing={len(failing_paths)})"
        )
        if logger is not None:
            logger.log(
                "review_list_updated",
                path=str(review_list_path),
                added=review_list_added,
                failing=len(failing_paths),
            )

    issues = (zarr_fail > 0) or (zarr_missing > 0) or (zarr_error > 0)
    print(
        "\nSummary: "
        f"zarr_scanned={zarr_scanned} "
        f"zarr_checked={zarr_checked} "
        f"zarr_pass={zarr_pass} "
        f"zarr_fail={zarr_fail} "
        f"zarr_missing={zarr_missing} "
        f"filtered_zarr_use={filtered_zarr_use} "
        f"errors={zarr_error} "
        f"review_list_added={review_list_added} "
        f"issues={'yes' if issues else 'no'}"
    )

    if logger is not None:
        logger.log(
            "run_end",
            zarr_scanned=zarr_scanned,
            zarr_checked=zarr_checked,
            zarr_pass=zarr_pass,
            zarr_fail=zarr_fail,
            zarr_missing=zarr_missing,
            filtered_zarr_use=filtered_zarr_use,
            errors=zarr_error,
            review_list_added=review_list_added,
            issues=issues,
        )
        logger.close()

    if zarr_checked == 0:
        return 1
    if args.strict and issues:
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
