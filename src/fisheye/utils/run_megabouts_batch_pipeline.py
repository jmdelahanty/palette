"""Batch-run subject-shape, Megabouts posture views, and bout classification.

The utility is conservative by design:

* default mode is dry-run;
* archives are processed serially by default, with opt-in archive-level
  parallelism;
* subject-shape is treated as a required prerequisite for Megabouts-compatible
  tail posture;
* each child stage remains responsible for detailed method provenance.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import zarr

from fisheye.shared.json_safety import write_json_atomic


DEFAULT_SUBJECT_SHAPE_RUN = "subject_shape_v3_snout_medialjoin_batch_20260505"
DEFAULT_TAIL_POSTURE_RUN = "tail_posture_megabouts_k11_batch_20260505"
DEFAULT_TRACK_RUN = "tk_hyst4_low2_latch_s005"
DEFAULT_SWIM_BOUT_RUN = "bouts_tk_hyst4_low2_latch_s005_peak_event_exp_tau025_prom4_dist010_w098"
DEFAULT_CLASSIFIER_RUN = "megabouts_cls_tk_hyst4_low2_latch_s005_peak_event_prom4_w098_mbprep_batch_20260505"
DEFAULT_CLASSIFIER_INPUT_MODE = "megabouts_preprocessed_full_timeseries"


@dataclass(frozen=True)
class ArchivePlan:
    zarr_path: str
    refined_subject_run: Optional[str]
    subject_shape_run: str
    tail_posture_run: str
    track_run: str
    swim_bout_run: str
    classifier_run: str
    classifier_input_mode: str
    run_subject_shape: bool
    run_tail_posture: bool
    run_classifier: bool
    skip_reason: str = ""


@dataclass
class ArchiveResult:
    zarr_path: str
    subject_shape_run: str
    tail_posture_run: str
    classifier_run: str
    classifier_input_mode: str
    subject_shape_status: str = "not_requested"
    tail_posture_status: str = "not_requested"
    classifier_status: str = "not_requested"
    validation_status: str = "not_run"
    log_path: str = ""
    error: str = ""


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _attrs(group_path: Path) -> dict[str, Any]:
    payload = _read_json(group_path / "zarr.json")
    raw = payload.get("attributes") if isinstance(payload, dict) else None
    return dict(raw) if isinstance(raw, dict) else {}


def _child_groups(group_path: Path) -> list[str]:
    if not group_path.is_dir():
        return []
    return sorted(path.name for path in group_path.iterdir() if path.is_dir())


def _latest_group_name(parent_path: Path) -> Optional[str]:
    attrs = _attrs(parent_path)
    for key in ("latest", "latest_materialized"):
        value = attrs.get(key)
        if isinstance(value, str) and value and (parent_path / value).is_dir():
            return value
    children = _child_groups(parent_path)
    return children[-1] if children else None


def _zarr_paths_from_report(report_path: Path) -> list[Path]:
    payload = _read_json(report_path)
    if payload is None:
        raise ValueError(f"Could not read JSON report: {report_path}")

    rows = payload.get("results")
    if not isinstance(rows, list):
        rows = payload.get("plans")
    if not isinstance(rows, list):
        raise ValueError(f"Report {report_path} does not contain a results or plans list.")

    seen: set[str] = set()
    zarr_paths: list[Path] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_path = row.get("zarr_path")
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        zarr_paths.append(path)
    if not zarr_paths:
        raise ValueError(f"Report {report_path} did not contain any zarr_path entries.")
    return zarr_paths


def _discover_analysis_zarrs(roots: Sequence[Path], *, include_smoke: bool) -> list[Path]:
    seen: set[str] = set()
    zarrs: list[Path] = []
    for root in roots:
        root = root.expanduser()
        candidates = [root] if root.name.endswith("_analysis.zarr") else sorted(root.rglob("*_analysis.zarr"))
        for candidate in candidates:
            if not include_smoke and "/smoke/" in str(candidate):
                continue
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            zarrs.append(candidate)
    return sorted(zarrs)


def _exists(zarr_path: Path, rel_path: str) -> bool:
    path = zarr_path / rel_path
    return (path / "zarr.json").exists() or path.exists()


def build_archive_plan(
    zarr_path: Path,
    *,
    subject_shape_run: str,
    tail_posture_run: str,
    track_run: str,
    swim_bout_run: str,
    classifier_run: str,
    classifier_input_mode: str,
    skip_existing: bool,
    force_classifier: bool,
) -> ArchivePlan:
    refined_subject_run = _latest_group_name(zarr_path / "refined_subject_masks_runs")
    subject_shape_exists = _exists(zarr_path, f"analysis/subject_shape_runs/{subject_shape_run}")
    tail_posture_exists = _exists(zarr_path, f"analysis/tail_posture_view_runs/{tail_posture_run}")
    classifier_exists = _exists(zarr_path, f"analysis/bout_classification_runs/{classifier_run}")
    track_exists = _exists(zarr_path, f"analysis/track_kinematics_runs/offline/{track_run}")
    swim_bout_exists = _exists(zarr_path, f"analysis/swim_bout_runs/{swim_bout_run}")

    skip_reasons: list[str] = []
    if refined_subject_run is None and not subject_shape_exists:
        skip_reasons.append("missing_refined_subject_masks_for_subject_shape")
    if not track_exists:
        skip_reasons.append("missing_track_kinematics_run")
    if not swim_bout_exists:
        skip_reasons.append("missing_swim_bout_run")

    can_subject_shape = refined_subject_run is not None or subject_shape_exists
    run_subject_shape = bool(can_subject_shape and not (skip_existing and subject_shape_exists))

    subject_shape_available = bool(subject_shape_exists or run_subject_shape)
    can_tail_posture = subject_shape_available
    run_tail_posture = bool(can_tail_posture and not (skip_existing and tail_posture_exists))

    tail_posture_available = bool(tail_posture_exists or run_tail_posture)
    can_classifier = bool(tail_posture_available and track_exists and swim_bout_exists)
    run_classifier = bool(can_classifier and (force_classifier or not (skip_existing and classifier_exists)))

    if not subject_shape_available:
        skip_reasons.append("subject_shape_unavailable")
    if not tail_posture_available:
        skip_reasons.append("tail_posture_unavailable")

    return ArchivePlan(
        zarr_path=str(zarr_path),
        refined_subject_run=refined_subject_run,
        subject_shape_run=subject_shape_run,
        tail_posture_run=tail_posture_run,
        track_run=track_run,
        swim_bout_run=swim_bout_run,
        classifier_run=classifier_run,
        classifier_input_mode=classifier_input_mode,
        run_subject_shape=run_subject_shape,
        run_tail_posture=run_tail_posture,
        run_classifier=run_classifier,
        skip_reason=", ".join(dict.fromkeys(skip_reasons)),
    )


def _write_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _run_command(cmd: Sequence[str], *, dry_run: bool, log_path: Optional[Path] = None) -> str:
    command_line = "+ " + " ".join(str(part) for part in cmd)
    if log_path is None:
        print(command_line, flush=True)
    else:
        _write_log(log_path, command_line)
    if dry_run:
        return "planned"
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    if log_path is None:
        completed = subprocess.run(list(cmd), check=False, env=env)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(list(cmd), check=False, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return "ok" if completed.returncode == 0 else f"failed_exit_{completed.returncode}"


def _subject_shape_command(plan: ArchivePlan, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.analysis.subject_shape_runs",
        plan.zarr_path,
        "--refined-run",
        str(plan.refined_subject_run),
        "--run-name",
        plan.subject_shape_run,
        "--chunk-size",
        str(args.subject_shape_chunk_size),
        "--execution-backend",
        args.subject_shape_execution_backend,
        "--scheduler",
        args.subject_shape_scheduler,
    ]
    if args.subject_shape_num_workers is not None:
        cmd.extend(["--num-workers", str(args.subject_shape_num_workers)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _tail_posture_command(plan: ArchivePlan, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.analysis.tail_posture_view_runs",
        plan.zarr_path,
        "--subject-shape-run",
        plan.subject_shape_run,
        "--run-name",
        plan.tail_posture_run,
        "--view-family",
        args.view_family,
        "--head-source",
        args.head_source,
        "--keypoint-count",
        str(args.keypoint_count),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _classifier_command(plan: ArchivePlan, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.analysis.megabouts_classifier",
        plan.zarr_path,
        "--run-name",
        plan.classifier_run,
        "--tail-posture-view-run",
        plan.tail_posture_run,
        "--track-kinematics-run",
        plan.track_run,
        "--track-scope",
        "offline",
        "--track-id",
        str(args.track_id),
        "--swim-bout-run",
        plan.swim_bout_run,
        "--speed-level",
        args.speed_level,
        "--heading-source",
        args.heading_source,
        "--bout-duration-s",
        str(args.bout_duration_s),
        "--min-tail-valid-fraction",
        str(args.min_tail_valid_fraction),
        "--min-traj-valid-fraction",
        str(args.min_traj_valid_fraction),
        "--max-consecutive-invalid-frames",
        str(args.max_consecutive_invalid_frames),
        "--traj-reference-index",
        str(args.traj_reference_index),
        "--device",
        args.device,
        "--classifier-input-mode",
        plan.classifier_input_mode,
    ]
    if args.bout_duration_frames is not None:
        cmd.extend(["--bout-duration-frames", str(args.bout_duration_frames)])
    if not args.align_traj_to_onset:
        cmd.append("--no-align-traj-to-onset")
    if args.exclude_cs:
        cmd.append("--exclude-CS")
    if args.megabouts_repo:
        cmd.extend(["--megabouts-repo", str(args.megabouts_repo)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _validate_plan_outputs(plan: ArchivePlan) -> tuple[str, str]:
    zarr_path = Path(plan.zarr_path)
    checks = [
        f"analysis/subject_shape_runs/{plan.subject_shape_run}/components/subject_body/tail_sample_xy",
        f"analysis/tail_posture_view_runs/{plan.tail_posture_run}/tail_angle_rad",
        f"analysis/tail_posture_view_runs/{plan.tail_posture_run}/valid",
        f"analysis/bout_classification_runs/{plan.classifier_run}/per_bout",
    ]
    missing = [rel for rel in checks if not _exists(zarr_path, rel)]
    if missing:
        return "failed", "missing " + ", ".join(missing)
    return "ok", ""


def _safe_log_stem(zarr_path: str) -> str:
    raw = Path(zarr_path).name
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in raw)


def _run_archive_plan(
    plan: ArchivePlan,
    *,
    args: argparse.Namespace,
    dry_run: bool,
    index: int,
    total: int,
    log_path: Optional[Path] = None,
) -> ArchiveResult:
    if log_path is None:
        print(f"\n[{index}/{total}] {plan.zarr_path}", flush=True)
    else:
        _write_log(log_path, f"[{index}/{total}] {plan.zarr_path}")

    result = ArchiveResult(
        zarr_path=plan.zarr_path,
        subject_shape_run=plan.subject_shape_run,
        tail_posture_run=plan.tail_posture_run,
        classifier_run=plan.classifier_run,
        classifier_input_mode=plan.classifier_input_mode,
        log_path=str(log_path) if log_path is not None else "",
    )
    try:
        if plan.skip_reason:
            if log_path is None:
                print(f"plan_notes: {plan.skip_reason}", flush=True)
            else:
                _write_log(log_path, f"plan_notes: {plan.skip_reason}")
        if plan.run_subject_shape:
            result.subject_shape_status = _run_command(_subject_shape_command(plan, args), dry_run=dry_run, log_path=log_path)
            if result.subject_shape_status != "ok" and not dry_run:
                raise RuntimeError(f"subject_shape {result.subject_shape_status}")
        if plan.run_tail_posture:
            result.tail_posture_status = _run_command(_tail_posture_command(plan, args), dry_run=dry_run, log_path=log_path)
            if result.tail_posture_status != "ok" and not dry_run:
                raise RuntimeError(f"tail_posture {result.tail_posture_status}")
        if plan.run_classifier:
            result.classifier_status = _run_command(_classifier_command(plan, args), dry_run=dry_run, log_path=log_path)
            if result.classifier_status != "ok" and not dry_run:
                raise RuntimeError(f"classifier {result.classifier_status}")
        if dry_run:
            result.validation_status = "planned"
        else:
            status, detail = _validate_plan_outputs(plan)
            result.validation_status = status
            if status != "ok":
                raise RuntimeError(detail)
            if args.consolidate_metadata:
                zarr.consolidate_metadata(plan.zarr_path)
    except Exception as exc:
        result.error = str(exc)
        if log_path is None:
            print(f"error: {exc}", file=sys.stderr, flush=True)
        else:
            _write_log(log_path, f"error: {exc}")
    return result


def _write_batch_json_report(path: Path, *, plans: Sequence[ArchivePlan], results: Sequence[ArchiveResult]) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plans": [asdict(plan) for plan in plans],
        "results": [asdict(result) for result in results],
    }
    write_json_atomic(path, payload, trailing_newline=False)


def _write_markdown_report(path: Path, *, plans: Sequence[ArchivePlan], results: Sequence[ArchiveResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Megabouts Batch Pipeline Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Planned archives: {len(plans)}",
        f"- Results: {len(results)}",
        f"- Subject-shape planned: {sum(1 for plan in plans if plan.run_subject_shape)}",
        f"- Tail-posture planned: {sum(1 for plan in plans if plan.run_tail_posture)}",
        f"- Classification planned: {sum(1 for plan in plans if plan.run_classifier)}",
        f"- Validation ok: {sum(1 for result in results if result.validation_status == 'ok')}",
        f"- Errors: {sum(1 for result in results if result.error)}",
        "",
        "## Results",
        "",
        "| Zarr | Subject shape | Tail posture | Classifier | Validation | Log | Error |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |".format(
                Path(result.zarr_path).name,
                result.subject_shape_status,
                result.tail_posture_status,
                result.classifier_status,
                result.validation_status,
                result.log_path,
                (result.error or "").replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("/nvme1/recordings")])
    parser.add_argument("--roots-from-report", type=Path)
    parser.add_argument("--apply", action="store_true", help="Run commands. Default is dry-run planning.")
    parser.add_argument("--pilot-size", type=int, default=None)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of independent analysis archives to process concurrently. Default preserves serial execution.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-archive subprocess logs when --jobs > 1.",
    )
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true", help="Replace target run names if present.")
    parser.add_argument(
        "--force-classifier",
        action="store_true",
        help="Run the classifier even when the target run group already exists. Use with --overwrite to replace it.",
    )
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Plan existing outputs too.")
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--subject-shape-run", default=DEFAULT_SUBJECT_SHAPE_RUN)
    parser.add_argument("--tail-posture-run", default=DEFAULT_TAIL_POSTURE_RUN)
    parser.add_argument("--track-run", default=DEFAULT_TRACK_RUN)
    parser.add_argument("--swim-bout-run", default=DEFAULT_SWIM_BOUT_RUN)
    parser.add_argument("--classifier-run", default=DEFAULT_CLASSIFIER_RUN)
    parser.add_argument(
        "--classifier-input-mode",
        default=DEFAULT_CLASSIFIER_INPUT_MODE,
        choices=("palette_prepared_fixed_windows", "megabouts_preprocessed_full_timeseries"),
    )
    parser.add_argument("--subject-shape-chunk-size", type=int, default=256)
    parser.add_argument(
        "--subject-shape-execution-backend",
        choices=("serial_driver", "dask_worker_chunks"),
        default="serial_driver",
    )
    parser.add_argument(
        "--subject-shape-scheduler",
        choices=("single-threaded", "threads", "processes", "distributed"),
        default="single-threaded",
    )
    parser.add_argument("--subject-shape-num-workers", type=int, default=None)
    parser.add_argument("--view-family", default="megabouts_compatible")
    parser.add_argument("--head-source", choices=("head_endpoint_xy", "snout_tip_xy"), default="head_endpoint_xy")
    parser.add_argument("--keypoint-count", type=int, default=11)
    parser.add_argument("--track-id", type=int, default=0)
    parser.add_argument("--speed-level", default="exponential")
    parser.add_argument("--heading-source", default="smoothed_heading_radians")
    parser.add_argument("--bout-duration-s", type=float, default=0.2)
    parser.add_argument("--bout-duration-frames", type=int, default=None)
    parser.add_argument("--min-tail-valid-fraction", type=float, default=0.75)
    parser.add_argument("--min-traj-valid-fraction", type=float, default=0.75)
    parser.add_argument("--max-consecutive-invalid-frames", type=int, default=2)
    parser.add_argument("--align-traj-to-onset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--traj-reference-index", type=int, default=0)
    parser.add_argument("--exclude-cs", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--megabouts-repo", type=Path, default=Path("~/gitrepos/megabouts").expanduser())
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    parser.add_argument("--consolidate-metadata", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    dry_run = not bool(args.apply)

    root_inputs = _zarr_paths_from_report(args.roots_from_report) if args.roots_from_report else args.roots
    all_zarrs = _discover_analysis_zarrs(root_inputs, include_smoke=bool(args.include_smoke))
    all_plans = [
        build_archive_plan(
            zarr_path,
            subject_shape_run=args.subject_shape_run,
            tail_posture_run=args.tail_posture_run,
            track_run=args.track_run,
            swim_bout_run=args.swim_bout_run,
            classifier_run=args.classifier_run,
            classifier_input_mode=args.classifier_input_mode,
            skip_existing=bool(args.skip_existing),
            force_classifier=bool(args.force_classifier),
        )
        for zarr_path in all_zarrs
    ]
    plans = [
        plan
        for plan in all_plans
        if any((plan.run_subject_shape, plan.run_tail_posture, plan.run_classifier))
    ]
    if args.pilot_size is not None:
        plans = plans[: max(0, int(args.pilot_size))]

    print(f"analysis_archives_discovered: {len(all_plans)}")
    print(f"archives_selected: {len(plans)}")
    print(f"mode: {'apply' if args.apply else 'dry-run'}")
    print(f"subject_shape_run: {args.subject_shape_run}")
    print(f"tail_posture_run: {args.tail_posture_run}")
    print(f"classifier_run: {args.classifier_run}")
    print(f"classifier_input_mode: {args.classifier_input_mode}")
    print(f"jobs: {max(1, int(args.jobs))}")

    results: list[ArchiveResult] = []
    exit_code = 0

    jobs = max(1, int(args.jobs))
    if jobs == 1 or dry_run:
        for idx, plan in enumerate(plans, start=1):
            result = _run_archive_plan(plan, args=args, dry_run=dry_run, index=idx, total=len(plans))
            results.append(result)
            if result.error:
                exit_code = 1
                if not bool(args.continue_on_error):
                    break
    else:
        log_dir = args.log_dir or Path("/tmp") / f"palette_megabouts_batch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"log_dir: {log_dir}")
        if not bool(args.continue_on_error):
            print("note: --jobs > 1 collects all submitted archive results before exiting on errors.", flush=True)

        future_to_index = {}
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            for idx, plan in enumerate(plans, start=1):
                log_path = log_dir / f"{idx:03d}_{_safe_log_stem(plan.zarr_path)}.log"
                future = executor.submit(
                    _run_archive_plan,
                    plan,
                    args=args,
                    dry_run=dry_run,
                    index=idx,
                    total=len(plans),
                    log_path=log_path,
                )
                future_to_index[future] = idx

            ordered_results: dict[int, ArchiveResult] = {}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                result = future.result()
                ordered_results[idx] = result
                status = "ERROR" if result.error else "ok"
                print(f"[{idx}/{len(plans)}] {status} {Path(result.zarr_path).name} log={result.log_path}", flush=True)
                if result.error:
                    exit_code = 1
        results = [ordered_results[idx] for idx in sorted(ordered_results)]

    if args.json_report:
        _write_batch_json_report(args.json_report, plans=plans, results=results)
    if args.markdown_report:
        _write_markdown_report(args.markdown_report, plans=plans, results=results)

    print("\nsummary:")
    print(json.dumps({"plans": len(plans), "results": len(results), "exit_code": exit_code}, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
