#!/usr/bin/env python3
"""Batch run detect-quality analysis for Palette Zarr archives."""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import zarr
from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.batch_logging import utc_now

try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
except Exception:  # pragma: no cover - rich is optional
    Console = None  # type: ignore
    Progress = None  # type: ignore
    SpinnerColumn = None  # type: ignore
    TextColumn = None  # type: ignore
    BarColumn = None  # type: ignore
    TimeElapsedColumn = None  # type: ignore
    TimeRemainingColumn = None  # type: ignore


@dataclass
class DetectQualityPlan:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    detect_run: Optional[str] = None
    quality_present: bool = False


_utc_now = utc_now
JsonLogger = SharedJsonLogger


def _resolve_root(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: List[Path]) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "detect_quality_batch"
    if roots:
        return roots[0] / "logs" / "organize_recordings" / "detect_quality_batch"
    return Path.cwd() / "logs" / "organize_recordings" / "detect_quality_batch"


_run_id = make_run_id


def _progress(console: Optional[Console], total: int):
    if console is None or Progress is None:
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _select_detect_run(root: zarr.Group, requested: Optional[str]) -> Optional[str]:
    detect_parent = root.get("detect_runs")
    if detect_parent is None:
        return None
    if requested:
        return requested if requested in detect_parent else None
    latest = detect_parent.attrs.get("latest")
    if latest:
        return str(latest)
    try:
        keys = sorted(detect_parent.group_keys())
    except Exception:
        keys = []
    return keys[-1] if keys else None


def _has_quality_report(detect_group: zarr.Group, quality_run_name: Optional[str]) -> bool:
    quality_parent = detect_group.get("quality_reports")
    if quality_parent is None:
        return False
    if quality_run_name:
        return quality_run_name in quality_parent
    latest = quality_parent.attrs.get("latest")
    if latest and str(latest) in quality_parent:
        return True
    try:
        return len(list(quality_parent.group_keys())) > 0
    except Exception:
        return False


def _build_plans(
    roots: List[Path],
    recursive: bool,
    detect_run: Optional[str],
    skip_existing: bool,
    *,
    quality_run_name: Optional[str] = None,
) -> List[DetectQualityPlan]:
    plans: List[DetectQualityPlan] = []
    for zarr_path in _iter_zarr(roots, recursive):
        if not zarr_path.exists():
            continue
        try:
            root = zarr.open_group(str(zarr_path), mode="r")
        except Exception as exc:
            plans.append(
                DetectQualityPlan(
                    zarr_path=zarr_path,
                    status="missing",
                    reason=f"zarr open failed: {exc}",
                )
            )
            continue

        selected = _select_detect_run(root, detect_run)
        if selected is None:
            plans.append(
                DetectQualityPlan(
                    zarr_path=zarr_path,
                    status="missing",
                    reason="detect_runs missing" if detect_run is None else "detect_run not found",
                )
            )
            continue

        detect_group = root[f"detect_runs/{selected}"]
        quality_present = _has_quality_report(detect_group, quality_run_name)
        if skip_existing and quality_present:
            plans.append(
                DetectQualityPlan(
                    zarr_path=zarr_path,
                    status="skipped",
                    reason=(
                        f"quality_reports/{quality_run_name} present"
                        if quality_run_name
                        else "quality_reports present"
                    ),
                    detect_run=selected,
                    quality_present=True,
                )
            )
            continue

        plans.append(
            DetectQualityPlan(
                zarr_path=zarr_path,
                status="ok",
                detect_run=selected,
                quality_present=quality_present,
            )
        )
    return plans


def _build_cmd(args: argparse.Namespace, zarr_path: Path, detect_run: Optional[str]) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.refinement.detect_quality",
        str(zarr_path),
        "--threshold",
        str(args.threshold),
        "--threshold-mode",
        str(args.threshold_mode),
        "--threshold-reference-width",
        str(args.threshold_reference_width),
    ]
    if detect_run:
        cmd.extend(["--run", detect_run])
    quality_run_name = getattr(args, "quality_run_name", None)
    if quality_run_name:
        cmd.extend(["--quality-run-name", quality_run_name])
    expected_subject_count = getattr(args, "expected_subject_count", None)
    if expected_subject_count is not None:
        cmd.extend(["--expected-subject-count", str(expected_subject_count)])
    if args.no_save:
        cmd.append("--no-save")
    else:
        cmd.append("--save")
    return cmd


def _print_plan(plans: List[DetectQualityPlan]) -> None:
    counts = {"ok": 0, "skipped": 0, "missing": 0}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
        print(f"{plan.zarr_path}")
        print(f"  status: {plan.status}")
        if plan.detect_run:
            print(f"  detect_run: {plan.detect_run}")
        if plan.reason:
            print(f"  reason: {plan.reason}")
    print("\nSummary:")
    print(f"  ok: {counts.get('ok', 0)}")
    print(f"  skipped: {counts.get('skipped', 0)}")
    print(f"  missing: {counts.get('missing', 0)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch run detect-quality analysis for Palette Zarr archives.")
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarr archives.")
    parser.add_argument("--apply", action="store_true", help="Run detect-quality (default is dry-run).")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not skip when quality reports already exist.")
    parser.add_argument("--detect-run", help="Specific detect run name (default: latest).")
    parser.add_argument(
        "--quality-run-name",
        help="Optional explicit quality report run name to write under each selected detect run.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Jump threshold value (interpreted by --threshold-mode).",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["scaled", "pixels", "normalized"],
        default="scaled",
        help="How to interpret --threshold (default: scaled).",
    )
    parser.add_argument(
        "--threshold-reference-width",
        type=float,
        default=640.0,
        help="Reference width for scaled threshold mode (default: 640).",
    )
    parser.add_argument(
        "--expected-subject-count",
        type=int,
        default=None,
        help=(
            "Expected total subjects per frame. Use this for multi-arena "
            "recordings so global multi-detection labels mean over-expected, "
            "not more than one."
        ),
    )
    parser.add_argument("--no-save", action="store_true", help="Analyze only; do not write quality report.")
    parser.add_argument("--json", action="store_true", help="Emit JSON lines for plan/results.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory to store JSONL logs.")

    args = parser.parse_args(argv)
    roots = _resolve_root(args.paths)
    skip_existing = not args.no_skip_existing

    plans = _build_plans(
        roots,
        args.recursive,
        args.detect_run,
        skip_existing,
        quality_run_name=args.quality_run_name,
    )
    if not plans:
        print("No zarr files found.")
        return 1

    if not args.apply:
        if args.json:
            for plan in plans:
                print(
                    json.dumps(
                        {
                            "zarr": str(plan.zarr_path),
                            "status": plan.status,
                            "detect_run": plan.detect_run,
                            "quality_run_name": args.quality_run_name,
                            "reason": plan.reason,
                        }
                    )
                )
        else:
            _print_plan(plans)
            print("\nUse --apply to run detect quality.")
        return 0

    log_dir = _resolve_log_dir(args.log_dir, roots)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    log_path = log_dir / f"detect_quality_batch_{run_id}.jsonl"
    logger = JsonLogger(log_path, run_id)
    print(f"Log file: {log_path}")
    logger.log(
        "run_start",
        roots=[str(r) for r in roots],
        recursive=bool(args.recursive),
        detect_run=args.detect_run,
        threshold=float(args.threshold),
        threshold_mode=str(args.threshold_mode),
        threshold_reference_width=float(args.threshold_reference_width),
        expected_subject_count=args.expected_subject_count,
        quality_run_name=args.quality_run_name,
        skip_existing=bool(skip_existing),
        no_save=bool(args.no_save),
    )

    ok = 0
    failed = 0
    skipped = 0
    missing = 0

    console = Console() if Console else None
    progress = _progress(console, len(plans))
    task_id = progress.add_task("detect_quality", total=len(plans)) if progress else None

    for plan in plans:
        if plan.status == "missing":
            missing += 1
            logger.log("recording_missing", zarr=str(plan.zarr_path), reason=plan.reason, detect_run=plan.detect_run)
            if args.json:
                print(
                    json.dumps(
                        {
                            "status": "missing",
                            "zarr": str(plan.zarr_path),
                            "reason": plan.reason,
                            "quality_run_name": args.quality_run_name,
                        }
                    )
                )
        elif plan.status == "skipped":
            skipped += 1
            logger.log("recording_skipped", zarr=str(plan.zarr_path), reason=plan.reason, detect_run=plan.detect_run)
            if args.json:
                print(
                    json.dumps(
                        {
                            "status": "skipped",
                            "zarr": str(plan.zarr_path),
                            "reason": plan.reason,
                            "quality_run_name": args.quality_run_name,
                        }
                    )
                )
        else:
            cmd = _build_cmd(args, plan.zarr_path, plan.detect_run)
            logger.log(
                "quality_start",
                zarr=str(plan.zarr_path),
                detect_run=plan.detect_run,
                quality_run_name=args.quality_run_name,
                cmd=cmd,
            )
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                ok += 1
                logger.log(
                    "quality_success",
                    zarr=str(plan.zarr_path),
                    detect_run=plan.detect_run,
                    quality_run_name=args.quality_run_name,
                    returncode=result.returncode,
                )
                if args.json:
                    print(
                        json.dumps(
                            {
                                "status": "ok",
                                "zarr": str(plan.zarr_path),
                                "detect_run": plan.detect_run,
                                "quality_run_name": args.quality_run_name,
                            }
                        )
                    )
            else:
                failed += 1
                logger.log(
                    "quality_failed",
                    zarr=str(plan.zarr_path),
                    detect_run=plan.detect_run,
                    quality_run_name=args.quality_run_name,
                    returncode=result.returncode,
                )
                if args.json:
                    print(
                        json.dumps(
                            {
                                "status": "failed",
                                "zarr": str(plan.zarr_path),
                                "detect_run": plan.detect_run,
                                "quality_run_name": args.quality_run_name,
                                "returncode": int(result.returncode),
                            }
                        )
                    )
        if progress:
            progress.advance(task_id)

    if progress:
        progress.stop()

    logger.log("run_end", ok=ok, failed=failed, skipped=skipped, missing=missing)
    logger.close()

    if not args.json:
        print("\nSummary:")
        print(f"  ok: {ok}")
        print(f"  failed: {failed}")
        print(f"  skipped: {skipped}")
        print(f"  missing: {missing}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
