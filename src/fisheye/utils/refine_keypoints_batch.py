from fisheye.shared.zarr_helpers import infer_zarr_use as _infer_zarr_use
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
from fisheye.utils.auto_keypoint_review import AutoReviewPolicy, apply_auto_review

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
class RefinePlan:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    keypoint_run: Optional[str] = None
    refined_present: bool = False


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
        return Path(env_root) / "refine_keypoints_batch"
    if roots:
        return roots[0] / "logs" / "refine_keypoints_batch"
    return Path.cwd() / "logs" / "refine_keypoints_batch"


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


def _load_paths_file(path: Path) -> List[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items: List[Path] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _decode_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _source_keypoints_run(run_group: object) -> Optional[str]:
    attrs = getattr(run_group, "attrs", None)
    if attrs is None:
        return None
    source = _decode_text(attrs.get("source_keypoints_run"))
    if source:
        return source
    return _decode_text(attrs.get("source_keypoint_run"))


def _has_matching_refined(root: zarr.Group, *, keypoint_run: str) -> bool:
    refined_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
    if refined_parent is None:
        return False
    latest = _decode_text(refined_parent.attrs.get("latest"))
    if latest and latest in refined_parent:
        try:
            if _source_keypoints_run(refined_parent[latest]) == keypoint_run:
                return True
        except Exception:
            pass
    try:
        run_names = list(refined_parent.group_keys())
    except Exception:
        return False
    for run_name in run_names:
        try:
            if _source_keypoints_run(refined_parent[run_name]) == keypoint_run:
                return True
        except Exception:
            continue
    return False


def _select_keypoint_run(root: zarr.Group, requested: Optional[str]) -> Optional[str]:
    keypoint_parent = root.get("keypoints_runs")
    if keypoint_parent is None:
        return None
    if requested:
        return requested if requested in keypoint_parent else None
    return _decode_text(keypoint_parent.attrs.get("latest"))


def _build_plans(
    roots: List[Path],
    recursive: bool,
    keypoint_run: Optional[str],
    skip_existing: bool,
    zarr_use_filter: str,
) -> List[RefinePlan]:
    plans: List[RefinePlan] = []
    for zarr_path in _iter_zarr(roots, recursive):
        if not zarr_path.exists():
            continue
        root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        if zarr_use_filter != "any":
            observed_use = _infer_zarr_use(root, zarr_path)
            if observed_use != zarr_use_filter:
                plans.append(
                    RefinePlan(
                        zarr_path=zarr_path,
                        status="skipped",
                        reason=f"zarr_use mismatch (wanted={zarr_use_filter}, found={observed_use or 'unknown'})",
                    )
                )
                continue
        selected = _select_keypoint_run(root, keypoint_run)
        if selected is None:
            plans.append(
                RefinePlan(
                    zarr_path=zarr_path,
                    status="missing",
                    reason="keypoints_runs missing" if keypoint_run is None else "keypoint_run not found",
                )
            )
            continue
        refined_present = _has_matching_refined(root, keypoint_run=selected)
        if skip_existing and refined_present:
            plans.append(
                RefinePlan(
                    zarr_path=zarr_path,
                    status="skipped",
                    reason=f"refined_keypoints run already exists for source '{selected}'",
                    keypoint_run=selected,
                    refined_present=True,
                )
            )
            continue
        plans.append(
            RefinePlan(
                zarr_path=zarr_path,
                status="ok",
                keypoint_run=selected,
                refined_present=refined_present,
            )
        )
    return plans


def _build_cmd(args: argparse.Namespace, zarr_path: Path, keypoint_run: Optional[str]) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.refinement.refine_keypoints",
        str(zarr_path),
        "--config",
        args.config,
    ]
    if keypoint_run:
        cmd.extend(["--keypoint-run", keypoint_run])
    if args.chunk_size is not None:
        cmd.extend(["--chunk-size", str(args.chunk_size)])
    if args.scheduler is not None:
        cmd.extend(["--scheduler", args.scheduler])
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
    if args.memory_limit is not None:
        cmd.extend(["--memory-limit", args.memory_limit])
    return cmd


def _build_auto_review_policy(args: argparse.Namespace) -> AutoReviewPolicy:
    tags = tuple(args.auto_review_disqualifying_tag) if args.auto_review_disqualifying_tag else AutoReviewPolicy.disqualifying_tags
    return AutoReviewPolicy(
        policy_id=str(args.auto_review_policy_id),
        policy_version=int(args.auto_review_policy_version),
        remaining_failures_max=int(args.auto_review_remaining_failures_max),
        refined_success_rate_min=float(args.auto_review_refined_success_rate_min),
        usable_keypoints_rate_min=float(args.auto_review_usable_rate_min),
        confidence_valid_rate_min=float(args.auto_review_confidence_valid_rate_min),
        geometry_valid_rate_min=float(args.auto_review_geometry_valid_rate_min),
        disqualifying_tags=tags,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch refine keypoints for Palette Zarr archives.")
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr or root path per line (comments with # allowed).",
    )
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarrs.")
    parser.add_argument("--apply", action="store_true", help="Run refinement (default is dry-run).")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Filter zarr archives by purpose (default: analysis).",
    )
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not skip when refined runs already exist.")
    parser.add_argument("--keypoint-run", help="Keypoint run name to refine (defaults to latest).")
    parser.add_argument("--config", default="configs/fisheye/default.yaml", help="Config path.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Refinement chunk size override.")
    parser.add_argument(
        "--scheduler",
        choices=["processes", "threads", "distributed"],
        default=None,
        help="Refinement scheduler override.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Refinement worker count override.")
    parser.add_argument("--memory-limit", default=None, help="Refinement worker memory limit override.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory to store JSONL logs.")
    parser.add_argument("--no-log", action="store_true", help="Disable JSONL logging in apply mode.")
    parser.add_argument("--json", action="store_true", help="Emit JSON lines for plans/results.")
    parser.add_argument(
        "--auto-review-full-recording",
        action="store_true",
        help=(
            "After successful refinement, write algorithmic keypoint_review_status with "
            "intended_use=full_recording using policy thresholds."
        ),
    )
    parser.add_argument("--auto-review-policy-id", default=AutoReviewPolicy.policy_id, help="Auto-review policy identifier.")
    parser.add_argument("--auto-review-policy-version", type=int, default=AutoReviewPolicy.policy_version, help="Auto-review policy version.")
    parser.add_argument("--auto-review-remaining-failures-max", type=int, default=AutoReviewPolicy.remaining_failures_max)
    parser.add_argument("--auto-review-refined-success-rate-min", type=float, default=AutoReviewPolicy.refined_success_rate_min)
    parser.add_argument("--auto-review-usable-rate-min", type=float, default=AutoReviewPolicy.usable_keypoints_rate_min)
    parser.add_argument("--auto-review-confidence-valid-rate-min", type=float, default=AutoReviewPolicy.confidence_valid_rate_min)
    parser.add_argument("--auto-review-geometry-valid-rate-min", type=float, default=AutoReviewPolicy.geometry_valid_rate_min)
    parser.add_argument(
        "--auto-review-disqualifying-tag",
        action="append",
        default=None,
        help="Reason tag that forces needs_review (repeatable).",
    )
    parser.add_argument("--auto-review-reviewer", help="Optional reviewer label for auto-review payload.")
    parser.add_argument("--auto-review-notes", help="Optional notes for auto-review payload.")
    parser.add_argument("--auto-review-overwrite-existing", action="store_true", help="Overwrite existing keypoint_review_status.")
    parser.add_argument(
        "--auto-review-no-latest",
        action="store_true",
        help="Do not update refined parent keypoint_review_status_latest pointer.",
    )

    args = parser.parse_args(argv)

    file_list_paths: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(_load_paths_file(path))

    roots = _resolve_root(list(args.paths) + file_list_paths)
    skip_existing = not args.no_skip_existing
    auto_review_enabled = bool(args.auto_review_full_recording)
    auto_policy = _build_auto_review_policy(args) if auto_review_enabled else None

    plans = _build_plans(roots, args.recursive, args.keypoint_run, skip_existing, args.zarr_use)
    if not plans:
        print("No zarr files found.")
        return 1

    if not args.apply:
        counts = {"ok": 0, "skipped": 0, "missing": 0}
        for plan in plans:
            counts[plan.status] = counts.get(plan.status, 0) + 1
        if args.json:
            for plan in plans:
                print(
                    json.dumps(
                        {
                            "zarr": str(plan.zarr_path),
                            "status": plan.status,
                            "reason": plan.reason,
                            "keypoint_run": plan.keypoint_run,
                            "refined_present": bool(plan.refined_present),
                        },
                        sort_keys=True,
                    )
                )
        else:
            print("Planned keypoint refinement (dry-run):")
            for plan in plans:
                print(f"{plan.zarr_path}")
                print(f"  status: {plan.status}")
                if plan.keypoint_run:
                    print(f"  keypoint_run: {plan.keypoint_run}")
                if plan.reason:
                    print(f"  reason: {plan.reason}")
            print("\nSummary:")
            print(f"  ok: {counts.get('ok', 0)}")
            print(f"  skipped: {counts.get('skipped', 0)}")
            print(f"  missing: {counts.get('missing', 0)}")
            print("\nUse --apply to run refinement.")
        return 0

    logger: Optional[JsonLogger] = None
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, roots)
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = _run_id()
        log_path = log_dir / f"refine_keypoints_batch_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            roots=[str(r) for r in roots],
            file_list=[str(p) for p in (args.file_list or [])],
            recursive=bool(args.recursive),
            zarr_use=str(args.zarr_use),
            auto_review_enabled=auto_review_enabled,
            auto_review_policy=(
                {
                    "policy_id": auto_policy.policy_id,
                    "policy_version": auto_policy.policy_version,
                    "remaining_failures_max": auto_policy.remaining_failures_max,
                    "refined_success_rate_min": auto_policy.refined_success_rate_min,
                    "usable_keypoints_rate_min": auto_policy.usable_keypoints_rate_min,
                    "confidence_valid_rate_min": auto_policy.confidence_valid_rate_min,
                    "geometry_valid_rate_min": auto_policy.geometry_valid_rate_min,
                    "disqualifying_tags": list(auto_policy.disqualifying_tags),
                }
                if auto_policy is not None
                else None
            ),
        )

    console = Console() if Console else None
    progress = _progress(console, len(plans))
    task_id = progress.add_task("refine_keypoints", total=len(plans)) if progress else None

    ok = skipped = missing = failed = 0
    for plan in plans:
        if plan.status == "missing":
            missing += 1
            if args.json:
                print(json.dumps({"status": "missing", "zarr": str(plan.zarr_path), "reason": plan.reason}))
            if logger is not None:
                logger.log("recording_missing", zarr=str(plan.zarr_path), reason=plan.reason)
        elif plan.status == "skipped":
            skipped += 1
            if args.json:
                print(json.dumps({"status": "skipped", "zarr": str(plan.zarr_path), "reason": plan.reason}))
            if logger is not None:
                logger.log("recording_skipped", zarr=str(plan.zarr_path), reason=plan.reason)
        else:
            cmd = _build_cmd(args, plan.zarr_path, plan.keypoint_run)
            if logger is not None:
                logger.log("refine_start", zarr=str(plan.zarr_path), cmd=cmd, keypoint_run=plan.keypoint_run)
            if not args.json:
                print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                auto_review_result: Optional[dict] = None
                auto_review_error: Optional[str] = None
                if auto_review_enabled and auto_policy is not None:
                    try:
                        auto_review_result = apply_auto_review(
                            plan.zarr_path,
                            source_keypoints_run=plan.keypoint_run,
                            policy=auto_policy,
                            intended_use="full_recording",
                            reviewer=args.auto_review_reviewer,
                            notes=args.auto_review_notes,
                            overwrite_existing=bool(args.auto_review_overwrite_existing),
                            update_latest=not bool(args.auto_review_no_latest),
                            dry_run=False,
                        )
                    except Exception as exc:
                        auto_review_error = str(exc)

                if auto_review_error is None:
                    ok += 1
                    if args.json:
                        payload = {
                            "status": "ok",
                            "zarr": str(plan.zarr_path),
                            "returncode": int(result.returncode),
                            "keypoint_run": plan.keypoint_run,
                        }
                        if auto_review_enabled:
                            payload["auto_review"] = auto_review_result
                        print(json.dumps(payload, sort_keys=True))
                    if logger is not None:
                        logger.log(
                            "refine_success",
                            zarr=str(plan.zarr_path),
                            returncode=result.returncode,
                            auto_review=auto_review_result,
                        )
                else:
                    failed += 1
                    if args.json:
                        print(
                            json.dumps(
                                {
                                    "status": "failed",
                                    "zarr": str(plan.zarr_path),
                                    "returncode": int(result.returncode),
                                    "keypoint_run": plan.keypoint_run,
                                    "failure_stage": "auto_review",
                                    "error": auto_review_error,
                                },
                                sort_keys=True,
                            )
                        )
                    if logger is not None:
                        logger.log(
                            "auto_review_failed",
                            zarr=str(plan.zarr_path),
                            returncode=result.returncode,
                            keypoint_run=plan.keypoint_run,
                            error=auto_review_error,
                        )
            else:
                failed += 1
                if args.json:
                    print(
                        json.dumps(
                            {
                                "status": "failed",
                                "zarr": str(plan.zarr_path),
                                "returncode": int(result.returncode),
                                "keypoint_run": plan.keypoint_run,
                            },
                            sort_keys=True,
                        )
                    )
                if logger is not None:
                    logger.log("refine_failed", zarr=str(plan.zarr_path), returncode=result.returncode)
        if progress:
            progress.advance(task_id)

    if progress:
        progress.stop()

    if logger is not None:
        logger.log("run_end", ok=ok, skipped=skipped, missing=missing, failed=failed)
        logger.close()

    if not args.json:
        print("\nSummary:")
        print(f"  ok: {ok}")
        print(f"  skipped: {skipped}")
        print(f"  missing: {missing}")
        print(f"  failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
