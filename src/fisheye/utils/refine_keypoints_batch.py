import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import zarr

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonLogger:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, event: str, **fields: object) -> None:
        payload = {"event": event, "ts_utc": _utc_now(), "run_id": self.run_id}
        payload.update(fields)
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


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


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{os.getpid()}"


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


def _iter_zarr(roots: List[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.suffix == ".zarr":
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("zarr/*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")


def _has_refined(root: zarr.Group) -> bool:
    refined_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
    if refined_parent is None:
        return False
    latest = refined_parent.attrs.get("latest")
    if latest:
        return True
    try:
        return len(list(refined_parent.group_keys())) > 0
    except Exception:
        return False


def _select_keypoint_run(root: zarr.Group, requested: Optional[str]) -> Optional[str]:
    keypoint_parent = root.get("keypoints_runs")
    if keypoint_parent is None:
        return None
    if requested:
        return requested if requested in keypoint_parent else None
    return keypoint_parent.attrs.get("latest")


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    purpose = root.attrs.get("zarr_purpose")
    if purpose is not None:
        value = str(purpose).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


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
        root = zarr.open_group(str(zarr_path), mode="r")
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
        refined_present = _has_refined(root)
        if skip_existing and refined_present:
            plans.append(
                RefinePlan(
                    zarr_path=zarr_path,
                    status="skipped",
                    reason="refined_keypoints_runs present",
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

    args = parser.parse_args(argv)

    file_list_paths: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(_load_paths_file(path))

    roots = _resolve_root(list(args.paths) + file_list_paths)
    skip_existing = not args.no_skip_existing

    plans = _build_plans(roots, args.recursive, args.keypoint_run, skip_existing, args.zarr_use)
    if not plans:
        print("No zarr files found.")
        return 1

    if not args.apply:
        print("Planned keypoint refinement (dry-run):")
        for plan in plans:
            print(f"{plan.zarr_path}")
            print(f"  status: {plan.status}")
            if plan.keypoint_run:
                print(f"  keypoint_run: {plan.keypoint_run}")
            if plan.reason:
                print(f"  reason: {plan.reason}")
        counts = {"ok": 0, "skipped": 0, "missing": 0}
        for plan in plans:
            counts[plan.status] = counts.get(plan.status, 0) + 1
        print("\nSummary:")
        print(f"  ok: {counts.get('ok', 0)}")
        print(f"  skipped: {counts.get('skipped', 0)}")
        print(f"  missing: {counts.get('missing', 0)}")
        print("\nUse --apply to run refinement.")
        return 0

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
    )

    console = Console() if Console else None
    progress = _progress(console, len(plans))
    task_id = progress.add_task("refine_keypoints", total=len(plans)) if progress else None

    ok = skipped = missing = failed = 0
    for plan in plans:
        if plan.status == "missing":
            missing += 1
            logger.log("recording_missing", zarr=str(plan.zarr_path), reason=plan.reason)
        elif plan.status == "skipped":
            skipped += 1
            logger.log("recording_skipped", zarr=str(plan.zarr_path), reason=plan.reason)
        else:
            cmd = _build_cmd(args, plan.zarr_path, plan.keypoint_run)
            logger.log("refine_start", zarr=str(plan.zarr_path), cmd=cmd, keypoint_run=plan.keypoint_run)
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                ok += 1
                logger.log("refine_success", zarr=str(plan.zarr_path), returncode=result.returncode)
            else:
                failed += 1
                logger.log("refine_failed", zarr=str(plan.zarr_path), returncode=result.returncode)
        if progress:
            progress.advance(task_id)

    if progress:
        progress.stop()

    logger.log("run_end", ok=ok, skipped=skipped, missing=missing, failed=failed)
    logger.close()

    print("\nSummary:")
    print(f"  ok: {ok}")
    print(f"  skipped: {skipped}")
    print(f"  missing: {missing}")
    print(f"  failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
