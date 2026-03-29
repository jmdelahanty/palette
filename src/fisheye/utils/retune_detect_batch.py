import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
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
class RetunePlan:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    refined_run: Optional[str] = None
    variant: Optional[str] = None
    output_group_exists: bool = False
    target_frames: Optional[List[int]] = None


_utc_now = utc_now
JsonLogger = SharedJsonLogger


def _resolve_root(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _load_paths_file(path: Path) -> List[Path]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    items: List[Path] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _normalize_path(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _load_frame_list(path: Path) -> dict[str, List[int]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Frame list must be JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame list must be a JSON object: {path}")
    parsed: dict[str, List[int]] = {}
    for key, value in data.items():
        if not isinstance(value, list):
            continue
        frames: List[int] = []
        for item in value:
            try:
                frames.append(int(item))
            except (TypeError, ValueError):
                continue
        if not frames:
            continue
        norm_key = _normalize_path(Path(str(key)))
        parsed[norm_key] = sorted(set(frames))
    return parsed


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: List[Path]) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "retune_detect_batch"
    if roots:
        return roots[0] / "logs" / "retune_detect_batch"
    return Path.cwd() / "logs" / "retune_detect_batch"


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


def _iter_zarr(roots: List[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.suffix == ".zarr":
            yield root
            continue
        if root.is_dir() and root.suffix == ".zarr":
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("zarr/*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")


def _select_refined_run(root: zarr.Group, requested: Optional[str]) -> Optional[str]:
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        return None
    if requested:
        return requested if requested in refined_parent else None
    return refined_parent.attrs.get("latest")


def _select_variant(refined_run: zarr.Group, requested: Optional[str]) -> Optional[str]:
    if requested and requested in refined_run:
        return requested
    if "interpolated" in refined_run:
        return "interpolated"
    if "filtered" in refined_run:
        return "filtered"
    try:
        keys = list(refined_run.group_keys())
    except Exception:
        keys = []
    return keys[0] if keys else None


def _background_ready(root: zarr.Group) -> bool:
    bg_parent = root.get("background_runs")
    if bg_parent is None:
        return False
    return bool(bg_parent.attrs.get("latest"))


def _detect_method_is_yolo(root: zarr.Group, refined_run: zarr.Group) -> bool:
    source_detect = refined_run.attrs.get("source_detect_run")
    if not source_detect:
        return False
    detect_group = root.get(f"detect_runs/{source_detect}")
    if detect_group is None:
        return False
    detection_method = str(detect_group.attrs.get("detection_method", "")).lower()
    pipeline_type = str(detect_group.attrs.get("pipeline_type", "")).lower()
    model_type = str(detect_group.attrs.get("model_type", "")).lower()
    return "yolo" in detection_method or "yolo" in pipeline_type or "yolo" in model_type


def _build_plans(
    roots: List[Path],
    recursive: bool,
    refined_run: Optional[str],
    variant: Optional[str],
    output_group: str,
    skip_existing: bool,
    force: bool,
    target_frames_by_zarr: Optional[dict[str, List[int]]] = None,
) -> List[RetunePlan]:
    plans: List[RetunePlan] = []
    for zarr_path in _iter_zarr(roots, recursive):
        if not zarr_path.exists():
            continue
        normalized_path = _normalize_path(zarr_path)
        target_frames = target_frames_by_zarr.get(normalized_path) if target_frames_by_zarr else None
        root = zarr.open_group(str(zarr_path), mode="r")
        selected = _select_refined_run(root, refined_run)
        if selected is None:
            plans.append(
                RetunePlan(
                    zarr_path=zarr_path,
                    status="missing",
                    reason="refined_detect_runs missing" if refined_run is None else "refined_run not found",
                )
            )
            continue
        refined_parent = root.get("refined_detect_runs")
        refined_group = refined_parent[selected]
        selected_variant = _select_variant(refined_group, variant)
        if not selected_variant:
            plans.append(
                RetunePlan(
                    zarr_path=zarr_path,
                    status="missing",
                    reason="no refined variants found",
                    refined_run=selected,
                )
            )
            continue
        if _detect_method_is_yolo(root, refined_group):
            plans.append(
                RetunePlan(
                    zarr_path=zarr_path,
                    status="skipped",
                    reason="yolo detect run (retune not supported)",
                    refined_run=selected,
                    variant=selected_variant,
                )
            )
            continue
        if not _background_ready(root):
            plans.append(
                RetunePlan(
                    zarr_path=zarr_path,
                    status="missing",
                    reason="background_runs missing or latest not set",
                    refined_run=selected,
                    variant=selected_variant,
                )
            )
            continue
        if "raw_video" not in root or "images_ds" not in root["raw_video"]:
            plans.append(
                RetunePlan(
                    zarr_path=zarr_path,
                    status="missing",
                    reason="raw_video/images_ds missing",
                    refined_run=selected,
                    variant=selected_variant,
                )
            )
            continue
        output_group_exists = output_group in refined_group
        if skip_existing and output_group_exists:
            plans.append(
                RetunePlan(
                    zarr_path=zarr_path,
                    status="skipped",
                    reason=f"output_group '{output_group}' exists (use --overwrite for incremental retune)",
                    refined_run=selected,
                    variant=selected_variant,
                    output_group_exists=True,
                    target_frames=target_frames,
                )
            )
            continue
        try:
            # If the output group already exists, retunes should be incremental
            # and missing counts should be computed from that group.
            base_group = refined_group[output_group] if output_group_exists else refined_group[selected_variant]
            n_frames = int(root["raw_video"]["images_ds"].shape[0])
            if target_frames:
                filtered = [f for f in target_frames if 0 <= f < n_frames]
                if not filtered:
                    plans.append(
                        RetunePlan(
                            zarr_path=zarr_path,
                            status="skipped",
                            reason="no targeted frames in range",
                            refined_run=selected,
                            variant=selected_variant,
                            output_group_exists=output_group_exists,
                            target_frames=[],
                        )
                    )
                    continue
                target_frames = sorted(set(filtered))
            frame_counts = base_group.get("frame_counts")
            if frame_counts is None:
                frame_indices = np.asarray(base_group["frame_indices"][:], dtype=np.int64)
                counts = np.bincount(frame_indices, minlength=n_frames)
            else:
                counts = np.asarray(frame_counts[:], dtype=np.int64)
                if counts.shape[0] != n_frames:
                    counts = counts[: min(n_frames, counts.shape[0])]
            if target_frames:
                plans.append(
                    RetunePlan(
                        zarr_path=zarr_path,
                        status="ok",
                        reason=f"targeted_frames={len(target_frames)}",
                        refined_run=selected,
                        variant=selected_variant,
                        output_group_exists=output_group_exists,
                        target_frames=target_frames,
                    )
                )
                continue
            if int(np.sum(counts == 0)) == 0 and not force:
                plans.append(
                    RetunePlan(
                        zarr_path=zarr_path,
                        status="skipped",
                        reason="no missing detections",
                        refined_run=selected,
                        variant=selected_variant,
                        target_frames=target_frames,
                    )
                )
                continue
            if int(np.sum(counts == 0)) == 0 and force:
                plans.append(
                    RetunePlan(
                        zarr_path=zarr_path,
                        status="ok",
                        reason="no missing detections (forced)",
                        refined_run=selected,
                        variant=selected_variant,
                        output_group_exists=output_group_exists,
                        target_frames=target_frames,
                    )
                )
                continue
        except Exception as exc:
            plans.append(
                RetunePlan(
                    zarr_path=zarr_path,
                    status="missing",
                    reason=f"failed to inspect refined frames: {exc}",
                    refined_run=selected,
                    variant=selected_variant,
                    target_frames=target_frames,
                )
            )
            continue
        plans.append(
            RetunePlan(
                zarr_path=zarr_path,
                status="ok",
                refined_run=selected,
                variant=selected_variant,
                output_group_exists=output_group_exists,
                target_frames=target_frames,
            )
        )
    return plans


def _build_cmd(args: argparse.Namespace, plan: RetunePlan) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.tune.detect_review",
        str(plan.zarr_path),
        "--retune-ui" if args.retune_ui else "--retune",
        "--config",
        args.config,
        "--output-group",
        args.output_group,
    ]
    if plan.refined_run:
        cmd.extend(["--refined-run", plan.refined_run])
    if args.variant:
        cmd.extend(["--variant", args.variant])
    if args.overwrite:
        cmd.append("--overwrite")
    if plan.target_frames:
        cmd.extend(["--frames", ",".join(str(frame) for frame in plan.target_frames)])
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.retune_score is not None:
        cmd.extend(["--retune-score", str(args.retune_score)])
    if args.retune_class_id is not None:
        cmd.extend(["--retune-class-id", str(args.retune_class_id)])
    if args.use_full_res:
        cmd.append("--use-full-res")
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch retune missing detections for Palette Zarr archives.")
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument(
        "--frame-list",
        type=Path,
        action="append",
        help="JSON file mapping zarr paths to frame lists for targeted retune.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarrs.")
    parser.add_argument("--apply", action="store_true", help="Run retune (default is dry-run).")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not skip when output group already exists.")
    parser.add_argument("--refined-run", help="Refined run name to retune (defaults to latest).")
    parser.add_argument("--variant", choices=["filtered", "interpolated"], help="Refined variant to retune.")
    parser.add_argument("--output-group", default="manual", help="Output group name for retuned detections.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output group.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retune even when no missing detections are found (useful with file lists).",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to retune.")
    parser.add_argument("--use-full-res", action="store_true", help="Use full-resolution frames for retune.")
    parser.add_argument(
        "--retune-ui",
        action="store_true",
        help="Run interactive retune UI per recording (requires --apply).",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Prompt before each recording (use with --retune-ui for human-in-the-loop).",
    )
    parser.add_argument("--config", default="pipeline_config.yaml", help="Config path.")
    parser.add_argument("--retune-score", type=float, default=1.0, help="Score for retuned detections.")
    parser.add_argument("--retune-class-id", type=int, default=0, help="Class id for retuned detections.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory to store JSONL logs.")

    args = parser.parse_args(argv)

    file_list_paths: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(_load_paths_file(path))

    frame_map: dict[str, List[int]] = {}
    if args.frame_list:
        for path in args.frame_list:
            data = _load_frame_list(path)
            for key, frames in data.items():
                if key in frame_map:
                    frame_map[key] = sorted(set(frame_map[key]).union(frames))
                else:
                    frame_map[key] = frames

    frame_list_roots = [Path(key) for key in frame_map.keys()]

    if args.paths:
        roots = _resolve_root(args.paths)
        roots.extend(file_list_paths)
        roots.extend(frame_list_roots)
    elif file_list_paths or frame_list_roots:
        roots = file_list_paths + frame_list_roots
    else:
        roots = _resolve_root(None)

    deduped: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    roots = deduped
    force = bool(args.force or args.file_list or args.frame_list)
    if force and not args.force and (args.file_list or args.frame_list):
        print("Note: file list provided; forcing retune even when no missing detections.")
    # --overwrite implies we should not skip existing output groups
    skip_existing = not args.no_skip_existing and not args.overwrite

    plans = _build_plans(
        roots,
        args.recursive,
        args.refined_run,
        args.variant,
        args.output_group,
        skip_existing,
        force,
        frame_map or None,
    )
    if not plans:
        print("No zarr files found.")
        return 1

    if not args.apply:
        print("Planned detection retune (dry-run):")
        for plan in plans:
            print(f"{plan.zarr_path}")
            print(f"  status: {plan.status}")
            if plan.refined_run:
                print(f"  refined_run: {plan.refined_run}")
            if plan.variant:
                print(f"  variant: {plan.variant}")
            if plan.target_frames:
                print(f"  target_frames: {len(plan.target_frames)}")
            if plan.reason:
                print(f"  reason: {plan.reason}")
        counts = {"ok": 0, "skipped": 0, "missing": 0}
        for plan in plans:
            counts[plan.status] = counts.get(plan.status, 0) + 1
        print("\nSummary:")
        print(f"  ok: {counts.get('ok', 0)}")
        print(f"  skipped: {counts.get('skipped', 0)}")
        print(f"  missing: {counts.get('missing', 0)}")
        print("\nUse --apply to run retune.")
        return 0

    log_dir = _resolve_log_dir(args.log_dir, roots)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    log_path = log_dir / f"retune_detect_batch_{run_id}.jsonl"
    logger = JsonLogger(log_path, run_id)
    print(f"Log file: {log_path}")
    logger.log(
        "run_start",
        roots=[str(r) for r in roots],
        recursive=bool(args.recursive),
        retune_ui=bool(args.retune_ui),
        prompt=bool(args.prompt),
        force=bool(force),
        frame_list=bool(frame_map),
    )

    console = Console() if Console else None
    progress = _progress(console, len(plans))
    task_id = progress.add_task("retune_detect", total=len(plans)) if progress else None

    ok = skipped = missing = failed = 0
    for plan in plans:
        if plan.status == "missing":
            missing += 1
            logger.log("recording_missing", zarr=str(plan.zarr_path), reason=plan.reason)
        elif plan.status == "skipped":
            skipped += 1
            logger.log(
                "recording_skipped",
                zarr=str(plan.zarr_path),
                reason=plan.reason,
                refined_run=plan.refined_run,
                variant=plan.variant,
                target_frames=len(plan.target_frames) if plan.target_frames else 0,
            )
        else:
            if args.prompt:
                response = input(
                    f"\nRetune {plan.zarr_path} (refined_run={plan.refined_run})? "
                    "[Enter]=run, s=skip, q=quit: "
                ).strip().lower()
                if response == "q":
                    logger.log("run_cancelled", zarr=str(plan.zarr_path))
                    break
                if response == "s":
                    skipped += 1
                    logger.log("recording_skipped", zarr=str(plan.zarr_path), reason="user_skip")
                    if progress:
                        progress.advance(task_id)
                    continue
            cmd = _build_cmd(args, plan)
            logger.log(
                "retune_start",
                zarr=str(plan.zarr_path),
                cmd=cmd,
                refined_run=plan.refined_run,
                target_frames=len(plan.target_frames) if plan.target_frames else 0,
            )
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                ok += 1
                logger.log("retune_success", zarr=str(plan.zarr_path), returncode=result.returncode)
            else:
                failed += 1
                logger.log("retune_failed", zarr=str(plan.zarr_path), returncode=result.returncode)
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
