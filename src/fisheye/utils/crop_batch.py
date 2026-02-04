import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import zarr
import yaml

from ..tracking.crop import (
    crop_detections,
    get_detection_source_info,
    get_crop_parameters,
    infer_detection_source_type,
)

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
class CropPlan:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    source_type: Optional[str] = None
    source_path: Optional[str] = None
    roi_size: Optional[Tuple[int, int]] = None
    preferred_policy: Optional[str] = None
    latest_crop: Optional[str] = None
    latest_signature: Optional[Dict[str, object]] = None


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
        return Path(env_root) / "crop_batch"
    if roots:
        return roots[0] / "logs" / "crop_batch"
    return Path.cwd() / "logs" / "crop_batch"


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


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _normalize_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().strip("/")
    return text or None


def _normalize_str(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").strip() or None
    text = str(value).strip()
    return text or None


def _normalize_roi(value: object) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _latest_crop_signature(root: zarr.Group) -> Tuple[Optional[str], Optional[Dict[str, object]]]:
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        return None, None
    latest = crop_parent.attrs.get("latest")
    if not latest or latest not in crop_parent:
        return None, None
    crop_group = crop_parent[latest]
    signature = {
        "detection_source_path": crop_group.attrs.get("detection_source_path"),
        "detection_source_type": crop_group.attrs.get("detection_source_type"),
        "detection_preferred_policy": crop_group.attrs.get("detection_preferred_policy"),
        "roi_size": crop_group.attrs.get("roi_size"),
        "status": crop_group.attrs.get("status"),
    }
    return str(latest), signature


def _diff_signature(
    desired: Dict[str, object],
    existing: Dict[str, object],
    compare_policy: bool,
) -> List[str]:
    diffs: List[str] = []
    desired_path = _normalize_path(desired.get("detection_source_path"))
    existing_path = _normalize_path(existing.get("detection_source_path"))
    if desired_path != existing_path:
        diffs.append("source_path")
    desired_type = _normalize_str(desired.get("detection_source_type"))
    existing_type = _normalize_str(existing.get("detection_source_type"))
    if desired_type != existing_type:
        diffs.append("source_type")
    desired_roi = _normalize_roi(desired.get("roi_size"))
    existing_roi = _normalize_roi(existing.get("roi_size"))
    if desired_roi != existing_roi:
        diffs.append("roi_size")
    if compare_policy:
        desired_policy = _normalize_str(desired.get("detection_preferred_policy"))
        existing_policy = _normalize_str(existing.get("detection_preferred_policy"))
        if desired_policy != existing_policy:
            diffs.append("preferred_policy")
    return diffs


def _build_plan(
    zarr_path: Path,
    config: Dict[str, Any],
    source_type: str,
    source_path: Optional[str],
    preferred_policy: Optional[str],
    force_new: bool,
) -> CropPlan:
    if not zarr_path.exists():
        return CropPlan(zarr_path=zarr_path, status="missing", reason="zarr not found")

    root = zarr.open_group(str(zarr_path), mode="r")
    try:
        resolved_path, _source_group, _detection_source, resolved_type = get_detection_source_info(
            root=root,
            source_type=source_type,
            source_path_override=source_path,
            console=None,
            preferred_policy=preferred_policy,
        )
    except ValueError as exc:
        return CropPlan(
            zarr_path=zarr_path,
            status="missing",
            reason=str(exc),
        )

    crop_params, _ = get_crop_parameters(root, config, console=None)
    roi_size = tuple(crop_params.get("roi_sz", [256, 256]))

    desired = {
        "detection_source_path": resolved_path,
        "detection_source_type": resolved_type,
        "detection_preferred_policy": preferred_policy,
        "roi_size": roi_size,
    }

    latest_crop, latest_signature = _latest_crop_signature(root)
    status = "ok"
    reason = None

    if latest_signature:
        latest_status = _normalize_str(latest_signature.get("status"))
        if latest_status and latest_status.lower() != "completed":
            reason = f"latest crop run '{latest_crop}' not completed"
        else:
            diffs = _diff_signature(
                desired,
                latest_signature,
                compare_policy=preferred_policy is not None,
            )
            if not diffs and not force_new:
                status = "skipped"
                reason = f"matches latest crop run '{latest_crop}'"
            elif diffs:
                reason = "differs: " + ", ".join(diffs)
            elif force_new:
                reason = "force-new"

    return CropPlan(
        zarr_path=zarr_path,
        status=status,
        reason=reason,
        source_type=resolved_type,
        source_path=resolved_path,
        roi_size=roi_size,
        preferred_policy=preferred_policy,
        latest_crop=latest_crop,
        latest_signature=latest_signature,
    )


def _resolve_targets(paths: List[Path], recursive: bool) -> List[Path]:
    seen: set[str] = set()
    ordered: List[Path] = []
    for path in _iter_zarr(paths, recursive):
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch crop ROIs for Palette Zarr recordings.")
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--file-list", type=Path, help="Text file with zarr paths to process.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarrs.")
    parser.add_argument("--apply", action="store_true", help="Run cropping (default is dry-run).")
    parser.add_argument("--force-new", action="store_true", help="Always create a new crop run.")
    parser.add_argument(
        "--source-type",
        type=str,
        default=None,
        choices=["detect", "filtered", "interpolated", "manual", "preferred", "auto"],
        help="Detection source (default: config value, otherwise preferred).",
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Explicit detection source path (e.g. detect_runs/<run> or refined_detect_runs/<run>/manual).",
    )
    parser.add_argument(
        "--preferred-policy",
        type=str,
        default=None,
        choices=["training", "full_recording"],
        help="Policy for preferred/auto source selection.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional crop config YAML (defaults to built-in/zarr defaults).",
    )
    parser.add_argument("--scheduler", choices=["processes", "threads", "distributed"], default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--acceleration", choices=["auto", "gpu", "cpu"], default=None)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for JSONL logs.")

    args = parser.parse_args(argv)

    explicit_roots: List[Path] = []
    if args.file_list:
        explicit_roots.extend(_load_paths_file(args.file_list))
    explicit_roots.extend(args.paths or [])
    if not explicit_roots:
        explicit_roots = _resolve_root(None)

    roots = _resolve_root(explicit_roots)
    zarr_paths = _resolve_targets(roots, args.recursive)
    if not zarr_paths:
        print("No zarr files found.")
        return 1

    config = _load_config(args.config)
    crop_cfg = config.get("crop", {}) or {}
    raw_source_type = args.source_type or crop_cfg.get("source_type") or "preferred"
    source_path = _normalize_path(args.source_path or crop_cfg.get("source_path"))
    source_type = infer_detection_source_type(source_path, raw_source_type)
    preferred_policy = args.preferred_policy or crop_cfg.get("preferred_policy")

    plans: List[CropPlan] = []
    for zarr_path in zarr_paths:
        plans.append(
            _build_plan(
                zarr_path=zarr_path,
                config=config,
                source_type=source_type,
                source_path=source_path,
                preferred_policy=preferred_policy,
                force_new=bool(args.force_new),
            )
        )

    if not args.apply:
        print("Planned crop runs (dry-run):")
        for plan in plans:
            print(f"{plan.zarr_path}")
            print(f"  status: {plan.status}")
            if plan.reason:
                print(f"  reason: {plan.reason}")
            if plan.source_type:
                print(f"  source_type: {plan.source_type}")
            if plan.source_path:
                print(f"  source_path: {plan.source_path}")
            if plan.roi_size:
                print(f"  roi_size: {plan.roi_size[0]}x{plan.roi_size[1]}")
            if plan.latest_crop:
                print(f"  latest_crop: {plan.latest_crop}")
                if plan.latest_signature:
                    latest_path = plan.latest_signature.get("detection_source_path")
                    latest_type = plan.latest_signature.get("detection_source_type")
                    latest_roi = plan.latest_signature.get("roi_size")
                    if latest_path or latest_type:
                        print(f"  latest_source: {latest_type} ({latest_path})")
                    if latest_roi:
                        roi = _normalize_roi(latest_roi)
                        if roi:
                            print(f"  latest_roi_size: {roi[0]}x{roi[1]}")
        counts = {"ok": 0, "skipped": 0, "missing": 0}
        for plan in plans:
            counts[plan.status] = counts.get(plan.status, 0) + 1
        print("\nSummary:")
        print(f"  ok: {counts.get('ok', 0)}")
        print(f"  skipped: {counts.get('skipped', 0)}")
        print(f"  missing: {counts.get('missing', 0)}")
        print("\nUse --apply to run cropping.")
        return 0

    log_dir = _resolve_log_dir(args.log_dir, roots)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    log_path = log_dir / f"crop_batch_{run_id}.jsonl"
    logger = JsonLogger(log_path, run_id)
    print(f"Log file: {log_path}")
    logger.log("run_start", roots=[str(r) for r in roots], recursive=bool(args.recursive))

    console = Console() if Console else None
    progress = _progress(console, len(plans))
    task_id = progress.add_task("crop_batch", total=len(plans)) if progress else None

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
                source_type=plan.source_type,
                source_path=plan.source_path,
            )
        else:
            logger.log(
                "crop_start",
                zarr=str(plan.zarr_path),
                source_type=plan.source_type,
                source_path=plan.source_path,
                roi_size=list(plan.roi_size) if plan.roi_size else None,
                preferred_policy=plan.preferred_policy,
            )
            try:
                results = crop_detections(
                    zarr_path=str(plan.zarr_path),
                    config=config,
                    source_type=plan.source_type or source_type,
                    source_path=plan.source_path,
                    preferred_policy=plan.preferred_policy,
                    scheduler=args.scheduler,
                    num_workers=args.num_workers,
                    console=console,
                    acceleration=args.acceleration,
                    use_gpu_allowed=not args.no_gpu,
                    force_cpu=args.force_cpu,
                    verbose=args.verbose,
                )
            except Exception as exc:
                failed += 1
                logger.log("crop_failed", zarr=str(plan.zarr_path), error=str(exc))
            else:
                ok += 1
                logger.log(
                    "crop_success",
                    zarr=str(plan.zarr_path),
                    total_crops=results.get("total_crops"),
                    detection_source_type=results.get("detection_source_type"),
                    detection_source_path=results.get("detection_source_path"),
                )
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
