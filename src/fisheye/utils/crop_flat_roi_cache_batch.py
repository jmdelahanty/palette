"""Local batch workflow for crop geometry plus flat ROI cache materialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fisheye.cli.shared_args import add_apply_dry_run_args, add_log_args, add_registry_discovery_args
from fisheye.shared.batch_logging import JsonLogger, make_run_id
from fisheye.shared.environment import resolve_log_dir, resolve_recording_roots
from fisheye.shared.flat_roi_cache import build_flat_roi_cache
from fisheye.shared.zarr_discovery import discover_registry_zarrs
from fisheye.tracking.crop import crop_detections, infer_detection_source_type
from fisheye.utils.crop_batch import (
    CropPlan,
    _build_plans,
    _load_config,
    _load_paths_file,
    _normalize_path,
    _require_completed_crop_result,
    _resolve_targets,
)


def _safe_component(value: object) -> str:
    text = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text) or "cache"


def _resolve_roots(paths: List[Path]) -> List[Path]:
    return resolve_recording_roots(paths or None)


def _resolve_cache_output_dir(*, args: argparse.Namespace, run_id: str) -> Path:
    if args.cache_output_dir is not None:
        return args.cache_output_dir.expanduser().resolve()
    workflow_id = args.workflow_id or f"crop_flat_roi_cache_{run_id}"
    return (
        args.cache_root.expanduser().resolve()
        / _safe_component(workflow_id)
        / "roi_cache"
    )


def _progress_callback_factory(*, progress_jsonl: Optional[Path], progress_stderr: bool):
    handle = None
    if progress_jsonl is not None:
        progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
        handle = progress_jsonl.open("a", encoding="utf-8", buffering=1)

    def emit(event: Dict[str, Any]) -> None:
        if handle is not None:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
            handle.flush()
        if progress_stderr:
            print(_format_progress_event(event), file=sys.stderr, flush=True)

    return emit, handle


def _format_progress_event(event: Dict[str, Any]) -> str:
    name = event.get("event")
    if name == "start":
        return (
            "flat_roi_cache_progress=start "
            f"rows={event.get('total_rois')} "
            f"batch_size={event.get('batch_size')} "
            f"manifest={event.get('manifest_path')}"
        )
    if name == "batch":
        progress = event.get("progress") if isinstance(event.get("progress"), dict) else {}
        batch = event.get("batch") if isinstance(event.get("batch"), dict) else {}
        return (
            "flat_roi_cache_progress=batch "
            f"batch={batch.get('index')} "
            f"rows={progress.get('rows_written')}/{progress.get('rows_total')} "
            f"elapsed_s={progress.get('elapsed_seconds')}"
        )
    if name == "complete":
        progress = event.get("progress") if isinstance(event.get("progress"), dict) else {}
        return (
            "flat_roi_cache_progress=complete "
            f"rows={progress.get('rows_written')}/{progress.get('rows_total')} "
            f"manifest={event.get('manifest_path')}"
        )
    if name == "reuse_existing":
        return f"flat_roi_cache_progress=reuse_existing manifest={event.get('manifest_path')}"
    if name == "backend_fallback":
        return (
            "flat_roi_cache_progress=backend_fallback "
            f"failed={event.get('failed_backend')} fallback={event.get('fallback_backend')}"
        )
    return f"flat_roi_cache_progress={name}"


def _resolve_crop_run_after_crop(plan: CropPlan, crop_result: Optional[Dict[str, Any]]) -> Optional[str]:
    if crop_result is not None:
        run_name = crop_result.get("run_name")
        if run_name:
            return str(run_name)
        return None
    if plan.status == "skipped" and plan.latest_crop:
        return str(plan.latest_crop)
    return None


def _cache_progress_path(log_dir: Optional[Path], zarr_path: Path, crop_run: str) -> Optional[Path]:
    if log_dir is None:
        return None
    return log_dir / f"{_safe_component(zarr_path.stem)}__{_safe_component(crop_run)}.flat_roi_cache.progress.jsonl"


def _plan_cache_note(plan: CropPlan) -> str:
    if plan.status == "skipped" and plan.latest_crop:
        return f"will build/reuse flat ROI cache for existing crop run {plan.latest_crop!r}"
    if plan.status == "ok":
        return "will create/reuse crop run, then build/reuse flat ROI cache for the resulting crop"
    return "no cache build planned"


def _registry_required_steps_for_source(source_type: str) -> List[str]:
    """Return SQL prefilter steps for registry discovery.

    The crop source can be a refined-detect run even when the raw ``detect``
    step status was not synchronized into the registry.  Keep ``auto`` broad and
    let zarr planning resolve/refuse the actual source, because registry SQL
    cannot express "refined_detect OR detect" with the shared helper.
    """

    if source_type == "refined":
        return ["refined_detect"]
    if source_type == "detect":
        return ["detect"]
    return []


def _discover_registry_zarrs_for_workflow(
    *,
    registry_path: Path,
    scope_paths: List[Path],
    source_type: str,
    rig_id: Optional[str],
    arena_id: Optional[str],
    camera_id: Optional[str],
    path_contains: Optional[str],
) -> List[Path]:
    return discover_registry_zarrs(
        registry_path=registry_path,
        scope_paths=scope_paths,
        zarr_use="analysis",
        rig_id=rig_id,
        arena_id=arena_id,
        camera_id=camera_id,
        path_contains=path_contains,
        require_steps_ok=_registry_required_steps_for_source(source_type),
        zarr_suffix="_analysis.zarr",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local batch workflow that creates/reuses crop_runs and builds flat_bin_v1 "
            "ROI caches for downstream pose/segmentation jobs."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--file-list", type=Path, help="Text file with zarr paths to process.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarrs.")
    add_apply_dry_run_args(
        parser,
        apply_help="Run crop creation and cache materialization.",
        dry_run_help="Show planned crop/cache work without writing.",
    )
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Filter zarr archives by purpose (default: analysis).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail fast on the first failed recording.")
    parser.add_argument("--force-new", action="store_true", help="Always create a new crop run.")
    parser.add_argument(
        "--crop-storage-mode",
        choices=["materialized", "geometry_only"],
        default="materialized",
        help="Crop persistence mode for newly written crop runs (default: materialized).",
    )
    parser.add_argument(
        "--source-type",
        choices=["auto", "refined", "detect", "manual", "filtered", "interpolated"],
        default=None,
        help="Detection source for crop generation (default: config value, otherwise detect).",
    )
    parser.add_argument(
        "--source-path",
        default=None,
        help="Explicit detection source path, e.g. refined_detect_runs/<run>/instances.",
    )
    parser.add_argument(
        "--selection-policy",
        choices=["training", "full_recording"],
        default=None,
        help="Policy for auto source selection.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional crop config YAML.")
    parser.add_argument("--scheduler", choices=["processes", "threads", "distributed"], default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--acceleration", choices=["auto", "gpu", "cpu"], default=None)
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration for crop generation.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU crop generation.")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/tmp/palette_roi_cache"),
        help="Root for workflow cache output when --cache-output-dir is not set.",
    )
    parser.add_argument(
        "--workflow-id",
        help="Workflow namespace under --cache-root (default: crop_flat_roi_cache_<run_id>).",
    )
    parser.add_argument(
        "--cache-output-dir",
        type=Path,
        help="Explicit directory for flat ROI cache manifests and payloads.",
    )
    parser.add_argument("--cache-batch-size", type=int, default=1024, help="ROI rows per cache write batch.")
    parser.add_argument(
        "--cache-decode-backend",
        choices=["auto", "pynvvc_luma", "read_slice"],
        default="auto",
        help="Flat-cache materialization backend.",
    )
    parser.add_argument(
        "--roi-live-acceleration",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Live ROI read acceleration for geometry-only crop runs.",
    )
    parser.add_argument(
        "--roi-live-gpu-chunk-frames",
        type=int,
        default=32,
        help="Frame batch size for GPU-accelerated live ROI reads.",
    )
    parser.add_argument("--sha256", action="store_true", help="Compute and record cache payload sha256.")
    parser.add_argument("--overwrite-cache", action="store_true", help="Overwrite existing matching cache files.")
    parser.add_argument(
        "--progress-stderr",
        action="store_true",
        help="Print compact flat-cache progress updates to stderr during apply.",
    )
    parser.add_argument(
        "--progress-interval-s",
        type=float,
        default=30.0,
        help="Emit cache progress at least this often in seconds.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=0,
        help="Emit cache progress every N batches; 0 disables count-based progress.",
    )
    add_log_args(
        parser,
        log_dir_help="Directory for JSONL logs (default: <recordings_root>/logs/crop_flat_roi_cache_batch).",
    )
    add_registry_discovery_args(parser)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    explicit_roots: List[Path] = []
    if args.file_list:
        explicit_roots.extend(_load_paths_file(args.file_list))
    explicit_roots.extend(args.paths or [])
    roots = _resolve_roots(explicit_roots)

    config = _load_config(args.config)
    crop_cfg = config.get("crop", {}) or {}
    raw_source_type = args.source_type or crop_cfg.get("source_type") or "detect"
    source_path = _normalize_path(args.source_path or crop_cfg.get("source_path"))
    source_type = infer_detection_source_type(source_path, raw_source_type)
    selection_policy = args.selection_policy or crop_cfg.get("selection_policy")

    if args.source == "registry":
        if not args.registry or not args.registry.exists():
            print(f"Error: registry database not found: {args.registry}", file=sys.stderr)
            return 1
        registry_zarrs = _discover_registry_zarrs_for_workflow(
            registry_path=args.registry,
            scope_paths=roots,
            source_type=source_type,
            rig_id=args.rig_id,
            arena_id=args.arena_id,
            camera_id=args.camera_id,
            path_contains=args.path_contains,
        )
        if args.emit_paths:
            for path in registry_zarrs:
                print(path)
            return 0
        zarr_paths = registry_zarrs
    else:
        zarr_paths = _resolve_targets(roots, args.recursive)

    if not zarr_paths:
        print("No zarr files found.")
        return 1

    plans = _build_plans(
        zarr_paths=zarr_paths,
        config=config,
        source_type=source_type,
        source_path=source_path,
        selection_policy=selection_policy,
        force_new=bool(args.force_new),
        zarr_use_filter=str(args.zarr_use),
        crop_storage_mode=args.crop_storage_mode,
    )

    if not args.apply:
        run_id = make_run_id()
        cache_output_dir = _resolve_cache_output_dir(args=args, run_id=run_id)
        print("Planned crop + flat ROI cache workflow (dry-run):")
        print(f"cache_output_dir: {cache_output_dir}")
        for plan in plans:
            print(f"{plan.zarr_path}")
            print(f"  crop_status: {plan.status}")
            if plan.reason:
                print(f"  crop_reason: {plan.reason}")
            if plan.source_type:
                print(f"  source_type: {plan.source_type}")
            if plan.source_path:
                print(f"  source_path: {plan.source_path}")
            if plan.crop_storage_mode:
                print(f"  crop_storage_mode: {plan.crop_storage_mode}")
            if plan.latest_crop:
                print(f"  latest_crop: {plan.latest_crop}")
            print(f"  cache_plan: {_plan_cache_note(plan)}")
        counts: Dict[str, int] = {}
        for plan in plans:
            counts[plan.status] = counts.get(plan.status, 0) + 1
        print("\nSummary:")
        for key in ("ok", "skipped", "missing", "invalid"):
            print(f"  {key}: {counts.get(key, 0)}")
        print("\nUse --apply to run crop/cache materialization.")
        return 0

    run_id = make_run_id()
    log_dir: Optional[Path] = None
    logger: Optional[JsonLogger] = None
    if not args.no_log:
        log_dir = resolve_log_dir(args.log_dir, roots, log_subdir="crop_flat_roi_cache_batch")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"crop_flat_roi_cache_batch_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
    cache_output_dir = _resolve_cache_output_dir(args=args, run_id=run_id)
    cache_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache output dir: {cache_output_dir}")

    def log(event: str, **fields: Any) -> None:
        if logger is not None:
            logger.log(event, **fields)

    log(
        "run_start",
        roots=[str(path) for path in roots],
        zarr_count=len(zarr_paths),
        plan_count=len(plans),
        source_type=source_type,
        source_path=source_path,
        selection_policy=selection_policy,
        crop_storage_mode=args.crop_storage_mode,
        cache_output_dir=str(cache_output_dir),
        cache_decode_backend=args.cache_decode_backend,
        roi_live_acceleration=args.roi_live_acceleration,
    )

    ok = skipped = missing = invalid = crop_failed = cache_failed = 0
    for plan in plans:
        crop_result: Optional[Dict[str, Any]] = None
        crop_run = None
        try:
            if plan.status == "missing":
                missing += 1
                log("recording_missing", zarr=str(plan.zarr_path), reason=plan.reason)
                if args.strict:
                    break
                continue
            if plan.status == "invalid":
                invalid += 1
                log("recording_invalid", zarr=str(plan.zarr_path), reason=plan.reason)
                if args.strict:
                    break
                continue
            if plan.status == "skipped":
                skipped += 1
                crop_run = _resolve_crop_run_after_crop(plan, None)
                log(
                    "crop_skipped",
                    zarr=str(plan.zarr_path),
                    reason=plan.reason,
                    crop_run=crop_run,
                    source_type=plan.source_type,
                    source_path=plan.source_path,
                )
            else:
                log(
                    "crop_start",
                    zarr=str(plan.zarr_path),
                    source_type=plan.source_type,
                    source_path=plan.source_path,
                    crop_storage_mode=plan.crop_storage_mode,
                )
                crop_result = crop_detections(
                    zarr_path=str(plan.zarr_path),
                    config=config,
                    source_type=plan.source_type or source_type,
                    source_path=plan.source_path,
                    selection_policy=plan.selection_policy,
                    scheduler=args.scheduler,
                    num_workers=args.num_workers,
                    console=None,
                    acceleration=args.acceleration,
                    crop_storage_mode=plan.crop_storage_mode,
                    use_gpu_allowed=not args.no_gpu,
                    force_cpu=args.force_cpu,
                    verbose=args.verbose,
                )
                crop_run = _require_completed_crop_result(crop_result)
                log(
                    "crop_success",
                    zarr=str(plan.zarr_path),
                    crop_run=crop_run,
                    total_crops=crop_result.get("total_crops"),
                    duration_seconds=crop_result.get("duration_seconds"),
                    crop_storage_mode=crop_result.get("crop_storage_mode"),
                )

            if not crop_run:
                cache_failed += 1
                log("cache_failed", zarr=str(plan.zarr_path), reason="could not resolve crop run after crop step")
                if args.strict:
                    break
                continue

            progress_jsonl = _cache_progress_path(log_dir, plan.zarr_path, crop_run)
            emit_progress, progress_handle = _progress_callback_factory(
                progress_jsonl=progress_jsonl,
                progress_stderr=bool(args.progress_stderr),
            )
            try:
                log(
                    "cache_start",
                    zarr=str(plan.zarr_path),
                    crop_run=crop_run,
                    output_dir=str(cache_output_dir),
                    progress_jsonl=str(progress_jsonl) if progress_jsonl is not None else None,
                )
                manifest = build_flat_roi_cache(
                    zarr_path=plan.zarr_path,
                    crop_run=crop_run,
                    output_dir=cache_output_dir,
                    batch_size=args.cache_batch_size,
                    overwrite=bool(args.overwrite_cache),
                    compute_sha256=bool(args.sha256),
                    roi_live_acceleration=args.roi_live_acceleration,
                    roi_live_gpu_chunk_frames=args.roi_live_gpu_chunk_frames,
                    roi_decode_backend=args.cache_decode_backend,
                    progress_callback=emit_progress,
                    progress_interval_seconds=float(args.progress_interval_s),
                    progress_every_batches=int(args.progress_every_batches),
                )
            finally:
                if progress_handle is not None:
                    progress_handle.close()

            array = manifest.get("array") if isinstance(manifest.get("array"), dict) else {}
            builder = manifest.get("builder") if isinstance(manifest.get("builder"), dict) else {}
            log(
                "cache_success",
                zarr=str(plan.zarr_path),
                crop_run=crop_run,
                manifest_path=manifest.get("manifest_path"),
                payload_path=array.get("bin_path"),
                shape=array.get("shape"),
                decode_backend_effective=builder.get("decode_backend_effective"),
                duration_seconds=builder.get("duration_seconds"),
            )
            ok += 1
            print(
                "ok "
                f"{plan.zarr_path} crop_run={crop_run} "
                f"manifest={manifest.get('manifest_path')}"
            )
        except Exception as exc:
            if crop_run is None:
                crop_failed += 1
                log("crop_failed", zarr=str(plan.zarr_path), error=str(exc))
            else:
                cache_failed += 1
                log("cache_failed", zarr=str(plan.zarr_path), crop_run=crop_run, error=str(exc))
            print(f"failed {plan.zarr_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.strict:
                break

    failed = missing + invalid + crop_failed + cache_failed
    log(
        "run_end",
        ok=ok,
        skipped_crops=skipped,
        missing=missing,
        invalid=invalid,
        crop_failed=crop_failed,
        cache_failed=cache_failed,
        failed=failed,
    )
    if logger is not None:
        logger.close()

    print("\nSummary:")
    print(f"  ok: {ok}")
    print(f"  skipped_crops: {skipped}")
    print(f"  missing: {missing}")
    print(f"  invalid: {invalid}")
    print(f"  crop_failed: {crop_failed}")
    print(f"  cache_failed: {cache_failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
