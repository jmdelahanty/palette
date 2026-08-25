from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import zarr
import yaml

from ..cli.shared_args import add_log_args
from ..cli.shared_args import add_registry_discovery_args
from ..shared.batch_logging import JsonLogger as SharedJsonLogger
from ..shared.batch_logging import make_run_id
from ..shared.batch_logging import utc_now
from ..shared.environment import resolve_log_dir
from ..shared.environment import resolve_recording_roots
from ..shared.observation_coordinate_publication import (
    load_persisted_ordinary_crop_observation_geometry,
)
from ..shared.zarr_discovery import discover_registry_zarrs
from ..tracking.crop import (
    _infer_archive_use,
    _normalize_crop_storage_mode,
    _preflight_ordinary_crop_coordinates,
    crop_detections,
    get_detection_source_info,
    get_crop_parameters,
    get_video_source,
    infer_detection_source_type,
    ordinary_crop_geometry_policy_from_parameters,
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
    padding_mode: Optional[str] = None
    padded_row_count: Optional[int] = None
    crop_storage_mode: Optional[str] = None
    selection_policy: Optional[str] = None
    latest_crop: Optional[str] = None
    latest_pointer: Optional[str] = None
    latest_signature: Optional[Dict[str, object]] = None

_utc_now = utc_now
JsonLogger = SharedJsonLogger


def _resolve_root(paths: Optional[List[Path]]) -> List[Path]:
    return resolve_recording_roots(paths)


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: List[Path]) -> Path:
    return resolve_log_dir(arg_log_dir, roots, log_subdir="crop_batch")


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


def _require_completed_crop_result(result: Mapping[str, Any]) -> str:
    """Return the committed run name or fail a zero-row/pseudo-success result."""

    try:
        total_crops = int(result.get("total_crops", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Crop result has an invalid total_crops value.") from exc
    run_name = _normalize_str(result.get("run_name"))
    if total_crops <= 0:
        raise RuntimeError("Crop produced zero rows and no normal run was committed.")
    if run_name is None:
        raise RuntimeError("Crop reported rows but no committed run_name.")
    return run_name


def _normalize_roi(value: object) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _infer_crop_run_storage_mode(crop_group: zarr.Group) -> str:
    explicit = _normalize_str(crop_group.attrs.get("crop_storage_mode"))
    if explicit in {"materialized", "geometry_only"}:
        return explicit
    if "roi_images" in crop_group:
        return "materialized"
    return "geometry_only"


def _desired_crop_storage_mode(
    *,
    root: zarr.Group,
    zarr_path: Path,
    config: Dict[str, Any],
    crop_params: Dict[str, Any],
    explicit_crop_storage_mode: Optional[str],
) -> str:
    if explicit_crop_storage_mode is not None:
        return _normalize_crop_storage_mode(explicit_crop_storage_mode)

    crop_cfg = config.get("crop", {}) or {}
    configured = crop_cfg.get("crop_storage_mode")
    if configured is not None:
        return _normalize_crop_storage_mode(configured)

    return _normalize_crop_storage_mode(crop_params.get("crop_storage_mode", "materialized"))


def _latest_crop_signature(
    root: zarr.Group,
    *,
    crop_storage_mode: str,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, object]]]:
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        return None, None, None
    latest_pointer = "latest_any" if crop_storage_mode == "geometry_only" else "latest_materialized"
    latest = crop_parent.attrs.get(latest_pointer)
    if not latest and crop_storage_mode == "materialized":
        latest_pointer = "latest"
        latest = crop_parent.attrs.get("latest")
    elif not latest and crop_storage_mode == "geometry_only":
        latest_pointer = "latest"
        latest = crop_parent.attrs.get("latest")
    if not latest or latest not in crop_parent:
        return None, latest_pointer, None
    crop_group = crop_parent[latest]
    signature = {
        "detection_source_path": crop_group.attrs.get("detection_source_path"),
        "detection_source_type": crop_group.attrs.get("detection_source_type"),
        "detection_selection_policy": crop_group.attrs.get(
            "detection_selection_policy",
            crop_group.attrs.get("detection_preferred_policy"),
        ),
        "roi_size": crop_group.attrs.get("roi_size"),
        "crop_geometry_policy_digest": crop_group.attrs.get(
            "crop_geometry_policy_digest"
        ),
        "crop_storage_mode": _infer_crop_run_storage_mode(crop_group),
        "status": crop_group.attrs.get("status"),
    }
    return str(latest), latest_pointer, signature


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
    if desired.get("crop_geometry_policy_digest") != existing.get(
        "crop_geometry_policy_digest"
    ):
        diffs.append("crop_geometry_policy")
    desired_storage_mode = _normalize_str(desired.get("crop_storage_mode"))
    existing_storage_mode = _normalize_str(existing.get("crop_storage_mode"))
    if desired_storage_mode != existing_storage_mode:
        diffs.append("crop_storage_mode")
    if compare_policy:
        desired_policy = _normalize_str(desired.get("detection_selection_policy"))
        existing_policy = _normalize_str(existing.get("detection_selection_policy"))
        if desired_policy != existing_policy:
            diffs.append("selection_policy")
    return diffs


def _build_plan(
    zarr_path: Path,
    config: Dict[str, Any],
    source_type: str,
    source_path: Optional[str],
    selection_policy: Optional[str],
    force_new: bool,
    crop_storage_mode: Optional[str],
) -> CropPlan:
    if not zarr_path.exists():
        return CropPlan(zarr_path=zarr_path, status="missing", reason="zarr not found")

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    try:
        resolved_path, _source_group, _detection_source, resolved_type = get_detection_source_info(
            root=root,
            source_type=source_type,
            source_path_override=source_path,
            console=None,
            selection_policy=selection_policy,
        )
    except ValueError as exc:
        return CropPlan(
            zarr_path=zarr_path,
            status="missing",
            reason=str(exc),
        )

    crop_params, _ = get_crop_parameters(root, config, console=None)
    crop_policy = ordinary_crop_geometry_policy_from_parameters(crop_params)
    roi_width, roi_height = crop_policy.fixed_size_wh or (0, 0)
    roi_size = (int(roi_height), int(roi_width))
    desired_crop_storage_mode = _desired_crop_storage_mode(
        root=root,
        zarr_path=zarr_path,
        config=config,
        crop_params=crop_params,
        explicit_crop_storage_mode=crop_storage_mode,
    )
    if desired_crop_storage_mode != "materialized":
        return CropPlan(
            zarr_path=zarr_path,
            status="invalid",
            reason=(
                "the materialized-crop publication profile requires "
                "crop_storage_mode=materialized; geometry-only sealed crops use "
                "their dedicated publication profile"
            ),
            source_type=resolved_type,
            source_path=resolved_path,
            roi_size=roi_size,
            crop_storage_mode=desired_crop_storage_mode,
            selection_policy=selection_policy,
        )
    source_parts = str(resolved_path).strip().strip("/").split("/")
    if (
        resolved_type != "detect"
        or len(source_parts) != 2
        or source_parts[0] != "detect_runs"
    ):
        return CropPlan(
            zarr_path=zarr_path,
            status="invalid",
            reason=(
                "the materialized-crop publication profile requires an exact "
                "detect_runs/<run> source; refined and sparse sources require a "
                "profile with an explicit row-selection contract"
            ),
            source_type=resolved_type,
            source_path=resolved_path,
            roi_size=roi_size,
            crop_storage_mode=desired_crop_storage_mode,
            selection_policy=selection_policy,
        )

    try:
        video_source_type, video_path = get_video_source(root, console=None)
        canonical_preflight = _preflight_ordinary_crop_coordinates(
            root,
            zarr_path=str(zarr_path),
            source_path=resolved_path,
            source_group=_source_group,
            video_source_type=video_source_type,
            video_path=video_path,
            policy=crop_policy,
        )
    except Exception as exc:
        return CropPlan(
            zarr_path=zarr_path,
            status="invalid",
            reason=(
                "ordinary crop canonical preflight failed "
                f"({type(exc).__name__}: {exc})"
            ),
            source_type=resolved_type,
            source_path=resolved_path,
            roi_size=roi_size,
            crop_storage_mode=desired_crop_storage_mode,
            selection_policy=selection_policy,
        )
    if canonical_preflight.row_count == 0:
        return CropPlan(
            zarr_path=zarr_path,
            status="missing",
            reason="validated canonical detection source contains zero rows",
            source_type=resolved_type,
            source_path=resolved_path,
            roi_size=roi_size,
            crop_storage_mode=desired_crop_storage_mode,
            selection_policy=selection_policy,
        )

    desired = {
        "detection_source_path": resolved_path,
        "detection_source_type": resolved_type,
        "detection_selection_policy": selection_policy,
        "roi_size": roi_size,
        "crop_geometry_policy_digest": crop_policy.payload_digest,
        "crop_storage_mode": desired_crop_storage_mode,
    }

    latest_crop, latest_pointer, latest_signature = _latest_crop_signature(
        root,
        crop_storage_mode=desired_crop_storage_mode,
    )
    status = "ok"
    reason = None

    if latest_signature:
        latest_status = _normalize_str(latest_signature.get("status"))
        if latest_status and latest_status.lower() != "completed":
            reason = f"latest crop run '{latest_crop}' not completed"
        else:
            latest_group = root["crop_runs"][latest_crop]
            if (
                latest_group.attrs.get("coordinate_contract") != "canonical_v2"
                or latest_group.attrs.get("stage_selector_eligible") is not True
            ):
                reason = (
                    f"latest crop run '{latest_crop}' is not an eligible canonical_v2 "
                    "run; a new canonical run is required"
                )
            else:
                try:
                    load_persisted_ordinary_crop_observation_geometry(
                        root,
                        f"crop_runs/{latest_crop}",
                    )
                except Exception as exc:
                    reason = (
                        f"latest crop run '{latest_crop}' failed canonical validation "
                        f"({type(exc).__name__}: {exc}); a new canonical run is required"
                    )
                else:
                    diffs = _diff_signature(
                        desired,
                        latest_signature,
                        compare_policy=selection_policy is not None,
                    )
                    if not diffs and not force_new:
                        status = "skipped"
                        reason = f"matches validated canonical crop run '{latest_crop}'"
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
        padding_mode=crop_policy.padding_mode.value,
        padded_row_count=(
            int(canonical_preflight.padded_row_count)
            if getattr(canonical_preflight, "padded_row_count", None) is not None
            else None
        ),
        crop_storage_mode=desired_crop_storage_mode,
        selection_policy=selection_policy,
        latest_crop=latest_crop,
        latest_pointer=latest_pointer,
        latest_signature=latest_signature,
    )


def _build_plans(
    zarr_paths: List[Path],
    config: Dict[str, Any],
    source_type: str,
    source_path: Optional[str],
    selection_policy: Optional[str],
    force_new: bool,
    zarr_use_filter: str,
    crop_storage_mode: Optional[str],
) -> List[CropPlan]:
    plans: List[CropPlan] = []
    for zarr_path in zarr_paths:
        if zarr_use_filter != "any":
            try:
                root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
            except Exception as exc:
                plans.append(
                    CropPlan(
                        zarr_path=zarr_path,
                        status="missing",
                        reason=f"zarr open failed: {exc}",
                    )
                )
                continue
            observed_use = _infer_archive_use(root, zarr_path)
            if observed_use != zarr_use_filter:
                plans.append(
                    CropPlan(
                        zarr_path=zarr_path,
                        status="skipped",
                        reason=f"zarr_use mismatch (wanted={zarr_use_filter}, found={observed_use or 'unknown'})",
                    )
                )
                continue

        plans.append(
            _build_plan(
                zarr_path=zarr_path,
                config=config,
                source_type=source_type,
                source_path=source_path,
                selection_policy=selection_policy,
                force_new=force_new,
                crop_storage_mode=crop_storage_mode,
            )
        )
    return plans


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


def _discover_zarrs_from_registry(
    *,
    registry_path: Path,
    scope_paths: Sequence[Path],
    rig_id: Optional[str] = None,
    arena_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    path_contains: Optional[str] = None,
    skip_existing: bool = False,
) -> List[Path]:
    """Query the registry for analysis zarr paths suitable for crop processing.

    When *skip_existing* is True, recordings whose ``crop`` step status is
    already ``'ok'`` are excluded at the SQL level.  Only recordings whose
    ``detect`` step is ``'ok'`` are returned, ensuring the detection
    prerequisite is met.
    """
    return discover_registry_zarrs(
        registry_path=registry_path,
        scope_paths=scope_paths,
        zarr_use="analysis",
        rig_id=rig_id,
        arena_id=arena_id,
        camera_id=camera_id,
        path_contains=path_contains,
        require_steps_ok=["detect"],
        exclude_step_ok="crop" if skip_existing else None,
        zarr_suffix="_analysis.zarr",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch crop ROIs for Palette Zarr recordings.")
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--file-list", type=Path, help="Text file with zarr paths to process.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarrs.")
    parser.add_argument("--apply", action="store_true", help="Run cropping (default is dry-run).")
    parser.add_argument(
        "--fail-on-invalid-plan",
        action="store_true",
        help="In dry-run mode, return nonzero when any plan is missing or invalid.",
    )
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter zarr archives by purpose (default: any).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on first missing/error recording during apply.",
    )
    parser.add_argument("--force-new", action="store_true", help="Always create a new crop run.")
    parser.add_argument(
        "--crop-storage-mode",
        choices=["materialized", "geometry_only"],
        default=None,
        help=(
            "Crop persistence mode for this materialized-crop publisher. "
            "Geometry-only sealed crops use their dedicated publisher."
        ),
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default=None,
        choices=["auto", "refined", "detect", "manual", "filtered", "interpolated"],
        help=(
            "Detection source (default: config value, otherwise detect). The "
            "materialized-crop profile requires an exact detect_runs/<run> source; "
            "auto/refined/sparse sources require an explicit row-selection profile."
        ),
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help=(
            "Explicit canonical detection source path: detect_runs/<run>."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        type=str,
        default=None,
        choices=["training", "full_recording"],
        help="Policy for auto source selection.",
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
    parser.add_argument(
        "--external-write-backend",
        choices=["standard", "kvikio"],
        default=None,
        help="Write backend for external-video crop runs.",
    )
    parser.add_argument(
        "--external-roi-storage",
        choices=["compressed", "uncompressed"],
        default=None,
        help="ROI storage mode for external-video crop runs.",
    )
    parser.add_argument(
        "--external-use-sharding",
        action="store_true",
        help="Enable sharding for external-video ROI writes.",
    )
    parser.add_argument(
        "--no-external-use-sharding",
        action="store_true",
        help="Disable sharding for external-video ROI writes.",
    )
    parser.add_argument(
        "--external-roi-chunk-size",
        type=int,
        default=None,
        help="Detection-axis chunk length for external-video ROI writes.",
    )
    parser.add_argument(
        "--external-roi-shard-size",
        type=int,
        default=None,
        help="Detection-axis shard length for external-video ROI writes.",
    )
    parser.add_argument(
        "--external-gpu-chunk-frames",
        type=int,
        default=None,
        help="Frame count decoded per GPU crop chunk (external-video mode).",
    )
    parser.add_argument(
        "--require-kvikio",
        action="store_true",
        help="Fail if kvikIO GDS writes cannot be enabled in external-video mode.",
    )
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help=(
            "Write exact per-recording outcomes, including the committed or "
            "validated reused crop_run, for dependent workflow jobs."
        ),
    )
    add_log_args(
        parser,
        log_dir_help="Directory for JSONL logs.",
        include_no_log=False,
    )
    # --- Registry discovery mode ---
    add_registry_discovery_args(parser)

    args = parser.parse_args(argv)

    explicit_roots: List[Path] = []
    if args.file_list:
        explicit_roots.extend(_load_paths_file(args.file_list))
    explicit_roots.extend(args.paths or [])
    if not explicit_roots:
        explicit_roots = _resolve_root(None)

    roots = _resolve_root(explicit_roots)

    config = _load_config(args.config)
    crop_cfg = config.get("crop", {}) or {}
    raw_source_type = args.source_type or crop_cfg.get("source_type") or "detect"
    source_path = _normalize_path(args.source_path or crop_cfg.get("source_path"))
    source_type = infer_detection_source_type(source_path, raw_source_type)
    selection_policy = args.selection_policy or crop_cfg.get("selection_policy")
    external_use_sharding = None
    if args.external_use_sharding and args.no_external_use_sharding:
        raise SystemExit("Choose either --external-use-sharding or --no-external-use-sharding, not both.")
    if args.external_use_sharding:
        external_use_sharding = True
    elif args.no_external_use_sharding:
        external_use_sharding = False

    if args.source == "registry":
        if not args.registry or not args.registry.exists():
            print(
                f"Error: registry database not found: {args.registry}",
                file=sys.stderr,
            )
            return 1
        registry_zarrs = _discover_zarrs_from_registry(
            registry_path=args.registry,
            scope_paths=roots,
            rig_id=args.rig_id,
            arena_id=args.arena_id,
            camera_id=args.camera_id,
            path_contains=args.path_contains,
            # Registry crop=ok may describe a legacy or ambiguous crop. Always
            # discover detect-ready archives and validate Zarr evidence.
            skip_existing=False,
        )
        if args.emit_paths:
            for p in registry_zarrs:
                print(p)
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
            if plan.padding_mode:
                print(f"  padding_mode: {plan.padding_mode}")
            if plan.padded_row_count is not None:
                print(f"  padded_roi_count: {plan.padded_row_count}")
            if plan.crop_storage_mode:
                print(f"  crop_storage_mode: {plan.crop_storage_mode}")
            if plan.latest_crop:
                print(f"  latest_crop: {plan.latest_crop}")
                if plan.latest_pointer:
                    print(f"  latest_pointer: {plan.latest_pointer}")
                if plan.latest_signature:
                    latest_path = plan.latest_signature.get("detection_source_path")
                    latest_type = plan.latest_signature.get("detection_source_type")
                    latest_roi = plan.latest_signature.get("roi_size")
                    latest_storage_mode = plan.latest_signature.get("crop_storage_mode")
                    if latest_path or latest_type:
                        print(f"  latest_source: {latest_type} ({latest_path})")
                    if latest_roi:
                        roi = _normalize_roi(latest_roi)
                        if roi:
                            print(f"  latest_roi_size: {roi[0]}x{roi[1]}")
                    if latest_storage_mode:
                        print(f"  latest_crop_storage_mode: {latest_storage_mode}")
        counts = {"ok": 0, "skipped": 0, "missing": 0, "invalid": 0}
        for plan in plans:
            counts[plan.status] = counts.get(plan.status, 0) + 1
        print("\nSummary:")
        print(f"  ok: {counts.get('ok', 0)}")
        print(f"  skipped: {counts.get('skipped', 0)}")
        print(f"  missing: {counts.get('missing', 0)}")
        print(f"  invalid: {counts.get('invalid', 0)}")
        print("\nUse --apply to run cropping.")
        if args.fail_on_invalid_plan and (
            counts.get("missing", 0) > 0 or counts.get("invalid", 0) > 0
        ):
            return 1
        return 0

    log_dir = _resolve_log_dir(args.log_dir, roots)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    log_path = log_dir / f"crop_batch_{run_id}.jsonl"
    logger = JsonLogger(log_path, run_id)
    print(f"Log file: {log_path}")
    logger.log(
        "run_start",
        roots=[str(r) for r in roots],
        recursive=bool(args.recursive),
        zarr_use=str(args.zarr_use),
        strict=bool(args.strict),
        external_write_backend=args.external_write_backend,
        external_roi_storage=args.external_roi_storage,
        crop_storage_mode=args.crop_storage_mode,
        require_kvikio=bool(args.require_kvikio),
    )

    console = Console() if Console else None
    progress = _progress(console, len(plans))
    task_id = progress.add_task("crop_batch", total=len(plans)) if progress else None

    ok = skipped = missing = invalid = failed = 0
    outcomes: List[Dict[str, Any]] = []
    strict_abort = False
    for plan in plans:
        if plan.status == "missing":
            missing += 1
            outcomes.append(
                {
                    "zarr_path": str(plan.zarr_path),
                    "status": "missing",
                    "reason": plan.reason,
                    "crop_run": None,
                }
            )
            logger.log("recording_missing", zarr=str(plan.zarr_path), reason=plan.reason)
            if args.strict:
                failed += 1
                strict_abort = True
                logger.log("run_abort_strict", zarr=str(plan.zarr_path), reason=plan.reason, status=plan.status)
        elif plan.status == "invalid":
            invalid += 1
            failed += 1
            outcomes.append(
                {
                    "zarr_path": str(plan.zarr_path),
                    "status": "invalid",
                    "reason": plan.reason,
                    "crop_run": None,
                }
            )
            logger.log("recording_invalid", zarr=str(plan.zarr_path), reason=plan.reason)
            if args.strict:
                strict_abort = True
                logger.log("run_abort_strict", zarr=str(plan.zarr_path), reason=plan.reason, status=plan.status)
        elif plan.status == "skipped":
            skipped += 1
            outcomes.append(
                {
                    "zarr_path": str(plan.zarr_path),
                    "status": "skipped",
                    "reason": plan.reason,
                    "crop_run": plan.latest_crop,
                }
            )
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
                padding_mode=plan.padding_mode,
                padded_roi_count=plan.padded_row_count,
                selection_policy=plan.selection_policy,
                crop_storage_mode=plan.crop_storage_mode,
            )
            try:
                results = crop_detections(
                    zarr_path=str(plan.zarr_path),
                    config=config,
                    source_type=plan.source_type or source_type,
                    source_path=plan.source_path,
                    selection_policy=plan.selection_policy,
                    scheduler=args.scheduler,
                    num_workers=args.num_workers,
                    console=console,
                    acceleration=args.acceleration,
                    external_write_backend=args.external_write_backend,
                    external_roi_storage=args.external_roi_storage,
                    external_use_sharding=external_use_sharding,
                    external_roi_chunk_size=args.external_roi_chunk_size,
                    external_roi_shard_size=args.external_roi_shard_size,
                    external_gpu_chunk_frames=args.external_gpu_chunk_frames,
                    external_require_kvikio=args.require_kvikio,
                    crop_storage_mode=plan.crop_storage_mode,
                    use_gpu_allowed=not args.no_gpu,
                    force_cpu=args.force_cpu,
                    verbose=args.verbose,
                )
                committed_run = _require_completed_crop_result(results)
            except Exception as exc:
                failed += 1
                outcomes.append(
                    {
                        "zarr_path": str(plan.zarr_path),
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "crop_run": None,
                    }
                )
                logger.log("crop_failed", zarr=str(plan.zarr_path), error=str(exc))
                if args.strict:
                    strict_abort = True
                    logger.log("run_abort_strict", zarr=str(plan.zarr_path), reason=str(exc), status="failed")
            else:
                ok += 1
                outcomes.append(
                    {
                        "zarr_path": str(plan.zarr_path),
                        "status": "ok",
                        "reason": None,
                        "crop_run": committed_run,
                    }
                )
                logger.log(
                    "crop_success",
                    zarr=str(plan.zarr_path),
                    run_name=committed_run,
                    total_crops=results.get("total_crops"),
                    detection_source_type=results.get("detection_source_type"),
                    detection_source_path=results.get("detection_source_path"),
                    crop_storage_mode=results.get("crop_storage_mode"),
                )
        if progress:
            progress.advance(task_id)
        if strict_abort:
            break

    if progress:
        progress.stop()

    logger.log("run_end", ok=ok, skipped=skipped, missing=missing, invalid=invalid, failed=failed)
    logger.close()

    if args.result_json is not None:
        result_path = args.result_json.expanduser()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "palette.crop_batch_result.v1",
            "outcomes": outcomes,
            "summary": {
                "ok": ok,
                "skipped": skipped,
                "missing": missing,
                "invalid": invalid,
                "failed": failed,
            },
        }
        temporary = result_path.with_name(result_path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(result_path)

    print("\nSummary:")
    print(f"  ok: {ok}")
    print(f"  skipped: {skipped}")
    print(f"  missing: {missing}")
    print(f"  invalid: {invalid}")
    print(f"  failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
