"""Batch-run subject-mask inference and smart finalization on recording Zarrs.

The utility is intentionally conservative:

* default mode is dry-run;
* existing subject-mask and refined-subject-mask runs are not overwritten;
* smoke archives are excluded by default;
* each archive is processed serially while the underlying stages retain their
  own provenance capture.
"""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now_date as _utc_now_compact
from fisheye.shared.json_safety import write_json_atomic
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.mask_store import MaskStoreError, open_mask_store
from fisheye.shared.run_provenance import (
    CLI_RUN_PROVENANCE_ATTR,
    RUN_PROVENANCE_ATTR,
    append_input_artifacts,
    build_run_provenance,
)
from fisheye.shared.subject_mask_registry_status import (
    emit_refined_subject_mask_stage_completion,
    emit_subject_mask_stage_completion,
)
from fisheye.shared.subject_mask_chunks import DEFAULT_MASK_PROBS_SHARD_ROIS
from fisheye.shared.workflow_profile import WorkflowProfiler
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    mark_run_complete,
    require_runs_parent,
)
from fisheye.shared.zarr_discovery import discover_registry_zarr_entries


RAW_COMPONENTS = ("subject_body", "eyes_union", "swim_bladder")
REFINED_COMPONENTS = ("subject_body", "eye_left", "eye_right", "swim_bladder")
SUBJECT_MASK_CANONICAL_OUTPUT_PARENT = "subject_mask_runs"
SUBJECT_MASK_SHARD_OUTPUT_PARENT = "subject_mask_shard_runs"
SUBJECT_MASK_OUTPUT_PARENTS = (SUBJECT_MASK_CANONICAL_OUTPUT_PARENT, SUBJECT_MASK_SHARD_OUTPUT_PARENT)
OUTPUT_RUN_PARENTS = (*SUBJECT_MASK_OUTPUT_PARENTS, "refined_subject_masks_runs")
MAX_ARTIFACT_FILENAME_CHARS = 220
DEFAULT_FINALIZE_DENSE_MASK_ROW_CHUNK = 128
@dataclass(frozen=True)
class ArchivePlan:
    zarr_path: str
    subject_run: str
    refined_run: str
    crop_run: Optional[str]
    assignment_keypoint_group: Optional[str]
    assignment_keypoint_run: Optional[str]
    has_subject_runs: bool
    has_refined_subject_runs: bool
    run_inference: bool
    run_finalization: bool
    skip_reason: str = ""
    subject_output_parent: str = SUBJECT_MASK_CANONICAL_OUTPUT_PARENT


@dataclass
class ArchiveResult:
    zarr_path: str
    subject_run: str
    refined_run: str
    planned_inference: bool
    planned_finalization: bool
    inference_status: str = "not_requested"
    finalization_status: str = "not_requested"
    validation_status: str = "not_run"
    output_staging_status: str = "not_requested"
    staged_zarr_path: str = ""
    publish_status: str = "not_requested"
    registry_refresh_status: str = "not_requested"
    subject_mask_performance_rows: Optional[int] = None
    subject_mask_component_quality_rows: Optional[int] = None
    workflow_profile_path: str = ""
    workflow_profile: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class OutputStagingContext:
    source_zarr_path: Path
    staged_zarr_path: Path
    staging_root: Path


@dataclass(frozen=True)
class RunGroupPublishPlan:
    source_path: Path
    tmp_path: Path
    target_path: Path
    overwrite: bool
    publish_backend: str
    storage_stats: Mapping[str, Any]
    copy_duration_seconds: float


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


_ZARR_METADATA_FILENAMES = frozenset({"zarr.json", ".zarray", ".zattrs", ".zgroup"})


def _empty_storage_counter() -> dict[str, int]:
    return {
        "file_count": 0,
        "metadata_file_count": 0,
        "payload_file_count": 0,
        "apparent_bytes": 0,
        "allocated_bytes": 0,
    }


def _accumulate_storage_file(counter: dict[str, int], *, filename: str, stat_result: os.stat_result) -> None:
    counter["file_count"] += 1
    if filename in _ZARR_METADATA_FILENAMES:
        counter["metadata_file_count"] += 1
    else:
        counter["payload_file_count"] += 1
    counter["apparent_bytes"] += int(stat_result.st_size)
    counter["allocated_bytes"] += int(getattr(stat_result, "st_blocks", 0)) * 512


def _run_group_storage_stats(run_group_path: Path) -> dict[str, Any]:
    """Inspect one materialized run group without opening it through zarr.

    The scan records filesystem-object pressure as well as the logical array
    layouts responsible for it.  It deliberately does not follow symlinked
    directories: staged output run groups are expected to be self-contained,
    and following an unexpected link could scan unrelated archive content.
    """

    path = run_group_path.expanduser()
    started = time.perf_counter()
    totals = _empty_storage_counter()
    top_level: dict[str, dict[str, int]] = {}
    arrays: dict[str, dict[str, Any]] = {}
    array_ancestor_by_directory: dict[Path, str | None] = {}
    stat_error_count = 0
    metadata_error_count = 0

    if not path.is_dir():
        raise FileNotFoundError(f"Run group not found for storage scan: {path}")

    for root_text, directory_names, filenames in os.walk(path, topdown=True, followlinks=False):
        root = Path(root_text)
        relative_directory = root.relative_to(path)
        relative_directory_text = relative_directory.as_posix() if relative_directory.parts else "."
        inherited_array = array_ancestor_by_directory.get(root.parent)
        current_array = inherited_array

        if "zarr.json" in filenames:
            metadata = _read_json(root / "zarr.json")
            if metadata is None:
                metadata_error_count += 1
            elif metadata.get("node_type") == "array":
                current_array = relative_directory_text
                chunk_grid = metadata.get("chunk_grid")
                chunk_configuration = (
                    chunk_grid.get("configuration")
                    if isinstance(chunk_grid, Mapping)
                    else None
                )
                arrays[current_array] = {
                    "path": current_array,
                    "shape": list(metadata.get("shape") or []),
                    "data_type": metadata.get("data_type"),
                    "chunk_shape": list(
                        chunk_configuration.get("chunk_shape") or []
                    )
                    if isinstance(chunk_configuration, Mapping)
                    else [],
                    "codecs": list(metadata.get("codecs") or []),
                    "chunk_key_encoding": metadata.get("chunk_key_encoding"),
                    **_empty_storage_counter(),
                }

        array_ancestor_by_directory[root] = current_array
        for directory_name in directory_names:
            array_ancestor_by_directory[root / directory_name] = current_array

        for filename in filenames:
            file_path = root / filename
            try:
                stat_result = file_path.lstat()
            except OSError:
                stat_error_count += 1
                continue

            _accumulate_storage_file(totals, filename=filename, stat_result=stat_result)
            relative_file = file_path.relative_to(path)
            top_key = relative_file.parts[0] if len(relative_file.parts) > 1 else "__root__"
            top_counter = top_level.setdefault(top_key, _empty_storage_counter())
            _accumulate_storage_file(top_counter, filename=filename, stat_result=stat_result)

            if current_array is not None and current_array in arrays:
                _accumulate_storage_file(
                    arrays[current_array],
                    filename=filename,
                    stat_result=stat_result,
                )

    sorted_arrays = [arrays[key] for key in sorted(arrays)]
    return {
        "schema": "palette_run_group_storage_stats_v1",
        "run_group_path": str(path),
        **totals,
        "array_count": int(len(sorted_arrays)),
        "arrays": sorted_arrays,
        "top_level": {key: top_level[key] for key in sorted(top_level)},
        "stat_error_count": int(stat_error_count),
        "metadata_error_count": int(metadata_error_count),
        "scan_duration_seconds": float(time.perf_counter() - started),
    }


def _latest_group_name(parent_path: Path) -> Optional[str]:
    attrs = _attrs(parent_path)
    for key in ("latest", "latest_materialized"):
        value = attrs.get(key)
        if isinstance(value, str) and value and (parent_path / value).is_dir():
            return value
    children = _child_groups(parent_path)
    return children[-1] if children else None


def _latest_crop_group_name(parent_path: Path) -> Optional[str]:
    attrs = _attrs(parent_path)
    for key in ("latest_any", "latest", "latest_materialized"):
        value = attrs.get(key)
        if isinstance(value, str) and value and (parent_path / value).is_dir():
            return value
    children = _child_groups(parent_path)
    return children[-1] if children else None


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


def _discover_registry_analysis_zarrs(
    roots: Sequence[Path],
    *,
    registry_path: Path,
    rig_id: Optional[str],
    arena_id: Optional[str],
    camera_id: Optional[str],
    path_contains: Optional[str],
    exclude_refined_subject_masks_ok: bool,
) -> list[Path]:
    entries = discover_registry_zarr_entries(
        registry_path=registry_path,
        scope_paths=roots,
        zarr_use="analysis",
        rig_id=rig_id,
        arena_id=arena_id,
        camera_id=camera_id,
        path_contains=path_contains,
        require_steps_ok=("crop", "keypoints"),
        exclude_step_ok="refined_subject_masks" if exclude_refined_subject_masks_ok else None,
        zarr_suffix="_analysis.zarr",
    )
    return [entry.zarr_path for entry in entries]


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


def _resolve_crop_run(zarr_path: Path) -> Optional[str]:
    return _latest_crop_group_name(zarr_path / "crop_runs")


def _resolve_assignment_keypoints(zarr_path: Path) -> tuple[Optional[str], Optional[str]]:
    refined_parent = zarr_path / "refined_keypoints_runs"
    refined_latest = _latest_group_name(refined_parent)
    if refined_latest:
        return "refined_keypoints_runs", refined_latest
    keypoint_parent = zarr_path / "keypoints_runs"
    keypoint_latest = _latest_group_name(keypoint_parent)
    if keypoint_latest:
        return "keypoints_runs", keypoint_latest
    return None, None


def build_archive_plan(
    zarr_path: Path,
    *,
    subject_run_name: str,
    refined_run_name: str,
    force_inference: bool,
    force_finalization: bool,
    workflow_stage: str = "all",
    subject_output_parent: str = SUBJECT_MASK_CANONICAL_OUTPUT_PARENT,
    crop_run_name: str | None = None,
    resolve_assignment_keypoints: bool = True,
) -> ArchivePlan:
    if workflow_stage not in {"all", "inference", "finalization"}:
        raise ValueError(f"workflow_stage must be all, inference, or finalization; got {workflow_stage!r}.")
    if subject_output_parent not in SUBJECT_MASK_OUTPUT_PARENTS:
        raise ValueError(
            f"subject_output_parent must be one of {SUBJECT_MASK_OUTPUT_PARENTS}; got {subject_output_parent!r}."
        )
    subject_parent = zarr_path / subject_output_parent
    refined_parent = zarr_path / "refined_subject_masks_runs"
    subject_children = _child_groups(subject_parent)
    refined_children = _child_groups(refined_parent)
    if crop_run_name is None:
        crop_run = _resolve_crop_run(zarr_path)
    else:
        requested_crop_run = str(crop_run_name).strip()
        crop_run = (
            requested_crop_run
            if requested_crop_run and (zarr_path / "crop_runs" / requested_crop_run).is_dir()
            else None
        )
    if resolve_assignment_keypoints:
        keypoint_group, keypoint_run = _resolve_assignment_keypoints(zarr_path)
    else:
        keypoint_group, keypoint_run = None, None

    has_subject_runs = bool(subject_children)
    has_refined_subject_runs = bool(refined_children)
    target_subject_run_exists = subject_run_name in subject_children
    target_refined_run_exists = refined_run_name in refined_children
    selected_subject_run = subject_run_name
    selected_subject_run_exists = target_subject_run_exists
    if workflow_stage == "finalization" and not target_subject_run_exists:
        latest_subject_run = _latest_group_name(subject_parent)
        if latest_subject_run is not None:
            selected_subject_run = latest_subject_run
            selected_subject_run_exists = True
    if workflow_stage == "inference":
        run_inference = bool(force_inference or not target_subject_run_exists)
        run_finalization = False
    elif workflow_stage == "finalization":
        run_inference = False
        run_finalization = bool(force_finalization or not target_refined_run_exists)
    else:
        run_inference = bool(force_inference or not has_subject_runs)
        run_finalization = bool(force_finalization or not has_refined_subject_runs)

    skip_reasons: list[str] = []
    if crop_run is None:
        run_inference = False
        run_finalization = False
        skip_reasons.append("missing_crop_run")
    if (keypoint_group is None or keypoint_run is None) and workflow_stage != "inference":
        run_inference = False
        run_finalization = False
        skip_reasons.append("missing_keypoint_assignment_source")
    if workflow_stage == "inference" and target_subject_run_exists and not force_inference:
        skip_reasons.append(f"target_{subject_output_parent}_run_present")
    elif workflow_stage == "all" and has_subject_runs and not force_inference:
        skip_reasons.append(f"{subject_output_parent}_present")
    if workflow_stage == "finalization" and target_refined_run_exists and not force_finalization:
        skip_reasons.append("target_refined_subject_masks_run_present")
    elif workflow_stage == "all" and has_refined_subject_runs and not force_finalization:
        skip_reasons.append("refined_subject_masks_runs_present")
    if workflow_stage == "finalization" and run_finalization and not selected_subject_run_exists:
        run_finalization = False
        skip_reasons.append("target_subject_mask_run_missing")
    if run_finalization and not run_inference and not has_subject_runs:
        run_finalization = False
        skip_reasons.append("cannot_finalize_without_subject_mask_run")

    return ArchivePlan(
        zarr_path=str(zarr_path),
        subject_run=selected_subject_run,
        refined_run=refined_run_name,
        subject_output_parent=subject_output_parent,
        crop_run=crop_run,
        assignment_keypoint_group=keypoint_group,
        assignment_keypoint_run=keypoint_run,
        has_subject_runs=has_subject_runs,
        has_refined_subject_runs=has_refined_subject_runs,
        run_inference=run_inference,
        run_finalization=run_finalization,
        skip_reason=", ".join(skip_reasons),
    )


def _selected_subject_run_for_finalization(plan: ArchivePlan) -> str:
    zarr_path = Path(plan.zarr_path)
    if plan.subject_output_parent != SUBJECT_MASK_CANONICAL_OUTPUT_PARENT:
        raise RuntimeError(
            "Refined subject-mask finalization requires canonical subject_mask_runs input; "
            f"got {plan.subject_output_parent!r}."
        )
    if plan.run_inference or (zarr_path / "subject_mask_runs" / plan.subject_run).is_dir():
        return plan.subject_run
    latest = _latest_group_name(zarr_path / "subject_mask_runs")
    if latest is None:
        raise RuntimeError(f"{zarr_path} has no subject_mask_runs to finalize.")
    return latest


def _inference_command(
    args: argparse.Namespace,
    plan: ArchivePlan,
    *,
    defer_registry_status: bool = False,
    roi_cache_expected_archive_path: str | Path | None = None,
) -> list[str]:
    subject_output_parent = getattr(args, "subject_output_parent", SUBJECT_MASK_CANONICAL_OUTPUT_PARENT)
    cmd = [
        sys.executable,
        "-m",
        "fisheye.segmentation.infer_unet_subject_masks",
        plan.zarr_path,
        "--resolve-model-from-registry",
        "--registry",
        str(args.registry),
        "--model-coverage-class",
        args.model_coverage_class,
        "--model-component-coverage-key",
        args.model_component_coverage_key,
        "--model-label-schema-id",
        args.model_label_schema_id,
        "--model-top-k",
        str(args.model_top_k),
        "--run-name",
        plan.subject_run,
        "--output-parent",
        subject_output_parent,
        "--crop-run",
        str(plan.crop_run),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--mask-probs-dtype",
        args.mask_probs_dtype,
        "--mask-probs-chunk-rois",
        str(args.mask_probs_chunk_rois),
        "--no-write-masks-roi",
        "--async-output",
        "--output-queue-size",
        str(args.output_queue_size),
        "--no-progress",
        "--roi-cache-policy",
        args.roi_cache_policy,
        "--roi-live-acceleration",
        args.roi_live_acceleration,
        "--roi-live-gpu-chunk-frames",
        str(args.roi_live_gpu_chunk_frames),
    ]
    if plan.assignment_keypoint_group is not None and plan.assignment_keypoint_run is not None:
        cmd.extend(
            [
                "--assignment-keypoint-group",
                str(plan.assignment_keypoint_group),
                "--assignment-keypoint-run",
                str(plan.assignment_keypoint_run),
            ]
        )
    mask_probs_shard_rois = getattr(args, "mask_probs_shard_rois", None)
    if mask_probs_shard_rois is not None:
        cmd.extend(["--mask-probs-shard-rois", str(int(mask_probs_shard_rois))])
    else:
        cmd.append("--no-mask-probs-sharding")
    if args.model_require_unique:
        cmd.append("--model-require-unique")
    if args.model_include_non_success:
        cmd.append("--model-include-non-success")
    if args.roi_cache_dir is not None:
        cmd.extend(["--roi-cache-dir", str(args.roi_cache_dir)])
    if args.roi_cache_manifest is not None:
        cmd.extend(["--roi-cache-manifest", str(args.roi_cache_manifest)])
    if roi_cache_expected_archive_path is not None:
        cmd.extend(["--roi-cache-expected-archive-path", str(roi_cache_expected_archive_path)])
    source_roi_cache_alias_manifest = getattr(args, "source_roi_cache_alias_manifest", None)
    if (
        source_roi_cache_alias_manifest is None
        and subject_output_parent == SUBJECT_MASK_SHARD_OUTPUT_PARENT
        and args.roi_cache_manifest is not None
    ):
        source_roi_cache_alias_manifest = args.roi_cache_manifest
    if source_roi_cache_alias_manifest is not None:
        cmd.extend(["--source-roi-cache-alias-manifest", str(source_roi_cache_alias_manifest)])
    source_roi_cache_row_index_path = getattr(args, "source_roi_cache_row_index_path", None)
    if source_roi_cache_row_index_path is not None:
        cmd.extend(["--source-roi-cache-row-index-path", str(source_roi_cache_row_index_path)])
    for attr_name, flag in (
        ("source_collection_id", "--source-collection-id"),
        ("source_collection_path", "--source-collection-path"),
        ("source_clip_id", "--source-clip-id"),
        ("source_clip_index", "--source-clip-index"),
        ("source_work_unit_id", "--source-work-unit-id"),
        ("source_shard_id", "--source-shard-id"),
    ):
        value = getattr(args, attr_name, None)
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.profile_timings:
        cmd.append("--profile-timings")
    if defer_registry_status:
        cmd.append("--defer-registry-status")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _finalization_command(
    args: argparse.Namespace,
    plan: ArchivePlan,
    *,
    defer_registry_status: bool = False,
) -> list[str]:
    subject_run = _selected_subject_run_for_finalization(plan)
    cmd = [
        sys.executable,
        "-m",
        "fisheye.refinement.finalize_subject_masks",
        plan.zarr_path,
        "--subject-run",
        subject_run,
        "--run-name",
        plan.refined_run,
        "--components",
        "subject_body",
        "eyes_union",
        "swim_bladder",
        "--chunk-size",
        str(args.finalize_chunk_size),
        "--metric-level",
        args.metric_level,
        "--mask-storage",
        args.mask_storage,
        "--mask-rle-validation-mode",
        args.mask_rle_validation_mode,
        "--execution-backend",
        args.finalize_execution_backend,
        "--assignment-keypoint-group",
        str(plan.assignment_keypoint_group),
        "--assignment-keypoints-run",
        str(plan.assignment_keypoint_run),
        "--json",
    ]
    if args.finalize_num_workers is not None:
        cmd.extend(["--num-workers", str(args.finalize_num_workers)])
    if args.finalize_dense_mask_row_chunk is not None:
        cmd.extend(["--dense-mask-row-chunk", str(args.finalize_dense_mask_row_chunk)])
    if args.write_eye_geometry:
        cmd.append("--write-eye-geometry")
    if args.write_component_contours:
        cmd.append("--write-component-contours")
    if args.write_sampled_component_contours:
        cmd.append("--write-sampled-component-contours")
        cmd.extend(["--sampled-contour-row-chunk", str(args.sampled_contour_row_chunk)])
        for value in args.sampled_contour_k:
            cmd.extend(["--sampled-contour-k", str(value)])
    if args.retain_source_seeds:
        cmd.append("--retain-source-seeds")
    cmd.extend(["--postcompute-backend", args.finalize_postcompute_backend])
    if args.finalize_postcompute_chunk_size is not None:
        cmd.extend(["--postcompute-chunk-size", str(args.finalize_postcompute_chunk_size)])
    if args.finalize_postcompute_num_workers is not None:
        cmd.extend(["--postcompute-num-workers", str(args.finalize_postcompute_num_workers)])
    if args.progress_dir is not None:
        progress_filename = _safe_artifact_filename(
            (Path(plan.zarr_path).stem, plan.refined_run),
            ".finalization.progress.jsonl",
        )
        progress_path = args.progress_dir.expanduser().resolve() / progress_filename
        cmd.extend(["--progress-jsonl", str(progress_path)])
    if defer_registry_status:
        cmd.append("--defer-registry-status")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _run_command(cmd: Sequence[str], *, dry_run: bool) -> str:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return "planned"
    completed = subprocess.run(list(cmd), check=False)
    return "ok" if completed.returncode == 0 else f"failed_exit_{completed.returncode}"


def _safe_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value)) or "item"


def _safe_artifact_filename(
    components: Sequence[str],
    suffix: str,
    *,
    max_chars: int = MAX_ARTIFACT_FILENAME_CHARS,
) -> str:
    safe_components = [_safe_path_component(component) for component in components if str(component)]
    stem = "__".join(safe_components) or "artifact"
    filename = f"{stem}{suffix}"
    if len(filename) <= max_chars:
        return filename

    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
    prefix_budget = max(16, max_chars - len(suffix) - len(digest) - 2)
    prefix = stem[:prefix_budget].rstrip("._-") or "artifact"
    return f"{prefix}__{digest}{suffix}"


def _workflow_profile_path(args: argparse.Namespace, plan: ArchivePlan) -> Path | None:
    profile_dir = args.workflow_profile_dir or args.progress_dir
    if profile_dir is None:
        return None
    run_name = plan.subject_run if plan.run_inference else plan.refined_run
    filename = _safe_artifact_filename(
        (Path(plan.zarr_path).stem, str(args.workflow_stage), run_name),
        ".workflow.profile.jsonl",
    )
    return profile_dir.expanduser().resolve() / filename


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _subject_mask_package_path(base_dir: Path, zarr_path: Path, run_name: str) -> Path:
    filename = _safe_artifact_filename((zarr_path.stem, run_name), ".tar")
    return base_dir.expanduser().resolve() / filename


def _create_run_group_tar_package(source_run_path: Path, package_path: Path) -> dict[str, Any]:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = package_path.with_name(f"{package_path.name}.tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()
    started = datetime.now(timezone.utc)
    with tarfile.open(tmp_path, mode="w") as tar:
        tar.add(source_run_path, arcname=source_run_path.name, recursive=True)
    os.replace(tmp_path, package_path)
    stat = package_path.stat()
    finished = datetime.now(timezone.utc)
    return {
        "schema": "palette_subject_mask_run_package_v1",
        "format": "tar",
        "temporary_handoff": True,
        "source_run": source_run_path.name,
        "artifact_path": str(package_path),
        "size_bytes": int(stat.st_size),
        "created_at_utc": finished.isoformat(),
        "duration_seconds": float((finished - started).total_seconds()),
    }


def _extract_run_group_tar_package(package_path: Path, target_parent: Path, expected_run_name: str) -> Path:
    target_parent.mkdir(parents=True, exist_ok=True)
    target_run_path = target_parent / expected_run_name
    if target_run_path.exists() or target_run_path.is_symlink():
        _remove_path(target_run_path)
    with tarfile.open(package_path, mode="r") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe tar member path in {package_path}: {member.name!r}")
            if not member_path.parts or member_path.parts[0] != expected_run_name:
                raise ValueError(
                    f"Unexpected tar member path in {package_path}: {member.name!r}; "
                    f"expected top-level {expected_run_name!r}."
                )
        tar.extractall(target_parent)
    if not target_run_path.is_dir():
        raise FileNotFoundError(f"Package did not extract expected run group: {target_run_path}")
    return target_run_path


def _default_output_staging_root() -> Path:
    user = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("LSB_JOBID") or f"manual_{os.getpid()}"
    job_index = os.environ.get("LSB_JOBINDEX")
    user_scratch = Path(f"/scratch/{user}")
    if user_scratch.is_dir() and os.access(user_scratch, os.W_OK | os.X_OK):
        base = user_scratch / str(job_id)
    else:
        base = Path(os.environ.get("TMPDIR") or "/tmp") / "palette" / str(user) / str(job_id)
    if job_index:
        base = base / f"array_{job_index}"
    return base / "subject_mask_output_staging"


def _open_group_mutable(path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode="a", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode="a")


def _copy_attrs_from_path(source_group_path: Path, target_group: zarr.Group) -> None:
    attrs = _attrs(source_group_path)
    if attrs:
        target_group.attrs.update(attrs)


def _prepare_output_staging_zarr(
    source_zarr_path: Path,
    *,
    plan: ArchivePlan,
    staging_root: Path,
    overwrite: bool,
    stage_finalization_input: bool = False,
) -> OutputStagingContext:
    source_zarr_path = source_zarr_path.expanduser().resolve()
    staged_filename = _safe_artifact_filename((source_zarr_path.stem, plan.subject_run), ".zarr")
    staged_zarr_path = staging_root.expanduser().resolve() / staged_filename
    if staged_zarr_path.exists() or staged_zarr_path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Staged output zarr already exists: {staged_zarr_path}")
        _remove_path(staged_zarr_path)
    staged_zarr_path.parent.mkdir(parents=True, exist_ok=True)

    local_root = zarr.open_group(str(staged_zarr_path), mode="w")
    _copy_attrs_from_path(source_zarr_path, local_root)
    for parent_name in OUTPUT_RUN_PARENTS:
        parent = local_root.require_group(parent_name)
        _copy_attrs_from_path(source_zarr_path / parent_name, parent)

    for child in source_zarr_path.iterdir():
        if child.name in OUTPUT_RUN_PARENTS or child.name == "zarr.json":
            continue
        target = staged_zarr_path / child.name
        if target.exists() or target.is_symlink():
            continue
        os.symlink(child, target, target_is_directory=child.is_dir())

    if plan.run_finalization and not plan.run_inference:
        subject_run = _selected_subject_run_for_finalization(plan)
        source_subject = source_zarr_path / "subject_mask_runs" / subject_run
        staged_subject = staged_zarr_path / "subject_mask_runs" / subject_run
        if not source_subject.is_dir():
            raise FileNotFoundError(f"Cannot stage finalization source subject-mask run: {source_subject}")
        if not staged_subject.exists() and not staged_subject.is_symlink():
            if stage_finalization_input:
                package = _attrs(source_subject).get("cluster_run_package")
                package_path = None
                if isinstance(package, dict):
                    raw_package_path = package.get("artifact_path")
                    if isinstance(raw_package_path, str) and raw_package_path:
                        package_path = Path(raw_package_path)
                if package_path is not None and package_path.is_file():
                    _extract_run_group_tar_package(package_path, staged_zarr_path / "subject_mask_runs", subject_run)
                else:
                    shutil.copytree(source_subject, staged_subject, symlinks=True)
            else:
                os.symlink(source_subject, staged_subject, target_is_directory=True)

    return OutputStagingContext(
        source_zarr_path=source_zarr_path,
        staged_zarr_path=staged_zarr_path,
        staging_root=staging_root,
    )


def _prepare_run_group_publish(
    *,
    staged_parent: Path,
    target_parent: Path,
    run_name: str,
    overwrite: bool,
) -> RunGroupPublishPlan:
    source = staged_parent / run_name
    if not source.is_dir():
        raise FileNotFoundError(f"Staged run group not found: {source}")
    target_parent.mkdir(parents=True, exist_ok=True)
    target = target_parent / run_name
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Target run group already exists: {target}")
        _remove_path(target)
    tmp = target_parent / f".{run_name}.publish_tmp.{os.getpid()}"
    if tmp.exists() or tmp.is_symlink():
        _remove_path(tmp)
    storage_stats = _run_group_storage_stats(source)
    copy_started = time.perf_counter()
    shutil.copytree(source, tmp, symlinks=True)
    return RunGroupPublishPlan(
        source_path=source,
        tmp_path=tmp,
        target_path=target,
        overwrite=overwrite,
        publish_backend="shutil.copytree",
        storage_stats=storage_stats,
        copy_duration_seconds=float(time.perf_counter() - copy_started),
    )


def _commit_run_group_publish(plan: RunGroupPublishPlan) -> None:
    if plan.target_path.exists() or plan.target_path.is_symlink():
        if not plan.overwrite:
            raise FileExistsError(f"Target run group already exists: {plan.target_path}")
        _remove_path(plan.target_path)
    os.replace(plan.tmp_path, plan.target_path)


def _staging_publish_payload(ctx: OutputStagingContext) -> dict[str, Any]:
    return {
        "schema": "palette_subject_mask_output_staging_v1",
        "policy": "node_local_write_publish_to_prfs",
        "source_zarr_path": str(ctx.source_zarr_path),
        "staged_zarr_path": str(ctx.staged_zarr_path),
        "staging_root": str(ctx.staging_root),
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "lsb_jobid": os.environ.get("LSB_JOBID"),
        "lsb_jobindex": os.environ.get("LSB_JOBINDEX"),
    }


def _subject_mask_publish_provenance(
    *,
    ctx: OutputStagingContext,
    plan: ArchivePlan,
    publish_payload: dict[str, Any],
    refined: bool,
) -> dict[str, Any]:
    input_run_ids: dict[str, Any] = {}
    if plan.crop_run:
        input_run_ids["crop"] = plan.crop_run
    if plan.assignment_keypoint_run:
        input_run_ids["assignment_keypoints"] = plan.assignment_keypoint_run
    if refined:
        input_run_ids["subject_mask"] = plan.subject_run

    return build_run_provenance(
        command="fisheye.utils.run_subject_mask_batch_pipeline",
        params={
            "zarr_path": plan.zarr_path,
            "subject_run": plan.subject_run,
            "refined_run": plan.refined_run,
            "crop_run": plan.crop_run,
            "assignment_keypoint_group": plan.assignment_keypoint_group,
            "assignment_keypoint_run": plan.assignment_keypoint_run,
            "run_inference": plan.run_inference,
            "run_finalization": plan.run_finalization,
            "subject_output_parent": plan.subject_output_parent,
            "source_zarr_path": str(ctx.source_zarr_path),
            "staged_zarr_path": str(ctx.staged_zarr_path),
            "staging_root": str(ctx.staging_root),
            "publish": publish_payload,
            "run_family": "refined_subject_masks_runs" if refined else plan.subject_output_parent,
        },
        input_run_ids=input_run_ids,
        cwd=Path.cwd(),
    )


def _existing_run_input_artifacts(run_group: Any) -> list[Mapping[str, Any]]:
    attrs = getattr(run_group, "attrs", {})
    for attr_name in (RUN_PROVENANCE_ATTR, CLI_RUN_PROVENANCE_ATTR):
        provenance = attrs.get(attr_name)
        if not isinstance(provenance, Mapping):
            continue
        artifacts = provenance.get("input_artifacts")
        if isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes, bytearray)):
            return [item for item in artifacts if isinstance(item, Mapping)]
    return []


def _refresh_subject_mask_registry_views(
    *,
    registry_path: Optional[Path],
    zarr_path: Path,
) -> dict[str, Any]:
    if registry_path is None:
        return {"registry_refresh_status": "skipped", "reason": "no_registry"}

    registry = Registry(registry_path)
    try:
        input_zarr_path = zarr_path.expanduser()
        zarr_path = input_zarr_path.resolve()
        query_paths = sorted({str(input_zarr_path), str(zarr_path)})
        if len(query_paths) == 1:
            query_paths.append(query_paths[0])
        row = registry.conn.execute(
            """
            SELECT dataset_id, recording_id, zarr_use
            FROM datasets
            WHERE zarr_path IN (?, ?)
            ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
            LIMIT 1;
            """,
            (query_paths[0], query_paths[1]),
        ).fetchone()
        if row is None:
            return {
                "registry_refresh_status": "skipped",
                "reason": "dataset_not_in_registry",
                "registry_path": str(registry_path),
                "zarr_path": str(zarr_path),
            }
        dataset_id = str(row["dataset_id"])
        performance_rows = int(
            registry.refresh_subject_mask_performance_for_dataset(
                dataset_id,
                zarr_path=zarr_path,
                recording_id=row["recording_id"],
                zarr_use=row["zarr_use"],
            )
        )
        component_quality_rows = int(
            registry.refresh_subject_mask_component_quality_for_dataset(
                dataset_id,
                zarr_path=zarr_path,
                recording_id=row["recording_id"],
                zarr_use=row["zarr_use"],
            )
        )
        return {
            "registry_refresh_status": "ok",
            "registry_path": str(registry_path),
            "dataset_id": dataset_id,
            "subject_mask_performance_rows": performance_rows,
            "subject_mask_component_quality_rows": component_quality_rows,
        }
    finally:
        registry.close()


def _publish_staged_outputs(
    ctx: OutputStagingContext,
    *,
    plan: ArchivePlan,
    overwrite: bool,
    handoff_package_dir: Optional[Path] = None,
    registry_path: Optional[Path] = None,
) -> dict[str, Any]:
    root = _open_group_mutable(ctx.source_zarr_path)
    if plan.run_inference:
        require_runs_parent(
            root,
            plan.subject_output_parent,
            completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
        )
    if plan.run_finalization:
        require_runs_parent(
            root,
            "refined_subject_masks_runs",
            completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
        )

    publish_plans: list[RunGroupPublishPlan] = []
    if plan.run_inference:
        publish_plans.append(_prepare_run_group_publish(
            staged_parent=ctx.staged_zarr_path / plan.subject_output_parent,
            target_parent=ctx.source_zarr_path / plan.subject_output_parent,
            run_name=plan.subject_run,
            overwrite=overwrite,
        ))
    if plan.run_finalization:
        publish_plans.append(_prepare_run_group_publish(
            staged_parent=ctx.staged_zarr_path / "refined_subject_masks_runs",
            target_parent=ctx.source_zarr_path / "refined_subject_masks_runs",
            run_name=plan.refined_run,
            overwrite=overwrite,
        ))
    published_run_groups: list[dict[str, Any]] = []
    for publish_plan in publish_plans:
        commit_started = time.perf_counter()
        _commit_run_group_publish(publish_plan)
        published_run_groups.append(
            {
                "run_group_path": str(publish_plan.target_path),
                "source_run_group_path": str(publish_plan.source_path),
                "run_name": str(publish_plan.target_path.name),
                "parent": str(publish_plan.target_path.parent.name),
                "publish_backend": str(publish_plan.publish_backend),
                "storage_stats": dict(publish_plan.storage_stats),
                "copy_duration_seconds": float(publish_plan.copy_duration_seconds),
                "commit_duration_seconds": float(time.perf_counter() - commit_started),
            }
        )

    root = _open_group_mutable(ctx.source_zarr_path)
    publish_payload = _staging_publish_payload(ctx)
    handoff_packages: list[dict[str, Any]] = []
    if plan.run_inference:
        subject_parent = require_runs_parent(
            root,
            plan.subject_output_parent,
            completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
        )
        subject_group = subject_parent[plan.subject_run]
        subject_group.attrs["cluster_output_staging"] = dict(publish_payload)
        subject_run_provenance = _subject_mask_publish_provenance(
            ctx=ctx,
            plan=plan,
            publish_payload=publish_payload,
            refined=False,
        )
        subject_run_provenance = append_input_artifacts(
            subject_run_provenance,
            _existing_run_input_artifacts(subject_group),
        )
        subject_group.attrs[RUN_PROVENANCE_ATTR] = dict(subject_run_provenance)
        subject_group.attrs[CLI_RUN_PROVENANCE_ATTR] = dict(subject_run_provenance)
        mark_run_complete(
            subject_group,
            parent_group=subject_parent,
            run_name=plan.subject_run,
            run_provenance=subject_run_provenance,
        )
        if handoff_package_dir is not None and plan.subject_output_parent == SUBJECT_MASK_CANONICAL_OUTPUT_PARENT:
            package_payload = _create_run_group_tar_package(
                ctx.source_zarr_path / "subject_mask_runs" / plan.subject_run,
                _subject_mask_package_path(handoff_package_dir, ctx.source_zarr_path, plan.subject_run),
            )
            subject_group.attrs["cluster_run_package"] = dict(package_payload)
            handoff_packages.append(dict(package_payload))
        if plan.subject_output_parent == SUBJECT_MASK_CANONICAL_OUTPUT_PARENT:
            if not emit_subject_mask_stage_completion(
                root,
                ctx.source_zarr_path,
                run_group=subject_group,
                run_name=plan.subject_run,
                source="runtime_subject_mask_write_local_publish",
                invalidate_on_ok=True,
            ):
                raise RuntimeError(f"Failed to emit registry status for subject_mask_runs/{plan.subject_run}")
    if plan.run_finalization:
        refined_parent = require_runs_parent(
            root,
            "refined_subject_masks_runs",
            completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
        )
        refined_group = refined_parent[plan.refined_run]
        refined_group.attrs["cluster_output_staging"] = dict(publish_payload)
        refined_run_provenance = _subject_mask_publish_provenance(
            ctx=ctx,
            plan=plan,
            publish_payload=publish_payload,
            refined=True,
        )
        refined_run_provenance = append_input_artifacts(
            refined_run_provenance,
            _existing_run_input_artifacts(refined_group),
        )
        refined_group.attrs[RUN_PROVENANCE_ATTR] = dict(refined_run_provenance)
        refined_group.attrs[CLI_RUN_PROVENANCE_ATTR] = dict(refined_run_provenance)
        mark_run_complete(
            refined_group,
            parent_group=refined_parent,
            run_name=plan.refined_run,
            run_provenance=refined_run_provenance,
        )
        refined_parent.attrs["refined_subject_mask_review_status_latest"] = plan.refined_run
        if not emit_refined_subject_mask_stage_completion(
            root,
            ctx.source_zarr_path,
            run_group=refined_group,
            run_name=plan.refined_run,
            source="runtime_refined_subject_mask_write_local_publish",
            invalidate_on_ok=True,
        ):
            raise RuntimeError(
                f"Failed to emit registry status for refined_subject_masks_runs/{plan.refined_run}"
            )
    if plan.subject_output_parent == SUBJECT_MASK_CANONICAL_OUTPUT_PARENT:
        registry_refresh = _refresh_subject_mask_registry_views(
            registry_path=registry_path,
            zarr_path=ctx.source_zarr_path,
        )
    else:
        registry_refresh = {
            "registry_refresh_status": "skipped",
            "reason": "noncanonical_subject_output_parent",
            "subject_output_parent": plan.subject_output_parent,
        }
    storage_stats = [
        dict(item.get("storage_stats") or {})
        for item in published_run_groups
    ]
    return {
        "published_run_groups": published_run_groups,
        "published_run_group_count": int(len(published_run_groups)),
        "publish_backend": (
            str(published_run_groups[0]["publish_backend"])
            if len(published_run_groups) == 1
            else "multiple"
        ),
        "publish_file_count": int(sum(int(item.get("file_count") or 0) for item in storage_stats)),
        "publish_apparent_bytes": int(
            sum(int(item.get("apparent_bytes") or 0) for item in storage_stats)
        ),
        "publish_allocated_bytes": int(
            sum(int(item.get("allocated_bytes") or 0) for item in storage_stats)
        ),
        "publish_storage_scan_duration_seconds": float(
            sum(float(item.get("scan_duration_seconds") or 0.0) for item in storage_stats)
        ),
        "publish_copy_duration_seconds": float(
            sum(float(item.get("copy_duration_seconds") or 0.0) for item in published_run_groups)
        ),
        "publish_commit_duration_seconds": float(
            sum(float(item.get("commit_duration_seconds") or 0.0) for item in published_run_groups)
        ),
        "handoff_packages": handoff_packages,
        "handoff_package_count": int(len(handoff_packages)),
        **registry_refresh,
    }


def _cleanup_output_staging(ctx: OutputStagingContext) -> None:
    if ctx.staged_zarr_path.exists() or ctx.staged_zarr_path.is_symlink():
        _remove_path(ctx.staged_zarr_path)
    try:
        ctx.staged_zarr_path.parent.rmdir()
    except OSError:
        pass


def _consolidate_metadata_quietly(zarr_path: str | Path) -> None:
    """Refresh consolidated metadata without leaking known sidecar warnings to stderr."""

    consolidate_metadata_capture_expected_warnings(zarr_path)


def _open_group(path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode="r")


def validate_outputs(
    zarr_path: Path,
    *,
    subject_run: str,
    refined_run: str,
    subject_output_parent: str = SUBJECT_MASK_CANONICAL_OUTPUT_PARENT,
    require_subject: bool = True,
    require_refined: bool = True,
) -> tuple[str, str]:
    root = _open_group(zarr_path)
    details: list[str] = []
    subject_parent = root.get(subject_output_parent)
    if subject_parent is None or subject_run not in subject_parent:
        if require_subject:
            return "failed", f"missing {subject_output_parent}/{subject_run}"
    else:
        subject = subject_parent[subject_run]
        raw_labels = tuple(str(label) for label in subject.attrs.get("mask_labels", ()))
        if any(label not in raw_labels for label in RAW_COMPONENTS):
            return "failed", f"subject mask labels {raw_labels!r} missing {RAW_COMPONENTS!r}"
        if "mask_probs_roi" not in subject:
            return "failed", f"{subject_output_parent}/{subject_run} missing mask_probs_roi"
        details.append(f"subject_mask_labels={raw_labels}")

    refined_parent = root.get("refined_subject_masks_runs")
    if refined_parent is None or refined_run not in refined_parent:
        if require_refined:
            return "failed", f"missing refined_subject_masks_runs/{refined_run}"
    else:
        refined = refined_parent[refined_run]
        refined_labels = tuple(str(label) for label in refined.attrs.get("mask_labels", ()))
        if any(label not in refined_labels for label in REFINED_COMPONENTS):
            return "failed", f"refined mask labels {refined_labels!r} missing {REFINED_COMPONENTS!r}"
        try:
            mask_store = open_mask_store(
                refined,
                source_path=f"{zarr_path}/refined_subject_masks_runs/{refined_run}",
                prefer="dense",
            )
        except (MaskStoreError, ValueError) as exc:
            return "failed", f"refined_subject_masks_runs/{refined_run} missing usable mask store: {exc}"
        if any(label not in mask_store.mask_labels for label in REFINED_COMPONENTS):
            return "failed", f"refined mask store labels {mask_store.mask_labels!r} missing {REFINED_COMPONENTS!r}"
        for component in REFINED_COMPONENTS:
            if f"components/{component}" not in refined:
                return "failed", f"refined_subject_masks_runs/{refined_run} missing components/{component}"
        details.append(f"refined_mask_labels={refined_labels}; refined_mask_store={mask_store.encoding}")
    return "ok", "; ".join(details)


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
        "# Subject Mask Batch Pipeline Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Planned archives: {len(plans)}",
        f"- Results: {len(results)}",
        f"- Inference requested: {sum(1 for plan in plans if plan.run_inference)}",
        f"- Finalization requested: {sum(1 for plan in plans if plan.run_finalization)}",
        f"- Validation ok: {sum(1 for result in results if result.validation_status == 'ok')}",
        f"- Errors: {sum(1 for result in results if result.error)}",
        "",
        "## Results",
        "",
        "| Zarr | Inference | Finalization | Validation | Publish | Output Staging | Error |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |".format(
                Path(result.zarr_path).name,
                result.inference_status,
                result.finalization_status,
                result.validation_status,
                result.publish_status,
                result.output_staging_status,
                (result.error or "").replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("/groups/johnson/johnsonlab/jeremy/recordings")])
    parser.add_argument("--source", choices=("filesystem", "registry"), default="filesystem")
    parser.add_argument("--emit-paths", action="store_true", help="Print selected zarr paths and exit.")
    parser.add_argument("--rig-id", help="Filter by rig_id in registry mode.")
    parser.add_argument("--arena-id", help="Filter by arena_id in registry mode.")
    parser.add_argument("--camera-id-filter", help="Filter by camera_id in registry mode.")
    parser.add_argument("--path-contains", help="Filter zarr_path by substring in registry mode.")
    parser.add_argument(
        "--roots-from-report",
        type=Path,
        help="Use unique zarr_path entries from a previous JSON report instead of positional roots.",
    )
    parser.add_argument("--apply", action="store_true", help="Run commands. Default is dry-run planning.")
    parser.add_argument("--pilot-size", type=int, default=None, help="Limit to the first N eligible archives.")
    parser.add_argument("--include-smoke", action="store_true", help="Include /smoke/ analysis Zarrs.")
    parser.add_argument("--run-label", default=f"batch_{_utc_now_compact()}")
    parser.add_argument(
        "--workflow-stage",
        choices=("all", "inference", "finalization"),
        default="all",
        help="Run both stages, only subject-mask inference, or only refined-subject finalization.",
    )
    parser.add_argument("--registry", type=Path, default=None, help="Registry SQLite path. Defaults to PALETTE_REGISTRY_PATH/config.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--crop-run",
        help="Explicit crop_runs child to use for inference instead of the archive's latest crop run.",
    )
    parser.add_argument(
        "--no-assignment-keypoints",
        action="store_true",
        help="Do not attach an assignment-keypoint run to inference-only raw subject masks.",
    )
    parser.add_argument(
        "--subject-output-parent",
        choices=SUBJECT_MASK_OUTPUT_PARENTS,
        default=SUBJECT_MASK_CANONICAL_OUTPUT_PARENT,
        help=(
            "Parent group for raw subject-mask inference output. "
            "Use subject_mask_shard_runs only for inference-only collection shards; "
            "refined finalization currently requires canonical subject_mask_runs."
        ),
    )
    parser.add_argument("--roi-cache-policy", choices=("never", "auto", "always"), default="auto")
    parser.add_argument("--roi-cache-dir", type=Path)
    parser.add_argument("--roi-cache-manifest", type=Path)
    parser.add_argument(
        "--source-roi-cache-alias-manifest",
        type=Path,
        help=(
            "Durable source flat-cache alias manifest to record in shard provenance. "
            "Defaults to --roi-cache-manifest for subject_mask_shard_runs."
        ),
    )
    parser.add_argument(
        "--source-roi-cache-row-index-path",
        type=Path,
        help="Optional durable flat-cache row-index path to record in shard provenance.",
    )
    parser.add_argument("--source-collection-id", help="Collection id for shard provenance.")
    parser.add_argument("--source-collection-path", help="Collection path for shard provenance.")
    parser.add_argument("--source-clip-id", help="Clip id for shard provenance.")
    parser.add_argument("--source-clip-index", type=int, help="Clip index for shard provenance.")
    parser.add_argument("--source-work-unit-id", help="Scheduler/work-unit id for shard provenance.")
    parser.add_argument("--source-shard-id", help="Stable shard id for shard provenance.")
    parser.add_argument("--roi-live-acceleration", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--roi-live-gpu-chunk-frames", type=int, default=32)
    parser.add_argument("--mask-probs-dtype", choices=("uint8", "float16"), default="uint8")
    parser.add_argument("--mask-probs-chunk-rois", type=int, default=32)
    probability_storage = parser.add_mutually_exclusive_group()
    probability_storage.add_argument(
        "--mask-probs-shard-rois",
        type=int,
        default=DEFAULT_MASK_PROBS_SHARD_ROIS,
        help="Outer shard rows for mask_probs_roi (default: 2048).",
    )
    probability_storage.add_argument(
        "--no-mask-probs-sharding",
        dest="mask_probs_shard_rois",
        action="store_const",
        const=None,
        help="Use ordinary probability chunks instead of the default indexed-sharded layout.",
    )
    parser.add_argument("--output-queue-size", type=int, default=2)
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Pass --profile-timings to subject-mask inference and persist timing_profile attrs.",
    )
    parser.add_argument("--model-coverage-class", default="dense_all_components")
    parser.add_argument("--model-component-coverage-key", default="body+eyes+swim_bladder")
    parser.add_argument("--model-label-schema-id", default="subject_v1_union")
    parser.add_argument("--model-top-k", type=int, default=5)
    parser.add_argument("--model-require-unique", action="store_true")
    parser.add_argument("--model-include-non-success", action="store_true")
    parser.add_argument("--metric-level", choices=("cheap", "full"), default="cheap")
    parser.add_argument(
        "--mask-storage",
        choices=(
            "dense_uint8",
            "dense_and_bitpacked",
            "dense_and_rle",
            "dense_bitpacked_and_rle",
        ),
        default="dense_uint8",
        help=(
            "Refined subject-mask physical storage passed to finalize_subject_masks. "
            "dense_uint8 preserves the historical dense masks_roi surface; dense_and_bitpacked "
            "writes dense plus derived compact bitpacked masks; "
            "dense_and_rle writes dense plus derived compact component RLE; "
            "dense_bitpacked_and_rle writes all three surfaces for validation/audit runs."
        ),
    )
    parser.add_argument(
        "--mask-rle-validation-mode",
        choices=("full", "invariants", "none"),
        default="invariants",
        help=(
            "Compact mask_rle validation policy passed to finalize_subject_masks. Batch/cluster "
            "production defaults to invariants so compact stores are structurally validated without "
            "decoding the entire dense logical mask surface."
        ),
    )
    parser.add_argument("--finalize-chunk-size", type=int, default=64)
    parser.add_argument(
        "--finalize-dense-mask-row-chunk",
        type=int,
        default=DEFAULT_FINALIZE_DENSE_MASK_ROW_CHUNK,
        help=(
            "Rows per physical Zarr chunk for dense refined masks_roi "
            f"(default: {DEFAULT_FINALIZE_DENSE_MASK_ROW_CHUNK}, current batch production candidate). "
            "Pass 16 explicitly for the historical small-chunk dense layout."
        ),
    )
    parser.add_argument(
        "--finalize-execution-backend",
        choices=("serial_driver", "process_shards"),
        default="process_shards",
    )
    parser.add_argument("--finalize-num-workers", type=int, default=8)
    parser.add_argument(
        "--finalize-postcompute-backend",
        choices=("serial", "process_shards"),
        default="process_shards",
        help=(
            "Backend for expensive finalizer postcompute artifacts. process_shards is the batch-workflow "
            "default for row-sharded eye geometry and component contour materialization; serial preserves "
            "the historical in-process path."
        ),
    )
    parser.add_argument(
        "--finalize-postcompute-chunk-size",
        type=int,
        help="Rows per postcompute shard. Defaults to --finalize-chunk-size inside the finalizer.",
    )
    parser.add_argument(
        "--finalize-postcompute-num-workers",
        type=int,
        help="Worker count for process-sharded postcompute. Defaults to --finalize-num-workers inside the finalizer.",
    )
    parser.add_argument("--write-eye-geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--write-component-contours",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write full ragged component contours (default: true).",
    )
    parser.add_argument(
        "--write-sampled-component-contours",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write fixed-K sampled display contours for Crimson (default: true).",
    )
    parser.add_argument(
        "--sampled-contour-row-chunk",
        type=int,
        default=1024,
        help="Physical row chunk for fixed-K sampled contour arrays (default: 1024).",
    )
    parser.add_argument(
        "--sampled-contour-k",
        action="append",
        default=[],
        metavar="COMPONENT=K",
        help="Override one sampled contour K; repeat per component.",
    )
    parser.add_argument(
        "--retain-source-seeds",
        action="store_true",
        help=(
            "Ask refined subject-mask finalization to retain dense source_seed_masks_roi debug arrays. "
            "Defaults off for production storage/write performance."
        ),
    )
    parser.add_argument("--force-inference", action="store_true", help="Run inference even if subject_mask_runs already exists.")
    parser.add_argument("--force-finalization", action="store_true", help="Run finalization even if refined_subject_masks_runs already exists.")
    parser.add_argument("--overwrite", action="store_true", help="Pass overwrite through to child stages.")
    parser.add_argument(
        "--stage-output-to-scratch",
        action="store_true",
        help=(
            "Write subject-mask and refined-subject run groups to a local staged "
            "zarr, then publish completed run groups to the canonical archive."
        ),
    )
    parser.add_argument(
        "--stage-finalization-input-to-scratch",
        action="store_true",
        help=(
            "For finalization-only jobs with output staging, copy the source "
            "subject-mask run into the local staged zarr instead of symlinking it."
        ),
    )
    parser.add_argument(
        "--handoff-package-dir",
        type=Path,
        help=(
            "Shared non-backed-up directory, typically on NRS, for temporary "
            "subject-mask run tar packages used by split CPU finalization jobs."
        ),
    )
    parser.add_argument(
        "--output-staging-dir",
        type=Path,
        help=(
            "Directory for staged output zarr overlays. Defaults to "
            "/scratch/$USER/$LSB_JOBID/... when available."
        ),
    )
    parser.add_argument(
        "--keep-staged-output",
        action="store_true",
        help="Do not remove the staged local zarr after a successful publish.",
    )
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    parser.add_argument(
        "--progress-dir",
        type=Path,
        help="Directory for append-only stage progress JSONL files.",
    )
    parser.add_argument(
        "--workflow-profile-dir",
        type=Path,
        help=(
            "Directory for append-only workflow profile JSONL files. Defaults to "
            "--progress-dir when that is provided."
        ),
    )
    parser.add_argument("--consolidate-metadata", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.registry = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    dry_run = not bool(args.apply)
    if args.subject_output_parent != SUBJECT_MASK_CANONICAL_OUTPUT_PARENT and args.workflow_stage != "inference":
        print(
            "--subject-output-parent subject_mask_shard_runs is inference-only; "
            "run canonical finalization after collection merge.",
            file=sys.stderr,
        )
        return 2

    subject_run = f"subject_masks_unet_registry_{args.run_label}"
    refined_run = f"refined_subject_masks_smart_finalizer_{args.run_label}"
    if args.roots_from_report is not None:
        root_inputs = _zarr_paths_from_report(args.roots_from_report)
        discovered_zarrs = _discover_analysis_zarrs(root_inputs, include_smoke=bool(args.include_smoke))
    elif args.source == "registry":
        discovered_zarrs = _discover_registry_analysis_zarrs(
            args.roots,
            registry_path=args.registry,
            rig_id=args.rig_id,
            arena_id=args.arena_id,
            camera_id=args.camera_id_filter,
            path_contains=args.path_contains,
            exclude_refined_subject_masks_ok=not bool(args.force_inference or args.force_finalization),
        )
    else:
        discovered_zarrs = _discover_analysis_zarrs(args.roots, include_smoke=bool(args.include_smoke))

    all_plans = [
        build_archive_plan(
            zarr_path,
            subject_run_name=subject_run,
            refined_run_name=refined_run,
            force_inference=bool(args.force_inference),
            force_finalization=bool(args.force_finalization),
            workflow_stage=str(args.workflow_stage),
            subject_output_parent=str(args.subject_output_parent),
            crop_run_name=args.crop_run,
            resolve_assignment_keypoints=not bool(args.no_assignment_keypoints),
        )
        for zarr_path in discovered_zarrs
    ]
    plans = [plan for plan in all_plans if plan.run_inference or plan.run_finalization]
    if args.pilot_size is not None:
        plans = plans[: max(0, int(args.pilot_size))]
    if args.roi_cache_manifest is not None and len(plans) > 1:
        print(
            "--roi-cache-manifest can only be used when exactly one archive is selected.",
            file=sys.stderr,
        )
        return 2
    if args.crop_run is not None and len(all_plans) > 1:
        print("--crop-run can only be used when exactly one archive is selected.", file=sys.stderr)
        return 2
    if args.emit_paths:
        for plan in plans:
            print(plan.zarr_path)
        return 0

    print(f"analysis_archives_discovered: {len(all_plans)}")
    print(f"archives_selected: {len(plans)}")
    print(f"mode: {'apply' if args.apply else 'dry-run'}")
    print(f"workflow_stage: {args.workflow_stage}")
    print(f"subject_output_parent: {args.subject_output_parent}")
    print(f"subject_run: {subject_run}")
    print(f"refined_run: {refined_run}")
    print(f"stage_output_to_scratch: {bool(args.stage_output_to_scratch)}")
    if args.stage_output_to_scratch:
        output_staging_root = (
            args.output_staging_dir.expanduser().resolve()
            if args.output_staging_dir is not None
            else _default_output_staging_root()
        )
        print(f"output_staging_root: {output_staging_root}")
    else:
        output_staging_root = None

    results: list[ArchiveResult] = []
    exit_code = 0
    for idx, plan in enumerate(plans, start=1):
        print(f"\n[{idx}/{len(plans)}] {plan.zarr_path}", flush=True)
        result = ArchiveResult(
            zarr_path=plan.zarr_path,
            subject_run=plan.subject_run,
            refined_run=plan.refined_run,
            planned_inference=plan.run_inference,
            planned_finalization=plan.run_finalization,
        )
        profile_path = _workflow_profile_path(args, plan)
        profiler = WorkflowProfiler(profile_path, schema_prefix="palette_subject_mask_workflow")
        if profile_path is not None:
            result.workflow_profile_path = str(profile_path)
        profiler.emit(
            "start",
            "archive_total",
            zarr_path=plan.zarr_path,
            workflow_stage=str(args.workflow_stage),
            planned_inference=bool(plan.run_inference),
            planned_finalization=bool(plan.run_finalization),
            dry_run=bool(dry_run),
        )
        staged_ctx: OutputStagingContext | None = None
        effective_plan = plan
        archive_started = time.perf_counter()
        try:
            if args.stage_output_to_scratch:
                if dry_run:
                    result.output_staging_status = "planned"
                else:
                    assert output_staging_root is not None
                    with profiler.phase(
                        "prepare_output_staging",
                        source_zarr_path=plan.zarr_path,
                        staging_root=str(output_staging_root),
                        stage_finalization_input=bool(args.stage_finalization_input_to_scratch),
                    ) as phase:
                        staged_ctx = _prepare_output_staging_zarr(
                            Path(plan.zarr_path),
                            plan=plan,
                            staging_root=output_staging_root,
                            overwrite=bool(args.overwrite),
                            stage_finalization_input=bool(args.stage_finalization_input_to_scratch),
                        )
                        phase["staged_zarr_path"] = str(staged_ctx.staged_zarr_path)
                    effective_plan = replace(plan, zarr_path=str(staged_ctx.staged_zarr_path))
                    result.output_staging_status = "staged"
                    result.staged_zarr_path = str(staged_ctx.staged_zarr_path)
                    print(f"staged_output_zarr: {staged_ctx.staged_zarr_path}", flush=True)

            if plan.run_inference:
                cmd = _inference_command(
                    args,
                    effective_plan,
                    defer_registry_status=staged_ctx is not None,
                    roi_cache_expected_archive_path=plan.zarr_path if staged_ctx is not None else None,
                )
                with profiler.phase(
                    "inference_subprocess",
                    command=cmd,
                    dry_run=bool(dry_run),
                    staged_output=staged_ctx is not None,
                ) as phase:
                    result.inference_status = _run_command(cmd, dry_run=dry_run)
                    phase["command_status"] = result.inference_status
                if result.inference_status != "ok" and not dry_run:
                    raise RuntimeError(f"inference {result.inference_status}")
            if plan.run_finalization:
                cmd = _finalization_command(
                    args,
                    effective_plan,
                    defer_registry_status=staged_ctx is not None,
                )
                with profiler.phase(
                    "finalization_subprocess",
                    command=cmd,
                    dry_run=bool(dry_run),
                    staged_output=staged_ctx is not None,
                ) as phase:
                    result.finalization_status = _run_command(cmd, dry_run=dry_run)
                    phase["command_status"] = result.finalization_status
                if result.finalization_status != "ok" and not dry_run:
                    raise RuntimeError(f"finalization {result.finalization_status}")
            if dry_run:
                result.validation_status = "planned"
                result.publish_status = "planned" if args.stage_output_to_scratch else "not_requested"
            else:
                validation_subject_run = plan.subject_run if plan.run_inference else _selected_subject_run_for_finalization(plan)
                validation_path = Path(effective_plan.zarr_path)
                with profiler.phase(
                    "validate_outputs",
                    validation_path=str(validation_path),
                    require_subject=bool(plan.run_inference or plan.run_finalization),
                    require_refined=bool(plan.run_finalization),
                ) as phase:
                    status, detail = validate_outputs(
                        validation_path,
                        subject_run=validation_subject_run,
                        refined_run=plan.refined_run,
                        subject_output_parent=plan.subject_output_parent,
                        require_subject=bool(plan.run_inference or plan.run_finalization),
                        require_refined=bool(plan.run_finalization),
                    )
                    phase["validation_status"] = status
                    phase["validation_detail"] = detail
                result.validation_status = status
                if status != "ok":
                    raise RuntimeError(detail)
                if staged_ctx is not None:
                    with profiler.phase(
                        "publish_staged_outputs",
                        staged_zarr_path=str(staged_ctx.staged_zarr_path),
                        target_zarr_path=plan.zarr_path,
                        handoff_package_dir=str(args.handoff_package_dir) if args.handoff_package_dir else None,
                    ) as phase:
                        phase.update(_publish_staged_outputs(
                            staged_ctx,
                            plan=plan,
                            overwrite=bool(args.overwrite),
                            handoff_package_dir=args.handoff_package_dir,
                            registry_path=args.registry,
                        ))
                        result.registry_refresh_status = str(
                            phase.get("registry_refresh_status", "not_requested")
                        )
                        result.subject_mask_performance_rows = phase.get("subject_mask_performance_rows")
                        result.subject_mask_component_quality_rows = phase.get(
                            "subject_mask_component_quality_rows"
                        )
                    result.publish_status = "ok"
                    with profiler.phase(
                        "validate_published_outputs",
                        validation_path=plan.zarr_path,
                        require_subject=bool(plan.run_inference or plan.run_finalization),
                        require_refined=bool(plan.run_finalization),
                    ) as phase:
                        status, detail = validate_outputs(
                            Path(plan.zarr_path),
                            subject_run=validation_subject_run,
                            refined_run=plan.refined_run,
                            subject_output_parent=plan.subject_output_parent,
                            require_subject=bool(plan.run_inference or plan.run_finalization),
                            require_refined=bool(plan.run_finalization),
                        )
                        phase["validation_status"] = status
                        phase["validation_detail"] = detail
                    result.validation_status = status
                    if status != "ok":
                        raise RuntimeError(f"published output validation failed: {detail}")
                    if not bool(args.keep_staged_output):
                        try:
                            with profiler.phase(
                                "cleanup_output_staging",
                                staged_zarr_path=str(staged_ctx.staged_zarr_path),
                            ):
                                _cleanup_output_staging(staged_ctx)
                            result.output_staging_status = "cleaned"
                        except Exception as cleanup_exc:
                            result.output_staging_status = f"cleanup_failed:{cleanup_exc}"
                            print(f"warning: staged output cleanup failed: {cleanup_exc}", file=sys.stderr)
                else:
                    result.publish_status = "not_requested"
                if args.consolidate_metadata:
                    with profiler.phase("consolidate_metadata", zarr_path=plan.zarr_path):
                        _consolidate_metadata_quietly(plan.zarr_path)
        except Exception as exc:
            result.error = str(exc)
            exit_code = 1
            print(f"error: {exc}", file=sys.stderr, flush=True)
            if staged_ctx is not None and not bool(args.keep_staged_output):
                try:
                    with profiler.phase(
                        "cleanup_output_staging_after_error",
                        staged_zarr_path=str(staged_ctx.staged_zarr_path),
                    ):
                        _cleanup_output_staging(staged_ctx)
                    result.output_staging_status = "cleaned_after_error"
                except Exception as cleanup_exc:
                    result.output_staging_status = f"cleanup_failed_after_error:{cleanup_exc}"
                    print(f"warning: staged output cleanup failed after error: {cleanup_exc}", file=sys.stderr)
            if not bool(args.continue_on_error):
                profiler.record_finish(
                    "archive_total",
                    {
                        "duration_seconds": float(time.perf_counter() - archive_started),
                        "status": "error",
                        "error": str(exc),
                    },
                )
                result.workflow_profile = profiler.summary()
                results.append(result)
                break
        profiler.record_finish(
            "archive_total",
            {
                "duration_seconds": float(time.perf_counter() - archive_started),
                "status": "error" if result.error else "ok",
                "error": result.error or "",
            },
        )
        result.workflow_profile = profiler.summary()
        results.append(result)

    if args.json_report:
        _write_batch_json_report(args.json_report, plans=plans, results=results)
    if args.markdown_report:
        _write_markdown_report(args.markdown_report, plans=plans, results=results)

    print("\nsummary:")
    print(json.dumps({"plans": len(plans), "results": len(results), "exit_code": exit_code}, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
