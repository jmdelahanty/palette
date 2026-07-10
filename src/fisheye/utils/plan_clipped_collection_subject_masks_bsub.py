#!/usr/bin/env python3
"""Plan clipped-collection subject-mask shard jobs and finalization dependencies."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence as AbcSequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from fisheye.shared.run_provenance import json_ready
from fisheye.utils.plan_clipped_collection_keypoints_bsub import (
    DEFAULT_GROUPS_REPO,
    DEFAULT_REGISTRY,
    _bsub_prefix,
    _collection_clip_ids,
    _default_alias_manifest,
    _discover_cache_manifests,
    _load_manifest,
    _open_root,
    _parse_bsub_job_id,
    _recording_dir_from_zarr,
    _replace_job_placeholders,
    _run_command,
    _sanitize_component,
    _shell_join,
    _utc_run_id,
)


DEFAULT_COMPONENTS = ("subject_body", "eyes_union", "swim_bladder")
DEFAULT_ASSIGNMENT_KEYPOINT_GROUP = "refined_keypoints_runs"
DEFAULT_CLIP_FINALIZER_PACKAGE_DIR = Path("/nrs/ahrens/palette_staging/refined_subject_mask_clip_packages")
FINALIZATION_MODES = ("collection_direct", "per_clip_packages")


def _row_index_path_from_manifest(manifest_path: Path, payload: Mapping[str, Any]) -> Path | None:
    row_index = payload.get("row_index")
    if not isinstance(row_index, Mapping):
        return None
    raw_path = row_index.get("path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _source_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    source = payload.get("source")
    return source if isinstance(source, Mapping) else {}


def _first_selection_value(source: Mapping[str, Any], key: str) -> str | None:
    selection = source.get("selection")
    if not isinstance(selection, Mapping):
        return None
    values = selection.get(key)
    if not isinstance(values, AbcSequence) or isinstance(values, (str, bytes)):
        return None
    clean = [str(value).strip() for value in values if str(value).strip()]
    return clean[0] if len(clean) == 1 else None


def _clip_index_from_source(source: Mapping[str, Any], clip_id: str) -> int | None:
    for key in ("clip_index", "bundle_child_clip_index", "source_clip_index"):
        value = source.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    suffix = str(clip_id).rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def _normalize_components(values: Sequence[str] | None) -> tuple[str, ...]:
    components = tuple(str(value).strip() for value in (values or DEFAULT_COMPONENTS) if str(value).strip())
    if not components:
        raise ValueError("At least one subject-mask finalizer component is required.")
    return components


@dataclass(frozen=True)
class SubjectMaskClipPlan:
    clip_id: str
    cache_manifest: Path | None
    cache_status: str
    proxy_crop_run: str
    alias_manifest: Path | None
    row_index_path: Path | None
    subject_mask_shard_run: str
    subject_mask_job_name: str
    proxy_command: list[str] | None
    subject_mask_command: list[str] | None
    subject_mask_bsub_command: list[str] | None
    refined_subject_mask_clip_run: str | None = None
    refined_subject_mask_package_path: Path | None = None
    finalizer_job_name: str | None = None
    finalizer_command: list[str] | None = None
    finalizer_bsub_command: list[str] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "cache_manifest": str(self.cache_manifest) if self.cache_manifest else None,
            "cache_status": self.cache_status,
            "proxy_crop_run": self.proxy_crop_run,
            "alias_manifest": str(self.alias_manifest) if self.alias_manifest else None,
            "row_index_path": str(self.row_index_path) if self.row_index_path else None,
            "subject_mask_shard_run": self.subject_mask_shard_run,
            "subject_mask_job_name": self.subject_mask_job_name,
            "proxy_command": self.proxy_command,
            "subject_mask_command": self.subject_mask_command,
            "subject_mask_bsub_command": self.subject_mask_bsub_command,
            "refined_subject_mask_clip_run": self.refined_subject_mask_clip_run,
            "refined_subject_mask_package_path": (
                str(self.refined_subject_mask_package_path) if self.refined_subject_mask_package_path else None
            ),
            "finalizer_job_name": self.finalizer_job_name,
            "finalizer_command": self.finalizer_command,
            "finalizer_bsub_command": self.finalizer_bsub_command,
        }


@dataclass(frozen=True)
class SubjectMaskWorkflowPlan:
    zarr_path: Path
    collection_id: str
    run_id: str
    run_label: str
    repo: Path
    recording_dir: Path
    cache_dir_root: Path
    log_dir: Path
    clips: tuple[SubjectMaskClipPlan, ...]
    finalization_mode: str
    clip_finalizer_package_dir: Path | None
    import_array_copy_workers: int
    merged_proxy_crop_run: str
    refined_subject_masks_run: str
    components: tuple[str, ...]
    merge_proxy_command: list[str]
    finalize_command: list[str]
    finalizer_bsub_command: list[str]
    collection_import_job_name: str | None
    collection_import_command: list[str]
    collection_import_bsub_command: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "palette.clipped_collection_subject_mask_bsub_plan.v1",
            "zarr_path": str(self.zarr_path),
            "collection_id": self.collection_id,
            "run_id": self.run_id,
            "run_label": self.run_label,
            "repo": str(self.repo),
            "recording_dir": str(self.recording_dir),
            "cache_dir_root": str(self.cache_dir_root),
            "log_dir": str(self.log_dir),
            "clips": [clip.to_json() for clip in self.clips],
            "finalization_mode": self.finalization_mode,
            "clip_finalizer_package_dir": (
                str(self.clip_finalizer_package_dir) if self.clip_finalizer_package_dir else None
            ),
            "import_array_copy_workers": int(self.import_array_copy_workers),
            "merged_proxy_crop_run": self.merged_proxy_crop_run,
            "refined_subject_masks_run": self.refined_subject_masks_run,
            "components": list(self.components),
            "merge_proxy_command": self.merge_proxy_command,
            "finalize_command": self.finalize_command,
            "finalizer_bsub_command": self.finalizer_bsub_command,
            "collection_import_job_name": self.collection_import_job_name,
            "collection_import_command": self.collection_import_command,
            "collection_import_bsub_command": self.collection_import_bsub_command,
            "missing_cache_clip_ids": [
                clip.clip_id for clip in self.clips if clip.cache_manifest is None
            ],
        }


def build_plan(
    *,
    zarr_path: Path,
    collection_id: str,
    cache_dir_root: Path,
    clip_ids: Sequence[str],
    all_clips: bool,
    work_unit_ids: Sequence[str],
    run_id: str | None,
    run_label: str | None,
    repo: Path,
    registry: Path,
    log_dir: Path | None,
    batch_size: int,
    device: str,
    queue: str,
    ncores: int,
    mem_gb: int,
    gpus: int,
    finalizer_queue: str,
    finalizer_ncores: int,
    finalizer_mem_gb: int,
    finalizer_num_workers: int,
    finalizer_chunk_size: int,
    finalizer_dense_mask_row_chunk: int | None,
    finalizer_execution_backend: str,
    finalizer_postcompute_backend: str,
    finalizer_postcompute_num_workers: int | None,
    finalizer_postcompute_chunk_size: int | None,
    metric_level: str,
    mask_storage: str,
    mask_rle_validation_mode: str,
    components: Sequence[str] | None,
    assignment_keypoint_group: str | None,
    assignment_keypoints_run: str | None,
    write_eye_geometry: bool,
    write_component_contours: bool,
    retain_source_seeds: bool,
    model_coverage_class: str,
    model_component_coverage_key: str,
    model_label_schema_id: str,
    model_top_k: int,
    model_require_unique: bool,
    model_include_non_success: bool,
    mask_probs_dtype: str,
    mask_probs_chunk_rois: int,
    output_queue_size: int,
    profile_timings: bool,
    allow_multiple_cache_manifests: bool,
    overwrite_proxies: bool,
    overwrite_shards: bool,
    overwrite_final_outputs: bool,
    defer_registry_status: bool,
    finalization_mode: str = "collection_direct",
    clip_finalizer_package_dir: Path | None = None,
    import_array_copy_workers: int = 1,
    write_sampled_component_contours: bool = False,
) -> SubjectMaskWorkflowPlan:
    zarr_path = zarr_path.expanduser().resolve()
    cache_dir_root = cache_dir_root.expanduser().resolve()
    root = _open_root(zarr_path, mode="r")
    resolved_clip_ids = list(clip_ids)
    if all_clips:
        resolved_clip_ids = _collection_clip_ids(root, collection_id, set(work_unit_ids))
    if not resolved_clip_ids:
        raise ValueError("No clip ids selected. Pass --all-clips or one or more --clip-id values.")
    if len(set(resolved_clip_ids)) != len(resolved_clip_ids):
        raise ValueError("Duplicate clip ids were supplied.")
    resolved_clip_ids = sorted(resolved_clip_ids)

    resolved_components = _normalize_components(components)
    resolved_run_id = _sanitize_component(run_id or _utc_run_id(), default="run")
    resolved_run_label = _sanitize_component(
        run_label or f"clipped_subject_masks_{resolved_run_id}",
        default="clipped_subject_masks",
    )
    resolved_finalization_mode = str(finalization_mode or "collection_direct").strip()
    if resolved_finalization_mode not in FINALIZATION_MODES:
        raise ValueError(
            f"Unsupported finalization mode {finalization_mode!r}; expected one of {FINALIZATION_MODES}."
        )
    resolved_import_array_copy_workers = max(1, int(import_array_copy_workers))
    resolved_clip_package_dir = (
        (clip_finalizer_package_dir or DEFAULT_CLIP_FINALIZER_PACKAGE_DIR)
        .expanduser()
        .resolve()
        / resolved_run_label
    )
    needs_eye_assignment = "eyes_union" in resolved_components
    if needs_eye_assignment and not assignment_keypoints_run:
        raise ValueError(
            "Finalizing eyes_union requires --assignment-keypoints-run. "
            "Pass a collection refined-keypoint run, or choose components without eyes_union."
        )

    recording_dir = _recording_dir_from_zarr(zarr_path)
    resolved_log_dir = (
        log_dir or (recording_dir / "logs" / "clipped_collection_subject_masks_bsub" / resolved_run_label)
    ).expanduser()
    manifests = _discover_cache_manifests(
        cache_dir_root=cache_dir_root,
        collection_id=collection_id,
        clip_ids=resolved_clip_ids,
        allow_multiple=allow_multiple_cache_manifests,
    )

    clip_plans: list[SubjectMaskClipPlan] = []
    for clip_id in resolved_clip_ids:
        safe_clip = _sanitize_component(clip_id, default="clip")
        proxy_crop_run = f"crop_proxy_{resolved_run_label}_{safe_clip}"
        subject_mask_shard_run = f"subject_masks_{resolved_run_label}_{safe_clip}"
        subject_mask_job_name = f"sm_{resolved_run_label}_{safe_clip}"
        manifest_path = manifests.get(clip_id)
        alias_manifest = _default_alias_manifest(manifest_path, proxy_crop_run) if manifest_path else None
        row_index_path: Path | None = None
        proxy_command: list[str] | None = None
        subject_mask_command: list[str] | None = None
        subject_mask_bsub_command: list[str] | None = None

        if manifest_path is not None and alias_manifest is not None:
            manifest_payload = _load_manifest(manifest_path)
            source = _source_mapping(manifest_payload)
            row_index_path = _row_index_path_from_manifest(manifest_path, manifest_payload)
            source_collection_path = str(source.get("collection_path") or "").strip() or None
            source_work_unit_id = _first_selection_value(source, "work_unit_ids")
            source_clip_index = _clip_index_from_source(source, clip_id)
            proxy_command = [
                "scripts/py",
                "-m",
                "fisheye.utils.create_clipped_collection_proxy_crop_run",
                str(zarr_path),
                str(manifest_path),
                "--proxy-run",
                proxy_crop_run,
                "--alias-manifest",
                str(alias_manifest),
                "--json",
            ]
            if overwrite_proxies:
                proxy_command.append("--overwrite")

            subject_mask_command = [
                "scripts/py",
                "-m",
                "fisheye.segmentation.infer_unet_subject_masks",
                str(zarr_path),
                "--resolve-model-from-registry",
                "--registry",
                str(registry),
                "--model-coverage-class",
                model_coverage_class,
                "--model-component-coverage-key",
                model_component_coverage_key,
                "--model-label-schema-id",
                model_label_schema_id,
                "--model-top-k",
                str(model_top_k),
                "--run-name",
                subject_mask_shard_run,
                "--output-parent",
                "subject_mask_shard_runs",
                "--crop-run",
                proxy_crop_run,
                "--source-collection-id",
                collection_id,
                "--source-clip-id",
                clip_id,
                "--source-shard-id",
                subject_mask_shard_run,
                "--source-roi-cache-alias-manifest",
                str(alias_manifest),
                "--device",
                device,
                "--batch-size",
                str(batch_size),
                "--mask-probs-dtype",
                mask_probs_dtype,
                "--mask-probs-chunk-rois",
                str(mask_probs_chunk_rois),
                "--no-write-masks-roi",
                "--async-output",
                "--output-queue-size",
                str(output_queue_size),
                "--no-progress",
                "--roi-cache-policy",
                "never",
                "--roi-cache-manifest",
                str(alias_manifest),
                "--roi-cache-expected-archive-path",
                str(zarr_path),
            ]
            if source_collection_path is not None:
                subject_mask_command.extend(["--source-collection-path", source_collection_path])
            if source_clip_index is not None:
                subject_mask_command.extend(["--source-clip-index", str(source_clip_index)])
            if source_work_unit_id is not None:
                subject_mask_command.extend(["--source-work-unit-id", source_work_unit_id])
            if row_index_path is not None:
                subject_mask_command.extend(["--source-roi-cache-row-index-path", str(row_index_path)])
            if model_require_unique:
                subject_mask_command.append("--model-require-unique")
            if model_include_non_success:
                subject_mask_command.append("--model-include-non-success")
            if profile_timings:
                subject_mask_command.append("--profile-timings")
            if overwrite_shards:
                subject_mask_command.append("--overwrite")

            subject_mask_bsub_command = [
                *_bsub_prefix(
                    job_name=subject_mask_job_name,
                    queue=queue,
                    ncores=ncores,
                    mem_gb=mem_gb,
                    gpus=gpus,
                    log_dir=resolved_log_dir,
                    dependency=None,
                ),
                "bash",
                "-lc",
                f"cd {_shell_join([repo])} && {_shell_join(subject_mask_command)}",
            ]

        clip_plans.append(
            SubjectMaskClipPlan(
                clip_id=clip_id,
                cache_manifest=manifest_path,
                cache_status="found" if manifest_path else "missing",
                proxy_crop_run=proxy_crop_run,
                alias_manifest=alias_manifest,
                row_index_path=row_index_path,
                subject_mask_shard_run=subject_mask_shard_run,
                subject_mask_job_name=subject_mask_job_name,
                proxy_command=proxy_command,
                subject_mask_command=subject_mask_command,
                subject_mask_bsub_command=subject_mask_bsub_command,
            )
        )

    merged_proxy_crop_run = f"crop_proxy_{resolved_run_label}_collection"
    refined_subject_masks_run = f"refined_subject_masks_{resolved_run_label}"
    merge_proxy_command = [
        "scripts/py",
        "-m",
        "fisheye.utils.merge_clipped_proxy_crop_runs",
        str(zarr_path),
        "--output-run",
        merged_proxy_crop_run,
        "--json",
    ]
    for clip in clip_plans:
        merge_proxy_command.extend(["--source-crop-run", clip.proxy_crop_run])

    finalize_command = [
        "scripts/py",
        "-m",
        "fisheye.refinement.finalize_subject_masks",
        str(zarr_path),
        "--target-crop-run",
        merged_proxy_crop_run,
        "--refined-run",
        refined_subject_masks_run,
        "--components",
        *resolved_components,
        "--chunk-size",
        str(finalizer_chunk_size),
        "--metric-level",
        metric_level,
        "--mask-storage",
        mask_storage,
        "--mask-rle-validation-mode",
        mask_rle_validation_mode,
        "--execution-backend",
        finalizer_execution_backend,
        "--num-workers",
        str(finalizer_num_workers),
        "--postcompute-backend",
        finalizer_postcompute_backend,
        "--json",
    ]
    if finalizer_dense_mask_row_chunk is not None:
        finalize_command.extend(["--dense-mask-row-chunk", str(finalizer_dense_mask_row_chunk)])
    if finalizer_postcompute_num_workers is not None:
        finalize_command.extend(["--postcompute-num-workers", str(finalizer_postcompute_num_workers)])
    if finalizer_postcompute_chunk_size is not None:
        finalize_command.extend(["--postcompute-chunk-size", str(finalizer_postcompute_chunk_size)])
    if assignment_keypoints_run:
        finalize_command.extend(
            [
                "--assignment-keypoint-group",
                assignment_keypoint_group or DEFAULT_ASSIGNMENT_KEYPOINT_GROUP,
                "--assignment-keypoints-run",
                assignment_keypoints_run,
            ]
        )
    if write_eye_geometry:
        finalize_command.append("--write-eye-geometry")
    if write_component_contours:
        finalize_command.append("--write-component-contours")
    if write_sampled_component_contours:
        finalize_command.extend(["--write-sampled-component-contours", "--sampled-contour-row-chunk", "1024"])
    if retain_source_seeds:
        finalize_command.append("--retain-source-seeds")
    if defer_registry_status:
        finalize_command.append("--defer-registry-status")
    else:
        finalize_command.extend(["--registry", str(registry)])
    for clip in clip_plans:
        finalize_command.extend(["--subject-shard-run", clip.subject_mask_shard_run])
    if overwrite_final_outputs:
        merge_proxy_command.append("--overwrite")
        finalize_command.append("--overwrite")

    shard_job_refs = [
        f"<jobid:{clip.subject_mask_job_name}>"
        for clip in clip_plans
        if clip.subject_mask_bsub_command is not None
    ]
    finalizer_dependency = " && ".join(f"done({name})" for name in shard_job_refs) if shard_job_refs else None
    finalizer_job_name = f"sm_finalize_{resolved_run_label}"
    finalizer_shell = f"cd {_shell_join([repo])} && {_shell_join(merge_proxy_command)} && {_shell_join(finalize_command)}"
    finalizer_bsub_command = [
        *_bsub_prefix(
            job_name=finalizer_job_name,
            queue=finalizer_queue,
            ncores=finalizer_ncores,
            mem_gb=finalizer_mem_gb,
            gpus=0,
            log_dir=resolved_log_dir,
            dependency=finalizer_dependency,
        ),
        "bash",
        "-lc",
        finalizer_shell,
    ]
    collection_import_job_name: str | None = None
    collection_import_command: list[str] = []
    collection_import_bsub_command: list[str] = []
    if resolved_finalization_mode == "per_clip_packages":
        packaged_clips: list[SubjectMaskClipPlan] = []
        for clip in clip_plans:
            if clip.subject_mask_bsub_command is None:
                packaged_clips.append(clip)
                continue
            safe_clip = _sanitize_component(clip.clip_id, default="clip")
            refined_clip_run = f"{refined_subject_masks_run}_{safe_clip}"
            package_path = resolved_clip_package_dir / f"{refined_clip_run}.tar.gz"
            clip_finalizer_job_name = f"sm_finalize_{resolved_run_label}_{safe_clip}"
            clip_finalize_command = [
                "scripts/py",
                "-m",
                "fisheye.utils.finalize_subject_mask_clip_package",
                "--source-zarr",
                str(zarr_path),
                "--subject-shard-run",
                clip.subject_mask_shard_run,
                "--target-crop-run",
                merged_proxy_crop_run,
                "--refined-run",
                refined_clip_run,
                "--package-path",
                str(package_path),
                "--chunk-size",
                str(finalizer_chunk_size),
                "--metric-level",
                metric_level,
                "--mask-storage",
                mask_storage,
                "--mask-rle-validation-mode",
                mask_rle_validation_mode,
                "--execution-backend",
                finalizer_execution_backend,
                "--num-workers",
                str(finalizer_num_workers),
                "--postcompute-backend",
                finalizer_postcompute_backend,
                "--json",
            ]
            for component in resolved_components:
                clip_finalize_command.extend(["--component", component])
            if finalizer_dense_mask_row_chunk is not None:
                clip_finalize_command.extend(["--dense-mask-row-chunk", str(finalizer_dense_mask_row_chunk)])
            if finalizer_postcompute_num_workers is not None:
                clip_finalize_command.extend(["--postcompute-num-workers", str(finalizer_postcompute_num_workers)])
            if finalizer_postcompute_chunk_size is not None:
                clip_finalize_command.extend(["--postcompute-chunk-size", str(finalizer_postcompute_chunk_size)])
            if assignment_keypoints_run:
                clip_finalize_command.extend(
                    [
                        "--assignment-keypoint-group",
                        assignment_keypoint_group or DEFAULT_ASSIGNMENT_KEYPOINT_GROUP,
                        "--assignment-keypoints-run",
                        assignment_keypoints_run,
                    ]
                )
            if not write_eye_geometry:
                clip_finalize_command.append("--no-write-eye-geometry")
            if not write_component_contours:
                clip_finalize_command.append("--no-write-component-contours")
            if not write_sampled_component_contours:
                clip_finalize_command.append("--no-write-sampled-component-contours")
            if retain_source_seeds:
                clip_finalize_command.append("--retain-source-seeds")
            if overwrite_final_outputs:
                clip_finalize_command.append("--overwrite")

            dependency = f"done(<jobid:{clip.subject_mask_job_name}>)"
            clip_finalizer_shell = f"cd {_shell_join([repo])} && {_shell_join(clip_finalize_command)}"
            clip_finalizer_bsub_command = [
                *_bsub_prefix(
                    job_name=clip_finalizer_job_name,
                    queue=finalizer_queue,
                    ncores=finalizer_ncores,
                    mem_gb=finalizer_mem_gb,
                    gpus=0,
                    log_dir=resolved_log_dir,
                    dependency=dependency,
                ),
                "bash",
                "-lc",
                clip_finalizer_shell,
            ]
            packaged_clips.append(
                replace(
                    clip,
                    refined_subject_mask_clip_run=refined_clip_run,
                    refined_subject_mask_package_path=package_path,
                    finalizer_job_name=clip_finalizer_job_name,
                    finalizer_command=clip_finalize_command,
                    finalizer_bsub_command=clip_finalizer_bsub_command,
                )
            )
        clip_plans = packaged_clips
        finalizer_bsub_command = []
        package_paths = [
            clip.refined_subject_mask_package_path
            for clip in clip_plans
            if clip.refined_subject_mask_package_path is not None
        ]
        if package_paths:
            collection_import_job_name = f"sm_import_{resolved_run_label}"
            collection_import_command = [
                "scripts/py",
                "-m",
                "fisheye.utils.import_refined_subject_mask_clip_packages",
                "--zarr",
                str(zarr_path),
                "--output-run",
                refined_subject_masks_run,
                "--expected-target-crop-run",
                merged_proxy_crop_run,
                "--array-copy-workers",
                str(resolved_import_array_copy_workers),
                "--json",
            ]
            for package_path in package_paths:
                collection_import_command.extend(["--package", str(package_path)])
            if overwrite_final_outputs:
                collection_import_command.append("--overwrite")
            import_dependency_refs = [
                f"<jobid:{clip.finalizer_job_name}>"
                for clip in clip_plans
                if clip.finalizer_job_name is not None
            ]
            import_dependency = (
                " && ".join(f"done({name})" for name in import_dependency_refs)
                if import_dependency_refs
                else None
            )
            collection_import_shell = f"cd {_shell_join([repo])} && {_shell_join(collection_import_command)}"
            collection_import_bsub_command = [
                *_bsub_prefix(
                    job_name=collection_import_job_name,
                    queue=finalizer_queue,
                    ncores=finalizer_ncores,
                    mem_gb=finalizer_mem_gb,
                    gpus=0,
                    log_dir=resolved_log_dir,
                    dependency=import_dependency,
                ),
                "bash",
                "-lc",
                collection_import_shell,
            ]

    return SubjectMaskWorkflowPlan(
        zarr_path=zarr_path,
        collection_id=collection_id,
        run_id=resolved_run_id,
        run_label=resolved_run_label,
        repo=repo,
        recording_dir=recording_dir,
        cache_dir_root=cache_dir_root,
        log_dir=resolved_log_dir,
        clips=tuple(clip_plans),
        finalization_mode=resolved_finalization_mode,
        clip_finalizer_package_dir=resolved_clip_package_dir if resolved_finalization_mode == "per_clip_packages" else None,
        import_array_copy_workers=resolved_import_array_copy_workers,
        merged_proxy_crop_run=merged_proxy_crop_run,
        refined_subject_masks_run=refined_subject_masks_run,
        components=resolved_components,
        merge_proxy_command=merge_proxy_command,
        finalize_command=finalize_command,
        finalizer_bsub_command=finalizer_bsub_command,
        collection_import_job_name=collection_import_job_name,
        collection_import_command=collection_import_command,
        collection_import_bsub_command=collection_import_bsub_command,
    )


def _print_plan(plan: SubjectMaskWorkflowPlan) -> None:
    print("Clipped collection subject-mask workflow dry-run")
    print(f"  zarr: {plan.zarr_path}")
    print(f"  collection_id: {plan.collection_id}")
    print(f"  run_label: {plan.run_label}")
    print(f"  clips: {len(plan.clips)}")
    print(f"  components: {', '.join(plan.components)}")
    print(f"  finalization_mode: {plan.finalization_mode}")
    print(f"  log_dir: {plan.log_dir}")
    if plan.clip_finalizer_package_dir is not None:
        print(f"  clip_finalizer_package_dir: {plan.clip_finalizer_package_dir}")
    missing = [clip.clip_id for clip in plan.clips if clip.cache_manifest is None]
    if missing:
        print(f"  missing_cache_manifests: {', '.join(missing)}")
    print()
    print("DAG")
    for clip in plan.clips:
        print(
            "  "
            f"cache[{clip.clip_id}] ({clip.cache_status}) "
            f"-> proxy[{clip.proxy_crop_run}] "
            f"-> subject_mask_shard[{clip.subject_mask_shard_run}]"
        )
    shard_refs = ", ".join(clip.subject_mask_shard_run for clip in plan.clips)
    if plan.finalization_mode == "per_clip_packages":
        package_refs = ", ".join(
            str(clip.refined_subject_mask_package_path)
            for clip in plan.clips
            if clip.refined_subject_mask_package_path is not None
        )
        print(f"  shards[{shard_refs}] -> merge_proxy[{plan.merged_proxy_crop_run}]")
        print(f"  shards[{shard_refs}] -> clip_packages[{package_refs}]")
        print(f"  merge_proxy[{plan.merged_proxy_crop_run}] -> import_refined_subject_masks[{plan.refined_subject_masks_run}]")
        print(f"  clip_packages -> import_refined_subject_masks[{plan.refined_subject_masks_run}]")
    else:
        print(f"  shards[{shard_refs}] -> merge_proxy[{plan.merged_proxy_crop_run}]")
        print(f"  merge_proxy[{plan.merged_proxy_crop_run}] -> refined_subject_masks[{plan.refined_subject_masks_run}]")
    print()
    print("Submit-side proxy commands")
    for clip in plan.clips:
        if clip.proxy_command is None:
            print(f"  # {clip.clip_id}: cache manifest missing; proxy command not available")
            continue
        print(f"  {_shell_join(clip.proxy_command)}")
    print()
    print("Subject-mask shard bsub commands")
    for clip in plan.clips:
        if clip.subject_mask_bsub_command is None:
            print(f"  # {clip.clip_id}: cache manifest missing; subject-mask shard job not available")
            continue
        print(f"  {_shell_join(clip.subject_mask_bsub_command)}")
    print()
    if plan.finalization_mode == "per_clip_packages":
        print("Clip finalizer/package bsub command templates")
        for clip in plan.clips:
            if clip.finalizer_bsub_command is None:
                print(f"  # {clip.clip_id}: finalizer package job not available")
                continue
            print(f"  {_shell_join(clip.finalizer_bsub_command)}")
        print()
        print("Collection import bsub command template")
        print(f"  {_shell_join(plan.collection_import_bsub_command)}")
    else:
        print("Finalizer bsub command template")
        print(f"  {_shell_join(plan.finalizer_bsub_command)}")


class CompletedProcessLike(Protocol):
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[..., CompletedProcessLike]


def _write_submission_snapshot(
    path: Path,
    *,
    plan: SubjectMaskWorkflowPlan,
    status: str,
    proxy_results: Sequence[Mapping[str, Any]],
    shard_results: Sequence[Mapping[str, Any]],
    job_ids_by_name: Mapping[str, str],
    finalizer: Mapping[str, Any] | None = None,
    merge_proxy: Mapping[str, Any] | None = None,
    clip_finalizers: Sequence[Mapping[str, Any]] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "palette.clipped_collection_subject_mask_bsub_submission.v1",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(path.with_name("submission_plan.json")),
        "zarr_path": str(plan.zarr_path),
        "collection_id": plan.collection_id,
        "run_id": plan.run_id,
        "run_label": plan.run_label,
        "log_dir": str(plan.log_dir),
        "finalization_mode": plan.finalization_mode,
        "proxy_results": list(proxy_results),
        "subject_mask_shard_results": list(shard_results),
        "job_ids_by_name": dict(job_ids_by_name),
    }
    if merge_proxy is not None:
        payload["merge_proxy"] = dict(merge_proxy)
    if finalizer is not None:
        payload["finalizer"] = dict(finalizer)
    if clip_finalizers is not None:
        payload["clip_finalizers"] = list(clip_finalizers)
    if error is not None:
        payload["error"] = dict(error)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_ready(payload)


def apply_plan(plan: SubjectMaskWorkflowPlan, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    """Create proxy runs and submit the planned clipped subject-mask LSF DAG."""

    missing = [clip.clip_id for clip in plan.clips if clip.cache_manifest is None]
    if missing:
        raise ValueError(f"Cannot apply plan; missing cache manifests for clips: {missing}")
    unavailable = [
        clip.clip_id
        for clip in plan.clips
        if clip.proxy_command is None or clip.subject_mask_bsub_command is None
    ]
    if unavailable:
        raise ValueError(f"Cannot apply plan; clips are missing proxy or subject-mask commands: {unavailable}")
    if plan.finalization_mode == "per_clip_packages":
        unavailable_finalizers = [
            clip.clip_id
            for clip in plan.clips
            if clip.finalizer_bsub_command is None or clip.finalizer_job_name is None
        ]
        if unavailable_finalizers:
            raise ValueError(
                f"Cannot apply plan; clips are missing finalizer package commands: {unavailable_finalizers}"
            )
        if not plan.collection_import_bsub_command or plan.collection_import_job_name is None:
            raise ValueError("Cannot apply per_clip_packages plan; collection import command is missing.")

    plan.log_dir.mkdir(parents=True, exist_ok=True)
    plan_payload = json_ready(plan.to_json())
    plan_path = plan.log_dir / "submission_plan.json"
    submission_path = plan.log_dir / "submission.json"
    plan_path.write_text(json.dumps(plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    proxy_results: list[dict[str, Any]] = []
    shard_results: list[dict[str, Any]] = []
    job_ids_by_name: dict[str, str] = {}
    finalizer_payload: dict[str, Any] | None = None
    merge_proxy_payload: dict[str, Any] | None = None
    clip_finalizer_results: list[dict[str, Any]] = []

    _write_submission_snapshot(
        submission_path,
        plan=plan,
        status="submitting",
        proxy_results=proxy_results,
        shard_results=shard_results,
        job_ids_by_name=job_ids_by_name,
    )

    try:
        for clip in plan.clips:
            assert clip.proxy_command is not None
            proxy_result = _run_command(clip.proxy_command, cwd=plan.repo, runner=runner)
            proxy_results.append(
                {
                    "clip_id": clip.clip_id,
                    "proxy_crop_run": clip.proxy_crop_run,
                    "alias_manifest": str(clip.alias_manifest) if clip.alias_manifest else None,
                    "command_result": proxy_result,
                }
            )
            _write_submission_snapshot(
                submission_path,
                plan=plan,
                status="submitting",
                proxy_results=proxy_results,
                shard_results=shard_results,
                job_ids_by_name=job_ids_by_name,
            )

        if plan.finalization_mode == "per_clip_packages":
            merge_result = _run_command(plan.merge_proxy_command, cwd=plan.repo, runner=runner)
            merge_proxy_payload = {
                "merged_proxy_crop_run": plan.merged_proxy_crop_run,
                "command": plan.merge_proxy_command,
                "command_result": merge_result,
            }
            _write_submission_snapshot(
                submission_path,
                plan=plan,
                status="submitting",
                proxy_results=proxy_results,
                shard_results=shard_results,
                job_ids_by_name=job_ids_by_name,
                merge_proxy=merge_proxy_payload,
                clip_finalizers=clip_finalizer_results,
            )

        for clip in plan.clips:
            assert clip.subject_mask_bsub_command is not None
            submit_result = _run_command(clip.subject_mask_bsub_command, cwd=plan.repo, runner=runner)
            job_id = _parse_bsub_job_id(submit_result["stdout"], submit_result["stderr"])
            job_ids_by_name[clip.subject_mask_job_name] = job_id
            shard_results.append(
                {
                    "clip_id": clip.clip_id,
                    "subject_mask_job_name": clip.subject_mask_job_name,
                    "subject_mask_job_id": job_id,
                    "subject_mask_shard_run": clip.subject_mask_shard_run,
                    "command_result": submit_result,
                }
            )
            _write_submission_snapshot(
                submission_path,
                plan=plan,
                status="submitting",
                proxy_results=proxy_results,
                shard_results=shard_results,
                job_ids_by_name=job_ids_by_name,
                merge_proxy=merge_proxy_payload,
                clip_finalizers=clip_finalizer_results,
            )

        if plan.finalization_mode == "per_clip_packages":
            for clip in plan.clips:
                if clip.finalizer_bsub_command is None or clip.finalizer_job_name is None:
                    raise ValueError(f"Clip {clip.clip_id} is missing a finalizer package command.")
                finalizer_command = _replace_job_placeholders(clip.finalizer_bsub_command, job_ids_by_name)
                finalizer_result = _run_command(finalizer_command, cwd=plan.repo, runner=runner)
                finalizer_job_id = _parse_bsub_job_id(finalizer_result["stdout"], finalizer_result["stderr"])
                job_ids_by_name[clip.finalizer_job_name] = finalizer_job_id
                clip_finalizer_results.append(
                    {
                        "clip_id": clip.clip_id,
                        "job_name": clip.finalizer_job_name,
                        "job_id": finalizer_job_id,
                        "command": finalizer_command,
                        "command_result": finalizer_result,
                        "output_run": clip.refined_subject_mask_clip_run,
                        "package_path": (
                            str(clip.refined_subject_mask_package_path)
                            if clip.refined_subject_mask_package_path
                            else None
                        ),
                        "merged_proxy_crop_run": plan.merged_proxy_crop_run,
                    }
                )
                _write_submission_snapshot(
                    submission_path,
                    plan=plan,
                    status="submitting",
                    proxy_results=proxy_results,
                    shard_results=shard_results,
                    job_ids_by_name=job_ids_by_name,
                    merge_proxy=merge_proxy_payload,
                    clip_finalizers=clip_finalizer_results,
                )
            assert plan.collection_import_job_name is not None
            collection_import_command = _replace_job_placeholders(plan.collection_import_bsub_command, job_ids_by_name)
            collection_import_result = _run_command(collection_import_command, cwd=plan.repo, runner=runner)
            collection_import_job_id = _parse_bsub_job_id(
                collection_import_result["stdout"],
                collection_import_result["stderr"],
            )
            job_ids_by_name[plan.collection_import_job_name] = collection_import_job_id
            finalizer_payload = {
                "job_name": plan.collection_import_job_name,
                "job_id": collection_import_job_id,
                "command": collection_import_command,
                "command_result": collection_import_result,
                "output_run": plan.refined_subject_masks_run,
                "merged_proxy_crop_run": plan.merged_proxy_crop_run,
                "source_package_count": len(clip_finalizer_results),
                "source_packages": [
                    item.get("package_path")
                    for item in clip_finalizer_results
                    if item.get("package_path") is not None
                ],
            }
        else:
            finalizer_command = _replace_job_placeholders(plan.finalizer_bsub_command, job_ids_by_name)
            finalizer_result = _run_command(finalizer_command, cwd=plan.repo, runner=runner)
            finalizer_job_id = _parse_bsub_job_id(finalizer_result["stdout"], finalizer_result["stderr"])
            finalizer_job_name = f"sm_finalize_{plan.run_label}"
            job_ids_by_name[finalizer_job_name] = finalizer_job_id
            finalizer_payload = {
                "job_name": finalizer_job_name,
                "job_id": finalizer_job_id,
                "command": finalizer_command,
                "command_result": finalizer_result,
                "output_run": plan.refined_subject_masks_run,
                "merged_proxy_crop_run": plan.merged_proxy_crop_run,
            }
    except Exception as exc:
        _write_submission_snapshot(
            submission_path,
            plan=plan,
            status="failed",
            proxy_results=proxy_results,
            shard_results=shard_results,
            job_ids_by_name=job_ids_by_name,
            finalizer=finalizer_payload,
            merge_proxy=merge_proxy_payload,
            clip_finalizers=clip_finalizer_results,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise

    submission = _write_submission_snapshot(
        submission_path,
        plan=plan,
        status="submitted",
        proxy_results=proxy_results,
        shard_results=shard_results,
        job_ids_by_name=job_ids_by_name,
        finalizer=finalizer_payload,
        merge_proxy=merge_proxy_payload,
        clip_finalizers=clip_finalizer_results,
    )
    submission["submitted_at_utc"] = submission["updated_at_utc"]
    submission_path.write_text(json.dumps(json_ready(submission), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    submission["submission_path"] = str(submission_path)
    return json_ready(submission)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path, help="Analysis Zarr archive.")
    parser.add_argument("--collection-id", required=True, help="Finalized clipped refined-detect collection id.")
    parser.add_argument("--cache-dir-root", required=True, type=Path, help="Root containing clipped flat ROI cache manifests.")
    parser.add_argument("--all-clips", action="store_true", help="Plan all clips from the finalized collection.")
    parser.add_argument("--clip-id", action="append", default=[], help="Clip id to include; repeatable.")
    parser.add_argument("--work-unit-id", action="append", default=[], help="Restrict --all-clips to selected work units.")
    parser.add_argument("--run-id", help="Stable run id. Defaults to UTC timestamp.")
    parser.add_argument("--run-label", help="Stable run label used in run/job names.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_GROUPS_REPO, help="Palette repo path used inside LSF jobs.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="Palette registry sqlite path.")
    parser.add_argument("--log-dir", type=Path, help="LSF/progress log dir.")
    parser.add_argument("--batch-size-sm", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--queue", default="gpu_l4")
    parser.add_argument("--ncores", type=int, default=8)
    parser.add_argument("--mem-gb", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--finalizer-queue", default="short")
    parser.add_argument("--finalizer-ncores", type=int, default=8)
    parser.add_argument("--finalizer-mem-gb", type=int, default=32)
    parser.add_argument("--finalizer-num-workers", type=int, default=8)
    parser.add_argument("--finalizer-chunk-size", type=int, default=256)
    parser.add_argument("--finalizer-dense-mask-row-chunk", type=int, default=128)
    parser.add_argument("--finalizer-execution-backend", default="process_shards")
    parser.add_argument("--finalizer-postcompute-backend", default="process_shards")
    parser.add_argument("--finalizer-postcompute-num-workers", type=int)
    parser.add_argument("--finalizer-postcompute-chunk-size", type=int)
    parser.add_argument("--metric-level", default="cheap")
    parser.add_argument("--mask-storage", default="dense_and_bitpacked")
    parser.add_argument("--mask-rle-validation-mode", default="invariants")
    parser.add_argument(
        "--components",
        nargs="+",
        default=list(DEFAULT_COMPONENTS),
        help="Finalized components to write (default: subject_body eyes_union swim_bladder).",
    )
    parser.add_argument("--assignment-keypoint-group", default=DEFAULT_ASSIGNMENT_KEYPOINT_GROUP)
    parser.add_argument("--assignment-keypoints-run", help="Collection keypoint/refined-keypoint run for eyes_union assignment.")
    parser.add_argument("--no-write-eye-geometry", action="store_true")
    parser.add_argument("--no-write-component-contours", action="store_true")
    parser.add_argument("--write-sampled-component-contours", action="store_true")
    parser.add_argument("--no-write-sampled-component-contours", action="store_true")
    parser.add_argument("--retain-source-seeds", action="store_true")
    parser.add_argument("--model-coverage-class", default="dense_all_components")
    parser.add_argument("--model-component-coverage-key", default="body+eyes+swim_bladder")
    parser.add_argument("--model-label-schema-id", default="subject_v1_union")
    parser.add_argument("--model-top-k", type=int, default=5)
    parser.add_argument("--model-require-unique", action="store_true")
    parser.add_argument("--model-include-non-success", action="store_true")
    parser.add_argument("--mask-probs-dtype", default="uint8")
    parser.add_argument("--mask-probs-chunk-rois", type=int, default=32)
    parser.add_argument("--output-queue-size", type=int, default=2)
    parser.add_argument("--profile-timings", action="store_true")
    parser.add_argument("--allow-multiple-cache-manifests", action="store_true")
    parser.add_argument("--overwrite-proxies", action="store_true")
    parser.add_argument("--overwrite-shards", action="store_true")
    parser.add_argument("--overwrite-final-outputs", action="store_true")
    parser.add_argument("--defer-registry-status", action="store_true")
    parser.add_argument(
        "--finalization-mode",
        choices=FINALIZATION_MODES,
        default="collection_direct",
        help=(
            "collection_direct submits one collection finalizer that writes to the analysis Zarr. "
            "per_clip_packages submits one CPU finalizer/package job per clip and writes NRS tarballs."
        ),
    )
    parser.add_argument(
        "--clip-finalizer-package-dir",
        type=Path,
        default=DEFAULT_CLIP_FINALIZER_PACKAGE_DIR,
        help="Base directory for per-clip refined subject-mask finalizer packages.",
    )
    parser.add_argument(
        "--import-array-copy-workers",
        type=int,
        default=1,
        help=(
            "For per_clip_packages collection import, number of chunk-owned row-copy workers "
            "used to write row-aligned arrays into the merged refined-mask run."
        ),
    )
    parser.add_argument("--plan-json", type=Path, help="Optional path to write JSON plan.")
    parser.add_argument("--json", action="store_true", help="Print JSON plan instead of text.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Plan only; no jobs are submitted.")
    mode.add_argument("--apply", action="store_true", help="Create proxies and submit the LSF DAG.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.all_clips and args.clip_id:
        raise SystemExit("Use either --all-clips or --clip-id, not both.")
    if not args.all_clips and not args.clip_id:
        raise SystemExit("Provide --all-clips or at least one --clip-id.")
    postcompute_workers = args.finalizer_postcompute_num_workers
    if postcompute_workers is None:
        postcompute_workers = int(args.finalizer_num_workers)
    plan = build_plan(
        zarr_path=args.zarr,
        collection_id=args.collection_id,
        cache_dir_root=args.cache_dir_root,
        clip_ids=args.clip_id,
        all_clips=bool(args.all_clips),
        work_unit_ids=args.work_unit_id,
        run_id=args.run_id,
        run_label=args.run_label,
        repo=args.repo,
        registry=args.registry,
        log_dir=args.log_dir,
        batch_size=int(args.batch_size_sm),
        device=args.device,
        queue=args.queue,
        ncores=int(args.ncores),
        mem_gb=int(args.mem_gb),
        gpus=int(args.gpus),
        finalizer_queue=args.finalizer_queue,
        finalizer_ncores=int(args.finalizer_ncores),
        finalizer_mem_gb=int(args.finalizer_mem_gb),
        finalizer_num_workers=int(args.finalizer_num_workers),
        finalizer_chunk_size=int(args.finalizer_chunk_size),
        finalizer_dense_mask_row_chunk=int(args.finalizer_dense_mask_row_chunk),
        finalizer_execution_backend=args.finalizer_execution_backend,
        finalizer_postcompute_backend=args.finalizer_postcompute_backend,
        finalizer_postcompute_num_workers=postcompute_workers,
        finalizer_postcompute_chunk_size=args.finalizer_postcompute_chunk_size,
        metric_level=args.metric_level,
        mask_storage=args.mask_storage,
        mask_rle_validation_mode=args.mask_rle_validation_mode,
        components=args.components,
        assignment_keypoint_group=args.assignment_keypoint_group,
        assignment_keypoints_run=args.assignment_keypoints_run,
        write_eye_geometry=not bool(args.no_write_eye_geometry),
        write_component_contours=not bool(args.no_write_component_contours),
        retain_source_seeds=bool(args.retain_source_seeds),
        model_coverage_class=args.model_coverage_class,
        model_component_coverage_key=args.model_component_coverage_key,
        model_label_schema_id=args.model_label_schema_id,
        model_top_k=int(args.model_top_k),
        model_require_unique=bool(args.model_require_unique),
        model_include_non_success=bool(args.model_include_non_success),
        mask_probs_dtype=args.mask_probs_dtype,
        mask_probs_chunk_rois=int(args.mask_probs_chunk_rois),
        output_queue_size=int(args.output_queue_size),
        profile_timings=bool(args.profile_timings),
        allow_multiple_cache_manifests=bool(args.allow_multiple_cache_manifests),
        overwrite_proxies=bool(args.overwrite_proxies),
        overwrite_shards=bool(args.overwrite_shards),
        overwrite_final_outputs=bool(args.overwrite_final_outputs),
        defer_registry_status=bool(args.defer_registry_status),
        finalization_mode=args.finalization_mode,
        clip_finalizer_package_dir=args.clip_finalizer_package_dir,
        import_array_copy_workers=max(1, int(args.import_array_copy_workers)),
        write_sampled_component_contours=bool(args.write_sampled_component_contours)
        and not bool(args.no_write_sampled_component_contours),
    )
    payload = json_ready(plan.to_json())
    if args.plan_json:
        args.plan_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.plan_json.expanduser().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.apply:
        submission = apply_plan(plan)
        if args.json:
            print(json.dumps(submission, indent=2, sort_keys=True))
        else:
            print("Submitted clipped collection subject-mask workflow")
            print(f"  plan: {submission['plan_path']}")
            print(f"  submission: {submission['submission_path']}")
            print(f"  subject-mask shards: {len(submission['subject_mask_shard_results'])}")
            if submission.get("clip_finalizers"):
                print(f"  clip finalizer jobs: {len(submission.get('clip_finalizers') or [])}")
            if submission.get("finalizer"):
                print(f"  finalizer job: {submission['finalizer']['job_id']}")
        return 0
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
