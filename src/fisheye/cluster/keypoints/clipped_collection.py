#!/usr/bin/env python3
"""Plan clipped-collection keypoint shard jobs and finalization dependencies.

This family-owned planner remains independent from the whole-recording planner;
both consume only the small shared LSF kernel.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import zarr

from fisheye.cluster.keypoints.common import (
    DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
    DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
    build_keypoint_run_names,
    resolve_keypoint_storage,
)
from fisheye.cluster.lsf import (
    CommandRunner,
    LsfDependency,
    LsfJob,
    LsfResources,
    LsfWorkflow,
    build_bsub_command,
    build_bsub_prefix as _shared_bsub_prefix,
    parse_bsub_job_id as _parse_bsub_job_id,
    resolve_job_id_placeholders as _replace_job_placeholders,
    run_command as _run_command,
    shell_join as _shell_join,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.shared.run_provenance import json_ready


DEFAULT_GROUPS_REPO = Path("/groups/johnson/johnsonlab/jeremy/gitrepos/palette")
DEFAULT_REGISTRY = Path("/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite")


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sanitize_component(value: object, *, default: str) -> str:
    text = str(value or "").strip()
    chars = [ch if ch.isalnum() or ch in "_.-" else "_" for ch in text]
    sanitized = "".join(chars).strip("_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized or default


def _open_root(path: Path, mode: str = "r") -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode)


def _recording_dir_from_zarr(zarr_path: Path) -> Path:
    parent = zarr_path.parent
    if parent.name == "zarr":
        return parent.parent
    return parent


def _collection_clip_ids(root: zarr.Group, collection_id: str, work_unit_ids: set[str]) -> list[str]:
    collection = root["experiment_index"]["finalized_runs"][collection_id]
    selected_runs = collection.attrs.get("selected_runs", [])
    clip_ids: list[str] = []
    for row in selected_runs:
        if not isinstance(row, Mapping):
            continue
        clip_id = str(row.get("clip_id") or "").strip()
        work_unit_id = str(row.get("work_unit_id") or "").strip()
        if not clip_id:
            continue
        if work_unit_ids and work_unit_id not in work_unit_ids:
            continue
        if clip_id not in clip_ids:
            clip_ids.append(clip_id)
    return sorted(clip_ids)


def _manifest_clip_ids(payload: Mapping[str, Any]) -> list[str]:
    source = payload.get("source")
    if not isinstance(source, Mapping):
        return []
    out: list[str] = []
    bundle_child = str(source.get("bundle_child_clip_id") or "").strip()
    if bundle_child:
        out.append(bundle_child)
    selection = source.get("selection")
    if isinstance(selection, Mapping):
        for value in selection.get("clip_ids") or []:
            text = str(value or "").strip()
            if text and text not in out:
                out.append(text)
    return out


def _load_manifest(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Manifest is not a JSON object: {path}")
    return payload


def _discover_cache_manifests(
    *,
    cache_dir_root: Path,
    collection_id: str,
    clip_ids: Sequence[str],
    allow_multiple: bool = False,
) -> dict[str, Path]:
    wanted = set(clip_ids)
    matches: dict[str, list[Path]] = {clip_id: [] for clip_id in clip_ids}
    for path in sorted(cache_dir_root.rglob("*.flat_roi_cache.json")):
        payload = _load_manifest(path)
        source = payload.get("source")
        if not isinstance(source, Mapping):
            continue
        if str(source.get("collection_id") or "") != collection_id:
            continue
        if payload.get("cache_complete") is not True:
            continue
        for clip_id in _manifest_clip_ids(payload):
            if clip_id in wanted:
                matches[clip_id].append(path)

    resolved: dict[str, Path] = {}
    duplicate_errors: list[str] = []
    for clip_id, paths in matches.items():
        if not paths:
            continue
        if len(paths) > 1 and not allow_multiple:
            duplicate_errors.append(
                f"{clip_id}: " + ", ".join(str(path) for path in paths)
            )
            continue
        resolved[clip_id] = max(paths, key=lambda path: path.stat().st_mtime)
    if duplicate_errors:
        joined = "\n".join(duplicate_errors)
        raise ValueError(
            "Multiple cache manifests matched a clip. Re-run with "
            "--allow-multiple-cache-manifests to select the newest by mtime.\n"
            f"{joined}"
        )
    return resolved


def _default_alias_manifest(manifest_path: Path, proxy_run: str) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}__{proxy_run}.alias.json")


def _bsub_prefix(
    *,
    job_name: str,
    queue: str,
    ncores: int,
    mem_gb: int,
    gpus: int,
    log_dir: Path,
    dependency: str | None = None,
) -> list[str]:
    """Compatibility wrapper for subject-mask callers during staged migration."""

    return _shared_bsub_prefix(
        job_name=job_name,
        resources=LsfResources(
            queue=queue,
            ncores=ncores,
            mem_gb=mem_gb,
            gpus=gpus,
        ),
        stdout_path=log_dir / f"{job_name}.%J.out",
        stderr_path=log_dir / f"{job_name}.%J.err",
        dependency_expression=dependency,
    )


@dataclass(frozen=True)
class ClipPlan:
    clip_id: str
    cache_manifest: Path | None
    cache_status: str
    proxy_crop_run: str
    alias_manifest: Path | None
    keypoint_shard_run: str
    keypoint_job_name: str
    proxy_command: list[str] | None
    keypoint_command: list[str] | None
    keypoint_bsub_command: list[str] | None

    def to_json(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "cache_manifest": str(self.cache_manifest) if self.cache_manifest else None,
            "cache_status": self.cache_status,
            "proxy_crop_run": self.proxy_crop_run,
            "alias_manifest": str(self.alias_manifest) if self.alias_manifest else None,
            "keypoint_shard_run": self.keypoint_shard_run,
            "keypoint_job_name": self.keypoint_job_name,
            "proxy_command": self.proxy_command,
            "keypoint_command": self.keypoint_command,
            "keypoint_bsub_command": self.keypoint_bsub_command,
        }


@dataclass(frozen=True)
class WorkflowPlan:
    zarr_path: Path
    collection_id: str
    run_id: str
    run_label: str
    repo: Path
    recording_dir: Path
    cache_dir_root: Path
    log_dir: Path
    keypoint_storage: dict[str, Any]
    clips: tuple[ClipPlan, ...]
    merged_proxy_crop_run: str
    keypoint_collection_run: str
    refined_keypoints_run: str
    merge_proxy_command: list[str]
    finalize_command: list[str]
    refine_command: list[str]
    finalizer_bsub_command: list[str]
    refine_bsub_command: list[str]
    lsf_workflow: LsfWorkflow

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "palette.clipped_collection_keypoint_bsub_plan.v1",
            "zarr_path": str(self.zarr_path),
            "collection_id": self.collection_id,
            "run_id": self.run_id,
            "run_label": self.run_label,
            "repo": str(self.repo),
            "recording_dir": str(self.recording_dir),
            "cache_dir_root": str(self.cache_dir_root),
            "log_dir": str(self.log_dir),
            "keypoint_storage": self.keypoint_storage,
            "clips": [clip.to_json() for clip in self.clips],
            "merged_proxy_crop_run": self.merged_proxy_crop_run,
            "keypoint_collection_run": self.keypoint_collection_run,
            "refined_keypoints_run": self.refined_keypoints_run,
            "merge_proxy_command": self.merge_proxy_command,
            "finalize_command": self.finalize_command,
            "refine_command": self.refine_command,
            "finalizer_bsub_command": self.finalizer_bsub_command,
            "refine_bsub_command": self.refine_bsub_command,
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
    pose_schema: str,
    batch_size: int,
    keypoint_roi_shard_rows: int | None = DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
    keypoint_frame_shard_rows: int = DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
    device: str,
    queue: str,
    ncores: int,
    mem_gb: int,
    gpus: int,
    finalizer_queue: str,
    finalizer_ncores: int,
    finalizer_mem_gb: int,
    refine_queue: str,
    refine_ncores: int,
    refine_mem_gb: int,
    refine_num_workers: int,
    refine_scheduler: str,
    refine_chunk_size: int,
    stage_roi_cache_to_scratch: bool,
    allow_multiple_cache_manifests: bool,
    overwrite_proxies: bool,
    overwrite_final_outputs: bool,
) -> WorkflowPlan:
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

    resolved_run_id = _sanitize_component(run_id or _utc_run_id(), default="run")
    resolved_run_label = _sanitize_component(run_label or f"clipped_keypoints_{resolved_run_id}", default="clipped_keypoints")
    recording_dir = _recording_dir_from_zarr(zarr_path)
    resolved_log_dir = (log_dir or (recording_dir / "logs" / "clipped_collection_keypoints_bsub" / resolved_run_label)).expanduser()
    keypoint_storage = resolve_keypoint_storage(
        roi_shard_rows=keypoint_roi_shard_rows,
        frame_shard_rows=keypoint_frame_shard_rows,
    )
    manifests = _discover_cache_manifests(
        cache_dir_root=cache_dir_root,
        collection_id=collection_id,
        clip_ids=resolved_clip_ids,
        allow_multiple=allow_multiple_cache_manifests,
    )

    clip_plans: list[ClipPlan] = []
    lsf_jobs: list[LsfJob] = []
    for clip_id in resolved_clip_ids:
        safe_clip = _sanitize_component(clip_id, default="clip")
        proxy_crop_run = f"crop_proxy_{resolved_run_label}_{safe_clip}"
        keypoint_shard_run = f"keypoint_shard_{resolved_run_label}_{safe_clip}"
        keypoint_job_name = f"kp_{resolved_run_label}_{safe_clip}"
        manifest_path = manifests.get(clip_id)
        alias_manifest = _default_alias_manifest(manifest_path, proxy_crop_run) if manifest_path else None
        proxy_command: list[str] | None = None
        keypoint_command: list[str] | None = None
        keypoint_bsub_command: list[str] | None = None
        if manifest_path is not None and alias_manifest is not None:
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
            keypoint_command = [
                "scripts/py",
                "-m",
                "fisheye.utils.run_keypoints_with_registry_model",
                "--recording-dir",
                str(recording_dir),
                "--output",
                str(zarr_path),
                "--registry",
                str(registry),
                "--run-name",
                keypoint_shard_run,
                "--output-parent",
                "keypoint_shard_runs",
                "--crop-run",
                proxy_crop_run,
                "--pose-schema",
                pose_schema,
                "--batch-size",
                str(batch_size),
                "--device",
                device,
                "--roi-cache-manifest",
                str(alias_manifest),
                "--progress-jsonl",
                str(resolved_log_dir / "progress_jsonl" / f"{safe_clip}.jsonl"),
            ]
            if keypoint_roi_shard_rows is None:
                keypoint_command.append("--no-keypoint-sharding")
            else:
                keypoint_command.extend(
                    [
                        "--keypoint-roi-shard-rows",
                        str(int(keypoint_roi_shard_rows)),
                        "--keypoint-frame-shard-rows",
                        str(int(keypoint_frame_shard_rows)),
                    ]
                )
            if stage_roi_cache_to_scratch:
                keypoint_command.append("--stage-roi-cache-to-scratch")
            keypoint_lsf_job = LsfJob(
                job_key=keypoint_job_name,
                job_name=keypoint_job_name,
                command=(
                    "bash",
                    "-lc",
                    f"cd {_shell_join([repo])} && {_shell_join(keypoint_command)}",
                ),
                resources=LsfResources(
                    queue=queue,
                    ncores=ncores,
                    mem_gb=mem_gb,
                    gpus=gpus,
                ),
                stdout_path=resolved_log_dir / f"{keypoint_job_name}.%J.out",
                stderr_path=resolved_log_dir / f"{keypoint_job_name}.%J.err",
                metadata={"keypoint_storage": keypoint_storage},
            )
            lsf_jobs.append(keypoint_lsf_job)
            keypoint_bsub_command = build_bsub_command(keypoint_lsf_job)
        clip_plans.append(
            ClipPlan(
                clip_id=clip_id,
                cache_manifest=manifest_path,
                cache_status="found" if manifest_path else "missing",
                proxy_crop_run=proxy_crop_run,
                alias_manifest=alias_manifest,
                keypoint_shard_run=keypoint_shard_run,
                keypoint_job_name=keypoint_job_name,
                proxy_command=proxy_command,
                keypoint_command=keypoint_command,
                keypoint_bsub_command=keypoint_bsub_command,
            )
        )

    merged_proxy_crop_run = f"crop_proxy_{resolved_run_label}_collection"
    family_run_names = build_keypoint_run_names(resolved_run_label)
    keypoint_collection_run = family_run_names.keypoint_run
    refined_keypoints_run = family_run_names.refined_keypoint_run
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
        "fisheye.utils.finalize_keypoint_shards",
        str(zarr_path),
        "--target-crop-run",
        merged_proxy_crop_run,
        "--output-run",
        keypoint_collection_run,
        "--json",
    ]
    for clip in clip_plans:
        finalize_command.extend(["--shard-run", clip.keypoint_shard_run])
    if overwrite_final_outputs:
        merge_proxy_command.append("--overwrite")
        finalize_command.append("--overwrite")
    refine_command = [
        "scripts/py",
        "-m",
        "fisheye.refinement.refine_keypoints",
        str(zarr_path),
        "--keypoint-run",
        keypoint_collection_run,
        "--run-name",
        refined_keypoints_run,
        "--chunk-size",
        str(refine_chunk_size),
        "--scheduler",
        refine_scheduler,
        "--num-workers",
        str(refine_num_workers),
        "--no-post-audit",
    ]

    shard_job_keys = [
        clip.keypoint_job_name
        for clip in clip_plans
        if clip.keypoint_bsub_command is not None
    ]
    finalizer_job_name = f"kp_finalize_{resolved_run_label}"
    refine_job_name = f"kp_refine_{resolved_run_label}"
    finalizer_shell = f"cd {_shell_join([repo])} && {_shell_join(merge_proxy_command)} && {_shell_join(finalize_command)}"
    finalizer_lsf_job = LsfJob(
        job_key=finalizer_job_name,
        job_name=finalizer_job_name,
        command=("bash", "-lc", finalizer_shell),
        resources=LsfResources(
            queue=finalizer_queue,
            ncores=finalizer_ncores,
            mem_gb=finalizer_mem_gb,
            gpus=0,
        ),
        stdout_path=resolved_log_dir / f"{finalizer_job_name}.%J.out",
        stderr_path=resolved_log_dir / f"{finalizer_job_name}.%J.err",
        dependency=(LsfDependency(tuple(shard_job_keys)) if shard_job_keys else None),
    )
    refine_lsf_job = LsfJob(
        job_key=refine_job_name,
        job_name=refine_job_name,
        command=(
            "bash",
            "-lc",
            f"cd {_shell_join([repo])} && {_shell_join(refine_command)}",
        ),
        resources=LsfResources(
            queue=refine_queue,
            ncores=refine_ncores,
            mem_gb=refine_mem_gb,
            gpus=0,
        ),
        stdout_path=resolved_log_dir / f"{refine_job_name}.%J.out",
        stderr_path=resolved_log_dir / f"{refine_job_name}.%J.err",
        dependency=LsfDependency((finalizer_job_name,)),
    )
    finalizer_bsub_command = build_bsub_command(finalizer_lsf_job)
    refine_bsub_command = build_bsub_command(refine_lsf_job)
    lsf_jobs.extend((finalizer_lsf_job, refine_lsf_job))
    lsf_workflow = LsfWorkflow(
        workflow_id=resolved_run_label,
        family="keypoints.clipped_collection",
        jobs=tuple(lsf_jobs),
        metadata={
            "plan_schema": "palette.clipped_collection_keypoint_bsub_plan.v1",
            "collection_id": collection_id,
            "zarr_path": str(zarr_path),
            "keypoint_storage": keypoint_storage,
        },
    )

    return WorkflowPlan(
        zarr_path=zarr_path,
        collection_id=collection_id,
        run_id=resolved_run_id,
        run_label=resolved_run_label,
        repo=repo,
        recording_dir=recording_dir,
        cache_dir_root=cache_dir_root,
        log_dir=resolved_log_dir,
        keypoint_storage=keypoint_storage,
        clips=tuple(clip_plans),
        merged_proxy_crop_run=merged_proxy_crop_run,
        keypoint_collection_run=keypoint_collection_run,
        refined_keypoints_run=refined_keypoints_run,
        merge_proxy_command=merge_proxy_command,
        finalize_command=finalize_command,
        refine_command=refine_command,
        finalizer_bsub_command=finalizer_bsub_command,
        refine_bsub_command=refine_bsub_command,
        lsf_workflow=lsf_workflow,
    )


def _print_plan(plan: WorkflowPlan) -> None:
    print("Clipped collection keypoint workflow dry-run")
    print(f"  zarr: {plan.zarr_path}")
    print(f"  collection_id: {plan.collection_id}")
    print(f"  run_label: {plan.run_label}")
    print(f"  clips: {len(plan.clips)}")
    print(f"  log_dir: {plan.log_dir}")
    print(f"  keypoint_storage: {plan.keypoint_storage['effective']}")
    missing = [clip.clip_id for clip in plan.clips if clip.cache_manifest is None]
    if missing:
        print(f"  missing_cache_manifests: {', '.join(missing)}")
    print()
    print("DAG")
    for clip in plan.clips:
        print(f"  cache[{clip.clip_id}] ({clip.cache_status}) -> proxy[{clip.proxy_crop_run}] -> shard[{clip.keypoint_shard_run}]")
    shard_refs = ", ".join(clip.keypoint_shard_run for clip in plan.clips)
    print(f"  shards[{shard_refs}] -> merge_proxy[{plan.merged_proxy_crop_run}] -> keypoints[{plan.keypoint_collection_run}]")
    print(f"  keypoints[{plan.keypoint_collection_run}] -> refined_keypoints[{plan.refined_keypoints_run}]")
    print()
    print("Submit-side proxy commands")
    for clip in plan.clips:
        if clip.proxy_command is None:
            print(f"  # {clip.clip_id}: cache manifest missing; proxy command not available")
            continue
        print(f"  {_shell_join(clip.proxy_command)}")
    print()
    print("Keypoint shard bsub commands")
    for clip in plan.clips:
        if clip.keypoint_bsub_command is None:
            print(f"  # {clip.clip_id}: cache manifest missing; keypoint shard job not available")
            continue
        print(f"  {_shell_join(clip.keypoint_bsub_command)}")
    print()
    print("Finalizer bsub command template")
    print(f"  {_shell_join(plan.finalizer_bsub_command)}")
    print()
    print("Refine bsub command template")
    print(f"  {_shell_join(plan.refine_bsub_command)}")


def _write_submission_snapshot(
    path: Path,
    *,
    plan: WorkflowPlan,
    status: str,
    proxy_results: Sequence[Mapping[str, Any]],
    shard_results: Sequence[Mapping[str, Any]],
    job_ids_by_name: Mapping[str, str],
    finalizer: Mapping[str, Any] | None = None,
    refine: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "palette.clipped_collection_keypoint_bsub_submission.v1",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(path.with_name("submission_plan.json")),
        "zarr_path": str(plan.zarr_path),
        "collection_id": plan.collection_id,
        "run_id": plan.run_id,
        "run_label": plan.run_label,
        "log_dir": str(plan.log_dir),
        "proxy_results": list(proxy_results),
        "keypoint_shard_results": list(shard_results),
        "job_ids_by_name": dict(job_ids_by_name),
    }
    if finalizer is not None:
        payload["finalizer"] = dict(finalizer)
    if refine is not None:
        payload["refine"] = dict(refine)
    if error is not None:
        payload["error"] = dict(error)
    return write_json_snapshot(path, payload)


def apply_plan(plan: WorkflowPlan, *, runner: CommandRunner = subprocess.run) -> dict[str, Any]:
    """Create proxy runs and submit the planned clipped keypoint LSF DAG."""

    missing = [clip.clip_id for clip in plan.clips if clip.cache_manifest is None]
    if missing:
        raise ValueError(f"Cannot apply plan; missing cache manifests for clips: {missing}")
    unavailable = [clip.clip_id for clip in plan.clips if clip.proxy_command is None or clip.keypoint_bsub_command is None]
    if unavailable:
        raise ValueError(f"Cannot apply plan; clips are missing proxy or keypoint commands: {unavailable}")

    plan.log_dir.mkdir(parents=True, exist_ok=True)
    (plan.log_dir / "progress_jsonl").mkdir(parents=True, exist_ok=True)
    plan_payload = json_ready(plan.to_json())
    plan_path = plan.log_dir / "submission_plan.json"
    submission_path = plan.log_dir / "submission.json"
    write_json_snapshot(plan_path, plan_payload)

    proxy_results: list[dict[str, Any]] = []
    shard_results: list[dict[str, Any]] = []
    job_ids_by_name: dict[str, str] = {}
    finalizer_payload: dict[str, Any] | None = None
    refine_payload: dict[str, Any] | None = None

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

        clips_by_job_name = {clip.keypoint_job_name: clip for clip in plan.clips}
        finalizer_job_name = f"kp_finalize_{plan.run_label}"
        refine_job_name = f"kp_refine_{plan.run_label}"

        def record_submitted_job(record: Mapping[str, Any]) -> None:
            nonlocal finalizer_payload, refine_payload
            job_name = str(record["job_key"])
            job_id = str(record["job_id"])
            command = [str(value) for value in record["bsub_command"]]
            command_result = dict(record["command_result"])
            job_ids_by_name[job_name] = job_id
            clip = clips_by_job_name.get(job_name)
            if clip is not None:
                shard_results.append(
                    {
                        "clip_id": clip.clip_id,
                        "keypoint_job_name": clip.keypoint_job_name,
                        "keypoint_job_id": job_id,
                        "keypoint_shard_run": clip.keypoint_shard_run,
                        "command_result": command_result,
                    }
                )
            elif job_name == finalizer_job_name:
                finalizer_payload = {
                    "job_name": finalizer_job_name,
                    "job_id": job_id,
                    "command": command,
                    "command_result": command_result,
                    "output_run": plan.keypoint_collection_run,
                    "merged_proxy_crop_run": plan.merged_proxy_crop_run,
                }
            elif job_name == refine_job_name:
                refine_payload = {
                    "job_name": refine_job_name,
                    "job_id": job_id,
                    "command": command,
                    "command_result": command_result,
                    "output_run": plan.refined_keypoints_run,
                }
            else:
                raise ValueError(
                    f"Shared LSF workflow submitted an unknown keypoint job: {job_name}"
                )
            _write_submission_snapshot(
                submission_path,
                plan=plan,
                status="submitting",
                proxy_results=proxy_results,
                shard_results=shard_results,
                job_ids_by_name=job_ids_by_name,
                finalizer=finalizer_payload,
                refine=refine_payload,
            )

        submit_lsf_workflow(
            plan.lsf_workflow,
            cwd=plan.repo,
            plan_path=plan.log_dir / "lsf_plan.json",
            submission_path=plan.log_dir / "lsf_submission.json",
            runner=runner,
            on_job_submitted=record_submitted_job,
        )
    except Exception as exc:
        _write_submission_snapshot(
            submission_path,
            plan=plan,
            status="failed",
            proxy_results=proxy_results,
            shard_results=shard_results,
            job_ids_by_name=job_ids_by_name,
            finalizer=finalizer_payload,
            refine=refine_payload,
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
        refine=refine_payload,
    )
    submission["submitted_at_utc"] = submission["updated_at_utc"]
    write_json_snapshot(submission_path, submission)
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
    parser.add_argument("--pose-schema", default="traditional_v2")
    parser.add_argument("--batch-size-kp", type=int, default=256)
    keypoint_storage_group = parser.add_mutually_exclusive_group()
    keypoint_storage_group.add_argument(
        "--keypoint-roi-shard-rows",
        type=int,
        default=DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
        help=(
            "Outer rows for indexed-sharded ROI arrays "
            f"(default: {DEFAULT_KEYPOINT_ROI_SHARD_ROWS})."
        ),
    )
    keypoint_storage_group.add_argument(
        "--no-keypoint-sharding",
        action="store_const",
        dest="keypoint_roi_shard_rows",
        const=None,
        help="Use ordinary chunks for keypoint shard outputs.",
    )
    parser.add_argument(
        "--keypoint-frame-shard-rows",
        type=int,
        default=DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
        help=(
            "Outer rows for indexed-sharded frame arrays "
            f"(default: {DEFAULT_KEYPOINT_FRAME_SHARD_ROWS})."
        ),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--queue", default="gpu_l4")
    parser.add_argument("--ncores", type=int, default=4)
    parser.add_argument("--mem-gb", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--finalizer-queue", default="short")
    parser.add_argument("--finalizer-ncores", type=int, default=4)
    parser.add_argument("--finalizer-mem-gb", type=int, default=16)
    parser.add_argument("--refine-queue", default="short")
    parser.add_argument("--refine-ncores", type=int, default=4)
    parser.add_argument("--refine-mem-gb", type=int, default=16)
    parser.add_argument("--refine-num-workers", type=int, default=4)
    parser.add_argument("--refine-scheduler", default="threads")
    parser.add_argument("--refine-chunk-size", type=int, default=2048)
    parser.add_argument("--no-stage-roi-cache-to-scratch", action="store_true")
    parser.add_argument("--allow-multiple-cache-manifests", action="store_true")
    parser.add_argument("--overwrite-proxies", action="store_true")
    parser.add_argument("--overwrite-final-outputs", action="store_true")
    parser.add_argument("--plan-json", type=Path, help="Optional path to write JSON plan.")
    parser.add_argument("--json", action="store_true", help="Print JSON plan instead of text.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Plan only; no jobs are submitted.")
    mode.add_argument("--apply", action="store_true", help="Create proxy runs and submit the LSF DAG.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.all_clips and args.clip_id:
        raise SystemExit("Use either --all-clips or --clip-id, not both.")
    if not args.all_clips and not args.clip_id:
        raise SystemExit("Provide --all-clips or at least one --clip-id.")
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
        pose_schema=args.pose_schema,
        batch_size=int(args.batch_size_kp),
        keypoint_roi_shard_rows=args.keypoint_roi_shard_rows,
        keypoint_frame_shard_rows=int(args.keypoint_frame_shard_rows),
        device=args.device,
        queue=args.queue,
        ncores=int(args.ncores),
        mem_gb=int(args.mem_gb),
        gpus=int(args.gpus),
        finalizer_queue=args.finalizer_queue,
        finalizer_ncores=int(args.finalizer_ncores),
        finalizer_mem_gb=int(args.finalizer_mem_gb),
        refine_queue=args.refine_queue,
        refine_ncores=int(args.refine_ncores),
        refine_mem_gb=int(args.refine_mem_gb),
        refine_num_workers=int(args.refine_num_workers),
        refine_scheduler=args.refine_scheduler,
        refine_chunk_size=int(args.refine_chunk_size),
        stage_roi_cache_to_scratch=not bool(args.no_stage_roi_cache_to_scratch),
        allow_multiple_cache_manifests=bool(args.allow_multiple_cache_manifests),
        overwrite_proxies=bool(args.overwrite_proxies),
        overwrite_final_outputs=bool(args.overwrite_final_outputs),
    )
    payload = json_ready(plan.to_json())
    if args.plan_json:
        write_json_snapshot(args.plan_json.expanduser(), payload)
    if args.apply:
        submission = apply_plan(plan)
        if args.json:
            print(json.dumps(submission, indent=2, sort_keys=True))
        else:
            print("Submitted clipped collection keypoint workflow")
            print(f"  plan: {submission['plan_path']}")
            print(f"  submission: {submission['submission_path']}")
            print(f"  keypoint shards: {len(submission['keypoint_shard_results'])}")
            print(f"  finalizer job: {submission['finalizer']['job_id']}")
            print(f"  refine job: {submission['refine']['job_id']}")
        return 0
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
