#!/usr/bin/env python3
"""Plan clipped-collection keypoint shard jobs and finalization dependencies."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import zarr

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


def _shell_join(argv: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


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
    args = [
        "bsub",
        "-J",
        job_name,
        "-n",
        str(ncores),
        "-R",
        f"rusage[mem={mem_gb}G]",
        "-oo",
        str(log_dir / f"{job_name}.%J.out"),
        "-eo",
        str(log_dir / f"{job_name}.%J.err"),
    ]
    if queue:
        args.extend(["-q", queue])
    if gpus > 0:
        args.extend(["-gpu", f"num={gpus}"])
    if dependency:
        args.extend(["-w", dependency])
    return args


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
    recording_dir: Path
    cache_dir_root: Path
    log_dir: Path
    clips: tuple[ClipPlan, ...]
    merged_proxy_crop_run: str
    keypoint_collection_run: str
    refined_keypoints_run: str
    merge_proxy_command: list[str]
    finalize_command: list[str]
    refine_command: list[str]
    finalizer_bsub_command: list[str]
    refine_bsub_command: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "palette.clipped_collection_keypoint_bsub_plan.v1",
            "zarr_path": str(self.zarr_path),
            "collection_id": self.collection_id,
            "run_id": self.run_id,
            "run_label": self.run_label,
            "recording_dir": str(self.recording_dir),
            "cache_dir_root": str(self.cache_dir_root),
            "log_dir": str(self.log_dir),
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
    manifests = _discover_cache_manifests(
        cache_dir_root=cache_dir_root,
        collection_id=collection_id,
        clip_ids=resolved_clip_ids,
        allow_multiple=allow_multiple_cache_manifests,
    )

    clip_plans: list[ClipPlan] = []
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
            if stage_roi_cache_to_scratch:
                keypoint_command.append("--stage-roi-cache-to-scratch")
            keypoint_bsub_command = [
                *_bsub_prefix(
                    job_name=keypoint_job_name,
                    queue=queue,
                    ncores=ncores,
                    mem_gb=mem_gb,
                    gpus=gpus,
                    log_dir=resolved_log_dir,
                    dependency=None,
                ),
                "bash",
                "-lc",
                f"cd {_shell_join([repo])} && {_shell_join(keypoint_command)}",
            ]
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
    keypoint_collection_run = f"keypoints_{resolved_run_label}"
    refined_keypoints_run = f"refined_keypoints_{resolved_run_label}"
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

    shard_job_refs = [
        f"<jobid:{clip.keypoint_job_name}>"
        for clip in clip_plans
        if clip.keypoint_bsub_command is not None
    ]
    finalizer_dependency = " && ".join(f"done({name})" for name in shard_job_refs) if shard_job_refs else None
    finalizer_job_name = f"kp_finalize_{resolved_run_label}"
    refine_job_name = f"kp_refine_{resolved_run_label}"
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
    refine_bsub_command = [
        *_bsub_prefix(
            job_name=refine_job_name,
            queue=refine_queue,
            ncores=refine_ncores,
            mem_gb=refine_mem_gb,
            gpus=0,
            log_dir=resolved_log_dir,
            dependency=f"done(<jobid:{finalizer_job_name}>)",
        ),
        "bash",
        "-lc",
        f"cd {_shell_join([repo])} && {_shell_join(refine_command)}",
    ]

    return WorkflowPlan(
        zarr_path=zarr_path,
        collection_id=collection_id,
        run_id=resolved_run_id,
        run_label=resolved_run_label,
        recording_dir=recording_dir,
        cache_dir_root=cache_dir_root,
        log_dir=resolved_log_dir,
        clips=tuple(clip_plans),
        merged_proxy_crop_run=merged_proxy_crop_run,
        keypoint_collection_run=keypoint_collection_run,
        refined_keypoints_run=refined_keypoints_run,
        merge_proxy_command=merge_proxy_command,
        finalize_command=finalize_command,
        refine_command=refine_command,
        finalizer_bsub_command=finalizer_bsub_command,
        refine_bsub_command=refine_bsub_command,
    )


def _print_plan(plan: WorkflowPlan) -> None:
    print("Clipped collection keypoint workflow dry-run")
    print(f"  zarr: {plan.zarr_path}")
    print(f"  collection_id: {plan.collection_id}")
    print(f"  run_label: {plan.run_label}")
    print(f"  clips: {len(plan.clips)}")
    print(f"  log_dir: {plan.log_dir}")
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
    parser.add_argument("--device", default="0")
    parser.add_argument("--queue", default="gpu_l4")
    parser.add_argument("--ncores", type=int, default=4)
    parser.add_argument("--mem-gb", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--finalizer-queue", default="normal")
    parser.add_argument("--finalizer-ncores", type=int, default=4)
    parser.add_argument("--finalizer-mem-gb", type=int, default=16)
    parser.add_argument("--refine-queue", default="normal")
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
    parser.add_argument("--dry-run", action="store_true", help="Plan only. Required in this first implementation.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.all_clips and args.clip_id:
        raise SystemExit("Use either --all-clips or --clip-id, not both.")
    if not args.all_clips and not args.clip_id:
        raise SystemExit("Provide --all-clips or at least one --clip-id.")
    if not args.dry_run:
        raise SystemExit("Only --dry-run is implemented for this wrapper slice; refusing to submit jobs.")
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
        args.plan_json.expanduser().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
