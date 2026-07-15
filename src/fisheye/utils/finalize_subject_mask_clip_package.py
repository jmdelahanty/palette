#!/usr/bin/env python3
"""Finalize one clipped subject-mask shard into an NRS handoff package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import socket
import tarfile
import time
from typing import Any, Sequence

import zarr

from fisheye.refinement.finalize_subject_masks import finalize_subject_mask_run
from fisheye.shared.refined_subject_mask_encoded_chunks import (
    ENCODED_MASK_PAYLOAD_NAME,
    ENCODED_PACKAGE_SCHEMA_ID,
    build_global_encoded_mask_payload,
)
from fisheye.shared.run_provenance import json_ready


PACKAGE_SCHEMA_ID = "palette_refined_subject_mask_clip_package_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_staging_root() -> Path:
    base = os.environ.get("TMPDIR") or os.environ.get("LOCAL_SCRATCH") or "/tmp"
    job_id = os.environ.get("LSB_JOBID") or str(os.getpid())
    job_index = os.environ.get("LSB_JOBINDEX")
    work_unit = f"{job_id}_{job_index}" if job_index else job_id
    return Path(base) / f"palette_refined_subject_mask_clip_package_{work_unit}"


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _stage_zarr_with_local_refined_parent(
    *,
    source_zarr: Path,
    staging_root: Path,
    staging_name: str,
    overwrite: bool,
) -> Path:
    source_zarr = source_zarr.expanduser().resolve()
    staging_root = staging_root.expanduser().resolve()
    staged_zarr = staging_root / f"{staging_name}.zarr"
    if staged_zarr.exists() or staged_zarr.is_symlink():
        if not overwrite:
            raise ValueError(f"Staging zarr already exists: {staged_zarr}")
        _remove_path(staged_zarr)
    staging_root.mkdir(parents=True, exist_ok=True)
    staged_zarr.mkdir(parents=True)

    root_meta = source_zarr / "zarr.json"
    if not root_meta.is_file():
        raise ValueError(f"Source zarr is missing root zarr.json: {source_zarr}")
    shutil.copy2(root_meta, staged_zarr / "zarr.json")

    for child in source_zarr.iterdir():
        if child.name in {"zarr.json", "refined_subject_masks_runs"}:
            continue
        os.symlink(child, staged_zarr / child.name, target_is_directory=child.is_dir())
    return staged_zarr


def _write_package(
    *,
    staged_zarr: Path,
    refined_run: str,
    package_path: Path,
    metadata: dict[str, Any],
    overwrite: bool,
    schema_id: str = PACKAGE_SCHEMA_ID,
    encoded_payload_path: Path | None = None,
) -> dict[str, Any]:
    run_path = staged_zarr / "refined_subject_masks_runs" / refined_run
    if not run_path.is_dir():
        raise ValueError(f"Finalized run group is missing: {run_path}")

    package_path = package_path.expanduser().resolve()
    if package_path.exists() and not overwrite:
        raise ValueError(f"Package already exists: {package_path}")
    package_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = package_path.with_name(f".{package_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    manifest = {
        **metadata,
        "schema_id": str(schema_id),
        "package_completion_status": "complete",
        "created_at_utc": _utc_now(),
        "package_path": str(package_path),
        "run_group_path": f"refined_subject_masks_runs/{refined_run}",
    }
    manifest_bytes = (json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(run_path, arcname=f"refined_subject_masks_runs/{refined_run}")
            if encoded_payload_path is not None:
                if not encoded_payload_path.is_dir():
                    raise ValueError(f"Encoded mask payload is missing: {encoded_payload_path}")
                tar.add(encoded_payload_path, arcname=ENCODED_MASK_PAYLOAD_NAME)
            info = tarfile.TarInfo("package.json")
            info.size = len(manifest_bytes)
            info.mtime = int(time.time())
            tar.addfile(info, BytesIO(manifest_bytes))
        os.replace(tmp_path, package_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "schema_id": str(schema_id),
        "artifact_path": str(package_path),
        "run_group_path": f"refined_subject_masks_runs/{refined_run}",
        "created_at_utc": manifest["created_at_utc"],
        "size_bytes": int(package_path.stat().st_size),
    }


def finalize_subject_mask_clip_package(
    *,
    source_zarr: Path,
    subject_shard_run: str,
    target_crop_run: str,
    refined_run: str,
    package_path: Path,
    staging_root: Path | None = None,
    components: Sequence[str] | None = None,
    chunk_size: int = 256,
    metric_level: str = "cheap",
    mask_storage: str = "dense_and_bitpacked",
    mask_rle_validation_mode: str = "invariants",
    dense_mask_row_chunk: int | None = None,
    execution_backend: str = "process_shards",
    num_workers: int | None = None,
    postcompute_backend: str = "process_shards",
    postcompute_num_workers: int | None = None,
    postcompute_chunk_size: int | None = None,
    assignment_keypoint_group: str | None = None,
    assignment_keypoints_run: str | None = None,
    write_eye_geometry: bool = True,
    write_component_contours: bool = True,
    write_sampled_component_contours: bool = True,
    retain_source_seeds: bool = False,
    global_mask_grid_manifest: Path | None = None,
    encoded_mask_copy_workers: int = 8,
    overwrite: bool = False,
    cleanup: bool = True,
) -> dict[str, Any]:
    source_zarr = source_zarr.expanduser().resolve()
    staging_root = staging_root or _default_staging_root()
    safe_run = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in refined_run).strip("_") or "refined"
    staged_zarr = _stage_zarr_with_local_refined_parent(
        source_zarr=source_zarr,
        staging_root=staging_root,
        staging_name=safe_run,
        overwrite=True,
    )
    started = time.perf_counter()
    try:
        root = zarr.open_group(str(staged_zarr), mode="a", use_consolidated=False)
        summary = finalize_subject_mask_run(
            root,
            zarr_path=staged_zarr,
            subject_shard_runs=[subject_shard_run],
            target_crop_run=target_crop_run,
            refined_run=refined_run,
            components=components,
            chunk_size=int(chunk_size),
            metric_level=metric_level,
            mask_storage=mask_storage,
            mask_rle_validation_mode=mask_rle_validation_mode,
            dense_mask_row_chunk=dense_mask_row_chunk,
            execution_backend=execution_backend,
            num_workers=num_workers,
            postcompute_backend=postcompute_backend,
            postcompute_num_workers=postcompute_num_workers,
            postcompute_chunk_size=postcompute_chunk_size,
            assignment_keypoint_group=assignment_keypoint_group,
            assignment_keypoints_run=assignment_keypoints_run,
            write_eye_geometry=bool(write_eye_geometry),
            write_component_contours=bool(write_component_contours),
            write_sampled_component_contours=bool(write_sampled_component_contours),
            retain_source_seeds=bool(retain_source_seeds),
            overwrite=bool(overwrite),
        )
        run = root["refined_subject_masks_runs"][refined_run]
        run.attrs["clip_package_source_zarr_path"] = str(source_zarr)
        run.attrs["clip_package_staged_zarr_path"] = str(staged_zarr)
        run.attrs["clip_package_subject_shard_run"] = str(subject_shard_run)
        run.attrs["clip_package_target_crop_run"] = str(target_crop_run)
        run.attrs["clip_package_host"] = socket.gethostname()
        run.attrs["clip_package_lsb_jobid"] = os.environ.get("LSB_JOBID")
        run.attrs["clip_package_lsb_jobindex"] = os.environ.get("LSB_JOBINDEX")
        encoded_payload_summary: dict[str, Any] | None = None
        encoded_payload_path: Path | None = None
        package_schema_id = PACKAGE_SCHEMA_ID
        if global_mask_grid_manifest is not None:
            encoded_payload_path = staged_zarr / ENCODED_MASK_PAYLOAD_NAME
            encoded_payload_summary = build_global_encoded_mask_payload(
                run_path=staged_zarr / "refined_subject_masks_runs" / refined_run,
                grid_manifest_path=global_mask_grid_manifest,
                payload_path=encoded_payload_path,
                copy_workers=int(encoded_mask_copy_workers),
            )
            package_schema_id = ENCODED_PACKAGE_SCHEMA_ID
            run.attrs["encoded_global_masks_roi"] = dict(json_ready(encoded_payload_summary))
        package = _write_package(
            staged_zarr=staged_zarr,
            refined_run=refined_run,
            package_path=package_path,
            metadata={
                "source_zarr_path": str(source_zarr),
                "staged_zarr_path": str(staged_zarr),
                "subject_shard_run": str(subject_shard_run),
                "target_crop_run": str(target_crop_run),
                "host": socket.gethostname(),
                "lsb_jobid": os.environ.get("LSB_JOBID"),
                "lsb_jobindex": os.environ.get("LSB_JOBINDEX"),
                "summary": summary,
                "encoded_global_masks_roi": encoded_payload_summary,
            },
            overwrite=bool(overwrite),
            schema_id=package_schema_id,
            encoded_payload_path=encoded_payload_path,
        )
        run.attrs["cluster_run_package"] = dict(package)
        duration_seconds = float(time.perf_counter() - started)
        return {
            "schema_id": package_schema_id,
            "status": "ok",
            "source_zarr_path": str(source_zarr),
            "staged_zarr_path": str(staged_zarr),
            "subject_shard_run": str(subject_shard_run),
            "target_crop_run": str(target_crop_run),
            "refined_run": str(refined_run),
            "package": package,
            "duration_seconds": duration_seconds,
            "summary": summary,
            "encoded_global_masks_roi": encoded_payload_summary,
            "cleanup": bool(cleanup),
        }
    finally:
        if cleanup:
            _remove_path(staged_zarr)
            try:
                staging_root.rmdir()
            except OSError:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zarr", required=True, type=Path)
    parser.add_argument("--subject-shard-run", required=True)
    parser.add_argument("--target-crop-run", required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--package-path", required=True, type=Path)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--component", dest="components", action="append")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--metric-level", default="cheap")
    parser.add_argument("--mask-storage", default="dense_and_bitpacked")
    parser.add_argument("--mask-rle-validation-mode", default="invariants")
    parser.add_argument("--dense-mask-row-chunk", type=int)
    parser.add_argument("--execution-backend", choices=("serial_driver", "process_shards"), default="process_shards")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--postcompute-backend", default="process_shards")
    parser.add_argument("--postcompute-num-workers", type=int)
    parser.add_argument("--postcompute-chunk-size", type=int)
    parser.add_argument("--assignment-keypoint-group")
    parser.add_argument("--assignment-keypoints-run")
    parser.add_argument("--no-write-eye-geometry", action="store_true")
    parser.add_argument("--no-write-component-contours", action="store_true")
    parser.add_argument("--write-sampled-component-contours", action="store_true", default=True)
    parser.add_argument("--no-write-sampled-component-contours", action="store_true")
    parser.add_argument("--retain-source-seeds", action="store_true")
    parser.add_argument("--global-mask-grid-manifest", type=Path)
    parser.add_argument("--encoded-mask-copy-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = finalize_subject_mask_clip_package(
        source_zarr=args.source_zarr,
        subject_shard_run=args.subject_shard_run,
        target_crop_run=args.target_crop_run,
        refined_run=args.refined_run,
        package_path=args.package_path,
        staging_root=args.staging_root,
        components=args.components,
        chunk_size=int(args.chunk_size),
        metric_level=args.metric_level,
        mask_storage=args.mask_storage,
        mask_rle_validation_mode=args.mask_rle_validation_mode,
        dense_mask_row_chunk=args.dense_mask_row_chunk,
        execution_backend=args.execution_backend,
        num_workers=args.num_workers,
        postcompute_backend=args.postcompute_backend,
        postcompute_num_workers=args.postcompute_num_workers,
        postcompute_chunk_size=args.postcompute_chunk_size,
        assignment_keypoint_group=args.assignment_keypoint_group,
        assignment_keypoints_run=args.assignment_keypoints_run,
        write_eye_geometry=not bool(args.no_write_eye_geometry),
        write_component_contours=not bool(args.no_write_component_contours),
        write_sampled_component_contours=bool(args.write_sampled_component_contours)
        and not bool(args.no_write_sampled_component_contours),
        retain_source_seeds=bool(args.retain_source_seeds),
        global_mask_grid_manifest=args.global_mask_grid_manifest,
        encoded_mask_copy_workers=int(args.encoded_mask_copy_workers),
        overwrite=bool(args.overwrite),
        cleanup=not bool(args.no_cleanup),
    )
    if args.json:
        print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    else:
        print(f"Finalized {result['refined_run']} -> {result['package']['artifact_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
