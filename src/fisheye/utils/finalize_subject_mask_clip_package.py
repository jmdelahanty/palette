#!/usr/bin/env python3
"""Finalize one clipped subject-mask shard into an NRS handoff package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tarfile
import time
from typing import Any, Sequence

import numpy as np
import zarr

from fisheye.refinement.finalize_subject_masks import finalize_subject_mask_run
from fisheye.refinement.finalize_subject_masks import (
    REFINED_SUBJECT_MASK_ATTEMPT_ATTR,
    REFINED_SUBJECT_MASK_SCIENTIFIC_IDENTITY_ATTR,
    REFINED_SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_ATTR,
)
from fisheye.shared.subject_mask_attempt import (
    validate_subject_mask_attempt,
    validate_subject_mask_scientific_identity,
)
from fisheye.shared.subject_mask_worker_receipt import (
    RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    validate_subject_mask_worker_semantic_receipt,
)
from fisheye.shared.refined_subject_mask_encoded_chunks import (
    ENCODED_MASK_PAYLOAD_NAME,
    ENCODED_PACKAGE_SCHEMA_ID,
    build_global_encoded_mask_payload,
)
from fisheye.shared.run_provenance import json_ready
from fisheye.shared.zarr.subject_mask_final_layout_units import (
    build_subject_mask_final_layout_unit_package,
)
from fisheye.shared.zarr.subject_mask_quality_partition import (
    compute_subject_mask_quality_partition,
)
from fisheye.shared.zarr.subject_mask_sampled_contour_worker_receipt import (
    write_subject_mask_sampled_contour_worker_receipt,
)
from fisheye.shared.zarr.subject_mask_schema import SubjectMaskDimensions

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


def _refined_worker_proof(staged_zarr: Path, run: zarr.Group) -> dict[str, Any]:
    science = run.attrs.get(REFINED_SUBJECT_MASK_SCIENTIFIC_IDENTITY_ATTR)
    attempt = run.attrs.get(REFINED_SUBJECT_MASK_ATTEMPT_ATTR)
    binding = run.attrs.get(REFINED_SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_ATTR)
    if not isinstance(science, dict) or validate_subject_mask_scientific_identity(
        science
    ):
        raise RuntimeError("Refined clip output lacks valid scientific identity.")
    if not isinstance(attempt, dict) or validate_subject_mask_attempt(attempt):
        raise RuntimeError("Refined clip output lacks valid attempt identity.")
    if not isinstance(binding, dict):
        raise RuntimeError("Refined clip output lacks semantic receipt binding.")
    relative = str(binding.get("relative_path") or "")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError("Refined clip semantic receipt path is unsafe.")
    receipt_bytes = (staged_zarr / relative).read_bytes()
    document_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if document_sha256 != binding.get("document_sha256"):
        raise RuntimeError("Refined clip semantic receipt document changed.")
    receipt = json.loads(receipt_bytes)
    validate_subject_mask_worker_semantic_receipt(
        receipt,
        scientific_identity=science,
        attempt=attempt,
        required_paths=REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    )
    if receipt.get("payload_digest") != binding.get("payload_digest"):
        raise RuntimeError("Refined clip semantic receipt payload changed.")
    return {
        "scientific_identity_digest": science["digest"],
        "attempt_id": attempt["payload"]["attempt_id"],
        "attempt_payload_digest": attempt["payload_digest"],
        "semantic_receipt_payload_digest": receipt["payload_digest"],
        "semantic_receipt_document_sha256": document_sha256,
        "semantic_receipt_relative_path": relative,
    }


def _worker_receipt(
    archive: Path,
    run: zarr.Group,
    *,
    required_paths: Sequence[str],
) -> dict[str, Any]:
    """Load and deeply validate one exact producer receipt."""

    science = run.attrs.get(REFINED_SUBJECT_MASK_SCIENTIFIC_IDENTITY_ATTR)
    attempt = run.attrs.get(REFINED_SUBJECT_MASK_ATTEMPT_ATTR)
    binding = run.attrs.get(REFINED_SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_ATTR)
    if not isinstance(science, dict) or validate_subject_mask_scientific_identity(
        science
    ):
        raise RuntimeError(f"{run.path} lacks valid scientific identity.")
    if not isinstance(attempt, dict) or validate_subject_mask_attempt(attempt):
        raise RuntimeError(f"{run.path} lacks valid attempt identity.")
    if not isinstance(binding, dict):
        raise RuntimeError(f"{run.path} lacks a semantic receipt binding.")
    relative = str(binding.get("relative_path") or "")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError(f"{run.path} semantic receipt path is unsafe.")
    receipt_bytes = (archive / relative).read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != binding.get("document_sha256"):
        raise RuntimeError(f"{run.path} semantic receipt document changed.")
    receipt = json.loads(receipt_bytes)
    validate_subject_mask_worker_semantic_receipt(
        receipt,
        scientific_identity=science,
        attempt=attempt,
        required_paths=tuple(required_paths),
    )
    if receipt.get("payload_digest") != binding.get("payload_digest"):
        raise RuntimeError(f"{run.path} semantic receipt payload changed.")
    return receipt


def _build_publication_evidence(
    *,
    root: zarr.Group,
    staged_zarr: Path,
    raw_run: zarr.Group,
    refined_run: zarr.Group,
    crop_run: zarr.Group,
    destination: Path,
    producer_commit: str,
    work_unit_id: str,
    work_unit_index: int,
    source_clip_id: str,
    source_clip_index: int,
    global_frame_start: int,
    global_frame_stop: int,
    quality_compute_workers: int,
) -> dict[str, Any]:
    """Seal all immutable inputs needed by receipt-composed publication."""

    del root
    raw_rows = np.asarray(raw_run["source_crop_row_ids"][:], dtype=np.int64)
    refined_rows = np.asarray(refined_run["source_crop_row_ids"][:], dtype=np.int64)
    if (
        raw_rows.size == 0
        or not np.array_equal(raw_rows, refined_rows)
        or np.any(np.diff(raw_rows) != 1)
    ):
        raise RuntimeError(
            "Publication evidence requires matching contiguous raw/refined crop rows."
        )
    global_start_row = int(raw_rows[0])
    n_rois = int(crop_run["instance_key"].shape[0])
    n_frames = int(crop_run["frame_row_offsets"].shape[0]) - 1
    raw_receipt = _worker_receipt(
        staged_zarr,
        raw_run,
        required_paths=RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    )
    refined_receipt = _worker_receipt(
        staged_zarr,
        refined_run,
        required_paths=REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    )
    raw_values = raw_run["mask_probs_roi"]
    refined_values = refined_run["masks_roi"]
    raw_dimensions = SubjectMaskDimensions(
        n_frames=n_frames,
        n_rois=n_rois,
        n_channels=int(raw_values.shape[1]),
        roi_height=int(raw_values.shape[2]),
        roi_width=int(raw_values.shape[3]),
    )
    refined_dimensions = SubjectMaskDimensions(
        n_frames=n_frames,
        n_rois=n_rois,
        n_channels=int(refined_values.shape[1]),
        roi_height=int(refined_values.shape[2]),
        roi_width=int(refined_values.shape[3]),
    )
    destination.mkdir(parents=True, exist_ok=False)
    raw_package = build_subject_mask_final_layout_unit_package(
        source_array=raw_values,
        source_crop_row_ids=raw_run["source_crop_row_ids"],
        destination=destination / "raw_final_layout_unit",
        kind="raw_probability_uint8",
        dimensions=raw_dimensions,
        global_start_row=global_start_row,
        source_run_path=str(raw_run.path).strip("/"),
        worker_receipt_payload_digest=str(raw_receipt["payload_digest"]),
        producer_commit=producer_commit,
        worker_array_validation_record=raw_receipt["payload"]["arrays"][
            "mask_probs_roi"
        ],
    )
    refined_package = build_subject_mask_final_layout_unit_package(
        source_array=refined_values,
        source_crop_row_ids=refined_run["source_crop_row_ids"],
        destination=destination / "refined_final_layout_unit",
        kind="refined_dense_core",
        dimensions=refined_dimensions,
        global_start_row=global_start_row,
        source_run_path=str(refined_run.path).strip("/"),
        worker_receipt_payload_digest=str(refined_receipt["payload_digest"]),
        producer_commit=producer_commit,
        worker_array_validation_record=refined_receipt["payload"]["arrays"][
            "masks_roi"
        ],
    )
    contour_receipt = write_subject_mask_sampled_contour_worker_receipt(
        refined_run,
        destination=destination / "sampled_contour_receipt.json",
        global_start_row=global_start_row,
        worker_receipt=refined_receipt,
        producer_commit=producer_commit,
    )
    source_frames = np.asarray(
        crop_run["source_acquisition_frame_index"][raw_rows], dtype=np.int64
    )

    class _QualityRun(dict[str, object]):
        pass

    quality_run = _QualityRun(
        masks_roi=refined_run["masks_roi"],
        available_channels=refined_run["available_channels"],
        instance_key=crop_run["instance_key"][raw_rows],
    )
    quality_run.path = refined_run.path  # type: ignore[attr-defined]
    quality_run.attrs = refined_run.attrs  # type: ignore[attr-defined]
    quality = compute_subject_mask_quality_partition(
        quality_run,
        source_acquisition_frame_index=source_frames,
        global_start_row=global_start_row,
        global_frame_start=int(global_frame_start),
        global_frame_stop=int(global_frame_stop),
        work_unit_id=work_unit_id,
        work_unit_index=int(work_unit_index),
        source_worker_receipt=refined_receipt,
        producer_commit=producer_commit,
        destination=destination / "quality_partition",
        compute_workers=int(quality_compute_workers),
    )
    return {
        "schema_id": "palette.subject_mask.clip_publication_evidence",
        "schema_version": 1,
        "producer_commit": producer_commit,
        "work_unit_id": work_unit_id,
        "work_unit_index": int(work_unit_index),
        "source_clip_id": source_clip_id,
        "source_clip_index": int(source_clip_index),
        "global_frame_interval": {
            "start_frame": int(global_frame_start),
            "stop_frame": int(global_frame_stop),
        },
        "global_row_interval": {
            "start_row": global_start_row,
            "stop_row": global_start_row + int(raw_rows.size),
        },
        "raw_final_layout_payload_digest": raw_package["payload_digest"],
        "refined_final_layout_payload_digest": refined_package["payload_digest"],
        "sampled_contour_payload_digest": contour_receipt["payload_digest"],
        "quality_partition_payload_digest": quality["payload_digest"],
    }


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
    publication_evidence_path: Path | None = None,
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
    manifest_bytes = (
        json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(run_path, arcname=f"refined_subject_masks_runs/{refined_run}")
            if encoded_payload_path is not None:
                if not encoded_payload_path.is_dir():
                    raise ValueError(
                        f"Encoded mask payload is missing: {encoded_payload_path}"
                    )
                tar.add(encoded_payload_path, arcname=ENCODED_MASK_PAYLOAD_NAME)
            if publication_evidence_path is not None:
                if not publication_evidence_path.is_dir():
                    raise ValueError(
                        "Subject-mask publication evidence directory is missing: "
                        f"{publication_evidence_path}"
                    )
                tar.add(publication_evidence_path, arcname="publication_evidence")
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
    require_production_proof: bool = False,
    publication_evidence_producer_commit: str | None = None,
    work_unit_id: str | None = None,
    work_unit_index: int | None = None,
    source_clip_id: str | None = None,
    source_clip_index: int | None = None,
    global_frame_start: int | None = None,
    global_frame_stop: int | None = None,
    quality_compute_workers: int = 4,
    overwrite: bool = False,
    cleanup: bool = True,
) -> dict[str, Any]:
    source_zarr = source_zarr.expanduser().resolve()
    staging_root = staging_root or _default_staging_root()
    safe_run = (
        "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in refined_run).strip(
            "_"
        )
        or "refined"
    )
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
            require_production_proof=bool(require_production_proof),
        )
        run = root["refined_subject_masks_runs"][refined_run]
        worker_proof = None
        if require_production_proof:
            if run.attrs.get("stage_selector_eligible") is not False:
                raise RuntimeError(
                    "Refined clip package output must be selector-ineligible."
                )
            worker_proof = _refined_worker_proof(staged_zarr, run)
        run.attrs["clip_package_source_zarr_path"] = str(source_zarr)
        run.attrs["clip_package_staged_zarr_path"] = str(staged_zarr)
        run.attrs["clip_package_subject_shard_run"] = str(subject_shard_run)
        run.attrs["clip_package_target_crop_run"] = str(target_crop_run)
        run.attrs["clip_package_host"] = socket.gethostname()
        run.attrs["clip_package_lsb_jobid"] = os.environ.get("LSB_JOBID")
        run.attrs["clip_package_lsb_jobindex"] = os.environ.get("LSB_JOBINDEX")
        encoded_payload_summary: dict[str, Any] | None = None
        encoded_payload_path: Path | None = None
        publication_evidence_summary: dict[str, Any] | None = None
        publication_evidence_path: Path | None = None
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
            run.attrs["encoded_global_masks_roi"] = dict(
                json_ready(encoded_payload_summary)
            )
        if publication_evidence_producer_commit is not None:
            if not require_production_proof:
                raise ValueError(
                    "Publication evidence requires --require-production-proof."
                )
            required = {
                "work_unit_id": work_unit_id,
                "work_unit_index": work_unit_index,
                "source_clip_id": source_clip_id,
                "source_clip_index": source_clip_index,
                "global_frame_start": global_frame_start,
                "global_frame_stop": global_frame_stop,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "Publication evidence lacks exact work-unit fields: "
                    + ", ".join(missing)
                )
            publication_evidence_path = staged_zarr / "publication_evidence"
            publication_evidence_summary = _build_publication_evidence(
                root=root,
                staged_zarr=staged_zarr,
                raw_run=root["subject_mask_shard_runs"][subject_shard_run],
                refined_run=run,
                crop_run=root["crop_runs"][target_crop_run],
                destination=publication_evidence_path,
                producer_commit=str(publication_evidence_producer_commit),
                work_unit_id=str(work_unit_id),
                work_unit_index=int(work_unit_index),
                source_clip_id=str(source_clip_id),
                source_clip_index=int(source_clip_index),
                global_frame_start=int(global_frame_start),
                global_frame_stop=int(global_frame_stop),
                quality_compute_workers=int(quality_compute_workers),
            )
            run.attrs["clip_publication_evidence"] = dict(
                json_ready(publication_evidence_summary)
            )
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
                "worker_proof": worker_proof,
                "encoded_global_masks_roi": encoded_payload_summary,
                "publication_evidence": publication_evidence_summary,
            },
            overwrite=bool(overwrite),
            schema_id=package_schema_id,
            encoded_payload_path=encoded_payload_path,
            publication_evidence_path=publication_evidence_path,
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
            "worker_proof": worker_proof,
            "encoded_global_masks_roi": encoded_payload_summary,
            "publication_evidence": publication_evidence_summary,
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
    parser.add_argument(
        "--execution-backend",
        choices=("serial_driver", "process_shards"),
        default="process_shards",
    )
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--postcompute-backend", default="process_shards")
    parser.add_argument("--postcompute-num-workers", type=int)
    parser.add_argument("--postcompute-chunk-size", type=int)
    parser.add_argument("--assignment-keypoint-group")
    parser.add_argument("--assignment-keypoints-run")
    parser.add_argument("--no-write-eye-geometry", action="store_true")
    parser.add_argument("--no-write-component-contours", action="store_true")
    parser.add_argument(
        "--write-sampled-component-contours", action="store_true", default=True
    )
    parser.add_argument("--no-write-sampled-component-contours", action="store_true")
    parser.add_argument("--retain-source-seeds", action="store_true")
    parser.add_argument("--global-mask-grid-manifest", type=Path)
    parser.add_argument("--encoded-mask-copy-workers", type=int, default=8)
    parser.add_argument("--require-production-proof", action="store_true")
    parser.add_argument("--publication-evidence-producer-commit")
    parser.add_argument("--work-unit-id")
    parser.add_argument("--work-unit-index", type=int)
    parser.add_argument("--source-clip-id")
    parser.add_argument("--source-clip-index", type=int)
    parser.add_argument("--global-frame-start", type=int)
    parser.add_argument("--global-frame-stop", type=int)
    parser.add_argument("--quality-compute-workers", type=int, default=4)
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
        require_production_proof=bool(args.require_production_proof),
        publication_evidence_producer_commit=(
            args.publication_evidence_producer_commit
        ),
        work_unit_id=args.work_unit_id,
        work_unit_index=args.work_unit_index,
        source_clip_id=args.source_clip_id,
        source_clip_index=args.source_clip_index,
        global_frame_start=args.global_frame_start,
        global_frame_stop=args.global_frame_stop,
        quality_compute_workers=int(args.quality_compute_workers),
        overwrite=bool(args.overwrite),
        cleanup=not bool(args.no_cleanup),
    )
    if args.json:
        print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    else:
        print(
            f"Finalized {result['refined_run']} -> {result['package']['artifact_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
