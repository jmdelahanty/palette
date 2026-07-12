"""Run one split subject-mask stage with optional node-local ROI-cache staging."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from fisheye.shared.flat_roi_cache import load_flat_roi_cache_manifest
from fisheye.utils import run_subject_mask_batch_pipeline as pipeline


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_flat_roi_cache_manifest(
    manifest_path: Path,
    *,
    staging_dir: Path | None,
) -> tuple[Path, dict[str, object]]:
    """Copy one complete flat cache to a private node-local directory."""

    source_manifest = manifest_path.expanduser().resolve()
    payload = load_flat_roi_cache_manifest(source_manifest)
    array = payload.get("array")
    if not isinstance(array, dict):
        raise ValueError(f"Flat ROI cache manifest lacks array metadata: {source_manifest}")
    raw_bin = str(array.get("bin_path") or "").strip()
    if not raw_bin:
        raise ValueError(f"Flat ROI cache manifest lacks array.bin_path: {source_manifest}")
    source_bin = Path(raw_bin).expanduser()
    if not source_bin.is_absolute():
        source_bin = source_manifest.parent / source_bin
    source_bin = source_bin.resolve()
    if not source_bin.is_file():
        raise FileNotFoundError(f"Flat ROI cache payload not found: {source_bin}")
    expected_bytes = int(array.get("total_bytes") or 0)
    actual_bytes = int(source_bin.stat().st_size)
    if expected_bytes and actual_bytes != expected_bytes:
        raise ValueError(
            f"Flat ROI cache size mismatch: expected {expected_bytes}, got {actual_bytes}."
        )
    if staging_dir is None:
        raise ValueError("Subject-mask cache staging requires --roi-cache-staging-dir.")
    target_dir = staging_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    local_manifest = target_dir / source_manifest.name
    local_bin = target_dir / source_bin.name
    if local_manifest == source_manifest or local_bin == source_bin:
        raise ValueError("ROI-cache staging directory must differ from the source directory.")

    started_at = _utc_now()
    started = time.perf_counter()
    tmp_bin = local_bin.with_name(f"{local_bin.name}.tmp.{os.getpid()}")
    shutil.copy2(source_bin, tmp_bin)
    os.replace(tmp_bin, local_bin)
    duration = float(time.perf_counter() - started)
    details: dict[str, object] = {
        "schema": "palette_roi_cache_staging_v1",
        "policy": "node_scratch_staged_flat_cache",
        "staged": True,
        "requested_manifest_path": str(source_manifest),
        "effective_manifest_path": str(local_manifest),
        "source_bin_path": str(source_bin),
        "effective_bin_path": str(local_bin),
        "staging_dir": str(target_dir),
        "host": socket.gethostname(),
        "lsb_jobid": os.environ.get("LSB_JOBID"),
        "copy": {
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "duration_seconds": duration,
            "size_bytes": actual_bytes,
            "mib_per_second": (
                (actual_bytes / (1024 * 1024)) / duration if duration > 0 else None
            ),
        },
    }
    staged = dict(payload)
    staged_array = dict(array)
    staged_array["bin_path"] = local_bin.name
    staged["array"] = staged_array
    staged["manifest_path"] = str(local_manifest)
    staged["staging"] = details
    tmp_manifest = local_manifest.with_name(
        f"{local_manifest.name}.tmp.{os.getpid()}"
    )
    tmp_manifest.write_text(
        json.dumps(staged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_manifest, local_manifest)
    return local_manifest, details


def _pipeline_args(args: argparse.Namespace, *, cache_manifest: Path | None) -> list[str]:
    command = [
        str(args.analysis_zarr),
        "--apply",
        "--registry",
        str(args.registry),
        "--run-label",
        args.run_label,
        "--workflow-stage",
        args.stage,
        "--crop-run",
        args.crop_run,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--mask-probs-dtype",
        "uint8",
        "--mask-probs-chunk-rois",
        "32",
        "--mask-probs-shard-rois",
        str(args.mask_probs_shard_rois),
        "--output-queue-size",
        "2",
        "--model-coverage-class",
        args.model_coverage_class,
        "--model-component-coverage-key",
        args.model_component_coverage_key,
        "--model-label-schema-id",
        args.model_label_schema_id,
        "--model-top-k",
        str(args.model_top_k),
        "--model-require-unique",
        "--roi-cache-policy",
        "never",
        "--profile-timings",
        "--stage-output-to-scratch",
        "--defer-registry-status",
        "--metric-level",
        "cheap",
        "--mask-storage",
        "dense_uint8",
        "--mask-rle-validation-mode",
        "invariants",
        "--finalize-chunk-size",
        "256",
        "--finalize-dense-mask-row-chunk",
        "256",
        "--finalize-execution-backend",
        "process_shards",
        "--finalize-num-workers",
        str(args.finalize_num_workers),
        "--finalize-postcompute-backend",
        "process_shards",
        "--write-eye-geometry",
        "--no-write-component-contours",
        "--write-sampled-component-contours",
    ]
    if args.progress_dir is not None:
        command.extend(["--progress-dir", str(args.progress_dir)])
    if args.handoff_package_dir is not None:
        command.extend(["--handoff-package-dir", str(args.handoff_package_dir)])
    if args.stage == "inference":
        command.append("--no-assignment-keypoints")
        if cache_manifest is None:
            raise ValueError("Inference requires an ROI-cache manifest.")
        command.extend(["--roi-cache-manifest", str(cache_manifest)])
    else:
        command.extend(
            [
                "--assignment-keypoint-group",
                "refined_keypoints_runs",
                "--assignment-keypoints-run",
                args.refined_keypoint_run,
                "--stage-finalization-input-to-scratch",
            ]
        )
    return command


def _cleanup_staged_cache(staged_manifest: Path) -> None:
    """Remove only the manifest and payload created by this process."""

    try:
        payload = json.loads(staged_manifest.read_text(encoding="utf-8"))
        array = payload.get("array")
        raw_bin = str(array.get("bin_path") or "") if isinstance(array, dict) else ""
        if raw_bin:
            payload_path = Path(raw_bin)
            if not payload_path.is_absolute():
                payload_path = staged_manifest.parent / payload_path
            payload_path.unlink(missing_ok=True)
        staged_manifest.unlink(missing_ok=True)
        staged_manifest.parent.rmdir()
    except OSError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inference", "finalization"))
    parser.add_argument("--analysis-zarr", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--refined-keypoint-run")
    parser.add_argument("--roi-cache-manifest", type=Path)
    parser.add_argument("--roi-cache-staging-dir", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mask-probs-shard-rois", type=int, default=2048)
    parser.add_argument("--finalize-num-workers", type=int, default=16)
    parser.add_argument("--model-coverage-class", default="dense_all_components")
    parser.add_argument(
        "--model-component-coverage-key",
        default="body+eyes+swim_bladder",
    )
    parser.add_argument("--model-label-schema-id", default="subject_v1_union")
    parser.add_argument("--model-top-k", type=int, default=5)
    parser.add_argument("--progress-dir", type=Path)
    parser.add_argument("--handoff-package-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "inference" and args.roi_cache_manifest is None:
        raise ValueError("Inference requires --roi-cache-manifest.")
    if args.stage == "finalization" and not args.refined_keypoint_run:
        raise ValueError("Finalization requires --refined-keypoint-run.")

    staged_manifest: Path | None = None
    staging_details: dict[str, object] = {}
    try:
        if args.stage == "inference":
            assert args.roi_cache_manifest is not None
            staged_manifest, staging_details = _stage_flat_roi_cache_manifest(
                args.roi_cache_manifest,
                staging_dir=args.roi_cache_staging_dir,
            )
        command = _pipeline_args(args, cache_manifest=staged_manifest)
        status = int(pipeline.main(command))
        if args.json:
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "status": status,
                        "analysis_zarr": str(args.analysis_zarr),
                        "run_label": args.run_label,
                        "refined_keypoint_run": args.refined_keypoint_run,
                        "roi_cache_staging": staging_details,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return status
    finally:
        if staged_manifest is not None:
            _cleanup_staged_cache(staged_manifest)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
