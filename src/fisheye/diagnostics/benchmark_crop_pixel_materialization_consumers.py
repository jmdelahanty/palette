"""Benchmark one bounded acquisition-crop materialization and its real consumers.

The source analysis archive and acquisition sidecars are opened read-only.  A
modern crop run, keyed pixel work package, and optional model shard outputs are
created only under node-local scratch.  The only durable output is a strict JSON
receipt (plus consumer logs) written to an explicit benchmark directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.crop_pixel_work_package import (
    open_crop_pixel_work_package,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.system_metadata import get_git_info


SCHEMA_ID = "palette.crop_pixel_materialization_consumer_canary"
SCHEMA_VERSION = 1
DEFAULT_ROW_COUNT = 2_048
DEFAULT_BATCH_ROWS = 256


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _max_rss_bytes(who: int) -> int:
    value = int(resource.getrusage(who).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _storage_stats(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "apparent_bytes": sum(int(item.stat().st_size) for item in files),
    }


def _require_node_local_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    forbidden = (Path("/groups"), Path("/nrs"), Path("/Volumes"))
    if any(resolved == prefix or prefix in resolved.parents for prefix in forbidden):
        raise ValueError(
            "Crop pixel canary scratch must be node-local, not a shared mount: "
            f"{resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _digest_crop_source(source: CropImageSource, *, batch_rows: int) -> str:
    digest = hashlib.sha256()
    for start in range(0, int(source.total_rois), max(1, int(batch_rows))):
        stop = min(start + max(1, int(batch_rows)), int(source.total_rois))
        values = np.ascontiguousarray(source.read_slice(start, stop), dtype=np.uint8)
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _run_logged(command: list[str], *, log_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    result = {
        "command": command,
        "log_path": str(log_path),
        "returncode": int(completed.returncode),
        "seconds": float(time.perf_counter() - started),
    }
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"Consumer command failed ({completed.returncode}): {' '.join(command)}\n"
            + "\n".join(tail)
        )
    return result


def _array(group: Any, name: str, dtype: Any) -> np.ndarray:
    if name not in group:
        raise ValueError(f"Consumer output is missing {name!r}.")
    return np.asarray(group[name][:], dtype=dtype)


def _validate_consumer_lineage(
    local_archive: Path,
    *,
    output_parent: str,
    run_name: str,
    expected_rows: np.ndarray,
    expected_keys: np.ndarray,
    expected_signatures: np.ndarray,
    package_id: str,
) -> dict[str, Any]:
    root = zarr.open_group(str(local_archive), mode="r", use_consolidated=False)
    path = f"{output_parent}/{run_name}"
    if path not in root:
        raise ValueError(f"Consumer did not publish {path}.")
    run = root[path]
    rows = _array(run, "source_crop_row_ids", np.int64).reshape(-1)
    keys = _array(run, "instance_key", np.uint64).reshape(-1)
    signatures = _array(run, "source_row_signature", np.uint8)
    if not np.array_equal(rows, expected_rows):
        raise ValueError(f"{path} source_crop_row_ids differ from the package.")
    if not np.array_equal(keys, expected_keys):
        raise ValueError(f"{path} instance_key values differ from the package.")
    if not np.array_equal(signatures, expected_signatures):
        raise ValueError(f"{path} source_row_signature differs from the package.")
    if str(run.attrs.get("source_crop_pixel_work_package_id") or "") != package_id:
        raise ValueError(f"{path} does not bind the exact pixel package ID.")
    return {
        "run_path": path,
        "row_count": int(rows.shape[0]),
        "instance_keys_unique": int(np.unique(keys).shape[0]) == int(keys.shape[0]),
        "source_crop_row_ids_sha256": hashlib.sha256(
            np.asarray(rows, dtype="<i8").tobytes(order="C")
        ).hexdigest(),
        "instance_key_sha256": hashlib.sha256(
            np.asarray(keys, dtype="<u8").tobytes(order="C")
        ).hexdigest(),
        "source_row_signature_sha256": hashlib.sha256(
            np.asarray(signatures, dtype=np.uint8).tobytes(order="C")
        ).hexdigest(),
        "pixel_materialization_id": package_id,
    }


def _consumer_commands(
    *,
    local_archive: Path,
    crop_run: str,
    package_manifest: Path,
    keypoint_model: Path,
    subject_mask_model: Path,
    keypoint_run: str,
    mask_run: str,
    batch_rows: int,
    device: str,
) -> tuple[list[str], list[str]]:
    keypoints = [
        sys.executable,
        "-m",
        "fisheye.detection.detect_keypoints_yolo",
        str(local_archive),
        "--model",
        str(keypoint_model),
        "--run-name",
        keypoint_run,
        "--output-parent",
        "keypoint_shard_runs",
        "--crop-run",
        crop_run,
        "--pose-schema",
        "traditional_v2",
        "--batch-size",
        str(batch_rows),
        "--imgsz",
        "512",
        "--device",
        device,
        "--roi-work-package-manifest",
        str(package_manifest),
        "--input-mode",
        "tensor",
        "--model-input-transform",
        "auto",
        "--coordinate-contract-mode",
        "legacy_noncanonical",
        "--profile-timings",
    ]
    masks = [
        sys.executable,
        "-m",
        "fisheye.segmentation.infer_unet_subject_masks",
        str(local_archive),
        str(subject_mask_model),
        "--run-name",
        mask_run,
        "--output-parent",
        "subject_mask_shard_runs",
        "--crop-run",
        crop_run,
        "--batch-size",
        str(batch_rows),
        "--device",
        device,
        "--model-input-size",
        "512",
        "--model-input-transform",
        "auto",
        "--roi-work-package-manifest",
        str(package_manifest),
        "--mask-probs-dtype",
        "uint8",
        "--profile-timings",
        "--defer-registry-status",
    ]
    return keypoints, masks


def _prepare_command(
    args: argparse.Namespace,
    *,
    scratch: Path,
    output_json: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "fisheye.diagnostics.prepare_crop_pixel_materialization_canary",
        "--source-analysis-zarr",
        str(args.source_analysis_zarr.expanduser().resolve()),
        "--recording-dir",
        str(args.recording_dir.expanduser().resolve()),
        "--crop-meta",
        str(args.crop_meta.expanduser().resolve()),
        "--crop-video",
        str(args.crop_video.expanduser().resolve()),
        "--scratch-root",
        str(scratch),
        "--output-json",
        str(output_json),
        "--crop-run",
        str(args.crop_run),
        "--row-count",
        str(int(args.row_count)),
        "--batch-rows",
        str(int(args.batch_rows)),
    ]
    if args.source_width is not None:
        command.extend(("--source-width", str(int(args.source_width))))
    if args.source_height is not None:
        command.extend(("--source-height", str(int(args.source_height))))
    return command


def run_canary(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_archive = args.source_analysis_zarr.expanduser().resolve()
    source_metadata_path = source_archive / "zarr.json"
    if not source_metadata_path.is_file():
        raise ValueError(f"Source is not a Zarr v3 archive: {source_archive}")
    source_metadata_sha256_before = _sha256_file(source_metadata_path)
    scratch = _require_node_local_scratch(args.scratch_root)
    prepare_json = scratch / "prepare.json"
    prepare_log = args.output_json.with_suffix(".prepare.log")
    prepare_command = _prepare_command(
        args,
        scratch=scratch,
        output_json=prepare_json,
    )
    prepare_process = _run_logged(prepare_command, log_path=prepare_log)
    prepared = json.loads(prepare_json.read_text(encoding="utf-8"))
    if prepared.get("status") != "complete":
        raise RuntimeError("Crop pixel preparation worker did not complete.")
    local_archive = Path(str(prepared["local_archive"])).resolve()
    package_manifest = Path(str(prepared["package_manifest"])).resolve()
    recording_identity = str(prepared["recording_identity"])
    reference = prepared["crop_run_reference"]
    total_rows = int(prepared["source_crop_total_rows"])
    selected_count = int(prepared["selected_rows"])
    selected_rows = np.arange(selected_count, dtype=np.int64)
    package = prepared["package"]
    phases = dict(prepared["timing_seconds"])
    phases["prepare_worker_process"] = float(prepare_process["seconds"])
    local_root = zarr.open_group(str(local_archive), mode="r", use_consolidated=False)

    opened = open_crop_pixel_work_package(
        package_manifest,
        expected_archive_path=local_archive,
        expected_crop_run=args.crop_run,
        root=local_root,
        verify_payload=True,
        verify_pixel_rows=True,
    )
    try:
        expected_rows = np.asarray(opened.crop_row_indices, dtype=np.int64)
        expected_keys = np.asarray(opened.instance_keys, dtype=np.uint64)
        expected_signatures = np.asarray(opened.source_row_signatures, dtype=np.uint8)
        package_id = str(opened.package_id)
    finally:
        opened.close()

    consumer_inputs: dict[str, Any] = {}
    for name in ("keypoints", "subject_masks"):
        phase = time.perf_counter()
        source = CropImageSource.open_work_package(
            local_root,
            manifest_path=package_manifest,
            zarr_path=local_archive,
            crop_run=args.crop_run,
        )
        try:
            digest = _digest_crop_source(source, batch_rows=int(args.batch_rows))
            consumer_inputs[name] = {
                "pixel_sha256": digest,
                "row_count": int(source.total_rois),
                "roi_shape": [int(value) for value in source.roi_shape],
                "source_crop_row_ids_match": bool(
                    np.array_equal(source.source_crop_row_ids, expected_rows)
                ),
                "pixel_materialization_id": source.pixel_materialization_id,
                "seconds": float(time.perf_counter() - phase),
            }
        finally:
            source.close()
    package_pixel_sha256 = str(package["array"]["sha256"])
    for name, evidence in consumer_inputs.items():
        if evidence["pixel_sha256"] != package_pixel_sha256:
            raise ValueError(f"{name} consumer pixels differ from the package payload.")
        if evidence["source_crop_row_ids_match"] is not True:
            raise ValueError(f"{name} consumer crop-row identity differs from package.")

    model_consumers: dict[str, Any] = {"executed": False}
    if args.keypoint_model is not None or args.subject_mask_model is not None:
        if args.keypoint_model is None or args.subject_mask_model is None:
            raise ValueError(
                "Pass both --keypoint-model and --subject-mask-model, or neither."
            )
        keypoint_model = args.keypoint_model.expanduser().resolve()
        subject_mask_model = args.subject_mask_model.expanduser().resolve()
        for model in (keypoint_model, subject_mask_model):
            if not model.is_file():
                raise ValueError(f"Model artifact is missing: {model}")
        keypoint_run = "crop_pixel_canary_keypoints"
        mask_run = "crop_pixel_canary_subject_masks"
        keypoint_command, mask_command = _consumer_commands(
            local_archive=local_archive,
            crop_run=args.crop_run,
            package_manifest=package_manifest,
            keypoint_model=keypoint_model,
            subject_mask_model=subject_mask_model,
            keypoint_run=keypoint_run,
            mask_run=mask_run,
            batch_rows=int(args.model_batch_rows),
            device=str(args.device),
        )
        keypoint_log = args.output_json.with_suffix(".keypoints.log")
        mask_log = args.output_json.with_suffix(".subject_masks.log")
        model_consumers = {
            "executed": True,
            "keypoints": _run_logged(keypoint_command, log_path=keypoint_log),
            "subject_masks": _run_logged(mask_command, log_path=mask_log),
            "keypoint_model": {
                "path": str(keypoint_model),
                "sha256": _sha256_file(keypoint_model),
            },
            "subject_mask_model": {
                "path": str(subject_mask_model),
                "sha256": _sha256_file(subject_mask_model),
            },
        }
        model_consumers["keypoints"]["lineage"] = _validate_consumer_lineage(
            local_archive,
            output_parent="keypoint_shard_runs",
            run_name=keypoint_run,
            expected_rows=expected_rows,
            expected_keys=expected_keys,
            expected_signatures=expected_signatures,
            package_id=package_id,
        )
        model_consumers["subject_masks"]["lineage"] = _validate_consumer_lineage(
            local_archive,
            output_parent="subject_mask_shard_runs",
            run_name=mask_run,
            expected_rows=expected_rows,
            expected_keys=expected_keys,
            expected_signatures=expected_signatures,
            package_id=package_id,
        )

    source_metadata_sha256_after = _sha256_file(source_metadata_path)
    if source_metadata_sha256_after != source_metadata_sha256_before:
        raise RuntimeError("Source archive metadata changed during the canary.")
    git = get_git_info(Path(__file__).resolve().parents[3])
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "source": {
            "analysis_zarr": str(source_archive),
            "analysis_root_zarr_json_sha256_before": source_metadata_sha256_before,
            "analysis_root_zarr_json_sha256_after": source_metadata_sha256_after,
            "recording_dir": str(args.recording_dir.expanduser().resolve()),
            "crop_meta": str(args.crop_meta.expanduser().resolve()),
            "crop_video": str(args.crop_video.expanduser().resolve()),
            "recording_identity": recording_identity,
        },
        "selection": {
            "policy": "contiguous_crop_row_prefix_v1",
            "requested_rows": int(args.row_count),
            "selected_rows": selected_count,
            "source_crop_total_rows": total_rows,
            "first_crop_row": int(selected_rows[0]),
            "last_crop_row": int(selected_rows[-1]),
        },
        "local_crop_publication": {
            **prepared["crop_result"],
            "crop_run_reference": reference,
            "selector_eligible": False,
        },
        "work_package": {
            "package_id": package_id,
            "pixel_sha256": package_pixel_sha256,
            "pixel_bytes": int(package["array"]["total_bytes"]),
            "pixel_contract": package["pixel_contract"],
            "source_binding_profile": package["source"]["source_binding_profile"],
            "manifest_sha256": _sha256_file(package_manifest),
        },
        "consumer_inputs": consumer_inputs,
        "model_consumers": model_consumers,
        "timing_seconds": {
            **phases,
            "total": float(time.perf_counter() - started),
        },
        "resources": {
            "peak_rss_bytes": max(
                _max_rss_bytes(resource.RUSAGE_SELF),
                _max_rss_bytes(resource.RUSAGE_CHILDREN),
            ),
            "parent_peak_rss_bytes": _max_rss_bytes(resource.RUSAGE_SELF),
            "child_peak_rss_bytes": _max_rss_bytes(resource.RUSAGE_CHILDREN),
            "local_store": _storage_stats(scratch),
        },
        "runtime": {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "lsb_jobid": os.environ.get("LSB_JOBID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_executable": sys.executable,
        },
        "palette": {
            "repository": str(Path(__file__).resolve().parents[3]),
            "git": git,
        },
        "publication": {
            "selector_eligible": False,
            "registry_registered": False,
            "production_state_changes": [],
            "durable_artifacts": [
                str(args.output_json.expanduser().resolve()),
                str(prepare_log),
                *(
                    [
                        str(args.output_json.with_suffix(".keypoints.log")),
                        str(args.output_json.with_suffix(".subject_masks.log")),
                    ]
                    if model_consumers["executed"]
                    else []
                ),
            ],
        },
    }
    payload["evidence_digest"] = canonical_json_sha256(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-analysis-zarr", type=Path, required=True)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--crop-meta", type=Path, required=True)
    parser.add_argument("--crop-video", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--crop-run", default="crop_pixel_canary_acquisition_v1")
    parser.add_argument("--row-count", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--source-width", type=int)
    parser.add_argument("--source-height", type=int)
    parser.add_argument("--keypoint-model", type=Path)
    parser.add_argument("--subject-mask-model", type=Path)
    parser.add_argument("--model-batch-rows", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.row_count) <= 0 or int(args.batch_rows) <= 0:
        raise SystemExit("--row-count and --batch-rows must be positive")
    args.output_json = args.output_json.expanduser().resolve()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_canary(args)
    except Exception as exc:
        failure = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "production_state_changes": [],
        }
        args.output_json.write_text(
            json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        raise
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
