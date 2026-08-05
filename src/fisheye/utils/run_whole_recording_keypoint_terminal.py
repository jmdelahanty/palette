#!/usr/bin/env python3
"""Run one cache-backed whole-recording pose model into terminal evidence.

The analysis archive is never used as an inference output.  A geometry-only
crop shell and the authenticated flat cache are staged to node-local scratch;
the completed noncanonical compute result is then copied to a new immutable
workflow artifact for strict keypoint-v2 finalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.model_input_transform import (
    MODEL_INPUT_TRANSFORM_CHOICES,
)
from fisheye.shared.pose_model_input_contract import (
    load_pose_model_input_contract,
    validate_pose_runtime_compatibility,
)
from fisheye.shared.pose_inference_failure import (
    POSE_INFERENCE_FAILURE_SCHEMA_ID,
    POSE_INFERENCE_FAILURE_SCHEMA_VERSION,
    pose_inference_failure_code_map_json,
    pose_inference_failure_histogram,
    validate_pose_inference_failure_codes,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array, utc_now
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.keypoint_manifest import KeypointPreprocessingReference
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import is_run_complete
from fisheye.utils.run_keypoints_with_registry_model import (
    run_keypoints_with_registry_model,
)


WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_ID = (
    "palette.keypoint.whole_recording_terminal"
)
WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_VERSION = 3
TERMINAL_RECEIPT_NAME = "terminal_receipt.json"
_SOURCE_ARRAY_PATHS = (
    "instance_key",
    "source_crop_row_ids",
    "source_acquisition_frame_index",
    "frame_indices",
    "keypoints_roi",
    "keypoints_img",
    "keypoint_confidences",
    "confidence",
    "pose_bbox_xyxy_roi",
    "pose_bbox_xyxy_img",
    "detection_success",
    "pose_failure_codes",
)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return value


def _require_node_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not (str(resolved).startswith("/scratch/") or str(resolved).startswith("/tmp/")):
        raise ValueError("scratch_root must be under /scratch or /tmp.")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _stage_crop_shell(
    analysis_zarr: Path,
    *,
    crop_run: str,
    destination: Path,
) -> Path:
    source = analysis_zarr / "crop_runs" / crop_run
    if not source.is_dir():
        raise FileNotFoundError(f"Crop run not found: {source}")
    local = destination / "analysis.zarr"
    root = zarr.open_group(str(local), mode="w-", zarr_format=3)
    root.create_group("crop_runs")
    shutil.copytree(source, local / "crop_runs" / crop_run, copy_function=shutil.copy2)
    return local


def _cache_evidence(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    array = manifest.get("array")
    if not isinstance(array, Mapping):
        raise ValueError("Flat-cache manifest lacks array metadata.")
    payload_sha = str(array.get("sha256") or "")
    if len(payload_sha) != 64:
        raise ValueError("Flat-cache manifest lacks its payload SHA-256.")
    payload_path = Path(str(array.get("bin_path") or "")).expanduser()
    if not payload_path.is_absolute():
        payload_path = manifest_path.parent / payload_path
    payload_path = payload_path.resolve()
    if not payload_path.is_file():
        raise FileNotFoundError(f"Flat-cache payload not found: {payload_path}")
    shape = tuple(int(value) for value in array.get("shape", ()))
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("Flat-cache manifest requires one positive [N,H,W] shape.")
    dtype = np.dtype(array.get("dtype"))
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    if payload_path.stat().st_size != expected_bytes:
        raise ValueError("Flat-cache payload size differs from its manifest.")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha,
        "payload_size_bytes": expected_bytes,
        "shape": list(shape),
        "dtype": dtype.str,
        "source": manifest.get("source"),
        "builder": manifest.get("builder"),
    }


def _publish_terminal_tree(
    *,
    local_archive: Path,
    source_run: str,
    destination: Path,
    receipt: Mapping[str, Any],
) -> None:
    output = destination.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Terminal artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    if temporary.exists():
        raise FileExistsError(f"Terminal temporary exists: {temporary}")
    try:
        root = zarr.open_group(str(temporary), mode="w-", zarr_format=3)
        root.attrs.update(
            {
                "schema_id": WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_ID,
                "schema_version": WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_VERSION,
                "status": "complete",
                "selector_eligible": False,
                "registry_registered": False,
            }
        )
        root.create_group("keypoint_terminal_runs")
        shutil.copytree(
            local_archive / "keypoint_shard_runs" / source_run,
            temporary / "keypoint_terminal_runs" / source_run,
            copy_function=shutil.copy2,
        )
        consolidate_metadata_capture_expected_warnings(temporary)
        check = zarr.open_group(
            str(temporary), mode="r", zarr_format=3, use_consolidated=True
        )
        run = check[f"keypoint_terminal_runs/{source_run}"]
        if (
            not is_run_complete(run)
            or run.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                "Terminal keypoint run lost its completed ineligible state."
            )
        write_json_atomic(temporary / TERMINAL_RECEIPT_NAME, receipt)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run_whole_recording_keypoint_terminal(
    *,
    recording_id: str,
    recording_dir: Path,
    analysis_zarr: Path,
    crop_run: str,
    cache_manifest: Path,
    registry: Path,
    model_set_id: str,
    model_run_id: str,
    expected_model_path: Path,
    expected_model_sha256: str,
    model_input_contract: Path,
    pose_schema: str,
    terminal_run_id: str,
    terminal_output: Path,
    scratch_root: Path,
    batch_size: int,
    device: str,
    input_mode: str,
    model_input_size: int,
    network_input_size: int,
    model_input_transform_mode: str,
    model_input_stride: int,
    progress_jsonl: Path | None = None,
    progress_every_batches: int = 1,
) -> Mapping[str, Any]:
    archive = analysis_zarr.expanduser().resolve()
    cache = cache_manifest.expanduser().resolve()
    cache_binding = _cache_evidence(cache)
    cache_shape = tuple(int(value) for value in cache_binding["shape"])
    model_contract = load_pose_model_input_contract(
        model_input_contract,
        model_path=expected_model_path,
        expected_set_id=model_set_id,
        expected_run_id=model_run_id,
        expected_model_sha256=expected_model_sha256,
    )
    runtime_plan = model_contract.plan_for_native_shape(
        (cache_shape[1], cache_shape[2])
    )
    expected_model_input_transform = runtime_plan.transform
    if (
        input_mode != runtime_plan.input_mode
        or model_input_transform_mode != expected_model_input_transform.name
        or int(model_input_size) != expected_model_input_transform.model_height
        or int(network_input_size) != runtime_plan.network_imgsz
        or int(model_input_stride) != runtime_plan.model_stride
    ):
        raise ValueError(
            "Worker arguments disagree with the digest-bound model-input runtime plan."
        )
    runtime_compatibility = validate_pose_runtime_compatibility(model_contract)
    runtime_ultralytics_version = str(
        runtime_compatibility["runtime_ultralytics_version"]
    )
    scratch = _require_node_scratch(scratch_root) / f"keypoint_{uuid.uuid4().hex}"
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        local_archive = _stage_crop_shell(
            archive, crop_run=crop_run, destination=scratch / "compute"
        )
        result = run_keypoints_with_registry_model(
            recording_dir=recording_dir,
            output=local_archive,
            registry=registry,
            set_id=model_set_id,
            model_run_id=model_run_id,
            require_unique=True,
            run_name=terminal_run_id,
            output_parent="keypoint_shard_runs",
            crop_run=crop_run,
            pose_schema=pose_schema,
            batch_size=batch_size,
            device=device,
            imgsz=int(network_input_size),
            model_input_size=int(model_input_size),
            expected_model_stride=model_input_stride,
            roi_cache_policy="always",
            roi_cache_manifest=cache,
            roi_cache_expected_archive_path=archive,
            stage_roi_cache_to_scratch=True,
            roi_cache_staging_dir=scratch / "cache",
            profile_timings=True,
            progress_jsonl=progress_jsonl,
            progress_every_batches=progress_every_batches,
            input_mode=runtime_plan.input_mode,
            model_input_transform_mode=model_input_transform_mode,
            coordinate_contract_mode="legacy_noncanonical",
            keypoint_roi_shard_rows=None,
        )
        if not result.ok or result.keypoint_run != terminal_run_id:
            raise RuntimeError(
                f"Terminal keypoint inference failed: {result.to_dict()}"
            )
        root = zarr.open_group(str(local_archive), mode="r", use_consolidated=False)
        run = root[f"keypoint_shard_runs/{terminal_run_id}"]
        if (
            not is_run_complete(run)
            or run.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                "Terminal compute output is not complete and ineligible."
            )
        missing = [path for path in _SOURCE_ARRAY_PATHS if path not in run]
        if missing:
            raise RuntimeError(f"Terminal compute output lacks arrays: {missing!r}")
        if run.attrs.get("input_mode_effective") != runtime_plan.input_mode:
            raise RuntimeError(
                "Whole-recording terminal inference changed the contract input mode."
            )
        if run.attrs.get("model_input_transform") != (
            expected_model_input_transform.to_attrs()
        ):
            raise RuntimeError(
                "Terminal model-input transform differs from the planned padding contract."
            )
        if run.attrs.get("model_input_stride") != model_input_stride:
            raise RuntimeError(
                "Terminal model stride differs from the verified plan contract."
            )
        parameters = run.attrs.get("parameters")
        if (
            not isinstance(parameters, Mapping)
            or parameters.get("imgsz") != int(network_input_size)
            or parameters.get("model_input_size") != int(model_input_size)
            or parameters.get("model_predict_rect") is not False
            or run.attrs.get("ultralytics_version")
            != runtime_ultralytics_version
        ):
            raise RuntimeError(
                "Terminal model-side preprocessing differs from its input contract."
            )
        array_hashes = {
            path: sha256_array(run[path][...]) for path in _SOURCE_ARRAY_PATHS
        }
        resolution = result.resolution_payload
        if not isinstance(resolution, Mapping):
            raise RuntimeError("Terminal inference lacks model-resolution provenance.")
        artifacts = resolution.get("artifacts")
        pose_binding = (
            artifacts.get("model_pose_schema_binding")
            if isinstance(artifacts, Mapping)
            else None
        )
        if not isinstance(pose_binding, Mapping):
            raise RuntimeError("Terminal inference lacks exact pose-model binding.")
        selected = resolution.get("selected")
        if not isinstance(selected, Mapping):
            raise RuntimeError("Terminal inference lacks selected model identity.")
        if (
            Path(str(selected.get("model_path") or "")).expanduser().resolve()
            != expected_model_path.expanduser().resolve()
            or selected.get("model_sha256") != expected_model_sha256
        ):
            raise RuntimeError(
                "Terminal inference selected a model different from its input contract."
            )
        staging = run.attrs.get("source_roi_cache_staging")
        staging_copy = staging.get("copy") if isinstance(staging, Mapping) else None
        if (
            not isinstance(staging, Mapping)
            or staging.get("staged") is not True
            or not isinstance(staging_copy, Mapping)
            or staging_copy.get("verification") != "single_pass_copy_stream_sha256_v1"
            or staging_copy.get("source_sha256") != staging_copy.get("staged_sha256")
        ):
            raise RuntimeError(
                "Terminal inference lacks authenticated node-local cache staging."
            )
        if staging_copy.get("source_sha256") != cache_binding["payload_sha256"]:
            raise RuntimeError(
                "Authenticated cache staging disagrees with the source manifest."
            )
        cache_binding["staging_verification"] = dict(staging)
        preprocessing = KeypointPreprocessingReference(
            profile_id="yolo_pose_flat_cache_v1",
            profile_version=1,
            input_mode="flat_bin_node_scratch",
            document={
                "source_pixel_dtype": cache_binding["dtype"],
                "source_pixel_shape": cache_binding["shape"],
                "source_pixel_contract": (
                    cache_binding["builder"].get("pixel_contract")
                    if isinstance(cache_binding["builder"], Mapping)
                    else None
                ),
                "cache_manifest_sha256": cache_binding["manifest_sha256"],
                "cache_payload_sha256": cache_binding["payload_sha256"],
                "model_input_mode": run.attrs.get("input_mode_effective"),
                "model_input_transform": run.attrs.get("model_input_transform"),
                "model_input_stride": run.attrs.get("model_input_stride"),
                "model_input_contract": model_contract.to_json(),
                "model_input_runtime": runtime_plan.to_json(),
                "runtime_compatibility": runtime_compatibility,
                "confidence_threshold": run.attrs.get("confidence_threshold"),
                "iou_threshold": run.attrs.get("iou_threshold"),
                "max_detections_per_roi": run.attrs.get("max_detections"),
                "coordinate_contract_mode": "legacy_noncanonical",
            },
        ).as_manifest()
        detection_success = np.asarray(run["detection_success"][:], dtype=bool)
        pose_failure_codes = np.asarray(
            run["pose_failure_codes"][:], dtype=np.uint8
        )
        validate_pose_inference_failure_codes(
            pose_failure_codes,
            pose_success=detection_success,
        )
        failure_code_histogram = pose_inference_failure_histogram(
            pose_failure_codes
        )
        payload = {
            "status": "complete",
            "created_at_utc": utc_now(),
            "recording_id": recording_id,
            "recording_dir": str(recording_dir.expanduser().resolve()),
            "analysis_zarr": str(archive),
            "crop_run": crop_run,
            "terminal_run_id": terminal_run_id,
            "terminal_group_path": f"keypoint_terminal_runs/{terminal_run_id}",
            "row_count": int(run["instance_key"].shape[0]),
            "terminal_success_count": int(np.count_nonzero(detection_success)),
            "terminal_failure_count": int(
                detection_success.size - np.count_nonzero(detection_success)
            ),
            "pose_failure_codes": {
                "schema_id": POSE_INFERENCE_FAILURE_SCHEMA_ID,
                "schema_version": POSE_INFERENCE_FAILURE_SCHEMA_VERSION,
                "array_path": "pose_failure_codes",
                "dtype": "uint8",
                "code_map": pose_inference_failure_code_map_json(),
                "histogram": failure_code_histogram,
                "success_alignment": "code_zero_iff_detection_success_true",
                "public_raw_v2_array": False,
            },
            "source_array_hashes": array_hashes,
            "cache": cache_binding,
            "model": {
                "set_id": model_set_id,
                "run_id": model_run_id,
                "path": selected.get("model_path"),
                "sha256": selected.get("model_sha256"),
                "pose_model_schema_binding": dict(pose_binding),
                "pose_model_schema_binding_digest": canonical_json_sha256(pose_binding),
                "input_contract": model_contract.to_json(),
                "input_runtime": runtime_plan.to_json(),
                "runtime_compatibility": runtime_compatibility,
            },
            "preprocessing": preprocessing,
            "row_terminal_semantics": (
                "every_crop_row_present_with_exact_pose_failure_code_v2"
            ),
            "selector_eligible": False,
            "registry_registered": False,
            "production_state_changes": [],
        }
        receipt = {
            "schema_id": WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_ID,
            "schema_version": WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_VERSION,
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "payload_digest": canonical_json_sha256(payload),
            "payload": payload,
        }
        _publish_terminal_tree(
            local_archive=local_archive,
            source_run=terminal_run_id,
            destination=terminal_output,
            receipt=receipt,
        )
        return receipt
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-set-id", required=True)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--expected-model-path", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--model-input-contract", type=Path, required=True)
    parser.add_argument("--pose-schema", default="traditional_v2")
    parser.add_argument("--terminal-run-id", required=True)
    parser.add_argument("--terminal-output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--input-mode", choices=("tensor", "numpy-list", "auto"), default="tensor"
    )
    parser.add_argument("--model-input-size", type=int, required=True)
    parser.add_argument("--network-input-size", type=int, required=True)
    parser.add_argument(
        "--model-input-transform-mode",
        choices=MODEL_INPUT_TRANSFORM_CHOICES,
        required=True,
    )
    parser.add_argument("--model-input-stride", type=int, required=True)
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--progress-every-batches", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_whole_recording_keypoint_terminal(
        recording_id=args.recording_id,
        recording_dir=args.recording_dir,
        analysis_zarr=args.analysis_zarr,
        crop_run=args.crop_run,
        cache_manifest=args.cache_manifest,
        registry=args.registry,
        model_set_id=args.model_set_id,
        model_run_id=args.model_run_id,
        expected_model_path=args.expected_model_path,
        expected_model_sha256=args.expected_model_sha256,
        model_input_contract=args.model_input_contract,
        pose_schema=args.pose_schema,
        terminal_run_id=args.terminal_run_id,
        terminal_output=args.terminal_output,
        scratch_root=args.scratch_root,
        batch_size=args.batch_size,
        device=args.device,
        input_mode=args.input_mode,
        model_input_size=args.model_input_size,
        network_input_size=args.network_input_size,
        model_input_transform_mode=args.model_input_transform_mode,
        model_input_stride=args.model_input_stride,
        progress_jsonl=args.progress_jsonl,
        progress_every_batches=args.progress_every_batches,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
