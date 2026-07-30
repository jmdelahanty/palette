#!/usr/bin/env python3
"""Persist exact terminal evidence for one clip-local YOLO keypoint result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.pose_model_schema_binding import (
    load_pose_model_schema_binding,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.clipped_keypoint_finalization import (
    CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_ID,
    CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_VERSION,
    clip_terminal_result_from_yolo_arrays,
    validate_clip_terminal_result_receipt,
)
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
)
from fisheye.shared.zarr.keypoint_manifest import (
    keypoint_preprocessing_from_manifest,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr_run_completion import is_run_complete


_SOURCE_ARRAY_PATHS = (
    "instance_key",
    "source_crop_row_ids",
    "frame_indices",
    "keypoints_roi",
    "keypoint_confidences",
    "confidence",
    "pose_bbox_xyxy_roi",
    "detection_success",
)
_SOURCE_CROP_ARRAY_PATHS = (
    "instance_key",
    "source_crop_row_ids",
    "frame_indices",
    "roi_coordinates_full",
)


def _read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _group(root: Any, path: str) -> Any:
    normalized = str(path).strip().strip("/")
    if not normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError("source_group_path must be one safe relative Zarr path.")
    group = root
    for part in normalized.split("/"):
        group = group[part]
    return group


def build_clip_terminal_receipt(
    *,
    analysis_zarr: Path,
    crop_run_id: str,
    source_group_path: str,
    clip_id: str,
    clip_index: int,
    pose_binding_path: Path,
    preprocessing_path: Path,
    input_package_manifest_path: Path,
) -> dict[str, object]:
    """Recompute one sidecar from immutable crop, model, cache, and result data."""

    archive = analysis_zarr.expanduser().resolve()
    crop = open_persisted_crop_geometry_publication(
        archive,
        run_id=crop_run_id,
    )
    binding = load_pose_model_schema_binding(pose_binding_path)
    preprocessing = keypoint_preprocessing_from_manifest(
        _read_json(preprocessing_path)
    )
    package_path = input_package_manifest_path.expanduser().resolve()
    if not package_path.is_file():
        raise FileNotFoundError(f"Input package manifest not found: {package_path}")
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    source = _group(root, source_group_path)
    if not is_run_complete(source):
        raise ValueError("Clip-local keypoint source is not complete.")
    if source.attrs.get("coordinate_contract_mode") != "legacy_noncanonical":
        raise ValueError(
            "Clip-local shard must retain its legacy_noncanonical source label; "
            "the terminal sidecar is the explicit canonicalization boundary."
        )
    provenance = source.attrs.get("provenance")
    resolution = (
        provenance.get("model_resolution")
        if isinstance(provenance, Mapping)
        else None
    )
    artifacts = resolution.get("artifacts") if isinstance(resolution, Mapping) else None
    observed_binding = (
        artifacts.get("model_pose_schema_binding")
        if isinstance(artifacts, Mapping)
        else None
    )
    if not isinstance(observed_binding, Mapping) or dict(observed_binding) != binding:
        raise ValueError(
            "Clip-local keypoints bind a different pose-model schema document."
        )
    expected_source_preprocessing = {
        "coordinate_contract_mode": "legacy_noncanonical",
        "input_mode_effective": source.attrs.get("input_mode_effective"),
        "model_input_transform": source.attrs.get("model_input_transform"),
    }
    if preprocessing.document.get("clip_source_contract") != (
        expected_source_preprocessing
    ):
        raise ValueError(
            "Preprocessing manifest does not exactly bind the clip source runtime "
            "input mode and model transform."
        )
    declared_package = source.attrs.get("source_crop_pixel_work_package_manifest")
    if (
        not isinstance(declared_package, str)
        or Path(declared_package).expanduser().resolve() != package_path
    ):
        raise ValueError(
            "Clip-local keypoints bind a different crop-pixel work package."
        )
    arrays = {path: source[path] for path in _SOURCE_ARRAY_PATHS}
    source_crop_run = str(source.attrs.get("source_crop_run") or "").strip()
    if not source_crop_run or "/" in source_crop_run:
        raise ValueError("Clip-local keypoints lack a safe source_crop_run binding.")
    source_crop_group_path = f"crop_runs/{source_crop_run}"
    source_crop = _group(root, source_crop_group_path)
    if not is_run_complete(source_crop):
        raise ValueError("Clip-local source crop is not complete.")
    source_crop_arrays = {path: source_crop[path] for path in _SOURCE_CROP_ARRAY_PATHS}
    roi_shape = source_crop.attrs.get("roi_shape") or source_crop.attrs.get("roi_size")
    if (
        not isinstance(roi_shape, (list, tuple))
        or len(roi_shape) != 2
        or any(type(value) is not int or value <= 0 for value in roi_shape)
    ):
        raise ValueError("Clip-local source crop lacks a valid fixed ROI shape [H,W].")
    source_crop_arrays["roi_sizes_full"] = np.broadcast_to(
        np.asarray([int(roi_shape[1]), int(roi_shape[0])], dtype=np.int32),
        (int(source_crop["instance_key"].shape[0]), 2),
    )
    result = clip_terminal_result_from_yolo_arrays(
        crop,
        arrays,
        source_crop_arrays=source_crop_arrays,
        clip_id=clip_id,
        clip_index=clip_index,
        pose_model_schema_binding=binding,
        preprocessing=preprocessing,
        input_package_manifest_digest=_sha256_file(package_path),
    )
    payload = {
        "status": "complete",
        "analysis_zarr": str(archive),
        "source_group_path": str(source_group_path).strip().strip("/"),
        "source_crop_group_path": source_crop_group_path,
        "source_coordinate_contract_mode": "legacy_noncanonical",
        "source_array_hashes": {
            path: sha256_array(arrays[path][...]) for path in _SOURCE_ARRAY_PATHS
        },
        "source_crop_array_hashes": {
            path: sha256_array(source_crop[path][...])
            for path in _SOURCE_CROP_ARRAY_PATHS
        },
        "source_crop_roi_shape_hw": [int(roi_shape[0]), int(roi_shape[1])],
        "input_package_manifest_path": str(package_path),
        "result": result.as_manifest(),
        "production_state_changes": [],
    }
    receipt = {
        "schema_id": CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_ID,
        "schema_version": CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    errors = validate_clip_terminal_result_receipt(receipt)
    if errors:
        raise ValueError("Clip terminal receipt is invalid: " + "; ".join(errors))
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--source-group", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--clip-index", type=int, required=True)
    parser.add_argument("--pose-binding", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--input-package-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = build_clip_terminal_receipt(
            analysis_zarr=args.analysis_zarr,
            crop_run_id=args.crop_run,
            source_group_path=args.source_group,
            clip_id=args.clip_id,
            clip_index=args.clip_index,
            pose_binding_path=args.pose_binding,
            preprocessing_path=args.preprocessing,
            input_package_manifest_path=args.input_package_manifest,
        )
    except Exception as exc:
        failed = {
            "schema_id": CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_ID,
            "schema_version": CLIPPED_KEYPOINT_RESULT_RECEIPT_SCHEMA_VERSION,
            "status": "failed",
            "clip_id": args.clip_id,
            "clip_index": args.clip_index,
            "source_group_path": args.source_group,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.output, failed)
        print(json.dumps(failed, sort_keys=True))
        return 1
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
