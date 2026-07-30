#!/usr/bin/env python3
"""Finalize terminal clip keypoints into a selector-ineligible v2 run chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.pose_model_schema_binding import load_pose_model_schema_binding
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.clipped_keypoint_finalization import (
    CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_ID,
    clip_terminal_result_from_yolo_arrays,
    publish_selector_ineligible_clipped_keypoint_chain,
    validate_clip_terminal_result_receipt,
)
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
)
from fisheye.shared.zarr.keypoint_manifest import (
    keypoint_preprocessing_from_manifest,
)
from fisheye.shared.zarr.refined_keypoint_manifest import (
    initial_refined_keypoint_snapshot_identity,
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


def _load_terminal_results(
    *,
    analysis_zarr: Path,
    crop: Any,
    receipt_paths: Sequence[Path],
    pose_model_schema_binding: Mapping[str, Any],
    preprocessing: Any,
) -> tuple[Any, ...]:
    archive = analysis_zarr.expanduser().resolve()
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    results = []
    for receipt_path in receipt_paths:
        receipt = _read_json(receipt_path)
        errors = validate_clip_terminal_result_receipt(receipt)
        if errors:
            raise ValueError(
                f"Invalid clip receipt {receipt_path}: " + "; ".join(errors)
            )
        payload = receipt["payload"]
        if Path(payload["analysis_zarr"]).expanduser().resolve() != archive:
            raise ValueError(
                f"Clip receipt {receipt_path} binds a different analysis archive."
            )
        source = _group(root, payload["source_group_path"])
        if not is_run_complete(source):
            raise ValueError(
                f"Clip receipt source is no longer complete: {receipt_path}"
            )
        if (
            source.attrs.get("coordinate_contract_mode")
            != (payload["source_coordinate_contract_mode"])
        ):
            raise ValueError(
                f"Clip source coordinate mode changed after receipt: {receipt_path}"
            )
        provenance = source.attrs.get("provenance")
        resolution = (
            provenance.get("model_resolution")
            if isinstance(provenance, Mapping)
            else None
        )
        artifacts = (
            resolution.get("artifacts") if isinstance(resolution, Mapping) else None
        )
        observed_binding = (
            artifacts.get("model_pose_schema_binding")
            if isinstance(artifacts, Mapping)
            else None
        )
        if (
            not isinstance(observed_binding, Mapping)
            or dict(observed_binding) != pose_model_schema_binding
        ):
            raise ValueError(
                f"Clip source model binding changed after receipt: {receipt_path}"
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
                f"Clip source preprocessing changed after receipt: {receipt_path}"
            )
        package_path = (
            Path(payload["input_package_manifest_path"]).expanduser().resolve()
        )
        declared_package = source.attrs.get("source_crop_pixel_work_package_manifest")
        if (
            not package_path.is_file()
            or not isinstance(declared_package, str)
            or Path(declared_package).expanduser().resolve() != package_path
            or _sha256_file(package_path)
            != payload["result"]["input_package_manifest_digest"]
        ):
            raise ValueError(
                f"Clip input package changed after receipt: {receipt_path}"
            )
        arrays = {path: source[path] for path in _SOURCE_ARRAY_PATHS}
        source_crop = _group(root, payload["source_crop_group_path"])
        expected_source_crop_path = (
            f"crop_runs/{str(source.attrs.get('source_crop_run') or '').strip()}"
        )
        if expected_source_crop_path != payload["source_crop_group_path"]:
            raise ValueError(
                f"Clip source crop binding changed after receipt: {receipt_path}"
            )
        if not is_run_complete(source_crop):
            raise ValueError(f"Clip source crop is no longer complete: {receipt_path}")
        roi_shape = payload["source_crop_roi_shape_hw"]
        observed_roi_shape = source_crop.attrs.get(
            "roi_shape"
        ) or source_crop.attrs.get("roi_size")
        if list(observed_roi_shape or []) != roi_shape:
            raise ValueError(
                f"Clip source crop ROI shape changed after receipt: {receipt_path}"
            )
        source_crop_arrays = {
            path: source_crop[path] for path in _SOURCE_CROP_ARRAY_PATHS
        }
        source_crop_arrays["roi_sizes_full"] = np.broadcast_to(
            np.asarray([int(roi_shape[1]), int(roi_shape[0])], dtype=np.int32),
            (int(source_crop["instance_key"].shape[0]), 2),
        )
        observed_hashes = {
            path: sha256_array(arrays[path][...]) for path in _SOURCE_ARRAY_PATHS
        }
        if observed_hashes != payload["source_array_hashes"]:
            raise ValueError(
                f"Clip result arrays changed after receipt: {receipt_path}"
            )
        observed_crop_hashes = {
            path: sha256_array(source_crop[path][...])
            for path in _SOURCE_CROP_ARRAY_PATHS
        }
        if observed_crop_hashes != payload["source_crop_array_hashes"]:
            raise ValueError(
                f"Clip source crop arrays changed after receipt: {receipt_path}"
            )
        persisted = payload["result"]
        result = clip_terminal_result_from_yolo_arrays(
            crop,
            arrays,
            source_crop_arrays=source_crop_arrays,
            clip_id=persisted["clip_id"],
            clip_index=persisted["clip_index"],
            pose_model_schema_binding=pose_model_schema_binding,
            preprocessing=preprocessing,
            input_package_manifest_digest=persisted["input_package_manifest_digest"],
        )
        if result.as_manifest() != persisted:
            raise ValueError(
                f"Clip result contract changed after receipt: {receipt_path}"
            )
        results.append(result)
    return tuple(results)


def finalize_clipped_keypoint_v2_bundle(
    *,
    analysis_zarr: Path,
    crop_archive: Path | None = None,
    refined_archive: Path | None = None,
    crop_run_id: str,
    clip_receipt_paths: Sequence[Path],
    pose_binding_path: Path,
    preprocessing_path: Path,
    bundle_root: Path,
    raw_run_id: str,
    quality_run_id: str,
    refined_run_id: str,
    body_frame_run_id: str,
    recording_identity: str,
    refined_lineage_id: str,
    refined_snapshot_id: str,
) -> dict[str, object]:
    if not clip_receipt_paths:
        raise ValueError("At least one --clip-receipt is required.")
    archive = analysis_zarr.expanduser().resolve()
    resolved_crop_archive = (
        archive if crop_archive is None else crop_archive.expanduser().resolve()
    )
    crop = open_persisted_crop_geometry_publication(
        resolved_crop_archive,
        run_id=crop_run_id,
        source_refined_archive=refined_archive,
    )
    binding = load_pose_model_schema_binding(pose_binding_path)
    preprocessing = keypoint_preprocessing_from_manifest(_read_json(preprocessing_path))
    clips = _load_terminal_results(
        analysis_zarr=archive,
        crop=crop,
        receipt_paths=tuple(path.expanduser().resolve() for path in clip_receipt_paths),
        pose_model_schema_binding=binding,
        preprocessing=preprocessing,
    )
    identity = initial_refined_keypoint_snapshot_identity(
        recording_identity=recording_identity,
        lineage_id=refined_lineage_id,
        snapshot_id=refined_snapshot_id,
    )
    chain = publish_selector_ineligible_clipped_keypoint_chain(
        crop,
        clips,
        pose_model_schema_binding=binding,
        preprocessing=preprocessing,
        bundle_root=bundle_root,
        raw_run_id=raw_run_id,
        quality_run_id=quality_run_id,
        refined_run_id=refined_run_id,
        body_frame_run_id=body_frame_run_id,
        refined_identity=identity,
    )
    return {
        "schema_id": CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_ID,
        "schema_version": 1,
        "status": "complete",
        "analysis_zarr": str(archive),
        "crop_archive": str(resolved_crop_archive),
        "crop_run_id": crop.run_id,
        "clip_receipt_paths": [
            str(path.expanduser().resolve()) for path in clip_receipt_paths
        ],
        "bundle_root": str(bundle_root.expanduser().resolve()),
        "finalization_receipt_path": str(chain.receipt_path),
        "finalization_receipt_digest": chain.receipt["payload_digest"],
        "outputs": chain.receipt["payload"]["outputs"],
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument(
        "--crop-archive",
        type=Path,
        help="Optional standalone crop-v2 archive; defaults to --analysis-zarr.",
    )
    parser.add_argument(
        "--refined-archive",
        type=Path,
        help=(
            "Optional standalone refined-detection source bound by crop-v2; "
            "defaults to the crop archive."
        ),
    )
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--clip-receipt", type=Path, action="append", required=True)
    parser.add_argument("--pose-binding", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--raw-run", required=True)
    parser.add_argument("--quality-run", required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--body-frame-run", required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--refined-lineage-id", required=True)
    parser.add_argument("--refined-snapshot-id", required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = finalize_clipped_keypoint_v2_bundle(
            analysis_zarr=args.analysis_zarr,
            crop_archive=args.crop_archive,
            refined_archive=args.refined_archive,
            crop_run_id=args.crop_run,
            clip_receipt_paths=args.clip_receipt,
            pose_binding_path=args.pose_binding,
            preprocessing_path=args.preprocessing,
            bundle_root=args.bundle_root,
            raw_run_id=args.raw_run,
            quality_run_id=args.quality_run,
            refined_run_id=args.refined_run,
            body_frame_run_id=args.body_frame_run,
            recording_identity=args.recording_identity,
            refined_lineage_id=args.refined_lineage_id,
            refined_snapshot_id=args.refined_snapshot_id,
        )
    except Exception as exc:
        result = {
            "schema_id": CLIPPED_KEYPOINT_FINALIZATION_SCHEMA_ID,
            "schema_version": 1,
            "status": "failed",
            "analysis_zarr": str(args.analysis_zarr),
            "crop_archive": str(args.crop_archive or args.analysis_zarr),
            "crop_run_id": args.crop_run,
            "bundle_root": str(args.bundle_root),
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    write_json_atomic(args.result_json, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
