"""Compare pose inference on native, synthetic, and real source-context ROIs.

This is a bounded diagnostic.  It never writes an analysis archive, selector,
or registry.  The real-context profile is materialized into caller-owned
scratch and the small JSON result is published atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.cluster.keypoints.common import validate_flat_roi_cache_binding
from fisheye.shared.flat_roi_cache import (
    load_flat_roi_cache_manifest,
    write_pynvvc_luma_roi_payload,
)
from fisheye.shared.pose_model_input_contract import (
    load_pose_model_input_contract,
    validate_pose_runtime_compatibility,
)


RESULT_SCHEMA_ID = "palette.pose_source_context_benchmark"
RESULT_SCHEMA_VERSION = 1
METADATA_MODE = "unconsolidated_explicit_diagnostic"


def select_sample_indices(
    total_rows: int,
    sample_count: int,
    *,
    mode: str,
) -> np.ndarray:
    """Select a deterministic, strictly increasing row sample."""

    total = int(total_rows)
    count = min(int(sample_count), total)
    if total <= 0 or count <= 0:
        raise ValueError("Pose context benchmark requires positive row/sample counts.")
    if mode == "first":
        return np.arange(count, dtype=np.int64)
    if mode == "even":
        # Bin midpoints avoid duplicate indices while covering the full axis.
        numerators = (2 * np.arange(count, dtype=np.int64) + 1) * total
        return np.asarray(numerators // (2 * count), dtype=np.int64)
    raise ValueError("Sample mode must be 'first' or 'even'.")


def derive_centered_context_coordinates(
    native_coordinates_xy: np.ndarray,
    *,
    native_shape_hw: tuple[int, int],
    context_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Translate native top-left coordinates to a larger centered window."""

    coordinates = np.asarray(native_coordinates_xy, dtype=np.int32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Native ROI coordinates must have exact shape [N, 2].")
    native_h, native_w = (int(native_shape_hw[0]), int(native_shape_hw[1]))
    context_h, context_w = (int(context_shape_hw[0]), int(context_shape_hw[1]))
    delta_h, delta_w = context_h - native_h, context_w - native_w
    if min(native_h, native_w) <= 0 or min(context_h, context_w) <= 0:
        raise ValueError("Native and context shapes must be positive.")
    if delta_h < 0 or delta_w < 0 or delta_h % 2 or delta_w % 2:
        raise ValueError(
            "Source-context extent must be a centered, non-smaller even expansion."
        )
    offset_y, offset_x = delta_h // 2, delta_w // 2
    translated = coordinates - np.asarray([offset_x, offset_y], dtype=np.int32)
    return translated, {
        "schema_id": "palette.pose_source_context_transform",
        "schema_version": 1,
        "native_shape_hw": [native_h, native_w],
        "context_shape_hw": [context_h, context_w],
        "context_top_left_from_native_top_left_xy": [-offset_x, -offset_y],
        "native_xy_from_context_xy": [-offset_x, -offset_y],
        "coordinate_semantics": "continuous_pixel_centers_xy",
        "source_pixels": "source_camera_luma_with_zero_only_outside_frame",
    }


def summarize_confidence_scores(
    scores: Sequence[float | None],
    *,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Summarize one maximum detection score per sampled ROI."""

    finite = np.asarray(
        [float(value) for value in scores if value is not None], dtype=np.float64
    )
    return {
        "sample_count": int(len(scores)),
        "detected_at_prediction_floor": int(finite.size),
        "max_confidence": float(finite.max()) if finite.size else None,
        "median_detected_confidence": (
            float(np.median(finite)) if finite.size else None
        ),
        "count_by_threshold": {
            format(float(threshold), ".12g"): int(
                sum(
                    value is not None and float(value) >= float(threshold)
                    for value in scores
                )
            )
            for threshold in thresholds
        },
    }


def _array_sha256(array: np.ndarray, *, dtype: np.dtype[Any]) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object.")
    return value


def _predict_profile(
    model: Any,
    pixels: np.ndarray,
    *,
    network_imgsz: int,
    prediction_floor: float,
    thresholds: Sequence[float],
    device: str,
) -> dict[str, Any]:
    batch = np.asarray(pixels, dtype=np.uint8)
    rgb = [np.repeat(image[:, :, None], 3, axis=2) for image in batch]
    results = tuple(
        model.predict(
            rgb,
            imgsz=int(network_imgsz),
            conf=float(prediction_floor),
            iou=0.7,
            max_det=300,
            rect=False,
            device=device,
            verbose=False,
            stream=True,
        )
    )
    if len(results) != int(batch.shape[0]):
        raise RuntimeError("Pose benchmark result cardinality differs from its sample.")
    scores: list[float | None] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        confidence = getattr(boxes, "conf", None)
        if confidence is None or int(confidence.numel()) == 0:
            scores.append(None)
        else:
            scores.append(float(confidence.max().detach().cpu()))
    return {
        "pixel_sha256": _array_sha256(batch, dtype=np.dtype("u1")),
        "shape": list(batch.shape),
        "summary": summarize_confidence_scores(scores, thresholds=thresholds),
    }


def benchmark_pose_source_context(
    *,
    analysis_zarr: Path,
    crop_run: str,
    cache_manifest: Path,
    model_path: Path,
    model_set_id: str,
    model_run_id: str,
    model_sha256: str,
    model_input_contract: Path,
    output_json: Path,
    scratch_dir: Path,
    sample_count: int,
    sample_mode: str,
    prediction_floor: float,
    thresholds: Sequence[float],
    device: str,
) -> dict[str, Any]:
    """Run the bounded comparison and atomically persist its result."""

    archive = analysis_zarr.expanduser().resolve()
    manifest_path = cache_manifest.expanduser().resolve()
    binding = validate_flat_roi_cache_binding(
        manifest_path=manifest_path,
        analysis_zarr=archive,
        crop_run=crop_run,
        min_roi_size=1,
    )
    contract = load_pose_model_input_contract(
        model_input_contract.expanduser().resolve(),
        model_path=model_path.expanduser().resolve(),
        expected_set_id=model_set_id,
        expected_run_id=model_run_id,
        expected_model_sha256=model_sha256,
    )
    runtime_compatibility = validate_pose_runtime_compatibility(contract)
    native_shape = (int(binding.shape[1]), int(binding.shape[2]))
    runtime_plan = contract.plan_for_native_shape(native_shape)
    if runtime_plan.transform.is_identity:
        raise ValueError("Context benchmark requires a smaller native crop extent.")

    thresholds_tuple = tuple(float(value) for value in thresholds)
    if not thresholds_tuple or any(
        not np.isfinite(value) or value <= 0.0 or value > 1.0
        for value in thresholds_tuple
    ):
        raise ValueError("Thresholds must be finite values in (0, 1].")
    if float(prediction_floor) > min(thresholds_tuple):
        raise ValueError("Prediction floor must not exceed a reported threshold.")

    indices = select_sample_indices(
        binding.shape[0], sample_count, mode=sample_mode
    )
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    crop = root["crop_runs"][crop_run]
    expected_rows = int(binding.shape[0])
    required = {
        "frame_indices": np.dtype("<i8"),
        "source_acquisition_frame_index": np.dtype("<i8"),
        "roi_coordinates_full": np.dtype("<i4"),
        "instance_key": np.dtype("<u8"),
    }
    sampled: dict[str, np.ndarray] = {}
    for name, dtype in required.items():
        array = crop[name]
        if int(array.shape[0]) != expected_rows or np.dtype(array.dtype) != dtype:
            raise ValueError(f"Crop array {name} differs from the exact v2 contract.")
        sampled[name] = np.asarray(array[indices], dtype=dtype)
    if not np.array_equal(
        sampled["frame_indices"], sampled["source_acquisition_frame_index"]
    ):
        raise ValueError("Crop frame axes disagree for source-camera decoding.")
    frames = sampled["source_acquisition_frame_index"]
    if np.any(np.diff(frames) < 0):
        raise ValueError("Sampled source frames must be nondecreasing for decoding.")

    cache_document = load_flat_roi_cache_manifest(manifest_path)
    source = _mapping(cache_document.get("source"), field="cache source")
    video_path = Path(str(source.get("frame_source_path") or "")).expanduser().resolve()
    identity = _mapping(
        source.get("frame_source_identity"), field="cache source frame identity"
    )
    raw_frame_shape = identity.get("frame_shape")
    if (
        not isinstance(raw_frame_shape, list)
        or len(raw_frame_shape) != 2
        or any(type(value) is not int or value <= 0 for value in raw_frame_shape)
    ):
        raise ValueError("Cache source frame identity lacks exact frame_shape.")
    video_shape = (int(raw_frame_shape[0]), int(raw_frame_shape[1]))

    native_coordinates = sampled["roi_coordinates_full"]
    context_coordinates, context_transform = derive_centered_context_coordinates(
        native_coordinates,
        native_shape_hw=native_shape,
        context_shape_hw=contract.training_source_shape_hw,
    )
    scratch = scratch_dir.expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    context_payload = scratch / "real_source_context.bin"
    context_receipt = write_pynvvc_luma_roi_payload(
        video_path=video_path,
        frame_indices=frames,
        roi_coordinates_full=context_coordinates,
        roi_shape=contract.training_source_shape_hw,
        video_shape=video_shape,
        output_path=context_payload,
        batch_size=max(1, int(indices.size)),
        overwrite=True,
    )

    cache_pixels = np.memmap(
        binding.payload_path,
        dtype=np.uint8,
        mode="r",
        shape=binding.shape,
        order="C",
    )
    native_pixels = np.asarray(cache_pixels[indices], dtype=np.uint8)
    synthetic_pixels = runtime_plan.transform.apply_numpy_luma_batch(native_pixels)
    context_pixels = np.memmap(
        context_payload,
        dtype=np.uint8,
        mode="r",
        shape=(int(indices.size), *contract.training_source_shape_hw),
        order="C",
    )

    from ultralytics import YOLO

    model = YOLO(str(model_path.expanduser().resolve()))
    profiles = {
        "native_cache": _predict_profile(
            model,
            native_pixels,
            network_imgsz=runtime_plan.network_imgsz,
            prediction_floor=prediction_floor,
            thresholds=thresholds_tuple,
            device=device,
        ),
        "synthetic_contract_padding": _predict_profile(
            model,
            synthetic_pixels,
            network_imgsz=runtime_plan.network_imgsz,
            prediction_floor=prediction_floor,
            thresholds=thresholds_tuple,
            device=device,
        ),
        "real_source_context": _predict_profile(
            model,
            np.asarray(context_pixels),
            network_imgsz=runtime_plan.network_imgsz,
            prediction_floor=prediction_floor,
            thresholds=thresholds_tuple,
            device=device,
        ),
    }
    result = {
        "schema_id": RESULT_SCHEMA_ID,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "metadata_mode": METADATA_MODE,
        "analysis_zarr": str(archive),
        "crop_run": crop_run,
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": binding.manifest_sha256,
        "cache_payload_sha256": binding.payload_sha256,
        "model": contract.to_json(),
        "runtime_compatibility": runtime_compatibility,
        "sample": {
            "mode": sample_mode,
            "count": int(indices.size),
            "indices": indices.tolist(),
            "frame_indices_sha256": _array_sha256(frames, dtype=np.dtype("<i8")),
            "instance_key_sha256": _array_sha256(
                sampled["instance_key"], dtype=np.dtype("<u8")
            ),
            "native_coordinates_sha256": _array_sha256(
                native_coordinates, dtype=np.dtype("<i4")
            ),
            "context_coordinates_sha256": _array_sha256(
                context_coordinates, dtype=np.dtype("<i4")
            ),
        },
        "context_transform": context_transform,
        "context_materialization": context_receipt,
        "prediction": {
            "floor": float(prediction_floor),
            "thresholds": list(thresholds_tuple),
            "network_imgsz": runtime_plan.network_imgsz,
            "device": device,
        },
        "profiles": profiles,
        "archive_mutation_performed": False,
        "selector_activation_performed": False,
        "registry_mutation_performed": False,
    }
    destination = output_json.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-set-id", required=True)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-input-contract", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--sample-mode", choices=("first", "even"), default="first")
    parser.add_argument("--prediction-floor", type=float, default=0.001)
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        dest="thresholds",
        default=None,
    )
    parser.add_argument("--device", default="0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    thresholds = args.thresholds or [0.25, 0.05, 0.01, 0.001]
    result = benchmark_pose_source_context(
        analysis_zarr=args.analysis_zarr,
        crop_run=args.crop_run,
        cache_manifest=args.cache_manifest,
        model_path=args.model_path,
        model_set_id=args.model_set_id,
        model_run_id=args.model_run_id,
        model_sha256=args.model_sha256,
        model_input_contract=args.model_input_contract,
        output_json=args.output_json,
        scratch_dir=args.scratch_dir,
        sample_count=args.sample_count,
        sample_mode=args.sample_mode,
        prediction_floor=args.prediction_floor,
        thresholds=thresholds,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
