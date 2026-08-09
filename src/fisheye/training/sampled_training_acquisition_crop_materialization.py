"""Materialize reviewed sampled rows from lossless acquisition crop videos.

The acquisition crop recorder owns a native 384x384 pixel window whose
placement may differ from Palette's later centered-crop rule.  This provider
therefore persists the recorder placement when it contains the reviewed box
and records an explicit, row-level full-frame fallback otherwise.  It never
silently mixes pixel authorities or resizes acquisition pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.crop_roi_layout import (
    build_canonical_crop_roi_layout,
    build_crop_roi_create_kwargs,
    crop_roi_layout_attrs,
)
from fisheye.shared.pynvvc_exact_seek import decode_one_frame_from_preceding_keyframe
from fisheye.shared.row_source_signature import build_row_source_signatures
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.array_contracts import (
    ArrayContract,
    DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
    INT64,
    UINT8,
)
from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_runtime import sha256_file
from fisheye.shared.zarr.crop_schema import CROP_GEOMETRY_SCHEMA_V1
from fisheye.shared.zarr.crop_storage import plan_crop_geometry_storage
from fisheye.shared.zarr.sampled_training_crop_materialization import (
    SampledTrainingCropPlan,
    build_sampled_training_crop_plan,
    write_by_physical_units,
    zero_padded_crop,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import TRAINING_IMMUTABLE_V1
from fisheye.shared.zarr.training_crop_materialization import (
    ACQUISITION_HYBRID_FALLBACK_REASON_CODE_MAP,
    ACQUISITION_HYBRID_PIXEL_SOURCE_CODE_MAP,
    SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER,
    SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_ID,
    SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_VERSION,
    TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE,
    TRAINING_CROP_MATERIALIZATION_PROVIDERS,
    TRAINING_CROP_MATERIALIZATION_SCHEMA_ID,
    build_training_crop_materialization_binding,
)
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.utils.export_acquisition_crop_pose_training_zarr import (
    load_crop_meta_table,
)

ACQUISITION_HYBRID_CROP_WRITER_SCHEMA_ID = (
    "palette.sampled_acquisition_crop_video_hybrid_writer"
)
ACQUISITION_HYBRID_CROP_WRITER_SCHEMA_VERSION = 1

PIXEL_SOURCE_CODE_MAP = ACQUISITION_HYBRID_PIXEL_SOURCE_CODE_MAP
FALLBACK_REASON_CODE_MAP = ACQUISITION_HYBRID_FALLBACK_REASON_CODE_MAP

_AUXILIARY_CONTRACTS = {
    "source_training_row_indices": ArrayContract(
        "palette.training_crop.source_training_row_index",
        1,
        INT64,
        ("n_instances",),
        ("instance",),
        "Compact sampled-training frame row that supplied this observation.",
    ),
    "source_crop_meta_row_indices": ArrayContract(
        "palette.training_crop.source_crop_meta_row_index",
        1,
        INT64,
        ("n_instances",),
        ("instance",),
        "Crop metadata row, or -1 when the full-frame fallback supplied pixels.",
    ),
    "source_crop_video_frame_indices": ArrayContract(
        "palette.training_crop.source_crop_video_frame_index",
        1,
        INT64,
        ("n_instances",),
        ("instance",),
        "Crop-video frame index, or -1 for a full-frame fallback row.",
    ),
    "source_crop_local_frame_ids": ArrayContract(
        "palette.training_crop.source_crop_local_frame_id",
        1,
        INT64,
        ("n_instances",),
        ("instance",),
        "Recorder-local crop frame identity, or -1 for fallback pixels.",
    ),
    "pixel_source_codes": ArrayContract(
        "palette.training_crop.pixel_source_code",
        1,
        UINT8,
        ("n_instances",),
        ("instance",),
        "Exact per-row pixel authority code.",
    ),
    "fallback_reason_codes": ArrayContract(
        "palette.training_crop.fallback_reason_code",
        1,
        UINT8,
        ("n_instances",),
        ("instance",),
        "Reason acquisition pixels could not represent the reviewed row.",
    ),
}


@dataclass(frozen=True)
class AcquisitionCropSource:
    recording_dir: Path
    video_path: Path
    metadata_path: Path
    summary_path: Path
    shape_hw: tuple[int, int]
    encoder_contract: Mapping[str, Any]
    video_stat: Mapping[str, Any]
    metadata_sha256: str
    summary_sha256: str


@dataclass(frozen=True)
class AcquisitionHybridCropPlan:
    base: SampledTrainingCropPlan
    arrays: Mapping[str, np.ndarray]
    source: AcquisitionCropSource
    row_signature_attrs: Mapping[str, Any]
    acquisition_row_count: int
    fallback_row_count: int

    @property
    def row_count(self) -> int:
        return self.base.row_count


def _resolve_manifest_path(
    recording_dir: Path, value: object, *, fallback_glob: str, label: str
) -> Path:
    if value:
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = recording_dir / candidate
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(
        (recording_dir / "derived" / "external_crop_recorder").glob(fallback_glob)
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} below {recording_dir}; found {len(matches)}."
        )
    return matches[0].resolve()


def resolve_acquisition_crop_source(
    recording_dir: str | Path,
    *,
    video_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> AcquisitionCropSource:
    """Resolve and fail closed on Orange's exact lossless crop declaration."""

    recording = Path(recording_dir).expanduser().resolve()
    manifest_path = recording / "recording_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Recording manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crop = (
        manifest.get("video_streams", {}).get("streams", {}).get("crop", {})
        if isinstance(manifest, Mapping)
        else {}
    )
    if not isinstance(crop, Mapping):
        raise ValueError("Recording manifest does not declare one crop stream.")
    video = _resolve_manifest_path(
        recording,
        video_path if video_path is not None else crop.get("video"),
        fallback_glob="*_crop_external.mp4",
        label="acquisition crop video",
    )
    metadata = _resolve_manifest_path(
        recording,
        metadata_path if metadata_path is not None else crop.get("metadata"),
        fallback_glob="*_crop_meta.csv",
        label="acquisition crop metadata",
    )
    summary = _resolve_manifest_path(
        recording,
        summary_path if summary_path is not None else crop.get("summary"),
        fallback_glob="*_crop_external_summary.json",
        label="acquisition crop summary",
    )
    document = json.loads(summary.read_text(encoding="utf-8"))
    encoder = document.get("video_metadata", {}).get("encoder", {})
    pixel = document.get("video_metadata", {}).get("source_pixel_contract", {})
    required_encoder = {
        "codec": "hevc",
        "tuning": "lossless",
        "rate_control_strategy": "lossless",
        "profile_name": "external_crop_hevc_lossless_gop1",
        "qp": 0,
        "resize_enabled": False,
        "requested_gop_length": 1,
        "resolved_gop_length": 1,
        "source_format": "mono8",
        "output_width": 384,
        "output_height": 384,
    }
    mismatches = {
        key: {"expected": expected, "observed": encoder.get(key)}
        for key, expected in required_encoder.items()
        if encoder.get(key) != expected
    }
    required_pixel = {
        "id": "orange.crop.mono8.v1",
        "dtype": "uint8",
        "value_range": "0_255",
        "pixel_format": "mono8",
        "memory_layout": "HxW",
        "width": 384,
        "height": 384,
    }
    mismatches.update(
        {
            f"source_pixel_contract.{key}": {
                "expected": expected,
                "observed": pixel.get(key),
            }
            for key, expected in required_pixel.items()
            if pixel.get(key) != expected
        }
    )
    if document.get("worker_failed") is not False:
        mismatches["worker_failed"] = {
            "expected": False,
            "observed": document.get("worker_failed"),
        }
    if int(document.get("encode_dropped", -1)) != 0:
        mismatches["encode_dropped"] = {
            "expected": 0,
            "observed": document.get("encode_dropped"),
        }
    if mismatches:
        raise ValueError(
            f"Acquisition crop stream is not the frozen lossless contract: {mismatches}"
        )
    stat = video.stat()
    encoder_contract = {
        "schema_id": "palette.orange_acquisition_crop_encoder_binding",
        "schema_version": 1,
        "encoder": {key: encoder[key] for key in required_encoder},
        "source_pixel_contract": {key: pixel[key] for key in required_pixel},
        "container_color_range": pixel.get("encoded_color_range"),
        "frames_encoded": int(document.get("frames_encoded", 0)),
        "encode_dropped": int(document.get("encode_dropped", 0)),
    }
    return AcquisitionCropSource(
        recording_dir=recording,
        video_path=video,
        metadata_path=metadata,
        summary_path=summary,
        shape_hw=(384, 384),
        encoder_contract=encoder_contract,
        video_stat={
            "strategy": "stat_v1",
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        metadata_sha256=sha256_file(metadata),
        summary_sha256=sha256_file(summary),
    )


def build_sampled_acquisition_crop_hybrid_plan(
    root: zarr.Group,
    *,
    refined_run_id: str,
    source: AcquisitionCropSource,
    allow_full_frame_fallback: bool = True,
) -> AcquisitionHybridCropPlan:
    """Map every reviewed positive row to one explicit pixel authority."""

    roi_height, roi_width = source.shape_hw
    base = build_sampled_training_crop_plan(
        root,
        refined_run_id=refined_run_id,
        roi_size_wh=(roi_width, roi_height),
    )
    metadata = load_crop_meta_table(source.metadata_path)
    valid_frames = metadata.frame_indices >= 0
    frame_values = metadata.frame_indices[valid_frames]
    if np.unique(frame_values).shape[0] != frame_values.shape[0]:
        raise ValueError("Crop metadata contains duplicate recording-frame identities.")
    frame_to_meta = {
        int(frame): int(row)
        for frame, row in zip(
            metadata.frame_indices[valid_frames], metadata.row_indices[valid_frames]
        )
    }
    if int(source.encoder_contract["frames_encoded"]) != int(
        metadata.frame_indices.shape[0]
    ):
        raise ValueError(
            "Lossless crop summary frame count differs from crop metadata rows."
        )

    arrays = {
        name: np.array(values, copy=True, order="C")
        for name, values in base.arrays.items()
    }
    row_count = base.row_count
    coordinates = np.asarray(arrays["roi_coordinates_full"], dtype=np.int32).copy()
    bbox_img = np.asarray(arrays["bbox_img_xyxy"], dtype=np.float32)
    acquisition_frames = np.asarray(
        arrays["source_acquisition_frame_index"], dtype=np.int64
    )
    local_frames = np.asarray(arrays["frame_indices"], dtype=np.int64)
    meta_rows = np.full(row_count, -1, dtype=np.int64)
    video_rows = np.full(row_count, -1, dtype=np.int64)
    local_crop_rows = np.full(row_count, -1, dtype=np.int64)
    source_codes = np.ones(row_count, dtype=np.uint8)
    reason_codes = np.zeros(row_count, dtype=np.uint8)

    for row in range(row_count):
        meta_row = frame_to_meta.get(int(acquisition_frames[row]))
        reason = 0
        recorded_xy: tuple[int, int] | None = None
        if meta_row is None:
            reason = 1
        elif bool(metadata.blank_frame[meta_row]):
            reason = 2
        elif not bool(metadata.has_detection[meta_row]):
            reason = 3
        elif (
            int(metadata.video_frame_indices[meta_row]) < 0
            or int(metadata.local_frame_ids[meta_row]) < 0
        ):
            reason = 4
        else:
            crop_xywh = np.asarray(metadata.crop_xywh[meta_row], dtype=np.float64)
            integral = np.rint(crop_xywh)
            if (
                not np.isfinite(crop_xywh).all()
                or not np.allclose(crop_xywh, integral, rtol=0.0, atol=1e-6)
                or tuple(int(value) for value in integral[2:])
                != (roi_width, roi_height)
            ):
                reason = 4
            else:
                x, y = (int(value) for value in integral[:2])
                box = bbox_img[row].astype(np.float64)
                if (
                    box[0] < x
                    or box[1] < y
                    or box[2] > x + roi_width
                    or box[3] > y + roi_height
                ):
                    reason = 5
                else:
                    recorded_xy = (x, y)
        if recorded_xy is not None:
            coordinates[row] = recorded_xy
            meta_rows[row] = int(meta_row)
            video_rows[row] = int(metadata.video_frame_indices[meta_row])
            local_crop_rows[row] = int(metadata.local_frame_ids[meta_row])
            source_codes[row] = 0
            reason_codes[row] = 0
        else:
            if not allow_full_frame_fallback:
                raise ValueError(
                    f"Reviewed row {row} requires fallback reason {reason}: "
                    f"{FALLBACK_REASON_CODE_MAP[str(reason)]}."
                )
            reason_codes[row] = np.uint8(reason)

    sizes = np.repeat(
        np.asarray([[roi_width, roi_height]], dtype=np.int32), row_count, axis=0
    )
    source_crop = np.concatenate(
        (coordinates.astype(np.float32), sizes.astype(np.float32)), axis=1
    )
    bbox_roi = bbox_img - np.concatenate(
        (coordinates.astype(np.float32), coordinates.astype(np.float32)), axis=1
    )
    if row_count and (
        np.any(bbox_roi[:, :2] < 0)
        or np.any(bbox_roi[:, 2] > roi_width)
        or np.any(bbox_roi[:, 3] > roi_height)
    ):
        raise ValueError(
            "A reviewed detection exceeds both its acquisition crop and fallback extent."
        )
    arrays.update(
        {
            "roi_coordinates_full": coordinates,
            "roi_sizes_full": sizes,
            "source_crop_xywh": source_crop,
            "bbox_roi_xyxy": bbox_roi,
            "source_training_row_indices": local_frames.copy(),
            "source_crop_meta_row_indices": meta_rows,
            "source_crop_video_frame_indices": video_rows,
            "source_crop_local_frame_ids": local_crop_rows,
            "pixel_source_codes": source_codes,
            "fallback_reason_codes": reason_codes,
        }
    )
    signatures = build_row_source_signatures(
        stage="sampled_acquisition_crop_video_hybrid_materialization",
        instance_keys=np.asarray(arrays["instance_key"], dtype=np.uint64),
        content_components={
            "source_refined_row_ids": arrays["source_refined_row_ids"],
            "frame_indices": arrays["frame_indices"],
            "bbox_norm_coords": arrays["bbox_norm_coords"],
            "roi_coordinates_full": coordinates,
            "roi_sizes_full": sizes,
            "pixel_source_codes": source_codes,
            "fallback_reason_codes": reason_codes,
        },
        compatibility_context={
            "provider": SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER,
            "source_refined_detect_run": refined_run_id,
            "source_frame_decision_digest": base.source_frame_decision_digest,
            "source_crop_meta_sha256": source.metadata_sha256,
            "source_crop_summary_sha256": source.summary_sha256,
            "source_crop_video_stat": dict(source.video_stat),
            "roi_size_wh": [roi_width, roi_height],
            "pixel_source_code_map": PIXEL_SOURCE_CODE_MAP,
            "fallback_reason_code_map": FALLBACK_REASON_CODE_MAP,
        },
    )
    arrays["source_row_signature"] = signatures.signatures
    return AcquisitionHybridCropPlan(
        base=base,
        arrays=arrays,
        source=source,
        row_signature_attrs=signatures.spec.to_attrs(),
        acquisition_row_count=int(np.count_nonzero(source_codes == 0)),
        fallback_row_count=int(np.count_nonzero(source_codes == 1)),
    )


def decode_acquisition_crop_frames_exact(
    video_path: Path,
    frame_indices: Sequence[int],
    *,
    expected_shape_hw: tuple[int, int],
    gpu_id: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Decode sparse GOP-1 frames with an exact seek proof for every row."""

    requested = sorted(set(int(value) for value in frame_indices))
    if not requested:
        return {}, {"backend": "not_run_no_acquisition_rows", "seek_count": 0}
    try:
        import PyNvVideoCodec as nvc  # type: ignore
    except Exception as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError(f"PyNvVideoCodec is unavailable: {exc}") from exc
    try:
        import torch

        use_torch = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - optional materializer
        torch = None  # type: ignore[assignment]
        use_torch = False
    if not use_torch:
        try:
            import cupy as cp
        except Exception as exc:  # pragma: no cover - workstation dependency
            raise RuntimeError(
                "Exact PyNvVideoCodec decode requires CUDA-enabled torch or CuPy."
            ) from exc
    demuxer = nvc.CreateDemuxer(filename=str(video_path))
    height, width = int(demuxer.Height()), int(demuxer.Width())
    if (height, width) != expected_shape_hw:
        raise RuntimeError(
            f"Crop video shape {(height, width)} differs from {expected_shape_hw}."
        )

    def materialize(frame: Any) -> np.ndarray:
        if use_torch:
            tensor = torch.from_dlpack(frame)
            result = (
                tensor[:height, :]
                .contiguous()
                .cpu()
                .numpy()
                .astype(np.uint8, copy=True)
            )
            del tensor
        else:
            array = cp.from_dlpack(frame)
            result = cp.asnumpy(array[:height, :]).astype(np.uint8, copy=False)
            del array
        return result

    frames: dict[int, np.ndarray] = {}
    proofs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for target in requested:
        decoder = nvc.CreateDecoder(
            gpuid=int(gpu_id), codec=demuxer.GetNvCodecId(), usedevicememory=True
        )
        frame, proof = decode_one_frame_from_preceding_keyframe(
            demuxer=demuxer,
            decoder=decoder,
            target_frame_index=target,
            materialize_frame=materialize,
        )
        if int(proof["target_packet_number"]) != 1:
            raise RuntimeError(
                f"Crop frame {target} is not independently seekable under GOP-1."
            )
        frames[target] = frame
        proofs.append(proof)
        del decoder
    return frames, {
        "backend": (
            "pynvvc_luma_exact_gop1_seek_torch"
            if use_torch
            else "pynvvc_luma_exact_gop1_seek_cupy"
        ),
        "gpu_id": int(gpu_id),
        "seek_count": len(proofs),
        "decoded_packet_count_total": sum(
            int(item["packets_submitted_through_target_output"]) for item in proofs
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "exact_frame_proof": (
            "demuxer_isSeekDone_exact_monotonic_pts_ordered_display_queue"
        ),
    }


def _write_auxiliary_array(
    run: zarr.Group, *, name: str, values: np.ndarray, row_count: int
) -> None:
    contract = _AUXILIARY_CONTRACTS[name]
    intent = contract.storage_intent(
        shape=values.shape,
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        access_unit_shape=(1,),
        growth_axis=0,
        shard_axes=(0,),
        name=name,
        dimensions={"n_instances": row_count},
    )
    storage = plan_storage(intent, TRAINING_IMMUTABLE_V1)
    destination = create_array_from_plan(
        run,
        name=name,
        contract=contract,
        plan=storage,
        fill_value=0,
        attributes={"artifact_class": "training_crop_pixel_lineage"},
    )
    write_by_physical_units(destination, values, plan=storage)


def write_sampled_acquisition_crop_hybrid(
    archive_path: str | Path,
    *,
    run_id: str,
    refined_run_id: str,
    source: AcquisitionCropSource,
    published_archive_path: str | Path | None = None,
    gpu_id: int = 0,
    allow_full_frame_fallback: bool = True,
    decoder: Callable[..., tuple[dict[int, np.ndarray], dict[str, Any]]] = (
        decode_acquisition_crop_frames_exact
    ),
) -> dict[str, Any]:
    """Write and verify one complete acquisition-hybrid crop run."""

    archive = Path(archive_path).expanduser().resolve()
    archive_identity = (
        archive
        if published_archive_path is None
        else Path(published_archive_path).expanduser().resolve()
    )
    candidate = str(run_id).strip()
    if not candidate or "/" in candidate or candidate.startswith("."):
        raise ValueError("run_id must be one safe non-hidden child-group name.")
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    plan = build_sampled_acquisition_crop_hybrid_plan(
        root,
        refined_run_id=refined_run_id,
        source=source,
        allow_full_frame_fallback=allow_full_frame_fallback,
    )
    parent = require_runs_parent(root, "crop_runs")
    if candidate in parent:
        raise FileExistsError(
            f"Training crop run already exists: crop_runs/{candidate}."
        )
    run = parent.create_group(candidate)
    mark_run_started(run, run_name=candidate, stage="crop")
    storage = plan_crop_geometry_storage(
        plan.base.dimensions, profile=TRAINING_IMMUTABLE_V1
    )
    roi_height, roi_width = source.shape_hw
    layout = build_canonical_crop_roi_layout(
        total_rois=plan.row_count,
        preferred_chunk_len=1,
        roi_storage="compressed",
        use_sharding=False,
    )
    run.attrs.update(
        {
            "status": "running",
            "stage_selector_eligible": False,
            "immutable_training_materialization": True,
            "artifact_class": "sampled_training_materialized_crops",
            "logical_schema_id": SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_ID,
            "logical_schema_version": SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_VERSION,
            "storage_plan": storage.as_manifest(),
            "training_materialization_schema": TRAINING_CROP_MATERIALIZATION_SCHEMA_ID,
            "training_materialization_provider": (
                SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER
            ),
            "training_materialization_provider_contract": list(
                TRAINING_CROP_MATERIALIZATION_PROVIDERS
            ),
            "source_crop_archive_path": str(archive_identity),
            "source_crop_run": plan.base.refined_run_id,
            "source_crop_path": (
                f"refined_detect_runs/{plan.base.refined_run_id}/instances"
            ),
            "source_crop_manifest_binding": {
                "authority_kind": "sampled_detection_review_plus_acquisition_crop_video",
                "source_refined_detect_run": plan.base.refined_run_id,
                "source_frame_decision_digest": plan.base.source_frame_decision_digest,
                "source_crop_meta_sha256": source.metadata_sha256,
                "source_crop_summary_sha256": source.summary_sha256,
            },
            "source_refined_detect_run": plan.base.refined_run_id,
            "source_frame_decision_path": plan.base.source_frame_decision_path,
            "source_frame_decision_digest": plan.base.source_frame_decision_digest,
            "source_images_path": "raw_video/images_full",
            "source_images_dtype": "uint8",
            "source_images_shape": list(plan.base.source_images_shape),
            "acquisition_crop_video_path": str(source.video_path),
            "acquisition_crop_video_stat": dict(source.video_stat),
            "acquisition_crop_meta_path": str(source.metadata_path),
            "acquisition_crop_meta_sha256": source.metadata_sha256,
            "acquisition_crop_summary_path": str(source.summary_path),
            "acquisition_crop_summary_sha256": source.summary_sha256,
            "acquisition_encoder_contract": dict(source.encoder_contract),
            "acquisition_crop_shape": [roi_height, roi_width],
            "pixel_source_code_map": PIXEL_SOURCE_CODE_MAP,
            "fallback_reason_code_map": FALLBACK_REASON_CODE_MAP,
            "fallback_policy": "sampled_images_full_zero_padded_v1",
            "source_pixels": "lossless_acquisition_crop_or_explicit_full_frame_fallback",
            "source_pixel_range": "0_255",
            "crop_storage_mode": "materialized",
            "coordinate_contract": "sampled_training_local_to_acquisition_v1",
            "height": plan.base.dimensions.source_height,
            "width": plan.base.dimensions.source_width,
            "roi_size": [roi_height, roi_width],
            "pixel_verification": "all_rows_byte_equal_to_declared_provider_v1",
            **crop_roi_layout_attrs(layout),
            **dict(plan.row_signature_attrs),
        }
    )

    bindings = {binding.path: binding for binding in CROP_GEOMETRY_SCHEMA_V1.bindings}
    for entry in storage.entries:
        name = entry.rule.path
        binding = bindings[name]
        contract = CROP_GEOMETRY_SCHEMA_V1.contracts.resolve(
            binding.contract_id, binding.contract_version
        )
        destination = create_array_from_plan(
            run,
            name=name,
            contract=contract,
            plan=entry.plan,
            fill_value=0,
            attributes={"artifact_class": "sampled_training_crop_geometry"},
        )
        write_by_physical_units(
            destination, np.asarray(plan.arrays[name]), plan=entry.plan
        )

    source_frame_contract = DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1
    source_frame_intent = source_frame_contract.storage_intent(
        shape=plan.base.source_frame_indices.shape,
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        access_unit_shape=(1,),
        growth_axis=0,
        shard_axes=(0,),
        name="source_frame_indices",
        dimensions={"n_instances": plan.row_count},
    )
    source_frame_plan = plan_storage(source_frame_intent, TRAINING_IMMUTABLE_V1)
    source_frame_array = create_array_from_plan(
        run,
        name="source_frame_indices",
        contract=source_frame_contract,
        plan=source_frame_plan,
        fill_value=0,
        attributes={"identity_semantics": "source_acquisition_frame_index"},
    )
    write_by_physical_units(
        source_frame_array, plan.base.source_frame_indices, plan=source_frame_plan
    )
    for name in _AUXILIARY_CONTRACTS:
        _write_auxiliary_array(
            run,
            name=name,
            values=np.asarray(plan.arrays[name]),
            row_count=plan.row_count,
        )

    video_rows = np.asarray(
        plan.arrays["source_crop_video_frame_indices"], dtype=np.int64
    )
    requested_video_rows = video_rows[video_rows >= 0]
    decoded, decode_evidence = decoder(
        source.video_path,
        requested_video_rows.tolist(),
        expected_shape_hw=source.shape_hw,
        gpu_id=int(gpu_id),
    )
    run.attrs["decode_backend"] = decode_evidence.get("backend")
    run.attrs["decode_evidence"] = decode_evidence
    roi_images = run.create_array(
        "roi_images",
        **build_crop_roi_create_kwargs(
            total_rois=plan.row_count,
            roi_sz=(roi_height, roi_width),
            layout=layout,
            overwrite=False,
        ),
    )
    source_images = root["raw_video/images_full"]
    local_frames = np.asarray(plan.arrays["frame_indices"], dtype=np.int64)
    coordinates = np.asarray(plan.arrays["roi_coordinates_full"], dtype=np.int32)
    source_codes = np.asarray(plan.arrays["pixel_source_codes"], dtype=np.uint8)
    for row in range(plan.row_count):
        if int(source_codes[row]) == 0:
            crop = decoded.get(int(video_rows[row]))
            if crop is None:
                raise RuntimeError(f"Missing decoded acquisition crop for row {row}.")
        else:
            frame = np.asarray(source_images[int(local_frames[row])], dtype=np.uint8)
            x, y = (int(value) for value in coordinates[row])
            crop = zero_padded_crop(frame, x=x, y=y, width=roi_width, height=roi_height)
        if crop.shape != (roi_height, roi_width) or crop.dtype != np.uint8:
            raise RuntimeError(f"Pixel provider returned invalid crop for row {row}.")
        roi_images[row] = crop
        if not np.array_equal(np.asarray(roi_images[row]), crop):
            raise RuntimeError(
                f"Persisted crop row {row} differs from its declared pixel provider."
            )

    run.attrs["summary_statistics"] = {
        "total_rois_cropped": plan.row_count,
        "acquisition_crop_video_rows": plan.acquisition_row_count,
        "sampled_images_full_fallback_rows": plan.fallback_row_count,
        "fallback_reason_counts": {
            FALLBACK_REASON_CODE_MAP[str(code)]: int(
                np.count_nonzero(plan.arrays["fallback_reason_codes"] == code)
            )
            for code in range(1, 6)
        },
        "pixel_rows_verified": plan.row_count,
    }
    run.attrs[TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE] = (
        build_training_crop_materialization_binding(run)
    )
    run.attrs["training_crop_materialization_binding_status"] = "strict_v1"
    run.attrs["status"] = "completed"
    mark_run_complete(
        run,
        run_name=candidate,
        run_provenance=build_writer_run_provenance(
            command=(
                "fisheye.training.sampled_training_acquisition_crop_materialization"
            ),
            params={
                "writer_schema_id": ACQUISITION_HYBRID_CROP_WRITER_SCHEMA_ID,
                "writer_schema_version": ACQUISITION_HYBRID_CROP_WRITER_SCHEMA_VERSION,
                "materialization_provider": (
                    SAMPLED_ACQUISITION_CROP_HYBRID_MATERIALIZATION_PROVIDER
                ),
                "fallback_policy": "sampled_images_full_zero_padded_v1",
                "pixel_verification": "all_rows_byte_equal_to_declared_provider_v1",
            },
            input_run_ids={
                "source_refined_detect_run": plan.base.refined_run_id,
                "source_frame_decision_path": plan.base.source_frame_decision_path,
                "source_crop_meta_sha256": source.metadata_sha256,
                "source_crop_summary_sha256": source.summary_sha256,
            },
        ),
    )
    run.attrs["status"] = "completed"
    run.attrs[TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE] = (
        build_training_crop_materialization_binding(run)
    )
    return {
        "schema_id": ACQUISITION_HYBRID_CROP_WRITER_SCHEMA_ID,
        "schema_version": ACQUISITION_HYBRID_CROP_WRITER_SCHEMA_VERSION,
        "status": "complete",
        "run_id": candidate,
        "row_count": plan.row_count,
        "roi_shape": [roi_height, roi_width],
        "acquisition_crop_video_rows": plan.acquisition_row_count,
        "sampled_images_full_fallback_rows": plan.fallback_row_count,
        "binding_digest": run.attrs[TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE][
            "payload_digest"
        ],
        "decode_evidence": decode_evidence,
        "storage_plan": storage.as_manifest(),
        "stage_selector_eligible": False,
    }


__all__ = [
    "ACQUISITION_HYBRID_CROP_WRITER_SCHEMA_ID",
    "ACQUISITION_HYBRID_CROP_WRITER_SCHEMA_VERSION",
    "AcquisitionCropSource",
    "AcquisitionHybridCropPlan",
    "FALLBACK_REASON_CODE_MAP",
    "PIXEL_SOURCE_CODE_MAP",
    "build_sampled_acquisition_crop_hybrid_plan",
    "decode_acquisition_crop_frames_exact",
    "resolve_acquisition_crop_source",
    "write_sampled_acquisition_crop_hybrid",
]
