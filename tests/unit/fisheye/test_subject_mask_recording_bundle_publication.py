from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest
import zarr

from fisheye.analysis.subject_shape_storage import (
    SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
)
from fisheye.analysis.subject_shape_runs import write_subject_shape_run_group
from fisheye.analysis_workflows.materializers.subject_shape import (
    build_subject_shape_materialization_plan,
    materialize_subject_shape,
)
from fisheye.cluster.subject_masks.publish_recording_bundle import (
    _refined_arrays,
    publish_recording_subject_mask_bundle,
)
from fisheye.shared.subject_mask_attempt import (
    build_subject_mask_attempt,
    build_subject_mask_scientific_identity,
)
from fisheye.shared.subject_mask_crop_placement import (
    normalize_subject_mask_crop_placement,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD,
    CANONICAL_SUBJECT_SHAPE_BUNDLE_PROFILE_ID,
    CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION,
    SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
    SUBJECT_SHAPE_SOURCE_BINDING_ATTR,
    SUBJECT_SHAPE_SOURCE_KIND_ATTR,
    SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR,
    SubjectShapeCoordinatePublicationError,
    activate_subject_shape_coordinate_publication,
    load_completed_ineligible_subject_shape_coordinate_publication,
    selector_snapshot,
)
from fisheye.shared.subject_mask_worker_receipt import (
    RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    build_subject_mask_worker_semantic_receipt,
)
from fisheye.shared.refined_subject_component_contours import (
    write_sampled_component_contour_arrays,
)
from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs
from fisheye.shared.import_video_metadata import (
    publish_external_video_acquisition_authority,
)
from fisheye.shared.source_video_metadata import build_source_video_metadata_v2
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.crop_manifest import (
    build_coordinate_crop_run_manifest,
    build_crop_row_source_signatures,
)
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropPlacementMode,
    CropSizeMode,
)
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR,
    activate_subject_mask_bundle,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    SubjectMaskCoreValidationMode,
)
from fisheye.shared.zarr.subject_mask_cache_publication import (
    validate_subject_mask_cache_run_manifest,
)
from fisheye.shared.zarr.subject_mask_final_layout_units import (
    build_subject_mask_final_layout_unit_package,
)
from fisheye.shared.zarr.subject_mask_quality_partition import (
    compute_subject_mask_quality_partition,
)
from fisheye.shared.zarr.subject_mask_bundle_coordinate_authority import (
    SubjectMaskBundleCoordinateAuthorityError,
    load_recording_subject_mask_coordinate_authority,
)
from fisheye.shared.zarr.subject_shape_bundle_source import (
    BoundSubjectShapeBundleSource,
    SubjectShapeBundleSourceError,
    load_subject_shape_bundle_source,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
    derive_subject_mask_frame_row_offsets,
    derive_subject_mask_metrics,
)
from fisheye.shared.zarr.subject_mask_sampled_contour_worker_receipt import (
    write_subject_mask_sampled_contour_worker_receipt,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    subject_mask_array_unit_document,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.utils.import_refined_subject_mask_clip_packages import (
    import_refined_subject_mask_clip_packages,
)
from tests.unit.fisheye.test_import_refined_subject_mask_clip_packages import (
    _write_package,
)
from tests.unit.fisheye.test_crop_manifest import (
    _arrays as _crop_manifest_arrays,
    _dimensions as _crop_manifest_dimensions,
    _metadata as _crop_manifest_metadata,
    _pixel as _crop_manifest_pixel,
    _policy as _crop_manifest_policy,
    _source as _crop_manifest_source,
)


def _create_array(group: zarr.Group, path: str, values: np.ndarray) -> None:
    parts = path.split("/")
    target = group
    for part in parts[:-1]:
        target = target.require_group(part)
    target.create_array(parts[-1], data=values, overwrite=True)


def _surfaces() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    masks = np.zeros((4, 4, 8, 8), dtype=np.uint8)
    masks[:, 0, 1:7, 1:7] = 1
    masks[:, 1, 2, 2] = 1
    masks[:, 2, 2, 5] = 1
    masks[:, 3, 5, 3] = 1
    metrics = derive_subject_mask_metrics(masks)
    probabilities = masks * np.uint8(255)
    raw = {
        "mask_probs_roi": probabilities,
        "available_channels": np.ones((4,), dtype=bool),
        "metrics/prob_max": (
            np.max(probabilities, axis=(2, 3)).astype(np.float32) / np.float32(255.0)
        ),
        **{f"metrics/{name}": value for name, value in metrics.items()},
    }
    refined = {
        "masks_roi": masks,
        "available_channels": np.ones((4,), dtype=bool),
        **{f"metrics/{name}": value for name, value in metrics.items()},
    }
    return raw, refined


def _seal_worker(
    draft_path: Path,
    run: zarr.Group,
    *,
    stage_kind: str,
    paths: tuple[str, ...],
    scientific_identity: dict[str, object] | None = None,
) -> None:
    row_count = int(run["source_crop_row_ids"].shape[0])
    run_path = str(run.path).strip("/")
    science = scientific_identity or build_subject_mask_scientific_identity(
        stage_kind=stage_kind,
        model={"artifact": "pytest"},
        crop={"run_id": "crop_001"},
        pixels={"digest": "a" * 64},
        row_identity={"rows": row_count, "run_path": run_path},
        inference_contract={"components": list(run.attrs["mask_labels"])},
        schema_version=1,
    )
    attempt = build_subject_mask_attempt(
        scientific_identity=science,
        run_path=run_path,
        attempt_id=str(uuid5(NAMESPACE_URL, f"pytest:{stage_kind}:{run_path}")),
    )
    arrays = {path: run[path] for path in paths}
    receipt = build_subject_mask_worker_semantic_receipt(
        stage_kind=stage_kind,
        run_path=str(run.path).strip("/"),
        scientific_identity=science,
        attempt=attempt,
        scope={"recording": "recording_001"},
        row_count=row_count,
        array_document=subject_mask_array_unit_document(arrays, paths, unit_rows=2),
        required_paths=paths,
        roi_aligned_paths=tuple(path for path in paths if path != "available_channels"),
    )
    receipt_bytes = canonical_json_bytes(receipt)
    relative = f"{str(run.path).strip('/')}/worker_semantic_receipt.json"
    receipt_path = draft_path / relative
    receipt_path.write_bytes(receipt_bytes)
    run.attrs.update(
        {
            "subject_mask_scientific_identity": science,
            "subject_mask_attempt": attempt,
            "subject_mask_worker_semantic_receipt_binding": {
                "schema_id": receipt["schema_id"],
                "schema_version": receipt["schema_version"],
                "payload_digest": receipt["payload_digest"],
                "relative_path": relative,
                "document_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "storage": "strict_json_sidecar_v1",
            },
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
        }
    )


def _draft(
    tmp_path: Path,
    *,
    raw_parent: str,
    raw_slices: dict[str, slice] | None = None,
    refined_slices: dict[str, slice] | None = None,
    split_eye_registry: bool = False,
) -> Path:
    path = tmp_path / "draft.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    crop = root.require_group("crop_runs").create_group("crop_001")
    for name, values in {
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_acquisition_frame_index": frames,
        "source_crop_xywh": np.asarray(
            [[0, 0, 8, 8], [1, 0, 8, 8], [0, 1, 8, 8], [1, 1, 8, 8]],
            dtype=np.float32,
        ),
        "frame_row_offsets": derive_subject_mask_frame_row_offsets(frames, n_frames=4),
    }.items():
        _create_array(crop, name, values)
    raw_values, refined_values = _surfaces()
    if split_eye_registry:
        refined_masks = refined_values["masks_roi"]
        raw_masks = np.stack(
            (
                refined_masks[:, 0],
                np.maximum(refined_masks[:, 1], refined_masks[:, 2]),
                refined_masks[:, 3],
            ),
            axis=1,
        )
        raw_metrics = derive_subject_mask_metrics(raw_masks)
        raw_probabilities = raw_masks * np.uint8(255)
        raw_values = {
            "mask_probs_roi": raw_probabilities,
            "available_channels": np.ones((3,), dtype=bool),
            "metrics/prob_max": (
                np.max(raw_probabilities, axis=(2, 3)).astype(np.float32)
                / np.float32(255.0)
            ),
            **{f"metrics/{name}": value for name, value in raw_metrics.items()},
        }
    row_values = {
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_acquisition_frame_index": frames,
    }
    raw_parent_group = root.require_group(raw_parent)
    refined_parent = root.require_group("refined_subject_masks_runs")
    refined_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    raw_labels = (
        ["subject_body", "eyes_union", "swim_bladder"]
        if split_eye_registry
        else refined_labels
    )
    for raw_name, row_slice in (
        raw_slices or {"raw_draft": slice(0, len(frames))}
    ).items():
        raw = raw_parent_group.create_group(raw_name)
        raw.attrs.update({"mask_labels": raw_labels, "mask_probability_threshold": 0.5})
        for name, values in row_values.items():
            _create_array(raw, name, values[row_slice])
        for name, values in raw_values.items():
            _create_array(
                raw,
                name,
                values if name == "available_channels" else values[row_slice],
            )
        _seal_worker(
            path,
            raw,
            stage_kind="raw_subject_mask",
            paths=RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
        )
    for refined_name, row_slice in (
        refined_slices or {"refined_draft": slice(0, len(frames))}
    ).items():
        refined = refined_parent.create_group(refined_name)
        refined.attrs["mask_labels"] = refined_labels
        for name, values in {
            "source_crop_row_ids": row_values["source_crop_row_ids"],
            **refined_values,
        }.items():
            _create_array(
                refined,
                name,
                values if name == "available_channels" else values[row_slice],
            )
        _seal_worker(
            path,
            refined,
            stage_kind="refined_subject_mask",
            paths=REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
        )
    return path


def _stamp_signed_hybrid_crop(root: zarr.Group) -> dict[str, object]:
    crop = root["crop_runs/crop_001"]
    source_crop_xywh = np.asarray(crop["source_crop_xywh"][:], dtype=np.float64)
    del crop["source_crop_xywh"]
    crop.create_array("source_crop_xywh", data=source_crop_xywh)
    crop.create_array(
        "roi_coordinates_full",
        data=np.asarray(source_crop_xywh[:, :2], dtype=np.int32),
    )
    crop.create_array(
        "roi_sizes_full",
        data=np.asarray(source_crop_xywh[:, 2:], dtype=np.int32),
    )
    frame_shape_authority = {
        "height": 1024,
        "width": 1280,
        "source": "test_fixture",
    }
    provider_record = {
        "schema_id": "palette.roi_pixel_provider_record.v1",
        "schema_version": 1,
        "crop_run": "crop_001",
        "frame_shape": [1024, 1280],
        "frame_shape_authority": frame_shape_authority,
    }
    crop.attrs.update(
        {
            "schema_id": "palette.hybrid_acquisition_offline_crop_run.v3",
            "source_pixels": "hybrid_acquisition_crop_video_offline_supplement",
            "provider_record": provider_record,
            "provider_record_sha256": canonical_json_sha256(provider_record),
            "source_full_frame_shape_authority": frame_shape_authority,
        }
    )
    _placement, normalization = normalize_subject_mask_crop_placement(
        crop,
        crop_run="crop_001",
        target_rows=np.arange(source_crop_xywh.shape[0], dtype=np.int64),
        values=source_crop_xywh,
    )
    assert normalization is not None
    return dict(normalization)


def test_recording_publisher_normalizes_signed_hybrid_crop_placement(
    tmp_path: Path,
) -> None:
    draft = _draft(tmp_path, raw_parent="subject_mask_shard_runs")
    root = zarr.open_group(str(draft), mode="a", use_consolidated=False)
    _stamp_signed_hybrid_crop(root)
    crop = root["crop_runs/crop_001"]
    source_crop_xywh = np.asarray(crop["source_crop_xywh"][:], dtype=np.float64)

    arrays = _refined_arrays(
        root["refined_subject_masks_runs/refined_draft"],
        crop,
        n_frames=4,
    )

    placement = np.asarray(arrays["source_crop_xywh"])
    assert placement.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(placement, source_crop_xywh)


def test_recording_bundle_publishes_normalized_signed_hybrid_placement(
    tmp_path: Path,
) -> None:
    draft = _draft(tmp_path, raw_parent="subject_mask_shard_runs")
    draft_root = zarr.open_group(str(draft), mode="a", use_consolidated=False)
    normalization = _stamp_signed_hybrid_crop(draft_root)
    draft_root["refined_subject_masks_runs/refined_draft"].attrs[
        "source_crop_xywh_normalization"
    ] = normalization
    analysis = tmp_path / "analysis_hybrid_placement.zarr"
    analysis_root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    analysis_root.attrs["recording_id"] = "recording_001"

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run="raw_draft",
        refined_draft_run="refined_draft",
        raw_run="raw_hybrid",
        refined_run="refined_hybrid",
        quality_run="quality_hybrid",
        bundle_id="bundle_hybrid",
        local_output_root=tmp_path / "hybrid_local_outputs",
        quality_scratch_root=tmp_path / "hybrid_quality_scratch",
        coordinate_contract_policy="legacy_allow_missing",
    )

    assert result["status"] == "complete"
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    assert published["subject_mask_runs/raw_hybrid/source_crop_xywh"].dtype == (
        np.dtype(np.float32)
    )
    refined = published["refined_subject_masks_runs/refined_hybrid"]
    assert refined["source_crop_xywh"].dtype == np.dtype(np.float32)
    assert refined.attrs["source_crop_xywh_normalization"] == normalization


def _install_worker_sampled_contours(
    draft_path: Path,
    *,
    refined_runs: tuple[str, ...],
) -> tuple[Path, ...]:
    root = zarr.open_group(str(draft_path), mode="a", use_consolidated=False)
    receipt_root = draft_path.parent / "sampled_contour_receipts"
    receipt_root.mkdir()
    receipt_paths: list[Path] = []
    for refined_name in refined_runs:
        run = root[f"refined_subject_masks_runs/{refined_name}"]
        components = SubjectMaskComponentRegistry(tuple(run.attrs["mask_labels"]))
        rows = int(run["masks_roi"].shape[0])
        for component_index, component in enumerate(components.labels):
            sample_count = {
                "subject_body": 128,
                "eye_left": 64,
                "eye_right": 64,
                "swim_bladder": 32,
            }[component]
            points = np.empty((rows, sample_count, 2), dtype=np.float32)
            points[..., 0] = np.arange(sample_count, dtype=np.float32)
            points[..., 1] = np.float32(component_index)
            write_sampled_component_contour_arrays(
                run.require_group("components").require_group(component),
                points_xy=points,
                valid=np.ones((rows,), dtype=bool),
                source_point_count=np.full((rows,), sample_count, dtype=np.int32),
                component=component,
                source_mask_run=refined_name,
                row_chunk=2,
            )
        run.attrs.update(
            {
                "sampled_component_contours_status": "computed",
                "derived_mask_caches_stale": False,
                "contours_stale": False,
            }
        )
        binding = run.attrs["subject_mask_worker_semantic_receipt_binding"]
        worker_receipt = json.loads(
            (draft_path / binding["relative_path"]).read_text(encoding="utf-8")
        )
        rows_global = np.asarray(run["source_crop_row_ids"][:], dtype=np.int64)
        destination = receipt_root / f"{refined_name}.json"
        write_subject_mask_sampled_contour_worker_receipt(
            run,
            destination=destination,
            global_start_row=int(rows_global[0]),
            worker_receipt=worker_receipt,
            producer_commit="a" * 40,
            unit_rows=2,
        )
        receipt_paths.append(destination)
    return tuple(receipt_paths)


def _install_worker_quality_partitions(
    draft_path: Path,
    *,
    refined_runs: tuple[str, ...],
) -> tuple[Path, ...]:
    root = zarr.open_group(str(draft_path), mode="r", use_consolidated=False)
    partition_root = draft_path.parent / "quality_partitions"
    partition_root.mkdir()
    paths: list[Path] = []
    crop = root["crop_runs/crop_001"]
    for index, refined_name in enumerate(refined_runs):
        run = root[f"refined_subject_masks_runs/{refined_name}"]
        binding = run.attrs["subject_mask_worker_semantic_receipt_binding"]
        receipt = json.loads(
            (draft_path / binding["relative_path"]).read_text(encoding="utf-8")
        )
        crop_rows = np.asarray(run["source_crop_row_ids"][:], dtype=np.int64)
        frames = np.asarray(
            crop["source_acquisition_frame_index"][crop_rows],
            dtype=np.int64,
        )

        class _QualityRun(dict[str, object]):
            pass

        quality_run = _QualityRun(
            masks_roi=run["masks_roi"],
            available_channels=run["available_channels"],
            instance_key=crop["instance_key"][crop_rows],
        )
        quality_run.path = run.path  # type: ignore[attr-defined]
        quality_run.attrs = run.attrs  # type: ignore[attr-defined]
        destination = partition_root / refined_name
        compute_subject_mask_quality_partition(
            quality_run,
            source_acquisition_frame_index=frames,
            global_start_row=int(crop_rows[0]),
            global_frame_start=int(frames.min()),
            global_frame_stop=int(frames.max()) + 1,
            work_unit_id=f"pytest_collection:clip_{int(frames.min())}",
            work_unit_index=index,
            source_worker_receipt=receipt,
            producer_commit="b" * 40,
            destination=destination,
            compute_workers=1,
            source_compute_block_bytes=512,
            receipt_unit_rows=2,
        )
        paths.append(destination)
    return tuple(paths)


def _install_composable_final_layout_packages(
    draft_path: Path,
    *,
    raw_runs: tuple[str, ...],
    refined_runs: tuple[str, ...],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    root = zarr.open_group(str(draft_path), mode="r", use_consolidated=False)
    package_root = draft_path.parent / "final_layout_packages"
    packages_by_kind: list[tuple[Path, ...]] = []
    for kind, family, names, payload_path in (
        (
            "raw_probability_uint8",
            "subject_mask_shard_runs",
            raw_runs,
            "mask_probs_roi",
        ),
        (
            "refined_dense_core",
            "refined_subject_masks_runs",
            refined_runs,
            "masks_roi",
        ),
    ):
        channels = int(root[f"{family}/{names[0]}/{payload_path}"].shape[1])
        dimensions = SubjectMaskDimensions(
            n_frames=4,
            n_rois=4,
            n_channels=channels,
            roi_height=8,
            roi_width=8,
        )
        paths: list[Path] = []
        for name in names:
            run = root[f"{family}/{name}"]
            binding = run.attrs["subject_mask_worker_semantic_receipt_binding"]
            receipt = json.loads(
                (draft_path / binding["relative_path"]).read_text(encoding="utf-8")
            )
            crop_rows = np.asarray(run["source_crop_row_ids"][:], dtype=np.int64)
            destination = package_root / kind / name
            build_subject_mask_final_layout_unit_package(
                source_array=run[payload_path],
                source_crop_row_ids=run["source_crop_row_ids"],
                destination=destination,
                kind=kind,
                dimensions=dimensions,
                global_start_row=int(crop_rows[0]),
                source_run_path=f"{family}/{name}",
                worker_receipt_payload_digest=receipt["payload_digest"],
                producer_commit="a" * 40,
                worker_array_validation_record=receipt["payload"]["arrays"][
                    payload_path
                ],
            )
            paths.append(destination)
        packages_by_kind.append(tuple(paths))
    return packages_by_kind[0], packages_by_kind[1]


def _install_crop_v2(
    draft_path: Path,
    *,
    signed_hybrid_provider_run: str | None = None,
) -> None:
    root = zarr.open_group(str(draft_path), mode="a", use_consolidated=False)
    crop = root.require_group("crop_runs").require_group("crop_001")
    arrays = _crop_manifest_arrays()
    for path, values in arrays.items():
        _create_array(crop, path, values)
    plan, direct, consolidated = _crop_manifest_metadata()
    policy = _crop_manifest_policy()
    if signed_hybrid_provider_run is not None:
        policy = CropGeometryPolicy(
            purpose="subject_analysis",
            size_mode=CropSizeMode.FIXED_PER_RUN,
            fixed_size_wh=(8, 8),
            padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
            placement_mode=CropPlacementMode.VERIFIED_EXPLICIT_PER_ROW,
            placement_authority={
                "schema_id": "palette.crop_geometry.explicit_origin_authority",
                "schema_version": 1,
                "authority_kind": "signed_hybrid_crop_provider",
                "run_id": signed_hybrid_provider_run,
                "provider_record_sha256": "1" * 64,
                "source_rowset_fingerprint": "2" * 64,
                "source_pixel_fingerprint": "3" * 64,
                "source_row_signature_spec_digest": "4" * 64,
            },
        )
        arrays["source_row_signature"] = build_crop_row_source_signatures(
            arrays,
            source=_crop_manifest_source(),
            policy=policy,
            pixel_authority=_crop_manifest_pixel(),
        ).signatures
    crop.attrs["run_manifest"] = build_coordinate_crop_run_manifest(
        run_id="crop_001",
        dimensions=_crop_manifest_dimensions(),
        policy=policy,
        storage_plan=plan,
        arrays=arrays,
        source=_crop_manifest_source(),
        pixel_authority=_crop_manifest_pixel(),
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=False,
    )


def _install_source_camera_authorities(
    root: zarr.Group,
    *,
    archive_path: Path,
) -> None:
    recording = archive_path.parent / "recording"
    recording.mkdir(exist_ok=True)
    video = recording / "cam2010095.mp4"
    video.write_bytes(b"subject-shape-bundle-test-video")
    source = {
        "source_path": str(video),
        "camera_id": "cam2010095",
        "width": 100,
        "height": 80,
        "total_frames": 4,
        "fps": 10.0,
        "codec": "h264",
        "pix_fmt": "yuv420p",
    }
    fingerprint = source_stat_fingerprint_attrs(
        video,
        attr_prefix="source_video",
        extra={
            "codec": source["codec"],
            "pix_fmt": source["pix_fmt"],
            "width": source["width"],
            "height": source["height"],
            "fps": source["fps"],
            "frame_count": source["total_frames"],
        },
    )
    root.attrs.update(
        {
            "camera_id": "cam2010095",
            "recording_path": str(recording),
            "source_video_path": str(video),
            "source_path": str(video),
            "source_video_metadata": build_source_video_metadata_v2(
                source,
                recording_path=recording,
                fingerprint_attrs=fingerprint,
            ),
        }
    )
    publish_external_video_acquisition_authority(root)


def _array_reference(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def _collection_partition_contract(start: int, stop: int) -> dict[str, object]:
    clip_id = f"clip_{start}"
    payload = {
        "role": "complete_collection_partition",
        "coverage_semantics": (
            "exact_complete_crop_rows_for_acquisition_frame_window_v1"
        ),
        "work_package_id": hashlib.sha256(clip_id.encode("utf-8")).hexdigest(),
        "collection": {
            "source_collection_id": "pytest_collection",
            "source_collection_path": "/pytest/collection.json",
            "source_clip_id": clip_id,
            "source_clip_index": start // 2,
            "source_work_unit_id": f"pytest_collection:{clip_id}",
            "source_shard_id": clip_id,
        },
        "frame_window": {
            "schema_id": "palette.acquisition_video_frame_window",
            "schema_version": 1,
            "recording_identity": "crop_manifest_test",
            "camera_identity": "cam2010095",
            "clip_id": clip_id,
            "actual_start_frame": start,
            "end_frame_exclusive": stop,
            "frame_count": stop - start,
            "clip_index_document_sha256": "e" * 64,
            "clip_video_sha256": "f" * 64,
        },
        "crop_rows": {
            "start": start,
            "stop": stop,
            "count": stop - start,
            "source_crop_total_rows": 4,
        },
        "validation": {
            "work_package_opened_and_content_verified": True,
            "row_interval_contiguous": True,
            "frame_offset_coverage_exact": True,
            "acquisition_frames_within_window": True,
        },
    }
    return {
        "schema_id": "palette.subject_mask.complete_collection_partition",
        "schema_version": 1,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def _upgrade_workers_to_coordinate_science_v2(
    draft_path: Path,
    *,
    worker_crop_run: str = "crop_001",
    bind_crop_manifest: bool = True,
    signed_hybrid_signature: Mapping[str, object] | None = None,
) -> None:
    root = zarr.open_group(str(draft_path), mode="a", use_consolidated=False)
    crop = root["crop_runs/crop_001"]
    crop_manifest = crop.attrs["run_manifest"]
    raw_parent = root["subject_mask_shard_runs"]
    raw_by_interval: dict[tuple[int, int], zarr.Group] = {}
    for run_name in sorted(raw_parent.keys()):
        run = raw_parent[run_name]
        rows = np.asarray(run["source_crop_row_ids"][:], dtype=np.int64)
        start = int(rows[0])
        stop = int(rows[-1]) + 1
        raw_by_interval[(start, stop)] = run
        coordinates = np.asarray(
            crop["source_crop_xywh"][start:stop, :2], dtype=np.int32
        )
        labels = list(run.attrs["mask_labels"])
        science = build_subject_mask_scientific_identity(
            stage_kind="raw_subject_mask",
            model={
                "artifact_role": "subject_mask_checkpoint",
                "artifact_sha256": "a" * 64,
                "artifact_size_bytes": 1024,
                "registry_set_id": "pytest_models",
                "registry_run_id": "pytest_raw",
                "label_schema_id": "pytest_subject_masks",
            },
            crop={
                "run_id": worker_crop_run,
                "run_group_path": worker_crop_run,
                "run_manifest": (
                    {
                        "schema_id": crop_manifest["schema_id"],
                        "schema_version": crop_manifest["schema_version"],
                        "payload_digest": crop_manifest["payload_digest"],
                    }
                    if bind_crop_manifest
                    else None
                ),
                "storage_mode": "geometry_only",
                "roi_shape_hw": [8, 8],
                "roi_coordinates_full": _array_reference(coordinates),
                "source_collection_id": "pytest_collection",
                "source_clip_id": f"clip_{start}",
                "source_clip_index": start // 2,
                "source_work_unit_id": f"pytest_collection:clip_{start}",
                "source_shard_id": f"clip_{start}",
                "collection_partition_contract": _collection_partition_contract(
                    start, stop
                ),
            },
            pixels={
                "profile": "pytest_pixels",
                "decoded_shape": [stop - start, 8, 8],
                "decoded_dtype": "uint8",
                "decoded_order": "C",
                "decoded_pixels_sha256": "b" * 64,
                "declared_pixels_sha256": "b" * 64,
                "cache_key": f"pytest_cache_{start}",
                "pixel_materialization_id": f"pytest_pixels_{start}",
                "pixel_contract": {
                    "schema": "palette_roi_pixel_contract_v1",
                    **(
                        {
                            "source_pixels": (
                                "hybrid_acquisition_crop_video_offline_supplement"
                            )
                        }
                        if signed_hybrid_signature is not None
                        else {}
                    ),
                },
                "work_package_role": "complete_collection_partition",
            },
            row_identity={
                "row_count": stop - start,
                "arrays": {
                    name: _array_reference(np.asarray(run[name][:]))
                    for name in (
                        "source_crop_row_ids",
                        "instance_key",
                        "source_acquisition_frame_index",
                    )
                },
            },
            inference_contract={
                "segmenter": "unet",
                "label_schema_id": "pytest_subject_masks",
                "mask_labels": labels,
                "model_input_transform": {
                    "name": "identity",
                    "native_shape_hw": [8, 8],
                    "model_shape_hw": [8, 8],
                    "pad_top": 0,
                    "pad_bottom": 0,
                    "pad_left": 0,
                    "pad_right": 0,
                    "coordinate_mapping": (
                        "native_xy = model_xy - [pad_left, pad_top]"
                    ),
                },
                "probability_semantics": "sigmoid_multilabel_logits",
                "probability_dtype": "uint8",
                "probability_encoding": "linear_uint8_0_255",
                "mask_probability_threshold": 0.5,
                "overlap_policy": "independent_sigmoid",
            },
        )
        _seal_worker(
            draft_path,
            run,
            stage_kind="raw_subject_mask",
            paths=RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
            scientific_identity=science,
        )

    refined_parent = root["refined_subject_masks_runs"]
    for run_name in sorted(refined_parent.keys()):
        run = refined_parent[run_name]
        rows = np.asarray(run["source_crop_row_ids"][:], dtype=np.int64)
        start = int(rows[0])
        stop = int(rows[-1]) + 1
        raw = raw_by_interval[(start, stop)]
        raw_science = raw.attrs["subject_mask_scientific_identity"]
        raw_binding = raw.attrs["subject_mask_worker_semantic_receipt_binding"]
        input_binding = {
            "run_path": str(raw.path).strip("/"),
            "run_manifest": None,
            "scientific_identity_digest": raw_science["digest"],
            "worker_semantic_receipt_binding": dict(raw_binding),
        }
        labels = list(run.attrs["mask_labels"])
        method = "smart_finalize_subject_masks_v1"
        science = build_subject_mask_scientific_identity(
            stage_kind="refined_subject_mask",
            model={
                "role": "deterministic_refinement_policy",
                "method": method,
                "source_input_binding": input_binding,
            },
            crop={
                "run_id": worker_crop_run,
                "source_crop_snapshot": (
                    {"source_crop_signature": dict(signed_hybrid_signature)}
                    if signed_hybrid_signature is not None
                    else {}
                ),
                "roi_shape_hw": [8, 8],
            },
            pixels={
                "semantic_input": "raw_subject_mask_surface",
                "surface_kind": "probability",
                "surface_path": "mask_probs_roi",
                "probability_encoding": "linear_uint8_0_255",
                "source_input_binding": input_binding,
            },
            row_identity={
                "row_count": stop - start,
                "arrays": {
                    "source_crop_row_ids": _array_reference(rows),
                    "instance_key": _array_reference(
                        np.asarray(crop["instance_key"][start:stop])
                    ),
                    "source_acquisition_frame_index": _array_reference(
                        np.asarray(crop["source_acquisition_frame_index"][start:stop])
                    ),
                    "source_crop_xywh": _array_reference(
                        np.asarray(crop["source_crop_xywh"][start:stop])
                    ),
                    "available_channels": _array_reference(
                        np.asarray(run["available_channels"][:])
                    ),
                },
            },
            inference_contract={
                "method": method,
                "finalization_semantics": ("smart_probability_to_refined_candidate"),
                "output_component_order": labels,
                "component_sources_and_policies": {
                    label: {"source": "pytest"} for label in labels
                },
                "eye_assignment_contract": None,
                "authoritative_output": "dense_uint8_masks_roi",
                "derived_cache_policy": (
                    "bitpacked_rle_metrics_contours_non_authoritative"
                ),
            },
        )
        _seal_worker(
            draft_path,
            run,
            stage_kind="refined_subject_mask",
            paths=REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
            scientific_identity=science,
        )


@pytest.mark.parametrize("raw_parent", ("subject_mask_runs", "subject_mask_shard_runs"))
def test_recording_bundle_publication_is_proof_bound_and_inactive(
    tmp_path: Path,
    raw_parent: str,
) -> None:
    draft = _draft(tmp_path, raw_parent=raw_parent)
    analysis = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent=raw_parent,
        raw_draft_run="raw_draft",
        refined_draft_run="refined_draft",
        raw_run="raw_001",
        refined_run="refined_001",
        quality_run="quality_001",
        bundle_id="bundle_001",
        local_output_root=tmp_path / "local_outputs",
        quality_scratch_root=tmp_path / "quality_scratch",
        coordinate_contract_policy="legacy_allow_missing",
        core_physical_unit_workers=2,
    )

    assert result["status"] == "complete"
    assert result["publication_execution"]["core_physical_unit_workers_requested"] == 2
    assert result["publication_execution"]["parallel_write_policy"] == (
        "bounded_threaded_disjoint_whole_physical_row_bands_v1"
    )
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in published.attrs
    assert (
        published["subject_mask_bundle_runs/bundle_001"].attrs[
            "palette_run_completion_status"
        ]
        == "complete"
    )
    for path in (
        "subject_mask_runs/raw_001",
        "refined_subject_masks_runs/refined_001",
        "subject_mask_quality_runs/quality_001",
    ):
        assert published[path].attrs["stage_selector_eligible"] is False
    for path in (
        "subject_mask_runs/raw_001",
        "refined_subject_masks_runs/refined_001",
    ):
        assert published[path].attrs["physical_unit_workers_requested"] == 2
        assert published[path].attrs["parallel_write_policy"] == (
            "bounded_threaded_disjoint_whole_physical_row_bands_v1"
        )


def test_recording_bundle_requires_crop_v2_by_default(tmp_path: Path) -> None:
    draft = _draft(tmp_path, raw_parent="subject_mask_shard_runs")
    analysis = tmp_path / "analysis_requires_crop_v2.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"

    with pytest.raises(ValueError, match="requires a persisted crop-v2"):
        publish_recording_subject_mask_bundle(
            analysis_zarr=analysis,
            draft_zarr=draft,
            crop_run="crop_001",
            raw_draft_parent="subject_mask_shard_runs",
            raw_draft_run="raw_draft",
            refined_draft_run="refined_draft",
            raw_run="raw_rejected",
            refined_run="refined_rejected",
            quality_run="quality_rejected",
            bundle_id="bundle_rejected",
            local_output_root=tmp_path / "rejected_outputs",
            quality_scratch_root=tmp_path / "rejected_scratch",
        )


def test_recording_bundle_signed_hybrid_rebase_is_explicit_and_fail_closed(
    tmp_path: Path,
) -> None:
    provider_run = "crop_hybrid_provider_001"
    signature = {
        "schema_id": "palette.hybrid_crop_provider.signature",
        "schema_version": 1,
        "source_pixels": "hybrid_acquisition_crop_video_offline_supplement",
        "provider_record_sha256": "1" * 64,
        "source_rowset_fingerprint": "2" * 64,
        "source_pixel_fingerprint": "3" * 64,
        "source_row_signature_spec_digest": "4" * 64,
    }
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
        refined_slices={
            "refined_clip_a": slice(0, 2),
            "refined_clip_b": slice(2, 4),
        },
    )
    _install_crop_v2(draft, signed_hybrid_provider_run=provider_run)
    _upgrade_workers_to_coordinate_science_v2(
        draft,
        worker_crop_run=provider_run,
        bind_crop_manifest=False,
        signed_hybrid_signature=signature,
    )
    analysis = tmp_path / "analysis_signed_hybrid_rebase.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "crop_manifest_test"
    _install_source_camera_authorities(root, archive_path=analysis)
    _install_crop_v2(analysis, signed_hybrid_provider_run=provider_run)
    work_units = (
        {
            "work_unit_id": "pytest_collection:clip_0",
            "work_unit_index": 0,
            "source_clip_id": "clip_0",
            "source_clip_index": 0,
            "frame_start": 0,
            "frame_stop": 2,
            "row_start": 0,
            "row_stop": 2,
        },
        {
            "work_unit_id": "pytest_collection:clip_2",
            "work_unit_index": 1,
            "source_clip_id": "clip_2",
            "source_clip_index": 1,
            "frame_start": 2,
            "frame_stop": 4,
            "row_start": 2,
            "row_stop": 4,
        },
    )
    common = {
        "analysis_zarr": analysis,
        "draft_zarr": draft,
        "crop_run": "crop_001",
        "raw_draft_parent": "subject_mask_shard_runs",
        "raw_draft_run": "raw_clip_a",
        "raw_draft_runs": ("raw_clip_a", "raw_clip_b"),
        "refined_draft_run": "refined_clip_a",
        "refined_draft_runs": ("refined_clip_a", "refined_clip_b"),
        "raw_run": "raw_signed_hybrid_rebase",
        "refined_run": "refined_signed_hybrid_rebase",
        "quality_run": "quality_signed_hybrid_rebase",
        "bundle_id": "bundle_signed_hybrid_rebase",
        "expected_work_units": work_units,
    }

    with pytest.raises(ValueError, match="differs from the crop-v2 authority"):
        publish_recording_subject_mask_bundle(
            **common,
            local_output_root=tmp_path / "rebase_rejected_outputs",
            quality_scratch_root=tmp_path / "rebase_rejected_quality",
        )

    _install_crop_v2(draft, signed_hybrid_provider_run="different_provider")
    _install_crop_v2(analysis, signed_hybrid_provider_run="different_provider")
    with pytest.raises(ValueError, match="differs from the crop-v2 authority"):
        publish_recording_subject_mask_bundle(
            **common,
            local_output_root=tmp_path / "wrong_provider_rejected_outputs",
            quality_scratch_root=tmp_path / "wrong_provider_rejected_quality",
            allow_signed_hybrid_crop_rebase=True,
        )

    _install_crop_v2(draft, signed_hybrid_provider_run=provider_run)
    _install_crop_v2(analysis, signed_hybrid_provider_run=provider_run)
    result = publish_recording_subject_mask_bundle(
        **common,
        local_output_root=tmp_path / "rebase_accepted_outputs",
        quality_scratch_root=tmp_path / "rebase_accepted_quality",
        allow_signed_hybrid_crop_rebase=True,
    )

    assert result["status"] == "complete"
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    manifest = published["subject_mask_runs/raw_signed_hybrid_rebase"].attrs[
        "run_manifest"
    ]
    assert manifest["schema_version"] == 4
    assert (
        manifest["payload"]["coordinate_dependencies"]["document"]["crop"]["run_path"]
        == "crop_runs/crop_001"
    )


def test_recording_bundle_publishes_coordinate_bound_members_and_subject_shape_v5(
    tmp_path: Path,
) -> None:
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
        refined_slices={
            "refined_clip_a": slice(0, 2),
            "refined_clip_b": slice(2, 4),
        },
    )
    _install_crop_v2(draft)
    _upgrade_workers_to_coordinate_science_v2(draft)
    analysis = tmp_path / "analysis_coordinate_v4.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "crop_manifest_test"
    _install_source_camera_authorities(root, archive_path=analysis)
    _install_crop_v2(analysis)

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run="raw_clip_a",
        raw_draft_runs=("raw_clip_b", "raw_clip_a"),
        refined_draft_run="refined_clip_a",
        refined_draft_runs=("refined_clip_b", "refined_clip_a"),
        raw_run="raw_coordinate_v4",
        refined_run="refined_coordinate_v4",
        quality_run="quality_coordinate_v4",
        bundle_id="bundle_coordinate_v4",
        local_output_root=tmp_path / "coordinate_outputs",
        quality_scratch_root=tmp_path / "coordinate_quality_scratch",
        expected_work_units=(
            {
                "work_unit_id": "pytest_collection:clip_0",
                "work_unit_index": 0,
                "source_clip_id": "clip_0",
                "source_clip_index": 0,
                "frame_start": 0,
                "frame_stop": 2,
                "row_start": 0,
                "row_stop": 2,
            },
            {
                "work_unit_id": "pytest_collection:clip_2",
                "work_unit_index": 1,
                "source_clip_id": "clip_2",
                "source_clip_index": 1,
                "frame_start": 2,
                "frame_stop": 4,
                "row_start": 2,
                "row_stop": 4,
            },
        ),
    )

    assert result["status"] == "complete"
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    raw_manifest = published["subject_mask_runs/raw_coordinate_v4"].attrs[
        "run_manifest"
    ]
    refined_manifest = published[
        "refined_subject_masks_runs/refined_coordinate_v4"
    ].attrs["run_manifest"]
    assert raw_manifest["schema_version"] == 4
    assert refined_manifest["schema_version"] == 4
    assert (
        refined_manifest["payload"]["coordinate_dependencies"]["document"]["raw_core"][
            "manifest_payload_digest"
        ]
        == raw_manifest["payload_digest"]
    )
    bundle_manifest = published["subject_mask_bundle_runs/bundle_coordinate_v4"].attrs[
        "run_manifest"
    ]
    coordinate_binding = bundle_manifest["payload"]["cross_binding"][
        "coordinate_contract"
    ]
    assert coordinate_binding["crop"]["run_path"] == "crop_runs/crop_001"
    assert (
        coordinate_binding["refined_raw_core_binding"]["manifest_payload_digest"]
        == raw_manifest["payload_digest"]
    )
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in published.attrs

    with pytest.raises(
        SubjectMaskBundleCoordinateAuthorityError,
        match="allow_inactive",
    ):
        load_recording_subject_mask_coordinate_authority(
            analysis,
            bundle_id="bundle_coordinate_v4",
        )
    inactive = load_recording_subject_mask_coordinate_authority(
        analysis,
        bundle_id="bundle_coordinate_v4",
        allow_inactive=True,
    )
    assert inactive.active is False
    assert inactive.crop_run_path == "crop_runs/crop_001"
    assert (
        inactive.refined_run.path == "refined_subject_masks_runs/refined_coordinate_v4"
    )
    assert inactive.camera_identity == "cam2010095"
    assert inactive.source_total_frames == 4
    assert (inactive.source_width, inactive.source_height) == (100, 80)
    assert (inactive.n_rois, inactive.roi_height, inactive.roi_width) == (4, 8, 8)
    assert inactive.assignment_keypoint_collection["mode"] == "not_used"
    np.testing.assert_array_equal(
        inactive.require_translation_only_offsets(),
        np.asarray(inactive.source_crop_xywh_node[:, :2], dtype=np.float64),
    )
    with pytest.raises(SubjectShapeBundleSourceError, match="cannot be constructed"):
        BoundSubjectShapeBundleSource()
    shape_source = load_subject_shape_bundle_source(
        analysis,
        bundle_id="bundle_coordinate_v4",
        allow_inactive=True,
    )
    assert shape_source.active is False
    assert shape_source.source_record["component_labels"] == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    offsets = shape_source.translation_offsets()
    np.testing.assert_array_equal(
        shape_source.transform_roi_points(np.zeros((4, 2), dtype=np.float32)),
        offsets,
    )
    np.testing.assert_array_equal(
        shape_source.transform_roi_boxes(np.zeros((4, 4), dtype=np.float32)),
        np.concatenate((offsets, offsets), axis=1),
    )
    with pytest.raises(
        SubjectMaskBundleCoordinateAuthorityError,
        match="allow_inactive=True",
    ):
        build_subject_shape_materialization_plan(
            analysis,
            scratch_root=tmp_path / "inactive_bundle_rejected",
            refined_run=None,
            subject_mask_bundle_id="bundle_coordinate_v4",
            run_name="must_not_plan",
            storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        )
    with pytest.raises(ValueError, match="differs from the selected"):
        build_subject_shape_materialization_plan(
            analysis,
            scratch_root=tmp_path / "conflicting_refined_rejected",
            refined_run="another_refined_run",
            subject_mask_bundle_id="bundle_coordinate_v4",
            allow_inactive_subject_mask_bundle=True,
            run_name="must_not_plan_either",
            storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        )
    with pytest.raises(ValueError, match="access-aware materializer"):
        write_subject_shape_run_group(
            published,
            zarr_path=analysis,
            subject_mask_bundle_id="bundle_coordinate_v4",
            allow_inactive_subject_mask_bundle=True,
            run_name="direct_bundle_v5_forbidden",
        )

    mutable = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    parent = mutable.get("analysis/subject_shape_runs")
    selectors_before = {
        name: (
            parent is not None and name in parent.attrs,
            parent.attrs.get(name) if parent is not None else None,
        )
        for name in ("latest", "latest_complete")
    }
    shape_result = materialize_subject_shape(
        analysis,
        scratch_root=tmp_path / "shape_bundle_scratch",
        refined_run=None,
        subject_mask_bundle_id="bundle_coordinate_v4",
        allow_inactive_subject_mask_bundle=True,
        run_name="shape_bundle_v5",
        storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        components=("subject_body", "swim_bladder", "eye_left", "eye_right"),
        block_rows=2,
        output_shard_rows=4,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_copy_workers=1,
        native_threads=1,
        copy_backend="python",
        apply=True,
        check_capacity=False,
    )
    assert shape_result["status"] == "complete"
    mutable = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    shape_parent = mutable["analysis/subject_shape_runs"]
    assert {
        name: (name in shape_parent.attrs, shape_parent.attrs.get(name))
        for name in ("latest", "latest_complete")
    } == selectors_before
    shape_run = mutable["analysis/subject_shape_runs/shape_bundle_v5"]
    assert shape_run.attrs["schema_version"] == (
        CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION
    )
    assert shape_run.attrs["method"] == CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD
    assert shape_run.attrs[SUBJECT_SHAPE_SOURCE_KIND_ATTR] == (
        SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
    )
    assert shape_run.attrs["stage_selector_eligible"] is False
    assert (
        shape_run.attrs[SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR]["logical_profile_id"]
        == CANONICAL_SUBJECT_SHAPE_BUNDLE_PROFILE_ID
    )
    owner = shape_run.attrs["subject_shape_publication_owner_uuid"]
    publication = load_completed_ineligible_subject_shape_coordinate_publication(
        mutable,
        "analysis/subject_shape_runs/shape_bundle_v5",
        expected_publication_owner=owner,
    )
    assert publication.source_binding is not None
    assert publication.source_binding.record_sha256 == shape_source.source_digest
    assert set(publication.source_binding.record["source_camera_authorities"]) == {
        "acquisition_frame",
        "continuous_pixel_frame",
        "pixel_edge_half_open_frame",
    }
    assert publication.manifest.record["schema_version"] == 2
    assert publication.derivation.record["schema_version"] == 2
    with pytest.raises(SubjectShapeCoordinatePublicationError, match="unpromoted"):
        activate_subject_shape_coordinate_publication(
            mutable,
            shape_parent,
            publication,
            run_name="shape_bundle_v5",
            owner=owner,
            snapshot=selector_snapshot(shape_parent),
        )

    activate_subject_mask_bundle(
        analysis_zarr=analysis,
        bundle_id="bundle_coordinate_v4",
    )
    active = load_recording_subject_mask_coordinate_authority(analysis)
    assert active.active is True
    assert active.authority_digest == inactive.authority_digest
    active_shape_source = load_subject_shape_bundle_source(analysis)
    assert active_shape_source.active is True
    assert active_shape_source.source_digest == shape_source.source_digest
    shape_source.assert_verified()
    reloaded = zarr.open_group(str(analysis), mode="a", use_consolidated=False)
    load_completed_ineligible_subject_shape_coordinate_publication(
        reloaded,
        "analysis/subject_shape_runs/shape_bundle_v5",
        expected_publication_owner=owner,
    )
    source_binding = reloaded[
        "analysis/subject_shape_runs/shape_bundle_v5/coordinate_records/source_binding"
    ]
    source_binding.attrs[SUBJECT_SHAPE_SOURCE_BINDING_ATTR] = {"tampered": True}
    with pytest.raises(SubjectShapeCoordinatePublicationError):
        load_completed_ineligible_subject_shape_coordinate_publication(
            reloaded,
            "analysis/subject_shape_runs/shape_bundle_v5",
            expected_publication_owner=owner,
        )


def test_recording_bundle_composes_multiple_raw_clip_shards_without_reordering(
    tmp_path: Path,
) -> None:
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
    )
    analysis = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run="raw_clip_b",
        # Deliberately reverse command order: canonical crop-row intervals,
        # rather than caller order, define the recording assembly.
        raw_draft_runs=("raw_clip_b", "raw_clip_a"),
        refined_draft_run="refined_draft",
        raw_run="raw_multi_001",
        refined_run="refined_multi_001",
        quality_run="quality_multi_001",
        bundle_id="bundle_multi_001",
        local_output_root=tmp_path / "local_multi_outputs",
        quality_scratch_root=tmp_path / "quality_multi_scratch",
        coordinate_contract_policy="legacy_allow_missing",
    )

    assert result["status"] == "complete"
    assert result["n_rois"] == 4
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    raw = published["subject_mask_runs/raw_multi_001"]
    np.testing.assert_array_equal(
        raw["source_crop_row_ids"][:], np.arange(4, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        raw["instance_key"][:], np.asarray([101, 102, 201, 301], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        raw["mask_probs_roi"][:, 0, 1, 1], np.full(4, 255, dtype=np.uint8)
    )
    assert raw.attrs["stage_selector_eligible"] is False
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in published.attrs


def test_recording_bundle_composes_multiple_refined_clip_shards_without_reordering(
    tmp_path: Path,
) -> None:
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
        refined_slices={
            "refined_clip_a": slice(0, 2),
            "refined_clip_b": slice(2, 4),
        },
    )
    analysis = tmp_path / "analysis_refined_shards.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run="raw_clip_b",
        raw_draft_runs=("raw_clip_b", "raw_clip_a"),
        refined_draft_run="refined_clip_b",
        refined_draft_runs=("refined_clip_b", "refined_clip_a"),
        raw_run="raw_multi_refined_001",
        refined_run="refined_multi_refined_001",
        quality_run="quality_multi_refined_001",
        bundle_id="bundle_multi_refined_001",
        local_output_root=tmp_path / "local_multi_refined_outputs",
        quality_scratch_root=tmp_path / "quality_multi_refined_scratch",
        coordinate_contract_policy="legacy_allow_missing",
    )

    assert result["status"] == "complete"
    assert result["n_rois"] == 4
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    refined = published["refined_subject_masks_runs/refined_multi_refined_001"]
    np.testing.assert_array_equal(
        refined["source_crop_row_ids"][:], np.arange(4, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        refined["instance_key"][:],
        np.asarray([101, 102, 201, 301], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        refined["masks_roi"][:, 0, 1, 1], np.ones(4, dtype=np.uint8)
    )
    assert refined.attrs["stage_selector_eligible"] is False
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in published.attrs


def test_recording_bundle_assembles_standard_sampled_contour_member(
    tmp_path: Path,
) -> None:
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
        refined_slices={
            "refined_clip_a": slice(0, 2),
            "refined_clip_b": slice(2, 4),
        },
    )
    receipts = _install_worker_sampled_contours(
        draft,
        refined_runs=("refined_clip_a", "refined_clip_b"),
    )
    analysis = tmp_path / "analysis_sampled_contours.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run="raw_clip_b",
        raw_draft_runs=("raw_clip_b", "raw_clip_a"),
        refined_draft_run="refined_clip_b",
        refined_draft_runs=("refined_clip_b", "refined_clip_a"),
        raw_run="raw_sampled_001",
        refined_run="refined_sampled_001",
        quality_run="quality_sampled_001",
        cache_run="sampled_contours_001",
        bundle_id="bundle_sampled_001",
        local_output_root=tmp_path / "local_sampled_outputs",
        quality_scratch_root=tmp_path / "quality_sampled_scratch",
        coordinate_contract_policy="legacy_allow_missing",
        sampled_contour_worker_receipts=tuple(reversed(receipts)),
        require_worker_sampled_contours=True,
        sampled_contour_producer_commit="a" * 40,
    )

    assert result["publication_execution"]["sampled_contour_source_mode"] == (
        "receipt_bound_worker_arrays"
    )
    assert result["cache_manifest_digest"] is not None
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    cache = published["subject_mask_cache_runs/sampled_contours_001"]
    manifest = cache.attrs["run_manifest"]
    assert manifest["schema_version"] == 2
    assert manifest["payload"]["write_receipt"]["worker_assembly"]["worker_count"] == 2
    assert cache["components/subject_body/sampled_contours/points_xy"].shape == (
        4,
        128,
        2,
    )
    bundle = published["subject_mask_bundle_runs/bundle_sampled_001"].attrs[
        "run_manifest"
    ]
    assert set(bundle["payload"]["members"]) == {
        "raw",
        "refined",
        "quality",
        "presentation_cache",
    }


def test_recording_bundle_composes_v5_dense_identity_through_coordinate_bundle(
    tmp_path: Path,
) -> None:
    raw_runs = ("raw_clip_a", "raw_clip_b")
    refined_runs = ("refined_clip_a", "refined_clip_b")
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
        refined_slices={
            "refined_clip_a": slice(0, 2),
            "refined_clip_b": slice(2, 4),
        },
    )
    _install_crop_v2(draft)
    _upgrade_workers_to_coordinate_science_v2(draft)
    contour_receipts = _install_worker_sampled_contours(
        draft, refined_runs=refined_runs
    )
    quality_partitions = _install_worker_quality_partitions(
        draft, refined_runs=refined_runs
    )
    raw_packages, refined_packages = _install_composable_final_layout_packages(
        draft,
        raw_runs=raw_runs,
        refined_runs=refined_runs,
    )
    analysis = tmp_path / "analysis_composable.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "crop_manifest_test"
    _install_source_camera_authorities(root, archive_path=analysis)
    _install_crop_v2(analysis)

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run=raw_runs[0],
        raw_draft_runs=raw_runs,
        refined_draft_run=refined_runs[0],
        refined_draft_runs=refined_runs,
        raw_run="raw_composable_001",
        refined_run="refined_composable_001",
        quality_run="quality_composable_001",
        cache_run="cache_composable_001",
        bundle_id="bundle_composable_001",
        local_output_root=tmp_path / "local_composable_outputs",
        quality_scratch_root=tmp_path / "quality_composable_scratch",
        expected_work_units=(
            {
                "work_unit_id": "pytest_collection:clip_0",
                "work_unit_index": 0,
                "source_clip_id": "clip_0",
                "source_clip_index": 0,
                "frame_start": 0,
                "frame_stop": 2,
                "row_start": 0,
                "row_stop": 2,
            },
            {
                "work_unit_id": "pytest_collection:clip_2",
                "work_unit_index": 1,
                "source_clip_id": "clip_2",
                "source_clip_index": 1,
                "frame_start": 2,
                "frame_stop": 4,
                "row_start": 2,
                "row_stop": 4,
            },
        ),
        core_validation_mode=SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE,
        raw_final_layout_unit_packages=raw_packages,
        refined_final_layout_unit_packages=refined_packages,
        require_complete_final_layout_units=True,
        sampled_contour_worker_receipts=contour_receipts,
        require_worker_sampled_contours=True,
        sampled_contour_producer_commit="a" * 40,
        quality_partition_roots=quality_partitions,
        require_worker_quality=True,
        quality_partition_producer_commit="b" * 40,
    )

    assert result["publication_execution"]["core_validation_mode"] == (
        SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE.value
    )
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    raw_manifest = published["subject_mask_runs/raw_composable_001"].attrs[
        "run_manifest"
    ]
    refined_manifest = published[
        "refined_subject_masks_runs/refined_composable_001"
    ].attrs["run_manifest"]
    assert raw_manifest["schema_version"] == (
        SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert refined_manifest["schema_version"] == (
        SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    refined_dense = refined_manifest["payload"]["logical_content"]["document"][
        "arrays"
    ]["masks_roi"]
    assert "sha256" not in refined_dense
    quality_manifest = published[
        "subject_mask_quality_runs/quality_composable_001"
    ].attrs["run_manifest"]
    quality_source = quality_manifest["payload"]["source_refined_subject_mask_snapshot"]
    cache_manifest = published["subject_mask_cache_runs/cache_composable_001"].attrs[
        "run_manifest"
    ]
    cache_source = cache_manifest["payload"]["source_refined_subject_mask_snapshot"]
    assert cache_source["dense_array_logical_identity_digest"] == (
        quality_source["dense_array_logical_identity_digest"]
    )
    assert quality_manifest["schema_version"] == 3
    assert quality_manifest["payload"]["write_receipt"]["schema_version"] == 4
    assert (
        quality_manifest["payload"]["write_receipt"]["source_compute_execution"]
        == "receipt_bound_partitions_with_verified_worker_units_v2"
    )
    assert (
        quality_manifest["payload"]["write_receipt"]["source_compute_block_count"] == 0
    )
    tampered_cache = copy.deepcopy(cache_manifest)
    tampered_cache["payload"]["source_refined_subject_mask_snapshot"][
        "worker_assembly_digest"
    ] = ("0" * 64)
    tampered_cache["payload_digest"] = canonical_json_sha256(tampered_cache["payload"])
    assert "subject-mask cache source/worker assembly digest differs" in (
        validate_subject_mask_cache_run_manifest(
            tampered_cache, source_manifest=refined_manifest
        )
    )
    bundle = published["subject_mask_bundle_runs/bundle_composable_001"].attrs[
        "run_manifest"
    ]
    assert bundle["payload"]["cross_binding"]["identity_policy"] == (
        "manifest_bound_composable_dense_identity_v2"
    )
    assert bundle["schema_version"] == 4
    assert (
        "ordered_source_verification"
        not in result["publication_execution"]["quality_phase_seconds"]
    )
    assert (
        "receipt_bound_source_verification"
        in result["publication_execution"]["quality_phase_seconds"]
    )
    authority = load_recording_subject_mask_coordinate_authority(
        analysis,
        bundle_id="bundle_composable_001",
        allow_inactive=True,
    )
    assert authority.active is False
    assert authority.crop_run_path == "crop_runs/crop_001"
    assert authority.refined_run.path == (
        "refined_subject_masks_runs/refined_composable_001"
    )


def test_recording_bundle_binds_distinct_raw_and_refined_component_registries(
    tmp_path: Path,
) -> None:
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
        refined_slices={
            "refined_clip_a": slice(0, 2),
            "refined_clip_b": slice(2, 4),
        },
        split_eye_registry=True,
    )
    analysis = tmp_path / "analysis_split_eye_registry.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_001"

    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run="raw_clip_a",
        raw_draft_runs=("raw_clip_a", "raw_clip_b"),
        refined_draft_run="refined_clip_a",
        refined_draft_runs=("refined_clip_a", "refined_clip_b"),
        raw_run="raw_split_eye_001",
        refined_run="refined_split_eye_001",
        quality_run="quality_split_eye_001",
        bundle_id="bundle_split_eye_001",
        local_output_root=tmp_path / "local_split_eye_outputs",
        quality_scratch_root=tmp_path / "quality_split_eye_scratch",
        coordinate_contract_policy="legacy_allow_missing",
    )

    assert result["status"] == "complete"
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    raw = published["subject_mask_runs/raw_split_eye_001"]
    refined = published["refined_subject_masks_runs/refined_split_eye_001"]
    quality = published["subject_mask_quality_runs/quality_split_eye_001"]
    assert raw.attrs["mask_labels"] == [
        "subject_body",
        "eyes_union",
        "swim_bladder",
    ]
    assert refined.attrs["mask_labels"] == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    assert raw["available_channels"].shape == (3,)
    assert refined["available_channels"].shape == (4,)
    quality_schema = quality.attrs["run_manifest"]["payload"]["logical_schema"]
    assert quality_schema["dimensions"]["n_channels"] == 4
    assert quality_schema["components"]["labels"] == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    bundle = published["subject_mask_bundle_runs/bundle_split_eye_001"]
    bundle_manifest = bundle.attrs["run_manifest"]
    assert bundle_manifest["schema_version"] == 2
    cross_binding = bundle_manifest["payload"]["cross_binding"]
    assert (
        cross_binding["component_registry_policy"]
        == "raw_and_refined_bound_independently_v1"
    )
    assert cross_binding["raw_components"]["labels"] == [
        "subject_body",
        "eyes_union",
        "swim_bladder",
    ]
    assert cross_binding["components"]["labels"] == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    assert (
        "available_channels"
        not in cross_binding["raw_refined_identity_array_values_sha256"]
    )
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in published.attrs


def test_two_clip_proof_import_flows_into_atomic_recording_bundle(
    tmp_path: Path,
) -> None:
    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_a": slice(0, 2), "raw_clip_b": slice(2, 4)},
    )
    mutable = zarr.open_group(str(draft), mode="a", use_consolidated=False)
    mutable.attrs["recording_frame_index_row_count"] = 4
    labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    package_a = _write_package(
        tmp_path,
        package_name="bundle_clip_a",
        run_name="refined_bundle_a",
        crop_row_ids=[0, 1],
        source_crop_run="crop_001",
        labels=labels,
        frame_indices=[0, 0],
        production_proof=True,
        roi_shape=(8, 8),
    )
    package_b = _write_package(
        tmp_path,
        package_name="bundle_clip_b",
        run_name="refined_bundle_b",
        crop_row_ids=[2, 3],
        source_crop_run="crop_001",
        labels=labels,
        frame_indices=[2, 3],
        production_proof=True,
        roi_shape=(8, 8),
    )
    imported = import_refined_subject_mask_clip_packages(
        zarr_path=draft,
        package_paths=(package_b, package_a),
        output_run="refined_recording_import",
        expected_target_crop_run="crop_001",
        require_production_proof=True,
        array_copy_workers=2,
    )
    assert imported["selector_eligible"] is False
    assert imported["source_validation_receipt_payload_digest"]

    analysis = tmp_path / "analysis_from_clips.zarr"
    analysis_root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    analysis_root.attrs["recording_id"] = "recording_001"
    result = publish_recording_subject_mask_bundle(
        analysis_zarr=analysis,
        draft_zarr=draft,
        crop_run="crop_001",
        raw_draft_parent="subject_mask_shard_runs",
        raw_draft_run="raw_clip_a",
        raw_draft_runs=("raw_clip_b", "raw_clip_a"),
        refined_draft_run="refined_recording_import",
        raw_run="raw_from_clips",
        refined_run="refined_from_clips",
        quality_run="quality_from_clips",
        bundle_id="bundle_from_clips",
        local_output_root=tmp_path / "local_clipped_outputs",
        quality_scratch_root=tmp_path / "quality_clipped_scratch",
        coordinate_contract_policy="legacy_allow_missing",
    )

    assert result["status"] == "complete"
    assert result["n_rois"] == 4
    published = zarr.open_group(str(analysis), mode="r", use_consolidated=False)
    assert SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR not in published.attrs
    for path in (
        "subject_mask_runs/raw_from_clips",
        "refined_subject_masks_runs/refined_from_clips",
        "subject_mask_quality_runs/quality_from_clips",
    ):
        assert published[path].attrs["stage_selector_eligible"] is False
    np.testing.assert_array_equal(
        published["refined_subject_masks_runs/refined_from_clips"][
            "source_crop_row_ids"
        ][:],
        np.arange(4, dtype=np.int64),
    )
