from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.shared.zarr.canonical_detection_manifest import (
    canonical_detection_logical_content_digest,
)
from fisheye.shared.zarr.clipped_refined_detection_finalization import (
    ClippedRefinedDetectionFinalizationError,
    prepare_clipped_refined_detection_snapshot,
    publish_selector_ineligible_clipped_refined_detection_snapshot,
    validate_clipped_refined_detection_finalization_receipt,
)
from fisheye.shared.zarr.crop_manifest import CropPixelAuthority
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.crop_shadow import (
    prepare_crop_geometry_from_refined_source,
    publish_selector_ineligible_crop_geometry_snapshot,
    validate_crop_geometry_shadow_publication,
)
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    CanonicalDetectionDimensions,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionBoundClipEvidence,
    RefinedDetectionSnapshotLineage,
    RefinedDetectionSourceIdentity,
    build_refined_detection_run_manifest,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_DECISION_CODE_MAP,
    SOURCE_KIND_CODE_MAP,
    RefinedDetectionClipBinding,
    RefinedDetectionClippedBinding,
    RefinedDetectionDimensions,
)
from fisheye.shared.zarr.refined_detection_storage import (
    plan_refined_detection_storage,
)
from tests.unit.fisheye.test_refined_detection_manifest import (
    _empty_arrays,
    _metadata_declarations,
)


RECORDING_IDENTITY = "clipped_recording_multi_subject"


def _clip(
    *,
    clip_index: int,
    parent_start: int,
    local_frame: int,
    refined_row_id: int,
    bbox: np.ndarray,
    class_id: int,
    instance_reason_label: str | None = None,
    source_reason_label: str | None = None,
) -> tuple[RefinedDetectionBoundClipEvidence, RefinedDetectionSourceIdentity]:
    dimensions = RefinedDetectionDimensions(
        n_frames=2,
        n_instances=1,
        n_source_detections=1,
        source_width=100,
        source_height=80,
    )
    arrays = _empty_arrays(dimensions)
    parent_frame = parent_start + local_frame
    frames = np.asarray([local_frame], dtype=np.int32)
    boxes = np.asarray(bbox, dtype=np.float32).reshape(1, 4)
    classes = np.asarray([class_id], dtype=np.int32)
    keys = mint_detection_instance_keys(
        recording_identity=RECORDING_IDENTITY,
        frame_indices=np.asarray([parent_frame], dtype=np.int32),
        bbox_norm_coords=boxes,
        class_ids=classes,
    )
    bbox_img, centers = derive_canonical_detection_geometry(
        boxes,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
    )
    scores = np.asarray([0.9 - clip_index * 0.1], dtype=np.float32)
    shared = {
        "frame_indices": frames,
        "source_acquisition_frame_index": frames.astype(np.int64),
        "instance_key": keys,
        "bbox_norm_coords": boxes,
        "bbox_img_xyxy": bbox_img,
        "centers_img_xy": centers,
        "scores": scores,
        "class_ids": classes,
    }
    arrays.update(
        {
            **{f"instances/{name}": value.copy() for name, value in shared.items()},
            "instances/refined_row_ids": np.asarray([refined_row_id], dtype=np.int64),
            "instances/score_valid": np.asarray([True], dtype=np.bool_),
            "instances/source_kind_codes": np.asarray(
                [SOURCE_KIND_CODE_MAP["raw_detect"]], dtype=np.uint8
            ),
            "instances/manual_edit_flags": np.asarray([False], dtype=np.bool_),
            "instances/source_detect_row_index": np.asarray([0], dtype=np.int64),
            "instances/reason_codes": np.asarray(
                [0 if instance_reason_label is None else 1],
                dtype=np.uint16,
            ),
            **{
                f"source_detections/{name}": value.copy()
                for name, value in shared.items()
            },
            "source_detections/source_detect_row_index": np.asarray(
                [0], dtype=np.int64
            ),
            "source_detections/decision_codes": np.asarray(
                [SOURCE_DECISION_CODE_MAP["accepted"]], dtype=np.uint8
            ),
            "source_detections/resolved_refined_row_id": np.asarray(
                [refined_row_id], dtype=np.int64
            ),
            "source_detections/reason_codes": np.asarray(
                [0 if source_reason_label is None else 1],
                dtype=np.uint16,
            ),
        }
    )
    offsets = np.asarray(
        [0, int(local_frame == 0), 1],
        dtype=np.int64,
    )
    arrays["instances/frame_row_offsets"] = offsets.copy()
    arrays["source_detections/frame_row_offsets"] = offsets.copy()
    REFINED_DETECTION_SCHEMA_V1.require(arrays, dimensions=dimensions)
    raw = RefinedDetectionSourceIdentity(
        run_id=f"detect_clip_{clip_index}",
        run_manifest_digest=f"{clip_index + 1:x}" * 64,
        logical_content_digest=f"{clip_index + 3:x}" * 64,
    )
    plans = plan_refined_detection_storage(dimensions)
    direct, consolidated = _metadata_declarations(dimensions, plans)
    manifest = build_refined_detection_run_manifest(
        run_id=f"refined_clip_{clip_index}",
        dimensions=dimensions,
        storage_plan=plans,
        lineage=RefinedDetectionSnapshotLineage(
            lineage_id=f"00000000-0000-4000-8000-0000000000{clip_index + 10:02d}",
            snapshot_id=f"00000000-0000-4000-8000-0000000000{clip_index + 20:02d}",
            recording_identity=RECORDING_IDENTITY,
            next_refined_row_id=refined_row_id + 1,
        ),
        source=raw,
        instance_reason_codes={
            0: "none",
            **({1: instance_reason_label} if instance_reason_label else {}),
        },
        source_reason_codes={
            0: "none",
            **({1: source_reason_label} if source_reason_label else {}),
        },
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=False,
    )
    return (
        RefinedDetectionBoundClipEvidence(
            clip_index=clip_index,
            manifest=manifest,
            arrays=arrays,
        ),
        raw,
    )


def _fixture(
    *,
    instance_reason_labels: tuple[str | None, str | None] = (None, None),
    source_reason_labels: tuple[str | None, str | None] = (None, None),
):
    clip_0, _ = _clip(
        clip_index=0,
        parent_start=0,
        local_frame=0,
        refined_row_id=7,
        bbox=np.asarray([0.25, 0.4, 0.2, 0.2]),
        class_id=1,
        instance_reason_label=instance_reason_labels[0],
        source_reason_label=source_reason_labels[0],
    )
    clip_1, _ = _clip(
        clip_index=1,
        parent_start=2,
        local_frame=1,
        refined_row_id=8,
        bbox=np.asarray([0.75, 0.6, 0.1, 0.2]),
        class_id=2,
        instance_reason_label=instance_reason_labels[1],
        source_reason_label=source_reason_labels[1],
    )
    evidence = (clip_0, clip_1)
    binding = RefinedDetectionClippedBinding(
        collection_id="collection_1",
        collection_manifest_digest="a" * 64,
        camera_serial="2010095",
        video_identity=RECORDING_IDENTITY,
        video_manifest_digest="b" * 64,
        recording_frame_index_digest="c" * 64,
        clips=tuple(
            RefinedDetectionClipBinding(
                clip_index=index,
                clip_id=f"clip_{index}",
                media_identity=f"clip_{index}.mp4",
                media_digest=f"{index + 4:x}" * 64,
                parent_frame_start=index * 2,
                parent_frame_stop=index * 2 + 2,
                frame_map_digest=f"{index + 6:x}" * 64,
                source_refined_run_id=evidence[index].manifest["payload"]["run_id"],
                source_refined_manifest_digest=evidence[index].manifest[
                    "payload_digest"
                ],
            )
            for index in range(2)
        ),
    )
    canonical_dimensions = CanonicalDetectionDimensions(
        n_frames=4,
        n_instances=2,
        source_width=100,
        source_height=80,
    )
    canonical_arrays: dict[str, np.ndarray] = {}
    for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
        name = path.split("/", 1)[1]
        if name == "frame_row_offsets":
            canonical_arrays[path] = np.asarray([0, 1, 1, 1, 2], dtype=np.int64)
        else:
            canonical_arrays[path] = np.concatenate(
                [
                    np.asarray(item.arrays[f"source_detections/{name}"])
                    for item in evidence
                ],
                axis=0,
            )
    canonical_arrays["instances/frame_indices"] = np.asarray([0, 3], dtype=np.int32)
    canonical_arrays["instances/source_acquisition_frame_index"] = np.asarray(
        [0, 3], dtype=np.int64
    )
    CANONICAL_DETECTION_SCHEMA_V1.require(
        canonical_arrays,
        dimensions=canonical_dimensions,
    )
    canonical_source = RefinedDetectionSourceIdentity(
        run_id="detect_recording",
        run_manifest_digest="d" * 64,
        logical_content_digest=canonical_detection_logical_content_digest(
            canonical_arrays,
            dimensions=canonical_dimensions,
        ),
    )
    return evidence, binding, canonical_arrays, canonical_dimensions, canonical_source


def test_prepares_complete_recording_pair_with_empty_frames() -> None:
    evidence, binding, canonical, canonical_dimensions, source = _fixture()
    prepared = prepare_clipped_refined_detection_snapshot(
        evidence,
        clipped_binding=binding,
        canonical_arrays=canonical,
        canonical_dimensions=canonical_dimensions,
        canonical_source=source,
        recording_identity=RECORDING_IDENTITY,
    )

    assert prepared.dimensions.n_frames == 4
    assert prepared.dimensions.n_instances == 2
    assert prepared.arrays["instances/frame_indices"].tolist() == [0, 3]
    assert prepared.arrays["instances/frame_row_offsets"].tolist() == [0, 1, 1, 1, 2]
    assert prepared.arrays["source_detections/source_detect_row_index"].tolist() == [
        0,
        1,
    ]
    assert prepared.arrays["instances/source_clip_indices"].tolist() == [0, 1]
    assert prepared.arrays["instances/source_refined_row_ids"].tolist() == [7, 8]
    assert len(prepared.source_collection.members) == 2


def test_merges_and_remaps_clip_local_reason_registries() -> None:
    evidence, binding, canonical, canonical_dimensions, source = _fixture(
        instance_reason_labels=("edge_case", "occluded"),
        source_reason_labels=("filtered_jump", "filtered_blip"),
    )

    prepared = prepare_clipped_refined_detection_snapshot(
        evidence,
        clipped_binding=binding,
        canonical_arrays=canonical,
        canonical_dimensions=canonical_dimensions,
        canonical_source=source,
        recording_identity=RECORDING_IDENTITY,
    )

    assert prepared.instance_reason_codes == {
        0: "none",
        1: "edge_case",
        2: "occluded",
    }
    assert prepared.arrays["instances/reason_codes"].tolist() == [1, 2]
    assert prepared.source_reason_codes == {
        0: "none",
        1: "filtered_blip",
        2: "filtered_jump",
    }
    assert prepared.arrays["source_detections/reason_codes"].tolist() == [2, 1]


def test_rejects_overlapping_global_refined_row_ids() -> None:
    evidence, binding, canonical, canonical_dimensions, source = _fixture()
    tampered_arrays = copy.deepcopy(evidence[1].arrays)
    tampered_arrays["instances/refined_row_ids"][0] = 7
    tampered_arrays["source_detections/resolved_refined_row_id"][0] = 7
    tampered = (
        evidence[0],
        RefinedDetectionBoundClipEvidence(
            clip_index=1,
            manifest=evidence[1].manifest,
            arrays=tampered_arrays,
        ),
    )
    with pytest.raises(
        ClippedRefinedDetectionFinalizationError,
        match="refined_row_ids overlap",
    ):
        prepare_clipped_refined_detection_snapshot(
            tampered,
            clipped_binding=binding,
            canonical_arrays=canonical,
            canonical_dimensions=canonical_dimensions,
            canonical_source=source,
            recording_identity=RECORDING_IDENTITY,
        )


def test_rejects_canonical_pair_mismatch() -> None:
    evidence, binding, canonical, canonical_dimensions, source = _fixture()
    changed = copy.deepcopy(canonical)
    changed["instances/scores"][0] = np.float32(0.1)
    with pytest.raises(
        ClippedRefinedDetectionFinalizationError,
        match="canonical logical digest",
    ):
        prepare_clipped_refined_detection_snapshot(
            evidence,
            clipped_binding=binding,
            canonical_arrays=changed,
            canonical_dimensions=canonical_dimensions,
            canonical_source=source,
            recording_identity=RECORDING_IDENTITY,
        )


def test_publishes_and_rebinds_clipped_snapshot_for_crop(tmp_path: Path) -> None:
    evidence, binding, canonical, canonical_dimensions, source = _fixture()
    prepared = prepare_clipped_refined_detection_snapshot(
        evidence,
        clipped_binding=binding,
        canonical_arrays=canonical,
        canonical_dimensions=canonical_dimensions,
        canonical_source=source,
        recording_identity=RECORDING_IDENTITY,
    )
    safe_root = tmp_path / "snapshots"
    publication = publish_selector_ineligible_clipped_refined_detection_snapshot(
        prepared,
        destination=safe_root / "refined.zarr",
        run_id="refined_recording",
        safe_root=safe_root,
        lineage_id="10000000-0000-4000-8000-000000000001",
        snapshot_id="20000000-0000-4000-8000-000000000002",
        recording_identity=RECORDING_IDENTITY,
    )

    assert (
        validate_clipped_refined_detection_finalization_receipt(publication.receipt)
        == ()
    )
    rebound = bind_refined_detection_crop_source(
        publication.snapshot.output_path,
        run_id="refined_recording",
        allow_selector_ineligible_benchmark=True,
        clipped_source_evidence=evidence,
    )
    assert rebound.dimensions == prepared.dimensions
    assert (
        rebound.logical_content_digest
        == publication.snapshot.receipt["logical_content_digest"]
    )
    crop_prepared = prepare_crop_geometry_from_refined_source(
        rebound,
        policy=CropGeometryPolicy(
            purpose="subject_analysis",
            size_mode=CropSizeMode.FIXED_PER_RUN,
            fixed_size_wh=(8, 8),
            padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
        ),
        pixel_authority=CropPixelAuthority(
            authority_id="fixture_source_video",
            authority_manifest_digest="e" * 64,
            recording_identity=RECORDING_IDENTITY,
            camera_identity="2010095",
            n_frames=4,
            source_width=100,
            source_height=80,
        ),
    )
    crop_root = tmp_path / "crops"
    crop = publish_selector_ineligible_crop_geometry_snapshot(
        crop_prepared,
        destination=crop_root / "crop.zarr",
        run_id="crop_recording",
        shadow_root=crop_root,
        coordinate_catalog=True,
    )
    assert validate_crop_geometry_shadow_publication(crop) == ()
    assert crop.arrays["frame_row_offsets"][:].tolist() == [0, 1, 1, 1, 2]
