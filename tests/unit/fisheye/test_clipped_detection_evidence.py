from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.cluster.clipped_detection_evidence import (
    ClipDetectionEvidenceInput,
    ClippedDetectionEvidenceInputs,
    build_clipped_detection_evidence_fragment,
    build_clipped_detection_storage_fragments,
)
from fisheye.detection.clipped_native_binding import (
    ClippedDetectionArtifactMember,
    bind_clipped_detection_artifacts,
)
from fisheye.detection.native_canonical_candidate import (
    write_native_clipped_detection_candidate,
)
from fisheye.shared.detect_reason_codec import encode_reason_bytes
from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.shared.zarr.clipped_detection_evidence import (
    allocate_automated_clip_refined_ids,
    publish_strict_clip_detection_evidence,
)
from fisheye.shared.zarr.detection_schema import (
    CanonicalDetectionDimensions,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.refined_detection_transition import (
    build_accept_all_refined_detection_root,
)
from fisheye.shared.zarr.refined_detection_schema import (
    SOURCE_DECISION_CODE_MAP,
    SOURCE_KIND_CODE_MAP,
)
from tests.unit.fisheye.test_clipped_storage_finalization_workflow import (
    _inputs as storage_inputs,
)


def _transition():
    dimensions = CanonicalDetectionDimensions(
        n_frames=3,
        n_instances=3,
        source_width=100,
        source_height=80,
    )
    frames = np.asarray([0, 2, 2], dtype=np.int32)
    boxes = np.asarray(
        [
            [0.2, 0.2, 0.1, 0.1],
            [0.5, 0.5, 0.2, 0.2],
            [0.8, 0.8, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    classes = np.asarray([1, 1, 2], dtype=np.int32)
    image_boxes, centers = derive_canonical_detection_geometry(
        boxes,
        source_width=100,
        source_height=80,
    )
    canonical = {
        "instances/frame_indices": frames,
        "instances/source_acquisition_frame_index": frames.astype(np.int64),
        "instances/instance_key": mint_detection_instance_keys(
            recording_identity="recording",
            frame_indices=frames,
            bbox_norm_coords=boxes,
            class_ids=classes,
        ),
        "instances/bbox_norm_coords": boxes,
        "instances/bbox_img_xyxy": image_boxes,
        "instances/centers_img_xy": centers,
        "instances/scores": np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        "instances/class_ids": classes,
        "instances/frame_row_offsets": np.asarray([0, 1, 1, 3], dtype=np.int64),
    }
    return build_accept_all_refined_detection_root(
        canonical,
        dimensions=dimensions,
        recording_identity="recording",
    )


def test_allocates_automated_ids_from_recording_canonical_rows() -> None:
    allocated = allocate_automated_clip_refined_ids(
        _transition(),
        canonical_row_start=7,
        recording_source_row_count=20,
    )

    assert allocated.arrays["instances/refined_row_ids"].tolist() == [7, 8, 9]
    assert allocated.arrays["source_detections/resolved_refined_row_id"].tolist() == [
        7,
        8,
        9,
    ]
    assert allocated.report["identity_allocation"] == {
        "policy": "recording_canonical_source_row_position_v1",
        "manual_rows_allowed": False,
        "canonical_source_row_start": 7,
        "canonical_source_row_stop": 10,
        "recording_next_refined_row_id": 20,
    }


def test_automated_allocator_rejects_manual_rows() -> None:
    transition = _transition()
    transition.arrays["instances/manual_edit_flags"][1] = True
    with pytest.raises(ValueError, match="recording-level delta allocator"):
        allocate_automated_clip_refined_ids(
            transition,
            canonical_row_start=0,
            recording_source_row_count=3,
        )


def test_publishes_clip_pair_bound_to_native_recording_canonical(
    tmp_path: Path,
) -> None:
    recording_identity = "recording:native-clip"
    local_frames = np.asarray([0, 2], dtype=np.int32)
    boxes = np.asarray(
        [[0.25, 0.25, 0.1, 0.1], [0.75, 0.75, 0.1, 0.1]],
        dtype=np.float64,
    )
    scores = np.asarray([0.9, 0.8], dtype=np.float32)
    classes = np.asarray([0, 0], dtype=np.int32)
    counts = np.asarray([1, 0, 1], dtype=np.int32)
    member = ClippedDetectionArtifactMember(
        work_unit_id="work_0",
        artifact_run_id="artifact_0",
        clip_id="clip_000000",
        clip_index=0,
        camera_serial="2010095",
        source_width=640,
        source_height=480,
        artifact_manifest_sha256="a" * 64,
        run_group_tree_sha256="b" * 64,
        parent_frame_indices=np.arange(3, dtype=np.int64),
        frame_indices=local_frames,
        bbox_norm_coords=boxes,
        scores=scores,
        class_ids=classes,
        artifact_row_id=np.arange(2, dtype=np.uint64),
        frame_counts=counts,
        n_detections=counts.copy(),
    )
    bound = bind_clipped_detection_artifacts(
        [member],
        recording_identity=recording_identity,
        n_frames=3,
        source_width=640,
        source_height=480,
    )
    recording = write_native_clipped_detection_candidate(
        bound,
        destination=tmp_path / "recording-canonical.zarr",
        run_id="recording_canonical",
        recording_identity=recording_identity,
        producer_id="fisheye.detection.detect_yolo",
        producer_version="test",
        source_frame_authority={
            "record_ref": "/analysis/acquisition_camera_frames/2010095@record",
            "record_sha256": "1" * 64,
        },
        source_pixel_authority={
            "record_ref": "/analysis/acquisition_camera_frames/2010095@record",
            "record_sha256": "1" * 64,
        },
        model_artifact_sha256="2" * 64,
        run_provenance={
            "schema_id": "test.provenance",
            "schema_version": 1,
        },
    )

    analysis = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = recording_identity
    raw = (
        root.require_group("clips")
        .require_group("clip_000000")
        .require_group("cameras")
        .require_group("2010095")
        .require_group("detection_artifact_runs")
        .create_group("artifact_0")
    )
    raw.attrs.update({"source_video_width": 640, "source_video_height": 480})
    keys = np.asarray(bound.arrays["instances/instance_key"])
    raw.create_array("frame_indices", data=local_frames)
    raw.create_array("bbox_norm_coords", data=boxes)
    raw.create_array("scores", data=scores)
    raw.create_array("class_ids", data=classes)
    raw.create_array("instance_key", data=keys)
    raw.create_array("frame_counts", data=counts)
    refined = (
        root["clips/clip_000000/cameras/2010095"]
        .require_group("refined_detect_runs")
        .create_group("legacy_refined")
    )
    instances = refined.create_group("instances")
    instances.create_array("frame_indices", data=local_frames[:1])
    instances.create_array("refined_row_ids", data=np.asarray([0], dtype=np.int64))
    instances.create_array("bbox_norm_coords", data=boxes[:1])
    instances.create_array(
        "source_kind_codes",
        data=np.asarray([SOURCE_KIND_CODE_MAP["raw_detect"]], dtype=np.int8),
    )
    instances.create_array("manual_edit_flags", data=np.asarray([False]))
    instances.create_array(
        "source_detect_row_index",
        data=np.asarray([0], dtype=np.int32),
    )
    instances.create_array("confidence_scores", data=scores[:1])
    instances.create_array("class_ids", data=classes[:1])
    instances.create_array("instance_key", data=keys[:1])
    instances.create_array(
        "reason_bytes",
        data=encode_reason_bytes(np.asarray(["clean"], dtype=object)),
    )
    source = refined.create_group("source_detections")
    source.create_array(
        "source_detect_row_index",
        data=np.arange(2, dtype=np.int32),
    )
    source.create_array("frame_indices", data=local_frames)
    source.create_array("bbox_norm_coords", data=boxes)
    source.create_array(
        "decision_codes",
        data=np.asarray(
            [
                SOURCE_DECISION_CODE_MAP["accepted"],
                SOURCE_DECISION_CODE_MAP["filtered"],
            ],
            dtype=np.int8,
        ),
    )
    source.create_array(
        "resolved_refined_row_id",
        data=np.asarray([0, -1], dtype=np.int64),
    )
    source.create_array("confidence_scores", data=scores)
    source.create_array("class_ids", data=classes)
    source.create_array("instance_key", data=keys)
    source.create_array(
        "reason_bytes",
        data=encode_reason_bytes(np.asarray(["clean", "filtered_blip"], dtype=object)),
    )

    output_root = tmp_path / ".palette_benchmarks" / "strict-clips"
    publication = publish_strict_clip_detection_evidence(
        analysis_zarr=analysis,
        source_detect_group_path=(
            "clips/clip_000000/cameras/2010095/detection_artifact_runs/artifact_0"
        ),
        source_refined_group_path=(
            "clips/clip_000000/cameras/2010095/refined_detect_runs/legacy_refined"
        ),
        recording_canonical_archive=recording.output_path,
        recording_canonical_run_id=recording.run_id,
        recording_identity=recording_identity,
        clip_id="clip_000000",
        clip_index=0,
        output_root=output_root,
        canonical_run_id="strict_canonical_0",
        refined_run_id="strict_refined_0",
    )

    assert publication.receipt["status"] == "complete"
    assert publication.receipt["selector_eligible"] is False
    assert (
        publication.refined.manifest["payload"]["snapshot_lineage"][
            "refined_row_id_allocator"
        ]["next_id"]
        == 2
    )
    refined_root = zarr.open_group(
        str(publication.refined.output_path),
        mode="r",
        use_consolidated=True,
    )
    strict = refined_root["refined_detect_runs/strict_refined_0"]
    assert strict["instances/refined_row_ids"][:].tolist() == [0]
    assert strict["source_detections/resolved_refined_row_id"][:].tolist() == [
        0,
        -1,
    ]


def test_evidence_fragment_freezes_selector_ineligible_chain(tmp_path: Path) -> None:
    bundle = tmp_path / ".palette_benchmarks" / "strict"
    inputs = ClippedDetectionEvidenceInputs(
        workflow_id="workflow",
        family="clipped",
        target_id="target",
        analysis_zarr=tmp_path / "analysis.zarr",
        recording_canonical_archive=tmp_path / "analysis.zarr",
        recording_canonical_run_id="canonical_recording",
        recording_identity="recording",
        detection_plan_path=tmp_path / "plan.json",
        collection_id="collection",
        recording_dir=tmp_path / "recording",
        bundle_root=bundle,
        clips=(
            ClipDetectionEvidenceInput(
                clip_index=0,
                clip_id="clip_000000",
                source_detect_group_path="clips/clip_000000/detect_runs/raw",
                source_refined_group_path=(
                    "clips/clip_000000/refined_detect_runs/refined"
                ),
                canonical_run_id="canonical_clip_0",
                refined_run_id="refined_clip_0",
            ),
        ),
        repo=tmp_path / "repo",
        run_root=tmp_path / "run",
        upstream_job_keys=("canonical", "refine"),
        required_artifacts=("canonical_detection", "refined_collection"),
    )

    module = build_clipped_detection_evidence_fragment(inputs)

    assert [job.job_key for job in module.fragment.jobs] == [
        "strict_detection_evidence_array:target",
        "strict_detection_binding:target",
    ]
    assert module.fragment.jobs[0].dependency.upstream_job_keys == (
        "canonical",
        "refine",
    )
    assert module.fragment.jobs[1].dependency.upstream_job_keys == (
        "strict_detection_evidence_array:target",
    )
    command = module.fragment.jobs[0].execution_group.tasks[0].command
    assert "fisheye.utils.publish_strict_clip_detection_evidence" in command
    assert module.fragment.metadata["manual_rows_allowed"] is False
    assert module.outputs.clips[0].archive == (
        bundle / "clip_000000_clip_000000" / "refined.zarr"
    )


def test_composition_binds_evidence_into_recording_storage(tmp_path: Path) -> None:
    bundle = tmp_path / ".palette_benchmarks" / "strict"
    evidence = ClippedDetectionEvidenceInputs(
        workflow_id="wf",
        family="analysis.clipped",
        target_id="sleepyfish",
        analysis_zarr=tmp_path / "recording_analysis.zarr",
        recording_canonical_archive=tmp_path / "canonical.zarr",
        recording_canonical_run_id="detect_recording",
        recording_identity="sleepyfish",
        detection_plan_path=tmp_path / "plan.json",
        collection_id="collection",
        recording_dir=tmp_path / "recording",
        bundle_root=bundle,
        clips=tuple(
            ClipDetectionEvidenceInput(
                clip_index=index,
                clip_id=f"clip_{index}",
                source_detect_group_path=f"clips/clip_{index}/detect_runs/raw",
                source_refined_group_path=(
                    f"clips/clip_{index}/refined_detect_runs/refined"
                ),
                canonical_run_id=f"canonical_clip_{index}",
                refined_run_id=f"strict_refined_clip_{index}",
            )
            for index in range(2)
        ),
        repo=tmp_path / "repo",
        run_root=tmp_path / "run",
        upstream_job_keys=("detect_recording", "legacy_refine"),
        required_artifacts=("canonical_detection", "refined_collection"),
    )

    modules = build_clipped_detection_storage_fragments(
        evidence,
        storage_inputs(tmp_path),
    )

    assert modules.evidence.outputs.artifact_key in modules.storage.fragment.requires
    first_storage_job = modules.storage.fragment.jobs[0]
    assert modules.evidence.outputs.terminal_job_key in (
        first_storage_job.dependency.upstream_job_keys
    )
    assert modules.storage.outputs.refined_run_id == "refined_recording"
    assert [item.run_id for item in modules.evidence.outputs.clips] == [
        "strict_refined_clip_0",
        "strict_refined_clip_1",
    ]
