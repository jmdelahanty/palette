from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fisheye.detection.clipped_native_binding import (
    ClippedDetectionArtifactMember,
    bind_clipped_detection_artifacts,
)
from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _member(
    *,
    clip_index: int,
    parent_start: int,
    frame_count: int,
    frame_indices: np.ndarray,
    bbox_norm_coords: np.ndarray,
) -> ClippedDetectionArtifactMember:
    rows = int(frame_indices.shape[0])
    counts = np.bincount(
        frame_indices.astype(np.int64),
        minlength=frame_count,
    ).astype(np.int32)
    return ClippedDetectionArtifactMember(
        work_unit_id=f"work_{clip_index}",
        artifact_run_id=f"artifact_{clip_index}",
        clip_id=f"clip_{clip_index:06d}",
        clip_index=clip_index,
        camera_serial="2010093",
        source_width=4512,
        source_height=4512,
        artifact_manifest_sha256=_DIGEST_A,
        run_group_tree_sha256=_DIGEST_B,
        parent_frame_indices=np.arange(
            parent_start,
            parent_start + frame_count,
            dtype=np.int64,
        ),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        bbox_norm_coords=np.asarray(bbox_norm_coords, dtype=np.float64),
        scores=np.linspace(0.8, 0.9, rows, dtype=np.float32),
        class_ids=np.zeros(rows, dtype=np.int32),
        artifact_row_id=np.arange(rows, dtype=np.uint64),
        frame_counts=counts,
        n_detections=counts.copy(),
    )


def _two_members() -> tuple[ClippedDetectionArtifactMember, ...]:
    return (
        _member(
            clip_index=0,
            parent_start=0,
            frame_count=3,
            frame_indices=np.asarray([0, 2], dtype=np.int32),
            bbox_norm_coords=np.asarray(
                [
                    [0.25, 0.25, 0.10, 0.12],
                    [0.40, 0.50, 0.08, 0.10],
                ],
                dtype=np.float64,
            ),
        ),
        _member(
            clip_index=1,
            parent_start=3,
            frame_count=2,
            frame_indices=np.asarray([0, 0, 1], dtype=np.int32),
            bbox_norm_coords=np.asarray(
                [
                    [0.55, 0.45, 0.10, 0.10],
                    [0.60, 0.40, 0.09, 0.08],
                    [0.70, 0.70, 0.12, 0.11],
                ],
                dtype=np.float64,
            ),
        ),
    )


def test_binding_emits_exact_canonical_v1_arrays_and_evidence() -> None:
    members = _two_members()
    result = bind_clipped_detection_artifacts(
        tuple(reversed(members)),
        recording_identity="recording:fixture",
        n_frames=5,
        source_width=4512,
        source_height=4512,
    )

    assert tuple(result.arrays) == CANONICAL_DETECTION_SCHEMA_V1.binding_paths
    assert result.dimensions.n_instances == 5
    assert result.arrays["instances/frame_indices"].tolist() == [0, 2, 3, 3, 4]
    assert result.arrays["instances/frame_row_offsets"].tolist() == [0, 1, 1, 2, 4, 5]
    assert result.arrays["instances/bbox_norm_coords"].dtype == np.dtype(np.float32)
    assert result.arrays["instances/source_acquisition_frame_index"].dtype == np.dtype(
        np.int64
    )
    assert result.arrays["instances/instance_key"].dtype == np.dtype(np.uint64)
    assert result.arrays["instances/instance_key"].flags.writeable is False
    CANONICAL_DETECTION_SCHEMA_V1.require(
        result.arrays,
        dimensions=result.dimensions,
    )

    evidence = result.binding_evidence["document"]
    assert evidence["camera_serial"] == "2010093"
    assert [item["clip_index"] for item in evidence["members"]] == [0, 1]
    assert evidence["members"][0]["canonical_row_start"] == 0
    assert evidence["members"][1]["canonical_row_stop"] == 5
    assert result.binding_evidence["digest_algorithm"] == "sha256_canonical_json_v1"


def test_split_binding_mints_same_keys_as_recording_level_input() -> None:
    result = bind_clipped_detection_artifacts(
        _two_members(),
        recording_identity="recording:fixture",
        n_frames=5,
        source_width=4512,
        source_height=4512,
    )
    expected = mint_detection_instance_keys(
        recording_identity="recording:fixture",
        frame_indices=result.arrays["instances/frame_indices"],
        bbox_norm_coords=result.arrays["instances/bbox_norm_coords"],
        class_ids=result.arrays["instances/class_ids"],
    )
    assert np.array_equal(result.arrays["instances/instance_key"], expected)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            {"frame_indices": np.asarray([0, 2], dtype=np.int64)},
            "frame_indices dtype",
        ),
        (
            {"bbox_norm_coords": np.ones((2, 4), dtype=np.float32)},
            "bbox_norm_coords dtype",
        ),
        (
            {"artifact_row_id": np.asarray([1, 0], dtype=np.uint64)},
            "dense run-local range",
        ),
    ],
)
def test_binding_rejects_noncanonical_artifact_contract(
    replacement: dict[str, np.ndarray],
    message: str,
) -> None:
    members = list(_two_members())
    members[0] = replace(members[0], **replacement)
    with pytest.raises(ValueError, match=message):
        bind_clipped_detection_artifacts(
            members,
            recording_identity="recording:fixture",
            n_frames=5,
            source_width=4512,
            source_height=4512,
        )


def test_binding_rejects_gap_or_overlap_in_parent_frame_coverage() -> None:
    members = list(_two_members())
    members[1] = replace(
        members[1],
        parent_frame_indices=np.asarray([4, 5], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="cover the canonical recording exactly once"):
        bind_clipped_detection_artifacts(
            members,
            recording_identity="recording:fixture",
            n_frames=5,
            source_width=4512,
            source_height=4512,
        )


def test_binding_rejects_mixed_cameras_and_duplicate_work_units() -> None:
    members = list(_two_members())
    with pytest.raises(ValueError, match="duplicate work_unit_id"):
        bind_clipped_detection_artifacts(
            [members[0], replace(members[1], work_unit_id=members[0].work_unit_id)],
            recording_identity="recording:fixture",
            n_frames=5,
            source_width=4512,
            source_height=4512,
        )

    with pytest.raises(ValueError, match="exactly one camera"):
        bind_clipped_detection_artifacts(
            [members[0], replace(members[1], camera_serial="2010094")],
            recording_identity="recording:fixture",
            n_frames=5,
            source_width=4512,
            source_height=4512,
        )
