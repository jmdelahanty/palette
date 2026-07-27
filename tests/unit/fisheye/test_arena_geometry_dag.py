from __future__ import annotations

from pathlib import Path

from fisheye.cluster.arena_geometry import (
    ArenaGeometrySelectionFragmentInputs,
    RegisteredDetectionGateFragmentInputs,
    build_arena_geometry_selection_fragment,
    build_registered_detection_gate_fragment,
)
from fisheye.cluster.recording_layout import (
    clipped_recording_target,
    whole_video_recording_target,
)


def test_geometry_fragments_are_layout_neutral_and_explicitly_ordered(
    tmp_path: Path,
) -> None:
    whole = whole_video_recording_target(
        target_id="whole",
        recording_id="recording-whole",
        recording_dir=tmp_path / "whole",
        analysis_zarr=tmp_path / "whole" / "analysis.zarr",
        video_path=tmp_path / "whole" / "video.mp4",
        camera_serial="2010093",
    )
    selected = build_arena_geometry_selection_fragment(
        ArenaGeometrySelectionFragmentInputs(
            workflow_id="geometry-test",
            family="recording_analysis",
            target=whole,
            repo=tmp_path / "repo",
            run_root=tmp_path / "run",
            candidate_run="candidate_1",
            selection_run="selection_1",
            selected_by="reviewer@example.org",
            decision_reason="reviewed",
            upstream_job_keys=("candidate:whole",),
            required_artifacts=("arena_geometry_candidate:whole",),
        )
    )
    gated = build_registered_detection_gate_fragment(
        RegisteredDetectionGateFragmentInputs(
            workflow_id="geometry-test",
            family="recording_analysis",
            target=whole,
            repo=tmp_path / "repo",
            run_root=tmp_path / "run",
            source_detection_group_path="detect_runs/detect_1",
            selection_run="selection_1",
            output_run="gate_1",
            upstream_job_keys=(
                selected.outputs.terminal_job_key,
                "detect:whole",
            ),
            required_artifacts=(
                selected.outputs.artifact_key,
                "raw_detection_work_units:whole",
            ),
        )
    )

    assert selected.fragment.metadata["recording_layout"] == "whole_video"
    assert gated.fragment.metadata["recording_layout"] == "whole_video"
    assert gated.fragment.metadata["row_identity"] == "instance_key"
    assert gated.fragment.metadata["raw_detections_preserved"] is True
    assert gated.fragment.requires == (
        "arena_geometry_selection:whole",
        "raw_detection_work_units:whole",
    )
    assert gated.fragment.jobs[0].dependency.upstream_job_keys == (
        "arena_geometry_selection:whole",
        "detect:whole",
    )


def test_gate_fragment_accepts_clipped_collection_adapter(tmp_path: Path) -> None:
    frame_index = tmp_path / "recording_frame_index.parquet"
    clipped = clipped_recording_target(
        target_id="clips",
        recording_id="recording-clips",
        recording_dir=tmp_path / "clips",
        analysis_zarr=tmp_path / "clips" / "analysis.zarr",
        work_units=(
            {
                "clip_id": "clip_000001",
                "clip_index": 0,
                "work_unit_id": "clip_000001:2010093",
                "camera_serial": "2010093",
                "source": {"video_path": str(tmp_path / "clip.mp4")},
            },
        ),
        recording_frame_index=frame_index,
    )
    module = build_registered_detection_gate_fragment(
        RegisteredDetectionGateFragmentInputs(
            workflow_id="geometry-test",
            family="recording_analysis",
            target=clipped,
            repo=tmp_path / "repo",
            run_root=tmp_path / "run",
            source_detection_group_path="detect_collection_sources/source_1",
            selection_run="selection_1",
            output_run="gate_1",
        )
    )

    assert module.fragment.metadata["recording_layout"] == "clipped_collection"
    assert (
        module.outputs.source_detection_group_path
        == "detect_collection_sources/source_1"
    )
