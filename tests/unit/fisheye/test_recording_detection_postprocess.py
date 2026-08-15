from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.cluster.recording_detection_postprocess import (
    RecordingDetectionPostprocessInputs,
    build_recording_detection_postprocess_fragment,
)
from fisheye.cluster.recording_layout import (
    clipped_recording_target,
    whole_video_recording_target,
)


def _inputs(tmp_path: Path, target, **overrides):
    values = {
        "workflow_id": "recording-postprocess",
        "family": "recording-analysis",
        "target": target,
        "repo": tmp_path / "repo",
        "run_root": tmp_path / "run",
        "source_detect_run": "detect_native",
        "quality_run": "quality_native",
        "refined_run": "refined_native",
        "registered_gate_requirement": "required",
        "registered_gate_run": "gate_exact",
        "upstream_job_keys": ("registered_detection_gate:target",),
        "required_artifacts": ("registered_detection_gate:target",),
    }
    values.update(overrides)
    return RecordingDetectionPostprocessInputs(**values)


def test_whole_video_and_clipped_use_same_canonical_postprocess_contract(
    tmp_path: Path,
) -> None:
    whole = whole_video_recording_target(
        target_id="whole",
        recording_id="recording",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "recording" / "analysis.zarr",
        video_path=tmp_path / "recording" / "video.mp4",
        camera_serial="2010093",
    )
    clipped = clipped_recording_target(
        target_id="clips",
        recording_id="recording",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "recording" / "analysis.zarr",
        recording_frame_index=tmp_path / "recording" / "recording_frame_index.parquet",
        work_units=(
            {
                "clip_id": "clip_000000",
                "clip_index": 0,
                "work_unit_id": "clip_000000:2010093",
                "camera_serial": "2010093",
                "source": {"video_path": str(tmp_path / "recording" / "clip.mp4")},
            },
        ),
    )

    whole_module = build_recording_detection_postprocess_fragment(
        _inputs(
            tmp_path,
            whole,
            canonicalize_legacy_source=True,
            canonical_source_run="detect_native_canonical_v3",
        )
    )
    clipped_module = build_recording_detection_postprocess_fragment(
        _inputs(tmp_path, clipped)
    )

    assert whole_module.outputs.input_detect_group_path == "detect_runs/detect_native"
    assert whole_module.outputs.source_detect_group_path == (
        "detect_runs/detect_native_canonical_v3"
    )
    assert clipped_module.outputs.input_detect_group_path == "detect_runs/detect_native"
    assert clipped_module.outputs.source_detect_group_path == "detect_runs/detect_native"

    for module in (whole_module, clipped_module):
        assert module.outputs.working_refined_group_path == (
            "refined_detect_runs/refined_native_working"
        )
        assert module.outputs.refined_group_path == "refined_detect_runs/refined_native"
        assert module.fragment.metadata["source_authority"] == (
            "canonical_recording_detect_run"
        )
        assert module.fragment.metadata["legacy_dish_mask_policy_coupled"] is False
        quality, refine = module.fragment.jobs[-2:]
        assert module.outputs.source_detect_group_path in quality.command
        rendered = " ".join(refine.command)
        assert "--registered-gate-requirement" in rendered
        assert "required" in rendered
        assert "--registered-gate-run" in rendered
        assert "gate_exact" in rendered
        assert "--selector-ineligible" in rendered
        assert "fisheye.utils.finalize_recording_refined_detection_v1" in rendered
        assert "--working-refined-run refined_native_working" in rendered
        assert "--output-run refined_native" in rendered
        assert "--selection-policy-id manual_review_only_v1" in rendered
        assert module.fragment.metadata["refined_finalization"] == (
            "immutable_refined_v1_snapshot"
        )
    assert whole_module.fragment.metadata["recording_layout"] == "whole_video"
    assert clipped_module.fragment.metadata["recording_layout"] == "clipped_collection"
    canonicalize = whole_module.fragment.jobs[0]
    rendered_canonicalize = " ".join(canonicalize.command)
    assert "fisheye.utils.publish_canonical_detection_successor" in (
        rendered_canonicalize
    )
    assert "--source-detect-group detect_runs/detect_native" in rendered_canonicalize
    assert "--successor-run detect_native_canonical_v3" in rendered_canonicalize
    assert whole_module.fragment.jobs[1].dependency.upstream_job_keys == (
        canonicalize.job_key,
    )
    assert clipped_module.fragment.jobs[0].dependency.upstream_job_keys == (
        "registered_detection_gate:target",
    )
    assert whole_module.fragment.metadata["canonicalize_legacy_source"] is True
    assert clipped_module.fragment.metadata["canonicalize_legacy_source"] is False


def test_required_mode_rejects_missing_exact_gate(tmp_path: Path) -> None:
    whole = whole_video_recording_target(
        target_id="whole",
        recording_id="recording",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "recording" / "analysis.zarr",
        video_path=tmp_path / "recording" / "video.mp4",
        camera_serial="2010093",
    )
    with pytest.raises(ValueError, match="exact gate run"):
        _inputs(tmp_path, whole, registered_gate_run=None)


def test_if_available_persists_explicit_unavailable_capability(tmp_path: Path) -> None:
    whole = whole_video_recording_target(
        target_id="whole",
        recording_id="recording",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "recording" / "analysis.zarr",
        video_path=tmp_path / "recording" / "video.mp4",
        camera_serial="2010093",
    )
    module = build_recording_detection_postprocess_fragment(
        _inputs(
            tmp_path,
            whole,
            registered_gate_requirement="if_available",
            registered_gate_run=None,
            upstream_job_keys=("detect_native_publish:whole",),
            required_artifacts=("canonical_detection:whole",),
        )
    )
    assert module.fragment.metadata["registered_gate_requirement"] == "if_available"
    assert module.fragment.metadata["registered_gate_run"] is None
    refine = module.fragment.jobs[1]
    assert "--registered-gate-run" not in " ".join(refine.command)


def test_legacy_canonicalization_requires_distinct_successor(tmp_path: Path) -> None:
    whole = whole_video_recording_target(
        target_id="whole",
        recording_id="recording",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "recording" / "analysis.zarr",
        video_path=tmp_path / "recording" / "video.mp4",
        camera_serial="2010093",
    )
    with pytest.raises(ValueError, match="successor run id"):
        _inputs(tmp_path, whole, canonicalize_legacy_source=True)
    with pytest.raises(ValueError, match="differ from its source"):
        _inputs(
            tmp_path,
            whole,
            canonicalize_legacy_source=True,
            canonical_source_run="detect_native",
        )


def test_active_canonical_source_is_enforced_by_quality_refine_and_finalize(
    tmp_path: Path,
) -> None:
    target = whole_video_recording_target(
        target_id="whole",
        recording_id="recording",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "recording" / "analysis.zarr",
        video_path=tmp_path / "recording" / "video.mp4",
        camera_serial="2010093",
    )
    module = build_recording_detection_postprocess_fragment(
        _inputs(
            tmp_path,
            target,
            require_active_canonical_source=True,
        )
    )

    quality, refine = module.fragment.jobs
    assert "--require-active-canonical-source" in quality.command
    assert " ".join(refine.command).count("--require-active-canonical-source") == 2
    assert module.outputs.require_active_canonical_source is True
    assert module.outputs.to_json()["require_active_canonical_source"] is True

    with pytest.raises(ValueError, match="cannot claim active canonical authority"):
        _inputs(
            tmp_path,
            target,
            canonicalize_legacy_source=True,
            canonical_source_run="detect_native_canonical_v3",
            require_active_canonical_source=True,
        )
