from __future__ import annotations

import json
from pathlib import Path

from fisheye.cluster.crop_snapshot import (
    CropSnapshotFragmentInputs,
    build_crop_snapshot_fragment,
)
from fisheye.shared.zarr.crop_schema import CropPaddingMode, CropSizeMode
from fisheye.utils import publish_crop_geometry_candidate as cli


def _inputs(tmp_path: Path, *, family: str) -> CropSnapshotFragmentInputs:
    return CropSnapshotFragmentInputs(
        workflow_id="crop_fixture",
        family=family,
        target_id="recording_a",
        analysis_zarr=tmp_path / "recording_analysis.zarr",
        repo=tmp_path / "repo",
        run_root=tmp_path / "run",
        run_id="crop_recording_a",
        purpose="keypoints",
        roi_width=512,
        roi_height=512,
        camera_id="2010093",
        upstream_job_keys=("refined_activate:recording_a",),
        required_artifacts=("approved_refined_detection:recording_a",),
    )


def test_crop_fragment_is_partition_independent_and_selector_ineligible(
    tmp_path: Path,
) -> None:
    module = build_crop_snapshot_fragment(
        _inputs(tmp_path, family="analysis.clipped")
    )
    job = module.fragment.jobs[0]
    command = " ".join(job.command)

    assert job.job_key == "crop_snapshot_publish:recording_a"
    assert job.dependency.upstream_job_keys == ("refined_activate:recording_a",)
    assert "fisheye.utils.publish_crop_geometry_candidate" in command
    assert "--run-id crop_recording_a" in command
    assert "--roi-width 512 --roi-height 512" in command
    assert module.fragment.requires == ("approved_refined_detection:recording_a",)
    assert module.fragment.metadata["lineage_profile"] == "full_acquisition"
    assert module.fragment.metadata["compute_partition_independent"] is True
    assert module.outputs.to_json()["selector_eligible"] is False


def test_crop_fragment_uses_same_publisher_for_whole_recording(tmp_path: Path) -> None:
    clipped = build_crop_snapshot_fragment(
        _inputs(tmp_path, family="analysis.clipped")
    )
    whole = build_crop_snapshot_fragment(
        _inputs(tmp_path, family="analysis.whole")
    )

    assert clipped.outputs.to_json() == whole.outputs.to_json()
    assert clipped.fragment.metadata == whole.fragment.metadata
    assert "fisheye.utils.publish_crop_geometry_candidate" in " ".join(
        whole.fragment.jobs[0].command
    )


def test_crop_candidate_cli_builds_fixed_policy_and_writes_receipt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_publish(**kwargs):
        captured.update(kwargs)
        return {
            "schema_id": "palette.crop_geometry.snapshot_publication",
            "status": "complete",
            "run_id": kwargs["run_id"],
            "selector_eligible": False,
        }

    monkeypatch.setattr(cli, "publish_crop_geometry_production_candidate", fake_publish)
    result_json = tmp_path / "receipt.json"

    exit_code = cli.main(
        [
            "--analysis-zarr",
            str(tmp_path / "analysis.zarr"),
            "--run-id",
            "crop_a",
            "--purpose",
            "keypoints",
            "--roi-width",
            "512",
            "--roi-height",
            "384",
            "--camera-id",
            "2010093",
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--result-json",
            str(result_json),
        ]
    )

    assert exit_code == 0
    assert json.loads(result_json.read_text(encoding="utf-8"))["status"] == "complete"
    policy = captured["policy"]
    assert policy.size_mode is CropSizeMode.FIXED_PER_RUN
    assert policy.fixed_size_wh == (512, 384)
    assert policy.padding_mode is CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME
    assert captured["expected_camera_identity"] == "2010093"

