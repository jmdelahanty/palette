from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.diagnostics.benchmark_crop_pixel_materialization_consumers import (
    _consumer_commands,
    _prepare_command,
    _require_node_local_scratch,
)


def test_canary_rejects_shared_scratch() -> None:
    with pytest.raises(ValueError, match="node-local"):
        _require_node_local_scratch(Path("/groups/example/canary"))


def test_consumer_commands_are_shard_only_and_use_one_package(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    package = tmp_path / "package.json"
    keypoints, masks = _consumer_commands(
        local_archive=archive,
        crop_run="crop_canary",
        package_manifest=package,
        keypoint_model=tmp_path / "pose.pt",
        subject_mask_model=tmp_path / "masks.pt",
        keypoint_run="kp",
        mask_run="sm",
        batch_rows=16,
        device="cuda:0",
    )

    assert keypoints[keypoints.index("--output-parent") + 1] == (
        "keypoint_shard_runs"
    )
    assert masks[masks.index("--output-parent") + 1] == (
        "subject_mask_shard_runs"
    )
    assert keypoints[keypoints.index("--roi-work-package-manifest") + 1] == str(
        package
    )
    assert masks[masks.index("--roi-work-package-manifest") + 1] == str(package)
    assert keypoints[keypoints.index("--coordinate-contract-mode") + 1] == (
        "legacy_noncanonical"
    )
    assert "--defer-registry-status" in masks


def test_prepare_command_is_a_separate_worker_process(tmp_path: Path) -> None:
    class _Args:
        source_analysis_zarr = tmp_path / "source.zarr"
        recording_dir = tmp_path / "recording"
        crop_meta = tmp_path / "crop.csv"
        crop_video = tmp_path / "crop.mp4"
        crop_run = "crop_canary"
        row_count = 2048
        batch_rows = 256
        source_width = None
        source_height = None

    command = _prepare_command(
        _Args(),
        scratch=tmp_path / "scratch",
        output_json=tmp_path / "prepare.json",
    )

    assert command[1:3] == [
        "-m",
        "fisheye.diagnostics.prepare_crop_pixel_materialization_canary",
    ]
    assert command[command.index("--row-count") + 1] == "2048"
