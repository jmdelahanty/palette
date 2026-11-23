"""Blob-detection artifact boxes become a refined run the web reviewer can open."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    resolve_latest_complete_run_name,
)
from fisheye.tune import detect_review_backend as backend
from fisheye.utils import seed_refined_detect_from_blob_artifact as seed

FRAMES = 4
# Frame 0: three fish, frame 1: none, frame 2: one, frame 3: two.
FRAME_INDICES = [0, 0, 0, 2, 3, 3]
BOXES = np.asarray(
    [
        [0.20, 0.20, 0.05, 0.04],
        [0.50, 0.30, 0.05, 0.04],
        [0.70, 0.80, 0.04, 0.05],
        [0.40, 0.40, 0.05, 0.05],
        [0.10, 0.90, 0.04, 0.04],
        [0.90, 0.10, 0.04, 0.04],
    ],
    dtype=np.float64,
)


def _archive(tmp_path: Path, *, complete: bool = True, counts_frames: int = FRAMES) -> Path:
    zarr_path = tmp_path / "rec_training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3, use_consolidated=False)
    root.attrs.update({"recording_id": "blob-seed-test"})
    raw = root.create_group("raw_video")
    raw.attrs["original_resolution"] = [640, 640]
    raw.create_array("images_ds", data=np.zeros((FRAMES, 64, 64), dtype=np.uint8))
    artifact = root.create_group("detection_artifact_runs").create_group("detect_blobs")
    artifact.create_array("frame_indices", data=np.asarray(FRAME_INDICES, dtype=np.int32))
    artifact.create_array("bbox_norm_coords", data=BOXES)
    counts = np.bincount(FRAME_INDICES, minlength=counts_frames).astype(np.int32)
    artifact.create_array("frame_counts", data=counts)
    if complete:
        artifact.attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_COMPLETE
    return zarr_path


def _tree(path: Path) -> dict[str, bytes]:
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def test_dry_run_reports_the_boxes_and_writes_nothing(tmp_path: Path) -> None:
    zarr_path = _archive(tmp_path)
    before = _tree(zarr_path)

    report = seed.seed_refined_detect(zarr_path, run_name="refined_seed")

    assert report["status"] == "would_write" and report["boxes"] == 6
    assert report["max_boxes_per_frame"] == 3 and report["frames_with_boxes"] == 3
    assert report["boxes_per_frame_histogram"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert _tree(zarr_path) == before


def test_every_box_reaches_the_web_reviewer_with_its_own_key(tmp_path: Path, monkeypatch) -> None:
    zarr_path = _archive(tmp_path)

    report = seed.seed_refined_detect(zarr_path, run_name="refined_seed", apply=True)

    assert report["status"] == "written"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root["refined_detect_runs"]
    assert resolve_latest_complete_run_name(parent) == "refined_seed"
    instances = parent["refined_seed"]["instances"]
    assert instances["frame_indices"][:].tolist() == FRAME_INDICES
    assert np.allclose(instances["bbox_norm_coords"][:], BOXES)
    keys = instances["instance_key"][:]
    assert np.unique(keys).shape[0] == 6 and not np.any(keys == 0)
    assert "detect_review_status" not in parent["refined_seed"].attrs

    # The labeling web app's detect_training session opens it in multi-box mode.
    monkeypatch.setattr(backend.detect_review_mod, "get_environment_info", lambda **_kwargs: {})
    session = backend.resolve_review_context(str(zarr_path), include_all=True)
    assert session.review_axis == "frame_instances"
    assert session.total_frames == FRAMES
    result = backend.apply_detection_collection(
        session,
        position=0,
        detections=[
            {"instance_key": str(int(keys[0])), "bbox_norm": [0.21, 0.20, 0.05, 0.04], "class_id": 0},
            {"instance_key": str(int(keys[1])), "bbox_norm": BOXES[1].tolist(), "class_id": 0},
            {"instance_key": None, "bbox_norm": [0.33, 0.66, 0.05, 0.05], "class_id": 0},
        ],
    )
    # One box moved, one left as it was, one new, one dropped.
    assert (result["added"], result["updated"], result["removed"]) == (1, 1, 1)


def test_an_existing_run_name_is_refused(tmp_path: Path) -> None:
    zarr_path = _archive(tmp_path)
    seed.seed_refined_detect(zarr_path, run_name="refined_seed", apply=True)

    with pytest.raises(ValueError, match="already exists"):
        seed.seed_refined_detect(zarr_path, run_name="refined_seed", apply=True)


def test_an_incomplete_artifact_is_refused(tmp_path: Path) -> None:
    zarr_path = _archive(tmp_path, complete=False)

    with pytest.raises(ValueError, match="is not complete"):
        seed.seed_refined_detect(zarr_path, apply=True)
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "refined_detect_runs" not in root


def test_an_artifact_for_another_frame_count_is_refused(tmp_path: Path) -> None:
    zarr_path = _archive(tmp_path, counts_frames=FRAMES + 2)

    with pytest.raises(ValueError, match="covers 6 frames"):
        seed.seed_refined_detect(zarr_path)


def test_a_downsample_that_changed_the_aspect_ratio_is_refused(tmp_path: Path) -> None:
    zarr_path = _archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="r+", use_consolidated=False)
    root["raw_video"].attrs["original_resolution"] = [480, 640]

    with pytest.raises(ValueError, match="aspect ratio"):
        seed.seed_refined_detect(zarr_path)


def test_the_command_is_a_dry_run_unless_apply_is_given(tmp_path: Path, capsys) -> None:
    zarr_path = _archive(tmp_path)

    assert seed.main([str(zarr_path), "--run-name", "refined_seed"]) == 0
    assert '"status": "would_write"' in capsys.readouterr().out
    assert seed.main([str(zarr_path), "--run-name", "refined_seed", "--apply"]) == 0
    assert '"status": "written"' in capsys.readouterr().out
