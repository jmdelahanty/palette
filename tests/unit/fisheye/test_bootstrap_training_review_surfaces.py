from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.refinement.refine_keypoints import (
    RefinedKeypointCoordinatePublicationUnavailable,
)
from fisheye.utils import bootstrap_training_review_surfaces as mod


def test_bootstrap_training_review_surfaces_fails_before_keypoint_publication(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str] = []

    def _fake_detect_keypoints_yolo(**kwargs):
        calls.append("detect")
        return kwargs["run_name"]

    monkeypatch.setattr(mod, "detect_keypoints_yolo", _fake_detect_keypoints_yolo)

    with pytest.raises(RefinedKeypointCoordinatePublicationUnavailable):
        mod.bootstrap_training_review_surfaces(
            zarr_path=tmp_path / "training.zarr",
            crop_run="crop_acq",
            pose_model=Path("/models/pose.pt"),
            registry=tmp_path / "registry.sqlite",
            run_id="red_scare_001",
        )

    assert calls == []
