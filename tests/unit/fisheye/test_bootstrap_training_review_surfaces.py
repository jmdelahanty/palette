from __future__ import annotations

from pathlib import Path

from fisheye.utils import bootstrap_training_review_surfaces as mod


def test_bootstrap_training_review_surfaces_uses_deterministic_run_handoffs(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}

    def _fake_detect_keypoints_yolo(**kwargs):
        calls["detect"] = kwargs
        return kwargs["run_name"]

    def _fake_create_refined_keypoint_run(*args, **kwargs):
        calls["refine_args"] = args
        calls["refine"] = kwargs
        return kwargs["run_name"]

    def _fake_subject_main(argv):
        calls["subject_argv"] = list(argv)

    def _fake_finalize_subject_masks(*args, **kwargs):
        calls["finalize_args"] = args
        calls["finalize"] = kwargs
        return {"status": "updated", "refined_run": kwargs["refined_run"]}

    monkeypatch.setattr(mod, "detect_keypoints_yolo", _fake_detect_keypoints_yolo)
    monkeypatch.setattr(mod, "create_refined_keypoint_run", _fake_create_refined_keypoint_run)
    monkeypatch.setattr(mod.infer_unet_subject_masks, "main", _fake_subject_main)
    monkeypatch.setattr(mod, "finalize_subject_masks", _fake_finalize_subject_masks)

    result = mod.bootstrap_training_review_surfaces(
        zarr_path=tmp_path / "training.zarr",
        crop_run="crop_acq",
        pose_model=Path("/models/pose.pt"),
        registry=tmp_path / "registry.sqlite",
        run_id="red_scare_001",
    )

    assert result.keypoints_run == "keypoints_training_review_red_scare_001"
    assert result.refined_keypoints_run == "refined_keypoints_training_review_red_scare_001"
    assert calls["detect"]["run_name"] == result.keypoints_run
    assert calls["detect"]["model_input_transform_mode"] == "auto"
    assert calls["detect"]["imgsz"] == 512
    assert calls["refine"]["keypoint_run"] == result.keypoints_run
    assert calls["refine"]["run_name"] == result.refined_keypoints_run

    subject_argv = calls["subject_argv"]
    assert subject_argv[subject_argv.index("--assignment-keypoint-run") + 1] == result.refined_keypoints_run
    assert subject_argv[subject_argv.index("--model-input-size") + 1] == "512"

    assert calls["finalize"]["subject_run"] == result.subject_masks_run
    assert calls["finalize"]["assignment_keypoints_run"] == result.refined_keypoints_run
    assert calls["finalize"]["refined_run"] == result.refined_subject_masks_run
    assert calls["finalize"]["registry"] == (tmp_path / "registry.sqlite").resolve()
