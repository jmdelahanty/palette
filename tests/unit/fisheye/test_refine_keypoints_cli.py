from __future__ import annotations

from fisheye.refinement import refine_keypoints as mod


def test_refine_keypoints_cli_forwards_run_name(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_create_refined_keypoint_run(**kwargs):
        captured.update(kwargs)
        return "refined_custom"

    monkeypatch.setattr(mod, "create_refined_keypoint_run", _fake_create_refined_keypoint_run)

    mod.main(
        [
            str(tmp_path / "recording.zarr"),
            "--keypoint-run",
            "keypoints_custom",
            "--run-name",
            "refined_custom",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--no-post-audit",
        ]
    )

    assert captured["keypoint_run"] == "keypoints_custom"
    assert captured["run_name"] == "refined_custom"

