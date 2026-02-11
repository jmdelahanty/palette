from __future__ import annotations

from pathlib import Path

from fisheye.utils import review_crops as mod


def test_main_forwards_review_flags_to_visualizer(monkeypatch, tmp_path: Path) -> None:
    plans = [
        mod.CropReviewPlan(
            zarr_path=Path("/tmp/rec_a_analysis.zarr"),
            crop_run="crop_001",
            status="ok",
            total_crops=10,
        ),
    ]

    monkeypatch.setattr(mod, "_build_plans", lambda *_args, **_kwargs: plans)

    seen_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool = False) -> None:  # noqa: ARG001
        seen_cmds.append(cmd)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(
        [
            str(tmp_path),
            "--recursive",
            "--no-prompt",
            "--review-state",
            "approved",
            "--review-method",
            "spotcheck",
            "--review-intended-use",
            "full_recording",
            "--reviewer",
            "tester",
            "--review-notes",
            "batch pass-through",
        ]
    )

    assert rc == 0
    assert len(seen_cmds) == 1
    cmd = seen_cmds[0]
    assert cmd[1:3] == ["-m", "fisheye.visualization.visualize_crops"]
    assert "--crop-run" in cmd
    assert "crop_001" in cmd
    assert "--review-state" in cmd
    assert "approved" in cmd
    assert "--review-method" in cmd
    assert "spotcheck" in cmd
    assert "--review-intended-use" in cmd
    assert "full_recording" in cmd
    assert "--reviewer" in cmd
    assert "tester" in cmd
    assert "--review-notes" in cmd
    assert "batch pass-through" in cmd
