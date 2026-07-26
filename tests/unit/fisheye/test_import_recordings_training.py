import json
import subprocess
import sys
from pathlib import Path

import h5py
import pytest

from fisheye.utils import import_recordings_training as mod


def _write_h5(path: Path, *, camera_id: str = "2010093") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["camera_id"] = camera_id


def _write_recording(root: Path, name: str, *, frame_count: int | None = None) -> Path:
    rec = root / name
    _write_h5(rec / "raw" / f"{name}.h5")
    cams = rec / "cams"
    cams.mkdir(parents=True, exist_ok=True)
    (cams / "Cam2010093_demo.mp4").write_bytes(b"")
    if frame_count is not None:
        (rec / "recording_manifest.json").write_text(
            json.dumps(
                {
                    "video_streams": {
                        "streams": {
                            "full": {
                                "frame_count": frame_count,
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
    return rec


def test_build_plans_targets_training_zarr_and_computes_frame_step(tmp_path: Path) -> None:
    rec = _write_recording(tmp_path, "rec_a_GoodCopBadCop", frame_count=139_877)

    plans = mod._build_plans(
        tmp_path,
        recursive=False,
        skip_existing=True,
        check_stimulus=False,
        requested_frame_step=None,
        target_sampled_frames=200,
        skip_tail_frames=200,
        path_contains="GoodCopBadCop",
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.recording_dir == rec
    assert plan.zarr_path == rec / "zarr" / "rec_a_GoodCopBadCop_training.zarr"
    assert plan.frame_step == 699
    assert plan.estimated_sampled_frames == 200
    assert plan.source_frame_count == 139_877
    assert plan.frame_count_source == "recording_manifest.video_streams.full.frame_count"


def test_build_plans_can_require_source_frame_count_for_pynvvc(tmp_path: Path) -> None:
    _write_recording(tmp_path, "rec_missing_count_GoodCopBadCop")

    plans = mod._build_plans(
        tmp_path,
        recursive=False,
        skip_existing=True,
        check_stimulus=False,
        requested_frame_step=10,
        target_sampled_frames=None,
        skip_tail_frames=0,
        path_contains="GoodCopBadCop",
        require_source_frame_count=True,
    )

    assert len(plans) == 1
    assert plans[0].status == "missing"
    assert "source frame count" in (plans[0].reason or "")


def test_build_plans_reads_legacy_external_summary_frame_count(tmp_path: Path) -> None:
    rec = _write_recording(tmp_path, "rec_b_GoodCopBadCop")
    summary_rel = "cams/Cam2010093_demo_external_summary.json"
    (rec / "recording_manifest.json").write_text(
        json.dumps({"files": {"cams": ["cams/Cam2010093_demo.mp4", summary_rel]}}),
        encoding="utf-8",
    )
    (rec / summary_rel).write_text(json.dumps({"frames_encoded": 143_447}), encoding="utf-8")

    plans = mod._build_plans(
        tmp_path,
        recursive=False,
        skip_existing=True,
        check_stimulus=False,
        requested_frame_step=None,
        target_sampled_frames=200,
        skip_tail_frames=200,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.frame_step == 717
    assert plan.estimated_sampled_frames == 200
    assert plan.source_frame_count == 143_447
    assert plan.frame_count_source == f"{summary_rel}:frames_encoded"


def test_target_sampled_frames_requires_fallback_when_count_missing(tmp_path: Path) -> None:
    _write_recording(tmp_path, "rec_c_GoodCopBadCop")

    plans = mod._build_plans(
        tmp_path,
        recursive=False,
        skip_existing=True,
        check_stimulus=False,
        requested_frame_step=None,
        target_sampled_frames=200,
        skip_tail_frames=200,
    )

    assert len(plans) == 1
    assert plans[0].status == "missing"
    assert "--target-sampled-frames" in (plans[0].reason or "")


def test_frame_step_is_fallback_when_target_count_missing_metadata(tmp_path: Path) -> None:
    _write_recording(tmp_path, "rec_d_GoodCopBadCop")

    plans = mod._build_plans(
        tmp_path,
        recursive=False,
        skip_existing=True,
        check_stimulus=False,
        requested_frame_step=700,
        target_sampled_frames=200,
        skip_tail_frames=200,
    )

    assert len(plans) == 1
    assert plans[0].status == "ok"
    assert plans[0].frame_step == 700
    assert plans[0].estimated_sampled_frames is None


def test_build_plans_limit_applies_after_path_filter(tmp_path: Path) -> None:
    _write_recording(tmp_path, "rec_a_other", frame_count=1000)
    target = _write_recording(tmp_path, "rec_b_GoodCopBadCop", frame_count=1000)
    _write_recording(tmp_path, "rec_c_GoodCopBadCop", frame_count=1000)

    plans = mod._build_plans(
        tmp_path,
        recursive=False,
        skip_existing=True,
        check_stimulus=False,
        requested_frame_step=None,
        target_sampled_frames=200,
        skip_tail_frames=0,
        path_contains="GoodCopBadCop",
        limit=1,
    )

    assert len(plans) == 1
    assert plans[0].recording_dir == target


def test_run_import_uses_pynvvc_backend_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _write_recording(tmp_path, "rec_pynvvc_GoodCopBadCop", frame_count=100)
    plan = mod._build_plans(
        tmp_path,
        recursive=False,
        skip_existing=True,
        check_stimulus=False,
        requested_frame_step=10,
        target_sampled_frames=None,
        skip_tail_frames=0,
        path_contains="GoodCopBadCop",
        require_source_frame_count=True,
    )[0]
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    success, returncode = mod._run_import(
        plan,
        config_path=Path("configs/fisheye/import_local.yaml"),
        overwrite=False,
        skip_tail_frames=0,
        decode_backend=mod.DECODE_BACKEND_PYNVVC_LUMA,
        gpu_id=2,
    )

    assert success is True
    assert returncode == 0
    cmd = calls[0]
    assert cmd[:3] == [sys.executable, "-m", "fisheye.utils.import_sampled_training_pynvvc"]
    assert str(rec / "cams" / "Cam2010093_demo.mp4") in cmd
    assert "--source-frame-count" in cmd
    assert "100" in cmd
    assert "--frame-step" in cmd
    assert "10" in cmd
    assert "--gpu-id" in cmd
    assert "2" in cmd


def test_run_acquisition_crop_video_append_uses_recording_scoped_run_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _write_recording(tmp_path, "rec_red_scare_RedScare", frame_count=100)
    plan = mod._build_plans(
        tmp_path,
        recursive=False,
        skip_existing=True,
        check_stimulus=False,
        requested_frame_step=10,
        target_sampled_frames=None,
        skip_tail_frames=0,
        path_contains="RedScare",
        require_source_frame_count=True,
    )[0]
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    success, returncode = mod._run_acquisition_crop_video_append(
        plan,
        run_name_prefix="crop_red_scare_acquisition_crop_video_training",
        overwrite_run=True,
        gpu_id=3,
    )

    assert success is True
    assert returncode == 0
    cmd = calls[0]
    assert cmd[:3] == [sys.executable, "-m", "fisheye.utils.append_acquisition_crop_video_training"]
    assert str(rec / "zarr" / "rec_red_scare_RedScare_training.zarr") in cmd
    assert "--recording-dir" in cmd
    assert str(rec) in cmd
    assert "--run-name" in cmd
    assert "crop_red_scare_acquisition_crop_video_training_rec_red_scare_RedScare" in cmd
    assert "--gpu-id" in cmd
    assert "3" in cmd
    assert "--overwrite-run" in cmd


def test_decord_decode_backend_is_not_supported(tmp_path: Path) -> None:
    _write_recording(tmp_path, "rec_legacy_GoodCopBadCop", frame_count=100)

    with pytest.raises(SystemExit, match="2"):
        mod.main([
            str(tmp_path),
            "--path-contains",
            "GoodCopBadCop",
            "--decode-backend",
            "legacy-decord",
            "--apply",
            "--no-log",
        ])
