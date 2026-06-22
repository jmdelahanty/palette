import json
from pathlib import Path

import h5py

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
