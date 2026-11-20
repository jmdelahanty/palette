"""Background run computed from a recording's own video keyframes."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.preprocessing import video_keyframe_median_background as stage
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import resolve_latest_complete_run_name

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required to encode and decode the test video",
)

CAMERA = "2010093"
SHAPE = (64, 96)
GOP = 5
FRAMES = 60


def _frame(index: int) -> np.ndarray:
    """A flat level that identifies the frame, with a fixed bright square."""

    image = np.full(SHAPE, 40 + 2 * index, dtype=np.uint8)
    image[8:24, 8:24] = 200
    return image


def _recording(tmp_path: Path, *, with_video: bool) -> Path:
    recording_dir = tmp_path / "rec"
    folder = recording_dir / "cams" / "acquisition" / "external_recorder"
    folder.mkdir(parents=True)
    (recording_dir / "recording_manifest.json").write_text(json.dumps({"camera_id": CAMERA}))
    video = folder / f"Cam{CAMERA}_external.mp4"
    if with_video:
        raw = b"".join(_frame(i).tobytes() for i in range(FRAMES))
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                "-s", f"{SHAPE[1]}x{SHAPE[0]}", "-r", "30", "-i", "-",
                # Motion JPEG: full-range yuvj420p like Orange's recorder output, and
                # available in every ffmpeg build. Every frame is a keyframe.
                "-c:v", "mjpeg", "-q:v", "1", "-pix_fmt", "yuvj420p", str(video),
            ],
            input=raw,
            check=True,
        )
    else:
        video.write_bytes(b"not decoded in this test")
    (folder / f"Cam{CAMERA}_external_keyframes.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_frames": FRAMES,
                "keyframe_video_frame_indices": list(range(0, FRAMES, 1 if with_video else GOP)),
            }
        )
    )
    zarr_path = recording_dir / "zarr" / "rec_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3, use_consolidated=False)
    root.create_group("raw_video")
    consolidate_metadata_capture_expected_warnings(zarr_path)
    return recording_dir


def test_keyframes_are_spread_evenly_and_include_both_ends_of_the_window() -> None:
    keyframes = list(range(0, 1000, 25))

    assert stage.select_keyframes(keyframes, count=5) == (0, 250, 500, 725, 975)
    assert stage.select_keyframes(keyframes, count=3, first_frame=110, last_frame=410) == (125, 275, 400)


def test_a_window_with_too_few_keyframes_is_refused() -> None:
    with pytest.raises(ValueError, match="holds 4 keyframes"):
        stage.select_keyframes(list(range(0, 1000, 25)), count=5, first_frame=0, last_frame=75)
    with pytest.raises(ValueError, match="at least 3"):
        stage.select_keyframes([0, 25], count=2)


def test_the_downsampled_copy_is_640_square_uint8() -> None:
    image = np.arange(1280 * 1280, dtype=np.uint32).reshape(1280, 1280) % 251
    ds = stage.downsample_background(image.astype(np.uint8))

    assert ds.shape == (640, 640) and ds.dtype == np.uint8
    with pytest.raises(ValueError, match="2-D uint8"):
        stage.downsample_background(image.astype(np.float32))


def test_the_run_holds_both_rasters_and_becomes_latest(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "a.zarr"), mode="w", zarr_format=3, use_consolidated=False)
    image = (np.arange(128 * 128).reshape(128, 128) % 200).astype(np.uint8)
    source = {"source_frame_indices": [0, 25, 50], "total_frames": 60, "video_relative_path": "v.mp4"}

    run_name = stage.write_video_keyframe_median_background(
        root, background_full=image, source=source, parameters={"method": stage.METHOD}
    )

    parent = root["background_runs"]
    assert run_name.startswith("background_video_keyframe_median_")
    assert resolve_latest_complete_run_name(parent, legacy_default=False) == run_name
    group = parent[run_name]
    assert np.array_equal(group["background_full"][:], image)
    assert np.array_equal(group["background_ds"][:], stage.downsample_background(image))
    attrs = dict(group.attrs)
    assert attrs["method"] == "video_keyframe_median" and attrs["n_frames_used"] == 3
    assert attrs["source_frame_indices"] == [0, 25, 50]
    assert attrs["background_source"] == "recording_video_keyframes"


def test_a_video_that_is_not_full_range_luma_is_refused(tmp_path: Path, monkeypatch) -> None:
    recording_dir = _recording(tmp_path, with_video=False)
    monkeypatch.setattr(
        stage,
        "probe_video",
        lambda path: {"pix_fmt": "yuv420p", "width": 96, "height": 64, "nb_frames": FRAMES},
    )

    with pytest.raises(ValueError, match="not a full-range luma source"):
        stage.plan_recording(recording_dir, count=5)


def test_a_keyframe_list_that_disagrees_with_the_video_is_refused(tmp_path: Path, monkeypatch) -> None:
    recording_dir = _recording(tmp_path, with_video=False)
    monkeypatch.setattr(
        stage,
        "probe_video",
        lambda path: {"pix_fmt": "yuvj420p", "width": 96, "height": 64, "nb_frames": FRAMES + 1},
    )

    with pytest.raises(ValueError, match="frame count disagrees"):
        stage.plan_recording(recording_dir, count=5)


def test_the_median_of_the_sampled_keyframes_is_published(tmp_path: Path, monkeypatch) -> None:
    recording_dir = _recording(tmp_path, with_video=False)
    monkeypatch.setattr(
        stage,
        "probe_video",
        lambda path: {"pix_fmt": "yuvj420p", "width": SHAPE[1], "height": SHAPE[0], "nb_frames": FRAMES},
    )

    def fake_decode(video_path, *, declared_keyframes, wanted, shape_hw, threads=4):
        return np.stack([_frame(index) for index in wanted]), "sha256:" + "ab" * 32

    monkeypatch.setattr(stage, "decode_keyframes", fake_decode)

    result = stage.compute_recording(recording_dir, count=5, first_frame=10, last_frame=55, window_note="adaptation")

    assert result["status"] == "written" and result["selected_as_latest"] is True
    assert result["source_frame_indices"] == [10, 20, 35, 45, 55]
    published = zarr.open_group(str(recording_dir / "zarr" / "rec_analysis.zarr"), mode="r", use_consolidated=True)
    group = published["background_runs"][result["run_name"]]
    # Median over frames 10, 20, 35, 45, 55 is frame 35's level; the square is constant.
    assert np.array_equal(group["background_full"][:], _frame(35))
    assert group["background_ds"].shape == (640, 640)
    assert group.attrs["parameters"]["window_note"] == "adaptation"
    assert group.attrs["video_keyframe_median"]["decoded_keyframes_sha256"] == "sha256:" + "ab" * 32


def test_the_dry_run_command_writes_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    recording_dir = _recording(tmp_path, with_video=False)
    monkeypatch.setattr(
        stage,
        "probe_video",
        lambda path: {"pix_fmt": "yuvj420p", "width": SHAPE[1], "height": SHAPE[0], "nb_frames": FRAMES},
    )

    assert stage.main([str(recording_dir), "--keyframes", "3", "--dry-run"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "would_write" and report["source_frame_indices"] == [0, 30, 55]
    root = zarr.open_group(str(recording_dir / "zarr" / "rec_analysis.zarr"), mode="r", use_consolidated=False)
    assert "background_runs" not in root


@needs_ffmpeg
def test_decoding_returns_exactly_the_wanted_keyframes(tmp_path: Path) -> None:
    recording_dir = _recording(tmp_path, with_video=True)
    video = recording_dir / "cams" / "acquisition" / "external_recorder" / f"Cam{CAMERA}_external.mp4"
    declared = list(range(FRAMES))

    probe = stage.probe_video(video)
    stack, digest = stage.decode_keyframes(
        video, declared_keyframes=declared, wanted=[5, 30, 55], shape_hw=SHAPE
    )

    assert probe["pix_fmt"] == "yuvj420p" and probe["nb_frames"] == FRAMES
    assert digest.startswith("sha256:")
    # Full range in and out: each frame's own level comes back (JPEG is near-exact
    # on flat images), so the right pictures were picked and no range was rescaled.
    for row, index in enumerate([5, 30, 55]):
        assert np.abs(stack[row].astype(int) - _frame(index).astype(int)).max() <= 3
        assert abs(int(stack[row][40, 60]) - (40 + 2 * index)) <= 1
    with pytest.raises(ValueError, match="not a declared keyframe"):
        stage.decode_keyframes(video, declared_keyframes=declared[::2], wanted=[7], shape_hw=SHAPE)


@needs_ffmpeg
def test_a_video_with_fewer_keyframes_than_declared_fails(tmp_path: Path) -> None:
    recording_dir = _recording(tmp_path, with_video=True)
    video = recording_dir / "cams" / "acquisition" / "external_recorder" / f"Cam{CAMERA}_external.mp4"
    declared = list(range(FRAMES * 2))  # twice as many as the video holds

    with pytest.raises(RuntimeError, match="decoding stopped"):
        stage.decode_keyframes(video, declared_keyframes=declared, wanted=[FRAMES * 2 - 1], shape_hw=SHAPE)
