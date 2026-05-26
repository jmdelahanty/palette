from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

from fisheye.utils.build_review_proxy_videos import (
    ReviewProxyOptions,
    build_ffmpeg_review_proxy_command,
    build_review_proxy_manifest,
    build_review_proxy_videos,
    resolve_review_proxy_encoder,
)


def _write_clip_index(recording_dir: Path) -> None:
    clip_dir = recording_dir / "clips" / "clip_000000"
    clip_dir.mkdir(parents=True)
    video = clip_dir / "Cam2010093.mp4"
    video.write_bytes(b"source video")
    payload = {
        "recording_id": "rec_a",
        "clips": [
            {
                "clip_id": "clip_000000",
                "clip_index": 0,
                "camera_artifacts": [
                    {
                        "camera_serial": "2010093",
                        "video_path": "clips/clip_000000/Cam2010093.mp4",
                        "frame_count": 120,
                        "fps": 30,
                        "source_width": 4512,
                        "source_height": 4512,
                    }
                ],
            }
        ],
    }
    (recording_dir / "recording_clip_index.json").write_text(json.dumps(payload), encoding="utf-8")


def _fake_ffprobe(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    payload = {
        "streams": [
            {
                "codec_name": "hevc",
                "width": 4512,
                "height": 4512,
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
                "nb_frames": "120",
                "duration": "4.0",
            }
        ]
    }
    return subprocess.CompletedProcess(list(command), 0, stdout=json.dumps(payload), stderr="")


def _fake_ffmpeg_encoders(*encoders: str):
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        lines = ["Encoders:"]
        lines.extend(f" V..... {encoder} fake encoder" for encoder in encoders)
        return subprocess.CompletedProcess(list(command), 0, stdout="\n".join(lines), stderr="")

    return runner


def test_build_ffmpeg_review_proxy_command_uses_faststart_h264(tmp_path: Path) -> None:
    command = build_ffmpeg_review_proxy_command(
        ffmpeg_bin="ffmpeg",
        source_video=tmp_path / "source.mp4",
        output_video=tmp_path / "proxy.mp4",
        proxy_width=1024,
        proxy_height=1024,
        encoder="libx264",
        overwrite=False,
    )

    assert command[:6] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    assert command[command.index("-vf") + 1] == "scale=1024:1024:flags=lanczos"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert command[-1] == str(tmp_path / "proxy.mp4")


def test_build_ffmpeg_review_proxy_command_uses_nvenc_quality_options(tmp_path: Path) -> None:
    command = build_ffmpeg_review_proxy_command(
        ffmpeg_bin="ffmpeg",
        source_video=tmp_path / "source.mp4",
        output_video=tmp_path / "proxy.mp4",
        proxy_width=1024,
        proxy_height=1024,
        encoder="h264_nvenc",
        overwrite=False,
    )

    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-preset") + 1] == "p3"
    assert command[command.index("-cq") + 1] == "23"
    assert "-crf" not in command


def test_build_ffmpeg_review_proxy_command_can_use_hwaccel_and_bilinear_scale(tmp_path: Path) -> None:
    command = build_ffmpeg_review_proxy_command(
        ffmpeg_bin="ffmpeg",
        source_video=tmp_path / "source.mp4",
        output_video=tmp_path / "proxy.mp4",
        proxy_width=1024,
        proxy_height=1024,
        encoder="h264_nvenc",
        hwaccel="cuda",
        scale_flags="bilinear",
        overwrite=False,
    )

    assert command[command.index("-hwaccel") + 1] == "cuda"
    assert command.index("-hwaccel") < command.index("-i")
    assert command[command.index("-vf") + 1] == "scale=1024:1024:flags=bilinear"


def test_resolve_review_proxy_encoder_prefers_libx264_then_nvenc() -> None:
    assert resolve_review_proxy_encoder("auto", runner=_fake_ffmpeg_encoders("h264_nvenc", "libx264")) == "libx264"
    assert resolve_review_proxy_encoder("auto", runner=_fake_ffmpeg_encoders("h264_nvenc")) == "h264_nvenc"
    assert resolve_review_proxy_encoder("h264_nvenc", runner=_fake_ffmpeg_encoders()) == "h264_nvenc"


def test_build_review_proxy_manifest_plans_clipped_proxy(tmp_path: Path) -> None:
    _write_clip_index(tmp_path)
    output_dir = tmp_path / "derived" / "review_proxy" / "video_detect" / "proxy_a"

    manifest = build_review_proxy_manifest(
        tmp_path,
        options=ReviewProxyOptions(output_dir=output_dir, proxy_run_id="proxy_a"),
        ffmpeg_runner=_fake_ffmpeg_encoders("h264_nvenc"),
        ffprobe_runner=_fake_ffprobe,
    )

    assert manifest["schema_version"] == "palette.review_proxy.video.v1"
    assert manifest["status"] == "planned"
    assert manifest["recording_id"] == "rec_a"
    assert manifest["encoder"] == "auto"
    assert manifest["resolved_encoder"] == "h264_nvenc"
    assert manifest["hwaccel"] is None
    assert manifest["scale_flags"] == "lanczos"
    assert manifest["clip_count"] == 1
    clip = manifest["clips"][0]
    assert clip["clip_id"] == "clip_000000"
    assert clip["clip_index"] == 0
    assert clip["camera_serial"] == "2010093"
    assert clip["source_width"] == 4512
    assert clip["source_height"] == 4512
    assert clip["proxy_width"] == 1024
    assert clip["proxy_height"] == 1024
    assert clip["frame_count"] == 120
    assert clip["fps"] == 30.0
    assert clip["encoder"] == "h264_nvenc"
    assert clip["hwaccel"] is None
    assert clip["scale_flags"] == "lanczos"
    assert clip["proxy_video_path"].endswith(
        "derived/review_proxy/video_detect/proxy_a/clips/clip_000000/Cam2010093_1024x1024_h264.mp4"
    )
    assert "-movflags +faststart" in clip["ffmpeg_command"]


def test_build_review_proxy_videos_apply_writes_manifest_and_proxy(tmp_path: Path) -> None:
    _write_clip_index(tmp_path)
    output_dir = tmp_path / "derived" / "review_proxy" / "video_detect" / "proxy_a"

    def fake_ffmpeg(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if "-encoders" in command:
            return _fake_ffmpeg_encoders("h264_nvenc")(command)
        Path(command[-1]).write_bytes(b"proxy video")
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    result = build_review_proxy_videos(
        tmp_path,
        options=ReviewProxyOptions(output_dir=output_dir, proxy_run_id="proxy_a", apply=True),
        ffmpeg_runner=fake_ffmpeg,
        ffprobe_runner=_fake_ffprobe,
    )

    assert result["status"] == "ok"
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proxy_path = Path(manifest["clips"][0]["proxy_video_path"])
    assert proxy_path.exists()
    assert proxy_path.read_bytes() == b"proxy video"
    assert manifest["video_outputs"][0]["bytes"] == len(b"proxy video")


def test_build_review_proxy_videos_defer_manifest_writes_proxy_without_final_manifest(tmp_path: Path) -> None:
    _write_clip_index(tmp_path)
    output_dir = tmp_path / "derived" / "review_proxy" / "video_detect" / "proxy_a"

    def fake_ffmpeg(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if "-encoders" in command:
            return _fake_ffmpeg_encoders("h264_nvenc")(command)
        Path(command[-1]).write_bytes(b"proxy video")
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    result = build_review_proxy_videos(
        tmp_path,
        options=ReviewProxyOptions(output_dir=output_dir, proxy_run_id="proxy_a", apply=True, defer_manifest=True),
        ffmpeg_runner=fake_ffmpeg,
        ffprobe_runner=_fake_ffprobe,
    )

    assert result["status"] == "ok"
    assert result["manifest_deferred"] is True
    assert result["manifest_written"] is False
    assert not Path(result["manifest_path"]).exists()
    proxy_path = output_dir / "clips" / "clip_000000" / "Cam2010093_1024x1024_h264.mp4"
    assert proxy_path.read_bytes() == b"proxy video"


def test_build_review_proxy_videos_write_manifest_only_requires_existing_proxy(tmp_path: Path) -> None:
    _write_clip_index(tmp_path)
    output_dir = tmp_path / "derived" / "review_proxy" / "video_detect" / "proxy_a"
    proxy_path = output_dir / "clips" / "clip_000000" / "Cam2010093_1024x1024_h264.mp4"
    proxy_path.parent.mkdir(parents=True)
    proxy_path.write_bytes(b"existing proxy")

    result = build_review_proxy_videos(
        tmp_path,
        options=ReviewProxyOptions(
            output_dir=output_dir,
            proxy_run_id="proxy_a",
            write_manifest_only=True,
            require_existing_proxies=True,
        ),
        ffmpeg_runner=_fake_ffmpeg_encoders("h264_nvenc"),
        ffprobe_runner=_fake_ffprobe,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["manifest_written"] is True
    assert manifest["video_outputs"][0]["status"] == "existing"
    assert manifest["video_outputs"][0]["bytes"] == len(b"existing proxy")


def test_build_review_proxy_videos_skip_existing_valid_proxy(tmp_path: Path) -> None:
    _write_clip_index(tmp_path)
    output_dir = tmp_path / "derived" / "review_proxy" / "video_detect" / "proxy_a"
    proxy_path = output_dir / "clips" / "clip_000000" / "Cam2010093_1024x1024_h264.mp4"
    proxy_path.parent.mkdir(parents=True)
    proxy_path.write_bytes(b"existing proxy")

    def fake_ffmpeg(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if "-encoders" in command:
            return _fake_ffmpeg_encoders("h264_nvenc")(command)
        raise AssertionError("ffmpeg should not run when skip-existing-valid is active")

    result = build_review_proxy_videos(
        tmp_path,
        options=ReviewProxyOptions(
            output_dir=output_dir,
            proxy_run_id="proxy_a",
            apply=True,
            skip_existing_valid=True,
        ),
        ffmpeg_runner=fake_ffmpeg,
        ffprobe_runner=_fake_ffprobe,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["video_outputs"][0]["status"] == "skipped_existing"
