from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils.submit_review_proxy_videos_sharded_bsub import (
    _partition_clip_ids,
    submit_review_proxy_videos_sharded,
)


def _write_multi_clip_index(recording_dir: Path, *, clip_count: int = 5) -> None:
    clips = []
    for clip_index in range(clip_count):
        clip_id = f"clip_{clip_index:06d}"
        clip_dir = recording_dir / "clips" / clip_id
        clip_dir.mkdir(parents=True)
        video = clip_dir / f"Cam2010093_{clip_id}.mp4"
        video.write_bytes(b"source video")
        clips.append(
            {
                "clip_id": clip_id,
                "clip_index": clip_index,
                "camera_artifacts": [
                    {
                        "camera_serial": "2010093",
                        "video_path": f"clips/{clip_id}/{video.name}",
                        "frame_count": 120,
                        "fps": 30,
                        "source_width": 4512,
                        "source_height": 4512,
                    }
                ],
            }
        )
    (recording_dir / "recording_clip_index.json").write_text(
        json.dumps({"recording_id": "rec_a", "clips": clips}),
        encoding="utf-8",
    )


def test_partition_clip_ids_balances_shards() -> None:
    shards = _partition_clip_ids(
        [f"clip_{index:06d}" for index in range(5)],
        shard_count=2,
        clips_per_shard=None,
    )

    assert [shard.clip_ids for shard in shards] == [
        ("clip_000000", "clip_000001"),
        ("clip_000002", "clip_000003", "clip_000004"),
    ]


def test_sharded_submitter_dry_run_writes_shard_and_finalizer_scripts(tmp_path: Path) -> None:
    _write_multi_clip_index(tmp_path, clip_count=5)
    output_dir = tmp_path / "derived" / "review_proxy" / "video_detect" / "proxy_a"
    run_dir = tmp_path / "runs" / "proxy_shards"

    result = submit_review_proxy_videos_sharded(
        recording_dir=tmp_path,
        output_dir=output_dir,
        proxy_run_id="proxy_a",
        run_dir=run_dir,
        repo_path=Path.cwd(),
        proxy_width=1024,
        proxy_height=1024,
        encoder="h264_nvenc",
        preset="veryfast",
        crf=23,
        hwaccel="cuda",
        scale_flags="bilinear",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        no_probe=True,
        overwrite=True,
        skip_existing_valid=True,
        camera_serials=(),
        clip_ids=(),
        shard_count=2,
        clips_per_shard=None,
        max_active=2,
        queue="gpu_l4",
        ncores=4,
        mem_gb=32,
        gpus=1,
        walltime="2:00",
        finalizer_queue="short",
        finalizer_ncores=2,
        finalizer_mem_gb=8,
        finalizer_walltime="1:00",
        submit=False,
    )

    assert result["status"] == "planned"
    assert result["shard_count"] == 2
    assert result["max_active"] == 2
    assert "--defer-manifest" in result["shards"][0]["command"]
    assert "--apply" in result["shards"][0]["command"]
    assert "--write-manifest-only" in result["finalizer_command"]
    assert "--require-existing-proxies" in result["finalizer_command"]
    assert "done(<shard_jobid>)" in result["finalizer_bsub_command_template"]
    assert Path(result["array_runner"]).exists()
    assert Path(result["finalizer_script"]).exists()
