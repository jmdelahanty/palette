from __future__ import annotations

import json
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).parents[3] / "scripts" / "submit_keypoints_batches_bsub.sh"


def _run_dry_run(tmp_path: Path, *storage_args: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    root = tmp_path / "recordings"
    (root / "rec" / "zarr" / "rec_analysis.zarr").mkdir(parents=True)
    log_dir = tmp_path / "logs"
    result = subprocess.run(
        [
            str(SCRIPT),
            "--root",
            str(root),
            "--source",
            "filesystem",
            "--batch-size",
            "1",
            "--log-dir",
            str(log_dir),
            "--run-id",
            "storage_test",
            "--dry-run",
            *storage_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=SCRIPT.parents[1],
    )
    summary = json.loads(
        (log_dir / "kp_storage_test" / "manifest_summary.json").read_text(
            encoding="utf-8"
        )
    )
    return result, summary


def test_batch_submitter_defaults_to_indexed_keypoint_shards(tmp_path: Path) -> None:
    result, summary = _run_dry_run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "--keypoint-roi-shard-rows 262144" in result.stdout
    assert "--keypoint-frame-shard-rows 262144" in result.stdout
    assert summary["keypoint_storage"]["effective"] == {
        "keypoint_storage_layout": "indexed_sharding_v1",
        "keypoint_storage_policy": "default_indexed_sharding_v1",
        "keypoint_roi_shard_rows": 262144,
        "keypoint_frame_shard_rows": 262144,
    }


def test_batch_submitter_forwards_regular_chunk_opt_out(tmp_path: Path) -> None:
    result, summary = _run_dry_run(tmp_path, "--no-keypoint-sharding")

    assert result.returncode == 0, result.stderr
    assert "--no-keypoint-sharding" in result.stdout
    assert "--keypoint-roi-shard-rows" not in result.stdout
    assert summary["keypoint_storage"]["effective"] == {
        "keypoint_storage_layout": "regular_chunks_v1",
        "keypoint_storage_policy": "explicit_regular_chunks_override",
        "keypoint_roi_shard_rows": None,
        "keypoint_frame_shard_rows": None,
    }


def test_batch_submitter_forwards_custom_shard_rows(tmp_path: Path) -> None:
    result, summary = _run_dry_run(
        tmp_path,
        "--keypoint-roi-shard-rows",
        "8192",
        "--keypoint-frame-shard-rows",
        "16384",
    )

    assert result.returncode == 0, result.stderr
    assert "--keypoint-roi-shard-rows 8192" in result.stdout
    assert "--keypoint-frame-shard-rows 16384" in result.stdout
    assert summary["keypoint_storage"]["effective"]["keypoint_roi_shard_rows"] == 8192
    assert summary["keypoint_storage"]["effective"]["keypoint_frame_shard_rows"] == 16384
