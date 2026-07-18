from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_submit_analytics_export_bsub_renders_fail_closed_job(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    collection = tmp_path / "collection.manifest.json"
    collection.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "shared" / "palette_analytics"
    log_dir = tmp_path / "logs"
    run_id = "chaser_v2_test_20260712T000000Z"

    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "submit_analytics_export_bsub.sh"),
            "--collection-manifest",
            str(collection),
            "--export-run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--palette-repo",
            str(repo),
            "--log-dir",
            str(log_dir),
            "--queue",
            "short",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mode=render-only" in result.stdout
    assert f"export_run_id={run_id}" in result.stdout
    assert "bsub_command=" in result.stdout
    run_dir = log_dir / f"analytics_export_{run_id}"
    job_script = run_dir / "run_analytics_export.sh"
    text = job_script.read_text(encoding="utf-8")
    assert "fisheye.utils.export_cross_recording_analytics" in text
    assert "fisheye.utils.validate_analytics_export" in text
    assert "fisheye.utils.compute_group_statistics" in text
    assert "Palette commit mismatch" in text
    assert str(output_root) in text
    assert "chaser_quadrant_occupancy_summary" in text
    assert "chaser_near_field_occupancy_summary" in text
    assert "chaser_cra_" not in text


def test_submit_analytics_export_bsub_rejects_unsafe_run_id(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    collection = tmp_path / "collection.manifest.json"
    collection.write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "submit_analytics_export_bsub.sh"),
            "--collection-manifest",
            str(collection),
            "--export-run-id",
            "../unsafe",
            "--palette-repo",
            str(repo),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "Unsafe --export-run-id" in result.stderr


def test_submit_analytics_export_bsub_uses_cluster_default_queue(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    collection = tmp_path / "collection.manifest.json"
    collection.write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "submit_analytics_export_bsub.sh"),
            "--collection-manifest",
            str(collection),
            "--export-run-id",
            "chaser_v2_default_queue_test",
            "--output-root",
            str(tmp_path / "shared" / "palette_analytics"),
            "--palette-repo",
            str(repo),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    bsub_line = next(
        line for line in result.stdout.splitlines() if line.startswith("bsub_command=")
    )
    assert " -q " not in bsub_line


def test_submit_analytics_export_bsub_sshes_only_bsub_command(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    collection = tmp_path / "collection.manifest.json"
    collection.write_text("{}\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ssh_args = tmp_path / "ssh_args.txt"
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_SSH_ARGS\"\n"
        "printf 'Job <123456> is submitted to queue <short>.\\n'\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    output_root = tmp_path / "shared" / "palette_analytics"
    log_dir = tmp_path / "logs"
    run_id = "chaser_v2_ssh_submit_test"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["FAKE_SSH_ARGS"] = str(ssh_args)

    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "submit_analytics_export_bsub.sh"),
            "--collection-manifest",
            str(collection),
            "--export-run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--palette-repo",
            str(repo),
            "--log-dir",
            str(log_dir),
            "--queue",
            "short",
            "--submit-host",
            "login1-citrus-poller",
            "--submit",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "job_id=123456" in result.stdout
    args = ssh_args.read_text(encoding="utf-8").splitlines()
    assert args[0] == "login1-citrus-poller"
    remote_command = args[1]
    assert remote_command.startswith("bsub ")
    assert "run_analytics_export.sh" in remote_command
    assert "export_cross_recording_analytics" not in remote_command
    assert "validate_analytics_export" not in remote_command
    submission = (
        log_dir / f"analytics_export_{run_id}" / "submission.txt"
    ).read_text(encoding="utf-8")
    assert "submit_mode=ssh_bsub" in submission
    assert "job_id=123456" in submission
