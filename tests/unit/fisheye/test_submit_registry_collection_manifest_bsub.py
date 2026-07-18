from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _base_args(repo: Path, tmp_path: Path, collection_id: str) -> list[str]:
    registry = tmp_path / "registry.sqlite"
    registry.touch()
    return [
        "bash",
        str(repo / "scripts" / "submit_registry_collection_manifest_bsub.sh"),
        "--collection-id",
        collection_id,
        "--collection-name",
        "All normalized chaser recordings",
        "--stimulus-mode",
        "CHASER",
        "--registry",
        str(registry),
        "--output-root",
        str(tmp_path / "palette_analytics"),
        "--palette-repo",
        str(repo),
    ]


def test_submitter_renders_cluster_manifest_job(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    collection_id = "all_chaser_test_v001"
    result = subprocess.run(
        _base_args(repo, tmp_path, collection_id),
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mode=render-only" in result.stdout
    assert "stimulus_mode=CHASER" in result.stdout
    run_dir = (
        tmp_path
        / "palette_analytics"
        / "logs"
        / "lsf"
        / f"collection_manifest_{collection_id}"
    )
    job = (run_dir / "run_collection_manifest.sh").read_text(encoding="utf-8")
    assert "fisheye.utils.build_virtual_collection_manifest" in job
    assert "fisheye.utils.virtual_collection_manifest" in job
    assert "--check-hash" in job
    assert "Palette commit mismatch" in job


def test_submitter_rejects_unsafe_collection_id(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        _base_args(repo, tmp_path, "../unsafe"),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "unsafe --collection-id" in result.stderr


def test_submitter_snapshots_exact_zarr_list_selection(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    zarr_list = tmp_path / "selected_zarrs.txt"
    zarr_list.write_text(
        "/groups/recordings/one_analysis.zarr\n"
        "# retained comment in immutable source snapshot\n"
        "/groups/recordings/two_analysis.zarr\n",
        encoding="utf-8",
    )
    collection_id = "selected_chaser_test_v001"
    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "submit_registry_collection_manifest_bsub.sh"),
            "--collection-id",
            collection_id,
            "--collection-name",
            "Selected normalized chaser recordings",
            "--zarr-list",
            str(zarr_list),
            "--profile",
            "chaser",
            "--output-root",
            str(tmp_path / "palette_analytics"),
            "--palette-repo",
            str(repo),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "source_mode=zarr_list" in result.stdout
    run_dir = (
        tmp_path
        / "palette_analytics"
        / "logs"
        / "lsf"
        / f"collection_manifest_{collection_id}"
    )
    snapshot = run_dir / "zarr_paths.txt"
    assert snapshot.read_text(encoding="utf-8") == zarr_list.read_text(
        encoding="utf-8"
    )
    job = (run_dir / "run_collection_manifest.sh").read_text(encoding="utf-8")
    assert "mapfile -t zarr_paths" in job
    assert 'cmd+=("${zarr_paths[@]}")' in job
    assert "zarr_list_sha256" in job


def test_submitter_requires_exactly_one_collection_source(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    zarr_list = tmp_path / "selected_zarrs.txt"
    zarr_list.write_text("/groups/recordings/one_analysis.zarr\n", encoding="utf-8")
    args = _base_args(repo, tmp_path, "ambiguous_source_v001")
    args.extend(["--zarr-list", str(zarr_list)])

    result = subprocess.run(
        args,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "select exactly one source" in result.stderr


def test_submitter_sshes_only_bsub_command(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
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
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["FAKE_SSH_ARGS"] = str(ssh_args)
    args = _base_args(repo, tmp_path, "all_chaser_ssh_test_v001")
    args.extend(["--submit-host", "login1-citrus-poller", "--submit"])

    result = subprocess.run(
        args,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "job_id=123456" in result.stdout
    captured = ssh_args.read_text(encoding="utf-8").splitlines()
    assert captured[0] == "login1-citrus-poller"
    assert captured[1].startswith("bsub ")
    assert "run_collection_manifest.sh" in captured[1]
    assert "build_virtual_collection_manifest" not in captured[1]
