from __future__ import annotations

from pathlib import Path
import subprocess


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "submit_tail_kinematics_materialization_bsub.sh"
)


def _build_clean_palette_checkout(path: Path) -> None:
    (path / "scripts").mkdir(parents=True)
    scripts_py = path / "scripts" / "py"
    scripts_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    scripts_py.chmod(0o755)
    module = (
        path
        / "src"
        / "fisheye"
        / "analysis_workflows"
        / "materializers"
        / "tail_kinematics.py"
    )
    module.parent.mkdir(parents=True)
    module.write_text("# fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Palette Tests",
            "-c",
            "user.email=palette-tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )


def test_submit_tail_kinematics_materialization_renders_fail_closed_job(
    tmp_path: Path,
) -> None:
    palette_repo = tmp_path / "palette-checkout"
    _build_clean_palette_checkout(palette_repo)
    zarr_path = tmp_path / "recording" / "zarr" / "analysis.zarr"
    zarr_path.mkdir(parents=True)
    (zarr_path / "zarr.json").write_text("{}\n", encoding="utf-8")
    log_dir = tmp_path / "logs"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--zarr",
            str(zarr_path),
            "--shape-run",
            "subject_shape_001",
            "--run-name",
            "tail_kinematics_001",
            "--palette-repo",
            str(palette_repo),
            "--log-dir",
            str(log_dir),
            "--scratch-base",
            "/scratch/palette-test",
            "--queue",
            "short",
            "--ncores",
            "3",
            "--output-shard-rows",
            "131072",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "mode=render-only" in result.stdout
    assert "shape_run=subject_shape_001" in result.stdout
    assert "run_name=tail_kinematics_001" in result.stdout
    assert "bsub_command=" in result.stdout
    assert " -n 3 " in result.stdout
    run_dirs = list(log_dir.glob("tail_kinematics_001_*"))
    assert len(run_dirs) == 1
    job_script = run_dirs[0] / "run_tail_kinematics_materialization.sh"
    text = job_script.read_text(encoding="utf-8")
    assert "Refusing tail-kinematics execution outside an LSF allocation" in text
    assert "fisheye.analysis_workflows.materializers.tail_kinematics" in text
    assert "--copy-backend rsync" in text
    assert "--execution-backend process_shards" in text
    assert '--output-shard-rows "${OUTPUT_SHARD_ROWS}"' in text
    assert "OUTPUT_SHARD_ROWS=131072" in text
    assert '--num-workers "${NCORES}"' in text
    assert "--apply" in text
    assert "/scratch/palette-test" in text
    assert "status_tmp=" in text
    subprocess.run(["bash", "-n", str(job_script)], check=True)


def test_submit_tail_kinematics_materialization_rejects_unsafe_run_name(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--zarr",
            str(tmp_path / "missing.zarr"),
            "--shape-run",
            "shape_001",
            "--run-name",
            "../unsafe",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "unsafe --run-name" in result.stderr
