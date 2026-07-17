from __future__ import annotations

from pathlib import Path
import subprocess


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "submit_eye_angle_materialization_layout_benchmark_bsub.sh"
)


def _build_clean_palette_checkout(path: Path) -> None:
    scripts_dir = path / "scripts"
    scripts_dir.mkdir(parents=True)
    scripts_py = scripts_dir / "py"
    scripts_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    scripts_py.chmod(0o755)
    module = (
        path
        / "src"
        / "fisheye"
        / "diagnostics"
        / "benchmark_eye_angle_materialization_layout.py"
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


def test_submit_eye_angle_layout_benchmark_renders_single_host_abba_job(
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
            "--subject-shape-run",
            "shape_001",
            "--keypoint-run",
            "keypoints_001",
            "--benchmark-id",
            "eye_layout_abba_001",
            "--palette-repo",
            str(palette_repo),
            "--log-dir",
            str(log_dir),
            "--queue",
            "short",
            "--ncores",
            "8",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mode=render-only" in result.stdout
    assert "requested_queue=short" in result.stdout
    assert "order=all_columns,semantic_16,semantic_16,all_columns" in result.stdout
    bsub_line = next(
        line for line in result.stdout.splitlines() if line.startswith("bsub_command=")
    )
    assert " -n 8 " in bsub_line
    assert " -q short " in bsub_line
    assert "span\\[hosts=1\\]" in bsub_line

    run_dir = log_dir / "eye_layout_abba_001"
    job_script = run_dir / "run_eye_angle_materialization_layout_benchmark.sh"
    subprocess.run(["bash", "-n", str(job_script)], check=True)
    text = job_script.read_text(encoding="utf-8")
    assert "benchmark_eye_angle_materialization_layout" in text
    assert '--order "${ORDER}"' in text
    assert '--output-root "${scratch_root}"' in text
    assert "Refusing execution outside LSF" in text
