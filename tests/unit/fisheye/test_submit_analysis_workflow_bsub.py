from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "submit_analysis_workflow_bsub.sh"
)


def _build_clean_palette_checkout(path: Path) -> None:
    scripts_dir = path / "scripts"
    scripts_dir.mkdir(parents=True)
    scripts_py = scripts_dir / "py"
    scripts_py.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    scripts_py.chmod(0o755)
    package = path / "src" / "fisheye" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("# fixture package\n", encoding="utf-8")
    module = path / "src" / "fisheye" / "utils" / "execute_analysis_workflow.py"
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


def test_submit_analysis_workflow_records_requested_and_runtime_resources(
    tmp_path: Path,
) -> None:
    palette_repo = tmp_path / "palette-checkout"
    _build_clean_palette_checkout(palette_repo)
    zarr_path = tmp_path / "recording" / "zarr" / "analysis.zarr"
    zarr_path.mkdir(parents=True)
    (zarr_path / "zarr.json").write_text("{}\n", encoding="utf-8")
    log_dir = tmp_path / "logs"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bsub = fake_bin / "bsub"
    fake_bsub.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'Job <123456> is submitted to queue <short>.\\n'\n",
        encoding="utf-8",
    )
    fake_bsub.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--zarr",
            str(zarr_path),
            "--execution-id",
            "runtime_provenance_test",
            "--target",
            "eye_angles",
            "--palette-repo",
            str(palette_repo),
            "--log-dir",
            str(log_dir),
            "--queue",
            "short",
            "--ncores",
            "5",
            "--mem-gb",
            "7",
            "--walltime",
            "00:30",
            "--submit",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "job_id=123456" in result.stdout
    run_dir = log_dir / "runtime_provenance_test_analysis.zarr_"
    submission = (run_dir / "submission.txt").read_text(encoding="utf-8")
    assert "requested_queue=short" in submission
    assert "requested_ncores=5" in submission
    assert "requested_mem_gb_per_slot=7" in submission
    assert "requested_walltime=00:30" in submission
    assert f"runtime_environment={run_dir / 'runtime_environment.txt'}" in submission

    job_script = run_dir / "run_analysis_workflow.sh"
    subprocess.run(["bash", "-n", str(job_script)], check=True)
    job_env = dict(env)
    job_env.update(
        {
            "LSB_JOBID": "123456",
            "LSB_QUEUE": "short",
            "LSB_HOSTS": "benchmark-host benchmark-host benchmark-host",
            "LSB_DJOB_NUMPROC": "5",
        }
    )
    rogue_package = tmp_path / "rogue" / "fisheye"
    rogue_package.mkdir(parents=True)
    (rogue_package / "__init__.py").write_text(
        "# wrong checkout\n",
        encoding="utf-8",
    )
    job_env["PYTHONPATH"] = str(rogue_package.parent)
    subprocess.run(["bash", str(job_script)], check=True, env=job_env)

    runtime = (run_dir / "runtime_environment.txt").read_text(encoding="utf-8")
    assert "schema_id=palette.analysis_workflow_runtime_environment.v1" in runtime
    assert "requested_queue=short" in runtime
    assert "effective_queue=short" in runtime
    assert "lsf_execution_hosts=benchmark-host benchmark-host benchmark-host" in runtime
    assert "requested_ncores=5" in runtime
    assert "allocated_slots=5" in runtime
    assert "cpu_model=" in runtime
    assert "cpu_model=unknown" not in runtime
    assert f"fisheye_source_file={palette_repo / 'src/fisheye/__init__.py'}" in runtime

    status = (run_dir / "status.txt").read_text(encoding="utf-8")
    assert "status=complete" in status
    assert "requested_queue=short" in status
    assert "effective_queue=short" in status
    assert "cpu_model=" in status
    assert "allocated_slots=5" in status


def test_submit_analysis_workflow_labels_unspecified_queue_as_cluster_default(
    tmp_path: Path,
) -> None:
    palette_repo = tmp_path / "palette-checkout"
    _build_clean_palette_checkout(palette_repo)
    zarr_path = tmp_path / "recording" / "zarr" / "analysis.zarr"
    zarr_path.mkdir(parents=True)
    (zarr_path / "zarr.json").write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--zarr",
            str(zarr_path),
            "--execution-id",
            "default_queue_test",
            "--target",
            "eye_angles",
            "--palette-repo",
            str(palette_repo),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "requested_queue=<cluster-default>" in result.stdout
    bsub_line = next(
        line for line in result.stdout.splitlines() if line.startswith("bsub_command=")
    )
    assert " -q " not in bsub_line
