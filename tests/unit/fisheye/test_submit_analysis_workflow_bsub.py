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
    telemetry_module = (
        path
        / "src"
        / "fisheye"
        / "diagnostics"
        / "run_with_resource_telemetry.py"
    )
    telemetry_module.parent.mkdir(parents=True)
    telemetry_module.write_text(
        """from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--summary-json', type=Path, required=True)
parser.add_argument('--samples-jsonl', type=Path, required=True)
parser.add_argument('--stdout-log', type=Path, required=True)
parser.add_argument('--requested-workers', type=int, required=True)
parser.add_argument('--allocated-slots', type=int, required=True)
parser.add_argument('--sample-interval-seconds')
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
completed = subprocess.run(command, check=False, text=True, capture_output=True)
args.stdout_log.write_text(completed.stdout + completed.stderr, encoding='utf-8')
args.samples_jsonl.write_text('{}\\n', encoding='utf-8')
args.summary_json.write_text(
    json.dumps({'exit_code': completed.returncode}) + '\\n',
    encoding='utf-8',
)
raise SystemExit(completed.returncode)
""",
        encoding="utf-8",
    )
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
    assert (
        f"resource_telemetry_summary={run_dir / 'resource_telemetry_summary.json'}"
        in submission
    )

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
    assert (
        f"resource_telemetry_summary={run_dir / 'resource_telemetry_summary.json'}"
        in status
    )
    assert (run_dir / "resource_telemetry_summary.json").is_file()
    assert (run_dir / "resource_telemetry_samples.jsonl").is_file()
    assert (run_dir / "workflow_stdout.log").is_file()


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


def test_submit_analysis_workflow_accepts_git_worktree_checkout(
    tmp_path: Path,
) -> None:
    source_repo = tmp_path / "palette-source"
    _build_clean_palette_checkout(source_repo)
    palette_worktree = tmp_path / "palette-worktree"
    subprocess.run(
        [
            "git",
            "-C",
            str(source_repo),
            "worktree",
            "add",
            "--detach",
            str(palette_worktree),
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (palette_worktree / ".git").is_file()

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
            "worktree_checkout_test",
            "--target",
            "track_kinematics",
            "--palette-repo",
            str(palette_worktree),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mode=render-only" in result.stdout
    job_script = (
        tmp_path
        / "logs"
        / "worktree_checkout_test_analysis.zarr_"
        / "run_analysis_workflow.sh"
    ).read_text(encoding="utf-8")
    assert f"PALETTE_REPO={palette_worktree}" in job_script
