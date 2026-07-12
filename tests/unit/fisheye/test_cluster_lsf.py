from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisheye.cluster.lsf import (
    LsfDependency,
    LsfDependencyCondition,
    LsfJob,
    LsfResources,
    LsfWorkflow,
    build_bsub_command,
    parse_bsub_job_id,
    render_dependency,
    resolve_job_id_placeholders,
    run_command,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.cluster.lsf.runtime import (
    build_runtime_command,
    expand_runtime_tokens,
    run_with_status,
)


def _job(
    job_key: str,
    *,
    dependency_keys: tuple[str, ...] = (),
) -> LsfJob:
    return LsfJob(
        job_key=job_key,
        job_name=job_key.replace(":", "_"),
        command=("bash", f"{job_key}.sh"),
        resources=LsfResources(queue="short", ncores=1, mem_gb=8),
        stdout_path=Path(f"/logs/{job_key}.%J.out"),
        stderr_path=Path(f"/logs/{job_key}.%J.err"),
        dependency=(LsfDependency(dependency_keys) if dependency_keys else None),
    )


def test_lsf_resources_validate_scheduler_values() -> None:
    resources = LsfResources(
        queue="gpu_l4",
        ncores=4,
        mem_gb=32,
        gpus=1,
        walltime="2:00",
        extra_lsf_args=("-R", "span[hosts=1]"),
    )

    assert resources.to_json() == {
        "queue": "gpu_l4",
        "ncores": 4,
        "mem_gb": 32,
        "gpus": 1,
        "walltime": "2:00",
        "extra_lsf_args": ["-R", "span[hosts=1]"],
    }
    with pytest.raises(ValueError, match="ncores"):
        LsfResources(queue="short", ncores=0, mem_gb=8)
    with pytest.raises(ValueError, match="mem_gb"):
        LsfResources(queue="short", ncores=1, mem_gb=0)
    with pytest.raises(ValueError, match="gpus"):
        LsfResources(queue="short", ncores=1, mem_gb=8, gpus=-1)


def test_dependency_rendering_is_structured_and_fail_closed() -> None:
    dependency = LsfDependency(("predict:a", "predict:b"))

    assert render_dependency(dependency) == (
        "done(<jobid:predict:a>) && done(<jobid:predict:b>)"
    )
    assert render_dependency(
        dependency,
        {"predict:a": "101", "predict:b": "102"},
    ) == "done(101) && done(102)"
    with pytest.raises(ValueError, match="predict:b"):
        render_dependency(dependency, {"predict:a": "101"})

    cleanup_dependency = LsfDependency(
        ("predict:a",),
        condition=LsfDependencyCondition.ALL_ENDED,
    )
    assert render_dependency(cleanup_dependency) == "ended(<jobid:predict:a>)"


def test_build_bsub_command_renders_resources_dependency_and_argv() -> None:
    job = LsfJob(
        job_key="refine:a",
        job_name="refine_a",
        command=("bash", "-lc", "scripts/py -m example --apply"),
        resources=LsfResources(
            queue="short",
            ncores=4,
            mem_gb=16,
            walltime="1:00",
        ),
        stdout_path=Path("/logs/refine_a.%J.out"),
        stderr_path=Path("/logs/refine_a.%J.err"),
        dependency=LsfDependency(("predict:a",)),
    )

    assert build_bsub_command(job) == [
        "bsub",
        "-J",
        "refine_a",
        "-n",
        "4",
        "-W",
        "1:00",
        "-R",
        "rusage[mem=16G]",
        "-oo",
        "/logs/refine_a.%J.out",
        "-eo",
        "/logs/refine_a.%J.err",
        "-q",
        "short",
        "-w",
        "done(<jobid:predict:a>)",
        "bash",
        "-lc",
        "scripts/py -m example --apply",
    ]
    resolved = build_bsub_command(job, {"predict:a": "123"})
    assert resolved[resolved.index("-w") + 1] == "done(123)"


def test_job_id_parsing_and_placeholder_resolution() -> None:
    assert parse_bsub_job_id("Job <123> is submitted to queue <short>.") == "123"
    assert parse_bsub_job_id("", "Job <456> is submitted to queue <gpu_l4>.") == "456"
    with pytest.raises(ValueError, match="Could not parse"):
        parse_bsub_job_id("submission rejected")

    command = ["bsub", "-w", "done(<jobid:a>) && done(<jobid:b>)"]
    assert resolve_job_id_placeholders(command, {"a": "11", "b": "12"}) == [
        "bsub",
        "-w",
        "done(11) && done(12)",
    ]
    with pytest.raises(ValueError, match="Unresolved"):
        resolve_job_id_placeholders(command, {"a": "11"})


def test_run_command_retains_diagnostics_and_raises_on_failure(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def success_runner(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="note")

    result = run_command(["bsub", "job.sh"], cwd=tmp_path, runner=success_runner)

    assert result == {
        "command": ["bsub", "job.sh"],
        "cwd": str(tmp_path),
        "returncode": 0,
        "stdout": "ok",
        "stderr": "note",
    }
    assert calls[0][1] == {
        "cwd": str(tmp_path),
        "text": True,
        "capture_output": True,
    }

    def failure_runner(argv, **kwargs):
        del argv, kwargs
        return SimpleNamespace(returncode=2, stdout="out", stderr="bad")

    with pytest.raises(RuntimeError, match="exit code 2"):
        run_command(["bsub", "job.sh"], cwd=tmp_path, runner=failure_runner)


def test_write_json_snapshot_is_normalized_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "bundle" / "submission.json"

    normalized = write_json_snapshot(
        path,
        {"path": tmp_path / "artifact", "status": "submitting"},
    )

    assert normalized == {
        "path": str(tmp_path / "artifact"),
        "status": "submitting",
    }
    assert json.loads(path.read_text(encoding="utf-8")) == normalized
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_lsf_workflow_validates_and_orders_dependency_graph() -> None:
    workflow = LsfWorkflow(
        workflow_id="workflow_a",
        family="keypoints.test",
        jobs=(
            _job("final", dependency_keys=("shard:a", "shard:b")),
            _job("shard:b", dependency_keys=("source",)),
            _job("shard:a", dependency_keys=("source",)),
            _job("source"),
        ),
    )

    assert [job.job_key for job in workflow.topological_jobs()] == [
        "source",
        "shard:b",
        "shard:a",
        "final",
    ]
    assert workflow.to_json()["submission_order"] == [
        "source",
        "shard:b",
        "shard:a",
        "final",
    ]

    with pytest.raises(ValueError, match="unknown job key"):
        LsfWorkflow(
            workflow_id="unknown_dependency",
            family="test",
            jobs=(_job("child", dependency_keys=("missing",)),),
        )
    with pytest.raises(ValueError, match="cycle"):
        LsfWorkflow(
            workflow_id="cycle",
            family="test",
            jobs=(
                _job("a", dependency_keys=("b",)),
                _job("b", dependency_keys=("a",)),
            ),
        )


def test_submit_lsf_workflow_records_incremental_dependency_state(tmp_path: Path) -> None:
    workflow = LsfWorkflow(
        workflow_id="workflow_a",
        family="keypoints.test",
        jobs=(
            _job("refine", dependency_keys=("predict",)),
            _job("predict"),
        ),
    )
    calls: list[list[str]] = []
    outputs = iter(
        [
            "Job <101> is submitted to queue <short>.",
            "Job <102> is submitted to queue <short>.",
        ]
    )

    def fake_runner(argv, **kwargs):
        del kwargs
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr=next(outputs))

    submitted_records: list[dict[str, object]] = []
    plan_path = tmp_path / "lsf_plan.json"
    submission_path = tmp_path / "lsf_submission.json"
    result = submit_lsf_workflow(
        workflow,
        cwd=tmp_path,
        plan_path=plan_path,
        submission_path=submission_path,
        runner=fake_runner,
        on_job_submitted=lambda record: submitted_records.append(dict(record)),
    )

    assert [call[call.index("-J") + 1] for call in calls] == ["predict", "refine"]
    assert calls[1][calls[1].index("-w") + 1] == "done(101)"
    assert result["status"] == "submitted"
    assert result["job_ids_by_key"] == {"predict": "101", "refine": "102"}
    assert [record["job_key"] for record in submitted_records] == [
        "predict",
        "refine",
    ]
    assert json.loads(plan_path.read_text(encoding="utf-8"))["schema"] == (
        "palette.lsf_workflow.v1"
    )
    saved = json.loads(submission_path.read_text(encoding="utf-8"))
    assert saved["schema"] == "palette.lsf_workflow_submission.v1"
    assert saved["status"] == "submitted"
    assert saved["jobs"][1]["dependency"] == "done(101)"


def test_submit_lsf_workflow_preserves_partial_state_on_parse_failure(
    tmp_path: Path,
) -> None:
    workflow = LsfWorkflow(
        workflow_id="workflow_failure",
        family="keypoints.test",
        jobs=(
            _job("predict"),
            _job("refine", dependency_keys=("predict",)),
        ),
    )
    outputs = iter(
        [
            "Job <201> is submitted to queue <short>.",
            "submission accepted without a parseable id",
        ]
    )

    def fake_runner(argv, **kwargs):
        del argv, kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr=next(outputs))

    submission_path = tmp_path / "lsf_submission.json"
    with pytest.raises(ValueError, match="Could not parse"):
        submit_lsf_workflow(
            workflow,
            cwd=tmp_path,
            plan_path=tmp_path / "lsf_plan.json",
            submission_path=submission_path,
            runner=fake_runner,
        )

    saved = json.loads(submission_path.read_text(encoding="utf-8"))
    assert saved["status"] == "submission_failed"
    assert saved["job_ids_by_key"] == {"predict": "201"}
    assert [record["job_key"] for record in saved["jobs"]] == ["predict"]
    assert saved["failed_job"]["job_key"] == "refine"
    assert saved["failed_job"]["command_result"]["returncode"] == 0


def test_runtime_command_uses_structured_argv_and_scheduler_tokens(tmp_path: Path) -> None:
    command = build_runtime_command(
        ("scripts/py", "-m", "example.worker", "--cache", "/scratch/<user>/<jobid>/cache"),
        status_path_template=tmp_path / "status" / "predict.<jobid>.json",
        workflow_id="workflow_a",
        family="keypoints.whole_recording",
        job_key="predict:recording_a",
        stage="keypoint_prediction",
        cwd=Path("/groups/repo"),
        environment_overrides={"PALETTE_DISABLE_REGISTRY_WRITES": "1"},
        cleanup_path_templates=("/scratch/<user>/<jobid>/cache",),
        expected_output_templates=(tmp_path / "outputs" / "run",),
    )

    assert command[:3] == ("scripts/py", "-m", "fisheye.cluster.lsf.runtime")
    assert command[command.index("--status-json") + 1].endswith(
        "status/predict.<jobid>.json"
    )
    assert "PALETTE_DISABLE_REGISTRY_WRITES=1" in command
    assert str(tmp_path / "outputs" / "run") in command
    separator = command.index("--")
    assert command[separator + 1 :] == (
        "scripts/py",
        "-m",
        "example.worker",
        "--cache",
        "/scratch/<user>/<jobid>/cache",
    )
    assert expand_runtime_tokens(
        "/scratch/<user>/<jobid>/<jobindex>",
        {"USER": "jeremy", "LSB_JOBID": "123", "LSB_JOBINDEX": "4"},
    ) == "/scratch/jeremy/123/4"


def test_runtime_status_records_success_without_exposing_environment_values(
    tmp_path: Path,
) -> None:
    status_template = tmp_path / "status" / "job.<jobid>.json"

    expected_output = tmp_path / "output"
    expected_output.mkdir()
    returncode = run_with_status(
        ("/bin/sh", "-c", "exit 0"),
        status_path_template=status_template,
        workflow_id="workflow_a",
        family="keypoints.whole_recording",
        job_key="predict:a",
        stage="keypoint_prediction",
        cwd=tmp_path,
        environment_overrides={"PALETTE_DISABLE_REGISTRY_WRITES": "secret-value"},
        expected_output_templates=(str(expected_output),),
        base_environment={"USER": "tester", "LSB_JOBID": "321", "PATH": "/usr/bin:/bin"},
    )

    assert returncode == 0
    saved = json.loads((tmp_path / "status" / "job.321.json").read_text(encoding="utf-8"))
    assert saved["schema"] == "palette.lsf_job_runtime_status.v1"
    assert saved["status"] == "succeeded"
    assert saved["returncode"] == 0
    assert saved["scheduler"]["job_id"] == "321"
    assert saved["environment_override_keys"] == ["PALETTE_DISABLE_REGISTRY_WRITES"]
    assert saved["expected_outputs"] == [
        {
            "exists": True,
            "expanded_path": str(expected_output),
            "requested_path": str(expected_output),
        }
    ]
    assert "secret-value" not in json.dumps(saved)


def test_runtime_status_propagates_worker_failure_and_refuses_unsafe_cleanup(
    tmp_path: Path,
) -> None:
    status_template = tmp_path / "status" / "job.<jobid>.json"

    returncode = run_with_status(
        ("/bin/sh", "-c", "exit 7"),
        status_path_template=status_template,
        workflow_id="workflow_a",
        family="keypoints.whole_recording",
        job_key="refine:a",
        stage="keypoint_refinement",
        cwd=tmp_path,
        cleanup_path_templates=(str(tmp_path / "not-job-scratch"),),
        base_environment={"USER": "tester", "LSB_JOBID": "654", "PATH": "/usr/bin:/bin"},
    )

    assert returncode == 7
    saved = json.loads((tmp_path / "status" / "job.654.json").read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["returncode"] == 7
    assert saved["cleanup"] == [
        {
            "allowed_root": "/scratch/tester/654",
            "expanded_path": str(tmp_path / "not-job-scratch"),
            "reason": "cleanup path is not a child of this LSF job scratch root",
            "requested_path": str(tmp_path / "not-job-scratch"),
            "status": "refused",
        }
    ]


def test_runtime_turns_missing_expected_output_into_failure(tmp_path: Path) -> None:
    returncode = run_with_status(
        ("/bin/sh", "-c", "exit 0"),
        status_path_template=tmp_path / "status.json",
        workflow_id="workflow_a",
        family="test",
        job_key="job:a",
        stage="test",
        cwd=tmp_path,
        expected_output_templates=(str(tmp_path / "missing"),),
        base_environment={"USER": "tester", "LSB_JOBID": "777", "PATH": "/usr/bin:/bin"},
    )

    assert returncode == 1
    saved = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["expected_outputs"][0]["exists"] is False
    assert "expected outputs are missing" in saved["error"]
