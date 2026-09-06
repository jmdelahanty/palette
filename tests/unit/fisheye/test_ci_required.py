from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import ci_required as gate


ROOT = Path(__file__).resolve().parents[3]
SHA = "a" * 40
MERGE_SHA = "b" * 40


@pytest.fixture
def context():
    return gate.Context("jmdelahanty/palette", 42, 1, SHA, MERGE_SHA, "pull_request")


@pytest.fixture
def run(context):
    return {
        "id": context.run_id,
        "run_attempt": context.attempt,
        "head_sha": context.head_sha,
        "event": context.event,
        "path": ".github/workflows/ci.yml",
        "repository": {"full_name": context.repository},
    }


@pytest.fixture
def jobs(context):
    return [
        {
            "id": index + 100,
            "run_id": context.run_id,
            "run_attempt": context.attempt,
            "head_sha": context.head_sha,
            "workflow_name": "CI",
            "name": name,
            "status": "completed",
            "conclusion": "success",
        }
        for index, name in enumerate(gate.required_names())
    ]


@pytest.fixture
def needs():
    return {name: {"result": "success"} for name in (*gate.SINGLE_JOBS, "tests")}


def test_exact_success_inventory_passes(context, run, jobs, needs):
    gate.validate_needs(needs)
    gate.validate_run(run, context)
    gate.validate_jobs(jobs, context)
    assert len(gate.required_names()) == 23


def test_push_context_requires_the_same_tested_commit():
    gate.Context("jmdelahanty/palette", 42, 1, SHA, SHA, "push")
    with pytest.raises(gate.GateError, match="Push head and tested commit"):
        gate.Context("jmdelahanty/palette", 42, 1, SHA, MERGE_SHA, "push")


@pytest.mark.parametrize("conclusion", [
    "failure", "skipped", "cancelled", "timed_out", "neutral", "stale",
    "action_required", "startup_failure", None, "",
])
@pytest.mark.parametrize("index", [0, 7, 22])
def test_non_success_is_blocking_even_when_needs_says_success(
    context, jobs, needs, conclusion, index,
):
    gate.validate_needs(needs)
    jobs[index]["conclusion"] = conclusion
    with pytest.raises(gate.GateError, match="not successful"):
        gate.validate_jobs(jobs, context)


@pytest.mark.parametrize("status", ["queued", "in_progress", "waiting", None])
def test_success_requires_completed_status(context, jobs, status):
    jobs[0]["status"] = status
    with pytest.raises(gate.GateError, match="not successful"):
        gate.validate_jobs(jobs, context)


@pytest.mark.parametrize("index", range(23))
def test_every_required_job_must_exist(context, jobs, index):
    del jobs[index]
    with pytest.raises(gate.GateError, match="Missing required jobs"):
        gate.validate_jobs(jobs, context)


def test_collapsed_skipped_matrix_does_not_count_as_sixteen_shards(context, jobs):
    jobs = jobs[:7] + [{**jobs[7], "name": "non-gpu tests (shard)", "conclusion": "skipped"}]
    with pytest.raises(gate.GateError):
        gate.validate_jobs(jobs, context)


@pytest.mark.parametrize("change", [
    {"run_id": 43}, {"run_attempt": 0}, {"run_attempt": 2},
    {"run_attempt": True}, {"run_id": "42"},
    {"head_sha": "c" * 40}, {"head_sha": MERGE_SHA},
    {"workflow_name": "unrelated"}, {"id": None},
])
def test_foreign_or_malformed_job_evidence_fails(context, jobs, change):
    jobs[0].update(change)
    with pytest.raises(gate.GateError):
        gate.validate_jobs(jobs, context)


def test_duplicate_names_and_ids_fail(context, jobs):
    with pytest.raises(gate.GateError, match="Duplicate"):
        gate.validate_jobs(jobs + [{**jobs[0], "id": 999}], context)
    jobs[1]["id"] = jobs[0]["id"]
    with pytest.raises(gate.GateError, match="Duplicate"):
        gate.validate_jobs(jobs, context)


def test_unexpected_jobs_fail(context, jobs):
    with pytest.raises(gate.GateError, match="Unexpected"):
        gate.validate_jobs(jobs + [{**jobs[0], "name": "new check", "id": 999}], context)


def test_running_gate_is_not_its_own_prerequisite(context, jobs):
    current = {**jobs[0], "name": "ci-required", "id": 999,
               "status": "in_progress", "conclusion": None}
    gate.validate_jobs(jobs + [current], context)
    with pytest.raises(gate.GateError, match="Duplicate"):
        gate.validate_jobs(jobs + [current, {**current, "id": 1000}], context)


def test_previous_terminal_gate_success_cannot_substitute_for_current_evidence(context, jobs):
    previous = {**jobs[0], "name": "ci-required", "id": 999}
    with pytest.raises(gate.GateError, match="Unexpected terminal gate state"):
        gate.validate_jobs(jobs + [previous], context)


@pytest.mark.parametrize("result", ["failure", "skipped", "cancelled", None])
def test_upstream_failure_or_skip_fails(needs, result):
    needs["generated-artifacts"]["result"] = result
    needs["tests"]["result"] = "skipped"
    with pytest.raises(gate.GateError, match="Dependencies not successful"):
        gate.validate_needs(needs)


@pytest.mark.parametrize("change", ["missing", "extra", "malformed"])
def test_needs_inventory_is_exact(needs, change):
    if change == "missing":
        del needs["tests"]
    elif change == "extra":
        needs["new-job"] = {"result": "success"}
    else:
        needs["tests"] = "success"
    with pytest.raises(gate.GateError):
        gate.validate_needs(needs)


@pytest.mark.parametrize("change", [
    {"id": 43}, {"run_attempt": 2}, {"head_sha": "c" * 40},
    {"run_attempt": True}, {"id": "42"},
    {"event": "pull_request_target"}, {"path": ".github/workflows/other.yml"},
    {"repository": {"full_name": "other/palette"}},
])
def test_run_identity_is_bound_to_invocation(context, run, change):
    run.update(change)
    with pytest.raises(gate.GateError, match="Run identity mismatch"):
        gate.validate_run(run, context)


def test_full_rerun_passes_but_partial_or_mixed_attempts_fail(context, run, jobs):
    rerun = gate.Context(context.repository, 42, 2, SHA, MERGE_SHA, "pull_request")
    run["run_attempt"] = 2
    for job in jobs:
        job["run_attempt"] = 2
    gate.validate_run(run, rerun)
    gate.validate_jobs(jobs, rerun)
    with pytest.raises(gate.GateError, match="Re-run all jobs"):
        gate.validate_jobs(jobs[7:], rerun)
    jobs[0]["run_attempt"] = 1
    with pytest.raises(gate.GateError, match="Re-run all jobs"):
        gate.validate_jobs(jobs, rerun)


def test_paginated_api_evidence_is_complete(context, jobs):
    pages = [{"total_count": 23, "jobs": jobs[:10]},
             {"total_count": 23, "jobs": jobs[10:]}]
    assert gate.jobs_from_pages(pages) == jobs
    gate.validate_jobs(gate.jobs_from_pages(pages), context)


@pytest.mark.parametrize("pages", [
    [], {}, [{"total_count": 2, "jobs": []}],
    [{"total_count": 0, "jobs": {}}], [{"jobs": []}],
    [{"total_count": 0, "jobs": []}, {"total_count": 1, "jobs": []}],
])
def test_incomplete_or_malformed_pages_fail(pages):
    with pytest.raises(gate.GateError):
        gate.jobs_from_pages(pages)


@pytest.fixture
def environ(needs):
    return {
        "GITHUB_REPOSITORY": "jmdelahanty/palette",
        "GITHUB_RUN_ID": "42", "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": MERGE_SHA, "GITHUB_EVENT_NAME": "pull_request",
        "CI_REQUIRED_HEAD_SHA": SHA, "CI_REQUIRED_NEEDS": json.dumps(needs),
    }


def test_cli_reads_exact_attempt_and_checks_checkout(monkeypatch, environ, run, jobs):
    requests = []

    def fake_command(args, **kwargs):
        requests.append(args)
        if args[0] == "git":
            return subprocess.CompletedProcess(args, 0, MERGE_SHA + "\n", "")
        payload = [{"total_count": len(jobs), "jobs": jobs}] if "--slurp" in args else run
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    monkeypatch.setattr(gate.subprocess, "run", fake_command)
    assert gate.main(environ) == 0
    assert [args[-1] for args in requests if args[0] == "gh"] == [
        "repos/jmdelahanty/palette/actions/runs/42",
        "repos/jmdelahanty/palette/actions/runs/42/attempts/1/jobs?per_page=100",
        "repos/jmdelahanty/palette/actions/runs/42",
    ]
    assert ["git", "rev-parse", "HEAD"] in requests


def test_run_attempt_rollover_fails(monkeypatch, environ, run, jobs):
    calls = 0

    def api(path, *, paginate=False):
        nonlocal calls
        calls += 1
        if paginate:
            return [{"total_count": len(jobs), "jobs": jobs}]
        return run if calls == 1 else {**run, "run_attempt": 2}

    monkeypatch.setattr(gate, "api_json", api)
    monkeypatch.setattr(gate, "checked_out_sha", lambda: MERGE_SHA)
    assert gate.main(environ) == 1


def test_wrong_checkout_fails_before_api(monkeypatch, environ):
    monkeypatch.setattr(gate, "checked_out_sha", lambda: SHA)
    monkeypatch.setattr(gate, "api_json", lambda *a, **k: pytest.fail("unexpected API"))
    assert gate.main(environ) == 1


@pytest.mark.parametrize("key,value", [
    ("GITHUB_RUN_ID", ""), ("GITHUB_RUN_ATTEMPT", "0"),
    ("GITHUB_REPOSITORY", "../other/path"), ("GITHUB_SHA", "short"),
    ("CI_REQUIRED_HEAD_SHA", ""), ("GITHUB_EVENT_NAME", "pull_request_target"),
    ("CI_REQUIRED_NEEDS", "invalid json"),
])
def test_invalid_invocation_fails_closed(environ, key, value):
    environ[key] = value
    assert gate.main(environ) == 1


def test_missing_environment_fails_closed(capsys):
    assert gate.main({}) == 1
    assert "ci-required: FAIL" in capsys.readouterr().err


def test_malformed_api_json_fails_closed(monkeypatch):
    monkeypatch.setattr(gate, "command_output", lambda args: "not JSON")
    with pytest.raises(gate.GateError, match="Malformed GitHub API JSON"):
        gate.api_json("repos/jmdelahanty/palette/actions/runs/42")


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(1, "gh", stderr="secret must not be printed"),
    subprocess.TimeoutExpired("gh", 60), FileNotFoundError("gh"),
])
def test_api_errors_fail_closed_without_leaking_output(monkeypatch, failure, capsys):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(gate.subprocess, "run", fail)
    with pytest.raises(gate.GateError) as error:
        gate.api_json("repos/jmdelahanty/palette/actions/runs/42")
    assert "secret" not in str(error.value)
    assert "secret" not in capsys.readouterr().err


def test_workflow_inventory_and_wiring_cannot_drift():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    assert set(jobs) == {*gate.SINGLE_JOBS, "tests", "ci-required"}
    assert {key: jobs[key]["name"] for key in gate.SINGLE_JOBS} == gate.SINGLE_JOBS
    assert jobs["tests"]["name"] == "non-gpu tests (shard ${{ matrix.shard }})"
    assert jobs["tests"]["strategy"] == {
        "fail-fast": False, "matrix": {"shard": list(range(gate.SHARD_COUNT))},
    }
    terminal = jobs["ci-required"]
    assert terminal["name"] == "ci-required"
    assert terminal["if"] == "always()"
    assert set(terminal["needs"]) == {*gate.SINGLE_JOBS, "tests"}
    assert len(terminal["needs"]) == len(gate.SINGLE_JOBS) + 1
    assert terminal["permissions"] == {"actions": "read", "contents": "read"}
    assert terminal["timeout-minutes"] <= 10
    assert terminal["env"]["PALETTE_PYTHON"] == "python"
    for job in jobs.values():
        assert job.get("continue-on-error", False) is False
        for step in job["steps"]:
            assert step.get("continue-on-error", False) is False
            if step.get("uses", "").startswith("actions/checkout@"):
                assert "ref" not in step.get("with", {})
    checkout = terminal["steps"][0]
    assert checkout["uses"] == "actions/checkout@v4"
    assert checkout["with"] == {"persist-credentials": False}
    verify = terminal["steps"][-1]
    assert "if" not in verify
    assert verify["run"] == "scripts/py scripts/ci_required.py"
    assert verify["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "CI_REQUIRED_NEEDS": "${{ toJSON(needs) }}",
        "CI_REQUIRED_HEAD_SHA": "${{ github.event.pull_request.head.sha || github.sha }}",
    }
    # PyYAML's YAML 1.1 loader reads the Actions `on` key as True.
    triggers = workflow.get("on", workflow.get(True))
    assert triggers == {"push": {"branches": ["main"]}, "pull_request": None}
