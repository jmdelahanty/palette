"""Workflow-runner intake slice: config, LSF budget, attach, done-only-from-Palette."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

from fisheye.flow import intake as flow_intake
from fisheye.flow.config import FlowConfigError, parse_config
from fisheye.flow.lsf import (
    BjobsCache,
    job_script,
    parse_bjobs,
    run_attempt,
    wait,
    write_json_atomic,
)

SHA = "a" * 64
COMMIT = "c" * 40
REPO_ROOT = Path(__file__).resolve().parents[3]


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _raw(tmp_path: Path, **overrides) -> dict:
    raw = {
        "schema": "palette.flow.intake_config.v1",
        "flow_root": str(tmp_path / "flow"),
        "staging_dir": str(tmp_path / "staging"),
        "destination_root": str(tmp_path / "recordings"),
        "registry": str(tmp_path / "registry.sqlite"),
        "registrar_config": str(tmp_path / "registrar.json"),
        "ops_deployment": str(tmp_path / "deployments" / "ops-current"),
        "deployments_root": str(tmp_path / "deployments"),
        "lsf_repo": str(tmp_path / "lsf-repo"),
        "lsf": {"submit_host": "login1-citrus-poller", "poll_s": 60},
    }
    raw.update(overrides)
    return raw


@pytest.fixture
def config(tmp_path):
    return parse_config(_raw(tmp_path))


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def _probe(step: str, sha: str = SHA, verdict=True, **extra) -> dict:
    document = {
        "schema": f"palette.intake.probe_{step}.v1",
        "snapshot_sha": sha,
        "verdict": verdict,
        "evidence_digest": "sha256:" + "d" * 64 if verdict else None,
        "producer_git_sha": COMMIT,
    }
    document.update(extra)
    return document


# --- config -----------------------------------------------------------------


def test_config_refuses_v1_state_and_staging_and_budget_below_five_minutes(tmp_path):
    with pytest.raises(FlowConfigError, match="v1 intake state"):
        parse_config(_raw(tmp_path, flow_root=str(tmp_path / "staging" / ".processing_state" / "x")))
    with pytest.raises(FlowConfigError, match="inside staging_dir"):
        parse_config(_raw(tmp_path, flow_root=str(tmp_path / "staging" / "flow")))
    with pytest.raises(FlowConfigError, match="bjobs_min_interval_s"):
        parse_config(_raw(tmp_path, lsf={"submit_host": "login1", "bjobs_min_interval_s": 60}))
    with pytest.raises(FlowConfigError, match="submit_host"):
        parse_config(_raw(tmp_path, lsf={"submit_host": "login1; rm -rf /"}))
    with pytest.raises(FlowConfigError, match="absolute"):
        parse_config(_raw(tmp_path, destination_root="recordings"))


# --- the login-node budget ---------------------------------------------------


def test_bjobs_cache_is_one_query_per_interval_across_callers(tmp_path):
    clock = Clock()
    calls = []

    def query():
        calls.append(clock())
        return _completed(stdout="101 RUN\n102 PEND\n")

    caches = [BjobsCache(tmp_path, min_interval_s=300, query=query, clock=clock) for _ in range(3)]
    for _ in range(60):  # an hour of 60 s polls from three waiting steps
        for cache in caches:
            assert cache.get()["jobs"]["101"] == "RUN"
        clock.sleep(60)
    assert len(calls) == 12
    assert all(b - a >= 300 for a, b in zip(calls, calls[1:]))


def test_failed_bjobs_query_still_consumes_the_interval(tmp_path):
    clock = Clock()
    calls = []

    def query():
        calls.append(clock())
        return _completed(returncode=255, stderr="ssh: connect failed")

    cache = BjobsCache(tmp_path, min_interval_s=300, query=query, clock=clock)
    for _ in range(10):
        document = cache.get()
        clock.sleep(30)
    assert len(calls) == 1
    assert document["ok"] is False and "connect failed" in document["error"]


def test_parse_bjobs_ignores_noise():
    assert parse_bjobs("No unfinished job found\n7 DONE\n8 exit\n") == {"7": "DONE", "8": "EXIT"}


def _submitted(attempt: Path, clock: Clock, job_id="4242") -> None:
    attempt.mkdir(parents=True)
    write_json_atomic(attempt / "submission.json", {"job_id": job_id, "submitted_at": clock()})


def test_wait_uses_nfs_evidence_and_never_queries_lsf_while_heartbeat_is_fresh(tmp_path, config):
    clock = Clock()
    attempt = tmp_path / "attempt-1"
    _submitted(attempt, clock)
    queries = []
    cache = BjobsCache(tmp_path / "lsf", min_interval_s=300,
                       query=lambda: queries.append(1) or _completed(stdout="4242 RUN\n"), clock=clock)

    def sleep(seconds):
        clock.sleep(seconds)
        (attempt / "heartbeat").touch()
        os.utime(attempt / "heartbeat", (clock(), clock()))
        if clock() - 1_000_000 > 1800:
            (attempt / "result.json").write_text(json.dumps(_probe("import")))
            (attempt / "exit_code").write_text("0\n")

    (attempt / "heartbeat").touch()
    os.utime(attempt / "heartbeat", (clock(), clock()))
    result = wait(attempt, settings=config.lsf, cache=cache, clock=clock, sleep=sleep)
    assert result.exit_code == 0 and result.document["verdict"] is True
    assert queries == []


def test_wait_reports_a_job_that_died_without_an_exit_record(tmp_path, config):
    clock = Clock()
    attempt = tmp_path / "attempt-1"
    _submitted(attempt, clock)
    (attempt / "lsf.out").write_text("TERM_RUNLIMIT\nResource usage summary:\n")
    cache = BjobsCache(tmp_path / "lsf", min_interval_s=300,
                       query=lambda: pytest.fail("no LSF query needed"), clock=clock)
    result = wait(attempt, settings=config.lsf, cache=cache, clock=clock, sleep=clock.sleep)
    assert result.exit_code == 1 and "without an exit record" in result.reason


def test_wait_fails_an_ended_job_with_no_evidence_after_one_grace_poll(tmp_path, config):
    clock = Clock()
    attempt = tmp_path / "attempt-1"
    _submitted(attempt, clock)
    cache = BjobsCache(tmp_path / "lsf", min_interval_s=300,
                       query=lambda: _completed(stdout="4242 EXIT\n"), clock=clock)
    result = wait(attempt, settings=config.lsf, cache=cache, clock=clock, sleep=clock.sleep)
    assert result.exit_code == 1 and "EXIT" in result.reason


# --- submit / attach ---------------------------------------------------------


def test_run_attempt_submits_once_then_attaches_after_a_controller_restart(tmp_path, config):
    clock = Clock()
    step_dir = tmp_path / "flow" / "intake" / SHA / "import"
    bsub_calls = []

    def bsub(command, cwd=None):
        bsub_calls.append(command)
        return _completed(stdout="Job <4242> is submitted to queue <short>.\n")

    cache = BjobsCache(tmp_path / "lsf", min_interval_s=300,
                       query=lambda: _completed(stdout="4242 PEND\n"), clock=clock)

    class ControllerKilled(Exception):
        pass

    def killed(_seconds):
        raise ControllerKilled

    with pytest.raises(ControllerKilled):
        run_attempt(step_dir, job_name="j", remote_repo=Path("/repo"),
                    argv=["-m", "fisheye.intake", "import-delivery", SHA, "--run-dir", "{ATTEMPT}/run"],
                    settings=config.lsf, cache=cache, bsub_runner=bsub, clock=clock, sleep=killed)
    attempt = step_dir / "attempt-1"
    script = (attempt / "job.sh").read_text()
    assert f"--run-dir {attempt}/run" in script
    assert bsub_calls[0][:2] == ["bsub", "-J"] and "-oo" in bsub_calls[0]

    def finish(seconds):
        clock.sleep(seconds)
        (attempt / "result.json").write_text(json.dumps(_probe("import")))
        (attempt / "exit_code").write_text("0\n")

    result = run_attempt(step_dir, job_name="j", remote_repo=Path("/repo"), argv=["x"],
                         settings=config.lsf, cache=cache, bsub_runner=bsub, clock=clock, sleep=finish)
    assert result.exit_code == 0 and result.attempt_dir == attempt
    assert len(bsub_calls) == 1  # attached, not resubmitted


def test_job_script_publishes_result_exit_code_and_heartbeat(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "py").write_text('#!/usr/bin/env bash\necho "{\\"args\\": \\"$*\\"}"\necho log >&2\nexit 75\n')
    (repo / "scripts" / "py").chmod(0o755)
    attempt = tmp_path / "attempt-1"
    attempt.mkdir()
    (attempt / "job.sh").write_text(job_script(remote_repo=repo, attempt_dir=attempt, argv=["-m", "x", "y z"]))
    completed = subprocess.run(["bash", str(attempt / "job.sh")], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 75
    assert (attempt / "exit_code").read_text().strip() == "75"
    assert json.loads((attempt / "result.json").read_text()) == {"args": "-m x y z"}
    assert (attempt / "payload.err").read_text().endswith("log\n")
    assert (attempt / "heartbeat").exists()


# --- Palette decides done ----------------------------------------------------


@pytest.mark.parametrize("document", [
    None,
    _probe("import", verdict=False),
    _probe("import", sha="b" * 64),
    _probe("register"),
    {**_probe("import"), "evidence_digest": None},
])
def test_exit_zero_without_a_true_matching_probe_writes_no_sentinel(config, document):
    code = flow_intake.record_outcome(config, SHA, "import", 0, document,
                                      deployment=Path("/d"), commit=COMMIT)
    assert code == 1
    assert not flow_intake.sentinel_path(config, SHA, "import").exists()


def test_outcomes_map_to_sentinel_hold_and_codes(config):
    assert flow_intake.record_outcome(config, SHA, "import", 0, _probe("import"),
                                      deployment=Path("/d"), commit=COMMIT) == 0
    sentinel = json.loads(flow_intake.sentinel_path(config, SHA, "import").read_text())
    assert sentinel["evidence_digest"].startswith("sha256:")
    assert sentinel["producer_git_sha"] == COMMIT
    assert flow_intake.record_outcome(config, SHA, "register", 75, {}, deployment=Path("/d"), commit=None) == 75
    assert flow_intake.record_outcome(config, SHA, "register", 1, None, deployment=Path("/d"), commit=None) == 1
    assert not flow_intake.refusal_path(config, SHA).exists()
    refused = {"error": "registrar_commit_mismatch", "receipt_producer_git_sha": COMMIT}
    assert flow_intake.record_outcome(config, SHA, "register", 65, refused,
                                      deployment=Path("/d"), commit=None) == 65
    hold = json.loads(flow_intake.refusal_path(config, SHA).read_text())
    assert hold["step"] == "register" and hold["result"] == refused


def _fake_exec(responses):
    calls = []

    def run(argv):
        calls.append(list(argv))
        for match, response in responses:
            if match(argv):
                return response(argv) if callable(response) else response
        raise AssertionError(f"unexpected command {argv}")

    return run, calls


def test_plan_skips_refused_legacy_and_unknown_registry_targets(config):
    targets = [
        {"snapshot_sha": "1" * 64, "legacy_mode": False, "import_recorded": False, "register_recorded": None},
        {"snapshot_sha": "2" * 64, "legacy_mode": True, "import_recorded": True, "register_recorded": False},
        {"snapshot_sha": "3" * 64, "legacy_mode": False, "import_recorded": True, "register_recorded": None},
        {"snapshot_sha": "4" * 64, "legacy_mode": False, "import_recorded": False, "register_recorded": False},
    ]
    flow_intake.refusal_path(config, "4" * 64).parent.mkdir(parents=True)
    flow_intake.refusal_path(config, "4" * 64).write_text("{}")
    discover = {"targets": targets, "legacy_markers": ["x"], "registry_error": None}
    run, calls = _fake_exec([(lambda a: "discover" in a, _completed(stdout=json.dumps(discover)))])
    planned = flow_intake.plan(config, runner=run)
    assert planned["drive"] == ["1" * 64]
    assert {i["snapshot_sha"] for i in planned["skipped"]} == {"2" * 64, "3" * 64}
    assert [i["snapshot_sha"] for i in planned["held"]] == ["4" * 64]
    assert "--destination-root" in calls[0] and str(config.destination_root) in calls[0]


def test_import_step_records_an_existing_import_without_touching_lsf(config):
    run, calls = _fake_exec([
        (lambda a: a[0] == "git", _completed(stdout=COMMIT + "\n")),
        (lambda a: "probe-import" in a, _completed(stdout=json.dumps(_probe("import")))),
    ])
    code = flow_intake.import_step(config, SHA, runner=run,
                                   bsub_runner=lambda *a, **k: pytest.fail("no bsub"))
    assert code == 0
    sentinel = json.loads(flow_intake.sentinel_path(config, SHA, "import").read_text())
    assert sentinel["source"] == "probe"


def test_register_runs_at_the_producer_commit_deployment(config, tmp_path):
    for name in ("ops-cccccccc", "ops-current"):
        (config.deployments_root / name).mkdir(parents=True)
    flow_intake.record_outcome(config, SHA, "import", 0, _probe("import"),
                               deployment=Path("/lsf"), commit=COMMIT)
    run, calls = _fake_exec([
        (lambda a: a[0] == "git", lambda a: _completed(
            stdout=(COMMIT if a[2].endswith("ops-cccccccc") else "e" * 40) + "\n")),
        (lambda a: "register-delivery" in a, _completed(stdout=json.dumps(_probe("register")))),
    ])
    assert flow_intake.register_step(config, SHA, runner=run) == 0
    register_call = next(c for c in calls if "register-delivery" in c)
    assert register_call[0] == str(config.deployments_root / "ops-cccccccc" / "scripts" / "py")
    assert "--config" in register_call and str(config.registrar_config) in register_call


def test_register_without_the_producer_deployment_fails_without_a_hold(config):
    flow_intake.record_outcome(config, SHA, "import", 0, _probe("import"),
                               deployment=Path("/lsf"), commit=COMMIT)
    run, _ = _fake_exec([])
    assert flow_intake.register_step(config, SHA, runner=run) == 1
    assert not flow_intake.refusal_path(config, SHA).exists()
    assert not flow_intake.sentinel_path(config, SHA, "register").exists()


# --- the Snakefile never owns data -------------------------------------------


def test_snakefile_outputs_are_runner_sentinels_only_and_rules_are_local():
    text = "\n".join(
        line for line in (REPO_ROOT / "workflows" / "intake.smk").read_text().splitlines()
        if not line.lstrip().startswith("#")
    ) + "\n"
    assert "directory(" not in text
    outputs = re.findall(r"output:\s*\n\s*(.+?),\n", text)
    assert outputs and all(o.startswith('INTAKE + "/{sha}/') and o.endswith('.done.json"') for o in outputs)
    rules = set(re.findall(r"^rule (\w+):", text, re.M))
    local = set(re.search(r"^localrules: (.+)$", text, re.M).group(1).replace(" ", "").split(","))
    assert rules <= local
    assert ".processing_state" not in text


# --- retry cap ----------------------------------------------------------------


def test_retry_cap_holds_after_consecutive_failures_and_held_does_not_count(config):
    for _ in range(2):
        assert flow_intake.record_outcome(config, SHA, "import", 1, None,
                                          deployment=Path("/d"), commit=None) == 1
        assert flow_intake.record_outcome(config, SHA, "import", 75, {},
                                          deployment=Path("/d"), commit=None) == 75
    assert not flow_intake.refusal_path(config, SHA).exists()
    assert flow_intake.record_outcome(config, SHA, "import", 1, {"error": "boom"},
                                      deployment=Path("/d"), commit=None) == 65
    hold = json.loads(flow_intake.refusal_path(config, SHA).read_text())
    assert hold["retry_cap"] is True and "boom" in hold["result"]["message"]
    assert flow_intake.import_step(config, SHA, runner=lambda a: pytest.fail("held")) == 65


def test_success_resets_the_failure_count(config):
    for _ in range(2):
        flow_intake.record_outcome(config, SHA, "import", 1, None, deployment=Path("/d"), commit=None)
    assert flow_intake.record_outcome(config, SHA, "import", 0, _probe("import"),
                                      deployment=Path("/d"), commit=COMMIT) == 0
    flow_intake.sentinel_path(config, SHA, "import").unlink()
    for _ in range(2):
        assert flow_intake.record_outcome(config, SHA, "import", 1, None,
                                          deployment=Path("/d"), commit=None) == 1
    assert not flow_intake.refusal_path(config, SHA).exists()


def test_missing_producer_deployment_counts_toward_the_cap(config):
    flow_intake.record_outcome(config, SHA, "import", 0, _probe("import"),
                               deployment=Path("/lsf"), commit=COMMIT)
    run, _ = _fake_exec([])
    codes = [flow_intake.register_step(config, SHA, runner=run) for _ in range(3)]
    assert codes[:2] == [1, 1]
    assert codes[2] == 65 and json.loads(flow_intake.refusal_path(config, SHA).read_text())["step"] == "register"


def test_submission_errors_count_toward_the_cap(config):
    run, _ = _fake_exec([
        (lambda a: a[0] == "git", _completed(stdout=COMMIT + "\n")),
        (lambda a: "probe-import" in a, _completed(returncode=1, stdout=json.dumps(_probe("import", verdict=False)))),
    ])

    def bsub(command, cwd=None):
        return _completed(returncode=255, stderr="ssh: Could not resolve hostname")

    assert flow_intake.import_step(config, SHA, runner=run, bsub_runner=bsub) == 1
    failures = json.loads(flow_intake.failures_path(config, SHA).read_text())
    assert failures["steps"]["import"]["count"] == 1
    assert "Could not resolve" in failures["steps"]["import"]["last_reason"]


# --- isolated synthetic trials --------------------------------------------------


def _registrar(tmp_path: Path, registry: str) -> Path:
    path = tmp_path / "registrar.json"
    path.write_text(json.dumps({"registry": registry}))
    return path


def test_synthetic_trials_refuse_the_canonical_registry(tmp_path):
    from fisheye.intake.delivery import CANONICAL_REGISTRY

    _registrar(tmp_path, str(CANONICAL_REGISTRY))
    with pytest.raises(FlowConfigError, match="canonical registry"):
        parse_config(_raw(tmp_path, registry=str(CANONICAL_REGISTRY),
                          allow_synthetic_isolated_registry=True))
    with pytest.raises(FlowConfigError, match="canonical registry"):
        parse_config(_raw(tmp_path, allow_synthetic_isolated_registry=True))
    _registrar(tmp_path, str(tmp_path / "other.sqlite"))
    with pytest.raises(FlowConfigError, match="same file"):
        parse_config(_raw(tmp_path, allow_synthetic_isolated_registry=True))
    with pytest.raises(FlowConfigError, match="true or false"):
        parse_config(_raw(tmp_path, allow_synthetic_isolated_registry="yes"))


def test_isolated_trial_passes_the_synthetic_flag_to_register(tmp_path):
    _registrar(tmp_path, str(tmp_path / "registry.sqlite"))
    config = parse_config(_raw(tmp_path, allow_synthetic_isolated_registry=True))
    (config.deployments_root / "ops-cccccccc").mkdir(parents=True)
    flow_intake.record_outcome(config, SHA, "import", 0, _probe("import"),
                               deployment=Path("/lsf"), commit=COMMIT)
    run, calls = _fake_exec([
        (lambda a: a[0] == "git", _completed(stdout=COMMIT + "\n")),
        (lambda a: "register-delivery" in a, _completed(stdout=json.dumps(_probe("register")))),
    ])
    assert flow_intake.register_step(config, SHA, runner=run) == 0
    assert "--allow-synthetic-isolated-registry" in next(c for c in calls if "register-delivery" in c)


def test_production_config_never_passes_the_synthetic_flag(config):
    (config.deployments_root / "ops-cccccccc").mkdir(parents=True)
    flow_intake.record_outcome(config, SHA, "import", 0, _probe("import"),
                               deployment=Path("/lsf"), commit=COMMIT)
    run, calls = _fake_exec([
        (lambda a: a[0] == "git", _completed(stdout=COMMIT + "\n")),
        (lambda a: "register-delivery" in a, _completed(stdout=json.dumps(_probe("register")))),
    ])
    flow_intake.register_step(config, SHA, runner=run)
    assert all("--allow-synthetic-isolated-registry" not in c for c in calls)
