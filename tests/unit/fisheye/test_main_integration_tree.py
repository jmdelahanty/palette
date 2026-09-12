from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import check_main_integration_tree as gate
from scripts import ci_required as ci_gate

ROOT = Path(__file__).resolve().parents[3]
LANDED_SHA = "a" * 40
BASE_SHA = "b" * 40
HEAD_SHA = "c" * 40
TESTED_SHA = "d" * 40
TREE_SHA = "e" * 40
HEAD_PARENT_SHA = "f" * 40
CHECK_ID = 601
CHECK_SUITE_ID = 701
RUN_ID = 801


@pytest.fixture
def context():
    return gate.Context(
        "jmdelahanty/palette",
        LANDED_SHA,
        "refs/heads/main",
        "push",
        22372296,
        gate.ACTIONS_APP_ID,
    )


@pytest.fixture
def landed():
    return gate.CommitIdentity(LANDED_SHA, TREE_SHA, (BASE_SHA, HEAD_SHA))


@pytest.fixture
def pull_identity():
    return gate.PullIdentity(17, BASE_SHA, HEAD_SHA, "feature", "jmdelahanty/palette")


@pytest.fixture
def ci_identity():
    return gate.CiIdentity(RUN_ID, 1, CHECK_ID, CHECK_SUITE_ID)


@pytest.fixture
def ruleset(context):
    return {
        "id": context.ruleset_id,
        "target": "branch",
        "source_type": "Repository",
        "source": context.repository,
        "enforcement": "active",
        "conditions": {"ref_name": {"exclude": [], "include": ["refs/heads/main"]}},
        "rules": [
            {"type": "deletion"},
            {"type": "non_fast_forward"},
            {
                "type": "required_status_checks",
                "parameters": {
                    "strict_required_status_checks_policy": True,
                    "do_not_enforce_on_create": False,
                    "required_status_checks": [
                        {"context": name, "integration_id": 15368}
                        for name in gate.REQUIRED_CI_CONTEXTS
                    ],
                },
            },
            {
                "type": "pull_request",
                "parameters": {"allowed_merge_methods": ["merge"]},
            },
        ],
        "bypass_actors": [],
    }


@pytest.fixture
def pull(context):
    return {
        "number": 17,
        "state": "closed",
        "merged": True,
        "merged_at": "2026-09-11T12:00:00Z",
        "merge_commit_sha": context.landed_sha,
        "base": {
            "ref": "main",
            "sha": BASE_SHA,
            "repo": {"full_name": context.repository},
        },
        "head": {
            "ref": "feature",
            "sha": HEAD_SHA,
            "repo": {"full_name": context.repository},
        },
    }


@pytest.fixture
def terminal_checks(context):
    check = {
        "id": CHECK_ID,
        "name": "ci-required",
        "head_sha": HEAD_SHA,
        "status": "completed",
        "conclusion": "success",
        "details_url": (
            f"https://github.com/{context.repository}/actions/runs/{RUN_ID}/job/{CHECK_ID}"
        ),
        "check_suite": {"id": CHECK_SUITE_ID},
        "app": {"id": context.actions_app_id},
    }
    return [{"total_count": 1, "check_runs": [check]}]


@pytest.fixture
def run(context):
    return {
        "id": RUN_ID,
        "name": "CI",
        "head_sha": HEAD_SHA,
        "head_branch": "feature",
        "path": ".github/workflows/ci.yml",
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "check_suite_id": CHECK_SUITE_ID,
        "run_attempt": 1,
        "repository": {"full_name": context.repository},
        "head_repository": {"full_name": context.repository},
    }


@pytest.fixture
def attestation(pull_identity, ci_identity):
    return {
        "schema": gate.ATTESTATION_SCHEMA,
        "repository": "jmdelahanty/palette",
        "workflow_path": gate.CI_WORKFLOW_PATH,
        "event": "pull_request",
        "run_id": ci_identity.run_id,
        "run_attempt": ci_identity.run_attempt,
        "candidate_head_sha": pull_identity.head_sha,
        "tested_sha": TESTED_SHA,
        "tested_tree_sha": TREE_SHA,
        "tested_parent_shas": [pull_identity.base_sha, pull_identity.head_sha],
        "pull_request": {
            "number": pull_identity.number,
            "base_ref": "main",
            "base_sha": pull_identity.base_sha,
            "head_ref": pull_identity.head_ref,
            "head_sha": pull_identity.head_sha,
        },
    }


def make_job_log(payload: object) -> bytes:
    record = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return (
        "2026-09-11T12:00:00Z setup\n"
        f"2026-09-11T12:00:01Z ci-required-attestation: {record}\n"
        "2026-09-11T12:00:02Z ci-required: PASS\n"
    ).encode()


def test_exact_same_tree_evidence_passes(
    context,
    landed,
    pull,
    pull_identity,
    terminal_checks,
    run,
    ci_identity,
    ruleset,
    attestation,
):
    gate.validate_ruleset(ruleset, context)
    assert gate.validate_associated_pull_requests([[{"number": 17}]]) == 17
    assert gate.validate_pull_request(pull, context, landed, 17) == pull_identity
    partial_ci = gate.validate_terminal_check(terminal_checks, context, HEAD_SHA)
    assert (
        gate.validate_workflow_run(run, context, pull_identity, partial_ci)
        == ci_identity
    )
    loaded = gate.attestation_from_job_log(make_job_log(attestation))
    assert (
        gate.validate_attestation(
            loaded,
            context,
            landed,
            pull_identity,
            ci_identity,
        )
        == TESTED_SHA
    )


@pytest.mark.parametrize(
    "ref,event",
    [
        ("refs/heads/feature", "push"),
        ("refs/heads/main", "pull_request"),
    ],
)
def test_context_rejects_anything_except_main_push(ref, event):
    with pytest.raises(gate.IntegrationError, match="push to main"):
        gate.Context(
            "jmdelahanty/palette",
            LANDED_SHA,
            ref,
            event,
            22372296,
            gate.ACTIONS_APP_ID,
        )


def test_landed_commit_requires_two_parents_ancestry_and_same_tree(
    monkeypatch, context
):
    monkeypatch.setattr(gate, "command_output", lambda args: LANDED_SHA + "\n")
    one_parent = gate.CommitIdentity(LANDED_SHA, TREE_SHA, (BASE_SHA,))
    monkeypatch.setattr(gate, "commit_identity", lambda sha: one_parent)
    with pytest.raises(gate.IntegrationError, match="two-parent"):
        gate.validate_landed_commit(context)

    landed = gate.CommitIdentity(LANDED_SHA, TREE_SHA, (BASE_SHA, HEAD_SHA))
    wrong_head = gate.CommitIdentity(HEAD_SHA, "0" * 40, (HEAD_PARENT_SHA,))
    monkeypatch.setattr(
        gate,
        "commit_identity",
        lambda sha: landed if sha == LANDED_SHA else wrong_head,
    )
    monkeypatch.setattr(gate, "require_ancestor", lambda *args: None)
    with pytest.raises(gate.IntegrationError, match="trees differ"):
        gate.validate_landed_commit(context)


def test_non_ancestor_is_blocking(monkeypatch):
    result = subprocess.CompletedProcess([], 1, "", "")
    monkeypatch.setattr(gate.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(gate.IntegrationError, match="not an ancestor"):
        gate.require_ancestor(BASE_SHA, HEAD_SHA)


@pytest.mark.parametrize(
    "change",
    [
        ("enforcement", "evaluate"),
        ("bypass_actors", [{"actor_id": 1}]),
        ("conditions", {"ref_name": {"exclude": [], "include": ["~ALL"]}}),
    ],
)
def test_ruleset_identity_scope_and_bypass_are_exact(context, ruleset, change):
    key, value = change
    ruleset[key] = value
    with pytest.raises(gate.IntegrationError):
        gate.validate_ruleset(ruleset, context)


@pytest.mark.parametrize(
    "change",
    [
        "not_strict",
        "wrong_app",
        "missing_terminal",
        "missing_upstream",
    ],
)
def test_ruleset_required_terminal_check_is_exact(context, ruleset, change):
    parameters = ruleset["rules"][2]["parameters"]
    if change == "not_strict":
        parameters["strict_required_status_checks_policy"] = False
    elif change == "wrong_app":
        parameters["required_status_checks"][-1]["integration_id"] = 1
    elif change == "missing_terminal":
        parameters["required_status_checks"].pop()
    else:
        parameters["required_status_checks"].pop(0)
    with pytest.raises(gate.IntegrationError):
        gate.validate_ruleset(ruleset, context)


@pytest.mark.parametrize("pulls", [[], [[{"number": 1}, {"number": 2}]], {}])
def test_associated_pull_request_must_be_unique(pulls):
    with pytest.raises(gate.IntegrationError):
        gate.validate_associated_pull_requests(pulls)


@pytest.mark.parametrize(
    "change",
    [
        ("state", "open"),
        ("merged", False),
        ("merge_commit_sha", "0" * 40),
        ("base_ref", "release"),
        ("base_sha", "0" * 40),
        ("head_sha", "0" * 40),
    ],
)
def test_pull_request_must_bind_exact_merge(context, landed, pull, change):
    key, value = change
    if key == "base_ref":
        pull["base"]["ref"] = value
    elif key == "base_sha":
        pull["base"]["sha"] = value
    elif key == "head_sha":
        pull["head"]["sha"] = value
    else:
        pull[key] = value
    with pytest.raises(gate.IntegrationError):
        gate.validate_pull_request(pull, context, landed, 17)


@pytest.mark.parametrize(
    "change",
    [
        ("conclusion", "failure"),
        ("status", "in_progress"),
        ("head_sha", "0" * 40),
        ("app", {"id": 1}),
        ("details_url", "https://example.test/not-a-run"),
    ],
)
def test_terminal_check_must_be_exact_success(context, terminal_checks, change):
    key, value = change
    terminal_checks[0]["check_runs"][0][key] = value
    with pytest.raises(gate.IntegrationError):
        gate.validate_terminal_check(terminal_checks, context, HEAD_SHA)


def test_absent_or_multiple_terminal_checks_fail(context, terminal_checks):
    terminal_checks[0]["total_count"] = 0
    terminal_checks[0]["check_runs"] = []
    with pytest.raises(gate.IntegrationError):
        gate.validate_terminal_check(terminal_checks, context, HEAD_SHA)


@pytest.mark.parametrize(
    "change",
    [
        ("event", "push"),
        ("path", ".github/workflows/other.yml"),
        ("head_sha", "0" * 40),
        ("head_branch", "other"),
        ("conclusion", "failure"),
        ("check_suite_id", 999),
        ("run_attempt", 0),
    ],
)
def test_workflow_run_must_bind_terminal_check(
    context,
    pull_identity,
    ci_identity,
    run,
    change,
):
    key, value = change
    run[key] = value
    with pytest.raises(gate.IntegrationError):
        gate.validate_workflow_run(run, context, pull_identity, ci_identity)


def test_workflow_run_recheck_rejects_a_new_attempt(
    context,
    pull_identity,
    ci_identity,
    run,
):
    run["run_attempt"] = ci_identity.run_attempt + 1
    with pytest.raises(gate.IntegrationError, match="attempt changed"):
        gate.validate_workflow_run(run, context, pull_identity, ci_identity)


def test_attestation_job_log_record_is_unique_and_well_formed(attestation):
    with pytest.raises(gate.IntegrationError, match="one exact"):
        gate.attestation_from_job_log(b"no evidence here\n")
    duplicate = make_job_log(attestation) + make_job_log(attestation)
    with pytest.raises(gate.IntegrationError, match="one exact"):
        gate.attestation_from_job_log(duplicate)
    with pytest.raises(gate.IntegrationError, match="Malformed CI evidence"):
        gate.attestation_from_job_log(b"ci-required-attestation: not-json\n")


@pytest.mark.parametrize(
    "change",
    [
        ("schema", "v0"),
        ("repository", "other/palette"),
        ("run_id", 999),
        ("candidate_head_sha", "0" * 40),
        ("tested_tree_sha", "0" * 40),
        ("tested_parent_shas", [HEAD_SHA, BASE_SHA]),
        ("pull_request.base_sha", "0" * 40),
        ("extra", "unexpected"),
    ],
)
def test_attestation_must_bind_exact_tree_and_candidate(
    context,
    landed,
    pull_identity,
    ci_identity,
    attestation,
    change,
):
    key, value = change
    if key.startswith("pull_request."):
        attestation["pull_request"][key.split(".", 1)[1]] = value
    else:
        attestation[key] = value
    with pytest.raises(gate.IntegrationError):
        gate.validate_attestation(
            attestation,
            context,
            landed,
            pull_identity,
            ci_identity,
        )


def test_verify_reads_exact_evidence_and_rechecks_immutable_claims(
    monkeypatch,
    context,
    landed,
    pull,
    terminal_checks,
    run,
    ruleset,
    attestation,
):
    head = gate.CommitIdentity(HEAD_SHA, TREE_SHA, (HEAD_PARENT_SHA, BASE_SHA))
    monkeypatch.setattr(gate, "validate_landed_commit", lambda _context: (landed, head))
    calls = []

    def api(path, *, paginate=False):
        calls.append((path, paginate))
        if "/rulesets/" in path:
            return ruleset
        if "/commits/" + LANDED_SHA + "/pulls" in path:
            return [[{"number": 17}]]
        if path.endswith("/pulls/17"):
            return pull
        if "/check-runs?" in path:
            return terminal_checks
        if path.endswith(f"/actions/runs/{RUN_ID}"):
            return run
        raise AssertionError(path)

    monkeypatch.setattr(gate, "api_json", api)
    monkeypatch.setattr(gate, "api_job_log", lambda path: make_job_log(attestation))
    result = gate.verify(
        {
            "GITHUB_REPOSITORY": context.repository,
            "GITHUB_SHA": context.landed_sha,
            "GITHUB_REF": context.ref,
            "GITHUB_EVENT_NAME": context.event,
            "PALETTE_MAIN_RULESET_ID": str(context.ruleset_id),
            "PALETTE_ACTIONS_APP_ID": str(context.actions_app_id),
        }
    )
    assert result == {
        "landed_sha": LANDED_SHA,
        "head_sha": HEAD_SHA,
        "tree_sha": TREE_SHA,
        "pull_request_number": 17,
        "ci_run_id": RUN_ID,
        "ci_run_attempt": 1,
        "tested_sha": TESTED_SHA,
        "ruleset_id": context.ruleset_id,
    }
    assert sum("/rulesets/" in path for path, _ in calls) == 2
    assert sum("/check-runs?" in path for path, _ in calls) == 2
    assert sum(path.endswith(f"/actions/runs/{RUN_ID}") for path, _ in calls) == 2


def test_cli_failure_is_fail_closed_without_secrets(monkeypatch, capsys):
    monkeypatch.setattr(
        gate,
        "verify",
        lambda env: (_ for _ in ()).throw(gate.IntegrationError("missing evidence")),
    )
    assert gate.main({}) == 1
    output = capsys.readouterr().err
    assert "main-integration: FAIL" in output
    assert "run full CI manually" in output


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, "gh", stderr="secret must not leak"),
        subprocess.TimeoutExpired("gh", 60),
        FileNotFoundError("gh"),
    ],
)
def test_api_failures_do_not_leak_command_output(monkeypatch, failure, capsys):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(gate.subprocess, "run", fail)
    with pytest.raises(gate.IntegrationError) as error:
        gate.api_json("repos/jmdelahanty/palette/rulesets/1")
    assert "secret" not in str(error.value)
    assert "secret" not in capsys.readouterr().err


def test_workflow_contract_uses_full_pr_ci_and_lightweight_main_gate_only():
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    integration = yaml.safe_load(
        (ROOT / ".github/workflows/main-integration.yml").read_text(),
    )
    ci_triggers = ci.get("on", ci.get(True))
    integration_triggers = integration.get("on", integration.get(True))
    assert ci_triggers == {"pull_request": None, "workflow_dispatch": None}
    assert integration_triggers == {"push": {"branches": ["main"]}}
    assert integration["permissions"] == {
        "actions": "read",
        "contents": "read",
        "pull-requests": "read",
    }
    assert set(integration["jobs"]) == {"main-integration"}
    job = integration["jobs"]["main-integration"]
    assert job["name"] == "main-integration"
    assert job["timeout-minutes"] <= 10
    assert job["env"] == {
        "PALETTE_PYTHON": "python",
        "PALETTE_ACTIONS_APP_ID": str(gate.ACTIONS_APP_ID),
        "PALETTE_MAIN_RULESET_ID": "22372296",
    }
    assert job["steps"][0] == {
        "name": "Check out exact integrated commit",
        "uses": "actions/checkout@v4",
        "with": {"fetch-depth": 0, "persist-credentials": False},
    }
    assert job["steps"][-1] == {
        "name": "Verify exact green same-tree integration",
        "env": {"GH_TOKEN": "${{ github.token }}"},
        "run": "scripts/py scripts/check_main_integration_tree.py",
    }
    assert all(step.get("continue-on-error", False) is False for step in job["steps"])
    assert set(gate.REQUIRED_CI_CONTEXTS) == {
        *ci_gate.required_names(),
        "ci-required",
    }
