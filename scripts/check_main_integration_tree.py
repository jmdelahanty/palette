#!/usr/bin/env python3
"""Accept a landed main merge only when its exact PR tree already passed full CI.

This is the user-owned-repository substitute for GitHub's unavailable merge
queue. It is intentionally fail closed. A failure means full CI must be run
manually on the current exact ``main`` commit before integration is complete.

Standard library only; the runner supplies git, gh and a read-only GH_TOKEN.
Only the ruleset reads use a short-lived, repository-scoped App token, because
GitHub omits bypass_actors from the response to read-only callers.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Mapping

ACTIONS_APP_ID = 15368
ATTESTATION_SCHEMA = "palette-ci-required-attestation-v1"
CI_WORKFLOW_PATH = ".github/workflows/ci.yml"
MAIN_REF = "refs/heads/main"
MAX_JOB_LOG_BYTES = 2_000_000
REQUIRED_CI_CONTEXTS = (
    "generated artifacts",
    "import boundaries",
    "file-size ratchet",
    "zarr open metadata modes",
    "observed metadata literals",
    "active contract freshness",
    "package and collection",
    *(f"non-gpu tests (shard {index})" for index in range(16)),
    "ci-required",
)


class IntegrationError(ValueError):
    """The landed commit lacks exact, sufficient integration evidence."""


@dataclass(frozen=True)
class Context:
    repository: str
    landed_sha: str
    ref: str
    event: str
    ruleset_id: int
    actions_app_id: int

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository):
            raise IntegrationError("Invalid repository identity")
        _sha(self.landed_sha, "landed commit")
        if self.ref != MAIN_REF or self.event != "push":
            raise IntegrationError("Main integration accepts only a push to main")
        if not all(
            type(value) is int and value > 0
            for value in (self.ruleset_id, self.actions_app_id)
        ):
            raise IntegrationError("Invalid ruleset or Actions app identity")


@dataclass(frozen=True)
class CommitIdentity:
    sha: str
    tree_sha: str
    parent_shas: tuple[str, ...]

    def __post_init__(self):
        _sha(self.sha, "commit")
        _sha(self.tree_sha, "commit tree")
        if not all(_is_sha(parent) for parent in self.parent_shas):
            raise IntegrationError("Malformed commit parents")


@dataclass(frozen=True)
class PullIdentity:
    number: int
    base_sha: str
    head_sha: str
    head_ref: str
    head_repository: str


@dataclass(frozen=True)
class CiIdentity:
    run_id: int
    run_attempt: int
    check_id: int
    check_suite_id: int


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _sha(value: object, label: str) -> str:
    if not _is_sha(value):
        raise IntegrationError(f"Malformed {label} SHA")
    return value


def _object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise IntegrationError(f"Malformed {label}")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise IntegrationError(f"Malformed {label}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise IntegrationError(f"Malformed {label}")
    return value


def command_output(args: list[str], *, token: str | None = None) -> str:
    try:
        env = None if token is None else {**os.environ, "GH_TOKEN": token}
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise IntegrationError(
            f"Unable to read integration evidence with {args[0]} "
            f"({type(exc).__name__})"
        ) from exc


def command_bytes(args: list[str]) -> bytes:
    try:
        data = subprocess.run(
            args,
            check=True,
            capture_output=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise IntegrationError(
            f"Unable to download integration evidence with {args[0]} "
            f"({type(exc).__name__})"
        ) from exc
    if not isinstance(data, bytes) or len(data) > MAX_JOB_LOG_BYTES:
        raise IntegrationError("Malformed or oversized terminal job log")
    return data


def _gh_api_args(
    path: str, *, paginate: bool = False, allow_escape_sequences: bool = False
) -> list[str]:
    args = [
        "gh",
        "api",
        "--hostname",
        "github.com",
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "X-GitHub-Api-Version: 2022-11-28",
    ]
    if allow_escape_sequences:
        args.append("--allow-escape-sequences")
    if paginate:
        args.extend(["--paginate", "--slurp"])
    return [*args, path]


def api_json(path: str, *, paginate: bool = False, token: str | None = None) -> object:
    if token is not None and not token.strip():
        raise IntegrationError("Missing ruleset credential")
    try:
        args = _gh_api_args(path, paginate=paginate)
        if token is None:
            return json.loads(command_output(args))
        return json.loads(command_output(args, token=token))
    except json.JSONDecodeError as exc:
        raise IntegrationError("Malformed GitHub API JSON") from exc


def api_job_log(path: str) -> bytes:
    # CI logs contain ANSI codes; gh otherwise refuses to emit them even to
    # captured stdout. The attestation parser still requires one exact record.
    return command_bytes(_gh_api_args(path, allow_escape_sequences=True))


def commit_identity(sha: str) -> CommitIdentity:
    lines = command_output(["git", "cat-file", "-p", sha]).splitlines()
    headers = lines[: lines.index("")] if "" in lines else lines
    trees = [line.removeprefix("tree ") for line in headers if line.startswith("tree ")]
    parents = tuple(
        line.removeprefix("parent ") for line in headers if line.startswith("parent ")
    )
    if len(trees) != 1:
        raise IntegrationError("Malformed commit object")
    return CommitIdentity(sha, trees[0], parents)


def require_ancestor(ancestor: str, descendant: str) -> None:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise IntegrationError("Unable to verify merge ancestry") from exc
    if result.returncode == 1:
        raise IntegrationError(
            "The exact main parent is not an ancestor of the PR head"
        )
    if result.returncode != 0:
        raise IntegrationError("Unable to verify merge ancestry")


def validate_landed_commit(context: Context) -> tuple[CommitIdentity, CommitIdentity]:
    if command_output(["git", "rev-parse", "HEAD"]).strip() != context.landed_sha:
        raise IntegrationError("Checkout does not match the landed main commit")
    landed = commit_identity(context.landed_sha)
    if len(landed.parent_shas) != 2:
        raise IntegrationError(
            "Same-tree integration requires a two-parent merge commit"
        )
    base_sha, head_sha = landed.parent_shas
    require_ancestor(base_sha, head_sha)
    head = commit_identity(head_sha)
    if landed.tree_sha != head.tree_sha:
        raise IntegrationError("Landed and PR-head trees differ")
    return landed, head


def list_pages(payload: object, label: str) -> list[dict]:
    if not isinstance(payload, list) or not payload:
        raise IntegrationError(f"Missing paginated {label}")
    records: list[dict] = []
    for page in payload:
        if not isinstance(page, list):
            raise IntegrationError(f"Malformed paginated {label}")
        if any(not isinstance(record, dict) for record in page):
            raise IntegrationError(f"Malformed {label}")
        records.extend(page)
    return records


def keyed_pages(payload: object, key: str, label: str) -> list[dict]:
    if not isinstance(payload, list) or not payload:
        raise IntegrationError(f"Missing paginated {label}")
    records: list[dict] = []
    total = None
    for page in payload:
        if not isinstance(page, dict) or not isinstance(page.get(key), list):
            raise IntegrationError(f"Malformed paginated {label}")
        count = page.get("total_count")
        if (
            type(count) is not int
            or count < 0
            or (total is not None and count != total)
        ):
            raise IntegrationError(f"Inconsistent {label} page totals")
        if any(not isinstance(record, dict) for record in page[key]):
            raise IntegrationError(f"Malformed {label}")
        total = count
        records.extend(page[key])
    if len(records) != total:
        raise IntegrationError(f"Incomplete paginated {label}")
    return records


def validate_ruleset(payload: object, context: Context) -> None:
    ruleset = _object(payload, "main ruleset")
    expected = {
        "id": context.ruleset_id,
        "target": "branch",
        "source_type": "Repository",
        "source": context.repository,
        "enforcement": "active",
        "bypass_actors": [],
    }
    if any(ruleset.get(key) != value for key, value in expected.items()):
        raise IntegrationError(
            "Main ruleset identity, enforcement, or bypass policy changed"
        )
    conditions = _object(ruleset.get("conditions"), "main ruleset conditions")
    refs = _object(conditions.get("ref_name"), "main ruleset ref conditions")
    if refs != {"exclude": [], "include": [MAIN_REF]}:
        raise IntegrationError("Main ruleset no longer targets only main")
    rules = ruleset.get("rules")
    if not isinstance(rules, list) or any(not isinstance(rule, dict) for rule in rules):
        raise IntegrationError("Malformed main ruleset rules")
    for required_type in ("deletion", "non_fast_forward", "pull_request"):
        if sum(rule.get("type") == required_type for rule in rules) != 1:
            raise IntegrationError(
                f"Main ruleset must contain one {required_type} rule"
            )
    status_rules = [
        rule for rule in rules if rule.get("type") == "required_status_checks"
    ]
    if len(status_rules) != 1:
        raise IntegrationError(
            "Main ruleset must contain one required-status-check rule"
        )
    parameters = _object(status_rules[0].get("parameters"), "required checks")
    if parameters.get("strict_required_status_checks_policy") is not True:
        raise IntegrationError("Main required checks are not strict")
    if parameters.get("do_not_enforce_on_create") is not False:
        raise IntegrationError("Main required checks do not apply on creation")
    checks = parameters.get("required_status_checks")
    if not isinstance(checks, list) or any(
        not isinstance(check, dict) for check in checks
    ):
        raise IntegrationError("Malformed required status checks")
    actual_checks = [
        (check.get("context"), check.get("integration_id")) for check in checks
    ]
    expected_checks = {(name, context.actions_app_id) for name in REQUIRED_CI_CONTEXTS}
    if (
        len(actual_checks) != len(expected_checks)
        or set(actual_checks) != expected_checks
    ):
        raise IntegrationError(
            "The exact GitHub Actions required-check inventory changed"
        )


def validate_associated_pull_requests(payload: object) -> int:
    pulls = list_pages(payload, "associated pull requests")
    if len(pulls) != 1:
        raise IntegrationError(
            "Landed commit must be associated with exactly one pull request"
        )
    return _positive_int(pulls[0].get("number"), "associated pull request number")


def validate_pull_request(
    payload: object,
    context: Context,
    landed: CommitIdentity,
    number: int,
) -> PullIdentity:
    pull = _object(payload, "pull request")
    base = _object(pull.get("base"), "pull request base")
    head = _object(pull.get("head"), "pull request head")
    base_repo = _object(base.get("repo"), "pull request base repository")
    head_repo = _object(head.get("repo"), "pull request head repository")
    base_sha, head_sha = landed.parent_shas
    expected = {
        "number": number,
        "state": "closed",
        "merged": True,
        "merge_commit_sha": context.landed_sha,
    }
    if any(pull.get(key) != value for key, value in expected.items()):
        raise IntegrationError("Pull request is not the exact merged integration")
    _text(pull.get("merged_at"), "pull request merge time")
    if (
        base.get("ref") != "main"
        or base.get("sha") != base_sha
        or base_repo.get("full_name") != context.repository
    ):
        raise IntegrationError("Pull request does not target the exact main parent")
    if head.get("sha") != head_sha:
        raise IntegrationError("Pull request head does not match the merge parent")
    return PullIdentity(
        number=number,
        base_sha=base_sha,
        head_sha=head_sha,
        head_ref=_text(head.get("ref"), "pull request head ref"),
        head_repository=_text(
            head_repo.get("full_name"), "pull request head repository"
        ),
    )


def validate_terminal_check(
    payload: object, context: Context, head_sha: str
) -> CiIdentity:
    checks = keyed_pages(payload, "check_runs", "terminal checks")
    if len(checks) != 1:
        raise IntegrationError("Expected exactly one latest ci-required check")
    check = checks[0]
    check_id = _positive_int(check.get("id"), "terminal check ID")
    app = _object(check.get("app"), "terminal check app")
    suite = _object(check.get("check_suite"), "terminal check suite")
    if (
        check.get("name") != "ci-required"
        or check.get("head_sha") != head_sha
        or check.get("status") != "completed"
        or check.get("conclusion") != "success"
        or app.get("id") != context.actions_app_id
    ):
        raise IntegrationError("Exact ci-required check is absent or unsuccessful")
    check_suite_id = _positive_int(suite.get("id"), "terminal check suite ID")
    pattern = (
        rf"https://github\.com/{re.escape(context.repository)}/actions/runs/"
        rf"([1-9][0-9]*)/job/{check_id}"
    )
    match = re.fullmatch(pattern, check.get("details_url", ""))
    if match is None:
        raise IntegrationError(
            "Terminal check does not identify its exact workflow run"
        )
    return CiIdentity(int(match.group(1)), 0, check_id, check_suite_id)


def validate_workflow_run(
    payload: object,
    context: Context,
    pull: PullIdentity,
    ci: CiIdentity,
) -> CiIdentity:
    run = _object(payload, "CI workflow run")
    repository = _object(run.get("repository"), "CI run repository")
    head_repository = _object(run.get("head_repository"), "CI run head repository")
    expected = {
        "id": ci.run_id,
        "name": "CI",
        "head_sha": pull.head_sha,
        "head_branch": pull.head_ref,
        "path": CI_WORKFLOW_PATH,
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "check_suite_id": ci.check_suite_id,
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise IntegrationError(
            "Terminal check is not bound to an exact successful PR CI run"
        )
    if (
        repository.get("full_name") != context.repository
        or head_repository.get("full_name") != pull.head_repository
    ):
        raise IntegrationError("CI workflow repository identity mismatch")
    attempt = _positive_int(run.get("run_attempt"), "CI run attempt")
    if ci.run_attempt not in {0, attempt}:
        raise IntegrationError("CI run attempt changed while evidence was read")
    return CiIdentity(ci.run_id, attempt, ci.check_id, ci.check_suite_id)


def attestation_from_job_log(data: bytes) -> object:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntegrationError("Malformed terminal job log") from exc
    prefix = "ci-required-attestation: "
    records = [
        line.split(prefix, maxsplit=1)[1]
        for line in text.splitlines()
        if prefix in line
    ]
    if len(records) != 1:
        raise IntegrationError("Terminal job log must contain one exact CI attestation")
    try:
        return json.loads(records[0])
    except json.JSONDecodeError as exc:
        raise IntegrationError("Malformed CI evidence attestation") from exc


def validate_attestation(
    payload: object,
    context: Context,
    landed: CommitIdentity,
    pull: PullIdentity,
    ci: CiIdentity,
) -> str:
    evidence = _object(payload, "CI evidence attestation")
    expected_keys = {
        "schema",
        "repository",
        "workflow_path",
        "event",
        "run_id",
        "run_attempt",
        "candidate_head_sha",
        "tested_sha",
        "tested_tree_sha",
        "tested_parent_shas",
        "pull_request",
    }
    if set(evidence) != expected_keys:
        raise IntegrationError("CI evidence attestation schema keys changed")
    tested_sha = _sha(evidence.get("tested_sha"), "attested tested commit")
    expected = {
        "schema": ATTESTATION_SCHEMA,
        "repository": context.repository,
        "workflow_path": CI_WORKFLOW_PATH,
        "event": "pull_request",
        "run_id": ci.run_id,
        "run_attempt": ci.run_attempt,
        "candidate_head_sha": pull.head_sha,
        "tested_tree_sha": landed.tree_sha,
        "tested_parent_shas": [pull.base_sha, pull.head_sha],
    }
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise IntegrationError("CI evidence is not bound to the landed PR tree")
    pull_evidence = _object(evidence.get("pull_request"), "attested pull request")
    if set(pull_evidence) != {"number", "base_ref", "base_sha", "head_ref", "head_sha"}:
        raise IntegrationError("Attested pull request schema keys changed")
    if pull_evidence != {
        "number": pull.number,
        "base_ref": "main",
        "base_sha": pull.base_sha,
        "head_ref": pull.head_ref,
        "head_sha": pull.head_sha,
    }:
        raise IntegrationError("CI evidence names a different pull request candidate")
    return tested_sha


def verify(environ: Mapping[str, str]) -> dict[str, object]:
    context = Context(
        repository=environ["GITHUB_REPOSITORY"],
        landed_sha=environ["GITHUB_SHA"],
        ref=environ["GITHUB_REF"],
        event=environ["GITHUB_EVENT_NAME"],
        ruleset_id=int(environ["PALETTE_MAIN_RULESET_ID"]),
        actions_app_id=int(environ["PALETTE_ACTIONS_APP_ID"]),
    )
    ruleset_token = environ["PALETTE_RULESET_TOKEN"]
    if not ruleset_token.strip():
        raise IntegrationError("Missing ruleset credential")
    landed, head = validate_landed_commit(context)
    ruleset_path = f"repos/{context.repository}/rulesets/{context.ruleset_id}"
    validate_ruleset(api_json(ruleset_path, token=ruleset_token), context)

    associated_path = (
        f"repos/{context.repository}/commits/{context.landed_sha}/pulls?per_page=100"
    )
    number = validate_associated_pull_requests(api_json(associated_path, paginate=True))
    pull = validate_pull_request(
        api_json(f"repos/{context.repository}/pulls/{number}"),
        context,
        landed,
        number,
    )
    if head.sha != pull.head_sha:
        raise IntegrationError("Loaded PR head does not match the merged head")

    checks_path = (
        f"repos/{context.repository}/commits/{pull.head_sha}/check-runs"
        f"?check_name=ci-required&app_id={context.actions_app_id}"
        "&filter=latest&per_page=100"
    )
    ci = validate_terminal_check(
        api_json(checks_path, paginate=True), context, pull.head_sha
    )
    run_path = f"repos/{context.repository}/actions/runs/{ci.run_id}"
    ci = validate_workflow_run(api_json(run_path), context, pull, ci)
    log_path = f"repos/{context.repository}/actions/jobs/{ci.check_id}/logs"
    tested_sha = validate_attestation(
        attestation_from_job_log(api_job_log(log_path)),
        context,
        landed,
        pull,
        ci,
    )

    # Fail if a new terminal check, rerun, or ruleset edit superseded the
    # accepted evidence while the proof was assembled.
    latest_ci = validate_terminal_check(
        api_json(checks_path, paginate=True),
        context,
        pull.head_sha,
    )
    if (
        latest_ci.run_id,
        latest_ci.check_id,
        latest_ci.check_suite_id,
    ) != (ci.run_id, ci.check_id, ci.check_suite_id):
        raise IntegrationError(
            "A newer ci-required check superseded the accepted evidence"
        )
    if validate_workflow_run(api_json(run_path), context, pull, ci) != ci:
        raise IntegrationError("CI run identity changed while evidence was read")
    validate_ruleset(api_json(ruleset_path, token=ruleset_token), context)
    return {
        "landed_sha": context.landed_sha,
        "head_sha": pull.head_sha,
        "tree_sha": landed.tree_sha,
        "pull_request_number": pull.number,
        "ci_run_id": ci.run_id,
        "ci_run_attempt": ci.run_attempt,
        "tested_sha": tested_sha,
        "ruleset_id": context.ruleset_id,
    }


def main(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    try:
        result = verify(env)
    except (IntegrationError, KeyError, ValueError) as exc:
        print(f"main-integration: FAIL: {exc}", file=sys.stderr)
        print(
            "main-integration: run full CI manually on the current exact main commit",
            file=sys.stderr,
        )
        return 1
    print(
        "main-integration: PASS: "
        f"landed={result['landed_sha']}, head={result['head_sha']}, "
        f"tree={result['tree_sha']}, pr={result['pull_request_number']}, "
        f"ci_run={result['ci_run_id']}/{result['ci_run_attempt']}, "
        f"tested={result['tested_sha']}, ruleset={result['ruleset_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
