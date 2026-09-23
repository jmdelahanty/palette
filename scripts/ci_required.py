#!/usr/bin/env python3
"""Require exact-success CI evidence from one complete workflow run attempt.

Standard library only; the runner supplies git, gh and a read-only GH_TOKEN.
The inventory is guarded against workflow drift by test_ci_required.py.
Partial reruns deliberately fail: use GitHub's "Re-run all jobs" instead.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

SINGLE_JOBS = {
    "generated-artifacts": "generated artifacts",
    "import-boundaries": "import boundaries",
    "file-size-ratchet": "file-size ratchet",
    "zarr-open-mode-ratchet": "zarr open metadata modes",
    "observed-metadata-literals": "observed metadata literals",
    "contract-freshness": "active contract freshness",
    "quality": "package and collection",
}
SHARD_COUNT = 16
RERUN_HELP = (
    'Use "Re-run all jobs"; evidence from partial/older attempts is not reused.'
)
ATTESTATION_SCHEMA = "palette-ci-required-attestation-v1"


class GateError(ValueError):
    """Missing, unsuccessful, ambiguous or incorrectly bound CI evidence."""


@dataclass(frozen=True)
class Context:
    repository: str
    run_id: int
    attempt: int
    head_sha: str
    tested_sha: str
    event: str

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository):
            raise GateError("Invalid repository identity")
        if not all(
            type(value) is int and value > 0 for value in (self.run_id, self.attempt)
        ):
            raise GateError("Invalid run ID or attempt")
        if not all(
            re.fullmatch(r"[0-9a-f]{40}", sha)
            for sha in (self.head_sha, self.tested_sha)
        ):
            raise GateError("Expected full candidate and tested commit SHAs")
        if self.event not in {"pull_request", "workflow_dispatch"}:
            raise GateError("Only pull_request and workflow_dispatch CI are supported")
        if self.event == "workflow_dispatch" and self.head_sha != self.tested_sha:
            raise GateError("Workflow-dispatch head and tested commit must agree")


@dataclass(frozen=True)
class CommitIdentity:
    tree_sha: str
    parent_shas: tuple[str, ...]

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.tree_sha):
            raise GateError("Malformed tested commit tree")
        if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in self.parent_shas):
            raise GateError("Malformed tested commit parents")


def required_names() -> tuple[str, ...]:
    return (
        *SINGLE_JOBS.values(),
        *(f"non-gpu tests (shard {i})" for i in range(SHARD_COUNT)),
    )


def validate_needs(needs: object) -> None:
    if not isinstance(needs, dict) or set(needs) != {*SINGLE_JOBS, "tests"}:
        raise GateError("Dependency inventory does not match the CI contract")
    failed = [
        name
        for name, job in needs.items()
        if not isinstance(job, dict) or job.get("result") != "success"
    ]
    if failed:
        raise GateError(f"Dependencies not successful: {', '.join(sorted(failed))}")


def validate_run(run: object, context: Context) -> None:
    expected = {
        "id": context.run_id,
        "run_attempt": context.attempt,
        "head_sha": context.head_sha,
        "event": context.event,
        "path": ".github/workflows/ci.yml",
    }
    if not isinstance(run, dict):
        raise GateError("Run identity mismatch: malformed run")
    mismatches = [
        key
        for key, value in expected.items()
        if type(run.get(key)) is not type(value) or run.get(key) != value
    ]
    repository = run.get("repository")
    if (
        not isinstance(repository, dict)
        or repository.get("full_name") != context.repository
    ):
        mismatches.append("repository")
    if mismatches:
        raise GateError(f"Run identity mismatch: {', '.join(mismatches)}. {RERUN_HELP}")


def validate_jobs(jobs: list[dict], context: Context) -> None:
    expected = set(required_names())
    seen_names: set[str] = set()
    seen_ids: set[int] = set()
    failed = []
    for job in jobs:
        if not isinstance(job, dict):
            raise GateError("Malformed job evidence")
        name, job_id = job.get("name"), job.get("id")
        if not isinstance(name, str) or type(job_id) is not int or job_id <= 0:
            raise GateError("Malformed job name or ID")
        if name in seen_names or job_id in seen_ids:
            raise GateError(f"Duplicate job evidence: {name}")
        seen_names.add(name)
        seen_ids.add(job_id)
        binding = {
            "run_id": context.run_id,
            "run_attempt": context.attempt,
            "head_sha": context.head_sha,
            "workflow_name": "CI",
        }
        if any(
            type(job.get(key)) is not type(value) or job.get(key) != value
            for key, value in binding.items()
        ):
            raise GateError(f"Job identity mismatch: {name}. {RERUN_HELP}")
        # The currently executing gate cannot be its own successful prerequisite.
        if name == "ci-required":
            if job.get("status") != "in_progress" or job.get("conclusion") is not None:
                raise GateError("Unexpected terminal gate state")
            continue
        if name not in expected:
            raise GateError(f"Unexpected job: {name}; update the reviewed CI contract")
        if job.get("status") != "completed" or job.get("conclusion") != "success":
            failed.append(f"{name} ({job.get('status')}/{job.get('conclusion')})")
    missing = expected - seen_names
    problems = []
    if missing:
        problems.append(
            f"Missing required jobs: {', '.join(sorted(missing))}. {RERUN_HELP}"
        )
    if failed:
        problems.append(f"Required jobs not successful: {', '.join(failed)}")
    if problems:
        raise GateError("\n".join(problems))


def jobs_from_pages(pages: object) -> list[dict]:
    if not isinstance(pages, list) or not pages:
        raise GateError("Missing paginated job evidence")
    jobs, total = [], None
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("jobs"), list):
            raise GateError("Malformed job page")
        count = page.get("total_count")
        if (
            type(count) is not int
            or count < 0
            or (total is not None and count != total)
        ):
            raise GateError("Inconsistent job page totals")
        total = count
        jobs.extend(page["jobs"])
    if len(jobs) != total:
        raise GateError("Incomplete paginated job evidence")
    return jobs


def command_output(args: list[str]) -> str:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        # Never echo credential-bearing command output or environment variables.
        raise GateError(
            f"Unable to read CI evidence with {args[0]} ({type(exc).__name__})"
        ) from exc


def checked_out_sha() -> str:
    return command_output(["git", "rev-parse", "HEAD"]).strip()


def checked_out_commit() -> CommitIdentity:
    lines = command_output(["git", "cat-file", "-p", "HEAD"]).splitlines()
    headers = lines[: lines.index("")] if "" in lines else lines
    trees = [line.removeprefix("tree ") for line in headers if line.startswith("tree ")]
    parents = tuple(
        line.removeprefix("parent ") for line in headers if line.startswith("parent ")
    )
    if len(trees) != 1:
        raise GateError("Malformed tested commit object")
    return CommitIdentity(trees[0], parents)


def event_payload(path: str) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError("Unable to read the workflow event payload") from exc


def _object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise GateError(f"Malformed {label}")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise GateError(f"Malformed {label}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateError(f"Malformed {label}")
    return value


def build_attestation(
    context: Context,
    payload: object,
    commit: CommitIdentity,
    ref: str,
) -> dict:
    event = _object(payload, "workflow event payload")
    repository = _object(event.get("repository"), "event repository")
    if repository.get("full_name") != context.repository:
        raise GateError("Event repository does not match the workflow repository")

    pull_request = None
    if context.event == "pull_request":
        pull = _object(event.get("pull_request"), "pull request event")
        base = _object(pull.get("base"), "pull request base")
        head = _object(pull.get("head"), "pull request head")
        number = event.get("number")
        if type(number) is not int or number <= 0 or pull.get("number") != number:
            raise GateError("Malformed pull request number")
        base_sha = _sha(base.get("sha"), "pull request base SHA")
        head_sha = _sha(head.get("sha"), "pull request head SHA")
        if head_sha != context.head_sha:
            raise GateError("Event pull request head does not match the candidate")
        if commit.parent_shas != (base_sha, head_sha):
            raise GateError("Tested merge parents do not match the pull request")
        pull_request = {
            "number": number,
            "base_ref": _text(base.get("ref"), "pull request base ref"),
            "base_sha": base_sha,
            "head_ref": _text(head.get("ref"), "pull request head ref"),
            "head_sha": head_sha,
        }
    else:
        if ref != "refs/heads/main":
            raise GateError("Manual full CI fallback must run on main")

    return {
        "schema": ATTESTATION_SCHEMA,
        "repository": context.repository,
        "workflow_path": ".github/workflows/ci.yml",
        "event": context.event,
        "run_id": context.run_id,
        "run_attempt": context.attempt,
        "candidate_head_sha": context.head_sha,
        "tested_sha": context.tested_sha,
        "tested_tree_sha": commit.tree_sha,
        "tested_parent_shas": list(commit.parent_shas),
        "pull_request": pull_request,
    }


def api_json(path: str, *, paginate: bool = False) -> object:
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
    if paginate:
        args.extend(["--paginate", "--slurp"])
    try:
        return json.loads(command_output([*args, path]))
    except json.JSONDecodeError as exc:
        raise GateError("Malformed GitHub API JSON") from exc


def main(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    try:
        context = Context(
            env["GITHUB_REPOSITORY"],
            int(env["GITHUB_RUN_ID"]),
            int(env["GITHUB_RUN_ATTEMPT"]),
            env["CI_REQUIRED_HEAD_SHA"],
            env["GITHUB_SHA"],
            env["GITHUB_EVENT_NAME"],
        )
        validate_needs(json.loads(env["CI_REQUIRED_NEEDS"]))
        if checked_out_sha() != context.tested_sha:
            raise GateError("Checkout does not match the workflow's tested commit")
        path = f"repos/{context.repository}/actions/runs/{context.run_id}"
        validate_run(api_json(path), context)
        pages = api_json(
            f"{path}/attempts/{context.attempt}/jobs?per_page=100", paginate=True
        )
        validate_jobs(jobs_from_pages(pages), context)
        # Refuse evidence if another attempt superseded this one during the reads.
        validate_run(api_json(path), context)
        commit = checked_out_commit()
        attestation = build_attestation(
            context,
            event_payload(env["GITHUB_EVENT_PATH"]),
            commit,
            env["GITHUB_REF"],
        )
    except (GateError, KeyError, ValueError) as exc:
        print(f"ci-required: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "ci-required-attestation: "
        + json.dumps(attestation, sort_keys=True, separators=(",", ":"))
    )
    print(
        f"ci-required: PASS: all {len(required_names())} required jobs succeeded; "
        f"run={context.run_id}, attempt={context.attempt}, "
        f"head={context.head_sha}, tested={context.tested_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
