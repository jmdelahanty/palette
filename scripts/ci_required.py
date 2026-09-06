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
RERUN_HELP = 'Use "Re-run all jobs"; evidence from partial/older attempts is not reused.'


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
        if not all(type(value) is int and value > 0 for value in (self.run_id, self.attempt)):
            raise GateError("Invalid run ID or attempt")
        if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (self.head_sha, self.tested_sha)):
            raise GateError("Expected full candidate and tested commit SHAs")
        if self.event not in {"push", "pull_request"}:
            raise GateError("Only push and pull_request CI are supported")
        if self.event == "push" and self.head_sha != self.tested_sha:
            raise GateError("Push head and tested commit must agree")


def required_names() -> tuple[str, ...]:
    return (*SINGLE_JOBS.values(), *(f"non-gpu tests (shard {i})" for i in range(SHARD_COUNT)))


def validate_needs(needs: object) -> None:
    if not isinstance(needs, dict) or set(needs) != {*SINGLE_JOBS, "tests"}:
        raise GateError("Dependency inventory does not match the CI contract")
    failed = [
        name for name, job in needs.items()
        if not isinstance(job, dict) or job.get("result") != "success"
    ]
    if failed:
        raise GateError(f"Dependencies not successful: {', '.join(sorted(failed))}")


def validate_run(run: object, context: Context) -> None:
    expected = {
        "id": context.run_id, "run_attempt": context.attempt,
        "head_sha": context.head_sha, "event": context.event,
        "path": ".github/workflows/ci.yml",
    }
    if not isinstance(run, dict):
        raise GateError("Run identity mismatch: malformed run")
    mismatches = [key for key, value in expected.items()
                  if type(run.get(key)) is not type(value) or run.get(key) != value]
    repository = run.get("repository")
    if not isinstance(repository, dict) or repository.get("full_name") != context.repository:
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
            "run_id": context.run_id, "run_attempt": context.attempt,
            "head_sha": context.head_sha, "workflow_name": "CI",
        }
        if any(type(job.get(key)) is not type(value) or job.get(key) != value
               for key, value in binding.items()):
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
        problems.append(f"Missing required jobs: {', '.join(sorted(missing))}. {RERUN_HELP}")
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
        if type(count) is not int or count < 0 or (total is not None and count != total):
            raise GateError("Inconsistent job page totals")
        total = count
        jobs.extend(page["jobs"])
    if len(jobs) != total:
        raise GateError("Incomplete paginated job evidence")
    return jobs


def command_output(args: list[str]) -> str:
    try:
        return subprocess.run(
            args, check=True, capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        # Never echo credential-bearing command output or environment variables.
        raise GateError(f"Unable to read CI evidence with {args[0]} ({type(exc).__name__})") from exc


def checked_out_sha() -> str:
    return command_output(["git", "rev-parse", "HEAD"]).strip()


def api_json(path: str, *, paginate: bool = False) -> object:
    args = ["gh", "api", "--hostname", "github.com",
            "-H", "Accept: application/vnd.github+json",
            "-H", "X-GitHub-Api-Version: 2022-11-28"]
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
            env["GITHUB_REPOSITORY"], int(env["GITHUB_RUN_ID"]),
            int(env["GITHUB_RUN_ATTEMPT"]), env["CI_REQUIRED_HEAD_SHA"],
            env["GITHUB_SHA"], env["GITHUB_EVENT_NAME"],
        )
        validate_needs(json.loads(env["CI_REQUIRED_NEEDS"]))
        if checked_out_sha() != context.tested_sha:
            raise GateError("Checkout does not match the workflow's tested commit")
        path = f"repos/{context.repository}/actions/runs/{context.run_id}"
        validate_run(api_json(path), context)
        pages = api_json(f"{path}/attempts/{context.attempt}/jobs?per_page=100", paginate=True)
        validate_jobs(jobs_from_pages(pages), context)
        # Refuse evidence if another attempt superseded this one during the reads.
        validate_run(api_json(path), context)
    except (GateError, KeyError, ValueError) as exc:
        print(f"ci-required: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"ci-required: PASS: all {len(required_names())} required jobs succeeded; "
          f"run={context.run_id}, attempt={context.attempt}, "
          f"head={context.head_sha}, tested={context.tested_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
