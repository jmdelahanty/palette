"""Submit one LSF job over SSH, then wait on NFS evidence.

Snakemake runs only on ws1 (its env is not visible to compute nodes), so a
cluster step is a local rule that calls :func:`run_attempt`. Login-node
contact is limited to (docs/design/2026-10-07-workflow-runner §5.1):

- one ``bsub`` per attempt;
- a shared, flock-guarded ``bjobs`` cache refreshed at most once per
  ``bjobs_min_interval_s`` (>= 300 s) across every waiting step, consulted
  only for jobs with no fresh NFS evidence (pending, or silent).

A running job proves liveness by touching ``heartbeat`` on NFS; it reports
its result as ``result.json`` plus ``exit_code`` (both renamed into place).
An LSF output footer with no ``exit_code`` means the job died: failed.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

from fisheye.cluster.lsf.backend import (
    build_ssh_bsub_runner,
    parse_bsub_job_id,
    shell_join,
)
from fisheye.flow.config import LsfSettings

ATTEMPT_SCHEMA = "palette.flow.lsf_attempt.v1"
ATTEMPT_PLACEHOLDER = "{ATTEMPT}"
LSF_FOOTER = "Resource usage summary"
ENDED_STATES = frozenset({"DONE", "EXIT"})
LIVE_STATES = frozenset({"PEND", "RUN", "PSUSP", "USUSP", "SSUSP", "PROV", "WAIT"})
EXIT_LOST = 1

Runner = Callable[..., Any]
Clock = Callable[[], float]


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json_atomic(path: Path, document: dict) -> None:
    _write_atomic(path, json.dumps(document, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# Shared bjobs cache: the only place LSF state is queried.


def build_ssh_bjobs_runner(submit_host: str) -> Runner:
    """Run exactly ``bjobs -a -noheader -o 'jobid stat'`` on the submit host."""

    def runner() -> subprocess.CompletedProcess[str]:
        remote = shell_join(["bjobs", "-a", "-noheader", "-o", "jobid stat"])
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", submit_host, remote],
            text=True,
            capture_output=True,
            timeout=120,
        )

    return runner


def parse_bjobs(stdout: str) -> dict[str, str]:
    jobs: dict[str, str] = {}
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit():
            jobs[fields[0]] = fields[1].upper()
    return jobs


class BjobsCache:
    """``<state_dir>/bjobs_cache.json``, refreshed at most once per interval.

    The interval is enforced across processes: refresh happens under an
    exclusive flock and only when the cached ``fetched_at`` is older than the
    interval, so any number of waiting steps or status readers share one
    ``bjobs`` call per interval.
    """

    def __init__(
        self,
        state_dir: Path,
        *,
        min_interval_s: int,
        query: Runner,
        clock: Clock = time.time,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.min_interval_s = min_interval_s
        self.query = query
        self.clock = clock
        self.path = self.state_dir / "bjobs_cache.json"

    def read(self) -> dict[str, Any] | None:
        try:
            document = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        return document if isinstance(document, dict) else None

    def get(self, *, refresh: bool = True) -> dict[str, Any] | None:
        cached = self.read()
        if not refresh or not self._stale(cached):
            return cached
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_dir / "bjobs.lock", "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            cached = self.read()  # another process may have refreshed meanwhile
            if not self._stale(cached):
                return cached
            attempted_at = self.clock()
            try:
                completed = self.query()
                ok = completed.returncode == 0 or bool(parse_bjobs(completed.stdout))
                error = None if ok else (completed.stderr or "").strip()[-500:]
                jobs = parse_bjobs(completed.stdout) if ok else {}
            except (OSError, subprocess.SubprocessError) as exc:
                ok, error, jobs = False, str(exc), {}
            # A failed query still consumes the interval: never retry-spam.
            document = {
                "fetched_at": attempted_at,
                "ok": ok,
                "error": error,
                "jobs": jobs if ok else (cached or {}).get("jobs", {}),
                "jobs_fetched_at": attempted_at if ok else (cached or {}).get("jobs_fetched_at"),
            }
            write_json_atomic(self.path, document)
            return document

    def _stale(self, cached: dict[str, Any] | None) -> bool:
        if not cached or not isinstance(cached.get("fetched_at"), (int, float)):
            return True
        return self.clock() - cached["fetched_at"] >= self.min_interval_s


# --------------------------------------------------------------------------
# One attempt: job script, submission, wait.


@dataclass(frozen=True)
class AttemptResult:
    attempt_dir: Path
    job_id: str | None
    exit_code: int
    document: dict | None
    reason: str


def job_script(*, remote_repo: Path, attempt_dir: Path, argv: Sequence[str]) -> str:
    """The LSF job: heartbeat on NFS, run ``scripts/py <argv>``, publish result."""

    a = shell_join([str(attempt_dir)])
    return f"""#!/usr/bin/env bash
set -uo pipefail
ATTEMPT={a}
cd {shell_join([str(remote_repo)])} || exit 70
( exec </dev/null >/dev/null 2>&1; while :; do touch "$ATTEMPT/heartbeat"; sleep 60; done ) &
HEARTBEAT_PID=$!
trap 'pkill -P "$HEARTBEAT_PID" 2>/dev/null; kill "$HEARTBEAT_PID" 2>/dev/null' EXIT
scripts/py {shell_join(list(argv))} >"$ATTEMPT/.result.json.tmp" 2>"$ATTEMPT/payload.err"
rc=$?
mv -f "$ATTEMPT/.result.json.tmp" "$ATTEMPT/result.json"
printf '%s\\n' "$rc" >"$ATTEMPT/.exit_code.tmp"
mv -f "$ATTEMPT/.exit_code.tmp" "$ATTEMPT/exit_code"
exit "$rc"
"""


def next_attempt_dir(step_dir: Path) -> Path:
    numbers = [
        int(path.name.split("-", 1)[1])
        for path in step_dir.glob("attempt-*")
        if path.name.split("-", 1)[1].isdigit()
    ]
    return step_dir / f"attempt-{max(numbers, default=0) + 1}"


def latest_attempt_dir(step_dir: Path) -> Path | None:
    candidates = [
        path for path in step_dir.glob("attempt-*") if path.name.split("-", 1)[1].isdigit()
    ]
    return max(candidates, key=lambda p: int(p.name.split("-", 1)[1]), default=None)


def _revalidate(directory: Path) -> None:
    """Open the directory so the NFS client revalidates it.

    ws1 caches negative lookups for up to about a minute, so a file a cluster
    node just created can look missing to a plain stat. Listing the parent
    refreshes it (found by the synthetic intake trial, 2026-10-09).
    """

    try:
        os.listdir(directory)
    except OSError:
        pass


def read_exit(attempt_dir: Path) -> tuple[int | None, dict | None]:
    _revalidate(attempt_dir)
    try:
        code = int((attempt_dir / "exit_code").read_text().strip())
    except (OSError, ValueError):
        return None, None
    try:
        document = json.loads((attempt_dir / "result.json").read_text())
    except (OSError, ValueError):
        document = None
    return code, document if isinstance(document, dict) else None


def footer_present(attempt_dir: Path) -> bool:
    try:
        return LSF_FOOTER in (attempt_dir / "lsf.out").read_text(errors="replace")
    except OSError:
        return False


def submit(
    *,
    attempt_dir: Path,
    job_name: str,
    remote_repo: Path,
    argv: Sequence[str],
    settings: LsfSettings,
    bsub_runner: Runner,
    clock: Clock = time.time,
) -> str:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    # ``{ATTEMPT}`` in argv names this attempt's directory (e.g. a fresh run dir).
    argv = [str(arg).replace(ATTEMPT_PLACEHOLDER, str(attempt_dir)) for arg in argv]
    script = attempt_dir / "job.sh"
    script.write_text(job_script(remote_repo=remote_repo, attempt_dir=attempt_dir, argv=argv))
    command = ["bsub", "-J", job_name, "-n", str(settings.ncores), "-W", settings.walltime,
               "-R", f"rusage[mem={settings.mem_gb}G]",
               "-oo", str(attempt_dir / "lsf.out"), "-eo", str(attempt_dir / "lsf.err")]
    if settings.queue:
        command += ["-q", settings.queue]
    command += ["bash", str(script)]
    submitted_at = clock()
    completed = bsub_runner(command, cwd=attempt_dir)
    stdout, stderr = getattr(completed, "stdout", ""), getattr(completed, "stderr", "")
    if getattr(completed, "returncode", 1) != 0:
        raise RuntimeError(f"bsub failed ({completed.returncode}): {stderr.strip()[-500:]}")
    job_id = parse_bsub_job_id(stdout, stderr)
    write_json_atomic(attempt_dir / "submission.json", {
        "schema": ATTEMPT_SCHEMA,
        "job_id": job_id,
        "job_name": job_name,
        "submitted_at": submitted_at,
        "remote_repo": str(remote_repo),
        "argv": list(argv),
        "bsub": command,
    })
    return job_id


def wait(
    attempt_dir: Path,
    *,
    settings: LsfSettings,
    cache: BjobsCache,
    clock: Clock = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> AttemptResult:
    submission = json.loads((attempt_dir / "submission.json").read_text())
    job_id = str(submission["job_id"])
    submitted_at = float(submission["submitted_at"])
    ended_seen_at: float | None = None
    while True:
        code, document = read_exit(attempt_dir)
        if code is not None:
            return AttemptResult(attempt_dir, job_id, code, document, "job reported")
        if footer_present(attempt_dir):
            return AttemptResult(attempt_dir, job_id, EXIT_LOST, None,
                                 "LSF job ended without an exit record (killed or crashed)")
        now = clock()
        if now - submitted_at > settings.max_wait_s:
            return AttemptResult(attempt_dir, job_id, EXIT_LOST, None,
                                 f"stopped waiting after {settings.max_wait_s}s (job may still run)")
        if not _heartbeat_fresh(attempt_dir, now, settings.heartbeat_stale_s):
            snapshot = cache.get()
            state = ((snapshot or {}).get("jobs") or {}).get(job_id)
            jobs_at = (snapshot or {}).get("jobs_fetched_at")
            if state in ENDED_STATES:
                # Output files can trail the scheduler over NFS: one grace poll.
                if ended_seen_at is None:
                    ended_seen_at = now
                elif now - ended_seen_at >= settings.poll_s:
                    return AttemptResult(attempt_dir, job_id, EXIT_LOST, None,
                                         f"LSF reports {state} but no exit record appeared")
            elif (
                state is None
                and isinstance(jobs_at, (int, float))
                and jobs_at - submitted_at > settings.bjobs_min_interval_s
            ):
                return AttemptResult(attempt_dir, job_id, EXIT_LOST, None,
                                     "LSF no longer knows the job and it left no evidence")
        sleep(settings.poll_s)


def _heartbeat_fresh(attempt_dir: Path, now: float, stale_s: int) -> bool:
    try:
        return now - (attempt_dir / "heartbeat").stat().st_mtime < stale_s
    except OSError:
        return False


def attachable(attempt_dir: Path | None) -> bool:
    """A submitted attempt with no exit record and no LSF footer: wait on it."""

    if attempt_dir is None or not (attempt_dir / "submission.json").is_file():
        return False
    code, _ = read_exit(attempt_dir)
    return code is None and not footer_present(attempt_dir)


def run_attempt(
    step_dir: Path,
    *,
    job_name: str,
    remote_repo: Path,
    argv: Sequence[str],
    settings: LsfSettings,
    cache: BjobsCache,
    bsub_runner: Runner | None = None,
    clock: Clock = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> AttemptResult:
    """Attach to a live earlier attempt, or submit a new one; then wait."""

    step_dir.mkdir(parents=True, exist_ok=True)
    latest = latest_attempt_dir(step_dir)
    if attachable(latest):
        print(f"attaching to {latest}", file=sys.stderr)
        attempt_dir = latest
    else:
        attempt_dir = next_attempt_dir(step_dir)
        runner = bsub_runner or build_ssh_bsub_runner(settings.submit_host)
        job_id = submit(attempt_dir=attempt_dir, job_name=job_name, remote_repo=remote_repo,
                        argv=argv, settings=settings, bsub_runner=runner, clock=clock)
        print(f"submitted LSF job {job_id} ({attempt_dir})", file=sys.stderr)
    return wait(attempt_dir, settings=settings, cache=cache, clock=clock, sleep=sleep)


__all__ = [
    "ATTEMPT_PLACEHOLDER",
    "AttemptResult",
    "BjobsCache",
    "attachable",
    "build_ssh_bjobs_runner",
    "job_script",
    "latest_attempt_dir",
    "parse_bjobs",
    "run_attempt",
    "submit",
    "wait",
    "write_json_atomic",
]
