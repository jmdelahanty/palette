"""Intake steps for ``workflows/intake.smk`` (runner design §6).

``python -m fisheye.flow.intake <plan|import|register|status> --config RUNNER.json``

Every Palette fact comes from the ``python -m fisheye.intake`` CLI, run from
a pinned deployment's ``scripts/py``: discovery and probes from the ws1 ops
deployment, imports on LSF from ``lsf_repo``, and registration from the ws1
deployment at the import receipt's ``producer_git_sha`` (§5.4).

Exit codes mirror the intake contract: 0 done (sentinel written), 65
refused (a ``refused.json`` hold is written; the operator clears it), 75
held by another live job, 1 retryable. The cron tick is the retry loop:
Snakemake runs with ``--retries 0 --keep-going``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

from fisheye.flow.config import FlowConfigError, IntakeFlowConfig, load_config
from fisheye.flow.lsf import (
    ATTEMPT_PLACEHOLDER,
    BjobsCache,
    build_ssh_bjobs_runner,
    latest_attempt_dir,
    read_exit,
    run_attempt,
    write_json_atomic,
)

EXIT_DONE, EXIT_FAILED, EXIT_REFUSED, EXIT_HELD = 0, 1, 65, 75
SENTINEL_SCHEMA = "palette.flow_job_done.v1"
REFUSAL_SCHEMA = "palette.flow_job_refused.v1"
FAILURES_SCHEMA = "palette.flow_job_failures.v1"
PLAN_SCHEMA = "palette.flow.intake_plan.v1"
PROBE_SCHEMAS = {
    "import": "palette.intake.probe_import.v",
    "register": "palette.intake.probe_register.v",
}

Exec = Callable[[Sequence[str]], subprocess.CompletedProcess]


def _exec(argv: Sequence[str]) -> subprocess.CompletedProcess:
    # stderr streams through to the Snakemake log; stdout is the JSON document.
    return subprocess.run(list(argv), stdout=subprocess.PIPE, stderr=None, text=True)


def intake_cli(deployment: Path, *args: str) -> list[str]:
    return [str(deployment / "scripts" / "py"), "-m", "fisheye.intake", *args]


def sentinel_path(config: IntakeFlowConfig, sha: str, step: str) -> Path:
    return config.delivery_dir(sha) / f"{step}.done.json"


def refusal_path(config: IntakeFlowConfig, sha: str) -> Path:
    return config.delivery_dir(sha) / "refused.json"


def failures_path(config: IntakeFlowConfig, sha: str) -> Path:
    return config.delivery_dir(sha) / "failures.json"


def _read_failures(config: IntakeFlowConfig, sha: str) -> dict:
    try:
        document = json.loads(failures_path(config, sha).read_text())
    except (OSError, ValueError):
        return {"schema": FAILURES_SCHEMA, "steps": {}}
    return document if isinstance(document, dict) and isinstance(document.get("steps"), dict) \
        else {"schema": FAILURES_SCHEMA, "steps": {}}


def _reset_failures(config: IntakeFlowConfig, sha: str, step: str) -> None:
    failures = _read_failures(config, sha)
    if failures["steps"].pop(step, None) is not None:
        write_json_atomic(failures_path(config, sha), failures)


def record_failure(config: IntakeFlowConfig, sha: str, step: str, reason: str, *, extra: dict | None = None) -> int:
    """Count one retryable failure; at the cap, hold the delivery for the operator.

    The cron tick is the retry loop, so without a cap a failure that never
    clears would resubmit work every tick. Deterministic refusals already exit
    65 from fisheye.intake; this is the backstop for everything else.
    """

    config.delivery_dir(sha).mkdir(parents=True, exist_ok=True)
    failures = _read_failures(config, sha)
    entry = failures["steps"].get(step) or {"count": 0}
    entry = {"count": int(entry.get("count", 0)) + 1, "last_reason": reason[-1000:],
             "last_at": time.time(), **(extra or {})}
    failures["steps"][step] = entry
    write_json_atomic(failures_path(config, sha), failures)
    if entry["count"] < config.max_consecutive_failures:
        return EXIT_FAILED
    write_json_atomic(refusal_path(config, sha), {
        "schema": REFUSAL_SCHEMA,
        "step": step,
        "snapshot_sha": sha,
        "recorded_at": time.time(),
        "retry_cap": True,
        "result": {"error": "retry_cap_reached",
                   "message": f"{entry['count']} consecutive {step} failures; last: {reason[-500:]}"},
        "clear": f"delete this file (and failures.json) after resolving; the next tick retries {sha}",
    })
    print(f"{step} {sha}: held after {entry['count']} consecutive failures", file=sys.stderr)
    return EXIT_REFUSED


def _git_head(deployment: Path, runner: Exec) -> str | None:
    completed = runner(["git", "-C", str(deployment), "rev-parse", "HEAD"])
    return completed.stdout.strip() if completed.returncode == 0 else None


def verdict_ok(document: Any, step: str, sha: str) -> bool:
    return (
        isinstance(document, dict)
        and str(document.get("schema", "")).startswith(PROBE_SCHEMAS[step])
        and document.get("verdict") is True
        and document.get("snapshot_sha") == sha
        and isinstance(document.get("evidence_digest"), str)
    )


def record_outcome(
    config: IntakeFlowConfig,
    sha: str,
    step: str,
    code: int,
    document: dict | None,
    *,
    deployment: Path,
    commit: str | None,
    extra: dict | None = None,
) -> int:
    """Turn one ``fisheye.intake`` result into a sentinel, a hold, or a code."""

    delivery = config.delivery_dir(sha)
    delivery.mkdir(parents=True, exist_ok=True)
    if code == EXIT_DONE:
        if not verdict_ok(document, step, sha):
            print(f"{step}: exit 0 without a true {step} probe for {sha}; not done", file=sys.stderr)
            return record_failure(config, sha, step, "exit 0 without a true probe")
        write_json_atomic(sentinel_path(config, sha, step), {
            "schema": SENTINEL_SCHEMA,
            "step": step,
            "snapshot_sha": sha,
            "evidence_digest": document["evidence_digest"],
            "producer_git_sha": document.get("producer_git_sha"),
            "deployment": str(deployment),
            "deployment_commit": commit,
            "recorded_at": time.time(),
            "probe": document,
            **(extra or {}),
        })
        _reset_failures(config, sha, step)
        return EXIT_DONE
    if code == EXIT_REFUSED:
        write_json_atomic(refusal_path(config, sha), {
            "schema": REFUSAL_SCHEMA,
            "step": step,
            "snapshot_sha": sha,
            "deployment": str(deployment),
            "deployment_commit": commit,
            "recorded_at": time.time(),
            "result": document,
            "clear": f"delete this file after resolving; the next tick retries {sha}",
            **(extra or {}),
        })
        return EXIT_REFUSED
    if code == EXIT_HELD:
        return EXIT_HELD  # another live job holds it: not a failure
    reason = (document or {}).get("error") if isinstance(document, dict) else None
    return record_failure(config, sha, step, f"exit {code}: {reason or 'no result document'}",
                          extra={"last_exit_code": code})


# --------------------------------------------------------------------------
# plan: which deliveries this tick should drive.


def plan(config: IntakeFlowConfig, *, runner: Exec = _exec) -> dict:
    completed = runner(intake_cli(
        config.ops_deployment, "discover",
        "--staging-dir", str(config.staging_dir),
        "--destination-root", str(config.destination_root),
        "--registry", str(config.registry),
    ))
    try:
        found = json.loads(completed.stdout)
    except ValueError as exc:
        raise RuntimeError(f"discover returned no JSON (exit {completed.returncode})") from exc
    if completed.returncode != EXIT_DONE:
        raise RuntimeError(f"discover failed (exit {completed.returncode}): {found.get('error')}")
    drive, held, skipped = [], [], []
    for target in found.get("targets", []):
        sha = target["snapshot_sha"]
        if refusal_path(config, sha).exists():
            held.append({"snapshot_sha": sha, "why": "refused; see refused.json"})
        elif target.get("legacy_mode"):
            skipped.append({"snapshot_sha": sha, "why": "legacy job-mode admission; manual"})
        elif target.get("register_recorded") is None and target.get("import_recorded"):
            skipped.append({"snapshot_sha": sha, "why": "registry state unknown this tick"})
        else:
            drive.append(sha)
    return {
        "schema": PLAN_SCHEMA,
        "planned_at": time.time(),
        "drive": sorted(drive),
        "held": held,
        "skipped": skipped,
        "registry_error": found.get("registry_error"),
        "refused_markers": found.get("refused_markers", []),
        "unreadable_states": found.get("unreadable_states", []),
        "legacy_marker_count": len(found.get("legacy_markers", [])),
    }


# --------------------------------------------------------------------------
# import: probe locally if the durable state already says complete,
# otherwise run import-delivery on LSF.


def import_step(
    config: IntakeFlowConfig,
    sha: str,
    *,
    runner: Exec = _exec,
    cache: BjobsCache | None = None,
    bsub_runner=None,
    clock=time.time,
    sleep=time.sleep,
) -> int:
    if refusal_path(config, sha).exists():
        return EXIT_REFUSED
    ops_commit = _git_head(config.ops_deployment, runner)
    probe = runner(intake_cli(
        config.ops_deployment, "probe-import", sha,
        "--destination-root", str(config.destination_root),
    ))
    try:
        probed = json.loads(probe.stdout)
    except ValueError:
        probed = None
    if probe.returncode == EXIT_DONE and verdict_ok(probed, "import", sha):
        # Already imported (e.g. by the cron path, or a lost sentinel).
        return record_outcome(config, sha, "import", EXIT_DONE, probed,
                              deployment=config.ops_deployment, commit=ops_commit,
                              extra={"source": "probe"})
    step_dir = config.delivery_dir(sha) / "import"
    argv = ["-m", "fisheye.intake", "import-delivery", sha,
            "--run-dir", f"{ATTEMPT_PLACEHOLDER}/run",
            "--destination-root", str(config.destination_root)]
    cache = cache or BjobsCache(config.lsf_state_dir,
                                min_interval_s=config.lsf.bjobs_min_interval_s,
                                query=build_ssh_bjobs_runner(config.lsf.submit_host),
                                clock=clock)
    try:
        result = run_attempt(
            step_dir,
            job_name=f"palette_flow_import_{sha[:16]}",
            remote_repo=config.lsf_repo,
            argv=argv,
            settings=config.lsf,
            cache=cache,
            bsub_runner=bsub_runner,
            clock=clock,
            sleep=sleep,
        )
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"import {sha}: could not submit or wait: {exc}", file=sys.stderr)
        return record_failure(config, sha, "import", f"submission: {exc}")
    print(f"import {sha}: exit {result.exit_code} ({result.reason})", file=sys.stderr)
    lsf_commit = _git_head(config.lsf_repo, runner)
    return record_outcome(config, sha, "import", result.exit_code, result.document,
                          deployment=config.lsf_repo, commit=lsf_commit,
                          extra={"lsf_job_id": result.job_id,
                                 "attempt_dir": str(result.attempt_dir)})


# --------------------------------------------------------------------------
# register: from the ws1 deployment at the receipt's producer commit.


def deployment_for_commit(config: IntakeFlowConfig, commit: str, *, runner: Exec = _exec) -> Path | None:
    for candidate in sorted(config.deployments_root.glob(f"ops-{commit[:8]}*")):
        if candidate.is_dir() and _git_head(candidate, runner) == commit:
            return candidate
    return None


def register_step(config: IntakeFlowConfig, sha: str, *, runner: Exec = _exec) -> int:
    if refusal_path(config, sha).exists():
        return EXIT_REFUSED
    try:
        imported = json.loads(sentinel_path(config, sha, "import").read_text())
    except (OSError, ValueError):
        print(f"register {sha}: no import sentinel", file=sys.stderr)
        return record_failure(config, sha, "register", "no import sentinel")
    commit = imported.get("producer_git_sha")
    if not isinstance(commit, str) or len(commit) != 40:
        print(f"register {sha}: import sentinel has no producer_git_sha", file=sys.stderr)
        return record_failure(config, sha, "register", "import sentinel has no producer_git_sha")
    deployment = deployment_for_commit(config, commit, runner=runner)
    if deployment is None:
        # An operator incident, not a refusal: it clears once the deployment exists.
        print(
            f"register {sha}: no ws1 deployment at producer commit {commit} under "
            f"{config.deployments_root} (create ops-{commit[:8]} with the deploy helper)",
            file=sys.stderr,
        )
        return record_failure(config, sha, "register", f"no ws1 deployment at producer commit {commit}")
    args = ["register-delivery", sha,
            "--config", str(config.registrar_config),
            "--destination-root", str(config.destination_root)]
    if config.allow_synthetic_isolated_registry:
        args.append("--allow-synthetic-isolated-registry")
    completed = runner(intake_cli(deployment, *args))
    try:
        document = json.loads(completed.stdout)
    except ValueError:
        document = None
    print(f"register {sha}: exit {completed.returncode} from {deployment}", file=sys.stderr)
    return record_outcome(config, sha, "register", completed.returncode, document,
                          deployment=deployment, commit=commit)


# --------------------------------------------------------------------------
# pin-check: may the runner move to another deployment now?

PIN_CHECK_SCHEMA = "palette.flow.intake_pin_check.v1"
_MIGRATION_PROBE = (
    "import fisheye.registry.migrations as m\n"
    "v = getattr(m, 'LATEST_MIGRATION_VERSION', None)\n"
    "if v is None: v = m.MIGRATION_METHODS[-1][0]\n"
    "print(int(v))\n"
)


def latest_migration(deployment: Path, *, runner: Exec = _exec) -> int:
    """The newest registry migration a deployment's code knows."""

    completed = runner([str(deployment / "scripts" / "py"), "-c", _MIGRATION_PROBE])
    try:
        return int(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"cannot read the registry migration of {deployment}") from exc


def _waiting_deliveries(config: IntakeFlowConfig, *, runner: Exec) -> dict[str, str | None]:
    """Deliveries imported but not yet registered -> their producer commit.

    Both paths count: the runner's own sentinels, and discover's view of
    deliveries the cron path imported.
    """

    waiting: dict[str, str | None] = {}
    root = config.intake_root
    for delivery in (p for p in root.glob("*") if p.is_dir()) if root.is_dir() else []:
        imported = sentinel_path(config, delivery.name, "import")
        if imported.is_file() and not sentinel_path(config, delivery.name, "register").is_file():
            try:
                waiting[delivery.name] = json.loads(imported.read_text()).get("producer_git_sha")
            except ValueError:
                waiting[delivery.name] = None
    completed = runner(intake_cli(
        config.ops_deployment, "discover",
        "--staging-dir", str(config.staging_dir),
        "--destination-root", str(config.destination_root),
        "--registry", str(config.registry),
    ))
    found = json.loads(completed.stdout)
    if found.get("registry_error"):
        raise RuntimeError(f"registry unreadable, cannot tell what is waiting: {found['registry_error']}")
    for target in found.get("targets", []):
        if target.get("import_recorded") and target.get("register_recorded") is not True:
            waiting.setdefault(target["snapshot_sha"], target.get("producer_git_sha"))
    return waiting


def pin_check(
    config: IntakeFlowConfig,
    target_ops: Path,
    target_lsf: Path,
    *,
    runner: Exec = _exec,
) -> tuple[int, dict]:
    """Refuse a pin move that would register old imports after a migration.

    Registration must run at each import's producer commit (§5.4). If the
    target pin adds registry migrations, new code migrates the live registry
    at its first registration; deliveries imported by older code would then
    register through code that predates those migrations (refused by the
    gateway's schema guard, or missing the new projections). So: drain first.
    """

    ops_commit = _git_head(target_ops, runner)
    lsf_commit = _git_head(target_lsf, runner)
    problems: list[str] = []
    if not ops_commit or not lsf_commit or ops_commit != lsf_commit:
        problems.append(f"target ws1 and LSF deployments are not one commit ({ops_commit} vs {lsf_commit})")
    current = latest_migration(config.ops_deployment, runner=runner)
    target = latest_migration(target_ops, runner=runner)
    blocking = []
    if target > current:
        for sha, producer in sorted(_waiting_deliveries(config, runner=runner).items()):
            deployment = deployment_for_commit(config, producer, runner=runner) if producer else None
            known = latest_migration(deployment, runner=runner) if deployment else None
            if known is None or known < target:
                blocking.append({"snapshot_sha": sha, "producer_git_sha": producer,
                                 "producer_migration": known})
    if blocking:
        problems.append(f"{len(blocking)} imported delivery(ies) must register before moving to "
                        f"migration {target}")
    report = {
        "schema": PIN_CHECK_SCHEMA,
        "target_ops": str(target_ops),
        "target_lsf": str(target_lsf),
        "target_commit": ops_commit,
        "current_migration": current,
        "target_migration": target,
        "adds_migrations": target > current,
        "blocking": blocking,
        "problems": problems,
        "safe": not problems,
    }
    return (EXIT_DONE if not problems else EXIT_REFUSED), report


# --------------------------------------------------------------------------
# status: NFS only; never contacts a login node.


def status_rows(config: IntakeFlowConfig) -> list[dict]:
    rows = []
    cache = BjobsCache(config.lsf_state_dir, min_interval_s=config.lsf.bjobs_min_interval_s,
                       query=lambda: None).read() or {}
    jobs = cache.get("jobs") or {}
    root = config.intake_root
    for delivery in sorted(p for p in root.glob("*") if p.is_dir()) if root.is_dir() else []:
        sha = delivery.name
        row: dict[str, Any] = {"snapshot_sha": sha, "import": "-", "register": "-", "detail": ""}
        for step in ("import", "register"):
            if sentinel_path(config, sha, step).is_file():
                row[step] = "done"
        refused = refusal_path(config, sha)
        if refused.is_file():
            try:
                hold = json.loads(refused.read_text())
            except ValueError:
                hold = {}
            row[hold.get("step", "import")] = "REFUSED"
            result = hold.get("result") or {}
            row["detail"] = str(result.get("error") or result.get("message") or "")[:80]
        attempt = latest_attempt_dir(delivery / "import")
        if attempt is not None and row["import"] not in ("done", "REFUSED"):
            code, _ = read_exit(attempt)
            try:
                job_id = json.loads((attempt / "submission.json").read_text())["job_id"]
            except (OSError, ValueError, KeyError):
                job_id = "?"
            lsf_state = jobs.get(str(job_id), "?")
            row["import"] = f"exit {code}" if code is not None else f"{attempt.name} {lsf_state}"
            row["detail"] = row["detail"] or f"job {job_id} {attempt}"
        rows.append(row)
    return rows


def _print_status(config: IntakeFlowConfig) -> None:
    rows = status_rows(config)
    cache = BjobsCache(config.lsf_state_dir, min_interval_s=config.lsf.bjobs_min_interval_s,
                       query=lambda: None).read()
    if cache:
        age = int(time.time() - cache.get("fetched_at", 0))
        print(f"LSF state cache: {age}s old, ok={cache.get('ok')}")
    if not rows:
        print(f"no deliveries under {config.intake_root}")
    for row in rows:
        print(f"{row['snapshot_sha'][:16]}  import={row['import']:<18} "
              f"register={row['register']:<8} {row['detail']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fisheye.flow.intake")
    parser.add_argument("command", choices=("plan", "import", "register", "status", "pin-check"))
    parser.add_argument("snapshot_sha", nargs="?")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--to-ops", type=Path, help="pin-check: target ws1 deployment")
    parser.add_argument("--to-lsf", type=Path, help="pin-check: target LSF deployment")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except FlowConfigError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if args.command == "plan":
        print(json.dumps(plan(config), indent=2, sort_keys=True))
        return EXIT_DONE
    if args.command == "status":
        _print_status(config)
        return EXIT_DONE
    if args.command == "pin-check":
        if not (args.to_ops and args.to_lsf):
            parser.error("pin-check needs --to-ops and --to-lsf")
        code, report = pin_check(config, args.to_ops, args.to_lsf)
        print(json.dumps(report, indent=2, sort_keys=True))
        return code
    if not args.snapshot_sha:
        parser.error(f"{args.command} needs a snapshot sha")
    step = import_step if args.command == "import" else register_step
    return step(config, args.snapshot_sha)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
