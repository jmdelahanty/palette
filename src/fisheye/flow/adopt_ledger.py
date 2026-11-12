"""One-time adoption of the Citrus v2 poller ledger at handover (runner design §6.3).

``python -m fisheye.flow.intake adopt-ledger --config RUNNER.json [--apply] [--abandon SHA:REASON ...]``

The cron poller/registrar pair records its view of each delivery in
``<state_dir>/<key>.*`` (keyed by marker path and bytes, not by snapshot) and
``<log_dir>/bsub_submissions/by_marker/<key>.job``. After the handover the
runner's own state under ``<flow_root>`` is the only authority on whether a
delivery is done or dead, so this reads that ledger once and turns it into
runner state:

- a sentinel only from a true ``fisheye.intake`` verdict for the same sha
  (``probe-register`` from the ws1 deployment at the receipt's producer
  commit, §5.4); the ledger alone never makes a delivery done;
- an operator hold citing the ledger for every terminal poller failure and
  every ledger claim the probes do not confirm;
- nothing for history: a confirmed delivery discover no longer lists.

It is dry-run by default. ``--apply`` refuses while any poller job is in
flight, the poller cron line is active, a delivery's keys disagree, or a
producer deployment is missing; it takes ``poller.lock`` and ``registrar.lock``
for the duration. It never writes to, moves or edits the ledger, and never
overwrites an existing ``refused.json`` or sentinel.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, Iterable

from fisheye.flow.config import V1_STATE_DIR_NAMES, IntakeFlowConfig
from fisheye.flow.lsf import write_json_atomic

REPORT_SCHEMA = "palette.flow.intake_adopt_ledger.v1"
POLLER_CRON_TAG = "# palette_citrus_v2_intake"
KNOWN_SUFFIXES = (
    "claimed", "submitted", "failed", "import_failed", "registered",
    "registration_failed", "registration_refused", "attached_pending", "attached_unresolved",
)
TERMINAL_FAILURES = ("import_failed", "registration_refused", "attached_unresolved")
LOCK_FILES = ("poller.lock", "registrar.lock")
_KEY = re.compile(r"^([0-9a-f]{64})\.([a-z_]+)$")

# Outcomes. Only the first three write runner state; the last three block --apply.
SENTINEL, SENTINEL_IMPORT_ONLY, HOLD_NEW = "sentinel", "sentinel (import only)", "hold (new)"
HOLD_EXISTING, SENTINEL_EXISTING = "hold (existing)", "sentinel (existing)"
RETIRED, RETIRED_UNCONFIRMED, UNCLAIMED = "retired", "retired (unconfirmed)", "unclaimed"
IN_FLIGHT, BLOCKED, CONFLICT = "in-flight", "blocked", "conflict"
WRITES = (SENTINEL, SENTINEL_IMPORT_ONLY, HOLD_NEW)
BLOCKING = (IN_FLIGHT, BLOCKED, CONFLICT)


class AdoptRefused(RuntimeError):
    pass


class ProbeUnavailable(RuntimeError):
    pass


@dataclass
class LedgerEntry:
    key: str
    snapshot_sha: str
    marker: str
    suffixes: list[str]
    job_id: str | None
    files: list[str] = field(default_factory=list)


def _key_value(text: str, name: str) -> str | None:
    found = re.findall(rf"^{name}=(.*?)\s*$", text, flags=re.MULTILINE)
    return found[-1] if found else None


def ledger_dirs(config: IntakeFlowConfig) -> tuple[Path, Path]:
    """The v2 ``state_dir`` and ``log_dir`` named by the registrar config."""

    try:
        registrar = json.loads(config.registrar_config.read_text())
        state_dir, log_dir = Path(registrar["state_dir"]), Path(registrar["log_dir"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AdoptRefused(f"registrar config {config.registrar_config} names no state_dir/log_dir: {exc}") from exc
    for name, path in (("state_dir", state_dir), ("log_dir", log_dir)):
        if path.name in V1_STATE_DIR_NAMES:
            raise AdoptRefused(f"{name} {path} is v1 intake state; the runner never reads it")
    return state_dir, log_dir


def read_ledger(state_dir: Path, log_dir: Path) -> tuple[list[LedgerEntry], list[str], list[str]]:
    """Every claimed key, plus grammar problems and keys without a claim."""

    by_key: dict[str, list[str]] = {}
    problems: list[str] = []
    for path in sorted(state_dir.iterdir()) if state_dir.is_dir() else []:
        if path.name in LOCK_FILES:
            continue
        match = _KEY.match(path.name)
        if not match or match.group(2) not in KNOWN_SUFFIXES:
            problems.append(f"unknown ledger file {path}")
            continue
        by_key.setdefault(match.group(1), []).append(match.group(2))
    if not state_dir.is_dir():
        problems.append(f"no ledger at {state_dir}")
    entries, orphans = [], []
    for key, suffixes in sorted(by_key.items()):
        if "claimed" not in suffixes:
            # The poller unlinks .claimed after a failed submission and retries.
            orphans.append(f"{key}: {', '.join(sorted(suffixes))} without .claimed")
            continue
        try:
            claimed = json.loads((state_dir / f"{key}.claimed").read_text())
            snapshot = str(claimed["snapshot_id"])
            marker = str(claimed["marker"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"unreadable {state_dir / (key + '.claimed')}: {exc}")
            continue
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", snapshot):
            problems.append(f"{key}.claimed has snapshot_id {snapshot!r}")
            continue
        job_id = None
        if "submitted" in suffixes:
            text = (state_dir / f"{key}.submitted").read_text(errors="replace")
            job_id = _key_value(text, "job_id")
            record = _key_value(text, "submitted_record")
            by_marker = Path(record) if record else log_dir / "bsub_submissions" / "by_marker" / f"{key}.job"
            if job_id is None and by_marker.is_file():
                # already_submitted=1: the launcher's own record names the job.
                job_id = _key_value(by_marker.read_text(errors="replace"), "job_id")
        files = [str(state_dir / f"{key}.{suffix}") for suffix in sorted(suffixes)]
        by_marker = log_dir / "bsub_submissions" / "by_marker" / f"{key}.job"
        if by_marker.is_file():
            files.append(str(by_marker))
        entries.append(LedgerEntry(key, snapshot.removeprefix("sha256:"), marker, sorted(suffixes), job_id, files))
    return entries, problems, orphans


def job_ended(log_dir: Path, key: str, job_id: str | None) -> bool | None:
    """True when the poller's LSF job left its end evidence; None if no run dir is found."""

    if not job_id:
        return None
    runs = [p for p in (log_dir / "bsub_submissions").glob(f"citrus_import_*_{key}") if p.is_dir()]
    if not runs:
        return None
    for run in runs:
        if any(run.glob(f"*.{job_id}.status.txt")):
            return True
        out = run / f"{job_id}.out"
        if out.is_file() and "Resource usage summary" in out.read_text(errors="replace"):
            return True
    return False


def poller_cron_active(read_crontab: Callable[[], str], tag: str = POLLER_CRON_TAG) -> bool:
    return any(line.rstrip().endswith(tag) and not line.lstrip().startswith("#")
               for line in read_crontab().splitlines())


def _read_crontab() -> str:
    completed = subprocess.run(["crontab", "-l"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return completed.stdout if completed.returncode == 0 else ""


def _siblings(marker: str) -> list[str]:
    session = Path(marker).parent
    if not session.parent.is_dir():
        return []
    return sorted(str(p) for p in session.parent.glob(f"{session.name}_*") if p.is_dir())


class Prober:
    """Probe verdicts once per sha, from the deployments the rules require."""

    def __init__(self, config: IntakeFlowConfig, runner) -> None:
        from fisheye.flow import intake as flow_intake

        self._flow, self._config, self._runner = flow_intake, config, runner
        self._cache: dict[tuple[str, str], tuple[dict | None, Path | None, str | None]] = {}

    def _probe(self, deployment: Path, step: str, sha: str, *extra: str) -> dict | None:
        """The verdict document when true, None when false; ProbeUnavailable when it could not run."""

        completed = self._runner(self._flow.intake_cli(
            deployment, f"probe-{step}", sha, "--destination-root", str(self._config.destination_root), *extra))
        try:
            document = json.loads(completed.stdout)
        except ValueError:
            document = None
        schema = str(document.get("schema", "")) if isinstance(document, dict) else ""
        if not schema.startswith(self._flow.PROBE_SCHEMAS[step]) or document.get("snapshot_sha") != sha:
            # No verdict at all (e.g. a deployment that predates the probe CLI):
            # never read as "not done", which would turn history into holds.
            raise ProbeUnavailable(f"probe-{step} gave no verdict from {deployment} (exit {completed.returncode})")
        return document if completed.returncode == 0 and self._flow.verdict_ok(document, step, sha) else None

    def imported(self, sha: str) -> tuple[dict | None, Path, str | None]:
        if ("import", sha) not in self._cache:
            ops = self._config.ops_deployment
            self._cache[("import", sha)] = (self._probe(ops, "import", sha), ops,
                                            self._flow._git_head(ops, self._runner))
        return self._cache[("import", sha)]  # type: ignore[return-value]

    def registered(self, sha: str, producer: str) -> tuple[dict | None, Path | None, str | None]:
        if ("register", sha) not in self._cache:
            deployment = self._flow.deployment_for_commit(self._config, producer, runner=self._runner)
            document = None if deployment is None else self._probe(
                deployment, "register", sha, "--registry", str(self._config.registry))
            self._cache[("register", sha)] = (document, deployment, producer)
        return self._cache[("register", sha)]


def classify(config: IntakeFlowConfig, entry: LedgerEntry, discoverable: set[str],
             prober: Prober, log_dir: Path) -> dict:
    try:
        return _classify(config, entry, discoverable, prober, log_dir)
    except ProbeUnavailable as exc:
        row = {"key": entry.key, "snapshot_sha": entry.snapshot_sha, "marker": entry.marker,
               "ledger": entry.suffixes, "ledger_files": entry.files,
               "discoverable": entry.snapshot_sha in discoverable}
        if entry.snapshot_sha in discoverable:
            return {**row, "outcome": BLOCKED, "why": str(exc)}
        return {**row, "outcome": RETIRED_UNCONFIRMED, "why": str(exc)}


def _classify(config: IntakeFlowConfig, entry: LedgerEntry, discoverable: set[str],
              prober: Prober, log_dir: Path) -> dict:
    from fisheye.flow.intake import refusal_path, sentinel_path

    sha = entry.snapshot_sha
    row = {"key": entry.key, "snapshot_sha": sha, "marker": entry.marker,
           "ledger": entry.suffixes, "ledger_files": entry.files, "discoverable": sha in discoverable}
    if refusal_path(config, sha).exists():
        return {**row, "outcome": HOLD_EXISTING}
    if sentinel_path(config, sha, "register").exists():
        return {**row, "outcome": SENTINEL_EXISTING}
    terminal = [s for s in TERMINAL_FAILURES if s in entry.suffixes]
    if terminal:
        return {**row, "outcome": HOLD_NEW, "why": f"poller ledger records {', '.join(terminal)}"}
    if "submitted" not in entry.suffixes:
        return {**row, "outcome": UNCLAIMED if sha in discoverable else RETIRED_UNCONFIRMED,
                "why": "claimed but never submitted"}
    ended = True if "registered" in entry.suffixes else job_ended(log_dir, entry.key, entry.job_id)
    if ended is False:
        return {**row, "outcome": IN_FLIGHT, "why": f"poller job {entry.job_id} has not ended"}
    imported, _, _ = prober.imported(sha)
    if imported is None and ended is None:
        return {**row, "outcome": BLOCKED,
                "why": f"no run directory for poller job {entry.job_id}; cannot tell whether it ended"}
    if imported is None:
        if sha not in discoverable:
            return {**row, "outcome": RETIRED_UNCONFIRMED, "why": "probe-import not true; not discoverable"}
        return {**row, "outcome": HOLD_NEW, "why": "ledger records a submission; probe-import is not true"}
    producer = imported.get("producer_git_sha")
    registered, deployment, _ = prober.registered(sha, producer) if producer else (None, None, None)
    if deployment is None:
        return {**row, "outcome": BLOCKED,
                "why": f"no ws1 deployment at producer commit {producer}; create ops-{str(producer)[:8]}"}
    if registered is not None:
        return {**row, "outcome": SENTINEL if sha in discoverable else RETIRED,
                "admission_mode": registered.get("admission_mode")}
    if "registered" in entry.suffixes:
        return {**row, "outcome": HOLD_NEW, "why": "ledger says registered; probe-register is not true"}
    if sha not in discoverable:
        return {**row, "outcome": RETIRED_UNCONFIRMED, "why": "imported, not registered, not discoverable"}
    return {**row, "outcome": SENTINEL_IMPORT_ONLY, "admission_mode": imported.get("admission_mode")}


def _merge(rows: list[dict]) -> str:
    outcomes = {row["outcome"] for row in rows}
    for blocking in BLOCKING:
        if blocking in outcomes:
            return blocking
    writes = outcomes & set(WRITES)
    if len(writes) > 1 or (writes and outcomes & {HOLD_EXISTING, SENTINEL_EXISTING}) \
            or (HOLD_NEW in outcomes and RETIRED in outcomes):
        return CONFLICT
    for outcome in (HOLD_EXISTING, SENTINEL_EXISTING, *WRITES, RETIRED, RETIRED_UNCONFIRMED, UNCLAIMED):
        if outcome in outcomes:
            return outcome
    return CONFLICT


def build_report(
    config: IntakeFlowConfig,
    *,
    abandon: Iterable[tuple[str, str]] = (),
    runner=None,
    read_crontab: Callable[[], str] | None = None,
) -> tuple[dict, Prober]:
    from fisheye.flow import intake as flow_intake

    runner = runner or flow_intake._exec
    state_dir, log_dir = ledger_dirs(config)
    entries, problems, orphans = read_ledger(state_dir, log_dir)
    targets = flow_intake.discover_targets(config, runner=runner)
    discoverable = {t["snapshot_sha"] for t in targets}
    prober = Prober(config, runner)
    rows = [classify(config, entry, discoverable, prober, log_dir) for entry in entries]
    by_sha: dict[str, list[dict]] = {}
    for row in rows:
        by_sha.setdefault(row["snapshot_sha"], []).append(row)
    abandoned = dict(abandon)
    deliveries = {}
    for sha in sorted(set(by_sha) | set(abandoned)):
        keyed = by_sha.get(sha, [])
        if sha in abandoned:
            if not keyed and sha not in discoverable:
                problems.append(f"--abandon {sha}: neither in the ledger nor discoverable")
                continue
            if flow_intake.refusal_path(config, sha).exists():
                outcome = HOLD_EXISTING
            elif any(r["outcome"] in (SENTINEL, SENTINEL_IMPORT_ONLY, SENTINEL_EXISTING, RETIRED) for r in keyed):
                outcome = CONFLICT  # Palette confirms what the operator wants abandoned
            else:
                outcome = HOLD_NEW
        else:
            outcome = _merge(keyed)
        deliveries[sha] = {"outcome": outcome, "keys": [r["key"] for r in keyed],
                           "abandon_reason": abandoned.get(sha)}
    unclaimed = sorted(t["snapshot_sha"] for t in targets
                       if t["snapshot_sha"] not in deliveries and not flow_intake.refusal_path(
                           config, t["snapshot_sha"]).exists())
    notes = [f"{sha}: {len(keys)} ledger keys" for sha, keys in
             ((s, d["keys"]) for s, d in deliveries.items()) if len(keys) > 1]
    for row in rows:
        if deliveries.get(row["snapshot_sha"], {}).get("outcome") in (HOLD_NEW, HOLD_EXISTING):
            for sibling in _siblings(row["marker"]):
                notes.append(f"{row['snapshot_sha']}: sibling session {sibling} (advisory; not a successor record)")
    archives = sorted(str(p) for p in state_dir.parent.glob(f"{state_dir.name}_*") if p.is_dir())
    archives += sorted(str(p) for p in (config.destination_root / ".transfer_intake").glob("superseded_*")
                       if p.is_dir())
    notes += [f"operator archive {path} ignored" for path in archives]
    cron_active = poller_cron_active(read_crontab or _read_crontab)
    blockers = [f"{sha}: {d['outcome']}" for sha, d in deliveries.items() if d["outcome"] in BLOCKING]
    if cron_active:
        blockers.append(f"the poller cron line ({POLLER_CRON_TAG}) is active; pause it first")
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": time.time(),
        "state_dir": str(state_dir),
        "log_dir": str(log_dir),
        "keys": rows,
        "deliveries": deliveries,
        "unclaimed": unclaimed,
        "orphan_keys": orphans,
        "notes": notes,
        "problems": problems,
        "blockers": blockers,
        "apply_allowed": not problems and not blockers,
    }, prober


def _hold(config: IntakeFlowConfig, sha: str, rows: list[dict], reason: str | None) -> None:
    from fisheye.flow.intake import REFUSAL_SCHEMA, refusal_path

    why = reason or "; ".join(sorted({r.get("why", "") for r in rows if r.get("why")}))
    files = sorted({f for r in rows for f in r["ledger_files"]})
    config.delivery_dir(sha).mkdir(parents=True, exist_ok=True)
    write_json_atomic(refusal_path(config, sha), {
        "schema": REFUSAL_SCHEMA,
        "step": "import",
        "snapshot_sha": sha,
        "recorded_at": time.time(),
        "operator_hold": True,
        "source": "adopt-ledger",
        "result": {"error": "operator_abandoned" if reason else "poller_ledger_not_confirmed",
                   "message": why, "ledger_files": files, "ledger_keys": [r["key"] for r in rows]},
        "clear": f"delete this file only after deciding {sha} should be imported; the next tick drives it",
    })


@contextmanager
def poller_locks(state_dir: Path):
    """Hold poller.lock and registrar.lock so no poller or registrar run overlaps."""

    handles = []
    try:
        for name in LOCK_FILES:
            path = state_dir / name
            if not path.is_file():
                raise AdoptRefused(f"no {path}; refusing to create files in the poller ledger")
            handle = open(path, "a")  # append mode: never truncate the poller's file
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AdoptRefused(f"{path} is held; a poller or registrar run is active") from exc
        yield
    finally:
        for handle in handles:
            handle.close()


def _write(config: IntakeFlowConfig, report: dict, prober: Prober) -> list[str]:
    from fisheye.flow import intake as flow_intake

    written = []
    for sha, delivery in report["deliveries"].items():
        outcome = delivery["outcome"]
        if outcome not in WRITES or flow_intake.refusal_path(config, sha).exists():
            continue  # never overwrite a hold
        if outcome == HOLD_NEW:
            _hold(config, sha, [r for r in report["keys"] if r["snapshot_sha"] == sha],
                  delivery.get("abandon_reason"))
            written.append(f"{sha}: {HOLD_NEW}")
            continue
        imported, ops, ops_commit = prober.imported(sha)
        extra = {"source": "adopt-ledger", "ledger_keys": delivery["keys"],
                 "admission_mode": imported.get("admission_mode")}
        if not flow_intake.sentinel_path(config, sha, "import").exists():
            flow_intake.record_outcome(config, sha, "import", flow_intake.EXIT_DONE, imported,
                                       deployment=ops, commit=ops_commit, extra=extra)
        if outcome == SENTINEL:
            registered, deployment, producer = prober.registered(sha, imported["producer_git_sha"])
            flow_intake.record_outcome(config, sha, "register", flow_intake.EXIT_DONE, registered,
                                       deployment=deployment, commit=producer, extra=extra)
        written.append(f"{sha}: {outcome}")
    return written


def adopt(
    config: IntakeFlowConfig,
    *,
    apply: bool = False,
    abandon: Iterable[tuple[str, str]] = (),
    runner=None,
    read_crontab: Callable[[], str] | None = None,
) -> dict:
    """The report; with ``apply``, recomputed and written under both poller locks."""

    if not apply:
        report, _ = build_report(config, abandon=abandon, runner=runner, read_crontab=read_crontab)
        return {**report, "applied": False, "written": []}
    state_dir, _ = ledger_dirs(config)
    with poller_locks(state_dir):
        # Verdicts are taken under the locks, so nothing moves between probe and write.
        report, prober = build_report(config, abandon=abandon, runner=runner, read_crontab=read_crontab)
        if not report["apply_allowed"]:
            raise AdoptRefused("; ".join(report["problems"] + report["blockers"]))
        written = _write(config, report, prober)
    return {**report, "applied": True, "written": written}


def parse_abandon(values: Iterable[str]) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        sha, _, reason = value.partition(":")
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or not reason.strip():
            raise AdoptRefused(f"--abandon needs SHA:REASON with a 64-hex sha, got {value!r}")
        parsed.append((sha, reason.strip()))
    return parsed
