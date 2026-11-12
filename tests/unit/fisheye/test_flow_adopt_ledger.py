"""adopt-ledger: one-time Citrus v2 poller ledger import at handover (design §6.3)."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from fisheye.flow import adopt_ledger as adopt
from fisheye.flow import intake as flow_intake
from fisheye.flow.config import parse_config

PRODUCER = "c" * 40
CURRENT = "e" * 40
NO_CRON = lambda: "*/5 * * * * true # other_job\n# */5 * * * * poll # palette_citrus_v2_intake\n"  # noqa: E731


def _sha(n: int) -> str:
    return f"{n:x}" * 64 if n < 16 else hashlib.sha256(str(n).encode()).hexdigest()


def _completed(returncode=0, stdout=""):
    return subprocess.CompletedProcess([], returncode, stdout, "")


def _probe(step: str, sha: str, verdict=True) -> dict:
    return {"schema": f"palette.intake.probe_{step}.v1", "snapshot_sha": sha, "verdict": verdict,
            "evidence_digest": "sha256:" + "d" * 64 if verdict else None,
            "producer_git_sha": PRODUCER, "admission_mode": "workstation"}


class World:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.staging = tmp_path / "staging"
        self.state = self.staging / ".processing_state_v2"
        self.logs = self.staging / ".processing_logs_v2"
        self.state.mkdir(parents=True)
        (self.logs / "bsub_submissions" / "by_marker").mkdir(parents=True)
        for lock in adopt.LOCK_FILES:
            (self.state / lock).write_text("")
        deployments = tmp_path / "deployments"
        for name in ("ops-current", f"ops-{PRODUCER[:8]}"):
            (deployments / name).mkdir(parents=True)
        (tmp_path / "registrar.json").write_text(json.dumps(
            {"state_dir": str(self.state), "log_dir": str(self.logs)}))
        self.config = parse_config({
            "schema": "palette.flow.intake_config.v1",
            "flow_root": str(tmp_path / "flow"),
            "staging_dir": str(self.staging),
            "destination_root": str(tmp_path / "recordings"),
            "registry": str(tmp_path / "registry.sqlite"),
            "registrar_config": str(tmp_path / "registrar.json"),
            "ops_deployment": str(deployments / "ops-current"),
            "deployments_root": str(deployments),
            "lsf_repo": str(tmp_path / "lsf-repo"),
            "lsf": {"submit_host": "login1-citrus-poller"},
        })
        self.discoverable: list[str] = []
        self.imported: set[str] = set()
        self.registered: set[str] = set()
        self.calls: list[list[str]] = []
        self.no_cli: set[str] = set()

    def key(self, sha: str, *suffixes: str, session: str | None = None, job: str | None = "100",
            ended: bool | None = True, already_submitted: bool = False) -> str:
        session = session or f"2026_10_09_{sha[:6]}"
        marker = self.staging / session / "_citrus_transfer_complete.json"
        key = hashlib.sha256(f"{marker}\0{sha}".encode()).hexdigest()
        (self.state / f"{key}.claimed").write_text(json.dumps(
            {"marker": str(marker), "marker_sha256": "f" * 64, "snapshot_id": f"sha256:{sha}", "command": []}))
        by_marker = self.logs / "bsub_submissions" / "by_marker" / f"{key}.job"
        for suffix in suffixes:
            path = self.state / f"{key}.{suffix}"
            if suffix == "submitted" and already_submitted:
                by_marker.write_text(f"job_id={job}\njob_name=x\nsession_dir={marker.parent}\n")
                path.write_text(f"already_submitted=1\nsubmitted_record={by_marker}\n")
            elif suffix == "submitted":
                path.write_text(f"session_dir={marker.parent}\njob_id={job}\n")
            else:
                path.write_text(json.dumps({"error": f"{suffix} record"}))
        if job and ended is not None:
            run = self.logs / "bsub_submissions" / f"citrus_import_20261011T000000Z_{session}_{key}"
            run.mkdir(parents=True)
            (run / f"{job}.out").write_text("Sender: LSF\n" + ("Resource usage summary:\n" if ended else ""))
        return key

    def run(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        if any(Path(argv[0]).parts[-3:-2] == (d,) for d in self.no_cli):
            return _completed(1, "")  # "No module named fisheye.intake"
        if argv[0] == "git":
            return _completed(stdout=(PRODUCER if PRODUCER[:8] in argv[2] else CURRENT) + "\n")
        if "discover" in argv:
            targets = [{"snapshot_sha": s} for s in self.discoverable]
            return _completed(stdout=json.dumps({"targets": targets, "registry_error": None}))
        for step, done in (("import", self.imported), ("register", self.registered)):
            if f"probe-{step}" in argv:
                sha = argv[argv.index(f"probe-{step}") + 1]
                return _completed(0 if sha in done else 1, json.dumps(_probe(step, sha, sha in done)))
        raise AssertionError(f"unexpected command {argv}")

    def adopt(self, **kwargs):
        kwargs.setdefault("read_crontab", NO_CRON)
        return adopt.adopt(self.config, runner=self.run, **kwargs)

    def ledger_snapshot(self) -> dict:
        return {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                for base in (self.state, self.logs) for p in sorted(base.rglob("*")) if p.is_file()}


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_registered_history_retires_and_writes_nothing(world):
    sha = _sha(1)
    world.key(sha, "submitted", "registered")
    world.imported.add(sha)
    world.registered.add(sha)
    report = world.adopt(apply=True)
    assert report["deliveries"][sha]["outcome"] == adopt.RETIRED
    assert report["written"] == [] and not world.config.delivery_dir(sha).exists()


def test_terminal_poller_failure_becomes_a_hold_citing_the_ledger(world):
    sha = _sha(2)
    key = world.key(sha, "submitted", "import_failed")
    world.discoverable.append(sha)
    report = world.adopt(apply=True)
    assert report["deliveries"][sha]["outcome"] == adopt.HOLD_NEW
    hold = json.loads(flow_intake.refusal_path(world.config, sha).read_text())
    assert hold["operator_hold"] is True and hold["source"] == "adopt-ledger"
    assert str(world.state / f"{key}.import_failed") in hold["result"]["ledger_files"]
    assert "import_failed" in hold["result"]["message"]
    assert flow_intake.plan(world.config, runner=world.run)["drive"] == []


@pytest.mark.parametrize("suffix", ["registration_refused", "attached_unresolved"])
def test_every_terminal_suffix_holds(world, suffix):
    sha = _sha(3)
    world.key(sha, "submitted", suffix)
    world.discoverable.append(sha)
    world.imported.add(sha)
    assert world.adopt()["deliveries"][sha]["outcome"] == adopt.HOLD_NEW


def test_import_only_sentinel_from_an_already_submitted_record(world):
    sha = _sha(4)
    world.key(sha, "submitted", "registration_failed", already_submitted=True, job="555")
    world.discoverable.append(sha)
    world.imported.add(sha)
    report = world.adopt(apply=True)
    assert report["deliveries"][sha]["outcome"] == adopt.SENTINEL_IMPORT_ONLY
    sentinel = json.loads(flow_intake.sentinel_path(world.config, sha, "import").read_text())
    assert sentinel["source"] == "adopt-ledger" and sentinel["admission_mode"] == "workstation"
    assert sentinel["probe"]["verdict"] is True
    assert not flow_intake.sentinel_path(world.config, sha, "register").exists()
    assert flow_intake.plan(world.config, runner=world.run)["drive"] == [sha]  # runner drives register


def test_register_probe_runs_at_the_producer_commit(world):
    sha = _sha(5)
    world.key(sha, "submitted", "registered")
    world.discoverable.append(sha)
    world.imported.add(sha)
    world.registered.add(sha)
    report = world.adopt(apply=True)
    assert report["deliveries"][sha]["outcome"] == adopt.SENTINEL
    register_calls = [c for c in world.calls if "probe-register" in c]
    assert register_calls and all(f"ops-{PRODUCER[:8]}" in c[0] for c in register_calls)
    sentinel = json.loads(flow_intake.sentinel_path(world.config, sha, "register").read_text())
    assert sentinel["deployment_commit"] == PRODUCER


def test_missing_producer_deployment_blocks_instead_of_holding(world):
    sha = _sha(6)
    world.key(sha, "submitted", "registered")
    world.discoverable.append(sha)
    world.imported.add(sha)
    (world.root / "deployments" / f"ops-{PRODUCER[:8]}").rmdir()
    report = world.adopt()
    assert report["deliveries"][sha]["outcome"] == adopt.BLOCKED and not report["apply_allowed"]
    with pytest.raises(adopt.AdoptRefused, match="blocked"):
        world.adopt(apply=True)
    assert not flow_intake.refusal_path(world.config, sha).exists()


def test_ledger_registered_but_probe_false_holds(world):
    sha = _sha(7)
    world.key(sha, "submitted", "registered")
    world.discoverable.append(sha)
    world.imported.add(sha)
    assert world.adopt()["deliveries"][sha]["outcome"] == adopt.HOLD_NEW


def test_in_flight_job_refuses_apply_and_writes_nothing(world):
    sha, other = _sha(8), _sha(9)
    world.key(sha, "submitted", ended=False)
    world.key(other, "submitted", "import_failed")
    world.discoverable += [sha, other]
    report = world.adopt()
    assert report["deliveries"][sha]["outcome"] == adopt.IN_FLIGHT
    with pytest.raises(adopt.AdoptRefused, match="in-flight"):
        world.adopt(apply=True)
    assert not world.config.intake_root.exists()


def test_no_run_dir_and_no_import_blocks(world):
    sha = _sha(10)
    world.key(sha, "submitted", ended=None)
    world.discoverable.append(sha)
    assert world.adopt()["deliveries"][sha]["outcome"] == adopt.BLOCKED


def test_existing_hold_is_reported_and_never_overwritten(world):
    sha = _sha(11)
    world.key(sha, "submitted", "import_failed")
    world.discoverable.append(sha)
    path = flow_intake.refusal_path(world.config, sha)
    path.parent.mkdir(parents=True)
    path.write_text('{"hand": "written"}')
    report = world.adopt(apply=True)
    assert report["deliveries"][sha]["outcome"] == adopt.HOLD_EXISTING
    assert path.read_text() == '{"hand": "written"}'


def test_two_keys_disagreeing_is_a_conflict(world):
    sha = _sha(12)
    world.key(sha, "submitted", "registered", session="a")
    world.key(sha, "submitted", "import_failed", session="a_resealed")
    world.imported.add(sha)
    world.registered.add(sha)
    report = world.adopt()
    assert report["deliveries"][sha]["outcome"] == adopt.CONFLICT
    assert any("2 ledger keys" in note for note in report["notes"])
    with pytest.raises(adopt.AdoptRefused, match="conflict"):
        world.adopt(apply=True)


def test_claimed_without_marker_and_unclaimed_markers(world):
    gone, fresh, claimed_live = _sha(13), _sha(14), _sha(15)
    world.key(gone, "failed")  # submission failed; marker since removed
    world.key(claimed_live)  # claimed, never submitted, still staged
    world.discoverable += [fresh, claimed_live]
    report = world.adopt()
    assert report["deliveries"][gone]["outcome"] == adopt.RETIRED_UNCONFIRMED
    assert report["deliveries"][claimed_live]["outcome"] == adopt.UNCLAIMED
    assert report["unclaimed"] == [fresh] and report["apply_allowed"]


def test_abandon_list_holds_with_the_operator_reason(world):
    sha, unknown = _sha(16), _sha(17)
    world.discoverable.append(sha)
    report = world.adopt(apply=True, abandon=[(sha, "staged by mistake")])
    assert report["deliveries"][sha]["outcome"] == adopt.HOLD_NEW
    hold = json.loads(flow_intake.refusal_path(world.config, sha).read_text())
    assert hold["result"]["error"] == "operator_abandoned"
    assert hold["result"]["message"] == "staged by mistake"
    assert "unclaimed" in report and sha not in report["unclaimed"]
    assert any(unknown in p for p in world.adopt(abandon=[(unknown, "x")])["problems"])
    with pytest.raises(adopt.AdoptRefused):
        adopt.parse_abandon(["abc:reason"])
    with pytest.raises(adopt.AdoptRefused):
        adopt.parse_abandon([f"{sha}:"])


def test_successor_siblings_and_archives_are_advisory_only(world):
    sha = _sha(18)
    world.key(sha, "submitted", "import_failed", session="2026_10_09_16_00_03")
    (world.staging / "2026_10_09_16_00_03_resealed").mkdir()
    (world.staging / ".processing_state_v2_superseded_20261011_dish_mismatch").mkdir()
    world.discoverable.append(sha)
    report = world.adopt(apply=True)
    hold = json.loads(flow_intake.refusal_path(world.config, sha).read_text())
    assert "successor" not in hold and "successor" not in hold["result"]
    assert "resealed" not in hold["result"]["message"]
    assert not any("resealed" in f for f in hold["result"]["ledger_files"])
    assert any("_resealed" in note and "advisory" in note for note in report["notes"])
    assert any("superseded_20261011" in note and "ignored" in note for note in report["notes"])


def test_active_poller_cron_or_held_lock_refuses_apply(world):
    sha = _sha(19)
    world.key(sha, "submitted", "import_failed")
    world.discoverable.append(sha)
    active = lambda: "*/5 * * * * cd x && poll >> log 2>&1 # palette_citrus_v2_intake\n"  # noqa: E731
    with pytest.raises(adopt.AdoptRefused, match="cron"):
        world.adopt(apply=True, read_crontab=active)
    with open(world.state / "poller.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(adopt.AdoptRefused, match="poller.lock is held"):
            world.adopt(apply=True)
    assert not flow_intake.refusal_path(world.config, sha).exists()


def test_unknown_ledger_grammar_and_v1_state_refuse(world):
    (world.state / f"{'a' * 64}.mystery").write_text("")
    report = world.adopt()
    assert report["problems"] and not report["apply_allowed"]
    (world.root / "registrar.json").write_text(json.dumps(
        {"state_dir": str(world.staging / ".processing_state"), "log_dir": str(world.logs)}))
    with pytest.raises(adopt.AdoptRefused, match="v1"):
        world.adopt()


def test_apply_never_changes_the_ledger(world):
    for n, suffixes in enumerate([("submitted", "registered"), ("submitted", "import_failed"),
                                  ("submitted", "registration_failed")], start=20):
        world.key(_sha(n), *suffixes)
        world.discoverable.append(_sha(n))
        world.imported.add(_sha(n))
    world.registered.add(_sha(20))
    before = world.ledger_snapshot()
    report = world.adopt(apply=True)
    assert len(report["written"]) == 3
    assert world.ledger_snapshot() == before


def test_cli_dry_run_writes_a_report_and_no_state(world, monkeypatch, capsys):
    sha = _sha(30)
    world.key(sha, "submitted", "import_failed")
    world.discoverable.append(sha)
    config_path = world.root / "runner.json"
    config_path.write_text(json.dumps({
        "schema": "palette.flow.intake_config.v1", "flow_root": str(world.config.flow_root),
        "staging_dir": str(world.staging), "destination_root": str(world.config.destination_root),
        "registry": str(world.config.registry), "registrar_config": str(world.config.registrar_config),
        "ops_deployment": str(world.config.ops_deployment), "deployments_root": str(world.config.deployments_root),
        "lsf_repo": str(world.config.lsf_repo), "lsf": {"submit_host": "login1-citrus-poller"}}))
    monkeypatch.setattr(flow_intake, "_exec", world.run)
    monkeypatch.setattr(adopt, "_read_crontab", NO_CRON)
    assert flow_intake.main(["adopt-ledger", "--config", str(config_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["applied"] is False and printed["deliveries"][sha]["outcome"] == adopt.HOLD_NEW
    assert not flow_intake.refusal_path(world.config, sha).exists()
    assert list((world.config.flow_root / "adopt_ledger").glob("report-*-dry-run.json"))


def test_producer_deployment_without_the_probe_cli_never_becomes_a_hold(world):
    retired, live = _sha(31), _sha(32)
    world.key(retired, "submitted", "registered")
    world.key(live, "submitted", "registered")
    world.discoverable.append(live)
    world.imported |= {retired, live}
    world.no_cli.add(f"ops-{PRODUCER[:8]}")  # predates fisheye.intake probe-register
    report = world.adopt()
    assert report["deliveries"][retired]["outcome"] == adopt.RETIRED_UNCONFIRMED
    assert report["deliveries"][live]["outcome"] == adopt.BLOCKED
    assert "no verdict" in next(k["why"] for k in report["keys"] if k["snapshot_sha"] == live)
    assert not report["apply_allowed"] and not world.config.intake_root.exists()
