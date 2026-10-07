"""Workstation registration of completed Citrus transfer imports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.utils import citrus_transfer_v2_poller as poller
from fisheye.utils import register_completed_imports as registrar

KEY = "b" * 64


def _config(tmp_path: Path) -> dict:
    config = {
        "staging_dir": str(tmp_path / "staging"), "state_dir": str(tmp_path / "state"),
        "log_dir": str(tmp_path / "logs"),
        "submit": {"transport": "local", "repo": "/groups/palette"},
        "registration": "workstation", "registry": str(tmp_path / "registry.sqlite"),
        "writer_host": "writer", "writer_lock_path": str(tmp_path / "writer.lock"),
        "shadow_temp_root": str(tmp_path / "shadows"), "shadow_backup_dir": str(tmp_path / "backups"),
    }
    (tmp_path / "state").mkdir()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return registrar.load_config(path)


def _submitted(tmp_path: Path, job_id: str = "777") -> None:
    (tmp_path / "state" / f"{KEY}.submitted").write_text(f"run_dir=x\njob_id={job_id}\n")


def _status(tmp_path: Path, job_id: str = "777", **payload) -> None:
    run_dir = tmp_path / "logs" / "bsub_submissions" / f"citrus_import_20261006T000000Z_session_{KEY}"
    run_dir.mkdir(parents=True)
    body = {"status": "complete", "import_complete": True,
            "zarr_paths": ["/rec/a.zarr", "/rec/b.zarr"],
            "plan": {"snapshot_id": "sha256:" + "d" * 64, "destination_root": "/rec",
                     "parents": [{"context": {"data_origin": "acquired"}}]}}
    body.update(payload)
    (run_dir / f"workflow-{job_id}").mkdir()
    (run_dir / f"workflow-{job_id}" / "citrus_session_import.status.json").write_text(json.dumps(body))


class Register:
    """Stands in for fisheye.intake.register_delivery: one call per delivery."""

    def __init__(self, fail: bool = False):
        self.calls: list[tuple[str, Path]] = []
        self.fail = fail

    def __call__(self, config, snapshot_sha, destination_root):
        self.calls.append((snapshot_sha, destination_root))
        if self.fail:
            raise RuntimeError("registry busy")
        return {zarr: f"dataset-{Path(zarr).name}" for zarr in ("/rec/a.zarr", "/rec/b.zarr")}


def test_pending_job_is_left_for_the_next_run(tmp_path):
    config = _config(tmp_path)
    _submitted(tmp_path)
    register = Register()
    assert registrar.register_completed(config, dry_run=False, register=register) == 0
    assert register.calls == [] and not (tmp_path / "state" / f"{KEY}.registered").exists()


def test_completed_import_is_registered_once(tmp_path):
    config = _config(tmp_path)
    _submitted(tmp_path)
    _status(tmp_path)
    register = Register()
    assert registrar.register_completed(config, dry_run=False, register=register) == 0
    # The whole delivery is one registration (one registry publication).
    assert register.calls == [("sha256:" + "d" * 64, Path("/rec"))]
    done = json.loads((tmp_path / "state" / f"{KEY}.registered").read_text())
    assert done["datasets"] == {"/rec/a.zarr": "dataset-a.zarr", "/rec/b.zarr": "dataset-b.zarr"}
    registrar.register_completed(config, dry_run=False, register=register)
    assert len(register.calls) == 1  # not registered again


def test_failed_import_is_recorded_for_review_not_retried(tmp_path):
    config = _config(tmp_path)
    _submitted(tmp_path)
    _status(tmp_path, status="failed", import_complete=False, error="boom")
    register = Register()
    registrar.register_completed(config, dry_run=False, register=register)
    registrar.register_completed(config, dry_run=False, register=register)
    assert register.calls == []
    assert json.loads((tmp_path / "state" / f"{KEY}.import_failed").read_text())["error"] == "boom"


def test_registration_error_is_retried_next_run(tmp_path):
    config = _config(tmp_path)
    _submitted(tmp_path)
    _status(tmp_path)
    assert registrar.register_completed(config, dry_run=False, register=Register(fail=True)) == 1
    assert (tmp_path / "state" / f"{KEY}.registration_failed").exists()
    assert registrar.register_completed(config, dry_run=False, register=Register()) == 0
    assert (tmp_path / "state" / f"{KEY}.registered").exists()
    assert not (tmp_path / "state" / f"{KEY}.registration_failed").exists()


@pytest.mark.parametrize("canonical", [True, False])
def test_synthetic_transfer_is_never_registered(tmp_path, monkeypatch, canonical):
    from fisheye.intake.delivery import CANONICAL_REGISTRY

    config = _config(tmp_path)
    if canonical:
        monkeypatch.setitem(config, "registry", str(CANONICAL_REGISTRY))
    _submitted(tmp_path)
    _status(tmp_path, plan={"snapshot_id": "sha256:" + "d" * 64, "destination_root": "/rec",
                            "parents": [{"context": {"data_origin": "synthetic"}}]})
    register = Register()
    assert registrar.register_completed(config, dry_run=False, register=register) == 1
    assert register.calls == []
    refused = json.loads((tmp_path / "state" / f"{KEY}.registration_refused").read_text())
    assert "synthetic" in refused["error"]
    assert not (tmp_path / "state" / f"{KEY}.registration_failed").exists()
    # Terminal: a refusal is never retried.
    assert registrar.register_completed(config, dry_run=False, register=register) == 0


def test_a_registration_held_by_another_run_is_left_for_the_next_run(tmp_path):
    from fisheye.intake.outcomes import IntakeHeld

    config = _config(tmp_path)
    _submitted(tmp_path)
    _status(tmp_path)

    def held(config, snapshot_sha, destination_root):
        raise IntakeHeld("held", lock_path="x.register.lock", holder=None)

    assert registrar.register_completed(config, dry_run=False, register=held) == 0
    assert sorted(p.name for p in (tmp_path / "state").iterdir()) == [f"{KEY}.submitted"]


def test_dry_run_writes_nothing(tmp_path):
    config = _config(tmp_path)
    _submitted(tmp_path)
    _status(tmp_path)
    register = Register()
    registrar.register_completed(config, dry_run=True, register=register)
    assert register.calls == [] and sorted(p.name for p in (tmp_path / "state").iterdir()) == [f"{KEY}.submitted"]


def test_config_requires_workstation_registration_and_writer_settings(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"state_dir": "s", "log_dir": "l"}))
    with pytest.raises(registrar.RegistrarRefusal, match="missing required keys"):
        registrar.load_config(path)
    _config(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    path.write_text(json.dumps({**config, "registration": "job"}))
    with pytest.raises(registrar.RegistrarRefusal, match="job mode is retired"):
        registrar.load_config(path)


def test_the_writer_host_may_be_named_short_or_fully_qualified(tmp_path, monkeypatch):
    from fisheye.intake import registration

    config = _config(tmp_path)
    monkeypatch.setattr(registration.socket, "gethostname", lambda: "delahantyj-ws1.hhmi.org")
    for name in ("delahantyj-ws1", "delahantyj-ws1.hhmi.org", "DELAHANTYJ-WS1.hhmi.org"):
        registrar._require_writer_host({**config, "writer_host": name})
    with pytest.raises(registrar.RegistrarRefusal, match="not the registry writer"):
        registrar._require_writer_host({**config, "writer_host": "delahantyj-ws2"})


def test_main_refuses_on_a_host_that_is_not_the_writer(tmp_path, capsys):
    _config(tmp_path)
    assert registrar.main(["--config", str(tmp_path / "config.json")]) == 2
    assert "is not the registry writer" in capsys.readouterr().out


def test_poller_dispatches_jobs_without_registering_in_workstation_mode(tmp_path):
    (tmp_path / "staging").mkdir()
    config = _config(tmp_path)
    command = poller.build_command(config, tmp_path / "session", KEY)
    assert "--no-register" in command[-1] and "--writer-host" not in command[-1]


def test_job_that_ended_without_status_is_a_failed_import_not_pending(tmp_path):
    config = _config(tmp_path)
    _submitted(tmp_path)
    run_dir = tmp_path / "logs" / "bsub_submissions" / f"citrus_import_20261007T000000Z_session_{KEY}"
    run_dir.mkdir(parents=True)
    (run_dir / "777.out").write_text("payload output\n")  # still running: no LSF report yet
    registrar.register_completed(config, dry_run=False, register=Register())
    assert not (tmp_path / "state" / f"{KEY}.import_failed").exists()

    (run_dir / "777.out").write_text("payload output\n\nResource usage summary:\n CPU time : 1 sec.\n")
    registrar.register_completed(config, dry_run=False, register=Register())
    failed = json.loads((tmp_path / "state" / f"{KEY}.import_failed").read_text())
    assert failed["error"] == "LSF job ended without writing its status JSON"


def test_a_job_that_found_the_delivery_held_is_attached_not_failed(tmp_path):
    # S3: the job script records payload_returncode=75 and the workflow made
    # no directory (the lock is taken before any side effect).
    config = _config(tmp_path)
    _submitted(tmp_path)
    run_dir = tmp_path / "logs" / "bsub_submissions" / f"citrus_import_20261006T000000Z_session_{KEY}"
    run_dir.mkdir(parents=True)
    (run_dir / "session.777.status.txt").write_text("job_id=777\npayload_returncode=75\n")
    register = Register()
    assert registrar.register_completed(config, dry_run=False, register=register) == 0
    assert not (tmp_path / "state" / f"{KEY}.import_failed").exists()
    assert register.calls == []
    # Any other ended-without-status exit is still a failed import.
    (run_dir / "session.777.status.txt").write_text("job_id=777\npayload_returncode=1\n")
    registrar.register_completed(config, dry_run=False, register=register)
    assert (tmp_path / "state" / f"{KEY}.import_failed").exists()


def test_the_newest_requeued_attempt_decides(tmp_path):
    # N6: an LSF requeue reuses the job id; each attempt has its own directory.
    config = _config(tmp_path)
    _submitted(tmp_path)
    _status(tmp_path, status="failed", import_complete=False, error="first attempt died")
    run_dir = next((tmp_path / "logs" / "bsub_submissions").iterdir())
    (run_dir / "workflow-777").rename(run_dir / "workflow-777-0-20261007T010000000000000Z-11")
    newer = run_dir / "workflow-777-0-20261007T020000000000000Z-12"
    newer.mkdir()
    body = {"status": "complete", "import_complete": True, "zarr_paths": ["/rec/a.zarr"],
            "plan": {"snapshot_id": "sha256:" + "d" * 64, "destination_root": "/rec",
                     "parents": [{"context": {"data_origin": "acquired"}}]}}
    (newer / "citrus_session_import.status.json").write_text(json.dumps(body))
    register = Register()
    assert registrar.register_completed(config, dry_run=False, register=register) == 0
    assert register.calls == [("sha256:" + "d" * 64, Path("/rec"))]
    assert not (tmp_path / "state" / f"{KEY}.import_failed").exists()


def test_a_registrar_commit_mismatch_is_terminal_with_the_needed_commit(tmp_path):
    # B1 + S1: never retried every 5 minutes; the record names the commit.
    from fisheye.intake.outcomes import RegistrarCommitMismatch

    config = _config(tmp_path)
    _submitted(tmp_path)
    _status(tmp_path)
    calls = []

    def mismatch(config, snapshot_sha, destination_root):
        calls.append(snapshot_sha)
        raise RegistrarCommitMismatch(
            "produced by another commit", code="registrar_commit_mismatch",
            details={"receipt_producer_git_sha": "a" * 40, "registrar_git_sha": "b" * 40},
        )

    assert registrar.register_completed(config, dry_run=False, register=mismatch) == 1
    assert registrar.register_completed(config, dry_run=False, register=mismatch) == 0
    assert len(calls) == 1
    refused = json.loads((tmp_path / "state" / f"{KEY}.registration_refused").read_text())
    assert refused["error"] == "registrar_commit_mismatch"
    assert refused["receipt_producer_git_sha"] == "a" * 40
    assert refused["registrar_git_sha"] == "b" * 40


def test_a_running_jobs_reserved_empty_status_is_pending_and_never_blocks_others(tmp_path):
    # P-1: the workflow reserves its status file empty and fills it at the end.
    config = _config(tmp_path)
    _submitted(tmp_path)
    run_dir = tmp_path / "logs" / "bsub_submissions" / f"citrus_import_20261006T000000Z_session_{KEY}"
    attempt = run_dir / "workflow-777-0-20261007T000000000000000Z-1"
    attempt.mkdir(parents=True)
    (attempt / "citrus_session_import.status.json").write_bytes(b"")
    other = "c" * 64
    (tmp_path / "state" / f"{other}.submitted").write_text("job_id=888\n")
    other_run = tmp_path / "logs" / "bsub_submissions" / f"citrus_import_20261006T000000Z_s_{other}"
    (other_run / "workflow-888").mkdir(parents=True)
    body = {"status": "complete", "import_complete": True, "zarr_paths": ["/rec/a.zarr"],
            "plan": {"snapshot_id": "sha256:" + "d" * 64, "destination_root": "/rec",
                     "parents": [{"context": {"data_origin": "acquired"}}]}}
    (other_run / "workflow-888" / "citrus_session_import.status.json").write_text(json.dumps(body))

    register = Register()
    assert registrar.register_completed(config, dry_run=False, register=register) == 0
    assert not (tmp_path / "state" / f"{KEY}.import_failed").exists()  # still running
    assert (tmp_path / "state" / f"{other}.registered").exists()

    # Once the job has ended, an empty status is a failed import.
    (run_dir / "777.out").write_text("Resource usage summary:\n")
    registrar.register_completed(config, dry_run=False, register=register)
    assert "empty or undecodable" in json.loads(
        (tmp_path / "state" / f"{KEY}.import_failed").read_text())["error"]


def test_one_broken_key_never_aborts_the_run(tmp_path):
    config = _config(tmp_path)
    _submitted(tmp_path)
    # Two run directories for one key: the lookup raises for this key only.
    for stamp in ("20261006T000000Z", "20261006T010000Z"):
        (tmp_path / "logs" / "bsub_submissions" / f"citrus_import_{stamp}_s_{KEY}" / "workflow-777").mkdir(parents=True)
    other = "c" * 64
    (tmp_path / "state" / f"{other}.submitted").write_text("job_id=888\n")
    other_run = tmp_path / "logs" / "bsub_submissions" / f"citrus_import_20261006T000000Z_s_{other}"
    (other_run / "workflow-888").mkdir(parents=True)
    body = {"status": "complete", "import_complete": True, "zarr_paths": ["/rec/a.zarr"],
            "plan": {"snapshot_id": "sha256:" + "d" * 64, "destination_root": "/rec",
                     "parents": [{"context": {"data_origin": "acquired"}}]}}
    (other_run / "workflow-888" / "citrus_session_import.status.json").write_text(json.dumps(body))
    assert registrar.register_completed(config, dry_run=False, register=Register()) == 1
    assert (tmp_path / "state" / f"{other}.registered").exists()



def _attached(tmp_path):
    config = _config(tmp_path)
    _submitted(tmp_path)
    (tmp_path / "state" / f"{KEY}.claimed").write_text(json.dumps({"snapshot_id": "sha256:" + "d" * 64}))
    run_dir = tmp_path / "logs" / "bsub_submissions" / f"citrus_import_20261006T000000Z_session_{KEY}"
    run_dir.mkdir(parents=True)
    (run_dir / "session.777.status.txt").write_text("job_id=777\npayload_returncode=75\n")
    return {**config, "destination_root": "/rec"}


def test_an_attached_key_registers_once_the_holder_imported_it(tmp_path):
    # SF-2(b): resolved from the claimed snapshot, through the import probe.
    config = _attached(tmp_path)
    probes = []
    register = Register()
    assert registrar.register_completed(
        config, dry_run=False, register=register,
        probe=lambda sha, root: probes.append((sha, root)) or True,
    ) == 0
    assert probes == [("sha256:" + "d" * 64, Path("/rec"))]
    assert register.calls == [("sha256:" + "d" * 64, Path("/rec"))]
    assert (tmp_path / "state" / f"{KEY}.registered").exists()


def test_an_attached_key_never_stalls_silently(tmp_path):
    config = {**_attached(tmp_path), "attached_max_ticks": 3}
    register = Register()
    for tick in range(2):
        assert registrar.register_completed(config, dry_run=False, register=register,
                                            probe=lambda sha, root: False) == 0
        pending = json.loads((tmp_path / "state" / f"{KEY}.attached_pending").read_text())
        assert pending["ticks"] == tick + 1
    assert registrar.register_completed(config, dry_run=False, register=register,
                                        probe=lambda sha, root: False) == 1
    unresolved = json.loads((tmp_path / "state" / f"{KEY}.attached_unresolved").read_text())
    assert unresolved["snapshot_id"] == "sha256:" + "d" * 64 and unresolved["ticks"] == 3
    assert not (tmp_path / "state" / f"{KEY}.attached_pending").exists()
    assert register.calls == []
    # Terminal: later runs leave it for the operator.
    assert registrar.register_completed(config, dry_run=False, register=register,
                                        probe=lambda sha, root: True) == 0
    assert register.calls == []


def test_a_transient_state_read_in_registration_is_retried_not_refused(tmp_path):
    # SF-1 at the registrar: a retryable error never writes the terminal refusal.
    from fisheye.intake.outcomes import IntakeTransient

    config = _config(tmp_path)
    _submitted(tmp_path)
    _status(tmp_path)

    def racing(config, snapshot_sha, destination_root):
        raise IntakeTransient("intake state kept changing while read")

    assert registrar.register_completed(config, dry_run=False, register=racing) == 1
    assert (tmp_path / "state" / f"{KEY}.registration_failed").exists()
    assert not (tmp_path / "state" / f"{KEY}.registration_refused").exists()
