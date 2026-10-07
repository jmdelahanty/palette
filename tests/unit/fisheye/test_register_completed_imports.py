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
            "plan": {"parents": [{"context": {"data_origin": "acquired"}}]}}
    body.update(payload)
    (run_dir / f"session.{job_id}.status.json").write_text(json.dumps(body))


class Register:
    def __init__(self, fail: bool = False):
        self.calls: list[tuple[Path, Path]] = []
        self.fail = fail

    def __call__(self, registry, zarr):
        self.calls.append((registry, zarr))
        if self.fail:
            raise RuntimeError("registry busy")
        return f"dataset-{zarr.name}"


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
    assert [str(z) for _r, z in register.calls] == ["/rec/a.zarr", "/rec/b.zarr"]
    done = json.loads((tmp_path / "state" / f"{KEY}.registered").read_text())
    assert done["datasets"] == {"/rec/a.zarr": "dataset-a.zarr", "/rec/b.zarr": "dataset-b.zarr"}
    registrar.register_completed(config, dry_run=False, register=register)
    assert len(register.calls) == 2  # not registered again


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


def test_synthetic_transfer_never_reaches_the_canonical_registry(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setitem(config, "registry", str(
        __import__("fisheye.utils.citrus_transfer_parent_workflow", fromlist=["x"]).CANONICAL_REGISTRY))
    _submitted(tmp_path)
    _status(tmp_path, plan={"parents": [{"context": {"data_origin": "synthetic"}}]})
    register = Register()
    assert registrar.register_completed(config, dry_run=False, register=register) == 1
    assert register.calls == []


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


def test_main_refuses_on_a_host_that_is_not_the_writer(tmp_path, capsys):
    _config(tmp_path)
    assert registrar.main(["--config", str(tmp_path / "config.json")]) == 2
    assert "is not the writer" in capsys.readouterr().out


def test_poller_dispatches_jobs_without_registering_in_workstation_mode(tmp_path):
    (tmp_path / "staging").mkdir()
    config = _config(tmp_path)
    command = poller.build_command(config, tmp_path / "session", KEY)
    assert "--no-register" in command[-1] and "--writer-host" not in command[-1]
