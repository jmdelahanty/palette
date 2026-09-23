from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from fisheye.utils import run_citrus_session_import as runner

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/recording_transfer_v2"


def _arguments(tmp_path):
    source = Path(shutil.copytree(FIXTURES / "rolling", tmp_path / "staging"))
    arguments = [
        str(source),
        "--transfer-v2",
        "--recording-only",
        "--recording-type",
        "behavior",
        "--recording-subtype",
        "free",
        "--behavior-mode",
        "free",
        "--dest-root",
        str(tmp_path / "recordings"),
        "--run-dir",
        str(tmp_path / "run"),
    ]
    return source, arguments


def test_transfer_v2_dry_run_writes_nothing_and_never_calls_legacy_organizer(
    tmp_path, monkeypatch, capsys
):
    source, arguments = _arguments(tmp_path)
    before = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    monkeypatch.setattr(
        runner,
        "_run_command",
        lambda *a, **k: pytest.fail("dry-run executed a command"),
    )
    assert runner.main([*arguments, "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["status"] == "organization_planned"
    assert len(payload["plan"]["parents"]) == 2
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "recordings").exists()
    assert {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    } == before


@pytest.mark.parametrize(
    "field", ["--recording-type", "--recording-subtype", "--behavior-mode"]
)
def test_transfer_v2_requires_explicit_scientific_context(tmp_path, field):
    _source, arguments = _arguments(tmp_path)
    position = arguments.index(field)
    del arguments[position : position + 2]
    with pytest.raises(SystemExit):
        runner.main(arguments)
    assert not (tmp_path / "recordings").exists()


def test_transfer_v2_failed_import_keeps_every_source_and_reports_incomplete(
    tmp_path, monkeypatch
):
    source, arguments = _arguments(tmp_path)
    from fisheye.utils import citrus_transfer_parent_workflow as workflow

    before = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    calls = []

    def failed(command, *, name, run_dir, pass_fds, env):
        assert len(pass_fds) == 1
        assert env["PALETTE_RECORDING_IMPORT_LEASE_FD"] == str(pass_fds[0])
        calls.append(command)
        return runner.CommandRecord(
            name, command, 7, str(run_dir / "stdout"), str(run_dir / "stderr")
        )

    monkeypatch.setattr(workflow, "_run_command", failed)
    assert runner.main([*arguments, "--apply"]) == 1
    assert len(calls) == 1
    assert "fisheye.utils.import_organized_recordings_analysis" in calls[0]
    assert {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    } == before
    status = json.loads(
        (tmp_path / "run/citrus_session_import.status.json").read_bytes()
    )
    assert status["status"] == "failed"
    assert status["staging_finalized"] is False
    assert status["import_complete"] is False
    assert (tmp_path / "run/organization_plan.json").is_file()
    assert not (source / "_palette_batch_disposition.json").exists()


@pytest.mark.parametrize(
    "location", ["inside_source", "source_ancestor", "inside_parent"]
)
def test_transfer_v2_logs_cannot_modify_source_or_parent_recording(tmp_path, location):
    source, arguments = _arguments(tmp_path)
    if location == "inside_source":
        output = source / "logs"
    elif location == "source_ancestor":
        output = tmp_path
    else:
        from fisheye.utils.organize_transfer_recordings import (
            build_transfer_organization_plan,
        )

        plan = build_transfer_organization_plan(
            source,
            destination_root=tmp_path / "recordings",
            recording_type="behavior",
            recording_subtype="free",
            behavior_mode="free",
        )
        output = Path(plan["parents"][0]["destination_dir"]) / "logs"
    arguments[arguments.index("--run-dir") + 1] = str(output)
    assert runner.main([*arguments, "--apply"]) == 1
    assert not (tmp_path / "recordings").exists()


@pytest.mark.parametrize(
    "target", ["registry", "source_dotdot", "existing_report", "coordinator_lock"]
)
def test_status_destination_never_overwrites_existing_or_reserved_evidence(
    tmp_path, monkeypatch, target
):
    source, arguments = _arguments(tmp_path)
    from fisheye.utils import citrus_transfer_parent_workflow as workflow
    from fisheye.utils.organize_transfer_recordings import (
        build_transfer_organization_plan,
        _state_directory,
    )

    monkeypatch.setattr(
        workflow,
        "_run_command",
        lambda command, name, run_dir, **kwargs: runner.CommandRecord(
            name, command, 7, "unused", "unused"
        ),
    )
    protected = None
    if target == "registry":
        protected = tmp_path / "registry.sqlite"
        protected.write_bytes(b"SQLite format 3\x00preserve this database")
        status = protected
        arguments.extend(["--register", "--registry", str(protected)])
    elif target == "source_dotdot":
        (tmp_path / "external").mkdir()
        status = tmp_path / "external/../staging/recording_session.json"
        protected = source / "recording_session.json"
    elif target == "existing_report":
        protected = tmp_path / "existing.json"
        protected.write_bytes(b"not owned by this invocation")
        status = protected
    else:
        plan = build_transfer_organization_plan(
            source,
            destination_root=tmp_path / "recordings",
            recording_type="behavior",
            recording_subtype="free",
            behavior_mode="free",
        )
        status = _state_directory(plan).with_suffix(".lock")
    before = protected.read_bytes() if protected is not None else None
    assert runner.main([*arguments, "--apply", "--status-json", str(status)]) == 1
    if protected is not None:
        assert protected.read_bytes() == before
    assert not (tmp_path / "recordings").exists()


def test_whole_workflow_lock_prevents_second_importer_invocation(tmp_path, monkeypatch):
    _source, arguments = _arguments(tmp_path)
    from fisheye.utils import citrus_transfer_parent_workflow as workflow

    calls = []

    def competing(command, *, name, run_dir, pass_fds, env):
        assert len(pass_fds) == 1
        assert env["PALETTE_RECORDING_IMPORT_LEASE_FD"] == str(pass_fds[0])
        calls.append(command)
        if len(calls) == 1:
            second = list(arguments)
            second[second.index("--run-dir") + 1] = str(tmp_path / "second-run")
            assert runner.main([*second, "--apply"]) == 1
        return runner.CommandRecord(name, command, 7, "unused", "unused")

    monkeypatch.setattr(workflow, "_run_command", competing)
    assert runner.main([*arguments, "--apply"]) == 1
    assert len(calls) == 1
