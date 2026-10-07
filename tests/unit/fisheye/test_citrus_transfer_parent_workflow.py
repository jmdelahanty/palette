from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from fisheye.utils import run_citrus_session_import as runner


@pytest.fixture(autouse=True)
def _placeholder_media_sync_assessment(monkeypatch):
    """Fixture videos are text placeholders; the real check runs in the canary."""
    from fisheye.diagnostics.video import container

    monkeypatch.setattr(
        container,
        "check_hevc_keyframe_flags",
        lambda path, **_: {
            "schema_id": "palette.video.sync_sample_assessment.v1",
            "codec": "h264",
            "container_inspection_status": "ok",
            "sync_sample_proof": "container_declared",
            "message": "placeholder media",
        },
    )


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/recording_transfer_v2"


def _arguments(tmp_path, name="rolling"):
    source = Path(shutil.copytree(FIXTURES / name, tmp_path / "staging"))
    arguments = [
        str(source),
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


def test_transfer_v2_single_video_dry_run_is_planned_not_refused(
    tmp_path, monkeypatch, capsys
):
    from tests.unit.fisheye.test_transfer_recording_organization import _resign

    source, arguments = _arguments(tmp_path, "whole")
    (source / "fixture.h5").unlink()  # placeholder bytes, not an H5 container
    _resign(source)
    monkeypatch.setattr(
        runner,
        "_run_command",
        lambda *a, **k: pytest.fail("dry-run executed a command"),
    )
    assert runner.main([*arguments, "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["recording_layout"] == "single_video"
    assert [parent["clip_count"] for parent in plan["parents"]] == [1]
    assert not (tmp_path / "recordings").exists()


@pytest.mark.parametrize(
    "flag",
    [
        ["--recording-type", "behavior"],
        ["--recording-subtype", "free"],
        ["--behavior-mode", "free"],
        ["--recording-only"],
    ],
)
def test_operator_context_flags_are_gone(tmp_path, flag):
    _source, arguments = _arguments(tmp_path)
    with pytest.raises(SystemExit):
        runner.main([*arguments, *flag])
    assert not (tmp_path / "recordings").exists()


def test_stimulus_import_follows_the_declared_intent():
    from fisheye.utils.citrus_transfer_parent_workflow import _plan_recording_only

    def plan(*intents):
        return {"parents": [{"context": {"recording_intent": i}} for i in intents]}

    assert _plan_recording_only(plan("recording_only", "recording_only"))
    assert not _plan_recording_only(plan("stimulus_experiment"))
    with pytest.raises(ValueError, match="one recording intent"):
        _plan_recording_only(plan("recording_only", "stimulus_experiment"))


def test_synthetic_transfer_never_registers_into_the_canonical_registry(tmp_path):
    from argparse import Namespace

    from fisheye.utils import citrus_transfer_parent_workflow as workflow

    plan = {"parents": [{"context": {"data_origin": "synthetic"}}]}
    canonical = Namespace(register=True, registry=workflow.CANONICAL_REGISTRY)
    with pytest.raises(ValueError, match="canonical registry"):
        workflow._refuse_synthetic_canonical_registration(plan, canonical)
    isolated = Namespace(register=True, registry=tmp_path / "isolated.sqlite")
    workflow._refuse_synthetic_canonical_registration(plan, isolated)
    acquired = {"parents": [{"context": {"data_origin": "acquired"}}]}
    workflow._refuse_synthetic_canonical_registration(acquired, canonical)


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


def test_retirement_replay_reports_the_zarr_paths_the_registrar_needs(tmp_path, monkeypatch):
    # A replay from the durable "retiring" journal goes straight to finalization.
    # It must report the same zarr_paths as a fresh import, or the workstation
    # registrar refuses the completed status and the delivery is never registered.
    from fisheye.utils import citrus_transfer_parent_workflow as workflow
    from fisheye.utils import organize_transfer_recordings as organizer
    from fisheye.utils import register_completed_imports as registrar

    _source, arguments = _arguments(tmp_path)
    monkeypatch.setattr(
        workflow,
        "_run_command",
        lambda command, *, name, run_dir, pass_fds, env: runner.CommandRecord(
            name, command, 7, "unused", "unused"
        ),
    )
    assert runner.main([*arguments, "--apply"]) == 1  # organizes, then the import fails
    plan_path = tmp_path / "run/organization_plan.json"
    plan = json.loads(plan_path.read_bytes())
    [state_path] = (tmp_path / "recordings/.transfer_intake").glob("*/organization_state.json")
    state = json.loads(state_path.read_bytes())
    state["status"] = "retiring"
    state_path.write_text(json.dumps(state))
    finalized = []

    def finalize(replayed, *, registry_path, require_stimulus):
        finalized.append(replayed["snapshot_id"])
        return {"import_receipts": {"r": "receipt"}, "retired_files": []}

    monkeypatch.setattr(workflow, "finalize_transfer_staging", finalize)
    replay = list(arguments)
    replay[replay.index("--run-dir") + 1] = str(tmp_path / "workflow-778")
    assert runner.main([*replay, "--apply", "--resume-transfer-plan", str(plan_path)]) == 0
    assert finalized == [plan["snapshot_id"]]
    status = json.loads((tmp_path / "workflow-778/citrus_session_import.status.json").read_bytes())
    assert status["status"] == "complete"
    assert "zarr_paths" in status, "replay omitted zarr_paths"
    expected = [
        str(Path(p["destination_dir"]) / "zarr" / f"{Path(p['destination_dir']).name}_analysis.zarr")
        for p in plan["parents"]
    ]
    assert [str(p) for p in organizer.parent_zarr_paths(plan)] == expected
    assert len(expected) == len(plan["parents"]) == 2
    assert status["zarr_paths"] == expected

    # The registrar accepts the replayed status exactly as it would a fresh one.
    key = "c" * 64
    config = {
        "staging_dir": str(tmp_path / "staging"), "state_dir": str(tmp_path / "state"),
        "log_dir": str(tmp_path / "logs"), "submit": {"transport": "local", "repo": "/r"},
        "registration": "workstation", "registry": str(tmp_path / "registry.sqlite"),
        "writer_host": "writer", "writer_lock_path": str(tmp_path / "writer.lock"),
        "shadow_temp_root": str(tmp_path / "shadows"), "shadow_backup_dir": str(tmp_path / "backups"),
    }
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / f"{key}.submitted").write_text("job_id=778\n")
    run_dir = tmp_path / "logs/bsub_submissions" / f"citrus_import_x_staging_{key}"
    run_dir.mkdir(parents=True)
    shutil.move(str(tmp_path / "workflow-778"), str(run_dir / "workflow-778"))
    registered = []
    assert registrar.register_completed(
        config, dry_run=False, register=lambda registry, zarr: registered.append(str(zarr)) or "d"
    ) == 0
    assert registered == expected
    assert (tmp_path / "state" / f"{key}.registered").is_file()
