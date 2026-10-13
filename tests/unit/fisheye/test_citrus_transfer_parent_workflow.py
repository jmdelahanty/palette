from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

from fisheye.intake import importing
from fisheye.intake.outcomes import EXIT_HELD, EXIT_REFUSED
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


def _failed_import(calls):
    """Stand-in for the in-process import owner: every parent fails."""

    def failed(plan, *, recording_only, lease_fd):
        assert os.fstat(lease_fd)  # the workflow lock, lent to the stimulus child
        calls.append(plan["snapshot_id"])
        return [
            importing.ParentImportResult(
                recording_id=parent["identity"]["recording_id"],
                camera_id=parent["identity"]["camera_id"],
                recording_dir=parent["destination_dir"],
                zarr_path="unused",
                outcome="failed",
                failed_step="ensure_analysis_archive",
                error="fixture failure",
            )
            for parent in plan["parents"]
        ]

    return failed


def test_transfer_v2_dry_run_writes_nothing_and_never_calls_legacy_organizer(
    tmp_path, monkeypatch, capsys
):
    source, arguments = _arguments(tmp_path)
    before = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    monkeypatch.setattr(
        importing, "import_parents", lambda *a, **k: pytest.fail("dry-run imported")
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
        importing, "import_parents", lambda *a, **k: pytest.fail("dry-run imported")
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
    from fisheye.intake.delivery import plan_recording_only

    def plan(*intents):
        return {"parents": [{"context": {"recording_intent": i}} for i in intents]}

    assert plan_recording_only(plan("recording_only", "recording_only"))
    assert not plan_recording_only(plan("stimulus_experiment"))
    with pytest.raises(ValueError, match="one recording intent"):
        plan_recording_only(plan("recording_only", "stimulus_experiment"))


def test_synthetic_transfer_never_registers_by_default_nor_into_the_canonical_registry(tmp_path):
    from fisheye.intake.delivery import CANONICAL_REGISTRY, refuse_synthetic_registration
    from fisheye.intake.outcomes import IntakeRefused

    plan = {"parents": [{"context": {"data_origin": "synthetic"}}]}
    with pytest.raises(IntakeRefused, match="never registered"):
        refuse_synthetic_registration(plan, registry=tmp_path / "isolated.sqlite")
    with pytest.raises(IntakeRefused, match="canonical registry"):
        refuse_synthetic_registration(plan, registry=CANONICAL_REGISTRY, allow_synthetic=True)
    refuse_synthetic_registration(plan, registry=tmp_path / "isolated.sqlite", allow_synthetic=True)
    acquired = {"parents": [{"context": {"data_origin": "acquired"}}]}
    refuse_synthetic_registration(acquired, registry=CANONICAL_REGISTRY)


def test_transfer_v2_failed_import_keeps_every_source_and_reports_incomplete(
    tmp_path, monkeypatch
):
    source, arguments = _arguments(tmp_path)
    before = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    calls = []
    monkeypatch.setattr(importing, "import_parents", _failed_import(calls))
    assert runner.main([*arguments, "--apply"]) == 1
    assert len(calls) == 1
    assert {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    } == before
    status = json.loads(
        (tmp_path / "run/citrus_session_import.status.json").read_bytes()
    )
    assert status["status"] == "failed"
    assert status["staging_finalized"] is False
    assert status["import_complete"] is False
    assert status["registry"] is None
    assert [parent["outcome"] for parent in status["parents"]] == ["failed", "failed"]
    assert "fixture failure" in status["error"]
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
    assert runner.main([*arguments, "--apply"]) == EXIT_REFUSED
    assert not (tmp_path / "recordings").exists()


@pytest.mark.parametrize("target", ["source_dotdot", "existing_report", "coordinator_lock"])
def test_status_destination_never_overwrites_existing_or_reserved_evidence(
    tmp_path, monkeypatch, target
):
    source, arguments = _arguments(tmp_path)
    from fisheye.utils.organize_transfer_recordings import (
        build_transfer_organization_plan,
        _state_directory,
    )

    monkeypatch.setattr(importing, "import_parents", _failed_import([]))
    protected = None
    if target == "source_dotdot":
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
    assert runner.main([*arguments, "--apply", "--status-json", str(status)]) == EXIT_REFUSED
    if protected is not None:
        assert protected.read_bytes() == before
    assert not (tmp_path / "recordings").exists()
    assert not (tmp_path / "run").exists()


def test_whole_workflow_lock_prevents_second_importer_invocation(tmp_path, monkeypatch):
    _source, arguments = _arguments(tmp_path)
    calls = []
    failed = _failed_import(calls)

    def competing(plan, *, recording_only, lease_fd):
        if not calls:
            second = list(arguments)
            second[second.index("--run-dir") + 1] = str(tmp_path / "second-run")
            # Held: exit 75, and the lock is taken before any side effect.
            assert runner.main([*second, "--apply"]) == EXIT_HELD
            assert not (tmp_path / "second-run").exists()
        return failed(plan, recording_only=recording_only, lease_fd=lease_fd)

    monkeypatch.setattr(importing, "import_parents", competing)
    assert runner.main([*arguments, "--apply"]) == 1
    assert len(calls) == 1


def test_retirement_replay_reports_the_zarr_paths_the_registrar_needs(tmp_path, monkeypatch):
    # A replay from the durable "retiring" journal goes straight to finalization.
    # It must report the same zarr_paths as a fresh import, or the workstation
    # registrar refuses the completed status and the delivery is never registered.
    from fisheye.utils import organize_transfer_recordings as organizer
    from fisheye.utils import register_completed_imports as registrar

    _source, arguments = _arguments(tmp_path)
    monkeypatch.setattr(importing, "import_parents", _failed_import([]))
    assert runner.main([*arguments, "--apply"]) == 1  # organizes, then the import fails
    plan_path = tmp_path / "run/organization_plan.json"
    plan = json.loads(plan_path.read_bytes())
    [state_path] = (tmp_path / "recordings/.transfer_intake").glob("*/organization_state.json")
    state = json.loads(state_path.read_bytes())
    state["status"] = "retiring"
    state_path.write_text(json.dumps(state))
    finalized = []
    receipts = {parent["identity"]["recording_id"]: "f" * 64 for parent in plan["parents"]}

    def finalize(replayed, *, registry_path, require_stimulus):
        assert registry_path is None  # the LSF side never names a registry
        finalized.append(replayed["snapshot_id"])
        return {**state, "status": "complete", "import_receipts": receipts, "retired_files": []}

    monkeypatch.setattr(organizer, "finalize_transfer_staging", finalize)
    monkeypatch.setattr(
        "fisheye.intake.probes.receipt_producer_git_sha", lambda zarr, receipt: "1" * 40
    )
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
    assert status["probe_import"]["zarr_paths"] == expected

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

    def register(config, snapshot_sha, destination_root):
        registered.append((snapshot_sha, destination_root))
        return {path: f"dataset-{index}" for index, path in enumerate(expected)}

    monkeypatch.setattr(
        "fisheye.intake.delivery.refuse_synthetic_registration", lambda *a, **k: None
    )
    monkeypatch.setattr(registrar, "refuse_synthetic_registration", lambda *a, **k: None)
    assert registrar.register_completed(config, dry_run=False, register=register) == 0
    assert registered == [(plan["snapshot_id"], Path(plan["destination_root"]))]
    assert (tmp_path / "state" / f"{key}.registered").is_file()
