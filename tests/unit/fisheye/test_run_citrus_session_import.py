from __future__ import annotations

import pytest

from fisheye.utils import run_citrus_session_import as import_mod


def test_the_subprocess_and_log_transport_is_gone() -> None:
    # Review F1: parents are handed to the recording import owner in-process
    # (fisheye.intake.importing.import_parents); nothing is parsed from logs.
    for retired in (
        "build_import_command",
        "_run_command",
        "_newest_jsonl",
        "_read_zarr_paths_from_import_log",
        "_verify_import_acknowledgments",
    ):
        assert not hasattr(import_mod, retired), retired


@pytest.mark.parametrize(
    "legacy_flag",
    [
        "--transfer-v2", "--run-h5-diagnostics", "--run-video-diagnostics", "--no-rename-cams",
        # Recording context now comes only from the producer's transfer snapshot.
        "--recording-only",
    ],
)
def test_legacy_poller_options_are_gone(legacy_flag, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        import_mod.main(["/unused", legacy_flag])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags", [["--register"], ["--registry", "/r.sqlite"], ["--register", "--registry", "/r.sqlite"]]
)
def test_job_mode_registration_is_retired(flags, capsys, monkeypatch) -> None:
    from fisheye.utils import citrus_transfer_parent_workflow as workflow

    monkeypatch.setattr(
        workflow, "run_transfer_parent_workflow", lambda args: pytest.fail("dispatched")
    )
    with pytest.raises(SystemExit) as exc:
        import_mod.main(["/session", "--apply", *flags])
    assert exc.value.code == 2
    assert "job-mode registration is retired" in capsys.readouterr().err


def test_runner_dispatches_only_to_transfer_parent_workflow(monkeypatch) -> None:
    from fisheye.utils import citrus_transfer_parent_workflow as workflow

    seen = []
    monkeypatch.setattr(workflow, "run_transfer_parent_workflow", lambda args: seen.append(args) or 0)
    assert import_mod.main(["/session"]) == 0
    assert seen[0].dry_run and not seen[0].apply
