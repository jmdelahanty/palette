"""Session importer routes a unified H5 to the explicit native profile.

A unified experimental H5 must reach ``import_stimulus_to_zarr`` with
``--source-profile`` and its external finalization receipt. Missing receipts,
unknown profiles and legacy-only options are refused before any subprocess
runs; legacy H5s keep their exact historical command.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import zarr

from fisheye.shared.unified_h5 import PROFILE
from fisheye.shared.unified_h5.reference import open_unified_source
from fisheye.utils import import_recording_analysis as mod
from tests.unit.fisheye.test_import_recording_analysis import (
    _acquisition_authority_updates,
    _opts,
    _write_current_manifest,
)
from tests.unit.fisheye.unified_h5_fixtures import emit_fixture, write_receipt


def _plan(tmp_path: Path, h5_path: Path, receipt: Path | None = None):
    return mod.RecordingAnalysisPlan(
        recording_dir=tmp_path,
        h5_path=h5_path,
        cam_video=tmp_path / "full.mp4",
        zarr_path=tmp_path / "analysis.zarr",
        finalization_receipt_path=receipt,
    )


def _options(**overrides):
    values = dict(
        import_video_metadata=True,
        video_metadata_overwrite=False,
        import_stimulus=True,
        stimulus_always=False,
        stimulus_run_name="candidate",
        stimulus_overwrite=False,
        stimulus_quiet=True,
    )
    values.update(overrides)
    return mod.RecordingImportOptions(**values)


def _capture_subprocess(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", run)
    return calls


def _legacy_h5(path: Path) -> Path:
    with h5py.File(path, "w") as h5:
        h5.attrs["session_uuid"] = "2026-08-12T21-59-55Z_arena_3"
    return path


def test_unified_h5_is_routed_with_profile_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_subprocess(monkeypatch)
    h5_path = emit_fixture(tmp_path, "base")
    receipt = write_receipt(tmp_path, "base")

    ok, _code, command = mod.run_stimulus_import(
        _plan(tmp_path, h5_path, receipt), _options()
    )

    assert ok
    assert command == [
        sys.executable,
        "-m",
        "fisheye.analysis.import_stimulus_to_zarr",
        str(h5_path),
        str(tmp_path / "analysis.zarr"),
        "--source-profile",
        PROFILE,
        "--finalization-receipt",
        str(receipt),
        "--run-name",
        "candidate",
        "--quiet",
    ]
    assert calls == [command]


def test_legacy_h5_command_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_subprocess(monkeypatch)
    h5_path = _legacy_h5(tmp_path / "legacy.h5")

    ok, _code, command = mod.run_stimulus_import(_plan(tmp_path, h5_path), _options())

    assert ok
    assert "--source-profile" not in command
    assert "--finalization-receipt" not in command
    assert calls == [command]


@pytest.mark.parametrize(
    ("receipt", "overrides", "reason"),
    [
        (False, {}, "unified_h5_requires_finalization_receipt"),
        (True, {"stimulus_overwrite": True}, "unified_h5_import_is_immutable_no_overwrite"),
        (
            True,
            {"stimulus_metadata_and_calibration_only": True},
            "unified_h5_has_no_metadata_only_import",
        ),
    ],
)
def test_unified_h5_refusals_happen_before_any_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: bool,
    overrides: dict,
    reason: str,
) -> None:
    calls = _capture_subprocess(monkeypatch)
    h5_path = emit_fixture(tmp_path, "base")
    receipt_path = write_receipt(tmp_path, "base") if receipt else None

    result = mod.run_stimulus_import(
        _plan(tmp_path, h5_path, receipt_path), _options(**overrides)
    )

    assert result == (False, 2, [reason])
    assert calls == []


def test_unknown_unified_profile_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_subprocess(monkeypatch)
    h5_path = emit_fixture(tmp_path, "base")
    with h5py.File(h5_path, "r+") as h5:
        h5["/metadata/session"].attrs["recording_artifact_profile"] = "unified_experimental_h5_v2"

    result = mod.run_stimulus_import(
        _plan(tmp_path, h5_path, write_receipt(tmp_path, "base")), _options()
    )

    assert result == (False, 2, ["unsupported_unified_h5_profile:unified_experimental_h5_v2"])
    assert calls == []


def test_routed_import_runs_the_real_importer(tmp_path: Path) -> None:
    h5_path = emit_fixture(tmp_path, "base")
    receipt = write_receipt(tmp_path, "base")
    plan = _plan(tmp_path, h5_path, receipt)

    ok, code, _command = mod.run_stimulus_import(plan, _options())

    assert (ok, code) == (True, 0)
    root = zarr.open_group(str(plan.zarr_path), mode="r", use_consolidated=True)
    candidate = open_unified_source(root, run_name="candidate")
    assert candidate.read_table("/frames/stimulus", start=0, stop=1).shape == (1,)


def test_unified_setup_without_native_import_is_logged_not_silently_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recording = tmp_path / "rec"
    (recording / "raw").mkdir(parents=True)
    h5_path = emit_fixture(recording / "raw", "base")
    plan = mod.RecordingAnalysisPlan(
        recording_dir=recording,
        h5_path=h5_path,
        cam_video=recording / "cams" / "cam.mp4",
        zarr_path=recording / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts()
    opts.import_video_metadata = True
    _write_current_manifest(plan.recording_dir)
    monkeypatch.setattr(mod, "git_identity", lambda **_kwargs: {"git_sha": "1" * 40, "git_dirty": False})
    monkeypatch.setattr(mod, "apply_video_metadata", lambda _plan, **_kwargs: _acquisition_authority_updates())
    monkeypatch.setattr(mod, "ensure_analysis_archive", lambda _plan: None)
    monkeypatch.setattr(mod, "apply_acquisition_frame_clock", lambda _plan: {})

    def legacy_setup(_plan):
        raise AssertionError("legacy subject reader must not run on a unified H5")

    monkeypatch.setattr(mod, "import_experiment_setup", legacy_setup)
    events: list[tuple[str, dict]] = []

    mod.process_recording_import(
        plan, opts, logger=lambda event, **fields: events.append((event, fields))
    )

    assert (
        "experiment_setup_not_projected",
        "unified_h5_metadata_requires_native_import",
    ) in [(event, fields.get("reason")) for event, fields in events]
