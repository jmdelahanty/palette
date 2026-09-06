from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.recording_preflight import (
    build_manifest_preflight_payload,
    preflight_gate_reason,
)


@pytest.mark.parametrize("section,field", [
    ("video", "status"), ("video", "media_status"),
    ("video", "tooling_status"), ("h5", "status"),
    ("h5", "core_status"), ("h5", "optional_status"),
    ("h5", "tooling_status"),
])
@pytest.mark.parametrize("failure", ["fail", "error"])
@pytest.mark.parametrize("summary", ["pass", "warn", "not_run"])
def test_summary_cannot_hide_any_component_failure(
    tmp_path: Path, section: str, field: str, failure: str, summary: str,
) -> None:
    payload = {"status": summary, section: {"status": "pass", field: failure}}
    (tmp_path / "recording_manifest.json").write_text(json.dumps({"preflight": payload}))
    assert preflight_gate_reason(tmp_path) is not None


@pytest.mark.parametrize("payload", [
    '{', '[]', '{"preflight":{"status":"fail","status":"pass"}}',
    '{"preflight":[]}', '{"preflight":{"status":"unknown"}}',
    '{"preflight":{"status":true}}', '{"preflight":{"video":[]}}',
    '{"preflight":{"status":"pass","video":{"error":"decoder crashed"}}}',
    '{"preflight":{"status":"pass","error":"diagnostics failed"}}',
])
def test_malformed_or_error_preflight_is_not_optional(tmp_path: Path, payload: str) -> None:
    (tmp_path / "recording_manifest.json").write_text(payload)
    assert preflight_gate_reason(tmp_path) is not None


@pytest.mark.parametrize("preflight", [
    None, {"status": "not_run", "video": None, "h5": None},
    {"status": "pass", "video": {"media_status": "pass", "tooling_status": "skip"}},
    {"status": "warn", "h5": {"core_status": "pass", "optional_status": "warn"}},
])
def test_contract_permitted_absence_or_warning_is_not_failure(tmp_path: Path, preflight) -> None:
    (tmp_path / "recording_manifest.json").write_text(json.dumps({"preflight": preflight}))
    assert preflight_gate_reason(tmp_path) is None


def test_preflight_api_has_no_failure_override(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="allow_failures"):
        preflight_gate_reason(tmp_path, allow_failures=True)


def test_manifest_builder_reports_optional_failure_as_failure() -> None:
    payload = build_manifest_preflight_payload(
        checked_at_utc="2026-09-06T00:00:00Z",
        h5={"status": "warn", "core_status": "pass", "optional_status": "fail"},
    )
    assert payload["status"] == "fail"


@pytest.mark.parametrize("module,args", [
    ("import_recording_analysis", ["--recording-dir", "/unused"]),
    ("run_recording_analysis_pipeline", ["--recording-dir", "/unused"]),
    ("import_organized_recordings_analysis", []),
    ("import_recordings_analysis", []),
    ("run_citrus_session_import", ["/unused"]),
])
def test_ingestion_clis_reject_failure_override(module, args, capsys) -> None:
    from importlib import import_module

    command = import_module(f"fisheye.utils.{module}")
    with pytest.raises(SystemExit) as exc:
        command.main([*args, "--allow-preflight-failures"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --allow-preflight-failures" in capsys.readouterr().err


@pytest.mark.parametrize("module,args", [
    ("import_recording_analysis", ["--recording-dir", "/unused"]),
    ("run_recording_analysis_pipeline", ["--recording-dir", "/unused"]),
    ("import_organized_recordings_analysis", []),
])
def test_ingestion_clis_cannot_disable_acquisition_metadata(module, args, capsys) -> None:
    from importlib import import_module

    command = import_module(f"fisheye.utils.{module}")
    with pytest.raises(SystemExit) as exc:
        command.main([*args, "--no-import-video-metadata"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --no-import-video-metadata" in capsys.readouterr().err
