from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import finalize_eye_mask_profile_artifacts as mod


def _row(
    tmp_path: Path,
    *,
    status: str = "ok",
    reason: str = "eligible",
    action: str = "render",
) -> mod.FinalizeRow:
    return mod.FinalizeRow(
        zarr_path=str(tmp_path / "rec_training.zarr"),
        zarr_use="training",
        profile_run="eye_mask_profile_001",
        review_state="approved",
        review_intended_use="training",
        review_method="manual",
        review_timestamp_utc="2026-02-25T00:00:01+00:00",
        source_stage_group="refined_eye_masks_runs",
        source_eye_mask_method="refine_eye_masks",
        artifact_signature="abc123",
        status=status,
        reason=reason,
        action=action,
    )


def test_dry_run_summary_uses_build_rows(monkeypatch, tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, status="ok", reason="eligible", action="render"),
        _row(tmp_path, status="skip", reason="review_state_mismatch", action="skip"),
    ]
    monkeypatch.setattr(mod, "_build_rows", lambda *args, **kwargs: rows)
    report_path = tmp_path / "report.json"

    rc = mod.main([str(tmp_path), "--json-report", str(report_path)])
    assert rc == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "dry-run"
    assert report["summary"]["scanned"] == 2
    assert report["summary"]["eligible"] == 1
    assert report["summary"]["would_finalize"] == 1
    assert report["summary"]["errors"] == 0


def test_apply_calls_finalize_row_and_counts_rendered(monkeypatch, tmp_path: Path) -> None:
    rows = [_row(tmp_path, status="ok", action="render")]
    monkeypatch.setattr(mod, "_build_rows", lambda *args, **kwargs: rows)
    finalized: list[tuple[str, int, bool]] = []

    def _fake_finalize(row: mod.FinalizeRow, *, visuals_dpi: int, force: bool) -> str:
        finalized.append((row.zarr_path, visuals_dpi, force))
        return "rendered"

    monkeypatch.setattr(mod, "_finalize_row", _fake_finalize)
    report_path = tmp_path / "report.json"

    rc = mod.main([str(tmp_path), "--apply", "--visuals-dpi", "222", "--json-report", str(report_path)])
    assert rc == 0
    assert finalized == [(rows[0].zarr_path, 222, False)]

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "apply"
    assert report["summary"]["rendered"] == 1
    assert report["summary"]["errors"] == 0


def test_apply_counts_unchanged_rows(monkeypatch, tmp_path: Path) -> None:
    rows = [_row(tmp_path, status="ok", action="unchanged")]
    monkeypatch.setattr(mod, "_build_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(mod, "_finalize_row", lambda *args, **kwargs: "unchanged")
    report_path = tmp_path / "report.json"

    rc = mod.main([str(tmp_path), "--apply", "--json-report", str(report_path)])
    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["unchanged"] == 1
    assert report["summary"]["rendered"] == 0


def test_apply_errors_return_nonzero(monkeypatch, tmp_path: Path) -> None:
    rows = [_row(tmp_path, status="ok", action="render")]
    monkeypatch.setattr(mod, "_build_rows", lambda *args, **kwargs: rows)

    def _raise_finalize(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_finalize_row", _raise_finalize)
    report_path = tmp_path / "report.json"

    rc = mod.main([str(tmp_path), "--apply", "--json-report", str(report_path)])
    assert rc == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["errors"] == 1
