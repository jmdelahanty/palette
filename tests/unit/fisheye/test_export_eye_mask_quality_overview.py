from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import export_eye_mask_quality_overview as mod


PNG_BYTES = b"\x89PNG\r\n\x1a\nFAKEPNG"


def _ready_row(tmp_path: Path) -> mod.ExportRow:
    return mod.ExportRow(
        zarr_path=str(tmp_path / "rec" / "zarr" / "rec_training.zarr"),
        zarr_use="training",
        artifact_name=mod.ARTIFACT_NAME,
        profile_run="eye_mask_profile_001",
        status="ready",
        reason="ok",
        output_path=str(tmp_path / "exports" / f"rec_training__eye_mask_profile_001__{mod.ARTIFACT_NAME}.png"),
        bytes_written=None,
    )


def test_export_eye_mask_quality_overview_writes_png(monkeypatch, tmp_path: Path) -> None:
    rows = [_ready_row(tmp_path)]
    monkeypatch.setattr(mod, "_collect_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(mod, "_load_artifact_bytes", lambda row: PNG_BYTES)

    output_dir = tmp_path / "exports"
    rc = mod.main(["--output-dir", str(output_dir)])
    assert rc == 0

    expected_path = output_dir / f"rec_training__eye_mask_profile_001__{mod.ARTIFACT_NAME}.png"
    assert expected_path.exists()
    assert expected_path.read_bytes() == PNG_BYTES


def test_export_eye_mask_quality_overview_list_mode_only_lists(monkeypatch, tmp_path: Path) -> None:
    rows = [_ready_row(tmp_path)]
    monkeypatch.setattr(mod, "_collect_rows", lambda *args, **kwargs: rows)
    report = tmp_path / "report.json"

    rc = mod.main(["--list", "--json-report", str(report), "--output-dir", str(tmp_path / "exports")])
    assert rc == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "list"
    assert payload["summary"]["listed"] == 1
    row = payload["rows"][0]
    assert row["status"] == "listed"
    assert row["reason"] == "list_mode"


def test_export_eye_mask_quality_overview_recursive_filters_zarr_use(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    rows = [_ready_row(tmp_path)]

    def _fake_collect_rows(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return rows

    monkeypatch.setattr(mod, "_collect_rows", _fake_collect_rows)
    monkeypatch.setattr(mod, "_load_artifact_bytes", lambda row: PNG_BYTES)
    report = tmp_path / "report.json"

    rc = mod.main(
        [
            str(tmp_path),
            "--recursive",
            "--zarr-use",
            "training",
            "--output-dir",
            str(tmp_path / "exports"),
            "--json-report",
            str(report),
        ]
    )
    assert rc == 0
    kwargs = captured["kwargs"]
    assert kwargs["recursive"] is True
    assert kwargs["zarr_use_filter"] == "training"


def test_export_eye_mask_quality_overview_view_mode_does_not_write_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rows = [_ready_row(tmp_path)]
    monkeypatch.setattr(mod, "_collect_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(mod, "_load_artifact_bytes", lambda row: PNG_BYTES)
    viewed_calls: list[tuple[int, str]] = []

    def _fake_view(png_bytes: bytes, *, title: str) -> None:
        viewed_calls.append((len(png_bytes), title))

    monkeypatch.setattr(mod, "_view_png_bytes", _fake_view)
    report = tmp_path / "report.json"

    rc = mod.main(["--view", "--json-report", str(report), "--output-dir", str(tmp_path / "exports")])
    assert rc == 0
    assert viewed_calls
    assert viewed_calls[0][0] == len(PNG_BYTES)
    assert "eye_mask_profile_001" in viewed_calls[0][1]
    assert not (tmp_path / "exports").exists()

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "view"
    assert payload["summary"]["viewed"] == 1
    assert payload["summary"]["exported"] == 0
    row = payload["rows"][0]
    assert row["status"] == "viewed"
    assert row["reason"] == "shown"


def test_export_eye_mask_quality_overview_reports_skip_row(monkeypatch, tmp_path: Path) -> None:
    rows = [
        mod.ExportRow(
            zarr_path=str(tmp_path / "missing.zarr"),
            zarr_use="training",
            artifact_name=mod.ARTIFACT_NAME,
            profile_run=None,
            status="skip",
            reason="profile_summary_missing",
            output_path=None,
            bytes_written=None,
        )
    ]
    monkeypatch.setattr(mod, "_collect_rows", lambda *args, **kwargs: rows)
    report = tmp_path / "report.json"

    rc = mod.main(["--json-report", str(report), "--output-dir", str(tmp_path / "exports")])
    assert rc == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["scanned"] == 1
    assert payload["summary"]["ready"] == 0
    assert payload["summary"]["skipped"] == 1
    row = payload["rows"][0]
    assert row["status"] == "skip"
    assert row["reason"] == "profile_summary_missing"


def test_render_profile_overview_png_returns_png_bytes() -> None:
    summary = {
        "dataset": {"dataset_id": "dataset_a", "recording_id": "rec_a", "zarr_use": "training"},
        "source": {
            "stage_group": "refined_eye_masks_runs",
            "eye_mask_method": "refine_eye_masks",
            "review_state": "approved",
            "review_intended_use": "training",
        },
        "quality": {
            "rows_total": 10,
            "rows_usable": 9,
            "successful_roi_pair_rate": 0.9,
            "reviewed_rate": 1.0,
            "excluded_rate": 0.1,
            "exclusion_reasons_json": '{"too_close": 1}',
        },
        "geometry": {
            "eye_separation": {"stats": {"p50": 12.2}},
            "ellipse_major": {"stats": {"p50": 8.1}},
            "ellipse_minor": {"stats": {"p50": 4.2}},
            "union_area": {"stats": {"p50": 96.0}},
            "area_lr_ratio": {"stats": {"p50": 1.05}},
        },
    }
    payload = mod.render_eye_mask_profile_overview_png(
        summary,
        zarr_name="recording.zarr",
        profile_run="eye_mask_profile_001",
    )
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(payload) > 1024


def test_extract_p50_supports_nested_stats() -> None:
    assert mod._extract_p50({"stats": {"p50": 1.23}}) == 1.23


def test_extract_reason_counts_supports_fallback_fields() -> None:
    assert mod._extract_reason_counts({"excluded_reasons": {"bad_fit": 2}}) == {"bad_fit": 2}
    assert mod._extract_reason_counts({"exclusion_reasons_json": '{"bad_fit": 3}'}) == {"bad_fit": 3}
