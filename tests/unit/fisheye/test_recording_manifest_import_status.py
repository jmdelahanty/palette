from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils.recording_manifest_import_status import (
    ManifestImportStatusUpdate,
    iter_updates_from_import_log,
    write_manifest_import_status,
)


def test_write_manifest_import_status_updates_top_level_fields(tmp_path: Path) -> None:
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "recording_manifest.json").write_text(
        json.dumps({"recording_name": "rec", "import_status": None}),
        encoding="utf-8",
    )

    result = write_manifest_import_status(
        ManifestImportStatusUpdate(
            recording_dir=rec,
            zarr_path=rec / "zarr" / "rec_analysis.zarr",
            status="ok",
            import_log=tmp_path / "import.jsonl",
            imported_at_utc="2026-06-21T23:18:53Z",
            import_run_id="run-1",
            registry_path=tmp_path / "registry.sqlite",
            registry_dataset_id="dataset-1",
            registry_synced_at_utc="2026-06-21T23:19:00Z",
        )
    )

    assert result.changed is True
    payload = json.loads((rec / "recording_manifest.json").read_text(encoding="utf-8"))
    assert payload["import_status"] == "ok"
    assert payload["import_log"] == str(tmp_path / "import.jsonl")
    assert payload["imported_at_utc"] == "2026-06-21T23:18:53Z"
    assert payload["analysis_import"]["run_id"] == "run-1"
    assert payload["registry_dataset_id"] == "dataset-1"


def test_iter_updates_from_import_log_reads_recording_ok_rows(tmp_path: Path) -> None:
    rec = tmp_path / "rec"
    zarr_path = rec / "zarr" / "rec_analysis.zarr"
    log_path = tmp_path / "import.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "recording_plan", "recording_dir": str(rec), "zarr_path": str(zarr_path)}),
                json.dumps(
                    {
                        "event": "recording_ok",
                        "recording_dir": str(rec),
                        "zarr_path": str(zarr_path),
                        "run_id": "run-1",
                        "ts_utc": "2026-06-21T23:18:53Z",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    updates = list(iter_updates_from_import_log(log_path))

    assert len(updates) == 1
    assert updates[0].recording_dir == rec
    assert updates[0].zarr_path == zarr_path
    assert updates[0].status == "ok"
    assert updates[0].imported_at_utc == "2026-06-21T23:18:53Z"
