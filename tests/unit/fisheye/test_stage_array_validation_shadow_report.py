from __future__ import annotations

import json
from pathlib import Path

from fisheye.registry.db import Registry
from fisheye.registry.status_ledger import upsert_recording_step_status
from fisheye.utils.report_stage_array_validation_shadow import build_shadow_validation_report


def test_shadow_validation_report_lists_invalid_rows(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.upsert_dataset(
            "dataset_invalid",
            session_uuid="dataset_invalid",
            zarr_path=tmp_path / "invalid.zarr",
            recording_id="rec",
        )
        registry.upsert_dataset(
            "dataset_ok",
            session_uuid="dataset_ok",
            zarr_path=tmp_path / "ok.zarr",
            recording_id="rec",
        )

        upsert_recording_step_status(
            registry,
            dataset_id="dataset_invalid",
            recording_id="rec",
            step_name="detect",
            status="ok",
            run_name="detect_001",
            details_json={
                "stage_array_validation_status": "invalid",
                "stage_array_validation_stage": "detect",
                "stage_array_validation_enforced": False,
                "stage_array_validation_errors": ["detect: missing required array 'frame_indices'"],
            },
        )
        upsert_recording_step_status(
            registry,
            dataset_id="dataset_ok",
            recording_id="rec",
            step_name="crop",
            status="ok",
            run_name="crop_001",
            details_json={
                "stage_array_validation_status": "ok",
                "stage_array_validation_stage": "crop",
                "stage_array_validation_enforced": False,
            },
        )

        report = build_shadow_validation_report(tmp_path / "registry.sqlite")

        assert report["matched_row_count"] == 1
        assert report["matched_validation_status_counts"] == {"invalid": 1}
        assert report["matched_stage_counts"] == {"detect": 1}
        assert report["rows"][0]["dataset_id"] == "dataset_invalid"
        assert report["rows"][0]["stage_array_validation_errors"] == [
            "detect: missing required array 'frame_indices'"
        ]
    finally:
        registry.close()


def test_shadow_validation_report_can_include_no_spec_and_filter_step(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.upsert_dataset(
            "dataset_custom",
            session_uuid="dataset_custom",
            zarr_path=tmp_path / "custom.zarr",
            recording_id="rec",
        )
        registry.upsert_dataset(
            "dataset_detect",
            session_uuid="dataset_detect",
            zarr_path=tmp_path / "detect.zarr",
            recording_id="rec",
        )
        upsert_recording_step_status(
            registry,
            dataset_id="dataset_custom",
            recording_id="rec",
            step_name="custom_stage",
            status="ok",
            run_name="custom_001",
            details_json={
                "stage_array_validation_status": "no_spec",
                "stage_array_validation_enforced": False,
            },
        )
        upsert_recording_step_status(
            registry,
            dataset_id="dataset_detect",
            recording_id="rec",
            step_name="detect",
            status="ok",
            run_name="detect_001",
            details_json={"stage_array_validation_status": "invalid"},
        )

        report = build_shadow_validation_report(
            tmp_path / "registry.sqlite",
            validation_statuses=("invalid", "no_spec"),
            step_name="custom_stage",
        )

        assert report["matched_row_count"] == 1
        assert report["matched_validation_status_counts"] == {"no_spec": 1}
        assert report["rows"][0]["step_name"] == "custom_stage"
    finally:
        registry.close()


def test_shadow_validation_report_ignores_malformed_details_json(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.upsert_dataset(
            "dataset_malformed",
            session_uuid="dataset_malformed",
            zarr_path=tmp_path / "bad.zarr",
            recording_id="rec",
        )
        with registry.conn:
            registry.conn.execute(
                """
                INSERT INTO recording_step_status (
                    dataset_id, recording_id, step_name, status, details_json, updated_utc
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                ("dataset_malformed", "rec", "detect", "ok", "{bad json", "2026-05-21T00:00:00Z"),
            )

        report = build_shadow_validation_report(tmp_path / "registry.sqlite")

        assert report["matched_row_count"] == 0
        json.dumps(report)
    finally:
        registry.close()
