from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import fisheye.utils.plan_validated_recording_behavior_bundle as subject


def test_parses_one_typed_absent_capability() -> None:
    capability, record = subject._capability_disposition(
        "gaze=unavailable:upstream_segmentation_quality"
    )

    assert capability == "gaze"
    assert record == {
        "state": "unavailable",
        "reason_code": "upstream_segmentation_quality",
        "detail": None,
    }


def test_rejects_reason_that_is_invalid_for_state() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="invalid for"):
        subject._capability_disposition("gaze=stale:upstream_segmentation_quality")


def test_main_prints_a_bounded_summary_not_the_complete_bundle(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        subject,
        "ensure_validated_recording_behavior_bundle",
        lambda *_args, **_kwargs: {
            "status": "complete_selector_ineligible_receipt_composition",
            "mode": "created",
            "bundle_path": "/tmp/bundle.json",
            "recording_id": "recording-1",
            "record_sha256": "a" * 64,
            "capabilities": {
                "provider_motion": {"state": "complete", "reason_code": None},
                "gaze": {
                    "state": "unavailable",
                    "reason_code": "upstream_segmentation_quality",
                },
            },
            "source_bindings": {"provider_motion": {"large": "not printed"}},
            "scientific_child_bindings": {"epoch_behavior": {}},
            "safety": {"selector_eligible": False, "production_authority": False},
        },
    )

    assert (
        subject.main(
            [
                "--projection-receipt",
                "/tmp/projection.json",
                "--palette-commit",
                "b" * 40,
                "--output-json",
                "/tmp/bundle.json",
                "--absent-capability",
                "gaze=unavailable:upstream_segmentation_quality",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert output["complete_capabilities"] == ["provider_motion"]
    assert output["noncomplete_capabilities"]["gaze"]["state"] == "unavailable"
    assert output["bundle_adapter_id"] == subject.RECORDING_BUNDLE_ADAPTER_ID
    assert "large" not in output


def test_main_routes_core_plus_chaser_through_the_same_bundle_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}

    def ensure(*args: object, **kwargs: object) -> dict[str, object]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {
            "status": "complete_selector_ineligible_receipt_composition",
            "mode": "created",
            "bundle_path": "/tmp/composite.json",
            "recording_id": "recording-1",
            "record_sha256": "a" * 64,
            "capabilities": {
                "kinematics_samples": {"state": "complete", "reason_code": None},
                "spatial_occupancy": {"state": "complete", "reason_code": None},
            },
            "source_bindings": {"provider_motion": {}},
            "scientific_child_bindings": {"spatial_occupancy": {}},
            "safety": {"selector_eligible": False, "production_authority": False},
        }

    monkeypatch.setattr(subject, "ensure_core_chaser_composite_bundle", ensure)

    assert (
        subject.main(
            [
                "--projection-receipt",
                "/tmp/projection.json",
                "--core-execution-report",
                "/tmp/core-report.json",
                "--palette-commit",
                "b" * 40,
                "--output-json",
                "/tmp/composite.json",
                "--expected-analysis-zarr",
                "/tmp/recording.zarr",
                "--expected-recording-id",
                "recording-1",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert output["bundle_adapter_id"] == subject.CORE_CHASER_BUNDLE_ADAPTER_ID
    assert observed["args"] == (
        Path("/tmp/core-report.json"),
        Path("/tmp/projection.json"),
    )
    assert observed["kwargs"] == {
        "palette_commit": "b" * 40,
        "output_json": Path("/tmp/composite.json"),
        "expected_analysis_zarr": Path("/tmp/recording.zarr"),
        "expected_recording_id": "recording-1",
    }


def test_core_plus_chaser_cli_rejects_profile_reinterpretation() -> None:
    with pytest.raises(SystemExit, match="cannot reinterpret"):
        subject.main(
            [
                "--projection-receipt",
                "/tmp/projection.json",
                "--core-execution-report",
                "/tmp/core-report.json",
                "--palette-commit",
                "b" * 40,
                "--output-json",
                "/tmp/composite.json",
                "--expected-analysis-zarr",
                "/tmp/recording.zarr",
                "--expected-recording-id",
                "recording-1",
                "--absent-capability",
                "gaze=unavailable:upstream_segmentation_quality",
            ]
        )
