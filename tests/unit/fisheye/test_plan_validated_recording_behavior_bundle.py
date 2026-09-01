from __future__ import annotations

import argparse
import json

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
    assert "large" not in output
