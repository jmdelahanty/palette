from __future__ import annotations

import time

import pytest

from fisheye.analysis_workflows.materializers.runtime_telemetry import (
    PhaseTelemetry,
    RUNTIME_TELEMETRY_IDENTITY_POLICY,
    RUNTIME_TELEMETRY_SCHEMA_ID,
)


def test_phase_telemetry_records_ordered_report_only_measurements() -> None:
    telemetry = PhaseTelemetry(
        materializer="unit_test",
        context={"requested_workers": 4},
    )

    with telemetry.phase("first"):
        time.sleep(0.001)
    with telemetry.phase("second"):
        sum(range(100))

    payload = telemetry.to_json()

    assert payload["schema_id"] == RUNTIME_TELEMETRY_SCHEMA_ID
    assert payload["schema_version"] == 1
    assert payload["identity_policy"] == RUNTIME_TELEMETRY_IDENTITY_POLICY
    assert payload["materializer"] == "unit_test"
    assert payload["execution"]["requested_workers"] == 4
    assert [phase["name"] for phase in payload["phases"]] == ["first", "second"]
    assert all(phase["outcome"] == "ok" for phase in payload["phases"])
    assert telemetry.duration_seconds("first") is not None
    assert telemetry.duration_seconds("missing") is None
    assert payload["wall_seconds"] >= payload["phases"][0]["wall_seconds"]


def test_phase_telemetry_records_failure_type_and_reraises() -> None:
    telemetry = PhaseTelemetry(materializer="unit_test")

    with pytest.raises(ValueError, match="expected"):
        with telemetry.phase("failure"):
            raise ValueError("expected")

    phase = telemetry.to_json()["phases"][0]
    assert phase["outcome"] == "error"
    assert phase["error_type"] == "ValueError"

