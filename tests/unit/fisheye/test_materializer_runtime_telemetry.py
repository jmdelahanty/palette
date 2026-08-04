from __future__ import annotations

from copy import deepcopy
import time

import pytest

from fisheye.analysis_workflows.materializers.runtime_telemetry import (
    PhaseTelemetry,
    RUNTIME_TELEMETRY_IDENTITY_POLICY,
    RUNTIME_TELEMETRY_SCHEMA_ID,
    require_runtime_telemetry,
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
    assert payload["schema_version"] == 2
    assert payload["identity_policy"] == RUNTIME_TELEMETRY_IDENTITY_POLICY
    assert payload["materializer"] == "unit_test"
    assert payload["execution"]["requested_workers"] == 4
    assert [phase["name"] for phase in payload["phases"]] == ["first", "second"]
    assert all(phase["outcome"] == "ok" for phase in payload["phases"])
    assert telemetry.duration_seconds("first") is not None
    assert telemetry.duration_seconds("missing") is None
    assert payload["wall_seconds"] >= payload["phases"][0]["wall_seconds"]


def test_nested_phase_telemetry_counts_only_top_level_wall_time() -> None:
    telemetry = PhaseTelemetry(materializer="unit_test")

    with telemetry.phase("publication"):
        with telemetry.phase("validation"):
            time.sleep(0.001)
        with telemetry.phase("inventory"):
            time.sleep(0.001)

    payload = telemetry.to_json()
    publication = next(
        phase for phase in payload["phases"] if phase["name"] == "publication"
    )

    assert payload["phase_parent_by_name"] == {
        "inventory": "publication",
        "publication": None,
        "validation": "publication",
    }
    assert payload["phase_wall_seconds_sum"] == publication["wall_seconds"]

    overlapping = deepcopy(payload)
    inventory = next(
        phase for phase in overlapping["phases"] if phase["name"] == "inventory"
    )
    validation = next(
        phase for phase in overlapping["phases"] if phase["name"] == "validation"
    )
    inventory["started_at_utc"] = validation["started_at_utc"]
    with pytest.raises(ValueError, match="sibling phase intervals overlap"):
        require_runtime_telemetry(overlapping, require_current=True)


def test_legacy_flat_runtime_is_auditable_but_timing_ineligible() -> None:
    telemetry = PhaseTelemetry(materializer="unit_test")
    with telemetry.phase("publication"):
        with telemetry.phase("validation"):
            pass
    legacy = telemetry.to_json()
    legacy["schema_version"] = 1
    legacy.pop("phase_parent_by_name")
    legacy["phase_wall_seconds_sum"] = sum(
        float(phase["wall_seconds"]) for phase in legacy["phases"]
    )

    require_runtime_telemetry(legacy)
    with pytest.raises(ValueError, match="timing-ineligible"):
        require_runtime_telemetry(legacy, require_current=True)


def test_runtime_telemetry_rejects_phase_outside_execution_interval() -> None:
    telemetry = PhaseTelemetry(materializer="unit_test")
    with telemetry.phase("first"):
        pass
    payload = telemetry.to_json()
    payload["phases"][0]["started_at_utc"] = "2000-01-01T00:00:00+00:00"
    payload["phases"][0]["finished_at_utc"] = "2000-01-01T00:00:00+00:00"

    with pytest.raises(ValueError, match="escapes the execution interval"):
        require_runtime_telemetry(payload, require_current=True)


def test_phase_telemetry_records_failure_type_and_reraises() -> None:
    telemetry = PhaseTelemetry(materializer="unit_test")

    with pytest.raises(ValueError, match="expected"):
        with telemetry.phase("failure"):
            raise ValueError("expected")

    phase = telemetry.to_json()["phases"][0]
    assert phase["outcome"] == "error"
    assert phase["error_type"] == "ValueError"
