from __future__ import annotations

from collections.abc import Iterator
import json

import pytest

from fisheye.analytics_exports.runtime_telemetry import (
    EXPORT_RUNTIME_PHASES,
    ExportRuntimePhaseRecorder,
    validate_export_runtime_telemetry,
)


def _clock(values: Iterator[float]):
    return lambda: next(values)


def test_runtime_recorder_freezes_exact_nonoverlapping_phase_contract() -> None:
    values = iter(float(index) for index in range(22))
    recorder = ExportRuntimePhaseRecorder(clock=_clock(values))
    for phase in EXPORT_RUNTIME_PHASES:
        with recorder.measure(phase):
            pass
    payload = recorder.snapshot()

    validate_export_runtime_telemetry(payload)
    assert payload["phase_order"] == list(EXPORT_RUNTIME_PHASES)
    assert payload["measured_phase_seconds"] == 10.0
    assert payload["wall_elapsed_seconds"] == 21.0
    assert payload["unmeasured_overhead_seconds"] == 11.0


def test_runtime_telemetry_survives_canonical_sorted_json_round_trip() -> None:
    values = iter(float(index) for index in range(22))
    recorder = ExportRuntimePhaseRecorder(clock=_clock(values))
    for phase in EXPORT_RUNTIME_PHASES:
        with recorder.measure(phase):
            pass

    decoded = json.loads(json.dumps(recorder.snapshot(), sort_keys=True))
    assert tuple(decoded["phases_seconds"]) != EXPORT_RUNTIME_PHASES
    validate_export_runtime_telemetry(decoded)


def test_runtime_recorder_rejects_nested_duplicate_missing_and_tampered_phases() -> (
    None
):
    values = iter(float(index) for index in range(100))
    recorder = ExportRuntimePhaseRecorder(clock=_clock(values))
    with recorder.measure(EXPORT_RUNTIME_PHASES[0]):
        with pytest.raises(RuntimeError, match="Cannot nest"):
            with recorder.measure(EXPORT_RUNTIME_PHASES[1]):
                pass
    with pytest.raises(RuntimeError, match="already measured"):
        with recorder.measure(EXPORT_RUNTIME_PHASES[0]):
            pass
    with pytest.raises(RuntimeError, match="omitted or reordered"):
        recorder.snapshot()

    complete_values = iter(float(index) for index in range(22))
    complete = ExportRuntimePhaseRecorder(clock=_clock(complete_values))
    for phase in EXPORT_RUNTIME_PHASES:
        with complete.measure(phase):
            pass
    payload = complete.snapshot()
    payload["phases_seconds"][EXPORT_RUNTIME_PHASES[0]] = -1.0
    with pytest.raises(ValueError, match="duration is invalid"):
        validate_export_runtime_telemetry(payload)
