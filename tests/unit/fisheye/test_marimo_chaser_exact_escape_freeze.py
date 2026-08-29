from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest

from apps.marimo.components.chaser_exact.escape_freeze import (
    ESCAPE_FREEZE_DISPLAY_RECIPE,
    ESCAPE_FREEZE_MAX_EVENT_POINTS,
    _display_indices,
    _escape_freeze_values,
    build_exact_escape_freeze_output,
)
from apps.marimo.components.chaser_exact.provenance import (
    build_projection_provenance,
)
from fisheye.analysis_workflows.escape_freeze_successor import (
    prepare_escape_freeze_successor,
)
from tests.unit.fisheye.test_escape_freeze_successor import _source


class _Ui:
    @staticmethod
    def table(records, **kwargs):
        return {"records": records, "kwargs": kwargs}


class _Marimo:
    ui = _Ui()

    @staticmethod
    def callout(value, **kwargs):
        return {"callout": value, **kwargs}

    @staticmethod
    def md(value):
        return {"markdown": value}

    @staticmethod
    def vstack(values):
        return values


def _projection(*, arrays: dict[str, np.ndarray] | None = None, source=None):
    prepared = prepare_escape_freeze_successor(_source() if source is None else source)
    payload = prepared.arrays if arrays is None else MappingProxyType(arrays)
    handle = SimpleNamespace(
        successor_kind="chaser_escape_freeze",
        deep_audited=True,
        scientific_manifest=prepared.manifest,
        run_path="analysis/chaser_escape_freeze_runs/escape-v1",
        manifest_sha256="a" * 64,
        scientific_payload_sha256=prepared.payload_digest,
        array=lambda name: payload[name],
    )
    spatial = SimpleNamespace(
        recording_id="recording-1",
        run_path="analysis/chaser_spatial_occupancy_runs/spatial-v1",
        manifest_sha256="b" * 64,
    )
    radials = tuple(
        SimpleNamespace(
            run_path=f"analysis/chaser_radial_near_field_runs/radial-{index}",
            manifest_sha256=str(index) * 64,
        )
        for index in (1, 2)
    )
    proofs = tuple(
        SimpleNamespace(provenance_record=lambda: {"verified": True}) for _ in range(2)
    )
    provenance = build_projection_provenance(
        spatial=spatial,
        radials=radials,
        relative_bindings=(
            {"run_path": "relative-1", "manifest_sha256": "1" * 64},
            {"run_path": "relative-2", "manifest_sha256": "2" * 64},
        ),
        relative_binding_proofs=proofs,
        escape_freeze=handle,
    )
    return SimpleNamespace(
        escape_freeze=handle,
        recording_id="recording-1",
        provenance=provenance,
    )


def test_escape_freeze_values_preserve_persisted_classes_events_and_sweep() -> None:
    values = _escape_freeze_values(_projection())

    assert values["n_trials"] == 1
    assert values["n_events"] == 1
    assert values["n_sweep_rows"] == 3
    assert values["trial_response_class_code"].tolist() == [1]
    assert values["event_high_turn"].tolist() == [True]
    assert values["recording_trace_usable_event_count"] == 1
    assert values["parameters"]["freeze_window_s"] == 2.0


def test_escape_freeze_renderer_uses_persisted_outcomes_and_declares_trace_gap() -> (
    None
):
    output = build_exact_escape_freeze_output(_Marimo, go, _projection())

    assert len(output) == 8
    outcomes, events, sensitivity = output[1:4]
    assert outcomes.layout.meta["escape_freeze_display"]["recipe_id"] == (
        ESCAPE_FREEZE_DISPLAY_RECIPE
    )
    assert outcomes.layout.meta["escape_freeze_display"]["response_classes"] == (
        "persisted_no_viewer_reclassification"
    )
    assert outcomes.layout.meta["escape_freeze_display"]["event_trace_samples"] == (
        "not_persisted_no_viewer_reconstruction"
    )
    assert len(outcomes.data) == 7
    assert len(events.data) == 5
    assert len(sensitivity.data) == 1
    assert "not reconstructed" in output[0]["callout"]
    assert output[5]["records"][0]["response_class"] == "speed_escape"
    assert output[7]["records"][0]["trace_status"] == "valid"


def test_escape_freeze_values_reject_changed_persisted_class() -> None:
    prepared = prepare_escape_freeze_successor(_source())
    arrays = {
        name: np.array(value, copy=True) for name, value in prepared.arrays.items()
    }
    arrays["trial_response_class_code"][0] = 2

    with pytest.raises(ValueError, match="classification"):
        _escape_freeze_values(_projection(arrays=arrays))


def test_escape_freeze_renderer_preserves_a_valid_zero_event_recording() -> None:
    projection = _projection(
        source=replace(_source(), escape_speed_threshold_mm_s=100.0)
    )

    values = _escape_freeze_values(projection)
    output = build_exact_escape_freeze_output(_Marimo, go, projection)

    assert values["n_events"] == 0
    assert values["recording_freeze_trial_count"] == 1
    assert output[7]["records"] == []
    assert "0 persisted speed-defined escape events" in output[0]["callout"]


def test_escape_freeze_event_projection_is_bounded_and_endpoint_preserving() -> None:
    size = ESCAPE_FREEZE_MAX_EVENT_POINTS + 101

    observed = _display_indices(size, limit=ESCAPE_FREEZE_MAX_EVENT_POINTS)

    assert observed.size == ESCAPE_FREEZE_MAX_EVENT_POINTS
    assert observed[0] == 0
    assert observed[-1] == size - 1
    assert np.all(np.diff(observed) > 0)
