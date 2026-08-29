from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest

from apps.marimo.components.chaser_exact.bout_response import (
    BOUT_RESPONSE_DISPLAY_RECIPE,
    BOUT_RESPONSE_MAX_POINTS_PER_SERIES,
    _bout_response_values,
    _display_indices,
    build_exact_bout_response_output,
)
from apps.marimo.components.chaser_exact.provenance import (
    build_projection_provenance,
)
from fisheye.analysis_workflows.generalized_bout_response_successor import (
    prepare_generalized_bout_response_successor,
)
from tests.unit.fisheye.test_generalized_bout_response_successor import _source


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


def _projection(
    *,
    body: bool = True,
    arrays: dict[str, np.ndarray] | None = None,
):
    prepared = prepare_generalized_bout_response_successor(_source(body=body))
    payload = prepared.arrays if arrays is None else MappingProxyType(arrays)
    handle = SimpleNamespace(
        successor_kind="generalized_chaser_bout_response",
        deep_audited=True,
        scientific_manifest=prepared.manifest,
        run_path=("analysis/generalized_chaser_bout_response_runs/bout-response-v1"),
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
        generalized_bout_response=handle,
    )
    return SimpleNamespace(
        generalized_bout_response=handle,
        recording_id="recording-1",
        provenance=provenance,
    )


def test_bout_response_values_preserve_exact_rows_and_persisted_summary() -> None:
    values = _bout_response_values(_projection())

    assert values["n_bouts"] == 2
    assert values["n_rows"] == 2
    assert values["n_summary"] == 6
    assert values["source_signal_level"] == "speed_filtered"
    assert values["controller_trial_row_id"].tolist() == [0, -1]
    assert values["attachment_reason_code"].tolist() == [0, 3]
    assert values["summary_bout_count"].tolist() == [0, 0, 1, 1, 0, 0]
    assert values["body_extension_present"] is True


def test_bout_response_renderer_uses_persisted_bins_rows_and_body_frame() -> None:
    output = build_exact_bout_response_output(_Marimo, go, _projection())

    assert len(output) == 6
    rate, kinematics, response = output[1:4]
    assert rate.layout.meta["bout_response_display"]["recipe_id"] == (
        BOUT_RESPONSE_DISPLAY_RECIPE
    )
    assert (
        rate.layout.meta["bout_response_display"]["scientific_recomputation"] is False
    )
    assert rate.layout.meta["bout_response_display"]["bout_resegmentation"] == (
        "prohibited"
    )
    assert len(rate.data) == 3
    assert len(kinematics.data) == 12
    distance_points = sum(len(trace.x) for trace in response.data[:3])
    directed_points = sum(len(trace.x) for trace in response.data[3:])
    assert distance_points == 2
    assert directed_points == 2
    table = output[5]
    assert len(table["records"]) == 6
    assert table["records"][2]["semantic_role"] == "chaser_training"
    assert table["records"][2]["bout_count"] == 1


def test_bout_response_renderer_keeps_absent_body_extension_explicit() -> None:
    output = build_exact_bout_response_output(
        _Marimo,
        go,
        _projection(body=False),
    )

    response = output[3]
    assert (
        response.layout.meta["bout_response_display"]["body_extension_present"] is False
    )
    assert "no motion-heading fallback" in output[0]["callout"]


def test_bout_response_values_reject_changed_attachment_evidence() -> None:
    prepared = prepare_generalized_bout_response_successor(_source())
    arrays = {
        name: np.array(value, copy=True) for name, value in prepared.arrays.items()
    }
    arrays["attachment_reason_code"][1] = 0

    with pytest.raises(ValueError, match="attachment evidence"):
        _bout_response_values(_projection(arrays=arrays))


def test_bout_response_display_projection_is_bounded_and_endpoint_preserving() -> None:
    source = np.arange(BOUT_RESPONSE_MAX_POINTS_PER_SERIES + 101, dtype=np.int64)

    observed = _display_indices(source)

    assert observed.size == BOUT_RESPONSE_MAX_POINTS_PER_SERIES
    assert observed[0] == source[0]
    assert observed[-1] == source[-1]
    assert np.all(np.diff(observed) > 0)
