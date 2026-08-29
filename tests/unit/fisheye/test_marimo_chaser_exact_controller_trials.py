from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest

from apps.marimo.components.chaser_exact.controller_trials import (
    CONTROLLER_TRIAL_DISPLAY_RECIPE,
    _controller_trial_values,
    _timed_gap_indices,
    build_exact_controller_trials_output,
)
from apps.marimo.components.chaser_exact.projection import RelativeFrameProjection
from apps.marimo.components.chaser_exact.provenance import (
    build_projection_provenance,
    plain,
)
from fisheye.analysis_workflows.controller_trial_successor import (
    prepare_controller_trial_successor,
)
from tests.unit.fisheye.test_controller_trial_successor import _source


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


def _relative(*, offset: float) -> RelativeFrameProjection:
    source = _source()
    n_rows = source.n_frames * source.n_chasers
    distance = np.arange(n_rows, dtype=np.float64) + offset
    arrays = {
        "acquisition_frame_id": source.acquisition_frame_id,
        "timestamp_ns": source.timestamp_ns,
        "timestamp_valid": source.timestamp_valid,
        "chaser_identity_code": source.chaser_identity_code,
        "chaser_occurrence_member": source.chaser_occurrence_member,
        "relative_distance_physical": distance,
        "relative_physical_valid": np.ones(n_rows, dtype=bool),
    }
    return RelativeFrameProjection(
        run_path=f"analysis/chaser_relative_frame_runs/relative-{offset}",
        run_name=f"relative-{offset}",
        recording_id=source.recording_id,
        manifest_sha256="a" * 64,
        n_frames=source.n_frames,
        n_chasers=source.n_chasers,
        source_authorities=MappingProxyType({}),
        arrays=MappingProxyType(arrays),
    )


def _projection(
    *,
    arrays: dict[str, np.ndarray] | None = None,
):
    prepared = prepare_controller_trial_successor(_source())
    payload = prepared.arrays if arrays is None else MappingProxyType(arrays)
    handle = SimpleNamespace(
        successor_kind="controller_chase_trials",
        deep_audited=True,
        scientific_manifest=prepared.manifest,
        run_path="analysis/controller_chase_trial_runs/controller-v1",
        manifest_sha256="b" * 64,
        scientific_payload_sha256=prepared.payload_digest,
        array=lambda name: payload[name],
    )
    spatial = SimpleNamespace(
        recording_id="recording-1",
        run_path="analysis/chaser_spatial_occupancy_runs/spatial-v1",
        manifest_sha256="c" * 64,
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
        controller_trials=handle,
    )
    return SimpleNamespace(
        controller_trials=handle,
        relatives=(_relative(offset=0.0), _relative(offset=0.5)),
        provider_ids=("keypoint.v1", "detection.v1"),
        recording_id="recording-1",
        provenance=provenance,
    )


def test_controller_trial_values_keep_gap_distinct_from_membership() -> None:
    values = _controller_trial_values(_projection())

    assert values["n_trials"] == 3
    assert np.flatnonzero(values["trial_gap_member"]).tolist() == [4]
    assert not bool(values["logged_active_trial_member"][4])
    assert values["trial_gap_reason_code_by_source_row"][4] == 4
    assert values["gap_registry"]["4"] == "explicit_controller_inactive"


def test_controller_trial_renderer_uses_sealed_membership_and_gap_evidence() -> None:
    output = build_exact_controller_trials_output(_Marimo, go, _projection())

    assert len(output) == 5
    full_figure = output[1]
    trial_figure = output[2]
    table = output[4]
    assert full_figure.layout.meta["controller_trial_display"]["recipe_id"] == (
        CONTROLLER_TRIAL_DISPLAY_RECIPE
    )
    assert (
        full_figure.layout.meta["controller_trial_display"][
            "legacy_trial_reconstruction"
        ]
        == "prohibited"
    )
    gap_traces = [
        trace for trace in trial_figure.data if trace.name == "retained nonmember gap"
    ]
    assert len(gap_traces) == 1
    assert gap_traces[0].customdata.tolist() == ["explicit_controller_inactive"]
    assert len(table["records"]) == 3
    assert table["records"][0]["active_member_count"] == 3
    assert table["records"][0]["gap_frame_count"] == 1
    assert table["records"][0]["untimed_gap_count"] == 0
    controller_binding = full_figure.layout.meta["controller_trial_binding"]
    assert controller_binding["source_relative_frame"] == plain(
        _projection().controller_trials.scientific_manifest["source_relative_frame"]
    )
    assert controller_binding["semantic_selection"] == plain(
        _projection().controller_trials.scientific_manifest["semantic_selection"]
    )


def test_timed_gap_indices_do_not_position_invalid_timestamp_rows() -> None:
    timestamp_valid = np.ones((10, 2), dtype=bool)
    timestamp_valid[6, :] = False

    observed = _timed_gap_indices(
        np.asarray([1, 3], dtype=np.int64),
        timestamp_valid=timestamp_valid,
        frame_indices=np.asarray([5, 6, 7, 8], dtype=np.int64),
        chaser_column=0,
    )

    assert observed.tolist() == [3]


@pytest.mark.parametrize("tamper", ["member_gap_overlap", "fallback_used"])
def test_controller_trial_values_reject_inconsistent_evidence(tamper: str) -> None:
    prepared = prepare_controller_trial_successor(_source())
    arrays = {
        name: np.array(value, copy=True) for name, value in prepared.arrays.items()
    }
    if tamper == "member_gap_overlap":
        arrays["trial_row_id_by_source_row"][4] = 0
        arrays["logged_active_trial_member"][4] = True
    else:
        arrays["fallback_used"][0] = True

    with pytest.raises(ValueError, match="membership or fail-closed evidence"):
        _controller_trial_values(_projection(arrays=arrays))


def test_controller_trial_values_reject_inconsistent_acquisition_boundary() -> None:
    prepared = prepare_controller_trial_successor(_source())
    arrays = {
        name: np.array(value, copy=True) for name, value in prepared.arrays.items()
    }
    arrays["start_acquisition_frame_id"][0] += 1

    with pytest.raises(ValueError, match="inconsistent sealed evidence"):
        _controller_trial_values(_projection(arrays=arrays))
