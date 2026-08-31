from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from fisheye.analysis.chaser_distance_runs import ChaserDistanceWindow
from fisheye.analysis_workflows.materializers.provider_epoch_behavior_summary import (
    ProviderEpochBehaviorSummaryError,
    _bind_protocol_semantic_row_identity,
    _bind_track_id,
    _make_per_epoch_fish,
    _safe_name,
    _source_bindings_sha256,
    _swim_bout_binding,
    _windows_from_protocol_semantic_binding,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    CHASER_WINDOW_ROLES,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _window() -> ChaserDistanceWindow:
    return ChaserDistanceWindow(
        window_id=0,
        label="pre_event",
        start_frame=0,
        end_frame=9,
        start_time_s=0.0,
        end_time_s=1.0,
        duration_s=1.0,
    )


def _semantic_selection_and_binding() -> tuple[SimpleNamespace, dict[str, object]]:
    source_labels = ("pre_event", "training_event", "post_event")
    intervals = tuple(
        SimpleNamespace(
            window_id=index,
            label=label,
            start_frame=index * 10,
            end_frame=(index + 1) * 10,
            source_interval_digest=str(index + 1) * 64,
        )
        for index, label in enumerate(source_labels)
    )
    selection = SimpleNamespace(
        selection_record={"selection_sha256": "8" * 64},
        run_path="analysis/stimulus_epoch_runs/semantic-epoch-v2",
        run_manifest_digest="1" * 64,
        run_manifest_payload_digest="2" * 64,
        source_epoch_logical_content_digest="3" * 64,
        source_epoch_lineage_hash="4" * 64,
        source_epoch_lineage_payload_digest="5" * 64,
        source_timeline_digest="6" * 64,
        selection_digest="8" * 64,
        fps=10.0,
        intervals=intervals,
    )
    epochs = [
        {
            "analysis_role": role,
            "window_id": index,
            "source_label": source_labels[index],
            "start_frame": index * 10,
            "end_frame_exclusive": (index + 1) * 10,
            "source_interval_sha256": str(index + 1) * 64,
        }
        for index, role in enumerate(CHASER_WINDOW_ROLES)
    ]
    roles = [
        {
            "analysis_role": role,
            "source_window_id": index,
            "source_interval_sha256": str(index + 1) * 64,
            "selected_start_frame": index * 10,
            "selected_end_frame_exclusive": (index + 1) * 10,
            "protocol_semantic_hash": f"sha256:{'b' * 64}",
            "protocol_semantic_step_index": 1,
            "protocol_semantic_step_ref": (
                "protocol_semantic_snapshot@recipe.steps[1]"
            ),
        }
        for index, role in enumerate(CHASER_WINDOW_ROLES)
    ]
    binding: dict[str, object] = {
        "protocol_semantic_hash": f"sha256:{'b' * 64}",
        "roles": list(CHASER_WINDOW_ROLES),
        "source_epoch_selection": {
            "source_epoch_run_path": selection.run_path,
            "source_epoch_run_manifest_sha256": selection.run_manifest_digest,
            "source_epoch_run_manifest_payload_sha256": (
                selection.run_manifest_payload_digest
            ),
            "source_epoch_logical_content_sha256": (
                selection.source_epoch_logical_content_digest
            ),
            "source_epoch_lineage_hash": selection.source_epoch_lineage_hash,
            "source_epoch_lineage_payload_sha256": (
                selection.source_epoch_lineage_payload_digest
            ),
            "source_timeline_digest": selection.source_timeline_digest,
            "selection_sha256": selection.selection_digest,
        },
        "selector_eligible": False,
        "production_authority": False,
        "position_suite_epochs": epochs,
        "position_suite_epochs_sha256": canonical_json_sha256(epochs),
        "semantic_role_bindings": roles,
        "semantic_role_bindings_sha256": canonical_json_sha256(roles),
    }
    return selection, binding


def test_protocol_semantic_windows_use_roles_and_selected_half_open_bounds() -> None:
    selection, binding = _semantic_selection_and_binding()

    windows = _windows_from_protocol_semantic_binding(selection, binding)

    assert tuple(window.label for window in windows) == CHASER_WINDOW_ROLES
    assert [(window.start_frame, window.end_frame) for window in windows] == [
        (0, 9),
        (10, 19),
        (20, 29),
    ]
    assert [window.duration_s for window in windows] == [1.0, 1.0, 1.0]


def test_protocol_semantic_windows_reject_rehashed_source_mismatch() -> None:
    selection, binding = _semantic_selection_and_binding()
    epochs = list(binding["position_suite_epochs"])
    epochs[1] = {**epochs[1], "source_interval_sha256": "f" * 64}
    binding["position_suite_epochs"] = epochs
    binding["position_suite_epochs_sha256"] = canonical_json_sha256(epochs)

    with pytest.raises(ProviderEpochBehaviorSummaryError, match="exact source"):
        _windows_from_protocol_semantic_binding(selection, binding)


def test_protocol_semantic_windows_reject_full_selection_record_as_identity() -> None:
    selection, binding = _semantic_selection_and_binding()
    binding["source_epoch_selection"] = selection.selection_record

    with pytest.raises(ProviderEpochBehaviorSummaryError, match="exact sources"):
        _windows_from_protocol_semantic_binding(selection, binding)


def test_protocol_semantic_identity_is_repeated_on_each_summary_row() -> None:
    _selection, binding = _semantic_selection_and_binding()
    rows = np.zeros(
        3,
        dtype=[("window_id", np.int32), ("bout_count", np.int32)],
    )
    rows["window_id"] = [0, 1, 2]

    bound = _bind_protocol_semantic_row_identity(rows, binding=binding)

    assert [value.rstrip(b"\x00") for value in bound["analysis_role"]] == [
        role.encode("utf-8") for role in CHASER_WINDOW_ROLES
    ]
    assert bound["protocol_semantic_step_index"].tolist() == [1, 1, 1]
    assert [value.rstrip(b"\x00") for value in bound["source_interval_sha256"]] == [
        str(index + 1).encode("utf-8") * 64 for index in range(3)
    ]
    assert all(
        value.rstrip(b"\x00") == f"sha256:{'b' * 64}".encode("utf-8")
        for value in bound["protocol_semantic_hash"]
    )


def test_protocol_semantic_identity_rejects_noncanonical_hash() -> None:
    _selection, binding = _semantic_selection_and_binding()
    binding["protocol_semantic_hash"] = "sha256:not-a-digest"
    roles = list(binding["semantic_role_bindings"])
    roles[0] = {
        **roles[0],
        "protocol_semantic_hash": "sha256:not-a-digest",
    }
    binding["semantic_role_bindings"] = roles
    rows = np.zeros(1, dtype=[("window_id", np.int32)])

    with pytest.raises(ProviderEpochBehaviorSummaryError, match="incomplete"):
        _bind_protocol_semantic_row_identity(rows, binding=binding)


def test_provider_epoch_summary_uses_valid_tracked_time_and_motion_validity() -> None:
    sample_valid = np.asarray(
        [True, True, True, True, True, False, False, False, False, False],
        dtype=bool,
    )
    transition_valid = np.asarray(
        [False, True, True, True, True, False, False, False, False, False],
        dtype=bool,
    )
    speed = np.asarray(
        [999.0, 10.0, 10.0, 10.0, 10.0, 999.0, 999.0, 999.0, 999.0, 999.0]
    )
    path = np.asarray([999.0, 1.0, 1.0, 1.0, 1.0, 999.0, 999.0, 999.0, 999.0, 999.0])
    track = SimpleNamespace(
        frame_indices=np.arange(10, dtype=np.int64),
        linear_sample_valid=sample_valid,
        sample_valid=sample_valid,
        transition_valid=transition_valid,
        speed_mm_by_level={"filtered": speed},
        frame_path_distance_mm_by_level={"filtered": path},
        smoothed_heading_degrees=np.linspace(0.0, 9.0, 10),
        heading_degrees=np.linspace(0.0, 9.0, 10),
    )
    bouts = np.zeros(
        2,
        dtype=[
            ("bout_id", np.int64),
            ("peak_frame", np.int64),
            ("start_frame", np.int64),
            ("end_frame", np.int64),
            ("duration_s", np.float64),
            ("path_length_mm", np.float64),
        ],
    )
    bouts[0] = (1, 1, 0, 2, 0.04, 0.2)
    bouts[1] = (2, 4, 3, 4, 0.06, 0.3)
    intervals = np.zeros(
        1,
        dtype=[
            ("interval_s", np.float64),
            ("prev_end_frame", np.int64),
            ("next_start_frame", np.int64),
            ("valid", bool),
        ],
    )
    intervals[0] = (0.1, 2, 3, True)
    tables = SimpleNamespace(bouts=bouts, inter_bout_intervals=intervals)

    row = _make_per_epoch_fish(
        windows=(_window(),),
        track=track,
        track_id=7,
        speed_level="filtered",
        swim_tables=tables,
        fps=10.0,
    )[0]

    assert int(row["track_id"]) == 7
    assert int(row["total_span_frames"]) == 10
    assert int(row["provider_sample_count"]) == 10
    assert int(row["valid_tracked_frame_count"]) == 5
    assert float(row["valid_tracked_duration_s"]) == 0.5
    assert int(row["motion_valid_sample_count"]) == 4
    assert int(row["speed_sample_count"]) == 4
    assert float(row["mean_speed_mm_s"]) == 10.0
    assert float(row["total_path_mm"]) == 4.0
    assert int(row["bout_count"]) == 2
    assert float(row["bout_rate_per_min"]) == 240.0
    assert int(row["inter_bout_interval_count"]) == 1
    assert float(row["inter_bout_interval_rate_per_min"]) == 120.0
    assert row["rate_denominator"].rstrip(b"\x00") == b"valid_tracked_duration_s"


def test_provider_epoch_bout_facts_bind_the_selected_track() -> None:
    source = np.zeros(
        2,
        dtype=[("window_id", np.int32), ("bout_id", np.int64)],
    )
    source["window_id"] = [0, 1]
    source["bout_id"] = [11, 12]

    bound = _bind_track_id(source, track_id=4)

    assert bound.dtype.names == ("track_id", "window_id", "bout_id")
    assert bound["track_id"].tolist() == [4, 4]
    assert bound["bout_id"].tolist() == [11, 12]


def test_source_binding_digest_normalizes_immutable_nested_mappings() -> None:
    frozen = MappingProxyType(
        {"epoch": MappingProxyType({"run": "epochs_1", "digest": "a" * 64})}
    )
    plain = {"epoch": {"run": "epochs_1", "digest": "a" * 64}}

    assert _source_bindings_sha256(frozen) == _source_bindings_sha256(plain)


def test_swim_bout_binding_requires_the_exact_provider_manifest_and_row_slice() -> None:
    frames = np.arange(10, dtype=np.int64)
    provider = SimpleNamespace(
        run_name="motion_1",
        provider_manifest_sha256="a" * 64,
        verification_digest="b" * 64,
        source_acquisition_frame_index=frames,
    )
    authority = {
        "motion_manifest_sha256": "a" * 64,
        "provider_verification_digest": "b" * 64,
        "track_id": 0,
        "track_row_start": 0,
        "track_row_stop": 10,
    }
    from fisheye.analysis.swim_bout_frame_axis import canonical_frame_axis_sha256

    tables = SimpleNamespace(
        run_name="bouts_1",
        run_path="analysis/swim_bout_runs/bouts_1",
        run_attrs={
            "source_track_kinematics_scope": "provider",
            "source_track_kinematics_run": "motion_1",
            "track_id": 0,
            "source_track_motion_manifest_sha256": "a" * 64,
            "source_track_motion_authority": authority,
            "frame_axis_contract": {
                "content_sha256": canonical_frame_axis_sha256(frames)
            },
            "lineage_hash": "c" * 64,
        },
        candidate=SimpleNamespace(candidate_id=0),
        signal=SimpleNamespace(signal_id=4, speed_level="speed_exponential"),
    )

    binding, lineage, frame_digest = _swim_bout_binding(
        tables,
        provider=provider,
        rows=slice(0, 10),
        track_id=0,
    )

    assert lineage == "c" * 64
    assert binding["source_track_motion_manifest_sha256"] == "a" * 64
    assert binding["track_row_stop"] == 10
    assert binding["frame_axis_sha256"] == frame_digest

    tables.run_attrs["source_track_motion_manifest_sha256"] = "d" * 64
    with pytest.raises(ProviderEpochBehaviorSummaryError, match="manifest"):
        _swim_bout_binding(
            tables,
            provider=provider,
            rows=slice(0, 10),
            track_id=0,
        )


@pytest.mark.parametrize("value", ["", "latest", "a/b", "../run", " run"])
def test_provider_epoch_summary_rejects_selector_or_path_names(value: str) -> None:
    with pytest.raises(ProviderEpochBehaviorSummaryError):
        _safe_name(value, label="fixture")
