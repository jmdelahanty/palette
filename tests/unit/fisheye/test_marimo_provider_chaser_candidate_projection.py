from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest

import apps.marimo.components.provider_chaser_candidate as component


class FakeArray:
    def __init__(self, values: object):
        self.values = np.asarray(values)

    def __getitem__(self, key: object) -> np.ndarray:
        return self.values[key]


class FakeGroup:
    def __init__(
        self,
        children: dict[str, object] | None = None,
        *,
        attrs: dict[str, object] | None = None,
    ):
        self.children = dict(children or {})
        self.attrs = dict(attrs or {})

    def __getitem__(self, key: str) -> object:
        if key in self.children:
            return self.children[key]
        current: object = self
        for part in key.split("/"):
            if not isinstance(current, FakeGroup):
                raise KeyError(key)
            current = current.children[part]
        return current

    def group_keys(self) -> list[str]:
        return [
            name
            for name, value in self.children.items()
            if isinstance(value, FakeGroup)
        ]


class FakeMarimo:
    @staticmethod
    def md(value: str) -> tuple[str, str]:
        return ("md", value)

    @staticmethod
    def vstack(values: list[object]) -> tuple[str, list[object]]:
        return ("vstack", values)


def _candidate_group() -> FakeGroup:
    n = 4
    chasers = 2
    source_stimulus_rows = np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64)
    fish = np.zeros((n, 2), dtype=np.float32)
    angle = np.deg2rad(10.0)
    chaser_xy = np.asarray(
        [
            [[-1.0, -np.sin(angle)], [1.0, 0.0]],
            [[-1.0, np.sin(angle)], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    arrays = {
        "samples/stimulus_frame_num": np.arange(n, dtype=np.int64),
        "samples/timestamp_ns": np.arange(n, dtype=np.int64) * 1_000_000,
        "samples/source_acquisition_frame_index": np.asarray(
            [10, 10, 11, 12], dtype=np.int64
        ),
        "samples/stimulus_epoch_window_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "samples/source_stimulus_run_row_index": source_stimulus_rows,
        "positions/source_position_run_row_index": np.asarray(
            [0, 0, 1, 2], dtype=np.int64
        ),
        "positions/fish_position_arena_xy": fish,
        "positions/fish_valid": np.ones(n, dtype=bool),
        "positions/chaser_position_arena_xy": chaser_xy,
        "positions/chaser_valid": np.ones((n, chasers), dtype=bool),
        "chasers/chaser_index": np.asarray([0, 1], dtype=np.int64),
        "distances/distance_mm": np.asarray(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0], [4.0, 7.0]], dtype=np.float32
        ),
    }
    return FakeGroup({path: FakeArray(values) for path, values in arrays.items()})


def _motion() -> SimpleNamespace:
    return SimpleNamespace(
        source_position_row_index=np.asarray([0, 1, 2], dtype=np.int64),
        source_acquisition_frame_index=np.asarray([10, 11, 12], dtype=np.int64),
        track_sample_key=np.asarray([[0, 10], [0, 11], [0, 12]], dtype=np.int64),
        arrays={"smoothed_heading_degrees": np.zeros(3, dtype=np.float32)},
        angular_sample_valid=np.ones(3, dtype=bool),
        body_frame_source_valid=np.ones(3, dtype=bool),
        timing_is_authoritative=False,
        verification_digest="motion-verification",
    )


def _sources() -> component._ResolvedProviderSources:
    bouts = np.asarray(
        [(7, 10, 12, 0.2, 1.5, 8.0, 10)],
        dtype=[
            ("bout_id", "i4"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("duration_s", "f8"),
            ("path_length_mm", "f8"),
            ("peak_physical_speed_mm_s", "f8"),
            ("peak_frame", "i8"),
        ],
    )
    tables = SimpleNamespace(
        bouts=bouts,
        candidate=SimpleNamespace(candidate_id=0),
        signal=SimpleNamespace(signal_id=4),
    )
    return component._ResolvedProviderSources(
        motion=_motion(),
        motion_run_path="analysis/track_kinematics_runs/provider/motion_v1",
        motion_manifest_sha256="b" * 64,
        swim_bout_tables=tables,
        swim_bout_run_path="analysis/swim_bout_runs/bouts_v1",
        swim_bout_array_manifest_sha256="c" * 64,
        swim_bout_frame_axis_contract_sha256="d" * 64,
    )


def _option() -> SimpleNamespace:
    return SimpleNamespace(
        renderer=component.PROVIDER_CHASER_CANDIDATE_RENDERER,
        run_path="analysis/provider_chaser_distance_candidate_runs/candidate_v1",
        attrs={component.MANIFEST_DIGEST_ATTR: "a" * 64},
        spec={"candidate_status": "unpromoted_selector_ineligible"},
    )


def test_projection_preserves_duplicate_stimulus_samples_and_exact_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate_group()
    root = FakeGroup()
    sources = _sources()
    attrs = {
        component.MANIFEST_DIGEST_ATTR: "a" * 64,
        "source_position_run_path": "analysis/subject_position_runs/observation/position_v1",
        "source_position_manifest_sha256": "e" * 64,
        "source_stimulus_run_path": "analysis/stimulus_runs/stimulus_v1",
        "source_position_estimator_id": "keypoint_triad.v1",
    }
    monkeypatch.setattr(
        component,
        "_require_candidate_option",
        lambda _path, _option: (
            Path("/tmp/fake.zarr"),
            "analysis/provider_chaser_distance_candidate_runs/candidate_v1",
            candidate,
            attrs,
        ),
    )
    monkeypatch.setattr(component, "open_zarr_root", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        component, "_load_source_bundle", lambda *_args, **_kwargs: sources
    )
    monkeypatch.setattr(
        component, "_epoch_labels", lambda _run: {0: "pre", 1: "chaser"}
    )
    monkeypatch.setattr(
        component,
        "_resolve_semantic_chaser_labels",
        lambda *_args, **_kwargs: ("aggressive", "inert"),
    )

    projection = component.load_provider_chaser_candidate_projection(
        Path("/tmp/fake.zarr"), _option()
    )

    assert projection.source_motion_row_index.tolist() == [0, 0, 1, 2]
    assert projection.source_acquisition_frame_index.tolist() == [10, 10, 11, 12]
    assert projection.chaser_labels == ("aggressive", "inert")
    assert projection.gaze_readiness == component.GazeReadiness
    assert projection.provenance["provider_motion"]["timing_is_authoritative"] is False
    assert projection.timestamp_ns.tolist() == [0, 1_000_000, 2_000_000, 3_000_000]
    assert projection.bout_rows[0]["mapped_stimulus_sample_count"] == 2
    assert projection.bout_rows[0]["onset_stimulus_sample_indices"] == (0, 1)
    assert projection.bout_rows[0]["support_count"] == 2
    assert (
        abs(abs(float(projection.bout_rows[0]["onset_bearing_mean_deg"])) - 180.0)
        < 1e-4
    )
    assert float(projection.bout_rows[0]["onset_bearing_resultant"]) > 0.9
    assert not projection.bearing_deg.flags.writeable


def test_invalid_provider_lineage_fails_closed() -> None:
    motion = _motion()
    with pytest.raises(ValueError, match="acquisition-frame lineage"):
        component._motion_row_indices(
            motion,
            candidate_position_rows=np.asarray([0], dtype=np.int64),
            candidate_acquisition_frames=np.asarray([999], dtype=np.int64),
        )


def test_semantic_roles_are_resolved_from_sealed_source_enum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = np.asarray(
        [(0, 1), (1, 3), (0, 1), (1, 3)],
        dtype=[("chaser_index", "i1"), ("chaser_behavior_class_id", "i1")],
    )
    enums = np.asarray(
        [(1, b"aggressive"), (3, b"inert")],
        dtype=[("id", "i1"), ("name", "S16")],
    )
    chaser_group = FakeGroup(
        attrs={
            "schema_id": "citrus.tracking.chaser_states",
            "schema_version": 5,
            "coordinate_descriptor_status": "canonical",
        }
    )
    stimulus = FakeGroup(
        {
            "tracking_data": FakeGroup({"chaser_states": chaser_group}),
            "enums": FakeGroup(),
        }
    )
    root = FakeGroup(
        {"analysis": FakeGroup({"stimulus_runs": FakeGroup({"stimulus_v1": stimulus})})}
    )
    monkeypatch.setattr(
        component,
        "load_structured_dataset",
        lambda _group, name: (
            (records, None) if name == "chaser_states" else (enums, None)
        ),
    )

    labels = component._resolve_semantic_chaser_labels(
        root,
        source_stimulus_run_path="analysis/stimulus_runs/stimulus_v1",
        source_row_indices=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        chaser_indices=np.asarray([0, 1], dtype=np.int64),
    )

    assert labels == ("aggressive", "inert")


def test_available_ids_require_exact_source_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    option = _option()
    archive = Path("/tmp/fake.zarr")
    monkeypatch.setattr(
        component,
        "_require_candidate_option",
        lambda _path, _option: (archive, "candidate", object(), {}),
    )
    monkeypatch.setattr(component, "open_zarr_root", lambda *_args, **_kwargs: object())
    no_bouts = SimpleNamespace(swim_bout_tables=None)
    monkeypatch.setattr(
        component, "_load_source_bundle", lambda *_args, **_kwargs: no_bouts
    )
    assert component.available_provider_chaser_candidate_analysis_ids(
        archive, option
    ) == (
        "static_artifacts",
        "provenance",
        "egocentric_bearing",
    )


def test_candidate_outputs_are_labeled_and_render_from_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate_group()
    sources = _sources()
    attrs = {
        component.MANIFEST_DIGEST_ATTR: "a" * 64,
        "source_position_run_path": "analysis/subject_position_runs/observation/position_v1",
        "source_position_manifest_sha256": "e" * 64,
        "source_stimulus_run_path": "analysis/stimulus_runs/stimulus_v1",
        "source_position_estimator_id": "keypoint_triad.v1",
    }
    monkeypatch.setattr(
        component, "open_zarr_root", lambda *_args, **_kwargs: FakeGroup()
    )
    monkeypatch.setattr(
        component, "_load_source_bundle", lambda *_args, **_kwargs: sources
    )
    monkeypatch.setattr(
        component, "_epoch_labels", lambda _run: {0: "pre", 1: "chaser"}
    )
    monkeypatch.setattr(
        component,
        "_resolve_semantic_chaser_labels",
        lambda *_args, **_kwargs: ("aggressive", "inert"),
    )
    projection = component._load_projection_from_candidate(
        Path("/tmp/fake.zarr"),
        "analysis/provider_chaser_distance_candidate_runs/candidate_v1",
        candidate,
        attrs,
        sources=sources,
    )

    bearing = component.build_provider_chaser_candidate_bearing_output(
        FakeMarimo, go, projection
    )
    bouts = component.build_provider_chaser_candidate_bout_response_output(
        FakeMarimo, go, projection
    )

    assert bearing[0] == "vstack"
    assert bouts[0] == "vstack"
    assert "unpromoted" in bearing[1][0][1].casefold()
    assert "do not encode the recorded stimulus color" in bouts[1][0][1]
    assert any(getattr(item, "data", ()) for item in bearing[1][1:])
    assert any(getattr(item, "data", ()) for item in bouts[1][1:])
