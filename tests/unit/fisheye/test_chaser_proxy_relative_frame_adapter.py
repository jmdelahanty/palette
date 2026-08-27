from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from fisheye.analysis_workflows import chaser_proxy_relative_frame_adapter as adapter
from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    select_chaser_input_provenance_proxy,
)
@dataclass(frozen=True)
class _NativeDimensions:
    total_frames: int = 4
    n_samples: int = 3
    n_chasers: int = 2


class _Native:
    recording_id = "recording-1"
    run_path = "analysis/provider_chaser_distance_candidate_runs/native-v1"
    manifest_sha256 = "a" * 64
    verification_digest = "b" * 64
    dimensions = _NativeDimensions()
    stimulus_frame_num = np.asarray([10, 11, 12], dtype=np.int64)
    timestamp_ns = np.asarray([100, 110, 120], dtype=np.int64)
    source_acquisition_frame_index = np.asarray([0, 0, 2], dtype=np.int64)
    source_stimulus_run_row_index = np.asarray(
        [[100, 101], [102, 103], [104, 105]], dtype=np.int64
    )
    source_stimulus_source_row_index = np.asarray(
        [[200, 201], [202, 203], [204, 205]], dtype=np.int64
    )
    chaser_index = np.asarray([0, 1], dtype=np.int16)
    chaser_position_arena_xy = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float64,
    )
    chaser_valid = np.ones((3, 2), dtype=bool)
    fish_position_source_camera_xy = np.asarray(
        [[20.0, 30.0], [20.0, 30.0], [40.0, 50.0]], dtype=np.float64
    )
    fish_valid = np.ones(3, dtype=bool)
    fish_source_position_run_row_index = np.asarray([50, 50, 52], dtype=np.int64)
    source_authority = {
        "schema_id": "palette.provider_chaser_distance_candidate_source_authority",
        "schema_version": 1,
        "recording_id": recording_id,
        "position": {
            "run_path": "analysis/subject_position_runs/position-v1",
            "manifest_sha256": "c" * 64,
            "decoded_content_sha256": "d" * 64,
            "estimator_id": "detection_bbox_centroid_v1",
            "estimator_sha256": "e" * 64,
            "policy_sha256": "f" * 64,
            "source_sha256": "0" * 64,
            "anatomy_sha256": "1" * 64,
            "coordinate_sha256": "2" * 64,
            "source_camera_frame": {
                "record_ref": "analysis/acquisition/source_camera_frame",
                "record_sha256": "3" * 64,
            },
        },
        "stimulus": {
            "run_path": "analysis/stimulus_runs/stimulus-v1",
            "row_identity": {"record_ref": "rows", "record_sha256": "4" * 64},
            "temporal_authority": {"record_ref": "time", "record_sha256": "5" * 64},
            "surface_manifest": {"record_ref": "surface", "record_sha256": "6" * 64},
            "output_manifest": {"record_ref": "output", "record_sha256": "7" * 64},
            "transform_manifest": {"record_ref": "transform", "record_sha256": "8" * 64},
            "source_camera_frame": {
                "record_ref": "analysis/acquisition/source_camera_frame",
                "record_sha256": "3" * 64,
            },
        },
        "stimulus_epoch": {
            "run_path": "analysis/stimulus_epoch_runs/epochs-v1",
            "schema_id": "palette.stimulus_epoch",
            "schema_version": 2,
            "manifest_sha256": "9" * 64,
            "metadata_equivalence": None,
        },
        "acquisition_frame_authority": {
            "record_ref": (
                "/analysis/acquisition_camera_frames/camera-1"
                "@acquisition_camera_frame"
            ),
            "record_sha256": "a" * 64,
        },
        "total_frames": 4,
        "stimulus_sample_count": 3,
        "fps": 100.0,
        "fps_authority": {"recording_id": recording_id},
        "pixels_per_mm_projector": 2.0,
        "temporal_join_policy": "preserve_native_v1",
        "numeric_transform": "typed_v1",
    }

    @property
    def source_stimulus_run_path(self) -> str:
        return self.source_authority["stimulus"]["run_path"]

    def assert_verified(self) -> None:
        return None


class _Proxy:
    def __init__(self, native: _Native) -> None:
        selected = select_chaser_input_provenance_proxy(native)
        self.recording_id = selected.recording_id
        self.run_path = "analysis/chaser_input_provenance_proxy_runs/proxy-v1"
        self.manifest_sha256 = "d" * 64
        self.acquisition_projection_record = selected.acquisition_projection_record
        self.acquisition_projection_record_sha256 = (
            selected.acquisition_projection_record_sha256
        )
        self.dimensions = SimpleNamespace(
            n_frames=selected.unique_acquisition_frame_count,
            n_candidates=selected.stimulus_frame_num.size
            if hasattr(selected, "stimulus_frame_num")
            else native.dimensions.n_samples,
            n_chasers=native.dimensions.n_chasers,
        )
        self.arrays = {
            name: np.asarray(getattr(selected, name))
            for name in (
                "acquisition_frame_index",
                "candidate_offsets",
                "candidate_sample_count",
                "candidate_native_sample_row_index",
                "selected",
                "selected_native_sample_row_index",
                "selected_stimulus_frame_num",
                "selected_source_stimulus_run_row_index",
                "selected_source_stimulus_source_row_index",
                "selected_timestamp_ns_session",
                "selected_chaser_index",
                "selected_chaser_position_xy",
                "selected_chaser_valid",
            )
        }

    @property
    def acquisition_frame_index(self) -> np.ndarray:
        return self.arrays["acquisition_frame_index"]

    @property
    def candidate_offsets(self) -> np.ndarray:
        return self.arrays["candidate_offsets"]

    @property
    def candidate_sample_count(self) -> np.ndarray:
        return self.arrays["candidate_sample_count"]

    @property
    def selected(self) -> np.ndarray:
        return self.arrays["selected"]

    @property
    def selected_native_sample_row_index(self) -> np.ndarray:
        return self.arrays["selected_native_sample_row_index"]

    @property
    def selected_source_stimulus_run_row_index(self) -> np.ndarray:
        return self.arrays["selected_source_stimulus_run_row_index"]

    @property
    def selected_chaser_position_xy(self) -> np.ndarray:
        return self.arrays["selected_chaser_position_xy"]

    @property
    def selected_chaser_valid(self) -> np.ndarray:
        return self.arrays["selected_chaser_valid"]

    @property
    def publication_binding_record(self) -> dict[str, object]:
        projection = self.acquisition_projection_record
        return {
            "schema_id": "palette.chaser_input_provenance_proxy_publication_binding",
            "schema_version": 1,
            "recording_id": self.recording_id,
            "run_path": self.run_path,
            "manifest_sha256": self.manifest_sha256,
            "acquisition_projection_record_sha256": self.acquisition_projection_record_sha256,
            "policy_id": projection["policy_id"],
            "temporal_alignment_class": projection["temporal_alignment_class"],
            "source_run_path": projection["source_run_path"],
            "source_manifest_sha256": projection["source_manifest_sha256"],
            "source_verification_digest": projection["source_verification_digest"],
            "n_frames": self.dimensions.n_frames,
            "n_candidates": self.dimensions.n_candidates,
            "n_chasers": self.dimensions.n_chasers,
            "selector_eligible": False,
            "selection": "none",
        }

    def array(self, name: str) -> np.ndarray:
        return self.arrays[name]


class _Group:
    def __init__(self, attrs: dict[str, object]) -> None:
        self.attrs = attrs


class _Root:
    def __init__(self) -> None:
        self.stimulus = _Group(
            {
                "protocol_json": {
                    "protocol_name": "fixture",
                    "steps": [
                        {
                            "parameters": {
                                "chasers": [
                                    {"chaser_index": 0, "enable_chase": True},
                                    {"chaser_index": 1, "enable_chase": False},
                                ]
                            }
                        }
                    ],
                }
            }
        )
        chaser_index = np.zeros(106, dtype=np.uint8)
        chaser_index[[103, 105]] = 1
        stimulus_frame = np.zeros(106, dtype=np.uint64)
        stimulus_frame[[102, 103]] = 11
        stimulus_frame[[104, 105]] = 12
        source_rows = np.arange(106, dtype=np.int64)
        source_rows[[102, 103, 104, 105]] = [202, 203, 204, 205]
        trial_id = np.zeros(106, dtype=np.uint64)
        trial_id[[102, 104]] = [1, 2]
        active = np.zeros(106, dtype=np.uint8)
        active[[102, 104]] = 1
        timestamp = np.zeros(106, dtype=np.int64)
        timestamp[[102, 103, 104, 105]] = [110, 110, 120, 120]
        self.chaser_states = {
            "chaser_index": chaser_index,
            "stimulus_frame_num": stimulus_frame,
            "source_row_indices": source_rows,
            "chase_trial_id": trial_id,
            "chase_sequence_active": active,
            "timestamp_ns_session": timestamp,
        }

    def __getitem__(self, path: str) -> object:
        if path == "analysis/stimulus_runs/stimulus-v1":
            return self.stimulus
        if path == (
            "analysis/stimulus_runs/stimulus-v1/tracking_data/chaser_states"
        ):
            return self.chaser_states
        raise KeyError(path)


class _Coordinate:
    mm_per_pixel = 0.25
    physical = SimpleNamespace(camera_id="camera-1")
    record = MappingProxyType(
        {
            "source_stimulus_run_ref": "/analysis/stimulus_runs/stimulus-v1",
            "stimulus_frame_transform_manifest": {
                "record_ref": "analysis/stimulus_runs/stimulus-v1/transform",
                "record_sha256": "f" * 64,
            },
            "selected_calibration": {
                "record_ref": "analysis/calibration/selected",
                "record_sha256": "0" * 64,
            },
            "arena_geometry": {
                "record_ref": "analysis/stimulus_runs/stimulus-v1/arena",
                "record_sha256": "1" * 64,
            },
            "arena_frame": {
                "record_ref": "analysis/stimulus_runs/stimulus-v1/arena_frame",
                "record_sha256": "2" * 64,
            },
            "selected_canvas_frame": {
                "record_ref": "analysis/stimulus_runs/stimulus-v1/canvas",
                "record_sha256": "4" * 64,
            },
            "source_camera_frame": {
                "record_ref": "analysis/acquisition/source_camera_frame",
                "record_sha256": "3" * 64,
            },
            "arena_to_source_camera_transform_chain": [
                {"record_ref": "transform/1", "record_sha256": "5" * 64},
                {"record_ref": "transform/2", "record_sha256": "6" * 64},
            ],
            "physical_authority": {
                "physical_frame": {
                    "record_ref": "analysis/physical/source_camera_mm",
                    "record_sha256": "7" * 64,
                }
            },
        }
    )

    def arena_to_source_camera_px(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) + np.asarray([10.0, 20.0])


def test_adapter_applies_typed_arena_to_camera_chain_with_exact_session_time(
    monkeypatch,
) -> None:
    native = _Native()
    proxy = _Proxy(native)
    root = _Root()
    timing = SimpleNamespace(
        recording_id="recording-1",
        frame_count=4,
        camera_id="camera-1",
        clock_run_path="analysis/acquisition_frame_clock_runs/clock-v1",
        clock_record_sha256="a" * 64,
        source_video_metadata_sha256="b" * 64,
        sha256="8" * 64,
    )
    acquisition_frame = SimpleNamespace(
        record=SimpleNamespace(
            recording_id="recording-1",
            camera_id="camera-1",
            frame_count=4,
            source_total_frames=4,
            source_video_metadata_sha256="b" * 64,
        )
    )
    subject = SimpleNamespace(
        subject_ids=("fish-uuid",),
        subject_identity_kind="uuid",
        group_path="analysis/subject_metadata_runs/subject-v1",
        record_sha256="9" * 64,
    )
    monkeypatch.setattr(
        adapter,
        "load_chaser_input_provenance_proxy_source_handle",
        lambda *args, **kwargs: proxy,
    )
    monkeypatch.setattr(
        adapter,
        "load_provider_chaser_stimulus_source_handle",
        lambda *args, **kwargs: native,
    )
    monkeypatch.setattr(adapter, "open_zarr_root", lambda *args, **kwargs: root)
    monkeypatch.setattr(
        adapter,
        "load_provider_recording_timing_authority",
        lambda *args, **kwargs: timing,
    )
    monkeypatch.setattr(
        adapter,
        "_load_native_acquisition_frame",
        lambda *args, **kwargs: acquisition_frame,
    )
    monkeypatch.setattr(
        adapter,
        "load_stimulus_physical_coordinate_authority",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        adapter,
        "require_bound_stimulus_physical_coordinate_authority",
        lambda value: value,
    )
    monkeypatch.setattr(
        adapter,
        "load_stimulus_response_coordinate_authority",
        lambda *args, **kwargs: _Coordinate(),
    )
    monkeypatch.setattr(
        adapter,
        "resolve_subject_metadata",
        lambda *args, **kwargs: subject,
    )

    profile = Path(adapter.__file__).resolve().parents[1] / "analysis/profiles/chaser_behavior_full_v3.yaml"
    bound = adapter.prepare_proxy_relative_frame(
        "/tmp/fixture.zarr",
        proxy_run_name="proxy-v1",
        analysis_profile_path=profile,
    )

    base = bound.prepared.base_arrays
    # The later native row wins for acquisition frame zero.  Flattening is
    # frame-major and chaser-minor, so both translated points are adjacent.
    assert np.array_equal(
        base["chaser_position_xy_px"][:2],
        np.asarray([[15.0, 26.0], [17.0, 28.0]], dtype=np.float32),
    )
    assert np.array_equal(
        base["fish_position_xy_px"][:2],
        np.asarray([[20.0, 30.0], [20.0, 30.0]], dtype=np.float32),
    )
    assert np.all(base["timestamp_valid"])
    assert base["timestamp_ns"].tolist() == [110, 110, 120, 120]
    assert base["trial_id"].tolist() == [1, -1, 2, -1]
    assert base["trial_valid"].tolist() == [True, False, True, False]
    assert base["active_state_code"].tolist() == [1, 0, 1, 0]
    assert base["relative_physical_valid"].tolist() == [True, True, True, True]
    controller = bound.prepared.manifest["context"]["controller_state"]["record"]
    assert controller["policy_id"] == (
        "exact_logged_chase_trial_id_and_active_state_v1"
    )
    assert controller["fallback"] == "prohibited_fail_closed"
    assert controller["position_validity_policy"] == (
        "controller_active_is_orthogonal_position_evidence_v1"
    )
    assert bound.prepared.manifest["active_position_validity_policy"] == {
        "policy_id": "controller_active_is_orthogonal_position_evidence_v1",
        "active_state_present": True,
        "active_state_surface": "base/active_state_code",
        "position_validity_semantics": (
            "controller activity is preserved as evidence and does not invalidate "
            "otherwise finite selected occurring fish/chaser geometry"
        ),
    }
    assert (
        bound.prepared.manifest["timing_policy"]["timestamp_field"]
        == "timestamp_ns_session"
    )
    transform = bound.prepared.manifest["context"][
        "arena_to_source_camera_transform"
    ]["record"]
    assert transform["from_coordinate_space"] == "arena_relative_canvas_px"
    assert transform["to_coordinate_space"] == "source_camera_image_px"
    assert transform["no_reflection_or_heuristic_flip"] is True
    assert bound.prepared.manifest["selector_eligible"] is False


def test_exact_controller_projection_rejects_active_row_without_logged_id() -> None:
    native = _Native()
    proxy = _Proxy(native)
    root = _Root()
    root.chaser_states["chase_trial_id"][102] = 0

    with pytest.raises(
        adapter.ChaserProxyRelativeFrameAdapterError,
        match="legacy contiguous-interval trial reconstruction is prohibited",
    ):
        adapter._exact_logged_controller_state(root, proxy=proxy, native=native)


def test_exact_controller_projection_rejects_missing_logged_field() -> None:
    native = _Native()
    proxy = _Proxy(native)
    root = _Root()
    del root.chaser_states["chase_trial_id"]

    with pytest.raises(
        adapter.ChaserProxyRelativeFrameAdapterError,
        match="legacy trial reconstruction is prohibited",
    ):
        adapter._exact_logged_controller_state(root, proxy=proxy, native=native)


def test_adapter_rejects_proxy_bound_to_another_native_authority(monkeypatch) -> None:
    native = _Native()
    proxy = _Proxy(native)
    projection = dict(proxy.acquisition_projection_record)
    projection["source_authority_digest"] = "f" * 64
    proxy.acquisition_projection_record = MappingProxyType(projection)
    monkeypatch.setattr(
        adapter,
        "load_chaser_input_provenance_proxy_source_handle",
        lambda *args, **kwargs: proxy,
    )
    monkeypatch.setattr(
        adapter,
        "load_provider_chaser_stimulus_source_handle",
        lambda *args, **kwargs: native,
    )

    profile = Path(adapter.__file__).resolve().parents[1] / "analysis/profiles/chaser_behavior_full_v3.yaml"
    try:
        adapter.prepare_proxy_relative_frame(
            "/tmp/fixture.zarr",
            proxy_run_name="proxy-v1",
            analysis_profile_path=profile,
        )
    except adapter.ChaserProxyRelativeFrameAdapterError as exc:
        assert "exact reopened native provider" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("mismatched native authority was accepted")


def test_native_acquisition_frame_loader_requires_exact_bound_pointer(
    monkeypatch,
) -> None:
    group_path = "analysis/acquisition_camera_frames/camera-1"
    authority_node = object()
    root = {group_path: authority_node}
    ownership = object()
    bound = SimpleNamespace(
        record_ref=f"/{group_path}@acquisition_camera_frame",
        record_sha256="a" * 64,
    )
    monkeypatch.setattr(
        adapter,
        "load_acquisition_import_ownership",
        lambda observed_root, observed_node: ownership,
    )
    monkeypatch.setattr(
        adapter,
        "load_acquisition_camera_frame",
        lambda observed_root, observed_node, *, import_ownership: bound,
    )

    assert (
        adapter._load_native_acquisition_frame(
            root,
            pointer={
                "record_ref": bound.record_ref,
                "record_sha256": bound.record_sha256,
            },
        )
        is bound
    )

    with pytest.raises(
        adapter.ChaserProxyRelativeFrameAdapterError,
        match="differs from its bound record",
    ):
        adapter._load_native_acquisition_frame(
            root,
            pointer={
                "record_ref": bound.record_ref,
                "record_sha256": "b" * 64,
            },
        )


def test_exact_body_projection_retains_absent_and_invalid_rows_without_fill() -> None:
    keys = adapter.AcquisitionFrameKeys(
        recording_id="recording-1",
        acquisition_frame_id=np.asarray([10, 11, 12, 13], dtype=np.int64),
        track_sample_id=np.asarray([10, 11, 12, 13], dtype=np.int64),
        row_axis_authority_id="relative-rows-v1",
        row_axis_authority_digest="relative-rows-digest",
        timestamp_ns=np.asarray([100, 200, 300, 400], dtype=np.int64),
    )
    body = SimpleNamespace(
        dimensions=SimpleNamespace(n_instances=3),
        frame_indices=np.asarray([10, 12, 13], dtype=np.int64),
        origin_xy=np.asarray([[1.0, 1.0], [np.nan, np.nan], [4.0, 4.0]]),
        forward_axis_xy=np.asarray([[1.0, 0.0], [np.nan, np.nan], [0.0, -1.0]]),
        left_axis_xy=np.asarray([[0.0, -1.0], [np.nan, np.nan], [-1.0, 0.0]]),
        axis_valid=np.asarray([True, False, True], dtype=bool),
        run_path="analysis/body_frame_runs/body-v1",
        run_manifest={"payload_digest": "a" * 64},
        verification_digest="b" * 64,
        recipe_id="keypoint-eye-axis-v1",
        recipe_digest="c" * 64,
    )
    authority_record = {
        "schema_id": "palette.position_body_frame_motion_source_authority",
        "schema_version": 1,
        "recording_id": "recording-1",
    }
    composition = SimpleNamespace(
        source_acquisition_frame_index=np.asarray([10, 12, 13], dtype=np.int64),
        body_frame_row_index=np.asarray([0, 1, 2], dtype=np.int64),
        authority_record=authority_record,
        authority_sha256=adapter.canonical_json_sha256(authority_record),
    )

    projected, record = adapter._project_body_frame_to_relative_axis(
        frames=keys.acquisition_frame_id,
        frame_keys=keys,
        body_frame=body,
        composition=composition,
        coordinate_authority_id="camera-v1",
        scale_authority_id="scale-v1",
        timing_authority_id="time-v1",
    )

    assert projected.source_row_index.tolist() == [0, -1, 1, 2]
    assert projected.axis_valid.tolist() == [True, False, False, True]
    assert np.isnan(projected.origin_xy[1:3]).all()
    assert record["missing_source_row_count"] == 1
    assert record["present_invalid_axis_count"] == 1
    assert record["valid_axis_count"] == 2
    assert record["interpolation"] == "prohibited"
    assert record["motion_heading_fallback"] == "prohibited"


def test_body_projection_rejects_duplicate_source_acquisition_frames() -> None:
    keys = adapter.AcquisitionFrameKeys(
        recording_id="recording-1",
        acquisition_frame_id=np.asarray([10], dtype=np.int64),
        track_sample_id=np.asarray([10], dtype=np.int64),
        row_axis_authority_id="relative-rows-v1",
        row_axis_authority_digest="relative-rows-digest",
        timestamp_ns=np.asarray([100], dtype=np.int64),
    )
    body = SimpleNamespace(
        dimensions=SimpleNamespace(n_instances=2),
        frame_indices=np.asarray([10, 10], dtype=np.int64),
    )
    composition = SimpleNamespace(
        source_acquisition_frame_index=np.asarray([10, 10], dtype=np.int64),
        body_frame_row_index=np.asarray([0, 1], dtype=np.int64),
    )

    with pytest.raises(
        adapter.ChaserProxyRelativeFrameAdapterError,
        match="multiple observations",
    ):
        adapter._project_body_frame_to_relative_axis(
            frames=keys.acquisition_frame_id,
            frame_keys=keys,
            body_frame=body,
            composition=composition,
            coordinate_authority_id="camera-v1",
            scale_authority_id="scale-v1",
            timing_authority_id="time-v1",
        )


def test_prepared_proxy_json_exposes_body_projection_coverage_record() -> None:
    record = {
        "schema_id": "palette.chaser_relative_frame.body_frame_projection_binding",
        "schema_version": 1,
        "recording_id": "recording-1",
        "relative_frame_count": 4,
        "exact_source_row_count": 3,
        "missing_source_row_count": 1,
        "present_invalid_axis_count": 1,
        "valid_axis_count": 2,
    }
    prepared = adapter.PreparedProxyRelativeFrame(
        prepared=SimpleNamespace(payload_digest="prepared-digest"),
        proxy_run_path="analysis/chaser_input_provenance_proxy_runs/proxy-v1",
        proxy_manifest_sha256="a" * 64,
        native_run_path="analysis/provider_chaser_distance_candidate_runs/native-v1",
        native_manifest_sha256="b" * 64,
        coordinate_lineage_sha256="c" * 64,
        timing_authority_sha256="d" * 64,
        subject_metadata_sha256="e" * 64,
        body_frame_run_path="analysis/body_frame_runs/body-v1",
        body_frame_manifest_sha256="f" * 64,
        body_frame_projection_sha256=adapter.canonical_json_sha256(record),
        body_frame_projection_record=record,
    )

    payload = prepared.to_json()

    assert payload["body_frame_projection"] == record
    assert payload["body_frame_projection_sha256"] == adapter.canonical_json_sha256(
        payload["body_frame_projection"]
    )
