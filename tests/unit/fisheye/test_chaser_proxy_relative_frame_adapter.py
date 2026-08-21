from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

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
                "/analysis/acquisition_frame_clock_runs/clock-v1"
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
                "selected_source_stimulus_run_row_index",
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

    def __getitem__(self, path: str) -> _Group:
        assert path == "analysis/stimulus_runs/stimulus-v1"
        return self.stimulus


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


def test_adapter_applies_typed_arena_to_camera_chain_without_timestamp_claim(
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
        sha256="8" * 64,
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
    assert not np.any(base["timestamp_valid"])
    assert bound.prepared.manifest["timing_policy"]["timestamp_field"] is None
    transform = bound.prepared.manifest["context"][
        "arena_to_source_camera_transform"
    ]["record"]
    assert transform["from_coordinate_space"] == "arena_relative_canvas_px"
    assert transform["to_coordinate_space"] == "source_camera_image_px"
    assert transform["no_reflection_or_heuristic_flip"] is True
    assert bound.prepared.manifest["selector_eligible"] is False


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
