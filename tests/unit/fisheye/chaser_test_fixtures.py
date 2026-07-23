"""Shared logical fixtures for downstream chaser-analysis unit tests.

Canonical publication and reader tests use real sealed Zarr-v3 templates.  The
downstream numerical and rendering suites instead patch the already-tested
reader boundary and consume detached logical arrays from their purpose-built
archives.  This keeps those tests focused on their own calculations without
teaching production readers to accept unsealed fixtures.
"""

from __future__ import annotations

import sys
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from fisheye.analysis import chaser_distance_io
from fisheye.shared.json_safety import decode_null_terminated_text


def _readonly(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _array(group: Any, path: str, *, dtype: Any | None = None) -> np.ndarray:
    return _readonly(group[path][:], dtype=dtype)


def _decode_rows(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        decode_null_terminated_text(np.asarray(row, dtype=np.uint8)).strip()
        for row in values
    )


class LogicalChaserDistanceFixture:
    """Detached test double for an already-verified chaser-distance snapshot."""

    def __init__(self, root: Any, run_name: str) -> None:
        self.run_name = run_name
        self.run_path = f"analysis/chaser_distance_runs/{run_name}"
        run = root[self.run_path]
        attrs = dict(run.attrs)

        self.recording_id = str(
            attrs.get("recording_id")
            or root.attrs.get("recording_id")
            or root.attrs.get("recording_name")
            or "test_recording"
        )
        self.authority_status = "verified_test_fixture"
        self.behavior_authority_status = "verified_test_fixture"
        self.source_detection_path = str(attrs.get("source_detection_path") or "")
        self.source_stimulus_run = str(attrs.get("source_stimulus_run") or "")
        self.source_stimulus_path = str(attrs.get("source_stimulus_path") or "")
        self.source_stimulus_epoch_run = attrs.get("source_stimulus_epoch_run")
        self.source_stimulus_epoch_path = attrs.get("source_stimulus_epoch_path")
        self.fps = float(attrs["fps"])
        self.total_frames = int(attrs["total_frames"])
        self.pixels_per_mm_projector = float(attrs["pixels_per_mm_projector"])
        self.coordinate_space_id = "arena_relative_canvas_px"
        self.coordinate_origin = "arena_top_left"
        self.positive_x = "right"
        self.positive_y = "down"
        self.reference_width_px = 0
        self.reference_height_px = 0
        self.pixel_convention = "continuous"
        self.arena_coordinate_descriptor = None
        self.source_camera_coordinate_descriptor = None
        self.coordinate_descriptor_sha256 = MappingProxyType({})
        self.measurement_descriptor_sha256 = MappingProxyType({})
        self.publication_seal_ref = f"/{self.run_path}@test_publication"
        self.publication_seal_sha256 = "0" * 64
        self.surface_manifest_ref = f"/{self.run_path}@test_surface_manifest"
        self.surface_manifest_sha256 = "1" * 64
        self.row_identity_ref = f"/{self.run_path}@test_row_identity"
        self.row_identity_sha256 = "2" * 64
        self.archive_identity = None

        self.camera_frame_id = _array(run, "frames/camera_frame_id", dtype=np.int64)
        self.stimulus_frame_num = _array(
            run,
            "frames/stimulus_frame_num",
            dtype=np.int64,
        )
        self.timestamp_ns = _array(run, "frames/timestamp_ns", dtype=np.int64)
        self.stimulus_epoch_window_id = _array(
            run,
            "frames/stimulus_epoch_window_id",
            dtype=np.int32,
        )
        self.stimulus_state_key = self.camera_frame_id
        self.source_detection_row_index = _readonly(
            np.arange(self.total_frames, dtype=np.int64)
        )
        self.fish_centroid_img_xy = _array(
            run,
            "positions/fish_centroid_img_xy",
        )
        self.fish_centroid_arena_xy = _array(
            run,
            "positions/fish_centroid_arena_xy",
        )
        self.chaser_arena_xy = _array(run, "positions/chaser_arena_xy")
        self.fish_valid = _array(run, "positions/fish_valid", dtype=bool)
        self.chaser_valid = _array(run, "positions/chaser_valid", dtype=bool)
        self.distance_px = _array(run, "distances/distance_px")
        self.distance_mm = _array(run, "distances/distance_mm")
        self.nearest_chaser_index = _array(
            run,
            "distances/nearest_chaser_index",
        )
        self.nearest_distance_mm = _array(
            run,
            "distances/nearest_distance_mm",
        )
        self.chaser_index = _array(run, "chasers/chaser_index")
        self.chaser_indices = self.chaser_index

        instance_rows = _array(run, "chasers/stimulus_instance_id_bytes")
        track_rows = _array(run, "chasers/source_track_key_bytes")
        self.stimulus_instance_ids = _decode_rows(instance_rows)
        self.source_track_keys = _decode_rows(track_rows)
        self.epoch_window_id = _array(run, "epoch_summary/window_id")
        self.epoch_label_bytes = _array(run, "epoch_summary/label_bytes")
        self.epoch_labels = _decode_rows(self.epoch_label_bytes)
        self.epoch_start_frame = _array(run, "epoch_summary/start_frame")
        self.epoch_end_frame = _array(run, "epoch_summary/end_frame")

    def authority_record(self) -> dict[str, Any]:
        return {
            "schema_id": "palette.test.chaser_distance_read_authority",
            "schema_version": 1,
            "run_ref": f"/{self.run_path}",
        }

    def require_behavior_authority(self) -> None:
        return None

    def require_arena_geometry_authority(self) -> None:
        return None

    def require_stimulus_protocol_authority(self, _semantic_label: str) -> None:
        return None

    def require_derived_surface_authority(self, _relative_path: str) -> None:
        return None


def _selected_run_name(root: Any, requested: str) -> str:
    parent = root["analysis/chaser_distance_runs"]
    explicit = str(requested).strip()
    if explicit and explicit != "latest":
        return explicit
    for selector in ("authoritative_run", "latest_complete", "latest"):
        value = parent.attrs.get(selector)
        if isinstance(value, str) and value.strip():
            return value.strip().rstrip("/").rsplit("/", 1)[-1]
    names = sorted(parent.group_keys())
    if not names:
        raise chaser_distance_io.ChaserDistanceReadError(
            "Archive has no chaser-distance run."
        )
    return str(names[-1])


@pytest.fixture
def logical_chaser_distance_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch downstream imports at the logical reader boundary for one test."""

    canonical_loader = chaser_distance_io.load_chaser_distance_run
    canonical_reject = (
        chaser_distance_io.reject_unsealed_chaser_derived_publication
    )

    def load_fixture(root: Any, *, run_name: str = "latest") -> Any:
        try:
            selected = _selected_run_name(root, run_name)
            run = root[f"analysis/chaser_distance_runs/{selected}"]
        except Exception:
            return canonical_loader(root, run_name=run_name)
        if run.attrs.get("coordinate_publication_status") != "legacy_unsealed":
            return canonical_loader(root, run_name=run_name)
        return LogicalChaserDistanceFixture(root, selected)

    def permit_test_publication(
        root: Any,
        *,
        run_name: str,
        run_path: str,
        relative_path: str,
    ) -> None:
        snapshot = load_fixture(root, run_name=run_name)
        if snapshot.run_path != str(run_path).strip("/"):
            raise chaser_distance_io.ChaserDistanceReadError(
                "Test fixture base run changed before derived publication."
            )
        if not str(relative_path).strip("/"):
            raise chaser_distance_io.ChaserDistanceReadError(
                "Test fixture derived path is empty."
            )

    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        if getattr(module, "load_chaser_distance_run", None) is canonical_loader:
            monkeypatch.setattr(module, "load_chaser_distance_run", load_fixture)
        if (
            getattr(module, "reject_unsealed_chaser_derived_publication", None)
            is canonical_reject
        ):
            monkeypatch.setattr(
                module,
                "reject_unsealed_chaser_derived_publication",
                permit_test_publication,
            )
