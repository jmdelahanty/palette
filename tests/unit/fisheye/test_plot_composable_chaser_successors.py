from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fisheye.utils.plot_composable_chaser_successors import (
    ComposableChaserPlotError,
    render_dashboard,
)


@dataclass
class _Handle:
    successor_kind: str
    recording_id: str
    scientific_payload_sha256: str
    scientific_manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    deep_audited: bool = True

    def array(self, name: str) -> np.ndarray:
        return self.arrays[name]


def _handles() -> tuple[_Handle, _Handle, _Handle]:
    controller_digest = "a" * 64
    bout_digest = "b" * 64
    controller = _Handle(
        successor_kind="controller_chase_trials",
        recording_id="recording-1",
        scientific_payload_sha256=controller_digest,
        scientific_manifest={},
        arrays={
            "start_acquisition_frame_id": np.asarray([100, 200], dtype=np.int64),
            "end_acquisition_frame_id_inclusive": np.asarray(
                [149, 259], dtype=np.int64
            ),
            "trial_ordinal": np.asarray([1, 2], dtype=np.int32),
            "logged_trial_id": np.asarray([11, 12], dtype=np.int64),
            "chaser_identity_code": np.asarray([1, 1], dtype=np.uint16),
            "gap_fraction": np.asarray([0.0, 0.1], dtype=np.float64),
        },
    )
    bout = _Handle(
        successor_kind="generalized_chaser_bout_response",
        recording_id="recording-1",
        scientific_payload_sha256=bout_digest,
        scientific_manifest={
            "sources": {
                "controller_trial_payload_sha256": controller_digest,
                "motion": {
                    "relative_frame_projection": {
                        "missing_relative_frame_count": 3
                    }
                },
            },
            "identity_registries": {
                "semantic_role": {"2": "chaser_training"}
            },
        },
        arrays={
            "summary_role_code": np.asarray([2, 2], dtype=np.uint8),
            "summary_chaser_identity_code": np.asarray([1, 1], dtype=np.uint16),
            "summary_distance_bin_index": np.asarray([0, 1], dtype=np.int16),
            "summary_distance_bin_start_mm": np.asarray([0, 8], dtype=np.float32),
            "summary_distance_bin_end_mm": np.asarray([8, np.inf], dtype=np.float32),
            "summary_bout_rate_per_min": np.asarray([1.5, 2.5], dtype=np.float64),
            "summary_bout_count": np.asarray([2, 3], dtype=np.int64),
        },
    )
    escape = _Handle(
        successor_kind="chaser_escape_freeze",
        recording_id="recording-1",
        scientific_payload_sha256="c" * 64,
        scientific_manifest={
            "sources": {
                "controller_trial_payload_sha256": controller_digest,
                "bout_response_payload_sha256": bout_digest,
            }
        },
        arrays={
            "trial_response_class_code": np.asarray([1, 2], dtype=np.uint8),
            "sweep_trial_row_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "sweep_speed_threshold_mm_s": np.asarray(
                [10, 20, 10, 20], dtype=np.float32
            ),
            "sweep_escape_event_rate_per_min": np.asarray(
                [3, 2, 1, 0], dtype=np.float64
            ),
        },
    )
    return controller, bout, escape


def test_render_dashboard_writes_png_and_pdf(tmp_path: Path) -> None:
    png, pdf = render_dashboard(*_handles(), output_stem=tmp_path / "dashboard")

    assert png.is_file() and png.stat().st_size > 0
    assert pdf.is_file() and pdf.stat().st_size > 0


def test_render_dashboard_rejects_stale_dependency(tmp_path: Path) -> None:
    controller, bout, escape = _handles()
    escape.scientific_manifest["sources"]["bout_response_payload_sha256"] = "f" * 64

    with pytest.raises(ComposableChaserPlotError, match="stale or mixed"):
        render_dashboard(controller, bout, escape, output_stem=tmp_path / "bad")
