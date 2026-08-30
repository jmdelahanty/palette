from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import plotly.graph_objects as go

from apps.marimo.components.chaser_exact.trajectory_overlays import (
    build_exact_trajectory_overlays_output,
)
from fisheye.visualization.chaser_appearance import (
    ChaserAppearance,
    ChaserAppearanceProjection,
)


class _Marimo:
    @staticmethod
    def callout(value: str, *, kind: str) -> dict[str, str]:
        return {"value": value, "kind": kind}

    @staticmethod
    def vstack(values: list[Any]) -> list[Any]:
        return values


class _Relative:
    n_frames = 6
    n_chasers = 2

    def __init__(self, *, fish_offset: float) -> None:
        frame = np.arange(self.n_frames, dtype=np.int64)
        fish = np.column_stack((100.0 + frame + fish_offset, 100.0 + frame))
        chaser = np.tile(
            np.asarray([[70.0, 70.0], [130.0, 70.0]], dtype=np.float64),
            (self.n_frames, 1, 1),
        )
        self.arrays = {
            "acquisition_frame_id": np.repeat(frame, self.n_chasers),
            "selection_member": np.ones(self.n_frames * self.n_chasers, dtype=bool),
            "fish_position_xy_px": np.repeat(fish, self.n_chasers, axis=0),
            "fish_position_valid": np.ones(self.n_frames * self.n_chasers, dtype=bool),
            "chaser_position_xy_px": chaser.reshape(-1, 2),
            "chaser_position_valid": np.ones(
                self.n_frames * self.n_chasers, dtype=bool
            ),
            "chaser_occurrence_member": np.ones(
                self.n_frames * self.n_chasers, dtype=bool
            ),
            "chaser_identity_code": np.tile(
                np.asarray([1, 2], dtype=np.uint16), self.n_frames
            ),
            "chaser_behavior_role_code": np.tile(
                np.asarray([1, 2], dtype=np.uint8), self.n_frames
            ),
        }

    def frame_chaser(self, name: str) -> np.ndarray:
        values = self.arrays[name]
        return values.reshape((self.n_frames, self.n_chasers) + values.shape[1:])

    def collapsed_frame(self, name: str) -> np.ndarray:
        return self.frame_chaser(name)[:, 0, ...]


def _appearance() -> ChaserAppearanceProjection:
    appearances = (
        ChaserAppearance(
            identity_code=1,
            chaser_index=0,
            identity="stimulus-v1:chaser_index:0",
            behavior_role_code=1,
            behavior_role="aggressive",
            experimental_color_rgba=(0.0, 0.0, 1.0, 1.0),
            experimental_color_hex="#0000ff",
            experimental_color_css="rgba(0,0,255,1)",
            plotly_role_symbol="star",
            matplotlib_role_marker="*",
            contrast_outline_hex="#ffffff",
        ),
        ChaserAppearance(
            identity_code=2,
            chaser_index=1,
            identity="stimulus-v1:chaser_index:1",
            behavior_role_code=2,
            behavior_role="inert",
            experimental_color_rgba=(0.0, 0.0, 1.0, 1.0),
            experimental_color_hex="#0000ff",
            experimental_color_css="rgba(0,0,255,1)",
            plotly_role_symbol="circle",
            matplotlib_role_marker="o",
            contrast_outline_hex="#ffffff",
        ),
    )
    return ChaserAppearanceProjection(
        recording_id="recording-1",
        source_stimulus_run_path="analysis/stimulus_runs/stimulus-v1",
        source_protocol_sha256="a" * 64,
        occurrence_binding_sha256="b" * 64,
        appearances=appearances,
        projection_sha256="c" * 64,
    )


def _projection() -> Any:
    return SimpleNamespace(
        relatives=(_Relative(fish_offset=0.0), _Relative(fish_offset=1.0)),
        chaser_appearance=_appearance(),
        radials=(
            SimpleNamespace(
                scientific_manifest={
                    "identity_registries": {
                        "behavior_role": {"1": "aggressive", "2": "inert"}
                    },
                    "arena": {
                        "center_x_px": 100.0,
                        "center_y_px": 100.0,
                        "radius_px": 50.0,
                    },
                }
            ),
        ),
        provider_ids=("keypoint.v1", "detection.v1"),
        epoch_records=(
            {
                "analysis_role": "chaser_pre",
                "start_frame": 0,
                "end_frame_exclusive": 2,
            },
            {
                "analysis_role": "chaser_training",
                "start_frame": 2,
                "end_frame_exclusive": 4,
            },
            {
                "analysis_role": "chaser_post",
                "start_frame": 4,
                "end_frame_exclusive": 6,
            },
        ),
        recording_id="recording-1",
        provenance={"bundle_manifest_sha256": "d" * 64},
    )


def test_trajectory_uses_protocol_color_and_independent_role_symbols() -> None:
    output = build_exact_trajectory_overlays_output(_Marimo, go, _projection())

    figure = output[1]
    chasers = [trace for trace in figure.data if "protocol chaser" in trace.name]
    assert len(chasers) == 12
    assert {trace.marker.color for trace in chasers} == {"rgba(0,0,255,1)"}
    assert {trace.marker.symbol for trace in chasers} == {"star", "circle"}
    assert {trace.name for trace in chasers} == {
        "aggressive · protocol chaser 0",
        "inert · protocol chaser 1",
    }
    appearance = figure.layout.meta["trajectory_chaser_appearance"]
    assert appearance["color_source"] == "sealed_protocol_rgba"
    assert appearance["index_palette_fallback"] == "prohibited"
