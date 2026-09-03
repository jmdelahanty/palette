from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from apps.marimo.components.chaser_exact.spatial_occupancy import (
    SPATIAL_OCCUPANCY_DISPLAY_RECIPE,
    _spatial_values,
    build_exact_spatial_occupancy_output,
)
from fisheye.visualization.chaser_appearance import (
    ChaserAppearance,
    ChaserAppearanceProjection,
)


class _SpatialHandle:
    successor_kind = "chaser_spatial_occupancy"
    deep_audited = True

    def __init__(self) -> None:
        shape = (2, 3, 2, 2)
        counts = np.zeros(shape, dtype=np.int64)
        counts[0, :, 0, 0] = 1
        counts[0, :, 0, 1] = 1
        counts[1, :, 0, 1] = 2
        candidate = np.full((2, 3), 4, dtype=np.int64)
        declared = np.full((2, 3), 3, dtype=np.int64)
        finite = np.full((2, 3), 3, dtype=np.int64)
        in_arena = np.full((2, 3), 2, dtype=np.int64)
        invalid = np.full((2, 3), 1, dtype=np.int64)
        out_of_arena = np.full((2, 3), 1, dtype=np.int64)
        self.arrays = {
            "occupancy_count": counts,
            "occupancy_density_valid_in_arena": counts / 2.0,
            "occupancy_fraction_candidate_epoch": counts / 4.0,
            "x_bin_edges_mm": np.asarray([-2.0, 0.0, 2.0]),
            "y_bin_edges_mm": np.asarray([-2.0, 0.0, 2.0]),
            "arena_bin_center_mask": np.asarray(
                [[True, False], [True, True]], dtype=bool
            ),
            "candidate_frame_count": candidate,
            "declared_valid_position_frame_count": declared,
            "finite_valid_position_frame_count": finite,
            "in_arena_position_frame_count": in_arena,
            "invalid_position_frame_count": invalid,
            "out_of_arena_position_frame_count": out_of_arena,
            "in_arena_coverage_fraction_candidate": in_arena / candidate,
            "in_arena_fraction_finite_valid": in_arena / finite,
        }
        self.scientific_manifest = {
            "dimensions": {
                "n_providers": 2,
                "n_epochs": 3,
                "grid_rows": 2,
                "grid_columns": 2,
            },
            "identity_registries": {
                "provider_role": {"0": "keypoint", "1": "detection"},
                "epoch_role": {
                    "0": "chaser_pre",
                    "1": "chaser_training",
                    "2": "chaser_post",
                },
            },
            "arena": {
                "center_x_px": 100.0,
                "center_y_px": 100.0,
                "radius_mm": 2.0,
                "mm_per_pixel": 0.02,
            },
            "grid": {
                "coordinate_orientation": "+x_right_+y_down",
                "normalization_policy_id": (
                    "valid_in_arena_and_candidate_epoch_denominators_v1"
                ),
                "bin_width_mm": 2.0,
            },
        }

    def array(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def require_verified_arrays(self, names) -> None:
        assert set(names).issubset(self.arrays)


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

    def __init__(self) -> None:
        frame = np.arange(self.n_frames, dtype=np.int64)
        identity = np.tile(np.asarray([1, 2], dtype=np.uint16), self.n_frames)
        role = np.tile(np.asarray([1, 2], dtype=np.uint8), self.n_frames)
        chaser = np.tile(
            np.asarray([[50.0, 100.0], [150.0, 100.0]], dtype=np.float64),
            (self.n_frames, 1, 1),
        )
        chaser[4:, :, 1] = 150.0
        self.arrays = {
            "acquisition_frame_id": np.repeat(frame, self.n_chasers),
            "selection_member": np.ones(self.n_frames * self.n_chasers, dtype=bool),
            "chaser_position_xy_px": chaser.reshape(-1, 2),
            "chaser_position_valid": np.ones(
                self.n_frames * self.n_chasers, dtype=bool
            ),
            "chaser_occurrence_member": np.ones(
                self.n_frames * self.n_chasers, dtype=bool
            ),
            "chaser_identity_code": identity,
            "chaser_behavior_role_code": role,
        }

    def frame_chaser(self, name: str) -> np.ndarray:
        values = self.arrays[name]
        return values.reshape((self.n_frames, self.n_chasers) + values.shape[1:])

    def collapsed_frame(self, name: str) -> np.ndarray:
        return self.frame_chaser(name)[:, 0, ...]


def _appearance() -> ChaserAppearanceProjection:
    values = []
    for identity_code, index, role_code, role, symbol in (
        (1, 0, 1, "aggressive", "star"),
        (2, 1, 2, "inert", "circle"),
    ):
        values.append(
            ChaserAppearance(
                identity_code=identity_code,
                chaser_index=index,
                identity=f"stimulus-v1:chaser_index:{index}",
                behavior_role_code=role_code,
                behavior_role=role,
                experimental_color_rgba=(0.0, 0.0, 1.0, 1.0),
                experimental_color_hex="#0000ff",
                experimental_color_css="rgba(0,0,255,1)",
                plotly_role_symbol=symbol,
                matplotlib_role_marker="*" if role == "aggressive" else "o",
                contrast_outline_hex="#ffffff",
            )
        )
    return ChaserAppearanceProjection(
        recording_id="recording-1",
        source_stimulus_run_path="analysis/stimulus_runs/stimulus-v1",
        source_protocol_sha256="a" * 64,
        occurrence_binding_sha256="b" * 64,
        appearances=tuple(values),
        projection_sha256="c" * 64,
    )


def _projection(handle: _SpatialHandle | None = None) -> Any:
    relative = _Relative()
    return SimpleNamespace(
        spatial=handle or _SpatialHandle(),
        recording_id="recording-1",
        provider_ids=("keypoint.v1", "detection.v1"),
        relatives=(relative, relative),
        chaser_appearance=_appearance(),
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
        provenance={"bundle_manifest_sha256": "a" * 64},
    )


def test_spatial_heatmap_uses_persisted_density_and_provider_difference() -> None:
    output = build_exact_spatial_occupancy_output(_Marimo, go, _projection())

    figure = output[1]
    assert len(figure.data) == 21
    keypoint_pre = np.asarray(figure.data[0].z)
    detection_pre = np.asarray(figure.data[1].z)
    difference_pre = np.asarray(figure.data[2].z)
    assert keypoint_pre.shape == (1, 1)
    assert keypoint_pre[0, 0] == 100.0
    assert detection_pre[0, 0] == 100.0
    assert difference_pre[0, 0] == 0.0
    customdata = np.asarray(figure.data[0].customdata)
    assert customdata[0, 0, 0] == 2
    assert bool(customdata[0, 0, 1])
    assert customdata[0, 0, 2] == 4.0
    assert np.isfinite(keypoint_pre[0, 0])
    display = figure.layout.meta["spatial_occupancy_display"]
    assert display["recipe_id"] == SPATIAL_OCCUPANCY_DISPLAY_RECIPE
    assert display["source_array"] == "occupancy_density_valid_in_arena"
    assert display["source_arrays"] == [
        "occupancy_density_valid_in_arena",
        "occupancy_fraction_candidate_epoch",
    ]
    assert display["scientific_recomputation"] is False
    assert display["interpolation"] == "prohibited"
    assert "missing=1" in figure.layout.annotations[0].text
    assert "out=1" in figure.layout.annotations[0].text
    assert display["provider_epoch_denominators"]["candidate_frame_count"][0][0] == 4
    assert (
        display["provider_epoch_denominators"]["invalid_position_frame_count"][0][0]
        == 1
    )
    assert display["default_normalization"] == "valid_in_arena"
    assert display["available_normalizations"] == [
        "valid_in_arena",
        "candidate_epoch",
    ]
    assert display["default_display_mode"] == "4_mm_valid_in_arena_robust_p98"
    assert display["available_display_modes"] == [
        "4_mm_valid_in_arena_robust_p98",
        "4_mm_valid_in_arena_full_range",
        "2_mm_valid_in_arena_robust_p98",
        "2_mm_valid_in_arena_full_range",
        "4_mm_candidate_epoch_robust_p98",
        "4_mm_candidate_epoch_full_range",
        "2_mm_candidate_epoch_robust_p98",
        "2_mm_candidate_epoch_full_range",
    ]
    assert (
        display["display_surfaces"]["2mm_valid_in_arena"]["count_aggregation"] == "none"
    )
    assert (
        display["display_surfaces"]["4mm_valid_in_arena"]["count_aggregation"]
        == "exact_2x2_sum"
    )
    assert display["display_surfaces"]["4mm_valid_in_arena"]["grid_shape"] == [
        1,
        1,
    ]
    assert (
        display["display_surfaces"]["2mm_candidate_epoch"]["source_array"]
        == "occupancy_fraction_candidate_epoch"
    )
    assert (
        display["display_surfaces"]["2mm_candidate_epoch"]["denominator"]
        == "candidate_frame_count"
    )
    density_scale = display["display_surfaces"]["4mm_valid_in_arena"][
        "color_scale_percent_per_bin"
    ]
    assert figure.layout.coloraxis.cmax == density_scale["robust_limit"]
    assert density_scale["full_range_reference_available"] is True
    buttons = figure.layout.updatemenus[0].buttons
    assert [button.label for button in buttons] == [
        "4 mm · valid in-arena · robust p98",
        "4 mm · valid in-arena · full range",
        "2 mm · valid in-arena · robust p98",
        "2 mm · valid in-arena · full range",
        "4 mm · candidate epoch · robust p98",
        "4 mm · candidate epoch · full range",
        "2 mm · candidate epoch · robust p98",
        "2 mm · candidate epoch · full range",
    ]
    coarse_trace_z = buttons[0].args[0]["z"]
    assert np.asarray(coarse_trace_z[0]).shape == (1, 1)
    assert np.asarray(coarse_trace_z[0])[0, 0] == 100.0
    candidate_trace_z = buttons[6].args[0]["z"]
    assert np.asarray(candidate_trace_z[0])[0, 1] == 25.0
    assert np.asarray(candidate_trace_z[1])[0, 1] == 50.0
    assert np.asarray(candidate_trace_z[2])[0, 1] == 25.0
    assert "candidate epoch" in buttons[6].args[0]["hovertemplate"][0]
    coarse_candidate_trace_z = buttons[4].args[0]["z"]
    assert np.asarray(coarse_candidate_trace_z[0])[0, 0] == 50.0
    overlay = display["chaser_location_overlay"]
    assert overlay["color_source"] == "sealed_protocol_rgba"
    assert overlay["role_encoding"] == "independent_marker_symbol_and_legend_text"
    marker_traces = [trace for trace in figure.data if trace.type == "scatter"]
    assert {trace.marker.color for trace in marker_traces} == {"rgba(0,0,255,1)"}
    assert {trace.marker.symbol for trace in marker_traces} == {"star", "circle"}


def test_spatial_projection_rejects_denominator_nonconservation() -> None:
    handle = _SpatialHandle()
    handle.arrays["in_arena_position_frame_count"] = np.full((2, 3), 3, dtype=np.int64)

    with pytest.raises(ValueError, match="conservation"):
        _spatial_values(_projection(handle))
