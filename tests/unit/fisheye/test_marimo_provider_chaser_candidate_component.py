from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import pytest
import zarr

from apps.marimo.components.analysis_catalog import group_specs_by_provider
from apps.marimo.components.provider_chaser_candidate import (
    available_provider_chaser_candidate_analysis_ids,
)
from apps.marimo.components.registry import (
    PROVIDER_CHASER_CANDIDATE_RENDERER,
    discover_provider_chaser_candidate_options,
)
from apps.marimo.components.static_artifacts import build_static_artifact_views
from fisheye.analysis.chaser_distance_runs import ChaserDistanceWindow
from fisheye.analysis.provider_chaser_distance_candidates import (
    ProviderChaserDistanceCandidate,
    _materialize_local,
)


RUN_NAME = "provider_canary_v1"
RUN_PATH = f"analysis/provider_chaser_distance_candidate_runs/{RUN_NAME}"
HISTOGRAM_ARTIFACT = f"{RUN_PATH}/visualizations/distance_histogram_png"
TRACE_ARTIFACT = f"{RUN_PATH}/visualizations/distance_trace_png"
HISTOGRAM_LABEL = "Distance histogram (stimulus samples)"
TRACE_LABEL = "Distance trace (stimulus samples)"


def _candidate() -> ProviderChaserDistanceCandidate:
    frame_count = 4
    windows = (
        ChaserDistanceWindow(
            window_id=0,
            label="pre",
            start_frame=0,
            end_frame=1,
            start_time_s=0.0,
            end_time_s=0.1,
            duration_s=0.2,
        ),
        ChaserDistanceWindow(
            window_id=1,
            label="chaser",
            start_frame=2,
            end_frame=3,
            start_time_s=0.2,
            end_time_s=0.3,
            duration_s=0.2,
        ),
    )
    arrays = {
        "samples/stimulus_frame_num": np.arange(frame_count, dtype=np.int64),
        "samples/source_acquisition_frame_index": np.arange(frame_count, dtype=np.int64),
        "samples/timestamp_ns": np.arange(frame_count, dtype=np.int64) * 100_000_000,
        "samples/stimulus_epoch_window_id": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "samples/source_stimulus_run_row_index": np.arange(frame_count, dtype=np.int64).reshape(-1, 1),
        "samples/source_stimulus_source_row_index": np.arange(100, 100 + frame_count, dtype=np.int64).reshape(-1, 1),
        "positions/source_position_run_row_index": np.arange(frame_count, dtype=np.int64),
        "positions/source_position_source_row_index": np.arange(frame_count, dtype=np.int64),
        "positions/source_position_instance_key": np.arange(10, 10 + frame_count, dtype=np.uint64),
        "positions/source_position_failure_reason_code": np.zeros(frame_count, dtype=np.uint16),
        "positions/fish_position_source_camera_xy": np.asarray(
            [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0]], dtype=np.float32
        ),
        "positions/fish_valid": np.ones(frame_count, dtype=bool),
        "positions/fish_position_arena_xy": np.asarray(
            [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0]], dtype=np.float32
        ),
        "positions/chaser_position_arena_xy": np.asarray(
            [[[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]],
            dtype=np.float32,
        ),
        "positions/chaser_valid": np.ones((frame_count, 1), dtype=bool),
        "chasers/chaser_index": np.asarray([0], dtype=np.int16),
        "distances/distance_px": np.asarray([[2.0], [3.0], [4.0], [5.0]], dtype=np.float32),
        "distances/distance_mm": np.asarray([[1.0], [1.5], [2.0], [2.5]], dtype=np.float32),
        "distances/nearest_chaser_index": np.zeros(frame_count, dtype=np.int16),
        "distances/nearest_distance_mm": np.asarray([1.0, 1.5, 2.0, 2.5], dtype=np.float32),
        "epoch_summary/window_id": np.asarray([0, 1], dtype=np.int32),
        "epoch_summary/label_bytes": np.zeros((2, 96), dtype=np.uint8),
        "epoch_summary/start_frame": np.asarray([0, 2], dtype=np.int64),
        "epoch_summary/end_frame": np.asarray([1, 3], dtype=np.int64),
        "epoch_summary/valid_frame_count": np.full((2, 1), 2, dtype=np.int64),
        "epoch_summary/mean_distance_mm": np.ones((2, 1), dtype=np.float32),
        "epoch_summary/min_distance_mm": np.ones((2, 1), dtype=np.float32),
        "epoch_summary/p05_distance_mm": np.ones((2, 1), dtype=np.float32),
        "epoch_summary/p50_distance_mm": np.ones((2, 1), dtype=np.float32),
        "epoch_summary/p95_distance_mm": np.ones((2, 1), dtype=np.float32),
        "epoch_summary/fraction_within_threshold": np.ones((2, 1), dtype=np.float32),
        "epoch_distributions/window_id": np.asarray([0, 1], dtype=np.int32),
        "epoch_distributions/chaser_index": np.asarray([0], dtype=np.int16),
        "epoch_distributions/bin_edges_mm": np.asarray([0.0, 2.0, 4.0], dtype=np.float32),
        "epoch_distributions/bin_centers_mm": np.asarray([1.0, 3.0], dtype=np.float32),
        "epoch_distributions/hist_counts": np.ones((2, 1, 2), dtype=np.uint32),
        "epoch_distributions/hist_density": np.full((2, 1, 2), 0.25, dtype=np.float32),
        "epoch_distributions/valid_sample_count": np.full((2, 1), 2, dtype=np.int64),
    }
    return ProviderChaserDistanceCandidate(
        source_zarr=Path("source.zarr"),
        run_name=RUN_NAME,
        recording_id="recording-1",
        position_run_path="analysis/subject_position_runs/observation/position_v1",
        position_manifest_sha256="a" * 64,
        position_estimator_id="keypoint_anatomical_triad_mean.v1",
        stimulus_run_path="analysis/stimulus_runs/stimulus_v1",
        stimulus_epoch_run_path="analysis/stimulus_epoch_runs/epochs_v1",
        total_frames=frame_count,
        fps=10.0,
        pixels_per_mm_projector=2.0,
        threshold_mm=20.0,
        distribution_bin_width_mm=2.0,
        windows=windows,
        arrays=arrays,
        source_authority=MappingProxyType({"exact_test_fixture": True}),
    )


def _materialized_candidate(tmp_path: Path) -> Path:
    archive = tmp_path / "recording_analysis.zarr"
    _materialize_local(_candidate(), local_zarr=archive)
    return archive


def _reconsolidate(path: Path) -> None:
    zarr.consolidate_metadata(str(path))


def test_provider_candidate_discovery_is_exact_and_grouped_as_unpromoted(tmp_path: Path) -> None:
    archive = _materialized_candidate(tmp_path)

    options = discover_provider_chaser_candidate_options(archive)

    assert len(options) == 1
    option = options[0]
    assert option.renderer == PROVIDER_CHASER_CANDIDATE_RENDERER
    assert option.run_path == RUN_PATH
    assert option.label
    assert "unpromoted" in option.label.casefold()
    assert "selector" in option.label.casefold()
    assert option.attrs["stage_selector_eligible"] is False
    assert option.spec["static_artifacts"] == {
        TRACE_LABEL: "visualizations/distance_trace_png",
        HISTOGRAM_LABEL: "visualizations/distance_histogram_png",
    }

    grouped = group_specs_by_provider(options)
    assert set(grouped) == {"stimulus_chaser_candidate"}
    assert grouped["stimulus_chaser_candidate"] == options


def test_provider_candidate_filters_by_exact_run_and_artifact_paths(tmp_path: Path) -> None:
    archive = _materialized_candidate(tmp_path)

    assert [item.run_path for item in discover_provider_chaser_candidate_options(
        archive, run_path_filter=RUN_PATH
    )] == [RUN_PATH]
    assert discover_provider_chaser_candidate_options(
        archive, run_path_filter=f"{RUN_PATH}/child"
    ) == []
    assert [item.run_path for item in discover_provider_chaser_candidate_options(
        archive, artifact_filter=TRACE_ARTIFACT
    )] == [RUN_PATH]
    assert discover_provider_chaser_candidate_options(
        archive, artifact_filter="missing_png"
    ) == []


def test_provider_candidate_exposes_only_static_artifacts_and_provenance(tmp_path: Path) -> None:
    archive = _materialized_candidate(tmp_path)
    option = discover_provider_chaser_candidate_options(archive)[0]

    assert available_provider_chaser_candidate_analysis_ids(archive, option) == (
        "static_artifacts",
        "provenance",
    )
    assert option.spec["static_artifacts"]
    assert option.spec["static_artifacts"][HISTOGRAM_LABEL].startswith("visualizations/")
    assert option.spec["static_artifacts"][TRACE_LABEL].startswith("visualizations/")


class _FakeMarimo:
    @staticmethod
    def md(value: str) -> tuple[str, str]:
        return ("md", value)

    @staticmethod
    def vstack(values: list[Any]) -> tuple[str, list[Any]]:
        return ("vstack", values)


def test_provider_candidate_static_pngs_render_through_shared_renderer(tmp_path: Path) -> None:
    archive = _materialized_candidate(tmp_path)
    option = discover_provider_chaser_candidate_options(archive)[0]

    views, errors = build_static_artifact_views(
        _FakeMarimo,
        zarr_path=archive,
        run_path=option.run_path,
        spec=option.spec,
    )

    assert errors == {}
    assert set(views) == {HISTOGRAM_LABEL, TRACE_LABEL}
    assert all(view[0] == "vstack" for view in views.values())
    assert views[HISTOGRAM_LABEL][1][0][1] == f"`{HISTOGRAM_ARTIFACT}`"
    assert views[TRACE_LABEL][1][0][1] == f"`{TRACE_ARTIFACT}`"
    assert views[HISTOGRAM_LABEL][1][1][1].startswith(
        f'<img alt="{HISTOGRAM_LABEL}" src="data:image/png;base64,'
    )
    assert views[TRACE_LABEL][1][1][1].startswith(
        f'<img alt="{TRACE_LABEL}" src="data:image/png;base64,'
    )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda run: run.attrs.__setitem__(
                "provider_chaser_distance_candidate_manifest", {"malformed": True}
            ),
            id="malformed-manifest",
        ),
        pytest.param(
            lambda run: run["distances/distance_mm"].__setitem__((0, 0), 999.0),
            id="tampered-array",
        ),
        pytest.param(
            lambda run: run.attrs.__setitem__("stage_selector_eligible", True),
            id="selector-eligible",
        ),
    ],
)
def test_provider_candidate_discovery_hides_invalid_or_promoted_runs(
    tmp_path: Path,
    mutation: Callable[[Any], None],
) -> None:
    archive = _materialized_candidate(tmp_path)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    mutation(root[RUN_PATH])
    _reconsolidate(archive)

    assert discover_provider_chaser_candidate_options(archive) == []
