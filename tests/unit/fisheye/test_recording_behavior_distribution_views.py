from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apps.marimo.components.recording_behavior_distributions import (
    discover_recording_behavior_distribution_options,
    load_recording_behavior_distribution_handle,
    recording_behavior_distribution_figure,
)
from fisheye.analysis_workflows.recording_behavior_distribution_storage import (
    load_recording_behavior_distribution_source_handle,
    write_recording_behavior_distribution_run,
)
from fisheye.group_statistics.recording_behavior_distribution_specs import (
    DEFAULT_RECORDING_DISTRIBUTION_METRICS,
)
from fisheye.group_statistics.recording_behavior_distribution_views import (
    RecordingBehaviorDistributionViewError,
    available_recording_distribution_scopes,
    build_recording_behavior_distribution_view,
)
from fisheye.group_statistics.recording_behavior_distributions import (
    RecordingBehaviorDistributionConfig,
    RecordingDistributionMetricInput,
    compute_recording_behavior_distributions,
)
from fisheye.group_statistics.recording_distribution_scopes import (
    exact_source_membership_masks,
    frame_interval_scope,
    whole_session_scope,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.visualization.recording_behavior_distributions import (
    render_recording_behavior_distribution_figure,
)


def _write_fixture(archive: Path, *, run_name: str = "recording-distributions-v1"):
    scopes = (
        whole_session_scope(),
        frame_interval_scope(
            scope_id="custom_window",
            scope_label="Custom 5–10 s",
            scope_family="named_fixture",
            scope_provider_id="fixture.named.v1",
            order=1,
            start_frame=10,
            end_frame_exclusive=20,
            source_binding={"interval_sha256": "1" * 64},
        ),
        frame_interval_scope(
            scope_id="empty_window",
            scope_label="Empty control",
            scope_family="named_fixture",
            scope_provider_id="fixture.named.v1",
            order=2,
            start_frame=20,
            end_frame_exclusive=30,
            source_binding={"interval_sha256": "2" * 64},
        ),
    )
    spec = next(
        item
        for item in DEFAULT_RECORDING_DISTRIBUTION_METRICS
        if item.metric_id == "bout.duration_s"
    )
    values = np.asarray([0.01, 0.09], dtype=np.float64)
    identity = {
        "source_run_path": np.asarray(["analysis/bouts/exact"] * 2, dtype=object),
        "source_manifest_sha256": np.asarray(["3" * 64] * 2, dtype=object),
    }
    result = compute_recording_behavior_distributions(
        RecordingBehaviorDistributionConfig(
            distribution_run_id=run_name,
            recording_id="recording-1",
            scopes=scopes,
            source_record={"bundle_sha256": "4" * 64},
        ),
        (
            RecordingDistributionMetricInput(
                spec=spec,
                values=values,
                valid=np.ones(2, dtype=bool),
                scope_projection=exact_source_membership_masks(
                    scopes, source_scope_id=["custom_window", None]
                ),
                source_identity_arrays=identity,
                source_identity_fallback={
                    "source_run_path": "analysis/bouts/exact",
                    "source_manifest_sha256": "3" * 64,
                },
                valid_duration_s_by_scope={
                    "whole_session": 10.0,
                    "custom_window": 5.0,
                    "empty_window": 5.0,
                },
            ),
        ),
    )
    write_recording_behavior_distribution_run(
        archive,
        run_name=run_name,
        result=result,
        run_provenance=build_writer_run_provenance(
            command="recording-distribution-view-test",
            params={"run_name": run_name},
            cwd=Path.cwd(),
        ),
    )
    return load_recording_behavior_distribution_source_handle(
        archive,
        run_name=run_name,
        expected_recording_id="recording-1",
    )


def test_view_restores_structural_zeros_and_dynamic_scope_labels(
    tmp_path: Path,
) -> None:
    handle = _write_fixture(tmp_path / "analysis.zarr")

    assert [
        row["scope_label"] for row in available_recording_distribution_scopes(handle)
    ] == ["Whole session", "Custom 5–10 s", "Empty control"]
    view = build_recording_behavior_distribution_view(
        handle,
        metric_id="bout.duration_s",
        weighting_id="event",
    )

    assert [series.scope_id for series in view.series] == [
        "whole_session",
        "custom_window",
        "empty_window",
    ]
    whole, custom, empty = view.series
    assert whole.grid_index.tolist() == [0, 1, 2, 3, 4]
    assert whole.bin_count.tolist() == [1, 0, 0, 0, 1]
    assert custom.bin_count.tolist() == [1, 0, 0, 0, 0]
    assert custom.fraction.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert np.all(np.isnan(empty.fraction))
    assert empty.support["support_state"] == "zero_denominator"
    assert view.manifest_sha256 != handle.manifest["payload_digest"]
    with pytest.raises(ValueError, match="read-only"):
        whole.bin_count[0] = 9


def test_view_scope_selection_fails_closed_for_unknown_scope(tmp_path: Path) -> None:
    handle = _write_fixture(tmp_path / "analysis.zarr")

    with pytest.raises(KeyError, match="Unknown recording distribution scopes"):
        build_recording_behavior_distribution_view(
            handle,
            metric_id="bout.duration_s",
            weighting_id="event",
            scope_ids=("invented_epoch",),
        )
    with pytest.raises(ValueError, match="weighting"):
        build_recording_behavior_distribution_view(
            handle,
            metric_id="bout.duration_s",
            weighting_id="frame",
        )


def test_static_and_interactive_renderers_consume_the_same_view(tmp_path: Path) -> None:
    handle = _write_fixture(tmp_path / "analysis.zarr")
    view = build_recording_behavior_distribution_view(
        handle,
        metric_id="bout.duration_s",
        weighting_id="event",
    )

    interactive = recording_behavior_distribution_figure(view)
    static = render_recording_behavior_distribution_figure(view)

    assert interactive.layout.meta["view_sha256"] == view.view_sha256
    assert interactive.layout.meta["viewer_rebinning"] == "prohibited"
    assert [annotation.text for annotation in interactive.layout.annotations].count(
        "No valid evidence"
    ) == 1
    assert static._suptitle.get_text().startswith("Canonical swim-bout duration")
    assert [axis.get_title() for axis in static.axes] == [
        "Whole session",
        "Custom 5–10 s",
        "Empty control",
    ]


def test_discovery_is_consolidated_and_selection_revalidates_exact_run(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "analysis.zarr"
    handle = _write_fixture(archive)

    options = discover_recording_behavior_distribution_options(archive)

    assert len(options) == 1
    assert discover_recording_behavior_distribution_options(
        archive, run_path_filter=options[0].run_path
    ) == options
    assert discover_recording_behavior_distribution_options(
        archive, run_path_filter="analysis/recording_behavior_distribution_runs/other"
    ) == ()
    selected = load_recording_behavior_distribution_handle(archive, options[0])
    assert selected.verification_digest == handle.verification_digest


def test_view_rejects_wrong_handle_type() -> None:
    with pytest.raises(
        RecordingBehaviorDistributionViewError,
        match="validated recording-distribution handle",
    ):
        available_recording_distribution_scopes(object())  # type: ignore[arg-type]
