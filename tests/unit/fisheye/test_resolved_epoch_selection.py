from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest
import zarr

import fisheye.analysis_workflows.resolved_epoch_selection as resolved_epoch_selection
from fisheye.analysis.stimulus_epoch_consumer import read_stimulus_epoch_snapshot
from fisheye.analysis_workflows.resolved_epoch_selection import (
    HALF_OPEN_FRAME_INTERVAL_CONVENTION,
    RESOLVED_EPOCH_SELECTION_SCHEMA_ID,
    ResolvedEpochSelectionError,
    resolve_exact_stimulus_epoch_selection,
)

pytest_plugins = ("tests.unit.fisheye.test_stimulus_epoch_consumer",)


def _copy_candidate(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def test_exact_v2_run_converts_to_half_open_atomic_intervals(
    published_candidate: Path,
) -> None:
    selection = resolve_exact_stimulus_epoch_selection(
        published_candidate,
        run_name="candidate",
    )

    assert selection.run_path == "analysis/stimulus_epoch_runs/candidate"
    assert selection.run_schema_id == "palette.stimulus_epoch_windows.v2"
    assert selection.run_schema_version == 2
    assert selection.native_frame_count == 30
    assert selection.fps == 10.0
    assert [interval.window_id for interval in selection.intervals] == [0, 1, 2]
    assert [(item.start_frame, item.end_frame) for item in selection.intervals] == [
        (0, 10),
        (10, 20),
        (20, 30),
    ]
    assert all(
        item.to_record()["frame_interval_convention"]
        == HALF_OPEN_FRAME_INTERVAL_CONVENTION
        for item in selection.intervals
    )
    assert selection.intervals[1].occurrence_identity["occurrence_id"] == (
        "analysis/stimulus_epoch_runs/candidate#window:1"
    )
    assert (
        selection.intervals[1].source_metadata_identity["source_start_event_frame"]
        == 10
    )


def test_labels_are_descriptive_and_never_analysis_roles(
    published_candidate: Path,
) -> None:
    selection = resolve_exact_stimulus_epoch_selection(
        published_candidate,
        run_name="candidate",
    )
    record = selection.selection_record

    assert record["schema_id"] == RESOLVED_EPOCH_SELECTION_SCHEMA_ID
    assert [item["label"] for item in record["intervals"]] == [
        "pre_event",
        "training_event",
        "post_event",
    ]
    assert all("analysis_role" not in item for item in record["intervals"])
    assert all("role" not in item for item in record["intervals"])


def test_selection_digest_is_stable_and_bound_to_canonical_record(
    published_candidate: Path,
) -> None:
    first = resolve_exact_stimulus_epoch_selection(
        published_candidate,
        run_name="candidate",
    )
    second = resolve_exact_stimulus_epoch_selection(
        published_candidate,
        run_name="candidate",
    )

    assert first.selection_digest == second.selection_digest
    assert first.selection_record["selection_sha256"] == first.selection_digest
    assert first.to_record(include_digest=False) == second.to_record(
        include_digest=False
    )


@pytest.mark.parametrize("run_name", ["latest", "latest_complete", ""])
def test_adapter_requires_one_explicit_run_name(
    published_candidate: Path,
    run_name: str,
) -> None:
    with pytest.raises(ResolvedEpochSelectionError, match="explicit"):
        resolve_exact_stimulus_epoch_selection(
            published_candidate,
            run_name=run_name,
        )


def test_adapter_rejects_stale_source_timeline_digest(
    tmp_path: Path,
    published_candidate: Path,
) -> None:
    archive = _copy_candidate(
        published_candidate,
        tmp_path / "stale-source-timeline.zarr",
    )
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    root["analysis/stimulus_runs/stimulus_1/events/camera_frame_id"][0] = np.int64(1)

    with pytest.raises(ResolvedEpochSelectionError, match="timeline fingerprint"):
        resolve_exact_stimulus_epoch_selection(archive, run_name="candidate")


def test_adapter_rejects_stale_planned_manifest_digest(
    published_candidate: Path,
) -> None:
    with pytest.raises(ResolvedEpochSelectionError, match="manifest digest"):
        resolve_exact_stimulus_epoch_selection(
            published_candidate,
            run_name="candidate",
            expected_run_manifest_digest="0" * 64,
        )


def test_adapter_preserves_noncontiguous_snapshot_without_inventing_membership(
    published_candidate: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = read_stimulus_epoch_snapshot(
        published_candidate,
        run_name="candidate",
    )
    segments = list(original.segments)
    segments[1] = replace(
        segments[1],
        start_frame=11,
        start_time_s=1.1,
        duration_s=0.9,
    )
    monkeypatch.setattr(
        resolved_epoch_selection,
        "read_stimulus_epoch_snapshot",
        lambda *_args, **_kwargs: replace(original, segments=tuple(segments)),
    )

    selection = resolve_exact_stimulus_epoch_selection(
        published_candidate,
        run_name="candidate",
    )

    assert [(item.start_frame, item.end_frame) for item in selection.intervals] == [
        (0, 10),
        (11, 20),
        (20, 30),
    ]


def test_adapter_rejects_overlap_and_invalid_timing_snapshot(
    published_candidate: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = read_stimulus_epoch_snapshot(
        published_candidate,
        run_name="candidate",
    )
    overlapping = list(original.segments)
    overlapping[1] = replace(
        overlapping[1],
        start_frame=9,
        start_time_s=0.9,
        duration_s=1.1,
    )
    monkeypatch.setattr(
        resolved_epoch_selection,
        "read_stimulus_epoch_snapshot",
        lambda *_args, **_kwargs: replace(
            original,
            segments=tuple(overlapping),
        ),
    )
    with pytest.raises(ResolvedEpochSelectionError, match="overlap"):
        resolve_exact_stimulus_epoch_selection(
            published_candidate,
            run_name="candidate",
        )

    invalid_timing = list(original.segments)
    invalid_timing[0] = replace(invalid_timing[0], start_time_s=float("nan"))
    monkeypatch.setattr(
        resolved_epoch_selection,
        "read_stimulus_epoch_snapshot",
        lambda *_args, **_kwargs: replace(
            original,
            segments=tuple(invalid_timing),
        ),
    )
    with pytest.raises(ResolvedEpochSelectionError, match="non-finite"):
        resolve_exact_stimulus_epoch_selection(
            published_candidate,
            run_name="candidate",
        )
