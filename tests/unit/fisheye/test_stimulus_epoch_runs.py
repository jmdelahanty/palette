from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.stimulus_epoch_runs import (
    StimulusEpochResult,
    StimulusEpochWindow,
    build_arg_parser,
    write_stimulus_epoch_run,
)


def _result(path: Path) -> StimulusEpochResult:
    window = StimulusEpochWindow(
        window_id=0,
        label="pre_event",
        start_frame=0,
        end_frame=9,
        start_time_s=0.0,
        end_time_s=1.0,
        duration_s=1.0,
        source_start_event_name="PRE_START",
        source_end_event_name="PRE_END",
        source_start_event_frame=0,
        source_end_event_frame=10,
        source_policy="inclusive_start_exclusive_end_event_boundary",
    )
    return StimulusEpochResult(
        zarr_path=str(path),
        recording_id="recording_1",
        run_name="legacy_ineligible",
        stimulus_run_name="stimulus_1",
        stimulus_path="analysis/stimulus_runs/stimulus_1",
        fps=10.0,
        total_frames=10,
        windows=(window,),
        protocol_profile_id="test_profile",
        protocol_profile_version=1,
        protocol_profile_sha256="a" * 64,
        protocol_profile_source="test_profile.yaml",
        source_adapter_id="test_adapter",
        source_adapter_version=1,
        role_resolver_id="test_roles",
        role_resolver_version=1,
        window_policy_id="test_windows",
        window_policy_version=1,
    )


def test_selector_ineligible_v1_write_preserves_parent_selectors(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    parent = root.require_group("analysis").require_group("stimulus_epoch_runs")
    parent.attrs.update(
        {
            "palette_completion_epoch": 2,
            "latest": "existing",
            "latest_complete": "existing_complete",
            "latest_pending": "existing_pending",
        }
    )
    before = dict(parent.attrs)

    path = write_stimulus_epoch_run(
        archive,
        _result(archive),
        selector_ineligible=True,
    )

    reopened = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert path == "analysis/stimulus_epoch_runs/legacy_ineligible"
    assert dict(reopened["analysis/stimulus_epoch_runs"].attrs) == before
    run = reopened["analysis/stimulus_epoch_runs/legacy_ineligible"]
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["parameters"]["publication_policy"] == (
        "selector_ineligible_nonpromoting_v1"
    )


def test_v1_writer_cli_exposes_selector_ineligible_flag() -> None:
    args = build_arg_parser().parse_args(
        ["archive.zarr", "--selector-ineligible"]
    )

    assert args.selector_ineligible is True
