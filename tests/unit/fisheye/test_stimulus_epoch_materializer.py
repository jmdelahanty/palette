from __future__ import annotations

from pathlib import Path

import pytest
import zarr

from fisheye.analysis_workflows.materializers import stimulus_epochs as materializer
from fisheye.analysis_workflows.materializers.stimulus_epochs import (
    build_stimulus_epoch_candidate_plan,
    materialize_stimulus_epoch_candidate,
)

from .test_stimulus_epoch_schema import create_legacy_stimulus_epoch_archive


def test_v2_materializer_requires_explicit_opt_in_for_ineligible_v1_source(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    root = create_legacy_stimulus_epoch_archive(archive)
    source = root["analysis/stimulus_epoch_runs/source"]
    source.attrs["stage_selector_eligible"] = False

    with pytest.raises(ValueError, match="explicitly selector-ineligible"):
        build_stimulus_epoch_candidate_plan(
            archive,
            source_run="source",
            run_name="candidate_without_opt_in",
            scratch_root=tmp_path / "scratch-without-opt-in",
        )

    parent = root["analysis/stimulus_epoch_runs"]
    selectors_before = dict(parent.attrs)
    result = materialize_stimulus_epoch_candidate(
        archive,
        source_run="source",
        run_name="candidate_with_opt_in",
        scratch_root=tmp_path / "scratch-with-opt-in",
        copy_backend="python",
        allow_selector_ineligible_source=True,
        apply=True,
    )

    assert result["status"] == "complete"
    assert result["allow_selector_ineligible_source"] is True
    assert result["source_lifecycle"] == {
        "completion_status": "complete",
        "stage_selector_eligible": False,
        "stage_selector_marker": "explicit_false",
        "selection_policy": (
            "exact_named_complete_legacy_v1_selector_ineligible_opt_in"
        ),
        "allow_selector_ineligible_source": True,
    }
    assert result["plan"]["source_lifecycle"] == result["source_lifecycle"]

    direct = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    assert dict(direct["analysis/stimulus_epoch_runs"].attrs) == selectors_before
    candidate = direct["analysis/stimulus_epoch_runs/candidate_with_opt_in"]
    assert candidate.attrs["source_stimulus_epoch_lifecycle"] == result[
        "source_lifecycle"
    ]
    assert (
        candidate.attrs["stimulus_epoch_run_manifest"]["payload"]["source_epoch"][
            "lifecycle"
        ]
        == result["source_lifecycle"]
    )
    assert (
        consolidated["analysis/stimulus_epoch_runs/candidate_with_opt_in"].attrs[
            "stimulus_epoch_run_manifest"
        ]
        == candidate.attrs["stimulus_epoch_run_manifest"]
    )


def test_v2_materializer_cli_exposes_selector_ineligible_source_flag() -> None:
    args = materializer._build_parser().parse_args(
        [
            "archive.zarr",
            "--source-run",
            "source",
            "--run-name",
            "candidate",
            "--allow-selector-ineligible-source",
        ]
    )

    assert args.allow_selector_ineligible_source is True
