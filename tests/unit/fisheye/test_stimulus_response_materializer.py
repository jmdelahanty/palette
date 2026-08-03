from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.stimulus_response import (
    ProtocolStep,
    _write_stimulus_response_compact_v3,
)
from fisheye.analysis.stimulus_response_storage import (
    STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
    STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR,
    STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis_workflows.materializers import stimulus_response as mod
from fisheye.shared.stimulus_coordinate_contract import canonical_mapping_digest
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr.stimulus_response_schema import (
    STIMULUS_RESPONSE_LAYOUT,
    STIMULUS_RESPONSE_SCHEMA_VERSION,
)


def _build_source(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "stimulus-materializer-fixture"


def _fake_writer(argv) -> None:
    output = Path(argv[argv.index("--output-zarr-path") + 1])
    run_name = argv[argv.index("--run-name") + 1]
    layout = (
        argv[argv.index("--layout") + 1]
        if "--layout" in argv
        else "compact_tabular_v2"
    )
    storage_profile_id = (
        argv[argv.index("--storage-profile") + 1]
        if "--storage-profile" in argv
        else None
    )
    if storage_profile_id is not None:
        assert "--no-write-zarr-artifacts" in argv
    root = zarr.open_group(str(output), mode="a", zarr_format=3)
    run = (
        root.require_group("analysis")
        .require_group("stimulus_response_runs")
        .create_group(run_name)
    )
    coordinate_lineage = {
        "schema_id": "palette.stimulus_response.coordinate_lineage",
        "schema_version": 1,
        "source_stimulus_run_ref": "/analysis/stimulus_runs/stim_1",
    }
    coordinate_lineage["record_sha256"] = canonical_mapping_digest(
        coordinate_lineage
    )
    run.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "schema_id": "palette.stimulus_response",
            "schema_version": (
                STIMULUS_RESPONSE_SCHEMA_VERSION
                if layout == STIMULUS_RESPONSE_LAYOUT
                else 2
            ),
            "method": "stimulus_response",
            "method_version": "stimulus_response.v3",
            "row_axis": "stimulus_steps",
            "layout": layout,
            "parameters": {"moving_threshold_mm_s": 2.0},
            "source_refs": {
                "source_track_kinematics_run": (
                    "analysis/track_kinematics_runs/offline/track_1"
                ),
                "source_stimulus_run": "analysis/stimulus_runs/stim_1",
                "stimulus_coordinate_lineage": coordinate_lineage,
            },
            "n_steps": 1,
            "n_fish": 1,
            "stage_selector_eligible": layout != STIMULUS_RESPONSE_LAYOUT,
            "provenance": {
                "stage": "stimulus_response",
                "parameters": {},
                "inputs": {},
            },
        }
    )
    if layout == STIMULUS_RESPONSE_LAYOUT:
        _write_stimulus_response_compact_v3(
            run,
            global_metrics={
                "fish_id": np.asarray([0], dtype=np.int32),
                "total_distance_mm": np.asarray([1.0], dtype=np.float32),
                "mean_speed_mm_s": np.asarray([1.0], dtype=np.float32),
                "total_active_s": np.asarray([1.0], dtype=np.float32),
                "fraction_moving": np.asarray([1.0], dtype=np.float32),
            },
            steps=[ProtocolStep(0, "baseline", "SOLID_BLACK", 4, 0, 10, 1.0)],
            step_metrics=[{
                "fish_id": np.asarray([0], dtype=np.int32),
                "total_distance_mm": np.asarray([1.0], dtype=np.float32),
                "mean_speed_mm_s": np.asarray([1.0], dtype=np.float32),
                "median_speed_mm_s": np.asarray([1.0], dtype=np.float32),
                "max_speed_mm_s": np.asarray([1.0], dtype=np.float32),
                "fraction_moving": np.asarray([1.0], dtype=np.float32),
                "coverage": np.asarray([1.0], dtype=np.float32),
            }],
            frame_annotations=None,
            step_bout_metrics=None,
            step_grating_data=None,
            step_concentric_data=None,
            step_loom_data=None,
            global_omr_metrics=None,
            storage_profile=(
                get_storage_profile(storage_profile_id)
                if storage_profile_id is not None
                else None
            ),
        )
    else:
        run.create_group("step_index")
        run.create_group("global_per_fish")


def test_plan_is_read_only(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)
    scratch = tmp_path / "scratch"

    result = mod.materialize_stimulus_response(
        source,
        scratch_root=scratch,
        run_name="response_1",
        apply=False,
    )

    assert result["status"] == "planned"
    assert result["mutates_archive"] is False
    assert not scratch.exists()


def test_plan_rejects_forwarded_layout_override(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)

    with pytest.raises(
        ValueError,
        match="materializer owns these writer arguments: --layout",
    ):
        mod.build_stimulus_response_materialization_plan(
            source,
            scratch_root=tmp_path / "scratch",
            run_name="response_1",
            layout=STIMULUS_RESPONSE_LAYOUT,
            writer_arguments=("--layout=compact_tabular_v2",),
        )


def test_plan_rejects_invalid_storage_profile_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)

    with pytest.raises(ValueError, match="requires compact-tabular-v3"):
        mod.build_stimulus_response_materialization_plan(
            source,
            scratch_root=tmp_path / "scratch-v2",
            run_name="response_1",
            storage_profile_id=STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
        )
    with pytest.raises(ValueError, match="Unsupported stimulus-response"):
        mod.build_stimulus_response_materialization_plan(
            source,
            scratch_root=tmp_path / "scratch-profile",
            run_name="response_1",
            layout=STIMULUS_RESPONSE_LAYOUT,
            storage_profile_id="unknown_profile",
        )
    with pytest.raises(
        ValueError,
        match="materializer owns these writer arguments: --storage-profile",
    ):
        mod.build_stimulus_response_materialization_plan(
            source,
            scratch_root=tmp_path / "scratch-forwarded",
            run_name="response_1",
            layout=STIMULUS_RESPONSE_LAYOUT,
            storage_profile_id=STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
            writer_arguments=("--storage-profile=published_http_v1",),
        )


def test_candidate_cli_requires_closed_no_artifact_bundle() -> None:
    with pytest.raises(SystemExit):
        mod.response_writer.main(
            [
                "missing.zarr",
                "--layout",
                STIMULUS_RESPONSE_LAYOUT,
                "--storage-profile",
                STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
            ]
        )


def test_materializer_computes_locally_and_publishes_atomically(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)
    monkeypatch.setattr(mod.response_writer, "main", _fake_writer)

    result = mod.materialize_stimulus_response(
        source,
        scratch_root=tmp_path / "scratch",
        run_name="response_1",
        writer_arguments=("--track-kinematics-run", "track_1"),
        copy_backend="python",
        apply=True,
        keep_scratch=True,
    )

    assert result["status"] == "complete"
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/stimulus_response_runs"]
    assert parent.attrs["latest"] == "response_1"
    assert parent.attrs["latest_complete"] == "response_1"
    run = parent["response_1"]
    assert run.attrs["stage_selector_eligible"] is True
    assert run.attrs["atomic_publication_owner_uuid"]
    assert "atomic_publication_tombstone" not in run.attrs
    assert run.attrs["cluster_output_staging"]["publisher_contract"] == {
        "schema_id": "palette.atomic_run_group_publisher",
        "schema_version": 1,
    }


def test_v3_materializer_publishes_selector_ineligible_without_pointers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)
    monkeypatch.setattr(mod.response_writer, "main", _fake_writer)

    result = mod.materialize_stimulus_response(
        source,
        scratch_root=tmp_path / "scratch",
        run_name="response_v3",
        layout=STIMULUS_RESPONSE_LAYOUT,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
    )

    assert result["status"] == "complete"
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/stimulus_response_runs"]
    assert "latest" not in parent.attrs
    assert "latest_complete" not in parent.attrs
    run = parent["response_v3"]
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["palette_run_completion_status"] == "complete"


def test_candidate_materializer_consolidates_without_selector_activation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)
    mutable_root = zarr.open_group(str(source), mode="a", zarr_format=3)
    existing_parent = mutable_root.require_group("analysis").require_group(
        "stimulus_response_runs"
    )
    existing_parent.create_group("existing")
    existing_parent.attrs.update(
        {"latest": "existing", "latest_complete": "existing"}
    )
    monkeypatch.setattr(mod.response_writer, "main", _fake_writer)

    result = mod.materialize_stimulus_response(
        source,
        scratch_root=tmp_path / "scratch",
        run_name="response_candidate",
        layout=STIMULUS_RESPONSE_LAYOUT,
        storage_profile_id=STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
    )

    assert result["status"] == "complete"
    direct_root = zarr.open_group(
        str(source),
        mode="r",
        use_consolidated=False,
    )
    consolidated_root = zarr.open_group(
        str(source),
        mode="r",
        use_consolidated=True,
    )
    direct_parent = direct_root["analysis/stimulus_response_runs"]
    assert direct_parent.attrs["latest"] == "existing"
    assert direct_parent.attrs["latest_complete"] == "existing"
    direct_run = direct_parent["response_candidate"]
    consolidated_run = consolidated_root[
        "analysis/stimulus_response_runs/response_candidate"
    ]
    assert direct_run.attrs["stage_selector_eligible"] is False
    assert direct_run.attrs["palette_run_completion_status"] == "complete"
    assert direct_run.attrs["analysis_storage_profile_id"] == (
        STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID
    )
    assert STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR in direct_run.attrs
    assert direct_run.attrs[STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR] == (
        consolidated_run.attrs[STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR]
    )


def test_validator_rejects_stale_stimulus_coordinate_lineage(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.zarr"
    _fake_writer(
        [
            "--output-zarr-path",
            str(output),
            "--run-name",
            "response_1",
        ]
    )
    run = zarr.open_group(
        str(output / "analysis" / "stimulus_response_runs" / "response_1"),
        mode="a",
        use_consolidated=False,
    )
    source_refs = dict(run.attrs["source_refs"])
    coordinate_lineage = dict(source_refs["stimulus_coordinate_lineage"])
    coordinate_lineage["source_stimulus_run_ref"] = (
        "/analysis/stimulus_runs/tampered"
    )
    source_refs["stimulus_coordinate_lineage"] = coordinate_lineage
    run.attrs["source_refs"] = source_refs

    result = mod._validate_stimulus_response_run(
        output / "analysis" / "stimulus_response_runs" / "response_1"
    )

    assert result["valid"] is False
    assert "stale stimulus_coordinate_lineage digest" in result["errors"]
