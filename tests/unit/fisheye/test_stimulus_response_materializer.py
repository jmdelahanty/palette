from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.analysis_workflows.materializers import stimulus_response as mod


def _build_source(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "stimulus-materializer-fixture"


def _fake_writer(argv) -> None:
    output = Path(argv[argv.index("--output-zarr-path") + 1])
    run_name = argv[argv.index("--run-name") + 1]
    root = zarr.open_group(str(output), mode="a", zarr_format=3)
    run = (
        root.require_group("analysis")
        .require_group("stimulus_response_runs")
        .create_group(run_name)
    )
    run.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "schema_id": "palette.stimulus_response",
            "schema_version": 2,
            "method": "stimulus_response",
            "method_version": "stimulus_response.v2",
            "row_axis": "stimulus_steps",
            "layout": "compact_tabular_v2",
            "parameters": {"moving_threshold_mm_s": 2.0},
            "source_refs": {
                "source_track_kinematics_run": (
                    "analysis/track_kinematics_runs/offline/track_1"
                ),
                "source_stimulus_run": "analysis/stimulus_runs/stim_1",
            },
            "n_steps": 1,
            "n_fish": 1,
            "provenance": {
                "stage": "stimulus_response",
                "parameters": {},
                "inputs": {},
            },
        }
    )
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
