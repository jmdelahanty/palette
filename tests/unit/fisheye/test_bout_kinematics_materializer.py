from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis import bout_kinematics as bout_writer_module
from fisheye.analysis.bout_kinematics import LAYOUT_COMPACT_TABULAR_V2
from fisheye.analysis.bout_kinematics_schema import (
    write_bout_kinematics_array_manifest,
)
from fisheye.analysis_workflows.materializers.bout_kinematics import (
    build_bout_kinematics_compute_plan,
    build_bout_kinematics_storage_plan,
    materialize_bout_kinematics_compute,
    materialize_bout_kinematics_storage,
    promote_bout_kinematics_candidate,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.columnar import write_columnar_dataset
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.shared.zarr_sharded_copy import SHARD_POLICY_MULTI_CHUNK_CAPPED


def _make_archive(tmp_path: Path) -> Path:
    path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    parent = require_runs_parent(
        root.require_group("analysis"),
        "bout_kinematics_runs",
    )
    run = parent.create_group("bout_source")
    mark_run_started(run, run_name="bout_source", stage="bout_kinematics")
    run.attrs.update(
        {
            "status": "complete",
            "schema_id": "analysis.bout_kinematics_runs",
            "schema_version": 7,
            "method": "linked_per_bout_heading_kinematics",
            "method_version": "bout_kinematics.v7",
            "layout": LAYOUT_COMPACT_TABULAR_V2,
            "heading_levels": ["heading_smoothed"],
            "analysis_levels": ["movement", "heading_smoothed"],
            "default_heading_level": "heading_smoothed",
            "parameters": {"fixture": True},
            "source_refs": {"source_swim_bout_run": "bouts_fixture"},
        }
    )
    rows = 5_000
    movement = np.zeros(
        rows,
        dtype=[
            ("bout_id", "i4"),
            ("analysis_level_id", "i2"),
            ("analysis_level_bytes", "S16"),
            ("physical_active_duration_s", "f4"),
        ],
    )
    movement["bout_id"] = np.arange(rows, dtype=np.int32)
    movement["analysis_level_bytes"] = b"movement"
    movement["physical_active_duration_s"] = np.linspace(0.0, 1.0, rows)
    write_columnar_dataset(
        run,
        "movement_metrics",
        movement,
        {"schema_id": "fixture.movement"},
        shard_rows=None,
    )

    heading = np.zeros(
        rows,
        dtype=[
            ("bout_id", "i4"),
            ("analysis_level_id", "i2"),
            ("analysis_level_bytes", "S32"),
            ("heading_level_id", "i2"),
            ("heading_level_bytes", "S32"),
            ("net_delta_heading_deg", "f4"),
        ],
    )
    heading["bout_id"] = np.arange(rows, dtype=np.int32)
    heading["analysis_level_bytes"] = b"heading_smoothed"
    heading["heading_level_bytes"] = b"heading_smoothed"
    heading["net_delta_heading_deg"] = np.linspace(-20.0, 20.0, rows)
    write_columnar_dataset(
        run,
        "heading_metrics",
        heading,
        {"schema_id": "fixture.heading"},
        shard_rows=None,
    )
    visualizations = run.create_group("visualizations")
    visualizations.create_array(
        "summary_png",
        data=np.arange(100, dtype=np.uint8),
        chunks=(100,),
    )
    provenance = build_writer_run_provenance(
        command="fixture",
        params={"fixture": True},
        input_run_ids={"source": "fixture"},
    )
    run.attrs["run_provenance"] = provenance
    mark_run_complete(
        run,
        parent_group=parent,
        run_name="bout_source",
        run_provenance=provenance,
    )
    return path


def test_bout_kinematics_storage_plan_is_non_mutating(tmp_path: Path) -> None:
    source = _make_archive(tmp_path)
    before = sorted(path.relative_to(source) for path in source.rglob("*"))

    plan = build_bout_kinematics_storage_plan(
        source,
        source_run="latest_complete",
        scratch_root=tmp_path / "scratch",
        run_name="bout_candidate",
    )

    assert plan.source_run_name == "bout_source"
    assert plan.latest_before == "bout_source"
    assert plan.latest_complete_before == "bout_source"
    assert plan.to_json()["promotion_policy"] == (
        "publish_named_candidate_without_pointer_update"
    )
    assert not (tmp_path / "scratch").exists()
    assert before == sorted(path.relative_to(source) for path in source.rglob("*"))


def test_materialize_bout_kinematics_storage_publishes_without_promotion(
    tmp_path: Path,
) -> None:
    source = _make_archive(tmp_path)
    scratch = tmp_path / "scratch"

    result = materialize_bout_kinematics_storage(
        source,
        source_run="bout_source",
        scratch_root=scratch,
        run_name="bout_candidate",
        output_shard_rows=8_192,
        workers=1,
        copy_backend="python",
        apply=True,
    )

    assert result["status"] == "complete"
    assert result["promoted"] is False
    assert result["local_validation"]["valid"] is True
    assert result["local_validation"]["logical_fingerprint"]["logical_sha256"] == (
        result["source_validation"]["logical_fingerprint"]["logical_sha256"]
    )
    assert not scratch.exists()

    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/bout_kinematics_runs"]
    assert parent.attrs["latest"] == "bout_source"
    assert parent.attrs["latest_complete"] == "bout_source"
    candidate = parent["bout_candidate"]
    assert candidate.attrs["palette_run_completion_status"] == "complete"
    assert candidate.attrs["palette_run_name"] == "bout_candidate"
    assert candidate.attrs["stage_selector_eligible"] is False
    assert candidate.attrs["atomic_publication_owner_uuid"]
    assert "atomic_publication_tombstone" not in candidate.attrs
    assert candidate.attrs["cluster_output_staging"]["promotion_policy"] == (
        "named_candidate_only_parent_pointers_unchanged"
    )
    assert candidate.attrs["physical_storage_layout"]["shard_policy"] == (
        SHARD_POLICY_MULTI_CHUNK_CAPPED
    )
    assert candidate["movement_metrics/bout_id"].shards is not None
    assert candidate["heading_metrics/heading_level_bytes"].shards is not None
    assert candidate["visualizations/summary_png"].shards is None


def test_promote_bout_candidate_validates_then_updates_both_pointers(
    tmp_path: Path,
) -> None:
    source = _make_archive(tmp_path)
    materialize_bout_kinematics_storage(
        source,
        source_run="bout_source",
        scratch_root=tmp_path / "storage-scratch",
        run_name="bout_candidate",
        output_shard_rows=8_192,
        workers=1,
        copy_backend="python",
        apply=True,
    )

    planned = promote_bout_kinematics_candidate(
        source,
        run_name="bout_candidate",
        apply=False,
    )
    assert planned["status"] == "planned"
    assert planned["latest_before"] == "bout_source"

    promoted = promote_bout_kinematics_candidate(
        source,
        run_name="bout_candidate",
        apply=True,
        approved_by="test",
        note="validated storage candidate",
    )

    assert promoted["status"] == "complete"
    assert promoted["promoted"] is True
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/bout_kinematics_runs"]
    assert parent.attrs["latest"] == "bout_candidate"
    assert parent.attrs["latest_complete"] == "bout_candidate"
    promoted_candidate = parent["bout_candidate"]
    assert promoted_candidate.attrs["stage_selector_eligible"] is True
    receipt = promoted_candidate.attrs["storage_promotion"]
    assert receipt["approved_by"] == "test"
    assert receipt["previous_latest"] == "bout_source"


def test_compute_plan_is_read_only_and_rejects_managed_writer_arguments(
    tmp_path: Path,
) -> None:
    source = _make_archive(tmp_path)
    before = sorted(path.relative_to(source) for path in source.rglob("*"))

    plan = build_bout_kinematics_compute_plan(
        source,
        scratch_root=tmp_path / "compute-scratch",
        run_name="bout_fresh",
        writer_arguments=("--speed-level", "exponential"),
    )

    assert plan.local_run_path == (
        tmp_path
        / "compute-scratch"
        / "bout-output.zarr"
        / "analysis"
        / "bout_kinematics_runs"
        / "bout_fresh"
    )
    assert not plan.scratch_root.exists()
    assert before == sorted(path.relative_to(source) for path in source.rglob("*"))

    with pytest.raises(ValueError, match="owns these writer arguments"):
        build_bout_kinematics_compute_plan(
            source,
            scratch_root=tmp_path / "other-scratch",
            run_name="bout_other",
            writer_arguments=("--run-name", "wrong"),
        )


def test_compute_materializer_publishes_and_promotes_local_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_archive(tmp_path)

    def fake_writer_main(argv: list[str]) -> int:
        output_zarr = Path(argv[argv.index("--output-zarr-path") + 1])
        run_name = argv[argv.index("--run-name") + 1]
        root = zarr.open_group(str(output_zarr), mode="w", zarr_format=3)
        parent = require_runs_parent(
            root.require_group("analysis"), "bout_kinematics_runs"
        )
        group = parent.create_group(run_name)
        mark_run_started(group, run_name=run_name, stage="bout_kinematics")
        group.attrs.update(
            {
                "status": "complete",
                "schema_id": "analysis.bout_kinematics_runs",
                "schema_version": 7,
                "method": "linked_per_bout_heading_kinematics",
                "method_version": "bout_kinematics.v7",
                "layout": LAYOUT_COMPACT_TABULAR_V2,
                "heading_levels": ["heading_smoothed"],
                "analysis_levels": ["movement", "heading_smoothed"],
                "default_heading_level": "heading_smoothed",
                "parameters": {"fixture": True},
                "source_refs": {"source_swim_bout_run": "bouts_fixture"},
                "physical_storage_layout": {
                    "shard_policy": SHARD_POLICY_MULTI_CHUNK_CAPPED
                },
            }
        )
        rows = 5_000
        movement = np.zeros(rows, dtype=bout_writer_module._movement_metrics_dtype())
        movement["bout_id"] = np.arange(rows, dtype=np.int32)
        heading = np.zeros(rows, dtype=bout_writer_module._metrics_dtype())
        heading["bout_id"] = np.arange(rows, dtype=np.int32)
        bout_writer_module._write_compact_bout_kinematics_tables(
            group,
            movement_metrics=movement,
            movement_attrs={},
            metrics_by_level={"heading_smoothed": heading},
            heading_levels=["heading_smoothed"],
            default_heading_level="heading_smoothed",
            heading_table_attrs={},
            eye_gaze_metrics=None,
            eye_gaze_attrs=None,
            output_shard_rows=8_192,
        )
        write_bout_kinematics_array_manifest(group)
        provenance = build_writer_run_provenance(
            command="fixture",
            params={"fixture": True},
            input_run_ids={"source": "fixture"},
        )
        group.attrs["run_provenance"] = provenance
        mark_run_complete(
            group,
            parent_group=parent,
            run_name=run_name,
            run_provenance=provenance,
        )
        return 0

    monkeypatch.setattr(
        "fisheye.analysis_workflows.materializers.bout_kinematics.bout_writer.main",
        fake_writer_main,
    )
    scratch = tmp_path / "compute-scratch"
    result = materialize_bout_kinematics_compute(
        source,
        scratch_root=scratch,
        run_name="bout_fresh",
        output_shard_rows=8_192,
        writer_arguments=("--speed-level", "exponential"),
        copy_backend="python",
        apply=True,
    )

    assert result["status"] == "complete"
    assert result["promoted"] is True
    assert not scratch.exists()
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/bout_kinematics_runs"]
    assert parent.attrs["latest"] == "bout_fresh"
    assert parent.attrs["latest_complete"] == "bout_fresh"
    fresh = parent["bout_fresh"]
    assert fresh.attrs["stage_selector_eligible"] is True
    assert fresh.attrs["atomic_publication_owner_uuid"]
    assert "atomic_publication_tombstone" not in fresh.attrs
    assert fresh.attrs["node_local_materialization"]["compute_output"] == (
        "node_local_zarr"
    )
    assert fresh.attrs["cluster_output_staging"]["promotion_policy"] == (
        "complete_ineligible_then_pointers_then_eligibility_final"
    )
