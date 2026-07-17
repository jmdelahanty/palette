from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.bout_kinematics import LAYOUT_COMPACT_TABULAR_V2
from fisheye.analysis_workflows.materializers.bout_kinematics import (
    build_bout_kinematics_storage_plan,
    materialize_bout_kinematics_storage,
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
    assert candidate.attrs["cluster_output_staging"]["promotion_policy"] == (
        "named_candidate_only_parent_pointers_unchanged"
    )
    assert candidate.attrs["physical_storage_layout"]["shard_policy"] == (
        SHARD_POLICY_MULTI_CHUNK_CAPPED
    )
    assert candidate["movement_metrics/bout_id"].shards is not None
    assert candidate["heading_metrics/heading_level_bytes"].shards is not None
    assert candidate["visualizations/summary_png"].shards is None
