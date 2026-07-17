from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis import detect_bouts_multi_level as bout_writer
from fisheye.analysis.swim_bout_frame_axis import build_frame_axis_contract
from fisheye.analysis_workflows.materializers import swim_bouts as mod


def _build_source(path: Path) -> np.ndarray:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    frames = np.arange(12, dtype=np.int64)
    track = (
        root.require_group("analysis")
        .require_group("track_kinematics_runs")
        .require_group("offline")
        .require_group("track_1")
        .require_group("tracks")
        .require_group("id_0")
    )
    track.create_array("frame_indices", data=frames, chunks=(6,))
    return frames


def _fake_writer(argv) -> int:
    source = Path(argv[0])
    output = Path(argv[argv.index("--output-zarr-path") + 1])
    run_name = argv[argv.index("--run-name") + 1]
    source_root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    frames = np.asarray(
        source_root[
            "analysis/track_kinematics_runs/offline/track_1/tracks/id_0/frame_indices"
        ][:],
        dtype=np.int64,
    )
    root = zarr.open_group(str(output), mode="a", zarr_format=3)
    run = (
        root.require_group("analysis")
        .require_group("swim_bout_runs")
        .create_group(run_name)
    )
    run.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "schema_id": "palette.swim_bout_runs",
            "schema_version": 8,
            "layout": "compact_tabular_v2",
            "method": "peak_event",
            "method_version": bout_writer.METHOD_VERSION,
            "row_axis": "swim_bout_rows",
            "parameters": {"method": "peak_event"},
            "source_refs": {
                "source_track_kinematics_path": (
                    "analysis/track_kinematics_runs/offline/track_1/tracks/id_0"
                )
            },
            "source_track_kinematics_run": "track_1",
            "track_id": 0,
            "frame_axis_contract": build_frame_axis_contract(
                frames,
                authoritative_path=(
                    "analysis/track_kinematics_runs/offline/track_1/"
                    "tracks/id_0/frame_indices"
                ),
                source_track_kinematics_run="track_1",
                track_id=0,
            ),
            "provenance": {
                "stage": "detect_bouts_multi_level",
                "parameters": {},
                "inputs": {},
            },
        }
    )
    indexes = run.create_group("indexes")
    indexes.create_group("candidates")
    indexes.create_group("signal_variants")
    run.create_group("tables").create_group("bouts")
    signals = run.create_group("signals")
    signals.create_array(
        "detector_signal_mm_s",
        data=np.zeros((1, frames.size), dtype=np.float32),
        chunks=(1, frames.size),
    )
    return 0


def test_plan_is_read_only(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)
    scratch = tmp_path / "scratch"

    result = mod.materialize_swim_bouts(
        source,
        scratch_root=scratch,
        run_name="bouts_1",
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
    monkeypatch.setattr(mod.bout_writer, "main", _fake_writer)

    result = mod.materialize_swim_bouts(
        source,
        scratch_root=tmp_path / "scratch",
        run_name="bouts_1",
        writer_arguments=("--layout", "compact_v2"),
        copy_backend="python",
        apply=True,
        keep_scratch=True,
    )

    assert result["status"] == "complete"
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/swim_bout_runs"]
    assert parent.attrs["latest"] == "bouts_1"
    assert parent.attrs["latest_complete"] == "bouts_1"
    run = parent["bouts_1"]
    assert run.attrs["cluster_output_staging"]["publisher_contract"] == {
        "schema_id": "palette.atomic_run_group_publisher",
        "schema_version": 1,
    }
