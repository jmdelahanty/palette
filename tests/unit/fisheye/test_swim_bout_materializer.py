from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis import detect_bouts_multi_level as bout_writer
from fisheye.analysis.swim_bout_frame_axis import build_frame_axis_contract
from fisheye.analysis import swim_bout_schema
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
    track.create_array(
        "source_acquisition_frame_index",
        data=frames,
        chunks=(6,),
    )
    return frames


def _fake_writer(argv) -> int:
    source = Path(argv[0])
    output = Path(argv[argv.index("--output-zarr-path") + 1])
    run_name = argv[argv.index("--run-name") + 1]
    source_root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    frames = np.asarray(
        source_root[
            "analysis/track_kinematics_runs/offline/track_1/tracks/id_0/"
            "source_acquisition_frame_index"
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
            "source_track_motion_manifest_sha256": "a" * 64,
            "track_id": 0,
            "default_signal_id": 4,
            "frame_axis_contract": build_frame_axis_contract(
                frames,
                authoritative_path=(
                    "analysis/track_kinematics_runs/offline/track_1/"
                    "tracks/id_0/source_acquisition_frame_index"
                ),
                source_track_kinematics_run="track_1",
                track_id=0,
                source_track_motion_manifest_sha256="a" * 64,
            ),
            "provenance": {
                "stage": "detect_bouts_multi_level",
                "parameters": {},
                "inputs": {},
            },
        }
    )
    required = swim_bout_schema._required_specs()
    for spec in required.values():
        parent = run
        parts = spec.path.split("/")
        for part in parts[:-1]:
            parent = parent.require_group(part)
        if spec.path == "signals/detector_signal_mm_s":
            data = np.zeros((1, frames.size), dtype=np.float32)
        elif spec.path == "signals/detector_signal_signal_ids":
            data = np.asarray([4], dtype=np.int32)
        else:
            shape = (1,) if len(spec.axes) == 1 else (1, 64)
            data = np.zeros(shape, dtype=np.dtype(spec.dtype))
        parent.create_array(parts[-1], data=data)
    for table_path in swim_bout_schema._COLUMNAR_TABLE_PATHS:
        prefix = table_path + "/"
        fields = [
            (path[len(prefix) :], spec)
            for path, spec in required.items()
            if path.startswith(prefix) and "/" not in path[len(prefix) :]
        ]
        table = run[table_path]
        table.attrs["storage_layout"] = "columnar"
        table.attrs["field_names"] = [name for name, _spec in fields]
        table.attrs["field_dtypes"] = {
            name: spec.logical_dtype for name, spec in fields
        }
    swim_bout_schema.write_swim_bout_array_manifest(run)
    return 0


def _fake_all_nan_writer(argv) -> int:
    result = _fake_writer(argv)
    output = Path(argv[argv.index("--output-zarr-path") + 1])
    run_name = argv[argv.index("--run-name") + 1]
    root = zarr.open_group(str(output), mode="a", use_consolidated=False)
    root[f"analysis/swim_bout_runs/{run_name}/signals/detector_signal_mm_s"][:] = np.nan
    return result


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


def test_writer_requires_finite_default_detector_but_permits_zero_activity() -> None:
    assert (
        bout_writer._require_finite_default_detector_signal(
            np.asarray([0.0, np.nan, 0.0], dtype=np.float32),
            default_level="speed_exponential",
            track_kinematics_run="track_1",
            track_id=0,
        )
        == 2
    )

    with pytest.raises(
        ValueError,
        match="Default swim-bout detector signal has no finite physical samples",
    ):
        bout_writer._require_finite_default_detector_signal(
            np.asarray([np.nan, np.nan], dtype=np.float32),
            default_level="speed_exponential",
            track_kinematics_run="track_uncalibrated",
            track_id=0,
        )


def test_materializer_computes_locally_and_publishes_atomically(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    frames = _build_source(source)
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
    assert result["publish"]["physical_copy"]["verification"] == (
        "sha256_all_physical_files"
    )
    assert result["publish"]["physical_copy"]["content_sha256"]
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/swim_bout_runs"]
    assert parent.attrs["latest"] == "bouts_1"
    assert parent.attrs["latest_complete"] == "bouts_1"
    run = parent["bouts_1"]
    assert run.attrs["stage_selector_eligible"] is True
    assert run.attrs["atomic_publication_owner_uuid"]
    assert "atomic_publication_tombstone" not in run.attrs
    assert run.attrs["cluster_output_staging"]["publisher_contract"] == {
        "schema_id": "palette.atomic_run_group_publisher",
        "schema_version": 1,
    }
    validation = result["local_materialization"]["local_validation"]
    assert validation["default_signal_id"] == 4
    assert validation["default_detector_row"] == 0
    assert validation["default_detector_finite_count"] == frames.size


def test_materializer_rejects_all_nan_default_detector_before_publish(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)
    monkeypatch.setattr(mod.bout_writer, "main", _fake_all_nan_writer)

    with np.testing.assert_raises_regex(
        RuntimeError,
        "default detector signal has no finite physical samples",
    ):
        mod.materialize_swim_bouts(
            source,
            scratch_root=tmp_path / "scratch",
            run_name="bouts_all_nan",
            writer_arguments=("--layout", "compact_v2"),
            copy_backend="python",
            apply=True,
            keep_scratch=True,
        )

    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert root.get("analysis/swim_bout_runs/bouts_all_nan") is None
