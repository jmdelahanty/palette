from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.diagnostics.benchmark_analytics_query_exports import (
    _FAMILIES,
    _read_workloads,
    _source_metadata_paths,
    build_request,
    main,
    require_request,
)


def _request(tmp_path: Path, *, family_id: str = "kinematics_samples"):
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    (archive / "zarr.json").write_text("{}\n", encoding="utf-8")
    source_runs: dict[str, object]
    publisher_parameters: dict[str, object]
    if family_id == "eye_trace_samples":
        source_runs = {"eye_angle_run": "eye_v7"}
        publisher_parameters = {"row_group_rows": 65_536}
    elif family_id == "kinematics_samples":
        source_runs = {
            "track_kinematics_run": "track_v1",
            "track_scope": "offline",
        }
        publisher_parameters = {
            "requested_sample_rate_hz": 10.0,
            "source_window_rows": 131_072,
            "row_group_rows": 65_536,
        }
    elif family_id == "activity_spatial_time_bins":
        source_runs = {
            "track_kinematics_run": "track_v1",
            "track_scope": "offline",
            "swim_bout_runs_by_track": {"0": "bout_v8"},
        }
        publisher_parameters = {
            "requested_bin_size_s": 5.0,
            "source_window_rows": 131_072,
            "row_group_rows": 65_536,
        }
    else:
        source_runs = {
            "tail_kinematics_run": "tail_v2",
            "subject_shape_run": "shape_v4",
            "track_kinematics_run": "track_v1",
            "track_scope": "offline",
        }
        publisher_parameters = {
            "source_window_rows": 16_384,
            "source_rows_per_part": 131_072,
            "row_group_rows": 65_536,
        }
    return build_request(
        family_id=family_id,
        scale_id="full_duration",
        zarr_path=archive,
        export_root=tmp_path / "palette_benchmarks" / "exports",
        scratch_root=tmp_path / "node_benchmarks" / "scratch",
        benchmark_output_dir=tmp_path / "palette_benchmarks" / "evidence",
        export_run_id=f"{family_id}_full_01",
        source_runs=source_runs,
        publisher_parameters=publisher_parameters,
        repetitions=3,
    )


@pytest.mark.parametrize("family_id", sorted(_FAMILIES))
def test_closed_request_accepts_each_exact_family(
    tmp_path: Path,
    family_id: str,
) -> None:
    request = _request(tmp_path, family_id=family_id)
    payload = require_request(request)

    assert payload["family_id"] == family_id
    assert payload["scale_id"] == "full_duration"
    assert payload["workload"]["repetitions"] == 3
    assert payload["resources"]["cache_state"] == "uncontrolled_fresh_process"


def test_request_rejects_recomputed_digest_with_extra_publisher_parameter(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    request["payload"]["publisher_parameters"]["mystery_rows"] = 123
    from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

    request["payload_digest"] = canonical_json_sha256(request["payload"])

    with pytest.raises(ValueError, match="publisher parameters are not exact"):
        require_request(request)


def test_representative_short_kinematics_requires_exact_200k_frame_window(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    (archive / "zarr.json").write_text("{}\n", encoding="utf-8")
    common = {
        "family_id": "kinematics_samples",
        "scale_id": "representative_short",
        "zarr_path": archive,
        "export_root": tmp_path / "palette_benchmarks" / "exports",
        "scratch_root": tmp_path / "node_benchmarks" / "scratch",
        "benchmark_output_dir": tmp_path / "palette_benchmarks" / "evidence",
        "export_run_id": "kinematics_short_01",
        "source_runs": {
            "track_kinematics_run": "track_v1",
            "track_scope": "offline",
        },
    }
    base_parameters = {
        "requested_sample_rate_hz": 10.0,
        "source_window_rows": 131_072,
        "row_group_rows": 65_536,
    }

    with pytest.raises(ValueError, match="requires an explicit frame range"):
        build_request(**common, publisher_parameters=base_parameters)
    with pytest.raises(ValueError, match="exactly 200000 frames"):
        build_request(
            **common,
            publisher_parameters={
                **base_parameters,
                "source_frame_start": 0,
                "source_frame_stop_exclusive": 199_999,
            },
        )

    request = build_request(
        **common,
        publisher_parameters={
            **base_parameters,
            "source_frame_start": 0,
            "source_frame_stop_exclusive": 200_000,
        },
    )
    assert require_request(request)["publisher_parameters"] == {
        **base_parameters,
        "source_frame_start": 0,
        "source_frame_stop_exclusive": 200_000,
    }


def test_full_duration_kinematics_rejects_frame_window(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request["payload"]["publisher_parameters"].update(
        {
            "source_frame_start": 0,
            "source_frame_stop_exclusive": 200_000,
        }
    )
    from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

    request["payload_digest"] = canonical_json_sha256(request["payload"])
    with pytest.raises(ValueError, match="Full-duration kinematics"):
        require_request(request)


def test_representative_short_activity_requires_exact_200k_frame_window(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    (archive / "zarr.json").write_text("{}\n", encoding="utf-8")
    common = {
        "family_id": "activity_spatial_time_bins",
        "scale_id": "representative_short",
        "zarr_path": archive,
        "export_root": tmp_path / "palette_benchmarks" / "exports",
        "scratch_root": tmp_path / "node_benchmarks" / "scratch",
        "benchmark_output_dir": tmp_path / "palette_benchmarks" / "evidence",
        "export_run_id": "activity_short_01",
        "source_runs": {
            "track_kinematics_run": "track_v1",
            "track_scope": "offline",
            "swim_bout_runs_by_track": {"0": "bout_v8"},
        },
    }
    base_parameters = {
        "requested_bin_size_s": 5.0,
        "source_window_rows": 131_072,
        "row_group_rows": 65_536,
    }

    with pytest.raises(ValueError, match="requires an explicit frame range"):
        build_request(**common, publisher_parameters=base_parameters)
    with pytest.raises(ValueError, match="exactly 200000 frames"):
        build_request(
            **common,
            publisher_parameters={
                **base_parameters,
                "source_frame_start": 0,
                "source_frame_stop_exclusive": 199_999,
            },
        )

    request = build_request(
        **common,
        publisher_parameters={
            **base_parameters,
            "source_frame_start": 0,
            "source_frame_stop_exclusive": 200_000,
        },
    )
    assert require_request(request)["publisher_parameters"] == {
        **base_parameters,
        "source_frame_start": 0,
        "source_frame_stop_exclusive": 200_000,
    }


def test_full_duration_activity_rejects_frame_window(tmp_path: Path) -> None:
    request = _request(tmp_path, family_id="activity_spatial_time_bins")
    request["payload"]["publisher_parameters"].update(
        {
            "source_frame_start": 0,
            "source_frame_stop_exclusive": 200_000,
        }
    )
    from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

    request["payload_digest"] = canonical_json_sha256(request["payload"])
    with pytest.raises(ValueError, match="Full-duration activity_spatial_time_bins"):
        require_request(request)


def test_request_rejects_nonbenchmark_publication_path(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    (archive / "zarr.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark-namespaced"):
        build_request(
            family_id="eye_trace_samples",
            scale_id="full_duration",
            zarr_path=archive,
            export_root=tmp_path / "exports",
            scratch_root=tmp_path / "benchmark_scratch",
            benchmark_output_dir=tmp_path / "benchmark_evidence",
            export_run_id="eye_full_01",
            source_runs={"eye_angle_run": "eye_v7"},
            publisher_parameters={"row_group_rows": 65_536},
        )


def test_build_request_cli_writes_closed_kinematics_request(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    (archive / "zarr.json").write_text("{}\n", encoding="utf-8")
    request_path = tmp_path / "benchmark_request.json"

    assert (
        main(
            [
                "build-request",
                "--family",
                "kinematics_samples",
                "--scale",
                "full_duration",
                "--zarr",
                str(archive),
                "--export-root",
                str(tmp_path / "palette_benchmarks" / "exports"),
                "--scratch-root",
                str(tmp_path / "node_benchmarks" / "scratch"),
                "--benchmark-output-dir",
                str(tmp_path / "palette_benchmarks" / "evidence"),
                "--export-run-id",
                "kinematics_full_01",
                "--track-kinematics-run",
                "track_v1",
                "--track-scope",
                "offline",
                "--requested-sample-rate-hz",
                "10",
                "--source-window-rows",
                "131072",
                "--output",
                str(request_path),
            ]
        )
        == 0
    )

    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert require_request(request)["family_id"] == "kinematics_samples"
    assert (
        json.loads(capsys.readouterr().out)["payload_digest"]
        == request["payload_digest"]
    )


def test_build_request_cli_transports_activity_frame_selection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    (archive / "zarr.json").write_text("{}\n", encoding="utf-8")
    request_path = tmp_path / "activity_request.json"

    assert (
        main(
            [
                "build-request",
                "--family",
                "activity_spatial_time_bins",
                "--scale",
                "representative_short",
                "--zarr",
                str(archive),
                "--export-root",
                str(tmp_path / "palette_benchmarks" / "exports"),
                "--scratch-root",
                str(tmp_path / "node_benchmarks" / "scratch"),
                "--benchmark-output-dir",
                str(tmp_path / "palette_benchmarks" / "evidence"),
                "--export-run-id",
                "activity_short_01",
                "--track-kinematics-run",
                "track_v1",
                "--track-scope",
                "offline",
                "--track-swim-bout-run",
                "0=bout_v8",
                "--requested-bin-size-s",
                "5",
                "--source-window-rows",
                "131072",
                "--source-frame-start",
                "100",
                "--source-frame-stop-exclusive",
                "200100",
                "--output",
                str(request_path),
            ]
        )
        == 0
    )

    request = json.loads(request_path.read_text(encoding="utf-8"))
    parameters = require_request(request)["publisher_parameters"]
    assert parameters["source_frame_start"] == 100
    assert parameters["source_frame_stop_exclusive"] == 200_100
    assert json.loads(capsys.readouterr().out)["payload_digest"] == request[
        "payload_digest"
    ]


def _eye_table(start: int, stop: int) -> pa.Table:
    frames = list(range(start, stop))
    count = len(frames)
    return pa.table(
        {
            "recording_id": ["recording_a"] * count,
            "source_acquisition_frame_index": frames,
            "left_eye_angle_deg": [float(value) for value in frames],
            "right_eye_angle_deg": [float(value + 1) for value in frames],
            "vergence_eye_angle_deg": [1.0] * count,
            "valid_frame": [True] * count,
        }
    )


def test_read_workloads_use_row_group_statistics_and_manifest_selected_parts(
    tmp_path: Path,
) -> None:
    first = tmp_path / "part-00000.parquet"
    second = tmp_path / "part-00001.parquet"
    pq.write_table(_eye_table(0, 50), first, row_group_size=10)
    pq.write_table(_eye_table(50, 100), second, row_group_size=10)

    result = _read_workloads(
        parts=(first, second),
        family=_FAMILIES["eye_trace_samples"],
        seed=17,
        random_frame_reads=8,
        window_count=4,
        window_frames=12,
    )

    assert result["axis"] == {
        "column": "source_acquisition_frame_index",
        "minimum": 0,
        "maximum": 99,
        "statistics_source": "parquet_row_group_min_max",
    }
    assert result["footer_open"]["parts"] == [
        {"path": str(first), "row_groups": 5, "rows": 50},
        {"path": str(second), "row_groups": 5, "rows": 50},
    ]
    assert result["random_frame_hot_columns"]["rows"] == 8
    assert result["random_frame_hot_columns"]["latency"]["count"] == 8
    assert result["windowed_frame_hot_columns"]["rows"] == 48
    assert result["windowed_frame_hot_columns"]["latency"]["count"] == 4
    assert result["full_scan"]["rows"] == 100
    assert result["full_scan"]["decoded_bytes"] > 0
    assert len(result["full_scan"]["logical_stream_sha256"]) == 64


def test_source_metadata_guard_covers_exact_track_subtree(tmp_path: Path) -> None:
    request = _request(tmp_path)
    archive = Path(request["payload"]["zarr_path"])
    parent = archive / "analysis" / "track_kinematics_runs"
    scope = parent / "offline"
    run = scope / "track_v1"
    array = run / "tracks" / "id_0" / "positions_mm"
    for path in (parent, scope, run, array):
        path.mkdir(parents=True, exist_ok=True)
        (path / "zarr.json").write_text(
            json.dumps({"node_type": "group"}) + "\n",
            encoding="utf-8",
        )

    guarded = _source_metadata_paths(request["payload"])

    assert guarded == tuple(sorted(guarded))
    assert archive / "zarr.json" in guarded
    assert parent / "zarr.json" in guarded
    assert scope / "zarr.json" in guarded
    assert run / "zarr.json" in guarded
    assert array / "zarr.json" in guarded


def test_axis_extent_fails_closed_without_parquet_statistics(tmp_path: Path) -> None:
    part = tmp_path / "part.parquet"
    pq.write_table(_eye_table(0, 5), part, write_statistics=False)

    with pytest.raises(ValueError, match="lacks row-group min/max statistics"):
        _read_workloads(
            parts=(part,),
            family=_FAMILIES["eye_trace_samples"],
            seed=17,
            random_frame_reads=1,
            window_count=1,
            window_frames=1,
        )
