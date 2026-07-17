from pathlib import Path

from fisheye.diagnostics.benchmark_clipped_roi_cache_nvdec import (
    aggregate_trials,
    parse_gnu_time,
    parse_telemetry,
    recommend_concurrency,
    summarize_bundle,
)


def _bundle(workers: int = 4) -> dict:
    children = []
    for index in range(2):
        children.append(
            {
                "status": "ok",
                "clip_id": f"clip_{index:06d}",
                "published_bin_size_bytes": 262_144_000,
                "row_index": {"row_count": 1000},
                "builder": {
                    "timing": {
                        "rows": 1000,
                        "decoded_frames": 1010,
                        "decode_seconds_total": 10.0,
                        "duration_seconds": 12.0,
                    }
                },
                "publisher": {"payload_copy_seconds": 1.5},
            }
        )
    return {
        "status": "ok",
        "host": "l4-node",
        "max_workers": workers,
        "requested_child_count": 2,
        "completed_child_count": 2,
        "children": children,
    }


def test_parse_telemetry_and_gnu_time(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.csv"
    telemetry_path.write_text(
        "2026/07/17 12:00:00.000, 0, NVIDIA L4, 5, 3, 75, 400, 40\n"
        "2026/07/17 12:00:01.000, 0, NVIDIA L4, 7, 4, 85, 450, 42\n",
        encoding="utf-8",
    )
    telemetry = parse_telemetry(telemetry_path)
    assert telemetry["sample_count"] == 2
    assert telemetry["decoder_utilization_percent"]["mean"] == 80
    assert telemetry["decoder_utilization_percent"]["max"] == 85

    time_path = tmp_path / "time.txt"
    time_path.write_text(
        "\tUser time (seconds): 12.5\n"
        "\tSystem time (seconds): 1.5\n"
        "\tPercent of CPU this job got: 175%\n"
        "\tMaximum resident set size (kbytes): 2048000\n",
        encoding="utf-8",
    )
    usage = parse_gnu_time(time_path)
    assert usage == {
        "user_seconds": 12.5,
        "system_seconds": 1.5,
        "cpu_percent": 175.0,
        "max_rss_kib": 2_048_000,
    }


def test_summarize_and_recommend_concurrency() -> None:
    telemetry = {
        "sample_count": 10,
        "decoder_utilization_percent": {
            "mean": 90.0,
            "median": 91.0,
            "p95": 99.0,
            "max": 100.0,
        },
    }
    summary = summarize_bundle(
        _bundle(),
        trial_seconds=20.0,
        telemetry=telemetry,
        resource_usage={"max_rss_kib": 2_048_000},
    )
    assert summary["rows"] == 2000
    assert summary["decoded_frames"] == 2020
    assert summary["aggregate_rows_per_second"] == 100
    assert summary["aggregate_decoded_frames_per_second"] == 101
    assert summary["weighted_child_decode_frames_per_second"] == 101

    trials = [
        {
            **summary,
            "status": "complete",
            "max_workers": 2,
            "aggregate_rows_per_second": 96.0,
            "aggregate_decoded_frames_per_second": 97.0,
        },
        {
            **summary,
            "status": "complete",
            "max_workers": 4,
            "aggregate_rows_per_second": 100.0,
            "aggregate_decoded_frames_per_second": 101.0,
        },
        {
            **summary,
            "status": "complete",
            "max_workers": 8,
            "aggregate_rows_per_second": 99.0,
            "aggregate_decoded_frames_per_second": 100.0,
        },
    ]
    aggregates = aggregate_trials(trials)
    recommendation = recommend_concurrency(aggregates)
    assert recommendation["fastest_max_workers"] == 4
    assert recommendation["efficient_max_workers"] == 2
