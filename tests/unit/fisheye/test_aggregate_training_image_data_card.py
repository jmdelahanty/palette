from __future__ import annotations

import json

from fisheye.utils.aggregate_training_image_data_card import build_training_image_data_card


def test_build_training_image_data_card_aggregates_metrics_and_histograms() -> None:
    manifest = {
        "set_id": "detect_sleepyfish_v001",
        "set_version": "1",
        "datasets": [
            {"dataset_id": "dataset_a", "zarr_path": "/tmp/a_training.zarr"},
            {"dataset_id": "dataset_b", "zarr_path": "/tmp/b_training.zarr"},
            {"dataset_id": "dataset_missing", "zarr_path": "/tmp/missing_training.zarr"},
        ],
    }
    rows = [
        {
            "dataset_id": "dataset_a",
            "profile_run": "profile_a",
            "recording_id": "rec_a",
            "zarr_use": "training",
            "source_frame_array": "raw_video/images_ds",
            "mean_intensity_p50": 100.0,
            "contrast_p50": 20.0,
            "sharpness_p50": 3.0,
            "clip_dark_rate_mean": 0.01,
            "clip_bright_rate_mean": 0.02,
            "dish_design": "palm",
        },
        {
            "dataset_id": "dataset_b",
            "profile_run": "profile_b",
            "recording_id": "rec_b",
            "zarr_use": "training",
            "source_frame_array": "raw_video/images_ds",
            "mean_intensity_p50": 120.0,
            "contrast_p50": 30.0,
            "sharpness_p50": 5.0,
            "clip_dark_rate_mean": 0.03,
            "clip_bright_rate_mean": 0.04,
            "dish_design": "polar",
        },
    ]
    summaries = [
        {"intensity_histogram": {"bin_edges": [0.0, 128.0, 256.0], "counts": [2, 3]}},
        {"intensity_histogram": {"bin_edges": [0.0, 128.0, 256.0], "counts": [5, 7]}},
    ]

    card = build_training_image_data_card(
        manifest=manifest,
        profile_rows=rows,
        profile_summaries=summaries,
        missing_profile_dataset_ids=["dataset_missing"],
        split="all",
    )

    assert card["schema_name"] == "training_image_data_card"
    assert card["selection"]["profiled_dataset_count"] == 2
    assert card["selection"]["missing_profile_dataset_count"] == 1
    assert card["metric_stats"]["mean_intensity_p50"]["p50"] == 110.0
    assert card["intensity_histogram_aggregate"]["counts"] == [7, 10]
    assert card["composition_counts"]["dish_design"] == {"palm": 1, "polar": 1}
    json.dumps(card, allow_nan=False)
