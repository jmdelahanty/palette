import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import write_reason_columns
from fisheye.tune.keypoint_review import _count_reason_tags, _update_postprocess_summary


def test_count_reason_tags_uses_reason_bytes_fallback(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test_keypoint_review_reason_fallback.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")

    labels = np.array(
        [
            "manual_correction|geometry_issue",
            "manual_correction",
            "clean",
        ],
        dtype=object,
    )
    write_reason_columns(refined, labels, chunk_size=2, overwrite=True)

    counts = _count_reason_tags(refined)

    assert counts["manual_correction"] == 2
    assert counts["geometry_issue"] == 1
    assert counts["clean"] == 1


def test_update_postprocess_summary_keeps_timestamp_when_stats_unchanged(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test_keypoint_review_summary_static.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    refined.create_array("keypoints_roi", shape=(2, 3, 2), dtype="f8")
    refined.create_array("refined_success", data=np.array([True, False], dtype=bool))

    _update_postprocess_summary(refined, print_summary=False)
    sentinel = "2026-02-01T12:00:00+00:00"
    summary = dict(refined.attrs["summary_statistics"])
    summary["postprocess_updated_utc"] = sentinel
    refined.attrs["summary_statistics"] = summary

    _update_postprocess_summary(refined, print_summary=False)
    summary_after = dict(refined.attrs["summary_statistics"])
    assert summary_after["postprocess_updated_utc"] == sentinel


def test_update_postprocess_summary_updates_timestamp_when_stats_change(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test_keypoint_review_summary_change.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    refined.create_array("keypoints_roi", shape=(2, 3, 2), dtype="f8")
    refined.create_array("refined_success", data=np.array([True, False], dtype=bool))

    _update_postprocess_summary(refined, print_summary=False)
    sentinel = "2026-02-01T12:00:00+00:00"
    summary = dict(refined.attrs["summary_statistics"])
    summary["postprocess_updated_utc"] = sentinel
    refined.attrs["summary_statistics"] = summary

    refined["refined_success"][1] = True
    _update_postprocess_summary(refined, print_summary=False)
    summary_after = dict(refined.attrs["summary_statistics"])
    assert summary_after["postprocess_updated_utc"] != sentinel
