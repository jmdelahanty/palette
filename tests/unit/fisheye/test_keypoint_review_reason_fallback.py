import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import write_reason_columns
from fisheye.tune.keypoint_review import _count_reason_tags


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
    write_reason_columns(refined, labels, chunk_size=2, include_reason_text=True, overwrite=True)
    del refined["reason"]

    counts = _count_reason_tags(refined)

    assert counts["manual_correction"] == 2
    assert counts["geometry_issue"] == 1
    assert counts["clean"] == 1
