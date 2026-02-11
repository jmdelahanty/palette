import numpy as np
import zarr

from fisheye.refinement.refine_keypoints import _write_reason_arrays
from fisheye.shared.detect_reason_codec import decode_reason_bytes


def test_write_reason_arrays_creates_reason_and_reason_bytes(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test_keypoint_reasons.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")

    labels = np.array(
        [
            "clean",
            "flip_corrected|geometry_issue",
            "manual_correction",
        ],
        dtype=object,
    )
    _write_reason_arrays(refined, labels, chunk_size=2)

    stored_text = np.asarray(refined["reason"][:], dtype=object).tolist()
    assert stored_text == labels.tolist()

    stored_bytes = np.asarray(refined["reason_bytes"][:], dtype=np.uint8)
    decoded = decode_reason_bytes(stored_bytes).tolist()
    assert decoded == labels.tolist()
    assert refined.attrs["reason_encoding"] == "utf8-null-terminated"
    assert refined.attrs["reason_fallback_order"] == ["reason_bytes", "reason", "detection_source"]
