import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import decode_reason_bytes
from fisheye.tune.detect_review import _write_manual_group


def test_write_manual_group_writes_reason_bytes_and_detection_source(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "detect_review_reason.zarr", mode="w")
    refined = root.create_group("refined_detect")

    frame_indices = np.array([0, 2], dtype=np.int32)
    bbox_norm = np.array([[0.5, 0.5, 0.1, 0.1], [0.4, 0.6, 0.2, 0.2]], dtype=np.float64)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 0], dtype=np.int32)
    frame_counts = np.array([1, 0, 1], dtype=np.int32)

    _write_manual_group(
        refined_run=refined,
        output_group="manual",
        frame_indices=frame_indices,
        bbox_norm=bbox_norm,
        scores=scores,
        class_ids=class_ids,
        retune_id=None,
        frame_counts=frame_counts,
        detection_source=None,
        reason=None,
        metadata={"manual_review_timestamp": "2026-02-09T00:00:00Z"},
        overwrite=False,
    )

    manual = refined["manual"]
    assert np.asarray(manual["detection_source"][:], dtype=np.int8).tolist() == [0, 0]
    assert "reason_bytes" in manual
    assert decode_reason_bytes(manual["reason_bytes"][:]).tolist() == ["clean", "clean"]
    assert "reason" not in manual
    assert manual.attrs["reason_authority"] == "reason_bytes"
    assert manual.attrs["reason_fallback_order"] == ["reason_bytes", "detection_source"]
    fields = list(manual.attrs.get("column_fields", []))
    assert "reason_bytes" in fields
    assert "reason" not in fields
    assert "detection_source" in fields
