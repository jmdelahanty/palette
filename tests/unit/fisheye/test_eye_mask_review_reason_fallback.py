import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import write_reason_columns
from fisheye.tune import eye_mask_review as mod


def test_count_reason_tags_uses_metrics_reason_bytes_fallback(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test_eye_mask_review_reason_fallback.zarr", mode="w")
    refined = root.create_group("refined_eye_masks_runs").create_group("refined_eye_masks_001")
    metrics = refined.create_group("metrics")

    labels = np.array(
        [
            "manual_correction|overlap",
            "manual_correction",
            "clean",
        ],
        dtype=object,
    )
    write_reason_columns(metrics, labels, chunk_size=2, include_reason_text=True, overwrite=True)
    del metrics["reason"]

    counts = mod._count_reason_tags(refined)

    assert counts["manual_correction"] == 2
    assert counts["overlap"] == 1
    assert counts["clean"] == 1


def test_update_postprocess_summary_derives_eye_separation_from_ellipse_centers(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test_eye_mask_review_summary_ellipse_sep.zarr", mode="w")
    refined = root.create_group("refined_eye_masks_runs").create_group("refined_eye_masks_001")
    masks = np.zeros((2, 2, 8, 8), dtype=np.uint8)
    masks[0, 0, :, :] = 1
    masks[0, 1, :, :] = 1
    masks[1, 0, :, :] = 1
    refined.create_array(
        "ellipse_success",
        data=np.array([[True, True], [True, False]], dtype=bool),
    )
    refined.create_array(
        "ellipse_params",
        data=np.array(
            [
                [[10.0, 10.0, 8.0, 4.0, 0.0], [14.0, 13.0, 7.0, 3.0, 0.0]],
                [[20.0, 20.0, 8.0, 4.0, 0.0], [25.0, 24.0, 7.0, 3.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )
    refined.create_array(
        "masks_roi",
        data=masks,
    )

    stats = mod._update_postprocess_summary(root, refined, print_summary=False)

    assert stats["successful_roi_pairs"] == 1
    assert refined.attrs["successful_roi_pairs"] == 1
    assert dict(refined.attrs["summary_statistics"])["postprocess"]["successful_roi_pairs"] == 1
