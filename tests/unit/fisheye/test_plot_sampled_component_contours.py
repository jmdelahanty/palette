from __future__ import annotations

import numpy as np
import pytest

from fisheye.diagnostics.plot_sampled_component_contours import (
    RoiImageSource,
    _normalize_image,
    _short_source_label,
    build_arg_parser,
    component_k,
    contour_similarity_metrics,
    parse_component_k,
    resolve_roi_image_source,
    resolve_pixel_to_mm,
    resample_closed_polyline,
)


def test_resample_closed_polyline_returns_fixed_k_samples() -> None:
    square = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ],
        dtype=np.float32,
    )

    sampled = resample_closed_polyline(square, 8)

    assert sampled.shape == (8, 2)
    np.testing.assert_allclose(sampled[0], [0.0, 0.0])
    np.testing.assert_allclose(sampled[2], [10.0, 0.0])
    np.testing.assert_allclose(sampled[4], [10.0, 10.0])
    np.testing.assert_allclose(sampled[6], [0.0, 10.0])


def test_resample_closed_polyline_handles_degenerate_contours() -> None:
    sampled_empty = resample_closed_polyline(np.empty((0, 2), dtype=np.float32), 3)
    sampled_single = resample_closed_polyline(np.asarray([[4.0, 5.0]], dtype=np.float32), 3)

    assert sampled_empty.shape == (3, 2)
    assert np.isnan(sampled_empty).all()
    np.testing.assert_allclose(sampled_single, np.asarray([[4.0, 5.0]] * 3, dtype=np.float32))


def test_contour_similarity_metrics_reports_pixel_distances() -> None:
    raw = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ],
        dtype=np.float32,
    )
    sampled = raw.copy()

    metrics = contour_similarity_metrics(raw, sampled, pixel_to_mm=0.02)

    assert metrics["raw_vertices"] == 4
    assert metrics["sampled_vertices"] == 4
    assert metrics["contour_perimeter_px"] == pytest.approx(40.0)
    assert metrics["contour_bbox_diagonal_px"] == pytest.approx(np.sqrt(200.0))
    assert metrics["raw_to_sampled_max_px"] == pytest.approx(0.0)
    assert metrics["symmetric_hausdorff_px"] == pytest.approx(0.0)
    assert metrics["pixel_to_mm"] == pytest.approx(0.02)
    assert metrics["raw_to_sampled_p95_mm"] == pytest.approx(0.0)


def test_parse_component_k_and_defaults() -> None:
    overrides = parse_component_k(["subject_body=512", "eye_left=48"])

    assert component_k("subject_body", overrides, default_k=32) == 512
    assert component_k("eye_left", overrides, default_k=32) == 48
    assert component_k("eye_right", overrides, default_k=32) == 64
    assert component_k("unknown_component", overrides, default_k=32) == 32


def test_parse_component_k_rejects_invalid_specs() -> None:
    with pytest.raises(Exception, match="Expected COMPONENT=K"):
        parse_component_k(["subject_body"])
    with pytest.raises(Exception, match="K must be positive"):
        parse_component_k(["subject_body=0"])


def test_cli_defaults_to_comparison_layout() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(["--zarr", "example.zarr", "--output", "out.png", "--comparison-k", "128"])

    assert args.layout == "comparison"
    assert args.comparison_k == [128]


def test_roi_image_source_maps_refined_rows_to_crop_rows() -> None:
    roi_images = np.stack(
        [
            np.full((4, 4), 10, dtype=np.uint8),
            np.full((4, 4), 90, dtype=np.uint8),
        ],
        axis=0,
    )
    source = RoiImageSource(
        crop_run_name="crop_test",
        roi_images=roi_images,
        source_crop_row_ids=np.asarray([1, 0], dtype=np.int64),
    )

    np.testing.assert_array_equal(source.image_for_refined_row(0), roi_images[1])
    np.testing.assert_array_equal(source.image_for_refined_row(1), roi_images[0])
    assert source.image_for_refined_row(-1) is None
    assert source.image_for_refined_row(2) is None


def test_resolve_roi_image_source_falls_back_to_materialized_roi_images(tmp_path) -> None:
    import zarr

    root = zarr.open_group(str(tmp_path / "contour_image_source.zarr"), mode="w")
    crop = root.create_group("crop_runs").create_group("crop_run")
    roi_images = np.stack(
        [
            np.full((4, 4), 10, dtype=np.uint8),
            np.full((4, 4), 90, dtype=np.uint8),
        ],
        axis=0,
    )
    crop.create_array("roi_images", data=roi_images, chunks=(1, 4, 4))
    run = root.create_group("refined_subject_masks_runs").create_group("run")
    run.attrs["source_crop_run"] = "crop_run"
    run.create_array("source_crop_row_ids", data=np.asarray([1, 0], dtype=np.int64), chunks=(2,))

    source = resolve_roi_image_source(root, run)

    assert source is not None
    assert source.crop_run_name == "crop_run"
    assert source.source_kind == "roi_images"
    assert source.row_position_fallback is False
    np.testing.assert_array_equal(source.image_for_refined_row(0), roi_images[1])


def test_resolve_pixel_to_mm_prefers_analysis_calibration(tmp_path) -> None:
    import zarr

    root = zarr.open_group(str(tmp_path / "calibration.zarr"), mode="w")
    root.attrs["pixel_to_mm"] = 0.5
    calibration = root.create_group("analysis").create_group("calibration")
    calibration.attrs["pixel_to_mm"] = 0.02

    scale, source = resolve_pixel_to_mm(root)

    assert scale == pytest.approx(0.02)
    assert source == "analysis/calibration.attrs[pixel_to_mm]"


def test_normalize_image_uses_robust_display_range() -> None:
    image = np.asarray([[0, 10, 20], [30, 40, 255]], dtype=np.uint8)

    normalized = _normalize_image(image)

    assert normalized.shape == image.shape
    assert float(normalized.min()) >= 0.0
    assert float(normalized.max()) <= 1.0


def test_short_source_label_preserves_source_kind() -> None:
    label = "crop:crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-09Z_arena_1_RedScare/materialized_crop_run"

    shortened = _short_source_label(label, max_chars=48)

    assert len(shortened) <= 48
    assert shortened.endswith("/materialized_crop_run")
    assert "..." in shortened
