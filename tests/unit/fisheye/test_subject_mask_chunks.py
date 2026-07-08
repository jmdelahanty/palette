from __future__ import annotations

from fisheye.shared.subject_mask_chunks import (
    refined_subject_mask_bitpacked_chunks,
    refined_subject_mask_metric_row_chunk,
    refined_subject_mask_storage_row_chunk,
    refined_subject_mask_storage_chunks,
    subject_mask_metric_row_chunk,
    subject_mask_storage_chunks,
)


def test_subject_mask_storage_chunks_use_full_roi_spatial_extent() -> None:
    assert subject_mask_storage_chunks(227, 512, 512) == (16, 1, 512, 512)


def test_subject_mask_storage_chunks_clamp_to_small_arrays() -> None:
    assert subject_mask_storage_chunks(2, 6, 8) == (2, 1, 6, 8)


def test_subject_mask_metric_row_chunk_uses_large_row_groups() -> None:
    assert subject_mask_metric_row_chunk(227) == 227


def test_refined_subject_mask_storage_chunks_use_editable_dense_policy() -> None:
    assert refined_subject_mask_storage_chunks(227, 512, 512) == (128, 1, 512, 512)


def test_refined_subject_mask_storage_chunks_adapt_to_crop_shape() -> None:
    assert refined_subject_mask_storage_chunks(227, 348, 348) == (128, 1, 348, 348)


def test_refined_subject_mask_bitpacked_chunks_use_cache_policy() -> None:
    assert refined_subject_mask_bitpacked_chunks(1000, 4, 512, 512) == (512, 4, 512, 64)
    assert refined_subject_mask_bitpacked_chunks(1000, 3, 348, 348) == (512, 3, 348, 44)


def test_refined_subject_mask_storage_chunks_accept_explicit_row_chunk() -> None:
    assert refined_subject_mask_storage_row_chunk(1000, 256) == 256
    assert refined_subject_mask_storage_chunks(1000, 512, 512, row_chunk=512) == (512, 1, 512, 512)


def test_refined_subject_mask_storage_chunks_clamp_explicit_row_chunk() -> None:
    assert refined_subject_mask_storage_row_chunk(32, 512) == 32
    assert refined_subject_mask_storage_chunks(32, 512, 512, row_chunk=512) == (32, 1, 512, 512)


def test_refined_subject_mask_metric_row_chunk_matches_current_policy() -> None:
    assert refined_subject_mask_metric_row_chunk(227) == 227
