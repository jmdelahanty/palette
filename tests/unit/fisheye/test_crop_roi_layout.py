from __future__ import annotations

from fisheye.shared.crop_roi_layout import (
    build_canonical_crop_roi_layout,
    build_crop_roi_create_kwargs,
    crop_roi_layout_attrs,
)


def test_build_canonical_crop_roi_layout_clamps_chunk_and_shard_lengths() -> None:
    layout = build_canonical_crop_roi_layout(
        total_rois=100,
        preferred_chunk_len=128,
        roi_storage="compressed",
        use_sharding=True,
        roi_shard_len=512,
    )

    assert layout.roi_storage == "compressed"
    assert layout.roi_chunk_len == 100
    assert layout.roi_use_sharding is True
    assert layout.roi_shard_len == 100


def test_build_crop_roi_create_kwargs_sets_full_roi_chunks() -> None:
    layout = build_canonical_crop_roi_layout(
        total_rois=200,
        preferred_chunk_len=32,
        roi_storage="compressed",
        use_sharding=False,
        roi_shard_len=None,
    )

    kwargs = build_crop_roi_create_kwargs(
        total_rois=200,
        roi_sz=(512, 512),
        layout=layout,
    )

    assert kwargs["shape"] == (200, 512, 512)
    assert kwargs["chunks"] == (32, 512, 512)
    assert kwargs["dtype"] == "uint8"
    assert kwargs["compressors"] is not None
    assert "shards" not in kwargs


def test_crop_roi_layout_attrs_omits_shard_len_when_unsharded() -> None:
    layout = build_canonical_crop_roi_layout(
        total_rois=50,
        preferred_chunk_len=16,
        roi_storage="uncompressed",
        use_sharding=False,
        roi_shard_len=None,
    )

    attrs = crop_roi_layout_attrs(layout)

    assert attrs == {
        "roi_storage": "uncompressed",
        "roi_chunk_len": 16,
        "roi_use_sharding": False,
    }
