from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from zarr.codecs import BloscCodec

DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN = 32
DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN = 128
SCRATCH_ROI_CACHE_LAYOUT_PROFILE = "scratch_v1"
_VALID_ROI_STORAGE = {"compressed", "uncompressed"}


@dataclass(frozen=True)
class CropRoiLayout:
    roi_storage: str
    roi_chunk_len: int
    roi_use_sharding: bool
    roi_shard_len: Optional[int] = None


def normalize_crop_roi_storage(value: object, *, default: str = "compressed") -> str:
    text = str(value or default).strip().lower()
    if text not in _VALID_ROI_STORAGE:
        return default
    return text


def build_canonical_crop_roi_layout(
    *,
    total_rois: int,
    preferred_chunk_len: int,
    roi_storage: object = "compressed",
    use_sharding: bool = False,
    roi_shard_len: Optional[int] = None,
) -> CropRoiLayout:
    total = max(1, int(total_rois))
    chunk_len = max(1, min(int(preferred_chunk_len), total))
    storage = normalize_crop_roi_storage(roi_storage)
    if use_sharding:
        if roi_shard_len is None:
            shard_len = max(chunk_len, chunk_len * 8)
        else:
            shard_len = max(chunk_len, int(roi_shard_len))
        shard_len = min(shard_len, total)
    else:
        shard_len = None
    return CropRoiLayout(
        roi_storage=storage,
        roi_chunk_len=chunk_len,
        roi_use_sharding=bool(use_sharding),
        roi_shard_len=shard_len,
    )


def build_scratch_roi_cache_layout(
    *,
    total_rois: int,
    preferred_chunk_len: int | None = None,
) -> CropRoiLayout:
    chunk_preference = (
        DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN if preferred_chunk_len is None else int(preferred_chunk_len)
    )
    return build_canonical_crop_roi_layout(
        total_rois=total_rois,
        preferred_chunk_len=chunk_preference,
        roi_storage="uncompressed",
        use_sharding=False,
        roi_shard_len=None,
    )


def crop_roi_layout_attrs(layout: CropRoiLayout) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "roi_storage": layout.roi_storage,
        "roi_chunk_len": int(layout.roi_chunk_len),
        "roi_use_sharding": bool(layout.roi_use_sharding),
    }
    if layout.roi_use_sharding and layout.roi_shard_len is not None:
        attrs["roi_shard_len"] = int(layout.roi_shard_len)
    return attrs


def build_crop_roi_create_kwargs(
    *,
    total_rois: int,
    roi_sz: Tuple[int, int],
    layout: CropRoiLayout,
    overwrite: bool = True,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "shape": (int(total_rois), int(roi_sz[0]), int(roi_sz[1])),
        "chunks": (int(layout.roi_chunk_len), int(roi_sz[0]), int(roi_sz[1])),
        "dtype": "uint8",
        "overwrite": bool(overwrite),
    }
    if layout.roi_storage == "compressed":
        kwargs["compressors"] = BloscCodec(typesize=1, cname="lz4", clevel=1, shuffle="bitshuffle")
    else:
        kwargs["compressors"] = None
    if layout.roi_use_sharding and layout.roi_shard_len is not None:
        kwargs["shards"] = (int(layout.roi_shard_len), int(roi_sz[0]), int(roi_sz[1]))
    return kwargs
