"""Utilities for loading eye-mask segmentation data with a consistent schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import dask.array as da
import numpy as np
import zarr

from fisheye.shared.mask_probability_encoding import decode_probability_values_from_attrs


@dataclass
class MaskBundle:
    """Container describing masks and probabilities for an eye-mask run."""

    binary: object  # numpy.ndarray or dask.array.Array
    probs: Optional[object]  # numpy.ndarray or dask.array.Array
    binary_identity: Literal["union", "lr"]
    probs_identity: Optional[Literal["union", "lr"]]
    provenance: Dict[str, object]


def _to_float01(arr: np.ndarray, *, run: zarr.Group, source_path: str) -> np.ndarray:
    return decode_probability_values_from_attrs(
        arr,
        attrs=run.attrs,
        source_path=source_path,
    )


def _to_uint01(arr: np.ndarray) -> np.ndarray:
    if isinstance(arr, da.Array):
        return (arr > 0).astype(np.uint8)
    return (arr > 0).astype(np.uint8, copy=False)


def _ensure_4d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr[:, None, :, :]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {arr.shape}")
    return arr


def _select_probability_source(run: zarr.Group) -> Optional[str]:
    if "mask_probs_roi_refined" in run:
        return "mask_probs_roi_refined"
    if "mask_probs_roi" in run:
        return "mask_probs_roi"
    return None


def _materialize_masks(
    run: zarr.Group,
    binary: np.ndarray,
    source_array: Optional[zarr.Array],
) -> None:
    N, C, H, W = binary.shape
    chunks = None
    compressors = None
    filters = None

    if source_array is not None:
        chunks = getattr(source_array, "chunks", None)
        compressors = getattr(source_array, "compressors", None)
        if compressors is None:
            compressor = getattr(source_array, "compressor", None)
            if compressor is not None:
                compressors = compressor
        filters = getattr(source_array, "filters", None)

    kwargs: Dict[str, object] = {}
    if chunks:
        kwargs["chunks"] = chunks
    else:
        kwargs["chunks"] = (min(512, N), C, H, W)
    if compressors is not None:
        if isinstance(compressors, tuple):
            kwargs["compressors"] = compressors
        else:
            kwargs["compressor"] = compressors
    if filters is not None:
        kwargs["filters"] = filters

    run.create_array(
        "masks_roi",
        data=binary.astype(np.uint8, copy=False),
        overwrite=True,
        **kwargs,
    )


def load_mask_bundle(
    run: zarr.Group,
    *,
    threshold: float = 0.5,
    prefer_probs: bool = True,
    materialize: bool = False,
    lazy: bool = False,
    lazy_chunks: Optional[Tuple[int, ...]] = None,
) -> MaskBundle:
    """Return a consistent mask/probability bundle for the provided run."""

    if lazy and materialize:
        raise ValueError("Cannot materialize masks when loading lazily.")

    keys = set(run.array_keys())
    provenance: Dict[str, object] = {
        "source": [],
        "threshold": float(threshold),
        "materialized": False,
    }

    def _load_array(arr: zarr.Array) -> object:
        if lazy:
            if lazy_chunks is None:
                return da.from_zarr(arr)
            return da.from_zarr(arr, chunks=lazy_chunks)
        return arr[:]

    def _load_probabilities(probs_key: str) -> object:
        raw_probs = _load_array(run[probs_key])
        source_path = f"{getattr(run, 'path', '<run>')}/{probs_key}"
        return _ensure_4d(_to_float01(raw_probs, run=run, source_path=source_path))

    if lazy_chunks is not None and len(lazy_chunks) != 4:
        raise ValueError("lazy_chunks must be a 4-tuple matching (N, C, H, W) layout.")

    probs: Optional[np.ndarray] = None
    probs_identity: Optional[Literal["union", "lr"]] = None
    probs_key = _select_probability_source(run) if prefer_probs else None

    if probs_key is not None:
        probs = _load_probabilities(probs_key)
        probs_identity = "lr" if probs.shape[1] == 2 else "union"
        provenance["source"].append(probs_key)

    binary: Optional[np.ndarray] = None
    binary_identity: Literal["union", "lr"]

    if "masks_roi" in keys:
        raw_binary = _load_array(run["masks_roi"])
        binary = _ensure_4d(_to_uint01(raw_binary))
        binary_identity = "lr" if binary.shape[1] == 2 else "union"
        provenance["source"].append("masks_roi")
        if probs is None and prefer_probs:
            # if masks exist but probabilities requested, fetch fallback
            fallback_probs_key = _select_probability_source(run)
            if fallback_probs_key is not None:
                probs = _load_probabilities(fallback_probs_key)
                probs_identity = "lr" if probs.shape[1] == 2 else "union"
                provenance["source"].append(fallback_probs_key)
    elif probs is not None:
        if isinstance(probs, da.Array):
            binary = (probs >= threshold).astype(np.uint8)
        else:
            binary = (probs >= threshold).astype(np.uint8, copy=False)
        binary_identity = "lr" if probs.shape[1] == 2 else "union"
        provenance["source"].append("synth_binary_from_probs")
        if materialize:
            source_array = run.get(probs_key) if probs_key else None
            _materialize_masks(run, binary.compute() if isinstance(binary, da.Array) else binary, source_array)
            provenance["materialized"] = True
    else:
        raise ValueError("Run provides neither 'masks_roi' nor probability masks.")

    return MaskBundle(
        binary=binary,
        probs=probs,
        binary_identity=binary_identity,
        probs_identity=probs_identity,
        provenance=provenance,
    )
