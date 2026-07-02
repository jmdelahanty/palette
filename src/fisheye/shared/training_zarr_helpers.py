"""Shared helpers for merged training-Zarr exporters."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8

from fisheye.registry.db import Registry, resolve_dataset_id
from fisheye.shared.type_conversions import normalize_attr as _as_text


def json_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except Exception:
            return []
        if isinstance(payload, list):
            return [str(item) for item in payload if item]
    return []


def json_dict(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except Exception:
            return None
        if isinstance(payload, dict):
            return payload
    return None


def resolve_source_dataset_id(
    *,
    source_root: zarr.Group,
    source_path: Path,
    registry_path: Optional[Path],
) -> str:
    """Resolve source dataset identity with detect/keypoint-aligned precedence."""
    explicit_dataset_id = _as_text(source_root.attrs.get("dataset_id"))
    if explicit_dataset_id:
        return str(explicit_dataset_id)

    dataset_id, _ = resolve_dataset_id(source_root, source_path)

    if registry_path is not None:
        registry = Registry(registry_path)
        try:
            registered_dataset_id = registry.scan_zarr(source_path)
            if registered_dataset_id:
                return str(registered_dataset_id)
        finally:
            registry.close()

    return str(dataset_id)


def write_string_array(group: zarr.Group, name: str, values: Sequence[str]) -> zarr.Array:
    arr = group.create_array(
        name,
        shape=(int(len(values)),),
        dtype=VariableLengthUTF8(),
        chunks=(max(1, min(65536, int(len(values)) or 1)),),
        overwrite=True,
    )
    arr[:] = np.asarray([str(v) for v in values], dtype=object)
    return arr


def normalized_split_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    train_v = max(0.0, float(train))
    val_v = max(0.0, float(val))
    test_v = max(0.0, float(test))
    total = train_v + val_v + test_v
    if total <= 0.0:
        raise ValueError("At least one split ratio must be > 0.")
    return train_v / total, val_v / total, test_v / total


def make_split_indices(
    total_samples: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = int(total_samples)
    if total <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    tr, vr, ter = normalized_split_ratios(train_ratio, val_ratio, test_ratio)
    order = np.random.default_rng(int(seed)).permutation(total).astype(np.int64, copy=False)
    train_count = int(round(float(total) * tr))
    val_count = int(round(float(total) * vr))
    train_count = max(0, min(train_count, total))
    val_count = max(0, min(val_count, total - train_count))
    test_count = total - train_count - val_count
    if ter <= 0.0:
        val_count = total - train_count
        test_count = 0

    train_idx = order[:train_count]
    val_idx = order[train_count : train_count + val_count]
    test_idx = order[train_count + val_count : train_count + val_count + test_count]
    return train_idx, val_idx, test_idx
