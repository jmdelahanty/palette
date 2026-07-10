"""Stable per-row fingerprints for dense binary mask surfaces."""

from __future__ import annotations

import hashlib

import numpy as np


def mask_row_fingerprint(mask: np.ndarray) -> np.uint64:
    """Return the contract BLAKE2b-64 fingerprint for one dense uint8 mask."""

    payload = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    digest = hashlib.blake2b(payload.tobytes(), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, byteorder="little", signed=False))


def batch_mask_row_fingerprints(masks: np.ndarray) -> np.ndarray:
    """Return stable fingerprints for a dense ``(N,H,W)`` mask block."""

    rows = np.asarray(masks, dtype=np.uint8)
    if rows.ndim != 3:
        raise ValueError(f"Expected component masks with shape (N,H,W), got {tuple(rows.shape)}")
    out = np.zeros((int(rows.shape[0]),), dtype=np.uint64)
    for row_idx in range(int(rows.shape[0])):
        out[row_idx] = mask_row_fingerprint(rows[row_idx])
    return out
