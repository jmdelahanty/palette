"""Content-derived detection instance keys.

Instance keys are minted once at detection time and copied verbatim through
downstream row-lineage arrays. They are intentionally not positional IDs.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

INSTANCE_KEY_ARRAY = "instance_key"
INSTANCE_KEY_ALGORITHM = "palette.blake2b64.recording_frame_bbox_class_v1"
INSTANCE_KEY_BBOX_QUANTIZATION = 1_000_000
INSTANCE_KEY_DUPLICATE_POLICY = "duplicate_base_keys_get_detect_time_occurrence_ordinal"


def resolve_recording_identity(attrs: Mapping[str, Any], *, fallback_path: str | Path | None = None) -> str:
    """Resolve the stable recording identity used as one instance-key input."""

    for name in ("recording_id", "recording_name", "session_uuid", "dataset_id"):
        value = attrs.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    if fallback_path is not None:
        return Path(fallback_path).expanduser().resolve().stem
    return "unknown_recording"


def _digest_uint64(payload: str) -> np.uint64:
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, byteorder="little", signed=False))


def mint_detection_instance_keys(
    *,
    recording_identity: str,
    frame_indices: np.ndarray,
    bbox_norm_coords: np.ndarray,
    class_ids: np.ndarray | None = None,
    bbox_quantization: int = INSTANCE_KEY_BBOX_QUANTIZATION,
) -> np.ndarray:
    """Mint deterministic per-detection keys from recording, frame, class, and bbox.

    The normal key payload is content-derived. If two rows have identical payloads,
    a detect-time duplicate ordinal is appended to those duplicate payloads so the
    final key remains unique within the run. Downstream stages must copy the key;
    they must not recompute it.
    """

    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    bboxes = np.asarray(bbox_norm_coords, dtype=np.float64).reshape(-1, 4)
    if bboxes.shape[0] != frames.shape[0]:
        raise ValueError("bbox_norm_coords row count does not match frame_indices.")
    classes = (
        np.zeros(frames.shape[0], dtype=np.int64)
        if class_ids is None
        else np.asarray(class_ids, dtype=np.int64).reshape(-1)
    )
    if classes.shape[0] != frames.shape[0]:
        raise ValueError("class_ids row count does not match frame_indices.")

    if frames.size == 0:
        return np.empty((0,), dtype=np.uint64)

    quant = int(bbox_quantization)
    if quant <= 0:
        raise ValueError("bbox_quantization must be positive.")
    quantized = np.rint(bboxes * float(quant)).astype(np.int64, copy=False)

    base_payloads = [
        "|".join(
            (
                str(recording_identity),
                str(int(frame)),
                str(int(cls)),
                str(int(x)),
                str(int(y)),
                str(int(w)),
                str(int(h)),
            )
        )
        for frame, cls, (x, y, w, h) in zip(frames, classes, quantized, strict=True)
    ]
    duplicate_counts = Counter(base_payloads)
    occurrences: dict[str, int] = {}
    out = np.empty(frames.shape[0], dtype=np.uint64)
    for idx, payload in enumerate(base_payloads):
        if duplicate_counts[payload] > 1:
            ordinal = occurrences.get(payload, 0)
            occurrences[payload] = ordinal + 1
            payload = f"{payload}|duplicate_ordinal={ordinal}"
        out[idx] = _digest_uint64(payload)

    unique_count = int(np.unique(out).shape[0])
    if unique_count != int(out.shape[0]):
        raise ValueError("instance_key hash collision detected within detection run.")
    return out


def instance_key_attrs(recording_identity: str) -> dict[str, Any]:
    """Return attrs documenting the instance-key minting policy."""

    return {
        "instance_key_algorithm": INSTANCE_KEY_ALGORITHM,
        "instance_key_recording_identity": str(recording_identity),
        "instance_key_bbox_quantization": int(INSTANCE_KEY_BBOX_QUANTIZATION),
        "instance_key_duplicate_policy": INSTANCE_KEY_DUPLICATE_POLICY,
    }
