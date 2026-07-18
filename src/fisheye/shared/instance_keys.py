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

# Payload context appended when curation mints keys for rows whose point of
# origin is manual curation (no source detect row). Namespacing the payload
# guarantees a manual box that quantizes identically to a detection cannot
# collide with that detection's copied key.
INSTANCE_KEY_CONTEXT_MANUAL_CURATION = "manual_curation"
INSTANCE_KEY_CONTEXT_MANUAL_CURATION_ROW_ID = "manual_curation_refined_row_id_v1"

# Per-row provenance for curated rowsets that mix copied and minted keys.
INSTANCE_KEY_ORIGIN_ARRAY = "instance_key_origin_codes"
INSTANCE_KEY_ORIGIN_CODE_MAP: dict[str, int] = {
    "copied_from_detect": 0,
    "minted_at_curation": 1,
}


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
    payload_context: str | None = None,
) -> np.ndarray:
    """Mint deterministic per-detection keys from recording, frame, class, and bbox.

    The normal key payload is content-derived. If two rows have identical payloads,
    a detect-time duplicate ordinal is appended to those duplicate payloads so the
    final key remains unique within the run. Downstream stages must copy the key;
    they must not recompute it.

    ``payload_context`` namespaces the hash payload for keys legitimately minted
    at a later point of origin (e.g. ``INSTANCE_KEY_CONTEXT_MANUAL_CURATION`` for
    hand-drawn curation boxes), so contexted keys cannot collide with detect-time
    keys minted from identical content. When ``None`` (the default) the payload is
    bit-identical to the historical detect-time format.
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
    context = str(payload_context).strip() if payload_context is not None else ""
    if context:
        base_payloads = [f"{payload}|context={context}" for payload in base_payloads]
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


def mint_manual_curation_instance_keys(
    *,
    recording_identity: str,
    refined_row_ids: np.ndarray,
    frame_indices: np.ndarray,
    bbox_norm_coords: np.ndarray,
    class_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Mint keys once for new manual observations using their stable row IDs.

    ``refined_row_id`` is included in the manual-origin namespace so deleting
    and later recreating an otherwise identical box cannot reuse the retired
    observation key. Surviving observations must copy their stored key rather
    than call this function again after an edit.
    """

    row_ids = np.asarray(refined_row_ids, dtype=np.int64).reshape(-1)
    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    bboxes = np.asarray(bbox_norm_coords, dtype=np.float64).reshape(-1, 4)
    classes = (
        np.zeros(frames.shape[0], dtype=np.int64)
        if class_ids is None
        else np.asarray(class_ids, dtype=np.int64).reshape(-1)
    )
    if not (row_ids.shape[0] == frames.shape[0] == bboxes.shape[0] == classes.shape[0]):
        raise ValueError("Manual curation key inputs must agree on row count.")
    if np.any(row_ids < 0):
        raise ValueError("Manual curation keys require assigned non-negative refined_row_ids.")
    if int(np.unique(row_ids).shape[0]) != int(row_ids.shape[0]):
        raise ValueError("Manual curation keys require unique refined_row_ids.")

    out = np.empty(row_ids.shape[0], dtype=np.uint64)
    for idx, row_id in enumerate(row_ids.tolist()):
        out[idx] = mint_detection_instance_keys(
            recording_identity=recording_identity,
            frame_indices=frames[idx : idx + 1],
            bbox_norm_coords=bboxes[idx : idx + 1],
            class_ids=classes[idx : idx + 1],
            payload_context=(
                f"{INSTANCE_KEY_CONTEXT_MANUAL_CURATION_ROW_ID}:"
                f"refined_row_id={int(row_id)}"
            ),
        )[0]

    if int(np.unique(out).shape[0]) != int(out.shape[0]):
        raise ValueError("Manual curation instance_key hash collision detected.")
    return out


def instance_key_attrs(
    recording_identity: str,
    *,
    frame_domain: str = "run_frame_index",
    frame_mapping_source: str | None = None,
    frame_mapping_sha256: str | None = None,
) -> dict[str, Any]:
    """Return attrs documenting the instance-key minting policy."""

    attrs: dict[str, Any] = {
        "instance_key_algorithm": INSTANCE_KEY_ALGORITHM,
        "instance_key_recording_identity": str(recording_identity),
        "instance_key_frame_domain": str(frame_domain),
        "instance_key_bbox_quantization": int(INSTANCE_KEY_BBOX_QUANTIZATION),
        "instance_key_duplicate_policy": INSTANCE_KEY_DUPLICATE_POLICY,
    }
    if frame_mapping_source:
        attrs["instance_key_frame_mapping_source"] = str(frame_mapping_source)
    if frame_mapping_sha256:
        attrs["instance_key_frame_mapping_sha256"] = str(frame_mapping_sha256)
    return attrs
