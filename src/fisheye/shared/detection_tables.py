"""Compatibility-neutral access to detection instance tables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

DETECTION_INSTANCE_REQUIRED_ARRAYS = (
    "frame_indices",
    "bbox_norm_coords",
)

_BBOX_CENTER_DERIVATION_SCHEMA = "palette.bbox_center_derivation"
_BBOX_CENTER_DERIVATION_VERSION = 2
_BBOX_CENTER_DERIVATION_OPERATION = (
    "half_open_xyxy_edges_to_continuous_midpoint_v2"
)


def _pixel_authority_pointer(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    record_ref = str(value.get("record_ref") or "").strip()
    digest = str(value.get("record_sha256") or "").strip().lower()
    digest = digest.removeprefix("sha256:")
    if not record_ref:
        raise ValueError(f"{label} lacks record_ref.")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} has an invalid record_sha256.")
    return {"record_ref": record_ref, "record_sha256": digest}


def resolve_detection_source_pixel_authority(
    run_attrs: Mapping[str, Any],
) -> dict[str, str] | None:
    """Resolve legacy or canonical-v2 source-camera point authority.

    Legacy canonical adapters carry ``source_evidence.source_pixel_authority``.
    Native canonical-v2 detection runs bind the same continuous point frame as
    ``bbox_center_derivation.destination_frame``.  When both are present they
    must agree exactly after SHA-256 prefix normalization.
    """

    pointers: list[dict[str, str]] = []
    source_evidence = run_attrs.get("source_evidence")
    if isinstance(source_evidence, Mapping) and (
        "source_pixel_authority" in source_evidence
    ):
        pointers.append(
            _pixel_authority_pointer(
                source_evidence.get("source_pixel_authority"),
                label="detection source_evidence source_pixel_authority",
            )
        )

    derivation = run_attrs.get("bbox_center_derivation")
    if isinstance(derivation, Mapping):
        if (
            derivation.get("schema_id") != _BBOX_CENTER_DERIVATION_SCHEMA
            or derivation.get("schema_version") != _BBOX_CENTER_DERIVATION_VERSION
            or derivation.get("operation") != _BBOX_CENTER_DERIVATION_OPERATION
        ):
            raise ValueError(
                "Detection bbox_center_derivation has an unsupported coordinate contract."
            )
        pointers.append(
            _pixel_authority_pointer(
                derivation.get("destination_frame"),
                label="detection bbox_center_derivation destination_frame",
            )
        )

    if not pointers:
        return None
    if any(pointer != pointers[0] for pointer in pointers[1:]):
        raise ValueError(
            "Detection source pixel-authority declarations do not agree."
        )
    return pointers[0]


def resolve_detection_instance_table(run: Any) -> Any:
    """Return ``instances`` for strict runs or the legacy run-root table.

    This is a reader boundary only.  It does not create aliases and it never
    changes which run is selected.
    """

    table = run.get("instances")
    if table is not None and all(
        name in table for name in DETECTION_INSTANCE_REQUIRED_ARRAYS
    ):
        return table
    return run


def read_detection_frame_counts(table: Any, *, n_frames: int) -> np.ndarray:
    """Read compatibility counts or derive them from the canonical CSR index."""

    count = int(n_frames)
    if count < 0:
        raise ValueError("n_frames cannot be negative.")
    if "frame_row_offsets" in table:
        offsets = np.asarray(table["frame_row_offsets"][:], dtype=np.int64)
        if offsets.shape != (count + 1,):
            raise ValueError(
                "Detection frame_row_offsets length differs from n_frames + 1."
            )
        if not offsets.size or int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError("Detection frame_row_offsets is malformed.")
        differences = np.diff(offsets)
        if differences.size and int(np.max(differences)) > np.iinfo(np.int32).max:
            raise ValueError("Per-frame detection cardinality exceeds int32.")
        return differences.astype(np.int32, copy=False)
    for name in ("frame_counts", "n_detections"):
        if name in table:
            values = np.asarray(table[name][:], dtype=np.int32)
            if values.shape != (count,):
                raise ValueError(f"Detection {name} length differs from n_frames.")
            return values
    frames = np.asarray(table["frame_indices"][:], dtype=np.int64)
    if frames.size and (np.any(frames < 0) or np.any(frames >= count)):
        raise ValueError("Detection frame_indices are outside n_frames.")
    return np.bincount(frames, minlength=count).astype(np.int32, copy=False)


__all__ = [
    "DETECTION_INSTANCE_REQUIRED_ARRAYS",
    "read_detection_frame_counts",
    "resolve_detection_instance_table",
    "resolve_detection_source_pixel_authority",
]
