from __future__ import annotations

from typing import Optional

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8

REASON_BYTES_ENCODING = "utf8-null-terminated"
REASON_BYTES_MIN_WIDTH = 16


def _labels_from_detection_source(detection_source: np.ndarray) -> np.ndarray:
    source = np.asarray(detection_source, dtype=np.int8)
    return np.where(source == 1, "interpolated", "clean").astype(object)


def encode_reason_bytes(reason: np.ndarray, *, min_width: int = REASON_BYTES_MIN_WIDTH) -> np.ndarray:
    labels = np.asarray(reason, dtype=str).reshape(-1)
    encoded = [label.encode("utf-8", "ignore") for label in labels.tolist()]
    max_len = max((len(item) for item in encoded), default=0)
    width = max(int(min_width), int(max_len) + 1)
    out = np.zeros((len(encoded), width), dtype=np.uint8)
    for row_idx, payload in enumerate(encoded):
        if not payload:
            continue
        trimmed = payload[: width - 1]
        out[row_idx, : len(trimmed)] = np.frombuffer(trimmed, dtype=np.uint8)
    return out


def decode_reason_bytes(reason_bytes: np.ndarray) -> np.ndarray:
    data = np.asarray(reason_bytes, dtype=np.uint8)
    if data.ndim != 2:
        raise ValueError(f"reason_bytes must be 2D, received shape={data.shape}.")
    labels = []
    for row in data:
        nul = np.where(row == 0)[0]
        end = int(nul[0]) if nul.size else int(row.shape[0])
        labels.append(bytes(row[:end]).decode("utf-8", "ignore"))
    return np.asarray(labels, dtype=object)


def write_reason_columns(
    group: zarr.Group,
    reason: np.ndarray,
    chunk_size: int,
    *,
    include_reason_text: bool = True,
    overwrite: bool = False,
) -> list[str]:
    labels = np.asarray(reason, dtype=str).reshape(-1)
    det_chunk = max(1, int(chunk_size))
    written_fields = ["reason_bytes"]

    reason_bytes = encode_reason_bytes(labels)
    group.create_array(
        "reason_bytes",
        data=reason_bytes,
        chunks=(det_chunk, int(reason_bytes.shape[1])),
        overwrite=overwrite,
    )

    if include_reason_text:
        reason_arr = group.create_array(
            "reason",
            shape=(int(labels.shape[0]),),
            chunks=(det_chunk,),
            dtype=VariableLengthUTF8(),
            fill_value="",
            overwrite=overwrite,
        )
        reason_arr[:] = labels
        written_fields.append("reason")

    group.attrs["reason_encoding"] = REASON_BYTES_ENCODING
    group.attrs["reason_bytes_width"] = int(reason_bytes.shape[1])
    group.attrs["reason_bytes_null_terminated"] = True
    group.attrs["reason_fallback_order"] = ["reason_bytes", "reason", "detection_source"]
    return written_fields


def read_reason_labels(group: zarr.Group) -> Optional[np.ndarray]:
    reason_bytes = group.get("reason_bytes")
    if reason_bytes is not None:
        return decode_reason_bytes(reason_bytes[:])

    reason = group.get("reason")
    if reason is not None:
        return np.asarray(reason[:], dtype=object)

    detection_source = group.get("detection_source")
    if detection_source is not None:
        return _labels_from_detection_source(detection_source[:])

    return None
