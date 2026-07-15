from __future__ import annotations

from typing import Any, Optional

import numpy as np
import zarr

REASON_BYTES_ENCODING = "utf8-null-terminated"
REASON_BYTES_MIN_WIDTH = 64


def _stamp_reason_contract(group: zarr.Group, width: int) -> None:
    group.attrs["reason_encoding"] = REASON_BYTES_ENCODING
    group.attrs["reason_authority"] = "reason_bytes"
    group.attrs["reason_bytes_width"] = int(width)
    group.attrs["reason_bytes_null_terminated"] = True
    group.attrs["reason_fallback_order"] = ["reason_bytes", "detection_source"]


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
    overwrite: bool = False,
) -> list[str]:
    """Write the canonical fixed-width reason column.

    ``reason`` remains a supported read-only compatibility surface for old
    archives.  New writes intentionally remove that variable-length mirror so
    callers cannot leave two independently mutable reason authorities behind.
    """
    labels = np.asarray(reason, dtype=str).reshape(-1)
    det_chunk = max(1, int(chunk_size))

    reason_bytes = encode_reason_bytes(labels)
    group.create_array(
        "reason_bytes",
        data=reason_bytes,
        chunks=(det_chunk, int(reason_bytes.shape[1])),
        overwrite=overwrite,
    )

    # Publish the new authority before retiring the legacy mirror.  A failed
    # reason_bytes write therefore leaves the historical column untouched.
    if "reason" in group:
        del group["reason"]

    _stamp_reason_contract(group, int(reason_bytes.shape[1]))
    return ["reason_bytes"]


def update_reason_rows(
    group: zarr.Group,
    row_indices: np.ndarray,
    reason: np.ndarray,
) -> None:
    row_indices_arr = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    labels = np.asarray(reason, dtype=str).reshape(-1)
    if row_indices_arr.shape[0] != labels.shape[0]:
        raise ValueError("row_indices and reason must have the same length.")
    if row_indices_arr.size == 0:
        return

    reason_arr = group.get("reason")
    reason_bytes_arr = group.get("reason_bytes")
    if "frame_indices" in group:
        row_count = int(group["frame_indices"].shape[0])
    elif reason_bytes_arr is not None:
        row_count = int(reason_bytes_arr.shape[0])
    elif reason_arr is not None:
        row_count = int(reason_arr.shape[0])
    else:
        row_count = int(np.max(row_indices_arr)) + 1
    if np.any(row_indices_arr < 0) or np.any(row_indices_arr >= row_count):
        raise ValueError("row_indices contain out-of-range values.")

    if reason_bytes_arr is None:
        existing_labels = read_reason_labels(group)
        full_labels = (
            np.asarray(existing_labels, dtype=object).reshape(-1)
            if existing_labels is not None and int(np.asarray(existing_labels).size) == row_count
            else np.full(row_count, "", dtype=object)
        )
        full_labels[row_indices_arr] = np.asarray(labels, dtype=object)
        write_reason_columns(
            group,
            full_labels,
            max(1, row_count),
            overwrite=True,
        )
        return

    width = int(reason_bytes_arr.shape[1]) if len(reason_bytes_arr.shape) > 1 else REASON_BYTES_MIN_WIDTH
    encoded = encode_reason_bytes(labels, min_width=width)
    if encoded.shape[1] > width:
        full_labels = decode_reason_bytes(reason_bytes_arr[:]).reshape(-1)
        full_labels[row_indices_arr] = np.asarray(labels, dtype=object)
        write_reason_columns(
            group,
            full_labels,
            max(1, row_count),
            overwrite=True,
        )
        return

    reason_bytes_arr[row_indices_arr, :] = encoded[:, :width]
    if "reason" in group:
        del group["reason"]
    _stamp_reason_contract(group, width)


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


class MutableReasonColumn:
    """A small 1-D edit adapter backed exclusively by ``reason_bytes``."""

    def __init__(self, group: zarr.Group, values: np.ndarray) -> None:
        self._group = group
        self._values = np.asarray(values, dtype=object).reshape(-1)
        self.shape = self._values.shape
        self.dtype = self._values.dtype

    @property
    def chunks(self) -> tuple[int, ...] | None:
        reason_bytes = self._group.get("reason_bytes")
        if reason_bytes is None or not reason_bytes.chunks:
            return None
        return (int(reason_bytes.chunks[0]),)

    def __len__(self) -> int:
        return int(self._values.shape[0])

    def __getitem__(self, item: Any) -> Any:
        return self._values[item]

    def __setitem__(self, item: Any, value: Any) -> None:
        self._values[item] = value
        selected = np.asarray(np.arange(len(self), dtype=np.int64)[item], dtype=np.int64).reshape(-1)
        if selected.size == 0:
            return
        update_reason_rows(
            self._group,
            selected,
            np.asarray(self._values[selected], dtype=object),
        )


def open_mutable_reason_column(
    group: zarr.Group,
    *,
    chunk_size: int,
) -> Optional[MutableReasonColumn]:
    """Open a mutable logical reason vector while canonicalizing legacy storage."""
    labels = read_reason_labels(group)
    if labels is None:
        return None
    labels = np.asarray(labels, dtype=object).reshape(-1)
    if "reason_bytes" not in group or "reason" in group:
        write_reason_columns(
            group,
            labels,
            chunk_size=max(1, int(chunk_size)),
            overwrite=True,
        )
    else:
        _stamp_reason_contract(group, int(group["reason_bytes"].shape[1]))
    return MutableReasonColumn(group, labels)
