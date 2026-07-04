"""Strict-JSON safety helpers for Zarr attrs and plot specs."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def decode_fixed_width_bytes(value: bytes | bytearray | np.bytes_, *, errors: str = "replace") -> str:
    """Decode UTF-8 bytes commonly stored as null-terminated fixed-width text."""

    return bytes(value).split(b"\x00", 1)[0].decode("utf-8", errors=errors)


def decode_null_terminated_text(value: Any, *, errors: str = "replace") -> str:
    """Decode Zarr/H5 scalar or fixed-width row text with null termination.

    This intentionally handles only text-like scalar values and uint8/int rows.
    Domain-specific decoders should stay local when they need event parsing,
    probability decoding, enum mapping, or fallback ordering.
    """

    if isinstance(value, str):
        return value.split("\x00", 1)[0]
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return decode_fixed_width_bytes(value, errors=errors)

    arr = np.asarray(value)
    if arr.ndim == 0:
        return decode_null_terminated_text(arr.item(), errors=errors)
    if arr.dtype.kind in ("u", "i"):
        raw: list[int] = []
        for item in arr.ravel():
            byte = int(item)
            if byte == 0:
                break
            raw.append(byte)
        return bytes(raw).decode("utf-8", errors=errors)
    if arr.dtype.kind == "S":
        return decode_fixed_width_bytes(arr.tobytes(), errors=errors)
    return str(value).split("\x00", 1)[0]


def json_attr_safe(value: Any) -> Any:
    """Return a value safe for strict JSON serialization in Zarr metadata.

    This helper is for attrs, plot specs, and report payloads. It is not the
    canonical run-lineage fingerprint normalizer, which intentionally has its
    own Unicode and transient-key rules.
    """

    if isinstance(value, np.generic):
        return json_attr_safe(value.item())
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return decode_null_terminated_text(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_attr_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_attr_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_attr_safe(item) for item in value]
    return str(value)


def json_attr_safe_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return a string-keyed strict-JSON-safe mapping."""

    return {str(key): json_attr_safe(value) for key, value in values.items()}


def strict_json_dumps(
    value: Any,
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
    separators: tuple[str, str] = (",", ":"),
) -> str:
    """Serialize with strict JSON rules after applying ``json_attr_safe``."""

    return json.dumps(
        json_attr_safe(value),
        allow_nan=False,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        separators=separators,
    )


def _atomic_text_write(path: Path, text: str, *, create_parents: bool = True) -> None:
    path = path.expanduser()
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = True,
    indent: int | None = 2,
    ensure_ascii: bool = True,
    trailing_newline: bool = True,
    create_parents: bool = True,
) -> None:
    """Atomically write one strict-JSON document."""

    text = json.dumps(
        json_attr_safe(payload),
        allow_nan=False,
        sort_keys=sort_keys,
        indent=indent,
        ensure_ascii=ensure_ascii,
    )
    if trailing_newline:
        text += "\n"
    _atomic_text_write(path, text, create_parents=create_parents)


def write_jsonl_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
    create_parents: bool = True,
) -> None:
    """Atomically write newline-delimited strict-JSON rows."""

    lines = [
        json.dumps(
            json_attr_safe_mapping(row),
            allow_nan=False,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
        )
        for row in rows
    ]
    text = "".join(f"{line}\n" for line in lines)
    _atomic_text_write(path, text, create_parents=create_parents)


__all__ = [
    "decode_fixed_width_bytes",
    "decode_null_terminated_text",
    "json_attr_safe",
    "json_attr_safe_mapping",
    "strict_json_dumps",
    "write_json_atomic",
    "write_jsonl_atomic",
]
