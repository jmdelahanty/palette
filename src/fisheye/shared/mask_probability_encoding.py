"""Shared probability-mask encoding validation and decoding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

try:  # pragma: no cover - dask is an optional runtime surface for this helper.
    import dask.array as da
except Exception:  # pragma: no cover
    da = None  # type: ignore[assignment]


PROBABILITY_ENCODING_UNIT_FLOAT = "unit_float"
PROBABILITY_ENCODING_LINEAR_UINT8 = "linear_uint8_0_255"
VALID_PROBABILITY_ENCODINGS = frozenset(
    {
        PROBABILITY_ENCODING_UNIT_FLOAT,
        PROBABILITY_ENCODING_LINEAR_UINT8,
    }
)


def normalize_probabilities_encoding(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in VALID_PROBABILITY_ENCODINGS:
        return text
    return None


def require_probabilities_encoding(
    value: object,
    *,
    source_path: str,
    observed_dtype: object,
) -> str:
    encoding = normalize_probabilities_encoding(value)
    dtype_text = str(np.dtype(observed_dtype))
    if encoding is None:
        raw = str(value or "").strip()
        detail = "missing" if not raw else f"unrecognized value {raw!r}"
        raise ValueError(
            f"{source_path} has {detail} for required probabilities_encoding; "
            f"observed dtype={dtype_text}."
        )
    return encoding


def probabilities_encoding_from_attrs(
    attrs: Mapping[str, object] | Any,
    *,
    source_path: str,
    observed_dtype: object,
) -> str:
    value = attrs.get("probabilities_encoding") if hasattr(attrs, "get") else None
    return require_probabilities_encoding(
        value,
        source_path=source_path,
        observed_dtype=observed_dtype,
    )


def decode_probability_values(
    values: Any,
    *,
    encoding: object,
    source_path: str,
) -> Any:
    arr = values if _is_dask_array(values) else np.asarray(values)
    dtype = np.dtype(arr.dtype)
    resolved = require_probabilities_encoding(
        encoding,
        source_path=source_path,
        observed_dtype=dtype,
    )
    _validate_probability_dtype(resolved, dtype=dtype, source_path=source_path)
    if resolved == PROBABILITY_ENCODING_LINEAR_UINT8:
        decoded = arr.astype(np.float32) / np.float32(255.0)
    elif _is_dask_array(arr):
        decoded = arr.astype(np.float32)
    else:
        decoded = arr.astype(np.float32, copy=False)
    return _clip_probability_values(decoded)


def decode_probability_values_from_attrs(
    values: Any,
    *,
    attrs: Mapping[str, object] | Any,
    source_path: str,
) -> Any:
    dtype = np.dtype(getattr(values, "dtype", None) or np.asarray(values).dtype)
    encoding = probabilities_encoding_from_attrs(
        attrs,
        source_path=source_path,
        observed_dtype=dtype,
    )
    return decode_probability_values(values, encoding=encoding, source_path=source_path)


def _validate_probability_dtype(encoding: str, *, dtype: np.dtype, source_path: str) -> None:
    if encoding == PROBABILITY_ENCODING_LINEAR_UINT8:
        if dtype != np.dtype(np.uint8):
            raise ValueError(
                f"{source_path} declares probabilities_encoding={encoding!r}, "
                f"which requires uint8 storage; observed dtype={dtype}."
            )
        return
    if encoding == PROBABILITY_ENCODING_UNIT_FLOAT:
        if not np.issubdtype(dtype, np.floating):
            raise ValueError(
                f"{source_path} declares probabilities_encoding={encoding!r}, "
                f"which requires floating-point storage; observed dtype={dtype}."
            )
        return
    raise AssertionError(f"Unexpected probability encoding after validation: {encoding}")


def _is_dask_array(value: Any) -> bool:
    return da is not None and isinstance(value, da.Array)


def _clip_probability_values(values: Any) -> Any:
    if _is_dask_array(values):
        assert da is not None
        cleaned = da.where(da.isnan(values), np.float32(0.0), values)
        cleaned = da.where(da.logical_and(da.isinf(cleaned), cleaned > 0), np.float32(1.0), cleaned)
        cleaned = da.where(da.logical_and(da.isinf(cleaned), cleaned < 0), np.float32(0.0), cleaned)
        return da.clip(cleaned, np.float32(0.0), np.float32(1.0))
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0, out=arr)
