"""Run-level lineage fingerprint helpers.

These helpers produce compact, deterministic fingerprints for derived-analysis
runs. They intentionally hash the scientific dependency state, not operational
details such as timestamps, hostnames, or output paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # NumPy is a project dependency, but keep this helper import-tolerant.
    import numpy as np
except Exception:  # pragma: no cover - only relevant in broken environments.
    np = None  # type: ignore[assignment]


LINEAGE_PAYLOAD_SCHEMA_ID = "palette.run_lineage_fingerprint_payload"
LINEAGE_PAYLOAD_SCHEMA_VERSION = 1
LINEAGE_ATTR_SCHEMA_ID = "palette.run_lineage_fingerprint_attrs"
LINEAGE_ATTR_SCHEMA_VERSION = 1
LINEAGE_CANONICALIZATION = "json_sorted_keys_run_lineage_v1"

FINGERPRINT_STATUSES = {"complete", "best_effort", "missing"}

LINEAGE_ATTR_NAMES = (
    "source_fingerprint",
    "source_lineage_hash",
    "lineage_hash",
    "fingerprint_status",
    "lineage_fingerprint_schema_id",
    "lineage_fingerprint_schema_version",
    "lineage_fingerprint_canonicalization",
    "lineage_payload_json",
)


class RunLineageFingerprintError(ValueError):
    """Raised when a run-lineage fingerprint payload is invalid."""


def _path_join(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _numpy_scalar_to_python(value: Any) -> Any:
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def normalize_lineage_value(value: Any, *, path: str = "$") -> Any:
    """Return a strict-JSON-safe, Unicode-normalized lineage value.

    Non-finite floats are normalized to ``None``. This keeps legacy Zarr attrs
    with accidental NaN/Infinity values from poisoning backfilled fingerprints;
    strict JSON serialization with ``allow_nan=False`` is still enforced.
    """

    value = _numpy_scalar_to_python(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.rstrip(b"\x00").decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.ndarray):
        return normalize_lineage_value(value.tolist(), path=path)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            key = unicodedata.normalize("NFC", str(raw_key))
            if key in out:
                raise RunLineageFingerprintError(
                    f"{path}: duplicate key after Unicode normalization: {key!r}"
                )
            out[key] = normalize_lineage_value(raw_item, path=_path_join(path, key))
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            normalize_lineage_value(item, path=_path_join(path, index))
            for index, item in enumerate(value)
        ]
    return str(value)


def canonical_lineage_json(payload: Mapping[str, Any]) -> str:
    """Serialize ``payload`` with the run-lineage canonical JSON rules."""

    normalized = normalize_lineage_value(payload)
    if not isinstance(normalized, Mapping):
        raise RunLineageFingerprintError("run-lineage payload must be a mapping")
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_run_lineage_hash(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 hash for a run-lineage payload."""

    return hashlib.sha256(canonical_lineage_json(payload).encode("utf-8")).hexdigest()


def build_run_lineage_payload(
    *,
    run_family: str,
    analysis_schema: Mapping[str, Any] | None = None,
    method: str | None = None,
    method_version: str | None = None,
    source_refs: Mapping[str, Any] | None = None,
    source_fingerprints: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    code: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical payload that should be hashed for one run.

    ``run_id`` and output path are intentionally excluded. The manifest records
    those separately; the fingerprint answers whether the meaningful source,
    method, schema, code revision, and parameter state is the same.
    """

    payload: dict[str, Any] = {
        "lineage_schema_id": LINEAGE_PAYLOAD_SCHEMA_ID,
        "lineage_schema_version": LINEAGE_PAYLOAD_SCHEMA_VERSION,
        "run_family": str(run_family),
        "analysis_schema": dict(analysis_schema or {}),
        "method": method,
        "method_version": method_version,
        "source_refs": dict(source_refs or {}),
        "source_fingerprints": dict(source_fingerprints or {}),
        "parameters": dict(parameters or {}),
        "code": dict(code or {}),
    }
    return normalize_lineage_value(payload)


def build_run_lineage_attrs(
    payload: Mapping[str, Any],
    *,
    fingerprint_status: str,
) -> dict[str, Any]:
    """Return strict-JSON-safe Zarr attrs for a run-lineage payload."""

    if fingerprint_status not in FINGERPRINT_STATUSES:
        raise RunLineageFingerprintError(
            f"fingerprint_status must be one of {sorted(FINGERPRINT_STATUSES)}, "
            f"got {fingerprint_status!r}"
        )
    lineage_json = canonical_lineage_json(payload)
    lineage_hash = hashlib.sha256(lineage_json.encode("utf-8")).hexdigest()
    return {
        "source_fingerprint": lineage_hash,
        "source_lineage_hash": lineage_hash,
        "lineage_hash": lineage_hash,
        "fingerprint_status": fingerprint_status,
        "lineage_fingerprint_schema_id": LINEAGE_ATTR_SCHEMA_ID,
        "lineage_fingerprint_schema_version": LINEAGE_ATTR_SCHEMA_VERSION,
        "lineage_fingerprint_canonicalization": LINEAGE_CANONICALIZATION,
        "lineage_payload_json": lineage_json,
    }


def write_run_lineage_attrs(
    run_group: Any,
    payload: Mapping[str, Any],
    *,
    fingerprint_status: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write run-lineage attrs to a Zarr-like group and return them.

    Existing fingerprint attrs are preserved unless ``overwrite`` is true. This
    keeps backfills non-destructive by default.
    """

    attrs = build_run_lineage_attrs(payload, fingerprint_status=fingerprint_status)
    target_attrs = run_group.attrs
    for key, value in attrs.items():
        if overwrite or target_attrs.get(key) is None:
            target_attrs[key] = value
    return attrs
