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


TRANSIENT_LINEAGE_KEYS = {
    "artifacts",
    "command",
    "created_at_utc",
    "created_utc",
    "environment",
    "git_branch",
    "host",
    "hostname",
    "latest",
    "operator",
    "output",
    "output_path",
    "platform",
    "run_name",
    "run_path",
    "scheduler",
    "timestamp_utc",
    "wall_time_s",
    "wall_time_seconds",
    "zarr_path",
}

TOP_LEVEL_PARAMETER_KEYS = {
    "boundary_frame_rounding_policy",
    "boundary_mode",
    "boundary_window_s",
    "default_heading_level",
    "default_level",
    "detection_method",
    "duration_frame_rounding_policy",
    "effective_min_bout_duration_s",
    "effective_min_gap_duration_s",
    "exponential_source_level",
    "exponential_tau_s",
    "fps",
    "gap_merge_min_gap_duration_s",
    "gap_merge_min_gap_source",
    "gap_merge_policy",
    "gap_merge_policy_active",
    "hysteresis_band_policy",
    "hysteresis_enabled",
    "hysteresis_high_px",
    "hysteresis_low_px",
    "hysteresis_min_frames",
    "method",
    "min_bout_duration_s",
    "min_gap_duration_s",
    "min_gap_frames",
    "moving_threshold_mm_s",
    "physical_active_boundary_constraint",
    "physical_active_boundary_margin_s",
    "physical_active_boundary_policy",
    "physical_active_signal_array",
    "physical_active_signal_level",
    "physical_active_threshold_mm_s",
    "pixel_to_mm",
    "projection_deadzone",
    "projection_speed_deadzone_mm_s",
    "radial_singularity_epsilon_mm",
    "resolved_boundary_window_frames",
    "resolved_min_bout_frames",
    "resolved_min_gap_frames",
    "shape_split_policy",
    "source_track_id",
    "stimulus_radial_polarity",
    "stimulus_radial_polarity_source",
    "stimulus_radial_sign",
    "threshold_crossing_interpolation",
    "track_id",
    "window_lengths_s",
}

TOP_LEVEL_PARAMETER_PREFIXES = (
    "early_",
    "hysteresis_",
    "min_peak_",
    "peak_",
    "smooth",
    "smoothing_",
    "threshold_",
)


def _path_join(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _coerce_mapping(value: Any) -> dict[str, Any]:
    value = _parse_jsonish(value)
    return dict(value) if isinstance(value, Mapping) else {}


def _attrs_dict(group: Any) -> dict[str, Any]:
    attrs = getattr(group, "attrs", {})
    try:
        return {str(key): _parse_jsonish(value) for key, value in attrs.items()}
    except AttributeError:
        return {}


def _strip_transient(value: Any) -> Any:
    value = _parse_jsonish(value)
    if isinstance(value, Mapping):
        return {
            str(key): _strip_transient(item)
            for key, item in value.items()
            if str(key) not in TRANSIENT_LINEAGE_KEYS
        }
    if isinstance(value, list):
        return [_strip_transient(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_transient(item) for item in value]
    return value


def _is_parameter_attr(key: str) -> bool:
    return key in TOP_LEVEL_PARAMETER_KEYS or key.startswith(TOP_LEVEL_PARAMETER_PREFIXES)


def _source_ref_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    refs = _coerce_mapping(attrs.get("source_refs"))
    for key, value in attrs.items():
        if not key.startswith("source_"):
            continue
        if key in {"source_fingerprint", "source_lineage_hash"}:
            continue
        refs[key] = value
    return _strip_transient(refs)


def _source_fingerprint_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if "fingerprint" not in key:
            continue
        if key in LINEAGE_ATTR_NAMES:
            continue
        out[key] = value
    refs = _coerce_mapping(attrs.get("source_refs"))
    for key, value in refs.items():
        if "fingerprint" in str(key):
            out[str(key)] = value
    return _strip_transient(out)


def _parameter_attrs(attrs: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _coerce_mapping(provenance.get("parameters"))
    parameters.update(_coerce_mapping(attrs.get("parameters")))
    for key, value in attrs.items():
        if _is_parameter_attr(key):
            parameters[key] = value
    return _strip_transient(parameters)


def _code_attrs(attrs: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    git = _coerce_mapping(provenance.get("git"))
    commit = git.get("commit") or git.get("commit_hash") or attrs.get("git_commit")
    dirty = git.get("is_dirty")
    if dirty is None:
        dirty = attrs.get("git_dirty") or attrs.get("git_is_dirty")
    code = {
        "git_commit": commit,
        "git_dirty": dirty,
    }
    return {key: value for key, value in code.items() if value is not None}


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


def build_best_effort_run_lineage_payload(run_family: str, run_group: Any) -> dict[str, Any]:
    """Build a best-effort lineage payload from attrs on a Zarr-like run group.

    This is intentionally conservative: it captures schema/method/source refs,
    parameters, and git state exposed by existing writer attrs, while stripping
    transient execution details. It is suitable for new v1 writer hooks and for
    metadata backfills where complete source-content hashes are not yet
    available.
    """

    attrs = _attrs_dict(run_group)
    provenance = _coerce_mapping(attrs.get("provenance"))
    analysis_schema = {
        "schema_id": attrs.get("schema_id"),
        "schema_version": attrs.get("schema_version") or attrs.get("output_schema_version"),
        "row_axis": attrs.get("row_axis"),
    }
    method = attrs.get("method") or attrs.get("detection_method") or provenance.get("stage")
    method_version = attrs.get("method_version") or provenance.get("version")
    return build_run_lineage_payload(
        run_family=run_family,
        analysis_schema=analysis_schema,
        method=None if method is None else str(method),
        method_version=None if method_version is None else str(method_version),
        source_refs=_source_ref_attrs(attrs),
        source_fingerprints=_source_fingerprint_attrs(attrs),
        parameters=_parameter_attrs(attrs, provenance),
        code=_code_attrs(attrs, provenance),
    )


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


def write_best_effort_run_lineage_attrs(
    run_group: Any,
    *,
    run_family: str,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write best-effort lineage fingerprint attrs for a newly created run."""

    payload = build_best_effort_run_lineage_payload(run_family, run_group)
    return write_run_lineage_attrs(
        run_group,
        payload,
        fingerprint_status="best_effort",
        overwrite=overwrite,
    )
