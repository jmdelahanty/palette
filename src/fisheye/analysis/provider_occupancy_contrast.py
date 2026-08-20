"""Pure, strict recording-level occupancy contrasts.

This module deliberately has no Zarr, registry, selector, or cohort surface.
It consumes two already validated immutable provider-neutral occupancy
summaries and returns only the treatment-minus-baseline difference.  The
summary contract is intentionally represented as strict JSON-compatible
metadata plus a numeric ``occupancy_fraction`` array so this function can be
used by a publisher without opening or rewriting either source run.

The required summary fields are documented by ``_COMPATIBILITY_FIELDS`` and
the source identity helpers below.  Identity records must contain an explicit
identifier and a lowercase SHA-256 digest.  If an identity record includes a
``payload`` and a digest, the digest is checked against that payload; this is
what makes a stale in-memory manifest fail closed rather than merely comparing
two copied labels.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SCHEMA_ID = "palette.provider_occupancy_contrast"
SCHEMA_VERSION = 1
OPERATION = "difference"
CONTRAST_FIELD = "occupancy_fraction_difference"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These fields are compared by exact canonical-JSON identity.  Grid edges are
# kept outside this tuple because they are numeric arrays and require exact
# array comparison rather than JSON object equality.
_COMPATIBILITY_FIELDS: tuple[str, ...] = (
    "schema_family",
    "schema_version",
    "provider_id",
    "estimator",
    "position_track_policy",
    "coordinate_frame",
    "transform",
    "geometry",
    "sample_unit",
    "denominator",
    "normalization",
    "recording_id",
    "subject_id",
    "timing_authority",
    "grid_policy",
    "edge_policy",
)

_EXACT_IDENTITY_COMPATIBILITY_FIELDS = frozenset(
    {
        "estimator",
        "position_track_policy",
        "coordinate_frame",
        "transform",
        "geometry",
        "timing_authority",
    }
)

_IDENTITY_DIGEST_KEYS: tuple[str, ...] = (
    "sha256",
    "manifest_sha256",
    "payload_digest",
    "digest",
)
_IDENTITY_ID_KEYS: tuple[str, ...] = (
    "id",
    "identity",
    "manifest_id",
    "selection_id",
    "occurrence_id",
    "run_id",
    "run_ref",
    "path",
)


class OccupancyContrastError(ValueError):
    """Raised when an occupancy contrast cannot be proven compatible."""


def _error(field: str, message: str) -> OccupancyContrastError:
    return OccupancyContrastError(f"{field}: {message}")


def _strict_copy(value: Any, *, field: str) -> Any:
    """Return a detached strict-JSON copy, rejecting NaN and non-JSON values."""

    try:
        # Going through canonical JSON both validates the value and avoids
        # retaining mutable nested source metadata in the returned result.
        import json

        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise _error(field, "must be strict JSON") from exc


def _require_mapping(summary: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = summary.get(field)
    if not isinstance(value, Mapping):
        raise _error(field, "must be one nonempty mapping")
    if not value:
        raise _error(field, "must not be empty")
    return value


def _require_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise _error(field, "must be one nonempty canonical string")
    return value


def _require_digest(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _error(field, "must be one lowercase SHA-256 digest")
    return value


def _canonical_identity(value: Any, *, field: str) -> dict[str, Any]:
    """Validate and detach one manifest/selection/occurrence identity.

    A digest is mandatory even when the source is represented by a path or run
    reference.  Optional payload verification catches a stale digest in a
    hand-built summary and gives callers a useful failure instead of silently
    accepting a changed source under an unchanged label.
    """

    if not isinstance(value, Mapping) or not value:
        raise _error(field, "must be one nonempty identity mapping")

    identity_key = next((key for key in _IDENTITY_ID_KEYS if key in value), None)
    digest_key = next((key for key in _IDENTITY_DIGEST_KEYS if key in value), None)
    if identity_key is None:
        raise _error(field, "is missing an explicit identity")
    if digest_key is None:
        raise _error(field, "is missing an explicit SHA-256 digest")

    identity = _require_text(value[identity_key], field=f"{field}.{identity_key}")
    digest = _require_digest(value[digest_key], field=f"{field}.{digest_key}")

    for key in _IDENTITY_ID_KEYS:
        if key in value:
            _require_text(value[key], field=f"{field}.{key}")
    for key in _IDENTITY_DIGEST_KEYS:
        if key in value:
            _require_digest(value[key], field=f"{field}.{key}")
            if value[key] != digest:
                raise _error(field, "contains conflicting digest identities")

    if "payload" in value:
        payload = _strict_copy(value["payload"], field=f"{field}.payload")
        if canonical_json_sha256(payload) != digest:
            raise _error(field, "payload digest is stale")

    result = _strict_copy(value, field=field)
    if not isinstance(result, dict):  # pragma: no cover - defensive
        raise _error(field, "must remain one mapping after canonicalization")
    # Ensure the selected aliases are retained in the returned record while
    # making the identity requirement visible to downstream consumers.
    result["identity"] = identity
    result["sha256"] = digest
    return result


def _canonical_occurrences(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _error("source_occurrences", "must be one nonempty sequence")
    if not value:
        raise _error("source_occurrences", "must not be empty")
    return [
        _canonical_identity(item, field=f"source_occurrences[{index}]")
        for index, item in enumerate(value)
    ]


def _canonical_selections(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = summary.get("source_selections")
    if value is None and "source_selection" in summary:
        value = [summary["source_selection"]]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _error("source_selections", "must be one nonempty sequence")
    if not value:
        raise _error("source_selections", "must not be empty")
    return [
        _canonical_identity(item, field=f"source_selections[{index}]")
        for index, item in enumerate(value)
    ]


def _canonical_compatibility_value(value: Any, *, field: str) -> Any:
    if type(value) is str:
        return _require_text(value, field=field)
    if isinstance(value, Mapping):
        if not value:
            raise _error(field, "must not be empty")
        return _strict_copy(value, field=field)
    if isinstance(value, (list, tuple)):
        if not value:
            raise _error(field, "must not be empty")
        return _strict_copy(value, field=field)
    raise _error(field, "must be a nonempty string, mapping, or sequence")


def _canonical_grid(summary: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    grid = _require_mapping(summary, "grid")
    if set(grid) != {"id", "sha256", "x_edges", "y_edges"}:
        raise _error(
            "grid",
            "must contain exactly id, sha256, x_edges, and y_edges",
        )
    grid_id = _require_text(grid["id"], field="grid.id")
    grid_sha256 = _require_digest(grid["sha256"], field="grid.sha256")
    try:
        x_edges = np.asarray(grid["x_edges"], dtype=np.float64)
        y_edges = np.asarray(grid["y_edges"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _error("grid", "edges must be finite numeric sequences") from exc
    if x_edges.ndim != 1 or y_edges.ndim != 1 or x_edges.size < 2 or y_edges.size < 2:
        raise _error("grid", "edges must be one-dimensional sequences with at least two values")
    if not np.isfinite(x_edges).all() or not np.isfinite(y_edges).all():
        raise _error("grid", "edges must be finite")
    if not np.all(np.diff(x_edges) > 0) or not np.all(np.diff(y_edges) > 0):
        raise _error("grid", "edges must be strictly increasing")

    grid_payload = {
        "id": grid_id,
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
    }
    if canonical_json_sha256(grid_payload) != grid_sha256:
        raise _error("grid", "digest is stale")
    grid_for_copy = {**grid_payload, "sha256": grid_sha256}
    grid_copy = _strict_copy(grid_for_copy, field="grid")
    if not isinstance(grid_copy, dict):  # pragma: no cover - defensive
        raise _error("grid", "must remain one mapping")
    # Normalize only the edge representation; values are retained exactly as
    # float64 in the result and compared by np.array_equal below.
    grid_copy["x_edges"] = x_edges.tolist()
    grid_copy["y_edges"] = y_edges.tolist()
    return grid_copy, x_edges, y_edges


def _canonical_fraction(summary: Mapping[str, Any]) -> np.ndarray:
    if "occupancy_fraction" not in summary:
        raise _error("occupancy_fraction", "is required")
    try:
        values = np.asarray(summary["occupancy_fraction"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _error("occupancy_fraction", "must be a numeric array") from exc
    if values.ndim != 2 or values.size == 0:
        raise _error("occupancy_fraction", "must be a nonempty two-dimensional array")
    if not np.isfinite(values).all():
        raise _error("occupancy_fraction", "must contain only finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise _error("occupancy_fraction", "values must lie in [0, 1]")
    return np.array(values, dtype=np.float64, copy=True)


def _positive_count(summary: Mapping[str, Any]) -> int:
    value = summary.get("valid_sample_count")
    if type(value) is not int or value <= 0:
        raise _error("valid_sample_count", "must be one positive exact integer")
    return value


def _validate_summary(summary: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    if not isinstance(summary, Mapping) or not summary:
        raise _error(arm, "must be one nonempty occupancy summary mapping")

    role = _require_text(summary.get("arm_role"), field=f"{arm}.arm_role")
    expected_role = "baseline" if arm == "baseline" else "treatment"
    if role != expected_role:
        raise _error(f"{arm}.arm_role", f"must be {expected_role!r}")

    grid, _x_edges, _y_edges = _canonical_grid(summary)
    normalized: dict[str, Any] = {
        "arm_role": role,
        "occupancy_fraction": _canonical_fraction(summary),
        "valid_sample_count": _positive_count(summary),
        "grid": grid,
        "source_manifest": _canonical_identity(
            summary.get("source_manifest"), field=f"{arm}.source_manifest"
        ),
        "source_selections": _canonical_selections(summary),
        "source_occurrences": _canonical_occurrences(summary.get("source_occurrences")),
    }

    for field in _COMPATIBILITY_FIELDS:
        if field not in summary:
            raise _error(f"{arm}.{field}", "is required")
        if field == "schema_version":
            value = summary[field]
            if type(value) is not int or value <= 0:
                raise _error(f"{arm}.{field}", "must be one positive exact integer")
            normalized[field] = value
        elif field in _EXACT_IDENTITY_COMPATIBILITY_FIELDS:
            normalized[field] = _canonical_identity(
                summary[field], field=f"{arm}.{field}"
            )
        else:
            normalized[field] = _canonical_compatibility_value(
                summary[field], field=f"{arm}.{field}"
            )

    # The summary schema must identify an exact version, not merely a display
    # label.  This is kept separate from schema_family so both can be compared.
    # Store the source occurrence and selection records exactly as detached
    # JSON, but deliberately do not retain source counts or other arrays.
    return normalized


def _require_equal(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any], field: str
) -> None:
    if canonical_json_sha256(baseline[field]) != canonical_json_sha256(treatment[field]):
        raise _error(field, "baseline and treatment identities differ")


def _require_exact_grid(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    baseline_grid = baseline["grid"]
    treatment_grid = treatment["grid"]
    baseline_x = np.asarray(baseline_grid["x_edges"], dtype=np.float64)
    baseline_y = np.asarray(baseline_grid["y_edges"], dtype=np.float64)
    treatment_x = np.asarray(treatment_grid["x_edges"], dtype=np.float64)
    treatment_y = np.asarray(treatment_grid["y_edges"], dtype=np.float64)
    if not np.array_equal(baseline_x, treatment_x) or not np.array_equal(
        baseline_y, treatment_y
    ):
        raise _error("grid.x_edges/y_edges", "baseline and treatment edges differ")
    if baseline["occupancy_fraction"].shape != treatment["occupancy_fraction"].shape:
        raise _error("occupancy_fraction.shape", "baseline and treatment shapes differ")
    expected_shape = (baseline_y.size - 1, baseline_x.size - 1)
    if baseline["occupancy_fraction"].shape != expected_shape:
        raise _error(
            "occupancy_fraction.shape",
            f"must equal the grid bin shape {expected_shape!r}",
        )
    return baseline_x, baseline_y


def compute_occupancy_contrast(
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute one strict ``treatment - baseline`` occupancy contrast.

    ``baseline`` and ``treatment`` are keyword-compatible positional arms but
    the names are part of this function's contract.  Their summaries must
    carry matching exact identities and ``arm_role`` values.  ``config`` is
    metadata for the fixed v1 policy only; it cannot select another operation,
    formula, normalization, or cohort aggregation.

    The returned ``occupancy_fraction_difference`` is a detached float64 array.
    Source occupancy arrays and source metadata are never modified or returned
    by reference.
    """

    baseline_value = _validate_summary(baseline, arm="baseline")
    treatment_value = _validate_summary(treatment, arm="treatment")

    for field in _COMPATIBILITY_FIELDS:
        _require_equal(baseline_value, treatment_value, field)
    baseline_x, baseline_y = _require_exact_grid(baseline_value, treatment_value)

    config_value: dict[str, Any]
    if config is None:
        config_value = {}
    elif not isinstance(config, Mapping):
        raise _error("config", "must be a mapping when provided")
    else:
        config_value = _strict_copy(config, field="config")
        if not isinstance(config_value, dict):  # pragma: no cover - defensive
            raise _error("config", "must remain one mapping")
        reserved = {
            "operation",
            "formula",
            "normalization",
            "cohort",
            "aggregation",
        }
        if reserved.intersection(config_value):
            raise _error(
                "config",
                "cannot override the fixed v1 difference operation or add cohort aggregation",
            )

    policy = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "operation": OPERATION,
        "formula": "treatment.occupancy_fraction - baseline.occupancy_fraction",
        "normalization": "source_occupancy_fraction_unchanged",
        "arm_roles": {"baseline": "baseline", "treatment": "treatment"},
        "compatibility_fields": list(_COMPATIBILITY_FIELDS),
        "grid_comparison": "exact_float64_edges_v1",
        "source_identity": "explicit_sha256_manifest_selection_occurrence_v1",
        "config": config_value,
    }
    policy_digest = canonical_json_sha256(policy)

    difference = np.subtract(
        treatment_value["occupancy_fraction"],
        baseline_value["occupancy_fraction"],
        dtype=np.float64,
    )

    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "operation": OPERATION,
        "formula": policy["formula"],
        "baseline_role": "baseline",
        "treatment_role": "treatment",
        "occupancy_fraction_difference": np.array(difference, dtype=np.float64, copy=True),
        "x_edges": np.array(baseline_x, dtype=np.float64, copy=True),
        "y_edges": np.array(baseline_y, dtype=np.float64, copy=True),
        "valid_sample_counts": {
            "baseline": baseline_value["valid_sample_count"],
            "treatment": treatment_value["valid_sample_count"],
        },
        "source_arms": {
            "baseline": {
                "role": "baseline",
                "source_manifest": deepcopy(baseline_value["source_manifest"]),
                "source_selections": deepcopy(baseline_value["source_selections"]),
                "source_occurrences": deepcopy(baseline_value["source_occurrences"]),
            },
            "treatment": {
                "role": "treatment",
                "source_manifest": deepcopy(treatment_value["source_manifest"]),
                "source_selections": deepcopy(treatment_value["source_selections"]),
                "source_occurrences": deepcopy(treatment_value["source_occurrences"]),
            },
        },
        "policy": policy,
        "policy_digest": policy_digest,
        "config_digest": canonical_json_sha256(config_value),
    }


build_provider_occupancy_contrast = compute_occupancy_contrast


__all__ = [
    "CONTRAST_FIELD",
    "OPERATION",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "OccupancyContrastError",
    "build_provider_occupancy_contrast",
    "compute_occupancy_contrast",
]
