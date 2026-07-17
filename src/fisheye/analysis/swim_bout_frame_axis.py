"""Versioned frame-axis references for swim-bout detector traces.

The authoritative frame axis belongs to the exact track-kinematics run used
to compute the detector trace.  New swim-bout runs can reference that array
without copying it.  An embedded copy remains a supported portability fallback
for standalone exports and historical compact-v2 runs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional

import numpy as np

FRAME_AXIS_CONTRACT_ATTR = "frame_axis_contract"
FRAME_AXIS_CONTRACT_SHA256_ATTR = "frame_axis_contract_sha256"
FRAME_AXIS_CONTRACT_SCHEMA_ID = "palette.swim_bout_frame_axis_reference"
FRAME_AXIS_CONTRACT_SCHEMA_VERSION = 1

FRAME_AXIS_STORAGE_REFERENCE = "reference"
FRAME_AXIS_STORAGE_EMBEDDED = "embedded"
FRAME_AXIS_STORAGE_CHOICES = (
    FRAME_AXIS_STORAGE_REFERENCE,
    FRAME_AXIS_STORAGE_EMBEDDED,
)
FRAME_AXIS_STORAGE_DEFAULT = FRAME_AXIS_STORAGE_REFERENCE

FRAME_AXIS_RESOLUTION_AUTHORITATIVE = "authoritative_zarr_path"
FRAME_AXIS_RESOLUTION_EMBEDDED = "embedded_fallback"
FRAME_AXIS_EMBEDDED_PATH = "signals/frame_indices"
FRAME_AXIS_HASH_REPRESENTATION = "little_endian_int64_c_order"


class SwimBoutFrameAxisError(ValueError):
    """Raised when a declared swim-bout frame axis cannot be resolved safely."""


def canonical_frame_axis_sha256(values: np.ndarray) -> str:
    """Hash frame-index values in a platform-independent canonical encoding."""

    canonical = np.asarray(values, dtype="<i8", order="C").reshape(-1)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def build_frame_axis_contract(
    values: np.ndarray,
    *,
    authoritative_path: str,
    source_track_kinematics_run: str,
    track_id: int,
    storage_mode: str = FRAME_AXIS_STORAGE_DEFAULT,
    authoritative_dtype: object = "int64",
) -> dict[str, Any]:
    """Build the persisted same-Zarr frame-axis reference contract."""

    if storage_mode not in FRAME_AXIS_STORAGE_CHOICES:
        expected = ", ".join(FRAME_AXIS_STORAGE_CHOICES)
        raise SwimBoutFrameAxisError(
            f"Unsupported frame-axis storage mode {storage_mode!r}; expected {expected}."
        )

    normalized_path = _normalize_contract_path(authoritative_path)
    run_name = str(source_track_kinematics_run).strip()
    if not run_name or run_name == "latest":
        raise SwimBoutFrameAxisError(
            "Frame-axis references require a resolved track-kinematics run name, not 'latest'."
        )

    frame_values = np.asarray(values, dtype=np.int64).reshape(-1)
    embedded_path = (
        FRAME_AXIS_EMBEDDED_PATH
        if storage_mode == FRAME_AXIS_STORAGE_EMBEDDED
        else None
    )
    resolution_order = [FRAME_AXIS_RESOLUTION_AUTHORITATIVE]
    if embedded_path is not None:
        resolution_order.append(FRAME_AXIS_RESOLUTION_EMBEDDED)

    return {
        "schema_id": FRAME_AXIS_CONTRACT_SCHEMA_ID,
        "schema_version": FRAME_AXIS_CONTRACT_SCHEMA_VERSION,
        "axis_kind": "camera_frame_index",
        "reference_scope": "same_zarr_root",
        "authoritative_path": normalized_path,
        "source_track_kinematics_run": run_name,
        "track_id": int(track_id),
        "shape": [int(frame_values.size)],
        "frame_count": int(frame_values.size),
        "authoritative_dtype": str(np.dtype(authoritative_dtype)),
        "resolved_dtype": "int64",
        "content_sha256": canonical_frame_axis_sha256(frame_values),
        "content_hash_representation": FRAME_AXIS_HASH_REPRESENTATION,
        "storage_mode": storage_mode,
        "embedded_path": embedded_path,
        "resolution_order": resolution_order,
        "reference_stability": "resolved_run_name_not_latest_pointer",
        "missing_authority_policy": (
            "use_declared_embedded_fallback"
            if embedded_path is not None
            else "fail_closed"
        ),
    }


def resolve_swim_bout_frame_axis(
    root: Any,
    run_group: Any,
    *,
    expected_length: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Resolve a swim-bout frame axis through its authoritative path.

    Runs without a frame-axis contract are treated as historical compact-v2
    runs and read from ``signals/frame_indices``.  Contracted runs always try
    the authoritative same-Zarr path first.  They use an embedded fallback only
    when the contract explicitly declares one and the authoritative node is
    absent; malformed or mismatched authoritative nodes fail closed.
    """

    attrs = _attrs_dict(run_group)
    raw_contract = attrs.get(FRAME_AXIS_CONTRACT_ATTR)
    if raw_contract is None:
        legacy = _optional_node(run_group, FRAME_AXIS_EMBEDDED_PATH)
        if legacy is None:
            return None
        return _read_axis(
            legacy,
            label=FRAME_AXIS_EMBEDDED_PATH,
            expected_length=expected_length,
        )
    if not isinstance(raw_contract, Mapping):
        raise SwimBoutFrameAxisError("frame_axis_contract must be a mapping.")

    contract = dict(raw_contract)
    _validate_contract_identity(contract, run_attrs=attrs)
    declared_count = _required_nonnegative_int(contract, "frame_count")
    if expected_length is not None and declared_count != int(expected_length):
        raise SwimBoutFrameAxisError(
            "Frame-axis contract length does not match detector trace: "
            f"contract={declared_count}, detector={int(expected_length)}."
        )

    authoritative_path = _normalize_contract_path(
        _required_text(contract, "authoritative_path")
    )
    authoritative = _optional_node(root, authoritative_path)
    if authoritative is not None:
        _validate_authoritative_metadata(
            authoritative,
            contract=contract,
            authoritative_path=authoritative_path,
        )
        return _read_axis(
            authoritative,
            label=authoritative_path,
            expected_length=declared_count,
        )

    embedded_path = contract.get("embedded_path")
    if embedded_path is None:
        raise SwimBoutFrameAxisError(
            f"Authoritative frame axis {authoritative_path!r} is missing and the "
            "contract declares no embedded fallback."
        )
    normalized_embedded = _normalize_contract_path(str(embedded_path))
    embedded = _optional_node(run_group, normalized_embedded)
    if embedded is None:
        raise SwimBoutFrameAxisError(
            f"Authoritative frame axis {authoritative_path!r} is missing and declared "
            f"embedded fallback {normalized_embedded!r} is also missing."
        )
    _validate_embedded_metadata(
        embedded,
        contract=contract,
        embedded_path=normalized_embedded,
    )
    return _read_axis(
        embedded,
        label=normalized_embedded,
        expected_length=declared_count,
    )


def _validate_contract_identity(
    contract: Mapping[str, Any],
    *,
    run_attrs: Mapping[str, Any],
) -> None:
    if contract.get("schema_id") != FRAME_AXIS_CONTRACT_SCHEMA_ID:
        raise SwimBoutFrameAxisError(
            f"Unsupported frame-axis contract schema_id {contract.get('schema_id')!r}."
        )
    if contract.get("schema_version") != FRAME_AXIS_CONTRACT_SCHEMA_VERSION:
        raise SwimBoutFrameAxisError(
            "Unsupported frame-axis contract schema_version "
            f"{contract.get('schema_version')!r}."
        )
    if contract.get("reference_scope") != "same_zarr_root":
        raise SwimBoutFrameAxisError(
            "Frame-axis reference_scope must be 'same_zarr_root'."
        )
    if contract.get("storage_mode") not in FRAME_AXIS_STORAGE_CHOICES:
        raise SwimBoutFrameAxisError(
            f"Unsupported frame-axis storage_mode {contract.get('storage_mode')!r}."
        )
    storage_mode = str(contract["storage_mode"])
    embedded_path = contract.get("embedded_path")
    expected_resolution_order = [FRAME_AXIS_RESOLUTION_AUTHORITATIVE]
    if storage_mode == FRAME_AXIS_STORAGE_EMBEDDED:
        if embedded_path != FRAME_AXIS_EMBEDDED_PATH:
            raise SwimBoutFrameAxisError(
                "Embedded frame-axis contracts must declare "
                f"embedded_path={FRAME_AXIS_EMBEDDED_PATH!r}."
            )
        expected_resolution_order.append(FRAME_AXIS_RESOLUTION_EMBEDDED)
    elif embedded_path is not None:
        raise SwimBoutFrameAxisError(
            "Reference-only frame-axis contracts must not declare an embedded path."
        )
    if contract.get("resolution_order") != expected_resolution_order:
        raise SwimBoutFrameAxisError(
            "Frame-axis resolution_order disagrees with storage_mode."
        )
    if contract.get("content_hash_representation") != FRAME_AXIS_HASH_REPRESENTATION:
        raise SwimBoutFrameAxisError(
            "Unsupported frame-axis content hash representation."
        )
    digest = _required_text(contract, "content_sha256")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
        raise SwimBoutFrameAxisError(
            "Frame-axis content_sha256 is not a SHA-256 digest."
        )

    source_run = _required_text(contract, "source_track_kinematics_run")
    run_source = run_attrs.get("source_track_kinematics_run")
    if run_source is not None and str(run_source) != source_run:
        raise SwimBoutFrameAxisError(
            "Frame-axis source run disagrees with swim-bout run lineage: "
            f"contract={source_run!r}, run={str(run_source)!r}."
        )
    contract_track_id = _required_nonnegative_int(contract, "track_id")
    run_track_id = run_attrs.get("track_id")
    if run_track_id is not None and int(run_track_id) != contract_track_id:
        raise SwimBoutFrameAxisError(
            "Frame-axis track id disagrees with swim-bout run lineage: "
            f"contract={contract_track_id}, run={int(run_track_id)}."
        )
    frame_count = _required_nonnegative_int(contract, "frame_count")
    if contract.get("shape") != [frame_count]:
        raise SwimBoutFrameAxisError(
            "Frame-axis contract shape must contain exactly frame_count."
        )
    if _required_text(contract, "resolved_dtype") != "int64":
        raise SwimBoutFrameAxisError("Frame-axis resolved_dtype must be 'int64'.")

    authoritative_path = _normalize_contract_path(
        _required_text(contract, "authoritative_path")
    )
    expected_tail = f"/{source_run}/tracks/id_{contract_track_id}/frame_indices"
    if not f"/{authoritative_path}".endswith(expected_tail):
        raise SwimBoutFrameAxisError(
            "Frame-axis authoritative_path disagrees with the declared source run "
            "or track id."
        )


def _validate_authoritative_metadata(
    node: Any,
    *,
    contract: Mapping[str, Any],
    authoritative_path: str,
) -> None:
    shape = getattr(node, "shape", None)
    if shape is None:
        raise SwimBoutFrameAxisError(
            f"Authoritative frame-axis path {authoritative_path!r} is not an array."
        )
    expected_shape = tuple(int(value) for value in contract.get("shape", ()))
    if tuple(int(value) for value in shape) != expected_shape:
        raise SwimBoutFrameAxisError(
            f"Authoritative frame-axis shape mismatch at {authoritative_path!r}: "
            f"expected={expected_shape}, actual={tuple(shape)}."
        )
    expected_dtype = np.dtype(_required_text(contract, "authoritative_dtype"))
    actual_dtype = np.dtype(getattr(node, "dtype", None))
    if actual_dtype != expected_dtype:
        raise SwimBoutFrameAxisError(
            f"Authoritative frame-axis dtype mismatch at {authoritative_path!r}: "
            f"expected={expected_dtype}, actual={actual_dtype}."
        )


def _validate_embedded_metadata(
    node: Any,
    *,
    contract: Mapping[str, Any],
    embedded_path: str,
) -> None:
    shape = getattr(node, "shape", None)
    if shape is None:
        raise SwimBoutFrameAxisError(
            f"Embedded frame-axis path {embedded_path!r} is not an array."
        )
    expected_shape = tuple(int(value) for value in contract.get("shape", ()))
    if tuple(int(value) for value in shape) != expected_shape:
        raise SwimBoutFrameAxisError(
            f"Embedded frame-axis shape mismatch at {embedded_path!r}: "
            f"expected={expected_shape}, actual={tuple(shape)}."
        )
    expected_dtype = np.dtype(_required_text(contract, "resolved_dtype"))
    actual_dtype = np.dtype(getattr(node, "dtype", None))
    if actual_dtype != expected_dtype:
        raise SwimBoutFrameAxisError(
            f"Embedded frame-axis dtype mismatch at {embedded_path!r}: "
            f"expected={expected_dtype}, actual={actual_dtype}."
        )


def _read_axis(
    node: Any,
    *,
    label: str,
    expected_length: Optional[int],
) -> np.ndarray:
    if getattr(node, "shape", None) is None:
        raise SwimBoutFrameAxisError(f"Frame-axis path {label!r} is not an array.")
    values = np.asarray(node[:], dtype=np.int64).reshape(-1)
    if expected_length is not None and values.size != int(expected_length):
        raise SwimBoutFrameAxisError(
            f"Frame-axis length mismatch at {label!r}: "
            f"expected={int(expected_length)}, actual={int(values.size)}."
        )
    return values


def _optional_node(group: Any, path: str) -> Any | None:
    current = group
    for part in _normalize_contract_path(path).split("/"):
        try:
            if part not in current:
                return None
            current = current[part]
        except Exception:
            return None
    return current


def _normalize_contract_path(path: str) -> str:
    normalized = str(path).strip().strip("/")
    parts = normalized.split("/") if normalized else []
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise SwimBoutFrameAxisError(f"Invalid Zarr-relative frame-axis path {path!r}.")
    if "latest" in parts:
        raise SwimBoutFrameAxisError(
            f"Frame-axis path must pin a resolved run name, not 'latest': {path!r}."
        )
    return "/".join(parts)


def _attrs_dict(group: Any) -> dict[str, Any]:
    attrs = getattr(group, "attrs", {})
    if hasattr(attrs, "asdict"):
        return dict(attrs.asdict())
    return dict(attrs)


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise SwimBoutFrameAxisError(f"Frame-axis contract is missing {key!r}.")
    return text


def _required_nonnegative_int(mapping: Mapping[str, Any], key: str) -> int:
    try:
        value = int(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise SwimBoutFrameAxisError(
            f"Frame-axis contract field {key!r} must be an integer."
        ) from exc
    if value < 0:
        raise SwimBoutFrameAxisError(
            f"Frame-axis contract field {key!r} must be non-negative."
        )
    return value


__all__ = [
    "FRAME_AXIS_CONTRACT_ATTR",
    "FRAME_AXIS_CONTRACT_SCHEMA_ID",
    "FRAME_AXIS_CONTRACT_SCHEMA_VERSION",
    "FRAME_AXIS_CONTRACT_SHA256_ATTR",
    "FRAME_AXIS_EMBEDDED_PATH",
    "FRAME_AXIS_STORAGE_CHOICES",
    "FRAME_AXIS_STORAGE_DEFAULT",
    "FRAME_AXIS_STORAGE_EMBEDDED",
    "FRAME_AXIS_STORAGE_REFERENCE",
    "SwimBoutFrameAxisError",
    "build_frame_axis_contract",
    "canonical_frame_axis_sha256",
    "resolve_swim_bout_frame_axis",
]
