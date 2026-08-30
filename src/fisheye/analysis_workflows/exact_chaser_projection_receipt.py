"""Composition receipts for receipt-bound exact-chaser interactive views.

One exact-chaser view joins several independently published immutable children.
Their validation receipts remain lineage-specific.  This record does not replace
those receipts: it binds their exact paths and digests into one closed projection
choice so a consumer cannot discover or mix children at read time.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    read_chaser_relative_frame_validation_receipt,
)
from fisheye.analysis_workflows.exact_immutable_child_validation_receipt import (
    read_exact_immutable_child_validation_receipt,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


RECEIPT_SCHEMA_ID = "palette.analysis.exact_chaser.projection_receipt"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_STATUS = "complete_selector_ineligible_receipt_composition"
VERIFICATION_MODE = "receipt_bound_targeted_array_rehash_v1"
EXACT_CHILD_KEYS = (
    "semantic_selection",
    "keypoint_radial",
    "detection_radial",
    "controller",
    "bout",
    "escape",
    "spatial_occupancy",
)
RELATIVE_CHILD_KEYS = ("keypoint", "detection")
POLICY = {
    "child_choice": "exact_receipt_path_and_record_sha256",
    "child_metadata_revalidation": "once_per_consumed_child_projection_load",
    "consumer_array_validation": "displayed_arrays_content_sha256",
    "archive_root_consolidated_metadata_reparse": False,
    "selector_resolution": False,
    "cache_authority": False,
}
SAFETY = {
    "selector_eligible": False,
    "production_authority": False,
    "registry_update": False,
    "selector_activation": False,
}
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class ExactChaserProjectionReceiptError(ValueError):
    """The projection receipt or one of its exact child receipts is invalid."""


def _fail(message: str) -> None:
    raise ExactChaserProjectionReceiptError(message)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _digest(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if _DIGEST_RE.fullmatch(result) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return result


def _commit(value: object) -> str:
    result = _text(value, field="palette_commit")
    if _COMMIT_RE.fullmatch(result) is None:
        _fail("palette_commit must be one full lowercase Git object ID.")
    return result


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactChaserProjectionReceiptError(
            f"Cannot read strict JSON object {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"JSON document is not one object: {path}")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_receipt_path(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _child_binding(receipt: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "receipt_path": str(path),
        "receipt_sha256": str(receipt["record_sha256"]),
        "run_path": str(receipt["run_path"]),
        "manifest_sha256": str(receipt["manifest_sha256"]),
        "payload_digest": str(receipt["payload_digest"]),
    }


def build_exact_chaser_projection_receipt(
    analysis_zarr: str | Path,
    *,
    exact_child_receipts: Mapping[str, str | Path],
    relative_frame_receipts: Mapping[str, str | Path],
    palette_commit: str,
    expected_recording_id: str | None = None,
) -> dict[str, Any]:
    """Validate and compose the closed receipt set for one exact projection."""

    archive = Path(analysis_zarr).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    if set(exact_child_receipts) != set(EXACT_CHILD_KEYS):
        _fail("Exact-child receipt keys are missing or unexpected.")
    if set(relative_frame_receipts) != set(RELATIVE_CHILD_KEYS):
        _fail("Relative-frame receipt keys are missing or unexpected.")
    recording_id = (
        _text(expected_recording_id, field="expected_recording_id")
        if expected_recording_id is not None
        else None
    )
    exact_bindings: dict[str, Any] = {}
    relative_bindings: dict[str, Any] = {}
    for key in EXACT_CHILD_KEYS:
        path = _canonical_receipt_path(
            exact_child_receipts[key], field=f"exact_child_receipts.{key}"
        )
        receipt = read_exact_immutable_child_validation_receipt(
            path,
            expected_analysis_zarr=archive,
            expected_recording_id=recording_id,
        )
        if recording_id is None:
            recording_id = str(receipt["recording_id"])
        exact_bindings[key] = _child_binding(receipt, path)
    assert recording_id is not None
    for key in RELATIVE_CHILD_KEYS:
        path = _canonical_receipt_path(
            relative_frame_receipts[key], field=f"relative_frame_receipts.{key}"
        )
        receipt = read_chaser_relative_frame_validation_receipt(
            path,
            expected_analysis_zarr=archive,
            expected_recording_id=recording_id,
        )
        relative_bindings[key] = _child_binding(receipt, path)
    body = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        "analysis_zarr": str(archive),
        "recording_id": recording_id,
        "exact_children": exact_bindings,
        "relative_frame_children": relative_bindings,
        "policy": POLICY,
        "safety": SAFETY,
        "software_authority": {
            "repository": "palette",
            "commit": _commit(palette_commit),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def _validate_binding_record(value: object, *, field: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "receipt_path",
        "receipt_sha256",
        "run_path",
        "manifest_sha256",
        "payload_digest",
    }:
        _fail(f"{field} must be one exact child binding.")
    return MappingProxyType(
        {
            "receipt_path": _text(
                value.get("receipt_path"), field=f"{field}.receipt_path"
            ),
            "receipt_sha256": _digest(
                value.get("receipt_sha256"), field=f"{field}.receipt_sha256"
            ),
            "run_path": _text(value.get("run_path"), field=f"{field}.run_path"),
            "manifest_sha256": _digest(
                value.get("manifest_sha256"), field=f"{field}.manifest_sha256"
            ),
            "payload_digest": _digest(
                value.get("payload_digest"), field=f"{field}.payload_digest"
            ),
        }
    )


def validate_exact_chaser_projection_receipt(
    receipt: object,
    *,
    expected_analysis_zarr: str | Path | None = None,
    expected_recording_id: str | None = None,
    validate_current_metadata: bool = True,
    validate_child_receipts: bool = True,
) -> Mapping[str, Any]:
    """Validate the composition and, by default, every child metadata generation."""

    if not isinstance(receipt, Mapping):
        _fail("Exact-chaser projection receipt must be one object.")
    value = _plain(receipt)
    persisted = value.pop("record_sha256", None)
    if _digest(persisted, field="record_sha256") != canonical_json_sha256(value):
        _fail("Exact-chaser projection receipt digest is stale.")
    required = {
        "schema_id",
        "schema_version",
        "status",
        "analysis_zarr",
        "recording_id",
        "exact_children",
        "relative_frame_children",
        "policy",
        "safety",
        "software_authority",
        "created_at_utc",
    }
    if set(value) != required:
        _fail("Exact-chaser projection receipt fields are missing or unexpected.")
    if (
        value["schema_id"] != RECEIPT_SCHEMA_ID
        or value["schema_version"] != RECEIPT_SCHEMA_VERSION
        or value["status"] != RECEIPT_STATUS
        or value["policy"] != POLICY
        or value["safety"] != SAFETY
    ):
        _fail("Exact-chaser projection receipt identity, policy, or safety is invalid.")
    archive = Path(_text(value["analysis_zarr"], field="analysis_zarr")).resolve()
    if str(archive) != value["analysis_zarr"]:
        _fail("Exact-chaser projection archive path is not canonical and absolute.")
    recording_id = _text(value["recording_id"], field="recording_id")
    if (
        expected_analysis_zarr is not None
        and archive != Path(expected_analysis_zarr).expanduser().resolve()
    ):
        _fail("Exact-chaser projection receipt names another archive.")
    if expected_recording_id is not None and recording_id != _text(
        expected_recording_id, field="expected_recording_id"
    ):
        _fail("Exact-chaser projection receipt names another recording.")
    software = value["software_authority"]
    if (
        not isinstance(software, Mapping)
        or set(software) != {"repository", "commit"}
        or software.get("repository") != "palette"
        or _commit(software.get("commit")) != software.get("commit")
    ):
        _fail("Exact-chaser projection receipt software authority is invalid.")
    _text(value["created_at_utc"], field="created_at_utc")
    exact = value["exact_children"]
    relative = value["relative_frame_children"]
    if not isinstance(exact, Mapping) or set(exact) != set(EXACT_CHILD_KEYS):
        _fail("Exact-chaser projection exact-child roster is inexact.")
    if not isinstance(relative, Mapping) or set(relative) != set(RELATIVE_CHILD_KEYS):
        _fail("Exact-chaser projection relative-frame roster is inexact.")
    for key in EXACT_CHILD_KEYS:
        binding = _validate_binding_record(exact[key], field=f"exact_children.{key}")
        if validate_child_receipts:
            path = _canonical_receipt_path(
                binding["receipt_path"], field=f"exact_children.{key}"
            )
            child = read_exact_immutable_child_validation_receipt(
                path,
                expected_analysis_zarr=archive,
                expected_recording_id=recording_id,
                expected_run_path=binding["run_path"],
                validate_current_metadata=validate_current_metadata,
            )
            if _child_binding(child, path) != dict(binding):
                _fail(f"Exact child {key!r} differs from its composition binding.")
    for key in RELATIVE_CHILD_KEYS:
        binding = _validate_binding_record(
            relative[key], field=f"relative_frame_children.{key}"
        )
        if validate_child_receipts:
            path = _canonical_receipt_path(
                binding["receipt_path"], field=f"relative_frame_children.{key}"
            )
            child = read_chaser_relative_frame_validation_receipt(
                path,
                expected_analysis_zarr=archive,
                expected_recording_id=recording_id,
                expected_run_name=binding["run_path"].rsplit("/", 1)[-1],
                expected_manifest_sha256=binding["manifest_sha256"],
                validate_current_metadata=validate_current_metadata,
            )
            if _child_binding(child, path) != dict(binding):
                _fail(
                    f"Relative-frame child {key!r} differs from its composition binding."
                )
    return MappingProxyType({**value, "record_sha256": persisted})


def read_exact_chaser_projection_receipt(
    path: str | Path, **expected: Any
) -> Mapping[str, Any]:
    return validate_exact_chaser_projection_receipt(
        _read_object(Path(path).expanduser().resolve()), **expected
    )


def ensure_exact_chaser_projection_receipt(
    analysis_zarr: str | Path,
    *,
    exact_child_receipts: Mapping[str, str | Path],
    relative_frame_receipts: Mapping[str, str | Path],
    palette_commit: str,
    output_json: str | Path,
    expected_recording_id: str | None = None,
) -> dict[str, Any]:
    """Create the composition receipt or validate and reuse its exact generation."""

    output = Path(output_json).expanduser().resolve()
    if output.exists():
        current = read_exact_chaser_projection_receipt(
            output,
            expected_analysis_zarr=analysis_zarr,
            expected_recording_id=expected_recording_id,
        )
        rebuilt = build_exact_chaser_projection_receipt(
            analysis_zarr,
            exact_child_receipts=exact_child_receipts,
            relative_frame_receipts=relative_frame_receipts,
            palette_commit=palette_commit,
            expected_recording_id=expected_recording_id,
        )
        for field in (
            "analysis_zarr",
            "recording_id",
            "exact_children",
            "relative_frame_children",
            "policy",
            "safety",
            "software_authority",
        ):
            if _plain(current[field]) != rebuilt[field]:
                _fail("Existing projection receipt binds another exact composition.")
        return {**_plain(current), "receipt_path": str(output), "mode": "reused_exact"}
    receipt = build_exact_chaser_projection_receipt(
        analysis_zarr,
        exact_child_receipts=exact_child_receipts,
        relative_frame_receipts=relative_frame_receipts,
        palette_commit=palette_commit,
        expected_recording_id=expected_recording_id,
    )
    write_json_atomic(output, receipt, overwrite=False)
    return {**receipt, "receipt_path": str(output), "mode": "created"}


__all__ = [
    "EXACT_CHILD_KEYS",
    "RELATIVE_CHILD_KEYS",
    "RECEIPT_SCHEMA_ID",
    "RECEIPT_SCHEMA_VERSION",
    "VERIFICATION_MODE",
    "ExactChaserProjectionReceiptError",
    "build_exact_chaser_projection_receipt",
    "ensure_exact_chaser_projection_receipt",
    "read_exact_chaser_projection_receipt",
    "validate_exact_chaser_projection_receipt",
]
