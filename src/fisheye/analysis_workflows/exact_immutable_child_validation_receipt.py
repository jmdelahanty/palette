"""Reusable receipts for exact selector-ineligible immutable Zarr children.

This receipt is for consumers that already know one exact child path.  It
never resolves a selector.  Creation streams and rehashes every array declared
by the child's scientific manifest; reuse validates the complete direct child
metadata generation without parsing the archive-root consolidated document.
Consumers still rehash the arrays they load through their typed source loader.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping

from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    STREAMING_HASH_BLOCK_BYTES,
    _metadata_inventory,
    _streaming_array_values_sha256,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root


RECEIPT_SCHEMA_ID = "palette.analysis.exact_immutable_child.validation_receipt"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_STATUS = "complete_selector_ineligible_direct_child_audit"
VERIFICATION_MODE = "receipt_bound_exact_direct_child_v1"
VALIDATION_POLICY = {
    "initial_array_validation": (
        "streaming_all_manifest_declared_arrays_content_sha256_v1"
    ),
    "reuse_metadata_validation": "all_direct_subtree_zarr_json_sha256",
    "consumer_array_validation": "typed_loader_content_sha256",
    "archive_root_consolidated_metadata_reparse": False,
    "exact_run_path_required": True,
    "selector_resolution": False,
}
EXPECTED_SAFETY = {
    "selector_eligible": False,
    "production_authority": False,
    "registry_update": False,
    "selector_activation": False,
}
RECEIPT_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "status",
        "analysis_zarr",
        "run_path",
        "recording_id",
        "manifest_attr",
        "manifest_digest_attr",
        "manifest",
        "manifest_sha256",
        "payload_digest",
        "array_declarations",
        "streaming_array_audit",
        "direct_metadata_inventory",
        "run_provenance_sha256",
        "completion_authority",
        "software_authority",
        "validation_policy",
        "safety",
        "created_at_utc",
    }
)

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SELECTOR_PARTS = {
    "latest",
    "latest_complete",
    "selected",
    "current",
    "active",
    "default",
}


class ExactImmutableChildValidationReceiptError(ValueError):
    """Raised when an exact-child receipt or its source is invalid."""


def _fail(message: str) -> None:
    raise ExactImmutableChildValidationReceiptError(message)


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


def _run_path(value: object) -> str:
    result = _text(value, field="run_path")
    path = PurePosixPath(result)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "analysis"
        or "\\" in result
        or any(part in {"", ".", ".."} or part in _SELECTOR_PARTS for part in path.parts)
    ):
        _fail("run_path must name one exact non-selector child below analysis/.")
    return path.as_posix()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactImmutableChildValidationReceiptError(
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


def _array_paths(group: Any, *, prefix: str = "") -> set[str]:
    paths = {
        f"{prefix}/{name}" if prefix else str(name)
        for name in group.array_keys()
    }
    for name in group.group_keys():
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        paths.update(_array_paths(group[name], prefix=child_prefix))
    return paths


def _streaming_manifest_array_audit(
    run_directory: Path,
    *,
    declarations: list[Mapping[str, Any]],
) -> dict[str, Any]:
    by_path: dict[str, Mapping[str, Any]] = {}
    for declaration in declarations:
        path = declaration.get("path")
        if type(path) is not str or not path or path in by_path:
            _fail("Manifest array paths are absent or duplicated.")
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
            _fail("Manifest array path leaves the exact immutable child.")
        by_path[path] = declaration
    run = open_zarr_root(run_directory, mode="r", use_consolidated=False)
    if _array_paths(run) != set(by_path):
        _fail("Exact child array inventory differs from its manifest.")
    bindings: dict[str, str] = {}
    for path, declaration in sorted(by_path.items()):
        expected = _digest(
            declaration.get("content_sha256"),
            field=f"array_declarations[{path}].content_sha256",
        )
        shape = declaration.get("shape")
        if not isinstance(shape, list) or any(
            type(value) is not int or value < 0 for value in shape
        ):
            _fail(f"Manifest array {path!r} shape is invalid.")
        observed = _streaming_array_values_sha256(
            run[path],
            expected_dtype=_text(
                declaration.get("dtype"),
                field=f"array_declarations[{path}].dtype",
            ),
            expected_shape=shape,
        )
        if observed != expected:
            _fail(f"Exact child array {path!r} content digest changed.")
        bindings[path] = observed
    return {
        "valid": True,
        "array_count": len(declarations),
        "policy_id": "streaming_all_manifest_declared_arrays_content_sha256_v1",
        "maximum_requested_block_bytes": STREAMING_HASH_BLOCK_BYTES,
        "content_bindings_sha256": canonical_json_sha256(bindings),
    }


def build_exact_immutable_child_validation_receipt(
    analysis_zarr: str | Path,
    *,
    run_path: str,
    manifest_attr: str,
    manifest_digest_attr: str,
    palette_commit: str,
    expected_recording_id: str | None = None,
) -> dict[str, Any]:
    """Stream-audit one exact child and build its reusable receipt."""

    archive = Path(analysis_zarr).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    exact_path = _run_path(run_path)
    child = archive / exact_path
    document = _read_object(child / "zarr.json")
    attrs = document.get("attributes")
    if document.get("node_type") != "group" or not isinstance(attrs, dict):
        _fail("Exact child zarr.json is not one direct Zarr group.")
    manifest_name = _text(manifest_attr, field="manifest_attr")
    digest_name = _text(manifest_digest_attr, field="manifest_digest_attr")
    manifest = attrs.get(manifest_name)
    if not isinstance(manifest, dict):
        _fail("Exact child manifest is absent.")
    manifest_sha = canonical_json_sha256(manifest)
    if attrs.get(digest_name) != manifest_sha:
        _fail("Exact child manifest digest is stale.")
    recording_id = _text(manifest.get("recording_id"), field="recording_id")
    if expected_recording_id is not None and recording_id != _text(
        expected_recording_id, field="expected_recording_id"
    ):
        _fail("Exact child belongs to another recording.")
    if manifest.get("run_path") != exact_path:
        _fail("Exact child manifest names another run path.")
    if (
        attrs.get("palette_run_completion_contract")
        != "palette.zarr_run_completion.v1"
        or attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("palette_run_name") != PurePosixPath(exact_path).name
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("production_authority") is not False
        or attrs.get("registry_update") is not False
        or attrs.get("selection") != "none"
    ):
        _fail("Exact child is not complete and selector-ineligible.")
    declarations = manifest.get("array_declarations")
    if not isinstance(declarations, list) or not declarations or not all(
        isinstance(value, Mapping) for value in declarations
    ):
        _fail("Exact child manifest lacks array declarations.")
    provenance = attrs.get("run_provenance")
    if not isinstance(provenance, dict):
        _fail("Exact child lacks run provenance.")
    array_audit = _streaming_manifest_array_audit(
        child,
        declarations=declarations,
    )
    body = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        "analysis_zarr": str(archive),
        "run_path": exact_path,
        "recording_id": recording_id,
        "manifest_attr": manifest_name,
        "manifest_digest_attr": digest_name,
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "payload_digest": _digest(
            manifest.get("payload_digest"), field="manifest.payload_digest"
        ),
        "array_declarations": declarations,
        "streaming_array_audit": array_audit,
        "direct_metadata_inventory": _metadata_inventory(child),
        "run_provenance_sha256": canonical_json_sha256(provenance),
        "completion_authority": {
            "contract": attrs["palette_run_completion_contract"],
            "status": attrs["palette_run_completion_status"],
            "completed_at_utc": _text(
                attrs.get("palette_run_completed_at_utc"),
                field="palette_run_completed_at_utc",
            ),
            "publication_owner_uuid": _text(
                attrs.get("atomic_publication_owner_uuid"),
                field="atomic_publication_owner_uuid",
            ),
        },
        "software_authority": {
            "repository": "palette",
            "commit": _commit(palette_commit),
        },
        "validation_policy": VALIDATION_POLICY,
        "safety": EXPECTED_SAFETY,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def validate_exact_immutable_child_validation_receipt(
    receipt: object,
    *,
    expected_analysis_zarr: str | Path | None = None,
    expected_run_path: str | None = None,
    expected_recording_id: str | None = None,
    expected_manifest_attr: str | None = None,
    expected_manifest_digest_attr: str | None = None,
    validate_current_metadata: bool = True,
) -> Mapping[str, Any]:
    """Strictly validate a receipt and its current direct metadata generation."""

    if not isinstance(receipt, Mapping):
        _fail("Exact-child receipt must be one object.")
    value = _plain(receipt)
    persisted = value.pop("record_sha256", None)
    if _digest(persisted, field="record_sha256") != canonical_json_sha256(value):
        _fail("Exact-child receipt digest is stale.")
    if set(value) != RECEIPT_FIELDS:
        _fail("Exact-child receipt field set is inexact.")
    if (
        value.get("schema_id") != RECEIPT_SCHEMA_ID
        or value.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or value.get("status") != RECEIPT_STATUS
        or value.get("validation_policy") != VALIDATION_POLICY
        or value.get("safety") != EXPECTED_SAFETY
    ):
        _fail("Exact-child receipt identity, policy, or safety is invalid.")
    archive = Path(_text(value.get("analysis_zarr"), field="analysis_zarr")).resolve()
    exact_path = _run_path(value.get("run_path"))
    recording_id = _text(value.get("recording_id"), field="recording_id")
    manifest_name = _text(value.get("manifest_attr"), field="manifest_attr")
    digest_name = _text(
        value.get("manifest_digest_attr"), field="manifest_digest_attr"
    )
    manifest = value.get("manifest")
    if not isinstance(manifest, dict):
        _fail("Exact-child receipt manifest is invalid.")
    manifest_sha = _digest(value.get("manifest_sha256"), field="manifest_sha256")
    payload_digest = _digest(value.get("payload_digest"), field="payload_digest")
    if (
        canonical_json_sha256(manifest) != manifest_sha
        or manifest.get("run_path") != exact_path
        or manifest.get("recording_id") != recording_id
        or manifest.get("payload_digest") != payload_digest
        or value.get("array_declarations") != manifest.get("array_declarations")
    ):
        _fail("Exact-child receipt manifest bindings are invalid.")
    if expected_analysis_zarr is not None and archive != Path(
        expected_analysis_zarr
    ).expanduser().resolve():
        _fail("Exact-child receipt names another archive.")
    if expected_run_path is not None and exact_path != _run_path(expected_run_path):
        _fail("Exact-child receipt names another run path.")
    if expected_recording_id is not None and recording_id != _text(
        expected_recording_id, field="expected_recording_id"
    ):
        _fail("Exact-child receipt names another recording.")
    if expected_manifest_attr is not None and manifest_name != _text(
        expected_manifest_attr, field="expected_manifest_attr"
    ):
        _fail("Exact-child receipt names another manifest attr.")
    if expected_manifest_digest_attr is not None and digest_name != _text(
        expected_manifest_digest_attr, field="expected_manifest_digest_attr"
    ):
        _fail("Exact-child receipt names another manifest digest attr.")
    audit = value.get("streaming_array_audit")
    declarations = value.get("array_declarations")
    if (
        not isinstance(audit, dict)
        or set(audit)
        != {
            "valid",
            "array_count",
            "policy_id",
            "maximum_requested_block_bytes",
            "content_bindings_sha256",
        }
        or audit.get("valid") is not True
        or audit.get("policy_id")
        != "streaming_all_manifest_declared_arrays_content_sha256_v1"
        or not isinstance(declarations, list)
        or not declarations
        or not all(isinstance(declaration, dict) for declaration in declarations)
        or audit.get("array_count") != len(declarations)
        or audit.get("maximum_requested_block_bytes")
        != STREAMING_HASH_BLOCK_BYTES
    ):
        _fail("Exact-child receipt streaming audit is invalid.")
    _digest(
        audit.get("content_bindings_sha256"),
        field="streaming_array_audit.content_bindings_sha256",
    )
    inventory = value.get("direct_metadata_inventory")
    if not isinstance(inventory, dict) or set(inventory) != {
        "policy_id",
        "file_count",
        "files",
        "inventory_sha256",
    }:
        _fail("Exact-child receipt metadata inventory is invalid.")
    files = inventory.get("files")
    if (
        inventory.get("policy_id")
        != "all_direct_subtree_zarr_json_files_sha256_v1"
        or not isinstance(files, list)
        or not files
        or inventory.get("file_count") != len(files)
        or canonical_json_sha256(files)
        != _digest(
            inventory.get("inventory_sha256"),
            field="direct_metadata_inventory.inventory_sha256",
        )
    ):
        _fail("Exact-child receipt metadata inventory bindings are invalid.")
    for record in files:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size_bytes", "sha256"}
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] < 0
        ):
            _fail("Exact-child receipt metadata file record is invalid.")
        _text(record.get("path"), field="direct_metadata_inventory.files.path")
        _digest(
            record.get("sha256"),
            field="direct_metadata_inventory.files.sha256",
        )
    if not any(record["path"] == "zarr.json" for record in files):
        _fail("Exact-child receipt metadata inventory lacks child root metadata.")
    _digest(value.get("run_provenance_sha256"), field="run_provenance_sha256")
    completion = value.get("completion_authority")
    if (
        not isinstance(completion, dict)
        or set(completion)
        != {"contract", "status", "completed_at_utc", "publication_owner_uuid"}
        or completion.get("contract") != "palette.zarr_run_completion.v1"
        or completion.get("status") != "complete"
    ):
        _fail("Exact-child receipt completion authority is invalid.")
    _text(completion.get("completed_at_utc"), field="completed_at_utc")
    _text(
        completion.get("publication_owner_uuid"),
        field="publication_owner_uuid",
    )
    _text(value.get("created_at_utc"), field="created_at_utc")
    software = value.get("software_authority")
    if (
        not isinstance(software, dict)
        or software.get("repository") != "palette"
        or _commit(software.get("commit")) != software.get("commit")
    ):
        _fail("Exact-child receipt software authority is invalid.")
    if validate_current_metadata:
        child = archive / exact_path
        document = _read_object(child / "zarr.json")
        attrs = document.get("attributes")
        completion = value.get("completion_authority")
        if not isinstance(attrs, dict) or not isinstance(completion, dict):
            _fail("Exact-child current metadata or completion authority is invalid.")
        if (
            attrs.get(manifest_name) != manifest
            or attrs.get(digest_name) != manifest_sha
            or canonical_json_sha256(attrs.get("run_provenance"))
            != value.get("run_provenance_sha256")
            or attrs.get("atomic_publication_owner_uuid")
            != completion.get("publication_owner_uuid")
            or attrs.get("palette_run_completion_status") != "complete"
            or _metadata_inventory(child) != value.get("direct_metadata_inventory")
        ):
            _fail("Receipt-bound exact child metadata generation changed.")
    return MappingProxyType({**value, "record_sha256": persisted})


def read_exact_immutable_child_validation_receipt(
    path: str | Path,
    **expected: Any,
) -> Mapping[str, Any]:
    receipt_path = Path(path).expanduser().resolve()
    return validate_exact_immutable_child_validation_receipt(
        _read_object(receipt_path), **expected
    )


def ensure_exact_immutable_child_validation_receipt(
    analysis_zarr: str | Path,
    *,
    run_path: str,
    manifest_attr: str,
    manifest_digest_attr: str,
    palette_commit: str,
    output_json: str | Path,
    expected_recording_id: str | None = None,
) -> dict[str, Any]:
    """Create a receipt or exactly validate and reuse its current generation."""

    output = Path(output_json).expanduser().resolve()
    if output.exists():
        receipt = read_exact_immutable_child_validation_receipt(
            output,
            expected_analysis_zarr=analysis_zarr,
            expected_run_path=run_path,
            expected_recording_id=expected_recording_id,
            expected_manifest_attr=manifest_attr,
            expected_manifest_digest_attr=manifest_digest_attr,
        )
        if receipt["software_authority"]["commit"] != _commit(palette_commit):
            _fail("Existing exact-child receipt belongs to another Palette commit.")
        return {**_plain(receipt), "receipt_path": str(output), "mode": "reused_exact"}
    receipt = build_exact_immutable_child_validation_receipt(
        analysis_zarr,
        run_path=run_path,
        manifest_attr=manifest_attr,
        manifest_digest_attr=manifest_digest_attr,
        palette_commit=palette_commit,
        expected_recording_id=expected_recording_id,
    )
    write_json_atomic(output, receipt, overwrite=False)
    return {**receipt, "receipt_path": str(output), "mode": "created"}


__all__ = [
    "RECEIPT_SCHEMA_ID",
    "RECEIPT_SCHEMA_VERSION",
    "VERIFICATION_MODE",
    "ExactImmutableChildValidationReceiptError",
    "build_exact_immutable_child_validation_receipt",
    "ensure_exact_immutable_child_validation_receipt",
    "read_exact_immutable_child_validation_receipt",
    "validate_exact_immutable_child_validation_receipt",
]
